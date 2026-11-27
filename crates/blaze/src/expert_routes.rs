pub(crate) fn group_by_expert<T: Copy>(
    routes: impl Iterator<Item = (usize, T)> + Clone,
    offsets: &mut [i32],
    cursors: &mut [i32],
    grouped: &mut [T],
) -> bool {
    if offsets.len() != cursors.len() + 1 || grouped.len() > i32::MAX as usize {
        return false;
    }
    offsets.fill(0);
    let counts = &mut offsets[1..];
    for (index, (expert, _)) in routes.clone().enumerate() {
        if index == grouped.len() {
            return false;
        }
        let Some(count) = counts.get_mut(expert) else {
            return false;
        };
        *count += 1;
    }
    for expert in 0..cursors.len() {
        offsets[expert + 1] += offsets[expert];
    }
    if *offsets.last().unwrap() as usize != grouped.len() {
        return false;
    }
    cursors.copy_from_slice(&offsets[..cursors.len()]);
    for (expert, route) in routes {
        let offset = &mut cursors[expert];
        grouped[*offset as usize] = route;
        *offset += 1;
    }
    true
}

pub(crate) unsafe extern "C" fn group_route_ids(
    ids: *const i32, routes: usize, experts: usize, offsets: *mut i32, cursors: *mut i32,
    grouped: *mut i32,
) -> std::ffi::c_int {
    if routes == 0 || routes > i32::MAX as usize || experts == 0 || experts >= i32::MAX as usize {
        return 1;
    }
    // HybridMoe owns separate, pre-sized arrays for these four slices.
    let (ids, offsets, cursors, grouped) = unsafe {
        (
            std::slice::from_raw_parts(ids, routes),
            std::slice::from_raw_parts_mut(offsets, experts + 1),
            std::slice::from_raw_parts_mut(cursors, experts),
            std::slice::from_raw_parts_mut(grouped, routes),
        )
    };
    let routes = ids.iter().enumerate().map(|(index, &expert)| (expert as usize, index as i32));
    if group_by_expert(routes, offsets, cursors, grouped) { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_keep_original_route_order_within_each_expert() {
        let ids = [2, 0, 2, 1, 0];
        let mut offsets = [0; 5];
        let mut cursors = [0; 4];
        let mut grouped = [-1; 5];
        let call = |ids: &[i32; 5], offsets: &mut [i32; 5], cursors: &mut [i32; 4], grouped: &mut [i32; 5]| unsafe {
            group_route_ids(ids.as_ptr(), ids.len(), cursors.len(), offsets.as_mut_ptr(), cursors.as_mut_ptr(), grouped.as_mut_ptr())
        };
        assert_eq!(call(&ids, &mut offsets, &mut cursors, &mut grouped), 0);
        assert_eq!(offsets, [0, 2, 3, 5, 5]);
        assert_eq!(grouped, [1, 4, 3, 0, 2]);
        assert_ne!(call(&[2, 0, -1, 1, 0], &mut offsets, &mut cursors, &mut grouped), 0);
    }
}
