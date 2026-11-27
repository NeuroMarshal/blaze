use crate::gpu::{Stream, check};
use anyhow::{Result, bail, ensure};
use blaze_sys as sys;
use std::ptr::null_mut;
use std::{ffi::CString, ops::Range, path::Path};

pub struct MoeEngine(*mut sys::blz_moe_engine);

unsafe impl Send for MoeEngine {}

impl MoeEngine {
    pub(crate) unsafe fn create(mut config: sys::blz_moe_engine_config) -> Result<Self> {
        config.group_routes = Some(crate::expert_routes::group_route_ids);
        config.rank_experts = Some(crate::expert_cache::rank_expert_ids);
        let mut engine = null_mut();
        let mut error = [0 as std::ffi::c_char; 1024];
        if unsafe { sys::blz_moe_engine_create(&config, &mut engine, error.as_mut_ptr(), error.len()) } != 0 {
            let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("MoE engine: {message}");
        }
        ensure!(!engine.is_null(), "MoE engine returned a null handle");
        Ok(Self(engine))
    }

    pub(crate) fn layers(&self) -> i32 {
        unsafe { sys::blz_moe_engine_layers(self.0) }
    }

    pub(crate) fn cpu_routes(&self) -> u64 {
        unsafe { sys::blz_moe_engine_cpu_routes(self.0) }
    }

    pub(crate) fn packed(
        directory: &Path, layers: Range<usize>, device: i32, budget: usize, top_k: usize,
        threads: usize, chunk: usize, stream_gpu_only: bool, all_resident: bool, caps: (f32, f32),
    ) -> Result<Self> {
        let model_dir = CString::new(directory.as_os_str().as_encoded_bytes())?;
        let heat = directory.join(format!("heat-{}-{}.bin", layers.start, layers.end));
        let heat = CString::new(heat.as_os_str().as_encoded_bytes())?;
        let wide_min = if stream_gpu_only { 1 } else { chunk.clamp(32, 256) };
        let wide_a8 = i32::from(std::env::var_os("BLAZE_MOE_WIDE_A8").is_some_and(|v| v == "1"));
        let options = CString::new(format!("layers={}-{},wide_tokens={chunk},wide_min_tokens={wide_min},wide_a8={wide_a8}",
            layers.start, layers.end))?;
        unsafe {
            Self::create(sys::blz_moe_engine_config {
                model_dir: model_dir.as_ptr(),
                topk: top_k as i32,
                gpu0: device,
                gpu1: -1,
                gpu0_gib: budget as f64 / (1u64 << 30) as f64,
                gpu1_gib: 0.0,
                ram_gib: -1.0,
                cpu_threads: threads as i32,
                cpu_cores: std::ptr::null(),
                heat_path: heat.as_ptr(),
                swap_options: if all_resident { c"enabled=0".as_ptr() } else { std::ptr::null() },
                options: options.as_ptr(),
                group_routes: None,
                rank_experts: None,
                gate_cap: caps.0,
                up_cap: caps.1,
            })
        }
    }

    pub(crate) unsafe fn forward(&self, layer: usize, x: *const u16, tokens: usize, ids: *const i32,
        weights: *const f32, out: *mut u16, stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_moe_engine_forward(self.0, layer as i32, x, tokens as i32, ids, weights, out, stream.0)
        }, "moe")
    }
}

impl Drop for MoeEngine {
    fn drop(&mut self) {
        unsafe { sys::blz_moe_engine_destroy(self.0) };
    }
}
