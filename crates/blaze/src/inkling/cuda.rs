use super::KvDtype;
use super::cpu::Routed;
use super::weights::{ExpertSlice, Experts, PackedMatrix, Packing};
use crate::gpu::{DevBuf, HostMapped, Stream, check};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Result, ensure};
use blaze_sys as sys;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ptr::null_mut;
use std::sync::Arc;

pub(super) struct Dense {
    weight: DevBuf,
    pub rows: usize,
    cols: usize,
}

impl Dense {
    pub fn load(ckpt: &Checkpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let t = ckpt.info(name)?;
        ensure!(
            t.dtype == Dtype::Bf16 && t.shape == [rows, cols],
            "{name}: expected BF16 [{rows}, {cols}], found {:?} {:?}",
            t.dtype,
            t.shape
        );
        Ok(Self {
            weight: DevBuf::from_slice(ckpt.bytes(name)?)?,
            rows,
            cols,
        })
    }

    pub fn stack(ckpt: &Checkpoint, names: &[(&str, usize)], cols: usize) -> Result<Self> {
        let rows = names.iter().map(|(_, n)| n).sum::<usize>();
        let weight = DevBuf::new(rows * cols * 2)?;
        let mut offset = 0;
        for &(name, n) in names {
            let t = ckpt.info(name)?;
            ensure!(
                t.dtype == Dtype::Bf16 && t.shape == [n, cols],
                "{name}: expected BF16 [{n}, {cols}], found {:?} {:?}",
                t.dtype,
                t.shape
            );
            weight.upload_at(offset, ckpt.bytes(name)?)?;
            offset += t.bytes();
        }
        Ok(Self { weight, rows, cols })
    }

    pub fn from_bf16(weight: &[u16], rows: usize, cols: usize) -> Result<Self> {
        ensure!(weight.len() == rows * cols, "Inkling: invalid dense matrix shape");
        Ok(Self {
            weight: DevBuf::from_slice(weight)?,
            rows,
            cols,
        })
    }

    pub(super) fn run(&self, gpu: &Compute, x: *const u16, out: *mut u16, m: usize) -> Result<()> {
        let (n, k) = (self.rows as i32, self.cols as i32);
        let weight = self.weight.ptr::<u16>();
        if m == 1 && (x as usize).is_multiple_of(16) {
            if self.rows > 4096 && crate::tune::wide_fits(m, self.cols, self.cols) {
                return check(
                    unsafe { sys::blz_gemv_wide(x.cast(), k.into(), weight.cast(), out.cast(), n.into(), 1, n, k, gpu.stream.0) },
                    "Inkling wide GEMV",
                );
            }
            if self.rows <= 4096 && crate::tune::skinny_fits(m, self.cols, self.cols) {
                return check(
                    unsafe {
                        sys::blz_skinny_gemm_ex(
                            x.cast(),
                            k.into(),
                            weight.cast(),
                            out.cast(),
                            n.into(),
                            1,
                            n,
                            k,
                            sys::BLZ_BF16,
                            gpu.stream.0,
                        )
                    },
                    "Inkling skinny GEMV",
                );
            }
        }
        check(
            unsafe {
                sys::blz_gemm_bf16(
                    gpu.gemm,
                    x,
                    self.cols as i64,
                    weight,
                    out,
                    self.rows as i64,
                    m as i32,
                    self.rows as i32,
                    self.cols as i32,
                    gpu.stream.0,
                )
            },
            "Inkling dense projection",
        )
    }
}

struct OnDevice {
    offset: usize,
    scale_offset: usize,
    alpha_offset: usize,
    packing: Packing,
    rows: usize,
    cols: usize,
}

impl OnDevice {
    fn store(buffer: &DevBuf, offset: usize, matrix: &PackedMatrix, data: ExpertSlice<'_>) -> Result<Self> {
        let scale_offset = offset + data.codes.len();
        let alpha_offset = scale_offset + data.scales.len();
        buffer.upload_at(offset, data.codes)?;
        buffer.upload_at(scale_offset, data.scales)?;
        if let Some(alpha) = data.alpha {
            ensure!(alpha_offset % 4 == 0, "Inkling NVFP4 expert scale alignment");
            buffer.upload_at(alpha_offset, &vec![alpha; matrix.rows])?;
        }
        Ok(Self {
            offset,
            scale_offset,
            alpha_offset,
            packing: matrix.packing,
            rows: matrix.rows,
            cols: matrix.cols,
        })
    }

    fn run(&self, buffer: &DevBuf, x: *const u16, out: *mut u16, m: usize, stream: &Stream) -> Result<()> {
        let (n, k) = (self.rows as i32, self.cols as i32);
        let codes = buffer.at(self.offset);
        let scales = buffer.at(self.scale_offset);
        let err = unsafe {
            match self.packing {
                Packing::Nvfp4 => sys::blz_nvfp4_gemv_mma(
                    x,
                    k.into(),
                    codes,
                    scales,
                    buffer.at(self.alpha_offset / 4),
                    out,
                    n.into(),
                    m as i32,
                    n,
                    k,
                    stream.0,
                ),
                Packing::Mxfp8 => sys::blz_mxfp8_gemv_mma(x, k.into(), codes, scales, out, n.into(), m as i32, n, k, stream.0),
            }
        };
        check(err, "Inkling packed expert GEMV")
    }

    fn offsets(&self) -> sys::blz_inkling_matrix {
        sys::blz_inkling_matrix {
            codes: self.offset as u64,
            scales: self.scale_offset as u64,
            alpha: self.alpha_offset as u64,
        }
    }
}

type GroupedExpertGemv = unsafe extern "C" fn(
    *const u16, i64, *const u8, *const sys::blz_inkling_matrix, *mut u16, i32, i32, i32, sys::cudaStream_t,
) -> sys::cudaError_t;

#[derive(Clone, Copy, Default)]
struct Heat {
    count: usize,
    epoch: usize,
}

impl Heat {
    fn at(self, epoch: usize) -> usize {
        self.count >> epoch.saturating_sub(self.epoch).min(usize::BITS as usize - 1)
    }

    fn add(&mut self, epoch: usize, observations: usize) {
        self.count = self.at(epoch).saturating_add(observations);
        self.epoch = epoch;
    }
}

struct DeviceExpert {
    gate_up: OnDevice,
    down: OnDevice,
    slot: usize,
    last_used: usize,
    heat: Heat,
}

struct ExpertPool {
    buffer: DevBuf,
    stride: usize,
    free_slots: Vec<usize>,
    entries: HashMap<(usize, usize), DeviceExpert>,
    hits: HashMap<(usize, usize), Heat>,
}

impl ExpertPool {
    fn new(bytes: usize, stride: usize) -> Result<Self> {
        let slots = bytes / stride;
        Ok(Self {
            buffer: DevBuf::new(slots * stride)?,
            stride,
            free_slots: (0..slots).rev().collect(),
            entries: HashMap::new(),
            hits: HashMap::new(),
        })
    }

    fn use_gpu(
        &mut self,
        ckpt: &Checkpoint,
        weights: &Experts,
        key: (usize, usize),
        tick: usize,
        epoch: usize,
        observations: Option<usize>,
        preserve_group: bool,
    ) -> Result<bool> {
        let mut heat = Heat::default();
        if let Some(observations) = observations {
            let hits = self.hits.entry(key).or_default();
            hits.add(epoch, observations);
            heat = *hits;
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = tick;
            entry.heat = heat;
            return Ok(true);
        }
        if observations.is_some() && heat.count < 2 {
            return Ok(false);
        }
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None => {
                let victim = self
                    .entries
                    .iter()
                    .filter(|(_, e)| !preserve_group || e.last_used != tick)
                    .min_by_key(|(key, e)| (e.heat.at(epoch), e.last_used, **key));
                let Some((&old, entry)) = victim else {
                    ensure!(observations.is_some(), "Inkling GPU expert slots exhausted");
                    return Ok(false);
                };
                let old_count = entry.heat.at(epoch);
                if observations.is_some() && heat.count < old_count.saturating_add(16.max(old_count / 4)) {
                    return Ok(false);
                }
                self.entries.remove(&old).unwrap().slot
            }
        };
        let id = key.1;
        let offset = slot * self.stride;
        let data = (|| -> Result<_> {
            let gate_up = OnDevice::store(&self.buffer, offset, &weights.gate_up, weights.gate_up.expert(ckpt, id)?)?;
            let offset_down = offset + weights.gate_up.gpu_bytes().next_multiple_of(64);
            let down = OnDevice::store(&self.buffer, offset_down, &weights.down, weights.down.expert(ckpt, id)?)?;
            Ok((gate_up, down))
        })();
        match data {
            Ok((gate_up, down)) => {
                self.entries.insert(
                    key,
                    DeviceExpert {
                        gate_up,
                        down,
                        slot,
                        last_used: tick,
                        heat,
                    },
                );
                Ok(true)
            }
            Err(err) => {
                // A partially copied record cannot be dispatched. Its slot is reusable on the next call.
                self.free_slots.push(slot);
                Err(err)
            }
        }
    }
}

pub(super) fn expert_size(weights: &Experts) -> usize {
    weights.gate_up.gpu_bytes().next_multiple_of(64) + weights.down.gpu_bytes().next_multiple_of(64)
}

struct ExpertCache {
    pools: [Option<ExpertPool>; 2],
    generation: usize,
    visits: usize,
}

impl ExpertCache {
    fn new(byte_limit: usize, sizes: [usize; 2], layers: [usize; 2], top_k: usize) -> Result<Self> {
        let minimum = (sizes[0] + sizes[1]) * top_k;
        ensure!(
            byte_limit >= minimum,
            "Inkling expert cache needs at least {} MiB for one route per format",
            minimum.div_ceil(1 << 20)
        );
        let mut pools = [None, None];
        let total = layers[0] * sizes[0] + layers[1] * sizes[1];
        for i in 0..2 {
            if sizes[i] > 0 {
                let budget = sizes[i] * top_k + (byte_limit - minimum) * (layers[i] * sizes[i]) / total;
                pools[i] = Some(ExpertPool::new(budget, sizes[i])?);
            }
        }
        Ok(Self {
            pools,
            generation: 0,
            visits: 0,
        })
    }
}

const EXPERT_BATCH: usize = 64;
// Count token visits, so a prefill chunk advances the heat clock by its length.
const HEAT_EPOCH_VISITS: usize = 48 * 512;

#[derive(Hash, Eq, PartialEq)]
enum ProjectionKey {
    Dense([usize; 5], [usize; 5]),
    Controller([usize; 3], [usize; 3]),
    Router(usize),
    Experts { pool: usize, hidden: usize, intermediate: usize, count: usize, caps: [u32; 2] },
}
const MAX_PROJECTION_GRAPHS: usize = 256;

struct ProjectionGraph(sys::cudaGraphExec_t);

impl ProjectionGraph {
    fn capture(stream: &Stream, enqueue: impl FnOnce() -> Result<()>) -> Option<Self> {
        if unsafe { sys::cudaStreamBeginCapture(stream.0, sys::CAPTURE_MODE_THREAD_LOCAL) } != 0 {
            return None;
        }
        let recorded = enqueue();
        let mut graph = null_mut();
        let ended = unsafe { sys::cudaStreamEndCapture(stream.0, &mut graph) };
        if graph.is_null() {
            return None;
        }
        let mut exec = null_mut();
        let instantiated = recorded.is_ok() && ended == 0
            && unsafe { sys::cudaGraphInstantiate(&mut exec, graph, 0) } == 0;
        unsafe { sys::cudaGraphDestroy(graph) };
        instantiated.then_some(Self(exec))
    }

    fn launch(&self, stream: &Stream) -> Result<()> {
        check(unsafe { sys::cudaGraphLaunch(self.0, stream.0) }, "Inkling projection graph")
    }
}

impl Drop for ProjectionGraph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cudaGraphExecDestroy(self.0) };
        }
    }
}

pub(super) struct Compute {
    pub stream: Stream,
    gemm: *mut sys::blz_gemm,
    input: DevBuf,
    result: DevBuf,
    gate: DevBuf,
    glu: DevBuf,
    sum: DevBuf,
    scatter_ids: DevBuf,
    scatter_weights: DevBuf,
    transfer: HostMapped,
    next_id: DevBuf,
    cache: ExpertCache,
    cpu: Option<Arc<rayon::ThreadPool>>,
    cpu_routes: u64,
    projection_graphs: Option<HashMap<ProjectionKey, Option<ProjectionGraph>>>,
    grouped_experts: bool,
}

impl Compute {
    pub fn new(
        max_in: usize,
        max_out: usize,
        hidden: usize,
        intermediate: usize,
        expert_bytes: usize,
        expert_sizes: [usize; 2],
        layer_visits: [usize; 2],
        top_k: usize,
        prefill_chunk: usize,
        cpu: Option<Arc<rayon::ThreadPool>>,
    ) -> Result<Self> {
        let stream = Stream::new()?;
        let batch = prefill_chunk.min(EXPERT_BATCH);
        let input = DevBuf::new((max_in * 3).max(hidden * prefill_chunk * 3) * 2)?;
        let result = DevBuf::new(max_out.max((hidden * 3 + max_in) * prefill_chunk).max(hidden * top_k) * 2)?;
        let expert_rows = batch.max(top_k);
        let gate = DevBuf::new((intermediate * 2 * expert_rows).max(max_in) * 2)?;
        let glu = DevBuf::new(intermediate * expert_rows * 2)?;
        let sum = DevBuf::new(prefill_chunk * hidden * 4)?;
        let scatter_ids = DevBuf::new(batch * 4)?;
        let scatter_weights = DevBuf::new(batch * 4)?;
        let transfer = HostMapped::new(input.bytes().max(result.bytes()))?;
        let next_id = DevBuf::new(4)?;
        let cache = ExpertCache::new(expert_bytes, expert_sizes, layer_visits, top_k)?;
        let mut gemm = null_mut();
        check(unsafe { sys::blz_gemm_create(&mut gemm) }, "Inkling cuBLAS handle")?;
        Ok(Self {
            stream,
            gemm,
            input,
            result,
            gate,
            glu,
            sum,
            scatter_ids,
            scatter_weights,
            transfer,
            next_id,
            cache,
            cpu,
            cpu_routes: 0,
            projection_graphs: std::env::var_os("BLAZE_INKLING_GRAPHS")
                .filter(|value| value == "1")
                .map(|_| HashMap::new()),
            grouped_experts: std::env::var_os("BLAZE_INKLING_GROUPED_MOE").is_some_and(|value| value == "1"),
        })
    }

    fn stage_input(&mut self, values: &[u16]) -> Result<()> {
        // SAFETY: all source u16 values are initialized and the staging buffer remains live until stream sync.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) };
        self.transfer.host_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        self.copy_staged_input(bytes.len())
    }

    fn copy_staged_input(&self, bytes: usize) -> Result<()> {
        check(
            unsafe {
                sys::cudaMemcpyAsync(
                    self.input.ptr(),
                    self.transfer.host_ptr(),
                    bytes,
                    sys::MEMCPY_H2D,
                    self.stream.0,
                )
            },
            "Inkling projection input",
        )
    }

    fn queue_result<T: Copy>(&self, result: *const T, count: usize) -> Result<()> {
        let bytes = count * std::mem::size_of::<T>();
        ensure!(bytes <= self.transfer.bytes(), "Inkling readback exceeds staging capacity");
        check(
            unsafe {
                sys::cudaMemcpyAsync(
                    self.transfer.host_ptr().cast_mut(),
                    result.cast(),
                    bytes,
                    sys::MEMCPY_D2H,
                    self.stream.0,
                )
            },
            "Inkling projection readback",
        )
    }

    fn read_result<T: Copy>(&self, count: usize) -> Vec<T> {
        // The caller has synchronized a copy of these values into the initialized, page-aligned buffer.
        unsafe { std::slice::from_raw_parts(self.transfer.host_ptr().cast::<T>(), count) }.to_vec()
    }

    pub fn linear_bf16(&mut self, matrix: &Dense, x: &[u16], m: usize) -> Result<Vec<u16>> {
        let rows = matrix.rows * m;
        ensure!(
            x.len() == matrix.cols * m && x.len() * 2 <= self.input.bytes() && rows * 2 <= self.result.bytes(),
            "Inkling projection shape exceeds scratch"
        );
        let work = (|| -> Result<()> {
            self.stage_input(x)?;
            matrix.run(self, self.input.ptr(), self.result.ptr(), m)?;
            self.queue_result::<u16>(self.result.ptr(), rows)
        })();
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result(rows))
    }

    pub fn project_group<const N: usize>(&mut self, matrices: [&Dense; N], inputs: [&[&[f32]]; N]) -> Result<[Vec<u16>; N]> {
        let m = inputs[0].len();
        let cols = matrices[0].cols;
        let rows = matrices.map(|w| w.rows * m);
        let output_values: usize = rows.iter().sum();
        ensure!(
            matches!(N, 3 | 5) && m != 0
                && inputs.iter().all(|x| x.len() == m && x.iter().all(|r| r.len() == cols))
                && matrices.iter().all(|w| w.cols == cols)
                && m * cols * 6 <= self.input.bytes()
                && output_values * 2 <= self.result.bytes(),
            "Inkling grouped projection shape exceeds scratch"
        );

        let host = self.transfer.host_ptr().cast_mut().cast::<u16>();
        let mut end = 0;
        let mut offsets = [0; 5];
        for i in 0..N {
            if let Some(j) = (0..i).find(|&j| std::ptr::eq(inputs[i], inputs[j]) ||
                inputs[i].iter().zip(inputs[j]).all(|(&a, &b)| std::ptr::eq(a, b))) {
                offsets[i] = offsets[j];
                continue;
            }
            offsets[i] = end;
            let size = m * cols;
            ensure!((end + size) * 2 <= self.input.bytes(), "Inkling grouped inputs exceed scratch");
            // The page-aligned staging area is idle and initialized before this write.
            let dst = unsafe { std::slice::from_raw_parts_mut(host.add(end), size) };
            for (out, v) in dst.iter_mut().zip(inputs[i].iter().flat_map(|r| r.iter())) {
                *out = to_bf16(*v);
            }
            end += size;
        }
        let enqueue = |gpu: &Self| -> Result<()> {
            gpu.copy_staged_input(end * 2)?;
            let mut out = 0;
            for i in 0..N {
                // Dynamic convolution taps use the decode GEMV for every prompt row.
                if i >= 3 && m > 1 {
                    for row in 0..m {
                        matrices[i].run(gpu, gpu.input.at(offsets[i] + row * cols), gpu.result.at(out + row * matrices[i].rows), 1)?;
                    }
                } else {
                    matrices[i].run(gpu, gpu.input.at(offsets[i]), gpu.result.at(out), m)?;
                }
                out += rows[i];
            }
            gpu.queue_result::<u16>(gpu.result.ptr(), output_values)
        };
        let work = if m == 1 {
            let weights = std::array::from_fn(|i| matrices.get(i).map_or(0, |w| w.weight.ptr::<u8>() as usize));
            let key = ProjectionKey::Dense(weights, offsets);
            self.run_projection(key, enqueue)
        } else {
            enqueue(self)
        };
        let synced = self.stream.sync();
        work?;
        synced?;
        let mut offset = 0;
        Ok(rows.map(|len| {
            // Read the requested range of the completed D2H copy.
            let dst = unsafe { std::slice::from_raw_parts(host.add(offset), len) }.to_vec();
            offset += len;
            dst
        }))
    }

    fn run_projection(&mut self, key: ProjectionKey, enqueue: impl Fn(&Self) -> Result<()>) -> Result<()> {
        let Some(graphs) = &self.projection_graphs else { return enqueue(self) };
        if let Some(cached) = graphs.get(&key) {
            return cached.as_ref().map_or_else(|| enqueue(self), |graph| graph.launch(&self.stream));
        }
        if graphs.len() >= MAX_PROJECTION_GRAPHS {
            return enqueue(self);
        }
        let captured = ProjectionGraph::capture(&self.stream, || enqueue(self));
        let result = captured.as_ref().map_or_else(|| enqueue(self), |graph| graph.launch(&self.stream));
        self.projection_graphs.as_mut().unwrap().insert(key, captured);
        result
    }

    pub fn controller_gates<const N: usize>(
        &mut self, down: [&Dense; N], up: [&Dense; N], inputs: [&[f32]; N],
    ) -> Result<Vec<f32>> {
        let cols = down[0].cols;
        let rows = up[0].rows;
        ensure!(
            matches!(N, 1 | 3) && inputs.iter().all(|x| x.len() == cols)
                && down.iter().all(|w| w.cols == cols && w.rows == 64)
                && up.iter().all(|w| w.rows == rows && w.cols == 64)
                && N * cols * 2 <= self.input.bytes() && N * 64 * 2 <= self.gate.bytes()
                && N * rows * 4 <= self.result.bytes(), "Inkling controller exceeds scratch"
        );
        let host = self.transfer.host_ptr().cast_mut().cast::<u16>();
        let dst = unsafe { std::slice::from_raw_parts_mut(host, N * cols) };
        for (dst, input) in dst.chunks_exact_mut(cols).zip(inputs) {
            for (dst, &value) in dst.iter_mut().zip(input) { *dst = to_bf16(value); }
        }
        let pointers = |matrices: [&Dense; N]| std::array::from_fn(|i| {
            matrices.get(i).map_or(0, |w| w.weight.ptr::<u16>() as usize)
        });
        let key = ProjectionKey::Controller(pointers(down), pointers(up));
        let work = self.run_projection(key, |gpu| {
            gpu.copy_staged_input(N * cols * 2)?;
            for (i, w) in down.iter().enumerate() {
                w.run(gpu, gpu.input.at(i * cols), gpu.gate.at(i * 64), 1)?;
            }
            let weights = |i: usize| up.get(i).map_or(std::ptr::null(), |w| w.weight.ptr::<u16>());
            check(unsafe {
                sys::blz_inkling_controller_up(
                    gpu.gate.ptr(), weights(0), weights(1), weights(2), gpu.result.ptr(),
                    rows as i32, N as i32, gpu.stream.0,
                )
            }, "Inkling controller gates")?;
            gpu.queue_result::<f32>(gpu.result.ptr(), N * rows)
        });
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result(N * rows))
    }

    pub fn router_logits(&mut self, weights: &Dense, activations: &[Vec<f32>]) -> Result<Vec<f32>> {
        let tokens = activations.len();
        let inputs = tokens * weights.cols;
        let outputs = tokens * weights.rows;
        ensure!(tokens > 0 && activations.iter().all(|x| x.len() == weights.cols)
            && inputs * 2 <= self.input.bytes() && outputs * 4 <= self.result.bytes(),
            "Inkling router exceeds scratch");
        let host = self.transfer.host_ptr().cast_mut().cast::<u16>();
        let dst = unsafe { std::slice::from_raw_parts_mut(host, inputs) };
        for (dst, &value) in dst.iter_mut().zip(activations.iter().flatten()) {
            *dst = to_bf16(value);
        }
        let enqueue = |gpu: &Self| {
            gpu.copy_staged_input(inputs * 2)?;
            check(unsafe {
                sys::blz_inkling_router_logits(weights.weight.ptr(), gpu.input.ptr(), gpu.result.ptr(),
                    weights.rows as i32, weights.cols as i32, tokens as i32, gpu.stream.0)
            }, "Inkling MoE router")?;
            gpu.queue_result::<f32>(gpu.result.ptr(), outputs)
        };
        let work = if tokens == 1 {
            self.run_projection(ProjectionKey::Router(weights.weight.ptr::<u8>() as usize), enqueue)
        } else {
            enqueue(self)
        };
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result(outputs))
    }

    pub fn linear(&mut self, matrix: &Dense, x: &[f32]) -> Result<Vec<f32>> {
        let x: Vec<_> = x.iter().copied().map(to_bf16).collect();
        Ok(self.linear_bf16(matrix, &x, 1)?.into_iter().map(from_bf16).collect())
    }

    pub fn argmax(&mut self, matrix: &Dense, x: &[f32], vocab: usize) -> Result<u32> {
        let x: Vec<_> = x.iter().copied().map(to_bf16).collect();
        let work = (|| -> Result<()> {
            self.stage_input(&x)?;
            matrix.run(self, self.input.ptr(), self.result.ptr(), 1)?;
            check(
                unsafe {
                    sys::blz_argmax_bf16(
                        self.result.ptr(),
                        1,
                        vocab as i32,
                        matrix.rows as i64,
                        self.next_id.ptr(),
                        self.stream.0,
                    )
                },
                "Inkling argmax",
            )?;
            self.queue_result::<i32>(self.next_id.ptr(), 1)
        })();
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result::<i32>(1)[0] as u32)
    }

    pub fn cpu_routes(&self) -> u64 {
        self.cpu_routes
    }

    fn moe(
        &mut self,
        ckpt: &Checkpoint,
        weights: &Experts,
        layer: usize,
        x: &[f32],
        routes: &[(usize, f32)],
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<Vec<f32>> {
        let x_bf16: Vec<_> = x.iter().map(|&x| to_bf16(x)).collect();
        self.cache.generation += 1;
        let tick = self.cache.generation;
        self.cache.visits += 1;
        let epoch = self.cache.visits / HEAT_EPOCH_VISITS;
        let pool = self.cache.pools[weights.gate_up.packing as usize].as_mut().unwrap();
        // Touch hits first. Cache misses then evict only experts outside this route group.
        for &(id, _) in routes {
            if let Some(entry) = pool.entries.get_mut(&(layer, id)) {
                entry.last_used = tick;
            }
        }
        let mut cold = Vec::new();
        let mut resident = Vec::new();
        for &(id, weight) in routes {
            if pool.use_gpu(ckpt, weights, (layer, id), tick, epoch, self.cpu.is_some().then_some(1), true)? {
                resident.push((id, weight));
            } else {
                cold.push((Routed::read(ckpt, weights, id)?, x, 0, weight));
            }
        }
        if resident.is_empty() {
            let mut total = vec![0.0; x.len()];
            let out_cpu = run_cold(self.cpu.as_deref(), cold, weights, (gate_cap, up_cap), &mut self.cpu_routes);
            merge_cold(&mut total, x.len(), out_cpu);
            return Ok(total);
        }

        let work = (|| -> Result<()> {
            let activation_bytes = std::mem::size_of_val(x_bf16.as_slice());
            let offset = activation_bytes.next_multiple_of(std::mem::align_of::<f32>());
            let weights_end = offset + resident.len() * std::mem::size_of::<f32>();
            let grouped = self.grouped_experts && (2..=65535).contains(&resident.len());
            let matrix_bytes = std::mem::size_of::<sys::blz_inkling_matrix>();
            let gate_meta = weights_end.next_multiple_of(std::mem::align_of::<sys::blz_inkling_matrix>());
            let down_meta = gate_meta + resident.len() * matrix_bytes;
            let end = if grouped { down_meta + resident.len() * matrix_bytes } else { weights_end };
            ensure!(end <= self.input.bytes() && end <= self.transfer.bytes(), "Inkling MoE input exceeds staging");
            let pool = self.cache.pools[weights.gate_up.packing as usize].as_ref().unwrap();
            let staging = self.transfer.host_slice_mut();
            // SAFETY: these initialized activation values are copied as native-endian bytes.
            let activations = unsafe { std::slice::from_raw_parts(x_bf16.as_ptr().cast::<u8>(), activation_bytes) };
            staging[..activation_bytes].copy_from_slice(activations);
            for (dst, (_, weight)) in staging[offset..weights_end].chunks_exact_mut(4).zip(&resident) {
                dst.copy_from_slice(&weight.to_ne_bytes());
            }
            if grouped {
                // SAFETY: the pinned buffer is page-aligned, the descriptor offsets are aligned,
                // and the capacity was checked above.
                unsafe {
                    let ptr = staging.as_mut_ptr();
                    let gate = ptr.add(gate_meta).cast::<sys::blz_inkling_matrix>();
                    let down = ptr.add(down_meta).cast::<sys::blz_inkling_matrix>();
                    for (row, &(id, _)) in resident.iter().enumerate() {
                        let expert = &pool.entries[&(layer, id)];
                        gate.add(row).write(expert.gate_up.offsets());
                        down.add(row).write(expert.down.offsets());
                    }
                }
            }
            let enqueue = |gpu: &Self| -> Result<()> {
                let pool = gpu.cache.pools[weights.gate_up.packing as usize].as_ref().unwrap();
                gpu.copy_staged_input(end)?;
                if grouped {
                    let run: GroupedExpertGemv = match weights.gate_up.packing {
                        Packing::Nvfp4 => sys::blz_inkling_grouped_nvfp4,
                        Packing::Mxfp8 => sys::blz_inkling_grouped_mxfp8,
                    };
                    let group_count = resident.len() as i32;
                    let launch = |matrix: &PackedMatrix, meta, x, x_step, y| {
                        check(
                            unsafe {
                                run(
                                    x, x_step, pool.buffer.ptr(), gpu.input.at::<u8>(meta).cast(), y,
                                    group_count, matrix.rows as i32, matrix.cols as i32, gpu.stream.0,
                                )
                            },
                            "Inkling grouped expert GEMV",
                        )
                    };
                    launch(&weights.gate_up, gate_meta, gpu.input.ptr(), 0, gpu.gate.ptr())?;
                    check(
                        unsafe {
                            sys::blz_inkling_swiglu(
                                gpu.gate.ptr(),
                                gpu.glu.ptr(),
                                group_count * weights.down.cols as i32,
                                gate_cap,
                                up_cap,
                                gpu.stream.0,
                            )
                        },
                        "Inkling expert activation",
                    )?;
                    launch(&weights.down, down_meta, gpu.glu.ptr(), weights.down.cols as i64, gpu.result.ptr())?;
                } else {
                    for (row, &(id, _)) in resident.iter().enumerate() {
                        let expert = &pool.entries[&(layer, id)];
                        expert.gate_up.run(&pool.buffer, gpu.input.ptr(), gpu.gate.ptr(), 1, &gpu.stream)?;
                        check(
                            unsafe {
                                sys::blz_inkling_swiglu(
                                    gpu.gate.ptr(),
                                    gpu.glu.ptr(),
                                    weights.down.cols as i32,
                                    gate_cap,
                                    up_cap,
                                    gpu.stream.0,
                                )
                            },
                            "Inkling expert activation",
                        )?;
                        expert.down.run(&pool.buffer, gpu.glu.ptr(), gpu.result.at(row * x.len()), 1, &gpu.stream)?;
                    }
                }
                check(
                    unsafe {
                        sys::blz_inkling_reduce_experts(
                            gpu.result.ptr(),
                            gpu.input.at(offset / std::mem::size_of::<f32>()),
                            gpu.sum.ptr(),
                            x.len() as i32,
                            resident.len() as i32,
                            gpu.stream.0,
                        )
                    },
                    "Inkling expert reduction",
                )?;
                gpu.queue_result::<f32>(gpu.sum.ptr(), x.len())
            };
            if grouped {
                let key = ProjectionKey::Experts {
                    pool: pool.buffer.ptr::<u8>() as usize,
                    hidden: x.len(),
                    intermediate: weights.down.cols,
                    count: resident.len(),
                    caps: [gate_cap.to_bits(), up_cap.to_bits()],
                };
                self.run_projection(key, enqueue)
            } else {
                enqueue(self)
            }
        })();
        let out_cpu = run_cold(self.cpu.as_deref(), cold, weights, (gate_cap, up_cap), &mut self.cpu_routes);
        let synced = self.stream.sync();
        work?;
        synced?;
        let mut total = self.read_result(x.len());
        merge_cold(&mut total, x.len(), out_cpu);
        Ok(total)
    }

    pub fn moe_batch(
        &mut self,
        ckpt: &Checkpoint,
        weights: &Experts,
        layer: usize,
        activations: &[Vec<f32>],
        routes: &[Vec<(usize, f32)>],
        caps: (f32, f32),
        experts: usize,
    ) -> Result<Vec<f32>> {
        if activations.len() == 1 {
            return self.moe(ckpt, weights, layer, &activations[0], &routes[0], caps.0, caps.1);
        }
        let hidden = weights.gate_up.cols;
        let intermediate = weights.down.cols;
        let mut by_expert = vec![Vec::new(); experts];
        for (row, route) in routes.iter().enumerate() {
            for &(id, w) in route {
                by_expert[id].push((row as i32, w));
            }
        }
        let activations_bf16: Vec<Vec<u16>> = activations
            .iter()
            .map(|row| row.iter().copied().map(to_bf16).collect())
            .collect();
        self.sum.zero(&self.stream)?;
        self.cache.generation += 1;
        let tick = self.cache.generation;
        self.cache.visits += activations.len();
        let epoch = self.cache.visits / HEAT_EPOCH_VISITS;
        let pool = self.cache.pools[weights.gate_up.packing as usize].as_mut().unwrap();
        let mut input = Vec::with_capacity(hidden * EXPERT_BATCH);
        let mut ids = Vec::with_capacity(EXPERT_BATCH);
        let mut probs = Vec::with_capacity(EXPERT_BATCH);
        let mut cold = Vec::new();
        for (id, rows) in by_expert.iter().enumerate().filter(|(_, rows)| !rows.is_empty()) {
            if !pool.use_gpu(
                ckpt,
                weights,
                (layer, id),
                tick,
                epoch,
                self.cpu.is_some().then_some(rows.len()),
                false,
            )? {
                for &(row, weight) in rows {
                    cold.push((
                        Routed::read(ckpt, weights, id)?,
                        activations[row as usize].as_slice(),
                        row as usize,
                        weight,
                    ));
                }
                continue;
            }
            let e = &pool.entries[&(layer, id)];
            for batch in rows.chunks(EXPERT_BATCH) {
                input.clear();
                ids.clear();
                probs.clear();
                for &(row, weight) in batch {
                    input.extend_from_slice(&activations_bf16[row as usize]);
                    ids.push(row);
                    probs.push(weight);
                }
                self.input.upload(&input)?;
                self.scatter_ids.upload(&ids)?;
                self.scatter_weights.upload(&probs)?;
                let n = batch.len();
                e.gate_up
                    .run(&pool.buffer, self.input.ptr(), self.gate.ptr(), n, &self.stream)?;
                check(
                    unsafe {
                        sys::blz_inkling_swiglu(
                            self.gate.ptr(),
                            self.glu.ptr(),
                            (intermediate * n) as i32,
                            caps.0,
                            caps.1,
                            self.stream.0,
                        )
                    },
                    "Inkling batched activation",
                )?;
                e.down.run(&pool.buffer, self.glu.ptr(), self.result.ptr(), n, &self.stream)?;
                check(
                    unsafe {
                        sys::blz_inkling_expert_scatter(
                            self.result.ptr(),
                            self.sum.ptr(),
                            self.scatter_ids.ptr(),
                            self.scatter_weights.ptr(),
                            n as i32,
                            hidden as i32,
                            self.stream.0,
                        )
                    },
                    "Inkling batched expert reduction",
                )?;
                self.stream.sync()?;
            }
        }
        let out_cpu = run_cold(self.cpu.as_deref(), cold, weights, caps, &mut self.cpu_routes);
        self.stream.sync()?;
        let mut total = self.sum.download(activations.len() * hidden)?;
        merge_cold(&mut total, hidden, out_cpu);
        Ok(total)
    }
}

type ColdRoute<'a> = (Routed<'a>, &'a [f32], usize, f32);

fn run_cold(
    cpu: Option<&rayon::ThreadPool>,
    cold: Vec<ColdRoute<'_>>,
    weights: &Experts,
    caps: (f32, f32),
    count: &mut u64,
) -> Vec<(usize, f32, Vec<f32>)> {
    if cold.is_empty() {
        return Vec::new();
    }
    *count = count.saturating_add(cold.len() as u64);
    cpu.unwrap().install(|| {
        cold.into_par_iter()
            .map(|(expert, x, row, weight)| (row, weight, expert.run(weights, x, caps)))
            .collect()
    })
}

fn merge_cold(output: &mut [f32], hidden: usize, cold: Vec<(usize, f32, Vec<f32>)>) {
    for (row, weight, values) in cold {
        for (dst, src) in output[row * hidden..][..hidden].iter_mut().zip(values) {
            *dst += src * weight;
        }
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        self.projection_graphs.take();
        unsafe { sys::blz_gemm_destroy(self.gemm) };
    }
}

pub(super) fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub(super) fn to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

const HEAD_DIM: usize = 128;
const MAX_TILES: usize = 256;

// A power-of-two global scale keeps the E4M3 step usable for small blocks while covering
// both the model's clipped K heads and its RMS-normalized V heads.
const KV_GLOBAL_SCALE: f32 = 1.0 / 16.0;
const KV_CODE_BYTES: usize = HEAD_DIM / 2;
const KV_SCALE_BYTES: usize = HEAD_DIM / 16;

pub(super) struct Kv {
    storage: DevBuf,
    capacity: usize,
    heads: usize,
    dtype: KvDtype,
}

impl Kv {
    pub fn row_bytes(heads: usize, dtype: KvDtype) -> usize {
        heads * match dtype {
            KvDtype::Bf16 => HEAD_DIM * 2,
            _ => KV_CODE_BYTES + KV_SCALE_BYTES,
        }
    }

    pub fn new(capacity: usize, heads: usize, host: bool, dtype: KvDtype) -> Result<Self> {
        let bytes = capacity
            .checked_mul(Self::row_bytes(heads, dtype) * 2)
            .ok_or_else(|| anyhow::anyhow!("Inkling KV size overflows"))?;
        let alloc = if host { DevBuf::host_mapped } else { DevBuf::new };
        Ok(Self {
            storage: alloc(bytes)?,
            capacity,
            heads,
            dtype,
        })
    }

    fn pointers(&self, slot: usize) -> [*mut u8; 4] {
        if self.dtype == KvDtype::Bf16 {
            let row_bytes = Self::row_bytes(self.heads, self.dtype);
            return [
                self.storage.at(slot * row_bytes),
                self.storage.at((self.capacity + slot) * row_bytes),
                null_mut(),
                null_mut(),
            ];
        }
        let rows = self.capacity * self.heads;
        let code_pool = rows * KV_CODE_BYTES;
        let scale_pool = rows * KV_SCALE_BYTES;
        let row = slot * self.heads;
        let codes = row * KV_CODE_BYTES;
        let scales = row * KV_SCALE_BYTES;
        [
            self.storage.at(codes),
            self.storage.at(code_pool + codes),
            self.storage.at(2 * code_pool + scales),
            self.storage.at(2 * code_pool + scale_pool + scales),
        ]
    }

    fn store(
        &self,
        position: usize,
        key: *const u16,
        value: *const u16,
        zero: &DevBuf,
        scale: &DevBuf,
        stream: &Stream,
    ) -> Result<()> {
        let [keys, values, key_scales, value_scales] = self.pointers(position % self.capacity);
        let stride = (self.heads * HEAD_DIM) as i64;
        if self.dtype == KvDtype::Bf16 {
            return check(
                unsafe {
                    sys::blz_kv_store_dense(
                        key.cast(), value.cast(), stride, stride, zero.ptr(), sys::BLZ_I32,
                        1, self.heads as i32, HEAD_DIM as i32, sys::BLZ_KV_BF16,
                        keys, values, scale.ptr(), scale.ptr(), stream.0,
                    )
                },
                "Inkling BF16 KV store",
            );
        }
        check(
            unsafe {
                sys::blz_nvfp4_kv_store(
                    key.cast(),
                    value.cast(),
                    stride,
                    stride,
                    sys::BLZ_BF16,
                    zero.ptr(),
                    sys::BLZ_I32,
                    1,
                    self.heads as i32,
                    HEAD_DIM as i32,
                    keys,
                    values,
                    key_scales,
                    value_scales,
                    scale.ptr(),
                    scale.ptr(),
                    i32::from(self.dtype == KvDtype::Nvfp4FourOverSix),
                    stream.0,
                )
            },
            "Inkling NVFP4 KV store",
        )
    }
}

pub(super) struct AttentionScratch {
    input: DevBuf,
    staging: HostMapped,
    relative: DevBuf,
    zero: DevBuf,
    kv_scale: DevBuf,
    tiles: DevBuf,
    heads: usize,
    kv_heads: usize,
}

impl AttentionScratch {
    pub fn new(heads: usize, kv_heads: usize, extent: usize, relative: usize) -> Result<Self> {
        let values = heads * (HEAD_DIM * 2 + relative + 1) + kv_heads * HEAD_DIM * 2;
        Ok(Self {
            input: DevBuf::new(values * 2 + 6 * 16)?,
            staging: HostMapped::new(values * 2 + 6 * 16)?,
            relative: DevBuf::new(heads * extent * 2)?,
            zero: DevBuf::from_slice(&[0i32])?,
            kv_scale: DevBuf::from_slice(&[KV_GLOBAL_SCALE])?,
            tiles: DevBuf::new(heads * MAX_TILES * (HEAD_DIM + 2) * 4)?,
            heads,
            kv_heads,
        })
    }

    pub fn run(
        &mut self,
        gpu: &mut Compute,
        kv: &Kv,
        q: &[u16],
        gate: &[u16],
        sink: &[u16],
        relative: &[u16],
        key: &[u16],
        value: &[u16],
        bias: &Dense,
        output: &Dense,
        position: usize,
    ) -> Result<Vec<f32>> {
        let mut offset = 0usize;
        let staging = self.staging.host_slice_mut();
        let [q, gate, sink, relative, key, value] = [q, gate, sink, relative, key, value].map(|row| {
            offset = offset.next_multiple_of(16);
            let ptr = self.input.at::<u16>(offset / 2);
            // SAFETY: these initialized u16 values are read as native-endian bytes.
            let bytes = unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), std::mem::size_of_val(row)) };
            staging[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
            ptr
        });
        let work = (|| -> Result<()> {
            check(
                unsafe {
                    sys::cudaMemcpyAsync(
                        self.input.ptr(),
                        self.staging.host_ptr(),
                        offset,
                        sys::MEMCPY_H2D,
                        gpu.stream.0,
                    )
                },
                "Inkling attention inputs",
            )?;
            kv.store(position, key, value, &self.zero, &self.kv_scale, &gpu.stream)?;
            bias.run(gpu, relative, self.relative.ptr(), self.heads)?;
            let tiles = kv.capacity.min(position + 1).div_ceil(64).clamp(1, MAX_TILES);
            let [keys, values, key_scales, value_scales] = kv.pointers(0);
            check(
                unsafe {
                    sys::blz_inkling_attention(
                        q,
                        keys,
                        values,
                        key_scales,
                        value_scales,
                        self.relative.ptr(),
                        sink,
                        gate,
                        gpu.gate.ptr(),
                        self.tiles.ptr(),
                        self.heads as i32,
                        self.kv_heads as i32,
                        kv.capacity as i32,
                        position as i32,
                        bias.rows as i32,
                        tiles as i32,
                        KV_GLOBAL_SCALE,
                        gpu.stream.0,
                    )
                },
                "Inkling relative attention",
            )?;
            output.run(gpu, gpu.gate.ptr(), gpu.result.ptr(), 1)?;
            gpu.queue_result::<u16>(gpu.result.ptr(), output.rows)
        })();
        let synced = gpu.stream.sync();
        work?;
        synced?;
        Ok(gpu.read_result::<u16>(output.rows).into_iter().map(from_bf16).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "requires a CUDA GPU and AVX2/FMA"]
    fn router_logits_match_host_for_prefill_and_decode() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) { return; }
        crate::gpu::set_device(0).unwrap();
        let mut gpu = Compute::new(96, 128, 96, 32, 0, [0; 2], [0; 2], 8, 3, None).unwrap();
        gpu.projection_graphs = Some(HashMap::new());
        let values = |i: usize| to_bf16(((i * 31 % 113) as f32 - 50.0) / (3 + i % 19) as f32);
        let weights: Vec<_> = (0..11 * 96).map(values).collect();
        let dense = Dense::from_bf16(&weights, 11, 96).unwrap();
        let weights: Vec<_> = weights.into_iter().map(from_bf16).collect();
        let inputs: Vec<Vec<_>> = (0..3).map(|row| {
            (0..96).map(|i| from_bf16(values(i + 7 * row))).collect()
        }).collect();
        for batch in [&inputs[..], &inputs[1..2], &inputs[2..]] {
            let actual = gpu.router_logits(&dense, batch).unwrap();
            let expected = batch.iter().flat_map(|x| weights.chunks_exact(96).map(move |w| super::super::forward::dot(w, x)));
            assert!(actual.into_iter().map(f32::to_bits).eq(expected.map(f32::to_bits)));
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "requires a CUDA GPU and AVX2/FMA"]
    fn controller_up_matches_host_dots() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) { return; }
        crate::gpu::set_device(0).unwrap();
        let stream = Stream::new().unwrap();
        let rows = 67;
        let values = |size, salt| (0..size).map(|i: usize| {
            to_bf16((((i * 31 + salt) % 101) as f32 - 50.0) / (3 + i % 19) as f32)
        }).collect::<Vec<_>>();
        let x = values(3 * 64, 17);
        let weights: [_; 3] = std::array::from_fn(|i| values(rows * 64, 23 * i));
        let matrices = weights.each_ref().map(|w| DevBuf::from_slice(w).unwrap());
        let input = DevBuf::from_slice(&x).unwrap();
        let output = DevBuf::new(3 * rows * 4).unwrap();
        check(unsafe {
            sys::blz_inkling_controller_up(input.ptr(), matrices[0].ptr(), matrices[1].ptr(), matrices[2].ptr(),
                output.ptr(), rows as i32, 3, stream.0)
        }, "controller parity").unwrap();
        stream.sync().unwrap();
        let actual = output.download::<f32>(3 * rows).unwrap();
        for (i, row) in actual.chunks_exact(rows).enumerate() {
            let h: Vec<_> = x[i * 64..(i + 1) * 64].iter().copied().map(from_bf16).collect();
            for (w, value) in weights[i].chunks_exact(64).zip(row) {
                let w: Vec<_> = w.iter().copied().map(from_bf16).collect();
                assert_eq!(super::super::forward::dot(&w, &h).to_bits(), value.to_bits());
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn convolution_taps_share_key_value_inputs() {
        crate::gpu::set_device(0).unwrap();
        let mut gpu = Compute::new(32, 512, 32, 32, 0, [0; 2], [0; 2], 8, 3, None).unwrap();
        gpu.projection_graphs = Some(HashMap::new());
        let weights: Vec<_> = (0..32 * 32).map(|i| to_bf16((i % 37) as f32 * 0.031 - 0.5)).collect();
        let projection = Dense::from_bf16(&weights, 32, 32).unwrap();
        let taps = Dense::from_bf16(&weights[..8 * 32], 8, 32).unwrap();
        let rows: Vec<Vec<_>> = (0..5).map(|r| (0..32).map(|i| ((i+1)*(r+1)) as f32 * 0.017).collect()).collect();
        let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
        for (offset, m) in [(0, 3), (0, 1), (1, 1)] {
            let q = &refs[offset..offset+m];
            let k = &refs[offset+1..offset+m+1];
            let v = &refs[offset+2..offset+m+2];
            let expected_qkv = gpu.project_group([&projection; 3], [q, k, v]).unwrap();
            let expected_taps = [k, v].map(|inputs| inputs.iter().flat_map(|x| {
                gpu.linear(&taps, x).unwrap().into_iter().map(to_bf16)
            }).collect::<Vec<_>>());
            let [a, b, c, d, e] = gpu.project_group(
                [&projection, &projection, &projection, &taps, &taps], [q, k, v, k, v],
            ).unwrap();
            assert_eq!([a, b, c], expected_qkv);
            assert_eq!([d, e], expected_taps);
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn grouped_projection_graph_replays_with_new_input() {
        crate::gpu::set_device(0).unwrap();
        let mut gpu = Compute::new(32, 32, 32, 32, 0, [0; 2], [0; 2], 8, 1, None).unwrap();
        let matrix: Vec<_> = (0..32 * 32).map(|i| to_bf16(if i % 33 == 0 { 1.0 } else { 0.0 })).collect();
        let matrix = Dense::from_bf16(&matrix, 32, 32).unwrap();
        let expected: Vec<_> = [1.0, 2.0].into_iter().map(|value| {
            let x = [value; 32];
            gpu.project_group([&matrix; 3], [&[x.as_slice()]; 3]).unwrap()
        }).collect();
        gpu.projection_graphs = Some(HashMap::new());
        for (value, expected) in [1.0, 2.0].into_iter().zip(expected) {
            let x = [value; 32];
            assert_eq!(gpu.project_group([&matrix; 3], [&[x.as_slice()]; 3]).unwrap(), expected);
        }
        assert!(gpu.projection_graphs.as_ref().unwrap().values().any(Option::is_some));
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn expert_reduction_preserves_accumulation_order() {
        crate::gpu::set_device(0).unwrap();
        let stream = Stream::new().unwrap();
        let width = 131;
        let weights = [0.73, -0.14, 0.067, 0.88, -0.861, 0.11, 0.37, 0.15];
        let rows: Vec<_> = (0..width * weights.len())
            .map(|i| to_bf16(((i * 19 % 251) as f32 - 117.0) * 0.013))
            .collect();
        let expected: Vec<_> = (0..width)
            .map(|col| weights.iter().enumerate().fold(0.0, |sum, (i, &weight)| {
                from_bf16(rows[i * width + col]).mul_add(weight, sum)
            }))
            .collect();
        let input = DevBuf::from_slice(&rows).unwrap();
        let weights_dev = DevBuf::from_slice(&weights).unwrap();
        let output = DevBuf::new(width * 4).unwrap();
        check(
            unsafe {
                sys::blz_inkling_reduce_experts(
                    input.ptr(), weights_dev.ptr(), output.ptr(), width as i32, weights.len() as i32, stream.0,
                )
            },
            "test Inkling expert reduction",
        )
        .unwrap();
        stream.sync().unwrap();
        assert_eq!(output.download::<f32>(width).unwrap(), expected);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn relative_attention_ring_and_sink_match_scalar_reference() {
        crate::gpu::set_device(0).unwrap();
        let s = Stream::new().unwrap();
        let sign = |i: usize| if i % 2 == 0 { 1.0 } else { -1.0 };
        let query: [f32; HEAD_DIM] = std::array::from_fn(|i| sign(i) * 0.25);
        let q = DevBuf::from_slice(&query.map(to_bf16)).unwrap();
        let gate = DevBuf::from_slice(&[0u16; HEAD_DIM]).unwrap();
        let b = [to_bf16(3.0f32.ln()), to_bf16(2.0f32.ln())];
        let relative = DevBuf::from_slice(&b).unwrap();
        let sink_logit = to_bf16(2.0f32.ln());
        let sink = DevBuf::from_slice(&[sink_logit]).unwrap();
        let out = DevBuf::new(HEAD_DIM * 2).unwrap();
        let scratch = DevBuf::new((HEAD_DIM + 2) * 2 * 4).unwrap();
        let zero = DevBuf::from_slice(&[0i32]).unwrap();
        let scale = DevBuf::from_slice(&[KV_GLOBAL_SCALE]).unwrap();
        let keys_in = DevBuf::new(HEAD_DIM * 2).unwrap();
        let values_in = DevBuf::new(HEAD_DIM * 2).unwrap();
        let key = |i: usize, value: f32| sign(i) * value * if i / 16 % 2 == 0 { 0.5 } else { 1.0 };
        let value = |i: usize, n: f32| sign(i) * n * if i / 16 % 2 == 0 { 0.5 } else { 1.0 };
        for (dtype, host) in [KvDtype::Bf16, KvDtype::Nvfp4, KvDtype::Nvfp4FourOverSix]
            .into_iter()
            .flat_map(|dtype| [(dtype, false), (dtype, true)])
        {
            let kv = Kv::new(2, 1, host, dtype).unwrap();
            for (position, k, v) in [(3, 0.5, -10.0), (4, -0.125, 2.0), (5, 0.25, 5.0)] {
                let ks: [u16; HEAD_DIM] = std::array::from_fn(|i| to_bf16(key(i, k)));
                let vs: [u16; HEAD_DIM] = std::array::from_fn(|i| to_bf16(value(i, v)));
                keys_in.upload(&ks).unwrap();
                values_in.upload(&vs).unwrap();
                kv.store(position, keys_in.ptr(), values_in.ptr(), &zero, &scale, &s).unwrap();
                s.sync().unwrap();
            }
            let [keys, values, key_scales, value_scales] = kv.pointers(0);
            for tiles in [1, 2] {
                check(
                    unsafe {
                        sys::blz_inkling_attention(
                            q.ptr(),
                            keys,
                            values,
                            key_scales,
                            value_scales,
                            relative.ptr(),
                            sink.ptr(),
                            gate.ptr(),
                            out.ptr(),
                            scratch.ptr(),
                            1,
                            1,
                            2,
                            5,
                            2,
                            tiles,
                            KV_GLOBAL_SCALE,
                            s.0,
                        )
                    },
                    "test Inkling attention",
                )
                .unwrap();
                s.sync().unwrap();
                let weight = |k, bias| {
                    let dot: f32 = query.iter().enumerate().map(|(i, &q)| q * key(i, k)).sum();
                    (dot / (HEAD_DIM as f32).sqrt() + from_bf16(bias)).exp()
                };
                let (near, far, sink) = (weight(0.25, b[0]), weight(-0.125, b[1]), from_bf16(sink_logit).exp());
                let expected: Vec<_> = (0..HEAD_DIM)
                    .map(|i| to_bf16((value(i, 5.0) * near + value(i, 2.0) * far) / (2.0 * (near + far + sink))))
                    .collect();
                let actual = out.download::<u16>(HEAD_DIM).unwrap();
                if dtype == KvDtype::Nvfp4 {
                    for (a, e) in actual.into_iter().zip(expected) {
                        assert!((from_bf16(a) - from_bf16(e)).abs() < 0.15, "{dtype:?}, host={host}, tiles={tiles}");
                    }
                } else {
                    assert_eq!(actual, expected, "{dtype:?}, host={host}, tiles={tiles}");
                }
            }
        }
    }
}
