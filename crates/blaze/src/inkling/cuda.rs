use super::cpu::Routed;
use super::weights::{ExpertSlice, Experts, PackedMatrix, Packing};
use crate::gpu::{DevBuf, Stream, check};
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
        check(
            unsafe {
                sys::blz_gemm_bf16(
                    gpu.gemm,
                    x,
                    self.cols as i64,
                    self.weight.ptr(),
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
}

struct DeviceExpert {
    gate_up: OnDevice,
    down: OnDevice,
    slot: usize,
    last_used: usize,
}

struct ExpertPool {
    buffer: DevBuf,
    stride: usize,
    next_slot: usize,
    entries: HashMap<(usize, usize), DeviceExpert>,
    hits: HashMap<(usize, usize), usize>,
}

impl ExpertPool {
    fn new(bytes: usize, stride: usize) -> Result<Self> {
        Ok(Self {
            buffer: DevBuf::new(bytes / stride * stride)?,
            stride,
            next_slot: 0,
            entries: HashMap::new(),
            hits: HashMap::new(),
        })
    }

    fn use_gpu(
        &mut self,
        ckpt: &Checkpoint,
        weights: &Experts,
        layer: usize,
        id: usize,
        tick: usize,
        observations: usize,
        cpu_enabled: bool,
    ) -> Result<bool> {
        if cpu_enabled {
            let key = (layer, id);
            let count = self.hits.entry(key).or_default();
            *count = count.saturating_add(observations);
            let count = *count;
            if !self.entries.contains_key(&key) {
                if count < 2 {
                    return Ok(false);
                }
                if self.next_slot >= self.buffer.bytes() / self.stride {
                    // Changing a resident expert costs PCIe traffic. Only replace a less-used slot when
                    // this expert has accumulated a clear lead, outside the active route group.
                    let replace = self
                        .entries
                        .iter()
                        .filter(|(_, e)| e.last_used != tick)
                        .min_by_key(|(key, e)| (self.hits.get(key).copied().unwrap_or(0), e.last_used))
                        .map(|(&key, _)| key);
                    let Some(old) = replace else {
                        return Ok(false);
                    };
                    if count < self.hits[&old].saturating_add(16) {
                        return Ok(false);
                    }
                    let victim = self.entries.get_mut(&old).unwrap();
                    victim.last_used = 0;
                }
            }
        }
        self.prepare(ckpt, weights, layer, id, tick)?;
        Ok(true)
    }

    fn prepare(&mut self, ckpt: &Checkpoint, weights: &Experts, layer: usize, id: usize, tick: usize) -> Result<()> {
        let key = (layer, id);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = tick;
            return Ok(());
        }
        let slot = if self.next_slot < self.buffer.bytes() / self.stride {
            let slot = self.next_slot;
            self.next_slot += 1;
            slot
        } else {
            let (&oldest, _) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .expect("non-empty expert pool");
            self.entries.remove(&oldest).unwrap().slot
        };
        let offset = slot * self.stride;
        let gate_up = OnDevice::store(&self.buffer, offset, &weights.gate_up, weights.gate_up.expert(ckpt, id)?)?;
        let offset_down = offset + weights.gate_up.gpu_bytes().next_multiple_of(64);
        let down = OnDevice::store(&self.buffer, offset_down, &weights.down, weights.down.expert(ckpt, id)?)?;
        self.entries.insert(
            key,
            DeviceExpert {
                gate_up,
                down,
                slot,
                last_used: tick,
            },
        );
        Ok(())
    }
}

pub(super) fn expert_size(weights: &Experts) -> usize {
    weights.gate_up.gpu_bytes().next_multiple_of(64) + weights.down.gpu_bytes().next_multiple_of(64)
}

struct ExpertCache {
    pools: [Option<ExpertPool>; 2],
    generation: usize,
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
        Ok(Self { pools, generation: 0 })
    }
}

const EXPERT_BATCH: usize = 64;

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
    cache: ExpertCache,
    cpu: Option<Arc<rayon::ThreadPool>>,
    cpu_routes: u64,
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
        let input = DevBuf::new(max_in.max(hidden * prefill_chunk) * 2)?;
        let result = DevBuf::new(max_out.max((hidden * 3 + max_in) * prefill_chunk) * 2)?;
        let gate = DevBuf::new((intermediate * 2 * batch).max(max_in) * 2)?;
        let glu = DevBuf::new(intermediate * batch * 2)?;
        let sum = DevBuf::new(prefill_chunk * hidden * 4)?;
        let scatter_ids = DevBuf::new(batch * 4)?;
        let scatter_weights = DevBuf::new(batch * 4)?;
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
            cache,
            cpu,
            cpu_routes: 0,
        })
    }

    pub fn linear_bf16(&self, matrix: &Dense, x: &[u16], m: usize) -> Result<Vec<u16>> {
        ensure!(
            x.len() == matrix.cols * m && matrix.rows * m * 2 <= self.result.bytes(),
            "Inkling projection shape exceeds scratch"
        );
        self.input.upload(x)?;
        matrix.run(self, self.input.ptr(), self.result.ptr(), m)?;
        self.stream.sync()?;
        self.result.download(matrix.rows * m)
    }

    pub fn linear_rows(&self, matrix: &Dense, rows: &[impl AsRef<[f32]>]) -> Result<Vec<u16>> {
        let x: Vec<_> = rows.iter().flat_map(|r| r.as_ref().iter().copied().map(to_bf16)).collect();
        self.linear_bf16(matrix, &x, rows.len())
    }

    pub fn linear(&self, matrix: &Dense, x: &[f32]) -> Result<Vec<f32>> {
        let x: Vec<_> = x.iter().map(|&x| to_bf16(x)).collect();
        Ok(self.linear_bf16(matrix, &x, 1)?.into_iter().map(from_bf16).collect())
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
            if pool.use_gpu(ckpt, weights, layer, id, tick, 1, self.cpu.is_some())? {
                resident.push((id, weight));
            } else {
                cold.push((Routed::read(ckpt, weights, id)?, x, 0, weight));
            }
        }
        self.input.upload(&x_bf16)?;
        self.sum.zero(&self.stream)?;
        for (id, weight) in resident {
            let e = &pool.entries[&(layer, id)];
            e.gate_up
                .run(&pool.buffer, self.input.ptr(), self.gate.ptr(), 1, &self.stream)?;
            check(
                unsafe {
                    sys::blz_inkling_swiglu(
                        self.gate.ptr(),
                        self.glu.ptr(),
                        weights.down.cols as i32,
                        gate_cap,
                        up_cap,
                        self.stream.0,
                    )
                },
                "Inkling expert activation",
            )?;
            e.down.run(&pool.buffer, self.glu.ptr(), self.result.ptr(), 1, &self.stream)?;
            check(
                unsafe {
                    sys::blz_inkling_expert_accumulate(self.result.ptr(), self.sum.ptr(), x.len() as i32, weight, self.stream.0)
                },
                "Inkling expert reduction",
            )?;
        }
        let out_cpu = run_cold(self.cpu.as_deref(), cold, weights, (gate_cap, up_cap), &mut self.cpu_routes);
        self.stream.sync()?;
        let mut total = self.sum.download(x.len())?;
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
        let pool = self.cache.pools[weights.gate_up.packing as usize].as_mut().unwrap();
        let mut input = Vec::with_capacity(hidden * EXPERT_BATCH);
        let mut ids = Vec::with_capacity(EXPERT_BATCH);
        let mut probs = Vec::with_capacity(EXPERT_BATCH);
        let mut cold = Vec::new();
        for (id, rows) in by_expert.iter().enumerate().filter(|(_, rows)| !rows.is_empty()) {
            if !pool.use_gpu(ckpt, weights, layer, id, tick, rows.len(), self.cpu.is_some())? {
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

pub(super) struct Kv {
    keys: DevBuf,
    values: DevBuf,
    capacity: usize,
    row_bytes: usize,
    q8: bool,
}

impl Kv {
    pub fn row_bytes(heads: usize, q8: bool) -> usize {
        heads * if q8 { HEAD_DIM + 4 } else { HEAD_DIM * 2 }
    }

    pub fn new(capacity: usize, heads: usize, host: bool, q8: bool) -> Result<Self> {
        let row_bytes = Self::row_bytes(heads, q8);
        let bytes = capacity * row_bytes;
        let alloc = if host { DevBuf::host_mapped } else { DevBuf::new };
        Ok(Self {
            keys: alloc(bytes)?,
            values: alloc(bytes)?,
            capacity,
            row_bytes,
            q8,
        })
    }

    pub fn store(&self, position: usize, key: &[u16], value: &[u16]) -> Result<()> {
        let off = (position % self.capacity) * self.row_bytes;
        if self.q8 {
            self.keys.upload_at(off, &quantize_kv(key))?;
            self.values.upload_at(off, &quantize_kv(value))
        } else {
            self.keys.upload_at(off, key)?;
            self.values.upload_at(off, value)
        }
    }
}

fn quantize_kv(row: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(row.len() / HEAD_DIM * (HEAD_DIM + 4));
    for head in row.chunks_exact(HEAD_DIM) {
        let max = head.iter().map(|&x| from_bf16(x).abs()).fold(0.0, f32::max);
        let scale = max / 127.0;
        let inv = if scale > 0.0 { scale.recip() } else { 0.0 };
        bytes.extend(head.iter().map(|&x| (from_bf16(x) * inv).round_ties_even() as i8 as u8));
        bytes.extend_from_slice(&scale.to_ne_bytes());
    }
    bytes
}

pub(super) struct AttentionScratch {
    query: DevBuf,
    gate: DevBuf,
    sink: DevBuf,
    relative: DevBuf,
    rel_input: DevBuf,
    tiles: DevBuf,
    heads: usize,
    kv_heads: usize,
}

impl AttentionScratch {
    pub fn new(heads: usize, kv_heads: usize, extent: usize, relative: usize) -> Result<Self> {
        Ok(Self {
            query: DevBuf::new(heads * HEAD_DIM * 2)?,
            gate: DevBuf::new(heads * HEAD_DIM * 2)?,
            sink: DevBuf::new(heads * 2)?,
            relative: DevBuf::new(heads * extent * 2)?,
            rel_input: DevBuf::new(heads * relative * 2)?,
            tiles: DevBuf::new(heads * MAX_TILES * (HEAD_DIM + 2) * 4)?,
            heads,
            kv_heads,
        })
    }

    pub fn run(
        &self,
        gpu: &Compute,
        kv: &Kv,
        q: &[u16],
        gate: &[u16],
        sink: &[u16],
        relative: &[u16],
        bias: &Dense,
        output: &Dense,
        position: usize,
    ) -> Result<Vec<f32>> {
        self.query.upload(q)?;
        self.gate.upload(gate)?;
        self.sink.upload(sink)?;
        self.rel_input.upload(relative)?;
        bias.run(gpu, self.rel_input.ptr(), self.relative.ptr(), self.heads)?;
        let tiles = kv.capacity.min(position + 1).div_ceil(64).clamp(1, MAX_TILES);
        check(
            unsafe {
                sys::blz_inkling_attention(
                    self.query.ptr(),
                    kv.keys.ptr(),
                    kv.values.ptr(),
                    self.relative.ptr(),
                    self.sink.ptr(),
                    self.gate.ptr(),
                    gpu.gate.ptr(),
                    self.tiles.ptr(),
                    self.heads as i32,
                    self.kv_heads as i32,
                    kv.capacity as i32,
                    position as i32,
                    bias.rows as i32,
                    tiles as i32,
                    i32::from(kv.q8),
                    gpu.stream.0,
                )
            },
            "Inkling relative attention",
        )?;
        output.run(gpu, gpu.gate.ptr(), gpu.result.ptr(), 1)?;
        gpu.stream.sync()?;
        Ok(gpu.result.download::<u16>(output.rows)?.into_iter().map(from_bf16).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn relative_attention_ring_and_sink_match_scalar_reference() {
        crate::gpu::set_device(0).unwrap();
        let s = Stream::new().unwrap();
        let q = DevBuf::from_slice(&[0u16; HEAD_DIM]).unwrap();
        let gate = DevBuf::from_slice(&[0u16; HEAD_DIM]).unwrap();
        let b = [to_bf16(3.0f32.ln()), to_bf16(2.0f32.ln())];
        let relative = DevBuf::from_slice(&b).unwrap();
        let sink_logit = to_bf16(2.0f32.ln());
        let sink = DevBuf::from_slice(&[sink_logit]).unwrap();
        let out = DevBuf::new(HEAD_DIM * 2).unwrap();
        let scratch = DevBuf::new((HEAD_DIM + 2) * 2 * 4).unwrap();
        for q8 in [false, true] {
            let kv = Kv::new(2, 1, q8, q8).unwrap();
            for (position, value) in [(3, -10.0), (4, 2.0), (5, 5.0)] {
                kv.store(position, &[0u16; HEAD_DIM], &[to_bf16(value); HEAD_DIM]).unwrap();
            }
            check(
                unsafe {
                    sys::blz_inkling_attention(
                        q.ptr(),
                        kv.keys.ptr(),
                        kv.values.ptr(),
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
                        2,
                        i32::from(q8),
                        s.0,
                    )
                },
                "test Inkling attention",
            )
            .unwrap();
            s.sync().unwrap();
            let (near, far, sink) = (from_bf16(b[0]).exp(), from_bf16(b[1]).exp(), from_bf16(sink_logit).exp());
            let expected = to_bf16((5.0 * near + 2.0 * far) / (2.0 * (near + far + sink)));
            assert_eq!(out.download::<u16>(HEAD_DIM).unwrap(), vec![expected; HEAD_DIM]);
        }
    }
}
