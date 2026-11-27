use super::KvDtype;
use crate::expert_runtime::MoeEngine;
use crate::graph::GraphCache;
#[cfg(test)]
use crate::graph::{Capture, CudaGraph};
use crate::gpu::{DevBuf, HostMapped, Stream, check};
use crate::matmul::{Bf16Kernel, Matmul};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use blaze_sys as sys;
#[cfg(test)]
use std::borrow::Cow;
#[cfg(test)]
use std::collections::HashMap;
use std::ptr::null_mut;

pub(super) struct Dense {
    weight: DevBuf,
    pub rows: usize,
    pub(super) cols: usize,
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
        self.run_prefix(gpu, x, out, m, self.rows, m == 1)
    }

    fn run_decode_rows(&self, gpu: &Compute, x: *const u16, out: *mut u16, count: usize) -> Result<()> {
        let max_rows = if self.rows > 4096 && crate::tune::wide_fits(1, self.cols, self.cols) {
            sys::BLZ_GEMV_WIDE_MAX_ROWS.min(sys::BLZ_GEMV_WIDE_MAX_SMEM / (self.cols * 2))
        } else if self.rows <= 4096 && crate::tune::skinny_fits(1, self.cols, self.cols) {
            sys::BLZ_SKINNY_EX_MAX_ROWS
        } else { 1 };
        let max_rows = if (x as usize).is_multiple_of(16) { max_rows } else { 1 };
        for first in (0..count).step_by(max_rows) {
            let batch = (count - first).min(max_rows);
            let input = x.wrapping_add(first * self.cols);
            self.run_prefix(gpu, input, out.wrapping_add(first * self.rows), batch, self.rows, true)?;
        }
        Ok(())
    }

    fn run_prefix(&self, gpu: &Compute, x: *const u16, out: *mut u16, m: usize, rows: usize, decode: bool) -> Result<()> {
        let product = Matmul {
            input: x, output: out, rows: m, columns: rows, inner: self.cols,
            input_stride: self.cols, output_stride: rows,
        };
        let kernel = if decode && (x as usize).is_multiple_of(16)
            && rows > 4096 && crate::tune::wide_fits(m, self.cols, self.cols)
        {
            Bf16Kernel::Wide
        } else if decode && (x as usize).is_multiple_of(16)
            && rows <= 4096 && crate::tune::skinny_fits(m, self.cols, self.cols)
        {
            Bf16Kernel::Skinny(0)
        } else {
            Bf16Kernel::Cublas
        };
        unsafe { product.bf16(gpu.gemm, self.weight.ptr(), kernel, &gpu.stream) }
    }
}

#[cfg(test)]
struct RecurrentBatch {
    #[cfg(test)]
    staging: HostMapped,
    projected_input: DevBuf,
    bank_staging: HostMapped,
    message_bank: DevBuf,
}

#[cfg(test)]
impl RecurrentBatch {
    fn new(rows: usize, width: usize) -> Result<Self> {
        Ok(Self {
            #[cfg(test)]
            staging: HostMapped::new(rows * 3 * width * 8)?,
            projected_input: DevBuf::new(rows * 3 * width * 2)?,
            bank_staging: HostMapped::new(rows * 2 * width * 4)?,
            message_bank: DevBuf::new(rows * 2 * width * 4)?,
        })
    }
}

#[cfg(test)]
pub(super) enum RecurrentMessages<'a> {
    Mixed(&'a [[Vec<f32>; 3]]),
    Bank {
        upload: Option<&'a [&'a [Vec<f32>]]>,
        weights: &'a [[[f32; 2]; 3]],
    },
}

#[cfg(test)]
#[derive(Hash, Eq, PartialEq)]
enum ProjectionKey {
    Dense([usize; 5], [usize; 5]),
    Controller([usize; 3], [usize; 3], usize),
    Router(usize),
    GatedRouter { weights: [usize; 4], eps: u32, tokens: usize },
}
#[cfg(test)]
const MAX_PROJECTION_GRAPHS: usize = 256;

pub(super) struct Compute {
    pub stream: Stream,
    gemm: *mut sys::blz_gemm,
    input: DevBuf,
    result: DevBuf,
    gate: DevBuf,
    transfer: HostMapped,
    next_id: DevBuf,
    controller_graphs: GraphCache<[usize; 8]>,
    experts: Option<MoeEngine>,
    #[cfg(test)]
    recurrent_batch: Option<RecurrentBatch>,
    #[cfg(test)]
    projection_graphs: Option<HashMap<ProjectionKey, Option<CudaGraph>>>,
    tiled_skinny: bool,
}

impl Compute {
    pub fn new(
        max_in: usize,
        max_out: usize,
        hidden: usize,
        prefill_chunk: usize,
    ) -> Result<Self> {
        let stream = Stream::new()?;
        let input = DevBuf::new((max_in * 3).max(hidden * prefill_chunk * 3) * 2)?;
        let result = DevBuf::new(max_out.max((hidden * 3 + max_in) * prefill_chunk) * 2)?;
        let gate = DevBuf::new((prefill_chunk * 3 * 128).max(hidden * 2).max(max_in * 2))?;
        let transfer = HostMapped::new(input.bytes().max(result.bytes()))?;
        let next_id = DevBuf::new(prefill_chunk * size_of::<i32>())?;
        #[cfg(test)]
        let recurrent_batch = (prefill_chunk >= 12)
            .then(|| RecurrentBatch::new(prefill_chunk, hidden)).transpose()?;
        let mut gemm = null_mut();
        check(unsafe { sys::blz_gemm_create(&mut gemm) }, "Inkling cuBLAS handle")?;
        Ok(Self {
            stream,
            gemm,
            input,
            result,
            gate,
            transfer,
            next_id,
            controller_graphs: GraphCache::new(96),
            experts: None,
            #[cfg(test)]
            recurrent_batch,
            #[cfg(test)]
            projection_graphs: std::env::var_os("BLAZE_INKLING_GRAPHS")
                .filter(|value| value == "1")
                .map(|_| HashMap::new()),
            tiled_skinny: prefill_chunk >= 256,
        })
    }

    fn stage_input(&mut self, values: &[u16]) -> Result<()> {
        // SAFETY: all source u16 values are initialized and the staging buffer remains live until stream sync.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) };
        self.transfer.host_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        self.copy_staged_input(bytes.len())
    }

    pub fn attach_expert_engine(&mut self, engine: MoeEngine) {
        self.experts = Some(engine);
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

    #[cfg(test)]
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

    #[cfg(test)]
    pub fn project_group<const N: usize>(&mut self, matrices: [&Dense; N], inputs: [&[&[f32]]; N]) -> Result<[Vec<u16>; N]> {
        self.project_rows(matrices, inputs, false)
    }

    #[cfg(test)]
    pub fn project_rows<const N: usize>(&mut self, matrices: [&Dense; N], inputs: [&[&[f32]]; N], verifying: bool) -> Result<[Vec<u16>; N]> {
        let (offsets, end) = self.stage_projections(matrices, inputs)?;
        let m = inputs[0].len();
        let rows = matrices.map(|w| w.rows * m);
        let output_values: usize = rows.iter().sum();
        let enqueue = |gpu: &Self| -> Result<()> {
            gpu.copy_staged_input(end * 2)?;
            gpu.enqueue_projections(matrices, &gpu.input, offsets, m, verifying, [0; N])?;
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
        let host = self.transfer.host_ptr().cast::<u16>();
        Ok(rows.map(|len| {
            let dst = unsafe { std::slice::from_raw_parts(host.add(offset), len) }.to_vec();
            offset += len;
            dst
        }))
    }

    #[cfg(test)]
    fn stage_projections<const N: usize>(&mut self, matrices: [&Dense; N], inputs: [&[&[f32]]; N]) -> Result<([usize; 5], usize)> {
        let m = inputs[0].len();
        let cols = matrices[0].cols;
        let output_values: usize = matrices.iter().map(|w| w.rows * m).sum();
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
        Ok((offsets, end))
    }

    fn enqueue_projections<const N: usize>(
        &self, matrices: [&Dense; N], input: &DevBuf, offsets: [usize; 5], rows: usize,
        verifying: bool, prefix: [usize; N],
    ) -> Result<[*mut u16; N]> {
        let total: usize = matrices.iter().zip(prefix).map(|(w, p)| w.rows * rows + p).sum();
        ensure!(total * 2 <= self.result.bytes(), "Inkling projection output exceeds scratch");
        let mut out = 0;
        let mut results = [null_mut(); N];
        for i in 0..N {
            out += prefix[i];
            let result = self.result.at(out);
            results[i] = result;
            if (verifying || i >= 3) && rows > 1 {
                matrices[i].run_decode_rows(self, input.at(offsets[i]), result, rows)?;
            } else {
                matrices[i].run(self, input.at(offsets[i]), result, rows)?;
            }
            out += matrices[i].rows * rows;
        }
        Ok(results)
    }

    #[cfg(test)]
    fn run_projection(&mut self, key: ProjectionKey, enqueue: impl Fn(&Self) -> Result<()>) -> Result<()> {
        let Some(graphs) = &self.projection_graphs else { return enqueue(self) };
        if let Some(cached) = graphs.get(&key) {
            return cached.as_ref().map_or_else(|| enqueue(self), |graph| graph.launch(&self.stream));
        }
        if graphs.len() >= MAX_PROJECTION_GRAPHS {
            return enqueue(self);
        }
        let captured = Capture::record(&self.stream, || enqueue(self))
            .and_then(Capture::instantiate).ok();
        let result = captured.as_ref().map_or_else(|| enqueue(self), |graph| graph.launch(&self.stream));
        self.projection_graphs.as_mut().unwrap().insert(key, captured);
        result
    }

    #[cfg(test)]
    pub fn controller_gates<const N: usize>(
        &mut self, down: [&Dense; N], up: [&Dense; N], inputs: [&[f32]; N],
    ) -> Result<Vec<f32>> {
        self.controller_gates_batch(down, up, &[inputs])
    }

    #[cfg(test)]
    pub fn controller_batch_limit<const N: usize>(&self, width: usize) -> usize {
        (self.input.bytes() / (N * width * 4))
            .min(self.result.bytes() / (N * width * 8))
            .min(self.gate.bytes() / (N * 64 * 2))
            .min(65535)
    }

    #[cfg(test)]
    pub fn controller_gates_batch<const N: usize>(
        &mut self, down: [&Dense; N], up: [&Dense; N], inputs: &[[&[f32]; N]],
    ) -> Result<Vec<f32>> {
        let cols = down[0].cols;
        let rows = up[0].rows;
        let count = inputs.len();
        let num_inputs = count * N * cols;
        let num_outputs = count * N * rows;
        ensure!(
            matches!(N, 1 | 3) && count > 0 && inputs.iter().flatten().all(|x| x.len() == cols)
                && down.iter().all(|w| w.cols == cols && w.rows == 64)
                && up.iter().all(|w| w.rows == rows && w.cols == 64)
                && num_inputs * 2 <= self.input.bytes() && count * N * 64 * 2 <= self.gate.bytes()
                && num_outputs * 4 <= self.result.bytes(), "Inkling controller exceeds scratch"
        );
        let host = self.transfer.host_ptr().cast_mut().cast::<u16>();
        let dst = unsafe { std::slice::from_raw_parts_mut(host, num_inputs) };
        for (dst, &input) in dst.chunks_exact_mut(cols).zip(inputs.iter().flatten()) {
            for (dst, &value) in dst.iter_mut().zip(input) {
                *dst = to_bf16(value);
            }
        }
        let pointers = |matrices: [&Dense; N]| std::array::from_fn(|i| {
            matrices.get(i).map_or(0, |w| w.weight.ptr::<u16>() as usize)
        });
        let key = ProjectionKey::Controller(pointers(down), pointers(up), count);
        let enqueue = |gpu: &Self| {
            gpu.copy_staged_input(num_inputs * 2)?;
            for first in (0..count).step_by(sys::BLZ_SKINNY_EX_MAX_ROWS) {
                let batch = (count - first).min(sys::BLZ_SKINNY_EX_MAX_ROWS);
                for (i, w) in down.iter().enumerate() {
                    let index = first * N + i;
                    if batch > 1 && crate::tune::skinny_fits(batch, cols, N * cols) {
                        check(unsafe {
                            sys::blz_skinny_gemm_ex(gpu.input.at::<u16>(index * cols).cast(), (N * cols) as i64,
                                w.weight.ptr(), gpu.gate.at::<u16>(index * 64).cast(), (N * 64) as i64, batch as i32,
                                64, cols as i32, sys::BLZ_BF16, gpu.stream.0)
                        }, "Inkling batched loop projection")?;
                    } else {
                        for row in 0..batch {
                            let index = index + row * N;
                            w.run(gpu, gpu.input.at(index * cols), gpu.gate.at(index * 64), 1)?;
                        }
                    }
                }
            }
            let weights = |i: usize| up.get(i).map_or(std::ptr::null(), |w| w.weight.ptr::<u16>());
            check(unsafe {
                sys::blz_inkling_controller_up(
                    gpu.gate.ptr(), weights(0), weights(1), weights(2), gpu.result.ptr(),
                    rows as i32, N as i32, count as i32, gpu.stream.0,
                )
            }, "Inkling controller gates")?;
            gpu.queue_result::<f32>(gpu.result.ptr(), num_outputs)
        };
        let work = if count == 1 { self.run_projection(key, enqueue) } else { enqueue(self) };
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result(num_outputs))
    }

    #[cfg(test)]
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

    #[cfg(test)]
    pub fn gate_batch_limit(&self, width: usize) -> usize {
        (self.input.bytes() / (12 * width)).min(self.result.bytes() / (8 * width))
            .min(self.gate.bytes() / 128).min(65535)
    }

    #[cfg(test)]
    fn recurrent_batch_limit(&self, width: usize) -> usize {
        self.controller_batch_limit::<3>(width).min(self.input.bytes() / (36 * width))
    }

    #[cfg(test)]
    fn enqueue_recurrent_attention(
        &self, down: [&Dense; 3], up: [&Dense; 3], norms: &DevBuf, width: usize, rows: usize,
        eps: f32, host: *const f32, bank: Option<*const f32>,
        projected: Option<(*mut u16, i32)>,
    ) -> Result<*mut u16> {
        let values = rows * 3 * width;
        check(unsafe {
            sys::cudaMemcpyAsync(self.input.ptr(), host.cast(), values * if bank.is_some() { 4 } else { 8 },
                sys::MEMCPY_H2D, self.stream.0)
        }, "Inkling recurrent attention upload")?;
        let states = self.input.ptr::<f32>();
        let messages = self.input.at::<f32>(values);
        let normalized = self.input.at::<u16>(4 * values);
        if let Some(raw) = bank {
            check(unsafe {
                sys::cudaMemcpyAsync(self.result.ptr(), host.add(values).cast(), rows * 6 * 4,
                    sys::MEMCPY_H2D, self.stream.0)
            }, "Inkling recurrent message weights")?;
            check(unsafe {
                sys::blz_inkling_controller_message_mix(raw, self.result.ptr(), messages, width as i32, rows as i32, self.stream.0)
            }, "Inkling recurrent message mix")?;
        }
        check(unsafe {
            sys::blz_inkling_controller_inputs(states, messages, normalized, width as i32, (rows * 3) as i32,
                eps, self.stream.0)
        }, "Inkling recurrent controller inputs")?;
        let cols = 2 * width;
        for (group, weights) in down.iter().enumerate() {
            if self.tiled_skinny && rows >= 32 {
                check(unsafe {
                    sys::blz_skinny_gemm_bf16_tiled(normalized.add(group * cols), (3 * cols) as i64,
                        weights.weight.ptr(), self.gate.at(group * 64), 3 * 64,
                        rows as i32, 64, cols as i32, self.stream.0)
                }, "Inkling recurrent controller row tiles")?;
            } else {
                for first in (0..rows).step_by(sys::BLZ_SKINNY_EX_MAX_ROWS) {
                    let batch = (rows - first).min(sys::BLZ_SKINNY_EX_MAX_ROWS);
                    let at = first * 3 + group;
                    check(unsafe {
                        sys::blz_skinny_gemm_ex(normalized.add(at * cols).cast(), (3 * cols) as i64,
                            weights.weight.ptr(), self.gate.at::<u16>(at * 64).cast(), 3 * 64,
                            batch as i32, 64, cols as i32, sys::BLZ_BF16, self.stream.0)
                    }, "Inkling recurrent controller down batch")?;
                }
            }
        }
        check(unsafe {
            sys::blz_inkling_controller_up(self.gate.ptr(), up[0].weight.ptr(), up[1].weight.ptr(), up[2].weight.ptr(),
                self.result.ptr(), cols as i32, 3, rows as i32, self.stream.0)
        }, "Inkling recurrent controller up batch")?;
        let (output, stride) = projected.unwrap_or((normalized, 0));
        check(unsafe {
            sys::blz_inkling_controller_qkv(self.result.ptr(), states, messages, norms.ptr(), output,
                width as i32, rows as i32, stride, eps, self.stream.0)
        }, "Inkling recurrent controller normalization")?;
        Ok(output)
    }

    #[cfg(test)]
    fn pack_recurrent_attention(dst: &mut [f32], width: usize, x: &[Vec<Vec<f32>>], messages: &[[Vec<f32>; 3]]) {
        let (states, msg) = dst.split_at_mut(x.len() * 3 * width);
        for (out, src) in states.chunks_exact_mut(width).zip(x.iter().flatten()) {
            out.copy_from_slice(src);
        }
        for (out, src) in msg.chunks_exact_mut(width).zip(messages.iter().flatten()) {
            out.copy_from_slice(src);
        }
    }

    #[cfg(test)]
    fn validate_recurrent_attention(x: &[Vec<Vec<f32>>], norms: &DevBuf) -> Result<usize> {
        let width = x.first().and_then(|row| row.first()).map_or(0, Vec::len);
        ensure!((32..=4096).contains(&width) && width.is_multiple_of(32)
            && x.iter().all(|row| row.len() == 3 && row.iter().all(|v| v.len() == width))
            && norms.bytes() == 3 * width * 4, "Inkling recurrent attention shape mismatch");
        Ok(width)
    }

    #[cfg(test)]
    #[cfg(test)]
    fn recurrent_attention_inputs(
        &mut self, down: [&Dense; 3], up: [&Dense; 3], x: &[Vec<Vec<f32>>],
        messages: &[[Vec<f32>; 3]], norms: &DevBuf, eps: f32,
    ) -> Result<Vec<u16>> {
        let width = Self::validate_recurrent_attention(x, norms)?;
        ensure!(x.len() <= self.recurrent_batch_limit(width), "Inkling recurrent attention exceeds scratch");
        let values = x.len() * 3 * width;
        let host = self.transfer.host_ptr().cast_mut().cast::<f32>();
        let dst = unsafe { std::slice::from_raw_parts_mut(host, values * 2) };
        Self::pack_recurrent_attention(dst, width, x, messages);
        let work = self.enqueue_recurrent_attention(down, up, norms, width, x.len(), eps, host, None, None)
            .and_then(|normalized| self.queue_result::<u16>(normalized, values));
        let synced = self.stream.sync();
        work?;
        synced?;
        Ok(self.read_result(values))
    }

    #[cfg(test)]
    pub fn queue_recurrent_attention(
        &mut self, down: [&Dense; 3], up: [&Dense; 3], x: &[Vec<Vec<f32>>],
        messages: RecurrentMessages<'_>, norms: &DevBuf, eps: f32,
    ) -> Result<bool> {
        if self.recurrent_batch.is_none() {
            return Ok(false);
        }
        let width = Self::validate_recurrent_attention(x, norms)?;
        let rows = x.len();
        let limit = self.recurrent_batch_limit(width);
        if limit <= 1 {
            return Ok(false);
        }
        let buffers = self.recurrent_batch.as_mut().unwrap();
        ensure!(rows * width * 24 <= buffers.staging.bytes()
            && rows * width * 6 <= buffers.projected_input.bytes(), "Inkling recurrent batch exceeds scratch");
        let host = buffers.staging.host_ptr().cast_mut().cast::<f32>();
        let (stride, use_bank, bank_bytes) = match messages {
            RecurrentMessages::Mixed(messages) => {
                ensure!(messages.len() == rows && messages.iter().flatten().all(|v| v.len() == width),
                    "Inkling mixed message shape mismatch");
                let stride = width * 6;
                let dst = unsafe { std::slice::from_raw_parts_mut(host, rows * stride) };
                for ((dst, x), messages) in dst.chunks_mut(limit * stride)
                    .zip(x.chunks(limit)).zip(messages.chunks(limit))
                {
                    Self::pack_recurrent_attention(dst, width, x, messages);
                }
                (stride, false, 0)
            }
            RecurrentMessages::Bank { upload, weights } => {
                let bytes = rows * width * 8;
                ensure!(weights.len() == rows && bytes <= buffers.bank_staging.bytes().min(buffers.message_bank.bytes()),
                    "Inkling message bank exceeds scratch");
                if let Some(bank) = upload {
                    ensure!(bank.len() == rows && bank.iter().all(|r| r.len() == 2 && r.iter().all(|m| m.len() == width)),
                        "Inkling message bank shape mismatch");
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(buffers.bank_staging.host_ptr().cast_mut().cast::<f32>(), rows * width * 2)
                    };
                    for (out, message) in dst.chunks_exact_mut(width).zip(bank.iter().flat_map(|row| row.iter())) {
                        out.copy_from_slice(message);
                    }
                }
                let stride = width * 3 + 6;
                let dst = unsafe { std::slice::from_raw_parts_mut(host, rows * stride) };
                for ((dst, states), weights) in dst.chunks_mut(limit * stride)
                    .zip(x.chunks(limit)).zip(weights.chunks(limit))
                {
                    let (out, mix) = dst.split_at_mut(states.len() * width * 3);
                    for (out, src) in out.chunks_exact_mut(width).zip(states.iter().flatten()) {
                        out.copy_from_slice(src);
                    }
                    for (dst, &src) in mix.iter_mut().zip(weights.iter().flatten().flatten()) {
                        *dst = src;
                    }
                }
                (stride, true, if upload.is_some() { bytes } else { 0 })
            }
        };
        let buffers = self.recurrent_batch.as_ref().unwrap();
        let enqueue = (|| -> Result<()> {
            if bank_bytes > 0 {
                check(unsafe {
                    sys::cudaMemcpyAsync(buffers.message_bank.ptr(), buffers.bank_staging.host_ptr(),
                        bank_bytes, sys::MEMCPY_H2D, self.stream.0)
                }, "Inkling recurrent message bank upload")?;
            }
            for first in (0..rows).step_by(limit) {
                let batch = (rows - first).min(limit);
                let bank = use_bank.then(|| buffers.message_bank.at::<f32>(first * width * 2).cast_const());
                let output = buffers.projected_input.at::<u16>(first * width);
                self.enqueue_recurrent_attention(down, up, norms, width, batch, eps,
                    host.wrapping_add(first * stride), bank, Some((output, rows as i32)))?;
            }
            Ok(())
        })();
        if enqueue.is_err() {
            let _ = self.stream.sync();
        }
        enqueue.map(|_| true)
    }

    #[cfg(test)]
    pub fn gated_router(
        &mut self, down: &Dense, up: &Dense, norm: &DevBuf, router: &Dense,
        input: Option<&[f32]>, x: &[f32], message: &[f32], eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let width = router.cols;
        let tokens = x.len() / width;
        let size = x.len();
        let scores = router.rows;
        let bytes = size * 12;
        let output_bytes = size * 2 + tokens * scores * 4;
        ensure!((32..=4096).contains(&width) && width.is_multiple_of(32)
            && (1..=self.gate_batch_limit(width)).contains(&tokens) && size == tokens*width
            && message.len() == size && input.is_none_or(|input| input.len() == size * 2)
            && down.cols == width * 2 && down.rows == 64 && up.rows == width * 2 && up.cols == 64
            && norm.bytes() == width * 4 && bytes.max(output_bytes) <= self.input.bytes()
            && width * 8 <= self.result.bytes() && self.gate.bytes() >= 128, "Inkling gated router exceeds scratch");
        let activation = self.input.ptr::<u16>();
        let staging = self.transfer.host_slice_mut();
        let (normalized, rest) = staging[..bytes].split_at_mut(size * 4);
        if let Some(input) = input {
            for (dst, value) in normalized.chunks_exact_mut(2).zip(input) {
                dst.copy_from_slice(&to_bf16(*value).to_ne_bytes());
            }
        }
        for (dst, src) in rest.chunks_exact_mut(size * 4).zip([x, message]) {
            let src = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), dst.len()) };
            dst.copy_from_slice(src);
        }
        let key = ProjectionKey::GatedRouter {
            weights: [down.weight.ptr::<u8>() as usize, up.weight.ptr::<u8>() as usize,
                      norm.ptr::<u8>() as usize, router.weight.ptr::<u8>() as usize],
            eps: eps.to_bits(), tokens,
        };
        let enqueue = |gpu: &Self| {
            if input.is_some() {
                gpu.copy_staged_input(bytes)?;
            } else {
                check(unsafe {
                    sys::cudaMemcpyAsync(gpu.input.at::<f32>(size).cast(),
                        gpu.transfer.host_ptr().cast::<u8>().add(size * 4).cast(), size * 8,
                        sys::MEMCPY_H2D, gpu.stream.0)
                }, "Inkling recurrent MLP inputs")?;
                check(unsafe {
                    sys::blz_inkling_controller_inputs(gpu.input.at(size), gpu.input.at(size * 2),
                        gpu.input.ptr(), width as i32, tokens as i32, eps, gpu.stream.0)
                }, "Inkling recurrent MLP input normalization")?;
            }
            if gpu.tiled_skinny && tokens >= 32 {
                check(unsafe {
                    sys::blz_skinny_gemm_bf16_tiled(gpu.input.ptr(), down.cols as i64, down.weight.ptr(),
                        gpu.gate.ptr(), down.rows as i64, tokens as i32,
                        down.rows as i32, down.cols as i32, gpu.stream.0)
                }, "Inkling MLP controller row tiles")?;
            } else {
                down.run_decode_rows(gpu, gpu.input.ptr(), gpu.gate.ptr(), tokens)?;
            }
            check(unsafe {
                sys::blz_inkling_controller_up(gpu.gate.ptr(), up.weight.ptr(), std::ptr::null(), std::ptr::null(),
                    gpu.result.ptr(), up.rows as i32, 1, tokens as i32, gpu.stream.0)
            }, "Inkling MLP controller projection")?;
            check(unsafe {
                sys::blz_inkling_controller_norm(gpu.result.ptr(), gpu.input.at(size), gpu.input.at(size * 2),
                    norm.ptr(), activation, width as i32, eps, tokens as i32, gpu.stream.0)
            }, "Inkling gated MLP normalization")?;
            let scores_out = gpu.input.at(size / 2);
            check(unsafe {
                sys::blz_inkling_router_logits(router.weight.ptr(), activation, scores_out,
                    scores as i32, width as i32, tokens as i32, gpu.stream.0)
            }, "Inkling gated MoE router")?;
            gpu.queue_result::<u8>(gpu.input.ptr(), output_bytes)
        };
        let work = if tokens == 1 && input.is_some() {
            self.run_projection(key, enqueue)
        } else {
            enqueue(self)
        };
        let synced = self.stream.sync();
        work?;
        synced?;
        let activation = self.read_result::<u16>(size).into_iter().map(from_bf16).collect();
        let host = self.transfer.host_ptr().cast::<u16>();
        let logits = unsafe { std::slice::from_raw_parts(host.add(size).cast::<f32>(), tokens*scores) }.to_vec();
        Ok((activation, logits))
    }

    pub fn argmax(&mut self, matrix: &Dense, x: &[f32], vocab: usize) -> Result<u32> {
        ensure!(vocab > 0 && vocab <= matrix.rows, "invalid Inkling readout vocabulary");
        let x: Vec<_> = x.iter().copied().map(to_bf16).collect();
        let work = (|| -> Result<()> {
            self.stage_input(&x)?;
            matrix.run_prefix(self, self.input.ptr(), self.result.ptr(), 1, vocab, true)?;
            check(
                unsafe {
                    sys::blz_argmax_bf16(
                        self.result.ptr(),
                        1,
                        vocab as i32,
                        vocab as i64,
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
        self.experts.as_ref().map_or(0, MoeEngine::cpu_routes)
    }

    pub fn enqueue_controller(
        &self, down: &Dense, up: &Dense, norm: *const f32, x: *const f32, message: *const f32,
        output: *mut u16, width: usize, rows: usize, eps: f32,
    ) -> Result<()> {
        ensure!(rows * width * 4 <= self.input.bytes() && rows * width * 8 <= self.result.bytes()
            && rows * 128 <= self.gate.bytes(), "Inkling device controller exceeds scratch");
        let enqueue = || {
        check(unsafe {
            sys::blz_inkling_controller_inputs(x, message, self.input.ptr(), width as i32, rows as i32, eps, self.stream.0)
        }, "Inkling controller input")?;
        if rows > 16 {
            down.run(self, self.input.ptr(), self.gate.ptr(), rows)?;
        } else {
            down.run_decode_rows(self, self.input.ptr(), self.gate.ptr(), rows)?;
        }
        check(unsafe {
            sys::blz_inkling_controller_up(self.gate.ptr(), up.weight.ptr(), null_mut(), null_mut(),
                self.result.ptr(), up.rows as i32, 1, rows as i32, self.stream.0)
        }, "Inkling controller projection")?;
        check(unsafe {
            sys::blz_inkling_controller_norm(self.result.ptr(), x, message, norm, output,
                width as i32, eps, rows as i32, self.stream.0)
        }, "Inkling controller residual")
        };
        if rows == 1 {
            let key = [
                down.weight.ptr::<u16>() as usize, up.weight.ptr::<u16>() as usize, norm as usize,
                x as usize, message as usize, output as usize, width, eps.to_bits() as usize,
            ];
            self.controller_graphs.launch(key, &self.stream, enqueue)
        } else {
            enqueue()
        }
    }

    pub fn enqueue_routes(&self, router: &Dense, x: *const u16, bias: &DevBuf, rows: usize,
        top_k: usize, scale: f32, ids: *mut i32, weights: *mut f32) -> Result<()> {
        ensure!(rows * router.rows * 4 <= self.result.bytes(), "Inkling router exceeds scratch");
        check(unsafe {
            sys::blz_inkling_router_logits(router.weight.ptr(), x, self.result.ptr(),
                router.rows as i32, router.cols as i32, rows as i32, self.stream.0)
        }, "Inkling router logits")?;
        check(unsafe {
            sys::blz_router_sigmoid_topk(self.result.ptr(), bias.ptr(), rows as i32, router.rows as i32,
                top_k as i32, scale, ids, weights, self.stream.0)
        }, "Inkling device top-k")
    }

    pub fn enqueue_experts(&self, layer: usize, x: *const u16, rows: usize, ids: *const i32,
        weights: *const f32, output: *mut u16) -> Result<()> {
        let experts = self.experts.as_ref().context("missing Inkling HybridMoe")?;
        unsafe { experts.forward(layer, x, rows, ids, weights, output, &self.stream) }
    }

    fn enqueue_device_scores(&self, head: &Dense, input: *const u16, vocab: usize, rows: usize) -> Result<()> {
        ensure!(rows * vocab * 2 <= self.result.bytes(), "Inkling verification head exceeds workspace");
        let width = head.cols;
        let max_rows = if vocab > 4096 && (input as usize).is_multiple_of(16)
            && crate::tune::wide_fits(1, width, width)
        {
            sys::BLZ_GEMV_WIDE_MAX_ROWS.min(sys::BLZ_GEMV_WIDE_MAX_SMEM / (width * 2))
        } else {
            1
        };
        for row in (0..rows).step_by(max_rows) {
            let batch = (rows - row).min(max_rows);
            head.run_prefix(self, input.wrapping_add(row * width), self.result.at(row * vocab), batch, vocab, true)?;
        }
        Ok(())
    }

    pub fn read_device_picks(&self, head: &Dense, input: *const u16, vocab: usize, rows: usize) -> Result<Vec<u32>> {
        ensure!(rows * 4 <= self.next_id.bytes(), "Inkling verification picks exceed workspace");
        self.enqueue_device_scores(head, input, vocab, rows)?;
        check(unsafe {
            sys::blz_argmax_bf16(self.result.ptr(), rows as i32, vocab as i32, vocab as i64, self.next_id.ptr(), self.stream.0)
        }, "Inkling argmax")?;
        self.queue_result::<u32>(self.next_id.ptr(), rows)?;
        self.stream.sync()?;
        Ok(self.read_result(rows))
    }

    pub fn read_device_scores(&self, head: &Dense, input: *const u16, vocab: usize, rows: usize) -> Result<Vec<u16>> {
        self.enqueue_device_scores(head, input, vocab, rows)?;
        let count = rows * vocab;
        self.queue_result::<u16>(self.result.ptr(), count)?;
        self.stream.sync()?;
        Ok(self.read_result(count))
    }

    pub fn linear_context(&self, device: i32) -> crate::linear::LinCtx {
        crate::linear::LinCtx {
            gemm: self.gemm,
            scratch: self.result.ptr(),
            scratch_elems: self.result.bytes() / 2,
            tune: std::sync::Arc::new(crate::tune::Tuning::open(device as usize)),
            route: crate::linear::GemmRoute::Auto,
            w4a4: crate::matmul::Nvfp4Rows::supported(),
        }
    }

}

impl Drop for Compute {
    fn drop(&mut self) {
        #[cfg(test)]
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
    window: usize,
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

    pub fn new(capacity: usize, window: usize, heads: usize, host: bool, dtype: KvDtype) -> Result<Self> {
        let bytes = capacity
            .checked_mul(Self::row_bytes(heads, dtype) * 2)
            .ok_or_else(|| anyhow::anyhow!("Inkling KV size overflows"))?;
        let alloc = if host { DevBuf::host_mapped } else { DevBuf::new };
        Ok(Self {
            storage: alloc(bytes)?,
            capacity,
            window,
            heads,
            dtype,
        })
    }

    pub(super) fn pointers(&self, slot: usize) -> [*mut u8; 4] {
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
        self.store_rows(position % self.capacity, key, value, zero.ptr(), 1, scale, stream)
    }

    fn store_rows(&self, base: usize, key: *const u16, value: *const u16, slots: *const i32,
                  count: usize, scale: &DevBuf, stream: &Stream) -> Result<()> {
        self.store_strided(base, key, value, self.heads * HEAD_DIM, slots, count, scale, stream)
    }

    pub(super) fn store_strided(&self, base: usize, key: *const u16, value: *const u16, stride: usize,
        slots: *const i32, count: usize, scale: &DevBuf, stream: &Stream) -> Result<()> {
        let [keys, values, key_scales, value_scales] = self.pointers(base);
        let stride = stride as i64;
        if self.dtype == KvDtype::Bf16 {
            return check(
                unsafe {
                    sys::blz_kv_store_dense(
                        key.cast(), value.cast(), stride, stride, slots.cast(), sys::BLZ_I32,
                        count as i64, self.heads as i32, HEAD_DIM as i32, sys::BLZ_KV_BF16,
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
                    slots.cast(),
                    sys::BLZ_I32,
                    count as i64,
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

pub(super) const PREFILL_ATTENTION_ROWS: usize = 16;

pub(super) fn prefill_attention_rows(chunk: usize) -> usize {
    chunk.min(if chunk >= 256 { 64 } else { PREFILL_ATTENTION_ROWS })
}

#[cfg(test)]
pub(super) struct AttentionRow<'a> {
    pub q: Vec<u16>,
    pub gate: &'a [u16],
    pub sink: &'a [u16],
    pub relative: Vec<u16>,
    pub key: Cow<'a, [u16]>,
    pub value: Cow<'a, [u16]>,
}

pub(super) struct AttentionConv<'a> {
    pub weights: &'a DevBuf,
    #[cfg(test)]
    pub history: [&'a [u16]; 2],
    #[cfg(test)]
    pub dynamic: Option<[&'a [u16]; 2]>,
    pub taps: usize,
    pub scale: f32,
    pub clip: f32,
    pub eps: f32,
}

pub(super) struct AttentionClip<'a> {
    pub weights: &'a DevBuf,
    pub relative: usize,
    pub threshold: f32,
    pub relative_threshold: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
}

#[cfg(test)]
impl AttentionRow<'_> {
    fn fields(&self) -> [&[u16]; 6] {
        [&self.q, self.gate, self.sink, &self.relative, &self.key, &self.value]
    }
}

pub(super) struct AttentionScratch {
    input: DevBuf,
    #[cfg(test)]
    staging: HostMapped,
    relative: DevBuf,
    zero: DevBuf,
    kv_scale: DevBuf,
    tiles: DevBuf,
    attended: DevBuf,
    filtered: DevBuf,
    #[cfg(test)]
    block_bytes: usize,
    batch: usize,
    projection_batch: usize,
    filtered_rows: usize,
    tensor_prefill: bool,
    heads: usize,
    kv_heads: usize,
}

impl AttentionScratch {
    pub fn new(heads: usize, kv_heads: usize, extent: usize, relative: usize, chunk: usize, taps: usize) -> Result<Self> {
        let values = heads * (HEAD_DIM * 2 + relative + 1) + kv_heads * HEAD_DIM * 2;
        let batch = chunk.min(PREFILL_ATTENTION_ROWS);
        let projection_batch = prefill_attention_rows(chunk);
        let filtered_rows = if projection_batch > batch { chunk } else { batch };
        let bytes = (values * batch + 2 * taps * kv_heads * (HEAD_DIM + batch)) * 2 + 9 * 16 + batch * 4;
        Ok(Self {
            input: DevBuf::new(bytes.next_multiple_of(16) * chunk.div_ceil(batch))?,
            #[cfg(test)]
            staging: HostMapped::new(bytes.next_multiple_of(16) * chunk.div_ceil(batch))?,
            relative: DevBuf::new(projection_batch * heads * extent * 2)?,
            zero: DevBuf::from_slice(&[0i32])?,
            kv_scale: DevBuf::from_slice(&[KV_GLOBAL_SCALE])?,
            tiles: DevBuf::new(batch * heads * MAX_TILES * (HEAD_DIM + 2) * 4)?,
            attended: DevBuf::new(projection_batch * heads * HEAD_DIM * 2)?,
            filtered: DevBuf::new(filtered_rows * 2 * kv_heads * HEAD_DIM * 2)?,
            #[cfg(test)]
            block_bytes: bytes.next_multiple_of(16),
            batch,
            projection_batch,
            filtered_rows,
            tensor_prefill: false,
            heads,
            kv_heads,
        })
    }

    pub fn set_prefill_attention(&mut self, mode: super::PrefillAttention) {
        self.tensor_prefill = mode == super::PrefillAttention::Tensor;
    }

    pub fn project_attention_device<const N: usize>(
        &self, gpu: &Compute, kv: &Kv, matrices: [&Dense; N], qkv: &DevBuf, offsets: [usize; 5],
        bias: &Dense, output: &Dense, start: usize, rows: usize, conv: &AttentionConv<'_>, clip: &AttentionClip<'_>,
        histories: &[DevBuf; 4], candidate: Option<&[DevBuf; 4]>, attn_weights: &DevBuf, attn_scale: f32,
        previous: *const f32, verify: bool, residual: *mut f32,
    ) -> Result<()> {
        let history = (conv.taps - 1).min(start);
        let width = self.kv_heads * HEAD_DIM;
        let history_values = if rows == 1 { 0 } else { width * history };
        let mut prefix = [0; N];
        prefix[1] = history_values;
        prefix[2] = history_values;
        let slot_offset = (rows * self.heads * (2 * HEAD_DIM + clip.relative + 1)).div_ceil(2);
        let attention_history = if rows == 1 { 0 } else { history * output.rows };
        ensure!((slot_offset + rows) * 4 <= self.input.bytes()
            && (attention_history + rows * output.rows) * 2 <= gpu.input.bytes(),
            "Inkling device attention exceeds scratch");
        let projections = gpu.enqueue_projections(matrices, qkv, offsets, rows, verify, prefix)?;
        let slots = self.input.at::<i32>(slot_offset);
        let raw_attn = gpu.input.at::<u16>(attention_history);
        if rows == 1 {
            self.project_tiles(gpu, kv, &projections, slots, bias, output, start, rows, conv, clip,
                raw_attn, Some(histories))?;
            return check(unsafe {
                sys::blz_inkling_decode_attention_residual(raw_attn, previous, attn_weights.ptr(), residual,
                    histories[2].ptr(), output.rows as i32, conv.taps as i32, history as i32, attn_scale, gpu.stream.0)
            }, "Inkling decode attention residual");
        }
        for (i, &raw) in projections[1..3].iter().enumerate() {
            let full = raw.wrapping_sub(history_values);
            copy_device(&gpu.stream, full, histories[i].ptr(), history_values)?;
            save_history(&gpu.stream, &histories[i], candidate.map(|buffers| &buffers[i]),
                full, history_values, rows * width, (conv.taps - 1) * width)?;
        }
        check(unsafe { sys::blz_inkling_kv_slots(slots, start as i32, kv.capacity as i32, rows as i32, gpu.stream.0) },
            "Inkling device KV slots")?;
        self.project_tiles(gpu, kv, &projections, slots, bias, output, start, rows, conv, clip, raw_attn, None)?;
        copy_device(&gpu.stream, gpu.input.ptr::<u16>(), histories[2].ptr(), attention_history)?;
        check(unsafe {
            sys::blz_inkling_prefill_attention_residual(gpu.input.ptr(), previous, attn_weights.ptr(), residual,
                output.rows as i32, conv.taps as i32, history as i32, rows as i32, attn_scale, gpu.stream.0)
        }, "Inkling device attention residual")?;
        save_history(&gpu.stream, &histories[2], candidate.map(|buffers| &buffers[2]),
            gpu.input.ptr(), attention_history, rows * output.rows, (conv.taps - 1) * output.rows)
    }

    fn project_tiles(
        &self, gpu: &Compute, kv: &Kv, projections: &[*mut u16], slots: *const i32,
        bias: &Dense, output: &Dense, start: usize, rows: usize,
        conv: &AttentionConv<'_>, clip: &AttentionClip<'_>, raw_attention: *mut u16, device_history: Option<&[DevBuf; 4]>,
    ) -> Result<()> {
        let width = self.kv_heads * HEAD_DIM;
        let history = (conv.taps - 1).min(start);
        let q_values = rows * self.heads * HEAD_DIM;
        let rel_values = rows * self.heads * clip.relative;
        let q = self.input.ptr::<u16>();
        let gate = self.input.at::<u16>(q_values);
        let relative = self.input.at::<u16>(2 * q_values);
        let sink = self.input.at::<u16>(2 * q_values + rel_values);
        check(unsafe {
            sys::blz_inkling_split_projection(projections[0], clip.weights.ptr(), q, gate, relative, sink,
                rows as i32, self.heads as i32, clip.relative as i32, start as i32,
                clip.threshold, clip.relative_threshold, clip.log_floor, clip.log_alpha, gpu.stream.0)
        }, "Inkling Q and relative clipping")?;

        if rows == 1 {
            if let Some(state) = device_history {
                let dynamic = |i| projections.get(i).map_or(std::ptr::null(), |p: &*mut u16| p.cast_const());
                check(unsafe {
                    sys::blz_inkling_decode_conv(projections[1], projections[2], dynamic(3), dynamic(4),
                        conv.weights.ptr(), self.filtered.ptr(), self.filtered.at(self.filtered_rows * width),
                        self.kv_heads as i32, conv.taps as i32, history as i32, state[0].ptr(), state[1].ptr(),
                        conv.scale, conv.clip, conv.eps, gpu.stream.0)
                }, "Inkling decode KV convolution")?;
            } else {
                self.convolve_kv(gpu, conv, projections, history, 0, history, 1)?;
            }
            kv.store(start, self.filtered.ptr(), self.filtered.at(self.filtered_rows * width),
                &self.zero, &self.kv_scale, &gpu.stream)?;
            self.attend_one(gpu, kv, [q, gate, sink, relative], bias, start)?;
            return output.run(gpu, self.attended.ptr(), raw_attention, 1);
        }

        let grouped = self.filtered_rows > self.batch;
        if grouped {
            self.convolve_kv(gpu, conv, projections, history, 0, history, rows)?;
        }
        let mut first = 0;
        while first < rows {
            let pos = start + first;
            let max_count = (rows - first).min(self.projection_batch);
            let max_tiles = kv.window.min(pos + max_count).div_ceil(64).clamp(1, MAX_TILES);
            let count = max_count.min(self.batch * MAX_TILES / max_tiles);
            let oldest = (pos + 1).saturating_sub(kv.window);
            ensure!(pos + count - oldest <= kv.capacity, "Inkling prefill would overwrite live KV");
            let filtered_at = if grouped { first * width } else {
                self.convolve_kv(gpu, conv, projections, history, first, (conv.taps - 1).min(pos), count)?;
                0
            };
            let keys = self.filtered.at(filtered_at);
            let values = self.filtered.at(self.filtered_rows * width + filtered_at);
            kv.store_rows(0, keys, values, slots.wrapping_add(first), count, &self.kv_scale, &gpu.stream)?;
            let query_offset = first * self.heads * HEAD_DIM;
            self.attend_tile(gpu, kv, [
                q.wrapping_add(query_offset), gate.wrapping_add(query_offset), sink.wrapping_add(first * self.heads),
                relative.wrapping_add(first * self.heads * clip.relative),
            ], bias, pos, count, false)?;
            let dst = raw_attention.wrapping_add(first * output.rows);
            if gpu.tiled_skinny && count >= 32 && output.cols.is_multiple_of(8) {
                check(unsafe {
                    sys::blz_skinny_gemm_bf16_tiled(self.attended.ptr(), output.cols as i64, output.weight.ptr(),
                        dst, output.rows as i64, count as i32,
                        output.rows as i32, output.cols as i32, gpu.stream.0)
                }, "Inkling attention output row tiles")?;
            } else {
                output.run_decode_rows(gpu, self.attended.ptr(), dst, count)?;
            }
            first += count;
        }
        Ok(())
    }

    fn convolve_kv(
        &self, gpu: &Compute, conv: &AttentionConv<'_>, projections: &[*mut u16],
        available_history: usize, first: usize, history: usize, count: usize,
    ) -> Result<()> {
        let width = self.kv_heads * HEAD_DIM;
        let offset = (first + available_history - history) * width;
        let raw = |i: usize| projections[i].wrapping_sub(available_history * width).wrapping_add(offset);
        let dynamic = |i: usize| projections.get(i).map_or(std::ptr::null(), |p| {
            p.wrapping_add(first * self.kv_heads * conv.taps).cast_const()
        });
        let key = self.filtered.ptr();
        let value = self.filtered.at(self.filtered_rows * width);
        check(unsafe {
            sys::blz_inkling_prefill_conv(raw(1), raw(2), dynamic(3), dynamic(4), conv.weights.ptr(), key, value,
                self.kv_heads as i32, conv.taps as i32, history as i32, count as i32,
                conv.scale, conv.clip, conv.eps, gpu.stream.0)
        }, "Inkling prefill KV convolution")
    }

    fn attend_tile(
        &self, gpu: &Compute, kv: &Kv, [q, gate, sink, relative]: [*const u16; 4], bias: &Dense,
        pos: usize, count: usize, verify: bool,
    ) -> Result<()> {
        bias.run(gpu, relative, self.relative.ptr(), count * self.heads)?;
        let [keys, values, ks, vs] = kv.pointers(0);
        check(unsafe {
            sys::blz_inkling_prefill_attention(q, keys, values, ks, vs, self.relative.ptr(), sink, gate,
                self.attended.ptr(), self.tiles.ptr(), self.heads as i32, self.kv_heads as i32,
                kv.capacity as i32, kv.window as i32, pos as i32, bias.rows as i32, count as i32,
                KV_GLOBAL_SCALE, i32::from(self.tensor_prefill && !verify && !kv.storage.is_host()), gpu.stream.0)
        }, "Inkling causal prefill attention")
    }

    fn attend_one(&self, gpu: &Compute, kv: &Kv, [q, gate, sink, relative]: [*const u16; 4],
        bias: &Dense, position: usize) -> Result<()> {
        bias.run(gpu, relative, self.relative.ptr(), self.heads)?;
        let tiles = kv.window.min(position + 1).div_ceil(64).clamp(1, MAX_TILES);
        let [keys, values, key_scales, value_scales] = kv.pointers(0);
        check(unsafe {
            sys::blz_inkling_attention(q, keys, values, key_scales, value_scales, self.relative.ptr(), sink, gate,
                self.attended.ptr(), self.tiles.ptr(), self.heads as i32, self.kv_heads as i32, kv.capacity as i32,
                kv.window as i32, position as i32, bias.rows as i32, tiles as i32, KV_GLOBAL_SCALE, gpu.stream.0)
        }, "Inkling relative attention")
    }

    #[cfg(test)]
    pub fn run_rows(&mut self, gpu: &mut Compute, kv: &Kv, rows: &[AttentionRow<'_>], bias: &Dense,
                    output: &Dense, start: usize, conv: Option<AttentionConv<'_>>, verify: bool) -> Result<Vec<f32>> {
        if let [row] = rows {
            let [q, gate, sink, relative, key, value] = row.fields();
            return self.run(gpu, kv, q, gate, sink, relative, key, value, bias, output, start);
        }

        let result_count = rows.len() * output.rows;
        ensure!(result_count * 2 <= gpu.result.bytes()
            && rows.len().div_ceil(self.batch) * self.block_bytes <= self.input.bytes(),
            "Inkling attention rows exceed scratch");
        let work = (|| -> Result<()> {
            for (block, chunk) in rows.chunks(self.batch).enumerate() {
                let first = block * self.batch;
                let pos = start + first;
                let count = chunk.len();
                let oldest = (pos + 1).saturating_sub(kv.window);
                ensure!(pos + count - oldest <= kv.capacity, "Inkling prefill would overwrite live KV");
                let base = block * self.block_bytes;
                let staging = &mut self.staging.host_slice_mut()[base..base + self.block_bytes];
                let mut offset: usize = 0;
                let history = conv.as_ref().map_or(0, |c| (c.taps - 1).min(pos));
                let width = self.kv_heads * HEAD_DIM;
                let mut stage = |src: &[u16], offset: &mut usize| {
                    // The staged copy reads initialized BF16 rows as bytes.
                    let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src)) };
                    staging[*offset..*offset + bytes.len()].copy_from_slice(bytes);
                    *offset += bytes.len();
                };
                let [q, gate, sink, relative, key, value] = std::array::from_fn(|field| {
                    offset = offset.next_multiple_of(16);
                    let ptr = self.input.at::<u16>((base + offset) / 2);
                    if let Some(c) = &conv && field >= 4 {
                        let from_state = history.saturating_sub(first) * width;
                        let initial = c.history[field - 4];
                        stage(&initial[initial.len() - from_state..], &mut offset);
                        for row in &rows[first.saturating_sub(history)..first] {
                            stage(row.fields()[field], &mut offset);
                        }
                    }
                    for row in chunk {
                        stage(row.fields()[field], &mut offset);
                    }
                    ptr
                });
                let dynamic = if let Some(c) = &conv && let Some(dynamic) = c.dynamic {
                    let width = self.kv_heads * c.taps;
                    dynamic.map(|src| {
                        offset = offset.next_multiple_of(16);
                        let ptr = self.input.at::<u16>((base + offset) / 2);
                        stage(&src[first * width..(first + count) * width], &mut offset);
                        ptr as *const u16
                    })
                } else {
                    [std::ptr::null(); 2]
                };
                let slots_at = offset.next_multiple_of(16);
                let mut offset = slots_at;
                for p in pos..pos + count {
                    staging[offset..offset + 4].copy_from_slice(&((p % kv.capacity) as i32).to_ne_bytes());
                    offset += 4;
                }
                check(unsafe {
                    sys::cudaMemcpyAsync(self.input.at::<u8>(base).cast(), self.staging.host_ptr().cast::<u8>().add(base).cast(),
                        offset, sys::MEMCPY_H2D, gpu.stream.0)
                }, "Inkling attention batch inputs")?;
                let (key, value) = if let Some(c) = &conv {
                    let keys = self.filtered.ptr();
                    let values = self.filtered.at(self.batch * width);
                    check(unsafe {
                        sys::blz_inkling_prefill_conv(key, value, dynamic[0], dynamic[1], c.weights.ptr(), keys, values,
                            self.kv_heads as i32, c.taps as i32, history as i32, count as i32,
                            c.scale, c.clip, c.eps, gpu.stream.0)
                    }, "Inkling prefill KV convolution")?;
                    (keys, values)
                } else {
                    (key, value)
                };
                kv.store_rows(0, key, value, self.input.at((base + slots_at) / 4), count, &self.kv_scale, &gpu.stream)?;
                self.attend_tile(gpu, kv, [q, gate, sink, relative], bias, pos, count, verify)?;
                output.run_decode_rows(gpu, self.attended.ptr(), gpu.result.at(first * output.rows), count)?;
            }
            gpu.queue_result::<u16>(gpu.result.ptr(), result_count)
        })();
        let synced = gpu.stream.sync();
        work?;
        synced?;
        Ok(gpu.read_result::<u16>(result_count).into_iter().map(from_bf16).collect())
    }

    #[cfg(test)]
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
            self.attend_one(gpu, kv, [q, gate, sink, relative], bias, position)?;
            output.run(gpu, self.attended.ptr(), gpu.result.ptr(), 1)?;

            gpu.queue_result::<u16>(gpu.result.ptr(), output.rows)
        })();
        let synced = gpu.stream.sync();
        work?;
        synced?;
        Ok(gpu.read_result::<u16>(output.rows).into_iter().map(from_bf16).collect())
    }
}

pub(super) fn copy_device<T>(stream: &Stream, destination: *mut T, source: *const T, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    // Callers provide non-overlapping ranges on the stream's device.
    check(unsafe {
        sys::cudaMemcpyAsync(destination.cast(), source.cast(), count * size_of::<T>(), sys::MEMCPY_D2D, stream.0)
    }, "Inkling device state copy")
}

pub(super) fn save_history(stream: &Stream, committed: &DevBuf, pending: Option<&DevBuf>,
    source: *const u16, prefix: usize, count: usize, capacity: usize) -> Result<()> {
    let total = prefix + count;
    if let Some(pending) = pending {
        ensure!(total * 2 <= pending.bytes(), "Inkling verification history exceeds workspace");
        copy_device(stream, pending.ptr(), source, total)
    } else {
        let keep = total.min(capacity);
        copy_device(stream, committed.ptr(), source.wrapping_add(total - keep), keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn tensor_prefill_matches_scalar_for_multiple_query_tiles_and_tail() -> Result<()> {
        crate::gpu::set_device(0)?;
        let stream = Stream::new()?;
        let (heads, rows, window, extent, start) = (3usize, 47usize, 192usize, 64usize, 250usize);
        let capacity = window + rows - 1;
        let kv = Kv::new(capacity, window, 1, false, KvDtype::Nvfp4FourOverSix)?;
        let values = |count: usize, salt: usize| -> Result<DevBuf> {
            let data: Vec<u16> = (0..count)
                .map(|i| to_bf16(((i.wrapping_mul(7919).wrapping_add(i / 31 * salt) % 509) as f32 - 254.0) / 512.0))
                .collect();
            DevBuf::from_slice(&data)
        };
        let slots: Vec<i32> = (start + 1 - window..start + rows)
            .map(|pos| (pos % capacity) as i32)
            .collect();
        let slots = DevBuf::from_slice(&slots)?;
        let scale = DevBuf::from_slice(&[KV_GLOBAL_SCALE])?;
        let keys = values(capacity * HEAD_DIM, 19)?;
        let vals = values(capacity * HEAD_DIM, 23)?;
        kv.store_rows(0, keys.ptr(), vals.ptr(), slots.ptr(), capacity, &scale, &stream)?;
        let count = rows * heads * HEAD_DIM;
        let query = values(count, 29)?;
        let gate = values(count, 31)?;
        let sink = values(rows * heads, 37)?;
        let relative = values(rows * heads * extent, 41)?;
        let partial = DevBuf::new(rows * heads * MAX_TILES * (HEAD_DIM + 2) * 4)?;
        let output = DevBuf::new(count * 2)?;
        let [k, v, ks, vs] = kv.pointers(0);
        let run = |tensor| -> Result<Vec<u16>> {
            check(unsafe {
                sys::blz_inkling_prefill_attention(query.ptr(), k, v, ks, vs, relative.ptr(), sink.ptr(),
                    gate.ptr(), output.ptr(), partial.ptr(), heads as i32, 1, capacity as i32, window as i32,
                    start as i32, extent as i32, rows as i32, KV_GLOBAL_SCALE, tensor, stream.0)
            }, "test tiled Inkling attention")?;
            stream.sync()?;
            output.download(count)
        };
        let scalar = run(0)?;
        let tensor = run(1)?;
        for (index, (&expected, &actual)) in scalar.iter().zip(&tensor).enumerate() {
            assert!((from_bf16(expected) - from_bf16(actual)).abs() < 0.002, "score at {index}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn batched_vocabulary_head_matches_single_rows() -> Result<()> {
        crate::gpu::set_device(0)?;
        let (rows, width, vocab) = (7usize, 2304usize, 4099usize);
        let make_values = |count: usize, salt: usize| {
            (0..count)
                .map(|i| to_bf16(((i * 37 + i / 19 * salt) % 509) as f32 / 1024.0 - 0.25))
                .collect::<Vec<_>>()
        };
        let matrix = Dense::from_bf16(&make_values(vocab * width, 13), vocab, width)?;
        let input = DevBuf::from_slice(&make_values(rows * width, 7))?;
        let compute = Compute::new(width, rows * vocab, width, rows)?;
        let batched = compute.read_device_scores(&matrix, input.ptr(), vocab, rows)?;
        for (row, got) in batched.chunks_exact(vocab).enumerate() {
            assert_eq!(got, compute.read_device_scores(&matrix, input.at(row * width), vocab, 1)?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn causal_prefill_matches_decode_across_tiles_and_ring_wrap() {
        crate::gpu::set_device(0).unwrap();
        let window = 160;
        let capacity = window + PREFILL_ATTENTION_ROWS - 1;
        let end = 225;
        let mut scratch = AttentionScratch::new(1, 1, window, 8, end, 4).unwrap();
        let mut gpu = Compute::new(128, 2048, 128, end).unwrap();
        let bias: Vec<_> = (0..window * 8).map(|i| to_bf16((i % 11) as f32 / 32.0)).collect();
        let bias = Dense::from_bf16(&bias, window, 8).unwrap();
        let weights: Vec<_> = (0..128 * 128).map(|i| to_bf16((i % 13) as f32 / 256.0)).collect();
        let output = Dense::from_bf16(&weights, 128, 128).unwrap();
        let row = |size, pos, salt| (0..size)
            .map(|i| to_bf16((((i * 37 + pos * salt) % 97) as f32 - 48.0) / 64.0))
            .collect::<Vec<_>>();
        let gates: Vec<_> = (0..end).map(|p| row(128, p, 3)).collect();
        let sinks: Vec<_> = (0..end).map(|p| row(1, p, 7)).collect();
        let rows: Vec<_> = (0..end).map(|p| AttentionRow {
            q: row(128, p, 11),
            gate: &gates[p],
            sink: &sinks[p],
            relative: row(8, p, 13),
            key: row(128, p, 17).into(),
            value: row(128, p, 19).into(),
        }).collect();
        for dtype in [KvDtype::Bf16, KvDtype::Nvfp4, KvDtype::Nvfp4FourOverSix] {
            let serial = Kv::new(capacity, window, 1, false, dtype).unwrap();
            let batch = Kv::new(capacity, window, 1, false, dtype).unwrap();
            let expected: Vec<_> = rows.iter().enumerate().flat_map(|(pos, row)| {
                scratch.run_rows(&mut gpu, &serial, std::slice::from_ref(row), &bias, &output, pos, None, true).unwrap()
            }).collect();
            let actual = scratch.run_rows(&mut gpu, &batch, &rows, &bias, &output, 0, None, true).unwrap();
            assert_eq!(actual, expected, "{dtype:?}");
            scratch.tensor_prefill = true;
            let tensor = scratch.run_rows(&mut gpu, &batch, &rows, &bias, &output, 0, None, false).unwrap();
            for (row, (&actual, &expected)) in tensor.iter().zip(&expected).enumerate() {
                assert!((actual - expected).abs() < 0.002, "{dtype:?} at {row}: {actual} vs {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "requires a CUDA GPU and AVX2/FMA"]
    fn recurrent_attention_batch_matches_host_normalization() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) { return; }
        crate::gpu::set_device(0).unwrap();
        let width = 2304;
        let count = 131;
        let mut gpu = Compute::new(2 * width, 12 * width, width, 512).unwrap();
        let values = |count, seed: usize| {
            (0..count).map(move |i| ((i * 641 + (i % 113) * seed) % 719) as f32 / 431.0 - 0.83)
        };
        let matrix = |rows, cols, seed| {
            let data = values(rows * cols, seed).map(|v| to_bf16(v * 0.025)).collect::<Vec<_>>();
            Dense::from_bf16(&data, rows, cols).unwrap()
        };
        let down = [23, 109, 403].map(|seed| matrix(64, width * 2, seed));
        let up = [57, 179, 929].map(|seed| matrix(width * 2, 64, seed));
        let norm = values(width * 3, 691).collect::<Vec<_>>();
        let norms = DevBuf::from_slice(&norm).unwrap();
        let x: Vec<Vec<Vec<f32>>> = (0..count).map(|row| {
            (0..3).map(|group| values(width, 19 * row + 33 * group).collect()).collect()
        }).collect();
        let messages: Vec<[Vec<f32>; 3]> = (0..count).map(|row| {
            std::array::from_fn(|group| values(width, 11 * row + 41 * group).collect())
        }).collect();
        let eps = 1e-6;
        let normalize = |row| super::super::forward::normalized(row, None, eps);
        let inputs: Vec<[Vec<f32>; 3]> = x.iter().zip(&messages).map(|(x, msg)| {
            std::array::from_fn(|group| normalize(x[group].as_slice()).into_iter()
                .chain(normalize(msg[group].as_slice())).collect())
        }).collect();
        let inputs: Vec<_> = inputs.iter().map(|rows| rows.each_ref().map(Vec::as_slice)).collect();
        let gates = gpu.controller_gates_batch(down.each_ref(), up.each_ref(), &inputs).unwrap();
        let expected: Vec<_> = x.iter().flatten().zip(messages.iter().flatten())
            .zip(gates.chunks_exact(2 * width)).enumerate().flat_map(|(i, ((row, message), gate))| {
                let mut row = row.clone();
                let (forget, gain) = gate.split_at(width);
                super::super::cpu::controller_mix(&mut row, message, forget, gain);
                let norm = &norm[(i % 3) * width..][..width];
                super::super::forward::normalized(&row, Some(norm), eps).into_iter().map(to_bf16)
            }).collect();
        let limit = gpu.recurrent_batch_limit(width);
        let actual: Vec<_> = x.chunks(limit).zip(messages.chunks(limit)).flat_map(|(x, messages)| {
            gpu.recurrent_attention_inputs(down.each_ref(), up.each_ref(), x, messages, &norms, eps).unwrap()
        }).collect();
        let different = actual.iter().zip(&expected).position(|(a, b)| a != b);
        assert_eq!(different.map(|i| (i / width, i % width, actual[i], expected[i])), None);
        gpu.recurrent_batch = Some(RecurrentBatch::new(count, width).unwrap());
        assert!(gpu.queue_recurrent_attention(down.each_ref(), up.each_ref(), &x,
            RecurrentMessages::Mixed(&messages), &norms, eps).unwrap());
        gpu.stream.sync().unwrap();
        let output = gpu.recurrent_batch.as_ref().unwrap().projected_input.download::<u16>(count * width * 3).unwrap();
        for group in 0..3 {
            for row in 0..count {
                let offset = (group * count + row) * width;
                let expected_offset = (row * 3 + group) * width;
                assert_eq!(&output[offset..][..width], &expected[expected_offset..][..width]);
            }
        }
        let raw: Vec<_> = x.iter().map(|row| &row[..2]).collect();
        let mixed: Vec<_> = raw.iter().map(|row| std::array::from_fn(|_| {
            row[0].iter().zip(&row[1]).map(|(a, b)| (a + b) * 0.5).collect::<Vec<_>>()
        })).collect();
        let mix_weights = vec![[[0.5f32; 2]; 3]; count];
        assert!(gpu.queue_recurrent_attention(down.each_ref(), up.each_ref(), &x,
            RecurrentMessages::Mixed(&mixed), &norms, eps).unwrap());
        gpu.stream.sync().unwrap();
        let cpu_message = gpu.recurrent_batch.as_ref().unwrap().projected_input.download::<u16>(count * width * 3).unwrap();
        for upload in [Some(raw.as_slice()), None] {
            assert!(gpu.queue_recurrent_attention(down.each_ref(), up.each_ref(), &x,
                RecurrentMessages::Bank { upload, weights: &mix_weights }, &norms, eps).unwrap());
            gpu.stream.sync().unwrap();
            let gpu_message = gpu.recurrent_batch.as_ref().unwrap().projected_input.download::<u16>(count * width * 3).unwrap();
            assert_eq!(gpu_message, cpu_message);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "requires a CUDA GPU and AVX2/FMA"]
    fn gated_router_replays_the_cpu_mix_and_norm() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) { return; }
        crate::gpu::set_device(0).unwrap();
        let width = 2304;
        let mut gpu = Compute::new(2 * width, 4 * width, width, 8).unwrap();
        gpu.projection_graphs = Some(HashMap::new());
        let values = |len, scale: f32| (0..len).map(|i: usize| {
            (((i * 47 % 179) as f32 - 89.0) / (9 + i % 31) as f32) * scale
        }).collect::<Vec<_>>();
        let dense = |rows, cols, scale| Dense::from_bf16(
            &values(rows * cols, scale).into_iter().map(to_bf16).collect::<Vec<_>>(), rows, cols,
        ).unwrap();
        let down = dense(64, 2 * width, 0.005);
        let up = dense(2 * width, 64, 0.15);
        let router = dense(128, width, 0.07);
        let weights = values(width, 0.43);
        let norm = DevBuf::from_slice(&weights).unwrap();
        let (mut inputs, mut xs, mut messages, mut activations, mut routes) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for factor in [0.007, 0.75, 9.0] {
            let x = values(width, factor);
            let mut msg = values(width, -0.4);
            msg.rotate_left(11);
            let input = values(2 * width, factor);
            let gates = gpu.controller_gates([&down], [&up], [&input]).unwrap();
            let (mut expected, eps) = (x.clone(), 1e-6);
            super::super::cpu::controller_mix(&mut expected, &msg, &gates[..width], &gates[width..]);
            let rms = (super::super::forward::dot(&expected, &expected) / width as f32 + eps).sqrt().recip();
            for (v, weight) in expected.iter_mut().zip(&weights) {
                *v = from_bf16(to_bf16(*v * rms * weight));
            }
            let scores = gpu.router_logits(&router, std::slice::from_ref(&expected)).unwrap();
            let (actual, logits) = gpu.gated_router(&down, &up, &norm, &router, Some(&input), &x, &msg, eps).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(logits, scores);
            inputs.extend(input);
            xs.extend(x);
            messages.extend(msg);
            activations.extend(expected);
            routes.extend(scores);
        }
        let batch = gpu.gated_router(&down, &up, &norm, &router, Some(&inputs), &xs, &messages, 1e-6).unwrap();
        assert_eq!(batch, (activations, routes));
        let normalized: Vec<_> = xs.chunks_exact(width).zip(messages.chunks_exact(width)).flat_map(|(x, m)| {
            super::super::forward::normalized(x, None, 1e-6).into_iter()
                .chain(super::super::forward::normalized(m, None, 1e-6))
        }).collect();
        let host = gpu.gated_router(&down, &up, &norm, &router, Some(&normalized), &xs, &messages, 1e-6).unwrap();
        let device = gpu.gated_router(&down, &up, &norm, &router, None, &xs, &messages, 1e-6).unwrap();
        assert_eq!(device, host);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "requires a CUDA GPU and AVX2/FMA"]
    fn router_logits_match_host_for_prefill_and_decode() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) { return; }
        crate::gpu::set_device(0).unwrap();
        let mut gpu = Compute::new(96, 128, 96, 3).unwrap();
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
                output.ptr(), rows as i32, 3, 1, stream.0)
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
        let mut gpu = Compute::new(32, 512, 32, 3).unwrap();
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
                gpu.linear_bf16(&taps, &x.iter().copied().map(to_bf16).collect::<Vec<_>>(), 1).unwrap().into_iter()
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
        let mut gpu = Compute::new(32, 32, 32, 1).unwrap();
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
    #[ignore = "requires CUDA"]
    fn tiled_skinny_prefill_matches_decode_order_for_strided_rows() {
        crate::gpu::set_device(0).unwrap();
        let (rows, columns, inner) = (67usize, 259usize, 256usize);
        let (input_stride, output_stride) = (inner + 32, columns + 5);
        let values = |len, salt| (0..len).map(|i| to_bf16(((i * salt % 97) as f32 - 48.0) / 31.0)).collect::<Vec<_>>();
        let input = DevBuf::from_slice(&values(rows * input_stride, 13)).unwrap();
        let weights = DevBuf::from_slice(&values(columns * inner, 61)).unwrap();
        let stream = Stream::new().unwrap();
        let eager = DevBuf::zeroed(rows * output_stride * 2, &stream).unwrap();
        let tiled = DevBuf::zeroed(rows * output_stride * 2, &stream).unwrap();
        for start in [0, 64] {
            let n = (rows - start).min(64);
            check(unsafe {
                sys::blz_skinny_gemm_cfg(input.at::<u16>(start * input_stride).cast(), input_stride as i64,
                    weights.ptr(), eager.at::<u16>(start * output_stride).cast(), output_stride as i64,
                    n as i32, columns as i32, inner as i32, sys::BLZ_BF16, 4, stream.0)
            }, "reference skinny projection").unwrap();
        }
        check(unsafe {
            sys::blz_skinny_gemm_bf16_tiled(input.ptr(), input_stride as i64, weights.ptr(), tiled.ptr(),
                output_stride as i64, rows as i32, columns as i32, inner as i32, stream.0)
        }, "tiled skinny projection").unwrap();
        stream.sync().unwrap();
        assert_eq!(eager.download::<u16>(rows * output_stride).unwrap(), tiled.download::<u16>(rows * output_stride).unwrap());
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
            let capacity = if host { 2 } else { 5 };
            let kv = Kv::new(capacity, 2, 1, host, dtype).unwrap();
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
                            capacity as i32,
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
