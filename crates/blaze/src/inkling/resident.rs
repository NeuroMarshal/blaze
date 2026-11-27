use super::super::cpu;
use super::super::cuda::{AttentionClip, AttentionConv, Compute, copy_device, save_history};
use super::{Controller, Input, Layer, Model, Prediction, Readout, Sequence, Shape, from_bf16};
use crate::gpu::{self, DevBuf, DeviceTransfer, HostMapped, Stream, check};
use crate::layer_walk::LoopEntry;
use anyhow::{Result, ensure};
use blaze_sys as sys;
use std::ptr::{null, null_mut};

struct LayerWeights {
    qkv_norm: DevBuf,
    mlp_norm: DevBuf,
    qkv_query: DevBuf,
    mlp_query: DevBuf,
    messages: Option<DevBuf>,
    kv: DevBuf,
    clip: DevBuf,
    attention: DevBuf,
    mlp: DevBuf,
    route_bias: DevBuf,
}

impl LayerWeights {
    fn new(layer: &Layer) -> Result<Self> {
        let qkv_norm: Vec<_> = if layer.local {
            layer.norm_q.repeat(3)
        } else {
            layer
                .norm_q
                .iter()
                .chain(&layer.norm_k)
                .chain(&layer.norm_v)
                .copied()
                .collect()
        };
        let clip: Vec<_> = layer
            .q_clip
            .iter()
            .chain(&layer.rel_clip)
            .copied()
            .collect();
        let taps = |kernel: &super::Conv| kernel.taps.iter().flatten().copied().collect::<Vec<_>>();
        let kv: Vec<_> = layer
            .k_conv
            .taps
            .iter()
            .chain(&layer.v_conv.taps)
            .flatten()
            .copied()
            .collect();
        Ok(Self {
            qkv_norm: DevBuf::from_slice(&qkv_norm)?,
            mlp_norm: DevBuf::from_slice(&layer.norm_mlp)?,
            qkv_query: DevBuf::from_slice(&layer.res_query)?,
            mlp_query: DevBuf::from_slice(&layer.res_mlp)?,
            messages: layer
                .loop_inputs
                .as_ref()
                .map(|c| {
                    DevBuf::from_slice(&c.queries.iter().flatten().copied().collect::<Vec<_>>())
                })
                .transpose()?,
            kv: DevBuf::from_slice(&kv)?,
            clip: DevBuf::from_slice(&clip)?,
            attention: DevBuf::from_slice(&taps(&layer.attn_conv))?,
            mlp: DevBuf::from_slice(&taps(&layer.mlp_conv))?,
            route_bias: DevBuf::from_slice(&layer.route_bias)?,
        })
    }
}

struct Workspace {
    hidden: DevBuf,
    bank: DevBuf,
    entry: DevBuf,
    messages: DevBuf,
    current: DevBuf,
    mixed: DevBuf,
    message: DevBuf,
    residual: DevBuf,
    rounded: DevBuf,
    input: DevBuf,
    raw_moe: DevBuf,
    routes: DevBuf,
    route_weights: DevBuf,
}

impl Workspace {
    fn new(s: &Shape, chunk: usize, depth: usize) -> Result<Self> {
        let values = s.hidden * chunk;
        Ok(Self {
            hidden: DevBuf::new(values * 4)?,
            bank: DevBuf::new(values * depth * 4)?,
            entry: DevBuf::new(values * 4)?,
            messages: DevBuf::new(values * depth * 4)?,
            current: DevBuf::new(values * 4)?,
            mixed: DevBuf::new(values * 4)?,
            message: DevBuf::new(values * 4)?,
            residual: DevBuf::new(values * 4)?,
            rounded: DevBuf::new(values * 2)?,
            input: DevBuf::new(values * 3 * 2)?,
            raw_moe: DevBuf::new((values + (s.conv - 1) * s.hidden) * 2)?,
            routes: DevBuf::new(chunk * s.top_k * 4)?,
            route_weights: DevBuf::new(chunk * s.top_k * 4)?,
        })
    }
}

pub(super) struct Executor {
    layers: Vec<LayerWeights>,
    workspace: Vec<Workspace>,
    handoff: Vec<DeviceTransfer>,
    staging: HostMapped,
    features: Option<HostMapped>,
    norm_embed: DevBuf,
    norm_final: DevBuf,
    query_final: DevBuf,
}

impl Executor {
    pub(super) fn supports(s: &Shape, layers: usize) -> bool {
        layers.div_ceil(s.block) < 32
            && s.hidden.is_multiple_of(32)
            && s.hidden <= 4096
            && s.experts <= 512
            && s.relative < 32
            && s.hidden <= s.q_width()
            && s.heads.is_multiple_of(8)
    }

    pub(super) fn new(
        first_device: i32,
        gpus: usize,
        placement: &[usize],
        layers: &[Layer],
        s: &Shape,
        chunk: usize,
        draft: bool,
        norm_embed: &[f32],
        norm_final: &[f32],
        query_final: &[f32],
    ) -> Result<Self> {
        let depth = layers.len().div_ceil(s.block);
        let mut weights = Vec::with_capacity(layers.len());
        for (layer, &rank) in layers.iter().zip(placement) {
            gpu::set_device(first_device + rank as i32)?;
            weights.push(LayerWeights::new(layer)?);
        }
        let mut workspace = Vec::with_capacity(gpus);
        let mut handoff = Vec::new();
        for rank in 0..gpus {
            gpu::set_device(first_device + rank as i32)?;
            workspace.push(Workspace::new(s, chunk, depth)?);
            if gpus > 1 {
                handoff.push(DeviceTransfer::new(
                    first_device + rank as i32,
                    first_device + (rank ^ 1) as i32,
                    4 * (depth + 1) * chunk * s.hidden,
                )?);
            }
        }
        gpu::set_device(first_device + placement[0] as i32)?;
        let norm_embed = DevBuf::from_slice(norm_embed)?;
        gpu::set_device(first_device + placement.last().copied().unwrap() as i32)?;
        Ok(Self {
            layers: weights,
            workspace,
            handoff,
            staging: HostMapped::new(chunk * (2 * s.hidden + 1))?,
            features: draft
                .then(|| HostMapped::new(4 * 4 * chunk * s.hidden))
                .transpose()?,
            norm_embed,
            norm_final: DevBuf::from_slice(norm_final)?,
            query_final: DevBuf::from_slice(query_final)?,
        })
    }
}

pub(super) fn histories(s: &Shape, rows: usize) -> Result<[DevBuf; 4]> {
    let size = |width| rows * width * 2;
    Ok([
        DevBuf::new(size(s.kv_width()))?,
        DevBuf::new(size(s.kv_width()))?,
        DevBuf::new(size(s.hidden))?,
        DevBuf::new(size(s.hidden))?,
    ])
}

fn queue_copy<T>(
    stream: &Stream,
    dst: *mut T,
    src: *const T,
    count: usize,
    kind: i32,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    check(
        unsafe {
            sys::cudaMemcpyAsync(
                dst.cast(),
                src.cast(),
                count * size_of::<T>(),
                kind,
                stream.0,
            )
        },
        "Inkling activation transfer",
    )
}

fn feature_ptr(buffer: &HostMapped, values: usize, tap: usize) -> *mut f32 {
    buffer
        .host_ptr()
        .cast_mut()
        .cast::<f32>()
        .wrapping_add(tap * values)
}

struct Chunk {
    rows: usize,
    start: usize,
    width: usize,
    eps: f32,
}

impl Chunk {
    fn values(&self) -> usize {
        self.rows * self.width
    }

    fn normalize(
        &self,
        stream: &Stream,
        input: *const f32,
        weight: *const f32,
        full: *mut f32,
        rounded: *mut u16,
    ) -> Result<()> {
        check(
            unsafe {
                sys::blz_inkling_norm_rows(
                    input,
                    weight,
                    full,
                    rounded,
                    self.rows as i32,
                    self.width as i32,
                    self.eps,
                    stream.0,
                )
            },
            "Inkling resident RMSNorm",
        )
    }

    fn mix(
        &self,
        stream: &Stream,
        bank: *const f32,
        depth: usize,
        current: *const f32,
        query: *const f32,
        output: *mut f32,
    ) -> Result<()> {
        check(
            unsafe {
                sys::blz_mix_normalized_rows(
                    bank,
                    current,
                    query,
                    output,
                    depth as i32,
                    self.rows as i32,
                    self.width as i32,
                    stream.0,
                )
            },
            "Inkling resident residual mix",
        )
    }

    fn expand(&self, stream: &Stream, input: *const u16, output: *mut f32) -> Result<()> {
        check(
            unsafe { sys::blz_inkling_expand_bf16(input, output, self.values(), stream.0) },
            "Inkling residual expansion",
        )
    }

    fn activate(
        &self,
        compute: &Compute,
        w: &Workspace,
        controller: Option<(&Controller, *const f32)>,
        message_depth: usize,
        input: *const f32,
        norm: *const f32,
        output: *mut u16,
    ) -> Result<()> {
        if let Some((controller, query)) = controller {
            self.mix(
                &compute.stream,
                w.messages.ptr(),
                message_depth,
                null(),
                query,
                w.message.ptr(),
            )?;
            compute.enqueue_controller(
                &controller.down,
                &controller.up,
                norm,
                input,
                w.message.ptr(),
                output,
                self.width,
                self.rows,
                self.eps,
            )
        } else {
            self.normalize(&compute.stream, input, norm, null_mut(), output)
        }
    }
}

impl Model {
    fn commit_verified_history(&self, seq: &Sequence, consumed: usize) -> Result<()> {
        let taps = self.cfg.conv - 1;
        let history = taps.min(seq.position);
        let total = history + consumed;
        let keep = total.min(taps);
        if keep == 0 {
            return Ok(());
        }
        let widths = [
            self.cfg.kv_width(),
            self.cfg.kv_width(),
            self.cfg.hidden,
            self.cfg.hidden,
        ];
        for (rank, device) in self.devices.iter().enumerate() {
            gpu::set_device(device.id)?;
            for (visit, state) in self.loops.visits.iter().zip(&seq.visits) {
                if self.placement[visit.layer] != rank {
                    continue;
                }
                let candidate = state.verification.as_ref().unwrap();
                for (i, width) in widths.into_iter().enumerate() {
                    copy_device(
                        &device.compute.stream,
                        state.history[i].ptr(),
                        candidate[i].at::<u16>((total - keep) * width),
                        keep * width,
                    )?;
                }
            }
        }
        for device in &self.devices {
            gpu::set_device(device.id)?;
            device.compute.stream.sync()?;
        }
        Ok(())
    }

    fn drain_resident(&self) {
        for (rank, device) in self.devices.iter().enumerate() {
            if gpu::set_device(device.id).is_ok() {
                let _ = device.compute.stream.sync();
                if let Some(handoff) = self.executor.handoff.get(rank) {
                    handoff.drain();
                }
            }
        }
    }

    pub(super) fn forward_gpu(
        &mut self,
        seq: &mut Sequence,
        input: &[Input],
        readout: Readout,
    ) -> Result<Option<Prediction>> {
        let result = self.enqueue_forward(seq, input, readout);
        if result.is_err() {
            self.drain_resident();
        }
        result
    }

    fn enqueue_forward(
        &mut self,
        seq: &mut Sequence,
        input: &[Input],
        readout: Readout,
    ) -> Result<Option<Prediction>> {
        let s = &self.cfg;
        let rows = input.len();
        ensure!(
            rows > 0
                && rows <= self.prefill_chunk
                && rows <= seq.capacity.saturating_sub(seq.position),
            "Inkling chunk exceeds workspace or sequence context"
        );
        let verifying = matches!(readout, Readout::Verify(_));
        ensure!(
            !verifying || rows > 1 && self.draft_limit() > 0,
            "invalid Inkling verification block"
        );
        let chunk = Chunk {
            rows,
            start: seq.position,
            width: s.hidden,
            eps: s.eps,
        };
        let values = chunk.values();
        let executor = &mut self.executor;
        let mut active = self.placement[0];
        gpu::set_device(self.devices[active].id)?;
        let width_bytes = 2 * s.hidden;
        let embedding = self.ckpt.bytes("model.llm.embed.weight")?;
        let (staging, image_flags) = executor.staging.host_slice_mut().split_at_mut(values * 2);
        for ((item, row), flag) in input
            .iter()
            .zip(staging.chunks_exact_mut(width_bytes))
            .zip(image_flags)
        {
            match item {
                Input::Token(id) => {
                    let id = *id as usize;
                    ensure!(
                        id < s.unpadded_vocab,
                        "Inkling: token {id} outside vocabulary"
                    );
                    row.copy_from_slice(&embedding[id * width_bytes..][..width_bytes]);
                    *flag = 0;
                }
                Input::Image(data) => {
                    ensure!(
                        data.len() == s.hidden,
                        "Inkling image embedding width mismatch"
                    );
                    for (dst, word) in row.chunks_exact_mut(2).zip(data) {
                        dst.copy_from_slice(&word.to_ne_bytes());
                    }
                    *flag = 1;
                }
            }
        }
        let workspace = &executor.workspace[active];
        let stream = &self.devices[active].compute.stream;
        queue_copy(
            stream,
            workspace.input.ptr(),
            executor.staging.host_ptr().cast::<u8>(),
            values * 2 + rows,
            sys::MEMCPY_H2D,
        )?;
        check(
            unsafe {
                sys::blz_inkling_embed_rows(
                    workspace.input.ptr(),
                    workspace.input.at(values * 2),
                    executor.norm_embed.ptr(),
                    workspace.hidden.ptr(),
                    rows as i32,
                    s.hidden as i32,
                    s.eps,
                    stream.0,
                )
            },
            "Inkling device embeddings",
        )?;
        let mut depth = 0;
        for (index, (visit, state)) in self.loops.visits.iter().zip(&seq.visits).enumerate() {
            let id = visit.layer;
            let step = &self.residual_walk[index];
            let rank = self.placement[id];
            if active != rank {
                let source = &executor.workspace[active];
                let destination = &executor.workspace[rank];
                executor.handoff[active].enqueue(
                    &self.devices[active].compute.stream,
                    &self.devices[rank].compute.stream,
                    &[
                        (&source.hidden, &destination.hidden, values * 4),
                        (&source.bank, &destination.bank, values * depth * 4),
                    ],
                )?;
                active = rank;
            }
            let device = &self.devices[rank];
            let compute = &device.compute;
            let stream = &compute.stream;
            let workspace = &executor.workspace[rank];
            let weights = &executor.layers[id];
            let layer = &self.layers[id];
            let histories = &state.history;
            let candidate = verifying.then(|| state.verification.as_ref().unwrap());
            if let Some(features) = &executor.features
                && let Some(tap) = [5, 23, 41].iter().position(|&tap| tap == index)
            {
                queue_copy(
                    stream,
                    feature_ptr(features, values, tap),
                    workspace.hidden.ptr(),
                    values,
                    sys::MEMCPY_D2H,
                )?;
            }
            depth = step.depth;
            match step.loop_entry {
                Some(LoopEntry::Save) => {
                    copy_device(
                        stream,
                        workspace.entry.ptr::<f32>(),
                        workspace.hidden.ptr(),
                        values,
                    )?;
                }
                Some(LoopEntry::Rewind {
                    first_message,
                    previous_blocks,
                }) => {
                    copy_device(
                        stream,
                        workspace.messages.ptr(),
                        workspace.bank.at::<f32>(first_message * values),
                        previous_blocks * values,
                    )?;
                    chunk.normalize(
                        stream,
                        workspace.hidden.ptr(),
                        null(),
                        workspace.messages.at(previous_blocks * values),
                        null_mut(),
                    )?;
                    copy_device(
                        stream,
                        workspace.hidden.ptr::<f32>(),
                        workspace.entry.ptr(),
                        values,
                    )?;
                }
                None => {}
            }
            let controller = (visit.pass > 0).then(|| layer.loop_inputs.as_ref().unwrap());
            let block_start = step.bank_slot.is_some();
            let mut offsets = [0usize; 5];
            if id != 0 {
                chunk.normalize(
                    stream,
                    workspace.hidden.ptr(),
                    null(),
                    workspace.current.ptr(),
                    null_mut(),
                )?;
            }
            let groups = if (id == 0 || layer.local) && controller.is_none() {
                1
            } else {
                3
            };
            for group in 0..groups {
                let input = if id == 0 {
                    workspace.hidden.ptr()
                } else {
                    let q = if layer.local { 0 } else { group };
                    chunk.mix(
                        stream,
                        workspace.bank.ptr(),
                        depth,
                        workspace.current.ptr(),
                        weights.qkv_query.at(q * s.hidden),
                        workspace.mixed.ptr(),
                    )?;
                    workspace.mixed.ptr()
                };
                let control = controller.map(|c| {
                    (
                        &c.controllers[group],
                        weights
                            .messages
                            .as_ref()
                            .unwrap()
                            .at::<f32>(group * s.hidden)
                            .cast_const(),
                    )
                });
                offsets[group] = group * values;
                chunk.activate(
                    compute,
                    workspace,
                    control,
                    step.messages,
                    input,
                    weights.qkv_norm.at(group * s.hidden),
                    workspace.input.at(offsets[group]),
                )?;
            }
            offsets[3] = offsets[1];
            offsets[4] = offsets[2];
            let conv = AttentionConv {
                weights: &weights.kv,
                #[cfg(test)]
                history: [&[], &[]],
                #[cfg(test)]
                dynamic: None,
                taps: s.conv,
                scale: layer.k_conv.scale,
                clip: s.qk_clip,
                eps: s.eps,
            };
            let clip = AttentionClip {
                weights: &weights.clip,
                relative: s.relative,
                threshold: s.qk_clip,
                relative_threshold: s.rel_clip,
                log_floor: s.log_floor,
                log_alpha: if layer.local { 0.0 } else { s.log_alpha },
            };
            let previous = if block_start {
                null()
            } else {
                workspace.hidden.ptr::<f32>().cast_const()
            };
            if let (Some(k), Some(v)) = (&layer.k_conv.dynamic, &layer.v_conv.dynamic) {
                device.scratch.project_attention_device(
                    compute,
                    &state.kv,
                    [&layer.query, &layer.key, &layer.value, k, v],
                    &workspace.input,
                    offsets,
                    &layer.bias,
                    &layer.output,
                    chunk.start,
                    rows,
                    &conv,
                    &clip,
                    histories,
                    candidate,
                    &weights.attention,
                    layer.attn_conv.scale,
                    previous,
                    verifying,
                    workspace.residual.ptr(),
                )?;
            } else {
                device.scratch.project_attention_device(
                    compute,
                    &state.kv,
                    [&layer.query, &layer.key, &layer.value],
                    &workspace.input,
                    offsets,
                    &layer.bias,
                    &layer.output,
                    chunk.start,
                    rows,
                    &conv,
                    &clip,
                    histories,
                    candidate,
                    &weights.attention,
                    layer.attn_conv.scale,
                    previous,
                    verifying,
                    workspace.residual.ptr(),
                )?;
            }
            if let Some(slot) = step.bank_slot {
                chunk.normalize(
                    stream,
                    workspace.hidden.ptr(),
                    null(),
                    workspace.bank.at(slot * values),
                    null_mut(),
                )?;
                depth += 1;
            }
            chunk.normalize(
                stream,
                workspace.residual.ptr(),
                null(),
                workspace.current.ptr(),
                null_mut(),
            )?;
            chunk.mix(
                stream,
                workspace.bank.ptr(),
                depth,
                workspace.current.ptr(),
                weights.mlp_query.ptr(),
                workspace.mixed.ptr(),
            )?;
            let control = controller.map(|c| {
                (
                    &c.controllers[3],
                    weights
                        .messages
                        .as_ref()
                        .unwrap()
                        .at::<f32>(3 * s.hidden)
                        .cast_const(),
                )
            });
            chunk.activate(
                compute,
                workspace,
                control,
                step.messages,
                workspace.mixed.ptr(),
                weights.mlp_norm.ptr(),
                workspace.input.ptr(),
            )?;
            compute.enqueue_routes(
                &layer.router,
                workspace.input.ptr(),
                &weights.route_bias,
                rows,
                s.top_k,
                s.route_scale,
                workspace.routes.ptr(),
                workspace.route_weights.ptr(),
            )?;
            let history = (s.conv - 1).min(chunk.start);
            let prefix = if rows == 1 { 0 } else { history * s.hidden };
            let raw = workspace.raw_moe.at::<u16>(prefix);
            compute.enqueue_experts(
                self.expert_layers[id],
                workspace.input.ptr(),
                rows,
                workspace.routes.ptr(),
                workspace.route_weights.ptr(),
                raw,
            )?;
            if rows == 1 {
                check(
                    unsafe {
                        sys::blz_inkling_decode_mlp_residual(
                            raw,
                            workspace.residual.ptr(),
                            weights.mlp.ptr(),
                            workspace.rounded.ptr(),
                            histories[3].ptr(),
                            s.hidden as i32,
                            s.conv as i32,
                            history as i32,
                            layer.mlp_conv.scale,
                            stream.0,
                        )
                    },
                    "Inkling decode MLP residual",
                )?;
            } else {
                copy_device(
                    stream,
                    workspace.raw_moe.ptr(),
                    histories[3].ptr::<u16>(),
                    prefix,
                )?;
                check(
                    unsafe {
                        sys::blz_inkling_mlp_residual_conv(
                            raw,
                            workspace.residual.ptr(),
                            workspace.raw_moe.ptr(),
                            weights.mlp.ptr(),
                            workspace.rounded.ptr(),
                            null_mut(),
                            s.hidden as i32,
                            s.conv as i32,
                            history as i32,
                            rows as i32,
                            0,
                            layer.mlp_conv.scale,
                            stream.0,
                        )
                    },
                    "Inkling resident MLP residual",
                )?;
                save_history(
                    stream,
                    &histories[3],
                    candidate.map(|buffers| &buffers[3]),
                    workspace.raw_moe.ptr(),
                    prefix,
                    values,
                    (s.conv - 1) * s.hidden,
                )?;
            }
            chunk.expand(stream, workspace.rounded.ptr(), workspace.hidden.ptr())?;
        }
        let device = &self.devices[active];
        let w = &executor.workspace[active];
        let stream = &device.compute.stream;
        let features = executor.features.as_ref();
        let mut consumed = rows;
        let mut result = None;
        if features.is_some() || !matches!(readout, Readout::None) {
            chunk.normalize(stream, w.hidden.ptr(), null(), w.current.ptr(), null_mut())?;
            chunk.mix(
                stream,
                w.bank.ptr(),
                depth,
                w.current.ptr(),
                executor.query_final.ptr(),
                w.mixed.ptr(),
            )?;
        }
        if let Some(features) = features {
            chunk.normalize(
                stream,
                w.mixed.ptr(),
                executor.norm_final.ptr(),
                w.current.ptr(),
                null_mut(),
            )?;
            queue_copy(
                stream,
                feature_ptr(features, values, 3),
                w.current.ptr(),
                values,
                sys::MEMCPY_D2H,
            )?;
        }
        match readout {
            Readout::None => stream.sync()?,
            Readout::Verify(draft) => {
                chunk.normalize(
                    stream,
                    w.mixed.ptr(),
                    executor.norm_final.ptr(),
                    null_mut(),
                    w.input.ptr(),
                )?;
                let picks = device.compute.read_device_picks(
                    &self.head,
                    w.input.ptr(),
                    s.unpadded_vocab,
                    rows,
                )?;
                let (kept, next) = crate::speculation::match_picks(draft, &picks);
                consumed = kept + 1;
                result = Some(Prediction::Verified(kept, next));
            }
            Readout::Greedy | Readout::Scores => {
                let last = Chunk { rows: 1, ..chunk };
                last.normalize(
                    stream,
                    w.mixed.at(values - s.hidden),
                    executor.norm_final.ptr(),
                    null_mut(),
                    w.input.ptr(),
                )?;
                result = Some(if matches!(readout, Readout::Greedy) {
                    let picks = device.compute.read_device_picks(
                        &self.head,
                        w.input.ptr(),
                        s.unpadded_vocab,
                        1,
                    )?;
                    Prediction::Token(picks[0])
                } else {
                    let mut logits = device
                        .compute
                        .read_device_scores(&self.head, w.input.ptr(), s.unpadded_vocab)?
                        .into_iter()
                        .map(from_bf16)
                        .collect::<Vec<_>>();
                    cpu::softcap(&mut logits, s.logit_cap);
                    Prediction::Scores(logits)
                });
            }
        }
        if let Some(features) = features {
            if let Some(dflash) = &mut self.dflash {
                dflash.append(&device.compute, features, seq.position, rows, consumed)?;
            } else if let (Some(dspark), Some(cache)) = (&self.draft, &mut seq.draft) {
                let data = (0..consumed)
                    .map(|index| {
                        std::array::from_fn(|tap| {
                            let row =
                                feature_ptr(features, values, tap).wrapping_add(index * s.hidden);
                            unsafe { std::slice::from_raw_parts(row, s.hidden) }.to_vec()
                        })
                    })
                    .collect::<Vec<_>>();
                dspark.append(&device.compute, cache, seq.position, &data)?;
            }
        }
        if verifying {
            self.commit_verified_history(seq, consumed)?;
        }
        seq.position += consumed;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA"]
    fn rejected_rows_cannot_replace_committed_convolution_history() -> Result<()> {
        gpu::set_device(0)?;
        let stream = Stream::new()?;
        let committed = DevBuf::from_slice(&[10u16, 11, 12])?;
        let pending = DevBuf::new(7 * size_of::<u16>())?;
        let source = DevBuf::from_slice(&[10u16, 11, 12, 20, 21, 22, 23])?;
        save_history(&stream, &committed, Some(&pending), source.ptr(), 3, 4, 3)?;
        stream.sync()?;
        assert_eq!(committed.download::<u16>(3)?, [10, 11, 12]);
        let keep = 3;
        let consumed = 2;
        copy_device(
            &stream,
            committed.ptr(),
            pending.at::<u16>(3 + consumed - keep),
            keep,
        )?;
        stream.sync()?;
        assert_eq!(committed.download::<u16>(3)?, [12, 20, 21]);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn sigmoid_routes_use_bias_only_for_ordering_and_preserve_expert_order() -> Result<()> {
        gpu::set_device(0)?;
        let stream = Stream::new()?;
        let logits = DevBuf::from_slice(&[0.0f32, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0])?;
        let bias = DevBuf::from_slice(&[0.0f32, 0.0, -1.0, 1.0])?;
        let ids = DevBuf::new(6 * size_of::<i32>())?;
        let weights = DevBuf::new(6 * size_of::<f32>())?;
        check(
            unsafe {
                sys::blz_router_sigmoid_topk(
                    logits.ptr(),
                    bias.ptr(),
                    2,
                    4,
                    3,
                    2.0,
                    ids.ptr(),
                    weights.ptr(),
                    stream.0,
                )
            },
            "sigmoid top-k test",
        )?;
        stream.sync()?;
        assert_eq!(ids.download::<i32>(6)?, [0, 1, 3, 0, 1, 3]);
        let actual = weights.download::<f32>(6)?;
        let tail = (1.0 + 2.0f32.exp()).recip();
        let scale = 2.0 / (1.0 + tail);
        let expected = [
            0.5 * scale,
            0.5 * scale,
            tail * scale,
            2.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
        ];
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        Ok(())
    }
}
