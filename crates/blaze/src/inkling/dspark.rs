use super::KvDtype;
use super::cuda::{Compute, Dense, Kv, from_bf16, to_bf16};
use super::forward::normalized;
use super::weights::read_float;
use crate::gpu::{DevBuf, check};
use crate::weights::Checkpoint;
use anyhow::{Result, ensure};
use blaze_sys as sys;
use serde_json::Value;

const H: usize = 2304;
const BLOCK: usize = 8;
const HEADS: usize = 24;
const KV_HEADS: usize = 8;
const D: usize = 128;
const Q: usize = HEADS * D;
const K: usize = KV_HEADS * D;
const FF: usize = 9216;
const R: usize = 256;
const WINDOW: usize = 1024;
const EPS: f32 = 1e-6;
const SCALE: f32 = 1.0 / 16.0;

pub(super) type Features = [Vec<f32>; 4];

struct Layer {
    ctx: Dense,
    qkv: Dense,
    out: Dense,
    gate_up: Dense,
    down: Dense,
    ctx_norm: Vec<f32>,
    ctx_k_norm: DevBuf,
    input_norm: Vec<f32>,
    q_norm: DevBuf,
    k_norm: DevBuf,
    post_norm: Vec<f32>,
}

pub(super) struct Draft {
    layers: Vec<Layer>,
    fuse: [[f32; 4]; 4],
    mask: Vec<f32>,
    final_norm: Vec<f32>,
    head_norm: Vec<f32>,
    rnn: Dense,
    bias: Vec<f32>,
    correction: Dense,
    correction_bias: Vec<f32>,
    inv_freq: DevBuf,
    input: DevBuf,
    output: DevBuf,
    attention: DevBuf,
    partial: DevBuf,
    slots: DevBuf,
    no_sink: DevBuf,
    open_gate: DevBuf,
    scale: DevBuf,
    pub limit: usize,
    pub head_vocab: usize,
}

pub(super) struct Cache {
    kv: Vec<Kv>,
    capacity: usize,
    conditioning: Vec<f32>,
}

impl Cache {
    fn slots(capacity: usize, layer: usize) -> usize {
        (if layer == 1 {
            capacity
        } else {
            capacity.min(WINDOW)
        }) + BLOCK
    }

    pub fn bytes(capacity: usize, format: KvDtype) -> usize {
        (0..3)
            .map(|i| Self::slots(capacity, i) * 2 * Kv::row_bytes(KV_HEADS, format))
            .sum()
    }

    pub fn new(capacity: usize, format: KvDtype) -> Result<Self> {
        let kv = (0..3)
            .map(|i| {
                Kv::new(
                    Self::slots(capacity, i),
                    Self::slots(capacity, i),
                    KV_HEADS,
                    false,
                    format,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            kv,
            capacity,
            conditioning: Vec::new(),
        })
    }
}

impl Draft {
    pub fn load(
        c: &Checkpoint,
        cfg: &Value,
        limit: usize,
        vocab: usize,
        head_vocab: usize,
    ) -> Result<Self> {
        for (key, expected) in [
            ("hidden_size", H),
            ("num_hidden_layers", 3),
            ("num_attention_heads", HEADS),
            ("num_key_value_heads", KV_HEADS),
            ("head_dim", D),
            ("intermediate_size", FF),
            ("markov_rank", R),
            ("block_size", BLOCK),
            ("sliding_window_size", WINDOW),
        ] {
            ensure!(
                cfg[key].as_u64() == Some(expected as u64),
                "unsupported DSpark {key}"
            );
        }
        ensure!(
            cfg["target_layer_ids"] == serde_json::json!([5, 23, 41, 48])
                && cfg["local_layer_ids"] == serde_json::json!([0, 2])
                && cfg["markov_head_type"] == "rnn_hidden"
                && cfg["rope_theta"].as_f64() == Some(1e6)
                && cfg["rms_norm_eps"].as_f64() == Some(1e-6),
            "unsupported Inkling DSpark configuration"
        );
        let raw = read_float(c, "fuse_weight", &[4, 4])?;
        let fuse = std::array::from_fn(|i| {
            let row = &raw[4 * i..4 * i + 4];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: [f32; 4] = std::array::from_fn(|j| (row[j] - max).exp());
            let sum: f32 = exp.iter().sum();
            exp.map(|x| x / sum)
        });
        let layers = (0..3)
            .map(|i| {
                let name = |t: &str| format!("layers.{i}.{t}.weight");
                let vector = |t, len| read_float(c, &name(t), &[len]);
                let head_norm = |t| {
                    DevBuf::from_slice(&vector(t, D)?.into_iter().map(to_bf16).collect::<Vec<_>>())
                };
                Ok(Layer {
                    ctx: Dense::stack(
                        c,
                        &[
                            (&name("self_attn.k_proj_ctx"), K),
                            (&name("self_attn.v_proj_ctx"), K),
                        ],
                        H,
                    )?,
                    qkv: Dense::stack(
                        c,
                        &[
                            (&name("self_attn.q_proj"), Q),
                            (&name("self_attn.k_proj"), K),
                            (&name("self_attn.v_proj"), K),
                        ],
                        H,
                    )?,
                    out: Dense::load(c, &name("self_attn.o_proj"), H, Q)?,
                    gate_up: Dense::stack(
                        c,
                        &[(&name("mlp.gate_proj"), FF), (&name("mlp.up_proj"), FF)],
                        H,
                    )?,
                    down: Dense::load(c, &name("mlp.down_proj"), H, FF)?,
                    ctx_norm: vector("dflare_norm", H)?,
                    ctx_k_norm: head_norm("self_attn.k_norm_ctx")?,
                    input_norm: vector("input_layernorm", H)?,
                    q_norm: head_norm("self_attn.q_norm")?,
                    k_norm: head_norm("self_attn.k_norm")?,
                    post_norm: vector("post_attention_layernorm", H)?,
                })
            })
            .collect::<Result<_>>()?;
        let table = c.info("markov_head.markov_w1.weight")?;
        ensure!(
            table.dtype == crate::weights::Dtype::Bf16 && table.shape == [vocab, R],
            "invalid DSpark token table"
        );
        Ok(Self {
            layers,
            fuse,
            limit,
            head_vocab,
            mask: read_float(c, "mask_embed", &[H])?,
            final_norm: read_float(c, "norm.weight", &[H])?,
            head_norm: read_float(c, "markov_head.dflare_norm.weight", &[H])?,
            rnn: Dense::load(c, "markov_head.rnn_proj.weight", 3 * R, 2 * R + 2 * H)?,
            bias: read_float(c, "markov_head.rnn_proj.bias", &[3 * R])?,
            correction: Dense::load(c, "markov_head.hidden_out.weight", H, R)?,
            correction_bias: read_float(c, "markov_head.hidden_out.bias", &[H])?,
            inv_freq: DevBuf::from_slice(&std::array::from_fn::<_, { D / 2 }, _>(|j| {
                1e6f32.powf(-2.0 * j as f32 / D as f32)
            }))?,
            input: DevBuf::new(BLOCK * FF * 2)?,
            output: DevBuf::new(BLOCK * 2 * FF * 2)?,
            attention: DevBuf::new(BLOCK * Q * 2)?,
            partial: DevBuf::new(BLOCK * HEADS * 256 * (D + 2) * 4)?,
            slots: DevBuf::new(BLOCK * 4)?,
            no_sink: DevBuf::from_slice(&[to_bf16(f32::NEG_INFINITY); HEADS])?,
            open_gate: DevBuf::from_slice(&vec![to_bf16(f32::INFINITY); Q])?,
            scale: DevBuf::from_slice(&[SCALE])?,
        })
    }

    fn project(&self, gpu: &Compute, matrix: &Dense, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        ensure!(
            rows > 0 && rows <= BLOCK && x.len() == matrix.cols * rows,
            "invalid DSpark projection shape"
        );
        let x: Vec<_> = x.iter().copied().map(to_bf16).collect();
        self.input.upload(&x)?;
        matrix.run(gpu, self.input.ptr(), self.output.ptr(), rows)?;
        gpu.stream.sync()?;
        Ok(self
            .output
            .download::<u16>(matrix.rows * rows)?
            .into_iter()
            .map(from_bf16)
            .collect())
    }

    fn mix(&self, row: &Features, weights: &[f32; 4], norm: &[f32]) -> Vec<f32> {
        let mut mix = vec![0.0; H];
        for (x, w) in row.iter().zip(weights) {
            for (out, v) in mix.iter_mut().zip(x) {
                *out += v * w;
            }
        }
        normalized(&mix, Some(norm), EPS)
    }

    fn normalize_heads(
        &self,
        gpu: &Compute,
        offset: usize,
        stride: usize,
        rows: usize,
        heads: usize,
        norm: &DevBuf,
        start: usize,
        cache_slots: Option<usize>,
    ) -> Result<()> {
        check(
            unsafe {
                sys::blz_dspark_norm_rope(
                    self.output.at(offset),
                    stride as i32,
                    rows as i32,
                    heads as i32,
                    norm.ptr(),
                    self.inv_freq.ptr(),
                    start as i32,
                    cache_slots.map_or(std::ptr::null_mut(), |_| self.slots.ptr()),
                    cache_slots.unwrap_or(1) as i32,
                    gpu.stream.0,
                )
            },
            "DSpark head normalization and RoPE",
        )
    }

    fn store(
        &self,
        gpu: &Compute,
        cache: &Cache,
        layer: usize,
        rows: usize,
        offset: usize,
        stride: usize,
    ) -> Result<()> {
        cache.kv[layer].store_strided(0, self.output.at(offset), self.output.at(offset + K),
            stride, self.slots.ptr(), rows, &self.scale, &gpu.stream)
    }

    pub fn append(
        &self,
        gpu: &Compute,
        cache: &mut Cache,
        start: usize,
        features: &[Features],
    ) -> Result<()> {
        for (chunk, rows) in features.chunks(BLOCK).enumerate() {
            let pos = start + chunk * BLOCK;
            for (i, l) in self.layers.iter().enumerate() {
                let inputs: Vec<_> = rows
                    .iter()
                    .flat_map(|r| self.mix(r, &self.fuse[i], &l.ctx_norm))
                    .map(to_bf16)
                    .collect();
                self.input.upload(&inputs)?;
                l.ctx
                    .run(gpu, self.input.ptr(), self.output.ptr(), rows.len())?;
                self.normalize_heads(
                    gpu,
                    0,
                    2 * K,
                    rows.len(),
                    KV_HEADS,
                    &l.ctx_k_norm,
                    pos,
                    Some(Cache::slots(cache.capacity, i)),
                )?;
                self.store(gpu, cache, i, rows.len(), 0, 2 * K)?;
                gpu.stream.sync()?;
            }
        }
        cache.conditioning = self.mix(features.last().unwrap(), &self.fuse[3], &self.head_norm);
        Ok(())
    }

    pub fn propose(
        &self,
        gpu: &mut Compute,
        cache: &Cache,
        ckpt: &Checkpoint,
        head: &Dense,
        anchor: u32,
        embedding: Vec<f32>,
        pos: usize,
        vocab: usize,
        limit: usize,
    ) -> Result<Vec<u32>> {
        let mut x = embedding;
        for _ in 1..BLOCK {
            x.extend_from_slice(&self.mask);
        }
        for (i, l) in self.layers.iter().enumerate() {
            let input: Vec<_> = x
                .chunks_exact(H)
                .flat_map(|r| normalized(r, Some(&l.input_norm), EPS))
                .map(to_bf16)
                .collect();
            let [k, v, ks, vs] = cache.kv[i].pointers(0);
            let slots = Cache::slots(cache.capacity, i);
            let stride = Q + 2 * K;
            self.input.upload(&input)?;
            l.qkv.run(gpu, self.input.ptr(), self.output.ptr(), BLOCK)?;
            self.normalize_heads(gpu, 0, stride, BLOCK, HEADS, &l.q_norm, pos, None)?;
            self.normalize_heads(gpu, Q, stride, BLOCK, KV_HEADS, &l.k_norm, pos, Some(slots))?;
            self.store(gpu, cache, i, BLOCK, Q, stride)?;
            check(
                unsafe {
                    sys::blz_dspark_attention(
                        self.output.ptr(),
                        stride as i32,
                        BLOCK as i32,
                        k,
                        v,
                        ks,
                        vs,
                        self.no_sink.ptr(),
                        self.open_gate.ptr(),
                        self.attention.ptr(),
                        self.partial.ptr(),
                        HEADS as i32,
                        KV_HEADS as i32,
                        slots as i32,
                        (pos + BLOCK - 1) as i32,
                        (pos + BLOCK).min(slots).div_ceil(64).clamp(1, 256) as i32,
                        SCALE,
                        gpu.stream.0,
                    )
                },
                "DSpark block attention",
            )?;
            l.out
                .run(gpu, self.attention.ptr(), self.input.ptr(), BLOCK)?;
            gpu.stream.sync()?;
            let out = self
                .input
                .download::<u16>(BLOCK * H)?
                .into_iter()
                .map(from_bf16);
            x.iter_mut().zip(out).for_each(|(dst, v)| *dst += v);
            let input: Vec<_> = x
                .chunks_exact(H)
                .flat_map(|r| normalized(r, Some(&l.post_norm), EPS))
                .collect();
            self.input
                .upload(&input.into_iter().map(to_bf16).collect::<Vec<_>>())?;
            l.gate_up
                .run(gpu, self.input.ptr(), self.output.ptr(), BLOCK)?;
            check(
                unsafe {
                    sys::blz_silu_mul(
                        self.output.ptr(),
                        (2 * FF) as i64,
                        self.output.at(FF),
                        (2 * FF) as i64,
                        BLOCK as i32,
                        FF as i32,
                        self.input.ptr(),
                        gpu.stream.0,
                    )
                },
                "DSpark SwiGLU",
            )?;
            l.down
                .run(gpu, self.input.ptr(), self.output.ptr(), BLOCK)?;
            gpu.stream.sync()?;
            let out = self
                .output
                .download::<u16>(BLOCK * H)?
                .into_iter()
                .map(from_bf16);
            x.iter_mut().zip(out).for_each(|(dst, v)| *dst += v);
        }
        let table = ckpt.bytes("markov_head.markov_w1.weight")?;
        let candidate_count = if self.head_vocab == 0 {
            vocab
        } else {
            vocab.min(self.head_vocab)
        };
        let mut previous = anchor;
        let mut state = [0.0; R];
        let mut draft = Vec::with_capacity(limit);
        for row in x.chunks_exact(H).skip(1).take(limit) {
            let hidden = normalized(row, Some(&self.final_norm), EPS);
            let offset = previous as usize * 2 * R;
            let mut joint = state.to_vec();
            joint.extend(
                table[offset..offset + 2 * R]
                    .chunks_exact(2)
                    .map(|b| from_bf16(u16::from_le_bytes([b[0], b[1]]))),
            );
            joint.extend_from_slice(&hidden);
            joint.extend_from_slice(&cache.conditioning);
            let mut rnn = self.project(gpu, &self.rnn, &joint, 1)?;
            rnn.iter_mut().zip(&self.bias).for_each(|(a, b)| *a += b);
            for j in 0..R {
                let gate = 1.0 / (1.0 + (-rnn[j]).exp());
                let cand = rnn[R + j].tanh();
                state[j] = cand + gate * (state[j] - cand);
            }
            let delta: Vec<_> = rnn[2 * R..].iter().map(|x| x.tanh()).collect();
            let correction = self.project(gpu, &self.correction, &delta, 1)?;
            let merged: Vec<_> = hidden
                .iter()
                .zip(correction)
                .zip(&self.correction_bias)
                .map(|((a, b), c)| a + b + c)
                .collect();
            previous = gpu.argmax(head, &merged, candidate_count)?;
            draft.push(previous);
        }
        Ok(draft)
    }
}
