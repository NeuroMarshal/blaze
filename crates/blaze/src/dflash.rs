//! DFlash2 block drafter, adapted from vLLM qwen3_dflash / qwen3_dflash2.
//! Tensor-parallel ranks merge vocabulary candidates before the selector walk.

use crate::gpu::{DevBuf, HostMapped, Stream, check};
use crate::graph::GraphCache;
use crate::linear::{Cut, LinCtx, Linear};
use crate::model::Model;
use crate::quant::QuantConfig;
use crate::speculation::{DraftItem, DraftPlan};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Result, anyhow, bail};
use blaze_sys as sys;
use serde_json::Value;
use std::ffi::c_int;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DFlashConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub layers: usize,
    pub eps: f32,
    pub theta: f64,
    pub window: usize,
    pub causal: bool,
    /// query tokens per block: the anchor and block - 1 mask tokens
    pub block: usize,
    pub mask_token: i64,
    pub group: usize,
    pub taps: usize,
    pub rank: usize,
    pub top_k: usize,
    /// target layers whose output the draft reads (DFlash ids)
    pub target_layers: Vec<usize>,
    pub embed_scale: f32,
    pub vocab: usize,
}

pub(crate) struct Target<'a> {
    pub hidden: usize,
    pub vocab: usize,
    pub local_vocab: usize,
    pub valid_vocab: usize,
    pub rank: usize,
    pub world: usize,
    pub sms: usize,
    pub ar: *mut sys::blz_host_ar,
    pub stream: &'a Stream,
    pub linear: &'a LinCtx,
    pub features: Option<&'a DevBuf>,
    pub embedding: *const u16,
    pub embedding_norm: Option<&'a DevBuf>,
    pub head: &'a dyn Fn(*const u16, usize, *mut u16) -> Result<()>,
}

impl Model {
    fn dflash_target<'a>(&'a self, head: &'a dyn Fn(*const u16, usize, *mut u16) -> Result<()>) -> Target<'a> {
        Target {
            hidden: self.cfg.hidden,
            vocab: self.cfg.vocab,
            local_vocab: self.dims.vocab,
            valid_vocab: self.dims.vocab,
            rank: self.tp.rank,
            world: self.tp.world,
            sms: self.sms(),
            ar: self.tp.ar,
            stream: &self.stream,
            linear: self.lin_ctx(),
            features: self.aux.as_ref(),
            embedding: self.embed_table().dev_ptr(),
            embedding_norm: None,
            head,
        }
    }
}

impl DFlashConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let arch = v["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "DFlash2DraftModel" {
            bail!("{}: architecture {arch} (DFlash2DraftModel expected)", dir.display());
        }
        let u = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("DFlash config: no {k}"));
        let d = &v["dflash_config"];
        let du = |k: &str| d[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("DFlash config: no dflash_config.{k}"));
        let layer_types: Vec<&str> = v["layer_types"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        if layer_types.iter().any(|t| *t != "sliding_attention") {
            bail!("DFlash config: layer types {layer_types:?} (sliding attention only supported)");
        }
        Ok(Self {
            hidden: u("hidden_size")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            inter: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            theta: v["rope_parameters"]["rope_theta"].as_f64().or_else(|| v["rope_theta"].as_f64()).unwrap_or(1e6),
            window: u("sliding_window")?,
            causal: v["is_causal"].as_bool().unwrap_or(false),
            block: du("block_size")?,
            mask_token: d["mask_token_id"].as_i64().ok_or_else(|| anyhow!("DFlash config: no mask_token_id"))?,
            group: du("conv_group_size")?,
            taps: du("conv_kernel_size")?,
            rank: du("selector_rank")?,
            top_k: du("selector_top_k")?,
            target_layers: d["target_layer_ids"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                .unwrap_or_default(),
            embed_scale: d["input_embedding_scale"].as_f64().unwrap_or(1.0) as f32,
            vocab: u("vocab_size")?,
        })
    }

    /// Target layers whose input (the residual stream) the target captures: one after each DFlash id (vLLM's
    /// get_eagle3_aux_layers_from_config).
    pub fn capture_layers(&self) -> Vec<usize> {
        self.target_layers.iter().map(|&l| l + 1).collect()
    }

    /// Tokens one call drafts.
    pub fn steps(&self) -> usize {
        self.block - 1
    }
}

/// Grouped convolution of one side pair: base kernels [2, taps, hidden] and the projection picking per-row deltas.
struct Conv {
    base: DevBuf,
    proj: Linear,
}

struct Layer {
    input_ln: DevBuf,
    post_ln: DevBuf,
    qkv: Linear,
    kv: Linear,
    q_norm: DevBuf,
    k_norm: DevBuf,
    o: Linear,
    gu: Linear,
    down: Linear,
    att_conv: Conv,
    mlp_conv: Conv,
    k_pool: DevBuf,
    v_pool: DevBuf,
}

/// Context rows of one forward of the KV precompute.
const ROWS: usize = 512;

struct BlockMeta {
    positions: *const i64,
    slots: *const i64,
    requests: *const i32,
    lengths: *const i32,
    tokens: *const i64,
    anchors: *const i64,
}

pub struct DFlash {
    graphs: GraphCache<(usize, usize)>,
    pub cfg: DFlashConfig,
    fc: Linear,
    hidden_norm: DevBuf,
    norm: DevBuf,
    layers: Vec<Layer>,
    hproj: Linear,
    pred: DevBuf,
    succ: DevBuf,
    inv_freq: DevBuf,
    one: DevBuf,
    no_scales: DevBuf,
    r2t: DevBuf,
    stride: usize,
    max_items: usize,
    heads: usize,
    kv_heads: usize,
    inter: usize,
    ar: *mut sys::blz_host_ar,
    rank: usize,
    world: usize,
    // context [ROWS, ...]
    ctx_h: DevBuf,
    ctx_n: DevBuf,
    ctx_kv: DevBuf,
    // block [items * block, ...]
    h: DevBuf,
    residual: DevBuf,
    x: DevBuf,
    coef: DevBuf,
    xc: DevBuf,
    qkv_out: DevBuf,
    attn: DevBuf,
    o_out: DevBuf,
    gu_out: DevBuf,
    act: DevBuf,
    partial_o: DevBuf,
    partial_ml: DevBuf,
    // candidates [items * steps, ...]
    hm: DevBuf,
    logits: DevBuf,
    cand_vals: DevBuf,
    cand_ids: DevBuf,
    cand_pairs: DevBuf,
    sel_ids: DevBuf,
    sel_unary: DevBuf,
    sel_hproj: DevBuf,
    drafts: DevBuf,
    draft_readback: HostMapped,
    /// per-call metadata: written into the pinned staging at a bump offset, copied to `meta` asynchronously (the
    /// staging is reused only after the call's final synchronization)
    staging: HostMapped,
    meta: DevBuf,
    block_meta_start: usize,
    staged: usize,
}

// The all-reduce handle is used only by the rank's own thread.
unsafe impl Send for DFlash {}

impl DFlash {
    /// Window splits of the block attention: about four CTAs per SM over the call's query tiles and KV heads, within
    /// the partial workspace (one CTA per tile walked the whole 2048-position window).
    fn attn_splits(&self, sms: usize, items: usize, q_len: usize) -> usize {
        let (heads, kvh, hd) = (self.heads, self.kv_heads, self.cfg.head_dim);
        let group = heads / kvh;
        let mt = (q_len * group).div_ceil(16).min(3);
        let tiles = q_len.div_ceil((16 * mt / group).max(1));
        let cap = self.partial_o.bytes() / (heads * hd * 4);
        let mut splits = (4 * sms / (items * tiles * kvh)).clamp(1, 16);
        while splits > 1
            && items * q_len * unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kvh as c_int, q_len as c_int, splits as c_int) } as usize > cap
        {
            splits /= 2;
        }
        splits
    }

    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize) -> Result<Self> {
        Self::load_target(dir, &model.dflash_target(&|_, _, _| unreachable!()), max_items, stride)
    }

    pub(crate) fn load_target(dir: &Path, target: &Target<'_>, max_items: usize, stride: usize) -> Result<Self> {
        let cfg = DFlashConfig::load(dir)?;
        let s = target.stream;
        let ckpt = Checkpoint::open(dir)?;
        let root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let hf_quant = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        let quant = QuantConfig::from_model_config(&root, hf_quant.as_ref())?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        if hd != 128 || cfg.vocab != target.vocab || h != target.hidden {
            bail!("DFlash: head dim {hd}, vocabulary {}, hidden {h} (128, the target's {} and {})", cfg.vocab, target.vocab, target.hidden);
        }
        let (rank, world) = (target.rank, target.world);
        if cfg.heads % world != 0 || cfg.kv_heads % world != 0 || cfg.inter % world != 0 {
            bail!("DFlash: heads / KV heads / MLP rows do not split {world} ways");
        }
        let (heads, kv_heads, inter) = (cfg.heads / world, cfg.kv_heads / world, cfg.inter / world);
        let vec = |name: &str| -> Result<DevBuf> {
            let t = ckpt.info(name)?;
            if t.dtype != Dtype::Bf16 {
                bail!("{name}: bf16 expected");
            }
            let b = DevBuf::zeroed(t.bytes(), s)?;
            b.upload(ckpt.bytes(name)?)?;
            Ok(b)
        };
        let lin = |mods: &[(String, Cut)]| Linear::load(&ckpt, &quant, mods, s);
        let part = |n: usize| Cut::Rows(vec![rank * n..(rank + 1) * n]);
        let (qr, kr) = (heads * hd, kv_heads * hd);
        let w = cfg.window;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let a = format!("{p}self_attn.");
            let conv = |name: &str| -> Result<Conv> {
                Ok(Conv { base: vec(&format!("{p}{name}.base_kernel"))?, proj: lin(&[(format!("{p}{name}.kernel_projection"), Cut::All)])? })
            };
            layers.push(Layer {
                input_ln: vec(&format!("{p}input_layernorm.weight"))?,
                post_ln: vec(&format!("{p}post_attention_layernorm.weight"))?,
                qkv: lin(&[(format!("{a}q_proj"), part(qr)), (format!("{a}k_proj"), part(kr)), (format!("{a}v_proj"), part(kr))])?,
                kv: lin(&[(format!("{a}k_proj"), part(kr)), (format!("{a}v_proj"), part(kr))])?,
                q_norm: vec(&format!("{a}q_norm.weight"))?,
                k_norm: vec(&format!("{a}k_norm.weight"))?,
                o: lin(&[(format!("{a}o_proj"), Cut::Cols(rank * qr..(rank + 1) * qr))])?,
                gu: lin(&[(format!("{p}mlp.gate_proj"), part(inter)), (format!("{p}mlp.up_proj"), part(inter))])?,
                down: lin(&[(format!("{p}mlp.down_proj"), Cut::Cols(rank * inter..(rank + 1) * inter))])?,
                att_conv: conv("attention_conv")?,
                mlp_conv: conv("mlp_conv")?,
                k_pool: DevBuf::zeroed(max_items * w * kr * 2, s)?,
                v_pool: DevBuf::zeroed(max_items * w * kr * 2, s)?,
            });
        }
        let freqs: Vec<f32> = (0..hd / 2).map(|i| cfg.theta.powf(-2.0 * i as f64 / hd as f64) as f32).collect();
        let table: Vec<i32> = (0..max_items).flat_map(|r| (0..stride).map(move |p| (r * w + p % w) as i32)).collect();
        let b = |n: usize| DevBuf::zeroed(n, s);
        let nq = max_items * cfg.block;
        let ns = max_items * cfg.steps();
        let groups = h / cfg.group;
        let v_local = target.local_vocab;
        // the block attention splits its window up to 16 ways (attn_splits)
        let parts = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kv_heads as c_int, cfg.block as c_int, 16) } as usize;
        let k = cfg.top_k;
        // a call's metadata: positions and slots of every context row (at most `window` per item), the blocks'
        // positions, slots, tokens, rows, lengths, anchors
        let block_meta_start = (max_items * (w * 16 + 128)).next_multiple_of(64);
        let meta_bytes = (block_meta_start + nq * 24 + max_items * 16 + 64 * 6).next_multiple_of(4096);
        Ok(Self {
            graphs: GraphCache::new(16),
            fc: lin(&[("fc".to_string(), Cut::All)])?,
            hidden_norm: vec("hidden_norm.weight")?,
            norm: vec("norm.weight")?,
            hproj: lin(&[("candidate_selector.hidden_projection".to_string(), Cut::All)])?,
            pred: vec("candidate_selector.predecessor_codebook")?,
            succ: vec("candidate_selector.successor_codebook")?,
            layers,
            inv_freq: DevBuf::from_slice(&freqs)?,
            one: DevBuf::from_slice(&[1.0f32])?,
            no_scales: b(16)?,
            r2t: DevBuf::from_slice(&table)?,
            stride,
            max_items,
            heads,
            kv_heads,
            inter,
            ar: if world > 1 { target.ar } else { std::ptr::null_mut() },
            rank,
            world,
            ctx_h: b(ROWS * h * 2)?,
            ctx_n: b(ROWS * h * 2)?,
            ctx_kv: b(ROWS * 2 * kr * 2)?,
            h: b(nq * h * 2)?,
            residual: b(nq * h * 2)?,
            x: b(nq * h * 2)?,
            coef: b(nq * 2 * cfg.taps * groups * 2)?,
            xc: b(nq * h * 2)?,
            qkv_out: b(nq * (qr + 2 * kr) * 2)?,
            attn: b(nq * qr * 2)?,
            o_out: b(nq * h * 2)?,
            gu_out: b(nq * 2 * inter * 2)?,
            act: b(nq * inter * 2)?,
            partial_o: b(nq * heads * parts * hd * 4)?,
            partial_ml: b(nq * heads * parts * 2 * 4)?,
            hm: b(ns * h * 2)?,
            logits: b(ns * v_local * 2)?,
            cand_vals: b(ns * k * 4)?,
            cand_ids: b(ns * k * 4)?,
            cand_pairs: b((world * ns * (k + 1) * 8).next_multiple_of(16))?,
            sel_ids: b(ns * k * 4)?,
            sel_unary: b(ns * k * 4)?,
            sel_hproj: b(ns * cfg.rank * 2)?,
            drafts: b(ns * 8)?,
            draft_readback: HostMapped::new(ns * size_of::<i64>())?,
            staging: HostMapped::new(meta_bytes)?,
            meta: b(meta_bytes)?,
            block_meta_start,
            staged: 0,
            cfg,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "DFlash2 draft: {} layers, block {}, window {}, top {} candidates, selector rank {}",
            self.cfg.layers, self.cfg.block, self.cfg.window, self.cfg.top_k, self.cfg.rank
        )
    }

    /// Copy `data` into the call's metadata (pinned staging, an async copy on the stream); its device pointer.
    fn stage_host<T: Copy>(&mut self, data: &[T], limit: usize) -> Result<*const T> {
        let bytes = std::mem::size_of_val(data);
        let at = self.staged;
        let next = (at + bytes).next_multiple_of(64);
        if next > limit {
            bail!("DFlash metadata of {} bytes exceeds its buffer", at + bytes);
        }
        // SAFETY: plain-old-data copy into the pinned staging buffer.
        self.staging.host_slice_mut()[at..at + bytes].copy_from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes) });
        self.staged = next;
        Ok(self.meta.at::<u8>(at).cast())
    }

    fn copy_meta(&self, s: &Stream, at: usize, bytes: usize) -> Result<()> {
        let host: *const u8 = self.staging.host_ptr().cast();
        check(
            unsafe { sys::cudaMemcpyAsync(self.meta.at::<u8>(at).cast(), host.add(at).cast(), bytes, sys::MEMCPY_H2D, s.0) },
            "DFlash meta copy",
        )
    }

    fn stage<T: Copy>(&mut self, s: &Stream, data: &[T]) -> Result<*const T> {
        let at = self.staged;
        let ptr = self.stage_host(data, self.block_meta_start)?;
        self.copy_meta(s, at, std::mem::size_of_val(data))?;
        Ok(ptr)
    }

    /// Extend the context with the last target step, then score the requested prefix of each draft block.
    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        anyhow::ensure!(plan.items.is_empty() || model.aux.is_some(), "DFlash without captured hidden states");
        let hidden = self.cfg.hidden;
        let head = |input, rows, output| model.lm_head().run(model.lin_ctx(), input, hidden,
            rows, output, model.dims.vocab, &model.stream);
        self.run_target(&model.dflash_target(&head), plan)
    }

    pub(crate) fn append_target(&mut self, target: &Target<'_>, item: &DraftItem) -> Result<()> {
        self.staged = 0;
        let submit = self.context(target, item);
        let waited = target.stream.sync();
        submit?;
        waited
    }

    pub(crate) fn run_target(&mut self, target: &Target<'_>, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        let n = plan.items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.max_items {
            bail!("DFlash draft of {n} items (max {})", self.max_items);
        }
        // the previous call ended with a synchronization: its staging is free
        self.staged = 0;
        let steps = plan.k.min(self.cfg.steps());
        let count = n * steps;
        let enqueue = (|| -> Result<()> {
            if target.features.is_some() {
                for it in &plan.items {
                    self.context(target, it)?;
                }
            }
            if steps > 0 {
                let meta = self.prepare_block(target.stream, &plan.items)?;
                let forward = || {
                    self.block(target, n, &meta, steps)?;
                    self.candidates(target, n, steps, meta.anchors)
                };
                if self.world == 1 {
                    self.graphs.launch((n, steps), target.stream, forward)?;
                } else {
                    forward()?;
                }
                check(unsafe {
                    sys::cudaMemcpyAsync(
                        self.draft_readback.host_ptr().cast_mut(),
                        self.drafts.ptr(),
                        count * size_of::<i64>(),
                        sys::MEMCPY_D2H,
                        target.stream.0,
                    )
                }, "DFlash draft readback")?;
            }
            Ok(())
        })();
        let waited = target.stream.sync();
        enqueue?;
        waited?;
        if steps == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let drafts = unsafe {
            std::slice::from_raw_parts(self.draft_readback.host_ptr().cast::<i64>(), count)
        };
        Ok(drafts.chunks_exact(steps).map(<[i64]>::to_vec).collect())
    }

    /// K/V of one item's context rows (its last `window` ones: no query reads further back) into every layer's ring.
    fn context(&mut self, target: &Target<'_>, it: &DraftItem) -> Result<()> {
        let cfg = self.cfg.clone();
        let (c, s, lin) = (&cfg, target.stream, target.linear);
        let (h, hd) = (c.hidden, c.head_dim);
        let kr = self.kv_heads * hd;
        let feats = c.target_layers.len() * h;
        let aux = target.features.ok_or_else(|| anyhow!("DFlash without captured hidden states"))?;
        let rows = it.tokens.len();
        let first = rows.saturating_sub(c.window);
        let w = c.window;
        for r0 in (first..rows).step_by(ROWS) {
            let m = (rows - r0).min(ROWS);
            let pos: Vec<i64> = (0..m).map(|j| (it.pos0 + r0 + j) as i64).collect();
            let loc: Vec<i64> = pos.iter().map(|&p| (it.row * w + p as usize % w) as i64).collect();
            let posp = self.stage(s, &pos)?;
            let locp = self.stage(s, &loc)?;
            // fc in pieces the NVFP4 GEMV takes directly (its BF16 widening would need more scratch than the
            // target's layers provide)
            for j in (0..m).step_by(sys::BLZ_NVFP4_MMA_MAX_ROWS) {
                let mj = (m - j).min(sys::BLZ_NVFP4_MMA_MAX_ROWS);
                self.fc.run(lin, aux.at::<u16>((it.aux_row + r0 + j) * feats), feats, mj, self.ctx_h.at::<u16>(j * h), h, s)?;
            }
            check(
                unsafe { sys::blz_rmsnorm(self.ctx_h.ptr(), std::ptr::null_mut(), self.hidden_norm.ptr(), c.eps, m as c_int, h as c_int, 0, self.ctx_n.ptr(), s.0) },
                "DFlash hidden norm",
            )?;
            for l in &self.layers {
                l.kv.run(lin, self.ctx_n.ptr(), h, m, self.ctx_kv.ptr(), 2 * kr, s)?;
                let k: *mut u16 = self.ctx_kv.ptr();
                let v: *const u16 = self.ctx_kv.at(kr);
                check(unsafe { sys::blz_head_rmsnorm(k, (2 * kr) as i64, m as c_int, self.kv_heads as c_int, hd as c_int, l.k_norm.ptr(), c.eps, s.0) }, "DFlash k norm")?;
                check(unsafe { sys::blz_rope_neox(k, (2 * kr) as i64, m as c_int, self.kv_heads as c_int, hd as c_int, posp, self.inv_freq.ptr(), s.0) }, "DFlash rope k")?;
                check(
                    unsafe {
                        sys::blz_kv_store_dense(k.cast(), v.cast(), (2 * kr) as i64, (2 * kr) as i64, locp.cast(), sys::BLZ_I64, m as i64,
                                                self.kv_heads as c_int, hd as c_int, sys::BLZ_KV_BF16, l.k_pool.ptr(), l.v_pool.ptr(),
                                                self.one.ptr(), self.one.ptr(), s.0)
                    },
                    "DFlash context kv",
                )?;
            }
        }
        Ok(())
    }

    fn prepare_block(&mut self, stream: &Stream, items: &[DraftItem]) -> Result<BlockMeta> {
        let bl = self.cfg.block;
        let w = self.cfg.window;
        let mask_token = self.cfg.mask_token;
        let n = items.len();
        let nq = n * bl;
        // query tokens and positions: the anchor (the token after the item's last row) and masks after it
        let mut tok = Vec::with_capacity(nq);
        let mut pos = Vec::with_capacity(nq);
        let mut loc = Vec::with_capacity(nq);
        let mut req = Vec::with_capacity(n);
        let mut kvl = Vec::with_capacity(n);
        for it in items {
            let p0 = it.pos0 + it.tokens.len();
            for j in 0..bl {
                tok.push(if j == 0 { *it.tokens.last().unwrap() } else { mask_token });
                pos.push((p0 + j) as i64);
                loc.push((it.row * w + (p0 + j) % w) as i64);
            }
            req.push(it.row as i32);
            kvl.push((p0 + bl) as i32);
        }
        let anchors: Vec<i64> = items.iter().map(|it| *it.tokens.last().unwrap()).collect();
        // Graphs retain these device addresses across windows of the same shape.
        self.staged = self.block_meta_start;
        let limit = self.meta.bytes();
        let meta = BlockMeta {
            positions: self.stage_host(&pos, limit)?,
            slots: self.stage_host(&loc, limit)?,
            requests: self.stage_host(&req, limit)?,
            lengths: self.stage_host(&kvl, limit)?,
            tokens: self.stage_host(&tok, limit)?,
            anchors: self.stage_host(&anchors, limit)?,
        };
        self.copy_meta(stream, self.block_meta_start, self.staged - self.block_meta_start)?;
        Ok(meta)
    }

    /// Runs the anchor and mask rows through every layer, then gathers the requested rows for the head.
    fn block(&self, target: &Target<'_>, n: usize, meta: &BlockMeta, steps: usize) -> Result<()> {
        let (c, s, lin) = (&self.cfg, target.stream, target.linear);
        let (h, hd, bl) = (c.hidden, c.head_dim, c.block);
        let nq = n * bl;
        let (qr, kr) = (self.heads * hd, self.kv_heads * hd);
        let ldq = qr + 2 * kr;
        let groups = h / c.group;
        check(
            unsafe { sys::blz_embed_gather(meta.tokens, nq as i64, target.embedding.cast(), c.vocab as i64, (h * 2) as i64, self.h.ptr(), s.0) },
            "DFlash embed",
        )?;
        if let Some(norm) = target.embedding_norm {
            check(unsafe { sys::blz_rmsnorm(self.h.ptr(), std::ptr::null_mut(), norm.ptr(), c.eps, nq as c_int, h as c_int, 0, self.x.ptr(), s.0) }, "DFlash target embedding norm")?;
            check(unsafe { sys::cudaMemcpyAsync(self.h.ptr(), self.x.ptr(), nq * h * 2, sys::MEMCPY_D2D, s.0) }, "DFlash normalized embedding")?;
        }
        if c.embed_scale != 1.0 {
            bail!("DFlash: input_embedding_scale {} is not supported yet", c.embed_scale);
        }
        let conv = |cv: &Conv, side: usize, x: *const u16, out: *mut u16| -> Result<()> {
            let delta: *const u16 = self.coef.at(side * c.taps * groups);
            check(
                unsafe {
                    sys::blz_dflash_grouped_conv(x, h as i64, delta, (2 * c.taps * groups) as i64, groups as i64,
                                                 cv.base.at::<u16>(side * c.taps * h), out, h as i64, nq as c_int, h as c_int,
                                                 bl as c_int, c.group as c_int, c.taps as c_int, s.0)
                },
                "DFlash conv",
            )
        };
        let all_reduce = |p: *mut u16| -> Result<()> {
            if self.ar.is_null() {
                return Ok(());
            }
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, p.cast(), (nq * h * 2) as i64, sys::BLZ_AR_BF16, s.0) }, "DFlash all-reduce")
        };
        for (li, l) in self.layers.iter().enumerate() {
            // x = input_norm(h (+ residual)); residual = the sum
            if li == 0 {
                check(unsafe { sys::cudaMemcpyAsync(self.residual.ptr(), self.h.ptr::<u8>().cast(), nq * h * 2, sys::MEMCPY_D2D, s.0) }, "DFlash residual")?;
                check(unsafe { sys::blz_rmsnorm(self.h.ptr(), std::ptr::null_mut(), l.input_ln.ptr(), c.eps, nq as c_int, h as c_int, 0, self.x.ptr(), s.0) }, "DFlash norm")?;
            } else {
                check(unsafe { sys::blz_rmsnorm(self.h.ptr(), self.residual.ptr(), l.input_ln.ptr(), c.eps, nq as c_int, h as c_int, 0, self.x.ptr(), s.0) }, "DFlash norm")?;
            }
            // attention conv (prepare), attention, conv (finish)
            l.att_conv.proj.run(lin, self.x.ptr(), h, nq, self.coef.ptr(), 2 * c.taps * groups, s)?;
            conv(&l.att_conv, 0, self.x.ptr(), self.xc.ptr())?;
            l.qkv.run(lin, self.xc.ptr(), h, nq, self.qkv_out.ptr(), ldq, s)?;
            let q: *mut u16 = self.qkv_out.ptr();
            let k: *mut u16 = self.qkv_out.at(qr);
            let v: *const u16 = self.qkv_out.at(qr + kr);
            check(unsafe { sys::blz_head_rmsnorm(q, ldq as i64, nq as c_int, self.heads as c_int, hd as c_int, l.q_norm.ptr(), c.eps, s.0) }, "DFlash q norm")?;
            check(unsafe { sys::blz_head_rmsnorm(k, ldq as i64, nq as c_int, self.kv_heads as c_int, hd as c_int, l.k_norm.ptr(), c.eps, s.0) }, "DFlash k norm")?;
            check(unsafe { sys::blz_rope_neox(q, ldq as i64, nq as c_int, self.heads as c_int, hd as c_int, meta.positions, self.inv_freq.ptr(), s.0) }, "DFlash rope q")?;
            check(unsafe { sys::blz_rope_neox(k, ldq as i64, nq as c_int, self.kv_heads as c_int, hd as c_int, meta.positions, self.inv_freq.ptr(), s.0) }, "DFlash rope k")?;
            check(
                unsafe {
                    sys::blz_kv_store_dense(k.cast(), v.cast(), ldq as i64, ldq as i64, meta.slots.cast(), sys::BLZ_I64, nq as i64, self.kv_heads as c_int,
                                            hd as c_int, sys::BLZ_KV_BF16, l.k_pool.ptr(), l.v_pool.ptr(), self.one.ptr(), self.one.ptr(), s.0)
                },
                "DFlash block kv",
            )?;
            let pool = sys::blz_nvfp4_pool {
                k_data: l.k_pool.ptr(),
                v_data: l.v_pool.ptr(),
                k_scale: self.no_scales.ptr(),
                v_scale: self.no_scales.ptr(),
                k_g: self.one.ptr(),
                v_g: self.one.ptr(),
                heads: self.kv_heads as c_int,
                dim: hd as c_int,
                format: sys::BLZ_KV_BF16,
            };
            check(
                unsafe {
                    sys::blz_mha_decode_window(
                        q.cast(),
                        ldq as i64,
                        hd as i64,
                        self.heads as c_int,
                        pool,
                        self.r2t.ptr(),
                        self.stride as i64,
                        sys::blz_index_ptr { ptr: meta.requests.cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: meta.lengths.cast(), kind: sys::BLZ_I32 },
                        n as c_int,
                        bl as c_int,
                        self.attn_splits(target.sms, n, bl) as c_int,
                        1.0 / (hd as f32).sqrt(),
                        c.window as c_int,
                        c.causal as c_int,
                        self.partial_o.ptr(),
                        self.partial_ml.ptr(),
                        sys::BLZ_BF16,
                        self.attn.ptr::<u16>().cast(),
                        qr as i64,
                        hd as i64,
                        s.0,
                    )
                },
                "DFlash attention",
            )?;
            l.o.run(lin, self.attn.ptr(), qr, nq, self.o_out.ptr(), h, s)?;
            all_reduce(self.o_out.ptr())?;
            conv(&l.att_conv, 1, self.o_out.ptr(), self.h.ptr())?;
            // x = post_norm(h + residual); MLP conv (prepare), MLP, conv (finish)
            check(unsafe { sys::blz_rmsnorm(self.h.ptr(), self.residual.ptr(), l.post_ln.ptr(), c.eps, nq as c_int, h as c_int, 0, self.x.ptr(), s.0) }, "DFlash post norm")?;
            l.mlp_conv.proj.run(lin, self.x.ptr(), h, nq, self.coef.ptr(), 2 * c.taps * groups, s)?;
            conv(&l.mlp_conv, 0, self.x.ptr(), self.xc.ptr())?;
            l.gu.run(lin, self.xc.ptr(), h, nq, self.gu_out.ptr(), 2 * self.inter, s)?;
            check(
                unsafe {
                    sys::blz_silu_mul(self.gu_out.ptr(), (2 * self.inter) as i64, self.gu_out.at(self.inter), (2 * self.inter) as i64, nq as c_int,
                                      self.inter as c_int, self.act.ptr(), s.0)
                },
                "DFlash silu",
            )?;
            l.down.run(lin, self.act.ptr(), self.inter, nq, self.o_out.ptr(), h, s)?;
            all_reduce(self.o_out.ptr())?;
            conv(&l.mlp_conv, 1, self.o_out.ptr(), self.h.ptr())?;
        }
        // final norm of h + residual; the mask rows (1 .. block of every item) to `hm`
        check(unsafe { sys::blz_rmsnorm(self.h.ptr(), self.residual.ptr(), self.norm.ptr(), c.eps, nq as c_int, h as c_int, 0, self.x.ptr(), s.0) }, "DFlash final norm")?;
        check(
            unsafe {
                sys::cudaMemcpy2DAsync(self.hm.ptr(), steps * h * 2, self.x.at::<u16>(h).cast(), bl * h * 2, steps * h * 2, n, sys::MEMCPY_D2D, s.0)
            },
            "DFlash mask rows",
        )
    }

    /// Top-k of the target's LM head at every mask row (merged over the ranks), the selector's projection, the walk.
    fn candidates(&self, target: &Target<'_>, n: usize, steps: usize, anchors: *const i64) -> Result<()> {
        let (c, s, lin) = (&self.cfg, target.stream, target.linear);
        let (h, k) = (c.hidden, c.top_k);
        let ns = n * steps;
        let v_local = target.local_vocab;
        (target.head)(self.hm.ptr(), ns, self.logits.ptr())?;
        check(
            unsafe {
                sys::blz_topk(self.logits.ptr(), v_local as i64, ns as c_int, target.valid_vocab as c_int, (self.rank * v_local) as c_int, k as c_int,
                              self.cand_vals.ptr(), self.cand_ids.ptr(), std::ptr::null_mut(), s.0)
            },
            "DFlash candidates",
        )?;
        let (ids, unary) = if self.world == 1 {
            (self.cand_ids.ptr(), self.cand_vals.ptr())
        } else {
            let k1 = k + 1;
            let bytes = (self.world * ns * k1 * 8).next_multiple_of(16);
            check(unsafe { sys::cudaMemsetAsync(self.cand_pairs.ptr(), 0, bytes, s.0) }, "DFlash rank staging")?;
            check(
                unsafe {
                    sys::blz_topk_pack(self.cand_vals.ptr(), self.cand_ids.ptr(), std::ptr::null(), ns as c_int, k as c_int,
                                       self.cand_pairs.at::<f32>(self.rank * ns * k1 * 2), s.0)
                },
                "DFlash pack",
            )?;
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, self.cand_pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "DFlash all-reduce")?;
            check(
                unsafe { sys::blz_cand_merge(self.cand_pairs.ptr(), self.world as c_int, ns as c_int, k as c_int,
                    self.sel_ids.ptr(), self.sel_unary.ptr(), s.0) },
                "DFlash merge",
            )?;
            (self.sel_ids.ptr(), self.sel_unary.ptr())
        };
        self.hproj.run(lin, self.hm.ptr(), h, ns, self.sel_hproj.ptr(), c.rank, s)?;
        check(
            unsafe {
                sys::blz_dflash_walk(ids, unary, self.sel_hproj.ptr(), anchors, self.pred.ptr(),
                                     self.succ.ptr(), n as c_int, steps as c_int, k as c_int, c.rank as c_int, self.drafts.ptr(), s.0)
            },
            "DFlash walk",
        )
    }
}
