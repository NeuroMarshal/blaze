//! DFlash2 block drafter, adapted from vLLM qwen3_dflash / qwen3_dflash2.
//! Tensor-parallel ranks merge vocabulary candidates before the selector walk.

use crate::eagle::{DraftItem, DraftPlan};
use crate::gpu::{DevBuf, check};
use crate::linear::{Cut, Linear};
use crate::model::Model;
use crate::quant::QuantConfig;
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

pub struct DFlash {
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
    q_tok: DevBuf,
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
    cand_lse: DevBuf,
    cand_pairs: DevBuf,
    sel_ids: DevBuf,
    sel_unary: DevBuf,
    sel_hproj: DevBuf,
    anchors: DevBuf,
    drafts: DevBuf,
    meta: DevBuf,
}

// The all-reduce handle is used only by the rank's own thread.
unsafe impl Send for DFlash {}

impl DFlash {
    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize) -> Result<Self> {
        let cfg = DFlashConfig::load(dir)?;
        let s = &model.stream;
        let ckpt = Checkpoint::open(dir)?;
        let root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let hf_quant = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        let quant = QuantConfig::from_model_config(&root, hf_quant.as_ref())?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        if hd != 128 || cfg.vocab != model.cfg.vocab || h != model.cfg.hidden {
            bail!("DFlash: head dim {hd}, vocabulary {}, hidden {h} (128, the target's {} and {})", cfg.vocab, model.cfg.vocab, model.cfg.hidden);
        }
        let (rank, world) = (model.tp.rank, model.tp.world);
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
        let v_local = model.dims.vocab;
        let parts = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kv_heads as c_int, cfg.block as c_int, 1) } as usize;
        let k = cfg.top_k;
        Ok(Self {
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
            ar: if world > 1 { model.tp.ar } else { std::ptr::null_mut() },
            rank,
            world,
            ctx_h: b(ROWS * h * 2)?,
            ctx_n: b(ROWS * h * 2)?,
            ctx_kv: b(ROWS * 2 * kr * 2)?,
            q_tok: b(nq * 8)?,
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
            cand_lse: b(ns * 4)?,
            cand_pairs: b((world * ns * (k + 1) * 8).next_multiple_of(16))?,
            sel_ids: b(ns * k * 4)?,
            sel_unary: b(ns * k * 4)?,
            sel_hproj: b(ns * cfg.rank * 2)?,
            anchors: b(max_items * 8)?,
            drafts: b(ns * 8)?,
            meta: b((ROWS.max(nq) * 3 + max_items * 2) * 8 + 4096)?,
            cfg,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "DFlash2 draft: {} layers, block {}, window {}, top {} candidates, selector rank {}",
            self.cfg.layers, self.cfg.block, self.cfg.window, self.cfg.top_k, self.cfg.rank
        )
    }

    /// Context K/V of the items' rows (the last step's hidden states) into every layer, then the drafts after each
    /// item (`steps()` tokens; `plan.k` is not used: a block is drafted whole).
    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        let n = plan.items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.max_items {
            bail!("DFlash draft of {n} items (max {})", self.max_items);
        }
        for it in &plan.items {
            self.context(model, it)?;
        }
        if plan.k == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        self.block(model, &plan.items)?;
        self.candidates(model, n)?;
        let s = &model.stream;
        s.sync()?;
        let steps = self.cfg.steps();
        let d: Vec<i64> = self.drafts.download(n * steps)?;
        Ok((0..n).map(|i| d[i * steps..(i + 1) * steps].to_vec()).collect())
    }

    /// K/V of one item's context rows (its last `window` ones: no query reads further back) into every layer's ring.
    fn context(&self, model: &Model, it: &DraftItem) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd) = (c.hidden, c.head_dim);
        let kr = self.kv_heads * hd;
        let feats = c.target_layers.len() * h;
        let aux = model.aux.as_ref().ok_or_else(|| anyhow!("DFlash without captured hidden states"))?;
        let rows = it.tokens.len();
        let first = rows.saturating_sub(c.window);
        let w = c.window;
        for r0 in (first..rows).step_by(ROWS) {
            let m = (rows - r0).min(ROWS);
            let pos: Vec<i64> = (0..m).map(|j| (it.pos0 + r0 + j) as i64).collect();
            let loc: Vec<i64> = pos.iter().map(|&p| (it.row * w + p as usize % w) as i64).collect();
            // metadata: positions, slots (one synchronous upload: the stream is idle between draft calls' parts)
            s.sync()?;
            self.meta.upload_at(0, &pos)?;
            self.meta.upload_at(m * 8, &loc)?;
            let posp: *const i64 = self.meta.ptr();
            let locp: *const i64 = self.meta.at::<u8>(m * 8).cast();
            self.fc.run(lin, aux.at::<u16>((it.aux_row + r0) * feats), feats, m, self.ctx_h.ptr(), h, s)?;
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

    /// The query blocks: [anchor, mask...] per item through every layer; leaves the mask rows' final hidden in `hm`.
    fn block(&self, model: &Model, items: &[DraftItem]) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd, bl) = (c.hidden, c.head_dim, c.block);
        let n = items.len();
        let nq = n * bl;
        let (qr, kr) = (self.heads * hd, self.kv_heads * hd);
        let ldq = qr + 2 * kr;
        let groups = h / c.group;
        let w = c.window;
        // query tokens and positions: the anchor (the token after the item's last row) and masks after it
        let mut tok = Vec::with_capacity(nq);
        let mut pos = Vec::with_capacity(nq);
        let mut loc = Vec::with_capacity(nq);
        let mut req = Vec::with_capacity(n);
        let mut kvl = Vec::with_capacity(n);
        for it in items {
            let p0 = it.pos0 + it.tokens.len();
            for j in 0..bl {
                tok.push(if j == 0 { *it.tokens.last().unwrap() } else { c.mask_token });
                pos.push((p0 + j) as i64);
                loc.push((it.row * w + (p0 + j) % w) as i64);
            }
            req.push(it.row as i32);
            kvl.push((p0 + bl) as i32);
        }
        s.sync()?;
        self.meta.upload_at(0, &pos)?;
        self.meta.upload_at(nq * 8, &loc)?;
        self.meta.upload_at(2 * nq * 8, &req)?;
        self.meta.upload_at(2 * nq * 8 + n * 4, &kvl)?;
        self.q_tok.upload(&tok)?;
        let anchors: Vec<i64> = items.iter().map(|it| *it.tokens.last().unwrap()).collect();
        self.anchors.upload(&anchors)?;
        let posp: *const i64 = self.meta.ptr();
        let locp: *const i64 = self.meta.at::<u8>(nq * 8).cast();
        let reqp: *const i32 = self.meta.at::<u8>(2 * nq * 8).cast();
        let kvlp: *const i32 = self.meta.at::<u8>(2 * nq * 8 + n * 4).cast();
        let table = model.embed_table();
        check(
            unsafe { sys::blz_embed_gather(self.q_tok.ptr(), nq as i64, table.dev_ptr(), c.vocab as i64, (h * 2) as i64, self.h.ptr(), s.0) },
            "DFlash embed",
        )?;
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
            check(unsafe { sys::blz_rope_neox(q, ldq as i64, nq as c_int, self.heads as c_int, hd as c_int, posp, self.inv_freq.ptr(), s.0) }, "DFlash rope q")?;
            check(unsafe { sys::blz_rope_neox(k, ldq as i64, nq as c_int, self.kv_heads as c_int, hd as c_int, posp, self.inv_freq.ptr(), s.0) }, "DFlash rope k")?;
            check(
                unsafe {
                    sys::blz_kv_store_dense(k.cast(), v.cast(), ldq as i64, ldq as i64, locp.cast(), sys::BLZ_I64, nq as i64, self.kv_heads as c_int,
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
                        sys::blz_index_ptr { ptr: reqp.cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: kvlp.cast(), kind: sys::BLZ_I32 },
                        n as c_int,
                        bl as c_int,
                        1,
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
        let steps = c.steps();
        check(
            unsafe {
                sys::cudaMemcpy2DAsync(self.hm.ptr(), steps * h * 2, self.x.at::<u16>(h).cast(), bl * h * 2, steps * h * 2, n, sys::MEMCPY_D2D, s.0)
            },
            "DFlash mask rows",
        )
    }

    /// Top-k of the target's LM head at every mask row (merged over the ranks), the selector's projection, the walk.
    fn candidates(&self, model: &Model, n: usize) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, k) = (c.hidden, c.top_k);
        let ns = n * c.steps();
        let v_local = model.dims.vocab;
        model.lm_head().run(lin, self.hm.ptr(), h, ns, self.logits.ptr(), v_local, s)?;
        check(
            unsafe {
                sys::blz_topk(self.logits.ptr(), v_local as i64, ns as c_int, v_local as c_int, (self.rank * v_local) as c_int, k as c_int,
                              self.cand_vals.ptr(), self.cand_ids.ptr(), std::ptr::null_mut(), s.0)
            },
            "DFlash candidates",
        )?;
        let k1 = k + 1;
        let bytes = (self.world * ns * k1 * 8).next_multiple_of(16);
        check(unsafe { sys::cudaMemsetAsync(self.cand_pairs.ptr(), 0, bytes, s.0) }, "memset")?;
        check(
            unsafe {
                sys::blz_topk_pack(self.cand_vals.ptr(), self.cand_ids.ptr(), self.cand_lse.ptr(), ns as c_int, k as c_int,
                                   self.cand_pairs.at::<f32>(self.rank * ns * k1 * 2), s.0)
            },
            "DFlash pack",
        )?;
        if !self.ar.is_null() {
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, self.cand_pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "DFlash all-reduce")?;
        }
        self.hproj.run(lin, self.hm.ptr(), h, ns, self.sel_hproj.ptr(), c.rank, s)?;
        s.sync()?;
        let pairs: Vec<f32> = self.cand_pairs.download(self.world * ns * k1 * 2)?;
        // merge: the k best (value, id) of the ranks' lists per row (ties: lower id)
        let (mut ids, mut vals) = (Vec::with_capacity(ns * k), Vec::with_capacity(ns * k));
        for r in 0..ns {
            let mut all: Vec<(f32, i32)> = (0..self.world)
                .flat_map(|w| (0..k).map(move |j| (w, j)))
                .map(|(w, j)| {
                    let base = ((w * ns + r) * k1 + j) * 2;
                    (pairs[base], pairs[base + 1] as i32)
                })
                .filter(|&(_, id)| id >= 0)
                .collect();
            all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            all.truncate(k);
            while all.len() < k {
                all.push((f32::NEG_INFINITY, 0));
            }
            ids.extend(all.iter().map(|x| x.1));
            vals.extend(all.iter().map(|x| x.0));
        }
        self.sel_ids.upload(&ids)?;
        self.sel_unary.upload(&vals)?;
        check(
            unsafe {
                sys::blz_dflash_walk(self.sel_ids.ptr(), self.sel_unary.ptr(), self.sel_hproj.ptr(), self.anchors.ptr(), self.pred.ptr(),
                                     self.succ.ptr(), n as c_int, c.steps() as c_int, k as c_int, c.rank as c_int, self.drafts.ptr(), s.0)
            },
            "DFlash walk",
        )
    }
}
