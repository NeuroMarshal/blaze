//! EAGLE-4 drafter (eagle-hq/EAGLE-4, checkpoint class LlamaForCausalLMEagle4), on the GPU of rank 0 next to the
//! target. One Llama layer whose attention reads concat(norm(embed(next token)), norm(hidden)): the hidden is the
//! target's hidden states after three layers (each normed, through `fc`) for positions the target computed, the
//! draft's own normed output for positions it drafts. Sliding-window attention (BF16 KV in a ring of `window`
//! positions per sequence row), a low-rank candidate head (rank-r scores, top C) re-scored with the full LM head at the
//! candidates' rows only (the head stays in pinned host memory).
//!
//! A position p of the draft takes the target's features of p and the token at p + 1 and predicts the token at p + 2.
//! After every target step the draft is extended over the step's kept rows (`DraftItem`), which also yields the first
//! draft token; then it runs alone for the rest of the chain, on the GPU without host round trips.

use crate::gpu::{DevBuf, HostMapped, check};
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
pub struct DraftConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub theta: f64,
    pub window: usize,
    /// target layers whose output the draft reads
    pub aux_layers: Vec<usize>,
    pub candidates: usize,
    pub norm_output: bool,
    pub fc_norm: bool,
    pub vocab: usize,
}

impl DraftConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let arch = v["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "LlamaForCausalLMEagle4" {
            bail!("{}: architecture {arch} (LlamaForCausalLMEagle4 expected)", dir.display());
        }
        let u = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("draft config: no {k}"));
        let head = &v["efficient_lm_head_config"];
        if head["type"].as_str() != Some("low_rank") {
            bail!("draft config: efficient_lm_head_config type {:?} (low_rank supported)", head["type"]);
        }
        if u("num_hidden_layers")? != 1 {
            bail!("draft config: {} layers (one supported)", u("num_hidden_layers")?);
        }
        let aux_layers: Vec<usize> = v["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        if aux_layers.len() != 3 {
            bail!("draft config: {} aux hidden states (3 supported)", aux_layers.len());
        }
        let window = if v["use_sliding_window"].as_bool().unwrap_or(false) { u("sliding_window")? } else { 0 };
        Ok(Self {
            hidden: u("hidden_size")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            inter: u("intermediate_size")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            theta: v["rope_parameters"]["rope_theta"].as_f64().or_else(|| v["rope_theta"].as_f64()).unwrap_or(1e6),
            window,
            aux_layers,
            candidates: head["num_candidates"].as_u64().unwrap_or(128) as usize,
            norm_output: v["norm_output"].as_bool().unwrap_or(false),
            fc_norm: v["fc_norm"].as_bool().unwrap_or(false) || v["enable_fc_norm"].as_bool().unwrap_or(false),
            vocab: u("vocab_size")?,
        })
    }

    /// Target layers whose input (the residual stream) the target captures: the layer after each aux layer.
    pub fn capture_layers(&self) -> Vec<usize> {
        // BLAZE_EAGLE_AUX_SHIFT (experiment): capture the input of layer aux + shift instead of aux + 1
        let shift = std::env::var("BLAZE_EAGLE_AUX_SHIFT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        self.aux_layers.iter().map(|&l| l + shift).collect()
    }
}

/// Rows of the last target step to extend the draft over, for one sequence.
#[derive(Clone, Debug)]
pub struct DraftItem {
    /// sequence row (the draft's KV ring)
    pub row: usize,
    /// first row of the item in the last target step (its captured hidden states)
    pub aux_row: usize,
    /// the token after each extended row (the target's input at the next position)
    pub tokens: Vec<i64>,
    /// position of the first extended row
    pub pos0: usize,
}

/// One draft call: extend every item, then draft `k` tokens after each (0: extend only).
#[derive(Clone, Debug, Default)]
pub struct DraftPlan {
    pub items: Vec<DraftItem>,
    pub k: usize,
}

/// Rows of one draft forward; the chain holds at most this many draft tokens.
const ROWS: usize = 512;
pub const MAX_CHAIN: usize = 16;

/// Per-forward metadata, laid out in one staging buffer copied once per call.
struct Fwd {
    /// (item, rows) spans in order, rows of the forward = their sum
    spans: Vec<(usize, usize)>,
    /// offsets in the staging / device metadata buffer: positions, locs, tokens (i64 per row), req, kv_len (i32 per span)
    pos: usize,
    loc: usize,
    tok: usize,
    req: usize,
    kvl: usize,
}

pub struct Draft {
    pub cfg: DraftConfig,
    fc_norm: Vec<DevBuf>,
    fc: Linear,
    input_ln: DevBuf,
    hidden_ln: DevBuf,
    post_ln: DevBuf,
    norm: DevBuf,
    qkv: Linear,
    o: Linear,
    gu: Linear,
    down: Linear,
    lr_down: Linear,
    lr_head: Linear,
    lm_head: HostMapped,
    inv_freq: DevBuf,
    one: DevBuf,
    k_pool: DevBuf,
    v_pool: DevBuf,
    no_scales: DevBuf,
    r2t: DevBuf,
    stride: usize,
    max_items: usize,
    // work [ROWS, ...]
    hcat: DevBuf,
    hin: DevBuf,
    emb: DevBuf,
    cat: DevBuf,
    qkv_out: DevBuf,
    attn: DevBuf,
    o_out: DevBuf,
    x2: DevBuf,
    gu_out: DevBuf,
    act: DevBuf,
    mlp: DevBuf,
    normed: DevBuf,
    partial_o: DevBuf,
    partial_ml: DevBuf,
    // chain [max_items, ...]
    last_normed: DevBuf,
    last_next: DevBuf,
    lr: DevBuf,
    scores: DevBuf,
    cand_vals: DevBuf,
    cand_ids: DevBuf,
    cand_logits: DevBuf,
    tok: DevBuf,
    record: DevBuf,
    staging: HostMapped,
    meta: DevBuf,
}

impl Draft {
    /// Load the drafter of `dir` on the current GPU (that of `model`), for `max_items` sequence rows and contexts of
    /// `stride` positions.
    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize) -> Result<Self> {
        let cfg = DraftConfig::load(dir)?;
        let s = &model.stream;
        let ckpt = Checkpoint::open(dir)?;
        let quant = QuantConfig::from_model_config(&serde_json::json!({}), None)?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        if hd != 128 || cfg.window == 0 {
            bail!("draft: head dim {hd}, window {} (128 and a sliding window supported)", cfg.window);
        }
        if cfg.vocab != model.cfg.vocab || h != model.cfg.hidden {
            bail!("draft: vocabulary {} / hidden {} differ from the target's {} / {}", cfg.vocab, h, model.cfg.vocab, model.cfg.hidden);
        }
        let vec = |name: &str| -> Result<DevBuf> {
            let t = ckpt.info(name)?;
            if t.dtype != Dtype::Bf16 {
                bail!("{name}: bf16 expected");
            }
            let b = DevBuf::zeroed(t.bytes(), s)?;
            b.upload(ckpt.bytes(name)?)?;
            Ok(b)
        };
        let lin = |names: &[&str]| Linear::load(&ckpt, &quant, &names.iter().map(|n| (n.to_string(), Cut::All)).collect::<Vec<_>>(), s);
        let fc_norm = if cfg.fc_norm { (0..3).map(|i| vec(&format!("fc_norm.{i}.weight"))).collect::<Result<_>>()? } else { Vec::new() };
        let lm = ckpt.info("lm_head.weight")?.clone();
        let mut lm_head = HostMapped::new(lm.bytes())?;
        lm_head.host_slice_mut().copy_from_slice(ckpt.bytes("lm_head.weight")?);
        let freqs: Vec<f32> = (0..hd / 2).map(|i| cfg.theta.powf(-2.0 * i as f64 / hd as f64) as f32).collect();
        // ring slot of position p of row r: r * window + p % window
        let w = cfg.window;
        let table: Vec<i32> = (0..max_items).flat_map(|r| (0..stride).map(move |p| (r * w + p % w) as i32)).collect();
        let kv_row = cfg.kv_heads * hd * 2;
        let b = |n: usize| DevBuf::zeroed(n, s);
        let (qd, kvd) = (cfg.heads * hd, cfg.kv_heads * hd);
        let parts_decode = unsafe { sys::blz_nvfp4_mha_parts(cfg.heads as c_int, cfg.kv_heads as c_int, 1, 64) } as usize;
        let partial_rows = (ROWS * cfg.heads).max(max_items * cfg.heads * parts_decode);
        let meta_bytes = (MAX_CHAIN + 8) * (3 * 8 * ROWS + 2 * 4 * max_items.max(1) + 256) + 4096;
        let c = cfg.candidates;
        Ok(Self {
            fc: lin(&["fc"])?,
            fc_norm,
            input_ln: vec("layers.0.input_layernorm.weight")?,
            hidden_ln: vec("layers.0.hidden_norm.weight")?,
            post_ln: vec("layers.0.post_attention_layernorm.weight")?,
            norm: vec("norm.weight")?,
            qkv: lin(&["layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj"])?,
            o: lin(&["layers.0.self_attn.o_proj"])?,
            gu: lin(&["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"])?,
            down: lin(&["layers.0.mlp.down_proj"])?,
            lr_down: lin(&["efficient_lm_head_down_proj"])?,
            lr_head: lin(&["efficient_lm_head"])?,
            lm_head,
            inv_freq: DevBuf::from_slice(&freqs)?,
            one: DevBuf::from_slice(&[1.0f32])?,
            k_pool: b(max_items * w * kv_row)?,
            v_pool: b(max_items * w * kv_row)?,
            no_scales: b(16)?,
            r2t: DevBuf::from_slice(&table)?,
            stride,
            max_items,
            hcat: b(ROWS * 3 * h * 2)?,
            hin: b(ROWS * h * 2)?,
            emb: b(ROWS * h * 2)?,
            cat: b(ROWS * 2 * h * 2)?,
            qkv_out: b(ROWS * (qd + 2 * kvd) * 2)?,
            attn: b(ROWS * qd * 2)?,
            o_out: b(ROWS * h * 2)?,
            x2: b(ROWS * h * 2)?,
            gu_out: b(ROWS * 2 * cfg.inter * 2)?,
            act: b(ROWS * cfg.inter * 2)?,
            mlp: b(ROWS * h * 2)?,
            normed: b(ROWS * h * 2)?,
            partial_o: b(partial_rows * hd * 4)?,
            partial_ml: b(partial_rows * 2 * 4)?,
            last_normed: b(max_items * h * 2)?,
            last_next: b(max_items * h * 2)?,
            lr: b(max_items * 128.max(c) * 2)?,
            scores: b(max_items * cfg.vocab * 2)?,
            cand_vals: b(max_items * c * 4)?,
            cand_ids: b(max_items * c * 4)?,
            cand_logits: b(max_items * c * 4)?,
            tok: b(max_items * 8)?,
            record: b(MAX_CHAIN * max_items * 8)?,
            staging: HostMapped::new(meta_bytes)?,
            meta: b(meta_bytes)?,
            cfg,
        })
    }

    /// Bytes of VRAM the drafter takes beyond its weights' share (KV ring and work buffers), for logging.
    pub fn describe(&self) -> String {
        format!(
            "EAGLE-4 draft: window {}, {} candidates, KV ring {:.0} MiB, LM head {:.1} GiB in pinned RAM",
            self.cfg.window,
            self.cfg.candidates,
            (self.k_pool.bytes() * 2) as f64 / (1u64 << 20) as f64,
            (self.cfg.vocab * self.cfg.hidden * 2) as f64 / (1u64 << 30) as f64
        )
    }

    fn put<T: Copy>(&mut self, offset: usize, values: &[T]) {
        let bytes = std::mem::size_of_val(values);
        let dst = &mut self.staging.host_slice_mut()[offset..offset + bytes];
        // SAFETY: plain-old-data copy into the pinned staging buffer.
        dst.copy_from_slice(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) });
    }

    /// Extend over `plan.items` (the last target step's rows, its hidden states captured by `model`), then draft
    /// `plan.k` tokens after each item. Returns the drafts per item.
    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        let n_items = plan.items.len();
        if n_items == 0 {
            return Ok(Vec::new());
        }
        if n_items > self.max_items || plan.k > MAX_CHAIN {
            bail!("draft of {n_items} items x {} tokens (max {} x {MAX_CHAIN})", plan.k, self.max_items);
        }
        let w = self.cfg.window;
        // forwards: the extend in chunks of ROWS rows (rows before the last `window` of an item are skipped: no draft
        // position reads them), then one forward per chain step
        let mut fwds: Vec<Fwd> = Vec::new();
        let mut off = 0usize;
        let mut alloc = |n: usize, sz: usize| {
            let at = off;
            off += (n * sz).next_multiple_of(64);
            at
        };
        let layout = |spans: Vec<(usize, usize)>, alloc: &mut dyn FnMut(usize, usize) -> usize| {
            let rows: usize = spans.iter().map(|s| s.1).sum();
            let n = spans.len();
            Fwd { pos: alloc(rows, 8), loc: alloc(rows, 8), tok: alloc(rows, 8), req: alloc(n, 4), kvl: alloc(n, 4), spans }
        };
        let mut skip = vec![0usize; n_items];
        let mut cur: Vec<(usize, usize)> = Vec::new();
        let mut cur_rows = 0;
        let mut ext_start: Vec<Vec<usize>> = Vec::new(); // per forward: first row of each span within its item
        let mut starts: Vec<usize> = Vec::new();
        for (i, it) in plan.items.iter().enumerate() {
            skip[i] = it.tokens.len().saturating_sub(w);
            let mut r = skip[i];
            while r < it.tokens.len() {
                let take = (it.tokens.len() - r).min(ROWS - cur_rows);
                cur.push((i, take));
                starts.push(r);
                cur_rows += take;
                r += take;
                if cur_rows == ROWS {
                    fwds.push(layout(std::mem::take(&mut cur), &mut alloc));
                    ext_start.push(std::mem::take(&mut starts));
                    cur_rows = 0;
                }
            }
        }
        if cur_rows > 0 {
            fwds.push(layout(std::mem::take(&mut cur), &mut alloc));
            ext_start.push(std::mem::take(&mut starts));
        }
        let n_ext = fwds.len();
        for _ in 1..plan.k {
            fwds.push(layout((0..n_items).map(|i| (i, 1)).collect(), &mut alloc));
        }
        if off > self.staging_len() {
            bail!("draft metadata of {off} bytes exceeds the staging buffer");
        }
        // fill the metadata
        for (f, fwd) in fwds.iter().enumerate() {
            let (mut pos, mut loc, mut tok, mut req, mut kvl) = (vec![], vec![], vec![], vec![], vec![]);
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                let it = &plan.items[i];
                let first = if f < n_ext {
                    let r0 = ext_start[f][si];
                    for r in r0..r0 + len {
                        tok.push(it.tokens[r]);
                    }
                    it.pos0 + r0
                } else {
                    // chain step j = f - n_ext + 1: the position after the extend's last, plus j - 1
                    it.pos0 + it.tokens.len() + (f - n_ext)
                };
                for p in first..first + len {
                    pos.push(p as i64);
                    loc.push((it.row * w + p % w) as i64);
                }
                req.push(it.row as i32);
                kvl.push((first + len) as i32);
            }
            self.put(fwd.pos, &pos);
            self.put(fwd.loc, &loc);
            self.put(fwd.tok, &tok);
            self.put(fwd.req, &req);
            self.put(fwd.kvl, &kvl);
        }
        let s = &model.stream;
        check(
            unsafe { sys::cudaMemcpyAsync(self.meta.ptr(), self.staging.host_ptr(), off, sys::MEMCPY_H2D, s.0) },
            "draft meta copy",
        )?;
        let (h, aux) = (self.cfg.hidden, model.aux.as_ref().ok_or_else(|| anyhow!("draft without captured hidden states"))?);
        // extend
        for (f, fwd) in fwds[..n_ext].iter().enumerate() {
            // inputs: the target's features of the rows, normed per layer, through fc
            let mut row = 0;
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                let a: *const u16 = aux.at((plan.items[i].aux_row + ext_start[f][si]) * 3 * h);
                for c in 0..3 {
                    let dst: *mut u16 = self.hcat.at((row * 3 + c) * h);
                    if self.cfg.fc_norm {
                        check(
                            unsafe {
                                sys::blz_rmsnorm_ld(a.add(c * h), (3 * h) as i64, self.fc_norm[c].ptr(), self.cfg.eps, len as c_int,
                                                    h as c_int, dst, (3 * h) as i64, s.0)
                            },
                            "fc norm",
                        )?;
                    } else {
                        check(
                            unsafe {
                                sys::cudaMemcpy2DAsync(dst.cast(), 3 * h * 2, a.add(c * h).cast(), 3 * h * 2, h * 2, len, sys::MEMCPY_D2D, s.0)
                            },
                            "fc input",
                        )?;
                    }
                }
                row += len;
            }
            self.fc.run(model.lin_ctx(), self.hcat.ptr(), 3 * h, row, self.hin.ptr(), h, s)?;
            let tok: *const i64 = self.meta.at::<u8>(fwd.tok).cast();
            self.layer(model, fwd, tok)?;
            // the last row of every item ending in this forward feeds the head and the chain
            let mut r = 0;
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                r += len;
                if ext_start[f][si] + len == plan.items[i].tokens.len() {
                    self.keep_last(i, r - 1, s)?;
                }
            }
        }
        if plan.k == 0 {
            return Ok(vec![Vec::new(); n_items]);
        }
        self.head(model, n_items, 0)?;
        for (j, fwd) in fwds[n_ext..].iter().enumerate() {
            check(
                unsafe { sys::cudaMemcpyAsync(self.hin.ptr(), self.last_next.ptr::<u8>().cast(), n_items * h * 2, sys::MEMCPY_D2D, s.0) },
                "chain hidden",
            )?;
            let tok: *const i64 = self.tok.ptr();
            self.layer(model, fwd, tok)?;
            for i in 0..n_items {
                self.keep_last(i, i, s)?;
            }
            self.head(model, n_items, j + 1)?;
        }
        s.sync()?;
        let rec: Vec<i64> = self.record.download(plan.k * n_items)?;
        Ok((0..n_items).map(|i| (0..plan.k).map(|j| rec[j * n_items + i]).collect()).collect())
    }

    fn staging_len(&self) -> usize {
        self.meta.bytes()
    }

    /// Row `r` of the forward's output becomes item `i`'s head input and next hidden.
    fn keep_last(&self, i: usize, r: usize, s: &crate::gpu::Stream) -> Result<()> {
        let h = self.cfg.hidden;
        let next = if self.cfg.norm_output { &self.normed } else { &self.hin };
        for (dst, src) in [(&self.last_normed, &self.normed), (&self.last_next, next)] {
            check(
                unsafe { sys::cudaMemcpyAsync(dst.at::<u16>(i * h).cast(), src.at::<u16>(r * h).cast(), h * 2, sys::MEMCPY_D2D, s.0) },
                "draft keep",
            )?;
        }
        Ok(())
    }

    /// The draft layer over the forward's rows: inputs `hin` (hidden) and the tokens at `tok`; leaves the normed
    /// output in `normed` and the un-normed sum in `hin`.
    fn layer(&self, model: &Model, fwd: &Fwd, tok: *const i64) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd) = (c.hidden, c.head_dim);
        let (qd, kvd) = (c.heads * hd, c.kv_heads * hd);
        let ldq = qd + 2 * kvd;
        let n: usize = fwd.spans.iter().map(|s| s.1).sum();
        let pos: *const i64 = self.meta.at::<u8>(fwd.pos).cast();
        let loc: *const i64 = self.meta.at::<u8>(fwd.loc).cast();
        let table = model.embed_table();
        check(
            unsafe {
                sys::blz_embed_gather(tok, n as i64, table.dev_ptr(), c.vocab as i64, (h * 2) as i64, self.emb.ptr(), s.0)
            },
            "draft embed",
        )?;
        check(
            unsafe {
                sys::blz_rmsnorm_ld(self.emb.ptr(), h as i64, self.input_ln.ptr(), c.eps, n as c_int, h as c_int, self.cat.ptr(),
                                    (2 * h) as i64, s.0)
            },
            "draft input norm",
        )?;
        check(
            unsafe {
                sys::blz_rmsnorm_ld(self.hin.ptr(), h as i64, self.hidden_ln.ptr(), c.eps, n as c_int, h as c_int,
                                    self.cat.at::<u16>(h), (2 * h) as i64, s.0)
            },
            "draft hidden norm",
        )?;
        self.qkv.run(lin, self.cat.ptr(), 2 * h, n, self.qkv_out.ptr(), ldq, s)?;
        let q: *mut u16 = self.qkv_out.ptr();
        let k: *mut u16 = self.qkv_out.at(qd);
        let v: *const u16 = self.qkv_out.at(qd + kvd);
        check(unsafe { sys::blz_rope_neox(q, ldq as i64, n as c_int, c.heads as c_int, hd as c_int, pos, self.inv_freq.ptr(), s.0) }, "draft rope q")?;
        check(unsafe { sys::blz_rope_neox(k, ldq as i64, n as c_int, c.kv_heads as c_int, hd as c_int, pos, self.inv_freq.ptr(), s.0) }, "draft rope k")?;
        check(
            unsafe {
                sys::blz_kv_store_dense(k.cast(), v.cast(), ldq as i64, ldq as i64, loc.cast(), sys::BLZ_I64, n as i64, c.kv_heads as c_int,
                                        hd as c_int, sys::BLZ_KV_BF16, self.k_pool.ptr(), self.v_pool.ptr(), self.one.ptr(), self.one.ptr(), s.0)
            },
            "draft kv store",
        )?;
        let pool = sys::blz_nvfp4_pool {
            k_data: self.k_pool.ptr(),
            v_data: self.v_pool.ptr(),
            k_scale: self.no_scales.ptr(),
            v_scale: self.no_scales.ptr(),
            k_g: self.one.ptr(),
            v_g: self.one.ptr(),
            heads: c.kv_heads as c_int,
            dim: hd as c_int,
            format: sys::BLZ_KV_BF16,
        };
        // leading one-row spans in one call, every longer span on its own
        let n1 = fwd.spans.iter().take_while(|sp| sp.1 == 1).count();
        let mut calls: Vec<(usize, usize, usize, usize)> = Vec::new(); // (first span, spans, q_len, first row)
        if n1 > 0 {
            calls.push((0, n1, 1, 0));
        }
        let mut row = n1;
        for (si, sp) in fwd.spans.iter().enumerate().skip(n1) {
            calls.push((si, 1, sp.1, row));
            row += sp.1;
        }
        for (first, count, q_len, r0) in calls {
            let splits = if q_len == 1 { (4 * model.sms() / (count * c.kv_heads)).clamp(1, 64) } else { 1 };
            check(
                unsafe {
                    sys::blz_mha_decode_window(
                        self.qkv_out.at::<u16>(r0 * ldq).cast(),
                        ldq as i64,
                        hd as i64,
                        c.heads as c_int,
                        pool,
                        self.r2t.ptr(),
                        self.stride as i64,
                        sys::blz_index_ptr { ptr: self.meta.at::<u8>(fwd.req + first * 4).cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: self.meta.at::<u8>(fwd.kvl + first * 4).cast(), kind: sys::BLZ_I32 },
                        count as c_int,
                        q_len as c_int,
                        splits as c_int,
                        1.0 / (hd as f32).sqrt(),
                        c.window as c_int,
                        self.partial_o.ptr(),
                        self.partial_ml.ptr(),
                        sys::BLZ_BF16,
                        self.attn.at::<u16>(r0 * qd).cast(),
                        qd as i64,
                        hd as i64,
                        s.0,
                    )
                },
                "draft attention",
            )?;
        }
        self.o.run(lin, self.attn.ptr(), qd, n, self.o_out.ptr(), h, s)?;
        // residual: hin += attention; x2 = norm(hin)
        check(
            unsafe { sys::blz_rmsnorm(self.o_out.ptr(), self.hin.ptr(), self.post_ln.ptr(), c.eps, n as c_int, h as c_int, 0, self.x2.ptr(), s.0) },
            "draft post norm",
        )?;
        self.gu.run(lin, self.x2.ptr(), h, n, self.gu_out.ptr(), 2 * c.inter, s)?;
        check(
            unsafe {
                sys::blz_silu_mul(self.gu_out.ptr(), (2 * c.inter) as i64, self.gu_out.at(c.inter), (2 * c.inter) as i64, n as c_int,
                                  c.inter as c_int, self.act.ptr(), s.0)
            },
            "draft silu",
        )?;
        self.down.run(lin, self.act.ptr(), c.inter, n, self.mlp.ptr(), h, s)?;
        // hin += mlp (the layer's output); normed = norm(hin)
        check(
            unsafe { sys::blz_rmsnorm(self.mlp.ptr(), self.hin.ptr(), self.norm.ptr(), c.eps, n as c_int, h as c_int, 0, self.normed.ptr(), s.0) },
            "draft final norm",
        )
    }

    /// Candidate head on `last_normed`: draft token of chain step `step` for every item, in `tok` and the record.
    fn head(&self, model: &Model, items: usize, step: usize) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, nc, rank) = (c.hidden, c.candidates, self.lr_down.n);
        self.lr_down.run(lin, self.last_normed.ptr(), h, items, self.lr.ptr(), rank, s)?;
        self.lr_head.run(lin, self.lr.ptr(), rank, items, self.scores.ptr(), c.vocab, s)?;
        check(
            unsafe {
                sys::blz_topk(self.scores.ptr(), c.vocab as i64, items as c_int, c.vocab as c_int, 0, nc as c_int, self.cand_vals.ptr(),
                              self.cand_ids.ptr(), std::ptr::null_mut(), s.0)
            },
            "draft candidates",
        )?;
        check(
            unsafe {
                sys::blz_candidate_logits(self.last_normed.ptr(), h as i64, items as c_int, h as c_int, self.lm_head.dev_ptr(),
                                          self.cand_ids.ptr(), nc as c_int, self.cand_logits.ptr(), s.0)
            },
            "draft candidate logits",
        )?;
        check(
            unsafe {
                sys::blz_candidate_pick(self.cand_logits.ptr(), self.cand_ids.ptr(), items as c_int, nc as c_int, self.tok.ptr(),
                                        self.record.at::<i64>(step * items), s.0)
            },
            "draft pick",
        )
    }
}
