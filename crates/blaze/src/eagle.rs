//! EAGLE-4 drafter (eagle-hq/EAGLE-4): target features at p and token p + 1 predict token p + 2.
//! The draft uses a BF16 KV ring; tensor-parallel ranks share the candidate selection.

use crate::gpu::{DevBuf, HostMapped, check};
use std::sync::Arc;
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

/// One draft call: extend every item, then draft `k` tokens after each (0: extend only), or with `tree` = (K, D) a
/// tree of depth D with K frontier rows per item instead of the chain.
#[derive(Clone, Debug, Default)]
pub struct DraftPlan {
    pub items: Vec<DraftItem>,
    pub k: usize,
    pub tree: Option<(usize, usize)>,
}

/// One item's draft tree (eagle_harness core/draft_tree.py): node n's token, path probability (the product of the
/// draft's probabilities from the root) and parent node (-1: a child of the root). Depth 1 at [0, K), the children of
/// depth d's frontier at K + (d - 1) K^2.
#[derive(Clone, Debug, Default)]
pub struct DraftTree {
    pub tokens: Vec<i64>,
    pub scores: Vec<f32>,
    pub parents: Vec<i32>,
    pub top_k: usize,
    pub depth: usize,
}

impl DraftTree {
    /// Depth of node n (1 for the root's children).
    pub fn depth_of(&self, n: usize) -> usize {
        let k = self.top_k;
        if n < k { 1 } else { 2 + (n - k) / (k * k) }
    }
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
    /// the low-rank head's and the LM head's rows of this rank's vocabulary shard [vocab_off, vocab_off + shard): the
    /// candidates of a shard, merged over the ranks (eagle_harness shard_candidates / merge_candidates)
    lr_head: Linear,
    lm_head: Arc<HostMapped>,
    lm_shard: Option<DevBuf>,
    vocab_off: usize,
    shard: usize,
    pairs: DevBuf,
    /// this rank's heads, KV heads and MLP rows
    heads: usize,
    kv_heads: usize,
    inter: usize,
    ar: *mut sys::blz_host_ar,
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
    graphs: std::collections::HashMap<DraftKey, DraftGraph>,
    /// draft trees: (K, D) of the buffers, the device tree, the frontier's hidden inputs, every row's top-K
    tree_shape: (usize, usize),
    tree: Option<TreeBufs>,
    /// drafting calls dumped so far (BLAZE_DRAFT_DUMP)
    dumped: usize,
    warmed: std::collections::HashSet<DraftKey>,
}

/// A captured draft call: chain length, per item (row in the target step, extended rows) or (MAX, 1) for a chain
/// without its extend, whether the extend is in the graph, the tree shape ((0, 0): a chain).
type DraftKey = (usize, Vec<(usize, usize)>, bool, (usize, usize));

/// Device buffers of the draft trees (blaze/tree.h).
struct TreeBufs {
    scores: DevBuf,
    tokens: DevBuf,
    parents: DevBuf,
    front_scores: DevBuf,
    front_tokens: DevBuf,
    front_node: DevBuf,
    ancestors: DevBuf,
    front_hidden: DevBuf,
    probs: DevBuf,
    cand_tokens: DevBuf,
    slot_base: usize,
    cap: usize,
}
const MAX_DRAFT_GRAPHS: usize = 64;
/// Extends of at most this many rows (after a verify or a decode step) are captured with the chain.
const GRAPH_EXTEND_ROWS: usize = 64;

struct DraftGraph {
    exec: sys::cudaGraphExec_t,
    graph: sys::cudaGraph_t,
}

impl Drop for DraftGraph {
    fn drop(&mut self) {
        unsafe {
            sys::cudaGraphExecDestroy(self.exec);
            sys::cudaGraphDestroy(self.graph);
        }
    }
}

// The all-reduce handle is used only by the rank's own thread.
unsafe impl Send for Draft {}

/// The drafter's LM head as a pinned host table (shared by every rank; only the candidates' rows are read).
pub fn load_lm_head(dir: &Path) -> Result<HostMapped> {
    let ckpt = Checkpoint::open(dir)?;
    let lm = ckpt.info("lm_head.weight")?.clone();
    if lm.dtype != Dtype::Bf16 {
        bail!("draft lm_head: bf16 expected");
    }
    let mut t = HostMapped::new(lm.bytes())?;
    t.host_slice_mut().copy_from_slice(ckpt.bytes("lm_head.weight")?);
    Ok(t)
}

impl Draft {
    /// Load this rank's part of the drafter of `dir` on the current GPU (that of `model`), for `max_items` sequence
    /// rows and contexts of `stride` positions.
    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize, lm_head: Arc<HostMapped>) -> Result<Self> {
        let cfg = DraftConfig::load(dir)?;
        let s = &model.stream;
        let ckpt = Checkpoint::open(dir)?;
        // an offline-quantized drafter (scripts/quantize_eagle4.py: ModelOpt MXFP8) carries hf_quant_config.json
        let hf_quant: Option<Value> = match std::fs::read_to_string(dir.join("hf_quant_config.json")) {
            Ok(t) => Some(serde_json::from_str(&t)?),
            Err(_) => None,
        };
        let quant = QuantConfig::from_model_config(&serde_json::json!({}), hf_quant.as_ref())?;
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
        let lin = |mods: &[(&str, Cut)]| Linear::load(&ckpt, &quant, &mods.iter().map(|(n, c)| (n.to_string(), c.clone())).collect::<Vec<_>>(), s);
        let fc_norm = if cfg.fc_norm { (0..3).map(|i| vec(&format!("fc_norm.{i}.weight"))).collect::<Result<_>>()? } else { Vec::new() };
        let (r, world) = (model.tp.rank, model.tp.world);
        if cfg.heads % world != 0 || cfg.kv_heads % world != 0 || cfg.inter % world != 0 || cfg.vocab % world != 0 {
            bail!("draft: heads / KV heads / MLP rows / vocabulary do not split {world} ways");
        }
        let shard = cfg.vocab / world;
        let (heads, kv_heads, inter) = (cfg.heads / world, cfg.kv_heads / world, cfg.inter / world);
        let part = |n: usize| Cut::Rows(vec![r * n..(r + 1) * n]);
        let (qr, kr) = (heads * hd, kv_heads * hd);
        let freqs: Vec<f32> = (0..hd / 2).map(|i| cfg.theta.powf(-2.0 * i as f64 / hd as f64) as f32).collect();
        // ring slot of position p of row r: r * window + p % window
        let w = cfg.window;
        let table: Vec<i32> = (0..max_items).flat_map(|r| (0..stride).map(move |p| (r * w + p % w) as i32)).collect();
        let kv_row = kv_heads * hd * 2;
        let b = |n: usize| DevBuf::zeroed(n, s);
        let (qd, kvd) = (heads * hd, kv_heads * hd);
        let parts_decode = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kv_heads as c_int, 1, 64) } as usize;
        let partial_rows = (ROWS * heads).max(max_items * heads * parts_decode);
        let (tk, td) = if model.opts.tree_width > 1 { (model.opts.tree_width, model.opts.tree_levels) } else { (0, 0) };
        if tk > 0 && (max_items * tk > ROWS || td < 2 || td > 32 || tk > 16) {
            bail!("draft tree of width {tk} and depth {td} for {max_items} items (K <= 16, 2 <= D <= 32, K * items <= {ROWS})");
        }
        let meta_bytes = (MAX_CHAIN.max(td) + 8) * (5 * 8 * ROWS + 256) + 4096;
        let c = cfg.candidates;
        // head rows of one call: an item each, or a tree's frontier rows
        let head_rows = max_items * tk.max(1);
        let ring_slots = max_items * w;
        let tree_slots = max_items * td * tk;
        let tree = if tk > 0 {
            let cap = tk + (td - 1) * tk * tk;
            Some(TreeBufs {
                scores: b(max_items * cap * 4)?,
                tokens: b(max_items * cap * 8)?,
                parents: b(max_items * cap * 4)?,
                front_scores: b(max_items * tk * 4)?,
                front_tokens: b(max_items * tk * 8)?,
                front_node: b(max_items * tk * 4)?,
                ancestors: b(max_items * tk * td * 4)?,
                front_hidden: b(max_items * tk * h * 2)?,
                probs: b(max_items * tk * tk * 4)?,
                cand_tokens: b(max_items * tk * tk * 8)?,
                slot_base: ring_slots,
                cap,
            })
        } else {
            None
        };
        Ok(Self {
            fc: lin(&[("fc", Cut::All)])?,
            fc_norm,
            input_ln: vec("layers.0.input_layernorm.weight")?,
            hidden_ln: vec("layers.0.hidden_norm.weight")?,
            post_ln: vec("layers.0.post_attention_layernorm.weight")?,
            norm: vec("norm.weight")?,
            qkv: lin(&[
                ("layers.0.self_attn.q_proj", part(qr)),
                ("layers.0.self_attn.k_proj", part(kr)),
                ("layers.0.self_attn.v_proj", part(kr)),
            ])?,
            o: lin(&[("layers.0.self_attn.o_proj", Cut::Cols(r * qr..(r + 1) * qr))])?,
            gu: lin(&[("layers.0.mlp.gate_proj", part(inter)), ("layers.0.mlp.up_proj", part(inter))])?,
            down: lin(&[("layers.0.mlp.down_proj", Cut::Cols(r * inter..(r + 1) * inter))])?,
            lr_down: lin(&[("efficient_lm_head_down_proj", Cut::All)])?,
            lr_head: lin(&[("efficient_lm_head", part(shard))])?,
            lm_shard: {
                let bytes = shard * h * 2;
                let (mut free, mut total) = (0usize, 0usize);
                check(unsafe { sys::cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
                let vram = model.opts.draft_head_vram.unwrap_or(free > bytes + (3usize << 29));
                if vram {
                    let b = DevBuf::zeroed(bytes, s)?;
                    check(
                        unsafe {
                            sys::cudaMemcpy(b.ptr(), lm_head.host_ptr().cast::<u8>().add(r * bytes).cast(), bytes, sys::MEMCPY_H2D)
                        },
                        "draft LM head shard",
                    )?;
                    Some(b)
                } else {
                    None
                }
            },
            vocab_off: r * shard,
            shard,
            pairs: b(world * max_items * tk.max(1) * c * 8)?,
            lm_head,
            heads,
            kv_heads,
            inter,
            ar: if world > 1 { model.tp.ar } else { std::ptr::null_mut() },
            inv_freq: DevBuf::from_slice(&freqs)?,
            one: DevBuf::from_slice(&[1.0f32])?,
            k_pool: b((ring_slots + tree_slots) * kv_row)?,
            v_pool: b((ring_slots + tree_slots) * kv_row)?,
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
            gu_out: b(ROWS * 2 * inter * 2)?,
            act: b(ROWS * inter * 2)?,
            mlp: b(ROWS * h * 2)?,
            normed: b(ROWS * h * 2)?,
            partial_o: b(partial_rows * hd * 4)?,
            partial_ml: b(partial_rows * 2 * 4)?,
            last_normed: b(max_items * h * 2)?,
            last_next: b(max_items * h * 2)?,
            lr: b(head_rows * 128.max(c) * 2)?,
            scores: b(head_rows * cfg.vocab * 2)?,
            cand_vals: b(head_rows * c * 4)?,
            cand_ids: b(head_rows * c * 4)?,
            cand_logits: b(head_rows * c * 4)?,
            tok: b(max_items * 8)?,
            record: b(MAX_CHAIN * max_items * 8)?,
            staging: HostMapped::new(meta_bytes)?,
            meta: b(meta_bytes)?,
            graphs: Default::default(),
            tree_shape: (tk, td),
            tree,
            dumped: 0,
            warmed: Default::default(),
            cfg,
        })
    }

    /// Bytes of VRAM the drafter takes beyond its weights' share (KV ring and work buffers), for logging.
    pub fn describe(&self) -> String {
        let tree = match self.tree_shape {
            (0, _) => "chains".to_string(),
            (k, d) => format!("trees of width {k}, depth {d}"),
        };
        let gib = (self.shard * self.cfg.hidden * 2) as f64 / (1u64 << 30) as f64;
        let head = if self.lm_shard.is_some() { format!("LM head shard {gib:.1} GiB in VRAM") } else { format!("LM head shard {gib:.1} GiB in pinned RAM") };
        format!(
            "EAGLE-4 draft: window {}, {} candidates, {tree}, KV {:.0} MiB, {head}",
            self.cfg.window,
            self.cfg.candidates,
            (self.k_pool.bytes() * 2) as f64 / (1u64 << 20) as f64,
        )
    }

    fn put<T: Copy>(&mut self, offset: usize, values: &[T]) {
        let bytes = std::mem::size_of_val(values);
        let dst = &mut self.staging.host_slice_mut()[offset..offset + bytes];
        // SAFETY: plain-old-data copy into the pinned staging buffer.
        dst.copy_from_slice(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) });
    }

    /// The metadata of a call in the staging buffer, copied to the device: the draft forwards after the extend (chain
    /// steps, or the tree's steps) first, at offsets that depend only on the shape, so that a captured call replays;
    /// then the extend in chunks of ROWS rows (rows before the last `window` of an item are skipped: no draft position
    /// reads them). Returns the forwards, how many of them the extend has, and each extend span's first row in its item.
    fn prepare(&mut self, model: &Model, plan: &DraftPlan) -> Result<(Vec<Fwd>, usize, Vec<Vec<usize>>)> {
        let n_items = plan.items.len();
        let w = self.cfg.window;
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
        // after the extend: chain step j (one row per item), or tree step j (K rows per item, one span each)
        let steps: Vec<Fwd> = match plan.tree {
            Some((k, d)) => (1..d).map(|_| layout((0..n_items * k).map(|r| (r / k, 1)).collect(), &mut alloc)).collect(),
            None => (1..plan.k).map(|_| layout((0..n_items).map(|i| (i, 1)).collect(), &mut alloc)).collect(),
        };
        let mut cur: Vec<(usize, usize)> = Vec::new();
        let mut cur_rows = 0;
        let mut ext_start: Vec<Vec<usize>> = Vec::new(); // per forward: first row of each span within its item
        let mut starts: Vec<usize> = Vec::new();
        for (i, it) in plan.items.iter().enumerate() {
            let mut r = it.tokens.len().saturating_sub(w);
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
        fwds.extend(steps);
        if off > self.staging_len() {
            bail!("draft metadata of {off} bytes exceeds the staging buffer");
        }
        // tree KV slots by the buffers' depth (the kernels' layout), whatever the call's depth
        let (tk, td) = match plan.tree {
            Some((k, _)) => (k, self.tree_shape.1),
            None => (0, 0),
        };
        let slot_base = self.tree.as_ref().map_or(0, |t| t.slot_base);
        for (f, fwd) in fwds.iter().enumerate() {
            let (mut pos, mut loc, mut tok, mut req, mut kvl) = (vec![], vec![], vec![], vec![], vec![]);
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                let it = &plan.items[i];
                let end = it.pos0 + it.tokens.len();
                if f >= n_ext && tk > 0 {
                    // tree step j: the frontier rows at depth j, each at its own KV slot, the committed window frozen
                    let (j, lane) = (f - n_ext + 1, si % tk);
                    pos.push((end + j - 1) as i64);
                    loc.push((slot_base + (i * td + j - 1) * tk + lane) as i64);
                    req.push(it.row as i32);
                    kvl.push((end + j) as i32);
                    continue;
                }
                let first = if f < n_ext {
                    let r0 = ext_start[f][si];
                    for r in r0..r0 + len {
                        tok.push(it.tokens[r]);
                    }
                    it.pos0 + r0
                } else {
                    // chain step j = f - n_ext + 1: the position after the extend's last, plus j - 1
                    end + (f - n_ext)
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
        Ok((fwds, n_ext, ext_start))
    }

    /// Extend over `plan.items` (the last target step's rows, its hidden states captured by `model`), then draft
    /// `plan.k` tokens after each item. Returns the drafts per item.
    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        let n_items = plan.items.len();
        if n_items == 0 {
            return Ok(Vec::new());
        }
        if n_items > self.max_items || plan.k > MAX_CHAIN || plan.tree.is_some() {
            bail!("draft of {n_items} items x {} tokens (max {} x {MAX_CHAIN}, chains)", plan.k, self.max_items);
        }
        let (fwds, n_ext, ext_start) = self.prepare(model, plan)?;
        let s = &model.stream;
        // small extends (after a verify or a decode step) and the chain replay as one CUDA graph per shape; a long
        // extend (a prompt's last window) runs eagerly, its chain through the chain graph
        let ext_rows: usize = fwds[..n_ext].iter().flat_map(|f| f.spans.iter().map(|s| s.1)).sum();
        if plan.k > 0 && n_ext == 1 && ext_rows <= GRAPH_EXTEND_ROWS && model.opts.draft_graphs {
            let key: DraftKey = (plan.k, plan.items.iter().map(|it| (it.aux_row, it.tokens.len())).collect(), true, (0, 0));
            self.graphed(model, key, |d| {
                d.extend(model, plan, &fwds[..n_ext], &ext_start)?;
                d.chain(model, n_items, &fwds[n_ext..])
            })?;
        } else {
            self.extend(model, plan, &fwds[..n_ext], &ext_start)?;
            if plan.k > 0 {
                let key: DraftKey = (plan.k, vec![(usize::MAX, 1); n_items], false, (0, 0));
                if model.opts.draft_graphs {
                    self.graphed(model, key, |d| d.chain(model, n_items, &fwds[n_ext..]))?;
                } else {
                    self.chain(model, n_items, &fwds[n_ext..])?;
                }
            }
        }
        if plan.k == 0 {
            return Ok(vec![Vec::new(); n_items]);
        }
        s.sync()?;
        let rec: Vec<i64> = self.record.download(plan.k * n_items)?;
        let drafts: Vec<Vec<i64>> = (0..n_items).map(|i| (0..plan.k).map(|j| rec[j * n_items + i]).collect()).collect();
        self.maybe_dump(model, plan, &drafts)?;
        Ok(drafts)
    }

    /// Extend over `plan.items`, then grow a draft tree of `plan.tree` = (K, D) after each (eagle_harness
    /// sglang/draft_steps.py): the extend's head gives the root's K children; step j runs the K frontier rows of depth
    /// j (each attends to the committed window and its own path), adds every row's top-K candidates as children, keeps
    /// the best K by path probability as the next frontier. One CUDA graph per shape; one host sync at the end.
    pub fn run_tree(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<DraftTree>> {
        let n_items = plan.items.len();
        let Some((k, d)) = plan.tree else { bail!("tree draft without a tree shape") };
        if n_items == 0 {
            return Ok(Vec::new());
        }
        // the buffers hold trees of the drafter's width and up to its depth; a call may grow fewer levels
        if n_items > self.max_items || k != self.tree_shape.0 || d < 2 || d > self.tree_shape.1 {
            bail!("tree draft of {n_items} items, shape {:?} (buffers: {} items, shape {:?})", (k, d), self.max_items, self.tree_shape);
        }
        let (fwds, n_ext, ext_start) = self.prepare(model, plan)?;
        let ext_rows: usize = fwds[..n_ext].iter().flat_map(|f| f.spans.iter().map(|s| s.1)).sum();
        if n_ext == 1 && ext_rows <= GRAPH_EXTEND_ROWS && model.opts.draft_graphs {
            let key: DraftKey = (0, plan.items.iter().map(|it| (it.aux_row, it.tokens.len())).collect(), true, (k, d));
            self.graphed(model, key, |dr| {
                dr.extend(model, plan, &fwds[..n_ext], &ext_start)?;
                dr.grow(model, n_items, &fwds[n_ext..])
            })?;
        } else {
            self.extend(model, plan, &fwds[..n_ext], &ext_start)?;
            let key: DraftKey = (0, vec![(usize::MAX, 1); n_items], false, (k, d));
            if model.opts.draft_graphs {
                self.graphed(model, key, |dr| dr.grow(model, n_items, &fwds[n_ext..]))?;
            } else {
                self.grow(model, n_items, &fwds[n_ext..])?;
            }
        }
        model.stream.sync()?;
        let t = self.tree.as_ref().unwrap();
        // every item's tree at its stride in the buffers; the nodes of the d levels grown come first
        let (cap, used) = (t.cap, k + (d - 1) * k * k);
        let scores: Vec<f32> = t.scores.download(n_items * cap)?;
        let tokens: Vec<i64> = t.tokens.download(n_items * cap)?;
        let parents: Vec<i32> = t.parents.download(n_items * cap)?;
        Ok((0..n_items)
            .map(|i| DraftTree {
                tokens: tokens[i * cap..i * cap + used].to_vec(),
                scores: scores[i * cap..i * cap + used].to_vec(),
                parents: parents[i * cap..i * cap + used].to_vec(),
                top_k: k,
                depth: d,
            })
            .collect())
    }

    /// The tree's device part after the extend (capturable): root candidates, then the D - 1 steps.
    fn grow(&self, model: &Model, items: usize, steps: &[Fwd]) -> Result<()> {
        let (h, s) = (self.cfg.hidden, &model.stream);
        let t = self.tree.as_ref().unwrap();
        let (k, d) = self.tree_shape;
        let dt = sys::blz_draft_tree {
            scores: t.scores.ptr(),
            tokens: t.tokens.ptr(),
            parents: t.parents.ptr(),
            front_scores: t.front_scores.ptr(),
            front_tokens: t.front_tokens.ptr(),
            front_node: t.front_node.ptr(),
            ancestors: t.ancestors.ptr(),
            slot_base: t.slot_base as i32,
            top_k: k as i32,
            max_depth: d as i32,
            cap: t.cap as i32,
        };
        self.head_topk(model, self.last_normed.ptr(), items, k, t.probs.ptr(), t.cand_tokens.ptr())?;
        check(
            unsafe {
                sys::blz_tree_begin(dt, items as c_int, t.probs.ptr(), t.cand_tokens.ptr(), self.last_next.ptr(), h as c_int,
                                    t.front_hidden.ptr(), s.0)
            },
            "tree begin",
        )?;
        let rows = items * k;
        let next = if self.cfg.norm_output { &self.normed } else { &self.hin };
        for (j, fwd) in steps.iter().enumerate() {
            let step = j + 1;
            check(
                unsafe { sys::cudaMemcpyAsync(self.hin.ptr(), t.front_hidden.ptr::<u8>().cast(), rows * h * 2, sys::MEMCPY_D2D, s.0) },
                "tree hidden",
            )?;
            self.layer_with(model, fwd, t.front_tokens.ptr(), Some(step))?;
            self.head_topk(model, self.normed.ptr(), rows, k, t.probs.ptr(), t.cand_tokens.ptr())?;
            check(
                unsafe {
                    sys::blz_tree_step(dt, items as c_int, step as c_int, t.probs.ptr(), t.cand_tokens.ptr(), next.ptr(), h as c_int,
                                       t.front_hidden.ptr(), s.0)
                },
                "tree step",
            )?;
        }
        Ok(())
    }

    /// Candidate head on `rows` rows of `input`: the top `k` of each row's softmax over its candidates (probs f32,
    /// tokens i64, [rows][k]).
    fn head_topk(&self, model: &Model, input: *const u16, rows: usize, k: usize, probs: *mut f32, tokens: *mut i64) -> Result<()> {
        self.head_merged(model, input, rows, k, probs, tokens, std::ptr::null_mut())
    }

    /// The candidate head (eagle_harness candidate_head): rank-r scores of this rank's vocabulary shard, its top C,
    /// their exact logits from the shard's LM head rows, every rank's candidates merged (the best C by logit), softmax
    /// over them; the first k per row (probs may be null; `record`, may be null, gets the first token).
    fn head_merged(&self, model: &Model, input: *const u16, rows: usize, k: usize, probs: *mut f32, tokens: *mut i64,
                   record: *mut i64) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, nc, rank) = (c.hidden, c.candidates, self.lr_down.n);
        let (shard, world) = (self.shard, model.tp.world);
        self.lr_down.run(lin, input, h, rows, self.lr.ptr(), rank, s)?;
        self.lr_head.run(lin, self.lr.ptr(), rank, rows, self.scores.ptr(), shard, s)?;
        check(
            unsafe {
                sys::blz_topk(self.scores.ptr(), shard as i64, rows as c_int, shard as c_int, 0, nc as c_int, self.cand_vals.ptr(),
                              self.cand_ids.ptr(), std::ptr::null_mut(), s.0)
            },
            "draft candidates",
        )?;
        let table: *const u16 = match &self.lm_shard {
            Some(b) => b.ptr(),
            None => unsafe { self.lm_head.dev_ptr::<u16>().add(self.vocab_off * h) },
        };
        check(
            unsafe {
                sys::blz_candidate_logits(input, h as i64, rows as c_int, h as c_int, table.cast(), self.cand_ids.ptr(), nc as c_int,
                                          self.cand_logits.ptr(), s.0)
            },
            "draft candidate logits",
        )?;
        check(
            unsafe {
                sys::blz_cand_pack(self.cand_logits.ptr(), self.cand_ids.ptr(), self.vocab_off as c_int, rows as c_int, nc as c_int,
                                   model.tp.rank as c_int, world as c_int, self.pairs.ptr(), s.0)
            },
            "draft candidate pack",
        )?;
        if !self.ar.is_null() {
            let bytes = (world * rows * nc * 8) as i64;
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, self.pairs.ptr(), bytes, sys::BLZ_AR_FP32, s.0) }, "draft candidate all-reduce")?;
        }
        check(
            unsafe {
                sys::blz_cand_merge_topk(self.pairs.ptr(), world as c_int, rows as c_int, nc as c_int, k as c_int, probs, tokens, record,
                                         s.0)
            },
            "draft candidate merge",
        )
    }

    /// BLAZE_DRAFT_DUMP=dir: the first drafting call's first item (its captured target features, tokens, positions)
    /// and drafts, and the head input of the last chain step, for a reference implementation (scripts/ref_eagle4.py);
    /// with BLAZE_DRAFT_DUMP_CALLS=n the first n calls in dir/00000, dir/00001, ... (calibration data of the offline
    /// quantization, scripts/quantize_eagle4.py).
    fn maybe_dump(&mut self, model: &Model, plan: &DraftPlan, drafts: &[Vec<i64>]) -> Result<()> {
        let Some(dir) = std::env::var_os("BLAZE_DRAFT_DUMP") else { return Ok(()) };
        let calls: usize = std::env::var("BLAZE_DRAFT_DUMP_CALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        if model.tp.rank != 0 || self.dumped >= calls || plan.items.is_empty() {
            return Ok(());
        }
        let dir = if calls == 1 { Path::new(&dir).to_path_buf() } else { Path::new(&dir).join(format!("{:05}", self.dumped)) };
        self.dumped += 1;
        let none = Vec::new();
        self.dump(model, &plan.items[0], drafts.first().unwrap_or(&none), &dir)
    }

    fn dump(&self, model: &Model, it: &DraftItem, drafts: &[i64], dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let h = self.cfg.hidden;
        let aux = model.aux.as_ref().ok_or_else(|| anyhow!("no captured hidden states"))?;
        let rows = it.tokens.len();
        let all: Vec<u16> = aux.download((it.aux_row + rows) * 3 * h)?;
        let feats = &all[it.aux_row * 3 * h..];
        let bytes: Vec<u8> = feats.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("features.bin"), bytes)?;
        let last: Vec<u16> = self.last_normed.download(h)?;
        std::fs::write(dir.join("first_normed.bin"), last.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let meta = serde_json::json!({"tokens": it.tokens, "pos0": it.pos0, "drafts": drafts, "hidden": h});
        std::fs::write(dir.join("meta.json"), meta.to_string())?;
        if std::env::var("BLAZE_DRAFT_DUMP_CALLS").is_err() {
            eprintln!("blaze: draft dump of {rows} rows and {} drafts in {}", drafts.len(), dir.display());
        }
        Ok(())
    }

    /// The extend forwards: the target's features of the items' rows (normed per layer, through fc) and their next
    /// tokens through the layer; the last row of every item feeds the head and the chain.
    fn extend(&self, model: &Model, plan: &DraftPlan, fwds: &[Fwd], ext_start: &[Vec<usize>]) -> Result<()> {
        let (h, s) = (self.cfg.hidden, &model.stream);
        let aux = model.aux.as_ref().ok_or_else(|| anyhow!("draft without captured hidden states"))?;
        for (f, fwd) in fwds.iter().enumerate() {
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
        Ok(())
    }

    /// Run `body` through a CUDA graph per `key` (as SGLang's EAGLE draft graph runner): one eager run, then the
    /// capture, then replays; eager past MAX_DRAFT_GRAPHS shapes.
    fn graphed(&mut self, model: &Model, key: DraftKey, body: impl Fn(&Self) -> Result<()>) -> Result<()> {
        let s = &model.stream;
        if let Some(g) = self.graphs.get(&key) {
            return check(unsafe { sys::cudaGraphLaunch(g.exec, s.0) }, "draft graph launch");
        }
        if self.graphs.len() >= MAX_DRAFT_GRAPHS || self.warmed.insert(key.clone()) {
            return body(self);
        }
        s.sync()?;
        check(unsafe { sys::cudaStreamBeginCapture(s.0, sys::CAPTURE_MODE_THREAD_LOCAL) }, "draft capture")?;
        let captured = body(self);
        let mut graph = std::ptr::null_mut();
        let end = unsafe { sys::cudaStreamEndCapture(s.0, &mut graph) };
        captured?;
        check(end, "draft capture end")?;
        let mut exec = std::ptr::null_mut();
        check(unsafe { sys::cudaGraphInstantiate(&mut exec, graph, 0) }, "draft graph instantiate")?;
        let g = DraftGraph { exec, graph };
        check(unsafe { sys::cudaGraphLaunch(g.exec, s.0) }, "draft graph launch")?;
        self.graphs.insert(key, g);
        Ok(())
    }

    fn chain(&self, model: &Model, items: usize, steps: &[Fwd]) -> Result<()> {
        let (h, s) = (self.cfg.hidden, &model.stream);
        self.head(model, items, 0)?;
        for (j, fwd) in steps.iter().enumerate() {
            check(
                unsafe { sys::cudaMemcpyAsync(self.hin.ptr(), self.last_next.ptr::<u8>().cast(), items * h * 2, sys::MEMCPY_D2D, s.0) },
                "chain hidden",
            )?;
            let tok: *const i64 = self.tok.ptr();
            self.layer(model, fwd, tok)?;
            for i in 0..items {
                self.keep_last(i, i, s)?;
            }
            self.head(model, items, j + 1)?;
        }
        Ok(())
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
        self.layer_with(model, fwd, tok, None)
    }

    /// `layer`; with `tree_step` = j the rows are a tree step's frontier (one span each): they attend to the committed
    /// window of their item, frozen at the extend's end, and to their own path's j slots (the tree's ancestors).
    fn layer_with(&self, model: &Model, fwd: &Fwd, tok: *const i64, tree_step: Option<usize>) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd) = (c.hidden, c.head_dim);
        let (heads, kv_heads) = (self.heads, self.kv_heads);
        let (qd, kvd) = (heads * hd, kv_heads * hd);
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
        check(unsafe { sys::blz_rope_neox(q, ldq as i64, n as c_int, heads as c_int, hd as c_int, pos, self.inv_freq.ptr(), s.0) }, "draft rope q")?;
        check(unsafe { sys::blz_rope_neox(k, ldq as i64, n as c_int, kv_heads as c_int, hd as c_int, pos, self.inv_freq.ptr(), s.0) }, "draft rope k")?;
        check(
            unsafe {
                sys::blz_kv_store_dense(k.cast(), v.cast(), ldq as i64, ldq as i64, loc.cast(), sys::BLZ_I64, n as i64, kv_heads as c_int,
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
            heads: kv_heads as c_int,
            dim: hd as c_int,
            format: sys::BLZ_KV_BF16,
        };
        if let Some(j) = tree_step {
            let t = self.tree.as_ref().unwrap();
            let splits = (4 * model.sms() / (n * kv_heads)).clamp(1, 64);
            check(
                unsafe {
                    sys::blz_mha_decode_tail(
                        self.qkv_out.ptr::<u16>().cast(),
                        ldq as i64,
                        hd as i64,
                        heads as c_int,
                        pool,
                        self.r2t.ptr(),
                        self.stride as i64,
                        sys::blz_index_ptr { ptr: self.meta.at::<u8>(fwd.req).cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: self.meta.at::<u8>(fwd.kvl).cast(), kind: sys::BLZ_I32 },
                        n as c_int,
                        splits as c_int,
                        1.0 / (hd as f32).sqrt(),
                        (c.window + j) as c_int,
                        t.ancestors.ptr(),
                        j as c_int,
                        self.tree_shape.1 as c_int,
                        self.partial_o.ptr(),
                        self.partial_ml.ptr(),
                        sys::BLZ_BF16,
                        self.attn.ptr::<u16>().cast(),
                        qd as i64,
                        hd as i64,
                        s.0,
                    )
                },
                "draft tree attention",
            )?;
            return self.finish_layer(model, n);
        }
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
            // about four CTAs per SM over the call's query tiles and KV heads, within the partial workspace
            let group = heads / kv_heads;
            let mt = (q_len * group).div_ceil(16).min(3);
            let tiles = q_len.div_ceil((16 * mt / group).max(1));
            let cap = self.partial_o.bytes() / (heads * hd * 4);
            let mut splits = (4 * model.sms() / (count * tiles * kv_heads)).clamp(1, 64);
            while splits > 1
                && count * q_len * unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kv_heads as c_int, q_len as c_int, splits as c_int) } as usize > cap
            {
                splits /= 2;
            }
            check(
                unsafe {
                    sys::blz_mha_decode_window(
                        self.qkv_out.at::<u16>(r0 * ldq).cast(),
                        ldq as i64,
                        hd as i64,
                        heads as c_int,
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
                        1,
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
        self.finish_layer(model, n)
    }

    /// The layer after its attention (in `attn`): o, residual + norm, MLP, residual + final norm.
    fn finish_layer(&self, model: &Model, n: usize) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd) = (c.hidden, c.head_dim);
        let (qd, inter) = (self.heads * hd, self.inter);
        self.o.run(lin, self.attn.ptr(), qd, n, self.o_out.ptr(), h, s)?;
        // residual: hin += attention (summed over the ranks); x2 = norm(hin)
        self.add_norm(self.o_out.ptr(), self.post_ln.ptr(), self.x2.ptr(), n, s)?;
        self.gu.run(lin, self.x2.ptr(), h, n, self.gu_out.ptr(), 2 * inter, s)?;
        check(
            unsafe {
                sys::blz_silu_mul(self.gu_out.ptr(), (2 * inter) as i64, self.gu_out.at(inter), (2 * inter) as i64, n as c_int,
                                  inter as c_int, self.act.ptr(), s.0)
            },
            "draft silu",
        )?;
        self.down.run(lin, self.act.ptr(), inter, n, self.mlp.ptr(), h, s)?;
        // hin += mlp (the layer's output); normed = norm(hin)
        self.add_norm(self.mlp.ptr(), self.norm.ptr(), self.normed.ptr(), n, s)
    }

    /// hin += x (all-reduced over the ranks first), out = rmsnorm(hin, w).
    fn add_norm(&self, x: *mut u16, w: *const u16, out: *mut u16, n: usize, s: &crate::gpu::Stream) -> Result<()> {
        let (h, eps) = (self.cfg.hidden, self.cfg.eps);
        if !self.ar.is_null() && n <= 64 {
            return check(unsafe { sys::blz_host_ar_rmsnorm(self.ar, x, self.hin.ptr(), w, eps, n as c_int, h as c_int, 0, out, s.0) }, "draft all-reduce norm");
        }
        // a long extend (the fused kernel takes 64 rows): the same bits from the all-reduce, then the norm
        if !self.ar.is_null() {
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, x.cast(), (n * h * 2) as i64, sys::BLZ_AR_BF16, s.0) }, "draft all-reduce")?;
        }
        check(unsafe { sys::blz_rmsnorm(x, self.hin.ptr(), w, eps, n as c_int, h as c_int, 0, out, s.0) }, "draft norm")
    }

    /// Candidate head on `last_normed`: draft token of chain step `step` for every item, in `tok` and the record.
    fn head(&self, model: &Model, items: usize, step: usize) -> Result<()> {
        self.head_merged(model, self.last_normed.ptr(), items, 1, std::ptr::null_mut(), self.tok.ptr(), self.record.at::<i64>(step * items))
    }

    #[allow(dead_code)]
    fn head_replicated(&self, model: &Model, items: usize, step: usize) -> Result<()> {
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
