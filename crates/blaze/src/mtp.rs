//! Checkpoint MTP drafter (vLLM qwen3_5_mtp / SGLang Qwen3_5ForCausalLMMTP).
//! A BF16 KV ring bounds draft context; target verification decides which tokens are accepted.

use crate::eagle::{DraftPlan, MAX_CHAIN};
use crate::gpu::{DevBuf, HostMapped, check};
use crate::linear::{Cut, Linear};
use crate::model::{HcBlock, Model, load_hc};
use crate::quant::QuantConfig;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Result, anyhow, bail};
use blaze_sys as sys;
use std::ffi::c_int;
use std::path::Path;

/// Rows of one draft forward.
const ROWS: usize = 512;
const MAX_GRAPHS: usize = 192;

#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub rotary_dim: usize,
    /// attention window of the KV ring, positions
    pub window: usize,
    pub vocab: usize,
}

/// Per-forward metadata offsets in the staging / device metadata buffer (as eagle.rs).
struct Fwd {
    /// (item, rows) spans in order
    spans: Vec<(usize, usize)>,
    pos: usize,
    loc: usize,
    tok: usize,
    req: usize,
    kvl: usize,
}

/// The layer between the target's hidden state and the head.
enum Body {
    Dense(Box<Dense>),
    Hyper(Box<Hyper>),
}

/// Qwen3.5: x = fc(concat(norm_e(embed), norm_h(hidden))), pre-norm attention and MLP, the final norm.
struct Dense {
    pre_emb: DevBuf,
    pre_hid: DevBuf,
    input_ln: DevBuf,
    post_ln: DevBuf,
    norm: DevBuf,
    fc: Linear,
    gu: Linear,
    down: Linear,
    /// this rank's MLP rows
    inter: usize,
    // work [ROWS, ...]
    emb: DevBuf,
    cat: DevBuf,
    hin: DevBuf,
    normed1: DevBuf,
    x2: DevBuf,
    gu_out: DevBuf,
    act: DevBuf,
    mlp: DevBuf,
}

/// Qwen4-Exp (SGLang qwen4_exp_mtp, its residual_linear_shared fusion): each stream s of the target's final streams H
/// (normed together, over all of them) through fc_hidden, plus fc_embedding of the normed embedding of the token,
///   S_s = fc_hidden(norm_h(H)_s) + fc_embedding(norm_e(embed(token)))
/// then the decoder layer on S: the attention and the routed MLP, each between its hyper-connection mix and combine. The
/// final mixer's mix of the streams is the head's input, the streams after the layer the next draft row's hidden input.
/// The routed experts run in the target's expert engine (its layer `moe_layer`, from the MTP checkpoint).
struct Hyper {
    pre_emb: DevBuf,
    pre_hid: DevBuf,
    fc_emb: Linear,
    fc_hid: Linear,
    attn_hc: HcBlock,
    mlp_hc: HcBlock,
    mixer: HcBlock,
    /// the router's logits, then the shared expert's gate logit
    gate: Linear,
    shared_gu: Linear,
    shared_down: Linear,
    has_shared_gate: bool,
    moe_layer: usize,
    experts: usize,
    top_k: usize,
    norm_topk: bool,
    shared_inter: usize,
    count: usize,
    lowrank: usize,
    // work [ROWS, ...]
    emb: DevBuf,
    eproj: DevBuf,
    hnorm: DevBuf,
    hproj: DevBuf,
    streams: DevBuf,
    block: DevBuf,
    hc_t: DevBuf,
    hc_ws: DevBuf,
    router: DevBuf,
    ids: DevBuf,
    weights: DevBuf,
    moe_out: DevBuf,
    sgu: DevBuf,
    sact: DevBuf,
    shared: DevBuf,
    x: DevBuf,
    /// [max_items, count * hidden]: each item's streams for its next draft row
    last_streams: DevBuf,
}

pub struct Mtp {
    pub cfg: MtpConfig,
    body: Body,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Linear,
    o: Linear,
    /// the head over the first `head_rows` rows of this rank's vocabulary shard (None: the target's own head over the
    /// whole shard) and the pairs a rank without rows contributes
    head_lin: Option<Linear>,
    head_rows: usize,
    /// token id of this rank's first head row
    head_first: usize,
    no_pairs: DevBuf,
    /// candidate head: ids per item and rank (cap per call), the target's BF16 head shard, logits of the candidates
    cand_cap: usize,
    table: Option<*const u16>,
    cand_logits: DevBuf,
    /// this call scores the candidates (the plan carries them) rather than the whole shard
    use_cands: bool,
    /// this rank's heads and KV heads
    heads: usize,
    kv_heads: usize,
    ar: *mut sys::blz_host_ar,
    one: DevBuf,
    no_scales: DevBuf,
    k_pool: DevBuf,
    v_pool: DevBuf,
    r2t: DevBuf,
    stride: usize,
    max_items: usize,
    // work [ROWS, ...]: the hidden inputs (width()), the attention's, the head's inputs
    hid: DevBuf,
    qkv_out: DevBuf,
    q_out: DevBuf,
    gate: DevBuf,
    attn: DevBuf,
    o_out: DevBuf,
    normed: DevBuf,
    partial_o: DevBuf,
    partial_ml: DevBuf,
    // chain [max_items, ...]
    last_normed: DevBuf,
    logits: DevBuf,
    pairs: DevBuf,
    pick: DevBuf,
    tok: DevBuf,
    record: DevBuf,
    staging: HostMapped,
    meta: DevBuf,
    graphs: std::collections::HashMap<(usize, usize, bool), Graph>,
    warmed: std::collections::HashSet<(usize, usize, bool)>,
    dumped: bool,
}

struct Graph {
    exec: sys::cudaGraphExec_t,
    graph: sys::cudaGraph_t,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            sys::cudaGraphExecDestroy(self.exec);
            sys::cudaGraphDestroy(self.graph);
        }
    }
}

// The all-reduce handle is used only by the rank's own thread.
unsafe impl Send for Mtp {}

/// VRAM of the MTP head on each rank beyond the target (weights, KV ring, work buffers), for `--kv-tokens auto`. `mtp`:
/// the MTP checkpoint (None: the target's); routed experts go to the expert engine's tiers, not counted here.
pub fn vram_estimate(dir: &Path, mtp: Option<&Path>, world: usize, window: usize, max_items: usize, head_vocab: usize) -> Result<usize> {
    let ckpt = Checkpoint::open(mtp.unwrap_or(dir))?;
    let world = world.max(1);
    let mut bytes = 0usize;
    for (name, t) in &ckpt.tensors {
        if name.starts_with("mtp.") && !name.contains(".mlp.experts.") {
            // fc and the norms are on every rank, the layer's projections split
            bytes += if name == "mtp.fc.weight" || name.contains("norm") { t.bytes() } else { t.bytes().div_ceil(world) };
        }
    }
    if head_vocab > 0
        && let Some(lm) = ckpt.get("lm_head.weight")
    {
        bytes += lm.bytes() / lm.shape[0].max(1) * head_vocab.div_ceil(world);
    }
    let cfg = crate::config::ModelConfig::load(dir)?;
    let kv = cfg.attn.kv_heads.div_ceil(world) * cfg.attn.head_dim * 2;
    let h = cfg.hidden;
    // two BF16 pools, the metadata and the work buffers of ROWS rows (about 14 hidden-wide buffers, the qkv and MLP ones;
    // five of the streams' width with hyper-connections)
    let streams = cfg.hc.as_ref().map_or(0, |hc| 5 * hc.count * h);
    let work = ROWS * (14 * h + streams + 3 * cfg.attn.heads * cfg.attn.head_dim + 4 * 17408usize.min(h * 4)) * 2 / world.max(1);
    Ok(bytes + 2 * max_items * window * kv + work + (96 << 20))
}

/// Does the checkpoint in `dir` carry an MTP head (`mtp.*` tensors)?
pub fn has_head(dir: &Path) -> Result<bool> {
    Ok(Checkpoint::open(dir)?.tensors.keys().any(|k| k.starts_with("mtp.")))
}

impl Mtp {
    /// Load this rank's part of the MTP head in `dir` (the target's checkpoint or an MTP one) on the current GPU, for
    /// `max_items` sequence rows and contexts of `stride` positions, with a KV ring of `window` positions per row.
    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize, window: usize) -> Result<Self> {
        let ckpt = Checkpoint::open(dir)?;
        let tc = &model.cfg;
        let (h, hd) = (tc.hidden, tc.attn.head_dim);
        let hyper = tc.hc.is_some();
        let inter = match (&tc.mlp, hyper) {
            (crate::config::MlpConfig::Dense { intermediate }, false) => *intermediate,
            (crate::config::MlpConfig::Moe { shared_intermediate, .. }, true) => *shared_intermediate,
            _ => bail!("MTP head: the Qwen3.5 dense layer or the Qwen4-Exp routed one between hyper-connection streams only"),
        };
        let cfg = MtpConfig {
            hidden: h,
            heads: tc.attn.heads,
            kv_heads: tc.attn.kv_heads,
            head_dim: hd,
            inter,
            eps: tc.rms_eps,
            rotary_dim: tc.rope.rotary_dim,
            window,
            vocab: tc.vocab,
        };
        if hd != 256 || window == 0 {
            bail!("MTP: head dim {hd}, window {window} (head dim 256 and a window supported)");
        }
        let s = &model.stream;
        let (r, world) = (model.tp.rank, model.tp.world);
        if cfg.heads % world != 0 || cfg.kv_heads % world != 0 || inter % world != 0 {
            bail!("MTP: heads / KV heads / MLP rows do not split {world} ways");
        }
        if hyper && world > 1 {
            bail!("MTP head of a hyper-connection model with tensor parallelism: not supported");
        }
        let need: &[&str] = if hyper {
            &["mtp.fc_hidden.weight", "mtp.fc_embedding.weight", "mtp.layers.0.self_attn.q_proj.weight", "mtp.hyper_connection_mixer.hc_norm.weight"]
        } else {
            &["mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight", "mtp.norm.weight"]
        };
        for name in need {
            if ckpt.get(name).is_none() {
                bail!("{}: no MTP head (tensor {name} missing)", dir.display());
            }
        }
        let vec = |name: &str, n: usize| -> Result<DevBuf> {
            let t = ckpt.info(name)?;
            if t.dtype != Dtype::Bf16 || t.numel() != n {
                bail!("{name}: bf16 [{n}] expected");
            }
            let b = DevBuf::zeroed(t.bytes(), s)?;
            b.upload(ckpt.bytes(name)?)?;
            Ok(b)
        };
        // the MTP tensors are BF16 in the NVFP4 checkpoints (hf_quant_config excludes `mtp*`; an MTP checkpoint's routed
        // experts are the expert engine's): no quantization rules
        let quant = QuantConfig::from_model_config(&serde_json::json!({}), None)?;
        let lin = |mods: Vec<(&str, Cut)>| Linear::load(&ckpt, &quant, &mods.into_iter().map(|(n, c)| (n.to_string(), c)).collect::<Vec<_>>(), s);
        let part = |n: usize| Cut::Rows(vec![r * n..(r + 1) * n]);
        let (heads, kv_heads, inter_r) = (cfg.heads / world, cfg.kv_heads / world, inter / world);
        let (qd, kvd) = (heads * hd, kv_heads * hd);
        let ldq = 2 * qd + 2 * kvd;
        let b = |n: usize| DevBuf::zeroed(n, s);
        let kv_row = kv_heads * hd * 2;
        let ring = max_items * window;
        let table: Vec<i32> = (0..max_items).flat_map(|row| (0..stride).map(move |p| (row * window + p % window) as i32)).collect();
        let parts_decode = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, kv_heads as c_int, 1, 64) } as usize;
        let partial_rows = (ROWS * heads).max(max_items * heads * parts_decode);
        let shard = model.dims.vocab;
        // a restricted head: the first `head_vocab` ids split evenly over the ranks (each reads its own rows of the
        // checkpoint's head, whatever the target's cut)
        let head_vocab = model.opts.mtp_head_vocab.min(tc.vocab);
        let per_rank = head_vocab.div_ceil(world);
        let head_first = if head_vocab == 0 { r * shard } else { r * per_rank };
        let head_rows = if head_vocab == 0 { shard } else { head_vocab.saturating_sub(r * per_rank).min(per_rank) };
        let head_lin = if head_vocab > 0 && head_rows > 0 {
            Some(lin(vec![("lm_head", Cut::Rows(vec![head_first..head_first + head_rows]))])?)
        } else {
            None
        };
        let head_table = model.lm_head().bf16_rows().filter(|_| head_vocab == 0 && model.opts.mtp_head_cands > 0);
        if model.opts.mtp_head_cands > 0 && head_table.is_none() && model.tp.rank == 0 {
            eprintln!("blaze: MTP candidate head: the target's LM head is not plain BF16 (or --mtp-head-vocab is set): the whole vocabulary");
        }
        let no_pairs: Vec<f32> = (0..max_items).flat_map(|_| [-3.0e38f32, 0.0]).collect();
        // metadata of one call (prepare): every item's candidate ids, the extend over at most one step's rows (a prefill
        // chunk and every item's chain plus one) in forwards of ROWS rows, the chain steps
        let cand_cap = if head_table.is_some() { model.opts.mtp_head_cands } else { 0 };
        let fwd_meta = 3 * ROWS * 8 + 2 * ((max_items + 1) * 4).next_multiple_of(64);
        let ext_fwds = (model.opts.max_chunk + max_items * (MAX_CHAIN + 1)).div_ceil(ROWS) + 1;
        let meta_bytes = (max_items * cand_cap * 4).next_multiple_of(64) + (ext_fwds + MAX_CHAIN) * fwd_meta + 4096;
        let body = if let Some(hc) = &tc.hc {
            let crate::config::MlpConfig::Moe { experts, top_k, norm_topk, .. } = tc.mlp else { unreachable!() };
            let moe_layer = model.mtp_moe_layer.ok_or_else(|| anyhow!("MTP head with routed experts: the expert engine holds no MTP layer"))?;
            let (count, lowrank) = (hc.count, hc.lowrank);
            let hw = count * h;
            let m = "mtp.layers.0.mlp.";
            let has_shared_gate = ckpt.get(&format!("{m}shared_expert_gate.weight")).is_some();
            let mut gate_mods = vec![(format!("{m}gate"), Cut::All)];
            if has_shared_gate {
                gate_mods.push((format!("{m}shared_expert_gate"), Cut::All));
            }
            let gate = Linear::load(&ckpt, &quant, &gate_mods, s)?;
            let ws = unsafe { sys::blz_hc_mix_workspace_bytes(ROWS as c_int, count as c_int, h as c_int, lowrank as c_int) };
            Body::Hyper(Box::new(Hyper {
                pre_emb: vec("mtp.pre_fc_norm_embedding.weight", h)?,
                pre_hid: vec("mtp.pre_fc_norm_hidden.weight", hw)?,
                fc_emb: lin(vec![("mtp.fc_embedding", Cut::All)])?,
                fc_hid: lin(vec![("mtp.fc_hidden", Cut::All)])?,
                attn_hc: load_hc(&ckpt, "mtp.layers.0.attn_hyper_connection.")?,
                mlp_hc: load_hc(&ckpt, "mtp.layers.0.mlp_hyper_connection.")?,
                mixer: load_hc(&ckpt, "mtp.hyper_connection_mixer.")?,
                router: b(ROWS * gate.n * 2)?,
                gate,
                shared_gu: lin(vec![(&format!("{m}shared_expert.gate_proj"), Cut::All), (&format!("{m}shared_expert.up_proj"), Cut::All)])?,
                shared_down: lin(vec![(&format!("{m}shared_expert.down_proj"), Cut::All)])?,
                has_shared_gate,
                moe_layer,
                experts,
                top_k,
                norm_topk,
                shared_inter: inter,
                count,
                lowrank,
                emb: b(ROWS * h * 2)?,
                eproj: b(ROWS * h * 2)?,
                hnorm: b(ROWS * hw * 2)?,
                hproj: b(ROWS * hw * 2)?,
                streams: b(ROWS * hw * 2)?,
                block: b(ROWS * h * 2)?,
                hc_t: b(ROWS * lowrank * 2)?,
                hc_ws: b(ws.max(16))?,
                ids: b(ROWS * top_k * 4)?,
                weights: b(ROWS * top_k * 4)?,
                moe_out: b(ROWS * h * 2)?,
                sgu: b(ROWS * 2 * inter * 2)?,
                sact: b(ROWS * inter * 2)?,
                shared: b(ROWS * h * 2)?,
                x: b(ROWS * h * 2)?,
                last_streams: b(max_items * hw * 2)?,
            }))
        } else {
            Body::Dense(Box::new(Dense {
                pre_emb: vec("mtp.pre_fc_norm_embedding.weight", h)?,
                pre_hid: vec("mtp.pre_fc_norm_hidden.weight", h)?,
                input_ln: vec("mtp.layers.0.input_layernorm.weight", h)?,
                post_ln: vec("mtp.layers.0.post_attention_layernorm.weight", h)?,
                norm: vec("mtp.norm.weight", h)?,
                fc: lin(vec![("mtp.fc", Cut::All)])?,
                gu: lin(vec![("mtp.layers.0.mlp.gate_proj", part(inter_r)), ("mtp.layers.0.mlp.up_proj", part(inter_r))])?,
                down: lin(vec![("mtp.layers.0.mlp.down_proj", Cut::Cols(r * inter_r..(r + 1) * inter_r))])?,
                inter: inter_r,
                emb: b(ROWS * h * 2)?,
                cat: b(ROWS * 2 * h * 2)?,
                hin: b(ROWS * h * 2)?,
                normed1: b(ROWS * h * 2)?,
                x2: b(ROWS * h * 2)?,
                gu_out: b(ROWS * 2 * inter_r * 2)?,
                act: b(ROWS * inter_r * 2)?,
                mlp: b(ROWS * h * 2)?,
            }))
        };
        let width = tc.hc.as_ref().map_or(h, |hc| hc.count * h);
        let m = Self {
            head_lin,
            head_rows,
            head_first,
            cand_cap,
            table: head_table,
            cand_logits: b(if head_table.is_some() { max_items * model.opts.mtp_head_cands * 4 } else { 16 })?,
            use_cands: false,
            no_pairs: DevBuf::from_slice(&no_pairs)?,
            body,
            q_norm: vec("mtp.layers.0.self_attn.q_norm.weight", hd)?,
            k_norm: vec("mtp.layers.0.self_attn.k_norm.weight", hd)?,
            qkv: lin(vec![
                ("mtp.layers.0.self_attn.q_proj", part(2 * qd)),
                ("mtp.layers.0.self_attn.k_proj", part(kvd)),
                ("mtp.layers.0.self_attn.v_proj", part(kvd)),
            ])?,
            o: lin(vec![("mtp.layers.0.self_attn.o_proj", Cut::Cols(r * qd..(r + 1) * qd))])?,
            heads,
            kv_heads,
            ar: if world > 1 { model.tp.ar } else { std::ptr::null_mut() },
            one: DevBuf::from_slice(&[1.0f32])?,
            no_scales: b(16)?,
            k_pool: b(ring * kv_row)?,
            v_pool: b(ring * kv_row)?,
            r2t: DevBuf::from_slice(&table)?,
            stride,
            max_items,
            hid: b(ROWS * width * 2)?,
            qkv_out: b(ROWS * ldq * 2)?,
            q_out: b(ROWS * qd * 2)?,
            gate: b(ROWS * qd * 2)?,
            attn: b(ROWS * qd * 2)?,
            o_out: b(ROWS * h * 2)?,
            normed: b(ROWS * h * 2)?,
            partial_o: b(partial_rows * hd * 4)?,
            partial_ml: b(partial_rows * 2 * 4)?,
            last_normed: b(max_items * h * 2)?,
            logits: b(max_items * shard * 2)?,
            pairs: b((world * max_items * 8).next_multiple_of(16))?,
            pick: b(max_items * 4)?,
            tok: b(max_items * 8)?,
            record: b(MAX_CHAIN * max_items * 8)?,
            staging: HostMapped::new(meta_bytes)?,
            meta: b(meta_bytes)?,
            graphs: Default::default(),
            warmed: Default::default(),
            dumped: false,
            cfg,
        };
        let mut weights = Vec::new();
        for l in m.linears() {
            weights.extend(l.tune_weights());
        }
        model.lin_ctx().tune.tune(&weights, model.lin_ctx().gemm, s)?;
        Ok(m)
    }

    /// The layer's projections (the head's aside).
    fn linears(&self) -> Vec<&Linear> {
        let mut v = vec![&self.qkv, &self.o];
        match &self.body {
            Body::Dense(d) => v.extend([&d.fc, &d.gu, &d.down]),
            Body::Hyper(y) => v.extend([&y.fc_emb, &y.fc_hid, &y.gate, &y.shared_gu, &y.shared_down]),
        }
        v
    }

    /// Values of a draft row's hidden input: the target's final normed hidden state, or its final streams.
    fn width(&self) -> usize {
        match &self.body {
            Body::Dense(_) => self.cfg.hidden,
            Body::Hyper(y) => y.count * self.cfg.hidden,
        }
    }

    pub fn describe(&self) -> String {
        let head = if self.head_lin.is_some() { format!(", head over {} vocabulary rows on this rank", self.head_rows) } else { String::new() };
        let routed = match &self.body {
            Body::Hyper(y) => format!(", its routed experts in the expert engine (layer {})", y.moe_layer),
            Body::Dense(_) => String::new(),
        };
        format!(
            "MTP head: window {}{head}, BF16 weights {:.0} MiB on this rank{routed}, KV ring {:.0} MiB",
            self.cfg.window,
            self.linears().iter().flat_map(|l| l.regions()).map(|r| r.1).sum::<usize>() as f64 / (1u64 << 20) as f64,
            (self.k_pool.bytes() * 2) as f64 / (1u64 << 20) as f64,
        )
    }

    fn put<T: Copy>(&mut self, offset: usize, values: &[T]) {
        let bytes = std::mem::size_of_val(values);
        let dst = &mut self.staging.host_slice_mut()[offset..offset + bytes];
        // SAFETY: plain-old-data copy into the pinned staging buffer.
        dst.copy_from_slice(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) });
    }

    /// The metadata of a call in the staging buffer, copied to the device: the chain steps first (at offsets that depend
    /// only on the shape, so that a captured call replays), then the extend in chunks of ROWS rows (an item's rows before
    /// its last `window` are skipped: no draft position reads them). Returns the forwards, how many are the extend's and
    /// each extend span's first row within its item.
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
        // the candidate ids of every item at the start of the metadata: their address depends on the shape only
        self.use_cands = self.cand_cap > 0 && plan.k > 0 && plan.cands.len() == n_items && plan.cands.iter().all(|c| !c.is_empty());
        let cand_at = if self.use_cands { alloc(n_items * self.cand_cap, 4) } else { 0 };
        let steps: Vec<Fwd> = (1..plan.k).map(|_| layout((0..n_items).map(|i| (i, 1)).collect(), &mut alloc)).collect();
        let mut cur: Vec<(usize, usize)> = Vec::new();
        let mut cur_rows = 0;
        let mut ext_start: Vec<Vec<usize>> = Vec::new();
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
        if off > self.meta.bytes() {
            bail!("MTP metadata of {off} bytes exceeds the staging buffer");
        }
        if self.use_cands {
            // this rank's share: the ids of its vocabulary shard, as row numbers of its head, padded with -1
            let (r, shard, cap) = (model.tp.rank, model.dims.vocab, self.cand_cap);
            let mut ids = vec![-1i32; n_items * cap];
            for (i, c) in plan.cands.iter().enumerate() {
                let (lo, hi) = (c.partition_point(|&t| (t as usize) < r * shard), c.partition_point(|&t| (t as usize) < (r + 1) * shard));
                for (d, &t) in ids[i * cap..(i + 1) * cap].iter_mut().zip(&c[lo..hi]) {
                    *d = t - (r * shard) as i32;
                }
            }
            self.put(cand_at, &ids);
        }
        for (f, fwd) in fwds.iter().enumerate() {
            let (mut pos, mut loc, mut tok, mut req, mut kvl) = (vec![], vec![], vec![], vec![], vec![]);
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                let it = &plan.items[i];
                let end = it.pos0 + it.tokens.len();
                let first = if f < n_ext {
                    let r0 = ext_start[f][si];
                    tok.extend_from_slice(&it.tokens[r0..r0 + len]);
                    it.pos0 + r0
                } else {
                    // chain step j = f - n_ext + 1: the row at the position after the extend's last, plus j - 1
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
        check(
            unsafe { sys::cudaMemcpyAsync(self.meta.ptr(), self.staging.host_ptr(), off, sys::MEMCPY_H2D, model.stream.0) },
            "MTP meta copy",
        )?;
        Ok((fwds, n_ext, ext_start))
    }

    /// Extend over `plan.items` (the last target step's rows, their final hidden states captured by `model`), then draft
    /// `plan.k` tokens after each item. Returns the drafts per item.
    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        let n_items = plan.items.len();
        if n_items == 0 {
            return Ok(Vec::new());
        }
        if n_items > self.max_items || plan.k > MAX_CHAIN || plan.tree.is_some() {
            bail!("MTP draft of {n_items} items x {} tokens (max {} x {MAX_CHAIN}, chains only)", plan.k, self.max_items);
        }
        let (fwds, n_ext, ext_start) = self.prepare(model, plan)?;
        let s = &model.stream;
        // the extend eagerly (its shape changes every call), the chain through one CUDA graph per (length, items)
        self.extend(model, plan, &fwds[..n_ext], &ext_start)?;
        if plan.k == 0 {
            return Ok(vec![Vec::new(); n_items]);
        }
        let dump = std::env::var_os("BLAZE_MTP_DUMP").filter(|_| !self.dumped && model.tp.rank == 0 && matches!(self.body, Body::Dense(_)));
        if let Some(dir) = dump {
            self.dumped = true;
            return self.run_dump(model, plan, &fwds[n_ext..], Path::new(&dir));
        }
        let key = (plan.k, n_items, self.use_cands);
        if model.opts.draft_graphs {
            self.graphed(model, key, |d| d.chain(model, n_items, &fwds[n_ext..]))?;
        } else {
            self.chain(model, n_items, &fwds[n_ext..])?;
        }
        s.sync()?;
        let rec: Vec<i64> = self.record.download(plan.k * n_items)?;
        Ok((0..n_items).map(|i| (0..plan.k).map(|j| rec[j * n_items + i]).collect()).collect())
    }

    /// BLAZE_MTP_DUMP=dir: the first drafting call of item 0 for the reference (scripts/ref_mtp.py): its tokens, first
    /// position and the target's final hidden states of its rows, the drafts, and the normed output of the draft after the
    /// extend and every chain step (the head's input), one file each.
    fn run_dump(&mut self, model: &Model, plan: &DraftPlan, steps: &[Fwd], dir: &Path) -> Result<Vec<Vec<i64>>> {
        let (h, s, n_items, k) = (self.cfg.hidden, &model.stream, plan.items.len(), plan.k);
        std::fs::create_dir_all(dir)?;
        let it = &plan.items[0];
        let aux = model.aux.as_ref().ok_or_else(|| anyhow!("no captured hidden states"))?;
        let all: Vec<u16> = aux.download((it.aux_row + it.tokens.len()) * h)?;
        let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("hidden.bin"), bytes(&all[it.aux_row * h..]))?;
        // eager chain with the head input kept after every step
        self.head(model, n_items, 0)?;
        s.sync()?;
        let mut normed = vec![self.last_normed.download::<u16>(n_items * h)?[..h].to_vec()];
        for (j, fwd) in steps.iter().enumerate() {
            self.next_hidden(n_items, s)?;
            self.layer(model, fwd, self.tok.ptr())?;
            for i in 0..n_items {
                self.keep_last(i, i, s)?;
            }
            self.head(model, n_items, j + 1)?;
            s.sync()?;
            normed.push(self.last_normed.download::<u16>(n_items * h)?[..h].to_vec());
        }
        let rec: Vec<i64> = self.record.download(k * n_items)?;
        let drafts: Vec<Vec<i64>> = (0..n_items).map(|i| (0..k).map(|j| rec[j * n_items + i]).collect()).collect();
        for (j, v) in normed.iter().enumerate() {
            std::fs::write(dir.join(format!("normed_{j}.bin")), bytes(v))?;
        }
        let meta = serde_json::json!({"tokens": it.tokens, "pos0": it.pos0, "drafts": drafts[0], "hidden": h, "window": self.cfg.window});
        std::fs::write(dir.join("meta.json"), meta.to_string())?;
        eprintln!("blaze: MTP dump of {} rows and {k} drafts in {}", it.tokens.len(), dir.display());
        Ok(drafts)
    }

    /// The extend forwards: the target's final hidden states of the items' rows and their next tokens through the layer;
    /// the last row of every item feeds the head and the chain.
    fn extend(&self, model: &Model, plan: &DraftPlan, fwds: &[Fwd], ext_start: &[Vec<usize>]) -> Result<()> {
        let (wd, s) = (self.width(), &model.stream);
        let aux = model.aux.as_ref().ok_or_else(|| anyhow!("MTP without captured hidden states"))?;
        for (f, fwd) in fwds.iter().enumerate() {
            let mut row = 0;
            for (si, &(i, len)) in fwd.spans.iter().enumerate() {
                let a: *const u16 = aux.at((plan.items[i].aux_row + ext_start[f][si]) * wd);
                check(unsafe { sys::cudaMemcpyAsync(self.hid.at::<u16>(row * wd).cast(), a.cast(), len * wd * 2, sys::MEMCPY_D2D, s.0) }, "MTP extend hidden")?;
                row += len;
            }
            let tok: *const i64 = self.meta.at::<u8>(fwd.tok).cast();
            self.layer(model, fwd, tok)?;
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

    /// Run `body` through a CUDA graph per `key`: one eager run, then the capture, then replays.
    fn graphed(&mut self, model: &Model, key: (usize, usize, bool), body: impl Fn(&Self) -> Result<()>) -> Result<()> {
        let s = &model.stream;
        if let Some(g) = self.graphs.get(&key) {
            return check(unsafe { sys::cudaGraphLaunch(g.exec, s.0) }, "MTP graph launch");
        }
        if self.graphs.len() >= MAX_GRAPHS || self.warmed.insert(key) {
            return body(self);
        }
        s.sync()?;
        check(unsafe { sys::cudaStreamBeginCapture(s.0, sys::CAPTURE_MODE_THREAD_LOCAL) }, "MTP capture")?;
        let captured = body(self);
        let mut graph = std::ptr::null_mut();
        let end = unsafe { sys::cudaStreamEndCapture(s.0, &mut graph) };
        captured?;
        check(end, "MTP capture end")?;
        let mut exec = std::ptr::null_mut();
        check(unsafe { sys::cudaGraphInstantiate(&mut exec, graph, 0) }, "MTP graph instantiate")?;
        let g = Graph { exec, graph };
        check(unsafe { sys::cudaGraphLaunch(g.exec, s.0) }, "MTP graph launch")?;
        self.graphs.insert(key, g);
        Ok(())
    }

    /// The head on the extend's last rows, then every chain step: the previous row's output and the token just picked
    /// through the layer, the head again.
    fn chain(&self, model: &Model, items: usize, steps: &[Fwd]) -> Result<()> {
        let s = &model.stream;
        self.head(model, items, 0)?;
        for (j, fwd) in steps.iter().enumerate() {
            self.next_hidden(items, s)?;
            self.layer(model, fwd, self.tok.ptr())?;
            for i in 0..items {
                self.keep_last(i, i, s)?;
            }
            self.head(model, items, j + 1)?;
        }
        Ok(())
    }

    /// Row `r` of the forward's head input becomes item `i`'s, its output (the normed row, or the streams) the item's
    /// next hidden input.
    fn keep_last(&self, i: usize, r: usize, s: &crate::gpu::Stream) -> Result<()> {
        let h = self.cfg.hidden;
        check(
            unsafe { sys::cudaMemcpyAsync(self.last_normed.at::<u16>(i * h).cast(), self.normed.at::<u16>(r * h).cast(), h * 2, sys::MEMCPY_D2D, s.0) },
            "MTP keep",
        )?;
        if let Body::Hyper(y) = &self.body {
            let hw = y.count * h;
            check(
                unsafe { sys::cudaMemcpyAsync(y.last_streams.at::<u16>(i * hw).cast(), y.streams.at::<u16>(r * hw).cast(), hw * 2, sys::MEMCPY_D2D, s.0) },
                "MTP keep streams",
            )?;
        }
        Ok(())
    }

    /// The hidden inputs of a chain step: every item's kept output.
    fn next_hidden(&self, items: usize, s: &crate::gpu::Stream) -> Result<()> {
        let (src, wd) = match &self.body {
            Body::Dense(_) => (self.last_normed.ptr::<u16>(), self.cfg.hidden),
            Body::Hyper(y) => (y.last_streams.ptr::<u16>(), y.count * self.cfg.hidden),
        };
        check(unsafe { sys::cudaMemcpyAsync(self.hid.ptr(), src.cast_const().cast(), items * wd * 2, sys::MEMCPY_D2D, s.0) }, "MTP chain hidden")
    }

    /// The draft layer over the forward's rows: hidden inputs in `hid`, tokens at `tok`; every row's head input in
    /// `normed` (and its streams in the hyper body's `streams`).
    fn layer(&self, model: &Model, fwd: &Fwd, tok: *const i64) -> Result<()> {
        match &self.body {
            Body::Dense(d) => self.layer_dense(model, d, fwd, tok),
            Body::Hyper(y) => self.layer_hyper(model, y, fwd, tok),
        }
    }

    fn layer_dense(&self, model: &Model, d: &Dense, fwd: &Fwd, tok: *const i64) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let h = c.hidden;
        let n: usize = fwd.spans.iter().map(|s| s.1).sum();
        // x = fc(concat(norm(embed(token)), norm(hidden)))
        check(
            unsafe { sys::blz_embed_gather(tok, n as i64, model.embed_table().dev_ptr(), c.vocab as i64, (h * 2) as i64, d.emb.ptr(), s.0) },
            "MTP embed",
        )?;
        check(
            unsafe { sys::blz_rmsnorm_ld_zc(d.emb.ptr(), h as i64, d.pre_emb.ptr(), c.eps, n as c_int, h as c_int, d.cat.ptr(), (2 * h) as i64, s.0) },
            "MTP embedding norm",
        )?;
        check(
            unsafe { sys::blz_rmsnorm_ld_zc(self.hid.ptr(), h as i64, d.pre_hid.ptr(), c.eps, n as c_int, h as c_int, d.cat.at::<u16>(h), (2 * h) as i64, s.0) },
            "MTP hidden norm",
        )?;
        d.fc.run(lin, d.cat.ptr(), 2 * h, n, d.hin.ptr(), h, s)?;
        check(
            unsafe { sys::blz_rmsnorm(d.hin.ptr(), std::ptr::null_mut(), d.input_ln.ptr(), c.eps, n as c_int, h as c_int, 1, d.normed1.ptr(), s.0) },
            "MTP input norm",
        )?;
        self.attention(model, fwd, d.normed1.ptr())?;
        // x += attention (summed over the ranks); x2 = norm(x)
        self.add_norm(d, self.o_out.ptr(), d.post_ln.ptr(), d.x2.ptr(), n, s)?;
        d.gu.run(lin, d.x2.ptr(), h, n, d.gu_out.ptr(), 2 * d.inter, s)?;
        check(
            unsafe { sys::blz_silu_mul(d.gu_out.ptr(), (2 * d.inter) as i64, d.gu_out.at(d.inter), (2 * d.inter) as i64, n as c_int, d.inter as c_int, d.act.ptr(), s.0) },
            "MTP silu",
        )?;
        d.down.run(lin, d.act.ptr(), d.inter, n, d.mlp.ptr(), h, s)?;
        // x += mlp; normed = mtp.norm(x)
        self.add_norm(d, d.mlp.ptr(), d.norm.ptr(), self.normed.ptr(), n, s)
    }

    /// The streams' mix for a block: their grouped norm into `hnorm` (the combine's gates read it), the block's input in
    /// `out`.
    fn hc_mix(&self, y: &Hyper, hc: &HcBlock, n: usize, out: *mut u16, s: &crate::gpu::Stream) -> Result<()> {
        let h = self.cfg.hidden;
        check(
            unsafe { sys::blz_grouped_rmsnorm(y.streams.ptr(), hc.norm.ptr(), self.cfg.eps, n as c_int, y.count as c_int, h as c_int, y.hnorm.ptr(), s.0) },
            "MTP hc norm",
        )?;
        check(
            unsafe {
                sys::blz_hc_mix(y.hnorm.ptr(), hc.down.ptr(), hc.up.ptr(), y.hc_t.ptr(), out, n as c_int, y.count as c_int, h as c_int,
                                y.lowrank as c_int, y.hc_ws.ptr(), s.0)
            },
            "MTP hc mix",
        )
    }

    /// A block's output back into the streams (gated injection, gates from the mix's norm).
    fn hc_combine(&self, y: &Hyper, hc: &HcBlock, out: *const u16, n: usize, s: &crate::gpu::Stream) -> Result<()> {
        let h = self.cfg.hidden;
        let inject = hc.inject.as_ref().ok_or_else(|| anyhow!("MTP: a hyper-connection block without block_inject_weight"))?;
        check(
            unsafe {
                sys::blz_hc_combine(y.streams.ptr(), y.hnorm.ptr(), inject.ptr(), out, h as i64, n as c_int, y.count as c_int, h as c_int, s.0)
            },
            "MTP hc combine",
        )
    }

    fn layer_hyper(&self, model: &Model, y: &Hyper, fwd: &Fwd, tok: *const i64) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let h = c.hidden;
        let hw = y.count * h;
        let n: usize = fwd.spans.iter().map(|s| s.1).sum();
        // S_s = fc_hidden(norm_h(H)_s) + fc_embedding(norm_e(embed(token)))
        check(
            unsafe { sys::blz_embed_gather(tok, n as i64, model.embed_table().dev_ptr(), c.vocab as i64, (h * 2) as i64, y.emb.ptr(), s.0) },
            "MTP embed",
        )?;
        check(
            unsafe { sys::blz_rmsnorm_ld_zc(y.emb.ptr(), h as i64, y.pre_emb.ptr(), c.eps, n as c_int, h as c_int, y.block.ptr(), h as i64, s.0) },
            "MTP embedding norm",
        )?;
        y.fc_emb.run(lin, y.block.ptr(), h, n, y.eproj.ptr(), h, s)?;
        check(
            unsafe { sys::blz_rmsnorm_ld_zc(self.hid.ptr(), hw as i64, y.pre_hid.ptr(), c.eps, n as c_int, hw as c_int, y.hnorm.ptr(), hw as i64, s.0) },
            "MTP hidden norm",
        )?;
        // every stream a row of fc_hidden
        y.fc_hid.run(lin, y.hnorm.ptr(), h, n * y.count, y.hproj.ptr(), h, s)?;
        check(unsafe { sys::blz_hc_widen(y.eproj.ptr(), n as c_int, y.count as c_int, h as c_int, y.streams.ptr(), s.0) }, "MTP widen")?;
        check(unsafe { sys::blz_add(y.streams.ptr(), y.hproj.ptr(), (n * hw) as i64, y.streams.ptr(), s.0) }, "MTP fusion")?;
        // attention between its mix and combine
        self.hc_mix(y, &y.attn_hc, n, y.block.ptr(), s)?;
        self.attention(model, fwd, y.block.ptr())?;
        self.hc_combine(y, &y.attn_hc, self.o_out.ptr(), n, s)?;
        // the routed MLP (the experts in the expert engine) and the shared expert, gated
        self.hc_mix(y, &y.mlp_hc, n, y.block.ptr(), s)?;
        y.gate.run(lin, y.block.ptr(), h, n, y.router.ptr(), y.gate.n, s)?;
        check(
            unsafe {
                sys::blz_router_topk(y.router.ptr(), y.gate.n as i64, n as c_int, y.experts as c_int, y.top_k as c_int, y.norm_topk as c_int,
                                     y.ids.ptr(), y.weights.ptr(), s.0)
            },
            "MTP router",
        )?;
        model.moe_forward(y.moe_layer, y.block.ptr(), n, y.ids.ptr(), y.weights.ptr(), y.moe_out.ptr())?;
        let si = y.shared_inter;
        y.shared_gu.run(lin, y.block.ptr(), h, n, y.sgu.ptr(), 2 * si, s)?;
        check(
            unsafe { sys::blz_silu_mul(y.sgu.ptr(), (2 * si) as i64, y.sgu.at(si), (2 * si) as i64, n as c_int, si as c_int, y.sact.ptr(), s.0) },
            "MTP shared silu",
        )?;
        y.shared_down.run(lin, y.sact.ptr(), si, n, y.shared.ptr(), h, s)?;
        if y.has_shared_gate {
            check(
                unsafe {
                    sys::blz_add_sigmoid_scaled(y.moe_out.ptr(), y.shared.ptr(), y.router.at(y.experts), y.gate.n as i64, n as c_int, h as c_int,
                                                y.x.ptr(), s.0)
                },
                "MTP shared gate",
            )?;
        } else {
            check(unsafe { sys::blz_add(y.moe_out.ptr(), y.shared.ptr(), (n * h) as i64, y.x.ptr(), s.0) }, "MTP shared add")?;
        }
        self.hc_combine(y, &y.mlp_hc, y.x.ptr(), n, s)?;
        // the head's input: the final mixer's mix (the streams stay for the next draft row)
        self.hc_mix(y, &y.mixer, n, self.normed.ptr(), s)
    }

    /// The layer's attention on x (its rows of the forward): gated query, q / k norms, partial rotary, KV into the ring;
    /// the output projection in `o_out` (this rank's partial sums).
    fn attention(&self, model: &Model, fwd: &Fwd, x: *const u16) -> Result<()> {
        let (c, s, lin) = (&self.cfg, &model.stream, model.lin_ctx());
        let (h, hd) = (c.hidden, c.head_dim);
        let (heads, kv_heads) = (self.heads, self.kv_heads);
        let (qd, kvd) = (heads * hd, kv_heads * hd);
        let ldq = 2 * qd + 2 * kvd;
        let n: usize = fwd.spans.iter().map(|s| s.1).sum();
        let pos: *const i64 = self.meta.at::<u8>(fwd.pos).cast();
        let loc: *const i64 = self.meta.at::<u8>(fwd.loc).cast();
        let (inv_freq, rope_scale) = model.rope_tables();
        // attention: gated query, q / k norms, partial rotary, KV into the ring
        self.qkv.run(lin, x, h, n, self.qkv_out.ptr(), ldq, s)?;
        let k: *mut u16 = self.qkv_out.at(2 * qd);
        let v: *const u16 = self.qkv_out.at(2 * qd + kvd);
        check(
            unsafe {
                sys::blz_attn_prep(self.qkv_out.ptr(), ldq as i64, k, ldq as i64, pos, self.q_norm.ptr(), self.k_norm.ptr(), c.eps, 1, n as c_int,
                                   heads as c_int, kv_heads as c_int, hd as c_int, c.rotary_dim as c_int, inv_freq, rope_scale,
                                   self.q_out.ptr(), self.gate.ptr(), s.0)
            },
            "MTP attention prep",
        )?;
        check(
            unsafe {
                sys::blz_kv_store_dense(k.cast(), v.cast(), ldq as i64, ldq as i64, loc.cast(), sys::BLZ_I64, n as i64, kv_heads as c_int, hd as c_int,
                                        sys::BLZ_KV_BF16, self.k_pool.ptr(), self.v_pool.ptr(), self.one.ptr(), self.one.ptr(), s.0)
            },
            "MTP kv store",
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
                        self.q_out.at::<u16>(r0 * qd).cast(),
                        qd as i64,
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
                "MTP attention",
            )?;
        }
        check(unsafe { sys::blz_mul_sigmoid(self.attn.ptr(), self.gate.ptr(), (n * qd) as i64, s.0) }, "MTP attention gate")?;
        self.o.run(lin, self.attn.ptr(), qd, n, self.o_out.ptr(), h, s)?;
        Ok(())
    }

    /// hin += x (all-reduced over the ranks first), out = rmsnorm(hin, w) with the zero-centered weight.
    fn add_norm(&self, d: &Dense, x: *mut u16, w: *const u16, out: *mut u16, n: usize, s: &crate::gpu::Stream) -> Result<()> {
        let (h, eps) = (self.cfg.hidden, self.cfg.eps);
        if !self.ar.is_null() && n <= 64 {
            return check(unsafe { sys::blz_host_ar_rmsnorm(self.ar, x, d.hin.ptr(), w, eps, n as c_int, h as c_int, 1, out, s.0) }, "MTP all-reduce norm");
        }
        if !self.ar.is_null() {
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, x.cast(), (n * h * 2) as i64, sys::BLZ_AR_BF16, s.0) }, "MTP all-reduce")?;
        }
        check(unsafe { sys::blz_rmsnorm(x, d.hin.ptr(), w, eps, n as c_int, h as c_int, 1, out, s.0) }, "MTP norm")
    }

    /// The candidate head: the logits of every item's candidate ids (rows of this rank's head shard, the metadata's first
    /// region), the best of them as this rank's pair, the pairs merged over the ranks.
    fn head_cands(&self, model: &Model, items: usize, step: usize) -> Result<()> {
        let (s, h, c) = (&model.stream, self.cfg.hidden, self.cand_cap);
        let (rank, world, shard) = (model.tp.rank, model.tp.world, model.dims.vocab);
        let ids: *const i32 = self.meta.ptr();
        let bytes = (world * items * 8).next_multiple_of(16);
        check(unsafe { sys::cudaMemsetAsync(self.pairs.ptr(), 0, bytes, s.0) }, "MTP memset pairs")?;
        check(
            unsafe {
                sys::blz_candidate_logits(self.last_normed.ptr(), h as i64, items as c_int, h as c_int, self.table.unwrap(), ids, c as c_int,
                                          self.cand_logits.ptr(), s.0)
            },
            "MTP candidate logits",
        )?;
        check(
            unsafe { sys::blz_cand_argmax(self.cand_logits.ptr(), ids, items as c_int, c as c_int, (rank * shard) as c_int, self.pairs.at::<f32>(rank * items * 2), s.0) },
            "MTP candidate argmax",
        )?;
        if !self.ar.is_null() {
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, self.pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "MTP head all-reduce")?;
        }
        check(unsafe { sys::blz_argmax_merge(self.pairs.ptr(), world as c_int, items as c_int, self.pick.ptr(), s.0) }, "MTP argmax merge")?;
        check(unsafe { sys::blz_widen_i32(self.pick.ptr(), self.tok.ptr(), self.record.at::<i64>(step * items), items as c_int, s.0) }, "MTP pick")
    }

    /// The target's LM head on `last_normed` (this rank's vocabulary shard, the (value, index) pairs merged over the
    /// ranks): the greedy draft token of every item in `tok` and in the record at `step`.
    fn head(&self, model: &Model, items: usize, step: usize) -> Result<()> {
        if self.use_cands {
            return self.head_cands(model, items, step);
        }
        let (s, h) = (&model.stream, self.cfg.hidden);
        let (rank, world, shard) = (model.tp.rank, model.tp.world, model.dims.vocab);
        let rows = self.head_rows;
        let bytes = (world * items * 8).next_multiple_of(16);
        check(unsafe { sys::cudaMemsetAsync(self.pairs.ptr(), 0, bytes, s.0) }, "MTP memset pairs")?;
        if rows == 0 {
            // this rank's shard lies beyond the restricted vocabulary: a pair no pick can beat
            check(
                unsafe { sys::cudaMemcpyAsync(self.pairs.at::<f32>(rank * items * 2).cast(), self.no_pairs.ptr(), items * 8, sys::MEMCPY_D2D, s.0) },
                "MTP empty pairs",
            )?;
        } else {
            match &self.head_lin {
                Some(l) => l.run(model.lin_ctx(), self.last_normed.ptr(), h, items, self.logits.ptr(), rows, s)?,
                None => model.lm_head().run(model.lin_ctx(), self.last_normed.ptr(), h, items, self.logits.ptr(), shard, s)?,
            }
            check(
                unsafe {
                    sys::blz_argmax_part(self.logits.ptr(), items as c_int, rows as c_int, rows as i64, self.head_first as c_int,
                                         self.pairs.at::<f32>(rank * items * 2), s.0)
                },
                "MTP argmax part",
            )?;
        }
        if !self.ar.is_null() {
            check(unsafe { sys::blz_host_ar_all_reduce(self.ar, self.pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "MTP head all-reduce")?;
        }
        check(unsafe { sys::blz_argmax_merge(self.pairs.ptr(), world as c_int, items as c_int, self.pick.ptr(), s.0) }, "MTP argmax merge")?;
        check(
            unsafe { sys::blz_widen_i32(self.pick.ptr(), self.tok.ptr(), self.record.at::<i64>(step * items), items as c_int, s.0) },
            "MTP pick",
        )
    }
}
