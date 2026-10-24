//! Qwen image preprocessing and vision towers with BF16 checkpoint weights.
//! The GPU path uses CUDA/cuBLAS; the CPU path uses AVX2/FMA with FP32 activations.

use crate::gpu::{self, DevBuf, Stream, check};
use crate::weights::{Checkpoint, Dtype, TensorInfo};
use anyhow::{Context, Result, anyhow, bail};
use blaze_sys as sys;
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

// configuration

/// The vision tower's shape (`vision_config` of config.json).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub channels: usize,
    pub num_pos: usize,
    /// width of the merged embeddings: the language model's hidden size
    pub out_hidden: usize,
    /// block MLP activation: GELU with the tanh approximation (else exact)
    pub act_tanh: bool,
    pub theta: f32,
    /// tensor name prefix ("model.visual.")
    pub prefix: String,
    pub image_token_id: i64,
}

impl VisionConfig {
    /// None for a checkpoint without a vision tower.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("config.json");
        let root: Value = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?)?;
        let Some(v) = root.get("vision_config") else { return Ok(None) };
        let kind = v.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !kind.starts_with("qwen3_5") {
            bail!("vision tower of type '{kind}' is not supported (Qwen3.5 family only)");
        }
        if v.get("deepstack_visual_indexes").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) {
            bail!("vision tower with deepstack features is not supported");
        }
        let n = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| anyhow!("vision_config: missing `{k}`"));
        let act_tanh = match v.get("hidden_act").and_then(Value::as_str).unwrap_or("gelu_pytorch_tanh") {
            "gelu_pytorch_tanh" | "gelu_tanh" | "gelu_new" => true,
            "gelu" => false,
            other => bail!("vision_config: activation '{other}' is not supported"),
        };
        let theta = v.pointer("/rope_parameters/rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32;
        let ckpt_prefix = if std::fs::read_dir(dir)?.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "safetensors")) {
            // both layouts exist in the wild: "model.visual." (transformers 5) and "visual."
            let c = Checkpoint::open(dir)?;
            if c.tensors.contains_key("model.visual.patch_embed.proj.weight") { "model.visual." } else { "visual." }
        } else {
            "model.visual."
        };
        Ok(Some(Self {
            depth: n("depth")?,
            hidden: n("hidden_size")?,
            heads: n("num_heads")?,
            intermediate: n("intermediate_size")?,
            patch: n("patch_size")?,
            temporal: n("temporal_patch_size")?,
            merge: n("spatial_merge_size")?,
            channels: n("in_channels").unwrap_or(3),
            num_pos: n("num_position_embeddings")?,
            out_hidden: n("out_hidden_size")?,
            act_tanh,
            theta,
            prefix: ckpt_prefix.into(),
            image_token_id: root.get("image_token_id").and_then(Value::as_i64).unwrap_or(-1),
        }))
    }

    /// Values per patch row of the encoder's input.
    pub fn patch_dim(&self) -> usize {
        self.channels * self.temporal * self.patch * self.patch
    }

    /// Bytes of the tower's weights (BF16).
    pub fn weight_bytes(&self) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        let block = 3 * h * h + 3 * h + h * h + h + h * i + i + i * h + h + 4 * h;
        let merger = 2 * h + 4 * h * 4 * h + 4 * h + self.out_hidden * 4 * h + self.out_hidden;
        2 * (self.depth * block + merger + self.patch_dim() * h + h + self.num_pos * h)
    }
}

/// How the image processor sizes and normalizes (preprocessor_config.json / processor_config.json of the checkpoint, with
/// the request's `mm_processor_kwargs` on top).
#[derive(Clone, Debug)]
pub struct ImageSettings {
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub patch: usize,
    pub merge: usize,
    pub temporal: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f64,
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
}

/// The processor fields a request or the server can override (vLLM's `mm_processor_kwargs`): `min_pixels` / `max_pixels`
/// and their aliases `size.shortest_edge` / `size.longest_edge`, `do_resize`, `do_rescale`, `do_normalize`,
/// `rescale_factor`, `image_mean`, `image_std`; also nested in `images_kwargs`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overrides {
    pub min_pixels: Option<u64>,
    pub max_pixels: Option<u64>,
    pub do_resize: Option<bool>,
    pub do_rescale: Option<bool>,
    pub do_normalize: Option<bool>,
    pub rescale_factor: Option<f64>,
    pub mean: Option<[f32; 3]>,
    pub std: Option<[f32; 3]>,
}

impl Overrides {
    /// One scope of kwargs. As in vLLM's Qwen-VL processor info, `size` holds the edges as pixel counts, `min_pixels` /
    /// `max_pixels` are equivalent and win over them; a null value overrides nothing.
    fn level(v: &Value, what: &str) -> std::result::Result<Self, String> {
        let obj = v.as_object().ok_or_else(|| format!("{what} must be an object"))?;
        let pixels = |key: &str, edge: &str| -> std::result::Result<Option<u64>, String> {
            let direct = obj.get(key).filter(|x| !x.is_null());
            let size = obj.get("size").and_then(|s| s.get(edge)).filter(|x| !x.is_null());
            match direct.or(size) {
                None => Ok(None),
                Some(x) => match x.as_u64().filter(|&n| n > 0) {
                    Some(n) => Ok(Some(n)),
                    None => Err(format!("{what}.{key} must be a positive integer")),
                },
            }
        };
        let flag = |key: &str| -> std::result::Result<Option<bool>, String> {
            match obj.get(key).filter(|x| !x.is_null()) {
                None => Ok(None),
                Some(x) => x.as_bool().map(Some).ok_or_else(|| format!("{what}.{key} must be a boolean")),
            }
        };
        let triple = |key: &str| -> std::result::Result<Option<[f32; 3]>, String> {
            match obj.get(key).filter(|x| !x.is_null()) {
                None => Ok(None),
                Some(x) => {
                    let a: Vec<f32> = x.as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|f| f as f32).collect()).unwrap_or_default();
                    <[f32; 3]>::try_from(a).map(Some).map_err(|_| format!("{what}.{key} must be a list of 3 numbers"))
                }
            }
        };
        Ok(Self {
            min_pixels: pixels("min_pixels", "shortest_edge")?,
            max_pixels: pixels("max_pixels", "longest_edge")?,
            do_resize: flag("do_resize")?,
            do_rescale: flag("do_rescale")?,
            do_normalize: flag("do_normalize")?,
            rescale_factor: obj.get("rescale_factor").and_then(Value::as_f64),
            mean: triple("image_mean")?,
            std: triple("image_std")?,
        })
    }

    /// A `mm_processor_kwargs` object: flat keys, with `images_kwargs` (the image scope) winning on conflicts.
    pub fn parse(v: &Value) -> std::result::Result<Self, String> {
        let flat = Self::level(v, "mm_processor_kwargs")?;
        match v.get("images_kwargs").filter(|x| !x.is_null()) {
            Some(nested) => Ok(flat.over(Self::level(nested, "mm_processor_kwargs.images_kwargs")?)),
            None => Ok(flat),
        }
    }

    /// `self` with the fields `hi` sets taken from `hi` (request over server over checkpoint).
    pub fn over(self, hi: Self) -> Self {
        Self {
            min_pixels: hi.min_pixels.or(self.min_pixels),
            max_pixels: hi.max_pixels.or(self.max_pixels),
            do_resize: hi.do_resize.or(self.do_resize),
            do_rescale: hi.do_rescale.or(self.do_rescale),
            do_normalize: hi.do_normalize.or(self.do_normalize),
            rescale_factor: hi.rescale_factor.or(self.rescale_factor),
            mean: hi.mean.or(self.mean),
            std: hi.std.or(self.std),
        }
    }
}

impl ImageSettings {
    /// The checkpoint's processor config: `size` edges (`shortest_edge` / `longest_edge`, pixel counts) or
    /// `min_pixels` / `max_pixels`, patch and merge sizes, mean and std. What the config leaves out takes the values of
    /// the Hugging Face Qwen2VLImageProcessor class (the processor's own defaults, as vLLM uses them).
    pub fn load(dir: &Path, cfg: &VisionConfig) -> Result<Self> {
        let read = |name: &str| -> Option<Value> { serde_json::from_str(&std::fs::read_to_string(dir.join(name)).ok()?).ok() };
        let j = read("preprocessor_config.json")
            .or_else(|| read("processor_config.json").and_then(|p| p.get("image_processor").cloned()))
            .unwrap_or(Value::Null);
        let size = j.get("size");
        let pick = |edge: &str, pixels: &str, default: u64| -> u64 {
            size.and_then(|s| s.get(edge)).and_then(Value::as_u64).or_else(|| j.get(pixels).and_then(Value::as_u64)).unwrap_or(default)
        };
        let arr3 = |key: &str, default: [f32; 3]| -> [f32; 3] {
            let a: Vec<f32> = j.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).map(|f| f as f32).collect()).unwrap_or_default();
            <[f32; 3]>::try_from(a).unwrap_or(default)
        };
        let us = |key: &str, d: usize| j.get(key).and_then(Value::as_u64).map(|x| x as usize).unwrap_or(d);
        let s = Self {
            min_pixels: pick("shortest_edge", "min_pixels", 56 * 56),
            max_pixels: pick("longest_edge", "max_pixels", 28 * 28 * 1280),
            patch: us("patch_size", cfg.patch),
            merge: us("merge_size", cfg.merge),
            temporal: us("temporal_patch_size", cfg.temporal),
            mean: arr3("image_mean", [0.481_454_66, 0.457_827_5, 0.408_210_73]),
            std: arr3("image_std", [0.268_629_54, 0.261_302_58, 0.275_777_11]),
            rescale: j.get("rescale_factor").and_then(Value::as_f64).unwrap_or(1.0 / 255.0),
            do_resize: j.get("do_resize").and_then(Value::as_bool).unwrap_or(true),
            do_rescale: j.get("do_rescale").and_then(Value::as_bool).unwrap_or(true),
            do_normalize: j.get("do_normalize").and_then(Value::as_bool).unwrap_or(true),
        };
        if s.patch != cfg.patch || s.merge != cfg.merge || s.temporal != cfg.temporal {
            bail!("the image processor's patch / merge / temporal sizes disagree with vision_config");
        }
        Ok(s)
    }

    pub fn apply(&self, o: &Overrides) -> Self {
        Self {
            min_pixels: o.min_pixels.unwrap_or(self.min_pixels),
            max_pixels: o.max_pixels.unwrap_or(self.max_pixels),
            do_resize: o.do_resize.unwrap_or(self.do_resize),
            do_rescale: o.do_rescale.unwrap_or(self.do_rescale),
            do_normalize: o.do_normalize.unwrap_or(self.do_normalize),
            rescale: o.rescale_factor.unwrap_or(self.rescale),
            mean: o.mean.unwrap_or(self.mean),
            std: o.std.unwrap_or(self.std),
            ..self.clone()
        }
    }

    /// The size an image of `w` x `h` is resized to, as the processor's `smart_resize` (HF Qwen2VL): both sides multiples
    /// of patch * merge, the pixel count within [min_pixels, max_pixels], the aspect ratio kept as well as the grid allows.
    pub fn target_size(&self, w: usize, h: usize) -> std::result::Result<(usize, usize), String> {
        let factor = self.patch * self.merge;
        if !self.do_resize {
            if h % factor != 0 || w % factor != 0 {
                return Err(format!("with do_resize false the image size {w}x{h} must be a multiple of {factor}"));
            }
            return Ok((h, w));
        }
        if w == 0 || h == 0 {
            return Err("the image is empty".into());
        }
        if w.max(h) as f64 / w.min(h) as f64 > 200.0 {
            return Err(format!("absolute aspect ratio must be smaller than 200, got {}", w.max(h) as f64 / w.min(h) as f64));
        }
        let f = factor as f64;
        let (hf, wf) = (h as f64, w as f64);
        let mut hb = ((hf / f).round_ties_even() * f) as u64;
        let mut wb = ((wf / f).round_ties_even() * f) as u64;
        if hb * wb > self.max_pixels {
            let beta = (hf * wf / self.max_pixels as f64).sqrt();
            hb = factor.max(((hf / beta / f).floor() * f) as usize) as u64;
            wb = factor.max(((wf / beta / f).floor() * f) as usize) as u64;
        } else if hb * wb < self.min_pixels {
            let beta = (self.min_pixels as f64 / (hf * wf)).sqrt();
            hb = ((hf * beta / f).ceil() * f) as u64;
            wb = ((wf * beta / f).ceil() * f) as u64;
        }
        Ok((hb as usize, wb as usize))
    }

    /// Merged tokens (rows of the encoder's output) of an image of `w` x `h`: (grid_h, grid_w) of the merged grid.
    pub fn merged_grid(&self, w: usize, h: usize) -> std::result::Result<(usize, usize), String> {
        let (th, tw) = self.target_size(w, h)?;
        Ok((th / self.patch / self.merge, tw / self.patch / self.merge))
    }
}

// image processor

/// A decoded image: RGB8, rows top to bottom.
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// vLLM's VLLM_MAX_IMAGE_PIXELS: the largest decoded image.
pub const MAX_DECODE_PIXELS: u64 = 178_956_970;

/// Decode PNG, JPEG, GIF (first frame), WebP or BMP. Transparency is composited over white (vLLM's default
/// `rgba_background_color`).
pub fn decode_image(bytes: &[u8]) -> std::result::Result<RgbImage, String> {
    let open = || image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| format!("cannot read the image: {e}"));
    let (w, h) = open()?.into_dimensions().map_err(|e| format!("cannot decode the image: {e}"))?;
    if u64::from(w) * u64::from(h) > MAX_DECODE_PIXELS {
        return Err(format!("image dimensions {w}x{h} ({} pixels) exceed the maximum of {MAX_DECODE_PIXELS} pixels", u64::from(w) * u64::from(h)));
    }
    let rgba = open()?.decode().map_err(|e| format!("cannot decode the image: {e}"))?.to_rgba8();
    let mut data = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.as_raw().chunks_exact(4) {
        let a = u32::from(px[3]);
        for &c in &px[..3] {
            data.push(if a == 255 { c } else { ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8 });
        }
    }
    Ok(RgbImage { w: w as usize, h: h as usize, data })
}

/// Taps of one output sample per axis: first input index and normalized weights (Pillow's / torch's antialiased bicubic,
/// a = -0.5, support widened by the shrink factor).
fn axis_taps(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let cubic = |x: f64| {
        let a = -0.5;
        let x = x.abs();
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    (0..output)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = (((center + support + 0.5) as i64).max(0) as usize).min(input);
            let mut w: Vec<f64> = (lo..hi).map(|i| cubic((i as f64 - center + 0.5) / filter_scale)).collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|x| *x /= total);
            }
            (lo, w.into_iter().map(|x| x as f32).collect())
        })
        .collect()
}

/// Resize RGB8 with the antialiased bicubic of the HF fast image processor (torchvision resize of a uint8 tensor: float
/// separable filter, rounded to the nearest integer).
fn resize_bicubic(src: &RgbImage, dw: usize, dh: usize) -> Vec<u8> {
    if src.w == dw && src.h == dh {
        return src.data.clone();
    }
    let (tx, ty) = (axis_taps(src.w, dw), axis_taps(src.h, dh));
    // horizontal pass: [h][dw][3] in f32
    let mut mid = vec![0f32; src.h * dw * 3];
    for y in 0..src.h {
        let row = &src.data[y * src.w * 3..(y + 1) * src.w * 3];
        for (x, (lo, w)) in tx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, &wk) in w.iter().enumerate() {
                let p = (lo + k) * 3;
                acc[0] += f32::from(row[p]) * wk;
                acc[1] += f32::from(row[p + 1]) * wk;
                acc[2] += f32::from(row[p + 2]) * wk;
            }
            mid[(y * dw + x) * 3..(y * dw + x) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut out = vec![0u8; dh * dw * 3];
    for (y, (lo, w)) in ty.iter().enumerate() {
        for x in 0..dw * 3 {
            let mut acc = 0f32;
            for (k, &wk) in w.iter().enumerate() {
                acc += mid[(lo + k) * dw * 3 + x] * wk;
            }
            out[y * dw * 3 + x] = acc.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// The encoder's input of one image.
pub struct Patches {
    /// [grid_h * grid_w, channels * temporal * patch * patch] bf16, normalized; patches in spatial-merge order (each
    /// merge x merge window of the grid consecutive); every row is laid out [channel][temporal][y][x] with the still
    /// image repeated over the temporal patch
    pub pixels: Vec<u16>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Patches {
    pub fn tokens(&self, merge: usize) -> usize {
        self.grid_h * self.grid_w / (merge * merge)
    }
}

/// Resize, rescale, normalize and patchify one image (HF Qwen2VLImageProcessor).
pub fn preprocess(img: &RgbImage, s: &ImageSettings) -> std::result::Result<Patches, String> {
    let (th, tw) = s.target_size(img.w, img.h)?;
    let px = if s.do_resize { resize_bicubic(img, tw, th) } else { img.data.clone() };
    let (p, m, t) = (s.patch, s.merge, s.temporal);
    let (gh, gw) = (th / p, tw / p);
    // x -> (x * rescale - mean) / std with rescale and mean fused as torchvision's fast path does:
    // (x - mean / rescale) / (std / rescale)
    let (rescale, mean, std) = (s.rescale, s.mean, s.std);
    let (m_f, s_f): ([f32; 3], [f32; 3]) = if s.do_rescale && s.do_normalize {
        let r = 1.0 / rescale;
        ([0, 1, 2].map(|c| (f64::from(mean[c]) * r) as f32), [0, 1, 2].map(|c| (f64::from(std[c]) * r) as f32))
    } else if s.do_normalize {
        (mean, std)
    } else {
        ([0.0; 3], [1.0; 3])
    };
    let rescale_only = s.do_rescale && !s.do_normalize;
    let norm = |v: u8, c: usize| -> f32 {
        if rescale_only { (f64::from(v) * rescale) as f32 } else { (f32::from(v) - m_f[c]) / s_f[c] }
    };
    let dim = s.temporal * p * p * 3;
    let mut out = vec![0u16; gh * gw * dim];
    let bw = gw / m;
    for i in 0..gh * gw {
        let (blk, j) = (i / (m * m), i % (m * m));
        let (row, col) = ((blk / bw) * m + j / m, (blk % bw) * m + j % m);
        let dst = &mut out[i * dim..(i + 1) * dim];
        for c in 0..3 {
            for y in 0..p {
                for x in 0..p {
                    let v = to_bf16(norm(px[((row * p + y) * tw + col * p + x) * 3 + c], c));
                    for tt in 0..t {
                        dst[((c * t + tt) * p + y) * p + x] = v;
                    }
                }
            }
        }
    }
    Ok(Patches { pixels: out, grid_h: gh, grid_w: gw })
}

// weights

fn tensor<'a>(ckpt: &'a Checkpoint, name: &str, numel: usize) -> Result<(&'a TensorInfo, &'a [u8])> {
    let info = ckpt.info(name)?;
    if info.dtype != Dtype::Bf16 {
        bail!("{name}: the vision tower needs BF16 weights");
    }
    if info.numel() != numel {
        bail!("{name}: {} values, expected {numel}", info.numel());
    }
    let b = ckpt.bytes(name)?;
    if b.as_ptr() as usize % 2 != 0 {
        bail!("{name}: tensor data is not 2-byte aligned");
    }
    Ok((info, b))
}

/// Tensor names of one block and the merger: (name, values) in a fixed order.
struct Names {
    patch_w: String,
    patch_b: String,
    pos: String,
    blocks: Vec<[String; 12]>,
    merger: [String; 6],
}

impl Names {
    fn new(c: &VisionConfig) -> Self {
        let p = &c.prefix;
        let blocks = (0..c.depth)
            .map(|i| {
                [
                    "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                    "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight",
                    "mlp.linear_fc2.bias",
                ]
                .map(|n| format!("{p}blocks.{i}.{n}"))
            })
            .collect();
        let merger = ["norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"].map(|n| format!("{p}merger.{n}"));
        Self { patch_w: format!("{p}patch_embed.proj.weight"), patch_b: format!("{p}patch_embed.proj.bias"), pos: format!("{p}pos_embed.weight"), blocks, merger }
    }

    /// Element counts of the block tensors (same order as `blocks`) and of the merger's.
    fn block_numel(c: &VisionConfig) -> [usize; 12] {
        let (h, i) = (c.hidden, c.intermediate);
        [h, h, 3 * h * h, 3 * h, h * h, h, h, h, i * h, i, h * i, h]
    }

    fn merger_numel(c: &VisionConfig) -> [usize; 6] {
        let m = c.hidden * c.merge * c.merge;
        [c.hidden, c.hidden, m * m, m, c.out_hidden * m, c.out_hidden]
    }
}

// GPU encoder

struct GpuLayer {
    w: [DevBuf; 12],
}

/// The tower on one GPU: BF16 weights resident, activations allocated per image.
pub struct GpuEncoder {
    cfg: VisionConfig,
    device: i32,
    patch_w: DevBuf,
    patch_b: DevBuf,
    pos: DevBuf,
    layers: Vec<GpuLayer>,
    merger: [DevBuf; 6],
    gemm: *mut sys::blz_gemm,
    stream: Stream,
    /// activation bytes of the last image
    pub scratch_bytes: std::cell::Cell<usize>,
}

// SAFETY: used only behind the Vision mutex; the raw cuBLAS handle is not tied to a thread.
unsafe impl Send for GpuEncoder {}

impl GpuEncoder {
    pub fn load(ckpt: &Checkpoint, cfg: &VisionConfig, device: i32) -> Result<Self> {
        gpu::set_device(device)?;
        let names = Names::new(cfg);
        let up = |name: &str, numel: usize| -> Result<DevBuf> { DevBuf::from_slice::<u8>(tensor(ckpt, name, numel)?.1) };
        let (h, k) = (cfg.hidden, cfg.patch_dim());
        let patch_w = up(&names.patch_w, h * k)?;
        let patch_b = up(&names.patch_b, h)?;
        let pos = up(&names.pos, cfg.num_pos * h)?;
        let numel = Names::block_numel(cfg);
        let mut layers = Vec::with_capacity(cfg.depth);
        for b in &names.blocks {
            let w = std::array::from_fn::<_, 12, _>(|i| up(&b[i], numel[i]));
            let mut it = Vec::with_capacity(12);
            for x in w {
                it.push(x?);
            }
            layers.push(GpuLayer { w: it.try_into().map_err(|_| anyhow!("block tensors"))? });
        }
        let mn = Names::merger_numel(cfg);
        let mut m = Vec::with_capacity(6);
        for i in 0..6 {
            m.push(up(&names.merger[i], mn[i])?);
        }
        let mut gemm = std::ptr::null_mut();
        if unsafe { sys::blz_gemm_create(&mut gemm) } != 0 {
            bail!("cuBLAS handle for the vision tower");
        }
        gpu::synchronize()?;
        Ok(Self {
            cfg: cfg.clone(),
            device,
            patch_w,
            patch_b,
            pos,
            layers,
            merger: m.try_into().map_err(|_| anyhow!("merger tensors"))?,
            gemm,
            stream: Stream::new()?,
            scratch_bytes: std::cell::Cell::new(0),
        })
    }

    /// Activation bytes of an image of `n` patches.
    pub fn scratch_for(cfg: &VisionConfig, n: usize) -> usize {
        n * (6 * cfg.hidden + (3 * cfg.hidden).max(cfg.intermediate)) * 2 + n / 4 * cfg.out_hidden * 2
    }

    pub fn encode(&self, px: &[u16], gh: usize, gw: usize) -> Result<Vec<u16>> {
        self.encode_stages(px, gh, gw, None)
    }

    /// `stages`: receives the activations after the position embedding, the first block and the last block (bf16, debugging).
    pub fn encode_stages(&self, px: &[u16], gh: usize, gw: usize, mut stages: Option<&mut Vec<Vec<u16>>>) -> Result<Vec<u16>> {
        let c = &self.cfg;
        gpu::set_device(self.device)?;
        let (n, h, i) = (gh * gw, c.hidden, c.intermediate);
        if px.len() != n * c.patch_dim() {
            bail!("{} pixel values for {n} patches", px.len());
        }
        let need = Self::scratch_for(c, n);
        let free = gpu::mem_info()?.0;
        if free < need + (64 << 20) {
            bail!("not enough free VRAM for an image of {n} patches: {:.0} MiB needed, {:.0} MiB free", need as f64 / 1048576.0, free as f64 / 1048576.0);
        }
        self.scratch_bytes.set(need);
        let s = self.stream.0;
        let b = |elems: usize| DevBuf::new(elems * 2);
        let (x, xn, q, k, v, o) = (b(n * h)?, b(n * h)?, b(n * h)?, b(n * h)?, b(n * h)?, b(n * h)?);
        let big = b(n * (3 * h).max(i))?;
        let nm = n / (c.merge * c.merge);
        let mh = h * c.merge * c.merge;
        let out = b(nm * c.out_hidden)?;
        big.upload_async(px, &self.stream)?;
        let g = |a: &DevBuf, lda: usize, w: &DevBuf, dst: &DevBuf, ldo: usize, m: usize, nn: usize, kk: usize| -> Result<()> {
            check(unsafe { sys::blz_gemm_bf16(self.gemm, a.ptr(), lda as i64, w.ptr(), dst.ptr(), ldo as i64, m as i32, nn as i32, kk as i32, s) }, "vision gemm")
        };
        g(&big, c.patch_dim(), &self.patch_w, &x, h, n, h, c.patch_dim())?;
        check(
            unsafe { sys::blz_vis_embed_pos(x.ptr(), self.patch_b.ptr(), self.pos.ptr(), (c.num_pos as f64).sqrt() as i32, gh as i32, gw as i32, c.merge as i32, h as i32, s) },
            "vision pos embed",
        )?;
        let snap = |x: &DevBuf, stages: &mut Option<&mut Vec<Vec<u16>>>| -> Result<()> {
            if let Some(st) = stages.as_deref_mut() {
                self.stream.sync()?;
                st.push(x.download::<u16>(n * h)?);
            }
            Ok(())
        };
        snap(&x, &mut stages)?;
        let hd = h / c.heads;
        for (li, l) in self.layers.iter().enumerate() {
            let w = &l.w;
            check(unsafe { sys::blz_vis_layernorm(x.ptr(), w[0].ptr(), w[1].ptr(), 1e-6, n as i32, h as i32, xn.ptr(), s) }, "vision ln1")?;
            g(&xn, h, &w[2], &big, 3 * h, n, 3 * h, h)?;
            check(
                unsafe { sys::blz_vis_qkv_rope(big.ptr(), w[3].ptr(), gh as i32, gw as i32, c.merge as i32, c.heads as i32, hd as i32, c.theta, q.ptr(), k.ptr(), v.ptr(), s) },
                "vision qkv",
            )?;
            check(unsafe { sys::blz_vis_attention(q.ptr(), k.ptr(), v.ptr(), n as i32, c.heads as i32, hd as i32, o.ptr(), s) }, "vision attention")?;
            g(&o, h, &w[4], &xn, h, n, h, h)?;
            check(unsafe { sys::blz_vis_bias_residual(xn.ptr(), w[5].ptr(), x.ptr(), n as i32, h as i32, x.ptr(), s) }, "vision proj residual")?;
            check(unsafe { sys::blz_vis_layernorm(x.ptr(), w[6].ptr(), w[7].ptr(), 1e-6, n as i32, h as i32, xn.ptr(), s) }, "vision ln2")?;
            g(&xn, h, &w[8], &big, i, n, i, h)?;
            check(unsafe { sys::blz_vis_bias_act(big.ptr(), w[9].ptr(), n as i32, i as i32, if c.act_tanh { 1 } else { 2 }, s) }, "vision gelu")?;
            g(&big, i, &w[10], &xn, h, n, h, i)?;
            check(unsafe { sys::blz_vis_bias_residual(xn.ptr(), w[11].ptr(), x.ptr(), n as i32, h as i32, x.ptr(), s) }, "vision mlp residual")?;
            if li == 0 || li + 1 == self.layers.len() {
                snap(&x, &mut stages)?;
            }
        }
        let m = &self.merger;
        check(unsafe { sys::blz_vis_layernorm(x.ptr(), m[0].ptr(), m[1].ptr(), 1e-6, n as i32, h as i32, xn.ptr(), s) }, "vision merger norm")?;
        g(&xn, mh, &m[2], &big, mh, nm, mh, mh)?;
        check(unsafe { sys::blz_vis_bias_act(big.ptr(), m[3].ptr(), nm as i32, mh as i32, 2, s) }, "vision merger gelu")?;
        g(&big, mh, &m[4], &out, c.out_hidden, nm, c.out_hidden, mh)?;
        check(unsafe { sys::blz_vis_bias_act(out.ptr(), m[5].ptr(), nm as i32, c.out_hidden as i32, 0, s) }, "vision merger bias")?;
        self.stream.sync()?;
        out.download::<u16>(nm * c.out_hidden)
    }
}

impl Drop for GpuEncoder {
    fn drop(&mut self) {
        unsafe { sys::blz_gemm_destroy(self.gemm) };
    }
}

// CPU encoder

/// The tower on the CPU: reads the checkpoint's BF16 tensors in place.
pub struct CpuEncoder {
    cfg: VisionConfig,
    threads: usize,
    // the weight table points into `_ckpt`'s maps and `_layers`' heap buffer
    w: sys::blz_vis_weights,
    _layers: Vec<sys::blz_vis_layer>,
    _ckpt: Checkpoint,
}

// SAFETY: read-only pointers into a memory map owned by the struct.
unsafe impl Send for CpuEncoder {}

impl CpuEncoder {
    pub fn load(dir: &Path, cfg: &VisionConfig, threads: usize) -> Result<Self> {
        let ckpt = Checkpoint::open(dir)?;
        let names = Names::new(cfg);
        let p = |name: &str, numel: usize| -> Result<*const u16> { Ok(tensor(&ckpt, name, numel)?.1.as_ptr().cast()) };
        let (h, k) = (cfg.hidden, cfg.patch_dim());
        let numel = Names::block_numel(cfg);
        let mut layers = Vec::with_capacity(cfg.depth);
        for b in &names.blocks {
            let t: Vec<*const u16> = (0..12).map(|i| p(&b[i], numel[i])).collect::<Result<_>>()?;
            layers.push(sys::blz_vis_layer {
                n1_w: t[0], n1_b: t[1], qkv_w: t[2], qkv_b: t[3], proj_w: t[4], proj_b: t[5], n2_w: t[6], n2_b: t[7],
                fc1_w: t[8], fc1_b: t[9], fc2_w: t[10], fc2_b: t[11],
            });
        }
        let mn = Names::merger_numel(cfg);
        let m: Vec<*const u16> = (0..6).map(|i| p(&names.merger[i], mn[i])).collect::<Result<_>>()?;
        let w = sys::blz_vis_weights {
            cfg: sys::blz_vis_cfg {
                depth: cfg.depth as i32,
                hidden: h as i32,
                heads: cfg.heads as i32,
                intermediate: cfg.intermediate as i32,
                patch: cfg.patch as i32,
                temporal: cfg.temporal as i32,
                merge: cfg.merge as i32,
                channels: cfg.channels as i32,
                side: (cfg.num_pos as f64).sqrt() as i32,
                out_hidden: cfg.out_hidden as i32,
                eps: 1e-6,
                theta: cfg.theta,
                act_tanh: cfg.act_tanh as i32,
            },
            patch_w: p(&names.patch_w, h * k)?,
            patch_b: p(&names.patch_b, h)?,
            pos_table: p(&names.pos, cfg.num_pos * h)?,
            layers: layers.as_ptr(),
            mrg_norm_w: m[0], mrg_norm_b: m[1], mrg_fc1_w: m[2], mrg_fc1_b: m[3], mrg_fc2_w: m[4], mrg_fc2_b: m[5],
        };
        Ok(Self { cfg: cfg.clone(), threads: threads.max(1), w, _layers: layers, _ckpt: ckpt })
    }

    pub fn encode(&self, px: &[u16], gh: usize, gw: usize) -> Result<Vec<u16>> {
        self.encode_stages(px, gh, gw, None)
    }

    /// `stages`: receives the activations after the position embedding, the first block and the last block (fp32).
    pub fn encode_stages(&self, px: &[u16], gh: usize, gw: usize, stages: Option<&mut Vec<f32>>) -> Result<Vec<u16>> {
        let c = &self.cfg;
        if px.len() != gh * gw * c.patch_dim() {
            bail!("{} pixel values for {} patches", px.len(), gh * gw);
        }
        let mut out = vec![0u16; gh * gw / (c.merge * c.merge) * c.out_hidden];
        let dbg = match stages {
            Some(v) => {
                v.resize(3 * gh * gw * c.hidden, 0.0);
                v.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };
        let rc = unsafe { sys::blz_vis_cpu_encode(&self.w, px.as_ptr(), gh as i32, gw as i32, self.threads as i32, out.as_mut_ptr(), dbg) };
        if rc != 0 {
            bail!("CPU vision encoder: unsupported shape (code {rc})");
        }
        Ok(out)
    }
}

// the handle

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionMode {
    Off,
    Gpu,
    Cpu,
    /// the GPU when the weights and a margin fit in its free memory after the model is loaded, else the CPU
    Auto,
}

impl std::str::FromStr for VisionMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("'{s}': expected off, gpu, cpu or auto")),
        }
    }
}

enum Encoder {
    Gpu(GpuEncoder),
    Cpu(CpuEncoder),
}

/// Image input of a served model: the processor settings of its checkpoint and the tower, serialized (one image at a
/// time through the encoder).
pub struct Vision {
    pub cfg: VisionConfig,
    pub settings: ImageSettings,
    enc: Mutex<Encoder>,
    pub on_gpu: bool,
}

/// Free VRAM kept after the tower's weights are loaded by `VisionMode::Auto`, for the activations of an image.
const AUTO_HEADROOM: usize = 1536 << 20;

impl Vision {
    /// The tower of the checkpoint in `dir`; None when `mode` is Off or the checkpoint has none.
    pub fn load(dir: &Path, mode: VisionMode, threads: usize, device: i32) -> Result<Option<Self>> {
        if mode == VisionMode::Off {
            return Ok(None);
        }
        let Some(cfg) = VisionConfig::load(dir)? else { return Ok(None) };
        let settings = ImageSettings::load(dir, &cfg)?;
        let gpu_fits = || -> bool {
            gpu::set_device(device).is_ok() && gpu::mem_info().map(|(free, _)| free > cfg.weight_bytes() + AUTO_HEADROOM).unwrap_or(false)
        };
        let on_gpu = match mode {
            VisionMode::Gpu => true,
            VisionMode::Cpu => false,
            _ => gpu_fits(),
        };
        let enc = if on_gpu {
            let ckpt = Checkpoint::open(dir)?;
            let t0 = Instant::now();
            let free0 = gpu::mem_info()?.0;
            let e = GpuEncoder::load(&ckpt, &cfg, device)?;
            eprintln!(
                "blaze: vision tower on GPU {device}: {:.0} MiB of weights in {:.1} s",
                free0.saturating_sub(gpu::mem_info()?.0) as f64 / 1048576.0,
                t0.elapsed().as_secs_f64()
            );
            Encoder::Gpu(e)
        } else {
            eprintln!("blaze: vision tower on the CPU ({threads} threads){}", if mode == VisionMode::Auto { ": not enough free VRAM" } else { "" });
            Encoder::Cpu(CpuEncoder::load(dir, &cfg, threads)?)
        };
        Ok(Some(Self { cfg, settings, enc: Mutex::new(enc), on_gpu }))
    }

    /// The embeddings of one image: [tokens, out_hidden] bf16.
    pub fn encode(&self, p: &Patches) -> Result<Vec<u16>> {
        let enc = self.enc.lock().map_err(|_| anyhow!("vision encoder poisoned"))?;
        match &*enc {
            Encoder::Gpu(e) => e.encode(&p.pixels, p.grid_h, p.grid_w),
            Encoder::Cpu(e) => e.encode(&p.pixels, p.grid_h, p.grid_w),
        }
    }

    /// Activation VRAM the GPU path needs for an image of `n` patches (0 on the CPU).
    pub fn scratch_bytes(&self, n: usize) -> usize {
        if self.on_gpu { GpuEncoder::scratch_for(&self.cfg, n) } else { 0 }
    }
}

// check against a reference

fn f32_tensor(c: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    let b = c.bytes(name)?;
    Ok(b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect())
}

/// (mean cosine, min cosine, max abs difference, relative L2 error) of `a` (bf16 rows of `width`) against `b`.
fn compare(a: &[u16], b: &[f32], width: usize) -> (f64, f64, f64, f64) {
    let (mut cs, mut mn, mut mx, mut num, mut den) = (0.0f64, 1.0f64, 0.0f64, 0.0f64, 0.0f64);
    let rows = b.len() / width;
    for r in 0..rows {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..width {
            let (x, y) = (f64::from(f32::from_bits(u32::from(a[r * width + i]) << 16)), f64::from(b[r * width + i]));
            dot += x * y;
            na += x * x;
            nb += y * y;
            mx = mx.max((x - y).abs());
            num += (x - y) * (x - y);
            den += y * y;
        }
        let c = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        cs += c;
        mn = mn.min(c);
    }
    (cs / rows as f64, mn, mx, (num / den.max(1e-30)).sqrt())
}

/// `blaze vision-check`: the encoder (GPU and / or CPU) on the reference's pixel values against its embeddings (the HF
/// tower in bf16 and fp32, scripts/vision_ref.py), with timings; with `image`, also this crate's own preprocessing against the
/// reference's pixel values and the whole path from the file.
pub fn check_reference(model: &Path, refdir: &Path, gpu: bool, cpu: bool, threads: usize, image: Option<&Path>) -> Result<()> {
    let cfg = VisionConfig::load(model)?.ok_or_else(|| anyhow!("{} has no vision tower", model.display()))?;
    let settings = ImageSettings::load(model, &cfg)?;
    let r = Checkpoint::open(refdir)?;
    let pv = f32_tensor(&r, "pixel_values")?;
    let grid = r.bytes("grid_thw")?;
    let (gh, gw) = (i64::from_le_bytes(grid[8..16].try_into()?) as usize, i64::from_le_bytes(grid[16..24].try_into()?) as usize);
    let pixels: Vec<u16> = pv.iter().map(|&f| to_bf16(f)).collect();
    let (hf_bf16, hf_f32) = (f32_tensor(&r, "embeds")?, r.get("embeds_f32").map(|_| f32_tensor(&r, "embeds_f32")).transpose()?);
    let w = cfg.out_hidden;
    println!("grid {gh} x {gw} = {} patches, {} tokens of {w}", gh * gw, gh * gw / 4);
    if let Some(f) = &hf_f32 {
        let a: Vec<u16> = hf_bf16.iter().map(|&x| to_bf16(x)).collect();
        let (c, mn, mx, rl) = compare(&a, f, w);
        println!("HF bf16 vs HF fp32 (the noise floor of a bf16 tower): cos mean {c:.5} min {mn:.4}, max abs {mx:.4}, rel L2 {rl:.4}");
    }
    let report = |label: &str, out: &[u16], secs: f64| {
        let (c, mn, mx, rl) = compare(out, &hf_bf16, w);
        print!("{label}: vs HF bf16: cos mean {c:.5} min {mn:.4}, max abs {mx:.4}, rel L2 {rl:.4}");
        if let Some(f) = &hf_f32 {
            let (c, mn, mx, rl) = compare(out, f, w);
            print!(" | vs HF fp32: cos mean {c:.5} min {mn:.4}, max abs {mx:.4}, rel L2 {rl:.4}");
        }
        println!(" | {:.0} ms", secs * 1e3);
    };
    let run = |label: &str, enc: &dyn Fn(&[u16]) -> Result<Vec<u16>>| -> Result<()> {
        let _ = enc(&pixels)?; // warm-up (kernel load, thread pool, page-ins)
        let t0 = Instant::now();
        let out = enc(&pixels)?;
        report(label, &out, t0.elapsed().as_secs_f64());
        Ok(())
    };
    // intermediate activations of the HF bf16 tower: after the position embedding, the first block, the last block
    let stage_ref: Vec<(&str, Vec<f32>)> = ["pos_embed", "block0", "last_hidden"].iter().filter_map(|n| f32_tensor(&r, n).ok().map(|v| (*n, v))).collect();
    let hh = cfg.hidden;
    let report_stages = |label: &str, got: Vec<Vec<f32>>| {
        for ((name, want), g) in stage_ref.iter().zip(got) {
            let a: Vec<u16> = g.iter().map(|&x| to_bf16(x)).collect();
            let (c, mn, mx, rl) = compare(&a, want, hh);
            println!("  {label} stage {name}: cos mean {c:.5} min {mn:.4}, max abs {mx:.4}, rel L2 {rl:.4}");
        }
    };
    if gpu {
        let ckpt = Checkpoint::open(model)?;
        let free0 = gpu::mem_info()?.0;
        let e = GpuEncoder::load(&ckpt, &cfg, 0)?;
        let loaded = free0.saturating_sub(gpu::mem_info()?.0);
        run("GPU", &|p| e.encode(p, gh, gw))?;
        let mut st = Vec::new();
        e.encode_stages(&pixels, gh, gw, Some(&mut st))?;
        report_stages("GPU", st.iter().map(|v| v.iter().map(|&b| f32::from_bits(u32::from(b) << 16)).collect()).collect());
        println!(
            "GPU VRAM: weights {:.0} MiB, activations of this image {:.0} MiB (the cuBLAS workspace and context come on top)",
            loaded as f64 / 1048576.0,
            e.scratch_bytes.get() as f64 / 1048576.0
        );
        if let Some(path) = image {
            check_image(path, &settings, &pv, gh, gw, &|p: &Patches| e.encode(&p.pixels, p.grid_h, p.grid_w), &hf_bf16, w)?;
        }
    }
    if cpu {
        let e = CpuEncoder::load(model, &cfg, threads)?;
        run(&format!("CPU ({threads} threads)"), &|p| e.encode(p, gh, gw))?;
        let mut st = Vec::new();
        e.encode_stages(&pixels, gh, gw, Some(&mut st))?;
        let n = gh * gw * hh;
        report_stages("CPU", st.chunks(n).map(<[f32]>::to_vec).collect());
        if let (Some(path), false) = (image, gpu) {
            check_image(path, &settings, &pv, gh, gw, &|p: &Patches| e.encode(&p.pixels, p.grid_h, p.grid_w), &hf_bf16, w)?;
        }
    }
    Ok(())
}

fn check_image(path: &Path, s: &ImageSettings, ref_pixels: &[f32], gh: usize, gw: usize, enc: &dyn Fn(&Patches) -> Result<Vec<u16>>, hf: &[f32], w: usize) -> Result<()> {
    let img = decode_image(&std::fs::read(path)?).map_err(|e| anyhow!(e))?;
    let t0 = Instant::now();
    let p = preprocess(&img, s).map_err(|e| anyhow!(e))?;
    println!("preprocess of {}x{}: {:.0} ms, grid {} x {} (reference {gh} x {gw})", img.w, img.h, t0.elapsed().as_secs_f64() * 1e3, p.grid_h, p.grid_w);
    if (p.grid_h, p.grid_w) != (gh, gw) {
        bail!("grid differs from the reference");
    }
    let (mut same, mut maxd) = (0usize, 0f32);
    for (a, &b) in p.pixels.iter().zip(ref_pixels) {
        let x = f32::from_bits(u32::from(*a) << 16);
        let d = (x - f32::from_bits(u32::from(to_bf16(b)) << 16)).abs();
        same += (d == 0.0) as usize;
        maxd = maxd.max(d);
    }
    println!("pixel values vs the reference processor: {:.4}% identical, max abs diff {maxd:.4} (one 8-bit level is 0.0078)", 100.0 * same as f64 / p.pixels.len() as f64);
    let out = enc(&p)?;
    let (c, mn, mx, rl) = compare(&out, hf, w);
    println!("embeddings from the file through this crate's preprocessing vs HF bf16: cos mean {c:.5} min {mn:.4}, max abs {mx:.4}, rel L2 {rl:.4}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ImageSettings {
        ImageSettings {
            min_pixels: 65536,
            max_pixels: 16777216,
            patch: 16,
            merge: 2,
            temporal: 2,
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale: 1.0 / 255.0,
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
        }
    }

    #[test]
    fn smart_resize_matches_the_reference_sizes() {
        let s = settings();
        // sizes printed by the HF processor for the test images (grids 14x20, 20x28, 30x40, 48x64 patches)
        assert_eq!(s.target_size(200, 150), Ok((224, 320)));
        assert_eq!(s.target_size(448, 336), Ok((320, 448)));
        assert_eq!(s.target_size(640, 480), Ok((480, 640)));
        assert_eq!(s.target_size(1024, 768), Ok((768, 1024)));
        // above max_pixels: scaled down; below min_pixels: scaled up
        let big = s.target_size(8000, 6000).unwrap();
        assert!(big.0 * big.1 <= 16777216 && big.0 % 32 == 0 && big.1 % 32 == 0);
        assert!(s.target_size(1, 1).is_err() || s.target_size(1, 1).unwrap().0 % 32 == 0);
        assert!(s.target_size(5000, 20).is_err());
    }

    #[test]
    fn kwargs_aliases_and_precedence() {
        let a = Overrides::parse(&serde_json::json!({"size": {"shortest_edge": 100, "longest_edge": 200}})).unwrap();
        assert_eq!((a.min_pixels, a.max_pixels), (Some(100), Some(200)));
        // min_pixels / max_pixels win over size within a level
        let b = Overrides::parse(&serde_json::json!({"min_pixels": 7, "size": {"shortest_edge": 100}})).unwrap();
        assert_eq!(b.min_pixels, Some(7));
        // images_kwargs wins over flat keys; null overrides nothing
        let c = Overrides::parse(&serde_json::json!({"max_pixels": 5, "min_pixels": null, "images_kwargs": {"max_pixels": 9}})).unwrap();
        assert_eq!((c.min_pixels, c.max_pixels), (None, Some(9)));
        assert!(Overrides::parse(&serde_json::json!({"max_pixels": -3})).is_err());
        let merged = a.over(Overrides { max_pixels: Some(5), ..Default::default() });
        assert_eq!((merged.min_pixels, merged.max_pixels), (Some(100), Some(5)));
    }

    #[test]
    fn patch_layout() {
        // 64 x 64 gradient: grid 4 x 4, merge order; the first window's patches are (0,0) (0,1) (1,0) (1,1)
        let img = RgbImage { w: 64, h: 64, data: (0..64 * 64 * 3).map(|i| (i / 3 % 64 + i / 3 / 64) as u8).collect() };
        let mut s = settings();
        s.do_resize = false;
        let p = preprocess(&img, &s).unwrap();
        assert_eq!((p.grid_h, p.grid_w), (4, 4));
        let dim = 2 * 16 * 16 * 3;
        // patch 1 is grid column 1: its first pixel is x = 16
        let first = |i: usize| f32::from_bits(u32::from(p.pixels[i * dim]) << 16);
        let expect = |v: f32| (v - 127.5) / 127.5;
        assert!((first(1) - expect(16.0)).abs() < 0.01);
        // patch 2 is grid row 1: y = 16
        assert!((first(2) - expect(16.0)).abs() < 0.01);
        // patch 4 starts the second window (grid column 2)
        assert!((first(4) - expect(32.0)).abs() < 0.01);
        // the temporal pair repeats
        assert_eq!(p.pixels[0], p.pixels[16 * 16]);
    }

    #[test]
    fn bicubic_of_a_constant_is_constant() {
        let img = RgbImage { w: 100, h: 80, data: vec![200; 100 * 80 * 3] };
        let out = resize_bicubic(&img, 64, 64);
        assert!(out.iter().all(|&v| v == 200));
    }
}
