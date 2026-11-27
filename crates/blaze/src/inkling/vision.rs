use super::cuda::{Compute, Dense, to_bf16};
use super::weights::read_float;
use crate::gpu::{DevBuf, check};
use crate::vision::decode_rgb;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use blaze_sys as sys;
use serde_json::Value;

const PREFIX: &str = "model.visual";

struct Norm {
    weight: DevBuf,
    bias: DevBuf,
}

impl Norm {
    fn load(ckpt: &Checkpoint, name: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            weight: tensor(ckpt, &format!("{name}.weight"), &[dim])?,
            bias: tensor(ckpt, &format!("{name}.bias"), &[dim])?,
        })
    }

    fn run(&self, in_: &DevBuf, out: &DevBuf, rows: i32, dim: i32, eps: f32, s: sys::cudaStream_t) -> Result<()> {
        check(
            unsafe { sys::blz_vis_layernorm(in_.ptr(), self.weight.ptr(), self.bias.ptr(), eps, rows, dim, out.ptr(), s) },
            "Inkling vision layer norm",
        )
    }
}

struct Residual {
    linear: Dense,
    bias: DevBuf,
    scale: DevBuf,
}

impl Residual {
    fn load(ckpt: &Checkpoint, name: &str, scale: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            linear: Dense::load(ckpt, &format!("{name}.weight"), rows, cols)?,
            bias: tensor(ckpt, &format!("{name}.bias"), &[rows])?,
            scale: tensor(ckpt, scale, &[rows])?,
        })
    }

    fn run(&self, gpu: &Compute, x: &DevBuf, state: &DevBuf, work: &DevBuf, rows: usize) -> Result<()> {
        self.linear.run(gpu, x.ptr(), work.ptr(), rows)?;
        check(
            unsafe {
                sys::blz_inkling_vision_residual(
                    state.ptr(),
                    work.ptr(),
                    self.bias.ptr(),
                    self.scale.ptr(),
                    rows as i32,
                    self.linear.rows as i32,
                    gpu.stream.0,
                )
            },
            "Inkling vision residual",
        )
    }
}

struct Layer {
    attn_norm: Norm,
    qkv: Dense,
    qkv_bias: DevBuf,
    attn_output: Residual,
    mlp_norm: Norm,
    mlp_in: Dense,
    mlp_bias: DevBuf,
    mlp_output: Residual,
}

impl Layer {
    fn load(ckpt: &Checkpoint, id: usize, dim: usize, ff: usize) -> Result<Self> {
        let stem = format!("{PREFIX}.backbone.layers.{id}");
        let name = |n: &str| format!("{stem}.{n}");
        let qkv = ["q_proj", "k_proj", "v_proj"].map(|m| name(&format!("self_attn.{m}.weight")));
        let qkv = Dense::stack(ckpt, &qkv.each_ref().map(|s| (s.as_str(), dim)), dim)?;
        let mut bias = Vec::with_capacity(3 * dim);
        for p in ["q_proj", "k_proj", "v_proj"] {
            bias.extend(
                read_float(ckpt, &name(&format!("self_attn.{p}.bias")), &[dim])?
                    .into_iter()
                    .map(to_bf16),
            );
        }
        Ok(Self {
            attn_norm: Norm::load(ckpt, &name("ln_1"), dim)?,
            qkv,
            qkv_bias: DevBuf::from_slice(&bias)?,
            attn_output: Residual::load(ckpt, &name("self_attn.out_proj"), &name("ls_1.gamma"), dim, dim)?,
            mlp_norm: Norm::load(ckpt, &name("ln_2"), dim)?,
            mlp_in: Dense::load(ckpt, &name("mlp.c_fc.weight"), ff, dim)?,
            mlp_bias: tensor(ckpt, &name("mlp.c_fc.bias"), &[ff])?,
            mlp_output: Residual::load(ckpt, &name("mlp.c_proj"), &name("ls_2.gamma"), dim, ff)?,
        })
    }
}

pub(super) struct Vision {
    layers: Vec<Layer>,
    input: Dense,
    class: Vec<u16>,
    position: DevBuf,
    norm: Norm,
    proj_in: Dense,
    proj_out: Dense,
    dim: usize,
    heads: usize,
    patch: usize,
    merge: usize,
    side: usize,
    eps: f32,
    ff: usize,
    max_tokens: usize,
}

impl Vision {
    pub fn load(ckpt: &Checkpoint, root: &Value, text_hidden: usize, max_tokens: usize) -> Result<Self> {
        let cfg = &root["vision_config"];
        ensure!(
            cfg["model_type"].as_str() == Some("inkling2_vision"),
            "Inkling: unsupported vision tower"
        );
        let int = |name: &str| {
            cfg[name]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v != 0)
                .with_context(|| format!("Inkling vision: invalid {name}"))
        };
        let dim = int("hidden_size")?;
        let heads = int("num_attention_heads")?;
        ensure!(dim == heads * 64, "Inkling vision requires 64-dimensional attention heads");
        ensure!(
            (1..=8192).contains(&max_tokens),
            "Inkling max image tokens must be in 1..=8192"
        );
        let patch = int("native_patch_size")?;
        let merge = int("adapter_merge_size")?;
        let side = int("posemb_grid_size")?;
        let ff = (cfg["mlp_ratio"].as_f64().context("Inkling vision: missing MLP ratio")? * dim as f64) as usize;
        let proj = int("projection_hidden_dim")?;
        let eps = cfg["layer_norm_eps"]
            .as_f64()
            .context("Inkling vision: missing layer norm epsilon")? as f32;
        ensure!(
            eps.is_finite() && eps > 0.0 && !cfg["projector_bias"].as_bool().unwrap_or(false),
            "Inkling vision: unsupported epsilon or projector bias"
        );
        let emb = format!("{PREFIX}.backbone.embeddings");
        let class = read_float(ckpt, &format!("{emb}.class_embedding"), &[dim])?;
        let cls_pos = read_float(ckpt, &format!("{emb}.cls_position_embedding"), &[dim])?;
        let class = class.into_iter().zip(cls_pos).map(|(a, b)| to_bf16(a + b)).collect();
        let layers = (0..int("num_hidden_layers")?)
            .map(|i| Layer::load(ckpt, i, dim, ff))
            .collect::<Result<_>>()?;
        Ok(Self {
            layers,
            input: Dense::load(ckpt, &format!("{emb}.patch_embedding.weight"), dim, patch * patch * 3)?,
            position: tensor(ckpt, &format!("{emb}.position_embedding"), &[side * side, dim])?,
            norm: Norm::load(ckpt, &format!("{PREFIX}.backbone.ln_pre"), dim)?,
            proj_in: Dense::load(ckpt, &format!("{PREFIX}.proj_in.weight"), proj, dim * merge * merge)?,
            proj_out: Dense::load(ckpt, &format!("{PREFIX}.proj_out.weight"), text_hidden, proj)?,
            class,
            dim,
            heads,
            patch,
            merge,
            side,
            eps,
            ff,
            max_tokens,
        })
    }

    pub fn encode(&self, gpu: &Compute, img: &crate::vision::RgbImage) -> Result<Vec<Vec<u16>>> {
        let (pixel_values, height, width) = patchify(img, self.patch, self.merge, self.max_tokens)?;
        let patches = height * width;
        let image_tokens = patches / (self.merge * self.merge);
        let rows = patches + 1;
        let dim = self.dim;
        let count = rows * dim;
        let s = gpu.stream.0;
        let input = DevBuf::from_slice(&pixel_values)?;
        let x = DevBuf::new(count * 2)?;
        let norm = DevBuf::new(count * 2)?;
        let packed = DevBuf::new(count * 6)?;
        let [q, k, v, temp] = [
            DevBuf::new(count * 2)?,
            DevBuf::new(count * 2)?,
            DevBuf::new(count * 2)?,
            DevBuf::new(count * 2)?,
        ];
        let work = DevBuf::new((rows * self.ff).max(image_tokens * self.proj_in.rows) * 2)?;
        let zero_bias = DevBuf::zeroed(self.proj_in.rows * 2, &gpu.stream)?;
        let result = DevBuf::new(image_tokens * self.proj_out.rows * 2)?;
        x.upload(&self.class)?;
        self.input.run(gpu, input.ptr(), x.at(dim), patches)?;
        check(
            unsafe {
                sys::blz_inkling_vision_pos(
                    x.at(dim),
                    self.position.ptr(),
                    height as i32,
                    width as i32,
                    self.merge as i32,
                    self.side as i32,
                    dim as i32,
                    s,
                )
            },
            "Inkling vision positions",
        )?;
        self.norm.run(&x, &x, rows as i32, dim as i32, self.eps, s)?;
        for layer in &self.layers {
            layer.attn_norm.run(&x, &norm, rows as i32, dim as i32, self.eps, s)?;
            layer.qkv.run(gpu, norm.ptr(), packed.ptr(), rows)?;
            check(
                unsafe {
                    sys::blz_inkling_vision_qkv(
                        packed.ptr(),
                        layer.qkv_bias.ptr(),
                        q.ptr(),
                        k.ptr(),
                        v.ptr(),
                        width as i32,
                        self.merge as i32,
                        self.heads as i32,
                        rows as i32,
                        64,
                        s,
                    )
                },
                "Inkling vision axial RoPE",
            )?;
            check(
                unsafe { sys::blz_vis_attention(q.ptr(), k.ptr(), v.ptr(), rows as i32, self.heads as i32, 64, norm.ptr(), s) },
                "Inkling vision attention",
            )?;
            layer.attn_output.run(gpu, &norm, &x, &temp, rows)?;
            layer.mlp_norm.run(&x, &norm, rows as i32, dim as i32, self.eps, s)?;
            layer.mlp_in.run(gpu, norm.ptr(), work.ptr(), rows)?;
            check(
                unsafe { sys::blz_vis_bias_act(work.ptr(), layer.mlp_bias.ptr(), rows as i32, self.ff as i32, 2, s) },
                "Inkling vision GELU",
            )?;
            layer.mlp_output.run(gpu, &work, &x, &temp, rows)?;
        }
        self.proj_in.run(gpu, x.at(dim), work.ptr(), image_tokens)?;
        check(
            unsafe {
                sys::blz_vis_bias_act(
                    work.ptr(),
                    zero_bias.ptr(),
                    image_tokens as i32,
                    self.proj_in.rows as i32,
                    2,
                    s,
                )
            },
            "Inkling vision projector GELU",
        )?;
        self.proj_out.run(gpu, work.ptr(), result.ptr(), image_tokens)?;
        gpu.stream.sync()?;
        let values = result.download::<u16>(image_tokens * self.proj_out.rows)?;
        Ok(values.chunks_exact(self.proj_out.rows).map(<[u16]>::to_vec).collect())
    }
}

pub(crate) fn decode_image(bytes: &[u8]) -> Result<crate::vision::RgbImage> {
    let img = decode_rgb(bytes, 4096 * 4096, |r, c| if (r / 8 + c / 8) % 2 == 0 { 204 } else { 153 })
        .map_err(anyhow::Error::msg)?;
    ensure!(img.w <= 4096 && img.h <= 4096, "Inkling images must fit within 4096x4096");
    Ok(img)
}

fn tensor(ckpt: &Checkpoint, name: &str, shape: &[usize]) -> Result<DevBuf> {
    let tensor = ckpt.info(name)?;
    ensure!(
        tensor.dtype == Dtype::Bf16 && tensor.shape == shape,
        "{name}: expected BF16 {shape:?}"
    );
    DevBuf::from_slice(ckpt.bytes(name)?)
}

fn patchify(img: &crate::vision::RgbImage, patch: usize, merge: usize, limit: usize) -> Result<(Vec<u16>, usize, usize)> {
    let side = patch * merge;
    let (h, w) = (img.h.div_ceil(side), img.w.div_ceil(side));
    ensure!(h * w <= limit, "Inkling image needs {} tokens, limit is {limit}", h * w);
    let pad = to_bf16(-1.0 - 2.0 / 255.0);
    let table: [u16; 256] = std::array::from_fn(|i| to_bf16(i as f32 * (2.0 / 255.0) - 1.0));
    let mut data = Vec::with_capacity(h * w * side * side * 3);
    for by in 0..h {
        for bx in 0..w {
            for dy in 0..merge {
                for dx in 0..merge {
                    for y in (by * merge + dy) * patch..(by * merge + dy + 1) * patch {
                        let x = (bx * merge + dx) * patch;
                        let valid = if y < img.h { img.w.saturating_sub(x).min(patch) } else { 0 };
                        if valid > 0 {
                            let start = (y * img.w + x) * 3;
                            data.extend(img.data[start..start + valid * 3].iter().map(|&p| table[p as usize]));
                        }
                        data.extend(std::iter::repeat_n(pad, (patch - valid) * 3));
                    }
                }
            }
        }
    }
    Ok((data, h * merge, w * merge))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_and_merges_partial_patches_in_raster_order() {
        let img = crate::vision::RgbImage {
            w: 3,
            h: 2,
            data: vec![255; 18],
        };
        let (p, h, w) = patchify(&img, 1, 2, 2).unwrap();
        assert_eq!((h, w), (2, 4));
        let white = to_bf16(1.0);
        let pad = to_bf16(-1.0 - 2.0 / 255.0);
        assert_eq!(
            p.chunks_exact(3).map(|r| r[0]).collect::<Vec<_>>(),
            [white, white, white, white, white, pad, white, pad]
        );
    }
}
