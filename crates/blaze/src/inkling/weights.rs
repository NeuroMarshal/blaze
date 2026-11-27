use crate::config::{MlpConfig, ModelConfig};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Packing {
    Nvfp4,
    Mxfp8,
}

pub(crate) struct PackedMatrix {
    pub packing: Packing,
    pub rows: usize,
    pub cols: usize,
    weight: String,
    scale: String,
    alpha: Vec<f32>,
}

pub(crate) struct ExpertSlice<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub alpha: Option<f32>,
}

impl PackedMatrix {
    fn load(ckpt: &Checkpoint, weight: String, experts: usize, rows: usize, cols: usize) -> Result<Self> {
        let scale = format!("{weight}_scale");
        let w = ckpt.info(&weight)?;
        let s = ckpt.info(&scale)?;
        let (packing, divisor, block) = match (w.dtype, s.dtype) {
            (Dtype::U8, Dtype::F8E4m3) => (Packing::Nvfp4, 2, 16),
            (Dtype::F8E4m3, Dtype::U8) => (Packing::Mxfp8, 1, 32),
            _ => anyhow::bail!(
                "{weight}: unsupported packed weights and scales ({:?}, {:?})",
                w.dtype,
                s.dtype
            ),
        };
        ensure!(
            cols % block == 0 && w.shape == [experts, rows, cols / divisor],
            "{weight}: unexpected shape {:?}",
            w.shape
        );
        ensure!(
            s.shape == [experts, rows, cols / block],
            "{scale}: unexpected shape {:?}",
            s.shape
        );
        let mut alpha = Vec::new();
        if packing == Packing::Nvfp4 {
            let denominator = read_float(ckpt, &format!("{weight}_outer_scale_denominator"), &[1])?[0];
            ensure!(
                denominator.is_finite() && denominator > 0.0,
                "{weight}: invalid NVFP4 scale denominator"
            );
            alpha = read_float(ckpt, &format!("{weight}_amax"), &[experts])?;
            for a in &mut alpha {
                ensure!(a.is_finite() && *a >= 0.0, "{weight}: invalid NVFP4 expert amax");
                *a /= denominator;
            }
        }
        Ok(Self {
            packing,
            rows,
            cols,
            weight,
            scale,
            alpha,
        })
    }

    pub fn expert<'a>(&self, ckpt: &'a Checkpoint, id: usize) -> Result<ExpertSlice<'a>> {
        let (divisor, block) = match self.packing {
            Packing::Nvfp4 => (2, 16),
            Packing::Mxfp8 => (1, 32),
        };
        let weight_stride = self.rows * (self.cols / divisor);
        let scale_stride = self.rows * (self.cols / block);
        let get = |name: &str, stride: usize| -> Result<&'a [u8]> {
            ckpt.bytes(name)?
                .get(id * stride..(id + 1) * stride)
                .with_context(|| format!("{name}: invalid expert {id}"))
        };
        Ok(ExpertSlice {
            codes: get(&self.weight, weight_stride)?,
            scales: get(&self.scale, scale_stride)?,
            alpha: self.alpha.get(id).copied(),
        })
    }

    pub fn gpu_bytes(&self) -> usize {
        let (divisor, block) = match self.packing {
            Packing::Nvfp4 => (2, 16),
            Packing::Mxfp8 => (1, 32),
        };
        self.rows * (self.cols / divisor + self.cols / block) + usize::from(self.packing == Packing::Nvfp4) * self.rows * 4
    }
}

pub(crate) struct Experts {
    pub gate_up: PackedMatrix,
    pub down: PackedMatrix,
}

impl Experts {
    pub fn load(ckpt: &Checkpoint, cfg: &ModelConfig, layer: usize) -> Result<Self> {
        let MlpConfig::Moe {
            experts, intermediate, ..
        } = cfg.mlp
        else {
            anyhow::bail!("Inkling layer {layer} requires routed experts");
        };
        let prefix = format!("model.llm.layers.{layer}.mlp.experts");
        let gate_rows = intermediate
            .checked_mul(2)
            .context("Inkling expert intermediate size overflows")?;
        let gate_up = PackedMatrix::load(ckpt, format!("{prefix}.w13_weight"), experts, gate_rows, cfg.hidden)?;
        let down = PackedMatrix::load(ckpt, format!("{prefix}.w2_weight"), experts, cfg.hidden, intermediate)?;
        Ok(Self { gate_up, down })
    }

    pub fn to_rg16(&self, ckpt: &Checkpoint, id: usize, record: &mut [u8]) -> Result<[f32; 3]> {
        let pack = |matrix: &PackedMatrix, output: &mut [u8]| {
            let src = matrix.expert(ckpt, id)?;
            crate::expert_rg16::convert_matrix(src.codes, src.scales, src.alpha, matrix.rows, matrix.cols, output)
        };
        let split = crate::expert_rg16::matrix_bytes(self.gate_up.rows, self.gate_up.cols);
        let (gate_up, down) = record.split_at_mut(split);
        let down_bytes = crate::expert_rg16::matrix_bytes(self.down.rows, self.down.cols);
        let gate_up = pack(&self.gate_up, gate_up)?;
        let down = pack(&self.down, &mut down[..down_bytes])?;
        Ok([gate_up, gate_up, down])
    }
}

pub(crate) fn read_float(ckpt: &Checkpoint, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = ckpt.info(name)?;
    ensure!(t.shape == shape, "{name}: expected {shape:?}, found {:?}", t.shape);
    let bytes = ckpt.bytes(name)?;
    Ok(match t.dtype {
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        _ => anyhow::bail!("{name}: expected BF16 or F32, found {:?}", t.dtype),
    })
}

pub(crate) fn packed_inventory(ckpt: &Checkpoint, cfg: &ModelConfig) -> Result<BTreeMap<Packing, usize>> {
    let mut inventory = BTreeMap::new();
    for layer in 0..cfg.layers.len() {
        let experts = Experts::load(ckpt, cfg, layer)?;
        for matrix in [experts.gate_up, experts.down] {
            *inventory.entry(matrix.packing).or_insert(0) += 1;
        }
    }
    Ok(inventory)
}
