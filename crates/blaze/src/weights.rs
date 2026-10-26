//! Memory-mapped safetensors shards indexed by tensor name.

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4m3,
    F8E5m2,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::F8E4m3,
            "F8_E5M2" => Self::F8E5m2,
            other => bail!("safetensors dtype {other} is not supported"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4m3 | Self::F8E5m2 => 1,
            Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub file: usize,
    /// Byte range in the file (header included in the offset).
    pub start: usize,
    pub end: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    pub fn bytes(&self) -> usize {
        self.end - self.start
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    maps: Vec<Mmap>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("listing {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no *.safetensors in {}", dir.display());
        }
        let mut maps = Vec::with_capacity(files.len());
        let mut tensors = BTreeMap::new();
        for (fi, path) in files.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the checkpoint is read-only for the lifetime of the process.
            let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
            if map.len() < 8 {
                bail!("{}: too short for a safetensors header", path.display());
            }
            let n = usize::try_from(u64::from_le_bytes(map[..8].try_into().unwrap()))?;
            let base = n.checked_add(8).filter(|&end| end <= map.len())
                .ok_or_else(|| anyhow!("{}: header extends beyond the file", path.display()))?;
            let header: Value = serde_json::from_slice(&map[8..base])
                .with_context(|| format!("{}: bad header", path.display()))?;
            for (name, v) in header.as_object().ok_or_else(|| anyhow!("{}: header is not an object", path.display()))? {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
                if offsets.len() != 2 { bail!("{name}: data_offsets must contain two integers"); }
                let offset = |v: &Value| -> Result<usize> {
                    v.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| anyhow!("{name}: invalid data offset"))
                };
                let (a, b) = (offset(&offsets[0])?, offset(&offsets[1])?);
                if a > b || b > map.len() - base { bail!("{name}: data range is outside the file"); }
                let shape = v["shape"].as_array().ok_or_else(|| anyhow!("{name}: missing shape"))?
                    .iter().map(|x| x.as_u64().and_then(|x| usize::try_from(x).ok())
                        .ok_or_else(|| anyhow!("{name}: invalid shape dimension"))).collect::<Result<Vec<_>>>()?;
                let info = TensorInfo {
                    file: fi,
                    start: base + a,
                    end: base + b,
                    dtype: Dtype::parse(v["dtype"].as_str().unwrap_or(""))?,
                    shape,
                };
                let bytes = info.shape.iter().try_fold(info.dtype.size(), |n, &dim| n.checked_mul(dim));
                if bytes != Some(info.bytes()) {
                    bail!("{name}: header range does not match its shape and dtype");
                }
                if tensors.insert(name.clone(), info).is_some() {
                    bail!("{name}: duplicate tensor across checkpoint files");
                }
            }
            maps.push(map);
        }
        Ok(Self { dir: dir.to_path_buf(), files, maps, tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.get(name).ok_or_else(|| anyhow!("checkpoint: missing tensor {name}"))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name)?;
        Ok(&self.maps[t.file][t.start..t.end])
    }

    /// The common prefix of the language model's tensors ("model.language_model." for multimodal checkpoints).
    pub fn text_prefix(&self) -> &'static str {
        if self.tensors.keys().any(|k| k.starts_with("model.language_model.")) {
            "model.language_model."
        } else {
            "model."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_checkpoint_ranges_return_errors_without_panicking() {
        let dir = std::env::temp_dir().join(format!("blaze-header-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("model.safetensors");
        for header in [
            serde_json::json!({"x":{"dtype":"U8","shape":[1],"data_offsets":[]}}),
            serde_json::json!({"x":{"dtype":"U8","shape":[1],"data_offsets":[4,1]}}),
            serde_json::json!({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,99]}}),
            serde_json::json!({"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}),
            serde_json::json!({"x":{"dtype":"F64","shape":[u64::MAX,2],"data_offsets":[0,1]}}),
        ] {
            let bytes = serde_json::to_vec(&header).unwrap();
            let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
            data.extend(bytes);
            data.push(42);
            std::fs::write(&path, data).unwrap();
            assert!(Checkpoint::open(&dir).is_err(), "accepted {header}");
        }
        for data in [vec![0; 3], u64::MAX.to_le_bytes().to_vec(), 100u64.to_le_bytes().to_vec()] {
            std::fs::write(&path, data).unwrap();
            assert!(Checkpoint::open(&dir).is_err());
        }
        let header = serde_json::json!({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}});
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut data = (bytes.len() as u64).to_le_bytes().to_vec();
        data.extend(bytes);
        data.push(42);
        std::fs::write(&path, data).unwrap();
        let c = Checkpoint::open(&dir).unwrap();
        assert_eq!(c.bytes("x").unwrap(), &[42]);
        drop(c);
        std::fs::copy(&path, dir.join("duplicate.safetensors")).unwrap();
        assert!(Checkpoint::open(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
