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
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: Value = serde_json::from_slice(&map[8..8 + n])
                .with_context(|| format!("{}: bad header", path.display()))?;
            let base = 8 + n;
            for (name, v) in header.as_object().ok_or_else(|| anyhow!("{}: header is not an object", path.display()))? {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
                let (a, b) = (offsets[0].as_u64().unwrap_or(0) as usize, offsets[1].as_u64().unwrap_or(0) as usize);
                let info = TensorInfo {
                    file: fi,
                    start: base + a,
                    end: base + b,
                    dtype: Dtype::parse(v["dtype"].as_str().unwrap_or(""))?,
                    shape: v["shape"].as_array().map(|s| s.iter().filter_map(Value::as_u64).map(|x| x as usize).collect()).unwrap_or_default(),
                };
                if info.end > map.len() || info.numel() * info.dtype.size() != info.bytes() {
                    bail!("{name}: header range does not match its shape and dtype");
                }
                tensors.insert(name.clone(), info);
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
