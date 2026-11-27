use crate::weights::Checkpoint;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const UNIT: usize = 144;
const HEADER_BYTES: usize = 4096;

pub(crate) fn matrix_bytes(rows: usize, columns: usize) -> usize {
    (rows / 16) * (columns / 16) * UNIT
}

pub(crate) fn record_bytes(hidden: usize, intermediate: usize) -> usize {
    (matrix_bytes(intermediate * 2, hidden) + matrix_bytes(hidden, intermediate)).next_multiple_of(256)
}

pub(crate) fn convert_matrix(
    codes: &[u8], scales: &[u8], multiplier: Option<f32>, rows: usize, columns: usize, output: &mut [u8],
) -> Result<f32> {
    let blocks = columns / 16;
    ensure!(rows.is_multiple_of(16) && columns.is_multiple_of(16) && blocks > 0
        && output.len() == matrix_bytes(rows, columns), "invalid rg16 matrix geometry");
    let (outer, max_exp) = match multiplier {
        Some(value) => {
            ensure!(codes.len() == rows * columns / 2 && scales.len() == rows * blocks, "invalid NVFP4 matrix");
            (value, 0)
        }
        None => {
            ensure!(blocks.is_multiple_of(2) && codes.len() == rows * columns
                && scales.len() == rows * blocks / 2, "invalid MXFP8 matrix");
            let exp = i32::from(max_block_exponent(scales)?).max(3);
            let outer = f32::from_bits(((exp - 2) as u32) << 23);
            (outer, exp)
        }
    };
    let table = multiplier.is_none().then(quantization_table);
    for (i, unit) in output.chunks_exact_mut(UNIT).enumerate() {
        let row_group = (i / blocks) * 16;
        let block = i % blocks;
        for row in 0..8 {
            let lo = row_group + row;
            let hi = lo + 8;
            if let Some(table) = table {
                let (lo_codes, lo_scale) = quantize_block(table, codes, scales, lo, block, columns, max_exp)?;
                let (hi_codes, hi_scale) = quantize_block(table, codes, scales, hi, block, columns, max_exp)?;
                unit[128 + row] = lo_scale;
                unit[136 + row] = hi_scale;
                for (column, (&a, &b)) in lo_codes.iter().zip(&hi_codes).enumerate() {
                    unit[(column / 4) * 32 + row * 4 + column % 4] = a | (b << 4);
                }
            } else {
                unit[128 + row] = scales[lo * blocks + block];
                unit[136 + row] = scales[hi * blocks + block];
                let lo_start = lo * (columns / 2) + block * 8;
                let hi_start = hi * (columns / 2) + block * 8;
                for column in 0..8 {
                    let a = codes[lo_start + column];
                    let b = codes[hi_start + column];
                    let off = (column / 2) * 32 + row * 4 + (column % 2) * 2;
                    unit[off] = (a & 15) | (b << 4);
                    unit[off + 1] = (a >> 4) | (b & 240);
                }
            }
        }
    }
    Ok(outer)
}

fn max_block_exponent(scales: &[u8]) -> Result<u8> {
    let exp = scales.iter().copied().max().context("empty MXFP8 scales")?;
    ensure!(exp != 255, "nonfinite MXFP8 block scale");
    Ok(exp)
}

const MIN_SHIFT: i32 = -20;
const MAX_SHIFT: i32 = 2;
const FP4_LEVELS: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];

struct Codebook {
    scale: u8,
    codes: [u8; 127],
}

fn e4m3(code: usize) -> f32 {
    let exponent = code >> 3;
    let fraction = (code & 7) as f32;
    if exponent == 0 { fraction / 512.0 } else { (1.0 + fraction / 8.0) * 2.0f32.powi(exponent as i32 - 7) }
}

fn nearest(levels: &[f32], value: f32) -> usize {
    let hi = levels.partition_point(|&x| x < value).min(levels.len() - 1);
    if hi == 0 { return hi; }
    let lo = hi - 1;
    let left = value - levels[lo];
    let right = levels[hi] - value;
    if left < right || (left == right && lo.is_multiple_of(2)) { lo } else { hi }
}

fn quantization_table() -> &'static [Codebook] {
    static TABLE: OnceLock<Vec<Codebook>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let levels = std::array::from_fn::<_, 127, _>(e4m3);
        (MIN_SHIFT..=MAX_SHIFT).flat_map(|shift| {
            let factor = 2.0f32.powi(shift);
            levels.map(|max| {
                let scale = nearest(&levels, (max * factor) / 6.0);
                let inverse = if scale == 0 { 0.0 } else { factor / levels[scale] };
                let codes = levels.map(|weight| nearest(&FP4_LEVELS, weight * inverse) as u8);
                Codebook { scale: scale as u8, codes }
            })
        }).collect()
    })
}

fn quantize_block(
    table: &[Codebook], codes: &[u8], scales: &[u8], row: usize, block: usize, columns: usize, max_exp: i32,
) -> Result<([u8; 16], u8)> {
    let at = row * columns + block * 16;
    let raw = &codes[at..at + 16];
    let peak = raw.iter().map(|v| v & 127).max().unwrap() as usize;
    ensure!(peak < 127, "nonfinite MXFP8 weight");
    let shift = i32::from(scales[at / 32]) - max_exp + 2;
    if shift < MIN_SHIFT || peak == 0 { return Ok(([0; 16], 0)); }
    let book = &table[(shift - MIN_SHIFT) as usize * 127 + peak];
    let quantized = std::array::from_fn(|i| book.codes[(raw[i] & 127) as usize] | ((raw[i] & 128) >> 4));
    Ok((quantized, book.scale))
}

fn directory(ckpt: &Checkpoint, layout: &[usize]) -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .context("cannot locate Blaze's expert cache")?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"blaze-rg16-v1:mx-fp4-rne-v1");
    for file in &ckpt.files {
        let path = file.canonicalize()?;
        let meta = path.metadata()?;
        hash.update(path.as_os_str().as_encoded_bytes());
        for value in [meta.dev(), meta.ino(), meta.len(), meta.mtime() as u64, meta.mtime_nsec() as u64] {
            hash.update(&value.to_le_bytes());
        }
    }
    for &layer in layout {
        hash.update(&(layer as u64).to_le_bytes());
    }
    Ok(base.join("blaze").join("rg16").join(hash.finalize().to_hex().as_str()))
}

pub(crate) fn cached_store(
    ckpt: &Checkpoint, layers: &[usize], experts: usize, hidden: usize, intermediate: usize, threads: usize,
    convert: impl Fn(usize, usize, &mut [u8]) -> Result<[f32; 3]> + Sync,
) -> Result<PathBuf> {
    let dir = directory(ckpt, layers)?;
    fs::create_dir_all(&dir)?;
    let lock = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(dir.join("convert.lock"))?;
    lock.lock()?;
    let stride = record_bytes(hidden, intermediate);
    let n = layers.len().checked_mul(experts).context("rg16 record count overflows")?;
    let records = n.checked_mul(stride).context("rg16 record bytes overflow")?;
    let geometry = records + 8;
    let end = geometry + n * 12;
    let header = serde_json::to_vec(&serde_json::json!({
        "blaze.rg16_v1.records": {"dtype":"U8", "shape":[layers.len(),experts,stride], "data_offsets":[0,records]},
        "blaze.rg16_v1.geometry": {"dtype":"I32", "shape":[2], "data_offsets":[records,geometry]},
        "blaze.rg16_v1.multipliers": {"dtype":"F32", "shape":[layers.len(),experts,3], "data_offsets":[geometry,end]}
    }))?;
    ensure!(header.len() < HEADER_BYTES - 8, "rg16 header exceeds its reserved space");
    let name = dir.join("experts.safetensors");
    if name.metadata().is_ok_and(|meta| meta.len() == (HEADER_BYTES + end) as u64) {
        return Ok(dir);
    }
    let temporary = dir.join("experts.partial");
    let result = write_store(&temporary, &header, &name, HEADER_BYTES + end, |file| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        pool.install(|| -> Result<()> {
            (0..n).into_par_iter().try_for_each_init(|| vec![0; stride], |record, index| -> Result<()> {
                let values = convert(layers[index / experts], index % experts, record)
                    .with_context(|| format!("packing rg16 expert {index}"))?;
                let bytes = values.map(f32::to_le_bytes).concat();
                file.write_all_at(record, (HEADER_BYTES + index * stride) as u64)?;
                file.write_all_at(&bytes, (HEADER_BYTES + geometry + index * 12) as u64)?;
                Ok(())
            })
        })?;
        let dims = [hidden as i32, intermediate as i32].map(i32::to_le_bytes).concat();
        file.write_all_at(&dims, (HEADER_BYTES + records) as u64)?;
        Ok(())
    });
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(dir)
}

fn write_store(
    tmp: &Path, header: &[u8], name: &Path, bytes: usize, write: impl FnOnce(&File) -> Result<()>,
) -> Result<()> {
    let file = File::create(tmp)?;
    file.set_len(bytes as u64)?;
    let mut padded = vec![b' '; HEADER_BYTES];
    padded[..8].copy_from_slice(&((HEADER_BYTES - 8) as u64).to_le_bytes());
    padded[8..8 + header.len()].copy_from_slice(header);
    file.write_all_at(&padded, 0)?;
    write(&file)?;
    file.sync_all()?;
    fs::rename(tmp, name)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(unit: &[u8], row: usize, column: usize) -> u8 {
        let byte = unit[column / 4 * 32 + (row % 8) * 4 + column % 4];
        (byte >> (4 * (row / 8))) & 15
    }

    #[test]
    fn nvfp4_codes_and_scales_survive_rg16_repack() {
        let codes: Vec<_> = (0..16 * 8).map(|i| (i * 19 + 11) as u8).collect();
        let scales: Vec<_> = (0..16).map(|i| (i + 19) as u8).collect();
        let mut unit = [0; UNIT];
        assert_eq!(convert_matrix(&codes, &scales, Some(0.0625), 16, 16, &mut unit).unwrap(), 0.0625);
        assert_eq!(&unit[128..], &scales);
        for row in 0..16 {
            for column in 0..16 {
                let expected = (codes[row * 8 + column / 2] >> (4 * (column % 2))) & 15;
                assert_eq!(unpack(&unit, row, column), expected);
            }
        }
    }

    #[test]
    fn mxfp8_converter_preserves_signs_and_rescales_blocks() {
        let mut codes = vec![0x38; 16 * 32];
        codes[1] = 0xb8;
        let scales: Vec<_> = (0..16).map(|i| if i < 8 { 127 } else { 130 }).collect();
        let mut packed = [0; 2 * UNIT];
        let outer = convert_matrix(&codes, &scales, None, 16, 32, &mut packed).unwrap();
        let dequant = |r: usize, c: usize| {
            let unit = &packed[c / 16 * UNIT..][..UNIT];
            let code = unpack(unit, r, c % 16);
            let sign = if code >= 8 { -1.0 } else { 1.0 };
            sign * FP4_LEVELS[(code & 7) as usize] * e4m3(unit[128 + r] as usize) * outer
        };
        assert!((dequant(0, 1) + 1.0).abs() < 0.07);
        assert!((dequant(0, 17) - 1.0).abs() < 0.07);
        assert!((dequant(8, 17) - 8.0).abs() < 0.56);
    }
}
