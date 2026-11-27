//! Device matrix multiplication shared by the model executors.

use crate::gpu::{Stream, check};
use crate::tune::{self, BF16_CUBLAS, BF16_MMA, BF16_SKINNY, BF16_WIDE};
use anyhow::Result;
use blaze_sys as sys;

#[derive(Clone, Copy)]
pub(crate) struct Nvfp4Rows {
    codes: *mut u8,
    scales: *mut u8,
    row_scales: *mut f32,
}

impl Nvfp4Rows {
    pub fn supported() -> bool {
        unsafe { sys::blz_nvfp4_gemm_w4a4_supported() != 0 }
    }

    pub fn bytes(rows: usize, inner: usize) -> usize {
        ((rows * (inner / 2)).next_multiple_of(64) + rows * (inner / 16)).next_multiple_of(64) + rows * 4
    }

    pub unsafe fn from_scratch(base: *mut u8, rows: usize, inner: usize) -> Self {
        let code_bytes = rows * (inner / 2);
        let scale_at = code_bytes.next_multiple_of(64);
        let scales_end = (scale_at + rows * (inner / 16)).next_multiple_of(64);
        Self {
            codes: base,
            scales: base.wrapping_add(scale_at),
            row_scales: base.wrapping_add(scales_end).cast(),
        }
    }

    pub unsafe fn quantize(self, input: *const u16, stride: usize, rows: usize, inner: usize,
        stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_nvfp4_quant_rows(input, stride as i64, rows as i32, inner as i32,
                self.codes, self.scales, self.row_scales, stream.0)
        }, "NVFP4 activation quantization")
    }

    pub unsafe fn quantize_residual(self, first: Self, input: *const u16, rows: usize, inner: usize,
        stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_nvfp4_quant_residual(input, inner as i64, rows as i32, inner as i32,
                first.codes, first.scales, first.row_scales, self.codes, self.scales, self.row_scales, stream.0)
        }, "NVFP4 activation residual quantization")
    }

    pub unsafe fn grouped(self, residual: Self, tiles: *const sys::blz_moe_matrix_tile, output: *mut u16,
        count: usize, columns: usize, inner: usize, stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_nvfp4_grouped_w4a4(self.codes, self.scales, self.row_scales, residual.codes, residual.scales,
                residual.row_scales, tiles, output, count as i32, columns as i32, inner as i32, stream.0)
        }, "NVFP4 native grouped matrix multiplication")
    }
}

pub(crate) struct Mxfp8Rows {
    codes: *mut u8,
    scales: *mut u8,
}

impl Mxfp8Rows {
    pub fn bytes(rows: usize, inner: usize) -> usize {
        rows * (inner + inner / 32)
    }

    pub unsafe fn from_scratch(base: *mut u8, rows: usize, inner: usize) -> Self {
        Self { codes: base, scales: base.wrapping_add(rows * inner) }
    }

    pub unsafe fn quantize(&self, input: *const u16, rows: usize, inner: usize, stream: &Stream) -> Result<()> {
        check(unsafe { sys::blz_mxfp8_quant_rows(input, self.codes, self.scales, rows as i32, inner as i32, stream.0) },
            "MXFP8 activation quantization")
    }

    pub unsafe fn grouped(&self, tiles: *const sys::blz_moe_matrix_tile, output: *mut u16,
        count: usize, columns: usize, inner: usize, stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_mxfp8_grouped_native(self.codes, self.scales, tiles, output, count as i32,
                columns as i32, inner as i32, stream.0)
        }, "MXFP8 native grouped matrix multiplication")
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Bf16Kernel {
    Cublas,
    Skinny(i32),
    Wide,
    Mma(i32),
}

pub(crate) struct Matmul {
    pub input: *const u16,
    pub output: *mut u16,
    pub rows: usize,
    pub columns: usize,
    pub inner: usize,
    pub input_stride: usize,
    pub output_stride: usize,
}

impl Matmul {
    pub unsafe fn nvfp4_native(&self, input: Nvfp4Rows, codes: *const u8, scales: *const u8, alpha: *const f32,
        stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_nvfp4_gemm_w4a4(input.codes, input.scales, input.row_scales, codes, scales, alpha,
                self.output, self.output_stride as i64, self.rows as i32, self.columns as i32, self.inner as i32, stream.0)
        }, "NVFP4 native matrix multiplication")
    }

    /// Device pointers must cover the indicated matrices, block scales and row multipliers.
    pub unsafe fn nvfp4(
        &self,
        weight: *const u8,
        scales: *const u8,
        alpha: *const f32,
        split: i32,
        tiles: i32,
        stream: &Stream,
    ) -> Result<()> {
        check(
            unsafe {
                sys::blz_nvfp4_gemv_mma_cfg(
                    self.input,
                    self.input_stride as i64,
                    weight,
                    scales,
                    alpha,
                    self.output,
                    self.output_stride as i64,
                    self.rows as i32,
                    self.columns as i32,
                    self.inner as i32,
                    split,
                    tiles,
                    stream.0,
                )
            },
            "blz_nvfp4_gemv_mma",
        )
    }

    pub fn tuned_bf16(&self, tuning: &tune::Tuning) -> Bf16Kernel {
        let Self {
            input,
            rows,
            columns,
            inner,
            input_stride,
            ..
        } = *self;
        let aligned = (input as usize).is_multiple_of(16);
        let skinny = aligned && tune::skinny_fits(rows, inner, input_stride);
        let mma = aligned && tune::mma_fits(rows, inner, input_stride);
        let wide = tune::wide_fits(rows, inner, input_stride);
        match tuning.get(tune::Kernel::Bf16, columns, inner, rows) {
            (BF16_SKINNY, cpb) if skinny => Bf16Kernel::Skinny(cpb),
            (BF16_WIDE, _) if wide => Bf16Kernel::Wide,
            (BF16_MMA, config) if mma => Bf16Kernel::Mma(config),
            (BF16_CUBLAS, _) => Bf16Kernel::Cublas,
            _ if wide && columns > 4096 => Bf16Kernel::Wide,
            _ if skinny && columns <= 4096 && rows <= 2 => Bf16Kernel::Skinny(0),
            _ if mma && columns <= 4096 => Bf16Kernel::Mma(0),
            _ => Bf16Kernel::Cublas,
        }
    }

    /// Device pointers must cover the indicated matrices until stream completion.
    pub unsafe fn bf16(
        &self,
        gemm: *mut sys::blz_gemm,
        weight: *const u16,
        kernel: Bf16Kernel,
        stream: &Stream,
    ) -> Result<()> {
        let Self {
            input,
            output,
            rows,
            columns,
            inner,
            input_stride,
            output_stride,
        } = *self;
        let (m, n, k) = (rows as i32, columns as i32, inner as i32);
        let (ldx, ldo) = (input_stride as i64, output_stride as i64);
        let (status, name) = unsafe {
            match kernel {
                Bf16Kernel::Cublas => (
                    sys::blz_gemm_bf16(gemm, input, ldx, weight, output, ldo, m, n, k, stream.0),
                    "blz_gemm_bf16",
                ),
                Bf16Kernel::Skinny(cpb) => (
                    sys::blz_skinny_gemm_cfg(
                        input.cast(),
                        ldx,
                        weight.cast(),
                        output.cast(),
                        ldo,
                        m,
                        n,
                        k,
                        sys::BLZ_BF16,
                        cpb,
                        stream.0,
                    ),
                    "blz_skinny_gemm",
                ),
                Bf16Kernel::Wide => (
                    sys::blz_gemv_wide(
                        input.cast(),
                        ldx,
                        weight.cast(),
                        output.cast(),
                        ldo,
                        m,
                        n,
                        k,
                        stream.0,
                    ),
                    "blz_gemv_wide",
                ),
                Bf16Kernel::Mma(config) => (
                    sys::blz_bf16_gemv_mma_cfg(
                        input,
                        ldx,
                        weight,
                        output,
                        ldo,
                        m,
                        n,
                        k,
                        config / 8,
                        config % 8,
                        stream.0,
                    ),
                    "blz_bf16_gemv_mma",
                ),
            }
        };
        check(status, name)
    }

    /// Device pointers must cover the indicated matrices and block scales until stream completion.
    pub unsafe fn mxfp8(
        &self,
        weight: *const u8,
        scales: *const u8,
        split: i32,
        tiles: i32,
        stream: &Stream,
    ) -> Result<()> {
        check(
            unsafe {
                sys::blz_mxfp8_gemv_mma_cfg(
                    self.input,
                    self.input_stride as i64,
                    weight,
                    scales,
                    self.output,
                    self.output_stride as i64,
                    self.rows as i32,
                    self.columns as i32,
                    self.inner as i32,
                    split,
                    tiles,
                    stream.0,
                )
            },
            "blz_mxfp8_gemv_mma",
        )
    }
}
