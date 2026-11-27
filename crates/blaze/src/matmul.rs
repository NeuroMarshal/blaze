//! Device matrix multiplication shared by the model executors.

use crate::gpu::{Stream, check};
use crate::tune::{self, BF16_CUBLAS, BF16_MMA, BF16_SKINNY, BF16_WIDE};
use anyhow::Result;
use blaze_sys as sys;

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
