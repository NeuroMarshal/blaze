//! Raw C ABI of the blaze kernels (kernels/include/blaze/*.h). Pointers are device or pinned host memory as the
//! headers state; every launch takes the caller's CUDA stream.
#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cudaStream_t = *mut c_void;
pub type cudaError_t = i32;

pub const CUDA_SUCCESS: cudaError_t = 0;

unsafe extern "C" {
    pub fn cudaGetErrorString(err: cudaError_t) -> *const std::ffi::c_char;
}
