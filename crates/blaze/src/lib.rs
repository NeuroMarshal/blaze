//! Quantized inference host: checkpoint loading, model execution, scheduling and serving.
//! CUDA C++ kernels are exposed through blaze-sys.

pub mod config;
pub mod inspect;
pub mod quant;
pub mod weights;
