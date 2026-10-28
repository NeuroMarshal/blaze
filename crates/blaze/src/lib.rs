//! Quantized inference host: checkpoint loading, model execution, scheduling and serving.
//! CUDA C++ kernels are exposed through blaze-sys.

#[macro_use]
pub mod logging;
pub mod telemetry;
pub mod service;
pub mod cli;
pub mod config;
pub mod dflash;
pub mod drafter;
pub mod eagle;
pub mod engine;
pub mod gpu;
pub mod health;
pub mod inspect;
pub mod linear;
pub mod mm;
pub mod model;
pub mod mtp;
pub mod quant;
pub mod sampler;
pub mod sched;
pub mod server;
pub mod structured;
pub mod suffix;
pub mod text;
pub mod tree;
pub mod vision;
pub mod tune;
pub mod attn_tune;
pub mod weights;
