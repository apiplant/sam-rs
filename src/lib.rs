//! Rust ([candle](https://github.com/huggingface/candle)) inference for Segment Anything models on
//! CPU and CUDA.

pub mod cpu_gemm;
pub mod dino;
pub mod download;
pub mod efficientsam;
pub mod image_proc;
pub mod kernels;
pub mod model;
pub mod sam1;
pub mod sam2;
pub mod sam3;
#[cfg(feature = "serve")]
pub mod serve;

pub use model::{Mask, Prompt, Sam, SamVariant};
