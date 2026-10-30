//! Original Segment Anything (ViT-B / ViT-L / ViT-H), Meta checkpoint layout.

pub mod decoder;
pub mod encoder;
pub mod prompt;
pub mod tinyvit;

pub use encoder::{ImageEncoder, EncoderConfig};
