//! Compares the TinyViT encoder with candle-transformers' reference (which uses the tanh GELU) on identical input.
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let img = image::open(std::env::args().nth(1).expect("image path"))?.to_rgb8();
    let px = sam_rs::image_proc::prepare_sam1(&img, &device, DType::F32)?.tensor;
    let w = sam_rs::download::fetch("lmz/candle-sam", "mobile_sam-tiny-vitt.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &device)? };
    let oracle = candle_transformers::models::segment_anything::tiny_vit::tiny_vit_5m(vb.pp("image_encoder"))?;
    let want = oracle.forward(&px)?;
    let mine = sam_rs::sam1::tinyvit::TinyVit::new(vb.pp("image_encoder"), true)?.forward(&px)?;
    let d = (&mine - &want)?.abs()?;
    let (max, mean) = (d.flatten_all()?.max(0)?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    let scale = want.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("tinyvit (tanh gelu) vs candle-transformers: max|d| {max:.6}  mean|d| {mean:.7}  mean|ref| {scale:.4}  rel {:.2e}  shape {:?}", mean / scale, mine.dims());
    let erf = sam_rs::sam1::tinyvit::TinyVit::new(vb.pp("image_encoder"), false)?.forward(&px)?;
    let d2 = (&erf - &want)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("exact-erf GELU vs the tanh reference:           mean|d| {d2:.7}  rel {:.2e}  (expected: the oracle approximates GELU)", d2 / scale);
    let _: Option<Tensor> = None;
    Ok(())
}
