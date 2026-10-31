//! Compares Grounding DINO with PyTorch, stage by stage, on the reference's own pixels and token ids.
//! Usage: dino_parity REF.safetensors [--f16]   (REF from scripts/ref_dino.py)
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sam_rs::dino::GroundingDino;

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
    let (g, w) = if g.dims() != w.dims() && g.elem_count() == w.elem_count() { (g.flatten_all()?, w.flatten_all()?) } else { (g, w) };
    if g.dims() != w.dims() {
        println!("{name:<16} SHAPE MISMATCH got {:?} want {:?}", g.dims(), w.dims());
        return Ok(());
    }
    let d = (&g - &w)?.abs()?;
    let (max, mean) = (d.flatten_all()?.max(0)?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{name:<16} max|d| {max:10.5}  mean|d| {mean:10.6}  mean|ref| {scale:8.4}  rel {:.2e}", mean / scale.max(1e-12));
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let device = Device::cuda_if_available(0)?;
    let dtype = if a.iter().any(|x| x == "--f16") && device.is_cuda() { DType::F16 } else { DType::F32 };
    let r = candle_core::safetensors::load(&a[1], &device)?;
    let model = GroundingDino::from_pretrained(&device, dtype)?;
    let px = r["pixel_values"].unsqueeze(0)?.to_dtype(dtype)?;
    let ids: Vec<u32> = r["input_ids"].to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec1()?;
    let mine = model.tokenize("a wheel. a truck.")?;
    println!("tokens {} {:?}", if mine == ids { "match" } else { "DIFFER" }, mine);
    let mut trace = vec![];
    let t = std::time::Instant::now();
    let img0 = image::open(a.get(2).filter(|s| !s.starts_with("--")).map(String::as_str).unwrap_or("/tmp/claude-60426/ptt/truck.jpg"))?.to_rgb8();
    diff("preprocess", &model.preprocess(&img0)?.squeeze(0)?, &r["pixel_values"])?;
    let raw = model.forward(&px, &ids, Some(&mut trace))?;
    model.sync()?;
    println!("forward {:.0?} on {:?} {dtype:?}", t.elapsed(), device.location());
    for (name, t) in &trace {
        if let Some(w) = r.get(*name) {
            diff(name, t, w)?;
        }
    }
    diff("logits", &raw.logits, &r["logits"])?;
    diff("boxes", &raw.boxes, &r["boxes"])?;
    let img = image::open(a.get(2).filter(|s| !s.starts_with("--")).map(String::as_str).unwrap_or("/tmp/claude-60426/ptt/truck.jpg"))?.to_rgb8();
    for d in model.postprocess(&raw, (img.height() as usize, img.width() as usize), 0.35, 0.25)? {
        println!("{:.3} {:?} {:?}", d.score, d.label, d.bbox.map(|v| (v * 10.0).round() / 10.0));
    }
    Ok(())
}
