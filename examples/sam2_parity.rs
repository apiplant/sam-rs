//! Compares SAM 2 with PyTorch.  Usage: sam2_parity REF.safetensors [variant]   (REF from scripts/ref_sam2.py)
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sam_rs::sam2::{Sam2, Sam2Variant};

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
    let (g, w) = if g.dims() != w.dims() && g.elem_count() == w.elem_count() { (g.flatten_all()?, w.flatten_all()?) } else { (g, w) };
    if g.dims() != w.dims() {
        println!("{name:<12} SHAPE MISMATCH got {:?} want {:?}", g.dims(), w.dims());
        return Ok(());
    }
    let d = (&g - &w)?.abs()?;
    let (max, mean) = (d.flatten_all()?.max(0)?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{name:<12} max|d| {max:10.5}  mean|d| {mean:10.6}  mean|ref| {scale:8.4}  rel {:.2e}", mean / scale.max(1e-12));
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let device = Device::cuda_if_available(0)?;
    let r = candle_core::safetensors::load(&a[1], &device)?;
    let variant = Sam2Variant::parse(a.get(2).map(String::as_str).unwrap_or("sam2.1-hiera-tiny")).expect("variant");
    let sam = Sam2::from_pretrained(variant, &device, DType::F32)?;
    let levels = sam.encode_pixels(&r["pixel_values"])?;
    for (i, l) in levels.iter().enumerate() {
        diff(&format!("fpn{i}"), l, &r[&format!("fpn{i}")])?;
    }
    let tf = sam.features_from_levels(&levels)?;
    diff("s0", &tf.s0, &r["s0"])?;
    diff("s1", &tf.s1, &r["s1"])?;
    diff("emb", &tf.emb, &r["emb"])?;
    let p = r["pt"].flatten_all()?.to_vec1::<f32>()?;
    let b = r["bx"].flatten_all()?.to_vec1::<f32>()?;
    type Case = (&'static str, Vec<(f32, f32, i32)>, Vec<[f32; 4]>, bool);
    let cases: Vec<Case> = vec![("pt", vec![(p[0], p[1], 1)], vec![], true), ("pt1", vec![(p[0], p[1], 1)], vec![], false), ("bx", vec![], vec![[b[0], b[1], b[2], b[3]]], true)];
    for (name, pts, boxes, multi) in cases {
        println!("-- {name}");
        let (sparse, pred) = sam.predict_raw(&tf, &pts, &boxes, multi)?;
        diff("sparse", &sparse.unsqueeze(1)?, &r[&format!("{name}_sparse")])?;
        diff("masks", &pred.masks, &r[&format!("{name}_masks")])?;
        let want = r[&format!("{name}_iou")].flatten_all()?.to_vec1::<f32>()?;
        println!("{:<12} iou got {:?} want {:?}  obj got {:.3} want {:.3}", "scores", pred.iou, want, pred.object_score, r[&format!("{name}_obj")].flatten_all()?.to_vec1::<f32>()?[0]);
    }
    Ok(())
}
