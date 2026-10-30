//! Compares the SAM 3 image tracker with PyTorch.  Usage: sam3_tracker_parity sam3_ref.safetensors sam3_tracker_ref.safetensors
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
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
    let (r, t) = (candle_core::safetensors::load(&a[1], &device)?, candle_core::safetensors::load(&a[2], &device)?);
    let sam = sam_rs::sam3::Sam3::from_file(&sam_rs::download::fetch("facebook/sam3", "model.safetensors")?, &sam_rs::download::fetch("facebook/sam3", "tokenizer.json")?, &device, DType::F32)?;
    let (tokens, _) = sam.encode_pixels_full(&r["pixel_values"])?;
    let img = sam_rs::sam3::ImageFeatures { fpn: vec![], tokens, orig: (1200, 1800) };
    let tf = sam.tracker_features(&img)?;
    diff("s0", &tf.s0, &t["t_s0"])?;
    diff("s1", &tf.s1, &t["t_s1"])?;
    diff("emb", &tf.emb, &t["t_emb"])?;

    let p = t["pt"].flatten_all()?.to_vec1::<f32>()?;
    let b = t["bx"].flatten_all()?.to_vec1::<f32>()?;
    type Case = (&'static str, Vec<(f32, f32, i32)>, Vec<[f32; 4]>, bool);
    let cases: Vec<Case> = vec![
        ("pt", vec![(p[0], p[1], 1)], vec![], true),
        ("pt1", vec![(p[0], p[1], 1)], vec![], false),
        ("bx", vec![], vec![[b[0], b[1], b[2], b[3]]], true),
    ];
    for (name, pts, boxes, multi) in cases {
        println!("-- {name}");
        let (sparse, pred) = sam.tracker_predict(&tf, &pts, &boxes, multi)?;
        diff("sparse", &sparse.unsqueeze(1)?, &t[&format!("{name}_sparse")])?;
        diff("masks", &pred.masks.unsqueeze(0)?.unsqueeze(0)?.reshape(t[&format!("{name}_masks")].dims())?, &t[&format!("{name}_masks")])?;
        let want_iou = t[&format!("{name}_iou")].flatten_all()?.to_vec1::<f32>()?;
        println!("{:<12} iou got {:?} want {:?}  obj got {:.3} want {:.3}", "scores", pred.iou, want_iou, pred.object_score, t[&format!("{name}_obj")].flatten_all()?.to_vec1::<f32>()?[0]);
    }
    Ok(())
}
