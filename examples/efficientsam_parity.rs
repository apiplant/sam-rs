//! Compares EfficientSAM with the ONNX exports run through onnxruntime.
//! Usage: efficientsam_parity REF.safetensors [efficientsam-ti|efficientsam-s]   (REF from scripts/ref_efficientsam.py)
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sam_rs::efficientsam::{EfficientSam, Variant};

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
    let r = candle_core::safetensors::load(&a[1], &device)?;
    let v = Variant::parse(a.get(2).map(String::as_str).unwrap_or("efficientsam-ti")).expect("variant");
    let sam = EfficientSam::from_pretrained(v, &device, DType::F32)?;
    // the oracle gets the raw [0,1] image; resize + normalize like the graph does
    let (_, h, w) = r["image"].dims3()?;
    let raw = (r["image"].to_device(&Device::Cpu)? * 255.0)?.round()?.permute((1, 2, 0))?.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
    let img = image::RgbImage::from_raw(w as u32, h as u32, raw).expect("image");
    let e = sam.embed(&img)?;
    diff("embedding", e.tensor(), &r["emb"].unsqueeze(0)?)?;
    // stage-by-stage against the dumped intermediates (single point prompt)
    if let Some(dbg_path) = a.get(3) {
        let d = candle_core::safetensors::load(dbg_path, &device)?;
        let coords = r["pt_coords"].to_vec2::<f32>()?;
        let (sx, sy) = (1024.0 / w as f32, 1024.0 / h as f32);
        let pts: Vec<(f32, f32, i32)> = coords.iter().map(|c| (c[0] * sx, c[1] * sy, 1)).collect();
        let sparse = sam.sparse(&pts)?;
        diff("sparse", &sparse, &d["sparse"])?;
        let mut trace = vec![];
        sam.decode_traced(e.tensor(), &sparse, &mut trace)?;
        for (i, t) in trace.iter().enumerate() {
            diff(&format!("ln{i}"), t, &d[&format!("ln{i}")])?;
        }
    }
    for name in ["pt", "pt2", "bx", "pt_inv"] {
        let coords = r[&format!("{name}_coords")].to_vec2::<f32>()?;
        let labels = r[&format!("{name}_labels")].to_vec1::<f32>()?;
        let sx = 1024.0 / w as f32;
        let sy = 1024.0 / h as f32;
        let pts: Vec<(f32, f32, i32)> = coords.iter().zip(&labels).map(|(c, &l)| (c[0] * sx, c[1] * sy, l as i32)).collect();
        println!("-- {name}  labels {labels:?}");
        let p = sam.decode(e.tensor(), &sam.sparse(&pts)?)?;
        if let Some(low) = r.get(&format!("{name}_low")) {
            diff("low-res", &p.masks.reshape(low.dims())?, low)?;
        }
        // reference masks are at original size: compare after the same bilinear upsample of ours
        let up = p.masks.to_dtype(DType::F32)?.unsqueeze(0)?.upsample_bilinear2d(h, w, false)?.squeeze(0)?;
        diff("masks", &up, &r[&format!("{name}_masks")])?;
        println!("{:<12} iou got {:?} want {:?}", "scores", p.iou, r[&format!("{name}_iou")].to_vec1::<f32>()?);
    }
    Ok(())
}
