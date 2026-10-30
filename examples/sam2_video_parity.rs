//! Compares the SAM 2 video tracker with PyTorch over a 6-frame clip.
//! Usage: sam2_video_parity REF.safetensors [variant]   (REF from scripts/ref_sam2_video.py)
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sam_rs::sam2::{Sam2, Sam2Variant};
use sam_rs::sam3::video::{FramePrompt, Session};

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?);
    if g.dims() != w.dims() {
        println!("{name:<14} SHAPE MISMATCH got {:?} want {:?}", g.dims(), w.dims());
        return Ok(());
    }
    let d = (&g - &w)?.abs()?;
    let (max, mean) = (d.max(0)?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{name:<14} max|d| {max:10.5}  mean|d| {mean:10.6}  mean|ref| {scale:8.4}  rel {:.2e}", mean / scale.max(1e-12));
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let device = Device::cuda_if_available(0)?;
    let r = candle_core::safetensors::load(&a[1], &device)?;
    let variant = Sam2Variant::parse(a.get(2).map(String::as_str).unwrap_or("sam2.1-hiera-tiny")).expect("variant");
    let sam = Sam2::from_pretrained(variant, &device, DType::F32)?;
    let p = r["point"].flatten_all()?.to_vec1::<f32>()?;
    let mut session = Session::new(None);
    session.add_prompt(0, FramePrompt { points: vec![(p[0], p[1], 1)] });
    for i in 0..6 {
        let e = sam.embed_pixels(&r["pix"].narrow(0, i, 1)?, (1200, 1500))?;
        if i == 1 {
            // isolate memory attention: hand it the reference's own frame-0 memory and pointer
            let mut s2 = Session::new(None);
            s2.add_prompt(0, FramePrompt { points: vec![(p[0], p[1], 1)] });
            let e0 = sam.embed_pixels(&r["pix"].narrow(0, 0, 1)?, (1200, 1500))?;
            sam.track_frame(&mut s2, &e0)?;
            s2.override_frame(0, r["mem0"].reshape((4096, 64))?.to_dtype(DType::BF16)?, r["mempos0"].reshape((4096, 64))?, r["f0_ptr"].reshape(256)?);
            diff("cond (ref mem)", &sam.conditioned_features(&s2, 1, &e)?, &r["cond2"])?;
            diff("cond (my mem)", &sam.conditioned_features(&session, 1, &e)?, &r["cond2"])?;
            let (m, iou, obj) = sam.decode_conditioned(&e, &r["cond2"])?;
            diff("decode(ref cond)", &m, &r["f1_pred"])?;
            println!("{:<14} decoder-only: iou {:?} obj {:.3} (ref obj {:.3})", "", iou, obj, r["f1_obj"].flatten_all()?.to_vec1::<f32>()?[0]);
        }
        let out = sam.track_frame(&mut session, &e)?;
        println!("-- frame {i}  obj {:.3} (ref {:.3})", out.object_score, r[&format!("f{i}_obj")].flatten_all()?.to_vec1::<f32>()?[0]);
        diff("pred_masks", &out.logits, &r[&format!("f{i}_pred")])?;
        let (g, w) = (out.logits.gt(0f32)?.flatten_all()?.to_vec1::<u8>()?, r[&format!("f{i}_pred")].gt(0f32)?.flatten_all()?.to_vec1::<u8>()?);
        let (inter, uni) = g.iter().zip(&w).fold((0u32, 0u32), |(a, b), (&x, &y)| (a + (x & y) as u32, b + (x | y) as u32));
        println!("{:<14} binary mask IoU vs reference {:.4}  ({} vs {} px)", "", inter as f32 / uni.max(1) as f32, g.iter().map(|&v| v as u32).sum::<u32>(), w.iter().map(|&v| v as u32).sum::<u32>());
        let (m, mp) = session.memory(i).expect("memory stored");
        diff("maskmem", &m.to_dtype(DType::F32)?, &r[&format!("mem{i}")])?;
        diff("maskmem_pos", mp, &r[&format!("mempos{i}")])?;
        diff("obj_pointer", session.object_pointer(i).expect("pointer"), &r[&format!("f{i}_ptr")])?;
    }
    Ok(())
}
