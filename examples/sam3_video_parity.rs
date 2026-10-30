//! Compares the SAM 3 video tracker with PyTorch over a 6-frame clip.
//! Usage: sam3_video_parity sam3_video_ref.safetensors   (from scripts/ref_sam3_video.py)
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sam_rs::sam3::video::{FramePrompt, Session};

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
    let (g, w) = if g.elem_count() == w.elem_count() { (g.flatten_all()?, w.flatten_all()?) } else { (g, w) };
    if g.dims() != w.dims() {
        println!("{name:<14} SHAPE MISMATCH got {:?} want {:?}", g.dims(), w.dims());
        return Ok(());
    }
    let d = (&g - &w)?.abs()?;
    let (max, mean) = (d.flatten_all()?.max(0)?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{name:<14} max|d| {max:10.5}  mean|d| {mean:10.6}  mean|ref| {scale:8.4}  rel {:.2e}", mean / scale.max(1e-12));
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let device = Device::cuda_if_available(0)?;
    let r = candle_core::safetensors::load(&a[1], &device)?;
    let sam = sam_rs::sam3::Sam3::from_file(&sam_rs::download::fetch("facebook/sam3", "model.safetensors")?, &sam_rs::download::fetch("facebook/sam3", "tokenizer.json")?, &device, DType::F32)?;
    let p = r["point"].flatten_all()?.to_vec1::<f32>()?;
    let mut session = Session::new(None);
    session.add_prompt(0, FramePrompt { points: vec![(p[0], p[1], 1)] });
    for i in 0..6 {
        let (tokens, _) = sam.encode_pixels_full(&r["pix"].narrow(0, i, 1)?)?;
        let img = sam_rs::sam3::ImageFeatures { fpn: vec![], tokens, orig: (1200, 1500) };
        let tf = sam.tracker_features(&img)?;
        if i == 1 {
            // isolate the memory attention: give it the reference's own frame-0 memory and pointer
            let mut s2 = Session::new(None);
            s2.add_prompt(0, FramePrompt { points: vec![(p[0], p[1], 1)] });
            let (tok0, _) = sam.encode_pixels_full(&r["pix"].narrow(0, 0, 1)?)?;
            let tf0 = sam.tracker_features(&sam_rs::sam3::ImageFeatures { fpn: vec![], tokens: tok0, orig: (1200, 1500) })?;
            sam.track_frame(&mut s2, &tf0)?;
            s2.override_frame(0, r["mem0"].reshape((5184, 64))?.to_dtype(DType::BF16)?, r["mempos0"].reshape((5184, 64))?, r["f0_ptr"].reshape(256)?);
            diff("cond (ref mem)", &sam.conditioned_features(&s2, 1, &tf)?, &r["cond2"])?;
        }
        if i > 0 {
            let c = sam.conditioned_features(&session, i, &tf)?;
            diff("conditioned", &c, &r[&format!("cond{}", i + 1)])?;
        }
        let out = sam.track_frame(&mut session, &tf)?;
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
