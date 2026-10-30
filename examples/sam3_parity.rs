//! Compares each SAM 3 stage with activations dumped from the PyTorch reference
//! (`ref_sam3.py` -> sam3_ref.safetensors).  Usage: sam3_parity REF.safetensors [cpu]
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn diff(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let (g, w) = (got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
    if g.dims() != w.dims() {
        println!("{name:<14} SHAPE MISMATCH got {:?} want {:?}", g.dims(), w.dims());
        return Ok(());
    }
    let d = (&g - &w)?.abs()?;
    let max = d.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean = d.mean_all()?.to_scalar::<f32>()?;
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{name:<14} max|d| {max:10.5}  mean|d| {mean:10.6}  mean|ref| {scale:8.4}  rel {:.2e}", mean / scale.max(1e-12));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = if args.get(2).map(|s| s == "cpu").unwrap_or(false) { Device::Cpu } else { Device::cuda_if_available(0)? };
    let r = candle_core::safetensors::load(&args[1], &device)?;
    let w = sam_rs::download::fetch("facebook/sam3", "model.safetensors")?;
    let t = sam_rs::download::fetch("facebook/sam3", "tokenizer.json")?;
    let sam = sam_rs::sam3::Sam3::from_file(&w, &t, &device, DType::F32)?;
    println!("device {:?}", device.location());

    let fpn = sam.encode_pixels(&r["pixel_values"])?;
    for i in 0..3 {
        diff(&format!("fpn{i}"), &fpn[i], &r[&format!("fpn{i}")])?;
    }
    let ids: Vec<u32> = r["input_ids"].to_vec2::<i64>()?[0].iter().map(|&v| v as u32).collect();
    let valid: Vec<u8> = r["attention_mask"].to_vec2::<i64>()?[0].iter().map(|&v| v as u8).collect();
    let (my_ids, my_valid) = sam.tokenize("truck")?;
    println!("tokenizer ids match: {}  mask match: {}", my_ids == ids, my_valid == valid);
    let text = sam.text_features(&ids)?;
    diff("text_proj", &text, &r["text_proj"])?;
    let n = valid.iter().filter(|&&v| v != 0).count();
    diff("text_proj[valid]", &text.narrow(1, 0, n)?, &r["text_proj"].narrow(1, 0, n)?)?;

    let ref_fpn: Vec<Tensor> = (0..3).map(|i| r[&format!("fpn{i}")].clone()).collect();
    for (label, f) in [("ref-fpn", &ref_fpn), ("my-fpn", &fpn)] {
        let o = sam.head(f, &r["text_proj"], &valid)?;
        println!("-- head with {label}");
        diff("pred_logits", &o.logits, &r["pred_logits"])?;
        diff("pred_boxes", &o.boxes, &r["pred_boxes"])?;
        diff("presence", &o.presence, &r["presence_logits"])?;
        diff("pred_masks", &o.masks, &r["pred_masks"])?;
    }
    // -- geometry (box) prompt: text "visual" + one positive exemplar box
    if r.contains_key("geo_boxes") {
        println!("-- geometry prompt");
        let b = r["geo_boxes"].to_vec3::<f32>()?[0][0].clone();
        let boxes = [[b[0], b[1], b[2], b[3]]];
        let labels: Vec<u32> = r["geo_labels"].to_vec2::<i64>()?[0].iter().map(|&v| v as u32).collect();
        let pos = sam.pos72_tokens();
        let gf = sam.geometry_features(&boxes, &labels, &ref_fpn[2], pos)?;
        diff("geo_feats", &gf, &r["geo_feats"])?;
        let ids: Vec<u32> = r["geo_ids"].to_vec2::<i64>()?[0].iter().map(|&v| v as u32).collect();
        let mut valid: Vec<u8> = r["geo_mask"].to_vec2::<i64>()?[0].iter().map(|&v| v as u8).collect();
        let text = Tensor::cat(&[&sam.text_features(&ids)?, &gf], 1)?;
        valid.extend(std::iter::repeat(1u8).take(gf.dim(1)?));
        let o = sam.head(&ref_fpn, &text, &valid)?;
        diff("geo_logits", &o.logits, &r["geo_logits"])?;
        diff("geo_boxes_out", &o.boxes, &r["geo_boxes_out"])?;
        diff("geo_presence", &o.presence, &r["geo_presence"])?;
        diff("geo_masks", &o.masks, &r["geo_masks"])?;
    }
    Ok(())
}
