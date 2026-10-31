//! Both SAM 2 generations must load (2.0 checkpoints lack the 2.1-only tensors) and track a short clip.
use candle_core::{DType, Device};
use sam_rs::sam2::{Sam2, Sam2Variant};
use sam_rs::sam3::video::Session;

#[test]
fn loads_and_tracks_both_generations() {
    for name in ["sam2-hiera-tiny", "sam2.1-hiera-tiny"] {
        let v = Sam2Variant::parse(name).unwrap();
        let repo = v.repo();
        if sam_rs::download::cached(&repo, "model.safetensors").is_none() || sam_rs::download::cached(&repo, "config.json").is_none() {
            eprintln!("skipped {name}: not cached");
            continue;
        }
        let m = Sam2::from_pretrained(v, &Device::Cpu, DType::F32).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let img = image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, 90]));
        let mut session = Session::new(None);
        session.add_prompt(0, m.prompt_points((64, 96), &[(48.0, 32.0, true)], &[]));
        for _ in 0..3 {
            let e = m.embed(&img).unwrap();
            let r = m.track_frame(&mut session, &e).unwrap();
            assert!(r.object_score.is_finite(), "{name}");
        }
    }
}
