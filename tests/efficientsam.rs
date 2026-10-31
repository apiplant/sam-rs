//! EfficientSAM tests. The checkpoint tests skip themselves when the weights are not in the cache.
use candle_core::{DType, Device};
use sam_rs::efficientsam::{EfficientSam, Variant};

// `cpu_gemm::set_enabled` is process-global, so tests that load a model must not interleave.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn cached(name: &str) -> Option<std::path::PathBuf> {
    let p = sam_rs::download::cache_dir()?.join("yunyangx--EfficientSAM").join(name);
    p.is_file().then_some(p)
}

fn test_image() -> image::RgbImage {
    image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, ((x + y) * 5 % 256) as u8]))
}

#[test]
fn variant_names() {
    assert_eq!(Variant::parse("efficientsam-ti"), Some(Variant::Tiny));
    assert_eq!(Variant::parse("efficientsam-s"), Some(Variant::Small));
    assert_eq!(Variant::parse("sam3"), None);
}

#[test]
fn pt_and_safetensors_agree() {
    let _g = LOCK.lock().unwrap();
    let (Some(pt), Some(st)) = (cached("efficient_sam_vitt.pt"), cached("efficient_sam_vitt.safetensors")) else {
        eprintln!("skipped: checkpoints not cached");
        return;
    };
    let dev = Device::Cpu;
    let img = test_image();
    let run = |p: &std::path::Path| {
        let m = EfficientSam::from_file(Variant::Tiny, p, &dev, DType::F32).unwrap();
        let e = m.embed(&img).unwrap();
        let masks = m.predict(&e, &[(40.0, 30.0, true)], &[]).unwrap();
        (e.tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap(), masks)
    };
    let (ea, ma) = run(&pt);
    let (eb, mb) = run(&st);
    assert_eq!(ea, eb);
    assert_eq!(ma.len(), mb.len());
    for (a, b) in ma.iter().zip(&mb) {
        assert_eq!((a.width, a.height), (96, 64));
        assert_eq!(a.data, b.data);
        assert_eq!(a.iou, b.iou);
    }
}

#[test]
fn prompts_produce_masks_in_image_size() {
    let _g = LOCK.lock().unwrap();
    let Some(st) = cached("efficient_sam_vitt.safetensors").or_else(|| cached("efficient_sam_vitt.pt")) else {
        eprintln!("skipped: checkpoint not cached");
        return;
    };
    let m = EfficientSam::from_file(Variant::Tiny, &st, &Device::Cpu, DType::F32).unwrap();
    let e = m.embed(&test_image()).unwrap();
    for (pts, boxes) in [(vec![(40.0, 30.0, true), (10.0, 10.0, false)], vec![]), (vec![], vec![[10.0, 10.0, 80.0, 50.0]])] {
        let masks = m.predict(&e, &pts, &boxes).unwrap();
        assert!(!masks.is_empty());
        for k in &masks {
            assert_eq!(k.data.len(), 96 * 64);
            assert!(k.iou.is_finite());
        }
    }
}

/// The bf16 CPU kernels (where the CPU has them) stay close to the plain f32 path.
#[test]
fn fast_cpu_path_matches_f32() {
    let _g = LOCK.lock().unwrap();
    let Some(st) = cached("efficient_sam_vitt.safetensors").or_else(|| cached("efficient_sam_vitt.pt")) else {
        eprintln!("skipped: checkpoint not cached");
        return;
    };
    let img = test_image();
    let embed = |fast: bool| {
        sam_rs::cpu_gemm::set_enabled(fast);
        let m = EfficientSam::from_file(Variant::Tiny, &st, &Device::Cpu, DType::F32).unwrap();
        m.embed(&img).unwrap().tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    let (fast, plain) = (embed(true), embed(false));
    sam_rs::cpu_gemm::set_enabled(true);
    let err: f32 = fast.iter().zip(&plain).map(|(a, b)| (a - b).abs()).sum::<f32>() / fast.len() as f32;
    let mag: f32 = plain.iter().map(|v| v.abs()).sum::<f32>() / plain.len() as f32;
    assert!(err / mag < 0.05, "relative error {}", err / mag);
}

/// The one-time `.pt` -> safetensors conversion yields the same tensors the converter script wrote.
#[test]
fn pt_conversion_round_trips() {
    let (Some(pt), Some(st)) = (cached("efficient_sam_vitt.pt"), cached("efficient_sam_vitt.safetensors")) else {
        eprintln!("skipped: checkpoints not cached");
        return;
    };
    let out = std::env::temp_dir().join(format!("sam-rs-conv-{}.safetensors", std::process::id()));
    sam_rs::download::convert_pt(&pt, &out).unwrap();
    let (a, b) = (candle_core::safetensors::load(&out, &Device::Cpu).unwrap(), candle_core::safetensors::load(&st, &Device::Cpu).unwrap());
    std::fs::remove_file(&out).ok();
    assert_eq!(a.len(), b.len());
    for (k, t) in &a {
        let d = (t - &b[k]).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0, "{k}");
    }
}
