//! Grounding DINO: the deformable-attention op against a scalar reference, the box helpers, and (when the checkpoint is
//! in the cache) a run of the whole network.
use candle_core::{DType, Device, Result, Tensor};
use sam_rs::dino::{ground, nms, Detection, GroundingDino};
use sam_rs::kernels::deform_attn;
use sam_rs::Mask;

/// `grid_sample(align_corners=False, padding_mode="zeros")` applied the slow way.
fn naive(value: &[f32], loc: &[f32], attn: &[f32], shapes: &[(usize, usize)], (q, h, d, p): (usize, usize, usize, usize)) -> Vec<f32> {
    let l = shapes.len();
    let s: usize = shapes.iter().map(|(a, b)| a * b).sum();
    let mut out = vec![0f32; q * h * d];
    for qi in 0..q {
        for hi in 0..h {
            let mut start = 0;
            for (li, &(hl, wl)) in shapes.iter().enumerate() {
                for pi in 0..p {
                    let base = ((qi * h + hi) * l + li) * p + pi;
                    let (x, y) = (loc[base * 2] * wl as f32 - 0.5, loc[base * 2 + 1] * hl as f32 - 0.5);
                    let at = |yy: i64, xx: i64, c: usize| -> f32 {
                        if yy < 0 || xx < 0 || yy >= hl as i64 || xx >= wl as i64 {
                            0.0
                        } else {
                            value[((start + yy as usize * wl + xx as usize) * h + hi) * d + c]
                        }
                    };
                    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
                    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                    for c in 0..d {
                        let v = at(y0, x0, c) * (1.0 - fx) * (1.0 - fy) + at(y0, x0 + 1, c) * fx * (1.0 - fy) + at(y0 + 1, x0, c) * (1.0 - fx) * fy + at(y0 + 1, x0 + 1, c) * fx * fy;
                        out[(qi * h + hi) * d + c] += attn[base] * v;
                    }
                }
                start += hl * wl;
            }
        }
    }
    assert!(start_sanity(s, shapes));
    out
}

fn start_sanity(s: usize, shapes: &[(usize, usize)]) -> bool {
    s == shapes.iter().map(|(a, b)| a * b).sum::<usize>()
}

#[test]
fn deform_attn_matches_scalar_reference() -> Result<()> {
    let shapes = [(7usize, 9usize), (4, 5), (2, 3)];
    let s: usize = shapes.iter().map(|(h, w)| h * w).sum();
    let (q, h, d, p) = (13usize, 4usize, 8usize, 3usize);
    let dev = Device::Cpu;
    let r = |shape: &[usize]| Tensor::randn(0f32, 1f32, shape, &dev);
    let value = r(&[1, s, h, d])?;
    // locations beyond [0, 1] exercise the zero padding
    let loc = (r(&[1, q, h, shapes.len(), p, 2])? * 0.6)?.affine(1.0, 0.5)?;
    let attn = r(&[1, q, h, shapes.len(), p])?;
    let got = deform_attn(&value, &loc, &attn, &shapes)?.flatten_all()?.to_vec1::<f32>()?;
    let flat = |t: &Tensor| t.flatten_all().and_then(|t| t.to_vec1::<f32>());
    let want = naive(&flat(&value)?, &flat(&loc)?, &flat(&attn)?, &shapes, (q, h, d, p));
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-4, "max error {err}");
    Ok(())
}

fn det(bbox: [f32; 4], score: f32) -> Detection {
    Detection { bbox, score, label: "x".into() }
}

#[test]
fn nms_keeps_the_best_of_overlapping_boxes() {
    let kept = nms(vec![det([0.0, 0.0, 10.0, 10.0], 0.6), det([1.0, 1.0, 11.0, 11.0], 0.9), det([50.0, 50.0, 60.0, 60.0], 0.5)], 0.5);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].score, 0.9);
    assert_eq!(kept[1].score, 0.5);
}

#[test]
fn ground_keeps_the_best_candidate_per_box() -> anyhow::Result<()> {
    let mask = |iou: f32, v: u8| Mask { width: 2, height: 1, data: vec![v, v], iou };
    let out = ground(vec![det([0.0, 0.0, 1.0, 1.0], 0.8)], |_| Ok(vec![mask(0.2, 1), mask(0.9, 7), mask(0.5, 3)]))?;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1.data, vec![7, 7]);
    Ok(())
}

#[test]
fn detects_without_errors_when_the_checkpoint_is_cached() {
    // tiny and base name the shared box head differently (`bbox_embed.0` vs `model.decoder.bbox_embed.0`)
    for repo in [sam_rs::dino::REPO_TINY, sam_rs::dino::REPO_BASE] {
        detect_with(repo);
    }
}

fn detect_with(repo: &str) {
    if ["config.json", "model.safetensors", "tokenizer.json"].iter().any(|f| sam_rs::download::cached(repo, f).is_none()) {
        eprintln!("skipped: {repo} is not cached");
        return;
    }
    let m = GroundingDino::from_repo(repo, &Device::Cpu, DType::F32).unwrap();
    let img = image::RgbImage::from_fn(160, 120, |x, y| image::Rgb([(x * 3) as u8, (y * 4) as u8, 120]));
    // a low threshold so something comes back even on a synthetic picture
    let found = m.detect(&img, "a red square. a person", 0.0, 0.0).unwrap();
    assert!(!found.is_empty());
    for d in &found {
        assert!(d.score.is_finite() && d.bbox.iter().all(|v| v.is_finite()));
    }
    assert!(m.detect(&img, "   ", 0.3, 0.2).is_err() || m.detect(&img, "   ", 0.3, 0.2).unwrap().is_empty());
}
