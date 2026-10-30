//! SAM 2 / SAM 2.1 (Hiera tiny, small, base+, large): point / box prompted segmentation of images, sharing the
//! prompt encoder and mask decoder with the SAM 3 image tracker.

pub mod hiera;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;

use crate::sam3::tracker::{Tracker, TrackerFeatures};
use crate::sam3::video::{FrameResult, Session, VideoTracker};

const SIZE: usize = 1024;
const SIDE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Tiny,
    Small,
    BasePlus,
    Large,
}

/// A SAM 2 checkpoint: size and generation (2.0 or 2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sam2Variant {
    pub size: Size,
    pub v2_1: bool,
}

impl Sam2Variant {
    /// `sam2.1-hiera-tiny`, `sam2-hiera-large`, and short forms `sam2-tiny`, `sam2.1-small`, `sam2-base-plus`.
    pub fn parse(s: &str) -> Option<Self> {
        let (v2_1, rest) = if let Some(r) = s.strip_prefix("sam2.1-") {
            (true, r)
        } else if let Some(r) = s.strip_prefix("sam2.0-") {
            (false, r)
        } else if let Some(r) = s.strip_prefix("sam2-") {
            // a bare `sam2-<size>` means the newer 2.1 weights; `sam2-hiera-<size>` is the original 2.0 repo name
            if r.starts_with("hiera-") { (false, r) } else { (true, r) }
        } else {
            return None;
        };
        let size = match rest.trim_start_matches("hiera-") {
            "tiny" | "t" => Size::Tiny,
            "small" | "s" => Size::Small,
            "base-plus" | "base_plus" | "b+" | "base+" => Size::BasePlus,
            "large" | "l" => Size::Large,
            _ => return None,
        };
        Some(Self { size, v2_1 })
    }

    pub fn repo(self) -> String {
        let size = match self.size {
            Size::Tiny => "tiny",
            Size::Small => "small",
            Size::BasePlus => "base-plus",
            Size::Large => "large",
        };
        format!("facebook/sam2{}-hiera-{size}", if self.v2_1 { ".1" } else { "" })
    }
}

pub struct Sam2 {
    enc: hiera::Hiera,
    tracker: Tracker,
    video: VideoTracker,
    pos: Tensor,
    pub device: Device,
    pub dtype: DType,
}

/// An embedded image, reusable across prompts.
pub struct Embedding {
    tf: TrackerFeatures,
    orig: (usize, usize),
}

impl Sam2 {
    pub fn from_dir(config: &std::path::Path, weights: &std::path::Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config).with_context(|| format!("reading {}", config.display()))?)?;
        let hc = hiera::HieraConfig::from_json(&cfg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        Ok(Self {
            enc: hiera::Hiera::new(&hc, vb.clone(), SIZE, device)?,
            tracker: Tracker::new(vb.clone(), SIDE, SIZE, device, dtype)?,
            video: VideoTracker::new(vb, SIDE, SIZE, None, device)?,
            pos: hiera::level_pos(SIDE, 256, device, dtype)?,
            device: device.clone(),
            dtype,
        })
    }

    /// Downloads the checkpoint on first use.
    pub fn from_pretrained(variant: Sam2Variant, device: &Device, dtype: DType) -> Result<Self> {
        let repo = variant.repo();
        let config = crate::download::fetch(&repo, "config.json")?;
        let weights = crate::download::fetch(&repo, "model.safetensors")?;
        Self::from_dir(&config, &weights, device, dtype)
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    /// Encoder levels for an already-preprocessed `[1, 3, 1024, 1024]` tensor (exposed for tests).
    pub fn encode_pixels(&self, px: &Tensor) -> Result<Vec<Tensor>> {
        Ok(self.enc.forward(&px.to_dtype(self.dtype)?)?)
    }

    pub fn features_from_levels(&self, levels: &[Tensor]) -> Result<TrackerFeatures> {
        Ok(self.tracker.features(levels)?)
    }

    /// Embedding of an already-preprocessed `[1, 3, 1024, 1024]` tensor (exposed for tests).
    pub fn embed_pixels(&self, px: &Tensor, orig: (usize, usize)) -> Result<Embedding> {
        let tf = self.features_from_levels(&self.encode_pixels(px)?)?;
        Ok(Embedding { tf, orig })
    }

    pub fn embed(&self, img: &RgbImage) -> Result<Embedding> {
        let px = crate::image_proc::prepare_square(img, SIZE, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], &self.device, self.dtype)?;
        let tf = self.features_from_levels(&self.encode_pixels(&px)?)?;
        Ok(Embedding { tf, orig: (img.height() as usize, img.width() as usize) })
    }

    /// Raw decoder prediction for prompts already in the 1024-px input space (exposed for tests).
    pub fn predict_raw(&self, tf: &TrackerFeatures, points: &[(f32, f32, i32)], boxes: &[[f32; 4]], multimask: bool) -> Result<(Tensor, crate::sam3::tracker::Prediction)> {
        let sparse = self.tracker.sparse(points, boxes, &self.device, self.dtype)?;
        let p = self.tracker.predict_sparse(tf, &sparse, multimask)?;
        Ok((sparse, p))
    }

    /// Memory-conditioned features `[1, 256, 64, 64]` for `frame_idx` given the session so far (exposed for tests).
    pub fn conditioned_features(&self, session: &Session, frame_idx: usize, e: &Embedding) -> Result<Tensor> {
        Ok(self.video.conditioned(session, frame_idx, &e.tf, &self.pos)?)
    }

    /// Decoder run on given conditioned features for a frame without prompt (exposed for tests): best-IoU mask logits.
    pub fn decode_conditioned(&self, e: &Embedding, cond: &Tensor) -> Result<(Tensor, Vec<f32>, f32)> {
        let tf = TrackerFeatures { s0: e.tf.s0.clone(), s1: e.tf.s1.clone(), emb: cond.clone(), raw: e.tf.raw.clone() };
        let (_, p) = self.predict_raw(&tf, &[], &[], true)?;
        let best = (0..p.iou.len()).max_by(|&a, &b| p.iou[a].total_cmp(&p.iou[b])).unwrap_or(0);
        Ok((p.masks.narrow(0, best, 1)?.squeeze(0)?, p.iou, p.object_score))
    }

    /// Advances `session` by one frame (streaming) and returns that frame's low-resolution mask logits `[256, 256]`.
    /// Add prompts (points in the 1024-px input space; use [`prompt_points`](Self::prompt_points)) with
    /// [`Session::add_prompt`] before processing the frame they belong to.
    pub fn track_frame(&self, session: &mut Session, e: &Embedding) -> Result<FrameResult> {
        Ok(self.video.track(&self.tracker, session, &e.tf, &self.pos, &self.device)?)
    }

    /// Converts prompts in original-image pixels (`orig = (height, width)`) to what [`Session::add_prompt`] expects:
    /// points as `(x, y, label)` with labels 1 / 0, a box as two corner points labelled 2 and 3.
    pub fn prompt_points(&self, orig: (usize, usize), points: &[(f32, f32, bool)], boxes: &[[f32; 4]]) -> crate::sam3::video::FramePrompt {
        let (sx, sy) = (SIZE as f32 / orig.1 as f32, SIZE as f32 / orig.0 as f32);
        let mut p = crate::sam3::video::FramePrompt::default();
        for &(x, y, fg) in points {
            p.points.push((x * sx, y * sy, fg as i32));
        }
        for b in boxes {
            p.points.push((b[0] * sx, b[1] * sy, 2));
            p.points.push((b[2] * sx, b[3] * sy, 3));
        }
        p
    }

    /// A tracked frame's mask logits as a binary mask at the original size.
    pub fn frame_mask(&self, r: &FrameResult, orig: (usize, usize)) -> Result<crate::Mask> {
        let data = binarize(&r.logits.unsqueeze(0)?, orig.0, orig.1)?.pop().expect("one mask");
        Ok(crate::Mask { width: orig.1, height: orig.0, data, iou: r.object_score })
    }

    /// Point / box prompts in original-image pixels. Returns up to three candidate masks (one with
    /// `multimask = false`) at the original resolution and the object-presence logit.
    pub fn predict(&self, e: &Embedding, points: &[(f32, f32, bool)], boxes: &[[f32; 4]], multimask: bool) -> Result<(Vec<crate::Mask>, f32)> {
        let (oh, ow) = e.orig;
        let (sx, sy) = (SIZE as f32 / ow as f32, SIZE as f32 / oh as f32);
        let pts: Vec<(f32, f32, i32)> = points.iter().map(|&(x, y, fg)| (x * sx, y * sy, fg as i32)).collect();
        let bxs: Vec<[f32; 4]> = boxes.iter().map(|b| [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]).collect();
        let (_, p) = self.predict_raw(&e.tf, &pts, &bxs, multimask)?;
        let bin = binarize(&p.masks, oh, ow)?;
        let masks = bin.into_iter().zip(p.iou).map(|(data, iou)| crate::Mask { width: ow, height: oh, data, iou }).collect();
        Ok((masks, p.object_score))
    }
}

/// Mask logits `[M, n, n]` -> binary masks at `oh x ow` (parallel on CPU, candle's interpolation on the GPU).
pub(crate) fn binarize(masks: &Tensor, oh: usize, ow: usize) -> Result<Vec<Vec<u8>>> {
    let (m, n, _) = masks.dims3()?;
    if masks.device().is_cpu() {
        let flat = masks.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        return Ok((0..m).map(|i| crate::image_proc::upsample_threshold(&flat[i * n * n..(i + 1) * n * n], n, oh, ow)).collect());
    }
    let up = masks.to_dtype(DType::F32)?.unsqueeze(0)?.upsample_bilinear2d(oh, ow, false)?;
    Ok(up.gt(0f32)?.to_dtype(DType::U8)?.squeeze(0)?.to_vec3::<u8>()?.into_iter().map(|m| m.into_iter().flatten().collect()).collect())
}
