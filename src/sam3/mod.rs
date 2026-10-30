//! SAM 3 (Segment Anything with Concepts): text-prompted instance segmentation. Port of the image path of
//! `facebook/sam3` (`Sam3Model`): ViT + FPN, CLIP text encoder, fusion encoder, DETR decoder, mask head.

pub mod detr;
pub mod geometry;
pub mod nn;
pub mod text;
pub mod tracker;
pub mod video;
pub mod vision;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;

pub const REPO: &str = "facebook/sam3";
const SIZE: usize = 1008;
const SIDE: usize = 72;

/// One detected instance, in original-image pixels.
pub struct Instance {
    pub score: f32,
    /// `[x0, y0, x1, y1]`
    pub bbox: [f32; 4],
    pub mask: crate::Mask,
}

/// Image features reused across text prompts.
pub struct ImageFeatures {
    /// FPN levels 288², 144², 72² (the 36² level is not used by the image path)
    pub fpn: Vec<Tensor>,
    /// ViT output `[1, 72, 72, 1024]`, shared with the tracker's neck
    pub tokens: Tensor,
    pub orig: (usize, usize),
}

/// Raw head outputs for one prompt.
pub struct Outputs {
    pub logits: Tensor,
    pub boxes: Tensor,
    pub presence: Tensor,
    pub masks: Tensor,
}

pub struct Sam3 {
    vision: vision::VisionEncoder,
    text: text::TextEncoder,
    enc: detr::Encoder,
    dec: detr::Decoder,
    scoring: detr::Scoring,
    mask: detr::MaskDecoder,
    geometry: geometry::GeometryEncoder,
    tracker: tracker::Tracker,
    tracker_neck: Vec<vision::Fpn>,
    video: video::VideoTracker,
    pos72: Tensor,
    tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub dtype: DType,
}

impl Sam3 {
    pub fn from_file(weights: &std::path::Path, tokenizer: &std::path::Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let d = vb.pp("detector_model");
        let mut tk = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tk.with_truncation(None).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            vision: vision::VisionEncoder::new(d.pp("vision_encoder"), device)?,
            text: text::TextEncoder::new(d.clone(), device, dtype)?,
            enc: detr::Encoder::new(d.pp("detr_encoder"))?,
            dec: detr::Decoder::new(d.pp("detr_decoder"))?,
            scoring: detr::Scoring::new(d.pp("dot_product_scoring"))?,
            mask: detr::MaskDecoder::new(d.pp("mask_decoder"))?,
            geometry: geometry::GeometryEncoder::new(d.pp("geometry_encoder"))?,
            tracker: tracker::Tracker::new(vb.pp("tracker_model"), SIDE, SIZE, device, dtype)?,
            tracker_neck: vision::VisionEncoder::tracker_neck(vb.pp("tracker_neck"))?,
            video: video::VideoTracker::new(vb.pp("tracker_model"), SIDE, SIZE, Some(4), device)?,
            pos72: vision::sine_pos(SIDE, SIDE, device, dtype)?.flatten_from(2)?.transpose(1, 2)?.contiguous()?,
            tokenizer: tk,
            device: device.clone(),
            dtype,
        })
    }

    /// Downloads `facebook/sam3` on first use. The repository is gated: see the README on `HF_TOKEN`.
    pub fn from_pretrained(device: &Device, dtype: DType) -> Result<Self> {
        let w = crate::download::fetch(REPO, "model.safetensors")?;
        let t = crate::download::fetch(REPO, "tokenizer.json")?;
        Self::from_file(&w, &t, device, dtype)
    }

    /// `[BOS, tokens..., EOS]` padded with EOS to 32, plus the 0/1 validity of each position.
    pub fn tokenize(&self, prompt: &str) -> Result<(Vec<u32>, Vec<u8>)> {
        let enc = self.tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        ids.truncate(text::MAX_LEN);
        let valid = vec![1u8; ids.len()];
        let mut valid = valid;
        ids.resize(text::MAX_LEN, 49407);
        valid.resize(text::MAX_LEN, 0);
        Ok((ids, valid))
    }

    /// Blocks until queued device work is done (for timing).
    pub fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    /// Position tokens of the 72x72 level `[1, 5184, 256]` (exposed for tests).
    pub fn pos72_tokens(&self) -> &Tensor {
        &self.pos72
    }

    /// Geometry encoder output for normalized `(cx, cy, w, h)` boxes (exposed for tests).
    pub fn geometry_features(&self, boxes: &[[f32; 4]], labels: &[u32], fpn72: &Tensor, pos: &Tensor) -> Result<Tensor> {
        Ok(self.geometry.forward(boxes, labels, fpn72, pos)?)
    }

    pub fn encode_pixels(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        Ok(self.encode_pixels_full(pixel_values)?.1)
    }

    /// ViT tokens and the three detector FPN levels.
    pub fn encode_pixels_full(&self, pixel_values: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (tokens, levels) = self.vision.forward(&pixel_values.to_dtype(self.dtype)?)?;
        Ok((tokens, levels.into_iter().take(3).collect()))
    }

    pub fn encode_image(&self, img: &RgbImage) -> Result<ImageFeatures> {
        let px = crate::image_proc::prepare_square(img, SIZE, [127.5; 3], [127.5; 3], &self.device, self.dtype)?;
        let (tokens, fpn) = self.encode_pixels_full(&px)?;
        Ok(ImageFeatures { fpn, tokens, orig: (img.height() as usize, img.width() as usize) })
    }

    pub fn text_features(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self.text.forward(ids)?)
    }

    /// Everything after the vision encoder, for one prompt.
    pub fn head(&self, fpn: &[Tensor], text: &Tensor, valid: &[u8]) -> Result<Outputs> {
        let mask = nn::key_padding_mask(valid, self.dtype, &self.device)?;
        let vis = fpn[2].flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let enc = self.enc.forward(&vis, &self.pos72, text, &mask)?;
        let dec = self.dec.forward(&enc, &self.pos72, text, &mask, SIDE, SIDE)?;
        let logits = self.scoring.forward(&dec.queries, text, valid)?;
        let masks = self.mask.forward(&dec.queries, &fpn[..2], &enc, text, &mask, SIDE)?;
        Ok(Outputs { logits, boxes: detr::boxes_xyxy(&dec.boxes)?, presence: dec.presence, masks })
    }

    /// Tracker features straight from an image, skipping the detector's FPN (what video tracking needs per frame).
    pub fn encode_for_tracking(&self, img: &RgbImage) -> Result<tracker::TrackerFeatures> {
        let px = crate::image_proc::prepare_square(img, SIZE, [127.5; 3], [127.5; 3], &self.device, self.dtype)?;
        let tokens = self.vision.tokens(&px.to_dtype(self.dtype)?)?;
        self.tracker_features_from_tokens(&tokens)
    }

    fn tracker_features_from_tokens(&self, tokens: &Tensor) -> Result<tracker::TrackerFeatures> {
        let f = self.tracker_neck.iter().take(3).map(|n| n.forward(tokens)).collect::<Result<Vec<_>, _>>()?;
        Ok(self.tracker.features(&f)?)
    }

    /// Mask logits of a tracked frame as a binary mask at the original size.
    pub fn frame_mask(&self, r: &video::FrameResult, orig: (usize, usize)) -> Result<crate::Mask> {
        let data = crate::sam2::binarize(&r.logits.unsqueeze(0)?, orig.0, orig.1)?.pop().expect("one mask");
        Ok(crate::Mask { width: orig.1, height: orig.0, data, iou: r.object_score })
    }

    /// Tracker features (the tracker's own FPN over the shared ViT tokens). Compute once per image.
    pub fn tracker_features(&self, f: &ImageFeatures) -> Result<tracker::TrackerFeatures> {
        self.tracker_features_from_tokens(&f.tokens)
    }

    /// Point / box prompts in original-image pixels, like SAM 1/2: `points` are `(x, y, positive)`, `boxes`
    /// `[x0, y0, x1, y1]`. Returns up to three candidate masks (or one with `multimask = false`) and the
    /// object-presence logit (positive means "an object is there").
    pub fn segment_points(&self, f: &ImageFeatures, tf: &tracker::TrackerFeatures, points: &[(f32, f32, bool)], boxes: &[[f32; 4]], multimask: bool) -> Result<(Vec<crate::Mask>, f32)> {
        let (oh, ow) = f.orig;
        let (sx, sy) = (SIZE as f32 / ow as f32, SIZE as f32 / oh as f32);
        let pts: Vec<(f32, f32, i32)> = points.iter().map(|&(x, y, fg)| (x * sx, y * sy, fg as i32)).collect();
        let bxs: Vec<[f32; 4]> = boxes.iter().map(|b| [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]).collect();
        let sparse = self.tracker.sparse(&pts, &bxs, &self.device, self.dtype)?;
        let p = self.tracker.predict_sparse(tf, &sparse, multimask)?;
        let bin = crate::sam2::binarize(&p.masks, oh, ow)?;
        let masks = bin.into_iter().zip(p.iou).map(|(data, iou)| crate::Mask { width: ow, height: oh, data, iou }).collect();
        Ok((masks, p.object_score))
    }

    /// Advances `session` by one frame (streaming) and returns that frame's low-resolution mask logits.
    /// `tf` are the frame's tracker features ([`tracker_features`](Self::tracker_features)). Add prompts with
    /// [`video::Session::add_prompt`] before processing the frame they belong to.
    pub fn track_frame(&self, session: &mut video::Session, tf: &tracker::TrackerFeatures) -> Result<video::FrameResult> {
        Ok(self.video.track(&self.tracker, session, tf, &self.pos72, &self.device)?)
    }

    /// Memory-conditioned features `[1, 256, 72, 72]` for `frame_idx` given the session so far (exposed for tests).
    pub fn conditioned_features(&self, session: &video::Session, frame_idx: usize, tf: &tracker::TrackerFeatures) -> Result<Tensor> {
        Ok(self.video.conditioned(session, frame_idx, tf, &self.pos72)?)
    }

    /// Raw tracker prediction for already-encoded prompts (exposed for tests).
    pub fn tracker_predict(&self, tf: &tracker::TrackerFeatures, points: &[(f32, f32, i32)], boxes: &[[f32; 4]], multimask: bool) -> Result<(Tensor, tracker::Prediction)> {
        let sparse = self.tracker.sparse(points, boxes, &self.device, self.dtype)?;
        let p = self.tracker.predict_sparse(tf, &sparse, multimask)?;
        Ok((sparse, p))
    }

    /// Instances matching `prompt` whose score (`sigmoid(logit) * sigmoid(presence)`) exceeds `threshold`.
    pub fn segment(&self, f: &ImageFeatures, prompt: &str, threshold: f32) -> Result<Vec<Instance>> {
        self.segment_with(f, prompt, &[], threshold)
    }

    /// Prompt features and validity for a text prompt plus optional exemplar boxes. `boxes`: pixel
    /// `([x0, y0, x1, y1], positive)` in the original image.
    pub fn prompt_features(&self, f: &ImageFeatures, prompt: &str, boxes: &[([f32; 4], bool)]) -> Result<(Tensor, Vec<u8>)> {
        let (ids, mut valid) = self.tokenize(prompt)?;
        let mut feats = self.text_features(&ids)?;
        if !boxes.is_empty() {
            let (oh, ow) = (f.orig.0 as f32, f.orig.1 as f32);
            let norm: Vec<[f32; 4]> = boxes.iter().map(|(b, _)| [(b[0] + b[2]) / 2.0 / ow, (b[1] + b[3]) / 2.0 / oh, (b[2] - b[0]) / ow, (b[3] - b[1]) / oh]).collect();
            let labels: Vec<u32> = boxes.iter().map(|(_, p)| *p as u32).collect();
            let geo = self.geometry.forward(&norm, &labels, &f.fpn[2], &self.pos72)?;
            valid.extend(std::iter::repeat(1u8).take(geo.dim(1)?));
            feats = Tensor::cat(&[&feats, &geo], 1)?;
        }
        Ok((feats, valid))
    }

    /// Like [`segment`](Self::segment) with exemplar boxes ("find more things like this"; positive or negative).
    /// Use the prompt `"visual"` for a purely visual query.
    pub fn segment_with(&self, f: &ImageFeatures, prompt: &str, boxes: &[([f32; 4], bool)], threshold: f32) -> Result<Vec<Instance>> {
        let (text, valid) = self.prompt_features(f, prompt, boxes)?;
        let o = self.head(&f.fpn, &text, &valid)?;
        let sig = |t: &Tensor| -> Result<Vec<f32>> { Ok(candle_nn::ops::sigmoid(&t.to_dtype(DType::F32)?)?.flatten_all()?.to_vec1::<f32>()?) };
        let presence = sig(&o.presence)?[0];
        let scores: Vec<f32> = sig(&o.logits)?.into_iter().map(|s| s * presence).collect();
        let boxes = o.boxes.to_dtype(DType::F32)?.squeeze(0)?.to_vec2::<f32>()?;
        let (oh, ow) = f.orig;
        let mut out = vec![];
        for (q, &s) in scores.iter().enumerate() {
            if s <= threshold {
                continue;
            }
            let m = candle_nn::ops::sigmoid(&o.masks.narrow(1, q, 1)?.to_dtype(DType::F32)?)?.upsample_bilinear2d(oh, ow, false)?;
            let data: Vec<u8> = m.gt(0.5f32)?.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
            let b = &boxes[q];
            out.push(Instance { score: s, bbox: [b[0] * ow as f32, b[1] * oh as f32, b[2] * ow as f32, b[3] * oh as f32], mask: crate::Mask { width: ow, height: oh, data, iou: s } });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }
}

