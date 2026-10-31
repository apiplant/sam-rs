//! HTTP server: every model family behind one JSON API, plus the web UI.
//!
//! The models live on one worker thread (CUDA state stays on the thread that created it, and the GPU is
//! single-tenant), which runs requests one at a time; the ntex handlers only parse requests and forward results.
//! Models load on first use and are kept in a small LRU. Endpoints (`GET /openapi.json` has the full description,
//! `GET /docs` renders it):
//!
//! - `GET /v1/info`, `GET /v1/models`, `POST /v1/models/{id}/load`, `DELETE /v1/models/{id}`
//! - `POST /v1/images?model=ID` (body: an image) embeds an image once and returns a session id;
//!   `POST /v1/images/{id}/segment` decodes points / boxes / SAM 3 text for it as often as you like;
//!   `DELETE /v1/images/{id}`
//! - `POST /v1/videos` creates a tracking session (SAM 2 and SAM 3: points / boxes, SAM 3 also text);
//!   `POST /v1/videos/{id}/frames` (body: a frame) returns that frame's masks; `DELETE /v1/videos/{id}`
//! - `GET /`: the web UI (built from `ui/` and embedded), `GET /health`

use std::sync::mpsc as std_mpsc;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use base64::Engine as _;
use candle_core::{DType, Device};
use image::RgbImage;
use ntex::http::StatusCode;
use ntex::util::Bytes;
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::efficientsam::{self, EfficientSam};
use crate::model::{self, Prompt, Sam, SamVariant};
use crate::sam2::{self, Sam2, Sam2Variant};
use crate::sam3::{self, text_track::TextState, video::Session, Sam3};
use crate::Mask;

// ------------------------------------------------------------------------------------------------ catalog

#[derive(Clone, Copy)]
enum Kind {
    Sam(SamVariant),
    Efficient(efficientsam::Variant),
    Sam2(Sam2Variant),
    Sam3,
}

struct Spec {
    id: String,
    name: String,
    family: &'static str,
    kind: Kind,
    size_mb: u32,
    gated: bool,
}

impl Spec {
    fn caps(&self) -> Value {
        match self.kind {
            Kind::Sam(_) | Kind::Efficient(_) => json!({ "points": true, "boxes": true, "text": false, "text_via_detector": true, "exemplars": false, "video": false }),
            Kind::Sam2(_) => json!({ "points": true, "boxes": true, "text": false, "text_via_detector": true, "exemplars": false, "video": true }),
            Kind::Sam3 => json!({ "points": true, "boxes": true, "text": true, "exemplars": true, "video": true }),
        }
    }

    /// The checkpoint files are in the cache (or, for custom weights, nothing is needed).
    fn local(&self) -> bool {
        use crate::download::cached;
        match self.kind {
            Kind::Sam(v) => {
                let (repo, file) = v.weights();
                cached(repo, file).is_some()
            }
            Kind::Efficient(v) => ["safetensors", "pt"].iter().any(|e| cached("yunyangx/EfficientSAM", &format!("{}.{e}", v.file())).is_some()),
            Kind::Sam2(v) => {
                let repo = v.repo();
                cached(&repo, "model.safetensors").is_some() && cached(&repo, "config.json").is_some()
            }
            Kind::Sam3 => cached(sam3::REPO, "model.safetensors").is_some() && cached(sam3::REPO, "tokenizer.json").is_some(),
        }
    }
}

fn catalog() -> Vec<Spec> {
    let mut v = vec![];
    let mut add = |id: &str, name: &str, family: &'static str, size_mb: u32, gated: bool| {
        let kind = parse_kind(id).expect("catalog ids parse");
        v.push(Spec { id: id.into(), name: name.into(), family, kind, size_mb, gated });
    };
    add("sam-vit-b", "SAM ViT-B", "SAM", 375, false);
    add("sam-vit-l", "SAM ViT-L", "SAM", 1250, false);
    add("sam-vit-h", "SAM ViT-H", "SAM", 2560, false);
    add("mobile-sam", "MobileSAM", "MobileSAM", 40, false);
    add("efficientsam-ti", "EfficientSAM ViT-Tiny", "EfficientSAM", 41, false);
    add("efficientsam-s", "EfficientSAM ViT-Small", "EfficientSAM", 106, false);
    for (gen, label) in [("sam2.1", "SAM 2.1"), ("sam2", "SAM 2")] {
        for (size, name, mb) in [("tiny", "Tiny", 160), ("small", "Small", 185), ("base-plus", "Base+", 323), ("large", "Large", 898)] {
            add(&format!("{gen}-hiera-{size}"), &format!("{label} Hiera {name}"), "SAM 2", mb, false);
        }
    }
    add("sam3", "SAM 3", "SAM 3", 3400, true);
    v
}

fn parse_kind(id: &str) -> Option<Kind> {
    if id == "sam3" {
        return Some(Kind::Sam3);
    }
    SamVariant::parse(id)
        .map(Kind::Sam)
        .or_else(|| efficientsam::Variant::parse(id).map(Kind::Efficient))
        .or_else(|| Sam2Variant::parse(id).map(Kind::Sam2))
}

// ------------------------------------------------------------------------------------------------ errors

/// A client mistake (400), or a missing object (404); anything else that fails is a 500.
#[derive(Debug)]
struct ApiError(StatusCode, String);

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.1)
    }
}

impl std::error::Error for ApiError {}

fn bad(msg: impl Into<String>) -> anyhow::Error {
    ApiError(StatusCode::BAD_REQUEST, msg.into()).into()
}

fn missing(msg: impl Into<String>) -> anyhow::Error {
    ApiError(StatusCode::NOT_FOUND, msg.into()).into()
}

fn fail(e: anyhow::Error) -> HttpResponse {
    let (status, msg) = match e.downcast_ref::<ApiError>() {
        Some(ApiError(s, m)) => (*s, m.clone()),
        None => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    HttpResponse::builder(status).content_type("application/json").body(json!({ "error": msg }).to_string())
}

fn ok(v: Value) -> HttpResponse {
    HttpResponse::Ok().content_type("application/json").body(v.to_string())
}

// ------------------------------------------------------------------------------------------------ state

enum Model {
    Sam(Sam),
    Efficient(EfficientSam),
    Sam2(Sam2),
    Sam3(Sam3),
}

enum Emb {
    Sam(model::Embedding),
    Efficient(efficientsam::Embedding),
    Sam2(sam2::Embedding),
    /// SAM 3: the detector's features, plus the tracker's (computed when a point prompt first needs them)
    Sam3(Box<sam3::ImageFeatures>, Option<Box<sam3::tracker::TrackerFeatures>>),
}

struct ImageSession {
    id: u64,
    model: String,
    width: usize,
    height: usize,
    emb: Emb,
    /// the image itself and, once a text prompt needed it, SAM 3's detector features for it
    img: RgbImage,
    detector: Option<Box<sam3::ImageFeatures>>,
}

#[derive(Default, Clone)]
struct VideoPrompt {
    points: Vec<(f32, f32, bool)>,
    boxes: Vec<[f32; 4]>,
}

enum Tracking {
    Sam2(Session),
    Sam3(Session),
    Text(TextState),
}

struct VideoSession {
    id: u64,
    model: String,
    frame: usize,
    prompt: VideoPrompt,
    tracking: Tracking,
}

pub struct Config {
    pub device: Device,
    pub dtype: DType,
    pub max_models: usize,
    pub allow_download: bool,
}

struct State {
    cfg: Config,
    /// least recently used first
    models: Vec<(String, Model)>,
    images: Vec<ImageSession>,
    videos: Vec<VideoSession>,
    next_id: u64,
}

const MAX_IMAGES: usize = 16;
const MAX_VIDEOS: usize = 4;

impl State {
    fn is_loaded(&self, id: &str) -> bool {
        self.models.iter().any(|(m, _)| m == id)
    }

    fn spec(id: &str) -> Result<Spec> {
        catalog().into_iter().find(|s| s.id == id).ok_or_else(|| missing(format!("unknown model {id:?} (GET /v1/models lists them)")))
    }

    /// Loads `id` if needed (downloading it unless disabled) and marks it most recently used.
    fn ensure(&mut self, id: &str, weights: Option<&str>) -> Result<()> {
        if let Some(i) = self.models.iter().position(|(m, _)| m == id) {
            if weights.is_none() {
                let m = self.models.remove(i);
                self.models.push(m);
                return Ok(());
            }
            self.models.remove(i); // new weights requested: reload
        }
        let spec = Self::spec(id)?;
        if weights.is_none() && !spec.local() && !self.cfg.allow_download {
            return Err(bad(format!("{id} is not downloaded and downloads are disabled (start the server without --no-download)")));
        }
        let t = Instant::now();
        let (device, dtype) = (self.cfg.device.clone(), self.cfg.dtype);
        let beside = |w: &std::path::Path, name: &str| -> Result<std::path::PathBuf> {
            let p = w.parent().unwrap_or(std::path::Path::new(".")).join(name);
            p.is_file().then_some(p.clone()).ok_or_else(|| bad(format!("{} must sit next to the weights", p.display())))
        };
        let w = weights.map(std::path::PathBuf::from);
        if let Some(w) = &w {
            if !w.is_file() {
                return Err(bad(format!("weights file {} not found", w.display())));
            }
        }
        let m = match (spec.kind, &w) {
            (Kind::Sam(v), Some(w)) => Model::Sam(Sam::from_file(w, v, &device, dtype)?),
            (Kind::Sam(v), None) => Model::Sam(Sam::from_pretrained(v, &device, dtype)?),
            (Kind::Efficient(v), Some(w)) => Model::Efficient(EfficientSam::from_file(v, w, &device, dtype)?),
            (Kind::Efficient(v), None) => Model::Efficient(EfficientSam::from_pretrained(v, &device, dtype)?),
            (Kind::Sam2(_), Some(w)) => Model::Sam2(Sam2::from_dir(&beside(w, "config.json")?, w, &device, dtype)?),
            (Kind::Sam2(v), None) => Model::Sam2(Sam2::from_pretrained(v, &device, dtype)?),
            (Kind::Sam3, Some(w)) => Model::Sam3(Sam3::from_file(w, &beside(w, "tokenizer.json")?, &device, dtype)?),
            (Kind::Sam3, None) => Model::Sam3(Sam3::from_pretrained(&device, dtype).context("loading SAM 3 (the weights are gated: set HF_TOKEN, see the README)")?),
        };
        eprintln!("loaded {id} in {:.2?}", t.elapsed());
        self.models.push((id.to_string(), m));
        while self.models.len() > self.cfg.max_models.max(1) {
            let (old, _) = self.models.remove(0);
            eprintln!("unloaded {old} (over --max-models)");
            self.drop_sessions_of(&old);
        }
        Ok(())
    }

    fn drop_sessions_of(&mut self, id: &str) {
        self.images.retain(|s| s.model != id);
        self.videos.retain(|s| s.model != id);
    }

    fn model(&self, id: &str) -> Result<&Model> {
        self.models.iter().find(|(m, _)| m == id).map(|(_, m)| m).ok_or_else(|| missing(format!("model {id} is no longer loaded")))
    }

    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn info(&self) -> Value {
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "device": if self.cfg.device.is_cuda() { "cuda" } else { "cpu" },
            "dtype": format!("{:?}", self.cfg.dtype).to_lowercase(),
            "cuda_build": cfg!(feature = "cuda"),
            "bf16_kernels": crate::cpu_gemm::enabled(),
            "hf_token": crate::download::hf_token().is_some(),
            "allow_download": self.cfg.allow_download,
            "max_models": self.cfg.max_models,
            "cache_dir": crate::download::cache_dir().map(|p| p.display().to_string()),
            "loaded": self.models.iter().map(|(m, _)| m.clone()).collect::<Vec<_>>(),
            "images": self.images.len(),
            "videos": self.videos.len(),
        })
    }

    fn models_json(&self) -> Value {
        Value::Array(
            catalog()
                .iter()
                .map(|s| {
                    json!({
                        "id": s.id, "name": s.name, "family": s.family, "size_mb": s.size_mb, "gated": s.gated,
                        "local": s.local(), "loaded": self.is_loaded(&s.id), "capabilities": s.caps(),
                    })
                })
                .collect(),
        )
    }

    // ---- images

    fn embed(&mut self, model: &str, bytes: &[u8]) -> Result<Value> {
        self.ensure(model, None)?;
        let img = decode(bytes)?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let t = Instant::now();
        let emb = match self.model(model)? {
            Model::Sam(m) => Emb::Sam(m.embed(&img)?),
            Model::Efficient(m) => Emb::Efficient(m.embed(&img)?),
            Model::Sam2(m) => Emb::Sam2(m.embed(&img)?),
            Model::Sam3(m) => Emb::Sam3(Box::new(m.encode_image(&img)?), None),
        };
        sync(self.model(model)?)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let id = self.new_id();
        self.images.push(ImageSession { id, model: model.to_string(), width: w, height: h, emb, img, detector: None });
        if self.images.len() > MAX_IMAGES {
            self.images.remove(0);
        }
        Ok(json!({ "id": id, "model": model, "width": w, "height": h, "encode_ms": ms }))
    }

    fn segment(&mut self, id: u64, req: SegmentRequest) -> Result<Value> {
        let pos = self.images.iter().position(|s| s.id == id).ok_or_else(|| missing(format!("image session {id} not found (it may have been evicted)")))?;
        let model_id = self.images[pos].model.clone();
        self.ensure(&model_id, None)?;
        let pos = self.images.iter().position(|s| s.id == id).ok_or_else(|| missing("image session was dropped"))?;
        let (w, h) = (self.images[pos].width, self.images[pos].height);
        let text_only = req.text.as_deref().is_some_and(|t| !t.trim().is_empty()) && req.points.is_empty() && req.boxes.is_empty() && req.neg_boxes.is_empty();
        if text_only && model_id != "sam3" {
            return self.segment_by_text(id, &model_id, req.text.as_deref().unwrap_or("").trim(), req.threshold.unwrap_or(0.3));
        }
        let points: Vec<(f32, f32, bool)> = req.points.iter().map(|p| (p.x, p.y, p.label.unwrap_or(1) != 0)).collect();
        let boxes = req.boxes.clone();
        let multimask = req.multimask.unwrap_or(true);
        let t = Instant::now();
        let entries: Vec<Entry>;
        let mut object_score = None;
        let mut kind = "masks";
        let State { models, images, .. } = self;
        let m = models.iter().find(|(m, _)| *m == model_id).map(|(_, m)| m).ok_or_else(|| missing("model unloaded"))?;
        match (m, &mut images[pos].emb) {
            (Model::Sam(m), Emb::Sam(e)) => {
                let prompt = if !points.is_empty() {
                    Prompt::Points(points)
                } else if let Some(b) = boxes.first() {
                    Prompt::Box(*b)
                } else {
                    return Err(bad("give points or a box"));
                };
                entries = m.predict(e, &prompt, multimask)?.into_iter().map(|k| Entry::mask(k, None)).collect();
            }
            (Model::Efficient(m), Emb::Efficient(e)) => {
                if points.is_empty() && boxes.is_empty() {
                    return Err(bad("give points and/or boxes"));
                }
                entries = m.predict(e, &points, &boxes)?.into_iter().map(|k| Entry::mask(k, None)).collect();
            }
            (Model::Sam2(m), Emb::Sam2(e)) => {
                if points.is_empty() && boxes.is_empty() {
                    return Err(bad("give points and/or boxes"));
                }
                let (masks, score) = m.predict(e, &points, &boxes, multimask)?;
                object_score = Some(score);
                entries = masks.into_iter().map(|k| Entry::mask(k, None)).collect();
            }
            (Model::Sam3(m), Emb::Sam3(f, tf)) => {
                let text = req.text.as_deref().map(str::trim).unwrap_or("");
                if !points.is_empty() || req.tracker.unwrap_or(false) {
                    if points.is_empty() && boxes.is_empty() {
                        return Err(bad("tracker mode needs points and/or boxes"));
                    }
                    if tf.is_none() {
                        *tf = Some(Box::new(m.tracker_features(f)?));
                    }
                    let (masks, score) = m.segment_points(f, tf.as_ref().expect("just set"), &points, &boxes, multimask)?;
                    object_score = Some(score);
                    entries = masks.into_iter().map(|k| Entry::mask(k, None)).collect();
                } else {
                    let exemplars: Vec<([f32; 4], bool)> = boxes.iter().map(|b| (*b, true)).chain(req.neg_boxes.iter().map(|b| (*b, false))).collect();
                    if text.is_empty() && exemplars.is_empty() {
                        return Err(bad("SAM 3 needs text and/or exemplar boxes (or points for the tracker)"));
                    }
                    let prompt = if text.is_empty() { "visual" } else { text };
                    kind = "instances";
                    entries = m
                        .segment_with(f, prompt, &exemplars, req.threshold.unwrap_or(0.3))?
                        .into_iter()
                        .map(|i| Entry { score: i.score, mask: i.mask, id: None, present: true })
                        .collect();
                }
            }
            _ => return Err(bad("the image session does not match the loaded model")),
        }
        sync(m)?;
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(json!({
            "kind": kind, "model": model_id, "width": w, "height": h,
            "object_score": object_score, "decode_ms": decode_ms,
            "results": entries.iter().map(Entry::json).collect::<Vec<_>>(),
        }))
    }

    /// Text prompts for models that have no text pathway (SAM, MobileSAM, EfficientSAM, SAM 2): SAM 3 finds the
    /// instances of the phrase, and `model_id` segments inside each detected box.
    fn segment_by_text(&mut self, id: u64, model_id: &str, text: &str, threshold: f32) -> Result<Value> {
        if self.cfg.max_models < 2 {
            return Err(bad("text prompts for this model use SAM 3 as the detector, which needs --max-models 2 or more"));
        }
        self.ensure("sam3", None).context("text prompts for this model use SAM 3 to find the objects first")?;
        self.ensure(model_id, None)?; // SAM 3 may have pushed the other model out; this also makes it the most recent
        let pos = self.images.iter().position(|s| s.id == id).ok_or_else(|| missing("image session was dropped (the model was reloaded)"))?;
        let (w, h) = (self.images[pos].width, self.images[pos].height);
        let t = Instant::now();
        let State { models, images, .. } = self;
        let find = |name: &str| models.iter().find(|(m, _)| m == name).map(|(_, m)| m);
        let (Some(Model::Sam3(det)), Some(model)) = (find("sam3"), find(model_id)) else { return Err(missing("a model was unloaded while the request ran")) };
        let sess = &mut images[pos];
        if sess.detector.is_none() {
            sess.detector = Some(Box::new(det.encode_image(&sess.img)?));
        }
        let found = det.segment(sess.detector.as_ref().expect("just set"), text, threshold)?;
        let mut entries = vec![];
        for inst in found.into_iter().take(32) {
            let b = inst.bbox;
            let masks = match (model, &sess.emb) {
                (Model::Sam(m), Emb::Sam(e)) => m.predict(e, &Prompt::Box(b), false)?,
                (Model::Efficient(m), Emb::Efficient(e)) => m.predict(e, &[], &[b])?,
                (Model::Sam2(m), Emb::Sam2(e)) => m.predict(e, &[], &[b], false)?.0,
                _ => return Err(bad("the image session does not match the loaded model")),
            };
            if let Some(best) = masks.into_iter().max_by(|a, b| a.iou.total_cmp(&b.iou)) {
                entries.push(Entry { score: inst.score, mask: best, id: None, present: true });
            }
        }
        sync(model)?;
        Ok(json!({
            "kind": "instances", "model": model_id, "detector": "sam3", "width": w, "height": h,
            "object_score": null, "decode_ms": t.elapsed().as_secs_f64() * 1e3,
            "results": entries.iter().map(Entry::json).collect::<Vec<_>>(),
        }))
    }

    // ---- video

    fn start_video(&mut self, req: VideoRequest) -> Result<Value> {
        let spec = Self::spec(&req.model)?;
        if !spec.caps()["video"].as_bool().unwrap_or(false) {
            return Err(bad(format!("{} has no video tracking", req.model)));
        }
        self.ensure(&req.model, None)?;
        let prompt = VideoPrompt { points: req.points.iter().map(|p| (p.x, p.y, p.label.unwrap_or(1) != 0)).collect(), boxes: req.boxes.clone() };
        let text = req.text.as_deref().map(str::trim).unwrap_or("");
        let tracking = match spec.kind {
            Kind::Sam2(_) => {
                if prompt.points.is_empty() && prompt.boxes.is_empty() {
                    return Err(bad("give points and/or a box for the object to track"));
                }
                Tracking::Sam2(Session::new(None))
            }
            Kind::Sam3 if !text.is_empty() && prompt.points.is_empty() && prompt.boxes.is_empty() => {
                Tracking::Text(TextState::new(text, req.threshold.unwrap_or(0.3)).detect_every(req.detect_every.unwrap_or(5)))
            }
            Kind::Sam3 => {
                if prompt.points.is_empty() && prompt.boxes.is_empty() {
                    return Err(bad("give text, or points and/or a box for the object to track"));
                }
                Tracking::Sam3(Session::new(None))
            }
            _ => unreachable!("video capability checked"),
        };
        let id = self.new_id();
        self.videos.push(VideoSession { id, model: req.model.clone(), frame: 0, prompt, tracking });
        if self.videos.len() > MAX_VIDEOS {
            self.videos.remove(0);
        }
        Ok(json!({ "id": id, "model": req.model }))
    }

    fn video_frame(&mut self, id: u64, bytes: &[u8]) -> Result<Value> {
        let pos = self.videos.iter().position(|s| s.id == id).ok_or_else(|| missing(format!("video session {id} not found")))?;
        let model_id = self.videos[pos].model.clone();
        self.ensure(&model_id, None)?;
        let pos = self.videos.iter().position(|s| s.id == id).ok_or_else(|| missing("video session was dropped"))?;
        let img = decode(bytes)?;
        let orig = (img.height() as usize, img.width() as usize);
        let t = Instant::now();
        let State { models, videos, .. } = self;
        let m = models.iter().find(|(m, _)| *m == model_id).map(|(_, m)| m).ok_or_else(|| missing("model unloaded"))?;
        let v = &mut videos[pos];
        let first = v.frame == 0;
        let entries: Vec<Entry> = match (m, &mut v.tracking) {
            (Model::Sam2(m), Tracking::Sam2(session)) => {
                let e = m.embed(&img)?;
                if first {
                    session.add_prompt(0, m.prompt_points(orig, &v.prompt.points, &v.prompt.boxes));
                }
                let r = m.track_frame(session, &e)?;
                vec![Entry { score: r.object_score, mask: m.frame_mask(&r, orig)?, id: Some(0), present: r.object_score > 0.0 }]
            }
            (Model::Sam3(m), Tracking::Sam3(session)) => {
                let tf = m.encode_for_tracking(&img)?;
                if first {
                    let (sx, sy) = (1008.0 / orig.1 as f32, 1008.0 / orig.0 as f32);
                    let mut p = sam3::video::FramePrompt::default();
                    for &(x, y, fg) in &v.prompt.points {
                        p.points.push((x * sx, y * sy, fg as i32));
                    }
                    for b in &v.prompt.boxes {
                        p.points.push((b[0] * sx, b[1] * sy, 2));
                        p.points.push((b[2] * sx, b[3] * sy, 3));
                    }
                    session.add_prompt(0, p);
                }
                let r = m.track_frame(session, &tf)?;
                vec![Entry { score: r.object_score, mask: m.frame_mask(&r, orig)?, id: Some(0), present: r.object_score > 0.0 }]
            }
            (Model::Sam3(m), Tracking::Text(state)) => state
                .step(m, &img)?
                .into_iter()
                .map(|o| Entry { score: o.detection_score.unwrap_or(o.object_score), mask: o.mask, id: Some(o.id), present: true })
                .collect(),
            _ => return Err(bad("the video session does not match the loaded model")),
        };
        let frame = v.frame;
        v.frame += 1;
        sync(m)?;
        Ok(json!({
            "frame": frame, "width": orig.1, "height": orig.0, "elapsed_ms": t.elapsed().as_secs_f64() * 1e3,
            "objects": entries.iter().map(Entry::json).collect::<Vec<_>>(),
        }))
    }
}

fn sync(m: &Model) -> Result<()> {
    match m {
        Model::Sam(m) => m.sync(),
        Model::Efficient(m) => m.sync(),
        Model::Sam2(m) => m.sync(),
        Model::Sam3(m) => m.sync(),
    }
}

fn decode(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.is_empty() {
        return Err(bad("the body must be an image (png, jpeg, webp or bmp)"));
    }
    image::load_from_memory(bytes).map(|i| i.to_rgb8()).map_err(|e| bad(format!("could not decode the image: {e}")))
}

// ------------------------------------------------------------------------------------------------ results

struct Entry {
    score: f32,
    mask: Mask,
    id: Option<usize>,
    present: bool,
}

impl Entry {
    fn mask(mask: Mask, id: Option<usize>) -> Self {
        Self { score: mask.iou, mask, id, present: true }
    }

    fn json(&self) -> Value {
        let m = &self.mask;
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (usize::MAX, usize::MAX, 0usize, 0usize, 0usize);
        for (y, row) in m.data.chunks(m.width.max(1)).enumerate() {
            let (mut first, mut last, mut n) = (usize::MAX, 0, 0);
            for (x, &v) in row.iter().enumerate() {
                if v > 0 {
                    n += 1;
                    if first == usize::MAX {
                        first = x;
                    }
                    last = x;
                }
            }
            if n > 0 {
                area += n;
                x0 = x0.min(first);
                x1 = x1.max(last);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
        let bbox = if area == 0 { Value::Null } else { json!([x0, y0, x1 + 1, y1 + 1]) };
        json!({ "id": self.id, "score": self.score, "present": self.present, "area": area, "bbox": bbox, "png": mask_png(m) })
    }
}

/// Base64 of an 8-bit grayscale PNG (0 / 255) of the mask.
fn mask_png(m: &Mask) -> String {
    use image::codecs::png::{CompressionType, FilterType, PngEncoder};
    use image::ImageEncoder;
    let gray: Vec<u8> = m.data.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect();
    let mut out = Vec::new();
    PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::NoFilter)
        .write_image(&gray, m.width as u32, m.height as u32, image::ExtendedColorType::L8)
        .expect("encoding to memory cannot fail");
    base64::engine::general_purpose::STANDARD.encode(out)
}

// ------------------------------------------------------------------------------------------------ requests

#[derive(Deserialize, Clone)]
struct PointIn {
    x: f32,
    y: f32,
    /// 1 foreground (default), 0 background
    label: Option<u8>,
}

#[derive(Deserialize, Default)]
struct SegmentRequest {
    #[serde(default)]
    points: Vec<PointIn>,
    #[serde(default)]
    boxes: Vec<[f32; 4]>,
    #[serde(default)]
    neg_boxes: Vec<[f32; 4]>,
    text: Option<String>,
    threshold: Option<f32>,
    multimask: Option<bool>,
    tracker: Option<bool>,
}

#[derive(Deserialize)]
struct VideoRequest {
    model: String,
    #[serde(default)]
    points: Vec<PointIn>,
    #[serde(default)]
    boxes: Vec<[f32; 4]>,
    text: Option<String>,
    threshold: Option<f32>,
    detect_every: Option<usize>,
}

#[derive(Deserialize)]
struct LoadRequest {
    weights: Option<String>,
}

// ------------------------------------------------------------------------------------------------ worker

type Task = Box<dyn FnOnce(&mut State) + Send>;

#[derive(Clone)]
struct Engine(std_mpsc::Sender<Task>);

impl Engine {
    /// Runs `f` on the worker thread and waits for its result.
    async fn call<T: Send + 'static>(&self, f: impl FnOnce(&mut State) -> Result<T> + Send + 'static) -> Result<T> {
        let (tx, rx) = oneshot::channel();
        self.0
            .send(Box::new(move |s| {
                // a panic in a model must not take the worker down with it
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(s))).unwrap_or_else(|_| Err(anyhow!("the model panicked")));
                let _ = tx.send(r);
            }))
            .map_err(|_| anyhow!("the engine thread stopped"))?;
        rx.await.map_err(|_| anyhow!("the engine thread stopped"))?
    }
}

fn path_id(req: &HttpRequest) -> Result<u64> {
    req.match_info().get("id").and_then(|s| s.parse().ok()).ok_or_else(|| bad("bad id"))
}

fn query(req: &HttpRequest, key: &str) -> Option<String> {
    req.query_string().split('&').filter_map(|kv| kv.split_once('=')).find(|(k, _)| *k == key).map(|(_, v)| percent_decode(v))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn json_body<T: for<'a> Deserialize<'a> + Default>(body: &Bytes) -> Result<T> {
    if body.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(T::default());
    }
    serde_json::from_slice(body).map_err(|e| bad(format!("bad JSON: {e}")))
}

async fn reply(r: Result<Value>) -> HttpResponse {
    match r {
        Ok(v) => ok(v),
        Err(e) => fail(e),
    }
}

/// The web UI's files (`ui/dist`) as `(path, bytes)`, embedded by build.rs; empty when it was not built.
static UI: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/ui.rs"));

const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>sam-rs API</title>
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1"></script>
  </body>
</html>
"#;

fn openapi() -> String {
    include_str!("openapi.json").replacen("\"VERSION\"", &format!("{:?}", env!("CARGO_PKG_VERSION")), 1)
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

async fn ui(req: HttpRequest) -> HttpResponse {
    let path = req.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match UI.iter().find(|(p, _)| *p == path) {
        Some((p, bytes)) => {
            let cache = if p.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
            HttpResponse::Ok().content_type(content_type(p)).header("cache-control", cache).body(Bytes::from_static(bytes))
        }
        None if UI.is_empty() && path == "index.html" => HttpResponse::NotFound()
            .content_type("text/plain; charset=utf-8")
            .body("this sam-rs was built without its web UI: run `pnpm install && pnpm build` in ui/, then rebuild\n"),
        None => HttpResponse::NotFound().content_type("text/plain; charset=utf-8").body("not found\n"),
    }
}

/// How `sam-rs serve` is configured.
pub struct Options {
    pub addr: String,
    pub cpu: bool,
    pub dtype: String,
    pub max_models: usize,
    pub allow_download: bool,
    /// Models to load before the server starts listening.
    pub preload: Vec<String>,
}

/// Starts the worker thread and serves until the process is stopped (Ctrl-C).
pub fn run(opts: Options) -> Result<()> {
    let (tx, rx) = std_mpsc::channel::<Task>();
    let (ready_tx, ready) = std_mpsc::channel::<Result<String>>();
    let o = (opts.cpu, opts.dtype.clone(), opts.max_models, opts.allow_download, opts.preload.clone());
    std::thread::Builder::new().name("sam-rs-engine".into()).spawn(move || {
        let (cpu, dtype, max_models, allow_download, preload) = o;
        let setup = || -> Result<State> {
            let device = if cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
            let dtype = match (device.is_cuda(), dtype.as_str()) {
                (true, "f16") => DType::F16,
                _ => DType::F32,
            };
            let mut s = State { cfg: Config { device, dtype, max_models, allow_download }, models: vec![], images: vec![], videos: vec![], next_id: 0 };
            for m in &preload {
                s.ensure(m, None).with_context(|| format!("preloading {m}"))?;
            }
            Ok(s)
        };
        match setup() {
            Ok(mut state) => {
                let _ = ready_tx.send(Ok(format!("{:?} {:?}", state.cfg.device.location(), state.cfg.dtype)));
                for task in rx {
                    task(&mut state);
                }
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    })?;
    let what = ready.recv().map_err(|_| anyhow!("engine thread died while starting"))??;
    eprintln!("engine ready on {what}");
    serve(Engine(tx), opts.addr)?;
    Ok(())
}

#[ntex::main(name = "sam-rs", signals = true)]
async fn serve(engine: Engine, addr: String) -> std::io::Result<()> {
    eprintln!("listening on http://{addr}  (web UI at /, API docs at /docs)");
    HttpServer::new(async move |_| {
        let e = engine.clone();
        // images and video frames are uploaded whole
        let cfg = web::WebAppConfig::new().set_state(web::types::PayloadConfig::new(64 << 20));
        let (e_info, e_models, e_load, e_unload, e_embed, e_seg, e_img_del, e_vid, e_frame, e_vid_del) =
            (e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone());
        App::new()
            .with_config(cfg)
            .service(web::resource("/health").to(async || "ok\n"))
            .service(web::resource("/openapi.json").to(async || HttpResponse::Ok().content_type("application/json").body(openapi())))
            .service(web::resource("/docs").to(async || HttpResponse::Ok().content_type("text/html; charset=utf-8").body(DOCS_HTML)))
            .service(web::resource("/v1/info").route(web::get().to(move || {
                let e = e_info.clone();
                async move { reply(e.call(|s| Ok(s.info())).await).await }
            })))
            .service(web::resource("/v1/models").route(web::get().to(move || {
                let e = e_models.clone();
                async move { reply(e.call(|s| Ok(s.models_json())).await).await }
            })))
            .service(web::resource("/v1/models/{id}/load").route(web::post().to(move |req: HttpRequest, body: Bytes| {
                let e = e_load.clone();
                async move {
                    let id = req.match_info().get("id").unwrap_or_default().to_string();
                    let r = async {
                        let l: LoadRequest = if body.is_empty() { LoadRequest { weights: None } } else { serde_json::from_slice(&body).map_err(|x| bad(format!("bad JSON: {x}")))? };
                        let t = Instant::now();
                        e.call(move |s| {
                            s.ensure(&id, l.weights.as_deref())?;
                            Ok(json!({ "id": id, "loaded": true, "load_ms": t.elapsed().as_secs_f64() * 1e3 }))
                        })
                        .await
                    };
                    reply(r.await).await
                }
            })))
            .service(web::resource("/v1/models/{id}").route(web::delete().to(move |req: HttpRequest| {
                let e = e_unload.clone();
                async move {
                    let id = req.match_info().get("id").unwrap_or_default().to_string();
                    reply(
                        e.call(move |s| {
                            let before = s.models.len();
                            s.models.retain(|(m, _)| *m != id);
                            s.drop_sessions_of(&id);
                            Ok(json!({ "id": id, "unloaded": s.models.len() < before }))
                        })
                        .await,
                    )
                    .await
                }
            })))
            .service(web::resource("/v1/images").route(web::post().to(move |req: HttpRequest, body: Bytes| {
                let e = e_embed.clone();
                async move {
                    let Some(model) = query(&req, "model") else { return fail(bad("need ?model=ID (GET /v1/models lists them)")) };
                    reply(e.call(move |s| s.embed(&model, &body)).await).await
                }
            })))
            .service(web::resource("/v1/images/{id}/segment").route(web::post().to(move |req: HttpRequest, body: Bytes| {
                let e = e_seg.clone();
                async move {
                    let r = async {
                        let id = path_id(&req)?;
                        let sreq: SegmentRequest = json_body(&body)?;
                        e.call(move |s| s.segment(id, sreq)).await
                    };
                    reply(r.await).await
                }
            })))
            .service(web::resource("/v1/images/{id}").route(web::delete().to(move |req: HttpRequest| {
                let e = e_img_del.clone();
                async move {
                    let r = async {
                        let id = path_id(&req)?;
                        e.call(move |s| {
                            let n = s.images.len();
                            s.images.retain(|i| i.id != id);
                            Ok(json!({ "deleted": s.images.len() < n }))
                        })
                        .await
                    };
                    reply(r.await).await
                }
            })))
            .service(web::resource("/v1/videos").route(web::post().to(move |body: Bytes| {
                let e = e_vid.clone();
                async move {
                    let r = async {
                        let vreq: VideoRequest = serde_json::from_slice(&body).map_err(|x| bad(format!("bad JSON: {x}")))?;
                        e.call(move |s| s.start_video(vreq)).await
                    };
                    reply(r.await).await
                }
            })))
            .service(web::resource("/v1/videos/{id}/frames").route(web::post().to(move |req: HttpRequest, body: Bytes| {
                let e = e_frame.clone();
                async move {
                    let r = async {
                        let id = path_id(&req)?;
                        e.call(move |s| s.video_frame(id, &body)).await
                    };
                    reply(r.await).await
                }
            })))
            .service(web::resource("/v1/videos/{id}").route(web::delete().to(move |req: HttpRequest| {
                let e = e_vid_del.clone();
                async move {
                    let r = async {
                        let id = path_id(&req)?;
                        e.call(move |s| {
                            let n = s.videos.len();
                            s.videos.retain(|v| v.id != id);
                            Ok(json!({ "deleted": s.videos.len() < n }))
                        })
                        .await
                    };
                    reply(r.await).await
                }
            })))
            .default_service(web::get().to(ui))
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("sam-rs"))?
    .workers(1)
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_parse() {
        let c = catalog();
        let mut ids: Vec<_> = c.iter().map(|s| s.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), c.len());
        assert!(c.iter().any(|s| matches!(s.kind, Kind::Sam3)));
        assert!(c.iter().filter(|s| matches!(s.kind, Kind::Sam2(_))).count() == 8);
    }

    #[test]
    fn decodes_query_values() {
        assert_eq!(percent_decode("a%20b+c%2Fd"), "a b c/d");
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn mask_bbox_and_area() {
        let m = Mask { width: 4, height: 3, data: vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], iou: 0.5 };
        let e = Entry::mask(m, None).json();
        assert_eq!(e["area"], 3);
        assert_eq!(e["bbox"], json!([1, 1, 3, 3]));
    }
}
