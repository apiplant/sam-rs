//! SAM 3 video tracker: SAM 2-style object tracking. Each frame is segmented with the image tracker's decoder after
//! its features have been conditioned, by memory attention, on the memories (and object pointers) of earlier frames.

use std::collections::{BTreeMap, BTreeSet};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::nn::{ln, sdpa};
use super::tracker::{Ff, Prediction, Tracker, TrackerFeatures};
use super::vision::{rope_tables_hd, sine_pos_nf};
use crate::kernels::layer_norm_2d;

const D: usize = 256;
const MEM: usize = 64;
const NUM_MASKMEM: usize = 7;
const MAX_POINTERS: usize = 16;
const NO_OBJ_SCORE: f64 = -1024.0;

fn lnorm2d(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    layer_norm_2d(x, &w.to_dtype(x.dtype())?, &b.to_dtype(x.dtype())?, 1e-6)
}

// ---------------------------------------------------------------------------------------- memory encoder

struct DownLayer {
    conv: Conv2d,
    ln_w: Tensor,
    ln_b: Tensor,
}

struct Fuser {
    dw: Conv2d,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Linear,
    pw2: Linear,
    scale: Tensor,
}

struct MemoryEncoder {
    down: Vec<DownLayer>,
    final_conv: Conv2d,
    feat_proj: Conv2d,
    fusers: Vec<Fuser>,
    proj: Conv2d,
}

impl MemoryEncoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        let md = vb.pp("mask_downsampler");
        let c3 = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };
        let chans = [1usize, 4, 16, 64, 256];
        let down = (0..4)
            .map(|i| {
                let l = md.pp(format!("layers.{i}"));
                Ok(DownLayer { conv: conv2d(chans[i], chans[i + 1], 3, c3, l.pp("conv"))?, ln_w: l.get(chans[i + 1], "layer_norm.weight")?, ln_b: l.get(chans[i + 1], "layer_norm.bias")? })
            })
            .collect::<Result<_>>()?;
        let dwc = Conv2dConfig { padding: 3, groups: D, ..Default::default() };
        let fusers = (0..2)
            .map(|i| {
                let l = vb.pp(format!("memory_fuser.layers.{i}"));
                Ok(Fuser {
                    dw: conv2d(D, D, 7, dwc, l.pp("depthwise_conv"))?,
                    ln_w: l.get(D, "layer_norm.weight")?,
                    ln_b: l.get(D, "layer_norm.bias")?,
                    pw1: linear(D, 1024, l.pp("pointwise_conv1"))?,
                    pw2: linear(1024, D, l.pp("pointwise_conv2"))?,
                    scale: l.get(D, "scale")?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            down,
            final_conv: conv2d(D, D, 1, Default::default(), md.pp("final_conv"))?,
            feat_proj: conv2d(D, D, 1, Default::default(), vb.pp("feature_projection"))?,
            fusers,
            proj: conv2d(D, MEM, 1, Default::default(), vb.pp("projection"))?,
        })
    }

    /// `pix`: `[1, 256, 72, 72]`, `mask`: `[1, 1, 1152, 1152]` (already scaled) -> `([1, 64, 72, 72], positions)`.
    fn forward(&self, pix: &Tensor, mask: &Tensor, side: usize) -> Result<(Tensor, Tensor)> {
        let mut m = mask.clone();
        for l in &self.down {
            m = lnorm2d(&l.conv.forward(&m)?, &l.ln_w, &l.ln_b)?.gelu_erf()?;
        }
        let m = self.final_conv.forward(&m)?;
        let mut x = (self.feat_proj.forward(pix)? + m)?;
        for f in &self.fusers {
            let y = lnorm2d(&f.dw.forward(&x)?, &f.ln_w, &f.ln_b)?.permute((0, 2, 3, 1))?.contiguous()?;
            let y = f.pw2.forward(&f.pw1.forward(&y)?.gelu_erf()?)?;
            let y = y.broadcast_mul(&f.scale.to_dtype(y.dtype())?)?.permute((0, 3, 1, 2))?.contiguous()?;
            x = (&x + y)?;
        }
        let feat = self.proj.forward(&x)?;
        let pos = sine_pos_nf(side, side, MEM / 2, feat.device(), feat.dtype())?;
        Ok((feat, pos))
    }
}

// ---------------------------------------------------------------------------------------- memory attention

/// `x * cos + rotate_pairwise(x) * sin` in f32 over `[1, L, 256]`; `cos`/`sin`: `[L, 256]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let (b, l, d) = x.dims3()?;
    let x2 = x.reshape((b, l, d / 2, 2))?;
    let rot = Tensor::cat(&[&x2.narrow(3, 1, 1)?.neg()?, &x2.narrow(3, 0, 1)?], 3)?.reshape((b, l, d))?;
    (x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?)?.to_dtype(dt)
}

struct RopeAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl RopeAttn {
    fn new(kv_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { q: linear(D, D, vb.pp("q_proj"))?, k: linear(kv_dim, D, vb.pp("k_proj"))?, v: linear(kv_dim, D, vb.pp("v_proj"))?, o: linear(D, D, vb.pp("o_proj"))? })
    }

    /// `q`: `[1, N, 256]` (rotated with `cos/sin`); `k`, `v` inputs `[1, M, kv]`. The first `M - n_exclude` keys are
    /// rotated with the tables tiled to their length; the last `n_exclude` (object pointers) are not.
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, cos: &Tensor, sin: &Tensor, n_exclude: usize) -> Result<Tensor> {
        let q = rope(&self.q.forward(q)?, cos, sin)?;
        let k = self.k.forward(k)?;
        let m = k.dim(1)?;
        let rot = m - n_exclude;
        let k = if rot == 0 {
            k
        } else {
            let reps = rot / cos.dim(0)?;
            let (ck, sk) = (Tensor::cat(&vec![cos; reps], 0)?, Tensor::cat(&vec![sin; reps], 0)?);
            let kr = rope(&k.narrow(1, 0, rot)?, &ck, &sk)?;
            if n_exclude == 0 { kr } else { Tensor::cat(&[&kr, &k.narrow(1, rot, n_exclude)?], 1)? }
        };
        let v = self.v.forward(v)?;
        let o = sdpa(&q.unsqueeze(1)?, &k.unsqueeze(1)?, &v.unsqueeze(1)?, None)?.squeeze(1)?;
        self.o.forward(&o)
    }
}

struct MemLayer {
    self_attn: RopeAttn,
    cross: RopeAttn,
    l1: Linear,
    l2: Linear,
    n1: LayerNorm,
    n2: LayerNorm,
    n3: LayerNorm,
}

struct MemoryAttention {
    layers: Vec<MemLayer>,
    out_ln: LayerNorm,
    cos: Tensor,
    sin: Tensor,
}

impl MemoryAttention {
    fn new(vb: VarBuilder, side: usize, device: &Device) -> Result<Self> {
        let layers = (0..4)
            .map(|i| {
                let l = vb.pp(format!("layers.{i}"));
                Ok(MemLayer {
                    self_attn: RopeAttn::new(D, l.pp("self_attn"))?,
                    cross: RopeAttn::new(MEM, l.pp("cross_attn_image"))?,
                    l1: linear(D, 2048, l.pp("linear1"))?,
                    l2: linear(2048, D, l.pp("linear2"))?,
                    n1: ln(D, 1e-5, l.pp("layer_norm1"))?,
                    n2: ln(D, 1e-5, l.pp("layer_norm2"))?,
                    n3: ln(D, 1e-5, l.pp("layer_norm3"))?,
                })
            })
            .collect::<Result<_>>()?;
        let (cos, sin) = rope_tables_hd(side, 1.0, D, 10000.0, device)?;
        Ok(Self { layers, out_ln: ln(D, 1e-5, vb.pp("layer_norm"))?, cos, sin })
    }

    /// `vision`, `vpos`: `[1, 5184, 256]`; `mem`, `mem_pos`: `[1, M, 64]` with the last `n_ptr` tokens object pointers.
    fn forward(&self, vision: &Tensor, vpos: &Tensor, mem: &Tensor, mem_pos: &Tensor, n_ptr: usize) -> Result<Tensor> {
        let mut x = (vision + (vpos * 0.1)?)?;
        for l in &self.layers {
            let y = l.n1.forward(&x)?;
            x = (&x + l.self_attn.forward(&y, &y, &y, &self.cos, &self.sin, 0)?)?;
            let y = l.n2.forward(&x)?;
            x = (&x + l.cross.forward(&y, &(mem + mem_pos)?, mem, &self.cos, &self.sin, n_ptr)?)?;
            let y = l.n3.forward(&x)?;
            x = (&x + l.l2.forward(&l.l1.forward(&y)?.relu()?)?)?;
        }
        self.out_ln.forward(&x)
    }
}

// ---------------------------------------------------------------------------------------- tracker + session

/// What a stored frame keeps for later frames.
struct FrameOut {
    object_pointer: Tensor, // [256]
    mem: Option<(Tensor, Tensor)>, // ([5184, 64] bf16 features, [5184, 64] positions)
}

/// Prompt for one frame: points `(x, y, label)` in 1008-px input space; labels 0/1 for points, 2/3 for box corners.
#[derive(Clone, Default)]
pub struct FramePrompt {
    pub points: Vec<(f32, f32, i32)>,
}

/// Tracking state of one object.
pub struct Session {
    next: usize,
    cond: BTreeMap<usize, FrameOut>,
    non_cond: BTreeMap<usize, FrameOut>,
    prompts: BTreeMap<usize, FramePrompt>,
    tracked: BTreeSet<usize>,
    /// total number of frames when the whole clip is known (affects the object-pointer time normalization)
    num_frames: Option<usize>,
}

impl Session {
    /// `num_frames`: `None` for streaming; `Some(n)` when the clip has `n` frames.
    pub fn new(num_frames: Option<usize>) -> Self {
        Self { next: 0, cond: BTreeMap::new(), non_cond: BTreeMap::new(), prompts: BTreeMap::new(), tracked: BTreeSet::new(), num_frames }
    }

    /// Prompt (replaces earlier prompts on that frame) for frame `frame_idx`, which must not have been processed yet.
    pub fn add_prompt(&mut self, frame_idx: usize, prompt: FramePrompt) {
        self.prompts.insert(frame_idx, prompt);
    }

    fn frame(&self, idx: usize) -> Option<&FrameOut> {
        self.cond.get(&idx).or_else(|| self.non_cond.get(&idx))
    }

    /// Stored memory of a processed frame, `(features [5184, 64], positions [5184, 64])` (for tests).
    pub fn memory(&self, idx: usize) -> Option<(&Tensor, &Tensor)> {
        self.frame(idx).and_then(|f| f.mem.as_ref()).map(|(a, b)| (a, b))
    }

    /// Stored object pointer `[256]` of a processed frame (for tests).
    pub fn object_pointer(&self, idx: usize) -> Option<&Tensor> {
        self.frame(idx).map(|f| &f.object_pointer)
    }

    /// Replaces the stored memory and pointer of a processed frame (for tests).
    pub fn override_frame(&mut self, idx: usize, features: Tensor, positions: Tensor, pointer: Tensor) {
        let out = FrameOut { object_pointer: pointer, mem: Some((features, positions)) };
        if self.cond.contains_key(&idx) {
            self.cond.insert(idx, out);
        } else {
            self.non_cond.insert(idx, out);
        }
    }

    /// Index the next processed frame will get.
    pub fn next_frame(&self) -> usize {
        self.next
    }
}

pub struct FrameResult {
    pub frame_idx: usize,
    /// low-resolution mask logits `[4*side, 4*side]` (288 for SAM 3, 256 for SAM 2) (the object is absent where `object_score <= 0`)
    pub logits: Tensor,
    pub object_score: f32,
}

pub struct VideoTracker {
    /// feature grid side (72 / 64), input size in pixels (1008 / 1024), and the conditioning-frame limit
    side: usize,
    img: usize,
    max_cond: Option<usize>,
    enc: MemoryEncoder,
    attn: MemoryAttention,
    obj_ptr: Ff,
    no_obj_ptr: Tensor,
    tproj: Linear,
    mem_tpos: Tensor, // [7, 64]
    occlusion: Tensor, // [64]
}

fn sine_pe_1d(pos: &[f32], dim: usize, device: &Device, dt: DType) -> Result<Tensor> {
    let pe = dim / 2;
    let dim_t: Vec<f32> = (0..pe).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / pe as f32)).collect();
    let mut out = vec![0f32; pos.len() * dim];
    for (r, &p) in pos.iter().enumerate() {
        for i in 0..pe {
            out[r * dim + i] = (p / dim_t[i]).sin();
            out[r * dim + pe + i] = (p / dim_t[i]).cos();
        }
    }
    Tensor::from_vec(out, (pos.len(), dim), device)?.to_dtype(dt)
}

impl VideoTracker {
    /// `m` holds `memory_encoder.*`, `memory_attention.*`, `object_pointer_proj.*` and friends (`tracker_model.` in the
    /// SAM 3 checkpoint, the checkpoint root for SAM 2). `max_cond`: how many conditioning frames memory attention may
    /// use (`None` = all).
    pub fn new(m: VarBuilder, side: usize, img: usize, max_cond: Option<usize>, device: &Device) -> Result<Self> {
        Ok(Self {
            side,
            img,
            max_cond,
            enc: MemoryEncoder::new(m.pp("memory_encoder"))?,
            attn: MemoryAttention::new(m.pp("memory_attention"), side, device)?,
            obj_ptr: Ff::new(D, D, D, 3, false, m.pp("object_pointer_proj"))?,
            no_obj_ptr: m.get((1, D), "no_object_pointer")?.reshape(D)?,
            tproj: linear(D, MEM, m.pp("temporal_positional_encoding_projection_layer"))?,
            mem_tpos: m.get((NUM_MASKMEM, 1, 1, MEM), "memory_temporal_positional_encoding")?.reshape((NUM_MASKMEM, MEM))?,
            occlusion: m.get((1, MEM), "occlusion_spatial_embedding_parameter")?.reshape(MEM)?,
        })
    }

    /// The reference's choice of which conditioning frames to attend to: the nearest before and after, then closest.
    fn select_cond<'a>(&self, frame_idx: usize, cond: &'a BTreeMap<usize, FrameOut>) -> (Vec<(usize, &'a FrameOut)>, Vec<usize>) {
        let max = self.max_cond.unwrap_or(usize::MAX);
        if cond.len() <= max {
            return (cond.iter().map(|(k, v)| (*k, v)).collect(), vec![]);
        }
        let mut sel: Vec<usize> = vec![];
        if let Some(&b) = cond.keys().filter(|&&t| t < frame_idx).max() {
            sel.push(b);
        }
        if let Some(&a) = cond.keys().filter(|&&t| t >= frame_idx).min() {
            sel.push(a);
        }
        let mut rest: Vec<usize> = cond.keys().filter(|t| !sel.contains(t)).copied().collect();
        rest.sort_by_key(|&t| t.abs_diff(frame_idx));
        sel.extend(rest.into_iter().take(max - sel.len()));
        let unselected: Vec<usize> = cond.keys().filter(|t| !sel.contains(t)).copied().collect();
        // reference iterates the selected dict in insertion order: before, after, then by distance
        (sel.iter().map(|t| (*t, &cond[t])).collect(), unselected)
    }

    /// Image features of `frame_idx` conditioned on the session's memory (forward-time tracking).
    pub(crate) fn conditioned(&self, s: &Session, frame_idx: usize, tf: &TrackerFeatures, vpos: &Tensor) -> Result<Tensor> {
        let (dev, dt) = (tf.raw.device(), tf.raw.dtype());
        let (selected, unselected) = self.select_cond(frame_idx, &s.cond);
        let mut order: Vec<(usize, &FrameOut)> = selected.into_iter().map(|(_, o)| (0usize, o)).collect();
        for off in (1..NUM_MASKMEM).rev() {
            let Some(prev) = frame_idx.checked_sub(off) else { continue };
            let out = s.non_cond.get(&prev).or_else(|| if unselected.contains(&prev) { s.cond.get(&prev) } else { None });
            if let Some(o) = out {
                order.push((off, o));
            }
        }
        let (mut mems, mut poss) = (vec![], vec![]);
        for (off, o) in &order {
            let Some((m, p)) = &o.mem else { continue };
            mems.push(m.to_dtype(dt)?);
            // reference indexes the temporal table with `offset - 1`, which wraps to the last row for offset 0
            let ti = if *off == 0 { NUM_MASKMEM - 1 } else { off - 1 };
            poss.push(p.to_dtype(dt)?.broadcast_add(&self.mem_tpos.get(ti)?.to_dtype(dt)?.unsqueeze(0)?)?);
        }

        // object pointers: conditioning frames at or before this one, then the previous non-conditioning frames
        let max_ptr = if s.num_frames.is_none() { MAX_POINTERS } else { MAX_POINTERS.min(s.num_frames.unwrap_or(MAX_POINTERS)) };
        let (mut offsets, mut ptrs) = (vec![], vec![]);
        for (&t, o) in &s.cond {
            if t <= frame_idx {
                offsets.push((frame_idx - t) as f32);
                ptrs.push(o.object_pointer.clone());
            }
        }
        for d in 1..max_ptr {
            let Some(r) = frame_idx.checked_sub(d) else { break };
            if let Some(n) = s.num_frames {
                if r >= n {
                    break;
                }
            }
            if let Some(o) = s.non_cond.get(&r) {
                offsets.push(d as f32);
                ptrs.push(o.object_pointer.clone());
            }
        }
        let mut n_ptr = 0;
        if !ptrs.is_empty() {
            let p = Tensor::stack(&ptrs, 0)?.to_dtype(dt)?; // [P, 256]
            let splits = D / MEM;
            let toks = p.reshape((ptrs.len() * splits, MEM))?; // each pointer split into 4 memory-width tokens
            let norm: Vec<f32> = offsets.iter().map(|o| o / (max_ptr as f32 - 1.0)).collect();
            let pe = self.tproj.forward(&sine_pe_1d(&norm, D, dev, dt)?)?; // [P, 64]
            let pe = pe.unsqueeze(1)?.broadcast_as((ptrs.len(), splits, MEM))?.reshape((ptrs.len() * splits, MEM))?;
            n_ptr = toks.dim(0)?;
            mems.push(toks);
            poss.push(pe);
        }
        let mem = Tensor::cat(&mems, 0)?.unsqueeze(0)?;
        let mem_pos = Tensor::cat(&poss, 0)?.unsqueeze(0)?;
        let vision = tf.raw.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let out = self.attn.forward(&vision, vpos, &mem, &mem_pos, n_ptr)?;
        out.transpose(1, 2)?.reshape((1, D, self.side, self.side))
    }

    /// Memory of a processed frame: `(features [5184, 64] bf16, positions [5184, 64])`.
    fn encode_memory(&self, tf: &TrackerFeatures, logits_hi: &Tensor, object_score: f32, from_prompt: bool) -> Result<(Tensor, Tensor)> {
        let dt = tf.raw.dtype();
        // a /16 downsampler must land on the feature grid: side * 16 (1152 for SAM 3, 1024 for SAM 2)
        let mem_size = self.side * 16;
        let big = logits_hi.upsample_bilinear2d(mem_size, mem_size, false)?;
        let m = if from_prompt { big.gt(0f32)?.to_dtype(DType::F32)? } else { candle_nn::ops::sigmoid(&big.to_dtype(DType::F32)?)? };
        let m = ((m * 20.0)? - 10.0)?.to_dtype(dt)?;
        let (mut feat, pos) = self.enc.forward(&tf.raw, &m, self.side)?;
        if object_score <= 0.0 {
            feat = feat.broadcast_add(&self.occlusion.to_dtype(dt)?.reshape((1, MEM, 1, 1))?)?;
        }
        let n = self.side * self.side;
        let tok = |t: &Tensor| t.flatten_from(2)?.transpose(1, 2)?.reshape((n, MEM));
        Ok((tok(&feat)?.to_dtype(DType::BF16)?, tok(&pos)?))
    }

    /// Processes the next frame of `s`. `tf`: its tracker features; `vpos`: the 72x72 sine positions `[1, 5184, 256]`.
    pub fn track(&self, tracker: &Tracker, s: &mut Session, tf: &TrackerFeatures, vpos: &Tensor, device: &Device) -> Result<FrameResult> {
        let dt = tf.raw.dtype();
        let idx = s.next;
        s.next += 1;
        let prompt = s.prompts.get(&idx).cloned();
        let is_init = prompt.is_some() && !s.tracked.contains(&idx);

        let pix = if is_init { tf.emb.clone() } else { self.conditioned(s, idx, tf, vpos)? };
        let feats = TrackerFeatures { s0: tf.s0.clone(), s1: tf.s1.clone(), emb: pix, raw: tf.raw.clone() };
        let points = prompt.as_ref().map(|p| p.points.clone()).unwrap_or_default();
        // multimask for a single click (or none), like the reference's `_use_multimask`
        let multimask = points.len() <= 1;
        let sparse = tracker.sparse(&points, &[], device, dt)?;
        let Prediction { masks, token, iou, object_score } = tracker.predict_sparse(&feats, &sparse, multimask)?;

        let obj = object_score > 0.0;
        let pick = if multimask { (0..iou.len()).max_by(|&a, &b| iou[a].total_cmp(&iou[b])).unwrap_or(0) } else { 0 };
        let low = masks.narrow(0, pick, 1)?.squeeze(0)?; // [4*side, 4*side]
        let low = if obj { low } else { (low.zeros_like()? + NO_OBJ_SCORE)? };

        let ptr = self.obj_ptr.forward(&token.unsqueeze(0)?)?.squeeze(0)?;
        let object_pointer = if obj { ptr } else { self.no_obj_ptr.to_dtype(ptr.dtype())? };

        let lo = 4 * self.side;
        let hi = low.to_dtype(DType::F32)?.reshape((1, 1, lo, lo))?.upsample_bilinear2d(self.img, self.img, false)?;
        let mem = self.encode_memory(tf, &hi, object_score, prompt.is_some())?;
        let out = FrameOut { object_pointer, mem: Some(mem) };
        if is_init {
            s.cond.insert(idx, out);
        } else {
            s.non_cond.insert(idx, out);
            s.tracked.insert(idx);
        }
        Ok(FrameResult { frame_idx: idx, logits: low, object_score })
    }
}
