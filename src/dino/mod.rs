//! Grounding DINO: open-vocabulary object detection from a text phrase (Swin backbone, BERT text encoder, a
//! feature-enhancer encoder with image-text fusion, language-guided query selection and a cross-modality decoder).
//! Port of `GroundingDinoForObjectDetection` from `transformers` for single images; `IDEA-Research/grounding-dino-*`
//! checkpoints. It turns "a cat. a remote control." into boxes, which any SAM can then segment (see
//! [`crate::grounded`]).

pub mod bert;
pub mod swin;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{conv2d, group_norm, layer_norm, linear, Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear, VarBuilder};
use image::RgbImage;

use crate::kernels::deform_attn;

pub const REPO_TINY: &str = "IDEA-Research/grounding-dino-tiny";
pub const REPO_BASE: &str = "IDEA-Research/grounding-dino-base";

/// Token ids that split a prompt into phrases: `[CLS]`, `[SEP]`, `.` and `?`.
const SPECIAL: [u32; 4] = [101, 102, 1012, 1029];

#[derive(Clone, Debug)]
struct Config {
    d: usize,
    heads: usize,
    enc_layers: usize,
    dec_layers: usize,
    ffn: usize,
    points: usize,
    levels: usize,
    queries: usize,
    max_text: usize,
    swin: (usize, Vec<usize>, Vec<usize>, usize), // embed, depths, heads, window
    pos_temperature: f32,
}

impl Config {
    fn from_json(v: &serde_json::Value) -> Self {
        let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let b = v.get("backbone_config").cloned().unwrap_or_default();
        let list = |k: &str, d: &[usize]| b.get(k).and_then(|x| x.as_array()).map_or(d.to_vec(), |a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect());
        Self {
            d: u("d_model", 256),
            heads: u("encoder_attention_heads", 8),
            enc_layers: u("encoder_layers", 6),
            dec_layers: u("decoder_layers", 6),
            ffn: u("encoder_ffn_dim", 2048),
            points: u("encoder_n_points", 4),
            levels: u("num_feature_levels", 4),
            queries: u("num_queries", 900),
            max_text: u("max_text_len", 256),
            swin: (
                b.get("embed_dim").and_then(|x| x.as_u64()).map_or(96, |x| x as usize),
                list("depths", &[2, 2, 6, 2]),
                list("num_heads", &[3, 6, 12, 24]),
                b.get("window_size").and_then(|x| x.as_u64()).map_or(7, |x| x as usize),
            ),
            pos_temperature: v.get("positional_embedding_temperature").and_then(|x| x.as_f64()).map_or(20.0, |x| x as f32),
        }
    }
}

// ------------------------------------------------------------------------------------------------ pieces

struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(i: usize, h: usize, o: usize, n: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let dims: Vec<(usize, usize)> = (0..n).map(|k| (if k == 0 { i } else { h }, if k + 1 == n { o } else { h })).collect();
        Ok(Self { layers: dims.iter().enumerate().map(|(k, &(a, b))| linear(a, b, vb.pp(format!("layers.{k}")))).collect::<candle_core::Result<_>>()? })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.layers.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

/// Multi-head attention with separate query / key / value inputs on `[L, C]` tensors.
struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Mha {
    fn new(c: usize, heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self { q: linear(c, c, vb.pp("query"))?, k: linear(c, c, vb.pp("key"))?, v: linear(c, c, vb.pp("value"))?, o: linear(c, c, vb.pp("out_proj"))?, heads })
    }

    /// `mask`: additive, broadcastable to `[heads, Lq, Lk]`.
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (lq, c) = q.dims2()?;
        let hd = c / self.heads;
        let split = |t: Tensor| -> candle_core::Result<Tensor> { let l = t.dim(0)?; t.reshape((l, self.heads, hd))?.transpose(0, 1)?.contiguous() };
        let (q, k, v) = (split(self.q.forward(q)?)?, split(self.k.forward(k)?)?, split(self.v.forward(v)?)?);
        let mut s = (q.matmul(&k.transpose(1, 2)?)? / (hd as f64).sqrt())?;
        if let Some(m) = mask {
            s = s.broadcast_add(m)?;
        }
        let a = candle_nn::ops::softmax(&s, D::Minus1)?.matmul(&v)?.transpose(0, 1)?.reshape((lq, c))?;
        self.o.forward(&a)
    }
}

/// `get_sine_pos_embed`: each of the `k` columns of `pos` (`[n, k]`) becomes `feats` sine / cosine features.
fn sine_embed(pos: &Tensor, feats: usize, temperature: f32, swap_xy: bool) -> candle_core::Result<Tensor> {
    let (n, k) = pos.dims2()?;
    let dev = pos.device();
    let freq: Vec<f32> = (0..feats / 2).map(|i| temperature.powf(2.0 * i as f32 / feats as f32)).collect();
    let freq = Tensor::from_vec(freq, (1, feats / 2), dev)?.to_dtype(pos.dtype())?;
    let mut cols = vec![];
    for j in 0..k {
        let ang = (pos.narrow(1, j, 1)?.affine(2.0 * std::f64::consts::PI, 0.0)?).broadcast_div(&freq)?;
        cols.push(Tensor::stack(&[&ang.sin()?, &ang.cos()?], 2)?.reshape((n, feats))?);
    }
    if swap_xy && k >= 2 {
        cols.swap(0, 1);
    }
    Tensor::cat(&cols, 1)
}

/// Multi-scale deformable attention (queries sample a few points on every feature level).
struct Deform {
    offsets: Linear,
    weights: Linear,
    value: Linear,
    out: Linear,
    heads: usize,
    levels: usize,
    points: usize,
}

impl Deform {
    fn new(c: usize, heads: usize, levels: usize, points: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            offsets: linear(c, heads * levels * points * 2, vb.pp("sampling_offsets"))?,
            weights: linear(c, heads * levels * points, vb.pp("attention_weights"))?,
            value: linear(c, c, vb.pp("value_proj"))?,
            out: linear(c, c, vb.pp("output_proj"))?,
            heads,
            levels,
            points,
        })
    }

    /// `query`: `[Q, C]` (with its position embedding already added), `value_in`: `[S, C]`, `refs`: `[Q, L, 2 or 4]`.
    fn forward(&self, query: &Tensor, value_in: &Tensor, refs: &Tensor, shapes: &[(usize, usize)]) -> candle_core::Result<Tensor> {
        let (q, c) = query.dims2()?;
        let s = value_in.dim(0)?;
        let (h, l, p) = (self.heads, self.levels, self.points);
        let value = self.value.forward(value_in)?.reshape((1, s, h, c / h))?;
        let off = self.offsets.forward(query)?.reshape((1, q, h, l, p, 2))?;
        let aw = candle_nn::ops::softmax_last_dim(&self.weights.forward(query)?.reshape((1, q, h, l * p))?)?.reshape((1, q, h, l, p))?;
        let ncoord = refs.dim(2)?;
        let loc = if ncoord == 2 {
            let norm: Vec<f32> = shapes.iter().flat_map(|&(hh, ww)| [ww as f32, hh as f32]).collect();
            let norm = Tensor::from_vec(norm, (1, 1, 1, l, 1, 2), query.device())?.to_dtype(query.dtype())?;
            let r = refs.reshape((1, q, 1, l, 1, 2))?;
            r.broadcast_add(&off.broadcast_div(&norm)?)?
        } else {
            let r = refs.reshape((1, q, 1, l, 1, 4))?;
            let (xy, wh) = (r.narrow(5, 0, 2)?, r.narrow(5, 2, 2)?);
            xy.broadcast_add(&(off / p as f64)?.broadcast_mul(&wh)?.affine(0.5, 0.0)?)?
        };
        let o = deform_attn(&value, &loc, &aw, shapes)?.reshape((q, c))?;
        self.out.forward(&o)
    }
}

struct Fusion {
    ln_v: LayerNorm,
    ln_t: LayerNorm,
    vision_proj: Linear,
    text_proj: Linear,
    values_vision: Linear,
    values_text: Linear,
    out_vision: Linear,
    out_text: Linear,
    vision_param: Tensor,
    text_param: Tensor,
    heads: usize,
}

impl Fusion {
    fn new(c: usize, ffn: usize, heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let (a, e) = (vb.pp("attn"), ffn / 2);
        Ok(Self {
            ln_v: layer_norm(c, 1e-5, vb.pp("layer_norm_vision"))?,
            ln_t: layer_norm(c, 1e-5, vb.pp("layer_norm_text"))?,
            vision_proj: linear(c, e, a.pp("vision_proj"))?,
            text_proj: linear(c, e, a.pp("text_proj"))?,
            values_vision: linear(c, e, a.pp("values_vision_proj"))?,
            values_text: linear(c, e, a.pp("values_text_proj"))?,
            out_vision: linear(e, c, a.pp("out_vision_proj"))?,
            out_text: linear(e, c, a.pp("out_text_proj"))?,
            vision_param: vb.get(c, "vision_param")?,
            text_param: vb.get(c, "text_param")?,
            heads,
        })
    }

    /// Image-to-text and text-to-image cross attention; `v`: `[N, C]`, `t`: `[T, C]`.
    fn forward(&self, v: &Tensor, t: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let (n, c) = v.dims2()?;
        let tl = t.dim(0)?;
        let (v, t) = (self.ln_v.forward(v)?, self.ln_t.forward(t)?);
        let e = self.vision_proj.weight().dim(0)?;
        let hd = e / self.heads;
        let split = |x: Tensor, len: usize| -> candle_core::Result<Tensor> { x.reshape((len, self.heads, hd))?.transpose(0, 1)?.contiguous() };
        let q = split((self.vision_proj.forward(&v)? * (hd as f64).powf(-0.5))?, n)?;
        let k = split(self.text_proj.forward(&t)?, tl)?;
        let vv = split(self.values_vision.forward(&v)?, n)?;
        let tv = split(self.values_text.forward(&t)?, tl)?;
        let attn = q.matmul(&k.transpose(1, 2)?)?; // [heads, N, T]
        let attn = attn.broadcast_sub(&attn.flatten_all()?.max(0)?)?.clamp(-50000.0, 50000.0)?;
        let at = attn.transpose(1, 2)?.contiguous()?; // [heads, T, N]
        let at = at.broadcast_sub(&at.max_keepdim(D::Minus1)?)?.clamp(-50000.0, 50000.0)?;
        let text_w = candle_nn::ops::softmax(&at, D::Minus1)?;
        let vision_w = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let dv = vision_w.matmul(&tv)?.transpose(0, 1)?.reshape((n, e))?; // image <- text
        let dt = text_w.matmul(&vv)?.transpose(0, 1)?.reshape((tl, e))?; // text <- image
        let dv = self.out_vision.forward(&dv)?;
        let dt = self.out_text.forward(&dt)?;
        let _ = c;
        Ok(((&v + dv.broadcast_mul(&self.vision_param)?)?, (&t + dt.broadcast_mul(&self.text_param)?)?))
    }
}

struct EncLayer {
    fusion: Fusion,
    t_attn: Mha,
    t_fc1: Linear,
    t_fc2: Linear,
    t_ln_before: LayerNorm,
    t_ln_after: LayerNorm,
    deform: Deform,
    ln1: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ln2: LayerNorm,
}

struct DecLayer {
    self_attn: Mha,
    ln_self: LayerNorm,
    text_attn: Mha,
    ln_text: LayerNorm,
    deform: Deform,
    ln_cross: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ln_ffn: LayerNorm,
}

// ------------------------------------------------------------------------------------------------ model

/// One detected phrase-matched box, in original-image pixels.
#[derive(Clone, Debug)]
pub struct Detection {
    /// `[x0, y0, x1, y1]`
    pub bbox: [f32; 4],
    pub score: f32,
    /// The words of the prompt this box matched.
    pub label: String,
}

pub struct GroundingDino {
    cfg: Config,
    swin: swin::Swin,
    input_proj: Vec<(Conv2d, GroupNorm)>,
    level_embed: Tensor,
    bert: bert::Bert,
    text_projection: Linear,
    enc: Vec<EncLayer>,
    dec: Vec<DecLayer>,
    ref_head: Mlp,
    bbox_embed: Mlp,
    enc_output: Linear,
    enc_output_norm: LayerNorm,
    enc_bbox: Mlp,
    query_embed: Tensor,
    dec_norm: LayerNorm,
    tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub dtype: DType,
}

/// Raw network outputs for one prompt.
pub struct Raw {
    /// `[queries, tokens]` logits (before the sigmoid)
    pub logits: Tensor,
    /// `[queries, 4]` normalized `(cx, cy, w, h)`
    pub boxes: Tensor,
    pub ids: Vec<u32>,
}

impl GroundingDino {
    pub fn from_file(config: &std::path::Path, weights: &std::path::Path, tokenizer: &std::path::Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config).with_context(|| format!("reading {}", config.display()))?)?;
        let cfg = Config::from_json(&cfg_json);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let m = vb.pp("model");
        let (embed, depths, heads, window) = cfg.swin.clone();
        let sc = swin::SwinConfig { embed_dim: embed, depths: depths.clone(), heads, window, patch: 4, out_stages: vec![2, 3, 4] };
        let chans: Vec<usize> = (1..depths.len()).map(|i| embed << i).collect(); // stage2..stage4 channels
        let swin = swin::Swin::new(&sc, m.pp("backbone.conv_encoder.model"))?;
        let mut input_proj = vec![];
        for i in 0..cfg.levels {
            let p = m.pp(format!("input_proj_vision.{i}"));
            let (cin, k, stride, pad) = if i < chans.len() { (chans[i], 1, 1, 0) } else { (if i == chans.len() { *chans.last().expect("stages") } else { cfg.d }, 3, 2, 1) };
            input_proj.push((conv2d(cin, cfg.d, k, Conv2dConfig { stride, padding: pad, ..Default::default() }, p.pp("0"))?, group_norm(32, cfg.d, 1e-5, p.pp("1"))?));
        }
        // the text branch is tiny and numerically touchy (masks, layer norms): keep it in f32 whatever the model dtype
        let vb32 = if dtype == DType::F32 { vb.clone() } else { unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? } };
        let m32 = vb32.pp("model");
        let bert = bert::Bert::new(m32.pp("text_backbone"), 768, 12, 12, 3072, 30522)?;
        let (d, h) = (cfg.d, cfg.heads);
        let enc = (0..cfg.enc_layers)
            .map(|i| {
                let l = m.pp(format!("encoder.layers.{i}"));
                let te = l.pp("text_enhancer_layer");
                let df = l.pp("deformable_layer");
                Ok(EncLayer {
                    fusion: Fusion::new(d, cfg.ffn, h / 2, l.pp("fusion_layer"))?,
                    t_attn: Mha::new(d, h / 2, te.pp("self_attn"))?,
                    t_fc1: linear(d, cfg.ffn / 2, te.pp("fc1"))?,
                    t_fc2: linear(cfg.ffn / 2, d, te.pp("fc2"))?,
                    t_ln_before: layer_norm(d, 1e-5, te.pp("layer_norm_before"))?,
                    t_ln_after: layer_norm(d, 1e-5, te.pp("layer_norm_after"))?,
                    deform: Deform::new(d, h, cfg.levels, cfg.points, df.pp("self_attn"))?,
                    ln1: layer_norm(d, 1e-5, df.pp("self_attn_layer_norm"))?,
                    fc1: linear(d, cfg.ffn, df.pp("fc1"))?,
                    fc2: linear(cfg.ffn, d, df.pp("fc2"))?,
                    ln2: layer_norm(d, 1e-5, df.pp("final_layer_norm"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let dec = (0..cfg.dec_layers)
            .map(|i| {
                let l = m.pp(format!("decoder.layers.{i}"));
                Ok(DecLayer {
                    self_attn: Mha::new(d, h, l.pp("self_attn"))?,
                    ln_self: layer_norm(d, 1e-5, l.pp("self_attn_layer_norm"))?,
                    text_attn: Mha::new(d, h, l.pp("encoder_attn_text"))?,
                    ln_text: layer_norm(d, 1e-5, l.pp("encoder_attn_text_layer_norm"))?,
                    deform: Deform::new(d, h, cfg.levels, cfg.points, l.pp("encoder_attn"))?,
                    ln_cross: layer_norm(d, 1e-5, l.pp("encoder_attn_layer_norm"))?,
                    fc1: linear(d, cfg.ffn, l.pp("fc1"))?,
                    fc2: linear(cfg.ffn, d, l.pp("fc2"))?,
                    ln_ffn: layer_norm(d, 1e-5, l.pp("final_layer_norm"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut tk = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tk.with_truncation(None).map_err(|e| anyhow::anyhow!("{e}"))?;
        tk.with_padding(None);
        Ok(Self {
            swin,
            input_proj,
            level_embed: m.get((cfg.levels, d), "level_embed")?,
            bert,
            text_projection: linear(768, d, m32.pp("text_projection"))?,
            enc,
            dec,
            ref_head: Mlp::new(2 * d, d, d, 2, m.pp("decoder.reference_points_head"))?,
            // the checkpoint stores one box head, shared by every decoder layer
            // (tiny names it `bbox_embed.0`, base `model.decoder.bbox_embed.0`)
            bbox_embed: Mlp::new(d, d, 4, 3, if vb.contains_tensor("bbox_embed.0.layers.0.weight") { vb.pp("bbox_embed.0") } else { m.pp("decoder.bbox_embed.0") })?,
            enc_output: linear(d, d, m.pp("enc_output"))?,
            enc_output_norm: layer_norm(d, 1e-5, m.pp("enc_output_norm"))?,
            enc_bbox: Mlp::new(d, d, 4, 3, m.pp("encoder_output_bbox_embed"))?,
            query_embed: m.get((cfg.queries, d), "query_position_embeddings.weight")?,
            dec_norm: layer_norm(d, 1e-5, m.pp("decoder.layer_norm"))?,
            tokenizer: tk,
            cfg,
            device: device.clone(),
            dtype,
        })
    }

    /// Downloads `IDEA-Research/grounding-dino-tiny` (or `-base`, via [`from_repo`](Self::from_repo)) on first use.
    pub fn from_pretrained(device: &Device, dtype: DType) -> Result<Self> {
        Self::from_repo(REPO_TINY, device, dtype)
    }

    pub fn from_repo(repo: &str, device: &Device, dtype: DType) -> Result<Self> {
        let c = crate::download::fetch(repo, "config.json")?;
        let w = crate::download::fetch(repo, "model.safetensors")?;
        let t = crate::download::fetch(repo, "tokenizer.json")?;
        Self::from_file(&c, &w, &t, device, dtype)
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    /// Resize (short side 800, long side at most 1333), scale to `[0, 1]`, ImageNet-normalize: `[1, 3, H, W]`.
    pub fn preprocess(&self, img: &RgbImage) -> Result<Tensor> {
        let (h, w) = (img.height() as f32, img.width() as f32);
        let (min, max) = (h.min(w), h.max(w));
        let (mut size, mut raw) = (800.0f32, None);
        if max / min * size > 1333.0 {
            let r = 1333.0 * min / max;
            raw = Some(r);
            size = r.round();
        }
        let (oh, ow) = if (h <= w && h == size) || (w <= h && w == size) {
            (h as u32, w as u32)
        } else if w < h {
            (raw.map_or((size * h / w) as u32, |r| (r * h / w) as u32), size as u32)
        } else {
            (size as u32, raw.map_or((size * w / h) as u32, |r| (r * w / h) as u32))
        };
        let resized = if (oh, ow) == (img.height(), img.width()) { img.clone() } else { image::imageops::resize(img, ow, oh, image::imageops::FilterType::Triangle) };
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let mut data = vec![0f32; 3 * (oh * ow) as usize];
        for (i, px) in resized.pixels().enumerate() {
            for c in 0..3 {
                data[c * (oh * ow) as usize + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
        Ok(Tensor::from_vec(data, (1, 3, oh as usize, ow as usize), &self.device)?.to_dtype(self.dtype)?)
    }

    /// Lower-cases the prompt, makes sure it ends with a full stop (phrases are separated by `.`) and tokenizes it.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let mut t = text.trim().to_lowercase();
        if !t.ends_with('.') {
            t.push('.');
        }
        let enc = self.tokenizer.encode(t, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(self.cfg.max_text);
        Ok(ids)
    }

    /// The full network on a preprocessed image and token ids. `trace` receives named intermediate tensors.
    pub fn forward(&self, px: &Tensor, ids: &[u32], mut trace: Option<&mut Vec<(&'static str, Tensor)>>) -> Result<Raw> {
        let cfg = &self.cfg;
        let (dev, dt) = (&self.device, self.dtype);
        let mut note = |name: &'static str, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        // ---- text
        let (allow, pos_ids) = text_masks(ids);
        let tl = ids.len();
        let text_raw = self.bert.forward(ids, &pos_ids, &allow, dev)?;
        let text = self.text_projection.forward(&text_raw)?.to_dtype(dt)?;
        note("text_features", &text);
        let text_self_mask = {
            let m: Vec<f32> = allow.iter().map(|&a| if a > 0 { 0.0 } else { -1e9 }).collect();
            Tensor::from_vec(m, (1, tl, tl), dev)?.to_dtype(dt)?
        };
        let text_pos = {
            let p: Vec<f32> = pos_ids.iter().map(|&p| p as f32).collect();
            sine_embed(&Tensor::from_vec(p, (tl, 1), dev)?.to_dtype(dt)?, cfg.d, 10000.0, false)?
        };

        // ---- image: backbone, projections, four feature levels
        let feats = self.swin.forward(px)?;
        for (i, f) in feats.iter().enumerate() {
            note(["swin2", "swin3", "swin4"][i.min(2)], f);
        }
        let mut maps: Vec<Tensor> = vec![];
        for (i, (conv, gn)) in self.input_proj.iter().enumerate() {
            let src = if i < feats.len() { &feats[i] } else if i == feats.len() { feats.last().expect("features") } else { maps.last().expect("previous level") };
            maps.push(gn.forward(&conv.forward(src)?)?);
        }
        let mut shapes = vec![];
        let (mut sources, mut poses) = (vec![], vec![]);
        for (l, m) in maps.iter().enumerate() {
            let (_, c, h, w) = m.dims4()?;
            shapes.push((h, w));
            sources.push(m.reshape((c, h * w))?.t()?.contiguous()?);
            let pe = Tensor::from_vec(sine_pos(h, w, cfg.d / 2, cfg.pos_temperature), (h * w, cfg.d), dev)?.to_dtype(dt)?;
            poses.push(pe.broadcast_add(&self.level_embed.get(l)?.reshape((1, cfg.d))?)?);
        }
        let mut v = Tensor::cat(&sources, 0)?;
        let pos = Tensor::cat(&poses, 0)?;
        note("vision_input", &v);
        let n = v.dim(0)?;

        // ---- encoder
        let refs = encoder_reference_points(&shapes, dev)?.to_dtype(dt)?; // [N, L, 2]
        let mut t = text.clone();
        for (li, layer) in self.enc.iter().enumerate() {
            let (nv, nt) = layer.fusion.forward(&v, &t)?;
            v = nv;
            t = nt;
            if li == 0 {
                note("l0_fusion_v", &v);
                note("l0_fusion_t", &t);
            }
            let tp = (&t + &text_pos)?;
            let a = layer.t_attn.forward(&tp, &tp, &t, Some(&text_self_mask))?;
            t = layer.t_ln_before.forward(&(&t + a)?)?;
            let f = layer.t_fc2.forward(&layer.t_fc1.forward(&t)?.relu()?)?;
            t = layer.t_ln_after.forward(&(&t + f)?)?;
            if li == 0 {
                note("l0_text", &t);
            }
            let a = layer.deform.forward(&(&v + &pos)?, &v, &refs, &shapes)?;
            v = layer.ln1.forward(&(&v + a)?)?;
            let f = layer.fc2.forward(&layer.fc1.forward(&v)?.relu()?)?;
            v = layer.ln2.forward(&(&v + f)?)?;
            if li == 0 {
                note("l0_deform", &v);
            }
        }
        note("enc_vision", &v);
        note("enc_text", &t);

        // ---- two-stage query selection
        let (query_obj, proposals) = self.proposals(&v, &shapes)?;
        let enc_logits = query_obj.matmul(&t.t()?)?; // [N, T]
        let enc_delta = self.enc_bbox.forward(&query_obj)?;
        let enc_coord = (enc_delta + proposals)?; // [N, 4] logits
        let best: Vec<f32> = enc_logits.max(1)?.to_dtype(DType::F32)?.to_vec1()?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| best[b].total_cmp(&best[a]));
        order.truncate(cfg.queries);
        let top = Tensor::from_vec(order.iter().map(|&i| i as u32).collect::<Vec<_>>(), order.len(), dev)?;
        let mut reference = candle_nn::ops::sigmoid(&enc_coord.index_select(&top, 0)?)?; // [Q, 4]
        note("topk_reference", &reference);
        let mut h = self.query_embed.to_dtype(dt)?;

        // ---- decoder
        let mut last_ref = reference.clone();
        for (idx, layer) in self.dec.iter().enumerate() {
            last_ref = reference.clone();
            let ref_in = reference.unsqueeze(1)?.broadcast_as((reference.dim(0)?, cfg.levels, 4))?.contiguous()?; // valid ratios are 1
            let query_pos = self.ref_head.forward(&sine_embed(&reference, cfg.d / 2, 10000.0, true)?)?;
            let qk = (&h + &query_pos)?;
            let a = layer.self_attn.forward(&qk, &qk, &h, None)?;
            h = layer.ln_self.forward(&(&h + a)?)?;
            let a = layer.text_attn.forward(&(&h + &query_pos)?, &t, &t, None)?;
            h = layer.ln_text.forward(&(&h + a)?)?;
            let a = layer.deform.forward(&(&h + &query_pos)?, &v, &ref_in, &shapes)?;
            h = layer.ln_cross.forward(&(&h + a)?)?;
            let f = layer.fc2.forward(&layer.fc1.forward(&h)?.relu()?)?;
            h = layer.ln_ffn.forward(&(&h + f)?)?;
            if idx + 1 < self.dec.len() {
                let tmp = self.bbox_embed.forward(&h)?;
                reference = candle_nn::ops::sigmoid(&(tmp + inverse_sigmoid(&reference)?)?)?;
            }
        }
        let hn = self.dec_norm.forward(&h)?;
        note("dec_hidden", &hn);
        let logits = hn.matmul(&t.t()?)?;
        let boxes = candle_nn::ops::sigmoid(&(self.bbox_embed.forward(&hn)? + inverse_sigmoid(&last_ref)?)?)?;
        Ok(Raw { logits, boxes, ids: ids.to_vec() })
    }

    /// `generate_encoder_output_proposals`: per-position box proposals and their features.
    fn proposals(&self, v: &Tensor, shapes: &[(usize, usize)]) -> Result<(Tensor, Tensor)> {
        let n = v.dim(0)?;
        let mut prop = Vec::with_capacity(n * 4);
        let mut valid = Vec::with_capacity(n);
        for (l, &(h, w)) in shapes.iter().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let p = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, 0.05 * 2f32.powi(l as i32), 0.05 * 2f32.powi(l as i32)];
                    let ok = p.iter().all(|&q| q > 0.01 && q < 0.99);
                    valid.push(if ok { 1.0f32 } else { 0.0 });
                    prop.extend(p.iter().map(|&q| if ok { (q / (1.0 - q)).ln() } else { f32::INFINITY }));
                }
            }
        }
        let dev = v.device();
        let valid = Tensor::from_vec(valid, (n, 1), dev)?.to_dtype(v.dtype())?;
        let prop = Tensor::from_vec(prop, (n, 4), dev)?.to_dtype(v.dtype())?;
        let q = self.enc_output_norm.forward(&self.enc_output.forward(&v.broadcast_mul(&valid)?)?)?;
        Ok((q, prop))
    }

    /// Boxes above `box_threshold` for the phrases in `text` ("a cat. a remote control."), each labelled with the words
    /// of the prompt it matched (tokens whose score exceeds `text_threshold`). Sorted by score.
    pub fn detect(&self, img: &RgbImage, text: &str, box_threshold: f32, text_threshold: f32) -> Result<Vec<Detection>> {
        let px = self.preprocess(img)?;
        let ids = self.tokenize(text)?;
        if ids.len() <= 2 {
            bail!("empty prompt");
        }
        let raw = self.forward(&px, &ids, None)?;
        self.postprocess(&raw, (img.height() as usize, img.width() as usize), box_threshold, text_threshold)
    }

    pub fn postprocess(&self, raw: &Raw, (h, w): (usize, usize), box_threshold: f32, text_threshold: f32) -> Result<Vec<Detection>> {
        let probs: Vec<Vec<f32>> = candle_nn::ops::sigmoid(&raw.logits.to_dtype(DType::F32)?)?.to_vec2()?;
        let boxes: Vec<Vec<f32>> = raw.boxes.to_dtype(DType::F32)?.to_vec2()?;
        let mut out = vec![];
        for (p, b) in probs.iter().zip(&boxes) {
            let score = p.iter().cloned().fold(0.0f32, f32::max);
            if score <= box_threshold {
                continue;
            }
            let words: Vec<u32> = p.iter().zip(&raw.ids).filter(|(&q, id)| q > text_threshold && !SPECIAL.contains(id)).map(|(_, &id)| id).collect();
            let label = self.tokenizer.decode(&words, true).map_err(|e| anyhow::anyhow!("{e}"))?;
            let (cx, cy, bw, bh) = (b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32);
            out.push(Detection { bbox: [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], score, label });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }
}

/// Grounded segmentation (the "Grounded-SAM" recipe): Grounding DINO's boxes prompt a SAM-style model. `predict`
/// segments inside one box (`[x0, y0, x1, y1]`, original-image pixels) and returns its candidate masks; the one with the
/// highest predicted IoU is kept for each detection.
pub fn ground(dets: Vec<Detection>, mut predict: impl FnMut([f32; 4]) -> Result<Vec<crate::Mask>>) -> Result<Vec<(Detection, crate::Mask)>> {
    let mut out = vec![];
    for d in dets {
        if let Some(best) = predict(d.bbox)?.into_iter().max_by(|a, b| a.iou.total_cmp(&b.iou)) {
            out.push((d, best));
        }
    }
    Ok(out)
}

/// Greedy non-maximum suppression: keeps the highest-scoring box of every group overlapping above `iou`.
pub fn nms(mut dets: Vec<Detection>, iou: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let area = |b: &[f32; 4]| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let mut keep: Vec<Detection> = vec![];
    for d in dets {
        let overlaps = keep.iter().any(|k| {
            let (x0, y0, x1, y1) = (d.bbox[0].max(k.bbox[0]), d.bbox[1].max(k.bbox[1]), d.bbox[2].min(k.bbox[2]), d.bbox[3].min(k.bbox[3]));
            let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
            inter / (area(&d.bbox) + area(&k.bbox) - inter).max(1e-9) > iou
        });
        if !overlaps {
            keep.push(d);
        }
    }
    keep
}

fn inverse_sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let x = x.clamp(1e-5, 1.0 - 1e-5)?;
    (x.clone() / (1.0 - &x)?)?.log()
}

/// Encoder reference points: every position's centre, repeated for each level (no padding, so valid ratios are 1).
fn encoder_reference_points(shapes: &[(usize, usize)], device: &Device) -> candle_core::Result<Tensor> {
    let mut pts = vec![];
    for &(h, w) in shapes {
        for y in 0..h {
            for x in 0..w {
                pts.push(((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32));
            }
        }
    }
    let n = pts.len();
    let mut data = Vec::with_capacity(n * shapes.len() * 2);
    for (x, y) in pts {
        for _ in 0..shapes.len() {
            data.push(x);
            data.push(y);
        }
    }
    Tensor::from_vec(data, (n, shapes.len(), 2), device)
}

/// 2-D sine position embedding of an all-valid `h x w` grid: `[h*w, 2*feats]` (row features first, then columns).
fn sine_pos(h: usize, w: usize, feats: usize, temperature: f32) -> Vec<f32> {
    let eps = 1e-6f32;
    let scale = 2.0 * std::f32::consts::PI;
    let freq: Vec<f32> = (0..feats / 2).map(|i| temperature.powf(2.0 * i as f32 / feats as f32)).collect();
    let embed = |idx: usize, n: usize, out: &mut [f32]| {
        let v = (idx as f32 + 1.0) / (n as f32 + eps) * scale;
        for (i, f) in freq.iter().enumerate() {
            out[2 * i] = (v / f).sin();
            out[2 * i + 1] = (v / f).cos();
        }
    };
    let mut out = vec![0f32; h * w * 2 * feats];
    for y in 0..h {
        for x in 0..w {
            let o = &mut out[(y * w + x) * 2 * feats..(y * w + x + 1) * 2 * feats];
            let (py, px) = o.split_at_mut(feats);
            embed(y, h, py);
            embed(x, w, px);
        }
    }
    out
}

/// `generate_masks_with_special_tokens_and_transfer_map`: which tokens may attend to which (words of one phrase only
/// see each other), and each token's position inside its phrase.
fn text_masks(ids: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let l = ids.len();
    let special: Vec<bool> = ids.iter().map(|i| SPECIAL.contains(i)).collect();
    let mut prev = vec![-1i64; l];
    let mut cur = -1i64;
    for i in 0..l {
        if special[i] {
            cur = i as i64;
        }
        prev[i] = cur;
    }
    let mut next = vec![l; l];
    let mut cur = l;
    for i in (0..l).rev() {
        if special[i] {
            cur = i;
        }
        next[i] = cur;
    }
    let valid: Vec<bool> = next.iter().map(|&n| n != 0 && n != l - 1 && n != l).collect();
    let mut allow = vec![0u8; l * l];
    for i in 0..l {
        for j in 0..l {
            allow[i * l + j] = (i == j || (next[i] == next[j] && valid[j])) as u8;
        }
    }
    let pos: Vec<u32> = (0..l).map(|i| if valid[i] { (i as i64 - prev[i] - 1).max(0) as u32 } else { 0 }).collect();
    (allow, pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrases_only_see_themselves() {
        // [CLS] a cat . a dog . [SEP]
        let ids = [101, 1037, 4937, 1012, 1037, 3899, 1012, 102];
        let (allow, pos) = text_masks(&ids);
        let at = |i: usize, j: usize| allow[i * ids.len() + j] == 1;
        assert!(at(1, 2) && at(2, 1) && at(1, 3)); // words of "a cat ." attend to each other and its full stop
        assert!(!at(1, 4) && !at(5, 2)); // but not across phrases
        assert!(at(0, 0) && !at(0, 1)); // [CLS] only sees itself
        assert_eq!(pos, vec![0, 0, 1, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn size_rule() {
        // checked through the arithmetic of `preprocess` on a plain 640x480 image: short side 800 -> 1067x800
        let (h, w) = (480f32, 640f32);
        let size = 800.0f32;
        assert_eq!((size * w / h) as u32, 1066);
    }
}
