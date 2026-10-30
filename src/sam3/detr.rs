//! Fusion encoder, DETR decoder (presence token, box refinement, box-RPB attention bias), dot-product
//! scoring and the mask head.

use candle_core::{Result, Tensor};
use candle_nn::{group_norm, linear, GroupNorm, LayerNorm, Linear, Module, VarBuilder};

use super::nn::{inverse_sigmoid, ln, Act, Attn, FastConv, Mlp, Mlp3};

const D: usize = 256;
const HEADS: usize = 8;
const FF: usize = 2048;
const LN_EPS: f64 = 1e-5;

// ------------------------------------------------------------------------------------------ encoder

struct EncLayer {
    ln1: LayerNorm,
    self_attn: Attn,
    cross_attn: Attn,
    ln2: LayerNorm,
    mlp: Mlp,
    ln3: LayerNorm,
}

pub struct Encoder {
    layers: Vec<EncLayer>,
}

impl Encoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layers = (0..6)
            .map(|i| {
                let l = vb.pp(format!("layers.{i}"));
                Ok(EncLayer {
                    ln1: ln(D, LN_EPS, l.pp("layer_norm1"))?,
                    self_attn: Attn::new(D, HEADS, l.pp("self_attn"))?,
                    cross_attn: Attn::new(D, HEADS, l.pp("cross_attn"))?,
                    ln2: ln(D, LN_EPS, l.pp("layer_norm2"))?,
                    mlp: Mlp::new(D, FF, Act::Relu, l.pp("mlp"))?,
                    ln3: ln(D, LN_EPS, l.pp("layer_norm3"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    /// `vision`, `pos`: `[1, N, 256]`; `text`: `[1, L, 256]`; `text_mask`: additive `[1,1,1,L]`.
    pub fn forward(&self, vision: &Tensor, pos: &Tensor, text: &Tensor, text_mask: &Tensor) -> Result<Tensor> {
        let mut x = vision.clone();
        for l in &self.layers {
            let h = l.ln1.forward(&x)?;
            let hp = (&h + pos)?;
            x = (&x + l.self_attn.forward(&hp, &hp, &h, None)?)?;
            let h = l.ln2.forward(&x)?;
            x = (&x + l.cross_attn.forward(&h, text, text, Some(text_mask))?)?;
            x = (&x + l.mlp.forward(&l.ln3.forward(&x)?)?)?;
        }
        Ok(x)
    }
}

// ------------------------------------------------------------------------------------------ decoder

struct DecLayer {
    self_attn: Attn,
    self_ln: LayerNorm,
    text_attn: Attn,
    text_ln: LayerNorm,
    vis_attn: Attn,
    vis_ln: LayerNorm,
    mlp: Mlp,
    mlp_ln: LayerNorm,
}

pub struct Decoder {
    layers: Vec<DecLayer>,
    out_ln: LayerNorm,
    box_head: Mlp3,
    query_embed: Tensor,
    reference_points: Tensor,
    presence_token: Tensor,
    presence_head: Mlp3,
    presence_ln: LayerNorm,
    ref_point_head: Mlp3,
    rpb_x: Mlp3,
    rpb_y: Mlp3,
}

pub struct DecoderOut {
    /// final-layer normalized query states `[1, Q, 256]`
    pub queries: Tensor,
    /// refined boxes of the last layer, `(cx, cy, w, h)` in [0,1], `[1, Q, 4]`
    pub boxes: Tensor,
    /// clamped presence logit `[1, 1]`
    pub presence: Tensor,
}

/// Interleaved `sin, cos, sin, cos, ...` embedding of `[1, Q]` coordinates into 128 features.
pub(crate) fn sine_1d(v: &Tensor) -> Result<Tensor> {
    let nf = 128usize;
    let dim_t: Vec<f32> = (0..nf).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / nf as f32)).collect();
    let dim_t = Tensor::from_vec(dim_t, (1, 1, nf), v.device())?.to_dtype(v.dtype())?;
    let p = (v.unsqueeze(2)? * (2.0 * std::f64::consts::PI))?.broadcast_div(&dim_t)?;
    let (b, q, _) = p.dims3()?;
    let p = p.reshape((b, q, nf / 2, 2))?;
    Tensor::cat(&[&p.narrow(3, 0, 1)?.sin()?, &p.narrow(3, 1, 1)?.cos()?], 3)?.reshape((b, q, nf))
}

/// `[1, Q, 4]` (cx, cy, w, h) -> `[1, Q, 512]` embedding ordered (y, x, w, h).
fn box_sine(boxes: &Tensor) -> Result<Tensor> {
    let c = |i: usize| boxes.narrow(2, i, 1)?.squeeze(2);
    Tensor::cat(&[sine_1d(&c(1)?)?, sine_1d(&c(0)?)?, sine_1d(&c(2)?)?, sine_1d(&c(3)?)?], 2)
}

fn cxcywh_to_xyxy(b: &Tensor) -> Result<Tensor> {
    let c = |i: usize| b.narrow(b.rank() - 1, i, 1);
    let (cx, cy, w, h) = (c(0)?, c(1)?, c(2)?, c(3)?);
    let (hw, hh) = ((&w * 0.5)?, (&h * 0.5)?);
    Tensor::cat(&[(&cx - &hw)?, (&cy - &hh)?, (&cx + &hw)?, (&cy + &hh)?], b.rank() - 1)
}

impl Decoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layers = (0..6)
            .map(|i| {
                let l = vb.pp(format!("layers.{i}"));
                Ok(DecLayer {
                    self_attn: Attn::new(D, HEADS, l.pp("self_attn"))?,
                    self_ln: ln(D, LN_EPS, l.pp("self_attn_layer_norm"))?,
                    text_attn: Attn::new(D, HEADS, l.pp("text_cross_attn"))?,
                    text_ln: ln(D, LN_EPS, l.pp("text_cross_attn_layer_norm"))?,
                    vis_attn: Attn::new(D, HEADS, l.pp("vision_cross_attn"))?,
                    vis_ln: ln(D, LN_EPS, l.pp("vision_cross_attn_layer_norm"))?,
                    mlp: Mlp::new(D, FF, Act::Relu, l.pp("mlp"))?,
                    mlp_ln: ln(D, LN_EPS, l.pp("mlp_layer_norm"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            layers,
            out_ln: ln(D, LN_EPS, vb.pp("output_layer_norm"))?,
            box_head: Mlp3::new(D, D, 4, 3, vb.pp("box_head"))?,
            query_embed: vb.get((200, D), "query_embed.weight")?,
            reference_points: vb.get((200, 4), "reference_points.weight")?,
            presence_token: vb.get((1, D), "presence_token.weight")?,
            presence_head: Mlp3::new(D, D, 1, 3, vb.pp("presence_head"))?,
            presence_ln: ln(D, LN_EPS, vb.pp("presence_layer_norm"))?,
            ref_point_head: Mlp3::new(2 * D, D, D, 2, vb.pp("ref_point_head"))?,
            rpb_x: Mlp3::new(2, D, HEADS, 2, vb.pp("box_rpb_embed_x"))?,
            rpb_y: Mlp3::new(2, D, HEADS, 2, vb.pp("box_rpb_embed_y"))?,
        })
    }

    /// Box relative-position bias `[1, heads, Q+1, h*w]` (zero row for the presence token).
    fn rpb(&self, ref_boxes: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (dev, dt) = (ref_boxes.device(), ref_boxes.dtype());
        let xyxy = cxcywh_to_xyxy(ref_boxes)?.squeeze(0)?; // [Q, 4]
        let q = xyxy.dim(0)?;
        let grid = |n: usize| -> Result<Tensor> { Tensor::from_vec((0..n).map(|i| i as f32 / n as f32).collect::<Vec<_>>(), (1, n, 1), dev)?.to_dtype(dt) };
        let pick = |a: usize, b: usize| -> Result<Tensor> { Tensor::cat(&[xyxy.narrow(1, a, 1)?, xyxy.narrow(1, b, 1)?], 1)?.reshape((q, 1, 2)) };
        let dy = grid(h)?.broadcast_sub(&pick(1, 3)?)?; // [Q, h, 2]
        let dx = grid(w)?.broadcast_sub(&pick(0, 2)?)?; // [Q, w, 2]
        let enc = |d: Tensor| -> Result<Tensor> {
            let d = (d * 8.0)?;
            let sign = (d.gt(0f64)?.to_dtype(d.dtype())? - d.lt(0f64)?.to_dtype(d.dtype())?)?;
            (sign * ((d.abs()? + 1.0)?.log()? / std::f64::consts::LN_2)?)? / 3.0
        };
        let ey = self.rpb_y.forward(&enc(dy)?)?; // [Q, h, heads]
        let ex = self.rpb_x.forward(&enc(dx)?)?; // [Q, w, heads]
        let m = ey.unsqueeze(2)?.broadcast_add(&ex.unsqueeze(1)?)?; // [Q, h, w, heads]
        let m = m.reshape((q, h * w, HEADS))?.permute((2, 0, 1))?; // [heads, Q, hw]
        let m = Tensor::cat(&[Tensor::zeros((HEADS, 1, h * w), dt, dev)?, m], 1)?;
        m.unsqueeze(0)
    }

    pub fn forward(&self, vision: &Tensor, vpos: &Tensor, text: &Tensor, text_mask: &Tensor, h: usize, w: usize) -> Result<DecoderOut> {
        let dt = vision.dtype();
        let mut refb = self.reference_points.to_dtype(dt)?.unsqueeze(0)?;
        refb = candle_nn::ops::sigmoid(&refb)?;
        let mut x = Tensor::cat(&[self.presence_token.to_dtype(dt)?.unsqueeze(0)?, self.query_embed.to_dtype(dt)?.unsqueeze(0)?], 1)?;
        let key_vis = (vision + vpos)?;
        let mut presence = None;
        for l in &self.layers {
            let qpos = self.ref_point_head.forward(&box_sine(&refb)?)?;
            let qpos = Tensor::cat(&[Tensor::zeros((1, 1, D), dt, vision.device())?, qpos], 1)?;
            let rpb = self.rpb(&refb, h, w)?;

            let qk = (&x + &qpos)?;
            x = l.self_ln.forward(&(&x + l.self_attn.forward(&qk, &qk, &x, None)?)?)?;
            let qk = (&x + &qpos)?;
            x = l.text_ln.forward(&(&x + l.text_attn.forward(&qk, text, text, Some(text_mask))?)?)?;
            let qk = (&x + &qpos)?;
            x = l.vis_ln.forward(&(&x + l.vis_attn.forward(&qk, &key_vis, vision, Some(&rpb))?)?)?;
            x = l.mlp_ln.forward(&(&x + l.mlp.forward(&x)?)?)?;

            let qh = x.narrow(1, 1, x.dim(1)? - 1)?;
            let delta = self.box_head.forward(&self.out_ln.forward(&qh)?)?;
            refb = candle_nn::ops::sigmoid(&(delta + inverse_sigmoid(&refb)?)?)?;
            let pr = self.presence_head.forward(&self.presence_ln.forward(&x.narrow(1, 0, 1)?)?)?.squeeze(2)?;
            presence = Some(pr.clamp(-10f64, 10f64)?);
        }
        let qh = x.narrow(1, 1, x.dim(1)? - 1)?;
        Ok(DecoderOut { queries: self.out_ln.forward(&qh)?, boxes: refb, presence: presence.expect("six layers") })
    }
}

pub fn boxes_xyxy(b: &Tensor) -> Result<Tensor> {
    cxcywh_to_xyxy(b)
}

// ------------------------------------------------------------------------------------------ scoring

pub struct Scoring {
    text_mlp: Mlp3,
    out_ln: LayerNorm,
    text_proj: Linear,
    query_proj: Linear,
}

impl Scoring {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            text_mlp: Mlp3::new(D, FF, D, 2, vb.pp("text_mlp"))?,
            out_ln: ln(D, LN_EPS, vb.pp("text_mlp_out_norm"))?,
            text_proj: linear(D, D, vb.pp("text_proj"))?,
            query_proj: linear(D, D, vb.pp("query_proj"))?,
        })
    }

    /// `queries`: `[1, Q, 256]`; `text`: `[1, L, 256]`; `valid`: 0/1 per token. Returns logits `[1, Q]`.
    pub fn forward(&self, queries: &Tensor, text: &Tensor, valid: &[u8]) -> Result<Tensor> {
        let t = self.out_ln.forward(&(self.text_mlp.forward(text)? + text)?)?;
        let m = Tensor::from_vec(valid.iter().map(|&v| v as f32).collect::<Vec<_>>(), (1, valid.len(), 1), t.device())?.to_dtype(t.dtype())?;
        let n = valid.iter().filter(|&&v| v != 0).count().max(1) as f64;
        let pooled = (t.broadcast_mul(&m)?.sum(1)? / n)?;
        let pt = self.text_proj.forward(&pooled)?.unsqueeze(2)?; // [1, 256, 1]
        let pq = self.query_proj.forward(queries)?; // [1, Q, 256]
        let s = (pq.matmul(&pt)?.squeeze(2)? * (1.0 / (D as f64).sqrt()))?;
        s.clamp(-12f64, 12f64)
    }
}

// ------------------------------------------------------------------------------------------ mask head

pub struct MaskDecoder {
    pixel_convs: Vec<FastConv>,
    pixel_norms: Vec<GroupNorm>,
    mask_embed: Vec<Linear>,
    instance_proj: FastConv,
    prompt_attn: Attn,
    prompt_ln: LayerNorm,
}

impl MaskDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let pd = vb.pp("pixel_decoder");
        Ok(Self {
            pixel_convs: (0..2).map(|i| FastConv::new(D, D, 3, pd.pp(format!("conv_layers.{i}")))).collect::<Result<_>>()?,
            pixel_norms: (0..2).map(|i| group_norm(8, D, 1e-5, pd.pp(format!("norms.{i}")))).collect::<Result<_>>()?,
            mask_embed: (0..3).map(|i| linear(D, D, vb.pp(format!("mask_embedder.layers.{i}")))).collect::<Result<_>>()?,
            instance_proj: FastConv::new(D, D, 1, vb.pp("instance_projection"))?,
            prompt_attn: Attn::new(D, HEADS, vb.pp("prompt_cross_attn"))?,
            prompt_ln: ln(D, LN_EPS, vb.pp("prompt_cross_attn_norm"))?,
        })
    }

    /// `queries`: `[1,Q,256]`, `fpn`: the 288² and 144² levels, `enc`: `[1, 72*72, 256]` encoder output,
    /// `text`/`text_mask`: prompt features. Returns mask logits `[1, Q, 288, 288]`.
    pub fn forward(&self, queries: &Tensor, fpn: &[Tensor], enc: &Tensor, text: &Tensor, text_mask: &Tensor, side: usize) -> Result<Tensor> {
        let n = enc.dim(1)?;
        let normed = self.prompt_ln.forward(enc)?;
        let enc = (enc + self.prompt_attn.forward(&normed, text, text, Some(text_mask))?)?;
        let mut prev = enc.transpose(1, 2)?.reshape((1, D, side, side))?;
        let _ = n;
        for (i, feat) in fpn.iter().rev().enumerate() {
            let (h, w) = (feat.dim(2)?, feat.dim(3)?);
            prev = (prev.upsample_nearest2d(h, w)? + feat)?;
            prev = self.pixel_norms[i].forward(&self.pixel_convs[i].forward(&prev)?)?.relu()?;
        }
        let inst = self.instance_proj.forward(&prev)?; // [1, 256, H, W]
        let mut q = queries.clone();
        for (i, l) in self.mask_embed.iter().enumerate() {
            q = l.forward(&q)?;
            if i + 1 < self.mask_embed.len() {
                q = q.relu()?;
            }
        }
        let (h, w) = (inst.dim(2)?, inst.dim(3)?);
        q.matmul(&inst.reshape((1, D, h * w))?)?.reshape((1, queries.dim(1)?, h, w))
    }
}

