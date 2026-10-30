//! SAM 3 image tracker: SAM 2-style point / box / mask prompting on top of the shared ViT tokens, through its own
//! FPN neck. Mask decoder with an object-score token, high-resolution skip features and dynamic multimask output.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, LayerNorm, Linear, Module, VarBuilder};

use super::nn::{ln, FastConv};
use crate::kernels::{add_par, layer_norm_2d, layer_norm_rows, mha_cpu};
use crate::sam1::decoder::Up2;

const D: usize = 256;

/// `ff(x)`: `act(proj_in(x))`, `layers`, `proj_out`; ReLU between; optional sigmoid on the output.
pub(crate) struct Ff {
    proj_in: Linear,
    layers: Vec<Linear>,
    proj_out: Linear,
    sigmoid: bool,
}

impl Ff {
    pub(crate) fn new(i: usize, h: usize, o: usize, n: usize, sigmoid: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: linear(i, h, vb.pp("proj_in"))?,
            layers: (0..n - 2).map(|j| linear(h, h, vb.pp(format!("layers.{j}")))).collect::<Result<_>>()?,
            proj_out: linear(h, o, vb.pp("proj_out"))?,
            sigmoid,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.proj_in.forward(x)?.relu()?;
        for l in &self.layers {
            x = l.forward(&x)?.relu()?;
        }
        let x = self.proj_out.forward(&x)?;
        if self.sigmoid {
            candle_nn::ops::sigmoid(&x)
        } else {
            Ok(x)
        }
    }
}

/// Attention with a down-projected inner dimension (`o_proj` named like the other SAM 3 attentions).
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attn {
    fn new(down: usize, vb: VarBuilder) -> Result<Self> {
        let inner = D / down;
        Ok(Self {
            q: linear(D, inner, vb.pp("q_proj"))?,
            k: linear(D, inner, vb.pp("k_proj"))?,
            v: linear(D, inner, vb.pp("v_proj"))?,
            o: linear(inner, D, vb.pp("o_proj"))?,
            heads: 8,
        })
    }

    /// `[1, Lq, 256]`, `[1, Lk, 256]` -> `[1, Lq, 256]`
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (lq, lk) = (q.dim(1)?, k.dim(1)?);
        // CPU: one parallel kernel over (query, head) instead of transposes + generic attention
        let (qp, kp, vp) = (self.q.forward(q)?, self.k.forward(k)?, self.v.forward(v)?);
        let inner = qp.dim(2)?;
        if let Some(o) = mha_cpu(&qp.reshape((lq, inner))?, &kp.reshape((lk, inner))?, &vp.reshape((lk, inner))?, self.heads)? {
            return self.o.forward(&o.reshape((1, lq, inner))?);
        }
        let split = |t: Tensor, l: usize| -> Result<Tensor> {
            let inner = t.dim(2)?;
            t.reshape((1, l, self.heads, inner / self.heads))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(self.q.forward(q)?, lq)?, split(self.k.forward(k)?, lk)?, split(self.v.forward(v)?, lk)?);
        let o = super::nn::sdpa(&q, &k, &v, None)?;
        let inner = o.dim(1)? * o.dim(3)?;
        self.o.forward(&o.transpose(1, 2)?.reshape((1, lq, inner))?)
    }
}

struct Block {
    self_attn: Attn,
    ln1: LayerNorm,
    t2i: Attn,
    ln2: LayerNorm,
    mlp: Ff,
    ln3: LayerNorm,
    ln4: LayerNorm,
    i2t: Attn,
}

struct Decoder {
    obj_token: Tensor,
    iou_token: Tensor,
    mask_tokens: Tensor,
    blocks: Vec<Block>,
    final_attn: Attn,
    final_ln: LayerNorm,
    up1: Up2,
    up_ln_w: Tensor,
    up_ln_b: Tensor,
    up2: Up2,
    hyper: Vec<Ff>,
    iou_head: Ff,
    obj_head: Ff,
    conv_s0: FastConv,
    conv_s1: FastConv,
}

/// Decoder output for one prompt set.
pub struct Prediction {
    /// mask logits `[M, 288, 288]`
    pub masks: Tensor,
    /// the mask token the object pointer is derived from `[256]`: of the best-IoU candidate when `multimask`, else token 0
    pub token: Tensor,
    pub iou: Vec<f32>,
    pub object_score: f32,
}

pub struct Tracker {
    /// feature grid side (72 for SAM 3, 64 for SAM 2) and input size in pixels (1008 / 1024)
    side: usize,
    img: f32,
    gauss: Tensor,
    point_embed: Tensor,
    not_a_point: Tensor,
    no_mask: Tensor,
    no_memory: Tensor,
    dec: Decoder,
    image_pe: Tensor,
}

/// Image features for the tracker: skip features and the image embedding.
pub struct TrackerFeatures {
    pub s0: Tensor,
    pub s1: Tensor,
    /// image embedding for a prompted frame: level 2 plus the learned no-memory embedding
    pub emb: Tensor,
    /// level 2 as produced by the neck `[1, 256, 72, 72]` (what the memory encoder and memory attention consume)
    pub raw: Tensor,
}

impl Tracker {
    /// `m` is the weight root holding `prompt_encoder.*`, `mask_decoder.*` and `no_memory_embedding`
    /// (`tracker_model.` in the SAM 3 checkpoint, the checkpoint root for SAM 2).
    pub fn new(m: VarBuilder, side: usize, img: usize, device: &Device, dtype: DType) -> Result<Self> {
        let md = m.pp("mask_decoder");
        let pe = m.pp("prompt_encoder");
        let t = md.pp("transformer");
        let blocks = (0..2)
            .map(|i| {
                let l = t.pp(format!("layers.{i}"));
                Ok(Block {
                    self_attn: Attn::new(1, l.pp("self_attn"))?,
                    ln1: ln(D, 1e-5, l.pp("layer_norm1"))?,
                    t2i: Attn::new(2, l.pp("cross_attn_token_to_image"))?,
                    ln2: ln(D, 1e-5, l.pp("layer_norm2"))?,
                    mlp: Ff::new(D, 2048, D, 2, false, l.pp("mlp"))?,
                    ln3: ln(D, 1e-5, l.pp("layer_norm3"))?,
                    ln4: ln(D, 1e-5, l.pp("layer_norm4"))?,
                    i2t: Attn::new(2, l.pp("cross_attn_image_to_token"))?,
                })
            })
            .collect::<Result<_>>()?;
        let dec = Decoder {
            obj_token: md.get((1, D), "obj_score_token.weight")?,
            iou_token: md.get((1, D), "iou_token.weight")?,
            mask_tokens: md.get((4, D), "mask_tokens.weight")?,
            blocks,
            final_attn: Attn::new(2, t.pp("final_attn_token_to_image"))?,
            final_ln: ln(D, 1e-5, t.pp("layer_norm_final_attn"))?,
            up1: Up2::new(D, D / 4, md.pp("upscale_conv1"))?,
            up_ln_w: md.get(D / 4, "upscale_layer_norm.weight")?,
            up_ln_b: md.get(D / 4, "upscale_layer_norm.bias")?,
            up2: Up2::new(D / 4, D / 8, md.pp("upscale_conv2"))?,
            hyper: (0..4).map(|i| Ff::new(D, D, D / 8, 3, false, md.pp(format!("output_hypernetworks_mlps.{i}")))).collect::<Result<_>>()?,
            iou_head: Ff::new(D, D, 4, 3, true, md.pp("iou_prediction_head"))?,
            obj_head: Ff::new(D, D, 1, 3, false, md.pp("pred_obj_score_head"))?,
            conv_s0: FastConv::new(D, D / 8, 1, md.pp("conv_s0"))?,
            conv_s1: FastConv::new(D, D / 4, 1, md.pp("conv_s1"))?,
        };
        let gauss = pe.get((2, 128), "shared_embedding.positional_embedding")?;
        let mut me = Self {
            side,
            img: img as f32,
            gauss,
            point_embed: pe.get((4, D), "point_embed.weight")?,
            not_a_point: pe.get((1, D), "not_a_point_embed.weight")?.reshape(D)?,
            no_mask: pe.get((1, D), "no_mask_embed.weight")?.reshape(D)?,
            no_memory: m.get((1, 1, D), "no_memory_embedding")?.reshape(D)?,
            dec,
            image_pe: Tensor::zeros(1, dtype, device)?,
        };
        me.image_pe = me.grid_pe(device, dtype)?;
        Ok(me)
    }

    /// coords in [0, 1] `[..., 2]` -> `[..., 256]`
    fn pe(&self, c: &Tensor) -> Result<Tensor> {
        let x = (c.affine(2.0, -1.0)?).broadcast_matmul(&self.gauss.to_dtype(c.dtype())?)?;
        let x = (x * (2.0 * std::f64::consts::PI))?;
        Tensor::cat(&[x.sin()?, x.cos()?], candle_core::D::Minus1)
    }

    /// Dense positional encoding of the 72x72 grid, `[1, 256, 72, 72]`.
    fn grid_pe(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let mut g = Vec::with_capacity(self.side * self.side * 2);
        let side = self.side;
        for y in 0..side {
            for x in 0..side {
                g.push((x as f32 + 0.5) / side as f32);
                g.push((y as f32 + 0.5) / side as f32);
            }
        }
        let g = Tensor::from_vec(g, (side, side, 2), device)?.to_dtype(dtype)?;
        self.pe(&g)?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()
    }

    /// Tracker features from the three neck levels (highest resolution first, each `[1, 256, h, w]`).
    pub fn features(&self, f: &[Tensor]) -> Result<TrackerFeatures> {
        let s0 = self.dec.conv_s0.forward(&f[0])?;
        let s1 = self.dec.conv_s1.forward(&f[1])?;
        let emb = f[2].broadcast_add(&self.no_memory.to_dtype(f[2].dtype())?.reshape((1, D, 1, 1))?)?;
        Ok(TrackerFeatures { s0, s1, emb, raw: f[2].clone() })
    }

    /// Sparse prompt tokens `[1, N, 256]`. `points`: `(x, y, label)` and `boxes`: `[x0, y0, x1, y1]`, in the
    /// 1008-px input space; labels 1 / 0 positive / negative.
    pub fn sparse(&self, points: &[(f32, f32, i32)], boxes: &[[f32; 4]], device: &Device, dt: DType) -> Result<Tensor> {
        let img = self.img;
        let norm = |x: f32, y: f32| [(x + 0.5) / img, (y + 0.5) / img];
        let mut rows: Vec<Tensor> = vec![];
        let mut pts: Vec<(f32, f32, i32)> = points.to_vec();
        if pts.is_empty() && boxes.is_empty() {
            // the reference substitutes one "no point" input when there is no prompt at all (and still pads below)
            pts.push((0.0, 0.0, -1));
        }
        if boxes.is_empty() {
            pts.push((-0.5, -0.5, -1)); // padding point
        }
        if !pts.is_empty() {
            let xy: Vec<f32> = pts.iter().flat_map(|p| norm(p.0, p.1)).collect();
            let pe = self.pe(&Tensor::from_vec(xy, (pts.len(), 2), device)?.to_dtype(dt)?)?;
            for (i, p) in pts.iter().enumerate() {
                rows.push(if p.2 == -1 {
                    self.not_a_point.to_dtype(dt)?
                } else {
                    (pe.get(i)? + self.point_embed.get(p.2.clamp(0, 3) as usize)?.to_dtype(dt)?)?
                });
            }
        }
        for b in boxes {
            let xy: Vec<f32> = [norm(b[0], b[1]), norm(b[2], b[3])].concat();
            let pe = self.pe(&Tensor::from_vec(xy, (2, 2), device)?.to_dtype(dt)?)?;
            rows.push((pe.get(0)? + self.point_embed.get(2)?.to_dtype(dt)?)?);
            rows.push((pe.get(1)? + self.point_embed.get(3)?.to_dtype(dt)?)?);
            rows.push(self.not_a_point.to_dtype(dt)?);
        }
        Tensor::stack(&rows, 0)?.unsqueeze(0)
    }

    pub fn predict_sparse(&self, f: &TrackerFeatures, sparse: &Tensor, multimask: bool) -> Result<Prediction> {
        let d = &self.dec;
        let dt = f.emb.dtype();
        let out_tokens = Tensor::cat(&[&d.obj_token, &d.iou_token, &d.mask_tokens], 0)?.to_dtype(dt)?.unsqueeze(0)?;
        let tokens = Tensor::cat(&[&out_tokens, sparse], 1)?; // [1, 6 + N, 256]
        let src = f.emb.broadcast_add(&self.no_mask.to_dtype(dt)?.reshape((1, D, 1, 1))?)?;
        let mut keys = src.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let kpe = self.image_pe.to_dtype(dt)?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let mut queries = tokens.clone();
        for (i, b) in d.blocks.iter().enumerate() {
            queries = if i == 0 {
                b.self_attn.forward(&queries, &queries, &queries)?
            } else {
                let q = (&queries + &tokens)?;
                (&queries + b.self_attn.forward(&q, &q, &queries)?)?
            };
            queries = b.ln1.forward(&queries)?;
            let (q, k) = ((&queries + &tokens)?, add_par(&keys, &kpe)?);
            queries = b.ln2.forward(&(&queries + b.t2i.forward(&q, &k, &keys)?)?)?;
            queries = b.ln3.forward(&(&queries + b.mlp.forward(&queries)?)?)?;
            let q = (&queries + &tokens)?;
            keys = layer_norm_rows(&add_par(&keys, &b.i2t.forward(&k, &q, &queries)?)?, b.ln4.weight(), b.ln4.bias().expect("layer norm has a bias"), 1e-5)?;
        }
        let (q, k) = ((&queries + &tokens)?, add_par(&keys, &kpe)?);
        let queries = d.final_ln.forward(&(&queries + d.final_attn.forward(&q, &k, &keys)?)?)?;

        let obj_tok = queries.narrow(1, 0, 1)?;
        let iou_tok = queries.narrow(1, 1, 1)?;
        let mask_toks = queries.narrow(1, 2, 4)?;
        let src = keys.transpose(1, 2)?.reshape((1, D, self.side, self.side))?;
        let up = layer_norm_2d(&(d.up1.forward(&src, false)? + &f.s1)?, &d.up_ln_w.to_dtype(dt)?, &d.up_ln_b.to_dtype(dt)?, 1e-6)?.gelu_erf()?;
        let up = (d.up2.forward(&up, false)? + &f.s0)?.gelu_erf()?;
        let (c, h, w) = (up.dim(1)?, up.dim(2)?, up.dim(3)?);
        let hyper = Tensor::stack(&(0..4).map(|i| d.hyper[i].forward(&mask_toks.narrow(1, i, 1)?.squeeze(1)?)).collect::<Result<Vec<_>>>()?, 1)?; // [1,4,32]
        let all = hyper.matmul(&up.reshape((1, c, h * w))?)?.reshape((4, h, w))?;
        let iou = d.iou_head.forward(&iou_tok.squeeze(1)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let obj = d.obj_head.forward(&obj_tok.squeeze(1)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0];
        let (masks, iou) = if multimask {
            (all.narrow(0, 1, 3)?, iou[1..].to_vec())
        } else {
            self.single(&all, &iou)?
        };
        let best = if multimask { 1 + (0..3).max_by(|&a, &b| iou[a].total_cmp(&iou[b])).expect("three candidates") } else { 0 };
        let token = mask_toks.narrow(1, best, 1)?.reshape(D)?;
        Ok(Prediction { masks, token, iou, object_score: obj })
    }

    /// Single-mask output with the stability fallback: keep token 0 if its mask is stable, else the best of the others.
    fn single(&self, all: &Tensor, iou: &[f32]) -> Result<(Tensor, Vec<f32>)> {
        let (delta, thresh) = (0.05f32, 0.98f32);
        let m0 = all.narrow(0, 0, 1)?;
        let area_i = m0.gt(delta)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        let area_u = m0.gt(-delta)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        let stability = if area_u > 0.0 { area_i / area_u } else { 1.0 };
        if stability >= thresh {
            return Ok((m0, vec![iou[0]]));
        }
        let best = (1..4).max_by(|&a, &b| iou[a].total_cmp(&iou[b])).expect("three candidates");
        Ok((all.narrow(0, best, 1)?, vec![iou[best]]))
    }
}

