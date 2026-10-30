//! Geometry (box) prompt encoder: each exemplar box is embedded from its raw coordinates, ROI-Align pooled
//! image features and a sine embedding of its centre, then refined with self- and image cross-attention.

use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, LayerNorm, Linear, Module, VarBuilder};

use super::detr::sine_1d;
use super::nn::{ln, Act, Attn, Mlp};

const D: usize = 256;
const ROI: usize = 7;

struct Layer {
    ln1: LayerNorm,
    self_attn: Attn,
    ln2: LayerNorm,
    cross_attn: Attn,
    ln3: LayerNorm,
    mlp: Mlp,
}

pub struct GeometryEncoder {
    label_embed: Tensor,
    cls_embed: Tensor,
    direct: Linear,
    /// `[256 * 7 * 7, 256]`: the 7x7 pooling convolution as a matrix
    pool_w: Tensor,
    pool_b: Tensor,
    pos_proj: Linear,
    vision_ln: LayerNorm,
    final_proj: Linear,
    prompt_ln: LayerNorm,
    layers: Vec<Layer>,
    out_ln: LayerNorm,
}

/// torchvision `roi_align` (aligned=False, adaptive sampling ratio) on an HWC feature map.
/// `boxes`: `(x1, y1, x2, y2)` in feature-map coordinates. Returns `[N, C * ROI * ROI]` ordered (c, ph, pw).
fn roi_align(feat: &[f32], h: usize, w: usize, c: usize, boxes: &[[f32; 4]]) -> Vec<f32> {
    let bilinear = |y: f32, x: f32, ch: usize| -> f32 {
        if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
            return 0.0;
        }
        let (mut y, mut x) = (y.max(0.0), x.max(0.0));
        let mut y_low = y as usize;
        let mut x_low = x as usize;
        let (y_high, x_high);
        if y_low >= h - 1 {
            y_high = h - 1;
            y_low = h - 1;
            y = y_low as f32;
        } else {
            y_high = y_low + 1;
        }
        if x_low >= w - 1 {
            x_high = w - 1;
            x_low = w - 1;
            x = x_low as f32;
        } else {
            x_high = x_low + 1;
        }
        let (ly, lx) = (y - y_low as f32, x - x_low as f32);
        let (hy, hx) = (1.0 - ly, 1.0 - lx);
        let at = |yy: usize, xx: usize| feat[(yy * w + xx) * c + ch];
        hy * hx * at(y_low, x_low) + hy * lx * at(y_low, x_high) + ly * hx * at(y_high, x_low) + ly * lx * at(y_high, x_high)
    };
    let mut out = vec![0f32; boxes.len() * c * ROI * ROI];
    for (n, b) in boxes.iter().enumerate() {
        let (roi_w, roi_h) = ((b[2] - b[0]).max(1.0), (b[3] - b[1]).max(1.0));
        let (bin_h, bin_w) = (roi_h / ROI as f32, roi_w / ROI as f32);
        let (gh, gw) = ((roi_h / ROI as f32).ceil() as usize, (roi_w / ROI as f32).ceil() as usize);
        let count = (gh * gw).max(1) as f32;
        for ch in 0..c {
            for ph in 0..ROI {
                for pw in 0..ROI {
                    let mut sum = 0f32;
                    for iy in 0..gh {
                        let y = b[1] + ph as f32 * bin_h + (iy as f32 + 0.5) * bin_h / gh as f32;
                        for ix in 0..gw {
                            let x = b[0] + pw as f32 * bin_w + (ix as f32 + 0.5) * bin_w / gw as f32;
                            sum += bilinear(y, x, ch);
                        }
                    }
                    out[((n * c + ch) * ROI + ph) * ROI + pw] = sum / count;
                }
            }
        }
    }
    out
}

impl GeometryEncoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layers = (0..3)
            .map(|i| {
                let l = vb.pp(format!("layers.{i}"));
                Ok(Layer {
                    ln1: ln(D, 1e-5, l.pp("layer_norm1"))?,
                    self_attn: Attn::new(D, 8, l.pp("self_attn"))?,
                    ln2: ln(D, 1e-5, l.pp("layer_norm2"))?,
                    cross_attn: Attn::new(D, 8, l.pp("cross_attn"))?,
                    ln3: ln(D, 1e-5, l.pp("layer_norm3"))?,
                    mlp: Mlp::new(D, 2048, Act::Relu, l.pp("mlp"))?,
                })
            })
            .collect::<Result<_>>()?;
        let pool = vb.pp("boxes_pool_project");
        Ok(Self {
            label_embed: vb.get((2, D), "label_embed.weight")?,
            cls_embed: vb.get((1, D), "cls_embed.weight")?,
            direct: linear(4, D, vb.pp("boxes_direct_project"))?,
            pool_w: pool.get((D, D, ROI, ROI), "weight")?.reshape((D, D * ROI * ROI))?.t()?.contiguous()?,
            pool_b: pool.get(D, "bias")?,
            pos_proj: linear(D + 2, D, vb.pp("boxes_pos_enc_project"))?,
            vision_ln: ln(D, 1e-5, vb.pp("vision_layer_norm"))?,
            final_proj: linear(D, D, vb.pp("final_proj"))?,
            prompt_ln: ln(D, 1e-5, vb.pp("prompt_layer_norm"))?,
            layers,
            out_ln: ln(D, 1e-5, vb.pp("output_layer_norm"))?,
        })
    }

    /// `boxes`: normalized `(cx, cy, w, h)`; `labels`: 1 positive / 0 negative; `fpn`: the 72x72 level
    /// `[1, 256, 72, 72]`; `vpos`: its sine positions as tokens `[1, 5184, 256]`. Returns `[1, N+1, 256]`
    /// (the boxes followed by the CLS token).
    pub fn forward(&self, boxes: &[[f32; 4]], labels: &[u32], fpn: &Tensor, vpos: &Tensor) -> Result<Tensor> {
        let (dev, dt) = (fpn.device(), fpn.dtype());
        let (h, w) = (fpn.dim(2)?, fpn.dim(3)?);
        let n = boxes.len();
        let tokens = fpn.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // [1, hw, 256]
        let normed = self.vision_ln.forward(&tokens)?;

        // ROI-Align runs on the host: a handful of boxes over a 72x72 map
        let feat = normed.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let xyxy: Vec<[f32; 4]> = boxes
            .iter()
            .map(|b| [(b[0] - b[2] / 2.0) * w as f32, (b[1] - b[3] / 2.0) * h as f32, (b[0] + b[2] / 2.0) * w as f32, (b[1] + b[3] / 2.0) * h as f32])
            .collect();
        let pooled = Tensor::from_vec(roi_align(&feat, h, w, D, &xyxy), (n, D * ROI * ROI), dev)?.to_dtype(dt)?;
        let pooled = pooled.matmul(&self.pool_w.to_dtype(dt)?)?.broadcast_add(&self.pool_b.to_dtype(dt)?)?;

        let b = Tensor::from_vec(boxes.iter().flatten().copied().collect::<Vec<_>>(), (1, n, 4), dev)?.to_dtype(dt)?;
        let col = |i: usize| b.narrow(2, i, 1).and_then(|t| t.squeeze(2));
        // position encoding of the centre, then the raw height and width
        let pos = Tensor::cat(&[sine_1d(&col(1)?)?, sine_1d(&col(0)?)?, col(3)?.unsqueeze(2)?, col(2)?.unsqueeze(2)?], 2)?;
        let ids = Tensor::from_vec(labels.to_vec(), n, dev)?;
        let label = self.label_embed.to_dtype(dt)?.index_select(&ids, 0)?.unsqueeze(0)?;
        let embed = (((self.direct.forward(&b)? + pooled.unsqueeze(0)?)? + self.pos_proj.forward(&pos)?)? + label)?;

        let mut x = Tensor::cat(&[embed, self.cls_embed.to_dtype(dt)?.unsqueeze(0)?], 1)?;
        x = self.prompt_ln.forward(&self.final_proj.forward(&x)?)?;
        let key = (&tokens + vpos)?;
        for l in &self.layers {
            let y = l.ln1.forward(&x)?;
            x = (&x + l.self_attn.forward(&y, &y, &y, None)?)?;
            let y = l.ln2.forward(&x)?;
            x = (&x + l.cross_attn.forward(&y, &key, &tokens, None)?)?;
            x = (&x + l.mlp.forward(&l.ln3.forward(&x)?)?)?;
        }
        self.out_ln.forward(&x)
    }
}
