//! EfficientSAM (ViT-Tiny / ViT-Small encoders, SAM-style prompt encoder and mask decoder).
//!
//! The architecture follows the ONNX exports published for the model; `examples/efficientsam_parity.rs` compares
//! against those graphs run through onnxruntime (`scripts/ref_efficientsam.py`).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, conv2d_no_bias, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};
use image::RgbImage;

use crate::cpu_gemm::Linear as FastLinear;
use crate::kernels::{bias_gelu, bias_residual};
use crate::sam1::encoder::LayerNorm2d;
use crate::sam2::hiera::bicubic;
use crate::sam3::nn::{ln, sdpa};

const SIZE: usize = 1024;
const SIDE: usize = 64;
const D: usize = 256;
const MAX_POINTS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Tiny,
    Small,
}

impl Variant {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "efficientsam-ti" | "efficientsam-tiny" | "efficient-sam-ti" | "efficientsam:10m" => Some(Self::Tiny),
            "efficientsam-s" | "efficientsam-small" | "efficient-sam-s" | "efficientsam:latest" => Some(Self::Small),
            _ => None,
        }
    }

    fn dims(self) -> (usize, usize, usize) {
        // (embed dim, depth, heads)
        match self {
            Self::Tiny => (192, 12, 3),
            Self::Small => (384, 12, 6),
        }
    }

    /// Checkpoint name without extension (`efficient_sam_vitt`).
    pub fn file(self) -> &'static str {
        match self {
            Self::Tiny => "efficient_sam_vitt",
            Self::Small => "efficient_sam_vits",
        }
    }
}

struct Block {
    n1: LayerNorm,
    qkv_w: FastLinear,
    qkv_b: Tensor,
    proj_w: FastLinear,
    proj_b: Tensor,
    n2: LayerNorm,
    fc1_w: FastLinear,
    fc1_b: Tensor,
    fc2_w: FastLinear,
    fc2_b: Tensor,
    heads: usize,
}

/// `[out, in]` checkpoint weight and bias -> (`x @ w` layer, bias).
fn fast_linear(vb: VarBuilder, inp: usize, out: usize) -> candle_core::Result<(FastLinear, Tensor)> {
    Ok((FastLinear::new(vb.get((out, inp), "weight")?.t()?.contiguous()?)?, vb.get(out, "bias")?))
}

impl Block {
    fn new(dim: usize, heads: usize, l: VarBuilder) -> candle_core::Result<Self> {
        let (qkv_w, qkv_b) = fast_linear(l.pp("attn.qkv"), dim, dim * 3)?;
        let (proj_w, proj_b) = fast_linear(l.pp("attn.proj"), dim, dim)?;
        let (fc1_w, fc1_b) = fast_linear(l.pp("mlp.fc1"), dim, dim * 4)?;
        let (fc2_w, fc2_b) = fast_linear(l.pp("mlp.fc2"), dim * 4, dim)?;
        Ok(Self { n1: ln(dim, 1e-6, l.pp("norm1"))?, qkv_w, qkv_b, proj_w, proj_b, n2: ln(dim, 1e-6, l.pp("norm2"))?, fc1_w, fc1_b, fc2_w, fc2_b, heads })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let n = h * w;
        let hd = c / self.heads;
        let y = self.n1.forward(x)?.reshape((n, c))?;
        let qkv = self.qkv_w.forward(&y)?.broadcast_add(&self.qkv_b)?.reshape((b, n, 3, self.heads, hd))?;
        let part = |i: usize| qkv.narrow(2, i, 1)?.squeeze(2)?.contiguous(); // [b, n, heads, hd]
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let o = match crate::kernels::flash_attn_cpu(&q, &k, &v)? {
            Some(o) => o,
            None => {
                let t = |t: &Tensor| t.transpose(1, 2);
                sdpa(&t(&q)?, &t(&k)?, &t(&v)?, None)?.transpose(1, 2)?
            }
        };
        let o = self.proj_w.forward(&o.reshape((n, c))?)?;
        let x = bias_residual(&x.reshape((n, c))?, &o, &self.proj_b)?;
        let m = bias_gelu(&self.fc1_w.forward(&self.n2.forward(&x)?)?, &self.fc1_b)?;
        bias_residual(&x, &self.fc2_w.forward(&m)?, &self.fc2_b)?.reshape((b, h, w, c))
    }
}

struct Encoder {
    patch: Conv2d,
    pos: Tensor,
    blocks: Vec<Block>,
    neck1: Conv2d,
    neck_ln1: LayerNorm2d,
    neck2: Conv2d,
    neck_ln2: LayerNorm2d,
}

impl Encoder {
    fn new(v: Variant, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        let (dim, depth, heads) = v.dims();
        let patch = conv2d(3, dim, 16, Conv2dConfig { stride: 16, ..Default::default() }, vb.pp("patch_embed.proj"))?;
        // the 14x14 pretraining grid (cls token dropped) is resized bicubically to 64x64
        let pe = vb.get((1, 197, dim), "pos_embed")?.to_dtype(DType::F32)?.narrow(1, 1, 196)?.reshape((14, 14, dim))?.permute((2, 0, 1))?.contiguous()?;
        let pos = bicubic(&pe.flatten_all()?.to_vec1::<f32>()?, dim, 14, 14, SIDE, SIDE);
        let pos = Tensor::from_vec(pos, (1, dim, SIDE, SIDE), device)?.permute((0, 2, 3, 1))?.contiguous()?.to_dtype(vb.dtype())?;
        let blocks = (0..depth).map(|i| Block::new(dim, heads, vb.pp(format!("blocks.{i}")))).collect::<candle_core::Result<_>>()?;
        Ok(Self {
            patch,
            pos,
            blocks,
            neck1: conv2d_no_bias(dim, D, 1, Default::default(), vb.pp("neck.0"))?,
            neck_ln1: LayerNorm2d::new(D, 1e-6, vb.pp("neck.1"))?,
            neck2: conv2d_no_bias(D, D, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("neck.2"))?,
            neck_ln2: LayerNorm2d::new(D, 1e-6, vb.pp("neck.3"))?,
        })
    }

    /// `[1, 3, 1024, 1024]` normalized -> `[1, 256, 64, 64]`
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.patch.forward(x)?.permute((0, 2, 3, 1))?.broadcast_add(&self.pos)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let x = self.neck_ln1.forward(&self.neck1.forward(&x)?)?;
        self.neck_ln2.forward(&self.neck2.forward(&x)?)
    }
}

// ------------------------------------------------------------------------------------------------ decoder

/// `Linear -> GELU` layers followed by a final linear `fc` (the checkpoint's `layers.N.0` + `fc` naming).
struct Mlp {
    layers: Vec<Linear>,
    fc: Linear,
}

impl Mlp {
    fn new(i: usize, h: usize, o: usize, n: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self { layers: (0..n).map(|j| linear(if j == 0 { i } else { h }, h, vb.pp(format!("layers.{j}.0")))).collect::<candle_core::Result<_>>()?, fc: linear(h, o, vb.pp("fc"))? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for l in &self.layers {
            x = l.forward(&x)?.gelu_erf()?;
        }
        self.fc.forward(&x)
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attn {
    fn new(inner: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self { q: linear(D, inner, vb.pp("q_proj"))?, k: linear(D, inner, vb.pp("k_proj"))?, v: linear(D, inner, vb.pp("v_proj"))?, o: linear(inner, D, vb.pp("out_proj"))?, heads: 8 })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
        let (lq, lk) = (q.dim(1)?, k.dim(1)?);
        let split = |t: Tensor, l: usize| -> candle_core::Result<Tensor> {
            let inner = t.dim(2)?;
            t.reshape((1, l, self.heads, inner / self.heads))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(self.q.forward(q)?, lq)?, split(self.k.forward(k)?, lk)?, split(self.v.forward(v)?, lk)?);
        let o = sdpa(&q, &k, &v, None)?;
        let inner = o.dim(1)? * o.dim(3)?;
        self.o.forward(&o.transpose(1, 2)?.reshape((1, lq, inner))?)
    }
}

struct Layer {
    self_attn: Attn,
    n1: LayerNorm,
    t2i: Attn,
    n2: LayerNorm,
    mlp_in: Linear,
    mlp_fc: Linear,
    n3: LayerNorm,
    n4: LayerNorm,
    i2t: Attn,
}

struct Decoder {
    iou_token: Tensor,
    mask_tokens: Tensor,
    layers: Vec<Layer>,
    final_attn: Attn,
    final_ln: LayerNorm,
    up1: crate::sam1::decoder::Up2,
    up_ln: (Tensor, Tensor),
    up2: crate::sam1::decoder::Up2,
    hyper: Vec<Mlp>,
    iou: Mlp,
}

pub struct Prediction {
    /// logits `[3, 256, 256]`
    pub masks: Tensor,
    pub iou: Vec<f32>,
}

pub struct EfficientSam {
    enc: Encoder,
    gauss: Tensor,
    invalid: Tensor,
    positive: Tensor,
    top_left: Tensor,
    bottom_right: Tensor,
    dec: Decoder,
    image_pe: Tensor,
    pub device: Device,
    pub dtype: DType,
}

pub struct Embedding {
    emb: Tensor,
    orig: (usize, usize),
}

impl Embedding {
    /// The `[1, 256, 64, 64]` image embedding (exposed for tests).
    pub fn tensor(&self) -> &Tensor {
        &self.emb
    }
}

/// Opens a `.safetensors` file, or an upstream PyTorch `.pt`/`.pth` checkpoint (tensors only, nothing is executed).
fn load_vb(path: &std::path::Path, device: &Device, dtype: DType) -> candle_core::Result<VarBuilder<'static>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pt" | "pth" | "bin") => {
            // some releases nest the weights under a "model" key
            match VarBuilder::from_pth(path, dtype, device) {
                Ok(vb) if vb.contains_tensor("image_encoder.patch_embed.proj.weight") => Ok(vb),
                _ => VarBuilder::from_pth_with_state(path, dtype, "model", device),
            }
        }
        _ => unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) },
    }
}

impl EfficientSam {
    pub fn from_file(v: Variant, weights: &std::path::Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = load_vb(weights, device, dtype)?;
        let pe = vb.pp("prompt_encoder");
        let md = vb.pp("mask_decoder");
        let t = md.pp("transformer");
        let layers = (0..2)
            .map(|i| {
                let l = t.pp(format!("layers.{i}"));
                Ok(Layer {
                    self_attn: Attn::new(D, l.pp("self_attn"))?,
                    n1: ln(D, 1e-5, l.pp("norm1"))?,
                    t2i: Attn::new(D / 2, l.pp("cross_attn_token_to_image"))?,
                    n2: ln(D, 1e-5, l.pp("norm2"))?,
                    mlp_in: linear(D, 2048, l.pp("mlp.layers.0.0"))?,
                    mlp_fc: linear(2048, D, l.pp("mlp.fc"))?,
                    n3: ln(D, 1e-5, l.pp("norm3"))?,
                    n4: ln(D, 1e-5, l.pp("norm4"))?,
                    i2t: Attn::new(D / 2, l.pp("cross_attn_image_to_token"))?,
                })
            })
            .collect::<candle_core::Result<_>>()?;
        let up = md.pp("final_output_upscaling_layers");
        let dec = Decoder {
            iou_token: md.get((1, D), "iou_token.weight")?,
            mask_tokens: md.get((4, D), "mask_tokens.weight")?,
            layers,
            final_attn: Attn::new(D / 2, t.pp("final_attn_token_to_image"))?,
            final_ln: ln(D, 1e-5, t.pp("norm_final_attn"))?,
            up1: crate::sam1::decoder::Up2::new(D, D / 4, up.pp("0.0"))?,
            up_ln: (up.get(D / 4, "0.1.weight")?, up.get(D / 4, "0.1.bias")?),
            up2: crate::sam1::decoder::Up2::new(D / 4, D / 8, up.pp("1.0"))?,
            hyper: (0..4).map(|i| Mlp::new(D, D, D / 8, 2, md.pp(format!("output_hypernetworks_mlps.{i}")))).collect::<candle_core::Result<_>>()?,
            iou: Mlp::new(D, D, 4, 2, md.pp("iou_prediction_head"))?,
        };
        let mut me = Self {
            enc: Encoder::new(v, vb.pp("image_encoder"), device)?,
            gauss: pe.get((2, 128), "pe_layer.positional_encoding_gaussian_matrix")?,
            invalid: pe.get((1, D), "invalid_points.weight")?.reshape(D)?,
            positive: pe.get((1, D), "point_embeddings.weight")?.reshape(D)?,
            top_left: pe.get((1, D), "bbox_top_left_embeddings.weight")?.reshape(D)?,
            bottom_right: pe.get((1, D), "bbox_bottom_right_embeddings.weight")?.reshape(D)?,
            dec,
            image_pe: Tensor::zeros(1, dtype, device)?,
            device: device.clone(),
            dtype,
        };
        me.image_pe = me.grid_pe()?;
        Ok(me)
    }

    /// Downloads the upstream `.pt` checkpoint on first use (a converted `.safetensors` in the cache is preferred).
    pub fn from_pretrained(v: Variant, device: &Device, dtype: DType) -> Result<Self> {
        let path = crate::download::fetch_efficientsam(v.file())?;
        Self::from_file(v, &path, device, dtype)
    }

    fn pe(&self, c: &Tensor) -> candle_core::Result<Tensor> {
        let x = (c.affine(2.0, -1.0)?).broadcast_matmul(&self.gauss.to_dtype(c.dtype())?)?;
        let x = (x * (2.0 * std::f64::consts::PI))?;
        Tensor::cat(&[x.sin()?, x.cos()?], candle_core::D::Minus1)
    }

    fn grid_pe(&self) -> candle_core::Result<Tensor> {
        let mut g = Vec::with_capacity(SIDE * SIDE * 2);
        for y in 0..SIDE {
            for x in 0..SIDE {
                g.push((x as f32 + 0.5) / SIDE as f32);
                g.push((y as f32 + 0.5) / SIDE as f32);
            }
        }
        let g = Tensor::from_vec(g, (SIDE, SIDE, 2), &self.device)?.to_dtype(self.dtype)?;
        self.pe(&g)?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    /// Encoder on an already-resized, normalized `[1, 3, 1024, 1024]` tensor (exposed for tests).
    pub fn encode_pixels(&self, px: &Tensor) -> Result<Tensor> {
        Ok(self.enc.forward(&px.to_dtype(self.dtype)?)?)
    }

    pub fn embed(&self, img: &RgbImage) -> Result<Embedding> {
        let px = crate::image_proc::prepare_square_plain(img, SIZE, [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0], [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], &self.device, self.dtype)?;
        Ok(Embedding { emb: self.encode_pixels(&px)?, orig: (img.height() as usize, img.width() as usize) })
    }

    /// Sparse tokens `[1, N, 256]` for points `(x, y, label)` in the 1024-px input space.
    /// Labels: -1 invalid, 0 negative, 1 positive, 2 box top-left, 3 box bottom-right.
    pub fn sparse(&self, points: &[(f32, f32, i32)]) -> Result<Tensor> {
        let dt = self.dtype;
        // the graph always feeds exactly 6 prompt tokens: at most the first 6 points, padded with invalid points
        // (coordinate -1, label -1) up to 6
        let mut points: Vec<(f32, f32, i32)> = points.iter().take(MAX_POINTS).copied().collect();
        points.resize(MAX_POINTS, (-1.0, -1.0, -1));
        let points = &points[..];
        let xy: Vec<f32> = points.iter().flat_map(|p| [(p.0 + 0.5) / SIZE as f32, (p.1 + 0.5) / SIZE as f32]).collect();
        let pe = self.pe(&Tensor::from_vec(xy, (points.len(), 2), &self.device)?.to_dtype(dt)?)?;
        let mut rows = vec![];
        for (i, p) in points.iter().enumerate() {
            let extra = match p.2 {
                -1 => Some(&self.invalid),
                1 => Some(&self.positive),
                2 => Some(&self.top_left),
                3 => Some(&self.bottom_right),
                _ => None,
            };
            rows.push(match extra {
                Some(e) => (pe.get(i)? + e.to_dtype(dt)?)?,
                None => pe.get(i)?,
            });
        }
        Ok(Tensor::stack(&rows, 0)?.unsqueeze(0)?)
    }

    /// Decoder on an embedding and sparse tokens: three candidate masks.
    pub fn decode(&self, emb: &Tensor, sparse: &Tensor) -> Result<Prediction> {
        self.decode_traced(emb, sparse, &mut Vec::new())
    }

    /// [`decode`](Self::decode) that also records every LayerNorm output in graph order (for tests).
    pub fn decode_traced(&self, emb: &Tensor, sparse: &Tensor, trace: &mut Vec<Tensor>) -> Result<Prediction> {
        let d = &self.dec;
        let dt = emb.dtype();
        let out_tokens = Tensor::cat(&[&d.iou_token, &d.mask_tokens], 0)?.to_dtype(dt)?.unsqueeze(0)?;
        let tokens = Tensor::cat(&[&out_tokens, sparse], 1)?;
        let mut keys = emb.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let kpe = self.image_pe.to_dtype(dt)?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let mut queries = tokens.clone();
        for (i, l) in d.layers.iter().enumerate() {
            // q = queries (+ prompt positions after the first layer); the residual is taken from q itself, and q, k and
            // v are all the same tensor
            let q = if i == 0 { queries.clone() } else { (&queries + &tokens)? };
            queries = (&q + l.self_attn.forward(&q, &q, &q)?)?;
            queries = l.n1.forward(&queries)?;
            trace.push(queries.clone());
            let (q, k) = ((&queries + &tokens)?, (&keys + &kpe)?);
            queries = l.n2.forward(&(&queries + l.t2i.forward(&q, &k, &keys)?)?)?;
            trace.push(queries.clone());
            let m = l.mlp_fc.forward(&l.mlp_in.forward(&queries)?.gelu_erf()?)?;
            queries = l.n3.forward(&(&queries + m)?)?;
            trace.push(queries.clone());
            let q = (&queries + &tokens)?;
            keys = l.n4.forward(&(&keys + l.i2t.forward(&k, &q, &queries)?)?)?;
            trace.push(keys.clone());
        }
        let (q, k) = ((&queries + &tokens)?, (&keys + &kpe)?);
        let queries = d.final_ln.forward(&(&queries + d.final_attn.forward(&q, &k, &keys)?)?)?;
        trace.push(queries.clone());
        let iou_tok = queries.narrow(1, 0, 1)?.squeeze(1)?;
        let mask_toks = queries.narrow(1, 1, 4)?;
        let src = keys.transpose(1, 2)?.reshape((1, D, SIDE, SIDE))?;
        // one-group GroupNorm: statistics over all channels and positions together, then a per-channel scale / shift
        let u0 = d.up1.forward(&src, false)?;
        let (c0, h0, w0) = (u0.dim(1)?, u0.dim(2)?, u0.dim(3)?);
        let u = u0.to_dtype(DType::F32)?.reshape((1, c0 * h0 * w0))?;
        let mean = u.mean_keepdim(1)?;
        let xc = u.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(1)?;
        let un = xc.broadcast_div(&(var + 1e-5)?.sqrt()?)?.reshape((1, c0, h0, w0))?;
        let un = un.broadcast_mul(&d.up_ln.0.to_dtype(DType::F32)?.reshape((1, c0, 1, 1))?)?.broadcast_add(&d.up_ln.1.to_dtype(DType::F32)?.reshape((1, c0, 1, 1))?)?;
        let up = un.to_dtype(dt)?.gelu_erf()?;
        let up = d.up2.forward(&up, false)?.gelu_erf()?;
        let (c, h, w) = (up.dim(1)?, up.dim(2)?, up.dim(3)?);
        let hyper = Tensor::stack(&(0..4).map(|i| d.hyper[i].forward(&mask_toks.narrow(1, i, 1)?.squeeze(1)?)).collect::<candle_core::Result<Vec<_>>>()?, 1)?;
        let all = hyper.matmul(&up.reshape((1, c, h * w))?)?.reshape((4, h, w))?;
        let iou = d.iou.forward(&iou_tok)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Prediction { masks: all.narrow(0, 1, 3)?, iou: iou[1..].to_vec() })
    }

    /// Point / box prompts in original-image pixels: `points` are `(x, y, positive)`, `boxes` `[x0, y0, x1, y1]`.
    /// Returns three candidate masks at the original resolution.
    pub fn predict(&self, e: &Embedding, points: &[(f32, f32, bool)], boxes: &[[f32; 4]]) -> Result<Vec<crate::Mask>> {
        let (oh, ow) = e.orig;
        let (sx, sy) = (SIZE as f32 / ow as f32, SIZE as f32 / oh as f32);
        let mut pts: Vec<(f32, f32, i32)> = points.iter().map(|&(x, y, fg)| (x * sx, y * sy, fg as i32)).collect();
        for b in boxes {
            pts.push((b[0] * sx, b[1] * sy, 2));
            pts.push((b[2] * sx, b[3] * sy, 3));
        }
        let p = self.decode(&e.emb, &self.sparse(&pts)?)?;
        let bin = crate::sam2::binarize(&p.masks, oh, ow)?;
        Ok(bin.into_iter().zip(p.iou).map(|(data, iou)| crate::Mask { width: ow, height: oh, data, iou }).collect())
    }
}

