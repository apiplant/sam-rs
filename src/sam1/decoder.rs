//! Two-way transformer mask decoder.

use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use std::sync::OnceLock;

use crate::kernels::{add_par, bias_gelu, layer_norm_rows, mha_cpu, nchw_add_to_tokens, pixel_shuffle2_cpu};

use super::encoder::LayerNorm2d;

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl Attn {
    fn new(dim: usize, internal: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: linear(dim, internal, vb.pp("q_proj"))?,
            k: linear(dim, internal, vb.pp("k_proj"))?,
            v: linear(dim, internal, vb.pp("v_proj"))?,
            out: linear(internal, dim, vb.pp("out_proj"))?,
            heads,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if q.dim(0)? == 1 {
            let (nq, nk) = (q.dim(1)?, k.dim(1)?);
            let (qp, kp, vp) = (self.q.forward(q)?, self.k.forward(k)?, self.v.forward(v)?);
            let e = qp.dim(2)?;
            if let Some(o) = mha_cpu(&qp.reshape((nq, e))?, &kp.reshape((nk, e))?, &vp.reshape((nk, e))?, self.heads)? {
                return self.out.forward(&o.reshape((1, nq, e))?);
            }
        }
        let split = |t: Tensor| -> Result<Tensor> {
            let (b, n, c) = t.dims3()?;
            t.reshape((b, n, self.heads, c / self.heads))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(self.q.forward(q)?)?, split(self.k.forward(k)?)?, split(self.v.forward(v)?)?);
        let hd = q.dim(3)?;
        let a = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (hd as f64).sqrt())?)?;
        let o = a.matmul(&v)?;
        let (b, h, n, d) = o.dims4()?;
        self.out.forward(&o.transpose(1, 2)?.reshape((b, n, h * d))?)
    }
}

struct Layer {
    self_attn: Attn,
    n1: LayerNorm,
    t2i: Attn,
    n2: LayerNorm,
    lin1: Linear,
    lin2: Linear,
    n3: LayerNorm,
    n4: LayerNorm,
    i2t: Attn,
}

struct Mlp(Vec<Linear>);

impl Mlp {
    fn new(i: usize, h: usize, o: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self((0..n).map(|j| linear(if j == 0 { i } else { h }, if j == n - 1 { o } else { h }, vb.pp(format!("layers.{j}")))).collect::<Result<_>>()?))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.0.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.0.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

/// ConvTranspose2d with kernel 2, stride 2 (no overlap): a matmul over input channels plus a pixel shuffle,
/// much faster than a general transposed convolution.
pub(crate) struct Up2 {
    /// `[Cout*4, Cin]`, rows ordered `(co, dy, dx)`
    wt: Tensor,
    /// `[Cin, Cout*4]` for the generic path
    w: Tensor,
    /// `[Cout*4]`: the conv bias repeated for each of the four taps
    b: Tensor,
    cout: usize,
}

impl Up2 {
    pub(crate) fn new(cin: usize, cout: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((cin, cout, 2, 2), "weight")?.reshape((cin, cout * 4))?;
        let b = vb.get(cout, "bias")?.reshape((cout, 1))?.broadcast_as((cout, 4))?.reshape(cout * 4)?.contiguous()?;
        Ok(Self { wt: w.t()?.contiguous()?, w, b, cout })
    }

    /// Same as [`forward`](Self::forward) from tokens `[1, H*W, Cin]`, which avoids transposing them to NCHW first.
    pub(crate) fn forward_tokens(&self, x: &Tensor, h: usize, w: usize, gelu: bool) -> Result<Tensor> {
        let c = x.dim(x.rank() - 1)?;
        let y = x.reshape((h * w, c))?.matmul(&self.w)?;
        if let Some(o) = pixel_shuffle2_cpu(&y, &self.b, h, w, gelu, true)? {
            return Ok(o);
        }
        self.forward(&x.reshape((h * w, c))?.t()?.reshape((1, c, h, w))?, gelu)
    }

    /// `x`: `[1, Cin, H, W]` -> `[1, Cout, 2H, 2W]`, with GELU applied when `gelu`.
    pub(crate) fn forward(&self, x: &Tensor, gelu: bool) -> Result<Tensor> {
        let (n, c, h, w) = x.dims4()?;
        if n == 1 {
            // [Cout*4, Cin] @ [Cin, H*W]: the input stays in NCHW, no permute
            let y = self.wt.matmul(&x.reshape((c, h * w))?)?;
            if let Some(o) = pixel_shuffle2_cpu(&y, &self.b, h, w, gelu, false)? {
                return Ok(o);
            }
        }
        let y = x.permute((0, 2, 3, 1))?.reshape((n * h * w, c))?.matmul(&self.w)?.broadcast_add(&self.b)?;
        let y = y.reshape((n, h, w, self.cout, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((n, self.cout, h * 2, w * 2))?;
        if gelu {
            y.gelu_erf()
        } else {
            Ok(y)
        }
    }
}

pub struct MaskDecoder {
    iou_token: Tensor,
    mask_tokens: Tensor,
    layers: Vec<Layer>,
    final_attn: Attn,
    norm_final: LayerNorm,
    up1: Up2,
    up_norm: LayerNorm2d,
    up2: Up2,
    hyper: Vec<Mlp>,
    iou_head: Mlp,
    key_pe: OnceLock<Tensor>,
}

impl MaskDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let d = 256;
        let t = vb.pp("transformer");
        let layers = (0..2)
            .map(|i| {
                let v = t.pp(format!("layers.{i}"));
                Ok(Layer {
                    self_attn: Attn::new(d, d, 8, v.pp("self_attn"))?,
                    n1: layer_norm(d, 1e-5, v.pp("norm1"))?,
                    t2i: Attn::new(d, d / 2, 8, v.pp("cross_attn_token_to_image"))?,
                    n2: layer_norm(d, 1e-5, v.pp("norm2"))?,
                    lin1: linear(d, 2048, v.pp("mlp.lin1"))?,
                    lin2: linear(2048, d, v.pp("mlp.lin2"))?,
                    n3: layer_norm(d, 1e-5, v.pp("norm3"))?,
                    n4: layer_norm(d, 1e-5, v.pp("norm4"))?,
                    i2t: Attn::new(d, d / 2, 8, v.pp("cross_attn_image_to_token"))?,
                })
            })
            .collect::<Result<_>>()?;
        let up = vb.pp("output_upscaling");
        Ok(Self {
            iou_token: vb.get((1, d), "iou_token.weight")?,
            mask_tokens: vb.get((4, d), "mask_tokens.weight")?,
            layers,
            final_attn: Attn::new(d, d / 2, 8, t.pp("final_attn_token_to_image"))?,
            norm_final: layer_norm(d, 1e-5, t.pp("norm_final_attn"))?,
            up1: Up2::new(d, d / 4, up.pp("0"))?,
            up_norm: LayerNorm2d::new(d / 4, 1e-6, up.pp("1"))?,
            up2: Up2::new(d / 4, d / 8, up.pp("3"))?,
            hyper: (0..4).map(|i| Mlp::new(d, d, d / 8, 3, vb.pp(format!("output_hypernetworks_mlps.{i}")))).collect::<Result<_>>()?,
            iou_head: Mlp::new(d, d, 4, 3, vb.pp("iou_prediction_head"))?,
            key_pe: OnceLock::new(),
        })
    }

    /// Returns `(masks [1, M, 256, 256] logits, iou [1, M])`.
    pub fn forward(&self, image_emb: &Tensor, image_pe: &Tensor, sparse: &Tensor, dense: &Tensor, multimask: bool) -> Result<(Tensor, Tensor)> {
        let dt = image_emb.dtype();
        let out_tokens = Tensor::cat(&[&self.iou_token, &self.mask_tokens], 0)?.to_dtype(dt)?.unsqueeze(0)?;
        let tokens = Tensor::cat(&[&out_tokens, sparse], 1)?;
        let (b, c, h, w) = image_emb.dims4()?;
        let mut keys = nchw_add_to_tokens(image_emb, dense)?;
        // the positional encoding never changes, so transpose it to tokens once
        if self.key_pe.get().is_none() {
            let _ = self.key_pe.set(image_pe.flatten_from(2)?.transpose(1, 2)?.contiguous()?);
        }
        let key_pe = self.key_pe.get().expect("set above");
        let mut queries = tokens.clone();
        let query_pe = tokens.clone();
        for (i, l) in self.layers.iter().enumerate() {
            queries = if i == 0 {
                l.self_attn.forward(&queries, &queries, &queries)?
            } else {
                let q = (&queries + &query_pe)?;
                (&queries + l.self_attn.forward(&q, &q, &queries)?)?
            };
            queries = l.n1.forward(&queries)?;
            let q = (&queries + &query_pe)?;
            let k = add_par(&keys, key_pe)?;
            queries = l.n2.forward(&(&queries + l.t2i.forward(&q, &k, &keys)?)?)?;
            let m = l.lin2.forward(&l.lin1.forward(&queries)?.relu()?)?;
            queries = l.n3.forward(&(&queries + m)?)?;
            let q = (&queries + &query_pe)?;
            keys = layer_norm_rows(&add_par(&keys, &l.i2t.forward(&k, &q, &queries)?)?, l.n4.weight(), l.n4.bias().expect("layer norm has a bias"), 1e-5)?;
        }
        let q = (&queries + &query_pe)?;
        let k = add_par(&keys, key_pe)?;
        let queries = self.norm_final.forward(&(&queries + self.final_attn.forward(&q, &k, &keys)?)?)?;

        let iou_tok = queries.narrow(1, 0, 1)?.squeeze(1)?;
        let mask_toks = queries.narrow(1, 1, 4)?;
        let _ = c;
        let zero = Tensor::zeros(1, dt, keys.device())?;
        let up = self.up2.forward(&bias_gelu(&self.up_norm.forward(&self.up1.forward_tokens(&keys, h, w, false)?)?, &zero)?, true)?;
        let (_, c2, uh, uw) = up.dims4()?;
        let hyper = Tensor::stack(&(0..4).map(|i| self.hyper[i].forward(&mask_toks.narrow(1, i, 1)?.squeeze(1)?)).collect::<Result<Vec<_>>>()?, 1)?;
        let masks = hyper.matmul(&up.reshape((b, c2, uh * uw))?)?.reshape((b, 4, uh, uw))?;
        let iou = self.iou_head.forward(&iou_tok)?;
        if multimask {
            Ok((masks.narrow(1, 1, 3)?, iou.narrow(1, 1, 3)?))
        } else {
            Ok((masks.narrow(1, 0, 1)?, iou.narrow(1, 0, 1)?))
        }
    }
}
