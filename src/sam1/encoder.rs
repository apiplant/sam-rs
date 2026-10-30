//! SAM ViT image encoder (windowed attention with decomposed relative position embeddings).

use candle_core::{Result, Tensor};

use crate::cpu_gemm::Linear;
use crate::kernels::{layer_norm_2d, attention_relpos_cpu, bias_gelu, bmm, bias_residual, qkv_split, relpos_softmax};
use candle_nn::{conv2d, conv2d_no_bias, layer_norm, Conv2d, Conv2dConfig, LayerNorm, Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub global_attn: &'static [usize],
    pub window: usize,
    pub out_chans: usize,
}

impl EncoderConfig {
    pub const VIT_B: Self = Self { embed_dim: 768, depth: 12, heads: 12, global_attn: &[2, 5, 8, 11], window: 14, out_chans: 256 };
    pub const VIT_L: Self = Self { embed_dim: 1024, depth: 24, heads: 16, global_attn: &[5, 11, 17, 23], window: 14, out_chans: 256 };
    pub const VIT_H: Self = Self { embed_dim: 1280, depth: 32, heads: 16, global_attn: &[7, 15, 23, 31], window: 14, out_chans: 256 };
}

/// LayerNorm over the channel dim of an NCHW tensor.
pub struct LayerNorm2d {
    w: Tensor,
    b: Tensor,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(c: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { w: vb.get(c, "weight")?, b: vb.get(c, "bias")?, eps })
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm_2d(x, &self.w, &self.b, self.eps as f32)
    }
}

struct Attention {
    qkv_w: Linear,
    qkv_b: Tensor,
    proj_w: Linear,
    proj_b: Tensor,
    heads: usize,
    scale: f32,
    /// `rel_pos_h` rows followed by `rel_pos_w` rows.
    rel: Tensor,
}

impl Attention {
    fn new(dim: usize, heads: usize, input: usize, vb: VarBuilder) -> Result<Self> {
        let hd = dim / heads;
        let rel_h = vb.get((2 * input - 1, hd), "rel_pos_h")?;
        let rel_w = vb.get((2 * input - 1, hd), "rel_pos_w")?;
        Ok(Self {
            qkv_w: Linear::new(vb.get((dim * 3, dim), "qkv.weight")?.t()?.contiguous()?)?,
            qkv_b: vb.get(dim * 3, "qkv.bias")?,
            proj_w: Linear::new(vb.get((dim, dim), "proj.weight")?.t()?.contiguous()?)?,
            proj_b: vb.get(dim, "proj.bias")?,
            heads,
            scale: (hd as f32).powf(-0.5),
            rel: Tensor::cat(&[&rel_h, &rel_w], 0)?,
        })
    }

    /// `x`: `[nW, H, W, C]` -> the output projection **without** its bias, `[nW, H, W, C]`
    /// (the caller fuses the bias into the residual add).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (nw, h, w, c) = x.dims4()?;
        let (nh, n) = (self.heads, h * w);
        let hd = c / nh;
        let y = self.qkv_w.forward(&x.reshape((nw * n, c))?)?;
        let (q, kt, v) = qkv_split(&y, &self.qkv_b, n, nh, hd)?;
        let ctx = match attention_relpos_cpu(&q, &kt, &v, &self.rel, h, w, self.scale)? {
            Some(ctx) => ctx,
            None => bmm(&relpos_softmax(&bmm(&q, &kt)?, &q, &self.rel, h, w, self.scale)?, &v)?,
        };
        let out = ctx.reshape((nw, nh, n, hd))?.permute((0, 2, 1, 3))?.reshape((nw * n, c))?;
        self.proj_w.forward(&out)?.reshape((nw, h, w, c))
    }
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    lin1_w: Linear,
    lin1_b: Tensor,
    lin2_w: Linear,
    lin2_b: Tensor,
    window: usize,
}

impl Block {
    fn new(cfg: &EncoderConfig, global: bool, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        let (window, input) = if global { (0, 64) } else { (cfg.window, cfg.window) };
        Ok(Self {
            norm1: layer_norm(d, 1e-6, vb.pp("norm1"))?,
            attn: Attention::new(d, cfg.heads, input, vb.pp("attn"))?,
            norm2: layer_norm(d, 1e-6, vb.pp("norm2"))?,
            lin1_w: Linear::new(vb.get((d * 4, d), "mlp.lin1.weight")?.t()?.contiguous()?)?,
            lin1_b: vb.get(d * 4, "mlp.lin1.bias")?,
            lin2_w: Linear::new(vb.get((d, d * 4), "mlp.lin2.weight")?.t()?.contiguous()?)?,
            lin2_b: vb.get(d, "mlp.lin2.bias")?,
            window,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let y = self.norm1.forward(x)?;
        let y = if self.window > 0 {
            let (parts, hp, wp) = window_partition(&y, self.window)?;
            let parts = self.attn.forward(&parts)?;
            window_unpartition(&parts, self.window, hp, wp, (h, w), b, c)?
        } else {
            self.attn.forward(&y)?
        };
        let x = bias_residual(x, &y, &self.attn.proj_b)?;
        let t = b * h * w;
        let m = bias_gelu(&self.lin1_w.forward(&self.norm2.forward(&x)?.reshape((t, c))?)?, &self.lin1_b)?;
        bias_residual(&x, &self.lin2_w.forward(&m)?.reshape((b, h, w, c))?, &self.lin2_b)
    }
}

fn window_partition(x: &Tensor, ws: usize) -> Result<(Tensor, usize, usize)> {
    let (b, h, w, c) = x.dims4()?;
    let (ph, pw) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
    let x = if ph > 0 { x.pad_with_zeros(1, 0, ph)? } else { x.clone() };
    let x = if pw > 0 { x.pad_with_zeros(2, 0, pw)? } else { x };
    let (hp, wp) = (h + ph, w + pw);
    let x = x.reshape((b, hp / ws, ws, wp / ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b * (hp / ws) * (wp / ws), ws, ws, c))?;
    Ok((x, hp, wp))
}

fn window_unpartition(x: &Tensor, ws: usize, hp: usize, wp: usize, hw: (usize, usize), b: usize, c: usize) -> Result<Tensor> {
    let x = x.reshape((b, hp / ws, wp / ws, ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, hp, wp, c))?;
    x.narrow(1, 0, hw.0)?.narrow(2, 0, hw.1)?.contiguous()
}

pub struct ImageEncoder {
    patch: Conv2d,
    pos: Tensor,
    blocks: Vec<Block>,
    neck0: Conv2d,
    neck1: LayerNorm2d,
    neck2: Conv2d,
    neck3: LayerNorm2d,
}

impl ImageEncoder {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        let patch = conv2d(3, d, 16, Conv2dConfig { stride: 16, ..Default::default() }, vb.pp("patch_embed.proj"))?;
        let pos = vb.get((1, 64, 64, d), "pos_embed")?;
        let blocks = (0..cfg.depth).map(|i| Block::new(cfg, cfg.global_attn.contains(&i), vb.pp(format!("blocks.{i}")))).collect::<Result<_>>()?;
        let c3 = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            patch,
            pos,
            blocks,
            neck0: conv2d_no_bias(d, cfg.out_chans, 1, Default::default(), vb.pp("neck.0"))?,
            neck1: LayerNorm2d::new(cfg.out_chans, 1e-6, vb.pp("neck.1"))?,
            neck2: conv2d_no_bias(cfg.out_chans, cfg.out_chans, 3, c3, vb.pp("neck.2"))?,
            neck3: LayerNorm2d::new(cfg.out_chans, 1e-6, vb.pp("neck.3"))?,
        })
    }

    /// `[1, 3, 1024, 1024]` -> `[1, 256, 64, 64]`
    pub fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let mut x = self.patch.forward(img)?.permute((0, 2, 3, 1))?.broadcast_add(&self.pos)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let x = self.neck1.forward(&self.neck0.forward(&x)?)?;
        self.neck3.forward(&self.neck2.forward(&x)?)
    }
}

