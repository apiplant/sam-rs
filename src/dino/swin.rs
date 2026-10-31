//! Swin Transformer backbone (as `SwinBackbone` in `transformers`): patch embedding, shifted-window attention
//! stages joined by patch merging, and a layer norm on each requested stage output. Batch size 1; tokens are
//! `[H*W, C]`.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{conv2d, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

pub struct SwinConfig {
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub heads: Vec<usize>,
    pub window: usize,
    pub patch: usize,
    /// Stage numbers (1-based, as in `stage2`) whose outputs are returned.
    pub out_stages: Vec<usize>,
}

struct Block {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    /// `[heads, ws*ws, ws*ws]`
    bias: Tensor,
    heads: usize,
    shift: usize,
    ws: usize,
}

fn roll2(x: &Tensor, dy: usize, dx: usize) -> Result<Tensor> {
    // [H, W, C] rolled by (dy, dx) towards smaller indices
    let (h, w, _) = x.dims3()?;
    let x = if dy % h != 0 { Tensor::cat(&[&x.narrow(0, dy % h, h - dy % h)?, &x.narrow(0, 0, dy % h)?], 0)? } else { x.clone() };
    if dx % w != 0 { Tensor::cat(&[&x.narrow(1, dx % w, w - dx % w)?, &x.narrow(1, 0, dx % w)?], 1) } else { Ok(x) }
}

impl Block {
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let c = x.dim(1)?;
        let ws = self.ws;
        let y = self.ln1.forward(x)?.reshape((h, w, c))?;
        let (ph, pw) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
        let y = if ph > 0 { y.pad_with_zeros(0, 0, ph)? } else { y };
        let y = if pw > 0 { y.pad_with_zeros(1, 0, pw)? } else { y };
        let (hp, wp) = (h + ph, w + pw);
        let y = if self.shift > 0 {
            // roll by -shift: element at i moves to i - shift
            roll2(&y, self.shift, self.shift)?
        } else {
            y
        };
        let (nh, nw) = (hp / ws, wp / ws);
        let win = y.reshape((nh, ws, nw, ws, c))?.permute((0, 2, 1, 3, 4))?.reshape((nh * nw, ws * ws, c))?;
        let hd = c / self.heads;
        let split = |t: Tensor| t.reshape((nh * nw, ws * ws, self.heads, hd))?.transpose(1, 2)?.contiguous();
        let (q, k, v) = (split(self.q.forward(&win)?)?, split(self.k.forward(&win)?)?, split(self.v.forward(&win)?)?);
        let mut s = (q.matmul(&k.transpose(2, 3)?)? / (hd as f64).sqrt())?.broadcast_add(&self.bias.unsqueeze(0)?)?;
        if self.shift > 0 {
            s = s.broadcast_add(&shift_mask(hp, wp, ws, self.shift, x.device())?.to_dtype(s.dtype())?.unsqueeze(1)?)?;
        }
        let a = candle_nn::ops::softmax(&s, D::Minus1)?.matmul(&v)?.transpose(1, 2)?.reshape((nh * nw, ws * ws, c))?;
        let a = self.out.forward(&a)?;
        let a = a.reshape((nh, nw, ws, ws, c))?.permute((0, 2, 1, 3, 4))?.reshape((hp, wp, c))?;
        let a = if self.shift > 0 { roll2(&a, hp - self.shift, wp - self.shift)? } else { a };
        let a = a.narrow(0, 0, h)?.narrow(1, 0, w)?.contiguous()?.reshape((h * w, c))?;
        let x = (x + a)?;
        let f = self.fc2.forward(&self.fc1.forward(&self.ln2.forward(&x)?)?.gelu_erf()?)?;
        x + f
    }
}

/// `[nW, ws*ws, ws*ws]` additive mask (0 / -100) separating the regions a shifted window wraps around.
fn shift_mask(hp: usize, wp: usize, ws: usize, shift: usize, device: &Device) -> Result<Tensor> {
    let region = |i: usize, n: usize| if i + ws < n { 0 } else if i + shift < n { 1 } else { 2 };
    let (nh, nw) = (hp / ws, wp / ws);
    let mut id = vec![0u8; nh * nw * ws * ws]; // window-major region ids
    for y in 0..hp {
        for x in 0..wp {
            let (wy, wx, iy, ix) = (y / ws, x / ws, y % ws, x % ws);
            id[((wy * nw + wx) * ws + iy) * ws + ix] = (region(y, hp) * 3 + region(x, wp)) as u8;
        }
    }
    let n = ws * ws;
    let mut m = vec![0f32; nh * nw * n * n];
    for w in 0..nh * nw {
        for i in 0..n {
            for j in 0..n {
                if id[w * n + i] != id[w * n + j] {
                    m[(w * n + i) * n + j] = -100.0;
                }
            }
        }
    }
    Tensor::from_vec(m, (nh * nw, n, n), device)
}

struct Merging {
    norm: LayerNorm,
    reduction: Linear,
}

impl Merging {
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<(Tensor, usize, usize)> {
        let c = x.dim(1)?;
        let mut y = x.reshape((h, w, c))?;
        if h % 2 == 1 {
            y = y.pad_with_zeros(0, 0, 1)?;
        }
        if w % 2 == 1 {
            y = y.pad_with_zeros(1, 0, 1)?;
        }
        let (h2, w2) = (y.dim(0)? / 2, y.dim(1)? / 2);
        let y = y.reshape((h2, 2, w2, 2, c))?;
        let part = |a: usize, b: usize| y.narrow(1, a, 1)?.narrow(3, b, 1)?.reshape((h2, w2, c));
        let cat = Tensor::cat(&[&part(0, 0)?, &part(1, 0)?, &part(0, 1)?, &part(1, 1)?], 2)?.reshape((h2 * w2, 4 * c))?;
        Ok((self.reduction.forward(&self.norm.forward(&cat)?)?, h2, w2))
    }
}

pub struct Swin {
    patch: Conv2d,
    embed_norm: LayerNorm,
    stages: Vec<(Vec<Block>, Option<Merging>)>,
    out_norms: Vec<(usize, LayerNorm)>,
    patch_size: usize,
}

impl Swin {
    /// `vb`: the `...conv_encoder.model` prefix.
    pub fn new(cfg: &SwinConfig, vb: VarBuilder) -> Result<Self> {
        let e = cfg.embed_dim;
        let ws = cfg.window;
        let patch = conv2d(3, e, cfg.patch, Conv2dConfig { stride: cfg.patch, ..Default::default() }, vb.pp("embeddings.patch_embeddings.projection"))?;
        let embed_norm = layer_norm(e, 1e-5, vb.pp("embeddings.norm"))?;
        let mut stages = vec![];
        for (si, (&depth, &heads)) in cfg.depths.iter().zip(&cfg.heads).enumerate() {
            let dim = e << si;
            let blocks = (0..depth)
                .map(|bi| {
                    let l = vb.pp(format!("encoder.layers.{si}.blocks.{bi}"));
                    let sa = l.pp("attention.self");
                    // relative position bias, expanded once: table[index] -> [heads, n, n]
                    let table = sa.get(((2 * ws - 1) * (2 * ws - 1), heads), "relative_position_bias_table")?;
                    let idx: Vec<u32> = relative_index(ws);
                    let n = ws * ws;
                    let bias = table.index_select(&Tensor::new(idx, table.device())?, 0)?.reshape((n, n, heads))?.permute((2, 0, 1))?.contiguous()?;
                    Ok(Block {
                        ln1: layer_norm(dim, 1e-5, l.pp("layernorm_before"))?,
                        q: linear(dim, dim, sa.pp("query"))?,
                        k: linear(dim, dim, sa.pp("key"))?,
                        v: linear(dim, dim, sa.pp("value"))?,
                        out: linear(dim, dim, l.pp("attention.output.dense"))?,
                        ln2: layer_norm(dim, 1e-5, l.pp("layernorm_after"))?,
                        fc1: linear(dim, 4 * dim, l.pp("intermediate.dense"))?,
                        fc2: linear(4 * dim, dim, l.pp("output.dense"))?,
                        bias,
                        heads,
                        shift: if bi % 2 == 0 { 0 } else { ws / 2 },
                        ws,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let merging = if si + 1 < cfg.depths.len() {
                let m = vb.pp(format!("encoder.layers.{si}.downsample"));
                Some(Merging { norm: layer_norm(4 * dim, 1e-5, m.pp("norm"))?, reduction: linear_no_bias(4 * dim, 2 * dim, m.pp("reduction"))? })
            } else {
                None
            };
            stages.push((blocks, merging));
        }
        let out_norms = cfg.out_stages.iter().map(|&s| Ok((s, layer_norm(e << (s - 1), 1e-5, vb.pp(format!("hidden_states_norms.stage{s}")))?))).collect::<Result<_>>()?;
        Ok(Self { patch, embed_norm, stages, out_norms, patch_size: cfg.patch })
    }

    /// `[1, 3, H, W]` -> the requested stage outputs as `[1, C, h, w]`, finest first.
    pub fn forward(&self, px: &Tensor) -> Result<Vec<Tensor>> {
        let (_, _, ih, iw) = px.dims4()?;
        let p = self.patch_size;
        let px = if iw % p != 0 { px.pad_with_zeros(3, 0, p - iw % p)? } else { px.clone() };
        let px = if ih % p != 0 { px.pad_with_zeros(2, 0, p - ih % p)? } else { px };
        let x = self.patch.forward(&px)?;
        let (_, c, mut h, mut w) = x.dims4()?;
        let mut x = self.embed_norm.forward(&x.reshape((c, h * w))?.t()?.contiguous()?)?;
        let mut outs = vec![];
        for (si, (blocks, merge)) in self.stages.iter().enumerate() {
            for b in blocks {
                x = b.forward(&x, h, w)?;
            }
            if let Some((_, norm)) = self.out_norms.iter().find(|(s, _)| *s == si + 1) {
                let c = x.dim(1)?;
                outs.push(norm.forward(&x)?.reshape((h, w, c))?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?);
            }
            if let Some(m) = merge {
                (x, h, w) = m.forward(&x, h, w)?;
            }
        }
        Ok(outs)
    }
}

fn relative_index(ws: usize) -> Vec<u32> {
    let n = ws * ws;
    let mut v = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let (yi, xi, yj, xj) = ((i / ws) as i64, (i % ws) as i64, (j / ws) as i64, (j % ws) as i64);
            let (dy, dx) = (yi - yj + ws as i64 - 1, xi - xj + ws as i64 - 1);
            v.push((dy * (2 * ws as i64 - 1) + dx) as u32);
        }
    }
    v
}

#[allow(dead_code)]
fn _dtype(_: DType) {}
