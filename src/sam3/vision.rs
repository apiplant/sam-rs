//! SAM 3 vision encoder: ViT with axial 2-D RoPE (24x24 windows, four global layers) and a 4-level FPN.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module, VarBuilder};

use super::nn::{ln, sdpa, FastConv};
use crate::cpu_gemm::Linear;
use crate::kernels::{bias_gelu, bias_residual, qkv_rope};
use crate::sam1::decoder::Up2;

const DIM: usize = 1024;
const HEADS: usize = 16;
const LAYERS: usize = 32;
const WINDOW: usize = 24;
const GLOBAL: [usize; 4] = [7, 15, 23, 31];
const GRID: usize = 72;
const THETA: f32 = 10000.0;

/// `(cos, sin)` f32 tables `[side*side, 64]` for an axial 2-D RoPE over a `side x side` grid, positions
/// multiplied by `scale`. The first 32 dims rotate with the column index, the last 32 with the row index.
fn rope_tables(side: usize, scale: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    rope_tables_hd(side, scale, DIM / HEADS, THETA, device)
}

/// Same as [`rope_tables`] for any head dimension `hd` and base `theta`.
pub(crate) fn rope_tables_hd(side: usize, scale: f32, hd: usize, theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let sp = hd / 2;
    let inv: Vec<f32> = (0..sp / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / sp as f32)).collect();
    let n = side * side;
    let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
    for r in 0..side {
        for c in 0..side {
            let t = r * side + c;
            for j in 0..sp {
                let (pos, f) = if j < sp / 2 { (c as f32 * scale, inv[j]) } else { (r as f32 * scale, inv[j - sp / 2]) };
                let a = pos * f;
                // repeat_interleave(2)
                for k in 0..2 {
                    cos[t * hd + 2 * j + k] = a.cos();
                    sin[t * hd + 2 * j + k] = a.sin();
                }
            }
        }
    }
    Ok((Tensor::from_vec(cos, (n, hd), device)?, Tensor::from_vec(sin, (n, hd), device)?))
}

struct Layer {
    ln1: LayerNorm,
    qkv_w: Linear,
    qkv_b: Tensor,
    o_w: Linear,
    o_b: Tensor,
    ln2: LayerNorm,
    fc1_w: Linear,
    fc1_b: Tensor,
    fc2_w: Linear,
    fc2_b: Tensor,
    window: usize,
    cos: Tensor,
    sin: Tensor,
}

/// `[out, in]` weights -> `[in, out]` contiguous, for `x @ w`.
fn wt(vb: &VarBuilder, out: usize, inp: usize, name: &str) -> Result<Tensor> {
    vb.get((out, inp), name)?.t()?.contiguous()
}

impl Layer {
    fn new(idx: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let global = GLOBAL.contains(&idx);
        let (side, scale) = if global { (GRID, WINDOW as f32 / GRID as f32) } else { (WINDOW, 1.0) };
        let (cos, sin) = rope_tables(side, scale, device)?;
        let a = vb.pp("attention");
        let w = Tensor::cat(&[wt(&a.pp("q_proj"), DIM, DIM, "weight")?, wt(&a.pp("k_proj"), DIM, DIM, "weight")?, wt(&a.pp("v_proj"), DIM, DIM, "weight")?], 1)?;
        let b = Tensor::cat(&[a.pp("q_proj").get(DIM, "bias")?, a.pp("k_proj").get(DIM, "bias")?, a.pp("v_proj").get(DIM, "bias")?], 0)?;
        let m = vb.pp("mlp");
        Ok(Self {
            ln1: ln(DIM, 1e-6, vb.pp("layer_norm1"))?,
            qkv_w: Linear::new(w.contiguous()?)?,
            qkv_b: b.contiguous()?,
            o_w: Linear::new(wt(&a.pp("o_proj"), DIM, DIM, "weight")?)?,
            o_b: a.pp("o_proj").get(DIM, "bias")?,
            ln2: ln(DIM, 1e-6, vb.pp("layer_norm2"))?,
            fc1_w: Linear::new(wt(&m.pp("fc1"), 4736, DIM, "weight")?)?,
            fc1_b: m.pp("fc1").get(4736, "bias")?,
            fc2_w: Linear::new(wt(&m.pp("fc2"), DIM, 4736, "weight")?)?,
            fc2_b: m.pp("fc2").get(DIM, "bias")?,
            window: if global { 0 } else { WINDOW },
            cos,
            sin,
        })
    }

    /// `x`: `[B, H, W, C]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = x.dims4()?;
        let y = self.ln1.forward(x)?;
        let y = if self.window > 0 { partition(&y, self.window)? } else { y };
        let (nw, hh, ww, _) = y.dims4()?;
        let (n, t) = (hh * ww, nw * hh * ww);
        let qkv = self.qkv_w.forward(&y.reshape((t, DIM))?)?;
        let packed = qkv_rope(&qkv, &self.qkv_b, &self.cos, &self.sin, n, HEADS, DIM / HEADS)?;
        let ctx = attention(&packed, nw, n)?;
        let a = self.o_w.forward(&ctx)?.reshape((nw, hh, ww, DIM))?;
        let a = if self.window > 0 { unpartition(&a, self.window, b, h, w)? } else { a };
        let x = bias_residual(x, &a, &self.o_b)?;
        let m = bias_gelu(&self.fc1_w.forward(&self.ln2.forward(&x)?.reshape((b * h * w, DIM))?)?, &self.fc1_b)?;
        bias_residual(&x, &self.fc2_w.forward(&m)?.reshape((b, h, w, DIM))?, &self.fc2_b)
    }
}

/// Self-attention over a packed `[3, T, H, D]` (q, k, v) with `T = nw * n`. Returns `[T, H*D]`.
fn attention(packed: &Tensor, nw: usize, n: usize) -> Result<Tensor> {
    let (hd, t) = (DIM / HEADS, nw * n);
    #[cfg(feature = "flash-attn")]
    if packed.device().is_cuda() && matches!(packed.dtype(), DType::F16 | DType::BF16) {
        let get = |i: usize| packed.get(i)?.reshape((nw, n, HEADS, hd));
        let o = candle_flash_attn::flash_attn(&get(0)?, &get(1)?, &get(2)?, (hd as f32).powf(-0.5), false)?;
        return o.reshape((t, DIM));
    }
    if let Some(o) = crate::kernels::flash_attn_cpu(&packed.get(0)?.reshape((nw, n, HEADS, hd))?, &packed.get(1)?.reshape((nw, n, HEADS, hd))?, &packed.get(2)?.reshape((nw, n, HEADS, hd))?)? {
        return o.reshape((t, DIM));
    }
    let get = |i: usize| packed.get(i)?.reshape((nw, n, HEADS, hd))?.transpose(1, 2);
    sdpa(&get(0)?, &get(1)?, &get(2)?, None)?.transpose(1, 2)?.reshape((t, DIM))
}

/// `[B, H, W, C]` -> `[B * (H/ws) * (W/ws), ws, ws, C]` (sizes are exact multiples here).
fn partition(x: &Tensor, ws: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    x.reshape((b, h / ws, ws, w / ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b * (h / ws) * (w / ws), ws, ws, c))
}

fn unpartition(x: &Tensor, ws: usize, b: usize, h: usize, w: usize) -> Result<Tensor> {
    let c = x.dim(3)?;
    x.reshape((b, h / ws, w / ws, ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, h, w, c))
}

/// One FPN level: optional up/down-sampling, then a 1x1 and a 3x3 conv.
pub(crate) struct Fpn {
    up: Vec<Up2>,
    pool: bool,
    proj1: FastConv,
    proj2: FastConv,
}

impl Fpn {
    pub(crate) fn new(scale: f32, vb: VarBuilder) -> Result<Self> {
        let s = vb.pp("scale_layers");
        let (up, mid) = if scale == 4.0 {
            (vec![Up2::new(DIM, DIM / 2, s.pp("0"))?, Up2::new(DIM / 2, DIM / 4, s.pp("2"))?], DIM / 4)
        } else if scale == 2.0 {
            (vec![Up2::new(DIM, DIM / 2, s.pp("0"))?], DIM / 2)
        } else {
            (vec![], DIM)
        };
        Ok(Self {
            up,
            pool: scale == 0.5,
            proj1: FastConv::new(mid, 256, 1, vb.pp("proj1"))?,
            proj2: FastConv::new(256, 256, 3, vb.pp("proj2"))?,
        })
    }

    /// `tokens`: `[1, 72, 72, C]` backbone output.
    pub(crate) fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_, h, w, _) = tokens.dims4()?;
        let mut x = match self.up.len() {
            0 => tokens.permute((0, 3, 1, 2))?.contiguous()?,
            1 => self.up[0].forward_tokens(tokens, h, w, false)?,
            _ => self.up[1].forward(&self.up[0].forward_tokens(tokens, h, w, true)?, false)?,
        };
        if self.pool {
            x = x.max_pool2d(2)?;
        }
        self.proj2.forward(&self.proj1.forward(&x)?)
    }
}

pub struct VisionEncoder {
    patch: Conv2d,
    pos: Tensor,
    ln_pre: LayerNorm,
    layers: Vec<Layer>,
    fpn: Vec<Fpn>,
}

impl VisionEncoder {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let bb = vb.pp("backbone");
        let e = bb.pp("embeddings");
        let patch = candle_nn::conv2d_no_bias(3, DIM, 14, Conv2dConfig { stride: 14, ..Default::default() }, e.pp("patch_embeddings.projection"))?;
        let pos = e.get((1, 576, DIM), "position_embeddings")?;
        // 24x24 pretraining grid tiled 3x3 to 72x72
        let pos = pos.reshape((24, 24, DIM))?;
        let pos = Tensor::cat(&[&pos, &pos, &pos], 0)?;
        let pos = Tensor::cat(&[&pos, &pos, &pos], 1)?.reshape((1, GRID, GRID, DIM))?;
        let layers = (0..LAYERS).map(|i| Layer::new(i, bb.pp(format!("layers.{i}")), device)).collect::<Result<_>>()?;
        let fpn = [4.0, 2.0, 1.0, 0.5].iter().enumerate().map(|(i, &s)| Fpn::new(s, vb.pp(format!("neck.fpn_layers.{i}")))).collect::<Result<_>>()?;
        Ok(Self { patch, pos, ln_pre: ln(DIM, 1e-6, bb.pp("layer_norm"))?, layers, fpn })
    }

    pub(crate) fn tracker_neck(vb: VarBuilder) -> Result<Vec<Fpn>> {
        [4.0, 2.0, 1.0, 0.5].iter().enumerate().map(|(i, &s)| Fpn::new(s, vb.pp(format!("fpn_layers.{i}")))).collect()
    }

    /// `[B,3,1008,1008]` -> backbone tokens `[B,72,72,1024]` and the FPN levels (288², 144², 72², 36²).
    pub fn forward(&self, img: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let x = self.tokens(img)?;
        let levels = self.fpn.iter().map(|f| f.forward(&x)).collect::<Result<Vec<_>>>()?;
        Ok((x, levels))
    }

    /// Backbone tokens only `[B,72,72,1024]` (what the tracker's own neck consumes).
    pub fn tokens(&self, img: &Tensor) -> Result<Tensor> {
        let x = self.patch.forward(img)?.permute((0, 2, 3, 1))?.broadcast_add(&self.pos.to_dtype(img.dtype())?)?;
        let mut x = self.ln_pre.forward(&x)?;
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        Ok(x)
    }
}

/// Sine position embedding `[1, 256, h, w]` (128 features per axis, normalized, scale 2*pi), channels = (y, x).
pub fn sine_pos(h: usize, w: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    sine_pos_nf(h, w, 128, device, dtype)
}

/// [`sine_pos`] with `nf` features per axis: `[1, 2*nf, h, w]`.
pub fn sine_pos_nf(h: usize, w: usize, nf: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let scale = 2.0 * std::f32::consts::PI;
    let dim_t: Vec<f32> = (0..nf).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / nf as f32)).collect();
    let enc = |v: f32, i: usize| -> f32 {
        let a = v / dim_t[i];
        if i % 2 == 0 { a.sin() } else { a.cos() }
    };
    let mut out = vec![0f32; 2 * nf * h * w];
    for y in 0..h {
        for x in 0..w {
            let ye = (y + 1) as f32 / (h as f32 + 1e-6) * scale;
            let xe = (x + 1) as f32 / (w as f32 + 1e-6) * scale;
            for i in 0..nf {
                out[(i * h + y) * w + x] = enc(ye, i);
                out[((nf + i) * h + y) * w + x] = enc(xe, i);
            }
        }
    }
    Tensor::from_vec(out, (1, 2 * nf, h, w), device)?.to_dtype(dtype)
}
