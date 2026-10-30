//! TinyViT-5M image encoder of MobileSAM (conv stem, MBConv stage, three windowed-attention stages, SAM neck).
//! Port of `tiny_vit_sam.py`; every conv + BatchNorm pair is folded into one biased conv at load time.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module, VarBuilder};

use super::encoder::LayerNorm2d;
use crate::cpu_gemm::{self, Linear as FastLinear};
use crate::kernels::{bias_gelu, dwconv3x3};
use crate::sam3::nn::{ln, sdpa};

const EMBED: [usize; 4] = [64, 128, 160, 320];
const DEPTHS: [usize; 4] = [2, 2, 6, 2];
const HEADS: [usize; 4] = [2, 4, 5, 10];
const WINDOWS: [usize; 4] = [7, 7, 14, 7];

#[derive(Clone, Copy)]
struct Act {
    tanh: bool,
}

impl Act {
    fn apply(self, x: &Tensor) -> Result<Tensor> {
        if self.tanh {
            x.gelu()
        } else {
            // parallel / fused exact GELU (a zero bias): candle's gelu_erf is single-threaded on CPU
            bias_gelu(x, &Tensor::zeros(1, x.dtype(), x.device())?)
        }
    }
}

/// `Conv2d` (no bias) followed by an eval-mode BatchNorm, folded into a single biased convolution.
fn conv_bn(cin: usize, cout: usize, k: usize, cfg: Conv2dConfig, vb: VarBuilder) -> Result<Conv2d> {
    let dt = vb.dtype();
    let get = |vb: &VarBuilder, shape: &[usize], name: &str| -> Result<Tensor> { vb.get_with_hints_dtype(shape.to_vec(), name, Default::default(), DType::F32) };
    let w = get(&vb.pp("c"), &[cout, cin / cfg.groups, k, k], "weight")?;
    let bn = vb.pp("bn");
    let (gamma, beta, mean, var) = (get(&bn, &[cout], "weight")?, get(&bn, &[cout], "bias")?, get(&bn, &[cout], "running_mean")?, get(&bn, &[cout], "running_var")?);
    let scale = (gamma / (var + 1e-5)?.sqrt()?)?;
    let w = w.broadcast_mul(&scale.reshape((cout, 1, 1, 1))?)?;
    let b = (beta - (mean * &scale)?)?;
    Ok(Conv2d::new(w.to_dtype(dt)?, Some(b.to_dtype(dt)?), cfg))
}

/// Folded depthwise 3x3 conv + BatchNorm, run by the dedicated kernel.
struct Dw {
    w: Tensor,
    b: Tensor,
    stride: usize,
}

impl Dw {
    fn new(c: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, stride, groups: c, ..Default::default() };
        let conv = conv_bn(c, c, 3, cfg, vb)?;
        Ok(Self { w: conv.weight().reshape((c, 9))?.contiguous()?, b: conv.bias().expect("folded bias").clone(), stride })
    }

    fn forward(&self, x: &Tensor, gelu: Option<Act>) -> Result<Tensor> {
        match gelu {
            Some(a) if !a.tanh => dwconv3x3(x, &self.w, &self.b, self.stride, true),
            Some(a) => a.apply(&dwconv3x3(x, &self.w, &self.b, self.stride, false)?),
            None => dwconv3x3(x, &self.w, &self.b, self.stride, false),
        }
    }
}

/// A stride-1 folded conv that runs on the bf16 CPU kernel when it can (1x1 / 3x3, f32, `cpu_gemm::enabled()`).
struct Pw {
    base: Conv2d,
    fast: Option<cpu_gemm::Conv>,
}

impl Pw {
    fn new(cin: usize, cout: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: k / 2, ..Default::default() };
        let base = conv_bn(cin, cout, k, cfg, vb)?;
        let fast = if base.weight().device().is_cpu() && base.weight().dtype() == DType::F32 && cpu_gemm::enabled() { Some(cpu_gemm::Conv::new(base.weight(), base.bias())?) } else { None };
        Ok(Self { base, fast })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.fast {
            Some(f) if x.dtype() == DType::F32 && x.device().is_cpu() => f.forward(x),
            _ => self.base.forward(x),
        }
    }
}

/// Linear layer on the bf16 CPU kernel when possible.
struct Lin {
    w: FastLinear,
    b: Tensor,
}

impl Lin {
    fn new(i: usize, o: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((o, i), "weight")?.t()?.contiguous()?;
        Ok(Self { w: FastLinear::new(w)?, b: vb.get(o, "bias")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let y = self.w.forward(&x.reshape((lead, dims[dims.len() - 1]))?)?.broadcast_add(&self.b.to_dtype(x.dtype())?)?;
        let mut out = dims;
        *out.last_mut().expect("rank >= 1") = y.dim(1)?;
        y.reshape(out)
    }
}

/// Bias-free neck convolution, on the bf16 CPU kernel when possible.
struct NeckConv {
    base: Conv2d,
    fast: Option<cpu_gemm::Conv>,
}

impl NeckConv {
    fn new(cin: usize, cout: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let base = candle_nn::conv2d_no_bias(cin, cout, k, Conv2dConfig { padding: k / 2, ..Default::default() }, vb)?;
        let fast = if base.weight().device().is_cpu() && base.weight().dtype() == DType::F32 && cpu_gemm::enabled() { Some(cpu_gemm::Conv::new(base.weight(), None)?) } else { None };
        Ok(Self { base, fast })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.fast {
            Some(f) if x.dtype() == DType::F32 && x.device().is_cpu() => f.forward(x),
            _ => self.base.forward(x),
        }
    }
}

struct MbConv {
    c1: Pw,
    c2: Dw,
    c3: Pw,
}

impl MbConv {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * 4;
        Ok(Self { c1: Pw::new(dim, hidden, 1, vb.pp("conv1"))?, c2: Dw::new(hidden, 1, vb.pp("conv2"))?, c3: Pw::new(hidden, dim, 1, vb.pp("conv3"))? })
    }

    fn forward(&self, x: &Tensor, a: Act) -> Result<Tensor> {
        let y = self.c3.forward(&self.c2.forward(&a.apply(&self.c1.forward(x)?)?, Some(a))?)?;
        a.apply(&(y + x)?)
    }
}

/// Down-sampling between stages: 1x1, depthwise 3x3 (stride 2, or 1 at 320 channels), 1x1; output as tokens.
struct PatchMerging {
    c1: Pw,
    c2: Dw,
    c3: Pw,
    res: usize,
}

impl PatchMerging {
    fn new(res: usize, dim: usize, out: usize, vb: VarBuilder) -> Result<Self> {
        let stride = if [320, 448, 576].contains(&out) { 1 } else { 2 };
        Ok(Self { c1: Pw::new(dim, out, 1, vb.pp("conv1"))?, c2: Dw::new(out, stride, vb.pp("conv2"))?, c3: Pw::new(out, out, 1, vb.pp("conv3"))?, res })
    }

    /// NCHW or tokens `[1, res*res, C]` in; tokens out.
    fn forward(&self, x: &Tensor, a: Act) -> Result<Tensor> {
        let x = if x.rank() == 3 { x.reshape((1, self.res, self.res, ()))?.permute((0, 3, 1, 2))?.contiguous()? } else { x.clone() };
        let y = self.c3.forward(&self.c2.forward(&a.apply(&self.c1.forward(&x)?)?, Some(a))?)?;
        y.flatten_from(2)?.transpose(1, 2)?.contiguous()
    }
}

struct Attention {
    norm: LayerNorm,
    qkv: Lin,
    proj: Lin,
    /// relative-offset biases gathered to `[1, heads, n, n]`
    ab: Tensor,
    heads: usize,
    key_dim: usize,
}

impl Attention {
    fn new(dim: usize, heads: usize, window: usize, vb: VarBuilder) -> Result<Self> {
        let key_dim = dim / heads;
        let points: Vec<(i64, i64)> = (0..window).flat_map(|x| (0..window).map(move |y| (x as i64, y as i64))).collect();
        let mut offsets = std::collections::HashMap::new();
        let mut idxs = Vec::with_capacity(points.len() * points.len());
        for &(x1, y1) in &points {
            for &(x2, y2) in &points {
                let off = ((x2 - x1).abs(), (y2 - y1).abs());
                let l = offsets.len();
                idxs.push(*offsets.entry(off).or_insert(l) as u32);
            }
        }
        let table = vb.get((heads, offsets.len()), "attention_biases")?;
        let idxs = Tensor::from_vec(idxs, points.len() * points.len(), table.device())?;
        let ab = table.index_select(&idxs, 1)?.reshape((1, heads, points.len(), points.len()))?;
        Ok(Self {
            norm: ln(dim, 1e-5, vb.pp("norm"))?,
            qkv: Lin::new(dim, dim * 3, vb.pp("qkv"))?,
            proj: Lin::new(dim, dim, vb.pp("proj"))?,
            ab,
            heads,
            key_dim,
        })
    }

    /// `x`: `[nw, n, C]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (nw, n, c) = x.dims3()?;
        // unlike a standard ViT, the fused projection is laid out per head: [head][q | k | v]
        let qkv = self.qkv.forward(&self.norm.forward(x)?)?.reshape((nw, n, self.heads, 3, self.key_dim))?;
        let part = |i: usize| qkv.narrow(3, i, 1)?.squeeze(3)?.transpose(1, 2)?.contiguous();
        let o = sdpa(&part(0)?, &part(1)?, &part(2)?, Some(&self.ab.to_dtype(x.dtype())?))?;
        self.proj.forward(&o.transpose(1, 2)?.reshape((nw, n, c))?)
    }
}

struct Block {
    attn: Attention,
    local: Dw,
    norm: LayerNorm,
    fc1: Lin,
    fc2: Lin,
    window: usize,
    res: usize,
}

impl Block {
    fn new(dim: usize, heads: usize, window: usize, res: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(dim, heads, window, vb.pp("attn"))?,
            local: Dw::new(dim, 1, vb.pp("local_conv"))?,
            norm: ln(dim, 1e-5, vb.pp("mlp.norm"))?,
            fc1: Lin::new(dim, dim * 4, vb.pp("mlp.fc1"))?,
            fc2: Lin::new(dim * 4, dim, vb.pp("mlp.fc2"))?,
            window,
            res,
        })
    }

    /// tokens `[1, res*res, C]`
    fn forward(&self, x: &Tensor, a: Act) -> Result<Tensor> {
        let (h, w) = (self.res, self.res);
        let (b, l, c) = x.dims3()?;
        let ws = self.window;
        let y = if h == ws && w == ws {
            self.attn.forward(x)?
        } else {
            let y = x.reshape((b, h, w, c))?;
            let (pb, pr) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
            let y = if pb > 0 { y.pad_with_zeros(1, 0, pb)? } else { y };
            let y = if pr > 0 { y.pad_with_zeros(2, 0, pr)? } else { y };
            let (ph, pw) = (h + pb, w + pr);
            let (nh, nw) = (ph / ws, pw / ws);
            let y = y.reshape((b, nh, ws, nw, ws, c))?.transpose(2, 3)?.contiguous()?.reshape((b * nh * nw, ws * ws, c))?;
            let y = self.attn.forward(&y)?;
            let y = y.reshape((b, nh, nw, ws, ws, c))?.transpose(2, 3)?.contiguous()?.reshape((b, ph, pw, c))?;
            y.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()?.reshape((b, l, c))?
        };
        let y = (y + x)?;
        let y = self.local.forward(&y.transpose(1, 2)?.reshape((b, c, h, w))?.contiguous()?, None)?.reshape((b, c, l))?.transpose(1, 2)?.contiguous()?;
        let m = self.fc2.forward(&a.apply(&self.fc1.forward(&self.norm.forward(&y)?)?)?)?;
        &y + m
    }
}

enum Stage {
    Conv { blocks: Vec<MbConv>, down: PatchMerging },
    Attn { blocks: Vec<Block>, down: Option<PatchMerging> },
}

pub struct TinyVit {
    stem1: Conv2d,
    stem2: Conv2d,
    stages: Vec<Stage>,
    neck1: NeckConv,
    neck_ln1: LayerNorm2d,
    neck2: NeckConv,
    neck_ln2: LayerNorm2d,
    act: Act,
}

impl TinyVit {
    /// `vb` is the `image_encoder` prefix. `tanh_gelu` reproduces candle's reference implementation (tests only);
    /// the original model uses the exact erf GELU.
    pub fn new(vb: VarBuilder, tanh_gelu: bool) -> Result<Self> {
        let s2 = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };
        let pe = vb.pp("patch_embed.seq");
        let mut stages = vec![];
        let l0 = vb.pp("layers.0");
        stages.push(Stage::Conv {
            blocks: (0..DEPTHS[0]).map(|i| MbConv::new(EMBED[0], l0.pp(format!("blocks.{i}")))).collect::<Result<_>>()?,
            down: PatchMerging::new(256, EMBED[0], EMBED[1], l0.pp("downsample"))?,
        });
        for i in 1..4 {
            let l = vb.pp(format!("layers.{i}"));
            let res = 256 / (1 << i.min(2));
            stages.push(Stage::Attn {
                blocks: (0..DEPTHS[i]).map(|j| Block::new(EMBED[i], HEADS[i], WINDOWS[i], res, l.pp(format!("blocks.{j}")))).collect::<Result<_>>()?,
                down: if i < 3 { Some(PatchMerging::new(res, EMBED[i], EMBED[i + 1], l.pp("downsample"))?) } else { None },
            });
        }
        let dt = vb.dtype();
        let _ = dt;
        Ok(Self {
            stem1: conv_bn(3, EMBED[0] / 2, 3, s2, pe.pp("0"))?,
            stem2: conv_bn(EMBED[0] / 2, EMBED[0], 3, s2, pe.pp("2"))?,
            stages,
            neck1: NeckConv::new(EMBED[3], 256, 1, vb.pp("neck.0"))?,
            neck_ln1: LayerNorm2d::new(256, 1e-6, vb.pp("neck.1"))?,
            neck2: NeckConv::new(256, 256, 3, vb.pp("neck.2"))?,
            neck_ln2: LayerNorm2d::new(256, 1e-6, vb.pp("neck.3"))?,
            act: Act { tanh: tanh_gelu },
        })
    }

    /// `[1, 3, 1024, 1024]` -> `[1, 256, 64, 64]`
    pub fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let a = self.act;
        let mut x = self.stem2.forward(&a.apply(&self.stem1.forward(img)?)?)?;
        for st in &self.stages {
            x = match st {
                Stage::Conv { blocks, down } => {
                    for b in blocks {
                        x = b.forward(&x, a)?;
                    }
                    down.forward(&x, a)?
                }
                Stage::Attn { blocks, down } => {
                    for b in blocks {
                        x = b.forward(&x, a)?;
                    }
                    match down {
                        Some(d) => d.forward(&x, a)?,
                        None => x,
                    }
                }
            };
        }
        let x = x.reshape((1, 64, 64, EMBED[3]))?.permute((0, 3, 1, 2))?.contiguous()?;
        let x = self.neck_ln1.forward(&self.neck1.forward(&x)?)?;
        self.neck_ln2.forward(&self.neck2.forward(&x)?)
    }
}
