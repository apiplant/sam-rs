//! SAM 2 image encoder: Hiera (hierarchical ViT with query pooling and windowed attention) + a lateral-conv FPN.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

use crate::cpu_gemm::Linear as FastLinear;
use crate::kernels::{bias_gelu, bias_residual};
use crate::sam3::nn::{ln, sdpa};
use crate::sam3::vision::sine_pos_nf;

/// Architecture of one SAM 2 checkpoint, read from its `config.json`.
#[derive(Clone, Debug)]
pub struct HieraConfig {
    pub embed_dim: usize,
    pub dims: Vec<usize>,
    pub blocks: Vec<usize>,
    pub heads: Vec<usize>,
    pub windows: Vec<usize>,
    pub global_blocks: Vec<usize>,
    pub query_stride: usize,
    pub pool_stages: usize,
    pub mlp_ratio: f64,
    pub bg_size: (usize, usize),
    pub channels: Vec<usize>,
    pub top_down: Vec<usize>,
    pub fpn_dim: usize,
}

impl HieraConfig {
    pub fn from_json(cfg: &serde_json::Value) -> anyhow::Result<Self> {
        use anyhow::Context;
        let vc = cfg.get("vision_config").context("config has no vision_config")?;
        let bb = vc.get("backbone_config").context("no backbone_config")?;
        let ints = |v: &serde_json::Value, k: &str| -> anyhow::Result<Vec<usize>> {
            v.get(k).and_then(|a| a.as_array()).with_context(|| format!("missing {k}"))?.iter().map(|x| x.as_u64().map(|u| u as usize).with_context(|| format!("bad {k}"))).collect()
        };
        let one = |v: &serde_json::Value, k: &str| -> anyhow::Result<usize> { v.get(k).and_then(|x| x.as_u64()).map(|u| u as usize).with_context(|| format!("missing {k}")) };
        let qs = ints(bb, "query_stride")?;
        let bg = ints(bb, "window_positional_embedding_background_size")?;
        Ok(Self {
            embed_dim: one(bb, "hidden_size")?,
            dims: ints(bb, "embed_dim_per_stage")?,
            blocks: ints(bb, "blocks_per_stage")?,
            heads: ints(bb, "num_attention_heads_per_stage")?,
            windows: ints(bb, "window_size_per_stage")?,
            global_blocks: ints(bb, "global_attention_blocks")?,
            query_stride: qs[0],
            pool_stages: one(bb, "num_query_pool_stages")?,
            mlp_ratio: bb.get("mlp_ratio").and_then(|x| x.as_f64()).unwrap_or(4.0),
            bg_size: (bg[0], bg[1]),
            channels: ints(vc, "backbone_channel_list")?,
            top_down: ints(vc, "fpn_top_down_levels")?,
            fpn_dim: one(vc, "fpn_hidden_size")?,
        })
    }
}

/// PyTorch `F.interpolate(mode="bicubic", align_corners=False)` of `[C, h, w]` (flattened, channel-major) to `[C, oh, ow]`.
pub(crate) fn bicubic(x: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let cc1 = |t: f32| ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let cc2 = |t: f32| ((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A;
    let weights = |t: f32| [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)];
    let axis = |n_in: usize, n_out: usize| -> Vec<([usize; 4], [f32; 4])> {
        let scale = n_in as f32 / n_out as f32;
        (0..n_out)
            .map(|o| {
                let src = (o as f32 + 0.5) * scale - 0.5;
                let f = src.floor();
                let idx = [-1i64, 0, 1, 2].map(|d| ((f as i64 + d).clamp(0, n_in as i64 - 1)) as usize);
                (idx, weights(src - f))
            })
            .collect()
    };
    let (ys, xs) = (axis(h, oh), axis(w, ow));
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        let plane = &x[ch * h * w..(ch + 1) * h * w];
        for (oy, (yi, yw)) in ys.iter().enumerate() {
            for (ox, (xi, xw)) in xs.iter().enumerate() {
                let mut acc = 0f32;
                for a in 0..4 {
                    let mut row = 0f32;
                    for b in 0..4 {
                        row += xw[b] * plane[yi[a] * w + xi[b]];
                    }
                    acc += yw[a] * row;
                }
                out[(ch * oh + oy) * ow + ox] = acc;
            }
        }
    }
    out
}

struct Block {
    ln1: LayerNorm,
    qkv_w: FastLinear,
    qkv_b: Tensor,
    proj_w: FastLinear,
    proj_b: Tensor,
    ln2: LayerNorm,
    fc1_w: FastLinear,
    fc1_b: Tensor,
    fc2_w: FastLinear,
    fc2_b: Tensor,
    res_proj: Option<Linear>,
    heads: usize,
    dim_out: usize,
    window: usize,
    stride: Option<usize>,
}

/// Max pool (kernel = stride = `s`) over the spatial dims of an NHWC tensor (even sizes).
fn pool(x: &Tensor, s: usize) -> Result<Tensor> {
    // candle's native pooling kernel is far faster than two strided `max` reductions
    x.permute((0, 3, 1, 2))?.contiguous()?.max_pool2d(s)?.permute((0, 2, 3, 1))?.contiguous()
}

fn partition(x: &Tensor, ws: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let (ph, pw) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
    let x = if ph > 0 { x.pad_with_zeros(1, 0, ph)? } else { x.clone() };
    let x = if pw > 0 { x.pad_with_zeros(2, 0, pw)? } else { x };
    let (hp, wp) = (h + ph, w + pw);
    let x = x.reshape((b, hp / ws, ws, wp / ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b * (hp / ws) * (wp / ws), ws, ws, c))?;
    Ok((x, (hp, wp)))
}

fn unpartition(x: &Tensor, ws: usize, (hp, wp): (usize, usize), (h, w): (usize, usize)) -> Result<Tensor> {
    let c = x.dim(3)?;
    let b = x.dim(0)? / ((hp / ws) * (wp / ws));
    let x = x.reshape((b, hp / ws, wp / ws, ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, hp, wp, c))?;
    x.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()
}

/// `[out, in]` weight -> `[in, out]` for `x @ w`.
fn wt(vb: &VarBuilder, out: usize, inp: usize) -> Result<Tensor> {
    vb.get((out, inp), "weight")?.t()?.contiguous()
}

/// Attention over `[nw, L, H, D]` q/k/v (q possibly pooled, so `Lq != Lk`) -> `[nw, Lq, H*D]`.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (nw, lq, h, d) = q.dims4()?;
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) {
        return candle_flash_attn::flash_attn(q, k, v, (d as f32).powf(-0.5), false)?.reshape((nw, lq, h * d));
    }
    if let Some(o) = crate::kernels::flash_attn_cpu(q, k, v)? {
        return o.reshape((nw, lq, h * d));
    }
    let t = |t: &Tensor| t.transpose(1, 2);
    sdpa(&t(q)?, &t(k)?, &t(v)?, None)?.transpose(1, 2)?.reshape((nw, lq, h * d))
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.ln1.forward(x)?;
        let mut residual = x.clone();
        if let Some(rp) = &self.res_proj {
            residual = rp.forward(&y)?;
            if let Some(s) = self.stride {
                residual = pool(&residual, s)?;
            }
        }
        let (h, w) = (x.dim(1)?, x.dim(2)?);
        let (y, pad) = if self.window > 0 { partition(&y, self.window)? } else { (y, (h, w)) };
        let (nw, hh, ww, dim) = y.dims4()?;
        let n = hh * ww;
        let hd = self.dim_out / self.heads;
        let qkv = self.qkv_w.forward(&y.reshape((nw * n, dim))?)?.broadcast_add(&self.qkv_b.to_dtype(y.dtype())?)?.reshape((nw, n, 3, self.heads, hd))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
        let (q, qh, qw) = match self.stride {
            Some(s) => {
                let qp = pool(&q.reshape((nw, hh, ww, self.heads * hd))?, s)?;
                let (qh, qw) = (qp.dim(1)?, qp.dim(2)?);
                (qp.reshape((nw, qh * qw, self.heads, hd))?, qh, qw)
            }
            None => (q, hh, ww),
        };
        // projection without its bias (it is fused into the residual add below)
        let o = self.proj_w.forward(&attend(&q, &k, &v)?.reshape((nw * qh * qw, self.dim_out))?)?.reshape((nw, qh, qw, self.dim_out))?;
        let o = if self.window > 0 {
            let ws = match self.stride {
                Some(s) => self.window / s,
                None => self.window,
            };
            let (rh, rw) = (residual.dim(1)?, residual.dim(2)?);
            let pad_hw = if self.stride.is_some() { (rh + (ws - rh % ws) % ws, rw + (ws - rw % ws) % ws) } else { pad };
            unpartition(&o, ws, pad_hw, (rh, rw))?
        } else {
            o
        };
        let x = bias_residual(&residual, &o, &self.proj_b)?;
        let (b, hx, wx, c) = x.dims4()?;
        let m = bias_gelu(&self.fc1_w.forward(&self.ln2.forward(&x)?.reshape((b * hx * wx, c))?)?, &self.fc1_b)?;
        bias_residual(&x, &self.fc2_w.forward(&m)?.reshape((b, hx, wx, c))?, &self.fc2_b)
    }
}

pub struct Hiera {
    patch: Conv2d,
    pos: Tensor,
    blocks: Vec<Block>,
    stage_ends: Vec<usize>,
    convs: Vec<Conv2d>,
    top_down: Vec<usize>,
}

impl Hiera {
    /// `vb` is the checkpoint root (`vision_encoder.backbone.*` and `vision_encoder.neck.*`). `image` is the input size.
    pub fn new(cfg: &HieraConfig, vb: VarBuilder, image: usize, device: &Device) -> Result<Self> {
        let bb = vb.pp("vision_encoder.backbone");
        let patch = conv2d(3, cfg.embed_dim, 7, Conv2dConfig { stride: 4, padding: 3, ..Default::default() }, bb.pp("patch_embed.projection"))?;
        // position embedding: bicubic-resized background + the window embedding tiled over the map
        let side = image / 4;
        let c = cfg.embed_dim;
        let bg = bb.get((1, c, cfg.bg_size.0, cfg.bg_size.1), "pos_embed")?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let win = bb.get((1, c, cfg.windows[0], cfg.windows[0]), "pos_embed_window")?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut pos = bicubic(&bg, c, cfg.bg_size.0, cfg.bg_size.1, side, side);
        let ws = cfg.windows[0];
        for ch in 0..c {
            for y in 0..side {
                for x in 0..side {
                    pos[(ch * side + y) * side + x] += win[(ch * ws + y % ws) * ws + x % ws];
                }
            }
        }
        let pos = Tensor::from_vec(pos, (1, c, side, side), device)?.permute((0, 2, 3, 1))?.contiguous()?;

        let mut blocks = vec![];
        let mut stage_ends = vec![];
        let mut total = 0;
        for (stage, &n) in cfg.blocks.iter().enumerate() {
            for b in 0..n {
                let first_of_later_stage = stage > 0 && b == 0;
                let dim = if first_of_later_stage { cfg.dims[stage - 1] } else { cfg.dims[stage] };
                let dim_out = cfg.dims[stage];
                let window = if cfg.global_blocks.contains(&total) { 0 } else if first_of_later_stage { cfg.windows[stage - 1] } else { cfg.windows[stage] };
                let stride = if stage > 0 && stage <= cfg.pool_stages && b == 0 { Some(cfg.query_stride) } else { None };
                let l = bb.pp(format!("blocks.{total}"));
                let hid = (dim_out as f64 * cfg.mlp_ratio) as usize;
                blocks.push(Block {
                    ln1: ln(dim, 1e-6, l.pp("layer_norm1"))?,
                    qkv_w: FastLinear::new(wt(&l.pp("attn.qkv"), dim_out * 3, dim)?)?,
                    qkv_b: l.pp("attn.qkv").get(dim_out * 3, "bias")?,
                    proj_w: FastLinear::new(wt(&l.pp("attn.proj"), dim_out, dim_out)?)?,
                    proj_b: l.pp("attn.proj").get(dim_out, "bias")?,
                    ln2: ln(dim_out, 1e-6, l.pp("layer_norm2"))?,
                    fc1_w: FastLinear::new(wt(&l.pp("mlp.proj_in"), hid, dim_out)?)?,
                    fc1_b: l.pp("mlp.proj_in").get(hid, "bias")?,
                    fc2_w: FastLinear::new(wt(&l.pp("mlp.proj_out"), dim_out, hid)?)?,
                    fc2_b: l.pp("mlp.proj_out").get(dim_out, "bias")?,
                    res_proj: if dim != dim_out { Some(linear(dim, dim_out, l.pp("proj"))?) } else { None },
                    heads: cfg.heads[stage],
                    dim_out,
                    window,
                    stride,
                });
                total += 1;
            }
            stage_ends.push(total - 1);
        }
        let neck = vb.pp("vision_encoder.neck");
        let convs = cfg.channels.iter().enumerate().map(|(i, &ch)| conv2d(ch, cfg.fpn_dim, 1, Default::default(), neck.pp(format!("convs.{i}")))).collect::<Result<_>>()?;
        Ok(Self { patch, pos, blocks, stage_ends, convs, top_down: cfg.top_down.clone() })
    }

    /// `[1, 3, S, S]` -> the three neck levels, highest resolution first (`[1, 256, S/4, S/4]`, `/8`, `/16`).
    pub fn forward(&self, img: &Tensor) -> Result<Vec<Tensor>> {
        let mut x = self.patch.forward(img)?.permute((0, 2, 3, 1))?.broadcast_add(&self.pos.to_dtype(img.dtype())?)?;
        let mut stages = vec![];
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x)?;
            if self.stage_ends.contains(&i) {
                stages.push(x.clone());
            }
        }
        let n = self.convs.len() - 1;
        let mut outs: Vec<Tensor> = vec![];
        let mut prev: Option<Tensor> = None;
        for i in (0..=n).rev() {
            let lateral = self.convs[n - i].forward(&stages[i].permute((0, 3, 1, 2))?.contiguous()?)?;
            let cur = match (&prev, self.top_down.contains(&i) && i != n) {
                (Some(p), true) => {
                    let (h, w) = (lateral.dim(2)?, lateral.dim(3)?);
                    (lateral + p.upsample_nearest2d(h, w)?)?
                }
                _ => lateral,
            };
            outs.push(cur.clone());
            prev = Some(cur);
        }
        // `outs` is deepest-first; keep the last three (i = 2, 1, 0) and put the finest first
        let k = outs.len();
        Ok(outs.split_off(k - 3).into_iter().rev().collect())
    }
}

/// Sine position embedding of the coarsest level as tokens `[1, side*side, 256]` (what memory attention adds).
pub fn level_pos(side: usize, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    sine_pos_nf(side, side, dim / 2, device, dtype)?.flatten_from(2)?.transpose(1, 2)?.contiguous()
}
