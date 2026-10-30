//! Small building blocks shared by the SAM 3 stages.

use candle_core::{DType, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone, Copy)]
pub enum Act {
    Relu,
    Gelu,
}

impl Act {
    pub fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Act::Relu => x.relu(),
            Act::Gelu => x.gelu_erf(),
        }
    }
}

pub fn ln(d: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    layer_norm(d, eps, vb)
}

/// Scaled dot-product attention over `[B, H, L, D]` tensors with an optional additive mask.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let scale = (q.dim(3)? as f64).powf(-0.5);
    let s = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let s = match mask {
        Some(m) => s.broadcast_add(m)?,
        None => s,
    };
    candle_nn::ops::softmax_last_dim(&s)?.matmul(&v.contiguous()?)
}

/// `Sam3Attention`: four biased projections, `heads` heads, optional additive mask.
pub struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attn {
    pub fn new(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_out(hidden, heads, vb, "o_proj")
    }

    /// Same, with a different name for the output projection (`out_proj` in CLIP).
    pub fn with_out(hidden: usize, heads: usize, vb: VarBuilder, out: &str) -> Result<Self> {
        Ok(Self {
            q: linear(hidden, hidden, vb.pp("q_proj"))?,
            k: linear(hidden, hidden, vb.pp("k_proj"))?,
            v: linear(hidden, hidden, vb.pp("v_proj"))?,
            o: linear(hidden, hidden, vb.pp(out))?,
            heads,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, lq, e) = q.dims3()?;
        let lk = k.dim(1)?;
        let (h, d) = (self.heads, e / self.heads);
        #[cfg(feature = "flash-attn")]
        if mask.is_none() && q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) {
            // FlashAttention reads the natural [B, L, H, D] layout, so no transposes are needed
            let proj = |l: &Linear, t: &Tensor, n: usize| -> Result<Tensor> { l.forward(t)?.reshape((b, n, h, d)) };
            let o = candle_flash_attn::flash_attn(&proj(&self.q, q, lq)?, &proj(&self.k, k, lk)?, &proj(&self.v, v, lk)?, (d as f32).powf(-0.5), false)?;
            return self.o.forward(&o.reshape((b, lq, e))?);
        }
        if mask.is_none() && lq == lk {
            let proj = |l: &Linear, t: &Tensor| -> Result<Tensor> { l.forward(t)?.reshape((b, lq, h, d)) };
            if let Some(o) = crate::kernels::flash_attn_cpu(&proj(&self.q, q)?, &proj(&self.k, k)?, &proj(&self.v, v)?)? {
                return self.o.forward(&o.reshape((b, lq, e))?);
            }
        }
        let split = |t: Tensor, l: usize| t.reshape((b, l, h, d))?.transpose(1, 2);
        let (q, k, v) = (split(self.q.forward(q)?, lq)?, split(self.k.forward(k)?, lk)?, split(self.v.forward(v)?, lk)?);
        let o = sdpa(&q, &k, &v, mask)?.transpose(1, 2)?.reshape((b, lq, e))?;
        self.o.forward(&o)
    }
}

/// `fc1 -> act -> fc2`.
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    act: Act,
}

impl Mlp {
    pub fn new(hidden: usize, inter: usize, act: Act, vb: VarBuilder) -> Result<Self> {
        Ok(Self { fc1: linear(hidden, inter, vb.pp("fc1"))?, fc2: linear(inter, hidden, vb.pp("fc2"))?, act })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.apply(&self.fc1.forward(x)?)?)
    }
}

/// `Sam3DecoderMLP`: 2 or 3 linear layers named `layer1..`, ReLU in between.
pub struct Mlp3 {
    layers: Vec<Linear>,
}

impl Mlp3 {
    pub fn new(i: usize, h: usize, o: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        let layers = (0..n)
            .map(|j| linear(if j == 0 { i } else { h }, if j + 1 == n { o } else { h }, vb.pp(format!("layer{}", j + 1))))
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.layers.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

/// Additive attention mask `[1, 1, 1, L]` from a 0/1 key-validity vector (`-inf`-like where invalid).
pub fn key_padding_mask(valid: &[u8], dtype: DType, device: &candle_core::Device) -> Result<Tensor> {
    let v: Vec<f32> = valid.iter().map(|&m| if m != 0 { 0.0 } else { -1e9 }).collect();
    Tensor::from_vec(v, (1, 1, 1, valid.len()), device)?.to_dtype(dtype)
}

pub fn inverse_sigmoid(x: &Tensor) -> Result<Tensor> {
    let x = x.clamp(0f64, 1f64)?;
    let x1 = x.clamp(1e-3, f64::MAX)?;
    let x2 = (1.0 - &x)?.clamp(1e-3, f64::MAX)?;
    (x1 / x2)?.log()
}

/// A stride-1 "same" convolution that uses the bf16 CPU kernel when it can (1x1 / 3x3, f32, CPU, `enabled()`).
pub struct FastConv {
    base: candle_nn::Conv2d,
    fast: Option<crate::cpu_gemm::Conv>,
}

impl FastConv {
    pub fn new(cin: usize, cout: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((cout, cin, k, k), "weight")?;
        let b = vb.get(cout, "bias")?;
        let cfg = candle_nn::Conv2dConfig { padding: k / 2, ..Default::default() };
        let fast = if w.device().is_cpu() && w.dtype() == DType::F32 && crate::cpu_gemm::enabled() && (k == 1 || k == 3) {
            Some(crate::cpu_gemm::Conv::new(&w, Some(&b))?)
        } else {
            None
        };
        Ok(Self { base: candle_nn::Conv2d::new(w, Some(b), cfg), fast })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.fast {
            Some(f) if x.device().is_cpu() && x.dtype() == DType::F32 && x.dim(0)? == 1 => f.forward(x),
            _ => self.base.forward(x),
        }
    }
}
