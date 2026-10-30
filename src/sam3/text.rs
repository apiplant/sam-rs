//! CLIP-style text encoder (24 layers, causal attention) followed by the 1024 -> 256 projection.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::nn::{ln, Act, Attn, Mlp};

const DIM: usize = 1024;
const LAYERS: usize = 24;
pub const MAX_LEN: usize = 32;

struct Layer {
    ln1: LayerNorm,
    attn: Attn,
    ln2: LayerNorm,
    mlp: Mlp,
}

pub struct TextEncoder {
    tok: Embedding,
    pos: Tensor,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    proj: Linear,
    causal: Tensor,
}

impl TextEncoder {
    /// `vb` is the `detector_model` root.
    pub fn new(vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let m = vb.pp("text_encoder.text_model");
        let layers = (0..LAYERS)
            .map(|i| {
                let l = m.pp(format!("encoder.layers.{i}"));
                Ok(Layer {
                    ln1: ln(DIM, 1e-5, l.pp("layer_norm1"))?,
                    attn: Attn::with_out(DIM, 16, l.pp("self_attn"), "out_proj")?,
                    ln2: ln(DIM, 1e-5, l.pp("layer_norm2"))?,
                    mlp: Mlp::new(DIM, 4096, Act::Gelu, l.pp("mlp"))?,
                })
            })
            .collect::<Result<_>>()?;
        let mut mask = vec![0f32; MAX_LEN * MAX_LEN];
        for i in 0..MAX_LEN {
            for j in i + 1..MAX_LEN {
                mask[i * MAX_LEN + j] = -1e9;
            }
        }
        Ok(Self {
            tok: embedding(49408, DIM, m.pp("embeddings.token_embedding"))?,
            pos: m.get((MAX_LEN, DIM), "embeddings.position_embedding.weight")?,
            layers,
            final_ln: ln(DIM, 1e-5, m.pp("final_layer_norm"))?,
            proj: linear(DIM, 256, vb.pp("text_projection"))?,
            causal: Tensor::from_vec(mask, (1, 1, MAX_LEN, MAX_LEN), device)?.to_dtype(dtype)?,
        })
    }

    /// Token ids (padded to 32) -> `[1, 32, 256]` prompt features.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let device = self.pos.device();
        let ids = Tensor::from_slice(ids, (1, ids.len()), device)?;
        let mut x = self.tok.forward(&ids)?.broadcast_add(&self.pos)?;
        for l in &self.layers {
            let y = l.ln1.forward(&x)?;
            x = (&x + l.attn.forward(&y, &y, &y, Some(&self.causal))?)?;
            x = (&x + l.mlp.forward(&l.ln2.forward(&x)?)?)?;
        }
        self.proj.forward(&self.final_ln.forward(&x)?)
    }
}
