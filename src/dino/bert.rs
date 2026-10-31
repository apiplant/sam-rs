//! The BERT text encoder of Grounding DINO (`bert-base-uncased`, no pooler). Unlike a stock BERT it takes a full
//! `[L, L]` boolean attention mask (which tokens may attend to which) and explicit position ids, because phrases
//! separated by special tokens must not see each other.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln1: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ln2: LayerNorm,
}

pub struct Bert {
    words: Embedding,
    positions: Embedding,
    types: Embedding,
    ln: LayerNorm,
    layers: Vec<Layer>,
    heads: usize,
}

impl Bert {
    pub fn new(vb: VarBuilder, hidden: usize, layers: usize, heads: usize, ffn: usize, vocab: usize) -> Result<Self> {
        let e = vb.pp("embeddings");
        let eps = 1e-12;
        let layers = (0..layers)
            .map(|i| {
                let l = vb.pp(format!("encoder.layer.{i}"));
                let a = l.pp("attention");
                Ok(Layer {
                    q: linear(hidden, hidden, a.pp("self.query"))?,
                    k: linear(hidden, hidden, a.pp("self.key"))?,
                    v: linear(hidden, hidden, a.pp("self.value"))?,
                    out: linear(hidden, hidden, a.pp("output.dense"))?,
                    ln1: layer_norm(hidden, eps, a.pp("output.LayerNorm"))?,
                    fc1: linear(hidden, ffn, l.pp("intermediate.dense"))?,
                    fc2: linear(ffn, hidden, l.pp("output.dense"))?,
                    ln2: layer_norm(hidden, eps, l.pp("output.LayerNorm"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            words: embedding(vocab, hidden, e.pp("word_embeddings"))?,
            positions: embedding(512, hidden, e.pp("position_embeddings"))?,
            types: embedding(2, hidden, e.pp("token_type_embeddings"))?,
            ln: layer_norm(hidden, eps, e.pp("LayerNorm"))?,
            layers,
            heads,
        })
    }

    /// `ids`, `position_ids`: `[L]`; `allow`: `[L, L]` 0/1 (1 = may attend). Returns `[L, hidden]` in f32.
    pub fn forward(&self, ids: &[u32], position_ids: &[u32], allow: &[u8], device: &Device) -> Result<Tensor> {
        let l = ids.len();
        let ids_t = Tensor::new(ids, device)?;
        let pos_t = Tensor::new(position_ids, device)?;
        let types = Tensor::zeros(l, DType::U32, device)?;
        let mut x = ((self.words.forward(&ids_t)? + self.positions.forward(&pos_t)?)? + self.types.forward(&types)?)?;
        x = self.ln.forward(&x)?;
        let mask: Vec<f32> = allow.iter().map(|&a| if a > 0 { 0.0 } else { -1e9 }).collect();
        let mask = Tensor::from_vec(mask, (1, l, l), device)?;
        let hd = x.dim(1)? / self.heads;
        for ly in &self.layers {
            let split = |t: Tensor| t.reshape((l, self.heads, hd))?.transpose(0, 1)?.contiguous();
            let (q, k, v) = (split(ly.q.forward(&x)?)?, split(ly.k.forward(&x)?)?, split(ly.v.forward(&x)?)?);
            let s = (q.matmul(&k.transpose(1, 2)?)? / (hd as f64).sqrt())?.broadcast_add(&mask)?;
            let a = candle_nn::ops::softmax(&s, D::Minus1)?.matmul(&v)?.transpose(0, 1)?.reshape((l, self.heads * hd))?;
            x = ly.ln1.forward(&(x + ly.out.forward(&a)?)?)?;
            let f = ly.fc2.forward(&ly.fc1.forward(&x)?.gelu_erf()?)?;
            x = ly.ln2.forward(&(x + f)?)?;
        }
        Ok(x)
    }
}
