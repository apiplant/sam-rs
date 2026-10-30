//! Prompt encoder: points / boxes -> sparse tokens, plus the dense positional encoding.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub const IMG: f32 = 1024.0;
pub const EMB: usize = 64;

pub struct PromptEncoder {
    gauss: Tensor,        // [2, 128]
    point_embed: Vec<Tensor>, // 4 x [256]
    not_a_point: Tensor,  // [256]
    no_mask: Tensor,      // [256]
}

impl PromptEncoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let point_embed = (0..4).map(|i| vb.get((1, 256), &format!("point_embeddings.{i}.weight"))?.reshape(256)).collect::<Result<_>>()?;
        Ok(Self {
            gauss: vb.get((2, 128), "pe_layer.positional_encoding_gaussian_matrix")?,
            point_embed,
            not_a_point: vb.get((1, 256), "not_a_point_embed.weight")?.reshape(256)?,
            no_mask: vb.get((1, 256), "no_mask_embed.weight")?.reshape(256)?,
        })
    }

    /// coords in [0,1], shape `[..., 2]` -> `[..., 256]`
    fn pe(&self, coords: &Tensor) -> Result<Tensor> {
        let c = (coords.affine(2.0, -1.0)?).broadcast_matmul(&self.gauss.to_dtype(coords.dtype())?)?;
        let c = (c * (2.0 * std::f64::consts::PI))?;
        Tensor::cat(&[c.sin()?, c.cos()?], candle_core::D::Minus1)
    }

    /// Dense positional encoding `[1, 256, 64, 64]`.
    pub fn dense_pe(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let n = EMB;
        let mut g = Vec::with_capacity(n * n * 2);
        for y in 0..n {
            for x in 0..n {
                g.push((x as f32 + 0.5) / n as f32);
                g.push((y as f32 + 0.5) / n as f32);
            }
        }
        let g = Tensor::from_vec(g, (n, n, 2), device)?.to_dtype(dtype)?;
        self.pe(&g)?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()
    }

    pub fn dense_no_mask(&self) -> Result<Tensor> {
        self.no_mask.reshape((1, 256, 1, 1))?.broadcast_as((1, 256, EMB, EMB))?.contiguous()
    }

    /// `points`: `(x, y, label)` in 1024-space; label 1 fg, 0 bg, 2/3 box corners.
    /// Returns sparse embeddings `[1, N, 256]`.
    pub fn sparse(&self, points: &[(f32, f32, i32)], device: &Device, dtype: DType) -> Result<Tensor> {
        let mut pts: Vec<(f32, f32, i32)> = points.to_vec();
        if !pts.iter().any(|p| p.2 >= 2) {
            pts.push((-0.5, -0.5, -1)); // padding point when there is no box
        }
        let n = pts.len();
        let mut xy = Vec::with_capacity(n * 2);
        for p in &pts {
            xy.push((p.0 + 0.5) / IMG);
            xy.push((p.1 + 0.5) / IMG);
        }
        let xy = Tensor::from_vec(xy, (n, 2), device)?.to_dtype(dtype)?;
        let pe = self.pe(&xy)?;
        let mut rows = Vec::with_capacity(n);
        for (i, p) in pts.iter().enumerate() {
            let r = pe.get(i)?;
            rows.push(match p.2 {
                -1 => self.not_a_point.to_dtype(dtype)?,
                l => (r + self.point_embed[l as usize].to_dtype(dtype)?)?,
            });
        }
        Tensor::stack(&rows, 0)?.unsqueeze(0)
    }
}
