//! SAM image preprocessing: resize the longest side to 1024, normalize, zero-pad to a square.

use candle_core::{DType, Device, Result, Tensor};
use image::RgbImage;

pub const IMG_SIZE: usize = 1024;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

/// A preprocessed image plus the geometry needed to map prompts and masks back.
pub struct Prepared {
    /// `[1, 3, 1024, 1024]` normalized, bottom/right zero padded.
    pub tensor: Tensor,
    pub orig: (usize, usize),
    /// Size after the longest-side resize, before padding.
    pub resized: (usize, usize),
    /// Original-pixel -> 1024-space scale factor.
    pub scale: f32,
}

pub fn prepare(img: &RgbImage, size: usize, mean: [f32; 3], std: [f32; 3], device: &Device, dtype: DType) -> Result<Prepared> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let scale = size as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale) + 0.5) as usize;
    let nh = ((h as f32 * scale) + 0.5) as usize;
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        // Upload the raw pixels; one kernel resizes, normalizes and pads.
        let src = Tensor::from_raw_buffer(img.as_raw(), DType::U8, &[h, w, 3], device)?;
        let tensor = crate::kernels::preprocess(&src, (nh, nw), size, mean, std, dtype)?;
        return Ok(Prepared { tensor, orig: (h, w), resized: (nh, nw), scale });
    }
    let data = resize_normalize(img, nh, nw, size, mean, std);
    let tensor = Tensor::from_vec(data, (1, 3, size, size), device)?.to_dtype(dtype)?;
    Ok(Prepared { tensor, orig: (h, w), resized: (nh, nw), scale })
}

pub fn prepare_sam1(img: &RgbImage, device: &Device, dtype: DType) -> Result<Prepared> {
    prepare(img, IMG_SIZE, MEAN, STD, device, dtype)
}

/// Squashes the image to `size x size` (no aspect preservation, no padding), as SAM 3 does.
pub fn prepare_square(img: &RgbImage, size: usize, mean: [f32; 3], std: [f32; 3], device: &Device, dtype: DType) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        let (h, w) = (img.height() as usize, img.width() as usize);
        let src = Tensor::from_raw_buffer(img.as_raw(), DType::U8, &[h, w, 3], device)?;
        return crate::kernels::preprocess(&src, (size, size), size, mean, std, dtype);
    }
    Tensor::from_vec(resize_normalize(img, size, size, size, mean, std), (1, 3, size, size), device)?.to_dtype(dtype)
}

/// Antialiased triangle-filter resize to `nh x nw`, normalization, and zero padding into `[3, size, size]`.
/// The same arithmetic as the CUDA `preprocess` kernel (PIL's bilinear), parallel over output rows.
fn resize_normalize(img: &RgbImage, nh: usize, nw: usize, size: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    use rayon::prelude::*;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let (sx, sy) = (w as f32 / nw as f32, h as f32 / nh as f32);
    let (fx, fy) = (sx.max(1.0), sy.max(1.0));
    let tri = |x: f32| -> f32 {
        let x = x.abs();
        if x < 1.0 { 1.0 - x } else { 0.0 }
    };
    let raw = img.as_raw();
    let mut out = vec![0f32; 3 * size * size];
    // one task per (channel, row)
    out.par_chunks_mut(size).enumerate().for_each(|(i, row)| {
        let (c, y) = (i / size, i % size);
        if y >= nh {
            return;
        }
        let cy = (y as f32 + 0.5) * sy;
        let (y0, y1) = (((cy - fy + 0.5) as i64).max(0) as usize, ((cy + fy + 0.5) as usize).min(h));
        let wy: Vec<f32> = (y0..y1).map(|yy| tri((yy as f32 + 0.5 - cy) / fy)).collect();
        let wsy: f32 = wy.iter().sum();
        for x in 0..nw {
            let cx = (x as f32 + 0.5) * sx;
            let (x0, x1) = (((cx - fx + 0.5) as i64).max(0) as usize, ((cx + fx + 0.5) as usize).min(w));
            let mut wsx = 0f32;
            let mut acc = 0f32;
            for xx in x0..x1 {
                wsx += tri((xx as f32 + 0.5 - cx) / fx);
            }
            for (k, yy) in (y0..y1).enumerate() {
                let mut r = 0f32;
                for xx in x0..x1 {
                    r += tri((xx as f32 + 0.5 - cx) / fx) * raw[(yy * w + xx) * 3 + c] as f32;
                }
                acc += wy[k] * r;
            }
            row[x] = (acc / (wsx * wsy) - mean[c]) / std[c];
        }
    });
    out
}

/// Bilinear (align_corners=false) resize of an `n x n` logit map to `oh x ow` and threshold at 0, in parallel.
pub fn upsample_threshold(logits: &[f32], n: usize, oh: usize, ow: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let tap = |dst: usize, scale: f32| -> (usize, usize, f32) {
        let src = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src as usize).min(n - 1);
        (i0, (i0 + 1).min(n - 1), src - i0 as f32)
    };
    let xt: Vec<_> = (0..ow).map(|x| tap(x, n as f32 / ow as f32)).collect();
    let mut out = vec![0u8; oh * ow];
    out.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = tap(y, n as f32 / oh as f32);
        let (r0, r1) = (&logits[y0 * n..(y0 + 1) * n], &logits[y1 * n..(y1 + 1) * n]);
        for (o, &(x0, x1, fx)) in row.iter_mut().zip(&xt) {
            let top = r0[x0] * (1.0 - fx) + r0[x1] * fx;
            let bot = r1[x0] * (1.0 - fx) + r1[x1] * fx;
            *o = (top * (1.0 - fy) + bot * fy > 0.0) as u8;
        }
    });
    out
}

/// Squashes the image to `size x size` with a plain bilinear (half-pixel, no antialiasing) resize, then normalizes.
/// This is what the EfficientSAM graphs do, and differs from [`prepare_square`] when shrinking.
pub fn prepare_square_plain(img: &RgbImage, size: usize, mean: [f32; 3], std: [f32; 3], device: &Device, dtype: DType) -> Result<Tensor> {
    use rayon::prelude::*;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let raw = img.as_raw();
    let (sx, sy) = (w as f32 / size as f32, h as f32 / size as f32);
    let tap = |dst: usize, scale: f32, n: usize| -> (usize, usize, f32) {
        let src = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src as usize).min(n - 1);
        (i0, (i0 + 1).min(n - 1), src - i0 as f32)
    };
    let xt: Vec<_> = (0..size).map(|x| tap(x, sx, w)).collect();
    let mut out = vec![0f32; 3 * size * size];
    out.par_chunks_mut(size).enumerate().for_each(|(i, row)| {
        let (c, y) = (i / size, i % size);
        let (y0, y1, fy) = tap(y, sy, h);
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1, fx) = xt[x];
            let px = |yy: usize, xx: usize| raw[(yy * w + xx) * 3 + c] as f32;
            let top = px(y0, x0) * (1.0 - fx) + px(y0, x1) * fx;
            let bot = px(y1, x0) * (1.0 - fx) + px(y1, x1) * fx;
            *o = ((top * (1.0 - fy) + bot * fy) - mean[c]) / std[c];
        }
    });
    Tensor::from_vec(out, (1, 3, size, size), device)?.to_dtype(dtype)
}
