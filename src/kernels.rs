//! Fused CUDA kernels (built from `kernels/*.cu`), each with a tensor-op fallback for CPU.

use candle_core::{Result, Tensor};
use candle_nn::Module as _;

/// Decomposed-relative-position attention softmax.
///
/// `softmax_{kh,kw}(scale * scores[bh, i*W+j, kh*W+kw] + q[bh,i*W+j,:]·rel_h[i-kh+H-1] + q[bh,i*W+j,:]·rel_w[j-kw+W-1])`
///
/// `scores`: `[BH, H*W, H*W]` raw `q·kᵀ`; `q`: `[BH, H*W, hd]`; `table`: `rel_pos_h` rows followed by
/// `rel_pos_w` rows, `[(2H-1)+(2W-1), hd]`. Returns `[BH, H*W, H*W]`.
pub fn relpos_softmax(scores: &Tensor, q: &Tensor, table: &Tensor, h: usize, w: usize, scale: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if scores.device().is_cuda() && h * w <= cuda::MAX_ROW {
        return scores.apply_op3_no_bwd(q, table, &cuda::RelposSoftmax { h, w, hd: q.dim(2)?, scale });
    }
    if scores.device().is_cpu() && scores.dtype() == candle_core::DType::F32 {
        return scores.contiguous()?.apply_op3_no_bwd(&q.contiguous()?, &table.contiguous()?, &cpu::RelposSoftmax { h, w, hd: q.dim(2)?, scale });
    }
    let (bh, hd) = (scores.dim(0)?, q.dim(2)?);
    let idx = |size: usize| -> Result<Tensor> {
        let mut v = Vec::with_capacity(size * size);
        for qi in 0..size {
            for ki in 0..size {
                v.push((qi + size - 1 - ki) as u32);
            }
        }
        Tensor::from_vec(v, size * size, q.device())
    };
    let th = table.narrow(0, 0, 2 * h - 1)?;
    let tw = table.narrow(0, 2 * h - 1, 2 * w - 1)?;
    let rh = th.index_select(&idx(h)?, 0)?.reshape((h, h, hd))?;
    let rw = tw.index_select(&idx(w)?, 0)?.reshape((w, w, hd))?;
    let rq = q.reshape((bh, h, w, hd))?;
    let rel_h = rq.permute((1, 0, 2, 3))?.reshape((h, bh * w, hd))?.matmul(&rh.transpose(1, 2)?.contiguous()?)?.reshape((h, bh, w, h))?.permute((1, 0, 2, 3))?;
    let rel_w = rq.permute((2, 0, 1, 3))?.reshape((w, bh * h, hd))?.matmul(&rw.transpose(1, 2)?.contiguous()?)?.reshape((w, bh, h, w))?.permute((1, 2, 0, 3))?;
    let a = scores
        .affine(scale as f64, 0.)?
        .reshape((bh, h, w, h, w))?
        .broadcast_add(&rel_h.unsqueeze(4)?)?
        .broadcast_add(&rel_w.unsqueeze(3)?)?;
    candle_nn::ops::softmax_last_dim(&a.reshape((bh, h * w, h * w))?)
}

mod cpu {
    use candle_core::{bail, CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape};
    use rayon::prelude::*;

    /// Same math as the CUDA kernel: bias vectors from `q` and the tables, then a row softmax, one row per task.
    pub struct RelposSoftmax {
        pub h: usize,
        pub w: usize,
        pub hd: usize,
        pub scale: f32,
    }

    impl CustomOp3 for RelposSoftmax {
        fn name(&self) -> &'static str {
            "relpos-softmax"
        }

        fn cpu_fwd(&self, s: &CpuStorage, sl: &Layout, q: &CpuStorage, ql: &Layout, t: &CpuStorage, tl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (CpuStorage::F32(s), CpuStorage::F32(q), CpuStorage::F32(t)) = (s, q, t) else { bail!("relpos-softmax: f32 only on CPU") };
            let (Some((s0, s1)), Some((q0, _)), Some((t0, _))) = (sl.contiguous_offsets(), ql.contiguous_offsets(), tl.contiguous_offsets()) else {
                bail!("relpos-softmax needs contiguous inputs")
            };
            let (h, w, hd, scale) = (self.h, self.w, self.hd, self.scale);
            let n = h * w;
            let (s, q, t) = (&s[s0..s1], &q[q0..], &t[t0..]);
            let (th, tw) = (&t[..(2 * h - 1) * hd], &t[(2 * h - 1) * hd..]);
            let mut out = vec![0f32; s.len()];
            out.par_chunks_mut(n).enumerate().for_each_init(
                || (vec![0f32; h], vec![0f32; w]),
                |(rh, rw), (row, o)| {
                    let (i, j) = ((row % n) / w, row % w);
                    let qr = &q[row * hd..(row + 1) * hd];
                    let dot = |tab: &[f32], r: usize| -> f32 { tab[r * hd..(r + 1) * hd].iter().zip(qr).map(|(a, b)| a * b).sum() };
                    for kh in 0..h {
                        rh[kh] = dot(th, i + h - 1 - kh);
                    }
                    for kw in 0..w {
                        rw[kw] = dot(tw, j + w - 1 - kw);
                    }
                    let sr = &s[row * n..(row + 1) * n];
                    let mut max = f32::NEG_INFINITY;
                    for kh in 0..h {
                        for kw in 0..w {
                            let v = sr[kh * w + kw] * scale + rh[kh] + rw[kw];
                            o[kh * w + kw] = v;
                            max = max.max(v);
                        }
                    }
                    let mut sum = 0f32;
                    for v in o.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    for v in o.iter_mut() {
                        *v *= inv;
                    }
                },
            );
            Ok((CpuStorage::F32(out), sl.shape().clone()))
        }
    }

    fn f32s<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
        match (s, l.contiguous_offsets()) {
            (CpuStorage::F32(v), Some((a, b))) => Ok(&v[a..b]),
            _ => bail!("expected contiguous f32 input"),
        }
    }

    /// Rows of queries handled per task; the `TILE x H*W` score block (512 KB for 4096 keys) stays in L2.
    const TILE: usize = 32;

    pub struct TiledAttention {
        pub h: usize,
        pub w: usize,
        pub hd: usize,
        pub scale: f32,
        pub table: Vec<f32>,
    }

    impl CustomOp3 for TiledAttention {
        fn name(&self) -> &'static str {
            "tiled-attention"
        }

        fn cpu_fwd(&self, q: &CpuStorage, ql: &Layout, kt: &CpuStorage, kl: &Layout, v: &CpuStorage, vl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (q, kt, v) = (f32s(q, ql)?, f32s(kt, kl)?, f32s(v, vl)?);
            let (h, w, hd, scale) = (self.h, self.w, self.hd, self.scale);
            let n = h * w;
            let bh = ql.shape().dims3()?.0;
            let (th, tw) = (&self.table[..(2 * h - 1) * hd], &self.table[(2 * h - 1) * hd..]);
            let tiles = n.div_ceil(TILE);
            // bf16 path: keys and values are packed once per head, then every tile runs both GEMMs on vdpbf16ps
            let packs = crate::cpu_gemm::enabled().then(|| {
                (0..bh)
                    .into_par_iter()
                    .map(|b| (crate::cpu_gemm::PackedB::from_slice(&kt[b * hd * n..(b + 1) * hd * n], hd, n), crate::cpu_gemm::PackedB::from_slice(&v[b * n * hd..(b + 1) * n * hd], n, hd)))
                    .collect::<Vec<_>>()
            });
            let mut out = vec![0f32; bh * n * hd];
            // one task per (head, query tile); each writes its own `rows * hd` block of `out`
            struct Out(*mut f32);
            // SAFETY: tasks write disjoint blocks (see below) and `out` outlives the parallel loop.
            unsafe impl Sync for Out {}
            impl Out {
                fn at(&self, off: usize) -> *mut f32 {
                    // SAFETY: callers stay inside the allocation.
                    unsafe { self.0.add(off) }
                }
            }
            let base = Out(out.as_mut_ptr());
            (0..bh * tiles).into_par_iter().for_each_init(
                || (vec![0f32; TILE * n], vec![0f32; h], vec![0f32; w]),
                |(sc, rh, rw), task| {
                    let (b, t) = (task / tiles, task % tiles);
                    let (r0, rows) = (t * TILE, TILE.min(n - t * TILE));
                    // SAFETY: rows [r0, r0+rows) of head b belong to this task alone.
                    let o = unsafe { std::slice::from_raw_parts_mut(base.at((b * n + r0) * hd), rows * hd) };
                    let qb = &q[(b * n + r0) * hd..];
                    let (ktb, vb) = (&kt[b * hd * n..], &v[b * n * hd..]);
                    if let Some(packs) = &packs {
                        let (kp, vp) = &packs[b];
                        // SAFETY: `enabled()` implies AVX-512 BF16 + F; buffers are sized rows*hd / rows*n / rows*hd.
                        unsafe {
                            kp.matmul_rows(&qb[..rows * hd], rows, &mut sc[..rows * n]);
                            for r in 0..rows {
                                let row = r0 + r;
                                let (i, j) = (row / w, row % w);
                                crate::cpu_gemm::rel_bias(&qb[r * hd..(r + 1) * hd], th, tw, h, w, hd, i, j, rh, rw);
                                crate::cpu_gemm::softmax_row(&mut sc[r * n..(r + 1) * n], rh, rw, h, w, scale);
                            }
                            vp.matmul_rows(&sc[..rows * n], rows, o);
                        }
                        return;
                    }
                    // scores[rows, n] = q_tile[rows, hd] @ kt[hd, n]
                    // SAFETY: slices hold rows*hd, hd*n and rows*n elements.
                    unsafe {
                        gemm::gemm(rows, n, hd, sc.as_mut_ptr(), 1, n as isize, false, qb.as_ptr(), 1, hd as isize, ktb.as_ptr(), 1, n as isize, 0.0, 1.0, false, false, false, gemm::Parallelism::None);
                    }
                    for r in 0..rows {
                        let row = r0 + r;
                        let (i, j) = (row / w, row % w);
                        let qr = &qb[r * hd..(r + 1) * hd];
                        let dot = |tab: &[f32], idx: usize| -> f32 { tab[idx * hd..(idx + 1) * hd].iter().zip(qr).map(|(a, b)| a * b).sum() };
                        for kh in 0..h {
                            rh[kh] = dot(th, i + h - 1 - kh);
                        }
                        for kw in 0..w {
                            rw[kw] = dot(tw, j + w - 1 - kw);
                        }
                        let s = &mut sc[r * n..(r + 1) * n];
                        let mut max = f32::NEG_INFINITY;
                        for kh in 0..h {
                            for kw in 0..w {
                                let x = &mut s[kh * w + kw];
                                *x = *x * scale + rh[kh] + rw[kw];
                                max = max.max(*x);
                            }
                        }
                        let mut sum = 0f32;
                        for x in s.iter_mut() {
                            *x = (*x - max).exp();
                            sum += *x;
                        }
                        let inv = 1.0 / sum;
                        for x in s.iter_mut() {
                            *x *= inv;
                        }
                    }
                    // out_tile[rows, hd] = probs[rows, n] @ v[n, hd]
                    // SAFETY: as above.
                    unsafe {
                        gemm::gemm(rows, hd, n, o.as_mut_ptr(), 1, hd as isize, false, sc.as_ptr(), 1, n as isize, vb.as_ptr(), 1, hd as isize, 0.0, 1.0, false, false, false, gemm::Parallelism::None);
                    }
                },
            );
            Ok((CpuStorage::F32(out), ql.shape().clone()))
        }
    }

    /// Same layout contract as the CUDA `qkv_split`: `[3, BH*n*hd]` holding q, kᵀ, v back to back.
    pub struct QkvSplit {
        pub n: usize,
        pub nh: usize,
        pub hd: usize,
    }

    impl CustomOp2 for QkvSplit {
        fn name(&self) -> &'static str {
            "qkv-split"
        }

        fn cpu_fwd(&self, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (y, b) = (f32s(y, yl)?, f32s(b, bl)?);
            let (n, nh, hd) = (self.n, self.nh, self.hd);
            let c = nh * hd;
            let total = y.len();
            let plane = total / 3;
            let mut out = vec![0f32; total];
            let base = SendPtr(out.as_mut_ptr());
            // one task per token; its q/k/v elements land in disjoint places of `out`
            (0..total / (3 * c)).into_par_iter().for_each(|t| {
                let (win, tok) = (t / n, t % n);
                for which in 0..3 {
                    for head in 0..nh {
                        let bh = win * nh + head;
                        for d in 0..hd {
                            let col = which * c + head * hd + d;
                            let dst = if which == 1 { (bh * hd + d) * n + tok } else { (bh * n + tok) * hd + d };
                            // SAFETY: (which, bh, tok, d) uniquely determines `dst`, and `t` fixes (win, tok).
                            unsafe { *base.at(which * plane + dst) = y[t * 3 * c + col] + b[col] };
                        }
                    }
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((3, plane))))
        }
    }

    struct SendPtr(*mut f32);
    // SAFETY: only used for provably disjoint writes inside a parallel loop.
    unsafe impl Sync for SendPtr {}
    impl SendPtr {
        fn at(&self, off: usize) -> *mut f32 {
            // SAFETY: callers stay inside the allocation.
            unsafe { self.0.add(off) }
        }
    }

    /// LayerNorm over the channel dim of an NCHW tensor, parallel over channel-planes of pixels.
    pub struct LayerNorm2d {
        pub eps: f32,
    }

    impl CustomOp3 for LayerNorm2d {
        fn name(&self) -> &'static str {
            "layernorm2d"
        }

        fn cpu_fwd(&self, x: &CpuStorage, xl: &Layout, w: &CpuStorage, wl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (x, w, b) = (f32s(x, xl)?, f32s(w, wl)?, f32s(b, bl)?);
            let (batch, c, h, wd) = xl.shape().dims4()?;
            let hw = h * wd;
            let mut out = vec![0f32; x.len()];
            let base = SendPtr(out.as_mut_ptr());
            // pixel blocks of 256: statistics are read along the (strided) channel axis
            let blocks = (batch * hw).div_ceil(256);
            (0..blocks).into_par_iter().for_each(|blk| {
                let lo = blk * 256;
                let hi = (lo + 256).min(batch * hw);
                for p in lo..hi {
                    let (bi, px) = (p / hw, p % hw);
                    let at = |ch: usize| (bi * c + ch) * hw + px;
                    let mut mean = 0f32;
                    for ch in 0..c {
                        mean += x[at(ch)];
                    }
                    mean /= c as f32;
                    let mut var = 0f32;
                    for ch in 0..c {
                        let d = x[at(ch)] - mean;
                        var += d * d;
                    }
                    let inv = 1.0 / (var / c as f32 + self.eps).sqrt();
                    for ch in 0..c {
                        // SAFETY: pixel p owns exactly the elements at(0..c).
                        unsafe { *base.at(at(ch)) = (x[at(ch)] - mean) * inv * w[ch] + b[ch] };
                    }
                }
            });
            Ok((CpuStorage::F32(out), xl.shape().clone()))
        }
    }

    /// CPU twin of the CUDA `qkv_rope`: bias add, 2-D RoPE on q and k, packed `[3, T, H, HD]` output.
    pub struct QkvRope {
        pub n: usize,
        pub heads: usize,
        pub hd: usize,
        pub sin: candle_core::Tensor,
    }

    impl CustomOp3 for QkvRope {
        fn name(&self) -> &'static str {
            "qkv-rope-cpu"
        }

        fn cpu_fwd(&self, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout, c: &CpuStorage, cl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (y, b, cos) = (f32s(y, yl)?, f32s(b, bl)?, f32s(c, cl)?);
            let (sin_t, sin_l) = self.sin.storage_and_layout();
            let candle_core::Storage::Cpu(sin_s) = &*sin_t else { bail!("sin table must be on the CPU") };
            let sin = f32s(sin_s, sin_l)?;
            let (n, heads, hd) = (self.n, self.heads, self.hd);
            let cdim = heads * hd;
            let rows = y.len() / (3 * cdim);
            let mut out = vec![0f32; y.len()];
            // out is [3, rows, C]: one task per token writes its q, k and v rows
            let base = SendPtr(out.as_mut_ptr());
            (0..rows).into_par_iter().for_each(|t| {
                let tok = t % n;
                for which in 0..3 {
                    let src = &y[t * 3 * cdim + which * cdim..][..cdim];
                    let bias = &b[which * cdim..][..cdim];
                    // SAFETY: (which, t) owns this C-wide row of `out`.
                    let dst = unsafe { std::slice::from_raw_parts_mut(base.at(which * rows * cdim + t * cdim), cdim) };
                    for h in 0..heads {
                        for j in 0..hd / 2 {
                            let i = h * hd + 2 * j;
                            let (x0, x1) = (src[i] + bias[i], src[i + 1] + bias[i + 1]);
                            if which < 2 {
                                let (cs, sn) = (cos[tok * hd + 2 * j], sin[tok * hd + 2 * j]);
                                dst[i] = x0 * cs - x1 * sn;
                                dst[i + 1] = x1 * cs + x0 * sn;
                            } else {
                                dst[i] = x0;
                                dst[i + 1] = x1;
                            }
                        }
                    }
                }
            });
            Ok((CpuStorage::F32(out), Shape::from(y.len())))
        }
    }

    pub struct Dw3 {
        pub stride: usize,
        pub gelu: bool,
    }

    impl CustomOp3 for Dw3 {
        fn name(&self) -> &'static str {
            "dwconv3x3"
        }

        fn cpu_fwd(&self, x: &CpuStorage, xl: &Layout, w: &CpuStorage, wl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (x, w, b) = (f32s(x, xl)?, f32s(w, wl)?, f32s(b, bl)?);
            let (_, c, h, wd) = xl.shape().dims4()?;
            let st = self.stride;
            let (oh, ow) = ((h + 2 - 3) / st + 1, (wd + 2 - 3) / st + 1);
            let mut out = vec![0f32; c * oh * ow];
            out.par_chunks_mut(ow).enumerate().for_each(|(row, o)| {
                let (ch, oy) = (row / oh, row % oh);
                let plane = &x[ch * h * wd..(ch + 1) * h * wd];
                let taps = &w[ch * 9..ch * 9 + 9];
                for (ox, v) in o.iter_mut().enumerate() {
                    let mut acc = b[ch];
                    for ky in 0..3 {
                        let iy = (oy * st + ky) as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = (ox * st + kx) as isize - 1;
                            if ix >= 0 && ix < wd as isize {
                                acc += plane[iy as usize * wd + ix as usize] * taps[ky * 3 + kx];
                            }
                        }
                    }
                    if self.gelu {
                        acc = 0.5 * acc * (1.0 + libm::erff(acc * std::f32::consts::FRAC_1_SQRT_2));
                    }
                    *v = acc;
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((1, c, oh, ow))))
        }
    }

    pub struct FlashAttn;

    impl CustomOp3 for FlashAttn {
        fn name(&self) -> &'static str {
            "flash-attn-cpu"
        }

        fn cpu_fwd(&self, q: &CpuStorage, ql: &Layout, k: &CpuStorage, kl: &Layout, v: &CpuStorage, vl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (q, k, v) = (f32s(q, ql)?, f32s(k, kl)?, f32s(v, vl)?);
            let (b, lq, h, d) = ql.shape().dims4()?;
            let (kb, lk, kh, kd) = kl.shape().dims4()?;
            if kl.shape() != vl.shape() || (kb, kh, kd) != (b, h, d) {
                bail!("flash-attn-cpu: incompatible q/k/v shapes {:?}/{:?}/{:?}", ql.shape(), kl.shape(), vl.shape());
            }
            // SAFETY: `flash_attn_cpu` only dispatches here when `cpu_gemm::enabled()` (AVX-512 BF16 + F).
            let out = unsafe { crate::cpu_gemm::flash_attention(q, k, v, b, lq, lk, h, d, (d as f32).powf(-0.5)) };
            Ok((CpuStorage::F32(out), ql.shape().clone()))
        }
    }

    pub struct Add;

    impl CustomOp2 for Add {
        fn name(&self) -> &'static str {
            "add-par"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (a, b) = (f32s(a, al)?, f32s(b, bl)?);
            let mut out = vec![0f32; a.len()];
            out.par_chunks_mut(1 << 14).enumerate().for_each(|(i, o)| {
                let base = i << 14;
                for (k, v) in o.iter_mut().enumerate() {
                    *v = a[base + k] + b[base + k];
                }
            });
            Ok((CpuStorage::F32(out), al.shape().clone()))
        }
    }

    pub struct LayerNormRows {
        pub eps: f32,
    }

    impl CustomOp3 for LayerNormRows {
        fn name(&self) -> &'static str {
            "layernorm-rows"
        }

        fn cpu_fwd(&self, x: &CpuStorage, xl: &Layout, w: &CpuStorage, wl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (x, w, b) = (f32s(x, xl)?, f32s(w, wl)?, f32s(b, bl)?);
            let c = w.len();
            let mut out = vec![0f32; x.len()];
            out.par_chunks_mut(c).zip(x.par_chunks(c)).for_each(|(o, x)| {
                let mean = x.iter().sum::<f32>() / c as f32;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
                let inv = 1.0 / (var + self.eps).sqrt();
                for k in 0..c {
                    o[k] = (x[k] - mean) * inv * w[k] + b[k];
                }
            });
            Ok((CpuStorage::F32(out), xl.shape().clone()))
        }
    }

    pub struct NchwAddToTokens;

    impl CustomOp2 for NchwAddToTokens {
        fn name(&self) -> &'static str {
            "nchw-add-to-tokens"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (a, b) = (f32s(a, al)?, f32s(b, bl)?);
            let (_, c, h, w) = al.shape().dims4()?;
            let hw = h * w;
            let mut out = vec![0f32; a.len()];
            out.par_chunks_mut(c).enumerate().for_each(|(p, o)| {
                for ch in 0..c {
                    o[ch] = a[ch * hw + p] + b[ch * hw + p];
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((1, hw, c))))
        }
    }

    pub struct Mha {
        pub heads: usize,
    }

    impl CustomOp3 for Mha {
        fn name(&self) -> &'static str {
            "mha"
        }

        fn cpu_fwd(&self, q: &CpuStorage, ql: &Layout, k: &CpuStorage, kl: &Layout, v: &CpuStorage, vl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (q, k, v) = (f32s(q, ql)?, f32s(k, kl)?, f32s(v, vl)?);
            let (nq, e) = ql.shape().dims2()?;
            let nk = kl.shape().dim(0)?;
            let hd = e / self.heads;
            let scale = (hd as f32).powf(-0.5);
            let mut out = vec![0f32; nq * e];
            // one task per (query, head)
            out.par_chunks_mut(hd).enumerate().for_each_init(
                || vec![0f32; nk],
                |sc, (task, o)| {
                    let (qi, head) = (task / self.heads, task % self.heads);
                    let qv = &q[qi * e + head * hd..][..hd];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..nk {
                        let kv = &k[j * e + head * hd..][..hd];
                        let d: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                        sc[j] = d;
                        max = max.max(d);
                    }
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - max).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    for j in 0..nk {
                        let p = sc[j] * inv;
                        let vv = &v[j * e + head * hd..][..hd];
                        for (a, b) in o.iter_mut().zip(vv) {
                            *a += p * b;
                        }
                    }
                },
            );
            Ok((CpuStorage::F32(out), ql.shape().clone()))
        }
    }

    pub struct PixelShuffle2 {
        pub h: usize,
        pub w: usize,
        pub gelu: bool,
        pub tokens: bool,
    }

    impl CustomOp2 for PixelShuffle2 {
        fn name(&self) -> &'static str {
            "pixel-shuffle2"
        }

        fn cpu_fwd(&self, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (y, b) = (f32s(y, yl)?, f32s(b, bl)?);
            let (h, w) = (self.h, self.w);
            let cout = b.len() / 4;
            let (oh, ow) = (2 * h, 2 * w);
            let mut out = vec![0f32; cout * oh * ow];
            out.par_chunks_mut(oh * ow).enumerate().for_each(|(co, plane)| {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let col = co * 4 + dy * 2 + dx;
                        let bias = b[col];
                        for yy in 0..h {
                            for xx in 0..w {
                                let p = yy * w + xx;
                                let mut v = if self.tokens { y[p * cout * 4 + col] } else { y[col * h * w + p] } + bias;
                                if self.gelu {
                                    v = 0.5 * v * (1.0 + libm::erff(v * std::f32::consts::FRAC_1_SQRT_2));
                                }
                                plane[(2 * yy + dy) * ow + 2 * xx + dx] = v;
                            }
                        }
                    }
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((1, cout, oh, ow))))
        }
    }

    /// Batched row-major matmul, parallel over the batch.
    pub struct Bmm;

    impl CustomOp2 for Bmm {
        fn name(&self) -> &'static str {
            "bmm"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (a, b) = (f32s(a, al)?, f32s(b, bl)?);
            let (batch, m, k) = al.shape().dims3()?;
            let n = bl.shape().dims3()?.2;
            let mut out = vec![0f32; batch * m * n];
            out.par_chunks_mut(m * n).enumerate().for_each(|(i, o)| {
                let (a, b) = (&a[i * m * k..], &b[i * k * n..]);
                // SAFETY: the three slices hold m*k, k*n and m*n elements of this batch entry.
                unsafe {
                    gemm::gemm(
                        m, n, k, o.as_mut_ptr(), 1, n as isize, false, a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
                        0.0, 1.0, false, false, false, gemm::Parallelism::None,
                    );
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((batch, m, n))))
        }
    }

    const CHUNK: usize = 1 << 14;

    /// `res + y + bias[col]`, parallel over rows of 16K elements.
    pub struct BiasResidual;

    impl CustomOp3 for BiasResidual {
        fn name(&self) -> &'static str {
            "bias-residual"
        }

        fn cpu_fwd(&self, r: &CpuStorage, rl: &Layout, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (r, y, b) = (f32s(r, rl)?, f32s(y, yl)?, f32s(b, bl)?);
            let c = b.len();
            let mut out = vec![0f32; r.len()];
            // CHUNK is a multiple of every channel count used by SAM (powers of two times small factors)
            let chunk = (CHUNK / c).max(1) * c;
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
                let base = i * chunk;
                for (k, v) in o.iter_mut().enumerate() {
                    *v = r[base + k] + y[base + k] + b[k % c];
                }
            });
            Ok((CpuStorage::F32(out), rl.shape().clone()))
        }
    }

    /// `gelu_erf(y + bias[col])`, parallel.
    pub struct BiasGelu;

    impl CustomOp2 for BiasGelu {
        fn name(&self) -> &'static str {
            "bias-gelu"
        }

        fn cpu_fwd(&self, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (y, b) = (f32s(y, yl)?, f32s(b, bl)?);
            let c = b.len();
            let mut out = vec![0f32; y.len()];
            let chunk = (CHUNK / c).max(1) * c;
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
                let base = i * chunk;
                for (k, v) in o.iter_mut().enumerate() {
                    let x = y[base + k] + b[k % c];
                    *v = 0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
                }
            });
            Ok((CpuStorage::F32(out), yl.shape().clone()))
        }
    }

    /// `y: [T, 3C]` + `bias[3C]` -> `[3, T, C]` (q, k, v planes), one task per token.
    pub struct BiasSplit3;

    impl CustomOp2 for BiasSplit3 {
        fn name(&self) -> &'static str {
            "bias-split3"
        }

        fn cpu_fwd(&self, y: &CpuStorage, yl: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (y, b) = (f32s(y, yl)?, f32s(b, bl)?);
            let c = b.len() / 3;
            let t = y.len() / (3 * c);
            let mut out = vec![0f32; y.len()];
            let base = SendPtr(out.as_mut_ptr());
            (0..t).into_par_iter().for_each(|i| {
                for which in 0..3 {
                    for k in 0..c {
                        // SAFETY: (which, i, k) uniquely determines the destination element.
                        unsafe { *base.at((which * t + i) * c + k) = y[i * 3 * c + which * c + k] + b[which * c + k] };
                    }
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((3, t, c))))
        }
    }

    /// NCHW `[1, C, H, W]` -> NHWC `[1, H, W, C]`, optionally adding an NHWC tensor of the same shape.
    pub struct NchwToNhwc;

    impl CustomOp2 for NchwToNhwc {
        fn name(&self) -> &'static str {
            "nchw-to-nhwc"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout, b: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
            let (a, b) = (f32s(a, al)?, f32s(b, bl)?);
            let (_, c, h, w) = al.shape().dims4()?;
            let hw = h * w;
            let mut out = vec![0f32; a.len()];
            out.par_chunks_mut(c).enumerate().for_each(|(p, o)| {
                for ch in 0..c {
                    o[ch] = a[ch * hw + p] + b[p * c + ch];
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((1, h, w, c))))
        }
    }

    /// NHWC `[1, H, W, C]` -> NCHW `[1, C, H, W]`.
    pub struct NhwcToNchw;

    impl CustomOp1 for NhwcToNchw {
        fn name(&self) -> &'static str {
            "nhwc-to-nchw"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout) -> Result<(CpuStorage, Shape)> {
            let a = f32s(a, al)?;
            let (_, h, w, c) = al.shape().dims4()?;
            let hw = h * w;
            let mut out = vec![0f32; a.len()];
            out.par_chunks_mut(hw).enumerate().for_each(|(ch, o)| {
                for (p, v) in o.iter_mut().enumerate() {
                    *v = a[p * c + ch];
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((1, c, h, w))))
        }
    }

    /// Max pool with kernel = stride = `s` over NHWC `[B, H, W, C]` (H, W divisible by `s`).
    pub struct MaxPoolNhwc {
        pub s: usize,
    }

    impl CustomOp1 for MaxPoolNhwc {
        fn name(&self) -> &'static str {
            "maxpool-nhwc"
        }

        fn cpu_fwd(&self, a: &CpuStorage, al: &Layout) -> Result<(CpuStorage, Shape)> {
            let a = f32s(a, al)?;
            let (b, h, w, c) = al.shape().dims4()?;
            let (s, oh, ow) = (self.s, h / self.s, w / self.s);
            let mut out = vec![f32::NEG_INFINITY; b * oh * ow * c];
            out.par_chunks_mut(ow * c).enumerate().for_each(|(row, o)| {
                let (bi, y) = (row / oh, row % oh);
                for x in 0..ow {
                    let dst = &mut o[x * c..(x + 1) * c];
                    for dy in 0..s {
                        for dx in 0..s {
                            let src = &a[((bi * h + y * s + dy) * w + x * s + dx) * c..][..c];
                            for (d, v) in dst.iter_mut().zip(src) {
                                *d = d.max(*v);
                            }
                        }
                    }
                }
            });
            Ok((CpuStorage::F32(out), Shape::from((b, oh, ow, c))))
        }
    }
}

/// `y: [T, 3C]` (qkv projection output without bias) + `bias[3C]` -> `[3, T, C]`: q, k, v planes with the bias added.
pub fn bias_split3(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if y.device().is_cpu() && y.dtype() == candle_core::DType::F32 {
        return y.contiguous()?.apply_op2_no_bwd(&bias.contiguous()?, &cpu::BiasSplit3);
    }
    let (t, c3) = y.dims2()?;
    y.broadcast_add(bias)?.reshape((t, 3, c3 / 3))?.permute((1, 0, 2))?.contiguous()
}

/// `[1, C, H, W]` -> `[1, H, W, C]` plus `add` (an NHWC tensor of that shape).
pub fn nchw_to_nhwc_add(x: &Tensor, add: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && add.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op2_no_bwd(&add.contiguous()?, &cpu::NchwToNhwc);
    }
    x.permute((0, 2, 3, 1))?.broadcast_add(&add.to_dtype(x.dtype())?)
}

/// `[1, H, W, C]` -> contiguous `[1, C, H, W]`.
pub fn nhwc_to_nchw(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.dim(0)? == 1 {
        return x.contiguous()?.apply_op1_no_bwd(&cpu::NhwcToNchw);
    }
    x.permute((0, 3, 1, 2))?.contiguous()
}

/// Max pool (kernel = stride = `s`) over an NHWC tensor with `H`, `W` divisible by `s`.
pub fn maxpool_nhwc(x: &Tensor, s: usize) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op1_no_bwd(&cpu::MaxPoolNhwc { s });
    }
    x.permute((0, 3, 1, 2))?.contiguous()?.max_pool2d(s)?.permute((0, 2, 3, 1))?.contiguous()
}

/// `y: [T, 3*nh*hd]` (bias-free qkv projection of `T = windows * n` tokens) -> `(q, kᵀ, v)` with shapes
/// `[BH, n, hd]`, `[BH, hd, n]`, `[BH, n, hd]` where `BH = windows * nh`; the bias is added on the way.
pub fn qkv_split(y: &Tensor, bias: &Tensor, n: usize, nh: usize, hd: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let t = y.dim(0)?;
    let bh = t / n * nh;
    #[cfg(feature = "cuda")]
    if y.device().is_cuda() {
        let packed = y.contiguous()?.apply_op2_no_bwd(bias, &cuda::QkvSplit { n, nh, hd })?;
        return Ok((
            packed.narrow(0, 0, 1)?.reshape((bh, n, hd))?,
            packed.narrow(0, 1, 1)?.reshape((bh, hd, n))?,
            packed.narrow(0, 2, 1)?.reshape((bh, n, hd))?,
        ));
    }
    if y.device().is_cpu() && y.dtype() == candle_core::DType::F32 {
        let packed = y.contiguous()?.apply_op2_no_bwd(&bias.contiguous()?, &cpu::QkvSplit { n, nh, hd })?;
        return Ok((
            packed.narrow(0, 0, 1)?.reshape((bh, n, hd))?,
            packed.narrow(0, 1, 1)?.reshape((bh, hd, n))?,
            packed.narrow(0, 2, 1)?.reshape((bh, n, hd))?,
        ));
    }
    let qkv = y.broadcast_add(bias)?.reshape((t / n, n, 3, nh, hd))?.permute((2, 0, 3, 1, 4))?.reshape((3, bh, n, hd))?;
    Ok((qkv.get(0)?.contiguous()?, qkv.get(1)?.transpose(1, 2)?.contiguous()?, qkv.get(2)?.contiguous()?))
}

/// Whole attention for one SAM block on CPU, tiled so the score matrix never leaves cache:
/// `softmax(scale * q·kᵀ + relpos(q)) · v`. `q`, `v`: `[BH, H*W, hd]`, `kt`: `[BH, hd, H*W]` (all f32,
/// contiguous). Returns `[BH, H*W, hd]`. `None` when the inputs are not CPU f32.
pub fn attention_relpos_cpu(q: &Tensor, kt: &Tensor, v: &Tensor, table: &Tensor, h: usize, w: usize, scale: f32) -> Result<Option<Tensor>> {
    if !(q.device().is_cpu() && q.dtype() == candle_core::DType::F32) {
        return Ok(None);
    }
    let table = table.flatten_all()?.to_vec1::<f32>()?;
    let op = cpu::TiledAttention { h, w, hd: q.dim(2)?, scale, table };
    Ok(Some(q.contiguous()?.apply_op3_no_bwd(&kt.contiguous()?, &v.contiguous()?, &op)?))
}

/// Mask-free multi-head attention on `[B, Lq, H, D]` queries and `[B, Lk, H, D]` keys/values (f32) using the bf16 CPU kernels. `None` when the
/// inputs are not CPU f32 or the CPU lacks AVX-512 BF16 (callers use the tensor path).
pub fn flash_attn_cpu(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Option<Tensor>> {
    if !(q.device().is_cpu() && q.dtype() == candle_core::DType::F32 && crate::cpu_gemm::enabled()) {
        return Ok(None);
    }
    Ok(Some(q.contiguous()?.apply_op3_no_bwd(&k.contiguous()?, &v.contiguous()?, &cpu::FlashAttn)?))
}

/// Batched `[B,m,k] x [B,k,n]`. On CPU with many small matrices (windowed attention) the batch is spread over
/// threads, one single-threaded GEMM per matrix, which is several times faster than candle's per-matrix threading.
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cpu() && a.dtype() == candle_core::DType::F32 && a.dim(0)? >= 32 {
        return a.contiguous()?.apply_op2_no_bwd(&b.contiguous()?, &cpu::Bmm);
    }
    a.matmul(b)
}

/// `a + b` for equal shapes, parallel on CPU f32 (candle's add is single-threaded).
pub fn add_par(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cpu() && a.dtype() == candle_core::DType::F32 && a.shape() == b.shape() {
        return a.contiguous()?.apply_op2_no_bwd(&b.contiguous()?, &cpu::Add);
    }
    a + b
}

/// LayerNorm over the last dim (`weight`, `bias`: `[C]`), parallel over rows on CPU f32.
pub fn layer_norm_rows(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op3_no_bwd(&weight.contiguous()?, &bias.contiguous()?, &cpu::LayerNormRows { eps });
    }
    use candle_core::Module as _;
    candle_nn::LayerNorm::new(weight.clone(), bias.clone(), eps as f64).forward(x)
}

/// `(emb + dense)` with both `[1, C, H, W]`, returned as tokens `[1, H*W, C]` (CPU f32 only).
pub fn nchw_add_to_tokens(emb: &Tensor, dense: &Tensor) -> Result<Tensor> {
    if emb.device().is_cpu() && emb.dtype() == candle_core::DType::F32 {
        return emb.contiguous()?.apply_op2_no_bwd(&dense.contiguous()?, &cpu::NchwAddToTokens);
    }
    (emb + dense)?.flatten_from(2)?.transpose(1, 2)?.contiguous()
}

/// Depthwise 3x3 convolution with padding 1 and stride 1 or 2, plus bias and an optional exact GELU.
/// `x`: `[1, C, H, W]`, `w`: `[C, 9]` (taps row-major), `b`: `[C]`. candle's grouped convolution launches kernels per
/// channel, which is far too slow for depthwise layers.
pub fn dwconv3x3(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize, gelu: bool) -> Result<Tensor> {
    let (n, c, h, wd) = x.dims4()?;
    if n != 1 {
        candle_core::bail!("dwconv3x3 supports batch 1");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return x.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &b.contiguous()?, &cuda::Dw3 { stride, gelu });
    }
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &b.contiguous()?, &cpu::Dw3 { stride, gelu });
    }
    let cfg = candle_nn::Conv2dConfig { padding: 1, stride, groups: c, ..Default::default() };
    let y = candle_nn::Conv2d::new(w.reshape((c, 1, 3, 3))?, Some(b.clone()), cfg).forward(x)?;
    let _ = (h, wd);
    if gelu { y.gelu_erf() } else { Ok(y) }
}

/// Multi-head attention over already-projected `q: [nq, E]`, `k, v: [nk, E]` (heads are contiguous
/// `E/heads` slices) -> `[nq, E]`. CPU f32 only; returns `None` otherwise so callers use the tensor path.
pub fn mha_cpu(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Result<Option<Tensor>> {
    if !(q.device().is_cpu() && q.dtype() == candle_core::DType::F32) {
        return Ok(None);
    }
    Ok(Some(q.contiguous()?.apply_op3_no_bwd(&k.contiguous()?, &v.contiguous()?, &cpu::Mha { heads })?))
}

/// `y: [Cout*4, H*W]` (rows ordered `(co, dy, dx)`; `[H*W, Cout*4]` when `tokens`) + `bias[Cout*4]` -> `[1, Cout, 2H, 2W]`, the output of a
/// kernel-2 stride-2 transposed convolution, optionally followed by GELU. CPU f32 only.
pub fn pixel_shuffle2_cpu(y: &Tensor, bias: &Tensor, h: usize, w: usize, gelu: bool, tokens: bool) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    if y.device().is_cuda() {
        let cout = bias.dim(0)? / 4;
        return Ok(Some(y.contiguous()?.apply_op2_no_bwd(&bias.contiguous()?, &cuda::PixelShuffle2 { cout, h, w, gelu, tokens })?));
    }
    if !(y.device().is_cpu() && y.dtype() == candle_core::DType::F32) {
        return Ok(None);
    }
    Ok(Some(y.contiguous()?.apply_op2_no_bwd(&bias.contiguous()?, &cpu::PixelShuffle2 { h, w, gelu, tokens })?))
}

/// LayerNorm over dim 1 of an NCHW tensor (SAM's `LayerNorm2d`).
pub fn layer_norm_2d(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op3_no_bwd(&weight.contiguous()?, &bias.contiguous()?, &cpu::LayerNorm2d { eps });
    }
    let dt = x.dtype();
    let x = x.to_dtype(candle_core::DType::F32)?;
    let mean = x.mean_keepdim(1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(1)?;
    let y = xc.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    let c = weight.dim(0)?;
    y.to_dtype(dt)?.broadcast_mul(&weight.reshape((1, c, 1, 1))?)?.broadcast_add(&bias.reshape((1, c, 1, 1))?)
}

/// SAM 3 ViT: `y: [T, 3*H*HD]` (fused QKV projection without bias) -> packed `[3, T, H, HD]` (q, k, v) with the
/// bias added and the 2-D RoPE applied to q and k. `cos`/`sin`: f32 `[N, HD]`, `N` = tokens per window
/// (`T` is a multiple of `N`; token `t` uses row `t % N`). The packed layout is what FlashAttention reads.
pub fn qkv_rope(y: &Tensor, bias: &Tensor, cos: &Tensor, sin: &Tensor, n: usize, heads: usize, hd: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if y.device().is_cuda() {
        let packed = y.contiguous()?.apply_op2_no_bwd(bias, &cuda::QkvRope { n, heads, hd, cos: cos.clone(), sin: sin.clone() })?;
        return packed.reshape((3, y.dim(0)?, heads, hd));
    }
    if y.device().is_cpu() && y.dtype() == candle_core::DType::F32 {
        let packed = y.contiguous()?.apply_op3_no_bwd(&bias.contiguous()?, &cos.contiguous()?, &cpu::QkvRope { n, heads, hd, sin: sin.contiguous()? })?;
        return packed.reshape((3, y.dim(0)?, heads, hd));
    }
    let t = y.dim(0)?;
    let dt = y.dtype();
    let y = y.broadcast_add(bias)?.reshape((t / n, n, 3, heads, hd))?.to_dtype(candle_core::DType::F32)?;
    let rope = |x: Tensor| -> Result<Tensor> {
        let (nw, n_, h_, d_) = x.dims4()?;
        let x2 = x.reshape((nw, n_, h_, d_ / 2, 2))?;
        let rot = Tensor::cat(&[&x2.narrow(4, 1, 1)?.neg()?, &x2.narrow(4, 0, 1)?], 4)?.reshape((nw, n_, h_, d_))?;
        let (c, s) = (cos.reshape((1, n_, 1, d_))?, sin.reshape((1, n_, 1, d_))?);
        x.broadcast_mul(&c)? + rot.broadcast_mul(&s)?
    };
    let q = rope(y.narrow(2, 0, 1)?.squeeze(2)?)?;
    let k = rope(y.narrow(2, 1, 1)?.squeeze(2)?)?;
    let v = y.narrow(2, 2, 1)?.squeeze(2)?;
    Tensor::stack(&[q, k, v], 0)?.reshape((3, t, heads, hd))?.to_dtype(dt)
}

/// `res + y + bias` (bias broadcast over the last dim).
pub fn bias_residual(res: &Tensor, y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if res.device().is_cuda() {
        return res.contiguous()?.apply_op3_no_bwd(&y.contiguous()?, bias, &cuda::BiasResidual);
    }
    if res.device().is_cpu() && res.dtype() == candle_core::DType::F32 {
        return res.contiguous()?.apply_op3_no_bwd(&y.contiguous()?, &bias.contiguous()?, &cpu::BiasResidual);
    }
    res + y.broadcast_add(bias)?
}

/// `gelu(y + bias)` (exact erf GELU, bias broadcast over the last dim).
pub fn bias_gelu(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if y.device().is_cuda() {
        return y.contiguous()?.apply_op2_no_bwd(bias, &cuda::BiasGelu);
    }
    if y.device().is_cpu() && y.dtype() == candle_core::DType::F32 {
        return y.contiguous()?.apply_op2_no_bwd(&bias.contiguous()?, &cpu::BiasGelu);
    }
    y.broadcast_add(bias)?.gelu_erf()
}

/// Resize-normalize-pad of an RGB image already on a CUDA device; see `kernels/sam.cu`.
/// `src`: u8 `[h, w, 3]`. Returns `[1, 3, size, size]` in `dtype`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn preprocess(src: &Tensor, new_hw: (usize, usize), size: usize, mean: [f32; 3], std: [f32; 3], dtype: candle_core::DType) -> Result<Tensor> {
    let out = src.apply_op1_no_bwd(&cuda::Preprocess { new_hw, size, mean, inv_std: [1. / std[0], 1. / std[1], 1. / std[2]], dtype })?;
    out.reshape((1, 3, size, size))
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{kernel_name, CudaDType, CudaStorage, WrapErr};
    use candle_core::{bail, CpuStorage, CudaDevice, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, WithDType};

    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sam.ptx"));
    const THREADS: usize = 256;
    const MAXI: usize = 32;
    /// Longest softmax row the kernel keeps in registers (`MAXI` values per thread).
    pub const MAX_ROW: usize = THREADS * MAXI;

    /// Runs `$f::<T>` for the storage's dtype.
    macro_rules! by_dtype {
        ($dt:expr, $f:ident($($a:expr),*)) => {
            match $dt {
                DType::F16 => $f::<half::f16>($($a),*),
                DType::BF16 => $f::<half::bf16>($($a),*),
                DType::F32 => $f::<f32>($($a),*),
                d => bail!("unsupported dtype {d:?} for a sam-rs CUDA kernel"),
            }
        };
    }

    fn view<'a, T: CudaDType>(s: &'a CudaStorage, l: &Layout) -> Result<CudaView<'a, T>> {
        match l.contiguous_offsets() {
            Some((a, b)) => Ok(s.as_cuda_slice::<T>()?.slice(a..b)),
            None => bail!("sam-rs kernels need contiguous inputs"),
        }
    }

    fn grid(total: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (total.div_ceil(256).min(1 << 16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    pub struct RelposSoftmax {
        pub h: usize,
        pub w: usize,
        pub hd: usize,
        pub scale: f32,
    }

    impl RelposSoftmax {
        fn run<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(&self, dev: &CudaDevice, s: &CudaStorage, sl: &Layout, q: &CudaStorage, ql: &Layout, t: &CudaStorage, tl: &Layout) -> Result<CudaStorage> {
            let (s, q, t) = (view::<T>(s, sl)?, view::<T>(q, ql)?, view::<T>(t, tl)?);
            let n = self.h * self.w;
            let rows = sl.shape().elem_count() / n;
            // Fewest threads that keep every row slice in registers; ITEMS = ceil(n / threads), a power of two.
            let threads = n.next_multiple_of(32).min(THREADS);
            let items = n.div_ceil(threads).next_power_of_two();
            let func = dev.get_or_load_custom_func(&kernel_name::<T>(&format!("relpos_softmax_i{items}")), "sam", PTX)?;
            let out = unsafe { dev.alloc::<T>(sl.shape().elem_count())? };
            let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: ((self.hd + self.h + self.w) * 4) as u32 };
            let (h, w, hd) = (self.h as i32, self.w as i32, self.hd as i32);
            let mut b = func.builder();
            b.arg(&s).arg(&q).arg(&t).arg(&out).arg(&h).arg(&w).arg(&hd).arg(&self.scale);
            unsafe { b.launch(cfg) }.w()?;
            Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
        }
    }

    impl CustomOp3 for RelposSoftmax {
        fn name(&self) -> &'static str {
            "relpos-softmax"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("relpos-softmax is CUDA only")
        }
        fn cuda_fwd(&self, s: &CudaStorage, sl: &Layout, q: &CudaStorage, ql: &Layout, t: &CudaStorage, tl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = s.device().clone();
            let out = by_dtype!(s.dtype(), run_relpos(self, &dev, s, sl, q, ql, t, tl))?;
            Ok((out, sl.shape().clone()))
        }
    }
    fn run_relpos<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &RelposSoftmax, dev: &CudaDevice, s: &CudaStorage, sl: &Layout, q: &CudaStorage, ql: &Layout, t: &CudaStorage, tl: &Layout) -> Result<CudaStorage> {
        op.run::<T>(dev, s, sl, q, ql, t, tl)
    }

    pub struct QkvSplit {
        pub n: usize,
        pub nh: usize,
        pub hd: usize,
    }

    fn run_qkv<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &QkvSplit, dev: &CudaDevice, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (yv, bv) = (view::<T>(y, yl)?, view::<T>(b, bl)?);
        let total = yl.shape().elem_count();
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("qkv_split"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (tot, n, nh, hd) = (total as i64, op.n as i32, op.nh as i32, op.hd as i32);
        let mut bld = func.builder();
        bld.arg(&yv).arg(&bv).arg(&out).arg(&tot).arg(&n).arg(&nh).arg(&hd);
        unsafe { bld.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp2 for QkvSplit {
        fn name(&self) -> &'static str {
            "qkv-split"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("qkv-split is CUDA only")
        }
        fn cuda_fwd(&self, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = y.device().clone();
            let total = yl.shape().elem_count();
            let out = by_dtype!(y.dtype(), run_qkv(self, &dev, y, yl, b, bl))?;
            Ok((out, Shape::from((3, total / 3))))
        }
    }

    pub struct BiasResidual;

    fn run_res<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(dev: &CudaDevice, r: &CudaStorage, rl: &Layout, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (rv, yv, bv) = (view::<T>(r, rl)?, view::<T>(y, yl)?, view::<T>(b, bl)?);
        let total = rl.shape().elem_count();
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("bias_residual"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (tot, c) = (total as i64, bl.shape().elem_count() as i32);
        let mut bld = func.builder();
        bld.arg(&rv).arg(&yv).arg(&bv).arg(&out).arg(&tot).arg(&c);
        unsafe { bld.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp3 for BiasResidual {
        fn name(&self) -> &'static str {
            "bias-residual"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("bias-residual is CUDA only")
        }
        fn cuda_fwd(&self, r: &CudaStorage, rl: &Layout, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            if rl.shape() != yl.shape() {
                bail!("bias-residual: shape mismatch {:?} vs {:?}", rl.shape(), yl.shape());
            }
            let dev = r.device().clone();
            let out = by_dtype!(r.dtype(), run_res(&dev, r, rl, y, yl, b, bl))?;
            Ok((out, rl.shape().clone()))
        }
    }

    pub struct BiasGelu;

    fn run_gelu<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(dev: &CudaDevice, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (yv, bv) = (view::<T>(y, yl)?, view::<T>(b, bl)?);
        let total = yl.shape().elem_count();
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("bias_gelu"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (tot, c) = (total as i64, bl.shape().elem_count() as i32);
        let mut bld = func.builder();
        bld.arg(&yv).arg(&bv).arg(&out).arg(&tot).arg(&c);
        unsafe { bld.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp2 for BiasGelu {
        fn name(&self) -> &'static str {
            "bias-gelu"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("bias-gelu is CUDA only")
        }
        fn cuda_fwd(&self, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = y.device().clone();
            let out = by_dtype!(y.dtype(), run_gelu(&dev, y, yl, b, bl))?;
            Ok((out, yl.shape().clone()))
        }
    }

    pub struct Dw3 {
        pub stride: usize,
        pub gelu: bool,
    }

    fn run_dw3<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &Dw3, dev: &CudaDevice, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (xv, wv, bv) = (view::<T>(x, xl)?, view::<T>(w, wl)?, view::<T>(b, bl)?);
        let (_, c, h, wd) = xl.shape().dims4()?;
        let (oh, ow) = ((h + 2 - 3) / op.stride + 1, (wd + 2 - 3) / op.stride + 1);
        let total = c * oh * ow;
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("dwconv3"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (ci, hi, wi, ohi, owi, st, g) = (c as i32, h as i32, wd as i32, oh as i32, ow as i32, op.stride as i32, op.gelu as i32);
        let mut bld = func.builder();
        bld.arg(&xv).arg(&wv).arg(&bv).arg(&out).arg(&ci).arg(&hi).arg(&wi).arg(&ohi).arg(&owi).arg(&st).arg(&g);
        unsafe { bld.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp3 for Dw3 {
        fn name(&self) -> &'static str {
            "dwconv3x3"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("dwconv3x3 CUDA op on CPU")
        }
        fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = x.device().clone();
            let (_, c, h, wd) = xl.shape().dims4()?;
            let out = by_dtype!(x.dtype(), run_dw3(self, &dev, x, xl, w, wl, b, bl))?;
            Ok((out, Shape::from((1, c, (h + 2 - 3) / self.stride + 1, (wd + 2 - 3) / self.stride + 1))))
        }
    }

    pub struct QkvRope {
        pub n: usize,
        pub heads: usize,
        pub hd: usize,
        pub cos: candle_core::Tensor,
        pub sin: candle_core::Tensor,
    }

    fn run_qkv_rope<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &QkvRope, dev: &CudaDevice, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (yv, bv) = (view::<T>(y, yl)?, view::<T>(b, bl)?);
        let (cs, cl) = op.cos.storage_and_layout();
        let (sn, sl) = op.sin.storage_and_layout();
        let (candle_core::Storage::Cuda(cs), candle_core::Storage::Cuda(sn)) = (&*cs, &*sn) else { bail!("rope tables must be on the GPU") };
        let (cv, sv) = (view::<f32>(cs, cl)?, view::<f32>(sn, sl)?);
        let total = yl.shape().elem_count();
        let rows = (total / (3 * op.heads * op.hd)) as i64;
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("qkv_rope"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (n, h, hd) = (op.n as i32, op.heads as i32, op.hd as i32);
        let mut bld = func.builder();
        bld.arg(&yv).arg(&bv).arg(&cv).arg(&sv).arg(&out).arg(&rows).arg(&n).arg(&h).arg(&hd);
        unsafe { bld.launch(grid(total / 2)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp2 for QkvRope {
        fn name(&self) -> &'static str {
            "qkv-rope"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("qkv-rope is CUDA only")
        }
        fn cuda_fwd(&self, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = y.device().clone();
            let total = yl.shape().elem_count();
            let out = by_dtype!(y.dtype(), run_qkv_rope(self, &dev, y, yl, b, bl))?;
            Ok((out, Shape::from(total)))
        }
    }

    pub struct PixelShuffle2 {
        pub cout: usize,
        pub h: usize,
        pub w: usize,
        pub gelu: bool,
        pub tokens: bool,
    }

    fn run_shuffle<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &PixelShuffle2, dev: &CudaDevice, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<CudaStorage> {
        let (yv, bv) = (view::<T>(y, yl)?, view::<T>(b, bl)?);
        let total = op.cout * op.h * op.w * 4;
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("pixel_shuffle2"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (c, h, w, g, t) = (op.cout as i32, op.h as i32, op.w as i32, op.gelu as i32, op.tokens as i32);
        let mut bld = func.builder();
        bld.arg(&yv).arg(&bv).arg(&out).arg(&c).arg(&h).arg(&w).arg(&g).arg(&t);
        unsafe { bld.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp2 for PixelShuffle2 {
        fn name(&self) -> &'static str {
            "pixel-shuffle2"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("pixel-shuffle2 CUDA op on CPU")
        }
        fn cuda_fwd(&self, y: &CudaStorage, yl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = y.device().clone();
            let out = by_dtype!(y.dtype(), run_shuffle(self, &dev, y, yl, b, bl))?;
            Ok((out, Shape::from((1, self.cout, 2 * self.h, 2 * self.w))))
        }
    }

    pub struct Preprocess {
        pub new_hw: (usize, usize),
        pub size: usize,
        pub mean: [f32; 3],
        pub inv_std: [f32; 3],
        pub dtype: DType,
    }

    fn run_pre<T: CudaDType + WithDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(op: &Preprocess, dev: &CudaDevice, src: &CudaStorage, l: &Layout) -> Result<CudaStorage> {
        let sv = view::<u8>(src, l)?;
        let (h, w, _) = l.shape().dims3()?;
        let total = 3 * op.size * op.size;
        let func = dev.get_or_load_custom_func(&kernel_name::<T>("preprocess"), "sam", PTX)?;
        let out = unsafe { dev.alloc::<T>(total)? };
        let (h, w, nh, nw, s) = (h as i32, w as i32, op.new_hw.0 as i32, op.new_hw.1 as i32, op.size as i32);
        let mut b = func.builder();
        b.arg(&sv).arg(&out).arg(&h).arg(&w).arg(&nh).arg(&nw).arg(&s);
        b.arg(&op.mean[0]).arg(&op.mean[1]).arg(&op.mean[2]).arg(&op.inv_std[0]).arg(&op.inv_std[1]).arg(&op.inv_std[2]);
        unsafe { b.launch(grid(total)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    impl CustomOp1 for Preprocess {
        fn name(&self) -> &'static str {
            "preprocess"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            bail!("preprocess is CUDA only")
        }
        fn cuda_fwd(&self, src: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = src.device().clone();
            let out = by_dtype!(self.dtype, run_pre(self, &dev, src, l))?;
            Ok((out, Shape::from(3 * self.size * self.size)))
        }
    }
}
