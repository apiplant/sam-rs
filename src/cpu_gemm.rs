//! bf16 GEMM for CPUs with AVX-512 BF16 (`vdpbf16ps`): bf16 inputs, f32 accumulation.
//!
//! Weights are converted once into panels of 32 output columns laid out as `[k-pair][column][2]`, so the
//! microkernel streams them with plain loads and broadcasts one `(a[2p], a[2p+1])` pair per row.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{bail, CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Whether this CPU can run the bf16 kernels.
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Turns the bf16 path on or off for models loaded afterwards (on by default where available).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    available() && ENABLED.load(Ordering::Relaxed)
}

const NR: usize = 32;
const MR: usize = 8;
const MC: usize = 64;

fn bf16_bits(x: f32) -> u16 {
    // round to nearest even
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A `[K, N]` weight matrix, pre-converted to bf16 panels.
pub struct PackedB {
    k: usize,
    n: usize,
    kp: usize,
    /// `panels * kp * NR` u32, each a (k even, k odd) bf16 pair
    data: Vec<u32>,
}

impl PackedB {
    /// `w`: `[K, N]` f32.
    pub fn new(w: &Tensor) -> Result<Self> {
        let (k, n) = w.dims2()?;
        Ok(Self::from_slice(&w.contiguous()?.flatten_all()?.to_vec1::<f32>()?, k, n))
    }

    /// From a row-major `[K, N]` f32 slice.
    pub fn from_slice(w: &[f32], k: usize, n: usize) -> Self {
        let kp = k.div_ceil(2);
        let panels = n.div_ceil(NR);
        let mut data = vec![0u32; panels * kp * NR];
        for p in 0..panels {
            for q in 0..kp {
                for c in 0..NR {
                    let col = p * NR + c;
                    if col >= n {
                        continue;
                    }
                    let lo = bf16_bits(w[2 * q * n + col]) as u32;
                    let hi = if 2 * q + 1 < k { bf16_bits(w[(2 * q + 1) * n + col]) as u32 } else { 0 };
                    data[(p * kp + q) * NR + c] = lo | (hi << 16);
                }
            }
        }
        Self { k, n, kp, data }
    }

    /// `c[rows, N] = a[rows, K] @ self`, single-threaded (callers parallelize over blocks).
    ///
    /// # Safety
    /// The CPU must support AVX-512 BF16 (`available()`).
    pub unsafe fn matmul_rows(&self, a: &[f32], rows: usize, c: &mut [f32]) {
        block(self, a, rows, c)
    }

    /// `x`: `[M, K]` f32 -> `[M, N]` f32.
    pub fn matmul(&self, x: &Tensor) -> Result<Tensor> {
        x.contiguous()?.apply_op1_no_bwd(&Bf16Matmul(self))
    }
}

struct Bf16Matmul<'a>(&'a PackedB);

impl CustomOp1 for Bf16Matmul<'_> {
    fn name(&self) -> &'static str {
        "bf16-matmul"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let b = self.0;
        let CpuStorage::F32(x) = s else { bail!("bf16-matmul expects f32 activations") };
        let (m, k) = l.shape().dims2()?;
        if k != b.k {
            bail!("bf16-matmul: K mismatch {k} vs {}", b.k);
        }
        let x = match l.contiguous_offsets() {
            Some((a, e)) => &x[a..e],
            None => bail!("bf16-matmul needs contiguous input"),
        };
        let mut out = vec![0f32; m * b.n];
        let n = b.n;
        out.par_chunks_mut(MC * n).enumerate().for_each(|(blk, c)| {
            let r0 = blk * MC;
            let rows = MC.min(m - r0);
            // SAFETY: only reached when `available()` is true (the caller checks before building a PackedB path).
            unsafe { block(b, &x[r0 * k..(r0 + rows) * k], rows, c) };
        });
        Ok((CpuStorage::F32(out), Shape::from((m, n))))
    }
}

/// `c[rows, n] = a[rows, k] @ B` for one block of rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn block(b: &PackedB, a: &[f32], rows: usize, c: &mut [f32]) {
    use std::arch::x86_64::*;
    let (k, n, kp) = (b.k, b.n, b.kp);
    let kpad = kp * 2;
    let padded_rows = rows.div_ceil(MR) * MR;
    // A block in bf16, zero padded in K and in the row tail
    let mut ab = vec![0u16; padded_rows * kpad];
    for r in 0..rows {
        let src = &a[r * k..(r + 1) * k];
        let dst = &mut ab[r * kpad..r * kpad + k];
        let mut i = 0;
        while i + 16 <= k {
            let v = _mm512_loadu_ps(src.as_ptr().add(i));
            let h: __m256bh = _mm512_cvtneps_pbh(v);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, std::mem::transmute::<__m256bh, __m256i>(h));
            i += 16;
        }
        for j in i..k {
            dst[j] = bf16_bits(src[j]);
        }
    }
    let panels = n.div_ceil(NR);
    let mut tmp = [0f32; MR * NR];
    for p in 0..panels {
        let bp = b.data.as_ptr().add(p * kp * NR);
        let cols = NR.min(n - p * NR);
        for r0 in (0..padded_rows).step_by(MR) {
            let ap = ab.as_ptr().add(r0 * kpad);
            let mut acc = [[_mm512_setzero_ps(); 2]; MR];
            for q in 0..kp {
                let b0 = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(bp.add(q * NR) as *const _));
                let b1 = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(bp.add(q * NR + 16) as *const _));
                for r in 0..MR {
                    let pair = (ap.add(r * kpad) as *const u32).add(q).read_unaligned();
                    let av = std::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(pair as i32));
                    acc[r][0] = _mm512_dpbf16_ps(acc[r][0], av, b0);
                    acc[r][1] = _mm512_dpbf16_ps(acc[r][1], av, b1);
                }
            }
            let live = MR.min(rows.saturating_sub(r0));
            for r in 0..live {
                _mm512_storeu_ps(tmp.as_mut_ptr().add(r * NR), acc[r][0]);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(r * NR + 16), acc[r][1]);
                std::ptr::copy_nonoverlapping(tmp.as_ptr().add(r * NR), c.as_mut_ptr().add((r0 + r) * n + p * NR), cols);
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn block(_: &PackedB, _: &[f32], _: usize, _: &mut [f32]) {
    unreachable!("bf16 kernels need x86_64")
}

/// A bias-free linear layer (`x @ w`, `w: [in, out]`) that uses the bf16 kernel when the weights live on a
/// capable CPU and the path is enabled, and an ordinary matmul otherwise.
pub struct Linear {
    w: Tensor,
    packed: Option<PackedB>,
}

impl Linear {
    pub fn new(w: Tensor) -> Result<Self> {
        let packed = if w.device().is_cpu() && w.dtype() == candle_core::DType::F32 && enabled() { Some(PackedB::new(&w)?) } else { None };
        Ok(Self { w, packed })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.packed {
            Some(p) if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.rank() == 2 => p.matmul(x),
            _ => x.matmul(&self.w),
        }
    }
}

/// Softmax over one attention row with the decomposed relative-position bias:
/// `s[kh*w + kw] <- softmax(s * scale + rh[kh] + rw[kw])`, in place. AVX-512 with a polynomial `exp`.
///
/// # Safety
/// The CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_row(s: &mut [f32], rh: &[f32], rw: &[f32], h: usize, w: usize, scale: f32) {
    use std::arch::x86_64::*;
    let n = h * w;
    let vscale = _mm512_set1_ps(scale);
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    for kh in 0..h {
        let add = _mm512_set1_ps(rh[kh]);
        let mut kw = 0;
        while kw < w {
            let m: __mmask16 = if w - kw >= 16 { 0xffff } else { ((1u32 << (w - kw)) - 1) as u16 };
            let p = s.as_mut_ptr().add(kh * w + kw);
            let x = _mm512_maskz_loadu_ps(m, p);
            let b = _mm512_maskz_loadu_ps(m, rw.as_ptr().add(kw));
            let v = _mm512_add_ps(_mm512_fmadd_ps(x, vscale, add), b);
            _mm512_mask_storeu_ps(p, m, v);
            vmax = _mm512_mask_max_ps(vmax, m, vmax, v);
            kw += 16;
        }
    }
    let max = _mm512_reduce_max_ps(vmax);
    let vm = _mm512_set1_ps(max);
    let mut vsum = _mm512_setzero_ps();
    let mut i = 0;
    while i < n {
        let m: __mmask16 = if n - i >= 16 { 0xffff } else { ((1u32 << (n - i)) - 1) as u16 };
        let p = s.as_mut_ptr().add(i);
        let e = exp_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, p), vm));
        _mm512_mask_storeu_ps(p, m, e);
        vsum = _mm512_mask_add_ps(vsum, m, vsum, e);
        i += 16;
    }
    let inv = _mm512_set1_ps(1.0 / _mm512_reduce_add_ps(vsum));
    i = 0;
    while i < n {
        let m: __mmask16 = if n - i >= 16 { 0xffff } else { ((1u32 << (n - i)) - 1) as u16 };
        let p = s.as_mut_ptr().add(i);
        _mm512_mask_storeu_ps(p, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, p), inv));
        i += 16;
    }
}

/// `exp(x)` for `x <= 0` (relative error ~1e-7): `2^(x log2 e)` with a degree-6 polynomial on the fraction.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn exp_ps(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let t = _mm512_max_ps(_mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)), _mm512_set1_ps(-126.0));
    let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(t);
    let r = _mm512_sub_ps(t, n);
    let mut p = _mm512_set1_ps(0.000154035303);
    for c in [0.00133335581, 0.00961812911, 0.0555041087, 0.240226507, 0.693147181, 1.0] {
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
    }
    _mm512_scalef_ps(p, n)
}

/// `rh[kh] = q . table_h[i + h-1 - kh]`, `rw[kw] = q . table_w[j + w-1 - kw]`.
///
/// # Safety
/// The CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rel_bias(q: &[f32], th: &[f32], tw: &[f32], h: usize, w: usize, hd: usize, i: usize, j: usize, rh: &mut [f32], rw: &mut [f32]) {
    for kh in 0..h {
        let t = &th[(i + h - 1 - kh) * hd..][..hd];
        rh[kh] = t.iter().zip(q).map(|(a, b)| a * b).sum();
    }
    for kw in 0..w {
        let t = &tw[(j + w - 1 - kw) * hd..][..hd];
        rw[kw] = t.iter().zip(q).map(|(a, b)| a * b).sum();
    }
}

/// candle sizes its GEMM thread pool from the *physical* core count unless `RAYON_NUM_THREADS` is set, which
/// leaves SMT threads idle and makes the decoder's many small matmuls about 2x slower. This sets the variable to
/// the number of logical cores when the user has not. Call it once at the start of `main`, before any other
/// thread exists (the binary does); the library never calls it on its own because it mutates the environment.
pub fn init_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        if let Ok(n) = std::thread::available_parallelism() {
            std::env::set_var("RAYON_NUM_THREADS", n.get().to_string());
        }
    }
}

/// In-place `softmax(s * scale)` over one row (AVX-512, polynomial `exp`).
///
/// # Safety
/// The CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_scaled(s: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = s.len();
    let vscale = _mm512_set1_ps(scale);
    let mask = |i: usize| -> __mmask16 { if n - i >= 16 { 0xffff } else { ((1u32 << (n - i)) - 1) as u16 } };
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < n {
        let m = mask(i);
        let v = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, s.as_ptr().add(i)), vscale);
        _mm512_mask_storeu_ps(s.as_mut_ptr().add(i), m, v);
        vmax = _mm512_mask_max_ps(vmax, m, vmax, v);
        i += 16;
    }
    let vm = _mm512_set1_ps(_mm512_reduce_max_ps(vmax));
    let mut vsum = _mm512_setzero_ps();
    i = 0;
    while i < n {
        let m = mask(i);
        let e = exp_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, s.as_ptr().add(i)), vm));
        _mm512_mask_storeu_ps(s.as_mut_ptr().add(i), m, e);
        vsum = _mm512_mask_add_ps(vsum, m, vsum, e);
        i += 16;
    }
    let inv = _mm512_set1_ps(1.0 / _mm512_reduce_add_ps(vsum));
    i = 0;
    while i < n {
        let m = mask(i);
        _mm512_mask_storeu_ps(s.as_mut_ptr().add(i), m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, s.as_ptr().add(i)), inv));
        i += 16;
    }
}

/// Multi-head attention without bias or mask on `[B, L, H, D]` f32 buffers, tiled so the `L x L` score matrix
/// never leaves cache: per tile, `q_tile @ K^T` and `P @ V` run on the bf16 kernel with an AVX-512 softmax.
///
/// # Safety
/// The CPU must support AVX-512 BF16 (`available()`).
pub unsafe fn flash_attention(q: &[f32], k: &[f32], v: &[f32], b: usize, lq: usize, lk: usize, h: usize, d: usize, scale: f32) -> Vec<f32> {
    let l = lk;
    const TILE: usize = 32;
    let row = h * d;
    // K^T and V packed once per (batch, head)
    let packs: Vec<(PackedB, PackedB)> = (0..b * h)
        .into_par_iter()
        .map(|bh| {
            let (bi, hi) = (bh / h, bh % h);
            let mut kt = vec![0f32; d * l];
            let mut vm = vec![0f32; l * d];
            for j in 0..l {
                let src = (bi * lk + j) * row + hi * d;
                for c in 0..d {
                    kt[c * l + j] = k[src + c];
                }
                vm[j * d..(j + 1) * d].copy_from_slice(&v[src..src + d]);
            }
            (PackedB::from_slice(&kt, d, l), PackedB::from_slice(&vm, l, d))
        })
        .collect();
    let tiles = lq.div_ceil(TILE);
    let mut out = vec![0f32; b * lq * row];
    struct Out(*mut f32);
    // SAFETY: tasks write disjoint (row, head) blocks of `out`, which outlives the loop.
    unsafe impl Sync for Out {}
    impl Out {
        fn at(&self, off: usize) -> *mut f32 {
            // SAFETY: callers stay inside the allocation.
            unsafe { self.0.add(off) }
        }
    }
    let base = Out(out.as_mut_ptr());
    (0..b * h * tiles).into_par_iter().for_each_init(
        || (vec![0f32; TILE * l], vec![0f32; TILE * d], vec![0f32; TILE * d]),
        |(sc, qt, ot), task| {
            let (bh, t) = (task / tiles, task % tiles);
            let (bi, hi) = (bh / h, bh % h);
            let (r0, rows) = (t * TILE, TILE.min(lq - t * TILE));
            for r in 0..rows {
                let src = (bi * lq + r0 + r) * row + hi * d;
                qt[r * d..(r + 1) * d].copy_from_slice(&q[src..src + d]);
            }
            let (kp, vp) = &packs[bh];
            kp.matmul_rows(&qt[..rows * d], rows, &mut sc[..rows * l]);
            for r in 0..rows {
                softmax_scaled(&mut sc[r * l..(r + 1) * l], scale);
            }
            vp.matmul_rows(&sc[..rows * l], rows, &mut ot[..rows * d]);
            for r in 0..rows {
                let dst = (bi * lq + r0 + r) * row + hi * d;
                std::ptr::copy_nonoverlapping(ot.as_ptr().add(r * d), base.at(dst), d);
            }
        },
    );
    out
}

/// A stride-1 "same" convolution (1x1 or 3x3) on the bf16 kernel: each task unfolds 64 output pixels into an
/// im2col block and multiplies it with the pre-packed weights, so the full unfolded matrix never exists.
pub struct Conv {
    packed: PackedB,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
}

impl Conv {
    /// `w`: `[Cout, Cin, k, k]` f32, `bias`: `[Cout]` (or `None`).
    pub fn new(w: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        let (cout, cin, k, k2) = w.dims4()?;
        if k != k2 || !(k == 1 || k == 3) {
            bail!("bf16 conv supports 1x1 and 3x3 kernels, got {k}x{k2}");
        }
        // weight as [K = Cin*k*k, N = Cout]; K order is (cin, ky, kx), matching the unfold below
        let wm = w.reshape((cout, cin * k * k))?.t()?.contiguous()?;
        Ok(Self {
            packed: PackedB::new(&wm)?,
            bias: match bias {
                Some(b) => b.flatten_all()?.to_vec1::<f32>()?,
                None => vec![0.0; cout],
            },
            cin,
            cout,
            k,
        })
    }

    /// `x`: `[1, Cin, H, W]` f32 -> `[1, Cout, H, W]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.contiguous()?.apply_op1_no_bwd(&ConvOp(self))
    }
}

struct ConvOp<'a>(&'a Conv);

impl CustomOp1 for ConvOp<'_> {
    fn name(&self) -> &'static str {
        "bf16-conv"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let c = self.0;
        let CpuStorage::F32(x) = s else { bail!("bf16-conv expects f32") };
        let (n, cin, h, w) = l.shape().dims4()?;
        if n != 1 || cin != c.cin {
            bail!("bf16-conv: expected [1, {}, H, W], got {:?}", c.cin, l.shape());
        }
        let x = match l.contiguous_offsets() {
            Some((a, e)) => &x[a..e],
            None => bail!("bf16-conv needs contiguous input"),
        };
        let (hw, k, pad) = (h * w, c.k, c.k / 2);
        let kk = cin * k * k;
        let mut out = vec![0f32; c.cout * hw];
        struct Out(*mut f32);
        // SAFETY: each task writes the pixels [p0, p0+rows) of every output channel, disjoint across tasks.
        unsafe impl Sync for Out {}
        impl Out {
            fn at(&self, off: usize) -> *mut f32 {
                // SAFETY: callers stay inside the allocation.
                unsafe { self.0.add(off) }
            }
        }
        let base = Out(out.as_mut_ptr());
        const PIX: usize = 64;
        (0..hw.div_ceil(PIX)).into_par_iter().for_each_init(
            || (vec![0f32; PIX * kk], vec![0f32; PIX * c.cout]),
            |(a, o), blk| {
                let (p0, rows) = (blk * PIX, PIX.min(hw - blk * PIX));
                for r in 0..rows {
                    let (py, px) = ((p0 + r) / w, (p0 + r) % w);
                    let row = &mut a[r * kk..(r + 1) * kk];
                    if k == 1 {
                        for ch in 0..cin {
                            row[ch] = x[ch * hw + p0 + r];
                        }
                    } else {
                        for ch in 0..cin {
                            for ky in 0..3 {
                                let yy = py as isize + ky as isize - pad as isize;
                                for kx in 0..3 {
                                    let xx = px as isize + kx as isize - pad as isize;
                                    row[ch * 9 + ky * 3 + kx] = if yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize { x[ch * hw + yy as usize * w + xx as usize] } else { 0.0 };
                                }
                            }
                        }
                    }
                }
                // SAFETY: `Conv` is only built on paths gated by `enabled()`.
                unsafe { c.packed.matmul_rows(&a[..rows * kk], rows, &mut o[..rows * c.cout]) };
                for co in 0..c.cout {
                    for r in 0..rows {
                        // SAFETY: see `Out`.
                        unsafe { *base.at(co * hw + p0 + r) = o[r * c.cout + co] + c.bias[co] };
                    }
                }
            },
        );
        Ok((CpuStorage::F32(out), Shape::from((1, c.cout, h, w))))
    }
}
