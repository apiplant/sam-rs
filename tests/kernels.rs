//! Each fused CUDA kernel must agree with its plain-tensor fallback (which is what the CPU build runs).
#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Result, Tensor};
use sam_rs::kernels::{bias_gelu, bias_residual, qkv_split, relpos_softmax};

fn gpu() -> Option<Device> {
    Device::new_cuda(0).ok()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_device(&Device::Cpu)?.to_dtype(DType::F32)? - b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

fn rand(shape: &[usize], dev: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?.to_device(dev)
}

#[test]
fn relpos_softmax_matches_fallback() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    // global (64x64) and windowed (14x14) attention geometries, plus an odd non-square one
    for (h, w, bh, hd) in [(14, 14, 6, 64), (64, 64, 2, 64), (7, 11, 3, 80)] {
        let n = h * w;
        let s = rand(&[bh, n, n], &Device::Cpu)?;
        let q = rand(&[bh, n, hd], &Device::Cpu)?;
        let t = rand(&[2 * h - 1 + 2 * w - 1, hd], &Device::Cpu)?;
        let on = |x: &Tensor| x.to_device(&g)?.to_dtype(DType::F16);
        // reference sees the same f16-rounded inputs, so only the kernel's own error is measured
        let round = |x: &Tensor| x.to_dtype(DType::F16)?.to_dtype(DType::F32);
        let want = relpos_softmax(&round(&s)?, &round(&q)?, &round(&t)?, h, w, 0.125)?;
        let got = relpos_softmax(&on(&s)?, &on(&q)?, &on(&t)?, h, w, 0.125)?;
        let d = max_abs_diff(&got, &want)?;
        assert!(d < 1e-3, "relpos_softmax {h}x{w}: max diff {d}");
    }
    Ok(())
}

#[test]
fn qkv_split_and_bias_ops_match_fallback() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let on = |x: &Tensor| x.to_device(&g)?.to_dtype(DType::F16);
    let (n, nh, hd, win) = (196, 12, 64, 3);
    let c = nh * hd;
    let y = rand(&[win * n, 3 * c], &Device::Cpu)?;
    let b = rand(&[3 * c], &Device::Cpu)?;
    let (q0, k0, v0) = qkv_split(&y, &b, n, nh, hd)?;
    let (q1, k1, v1) = qkv_split(&on(&y)?, &on(&b)?, n, nh, hd)?;
    for (a, b) in [(&q1, &q0), (&k1, &k0), (&v1, &v0)] {
        assert_eq!(a.dims(), b.dims());
        assert!(max_abs_diff(a, b)? < 2e-2);
    }

    let res = rand(&[1, 8, 8, c], &Device::Cpu)?;
    let y = rand(&[1, 8, 8, c], &Device::Cpu)?;
    let bias = rand(&[c], &Device::Cpu)?;
    assert!(max_abs_diff(&bias_residual(&on(&res)?, &on(&y)?, &on(&bias)?)?, &bias_residual(&res, &y, &bias)?)? < 2e-2);
    let y = rand(&[64, 4 * c], &Device::Cpu)?;
    let bias = rand(&[4 * c], &Device::Cpu)?;
    assert!(max_abs_diff(&bias_gelu(&on(&y)?, &on(&bias)?)?, &bias_gelu(&y, &bias)?)? < 2e-2);
    Ok(())
}

#[test]
fn deform_attn_matches_cpu() -> Result<()> {
    use sam_rs::kernels::deform_attn;
    let Some(g) = gpu() else { return Ok(()) };
    let shapes = [(9usize, 12usize), (5, 6), (3, 3), (2, 2)];
    let s: usize = shapes.iter().map(|(h, w)| h * w).sum();
    let (b, q, h, d, l, p) = (1usize, 37usize, 8usize, 32usize, 4usize, 4usize);
    let value = rand(&[b, s, h, d], &Device::Cpu)?;
    // sampling locations spread beyond [0, 1] so the zero padding is exercised
    let loc = (rand(&[b, q, h, l, p, 2], &Device::Cpu)? * 0.5)?.affine(1.0, 0.5)?;
    let attn = candle_nn::ops::softmax_last_dim(&rand(&[b, q, h, l * p], &Device::Cpu)?)?.reshape((b, q, h, l, p))?;
    let want = deform_attn(&value, &loc, &attn, &shapes)?;
    for dt in [DType::F32, DType::F16] {
        let got = deform_attn(&value.to_device(&g)?.to_dtype(dt)?, &loc.to_device(&g)?.to_dtype(dt)?, &attn.to_device(&g)?.to_dtype(dt)?, &shapes)?;
        let tol = if dt == DType::F32 { 1e-4 } else { 5e-2 };
        let e = max_abs_diff(&got, &want)?;
        assert!(e < tol, "{dt:?}: max diff {e}");
    }
    Ok(())
}
