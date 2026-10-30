//! The bf16 CPU kernels must stay within bf16 rounding of the f32 reference.
use candle_core::{Device, Result, Tensor};
use sam_rs::cpu_gemm::{self, PackedB};

fn rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let e = (got - want)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    Ok(e / want.abs()?.mean_all()?.to_scalar::<f32>()?)
}

#[test]
fn bf16_matmul_matches_f32() -> Result<()> {
    if !cpu_gemm::available() {
        return Ok(());
    }
    let d = Device::Cpu;
    // odd sizes exercise the row tail, the column panel tail and the odd-K pad
    for (m, k, n) in [(64, 768, 2304), (13, 77, 45), (200, 3072, 768), (1, 5, 1)] {
        let x = Tensor::randn(0f32, 1., (m, k), &d)?;
        let w = Tensor::randn(0f32, 1., (k, n), &d)?;
        // bf16 rounding scales with the magnitude of the summed terms, not with the (possibly cancelling) result
        let (got, want) = (PackedB::new(&w)?.matmul(&x)?, x.matmul(&w)?);
        let scale = x.abs()?.matmul(&w.abs()?)?.mean_all()?.to_scalar::<f32>()?;
        let err = (&got - &want)?.abs()?.mean_all()?.to_scalar::<f32>()? / scale;
        assert!(err < 4e-3, "{m}x{k}x{n}: error relative to |x||w| is {err}");
    }
    Ok(())
}

#[test]
fn bf16_attention_matches_f32_attention() -> Result<()> {
    if !cpu_gemm::available() {
        return Ok(());
    }
    let d = Device::Cpu;
    for (h, w, bh, hd) in [(14, 14, 6, 64), (64, 64, 2, 64), (7, 11, 3, 80)] {
        let n = h * w;
        let q = Tensor::randn(0f32, 1., (bh, n, hd), &d)?;
        let kt = Tensor::randn(0f32, 1., (bh, hd, n), &d)?;
        let v = Tensor::randn(0f32, 1., (bh, n, hd), &d)?;
        let t = Tensor::randn(0f32, 1., (2 * h - 1 + 2 * w - 1, hd), &d)?;
        cpu_gemm::set_enabled(true);
        let fast = sam_rs::kernels::attention_relpos_cpu(&q, &kt, &v, &t, h, w, 0.125)?.unwrap();
        cpu_gemm::set_enabled(false);
        let exact = sam_rs::kernels::attention_relpos_cpu(&q, &kt, &v, &t, h, w, 0.125)?.unwrap();
        cpu_gemm::set_enabled(true);
        let err = rel_err(&fast, &exact)?;
        assert!(err < 1.5e-2, "{h}x{w}: relative error {err}");
    }
    Ok(())
}

#[test]
fn bf16_conv_matches_f32_conv() -> Result<()> {
    if !cpu_gemm::available() {
        return Ok(());
    }
    let d = Device::Cpu;
    // odd spatial sizes and channel counts exercise the pixel-block tail and the zero padding at the borders
    for (cin, cout, k, h, w) in [(256, 256, 3, 37, 41), (64, 32, 1, 23, 19), (5, 7, 3, 9, 9), (256, 256, 3, 72, 72)] {
        let x = Tensor::randn(0f32, 1., (1, cin, h, w), &d)?;
        let wt = Tensor::randn(0f32, 0.1, (cout, cin, k, k), &d)?;
        let b = Tensor::randn(0f32, 1., cout, &d)?;
        let want = x.conv2d(&wt, k / 2, 1, 1, 1)?.broadcast_add(&b.reshape((1, cout, 1, 1))?)?;
        let got = cpu_gemm::Conv::new(&wt, Some(&b))?.forward(&x)?;
        assert_eq!(got.dims(), want.dims());
        // error relative to the magnitude of the summed terms (|x| conv |w|), which is what bf16 rounding scales with
        let scale = x.abs()?.conv2d(&wt.abs()?, k / 2, 1, 1, 1)?.mean_all()?.to_scalar::<f32>()?;
        let err = (&got - &want)?.abs()?.mean_all()?.to_scalar::<f32>()? / scale;
        assert!(err < 4e-3, "{cin}->{cout} k{k} {h}x{w}: error relative to |x||w| is {err}");
    }
    Ok(())
}
