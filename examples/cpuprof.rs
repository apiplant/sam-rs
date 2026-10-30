//! Times the CPU building blocks of one ViT-B block to see where encoder time goes.
use std::time::Instant;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, ops};

fn time<T>(name: &str, flops: f64, n: usize, mut f: impl FnMut() -> Result<T>) -> Result<()> {
    f()?;
    let t = Instant::now();
    for _ in 0..n {
        f()?;
    }
    let s = t.elapsed().as_secs_f64() / n as f64;
    if flops > 0.0 {
        println!("{name:<28} {:8.2} ms  {:7.1} GFLOP/s", s * 1e3, flops / s / 1e9);
    } else {
        println!("{name:<28} {:8.2} ms", s * 1e3);
    }
    Ok(())
}

fn main() -> Result<()> {
    let d = Device::Cpu;
    let x = Tensor::randn(0f32, 1., (4096, 768), &d)?;
    let w = Tensor::randn(0f32, 1., (768, 2304), &d)?;
    let w4 = Tensor::randn(0f32, 1., (768, 3072), &d)?;
    time("matmul qkv 4096x768x2304", 2. * 4096. * 768. * 2304., 3, || x.matmul(&w))?;
    time("matmul mlp 4096x768x3072", 2. * 4096. * 768. * 3072., 3, || x.matmul(&w4))?;
    if sam_rs::cpu_gemm::available() {
        let pb = sam_rs::cpu_gemm::PackedB::new(&w)?;
        let pb4 = sam_rs::cpu_gemm::PackedB::new(&w4)?;
        time("BF16 qkv 4096x768x2304", 2. * 4096. * 768. * 2304., 5, || pb.matmul(&x))?;
        time("BF16 mlp 4096x768x3072", 2. * 4096. * 768. * 3072., 5, || pb4.matmul(&x))?;
        let want = x.matmul(&w)?;
        let got = pb.matmul(&x)?;
        let err = (&got - &want)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let mag = want.abs()?.mean_all()?.to_scalar::<f32>()?;
        println!("bf16 vs f32 matmul: mean abs err {err:.4} (mean |value| {mag:.3}, rel {:.2e})", err / mag);
    }
    let q = Tensor::randn(0f32, 1., (12, 4096, 64), &d)?;
    let kt = Tensor::randn(0f32, 1., (12, 64, 4096), &d)?;
    time("bmm global qk 12x4096^2x64", 2. * 12. * 4096. * 4096. * 64., 2, || q.matmul(&kt))?;
    let qw = Tensor::randn(0f32, 1., (300, 196, 64), &d)?;
    let ktw = Tensor::randn(0f32, 1., (300, 64, 196), &d)?;
    time("bmm window qk 300x196^2x64", 2. * 300. * 196. * 196. * 64., 3, || qw.matmul(&ktw))?;
    let xl = Tensor::randn(0f32, 1., (1, 64, 64, 768), &d)?;
    let ln = LayerNorm::new(Tensor::ones(768, DType::F32, &d)?, Tensor::zeros(768, DType::F32, &d)?, 1e-6);
    time("layernorm 64x64x768", 0., 5, || ln.forward(&xl))?;
    let m = x.matmul(&w4)?;
    time("gelu_erf 4096x3072", 0., 5, || m.gelu_erf())?;
    let b = Tensor::randn(0f32, 1., 3072, &d)?;
    time("broadcast_add bias", 0., 5, || m.broadcast_add(&b))?;
    time("permute+contiguous", 0., 5, || xl.reshape((1, 64, 64, 12, 64))?.permute((0, 3, 1, 2, 4))?.contiguous())?;
    let s = Tensor::randn(0f32, 1., (12, 4096, 4096), &d)?;
    time("softmax 12x4096x4096", 0., 2, || ops::softmax_last_dim(&s))?;
    let img = Tensor::randn(0f32, 1., (1, 3, 1024, 1024), &d)?;
    let pw = Tensor::randn(0f32, 1., (768, 3, 16, 16), &d)?;
    time("patch conv 16x16 s16", 2. * 4096. * 768. * 768., 2, || img.conv2d(&pw, 0, 16, 1, 1))?;
    let nk = Tensor::randn(0f32, 1., (256, 768, 1, 1), &d)?;
    let fm = Tensor::randn(0f32, 1., (1, 768, 64, 64), &d)?;
    time("neck conv1x1", 2. * 4096. * 768. * 256., 2, || fm.conv2d(&nk, 0, 1, 1, 1))?;
    let n3 = Tensor::randn(0f32, 1., (256, 256, 3, 3), &d)?;
    let f2 = Tensor::randn(0f32, 1., (1, 256, 64, 64), &d)?;
    time("neck conv3x3", 2. * 4096. * 256. * 256. * 9., 2, || f2.conv2d(&n3, 1, 1, 1, 1))?;
    let src = Tensor::randn(0f32, 1., (1, 256, 64, 64), &d)?;
    let k1 = Tensor::randn(0f32, 1., (256, 64, 2, 2), &d)?;
    time("convT 256->64 @64x64", 0., 3, || src.conv_transpose2d(&k1, 0, 0, 2, 1))?;
    let s2 = Tensor::randn(0f32, 1., (1, 64, 128, 128), &d)?;
    let k2 = Tensor::randn(0f32, 1., (64, 32, 2, 2), &d)?;
    time("convT 64->32 @128x128", 0., 3, || s2.conv_transpose2d(&k2, 0, 0, 2, 1))?;
    let src = Tensor::randn(0f32, 1., (1, 256, 64, 64), &d)?;
    let k1 = Tensor::randn(0f32, 1., (256, 64, 2, 2), &d)?;
    time("convT 256->64 @64x64", 0., 3, || src.conv_transpose2d(&k1, 0, 0, 2, 1))?;
    let s2 = Tensor::randn(0f32, 1., (1, 64, 128, 128), &d)?;
    let k2 = Tensor::randn(0f32, 1., (64, 32, 2, 2), &d)?;
    time("convT 64->32 @128x128", 0., 3, || s2.conv_transpose2d(&k2, 0, 0, 2, 1))?;
    Ok(())
}
