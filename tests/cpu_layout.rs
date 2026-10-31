//! The parallel CPU layout kernels must match the plain tensor ops they replace.
use candle_core::{Device, Result, Tensor};
use sam_rs::kernels::{bias_split3, maxpool_nhwc, nchw_to_nhwc_add, nhwc_to_nchw};

fn rand(shape: &[usize]) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, shape, &Device::Cpu)
}

fn same(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let d = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(d == 0.0, "max diff {d}");
    Ok(())
}

#[test]
fn bias_split3_matches_reshape() -> Result<()> {
    let (y, b) = (rand(&[37, 3 * 24])?, rand(&[3 * 24])?);
    let want = y.broadcast_add(&b)?.reshape((37, 3, 24))?.permute((1, 0, 2))?.contiguous()?;
    same(&bias_split3(&y, &b)?, &want)
}

#[test]
fn layout_changes_match_permute() -> Result<()> {
    let x = rand(&[1, 5, 6, 7])?; // NCHW
    let pos = rand(&[1, 6, 7, 5])?;
    same(&nchw_to_nhwc_add(&x, &pos)?, &(x.permute((0, 2, 3, 1))? + &pos)?)?;
    let y = rand(&[1, 6, 7, 5])?; // NHWC
    same(&nhwc_to_nchw(&y)?, &y.permute((0, 3, 1, 2))?.contiguous()?)
}

#[test]
fn maxpool_nhwc_matches_candle() -> Result<()> {
    for (b, h, w, c, s) in [(1, 8, 8, 3, 2), (3, 12, 6, 17, 2), (2, 9, 6, 4, 3)] {
        let x = rand(&[b, h, w, c])?;
        let want = x.permute((0, 3, 1, 2))?.contiguous()?.max_pool2d(s)?.permute((0, 2, 3, 1))?.contiguous()?;
        same(&maxpool_nhwc(&x, s)?, &want)?;
    }
    Ok(())
}
