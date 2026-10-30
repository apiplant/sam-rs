//! Compiles `kernels/*.cu` to PTX when the `cuda` feature is on. PTX (rather than a cubin) is forward
//! compatible: the driver JIT-compiles it for whatever GPU runs the binary.

use std::path::PathBuf;
use std::process::Command;

fn compute_cap() -> String {
    if let Ok(c) = std::env::var("CUDA_COMPUTE_CAP") {
        return c.trim().replace('.', "");
    }
    Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().replace('.', "")))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "80".into())
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let nvcc = std::env::var("CUDA_PATH").map(|p| format!("{p}/bin/nvcc")).unwrap_or_else(|_| "nvcc".into());
    let cap = compute_cap();
    for name in ["sam"] {
        let status = Command::new(&nvcc)
            .args(["--ptx", "-O3", "--use_fast_math", "-std=c++17", &format!("-arch=compute_{cap}")])
            .arg(format!("kernels/{name}.cu"))
            .arg("-o")
            .arg(out.join(format!("{name}.ptx")))
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc failed on kernels/{name}.cu");
    }
}
