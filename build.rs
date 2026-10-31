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

/// Writes `$OUT_DIR/ui.rs`: the built web UI (`ui/dist`, from `pnpm build` in `ui/`) as `(path, bytes)` pairs for
/// `sam-rs serve` to embed. Without a build the list is empty and `/` says how to make one.
fn embed_ui(out: &std::path::Path) {
    fn walk(dir: &std::path::Path, root: &std::path::Path, files: &mut Vec<(String, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, files);
            } else {
                files.push((p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p.canonicalize().unwrap()));
            }
        }
    }
    let root = PathBuf::from("ui/dist");
    println!("cargo:rerun-if-changed=ui/dist");
    println!("cargo:rerun-if-env-changed=SAM_RS_REQUIRE_UI");
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    if files.is_empty() && std::env::var_os("SAM_RS_REQUIRE_UI").is_some_and(|v| v != "0") {
        panic!("SAM_RS_REQUIRE_UI is set but ui/dist is empty: run `pnpm install && pnpm build` in ui/");
    }
    let mut src = String::from("&[\n");
    for (path, abs) in &files {
        println!("cargo:rerun-if-changed={}", abs.display());
        src += &format!("    ({path:?}, include_bytes!({:?}) as &[u8]),\n", abs.display().to_string());
    }
    src += "]\n";
    std::fs::write(out.join("ui.rs"), src).unwrap();
}

fn main() {
    embed_ui(&PathBuf::from(std::env::var("OUT_DIR").unwrap()));
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
