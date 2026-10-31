//! Downloads checkpoints from Hugging Face into `$XDG_CACHE_HOME/sam-rs/<model>/` (default
//! `~/.cache/sam-rs/...`). Files are fetched to a `.part` sibling and renamed into place once
//! complete, so a killed download never leaves a checkpoint that looks done.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// `$XDG_CACHE_HOME/sam-rs`, defaulting to `~/.cache/sam-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("sam-rs"))
}

/// Hugging Face access token: `$HF_TOKEN` (or `$HUGGING_FACE_HUB_TOKEN`), else the file written by
/// `huggingface-cli login` / `hf auth login` (`$HF_HOME/token`, default `~/.cache/huggingface/token`).
/// Only gated models (SAM 3) need one.
pub fn hf_token() -> Option<String> {
    let clean = |s: String| Some(s.trim().to_owned()).filter(|s| !s.is_empty());
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = std::env::var(var).ok().and_then(clean) {
            return Some(t);
        }
    }
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")))?;
    std::fs::read_to_string(home.join("token")).ok().and_then(clean)
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension(match dest.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });

    let mut req = ureq::get(url);
    let token = hf_token();
    if let Some(t) = &token {
        // ureq drops this header when Hugging Face redirects to its CDN, so the token stays on huggingface.co.
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code @ (401 | 403), _)) => {
            let hint = if token.is_none() {
                "this model is gated: accept its terms on its Hugging Face page, create a read token at https://huggingface.co/settings/tokens and export HF_TOKEN=hf_..."
            } else {
                "HF_TOKEN is set but was refused: request access on the model's Hugging Face page (approval can be manual) and make sure the token may read gated repos"
            };
            bail!("HTTP {code} for {url}: {hint}");
        }
        Err(e) => return Err(e).with_context(|| format!("downloading {url}")),
    };
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut written: u64 = 0;
    let mut last_report = 0u64;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 32 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!("\r  {name} done ({} MiB)          ", written >> 20);
    Ok(())
}

/// Returns the local path of `file` from Hugging Face repo `repo`, downloading it on first use.
pub fn fetch(repo: &str, file: &str) -> Result<PathBuf> {
    let dir = cache_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?.join(repo.replace('/', "--"));
    let dest = dir.join(file);
    if !dest.is_file() {
        eprintln!("Downloading {repo}/{file} from https://huggingface.co/{repo}");
        fetch_to_file(&format!("https://huggingface.co/{repo}/resolve/main/{file}"), &dest)?;
    }
    Ok(dest)
}

/// Local path of `file` from `repo` when it is already in the cache (never downloads).
pub fn cached(repo: &str, file: &str) -> Option<PathBuf> {
    let p = cache_dir()?.join(repo.replace('/', "--")).join(file);
    p.is_file().then_some(p)
}

/// Path of the EfficientSAM checkpoint `name` (`efficient_sam_vitt` / `efficient_sam_vits`) as a `.safetensors` file.
/// The upstream release is a PyTorch `.pt`; on first use it is downloaded and converted once (f32, written to a
/// `.part` file and renamed) into the cache. candle's pickle reader only builds tensors, it never runs pickled code.
/// If the conversion fails the `.pt` itself is returned, and the loader reads it directly.
pub fn fetch_efficientsam(name: &str) -> Result<PathBuf> {
    let dir = cache_dir().context("no cache directory available")?.join("yunyangx--EfficientSAM");
    let st = dir.join(format!("{name}.safetensors"));
    if st.is_file() {
        return Ok(st);
    }
    let pt = fetch("yunyangx/EfficientSAM", &format!("{name}.pt"))?;
    match convert_pt(&pt, &st) {
        Ok(()) => Ok(st),
        Err(e) => {
            eprintln!("could not convert {} to safetensors ({e:#}); reading the .pt directly", pt.display());
            Ok(pt)
        }
    }
}

/// Converts a PyTorch checkpoint (a flat state dict, or one nested under `"model"`) to a safetensors file.
pub fn convert_pt(pt: &Path, dest: &Path) -> Result<()> {
    use candle_core::{pickle, DType};
    let mut tensors = pickle::read_all(pt).with_context(|| format!("reading {}", pt.display()))?;
    if !tensors.iter().any(|(k, _)| k.ends_with("patch_embed.proj.weight")) {
        tensors = pickle::read_all_with_key(pt, Some("model")).with_context(|| format!("reading {} (key \"model\")", pt.display()))?;
    }
    if tensors.is_empty() {
        bail!("no tensors found in {}", pt.display());
    }
    let map = tensors
        .into_iter()
        .map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?.contiguous()?)))
        .collect::<candle_core::Result<std::collections::HashMap<_, _>>>()?;
    let part = dest.with_extension("safetensors.part");
    candle_core::safetensors::save(&map, &part).with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, dest)?;
    eprintln!("converted {} -> {} ({} tensors)", pt.display(), dest.display(), map.len());
    Ok(())
}
