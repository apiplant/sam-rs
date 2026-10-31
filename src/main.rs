use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use sam_rs::{Prompt, Sam, SamVariant};

#[derive(Parser)]
#[command(version, about = "Segment Anything inference in Rust (candle), CPU and CUDA")]
struct Args {
    /// Input image (not needed with --video)
    #[arg(long, required_unless_present = "video", default_value = "")]
    image: String,
    /// Model: sam-vit-b | sam-vit-l | sam-vit-h | mobile-sam | efficientsam-ti | efficientsam-s | sam2.1-hiera-{tiny,small,base-plus,large} | sam2-hiera-* | sam3
    #[arg(long, default_value = "sam-vit-b")]
    model: String,
    /// Use this checkpoint file instead of the cached / downloaded one (Meta layout, .safetensors)
    #[arg(long)]
    weights: Option<String>,
    /// Text prompt (SAM 3): find and segment every instance of this concept
    #[arg(long)]
    text: Option<String>,
    /// Minimum instance score for --text
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,
    /// Compute dtype on CUDA: f16 (default) or f32
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// Point prompts "x,y" (foreground) or "x,y,0" (background); repeatable
    #[arg(long = "point")]
    points: Vec<String>,
    /// Box prompt "x0,y0,x1,y1". SAM 3: repeatable, positive exemplar boxes ("find more like this")
    #[arg(long = "box")]
    boxes: Vec<String>,
    /// SAM 3: track the --point/--box object through the frames (images, sorted by name) of this directory
    #[arg(long)]
    video: Option<String>,
    /// SAM 3 --video: index of the frame the prompt belongs to (tracking runs forward from there)
    #[arg(long, default_value_t = 0)]
    prompt_frame: usize,
    /// Text detector for SAM / MobileSAM / EfficientSAM / SAM 2: grounding-dino-tiny | grounding-dino-base. With --text,
    /// Grounding DINO finds the boxes and the model segments inside each one
    #[arg(long)]
    detector: Option<String>,
    /// Grounding DINO: minimum box score
    #[arg(long, default_value_t = 0.35)]
    box_threshold: f32,
    /// Grounding DINO: minimum token score for a word to label a box
    #[arg(long, default_value_t = 0.25)]
    text_threshold: f32,
    /// SAM 3 --video --text: run the detector every N frames to find objects that appear later
    #[arg(long, default_value_t = 5)]
    detect_every: usize,
    /// SAM 3: treat --point / --box as SAM-style prompts for one object (the tracker) instead of text/exemplars
    #[arg(long)]
    tracker: bool,
    /// SAM 3: negative exemplar box "x0,y0,x1,y1" (repeatable)
    #[arg(long = "neg-box")]
    neg_boxes: Vec<String>,
    /// Save the best mask as a PNG
    #[arg(long)]
    out: Option<String>,
    /// Save this candidate (0-2) instead of the one with the highest predicted IoU
    #[arg(long)]
    index: Option<usize>,
    /// Return a single mask instead of three candidates
    #[arg(long)]
    single: bool,
    /// Force CPU
    #[arg(long)]
    cpu: bool,
    /// CPU: use exact f32 instead of the bf16 kernels (AVX-512 BF16 CPUs only; slower, bit-for-bit f32)
    #[arg(long)]
    f32: bool,
    /// Time the encoder and decoder over N runs
    #[arg(long)]
    bench: Option<usize>,
}

fn nums(s: &str) -> Result<Vec<f32>> {
    s.split(',').map(|v| v.trim().parse::<f32>().with_context(|| format!("bad number {v:?}"))).collect()
}

/// `sam-rs serve`: the HTTP API and web UI.
#[cfg(feature = "serve")]
#[derive(Parser)]
#[command(name = "sam-rs serve", about = "Serve every model over an HTTP API, with a web UI at /")]
struct ServeArgs {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
    /// Force CPU
    #[arg(long)]
    cpu: bool,
    /// CPU: exact f32 instead of the bf16 kernels
    #[arg(long)]
    f32: bool,
    /// Compute dtype on CUDA: f16 (default) or f32
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// How many models stay loaded at once; the least recently used is dropped first
    #[arg(long, default_value_t = 2)]
    max_models: usize,
    /// Load this model before listening (repeatable)
    #[arg(long = "preload")]
    preload: Vec<String>,
    /// Never download checkpoints: only models already in the cache can be loaded
    #[arg(long)]
    no_download: bool,
}

fn main() -> Result<()> {
    sam_rs::cpu_gemm::init_threads();
    #[cfg(feature = "serve")]
    if std::env::args().nth(1).as_deref() == Some("serve") {
        let a = ServeArgs::parse_from(std::env::args().skip(1));
        if a.f32 {
            sam_rs::cpu_gemm::set_enabled(false);
        }
        return sam_rs::serve::run(sam_rs::serve::Options { addr: a.addr, cpu: a.cpu, dtype: a.dtype, max_models: a.max_models, allow_download: !a.no_download, preload: a.preload });
    }
    let a = Args::parse();
    if a.f32 {
        sam_rs::cpu_gemm::set_enabled(false);
    }
    if a.model.starts_with("grounding-dino") {
        return run_dino(&a);
    }
    if a.detector.is_some() {
        return run_grounded(&a);
    }
    if let Some(v) = sam_rs::efficientsam::Variant::parse(&a.model) {
        return run_efficientsam(&a, v);
    }
    if let Some(v) = sam_rs::sam2::Sam2Variant::parse(&a.model) {
        return run_sam2(&a, v);
    }
    if a.model.starts_with("sam3") {
        return run_sam3(&a);
    }
    let variant = SamVariant::parse(&a.model).with_context(|| format!("unknown model {}", a.model))?;
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    let t = Instant::now();
    let sam = match &a.weights {
        Some(w) => Sam::from_file(std::path::Path::new(w), variant, &device, dtype)?,
        None => Sam::from_pretrained(variant, &device, dtype)?,
    };
    eprintln!("loaded in {:.2?} on {:?} ({dtype:?})", t.elapsed(), device.location());

    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let prompt = if let Some(b) = a.boxes.first() {
        let v = nums(b)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        Prompt::Box([v[0], v[1], v[2], v[3]])
    } else if !a.points.is_empty() {
        let mut p = vec![];
        for s in &a.points {
            let v = nums(s)?;
            if v.len() < 2 {
                bail!("--point needs x,y[,label]");
            }
            p.push((v[0], v[1], v.get(2).map_or(true, |l| *l != 0.0)));
        }
        Prompt::Points(p)
    } else {
        Prompt::Points(vec![(img.width() as f32 / 2.0, img.height() as f32 / 2.0, true)])
    };

    let t = Instant::now();
    let emb = sam.embed(&img)?;
    let masks = sam.predict(&emb, &prompt, !a.single)?;
    eprintln!("embed + decode: {:.2?}", t.elapsed());
    if let Some(n) = a.bench {
        let t = Instant::now();
        for _ in 0..n {
            sam_rs::image_proc::prepare_sam1(&img, &sam.device, sam.dtype)?;
        }
        sam.sync()?;
        eprintln!("preprocess alone: {:.1} ms", t.elapsed().as_secs_f64() / n as f64 * 1e3);
        let (mut te, mut td) = (0.0, 0.0);
        for _ in 0..n {
            let t = Instant::now();
            let e = sam.embed(&img)?;
            sam.sync()?;
            te += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sam.predict(&e, &prompt, !a.single)?;
            td += t.elapsed().as_secs_f64();
        }
        eprintln!("bench x{n}: encoder {:.1} ms, decoder {:.1} ms", te / n as f64 * 1e3, td / n as f64 * 1e3);
    }
    let best = match a.index {
        Some(i) => masks.get(i).with_context(|| format!("--index {i}: only {} masks", masks.len()))?,
        None => masks.iter().max_by(|a, b| a.iou.total_cmp(&b.iou)).context("no masks")?,
    };
    for (i, m) in masks.iter().enumerate() {
        println!("mask {i}: iou {:.3}, area {}", m.iou, m.data.iter().filter(|&&v| v > 0).count());
    }
    if let Some(out) = &a.out {
        let buf: Vec<u8> = best.data.iter().map(|&v| v * 255).collect();
        image::GrayImage::from_raw(best.width as u32, best.height as u32, buf).context("mask buffer")?.save(out)?;
    }
    Ok(())
}

fn run_sam3(a: &Args) -> Result<()> {
    use sam_rs::sam3::Sam3;
    if a.video.is_some() {
        return run_sam3_video(a);
    }
    if !a.points.is_empty() || a.tracker {
        return run_sam3_tracker(a);
    }
    let mut exemplars: Vec<([f32; 4], bool)> = vec![];
    for (list, positive) in [(&a.boxes, true), (&a.neg_boxes, false)] {
        for b in list {
            let v = nums(b)?;
            if v.len() != 4 {
                bail!("boxes need x0,y0,x1,y1");
            }
            exemplars.push(([v[0], v[1], v[2], v[3]], positive));
        }
    }
    // a purely visual query uses the placeholder text the model was trained with
    let prompt = a.text.as_deref().unwrap_or(if exemplars.is_empty() { "" } else { "visual" });
    if prompt.is_empty() {
        bail!("SAM 3 needs --text and/or --box");
    }
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let t = Instant::now();
    let sam = Sam3::from_pretrained(&device, dtype)?;
    eprintln!("loaded in {:.2?} on {:?} ({dtype:?})", t.elapsed(), device.location());
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let t = Instant::now();
    let feats = sam.encode_image(&img)?;
    let found = sam.segment_with(&feats, prompt, &exemplars, a.threshold)?;
    eprintln!("encode + segment: {:.2?}", t.elapsed());
    if let Some(n) = a.bench {
        let (mut te, mut th) = (0.0, 0.0);
        for _ in 0..n {
            let t = Instant::now();
            let f = sam.encode_image(&img)?;
            sam.sync()?;
            te += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sam.segment_with(&f, prompt, &exemplars, a.threshold)?;
            th += t.elapsed().as_secs_f64();
        }
        eprintln!("bench x{n}: image encoder {:.1} ms, text + head + masks {:.1} ms", te / n as f64 * 1e3, th / n as f64 * 1e3);
    }
    println!("{} instance(s) for {prompt:?}", found.len());
    for (i, f) in found.iter().enumerate() {
        println!("  #{i}: score {:.3} box [{:.0}, {:.0}, {:.0}, {:.0}] area {}", f.score, f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.mask.data.iter().filter(|&&v| v > 0).count());
    }
    if let (Some(out), Some(best)) = (&a.out, found.first()) {
        let buf: Vec<u8> = best.mask.data.iter().map(|&v| v * 255).collect();
        image::GrayImage::from_raw(best.mask.width as u32, best.mask.height as u32, buf).context("mask buffer")?.save(out)?;
    }
    Ok(())
}

/// SAM 3's image tracker: SAM 1/2-style point and box prompts for a single object.
fn run_sam3_tracker(a: &Args) -> Result<()> {
    use sam_rs::sam3::Sam3;
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let sam = Sam3::from_pretrained(&device, dtype)?;
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let mut points = vec![];
    for s in &a.points {
        let v = nums(s)?;
        if v.len() < 2 {
            bail!("--point needs x,y[,label]");
        }
        points.push((v[0], v[1], v.get(2).map_or(true, |l| *l != 0.0)));
    }
    let mut boxes = vec![];
    for s in &a.boxes {
        let v = nums(s)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        boxes.push([v[0], v[1], v[2], v[3]]);
    }
    if points.is_empty() && boxes.is_empty() {
        bail!("give --point and/or --box");
    }
    let t = Instant::now();
    let feats = sam.encode_image(&img)?;
    let tf = sam.tracker_features(&feats)?;
    let (masks, obj) = sam.segment_points(&feats, &tf, &points, &boxes, !a.single)?;
    eprintln!("encode + tracker: {:.2?}", t.elapsed());
    if let Some(n) = a.bench {
        let (mut te, mut td) = (0.0, 0.0);
        for _ in 0..n {
            let t = Instant::now();
            let f = sam.encode_image(&img)?;
            let tf = sam.tracker_features(&f)?;
            sam.sync()?;
            te += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sam.segment_points(&f, &tf, &points, &boxes, !a.single)?;
            td += t.elapsed().as_secs_f64();
        }
        eprintln!("bench x{n}: image + tracker features {:.1} ms, prompt decode {:.1} ms", te / n as f64 * 1e3, td / n as f64 * 1e3);
    }
    println!("object score {obj:.2} ({})", if obj > 0.0 { "object present" } else { "no object" });
    for (i, m) in masks.iter().enumerate() {
        println!("mask {i}: iou {:.3}, area {}", m.iou, m.data.iter().filter(|&&v| v > 0).count());
    }
    let best = match a.index {
        Some(i) => masks.get(i).with_context(|| format!("--index {i}: only {} masks", masks.len()))?,
        None => masks.iter().max_by(|x, y| x.iou.total_cmp(&y.iou)).context("no masks")?,
    };
    if let Some(out) = &a.out {
        let buf: Vec<u8> = best.data.iter().map(|&v| v * 255).collect();
        image::GrayImage::from_raw(best.width as u32, best.height as u32, buf).context("mask buffer")?.save(out)?;
    }
    Ok(())
}

/// SAM 3 video tracker over a directory of frames; writes one mask PNG per frame into `--out DIR`.
fn run_sam3_video(a: &Args) -> Result<()> {
    use sam_rs::sam3::{video::{FramePrompt, Session}, Sam3};
    let dir = a.video.as_deref().context("--video DIR")?;
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {dir}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref(), Some("png" | "jpg" | "jpeg" | "bmp" | "webp")))
        .collect();
    files.sort();
    if a.prompt_frame >= files.len() {
        bail!("--prompt-frame {} but only {} frames in {dir}", a.prompt_frame, files.len());
    }
    if let Some(text) = a.text.as_deref().filter(|_| a.points.is_empty() && a.boxes.is_empty()) {
        return run_sam3_text_video(a, text, &files);
    }
    let mut prompt = FramePrompt::default();
    let first = image::open(&files[a.prompt_frame])?.to_rgb8();
    let (sx, sy) = (1008.0 / first.width() as f32, 1008.0 / first.height() as f32);
    for s in &a.points {
        let v = nums(s)?;
        if v.len() < 2 {
            bail!("--point needs x,y[,label]");
        }
        prompt.points.push((v[0] * sx, v[1] * sy, v.get(2).map_or(1, |l| (*l != 0.0) as i32)));
    }
    for s in &a.boxes {
        let v = nums(s)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        // in video a box is two corner points labelled 2 and 3
        prompt.points.push((v[0] * sx, v[1] * sy, 2));
        prompt.points.push((v[2] * sx, v[3] * sy, 3));
    }
    if prompt.points.is_empty() {
        bail!("give --point and/or --box for the object to track");
    }
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let sam = Sam3::from_pretrained(&device, dtype)?;
    let mut session = Session::new(Some(files.len() - a.prompt_frame));
    session.add_prompt(0, prompt);
    if let Some(out) = &a.out {
        std::fs::create_dir_all(out)?;
    }
    let t_all = Instant::now();
    for (i, path) in files.iter().enumerate().skip(a.prompt_frame) {
        let t = Instant::now();
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let tf = sam.encode_for_tracking(&img)?;
        let r = sam.track_frame(&mut session, &tf)?;
        let mask = sam.frame_mask(&r, (img.height() as usize, img.width() as usize))?;
        let area = mask.data.iter().filter(|&&v| v > 0).count();
        println!("frame {i}: object score {:6.2} ({}), mask area {area}  [{:.0?}]", r.object_score, if r.object_score > 0.0 { "present" } else { "absent" }, t.elapsed());
        if let Some(out) = &a.out {
            let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{i}"));
            let buf: Vec<u8> = mask.data.iter().map(|&v| v * 255).collect();
            image::GrayImage::from_raw(mask.width as u32, mask.height as u32, buf).context("mask buffer")?.save(std::path::Path::new(out).join(format!("{name}_mask.png")))?;
        }
    }
    eprintln!("{} frames in {:.2?}", files.len() - a.prompt_frame, t_all.elapsed());
    Ok(())
}

/// SAM 2 / 2.1: point and box prompts, same flags as SAM 1.
fn run_sam2(a: &Args, variant: sam_rs::sam2::Sam2Variant) -> Result<()> {
    use sam_rs::sam2::Sam2;
    if a.video.is_some() {
        return run_sam2_video(a, variant);
    }
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let t = Instant::now();
    let sam = Sam2::from_pretrained(variant, &device, dtype)?;
    eprintln!("loaded {} in {:.2?} on {:?} ({dtype:?})", variant.repo(), t.elapsed(), device.location());
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let mut points = vec![];
    for s in &a.points {
        let v = nums(s)?;
        if v.len() < 2 {
            bail!("--point needs x,y[,label]");
        }
        points.push((v[0], v[1], v.get(2).map_or(true, |l| *l != 0.0)));
    }
    let mut boxes = vec![];
    for s in &a.boxes {
        let v = nums(s)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        boxes.push([v[0], v[1], v[2], v[3]]);
    }
    if points.is_empty() && boxes.is_empty() {
        points.push((img.width() as f32 / 2.0, img.height() as f32 / 2.0, true));
    }
    let t = Instant::now();
    let emb = sam.embed(&img)?;
    let (masks, obj) = sam.predict(&emb, &points, &boxes, !a.single)?;
    eprintln!("embed + decode: {:.2?}", t.elapsed());
    if let Some(n) = a.bench {
        let (mut te, mut td) = (0.0, 0.0);
        for _ in 0..n {
            let t = Instant::now();
            let e = sam.embed(&img)?;
            sam.sync()?;
            te += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sam.predict(&e, &points, &boxes, !a.single)?;
            td += t.elapsed().as_secs_f64();
        }
        eprintln!("bench x{n}: encoder {:.1} ms, decoder {:.1} ms", te / n as f64 * 1e3, td / n as f64 * 1e3);
    }
    println!("object score {obj:.2} ({})", if obj > 0.0 { "object present" } else { "no object" });
    for (i, m) in masks.iter().enumerate() {
        println!("mask {i}: iou {:.3}, area {}", m.iou, m.data.iter().filter(|&&v| v > 0).count());
    }
    let best = match a.index {
        Some(i) => masks.get(i).with_context(|| format!("--index {i}: only {} masks", masks.len()))?,
        None => masks.iter().max_by(|x, y| x.iou.total_cmp(&y.iou)).context("no masks")?,
    };
    if let Some(out) = &a.out {
        let buf: Vec<u8> = best.data.iter().map(|&v| v * 255).collect();
        image::GrayImage::from_raw(best.width as u32, best.height as u32, buf).context("mask buffer")?.save(out)?;
    }
    Ok(())
}

/// SAM 2 video tracking over a directory of frames; writes one mask PNG per frame into `--out DIR`.
fn run_sam2_video(a: &Args, variant: sam_rs::sam2::Sam2Variant) -> Result<()> {
    use sam_rs::{sam2::Sam2, sam3::video::Session};
    let dir = a.video.as_deref().context("--video DIR")?;
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {dir}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref(), Some("png" | "jpg" | "jpeg" | "bmp" | "webp")))
        .collect();
    files.sort();
    if a.prompt_frame >= files.len() {
        bail!("--prompt-frame {} but only {} frames in {dir}", a.prompt_frame, files.len());
    }
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let sam = Sam2::from_pretrained(variant, &device, dtype)?;
    let first = image::open(&files[a.prompt_frame])?.to_rgb8();
    let orig = (first.height() as usize, first.width() as usize);
    let mut points = vec![];
    for s in &a.points {
        let v = nums(s)?;
        if v.len() < 2 {
            bail!("--point needs x,y[,label]");
        }
        points.push((v[0], v[1], v.get(2).map_or(true, |l| *l != 0.0)));
    }
    let mut boxes = vec![];
    for s in &a.boxes {
        let v = nums(s)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        boxes.push([v[0], v[1], v[2], v[3]]);
    }
    if points.is_empty() && boxes.is_empty() {
        bail!("give --point and/or --box for the object to track");
    }
    let mut session = Session::new(Some(files.len() - a.prompt_frame));
    session.add_prompt(0, sam.prompt_points(orig, &points, &boxes));
    if let Some(out) = &a.out {
        std::fs::create_dir_all(out)?;
    }
    let t_all = Instant::now();
    for (i, path) in files.iter().enumerate().skip(a.prompt_frame) {
        let t = Instant::now();
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let e = sam.embed(&img)?;
        let r = sam.track_frame(&mut session, &e)?;
        let mask = sam.frame_mask(&r, (img.height() as usize, img.width() as usize))?;
        let area = mask.data.iter().filter(|&&v| v > 0).count();
        println!("frame {i}: object score {:6.2} ({}), mask area {area}  [{:.0?}]", r.object_score, if r.object_score > 0.0 { "present" } else { "absent" }, t.elapsed());
        if let Some(out) = &a.out {
            let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{i}"));
            let buf: Vec<u8> = mask.data.iter().map(|&v| v * 255).collect();
            image::GrayImage::from_raw(mask.width as u32, mask.height as u32, buf).context("mask buffer")?.save(std::path::Path::new(out).join(format!("{name}_mask.png")))?;
        }
    }
    eprintln!("{} frames in {:.2?}", files.len() - a.prompt_frame, t_all.elapsed());
    Ok(())
}

/// EfficientSAM: point and box prompts, three candidate masks.
fn run_efficientsam(a: &Args, variant: sam_rs::efficientsam::Variant) -> Result<()> {
    use sam_rs::efficientsam::EfficientSam;
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let sam = EfficientSam::from_pretrained(variant, &device, dtype)?;
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let mut points = vec![];
    for s in &a.points {
        let v = nums(s)?;
        if v.len() < 2 {
            bail!("--point needs x,y[,label]");
        }
        points.push((v[0], v[1], v.get(2).map_or(true, |l| *l != 0.0)));
    }
    let mut boxes = vec![];
    for s in &a.boxes {
        let v = nums(s)?;
        if v.len() != 4 {
            bail!("--box needs x0,y0,x1,y1");
        }
        boxes.push([v[0], v[1], v[2], v[3]]);
    }
    if points.is_empty() && boxes.is_empty() {
        points.push((img.width() as f32 / 2.0, img.height() as f32 / 2.0, true));
    }
    let t = Instant::now();
    let emb = sam.embed(&img)?;
    let masks = sam.predict(&emb, &points, &boxes)?;
    eprintln!("embed + decode: {:.2?}", t.elapsed());
    if let Some(n) = a.bench {
        let (mut te, mut td) = (0.0, 0.0);
        for _ in 0..n {
            let t = Instant::now();
            let e = sam.embed(&img)?;
            sam.sync()?;
            te += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sam.predict(&e, &points, &boxes)?;
            td += t.elapsed().as_secs_f64();
        }
        eprintln!("bench x{n}: encoder {:.1} ms, decoder {:.1} ms", te / n as f64 * 1e3, td / n as f64 * 1e3);
    }
    for (i, m) in masks.iter().enumerate() {
        println!("mask {i}: iou {:.3}, area {}", m.iou, m.data.iter().filter(|&&v| v > 0).count());
    }
    let best = match a.index {
        Some(i) => masks.get(i).with_context(|| format!("--index {i}: only {} masks", masks.len()))?,
        None => masks.iter().max_by(|x, y| x.iou.total_cmp(&y.iou)).context("no masks")?,
    };
    if let Some(out) = &a.out {
        let buf: Vec<u8> = best.data.iter().map(|&v| v * 255).collect();
        image::GrayImage::from_raw(best.width as u32, best.height as u32, buf).context("mask buffer")?.save(out)?;
    }
    Ok(())
}

/// SAM 3 text-prompted detect-and-track: every instance of `--text` is found, then followed through the frames.
/// Writes `NAME_obj<ID>_mask.png` per object and frame into `--out DIR`.
fn run_sam3_text_video(a: &Args, text: &str, files: &[std::path::PathBuf]) -> Result<()> {
    use sam_rs::sam3::{text_track::TextTracker, Sam3};
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let sam = Sam3::from_pretrained(&device, dtype)?;
    let mut tracker = TextTracker::new(&sam, text, a.threshold).detect_every(a.detect_every);
    if let Some(out) = &a.out {
        std::fs::create_dir_all(out)?;
    }
    let t_all = Instant::now();
    for (i, path) in files.iter().enumerate().skip(a.prompt_frame) {
        let t = Instant::now();
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let objects = tracker.step(&img)?;
        let summary: Vec<String> = objects.iter().map(|o| format!("#{} {} area {}{}", o.id, o.label, o.mask.data.iter().filter(|&&v| v > 0).count(), if o.detection_score.is_some() { " (detected)" } else { "" })).collect();
        println!("frame {i}: {} object(s) [{}]  [{:.0?}]", objects.len(), summary.join(", "), t.elapsed());
        if let Some(out) = &a.out {
            let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{i}"));
            for o in &objects {
                let buf: Vec<u8> = o.mask.data.iter().map(|&v| v * 255).collect();
                image::GrayImage::from_raw(o.mask.width as u32, o.mask.height as u32, buf).context("mask buffer")?.save(std::path::Path::new(out).join(format!("{name}_obj{}_mask.png", o.id)))?;
            }
        }
    }
    eprintln!("{} frames in {:.2?}", files.len() - a.prompt_frame, t_all.elapsed());
    Ok(())
}

fn dino_repo(name: &str) -> Result<&'static str> {
    match name {
        "grounding-dino-tiny" => Ok(sam_rs::dino::REPO_TINY),
        "grounding-dino-base" => Ok(sam_rs::dino::REPO_BASE),
        _ => bail!("unknown detector {name:?} (grounding-dino-tiny | grounding-dino-base)"),
    }
}

fn load_dino(name: &str, a: &Args) -> Result<sam_rs::dino::GroundingDino> {
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let t = Instant::now();
    // f32: the detector is small, and f16 is no faster here
    let m = sam_rs::dino::GroundingDino::from_repo(dino_repo(name)?, &device, DType::F32)?;
    eprintln!("loaded {name} in {:.2?} on {:?}", t.elapsed(), device.location());
    Ok(m)
}

/// Grounding DINO on its own: boxes for the phrases in `--text` ("a cat. a remote control.").
fn run_dino(a: &Args) -> Result<()> {
    let text = a.text.as_deref().context("--text \"a cat. a remote control.\" (phrases separated by full stops)")?;
    let dino = load_dino(&a.model, a)?;
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let t = Instant::now();
    let dets = dino.detect(&img, text, a.box_threshold, a.text_threshold)?;
    eprintln!("detect: {:.2?}", t.elapsed());
    for d in &dets {
        println!("{:.3} {:?} [{:.1}, {:.1}, {:.1}, {:.1}]", d.score, d.label, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]);
    }
    if dets.is_empty() {
        println!("no boxes above --box-threshold {}", a.box_threshold);
    }
    if let Some(out) = &a.out {
        let mut canvas = img.clone();
        let colors = [[115u8, 224, 201], [124, 140, 255], [255, 180, 84], [255, 122, 156], [182, 227, 92]];
        for (i, d) in dets.iter().enumerate() {
            let c = image::Rgb(colors[i % colors.len()]);
            let (x0, y0, x1, y1) = (d.bbox[0].max(0.0) as i32, d.bbox[1].max(0.0) as i32, (d.bbox[2] as i32).min(img.width() as i32 - 1), (d.bbox[3] as i32).min(img.height() as i32 - 1));
            for t in 0..3 {
                for x in x0..=x1 {
                    for y in [y0 + t, y1 - t] {
                        if y >= 0 && y < img.height() as i32 {
                            canvas.put_pixel(x as u32, y as u32, c);
                        }
                    }
                }
                for y in y0..=y1 {
                    for x in [x0 + t, x1 - t] {
                        if x >= 0 && x < img.width() as i32 {
                            canvas.put_pixel(x as u32, y as u32, c);
                        }
                    }
                }
            }
        }
        canvas.save(out)?;
    }
    Ok(())
}

/// Grounded segmentation: Grounding DINO turns `--text` into boxes and `--model` (SAM, MobileSAM, EfficientSAM or SAM 2)
/// segments inside each. `--out FILE.png` writes one mask per detection (`FILE_1.png`, `FILE_2.png`, ...).
fn run_grounded(a: &Args) -> Result<()> {
    use sam_rs::{dino, efficientsam::EfficientSam, model::{Prompt, Sam, SamVariant}, sam2::Sam2};
    let text = a.text.as_deref().context("--detector needs --text \"a cat. a remote control.\"")?;
    let device = if a.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = match (device.is_cuda(), a.dtype.as_str()) {
        (true, "f16") => DType::F16,
        _ => DType::F32,
    };
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image))?.to_rgb8();
    let detector = load_dino(a.detector.as_deref().unwrap_or_default(), a)?;
    let t = Instant::now();
    let dets = detector.detect(&img, text, a.box_threshold, a.text_threshold)?;
    eprintln!("detect: {:.2?} ({} boxes)", t.elapsed(), dets.len());
    drop(detector);
    let t = Instant::now();
    let grounded = if let Some(v) = sam_rs::efficientsam::Variant::parse(&a.model) {
        let m = EfficientSam::from_pretrained(v, &device, dtype)?;
        let e = m.embed(&img)?;
        dino::ground(dets, |b| m.predict(&e, &[], &[b]).map_err(Into::into))?
    } else if let Some(v) = sam_rs::sam2::Sam2Variant::parse(&a.model) {
        let m = Sam2::from_pretrained(v, &device, dtype)?;
        let e = m.embed(&img)?;
        dino::ground(dets, |b| Ok(m.predict(&e, &[], &[b], false)?.0))?
    } else if a.model.starts_with("sam3") {
        bail!("SAM 3 takes --text itself: drop --detector");
    } else {
        let v = SamVariant::parse(&a.model).with_context(|| format!("unknown model {}", a.model))?;
        let m = Sam::from_pretrained(v, &device, dtype)?;
        let e = m.embed(&img)?;
        dino::ground(dets, |b| m.predict(&e, &Prompt::Box(b), false))?
    };
    eprintln!("segment: {:.2?}", t.elapsed());
    for (i, (d, m)) in grounded.iter().enumerate() {
        println!("{}: {:?} score {:.3}, mask iou {:.3}, area {}", i + 1, d.label, d.score, m.iou, m.data.iter().filter(|&&v| v > 0).count());
        if let Some(out) = &a.out {
            let p = std::path::Path::new(out);
            let name = format!("{}_{}.png", p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mask".into()), i + 1);
            let buf: Vec<u8> = m.data.iter().map(|&v| v * 255).collect();
            image::GrayImage::from_raw(m.width as u32, m.height as u32, buf).context("mask buffer")?.save(p.with_file_name(name))?;
        }
    }
    Ok(())
}
