/** Everything the home page says about the project, in one place. */

export interface Card {
  name: string;
  href: string;
  tagline: string;
  body: string;
}

export interface Feature {
  title: string;
  body: string;
}

export const SITE = {
  pkg: "sam-rs",
  hasCuda: true,
  bins: ["sam-rs"],
  badges: ["Rust", "candle", "CUDA", "Apache-2.0"],
  hero: { pre: "Segment Anything, ", accent: "in Rust", post: "." },
  lead: "Click a point, draw a box or name a concept and get masks back: SAM, MobileSAM, EfficientSAM, SAM 2 and SAM 3, plus Grounding DINO for text on any of them, from one binary and one library, on CPU or CUDA, with video tracking. No Python.",
  heroNote: "Checkpoints download from Hugging Face on first run (SAM 3 is gated and needs an HF_TOKEN).",
  demo: null as { href: string; label: string } | null,
  cargo: `cargo install sam-rs                          # CPU only (bf16 kernels where the CPU has them)
cargo install sam-rs --features cuda          # CUDA (f16)`,
  terminal: {
    title: "sam-rs · truck.jpg 1800x1200, RTX 4090",
    command: "sam-rs --model sam2.1-hiera-tiny --image truck.jpg \\\n  --point 500,375 --out mask.png",
    outputLang: "text",
    output: `object score 19.36 (object present)
mask 0: iou 0.042, area 631329
mask 1: iou 0.952, area 22109
mask 2: iou 0.493, area 46080`,
  },
  cardsTitle: "Five model families, one interface",
  cardsLead: "Pick a model with --model; the prompts, the output and the library calls stay the same shape.",
  cards: [
    {
      name: "sam-rs",
      href: "/docs/cli",
      tagline: "Image in, masks out",
      body: "One binary: load a model, prompt it with points, boxes or text, print the candidate masks with their predicted IoU and save the best one as a PNG. Point it at a folder of frames to track an object through a video.",
    },
    {
      name: "Models & prompts",
      href: "/docs/models",
      tagline: "SAM · MobileSAM · EfficientSAM · SAM 2 · SAM 3",
      body: "What each family accepts (points, boxes, text, exemplar boxes, video), which checkpoints it uses, how big they are and how fast each one runs.",
    },
  ] as Card[],
  featuresTitle: "What it does",
  featuresLead: "From-scratch candle ports, each checked against the reference implementation.",
  features: [
    {
      title: "SAM, MobileSAM, EfficientSAM",
      body: "ViT-B/L/H, the TinyViT-based MobileSAM and EfficientSAM ViT-Ti/S, with point and box prompts and three candidate masks.",
    },
    {
      title: "SAM 2 and 2.1",
      body: "All four Hiera sizes of both generations, matching PyTorch to about 1e-6, plus video tracking with binary mask IoU 1.0000 against the reference.",
    },
    {
      title: "SAM 3 concepts",
      body: "Type \"truck\" or \"wheel\" and get every instance with a score, a box and a mask. Exemplar boxes (positive and negative) work too.",
    },
    {
      title: "Text for every SAM",
      body: "Grounding DINO turns phrases into boxes and any SAM, MobileSAM, EfficientSAM or SAM 2 segments inside them: Grounded-SAM, from one binary, checked against PyTorch.",
    },
    {
      title: "Video tracking",
      body: "Prompt one frame with points, a box or a text concept; SAM 3 follows every matching object through the rest of the clip and picks up new ones as they appear.",
    },
    {
      title: "Fast on CUDA",
      body: "Fused kernels and optional FlashAttention-2: SAM ViT-B in 45 ms end to end on an RTX 4090, about 11x faster than the int8 ONNX route.",
    },
    {
      title: "Fast on CPU",
      body: "bf16 kernels on AVX-512 BF16 CPUs: ViT-B in 0.8 s, SAM 2 tiny in 0.46 s, EfficientSAM-Ti in 0.3 s, all with almost no loss against f32.",
    },
    {
      title: "Loads what exists",
      body: "Meta and Hugging Face transformers checkpoint layouts, safetensors, and the PyTorch .pt files EfficientSAM ships, converted once on download without running pickled code.",
    },
    {
      title: "Library and CLI",
      body: "Embed an image once, then decode as many prompts as you like. The binary is a thin wrapper over the same API.",
    },
    {
      title: "No Python",
      body: "Tokenizer, preprocessing, downloads and every model are Rust. One static binary per platform, or cargo install.",
    },
  ] as Feature[],
  lib: {
    lead: "Embed the image once, then decode as many prompts as you like. Every model family has its own module with the same shape; CUDA is an opt-in feature, passed down to candle.",
    add: `cargo add sam-rs`,
    caption: "src/main.rs",
    snippet: `use candle_core::{DType, Device};
use sam_rs::{Prompt, Sam, SamVariant};

// Downloads the checkpoint into ~/.cache/sam-rs on first use.
let device = Device::cuda_if_available(0)?;              // Cpu unless you enable the \`cuda\` feature
let sam = Sam::from_pretrained(SamVariant::VitB, &device, DType::F16)?;

let image = image::open("truck.jpg")?.to_rgb8();
let embedding = sam.embed(&image)?;                      // the expensive part, once per image
let masks = sam.predict(&embedding, &Prompt::Points(vec![(500.0, 375.0, true)]), true)?;
for m in &masks {
    println!("iou {:.3}, {}x{}", m.iou, m.width, m.height);
}`,
  },
};
