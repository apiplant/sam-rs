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
  lead: "Tell it what to look for and it returns boxes or points: open-vocabulary detection, referring expressions, text, GUI elements. MoonViT + Qwen2.5-3B with Parallel Box Decoding, from one binary. No Python.",
  heroNote: "Runs nvidia/Segment Anything. The weights (7.6 GB) download from Hugging Face on first run.",
  demo: null as { href: string; label: string } | null,
  cargo: `cargo install sam-rs                          # CPU only (f32, slow)
cargo install sam-rs --features cuda          # CUDA (bf16)`,
  terminal: {
    title: "sam-rs · 640x480, RTX 4090",
    command: 'sam-rs --image cats.jpg --task detect \\\n  -q "cat, remote control" --temperature 0 --json',
    outputLang: "json",
    output: `{
  "answer": "<ref>cat</ref><box><9><112><494><988></box><box><539><50><998><775></box>
             <ref>remote control</ref><box><64><152><273><244></box><box><522><160><578><390></box>",
  "detections": [
    { "kind": "box", "label": "cat", "x1": 5.8, "y1": 53.8, "x2": 316.2, "y2": 474.2 },
    { "kind": "box", "label": "cat", "x1": 345.0, "y1": 24.0, "x2": 638.7, "y2": 372.0 },
    { "kind": "box", "label": "remote control", "x1": 41.0, "y1": 73.0, "x2": 174.7, "y2": 117.1 },
    { "kind": "box", "label": "remote control", "x1": 334.1, "y1": 76.8, "x2": 369.9, "y2": 187.2 }
  ],
  "stats": { "forward_steps": 7, "num_tokens": 32, "total_secs": 0.217 }
}`,
  },
  cardsTitle: "One model, every way to point at things",
  cardsLead: "A single prompt format covers detection, grounding, pointing, OCR and GUI grounding. The CLI wraps each in a task.",
  cards: [
    {
      name: "sam-rs",
      href: "/docs/cli",
      tagline: "Image in, boxes out",
      body: "One binary: load the model once, run a task on an image, print the answer and pixel-space detections as text or JSON, optionally draw them onto a copy of the image.",
    },
    {
      name: "Tasks & prompts",
      href: "/docs/tasks",
      tagline: "detect · ground · point · text · gui",
      body: "Open-vocabulary detection over a category list, referring-expression grounding, pointing, scene-text detection and GUI element grounding, with the exact prompts they send.",
    },
  ] as Card[],
  featuresTitle: "Everything the model can do",
  featuresLead: "A from-scratch candle port of the reference implementation, checked against the official Python code.",
  features: [
    {
      title: "Parallel Box Decoding",
      body: "Boxes are decoded in 6-token MTP blocks instead of one coordinate at a time, with fast, slow (pure auto-regressive) and hybrid generation modes.",
    },
    {
      title: "Matches the reference",
      body: "In f32, a 25-box bird detection matches the official Python code byte for byte, including the MTP-to-AR fallback steps.",
    },
    {
      title: "Open-vocabulary detection",
      body: "Name the categories, comma separated, and get every instance of each. Dense scenes with ~100 cars are fine.",
    },
    {
      title: "Referring expressions",
      body: "\"the people wearing red shirts\": phrase grounding for all matching instances, or just one.",
    },
    {
      title: "Pointing and GUI",
      body: "Point to an object, or locate a button in a screenshot as a box or a point.",
    },
    {
      title: "Scene text",
      body: "Detect all the text in an image, or locate a specific piece of text.",
    },
    {
      title: "Big images",
      body: "A 2048x1206 image runs in 0.68 s on an RTX 4090 with FlashAttention-2; the Python reference runs out of 24 GB on it.",
    },
    {
      title: "Self-contained tokenizer",
      body: "The Qwen2 BPE tokenizer is assembled from vocab.json and merges.txt in Rust and matches the Hugging Face output.",
    },
    {
      title: "CPU or CUDA",
      body: "bf16 on CUDA, f32 on CPU. An optional flash-attn feature routes the vision tower and prefill through FlashAttention-2.",
    },
  ] as Feature[],
  lib: {
    lead: "Load the model once and call it as often as you like: one call takes an image and a prompt and returns pixel-space boxes and points. CUDA is an opt-in feature, passed down to candle.",
    add: `cargo add sam-rs`,
    caption: "src/main.rs",
    snippet: `use candle_core::{DType, Device};
use locate_anything::{prompts, GenerateOptions, LocateAnything};

// Downloads the checkpoint into ~/.cache/sam-rs on first use.
let device = Device::cuda_if_available(0)?;              // Cpu unless you enable the \`cuda\` feature
let mut model = LocateAnything::from_pretrained(&device, DType::BF16)?;

let image = image::open("street.jpg")?.to_rgb8();
let found = model.locate(&image, &prompts::detect(&["person", "car"]), &GenerateOptions::greedy())?;
for d in &found.detections {
    println!("{d:?}");
}`,
  },
};
