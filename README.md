# sam-rs

Rust ([candle](https://github.com/huggingface/candle)) inference for the Segment Anything family on
**CPU and CUDA**, with fused CUDA kernels. A library plus one binary, no Python at runtime. Inspired by
[osam](https://github.com/wkentaro/osam).

| Model | Status |
| --- | --- |
| SAM ViT-B / ViT-L / ViT-H (`sam-vit-b`, `sam-vit-l`, `sam-vit-h`) | working, CPU + CUDA |
| **SAM 2 / 2.1** (Hiera tiny, small, base+, large): points and boxes, video tracking | working, CPU + CUDA |
| **SAM 3** (`sam3`): text-prompted instance segmentation, video tracking (points, boxes or text) | working, CPU + CUDA (gated weights, see below) |
| **MobileSAM** (`mobile-sam`) | working, CPU + CUDA |
| **EfficientSAM** (`efficientsam-ti`, `efficientsam-s`) | working, CPU + CUDA |

CUDA encoder time on an RTX 4090, f16, same image: ViT-B 38 ms, ViT-L 74 ms, ViT-H 96 ms (decode adds ~7 ms). ViT-B
checkpoints load in either the Meta layout (`lmz/candle-sam`) or the Hugging Face `transformers` layout
(`facebook/sam-vit-*`); ViT-L and ViT-H use the latter. Candidate masks are not in a fixed semantic order across
checkpoints (index 0 is the whole object for ViT-B but a part for ViT-H), so pick by the predicted IoU or
inspect all three.

## MobileSAM and EfficientSAM

```sh
sam-rs --model mobile-sam --image truck.jpg --point 500,375
sam-rs --model efficientsam-ti --image truck.jpg --point 500,375
```

MobileSAM (TinyViT-5M encoder + SAM ViT-B decoder) matches candle-transformers' reference encoder to 1.4e-6 (using the
tanh GELU that reference uses; the model itself uses the exact GELU, which is the default here). About 18 ms on the
4090 and ~0.5 s on CPU. Depthwise convolutions run on a dedicated kernel because candle's grouped convolution launches
per-channel kernels.

EfficientSAM was ported against the ONNX exports run through onnxruntime (`scripts/ref_efficientsam.py`,
`examples/efficientsam_parity.rs`): ViT-Ti matches on every prompt type, with low-res masks to 1.6e-5 and IoU scores to six
digits. ViT-S loads and runs but has no reference check. The upstream checkpoints are PyTorch `.pt` files; `sam-rs`
downloads them from `yunyangx/EfficientSAM` on first use and converts them once to a `.safetensors` file next to
the `.pt` (f32, using candle's pickle reader, which only builds tensors and never executes pickled code), so like the
other models it needs no manual step. `scripts/convert_efficientsam.py` still does the same conversion in Python.

ViT-Ti: ~21 ms encoder + ~7 ms decode on CUDA f16; ~0.29 s encoder on CPU (the bf16 kernels where the CPU has them;
about 1.3 s in exact f32). The prompt is always padded to exactly 6 points, as in the reference graph.

## SAM 2 / 2.1

Point and box prompts for one object per image, and video tracking, from the four Hiera sizes of both generations
(`sam2.1-hiera-{tiny,small,base-plus,large}` and `sam2-hiera-*`; `sam2-tiny` etc. mean the 2.1 weights). The
checkpoints are the public Hugging Face ones (about 160 MB to 900 MB) and download on first use.

```sh
sam-rs --model sam2.1-hiera-tiny --image truck.jpg --point 500,375 --out mask.png
sam-rs --model sam2.1-hiera-large --image truck.jpg --box 86,282,1710,851 --single
sam-rs --model sam2.1-hiera-tiny --video frames/ --point 500,375 --out masks/     # track through a clip
```

Every size matches the PyTorch reference (`examples/sam2_parity.rs`, `scripts/ref_sam2.py`): the Hiera + neck
features to ~3e-6 relative error, masks to ~1e-6, and the IoU / object scores to six digits. The video tracker
(`examples/sam2_video_parity.rs`, `scripts/ref_sam2_video.py`) agrees on a 6-frame clip with binary mask IoU 1.0000
on every frame and mask logits to ~4e-6.

`truck.jpg`, one point prompt, three masks returned. RTX 4090 / Ryzen 9 7950X, median of repeated runs:

| | encoder | decode | total |
| --- | ---: | ---: | ---: |
| osam 0.9.2, CUDA provider (ONNX), `sam2:tiny` | 39 ms | 8 ms | 47 ms |
| **sam-rs CUDA f16 + FlashAttention, 2.1 tiny** | 14 ms | 6.5 ms | **~21 ms** |
| sam-rs CUDA f16 + FlashAttention, 2.1 small / base+ / large | 16 / 24 / 44 ms | 6.5 ms | ~23 / 30 / 50 ms |
| sam-rs CUDA f16, plain attention, 2.1 tiny / large | 23 / 64 ms | 6.5 ms | ~29 / 70 ms |
| osam 0.9.2, CPU provider (ONNX), `sam2:tiny` | 678 ms | 27 ms | 705 ms |
| sam-rs CPU (bf16 kernels), 2.1 tiny | 421 ms | 42 ms | ~463 ms |
| sam-rs CPU (bf16 kernels), 2.1 large | 1800 ms | 42 ms | ~1.85 s |

On CPU the tiny model is about 1.5x faster than osam. The layout changes around the Hiera blocks (QKV split, window
pooling, NCHW/NHWC conversions) were single-threaded candle copies and now run as parallel kernels
(`tests/cpu_layout.rs` checks them against the tensor ops); the large model is dominated by the GEMMs. osam's larger SAM 2 models were not benchmarked (their download had not finished). Video: about 42-50 ms
per frame for 2.1 tiny on CUDA f16 including PNG decoding.

## SAM 3: segment every instance of a concept

SAM 3 takes a short text prompt and returns every matching instance with a score, a box and a mask.

```sh
sam-rs --model sam3 --text "truck" --image truck.jpg --out mask.png      # best instance's mask
sam-rs --model sam3 --text "wheel" --image truck.jpg --threshold 0.5
```

The checkpoint (`facebook/sam3`, 3.4 GB) is gated; see [HF_TOKEN](#gated-models-and-hf_token-sam-3). The port covers
the image path of `Sam3Model` (ViT with 2-D RoPE + FPN, CLIP text encoder, fusion encoder, DETR decoder with presence
token and box-RPB attention, mask head), the geometry (box) prompt encoder, and the image tracker. Every stage matches
the PyTorch reference to about 1e-6 relative error (`examples/sam3_parity.rs` and `examples/sam3_tracker_parity.rs`,
reference activations from `scripts/ref_sam3.py` and `scripts/ref_sam3_tracker.py`), and the detection score on the
test image (0.932) equals the reference's.

Three ways to prompt it:

```sh
# 1. text: every instance of a concept
sam-rs --model sam3 --text "wheel" --image truck.jpg --threshold 0.5

# 2. exemplar boxes ("find more like this"), alone or combined with text; --neg-box marks what to exclude
sam-rs --model sam3 --box 86,282,1710,851 --image truck.jpg                    # text defaults to "visual"
sam-rs --model sam3 --text wheel --box 443,626,676,852 --image truck.jpg
sam-rs --model sam3 --text wheel --neg-box 443,626,676,852 --image truck.jpg

# 3. SAM 1/2-style points or a box for ONE object (the tracker); prints candidates, an object-presence score
sam-rs --model sam3 --point 500,375 --image truck.jpg --out mask.png
sam-rs --model sam3 --tracker --box 86,282,1710,851 --image truck.jpg
```

### SAM 3 video tracking

Click on (or box) an object in one frame and follow it through the rest of the clip. Frames are read from a directory
of images, sorted by name; extract them from a video with, for example, `ffmpeg -i clip.mp4 frames/%05d.png`.

```sh
sam-rs --model sam3 --video frames/ --point 500,375 --out masks/           # prompt on frame 0
sam-rs --model sam3 --video frames/ --box 86,282,1710,851 --prompt-frame 12 --out masks/
```

It prints the object score and mask area of every frame and writes `<frame>_mask.png` per frame. A positive object
score means the object is present. In code: `Session::new`, `add_prompt`, then `Sam3::track_frame` per frame (see
`examples/sam3_video_parity.rs`); the memory is a handful of `5184 x 64` tensors per frame, kept in bf16.

The tracker is a port of `Sam3TrackerVideoModel` (memory encoder, 4-layer memory attention with 2-D RoPE, object
pointers, the reference's conditioning-frame and memory-offset selection). Against PyTorch on a synthetic 6-frame clip
the per-frame masks agree at binary IoU 1.0000 on every frame, mask logits to ~1e-4 relative, and object scores to three
decimals. Memory attention matches to 5e-6 when both implementations are given the same memory; the remaining ~3e-4
in the conditioned features is bf16 rounding of the stored memories (PyTorch keeps them in bf16 too, so a ~1e-6
upstream difference occasionally flips an entry by a whole bf16 step). Speed, RTX 4090, f16 + FlashAttention: about
85-100 ms per frame including PNG decoding; CPU bf16: 3-4 s per frame.

#### Text-prompted detect-and-track

Give `--text` instead of a point or box and every instance of the concept is found and followed:

```sh
sam-rs --model sam3 --video frames/ --text "wheel" --detect-every 5 --out masks/
```

The detector runs on the first frame and then every `--detect-every` frames; each detection that no live track
explains (mask IoU < 0.5) starts a tracker session seeded with its box, so objects that enter later are picked up.
A track absent for 8 frames in a row is dropped. One `<frame>_obj<ID>_mask.png` is written per object and frame. In
code: `sam3::text_track::TextTracker::new(&sam, "wheel", 0.3).detect_every(5)` and `step(&frame)` per frame.
The association is a greedy mask-IoU rule, not `Sam3VideoModel`'s matching logic, and it has not been compared with the
reference, so identities can differ from it in crowded or heavily occluded scenes (on a 10-frame pan over `truck.jpg`
it finds three wheels on frame 0, picks up a fourth when it enters, and keeps the ids stable). About 130-350 ms per
frame on CUDA f16 (detector frames cost more), 3-4 s on CPU.

Limits (SAM 3 and SAM 2 video): one object per point/box session (run several sessions for several objects); forward
tracking only (no reverse or re-conditioning pass); mask prompts are not ported. Frames are squashed to 1008x1008 like
the image model.

For the best CUDA speed build with `--features flash-attn` (first build takes several minutes).

## Server and web UI

```sh
sam-rs serve                                  # http://127.0.0.1:8080: web UI at /, API reference at /docs
sam-rs serve --addr 0.0.0.0:9000 --preload sam2.1-hiera-tiny --max-models 3
```

One HTTP API (ntex) in front of every model, and a web UI (Solid 2, in `ui/`, embedded in the binary) to use it:
pick a model (downloaded ones first, with a toggle for the rest), open or paste an image, click for points
(right-click or alt for background), drag for boxes (shift-drag excludes, SAM 3), type a concept for SAM 3, and
drop a clip or a set of frames in Video mode to track objects through it. `--no-download` limits the server to
models already in the cache, and "Load local weights" loads a `.safetensors` path on the server.

The models live on one worker thread (the GPU is single-tenant), so requests queue; models load on first use and the
least recently used is dropped past `--max-models` (default 2). The API embeds an image once and decodes prompts against
that session, which is what makes interactive clicking fast (SAM ViT-B on an RTX 4090: ~45 ms to embed, ~25 ms per
prompt including PNG encoding):

```sh
curl -s localhost:8080/v1/models                                             # catalog: local / loaded / gated, capabilities
id=$(curl -s -X POST --data-binary @truck.jpg 'localhost:8080/v1/images?model=sam2.1-hiera-tiny' | jq .id)
curl -s -X POST -d '{"points":[{"x":500,"y":375}]}' localhost:8080/v1/images/$id/segment   # masks as base64 PNGs
curl -s -X POST -d '{"text":"wheel","threshold":0.5}' localhost:8080/v1/images/$(curl -s -X POST --data-binary @truck.jpg 'localhost:8080/v1/images?model=sam3' | jq .id)/segment
v=$(curl -s -X POST -d '{"model":"sam3","text":"wheel"}' localhost:8080/v1/videos | jq .id)       # video tracking session
curl -s -X POST --data-binary @frame00.jpg localhost:8080/v1/videos/$v/frames                   # then each frame in order
```

`GET /openapi.json` is the full description. The UI is built with `pnpm install && pnpm build` in `ui/` (`SAM_RS_REQUIRE_UI=1`
makes `cargo build` fail without it; release builds set it) and embedded by `build.rs`; `pnpm dev` there proxies to a
running server. Build without the server and its dependencies with `--no-default-features`.

## Build

```sh
cargo build --release                  # CPU only (f32)
cargo build --release --features cuda  # CPU + CUDA (f16) with the fused kernels
```

The `cuda` feature compiles `kernels/*.cu` to PTX with `nvcc` in `build.rs`. Set `CUDA_COMPUTE_CAP` (for example
`80`) to pick the target architecture; otherwise it asks `nvidia-smi`, and falls back to `80`. PTX is JIT-compiled
by the driver for newer GPUs.

## Usage

```sh
# one foreground point, three candidate masks, best one saved as a PNG
sam-rs --image truck.jpg --point 500,375 --out mask.png

# several points (a trailing 0 marks a background point) or a box
sam-rs --image truck.jpg --point 500,375 --point 1250,600,0
sam-rs --image truck.jpg --box 75,275,1725,850 --single

# pick a specific candidate (0-2), another model, or exact f32 on CPU
sam-rs --image truck.jpg --point 500,375 --model sam-vit-h --index 0 --out mask.png
sam-rs --image truck.jpg --point 500,375 --cpu --f32

# time the encoder and the decoder
sam-rs --image truck.jpg --point 500,375 --bench 20
```

CUDA is used when the binary was built with `--features cuda` and a GPU is present; `--cpu` forces the CPU.

## Performance

SAM ViT-B, `truck.jpg` (1800x1200), one point prompt, three masks returned, RTX 4090 + Ryzen 9 7950X (16 cores).
Median of repeated runs after warm-up; "embed + decode" is everything from the decoded image to the masks at the
original resolution.

| | embed + decode | encoder | decode + mask upsample |
| --- | ---: | ---: | ---: |
| osam 0.9.2, CUDA provider (onnxruntime-gpu, int8 ONNX) | 495 ms | 482 ms | 13 ms |
| **sam-rs, CUDA (f16, fused kernels)** | **45 ms** | 38 ms | 7 ms |
| osam 0.9.2, CPU provider (int8 ONNX) | 1414 ms | 1387 ms | 27 ms |
| **sam-rs, CPU (bf16 kernels, default on AVX-512 BF16 CPUs)** | **793 ms** | 761 ms | 32 ms |
| sam-rs, CPU (`--f32`, exact) | 1370 ms | 1338 ms | 32 ms |

What the table does and does not say:

* **CUDA: about 11x faster than osam** on this setup. osam runs the int8-quantized ONNX export under onnxruntime's
  CUDA provider, which logs that it inserted 200 and 143 `Memcpy` nodes into the encoder and decoder graphs
  ("might have negative impact on performance"). I did not profile osam further, so I can't say how much of the gap
  that explains; an fp16 export would likely narrow it.
* **CPU: about 1.8x faster than osam** with the default bf16 kernels (`vdpbf16ps`, f32 accumulation, ~2.9 TFLOP/s
  on this chip against ~1 TFLOP/s for f32 GEMM). Weights are packed to bf16 once at load. The bf16 masks match the
  f32 reference at IoU 0.99995 on this image, so unlike osam's int8 model the speedup costs almost no fidelity. On
  CPUs without AVX-512 BF16 the f32 path runs, which is about on par with osam (1.37 s vs 1.41 s here); `--f32`
  forces it. The 32 ms decode returns three full-resolution masks; a single mask (`--single`) takes ~25 ms against
  osam's 27 ms for its one mask. int8 (VNNI) kernels are not written; bf16 already beat the target without int8's
  accuracy cost.
* **Threads:** candle sizes its GEMM pool from the *physical* core count unless `RAYON_NUM_THREADS` is set, which
  made the decoder ~2x slower on SMT CPUs. The binary calls `sam_rs::cpu_gemm::init_threads()` to use all logical
  cores; library users should call it once at startup (or set the variable) for the same effect.
* Not measured: PyTorch SAM, the reference most people mean by "the Python version" (torch does not import on the
  benchmark machine). Only osam was compared.
* Accuracy: CUDA f16 and CPU f32 masks agree at IoU 0.9985 on this image. `tests/kernels.rs` checks every fused
  kernel against its tensor-op fallback.

How the CUDA number got there (encoder only, same image): straightforward port 93 ms, fused relative-position
softmax 62 ms, rel-pos folded into the kernel plus fused QKV split / bias+residual / bias+GELU 55 ms, GPU
preprocessing 38 ms. On CPU the encoder went from 3.9 s to 1.45 s with a tiled attention that keeps scores in
cache, a parallel GELU, fused QKV split, and finally the bf16 GEMM kernels for the linears and both attention
GEMMs plus an AVX-512 `exp`.

### SAM 3 timings

`truck.jpg`, prompt `"truck"`, RTX 4090 / Ryzen 9 7950X. The model returns the same single instance everywhere
(score 0.932; masks of the f16, CPU-bf16 and f32 runs agree at IoU > 0.9999).

| | image encoder | text + head + masks | total |
| --- | ---: | ---: | ---: |
| PyTorch + transformers 5.22, CPU, f32, 16 threads | 5040 ms | 650 ms | ~5.7 s |
| **sam-rs CPU (bf16 kernels, incl. bf16 convolutions)** | 2680 ms | 820 ms | **~3.5 s** |
| sam-rs CUDA f16, plain attention | 143 ms | 50 ms | ~190 ms |
| **sam-rs CUDA f16 + FlashAttention** | 63 ms | 28 ms | **~90 ms** |

osam 0.9.2's SAM 3 (four ONNX graphs, text prompt `"truck"`) takes 218 ms end to end on CUDA (172 ms image encoder, 46 ms
text + decode) and 6.86 s on CPU (5.97 s image encoder, 0.90 s text + decode), on the same machine. So the CUDA +
FlashAttention row above is about 2.4x faster and the bf16 CPU row about 2x faster. Not measured: PyTorch on the GPU
(only CPU torch is installed here).

## Models and downloads

Checkpoints live in `~/.cache/sam-rs/<hf-repo>/` (`$XDG_CACHE_HOME/sam-rs/...`, one directory per Hugging Face
repository with `/` written as `--`). The first run downloads what it needs from Hugging Face and reuses it
afterwards. A file is fetched to a `.part` sibling and renamed once complete, so an interrupted download never
looks finished.

```
~/.cache/sam-rs/
├── lmz--candle-sam/sam_vit_b_01ec64.safetensors      # sam-vit-b (375 MB)
├── facebook--sam2.1-hiera-large/model.safetensors
└── facebook--sam3/model.safetensors                  # gated, see below
```

If you already have a checkpoint, skip the download by putting it at that path (or symlinking it), or pass the
file directly with `--weights FILE`.

### Gated models and `HF_TOKEN` (SAM 3)

Most checkpoints are public. **SAM 3 and SAM 3.1 (`facebook/sam3`, `facebook/sam3.1`) are gated**: Hugging Face
refuses the download until you have accepted Meta's terms and the repository authors have approved your account.
Without that you get `HTTP 403` and a hint from `sam-rs`.

1. Sign in at huggingface.co and open <https://huggingface.co/facebook/sam3>. Click **Request access** and accept
   the terms. Do the same on <https://huggingface.co/facebook/sam3.1> if you want 3.1. Approval is manual and can
   take a while; the page shows "awaiting a review" until it is granted.
2. Create an access token at <https://huggingface.co/settings/tokens>. A **read** token works. For a
   fine-grained token, tick **Read access to contents of all public gated repos you can access**.
3. Give the token to `sam-rs` in either of these ways:

   ```sh
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx     # or HUGGING_FACE_HUB_TOKEN
   # or log in once; sam-rs reads the token file that the Hugging Face CLI writes
   hf auth login                                    # older CLIs: huggingface-cli login
   ```

   Resolution order: `$HF_TOKEN`, `$HUGGING_FACE_HUB_TOKEN`, then `$HF_HOME/token` (default
   `~/.cache/huggingface/token`).

The token is sent only to `huggingface.co` (the HTTP client drops the header when Hugging Face redirects to its
CDN), is never written to disk by `sam-rs`, and is only attached when set, so public models are unaffected.
Once a gated checkpoint is in the cache you no longer need the token to run it.

## Library

```rust
use candle_core::{DType, Device};
use sam_rs::{Prompt, Sam, SamVariant};

let device = Device::cuda_if_available(0)?;
let sam = Sam::from_pretrained(SamVariant::VitB, &device, DType::F16)?; // downloads on first use
let image = image::open("truck.jpg")?.to_rgb8();

let embedding = sam.embed(&image)?;                       // the expensive part, once per image
let masks = sam.predict(&embedding, &Prompt::Points(vec![(500.0, 375.0, true)]), true)?;
for m in &masks {
    println!("iou {:.3}, {}x{}", m.iou, m.width, m.height);
}
# anyhow::Ok(())
```

## Fused kernels

Profiling the straightforward port showed the GPU spending 61% of its time in `badd` (the broadcast adds that
apply SAM's decomposed relative-position bias to the attention scores), not in matrix multiplies. The CUDA build
therefore replaces whole op chains with single kernels (`kernels/sam.cu`):

* **relpos softmax**: computes both relative-position bias vectors from `q` and the tables in shared memory and
  applies them inside the softmax, so the 5-D broadcast adds, two small matmuls and their permutes disappear.
* **qkv split**: adds the QKV bias and writes `q`, `kᵀ`, `v` directly in attention layout.
* **bias + residual** and **bias + GELU**: one pass each instead of two or three.
* **preprocess**: antialiased resize, normalization and padding on the GPU instead of the CPU.

Each op has a plain-tensor fallback (used for CPU, or when a shape is outside a kernel's range). The CPU build
has its own parallel Rust versions of the same fusions, including a tiled attention that never materializes the
`N x N` score matrix.
