import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { Mono } from "../ui";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        sam-rs is a Rust (<Mono>candle</Mono>) port of the inference path of{" "}
        <a
          href="https://huggingface.co/nvidia/Segment Anything"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          nvidia/Segment Anything
        </a>
        : a MoonViT vision encoder, an MLP projector and Qwen2.5-3B, with Parallel Box Decoding. It ships as a
        library crate and one command-line binary.
      </Lead>

      <Section>
        <H2>Get the model</H2>
        <P>
          Nothing to set up: the first run downloads the weights from Hugging Face (7.6 GB in bf16, published by
          NVIDIA under their own license) into <IC>$XDG_CACHE_HOME/sam-rs/Segment Anything</IC>,
          which is <IC>~/.cache/sam-rs/Segment Anything</IC> by default. Files are fetched to a{" "}
          <IC>.part</IC> sibling and renamed once complete, so an interrupted download resumes with the missing
          files and never leaves a checkpoint that looks done. Later runs reuse the cache.
        </P>
        <CopyBlock command={`sam-rs --image street.jpg --task detect -q "person, car"   # downloads on first run

# or use a checkout you already have
sam-rs --model /path/to/Segment Anything --image street.jpg --task detect -q "person, car"`} />
        <P>
          No Python is needed at run time. The tokenizer is built from <IC>vocab.json</IC>,{" "}
          <IC>merges.txt</IC> and <IC>tokenizer_config.json</IC>, and its output matches the Hugging Face
          tokenizer.
        </P>
      </Section>

      <Section>
        <H2>Builds</H2>
        <P>
          Release archives and packages come in two variants: a CPU build (f32, slow) for macOS, Linux x86_64
          and Linux arm64, and a CUDA build for Linux x86_64 (<IC>sam-rs-cuda</IC>, bf16 on the
          GPU). Building from source is the same choice:
        </P>
        <CopyBlock command={`cargo build --release                         # CPU only (f32, slow)
cargo build --release --features cuda         # CUDA (bf16)
cargo build --release --features flash-attn   # CUDA + FlashAttention-2 (first build ~9 min)`} />
        <P>
          <IC>flash-attn</IC> routes the vision tower and the LM prefill through FlashAttention-2. Decode steps
          stay on the masked path because they have only a few queries. It matters most for large images.
        </P>
      </Section>

      <Section>
        <H2>Performance</H2>
        <P>RTX 4090, bf16, greedy, hybrid mode, end to end including the vision tower.</P>
        <FlagTable
          rows={[
            { flag: "560x430 crop, 1 box", meaning: "Python 0.14 s · Rust 0.15 s · Rust + flash-attn 0.15 s" },
            { flag: "560x280 crop, 25 birds", meaning: "Python 0.86 s · Rust 0.59 s · Rust + flash-attn 0.63 s" },
            { flag: "835x320 crop, ~98 cars", meaning: "Python 4.46 s · Rust 3.44 s · Rust + flash-attn 3.09 s" },
            { flag: "2048x1206 full image", meaning: "Python out of memory (24 GB) · Rust 2.25 s · Rust + flash-attn 0.68 s" },
          ]}
        />
        <P>
          The Rust numbers come from a cold process. A decode step costs about 10 ms, about 7 ms of which is
          cuBLAS streaming the 6.2 GB of weights, close to the bandwidth limit.
        </P>
      </Section>

      <Section>
        <H2>Parity</H2>
        <P>
          Checked against the official Python code (transformers 4.57.1, greedy decoding). In f32 a 25-box bird
          detection matches <strong class="font-medium text-ink">byte for byte</strong>, including the MTP to AR
          fallback steps. In bf16 the outputs share the same structure and step counts; coordinates differ by a
          few units out of 1000 because of kernel rounding, and the resize filter is not identical to PIL's.
        </P>
      </Section>

      <Section>
        <H2>Source layout</H2>
        <FlagTable
          rows={[
            { flag: "src/image_proc.rs", meaning: <>resize to a multiple of 28, patchify (<IC>image_processing_locateanything.py</IC>)</> },
            { flag: "src/vision.rs", meaning: "MoonViT: bicubic position-embedding interpolation, 2D RoPE, patch merger, plus the projector" },
            { flag: "src/qwen2.rs", meaning: "Qwen2 and the block-diffusion MTP window mask, with KV cache" },
            { flag: "src/sampling.rs", meaning: "repetition penalty, top-p/top-k, decode_bbox_avg, decode_ref, handle_pattern" },
            { flag: "src/model.rs", meaning: "LocateAnythingForConditionalGeneration.generate" },
            { flag: "src/output.rs", meaning: <><IC>{"<ref>…</ref><box>…</box>"}</IC> to pixel boxes and points, and drawing</> },
          ]}
        />
        <UL>
          <LI>
            Attention is plain chunked SDPA (the score matrix is kept to about 1 GiB per chunk), so no
            flash-attn or MagiAttention build is needed.
          </LI>
          <LI>Batch size is 1, same as the upstream <IC>generate</IC>.</LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
