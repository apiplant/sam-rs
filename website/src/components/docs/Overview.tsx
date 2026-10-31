import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { Mono } from "../ui";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        sam-rs is a Rust (<Mono>candle</Mono>) implementation of the Segment Anything family: SAM, MobileSAM,
        EfficientSAM, SAM 2 / 2.1 and SAM 3, on CPU and CUDA. It ships as a library crate and one command-line
        binary, and needs no Python at run time. It was inspired by{" "}
        <a
          href="https://github.com/wkentaro/osam"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          osam
        </a>
        .
      </Lead>

      <Section>
        <H2>Get the models</H2>
        <P>
          Nothing to set up: the first run downloads the checkpoint it needs from Hugging Face into{" "}
          <IC>$XDG_CACHE_HOME/sam-rs/&lt;repo&gt;</IC>, which is <IC>~/.cache/sam-rs/&lt;repo&gt;</IC> by default
          (one directory per repository, with <IC>/</IC> written as <IC>--</IC>). A file is fetched to a{" "}
          <IC>.part</IC> sibling and renamed once complete, so an interrupted download never looks finished.
          Put a checkpoint at that path, or pass <IC>--weights FILE</IC>, to skip the download.
        </P>
        <CopyBlock command={`sam-rs --image truck.jpg --point 500,375            # downloads sam-vit-b (375 MB) on first run`} />
        <P>
          <strong class="font-medium text-ink">SAM 3 is gated.</strong> Accept Meta&apos;s terms on{" "}
          <IC>huggingface.co/facebook/sam3</IC>, wait for approval, create a read token and export it. The token is
          sent only to huggingface.co and is not needed once the checkpoint is cached.
        </P>
        <CopyBlock command={`export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx     # or: hf auth login
sam-rs --model sam3 --text "wheel" --image truck.jpg`} />
      </Section>

      <Section>
        <H2>Builds</H2>
        <P>
          Release archives and packages come in two variants: a CPU build for macOS, Linux x86_64 and Linux
          arm64, and a CUDA build for Linux x86_64 (<IC>sam-rs-cuda</IC>, f16 on the GPU). Building from source
          is the same choice:
        </P>
        <CopyBlock command={`cargo build --release                         # CPU only
cargo build --release --features cuda         # CUDA (f16) with the fused kernels
cargo build --release --features flash-attn   # CUDA + FlashAttention-2 (long first build)`} />
        <P>
          The <IC>cuda</IC> feature compiles <IC>kernels/*.cu</IC> to PTX with <IC>nvcc</IC>; set{" "}
          <IC>CUDA_COMPUTE_CAP</IC> (for example <IC>80</IC>) to pick the architecture. On CPUs with AVX-512 BF16
          the linear layers, convolutions and attention run on bf16 kernels with f32 accumulation (about 3x the
          f32 GEMM throughput, masks at IoU 0.99995 against f32); <IC>--f32</IC> forces exact f32, and CPUs
          without the extension use it automatically.
        </P>
      </Section>

      <Section>
        <H2>Performance</H2>
        <P>
          <Mono>truck.jpg</Mono> (1800x1200), one point prompt, three masks, RTX 4090 and a Ryzen 9 7950X. Median of
          repeated runs after warm-up, against osam 0.9.2 (ONNX) where it was measured.
        </P>
        <FlagTable
          rows={[
            { flag: "SAM ViT-B, CUDA", meaning: "osam 495 ms · sam-rs f16 45 ms (encoder 38 ms, decode 7 ms)" },
            { flag: "SAM ViT-B, CPU", meaning: "osam 1414 ms · sam-rs bf16 793 ms · sam-rs --f32 1370 ms" },
            { flag: "SAM 2.1 tiny, CUDA", meaning: "osam 47 ms · sam-rs f16 + FlashAttention ~21 ms (large ~50 ms)" },
            { flag: "SAM 2.1 tiny, CPU", meaning: "osam 705 ms · sam-rs bf16 ~463 ms (large ~1.85 s)" },
            { flag: "SAM 3 \"truck\", CUDA", meaning: "osam 218 ms · sam-rs f16 + FlashAttention ~90 ms" },
            { flag: "SAM 3 \"truck\", CPU", meaning: "PyTorch 5.7 s · osam 6.9 s · sam-rs bf16 ~3.5 s" },
            { flag: "EfficientSAM-Ti", meaning: "CUDA f16 21 ms encoder + 7 ms decode · CPU bf16 ~0.3 s encoder" },
            { flag: "MobileSAM", meaning: "CUDA f16 ~18 ms · CPU ~0.5 s" },
          ]}
        />
        <P>
          Not measured: PyTorch on the GPU, and PyTorch SAM (torch does not import on the benchmark machine).
          Video: 42 to 50 ms per frame for SAM 2.1 tiny and 85 to 100 ms for SAM 3 on CUDA, including PNG
          decoding.
        </P>
      </Section>

      <Section>
        <H2>Parity</H2>
        <P>
          Each port is checked against its reference by an example that compares intermediate activations
          (<IC>examples/*_parity.rs</IC>, with the reference dumps produced by <IC>scripts/ref_*.py</IC>).
        </P>
        <UL>
          <LI>SAM 2 / 2.1: features to about 3e-6 relative error, masks to 1e-6, IoU and object scores to six digits; the video tracker agrees at binary mask IoU 1.0000 on every frame of a 6-frame clip.</LI>
          <LI>SAM 3: every stage to about 1e-6 relative error, detection score 0.932 equal to the reference; the video tracker at binary mask IoU 1.0000 on a 6-frame clip.</LI>
          <LI>MobileSAM: encoder to 1.4e-6 against candle-transformers.</LI>
          <LI>EfficientSAM ViT-Ti: low-res masks to 1.6e-5 and IoU scores to six digits against the ONNX export on point, box and invalid-point prompts. ViT-S runs but has no reference check.</LI>
          <LI>SAM 3 text-prompted video tracking has no reference comparison: it combines the detector and the tracker with a simple association rule.</LI>
        </UL>
      </Section>

      <Section>
        <H2>Source layout</H2>
        <FlagTable
          rows={[
            { flag: "src/model.rs", meaning: "The SAM / MobileSAM API: Sam, SamVariant, Prompt, Mask" },
            { flag: "src/sam1/", meaning: "ViT image encoder, TinyViT, prompt encoder, two-way mask decoder" },
            { flag: "src/efficientsam/", meaning: "EfficientSAM encoders, prompt encoder and decoder" },
            { flag: "src/sam2/", meaning: "Hiera encoder and the SAM 2 model, including video" },
            { flag: "src/sam3/", meaning: "Vision tower, CLIP text encoder, DETR detector, mask head, tracker, video tracker, text-prompted tracking" },
            { flag: "src/kernels.rs, kernels/sam.cu", meaning: "Fused CUDA kernels and their parallel CPU versions" },
            { flag: "src/cpu_gemm.rs", meaning: "bf16 GEMM, convolution and attention kernels for AVX-512 BF16" },
            { flag: "src/download.rs", meaning: "Hugging Face downloads and the cache" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}
