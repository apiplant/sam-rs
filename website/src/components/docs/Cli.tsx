import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>sam-rs</H1>
      <Lead>
        Load the model, run one task on one image, print the answer and the detections in pixel space.{" "}
        <IC>--task</IC> picks how your <IC>--query</IC> becomes a prompt; see{" "}
        <a href="/docs/tasks" class="text-accent hover:text-accent-dim">
          Tasks &amp; prompts
        </a>
        .
      </Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`# detection (comma-separated categories -> joined with </c>)
sam-rs --image street.jpg --task detect -q "person, car, bicycle" --draw out.png

# referring expression / pointing / OCR / GUI
sam-rs --image img.jpg --task ground -q "people wearing red shirts"
sam-rs --image img.jpg --task point  -q "the traffic light"
sam-rs --image img.jpg --task detect-text
sam-rs --image ui.png  --task gui    -q "the search button"

# JSON output, greedy, pure-AR decoding, step trace
sam-rs --image img.jpg -q "ship" --json --temperature 0 --mode slow -v`}
        />
      </Section>

      <Section>
        <H2>Options</H2>
        <FlagTable
          rows={[
            { flag: "--image FILE", meaning: "The image to search (jpeg, png, webp, bmp)." },
            { flag: "--task NAME", meaning: "detect, ground, ground-single, text, detect-text, gui, point or raw. Default: detect." },
            { flag: "-q, --query TEXT", meaning: <>What to look for. Required for every task except <IC>detect-text</IC>.</> },
            { flag: "--model DIR", meaning: "A local model directory (a clone of nvidia/Segment Anything). Without it the checkpoint is downloaded into the cache directory on first use and reused." },
            { flag: "--mode fast|slow|hybrid", meaning: "fast: MTP blocks only. slow: pure auto-regressive. hybrid (default): MTP first, AR fallback on malformed or uncertain boxes." },
            { flag: "--draw FILE", meaning: "Write a copy of the image with the detections drawn on it, one colour per label." },
            { flag: "--json", meaning: "Print prompt, answer, detections and stats as JSON instead of the readable summary." },
            { flag: "--max-patches N", meaning: "Vision budget in 14x14 patches before downscaling (default 25600, about 6400 LLM tokens). Lower it for faster runs on big images." },
            { flag: "--max-new-tokens N", meaning: "Generation limit (default 8192)." },
            { flag: "--temperature T", meaning: "0 = greedy. Default 0.7, like the upstream worker." },
            { flag: "--top-p P, --top-k K", meaning: "Nucleus and top-k sampling. top-p defaults to 0.9." },
            { flag: "--repetition-penalty R", meaning: "Default 1.1." },
            { flag: "--seed N", meaning: "Sampling seed." },
            { flag: "--cpu", meaning: "Run on the CPU even when a GPU is available (f32)." },
            { flag: "--f32", meaning: "f32 instead of bf16 on the GPU: slower, for numerical checks." },
            { flag: "-v, --verbose", meaning: "Print each decoding step (mtp or ar) to stderr." },
          ]}
        />
        <P>
          Diagnostics (model load time, patch grid, token and step counts, timings) go to stderr, so stdout
          stays clean for piping.
        </P>
      </Section>
    </DocsLayout>
  );
}
