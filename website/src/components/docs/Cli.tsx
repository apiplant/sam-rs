import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>sam-rs</H1>
      <Lead>
        Load a model, prompt it with points, a box or text, print the candidate masks and save the best one. The
        prompts each model accepts are listed under{" "}
        <a href="/docs/models" class="text-accent hover:text-accent-dim">
          Models &amp; prompts
        </a>
        .
      </Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`# one foreground point, three candidate masks, the best saved as a PNG
sam-rs --image truck.jpg --point 500,375 --out mask.png

# several points (a trailing 0 marks a background point), or a box
sam-rs --image truck.jpg --point 500,375 --point 1250,600,0
sam-rs --image truck.jpg --box 75,275,1725,850 --single

# other models
sam-rs --model sam-vit-h --image truck.jpg --point 500,375 --index 0
sam-rs --model mobile-sam --image truck.jpg --point 500,375
sam-rs --model sam2.1-hiera-large --image truck.jpg --box 86,282,1710,851

# SAM 3: every instance of a concept
sam-rs --model sam3 --text "wheel" --image truck.jpg --threshold 0.5

# video: a directory of frames, sorted by name
sam-rs --model sam2.1-hiera-tiny --video frames/ --point 500,375 --out masks/
sam-rs --model sam3 --video frames/ --text "wheel" --out masks/`}
        />
      </Section>

      <Section>
        <H2>Options</H2>
        <FlagTable
          rows={[
            { flag: "--image FILE", meaning: "The image to segment (jpeg, png, webp, bmp). Not needed with --video." },
            { flag: "--model NAME", meaning: <>sam-vit-b (default), sam-vit-l, sam-vit-h, mobile-sam, efficientsam-ti, efficientsam-s, sam2.1-hiera-&#123;tiny,small,base-plus,large&#125;, sam2-hiera-*, sam3.</> },
            { flag: "--weights FILE", meaning: "Use this .safetensors checkpoint instead of the cached or downloaded one (Meta layout)." },
            { flag: "--point X,Y[,0]", meaning: "Point prompt in image pixels; a trailing 0 makes it a background point. Repeatable." },
            { flag: "--box X0,Y0,X1,Y1", meaning: "Box prompt in image pixels. With SAM 3 it is an exemplar box (\"find more like this\") and repeatable." },
            { flag: "--text TEXT", meaning: "SAM 3: find and segment every instance of this concept. With --video it starts text-prompted detect-and-track." },
            { flag: "--neg-box BOX", meaning: "SAM 3: negative exemplar box. Repeatable." },
            { flag: "--threshold S", meaning: "SAM 3: minimum instance score for --text. Default 0.3." },
            { flag: "--tracker", meaning: "SAM 3: treat --point / --box as SAM-style prompts for one object instead of text or exemplars." },
            { flag: "--video DIR", meaning: "Track through the frames (images sorted by name) of this directory. SAM 2 and SAM 3." },
            { flag: "--prompt-frame N", meaning: "Video: the frame the prompt belongs to; tracking runs forward from there. Default 0." },
            { flag: "--detect-every N", meaning: "SAM 3 video with --text: re-run the detector every N frames to find objects that appear later. Default 5." },
            { flag: "--out PATH", meaning: "Save the best mask as a PNG; with --video, a directory that gets one mask PNG per frame (and per object for --text)." },
            { flag: "--index N", meaning: "Save candidate N (0-2) instead of the one with the highest predicted IoU." },
            { flag: "--single", meaning: "Return one mask instead of three candidates." },
            { flag: "--dtype f16|f32", meaning: "Compute dtype on CUDA. Default f16." },
            { flag: "--cpu", meaning: "Run on the CPU even when a GPU is available." },
            { flag: "--f32", meaning: "CPU: exact f32 instead of the bf16 kernels (AVX-512 BF16 CPUs only; slower)." },
            { flag: "--bench N", meaning: "Time the encoder and the decoder over N runs." },
          ]}
        />
        <P>
          Diagnostics (model load time, device, timings) go to stderr, so stdout stays clean for piping. Masks
          are not in a fixed semantic order across checkpoints, so pick by the predicted IoU (the default) or
          inspect all three with <IC>--index</IC>.
        </P>
      </Section>
    </DocsLayout>
  );
}
