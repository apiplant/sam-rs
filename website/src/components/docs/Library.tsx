import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>sam-rs</IC> crate is the engine behind the CLI: model loading, image preprocessing,
        generation with Parallel Box Decoding, and output parsing.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <CopyBlock command={`cargo add sam-rs                      # CPU only, no CUDA toolchain needed
cargo add sam-rs --features cuda      # + CUDA`} />
      </Section>

      <Section>
        <H2>Detect objects</H2>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::{DType, Device};
use locate_anything::{prompts, GenerateOptions, LocateAnything};

// Downloads the checkpoint into ~/.cache/sam-rs on first use (about 7.6 GB);
// LocateAnything::load(path, &device, dtype) takes a local clone instead.
let device = Device::cuda_if_available(0)?;       // the CPU unless the \`cuda\` feature is on
let mut model = LocateAnything::from_pretrained(&device, DType::BF16)?;   // DType::F32 on the CPU

let image = image::open("street.jpg")?.to_rgb8();
let found = model.locate(&image, &prompts::detect(&["person", "car"]), &GenerateOptions::greedy())?;

for d in &found.detections {
    println!("{d:?}");
}
println!("{} forward steps, {:.2}s", found.stats.forward_steps, found.stats.total_secs);`}</Pre>
        <P>
          <IC>found.detections</IC> holds <IC>Detection::Box</IC> or <IC>Detection::Point</IC> values in pixel
          coordinates; <IC>output::draw</IC> paints them onto an <IC>RgbImage</IC>.{" "}
          <IC>GenerateOptions::greedy()</IC> gives the same answer every run; <IC>GenerateOptions::default()</IC>
          samples like the upstream worker. <IC>locate_anything::prompts</IC> builds the prompt for each task:{" "}
          <IC>detect</IC>, <IC>ground</IC>, <IC>ground_single</IC>, <IC>text</IC>, <IC>detect_text</IC>,{" "}
          <IC>gui</IC> and <IC>point</IC>.
        </P>
      </Section>

      <Section>
        <H2>Pieces</H2>
        <P>
          <IC>model.encode_image</IC> runs the vision tower alone and <IC>model.build_prompt</IC> tokenizes a
          question with the right number of image tokens: both are public, for callers that want to drive the
          stages themselves.
        </P>
      </Section>
    </DocsLayout>
  );
}
