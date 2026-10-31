import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>sam-rs</IC> crate is the engine behind the CLI: checkpoint download and loading, image
        preprocessing, the models, and mask post-processing. Embed an image once, then decode many prompts.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <CopyBlock command={`cargo add sam-rs                      # CPU only, no CUDA toolchain needed
cargo add sam-rs --features cuda      # + CUDA
cargo add sam-rs --features flash-attn  # + FlashAttention-2 (long first build)`} />
        <P>
          Call <IC>sam_rs::cpu_gemm::init_threads()</IC> once at startup on the CPU: it sizes the GEMM pool from
          the logical cores instead of candle&apos;s physical-core default.
        </P>
      </Section>

      <Section>
        <H2>SAM, MobileSAM</H2>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::{DType, Device};
use sam_rs::{Prompt, Sam, SamVariant};

let device = Device::cuda_if_available(0)?;       // the CPU unless the \`cuda\` feature is on
let sam = Sam::from_pretrained(SamVariant::VitB, &device, DType::F16)?;   // DType::F32 on the CPU

let image = image::open("truck.jpg")?.to_rgb8();
let embedding = sam.embed(&image)?;               // the expensive part, once per image
let masks = sam.predict(&embedding, &Prompt::Points(vec![(500.0, 375.0, true)]), true)?;
for m in &masks {
    println!("iou {:.3}, {}x{}", m.iou, m.width, m.height);
}`}</Pre>
        <P>
          <IC>Mask</IC> holds a row-major binary mask at the original image size and its predicted IoU.{" "}
          <IC>Prompt::Box([x0, y0, x1, y1])</IC> prompts with a box; the last argument of <IC>predict</IC> asks
          for three candidates instead of one. <IC>Sam::from_file</IC> loads a local checkpoint.
        </P>
      </Section>

      <Section>
        <H2>The other families</H2>
        <P>
          Each family has its own module with the same shape (<IC>from_pretrained</IC>, <IC>embed</IC>,{" "}
          <IC>predict</IC>): <IC>sam_rs::efficientsam::EfficientSam</IC>, <IC>sam_rs::sam2::Sam2</IC> and{" "}
          <IC>sam_rs::sam3::Sam3</IC>.
        </P>
        <Pre caption="SAM 3: every instance of a concept" lang="rust">{`use sam_rs::sam3::Sam3;

let sam = Sam3::from_pretrained(&device, DType::F16)?;      // gated: needs HF_TOKEN the first time
let features = sam.encode_image(&image)?;
for inst in sam.segment(&features, "wheel", 0.5)? {
    println!("{:.3} {:?} {} px", inst.score, inst.bbox, inst.mask.data.iter().filter(|&&v| v > 0).count());
}`}</Pre>
        <Pre caption="SAM 3: text-prompted video tracking" lang="rust">{`use sam_rs::sam3::text_track::TextTracker;

let mut tracker = TextTracker::new(&sam, "wheel", 0.3).detect_every(5);
for frame in frames {                                       // image::RgbImage, in order
    for obj in tracker.step(&frame)? {
        println!("object {} score {:.1} {}", obj.id, obj.object_score, obj.mask.data.len());
    }
}`}</Pre>
        <P>
          For a single object, <IC>sam3::video::Session</IC> with <IC>add_prompt</IC> and{" "}
          <IC>Sam3::track_frame</IC> follows it frame by frame; <IC>sam2::Sam2</IC> offers the same for SAM 2.
        </P>
      </Section>
    </DocsLayout>
  );
}
