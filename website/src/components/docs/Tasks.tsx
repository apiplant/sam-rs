import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section, FlagTable } from "./Prose";

export function DocsModels() {
  return (
    <DocsLayout>
      <H1>Models &amp; prompts</H1>
      <Lead>
        Choose a family with <IC>--model</IC>. They all take points and boxes in image pixels; SAM 3 also takes
        text and exemplar boxes, and SAM 2 and SAM 3 track through video.
      </Lead>

      <Section>
        <H2>Models</H2>
        <FlagTable
          rows={[
            { flag: "sam-vit-b / -l / -h", meaning: "Segment Anything: points and boxes, three candidate masks. 375 MB to 2.5 GB. ViT-B loads in the Meta or the transformers layout; ViT-L and ViT-H use the transformers one." },
            { flag: "mobile-sam", meaning: "MobileSAM: a TinyViT-5M encoder with the SAM ViT-B decoder. About 18 ms on an RTX 4090 and 0.5 s on the CPU." },
            { flag: "efficientsam-ti / -s", meaning: "EfficientSAM ViT-Tiny and ViT-Small. The upstream .pt checkpoints are downloaded and converted to safetensors once, automatically. ViT-Ti: 21 ms encoder on CUDA, 0.3 s on the CPU." },
            { flag: "sam2.1-hiera-{tiny,small,base-plus,large}", meaning: <>SAM 2.1; <IC>sam2-hiera-*</IC> are the original SAM 2 weights. Points, boxes and video tracking. 160 to 900 MB.</> },
            { flag: "grounding-dino-tiny / -base", meaning: "Grounding DINO: phrases to boxes (a detector, not a segmenter). Pair it with any model above to prompt it with text. Tiny is 690 MB, base 940 MB." },
            { flag: "sam3", meaning: "SAM 3: text-prompted instance segmentation (score, box and mask per instance), exemplar boxes, a SAM-style tracker and video. 3.4 GB, gated on Hugging Face." },
          ]}
        />
      </Section>

      <Section>
        <H2>Points, boxes and candidates</H2>
        <P>
          A point is <IC>x,y</IC>, a background point <IC>x,y,0</IC>; a box is <IC>x0,y0,x1,y1</IC>. The model
          returns three candidate masks with a predicted IoU each (one with <IC>--single</IC>), because a click
          is ambiguous: it can mean a part, an object or the thing the object is on. In code,{" "}
          <IC>Prompt::Points</IC> and <IC>Prompt::Box</IC> carry the same values.
        </P>
        <Pre caption="--point 500,375 on truck.jpg" lang="text">{`object score 19.36 (object present)
mask 0: iou 0.042, area 631329
mask 1: iou 0.952, area 22109
mask 2: iou 0.493, area 46080`}</Pre>
      </Section>

      <Section>
        <H2>SAM 3 concepts</H2>
        <P>
          <IC>--text</IC> takes a short noun phrase and returns every matching instance whose score is above{" "}
          <IC>--threshold</IC>. Exemplar boxes mean &ldquo;find more like this&rdquo;; they work alone (the text
          defaults to <IC>visual</IC>) or combined with text, and <IC>--neg-box</IC> excludes what a box shows.
        </P>
        <Pre caption="three ways to prompt SAM 3" lang="bash">{`sam-rs --model sam3 --text "wheel" --image truck.jpg --threshold 0.5
sam-rs --model sam3 --text wheel --box 443,626,676,852 --image truck.jpg
sam-rs --model sam3 --point 500,375 --image truck.jpg      # SAM-style, one object (the tracker)`}</Pre>
      </Section>

      <Section>
        <H2>Text for every SAM: Grounding DINO</H2>
        <P>
          Only SAM 3 understands text; the other checkpoints contain no text tensors. Grounding DINO fills the gap the way
          Grounded-SAM does: it turns phrases (<IC>a wheel. a truck.</IC>, separated by full stops) into boxes, and the model you
          picked segments inside each. Each result carries the words of the prompt its box matched.
        </P>
        <Pre caption="boxes only, then boxes to masks" lang="bash">{`sam-rs --model grounding-dino-tiny --image truck.jpg --text "a wheel. a truck." --out boxes.png
sam-rs --model sam2.1-hiera-tiny --detector grounding-dino-tiny --image truck.jpg --text "a wheel. a truck." --out masks.png`}</Pre>
        <P>
          <IC>--box-threshold</IC> (0.35) is the minimum box score and <IC>--text-threshold</IC> (0.25) the minimum token score for a word
          to label a box. Both models stay loaded, so the server needs <IC>--max-models 2</IC> or more. In the web UI pick a{" "}
          <IC>Text detector</IC> under Prompt.
        </P>
      </Section>

      <Section>
        <H2>Video</H2>
        <P>
          Frames are read from a directory of images sorted by name; <IC>ffmpeg -i clip.mp4 frames/%05d.png</IC>{" "}
          extracts them. A point or box on one frame follows one object forward through the clip (SAM 2 and SAM
          3). Several concepts at once (<IC>--text "wheel. person in red"</IC>) are found separately and labelled, and SAM 2 can track by text through a Grounding DINO detector. With SAM 3 and <IC>--text</IC>, the detector finds every instance on the first frame and then every{" "}
          <IC>--detect-every</IC> frames; each detection that no live track explains starts a new tracker
          session, and a track absent for 8 frames in a row is dropped. One mask PNG per object and frame is
          written.
        </P>
        <P>
          The text-prompted association is a greedy mask-IoU rule rather than the reference model&apos;s matching
          logic, and it has not been compared with the reference, so identities can differ from it in crowded or
          heavily occluded scenes. Tracking runs forward only; mask prompts are not ported.
        </P>
      </Section>
    </DocsLayout>
  );
}
