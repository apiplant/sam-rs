import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section, FlagTable } from "./Prose";

export function DocsTasks() {
  return (
    <DocsLayout>
      <H1>Tasks &amp; prompts</H1>
      <Lead>
        The model takes one prompt format for everything. A task is just a template around your query; use{" "}
        <IC>--task raw</IC> to send your own prompt verbatim.
      </Lead>

      <Section>
        <H2>Tasks</H2>
        <FlagTable
          rows={[
            { flag: "detect", meaning: <>Detection or layout analysis. The query is a comma-separated category list, joined with <IC>{"</c>"}</IC>: <IC>Locate all the instances that matches the following description: person{"</c>"}car.</IC></> },
            { flag: "ground", meaning: <>Phrase grounding, all matching instances: <IC>Locate all the instances that match the following description: Q.</IC></> },
            { flag: "ground-single", meaning: <>Phrase grounding, one instance: <IC>Locate a single instance that matches the following description: Q.</IC></> },
            { flag: "text", meaning: <>Locate a piece of text: <IC>Please locate the text referred as Q.</IC></> },
            { flag: "detect-text", meaning: <>Scene-text detection, no query: <IC>Detect all the text in box format.</IC></> },
            { flag: "gui", meaning: <>GUI element grounding, as a box: <IC>Locate the region that matches the following description: Q.</IC></> },
            { flag: "point", meaning: <>Pointing, and GUI grounding as a point: <IC>Point to: Q.</IC></> },
            { flag: "raw", meaning: "The query is sent as the prompt, unchanged." },
          ]}
        />
      </Section>

      <Section>
        <H2>What the model answers</H2>
        <P>
          Boxes and points come back as tokens with integer coordinates in [0, 1000], relative to the image. The
          binary parses them into pixel coordinates of the original image.
        </P>
        <Pre caption="--task ground-single -q &quot;the cat sleeping on the right&quot;" lang="text">{`<ref>the cat sleeping on the right</ref><box><541><52><998><775></box>`}</Pre>
        <Pre caption="detections" lang="json">{`{"kind":"box","label":"the cat sleeping on the right","x1":346.24,"y1":24.96,"x2":638.72,"y2":372.0}`}</Pre>
        <P>Pointing answers with two coordinates instead of four, on a 640x480 image:</P>
        <Pre caption="--task point -q &quot;the cat on the left&quot;" lang="text">{`<ref>the cat on the left</ref><box><254><390></box>
{"kind":"point","label":"the cat on the left","x":162.56,"y":187.2}`}</Pre>
        <P>
          A category with nothing to find is answered with <IC>{"<box>None</box>"}</IC> and produces no
          detection.
        </P>
      </Section>

      <Section>
        <H2>Parallel Box Decoding</H2>
        <P>
          Instead of emitting a box as four sequential coordinate tokens, the model predicts them in 6-token
          blocks (multi-token prediction), so a dense scene needs far fewer forward steps. In{" "}
          <IC>hybrid</IC> mode the runtime falls back to one-token auto-regressive steps whenever a block is
          malformed or uncertain; <IC>fast</IC> never falls back, <IC>slow</IC> never uses MTP. The step trace
          (<IC>-v</IC>) shows which was used for each step.
        </P>
      </Section>
    </DocsLayout>
  );
}
