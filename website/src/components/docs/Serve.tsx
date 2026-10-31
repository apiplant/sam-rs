import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsServe() {
  return (
    <DocsLayout>
      <H1>Server &amp; web UI</H1>
      <Lead>
        <IC>sam-rs serve</IC> puts every model behind one HTTP API (ntex) and serves a web UI at <IC>/</IC>. The API
        reference renders at <IC>/docs</IC>, from <IC>/openapi.json</IC>.
      </Lead>

      <Section>
        <H2>Run it</H2>
        <CopyBlock command={`sam-rs serve                                   # http://127.0.0.1:8080
sam-rs serve --addr 0.0.0.0:9000 --preload sam2.1-hiera-tiny --max-models 3`} />
        <FlagTable
          rows={[
            { flag: "--addr HOST:PORT", meaning: "Where to listen. Default 127.0.0.1:8080." },
            { flag: "--preload MODEL", meaning: "Load a model before listening. Repeatable." },
            { flag: "--max-models N", meaning: "Models kept in memory; the least recently used is dropped first. Default 2." },
            { flag: "--no-download", meaning: "Only models already in the cache can be loaded." },
            { flag: "--cpu, --f32, --dtype", meaning: "As for the CLI." },
          ]}
        />
        <P>
          The models live on one worker thread, because the GPU is single-tenant: requests queue and run one at a
          time. Models load on first use.
        </P>
      </Section>

      <Section>
        <H2>The web UI</H2>
        <P>
          Pick a model (downloaded ones first), open, drop or paste an image, and click for foreground points
          (right-click or alt for background), drag for boxes (shift-drag excludes, SAM 3), or type a concept for
          SAM 3. Masks update as you prompt, because the image is embedded once and each prompt only runs the
          decoder. In Video mode, drop a clip or a set of frames, prompt the first frame and press Track: SAM 2
          and SAM 3 follow the object, and SAM 3 with a concept follows every instance. Every mask can be
          downloaded as a PNG, and <IC>Load local weights</IC> loads a <IC>.safetensors</IC> file from the
          server&apos;s disk.
        </P>
      </Section>

      <Section>
        <H2>The API</H2>
        <CopyBlock command={`curl -s localhost:8080/v1/models           # local / loaded / gated, and what each model accepts
id=$(curl -s -X POST --data-binary @truck.jpg 'localhost:8080/v1/images?model=sam2.1-hiera-tiny' | jq .id)
curl -s -X POST -d '{"points":[{"x":500,"y":375}]}' localhost:8080/v1/images/$id/segment

v=$(curl -s -X POST -d '{"model":"sam3","text":"wheel"}' localhost:8080/v1/videos | jq .id)
curl -s -X POST --data-binary @frame00.jpg localhost:8080/v1/videos/$v/frames`} />
        <FlagTable
          rows={[
            { flag: "GET /v1/info, /v1/models", meaning: "Device, dtype, loaded models; the catalog with sizes and capabilities." },
            { flag: "POST /v1/models/{id}/load", meaning: "Download and load a model, optionally from a local weights path. DELETE /v1/models/{id} unloads it." },
            { flag: "POST /v1/images?model=ID", meaning: "Embed an image (the body) and get a session id." },
            { flag: "POST /v1/images/{id}/segment", meaning: "Points, boxes, and for SAM 3 text and exemplar boxes. Masks come back as base64 grayscale PNGs with score, area and bbox." },
            { flag: "POST /v1/videos, /v1/videos/{id}/frames", meaning: "Start tracking with points, a box or SAM 3 text, then post each frame in order." },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}
