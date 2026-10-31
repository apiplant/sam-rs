/* The HTTP API of `sam-rs serve`; /openapi.json describes the same endpoints. */

export interface Capabilities {
  points: boolean;
  boxes: boolean;
  text: boolean;
  exemplars: boolean;
  video: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  size_mb: number;
  gated: boolean;
  local: boolean;
  loaded: boolean;
  capabilities: Capabilities;
}

export interface Info {
  version: string;
  device: "cuda" | "cpu";
  dtype: string;
  cuda_build: boolean;
  bf16_kernels: boolean;
  hf_token: boolean;
  allow_download: boolean;
  max_models: number;
  cache_dir: string | null;
  loaded: string[];
}

export interface ImageSession {
  id: number;
  model: string;
  width: number;
  height: number;
  encode_ms: number;
}

export interface Point {
  x: number;
  y: number;
  /** 1 foreground, 0 background */
  label: 0 | 1;
}

export type Box = [number, number, number, number];

export interface SegmentRequest {
  points?: Point[];
  boxes?: Box[];
  neg_boxes?: Box[];
  text?: string;
  threshold?: number;
  multimask?: boolean;
  tracker?: boolean;
}

export interface Result {
  id: number | null;
  score: number;
  present: boolean;
  area: number;
  bbox: [number, number, number, number] | null;
  /** base64 grayscale PNG */
  png: string;
}

export interface SegmentResponse {
  kind: "masks" | "instances";
  model: string;
  width: number;
  height: number;
  object_score: number | null;
  decode_ms: number;
  results: Result[];
}

export interface VideoRequest {
  model: string;
  points?: Point[];
  boxes?: Box[];
  text?: string;
  threshold?: number;
  detect_every?: number;
}

export interface FrameResponse {
  frame: number;
  width: number;
  height: number;
  elapsed_ms: number;
  objects: Result[];
}

async function check(r: Response): Promise<Response> {
  if (!r.ok) {
    const t = (await r.text()).trim();
    let msg = t;
    try {
      msg = JSON.parse(t).error ?? t;
    } catch {}
    throw new Error(msg || `${r.status} ${r.statusText}`);
  }
  return r;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } });

export const info = async (): Promise<Info> => (await check(await fetch("/v1/info"))).json();
export const models = async (): Promise<ModelInfo[]> => (await check(await fetch("/v1/models"))).json();

export async function loadModel(id: string, weights?: string): Promise<void> {
  await check(await fetch(`/v1/models/${encodeURIComponent(id)}/load`, json(weights ? { weights } : {})));
}

export async function unloadModel(id: string): Promise<void> {
  await check(await fetch(`/v1/models/${encodeURIComponent(id)}`, { method: "DELETE" }));
}

/** Embeds an image once; segment against the returned session as often as you like. */
export async function embed(model: string, image: Blob, signal?: AbortSignal): Promise<ImageSession> {
  return (await check(await fetch(`/v1/images?model=${encodeURIComponent(model)}`, { method: "POST", body: image, signal }))).json();
}

export async function segment(session: number, req: SegmentRequest, signal?: AbortSignal): Promise<SegmentResponse> {
  return (await check(await fetch(`/v1/images/${session}/segment`, { ...json(req), signal }))).json();
}

export async function startVideo(req: VideoRequest): Promise<{ id: number }> {
  return (await check(await fetch("/v1/videos", json(req)))).json();
}

export async function trackFrame(session: number, frame: Blob, signal?: AbortSignal): Promise<FrameResponse> {
  return (await check(await fetch(`/v1/videos/${session}/frames`, { method: "POST", body: frame, signal }))).json();
}

export async function endVideo(session: number): Promise<void> {
  await fetch(`/v1/videos/${session}`, { method: "DELETE" }).catch(() => {});
}
