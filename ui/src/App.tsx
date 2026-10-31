import { For, Show, createEffect, createMemo, createSignal, onSettled } from "solid-js";
import * as api from "./api";
import type { Box, FrameResponse, Info, ModelInfo, Point, Result, SegmentResponse } from "./api";
import { Stage, type Overlay } from "./Stage";
import { Icon, Logo, Segmented, Slider, Spinner, Toggle } from "./ui";
import { btn, btnGhost, btnPrimary, btnSm, btnSmIcon, btnStop, chip, errorBox, input, kbd, notice, panel, panelHead, panelTitle, sectionLabel, colorOf } from "./styles";

type Mode = "image" | "video";

interface Loaded {
  blob: Blob;
  url: string;
  w: number;
  h: number;
  name: string;
}

interface TrackedFrame {
  url: string;
  objects: Result[];
  ms: number;
}

type Source = { kind: "video"; el: HTMLVideoElement; duration: number } | { kind: "frames"; files: File[] };

const STORE = "sam-rs-ui";
const MAX_SIDE = 1280; // video frames are downscaled to this before upload

function loadSaved(): Record<string, any> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function toBlob(src: CanvasImageSource, w: number, h: number, type = "image/jpeg"): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d")!.drawImage(src, 0, 0, w, h);
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("could not encode the frame"))), type, 0.92));
}

const scaled = (w: number, h: number) => {
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  return { w: Math.round(w * s), h: Math.round(h * s) };
};

async function openImage(file: Blob, name: string, downscale = false): Promise<Loaded> {
  const bmp = await createImageBitmap(file);
  let blob = file;
  let { width: w, height: h } = bmp;
  if (downscale) {
    const s = scaled(w, h);
    if (s.w !== w) {
      blob = await toBlob(bmp, s.w, s.h);
      [w, h] = [s.w, s.h];
    }
  }
  bmp.close();
  return { blob, url: URL.createObjectURL(blob), w, h, name };
}

function seek(el: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((res, rej) => {
    const ok = () => (el.removeEventListener("error", bad), res());
    const bad = () => (el.removeEventListener("seeked", ok), rej(new Error("could not seek the video")));
    el.addEventListener("seeked", ok, { once: true });
    el.addEventListener("error", bad, { once: true });
    el.currentTime = Math.min(t, Math.max(0, el.duration - 0.001));
  });
}

export function App() {
  const saved = loadSaved();
  const [info, setInfo] = createSignal<Info | null>(null);
  const [online, setOnline] = createSignal(true);
  const [models, setModels] = createSignal<ModelInfo[]>([]);
  const [modelId, setModelId] = createSignal<string>(saved.model ?? "");
  const [localOnly, setLocalOnly] = createSignal<boolean>(saved.localOnly ?? true);
  const [mode, setMode] = createSignal<Mode>("image");
  const [error, setError] = createSignal("");
  const [weights, setWeights] = createSignal("");

  // prompts
  const [image, setImage] = createSignal<Loaded | null>(null);
  const [points, setPoints] = createSignal<Point[]>([]);
  const [boxes, setBoxes] = createSignal<Box[]>([]);
  const [negBoxes, setNegBoxes] = createSignal<Box[]>([]);
  const [text, setText] = createSignal("");
  const [threshold, setThreshold] = createSignal<number>(saved.threshold ?? 0.5);
  const [multimask, setMultimask] = createSignal<boolean>(saved.multimask ?? true);
  const [tracker, setTracker] = createSignal(false);
  const [detector, setDetector] = createSignal<string>(saved.detector ?? "");
  const [boxThreshold, setBoxThreshold] = createSignal<number>(saved.boxThreshold ?? 0.35);

  // image mode
  const [session, setSession] = createSignal<api.ImageSession | null>(null);
  const [resp, setResp] = createSignal<SegmentResponse | null>(null);
  const [hidden, setHidden] = createSignal<number[]>([]);
  const [picked, setPicked] = createSignal<number | null>(null);
  const [busy, setBusy] = createSignal("");

  // video mode
  const [source, setSource] = createSignal<Source | null>(null);
  const [fps, setFps] = createSignal<number>(saved.fps ?? 6);
  const [maxFrames, setMaxFrames] = createSignal<number>(saved.maxFrames ?? 48);
  const [detectEvery, setDetectEvery] = createSignal<number>(saved.detectEvery ?? 5);
  const [frames, setFrames] = createSignal<TrackedFrame[]>([]);
  const [cur, setCur] = createSignal(0);
  const [tracking, setTracking] = createSignal(false);
  const [progress, setProgress] = createSignal({ done: 0, total: 0 });
  const [playing, setPlaying] = createSignal(false);

  const spec = createMemo(() => models().find((m) => m.id === modelId()) ?? null);
  const caps = () => spec()?.capabilities;
  const isSam3 = () => spec()?.family === "SAM 3";
  const detectors = () => models().filter((m) => m.capabilities.detector);
  /** text goes to SAM 3 directly, or to a Grounding DINO detector in front of any other segmenter */
  const useDetector = () => !caps()?.text && !!detector();
  const hasPrompt = () => points().length + boxes().length + negBoxes().length > 0 || ((isSam3() || useDetector()) && text().trim().length > 0);

  const refresh = async () => {
    try {
      const [i, m] = await Promise.all([api.info(), api.models()]);
      setInfo(i);
      setModels(m);
      setOnline(true);
      if (!m.some((x) => x.id === modelId())) {
        const first = m.find((x) => x.loaded) ?? m.find((x) => x.local) ?? m[0];
        if (first) setModelId(first.id);
      }
    } catch (e) {
      setOnline(false);
      setError(`Cannot reach the server: ${(e as Error).message}`);
    }
  };

  createEffect(
    () => JSON.stringify({ model: modelId(), detector: detector(), boxThreshold: boxThreshold(), localOnly: localOnly(), threshold: threshold(), multimask: multimask(), fps: fps(), maxFrames: maxFrames(), detectEvery: detectEvery() }),
    (json) => {
      try {
        localStorage.setItem(STORE, json);
      } catch {}
    },
  );

  // ---- files

  async function openFiles(files: File[]) {
    setError("");
    try {
      const vids = files.filter((f) => f.type.startsWith("video/"));
      const imgs = files.filter((f) => f.type.startsWith("image/")).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      if (!vids.length && !imgs.length) throw new Error("that is neither an image nor a video");
      resetAll();
      if (vids.length) {
        const f = vids[0];
        const el = document.createElement("video");
        el.muted = true;
        el.preload = "auto";
        el.playsInline = true;
        el.src = URL.createObjectURL(f);
        await new Promise<void>((res, rej) => {
          el.onloadeddata = () => res();
          el.onerror = () => rej(new Error("the browser cannot decode that video"));
        });
        const { w, h } = scaled(el.videoWidth, el.videoHeight);
        await seek(el, 0);
        const first = await openImage(await toBlob(el, w, h), f.name);
        setSource({ kind: "video", el, duration: el.duration });
        setImage(first);
        setMode("video");
      } else if (imgs.length > 1 || mode() === "video") {
        setSource({ kind: "frames", files: imgs });
        setImage(await openImage(imgs[0], imgs[0].name, true));
        setMode("video");
      } else {
        setSource(null);
        setImage(await openImage(imgs[0], imgs[0].name));
        setMode("image");
      }
      if (mode() === "video" && !caps()?.video) {
        const m = models().find((x) => x.capabilities.video && x.local) ?? models().find((x) => x.capabilities.video);
        if (m) setModelId(m.id);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function resetAll() {
    stopTrack();
    setPoints([]);
    setBoxes([]);
    setNegBoxes([]);
    setSession(null);
    setResp(null);
    setHidden([]);
    setPicked(null);
    setFrames([]);
    setCur(0);
    setProgress({ done: 0, total: 0 });
  }

  function switchMode(m: Mode) {
    if (m === mode()) return;
    setMode(m);
    setError("");
    if (m === "video" && !caps()?.video) {
      const v = models().find((x) => x.capabilities.video && x.local) ?? models().find((x) => x.capabilities.video);
      if (v) setModelId(v.id);
    }
  }

  // ---- image mode: embed once per (image, model), segment on every prompt change

  let embedCtl: AbortController | null = null;
  createEffect(
    () => ({ img: image(), model: modelId(), mode: mode() }),
    ({ img, model, mode }) => {
      embedCtl?.abort();
      setSession(null);
      setResp(null);
      if (!img || !model || mode !== "image") return;
      const ctl = (embedCtl = new AbortController());
      void (async () => {
        setError("");
        const m = models().find((x) => x.id === model);
        setBusy(m?.loaded ? "Encoding the image…" : m?.local ? `Loading ${m.name}…` : `Downloading ${m?.name ?? model}…`);
        try {
          const s = await api.embed(model, img.blob, ctl.signal);
          if (ctl.signal.aborted) return;
          setSession(s);
          void refresh();
        } catch (e) {
          if (!ctl.signal.aborted) setError((e as Error).message);
        } finally {
          if (!ctl.signal.aborted) setBusy("");
        }
      })();
    },
  );

  let segTimer: ReturnType<typeof setTimeout> | undefined;
  let segGen = 0;
  createEffect(
    () => ({ s: session(), p: points(), b: boxes(), n: negBoxes(), t: text(), th: threshold(), mm: multimask(), tr: tracker(), mode: mode(), det: useDetector() && points().length + boxes().length + negBoxes().length === 0 && text().trim() ? detector() : "", bt: boxThreshold() }),
    ({ s, p, b, n, t, th, mm, tr, mode, det, bt }) => {
      clearTimeout(segTimer);
      if (!s || mode !== "image") return;
      if (!hasPrompt()) {
        setResp(null);
        return;
      }
      segTimer = setTimeout(async () => {
        const gen = ++segGen;
        setBusy((x) => x || "Segmenting…");
        try {
          const r = det
            ? await api.segment(s.id, { text: t, detector: det, threshold: bt })
            : await api.segment(s.id, { points: p, boxes: b, neg_boxes: n, text: isSam3() ? t : undefined, threshold: th, multimask: mm, tracker: isSam3() ? tr : undefined });
          if (gen !== segGen) return;
          setError("");
          setResp(r);
          setHidden([]);
          setPicked(r.kind === "masks" ? bestIndex(r) : null);
        } catch (e) {
          if (gen === segGen) setError((e as Error).message);
        } finally {
          if (gen === segGen) setBusy("");
        }
      }, 120);
    },
  );

  const bestIndex = (r: SegmentResponse) => r.results.reduce((best, x, i) => (x.score > r.results[best].score ? i : best), 0);

  // ---- video mode

  let trackCtl: AbortController | null = null;
  const frameCount = () => {
    const s = source();
    if (!s) return 0;
    return s.kind === "video" ? Math.min(maxFrames(), Math.max(1, Math.floor(s.duration * fps()))) : Math.min(maxFrames(), s.files.length);
  };

  async function grab(i: number): Promise<Blob> {
    const s = source()!;
    const img = image()!;
    if (s.kind === "video") {
      await seek(s.el, i / fps());
      return toBlob(s.el, img.w, img.h);
    }
    const bmp = await createImageBitmap(s.files[i]);
    const b = await toBlob(bmp, img.w, img.h);
    bmp.close();
    return b;
  }

  function stopTrack() {
    trackCtl?.abort();
    trackCtl = null;
    setTracking(false);
    setPlaying(false);
  }

  async function track() {
    const img = image();
    if (!img || !source() || !spec()) return;
    setError("");
    setFrames([]);
    setCur(0);
    const ctl = (trackCtl = new AbortController());
    const total = frameCount();
    setProgress({ done: 0, total });
    setTracking(true);
    let vid: number | null = null;
    try {
      const noBoxPrompt = points().length === 0 && boxes().length === 0;
      const vs = await api.startVideo({
        model: modelId(),
        points: points(),
        boxes: boxes(),
        text: (isSam3() || useDetector()) && noBoxPrompt ? text().trim() || undefined : undefined,
        detector: useDetector() && noBoxPrompt ? detector() : undefined,
        threshold: useDetector() ? boxThreshold() : threshold(),
        detect_every: detectEvery(),
      });
      vid = vs.id;
      for (let i = 0; i < total && !ctl.signal.aborted; i++) {
        const blob = await grab(i);
        const r: FrameResponse = await api.trackFrame(vs.id, blob, ctl.signal);
        setFrames((f) => [...f, { url: URL.createObjectURL(blob), objects: r.objects, ms: r.elapsed_ms }]);
        setCur(i);
        setProgress({ done: i + 1, total });
      }
      void refresh();
    } catch (e) {
      if (!ctl.signal.aborted) setError((e as Error).message);
    } finally {
      if (vid !== null) void api.endVideo(vid);
      if (trackCtl === ctl) setTracking(false);
    }
  }

  // playback of the tracked frames
  createEffect(
    () => ({ on: playing(), n: frames().length, fps: fps() }),
    ({ on, n, fps }) => {
      if (!on || n < 2) return;
      let alive = true;
      void (async () => {
        while (alive) {
          await sleep(1000 / Math.max(1, fps));
          if (!alive) break;
          setCur((c) => (c + 1 >= frames().length ? 0 : c + 1));
        }
      })();
      return () => {
        alive = false;
      };
    },
  );

  // with a text detector the prompt is the text alone: drop clicks and boxes left over from another model
  createEffect(
    () => useDetector(),
    (on) => {
      if (on) {
        setPoints([]);
        setBoxes([]);
        setNegBoxes([]);
      }
    },
  );

  // ---- prompts

  const addPoint = (p: Point) => {
    if (mode() === "video" && frames().length) setFrames([]);
    setPoints((l) => [...l, p]);
  };
  const addBox = (b: Box, neg: boolean) => (neg ? setNegBoxes((l) => [...l, b]) : setBoxes((l) => [...l, b]));
  const undo = () => {
    if (negBoxes().length && !boxes().length && !points().length) setNegBoxes((l) => l.slice(0, -1));
    else if (boxes().length && (!points().length || boxes().length >= points().length)) setBoxes((l) => l.slice(0, -1));
    else setPoints((l) => l.slice(0, -1));
  };
  const clearPrompts = () => {
    setPoints([]);
    setBoxes([]);
    setNegBoxes([]);
    setText("");
    // in video mode the tracked frames lock the stage: clearing also goes back to the first frame, ready to prompt again
    stopTrack();
    setFrames([]);
    setCur(0);
    setProgress({ done: 0, total: 0 });
  };

  // ---- model actions

  const [busyModel, setBusyModel] = createSignal("");
  async function loadWeights() {
    if (!modelId() || !weights().trim()) return;
    setBusyModel(modelId());
    try {
      await api.loadModel(modelId(), weights().trim());
      setError("");
      await refresh();
      // re-embed with the new weights
      const img = image();
      if (img) setImage({ ...img });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyModel("");
    }
  }

  async function unload(id: string) {
    setBusyModel(id);
    try {
      await api.unloadModel(id);
      await refresh();
    } finally {
      setBusyModel("");
    }
  }

  onSettled(() => {
    void refresh();
    const timer = setInterval(() => {
      fetch("/health")
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    }, 8000);
    const key = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.key === "z" && (e.ctrlKey || e.metaKey)) || e.key === "Backspace") {
        e.preventDefault();
        undo();
      } else if (e.key === "Escape") clearPrompts();
      else if (e.key === " " && mode() === "video" && frames().length > 1) {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight" && frames().length) setCur((c) => Math.min(frames().length - 1, c + 1));
      else if (e.key === "ArrowLeft" && frames().length) setCur((c) => Math.max(0, c - 1));
    };
    const paste = (e: ClipboardEvent) => {
      const files = [...(e.clipboardData?.files ?? [])];
      if (files.length) void openFiles(files);
    };
    window.addEventListener("keydown", key);
    window.addEventListener("paste", paste);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", key);
      window.removeEventListener("paste", paste);
    };
  });

  // ---- what the stage shows

  const stageSrc = () => (mode() === "video" && frames().length ? frames()[Math.min(cur(), frames().length - 1)].url : (image()?.url ?? null));

  const overlays = createMemo<Overlay[]>(() => {
    if (mode() === "video") {
      const f = frames()[cur()];
      return f ? f.objects.filter((o) => o.present).map((o) => ({ png: o.png, color: colorOf(o.id ?? 0), label: o.label ? `${o.label} #${(o.id ?? 0) + 1}` : o.id !== null && frames().some((x) => x.objects.length > 1) ? `#${o.id + 1}` : undefined, bbox: o.bbox })) : [];
    }
    const r = resp();
    if (!r) return [];
    return r.results
      .map((o, i) => ({ o, i }))
      .filter(({ i }) => (r.kind === "masks" ? (picked() === null ? true : picked() === i) : !hidden().includes(i)))
      .map(({ o, i }) => ({ png: o.png, color: colorOf(i), label: r.kind === "instances" ? `${o.label ? o.label + " " : ""}${(o.score * 100).toFixed(0)}%` : undefined, bbox: r.kind === "instances" ? o.bbox : null }));
  });

  const family = (m: ModelInfo) => m.family;
  const groups = createMemo(() => {
    const out: { family: string; items: ModelInfo[] }[] = [];
    for (const m of models()) {
      if (m.capabilities.detector) continue;
      if (localOnly() && !m.local && !m.loaded && m.id !== modelId()) continue;
      let g = out.find((x) => x.family === family(m));
      if (!g) out.push((g = { family: family(m), items: [] }));
      g.items.push(m);
    }
    return out;
  });

  const needsDownload = () => spec() && !spec()!.local && !spec()!.loaded;

  return (
    <div class="flex h-full flex-col">
      {/* header */}
      <header class="flex h-12 flex-none items-center gap-3 border-b border-ink-800 bg-ink-950 px-4">
        <Logo />
        <span class="text-[15px] font-semibold tracking-tight text-ink-100">
          sam<span class="bg-gradient-to-r from-accent to-accent2 bg-clip-text text-transparent">-rs</span>
        </span>
        <Show when={info()}>
          {(i) => (
            <span class={chip} title={`${i().dtype} · v${i().version}${i().bf16_kernels && i().device === "cpu" ? " · bf16 kernels" : ""}`}>
              <Icon name={i().device === "cuda" ? "bolt" : "cpu"} size={12} class="text-accent" />
              {i().device === "cuda" ? "CUDA" : "CPU"} · {i().dtype}
            </span>
          )}
        </Show>
        <div class="ml-auto flex items-center gap-1.5">
          <span class={`size-2 rounded-full ${online() ? "bg-ok" : "bg-rec animate-pulse-soft"}`} title={online() ? "Server online" : "Server unreachable"} />
          <a class={`${btnGhost} !px-2.5`} href="/docs" target="_blank" rel="noreferrer">
            <Icon name="book" size={14} />
            API
          </a>
        </div>
      </header>

      <div class="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[300px_minmax(0,1fr)_320px]">
        {/* left: mode, model, prompt options */}
        <aside class={`${panel} overflow-hidden`}>
          <div class="flex min-h-0 flex-1 flex-col overflow-auto">
            <div class="p-3">
              <Segmented
                value={mode()}
                onChange={switchMode}
                options={[
                  { id: "image", label: "Image", icon: "image" },
                  { id: "video", label: "Video", icon: "video" },
                ]}
              />
            </div>

            <section class="border-t border-ink-750 p-3">
              <div class={sectionLabel}>
                <span>Model</span>
                <label class="flex cursor-pointer items-center gap-1.5 font-normal normal-case tracking-normal text-ink-300">
                  <input type="checkbox" class="accent-[var(--color-accent)]" checked={localOnly()} onChange={(e) => setLocalOnly(e.currentTarget.checked)} />
                  downloaded only
                </label>
              </div>
              <div class="flex max-h-[34vh] flex-col gap-3 overflow-auto pr-1">
                <For each={groups()}>
                  {(g) => (
                    <div>
                      <div class="mb-1 px-1 text-[10.5px] font-semibold uppercase tracking-[0.1em] text-ink-500">{g.family}</div>
                      <div class="flex flex-col gap-1">
                        <For each={g.items}>
                          {(m) => {
                            const disabled = () => mode() === "video" && !m.capabilities.video;
                            return (
                              <button
                                class="flex w-full cursor-pointer items-center gap-2 rounded-lg border border-transparent px-2 py-1.5 text-left transition-colors enabled:hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-40 aria-pressed:border-accent/50 aria-pressed:bg-accent/10"
                                aria-pressed={modelId() === m.id ? "true" : "false"}
                                disabled={disabled()}
                                title={disabled() ? "No video tracking for this model" : m.id}
                                onClick={() => setModelId(m.id)}
                              >
                                <span class={`size-2 flex-none rounded-full ${m.loaded ? "bg-ok" : m.local ? "bg-ink-400" : "border border-ink-500"}`} title={m.loaded ? "loaded" : m.local ? "downloaded" : "not downloaded"} />
                                <span class="min-w-0 flex-1 truncate text-ink-100">{m.name}</span>
                                <Show when={m.gated}>
                                  <Icon name="lock" size={11} class="flex-none text-warn" />
                                </Show>
                                <span class="flex-none font-mono text-[11px] text-ink-400">{m.size_mb >= 1000 ? `${(m.size_mb / 1000).toFixed(1)} GB` : `${m.size_mb} MB`}</span>
                              </button>
                            );
                          }}
                        </For>
                      </div>
                    </div>
                  )}
                </For>
              </div>
              <Show when={spec()}>
                {(s) => (
                  <div class="mt-3 flex flex-wrap items-center gap-1.5">
                    <Show when={s().capabilities.points}>
                      <span class={chip}>points</span>
                    </Show>
                    <Show when={s().capabilities.boxes}>
                      <span class={chip}>boxes</span>
                    </Show>
                    <Show when={s().capabilities.text}>
                      <span class={chip}>text</span>
                    </Show>
                    <Show when={s().capabilities.video}>
                      <span class={chip}>video</span>
                    </Show>
                    <Show when={s().loaded}>
                      <button class={`${btn} ${btnSm} ml-auto`} disabled={busyModel() === s().id} onClick={() => unload(s().id)} title="Free the model's memory">
                        Unload
                      </button>
                    </Show>
                  </div>
                )}
              </Show>
              <Show when={needsDownload()}>
                <p class="mt-2 flex items-start gap-1.5 text-[12px] leading-snug text-ink-400">
                  <Icon name="cloud" size={13} class="mt-px flex-none" />
                  {info()?.allow_download === false ? "Not downloaded, and downloads are disabled on this server." : "Downloads from Hugging Face the first time you use it."}
                </p>
              </Show>
              <Show when={spec()?.gated && !info()?.hf_token && needsDownload()}>
                <p class={`${notice} mt-2 text-[12px]`}>SAM 3 is gated: accept the terms on Hugging Face and start the server with HF_TOKEN set.</p>
              </Show>
              <details class="mt-3 text-ink-300">
                <summary class="cursor-pointer select-none text-[12px] text-ink-400 hover:text-ink-200">Load local weights…</summary>
                <div class="mt-2 flex gap-1.5">
                  <input class={input} placeholder="/path/to/model.safetensors" value={weights()} onInput={(e) => setWeights(e.currentTarget.value)} />
                  <button class={btn} disabled={!weights().trim() || !!busyModel()} onClick={loadWeights}>
                    Load
                  </button>
                </div>
                <p class="mt-1.5 text-[11.5px] leading-snug text-ink-500">A path on the server's disk. SAM 2 needs config.json next to it, SAM 3 tokenizer.json.</p>
              </details>
            </section>

            <section class="border-t border-ink-750 p-3">
              <div class={sectionLabel}>
                <span>Prompt</span>
                <button class="cursor-pointer font-normal normal-case tracking-normal text-ink-400 hover:text-ink-100" onClick={clearPrompts}>
                  clear
                </button>
              </div>
              <Show when={caps()?.text}>
                <label class="mb-1 block text-ink-300">Concepts (SAM 3)</label>
                <input class={input} placeholder="e.g. wheel. person in red" value={text()} onInput={(e) => setText(e.currentTarget.value)} />
                <p class="mt-1.5 text-[12px] leading-snug text-ink-400">Several at once, separated by full stops or commas. Each is found, labelled and, in video, tracked.</p>
                <Slider label="Score threshold" value={threshold()} min={0.01} max={0.99} step={0.05} numeric onChange={setThreshold} />
                <Toggle label="Tracker mode" hint="Boxes prompt one object, like SAM 2, instead of marking examples" checked={tracker()} onChange={setTracker} />
              </Show>
              <Show when={spec() && !caps()?.text}>
                <label class="mb-1 block text-ink-300">Text detector</label>
                <select
                  class={`${input} cursor-pointer`}
                  value={detector()}
                  onChange={(e) => setDetector(e.currentTarget.value)}
                >
                  <option value="">None: clicks and boxes only</option>
                  <For each={detectors()}>
                    {(d) => (
                      <option value={d.id}>
                        {d.name}
                        {d.local ? "" : " (downloads)"}
                      </option>
                    )}
                  </For>
                </select>
                <Show
                  when={useDetector()}
                  fallback={<p class="mt-1.5 text-[12px] leading-snug text-ink-400">{spec()!.name} has no text input of its own. Pick Grounding DINO to type phrases: it finds the boxes and {spec()!.name} cuts each one out.</p>}
                >
                  <label class="mb-1 mt-3 block text-ink-300">Phrases</label>
                  <input class={input} placeholder="a wheel. a truck." value={text()} onInput={(e) => setText(e.currentTarget.value)} />
                  <p class="mt-1.5 text-[12px] leading-snug text-ink-400">Separate phrases with full stops. The image takes text only while a detector is selected (clicks and boxes are off). Grounding DINO and {spec()!.name} both stay loaded.</p>
                  <Slider label="Box threshold" value={boxThreshold()} min={0.01} max={0.99} step={0.05} numeric onChange={setBoxThreshold} />
                </Show>
              </Show>
              <Toggle label="Three candidates" hint="Ambiguous clicks return several masks to pick from" checked={multimask()} onChange={setMultimask} />
              <ul class="mt-2 space-y-1.5 text-[12px] leading-snug text-ink-400">
                <li>
                  <span class={kbd}>click</span> foreground point · <span class={kbd}>right-click</span> or <span class={kbd}>alt</span> background
                </li>
                <li>
                  <span class={kbd}>drag</span> a box
                  <Show when={caps()?.exemplars}>
                    {" "}
                    · <span class={kbd}>shift</span>+drag excludes (SAM 3)
                  </Show>
                </li>
                <li>
                  <span class={kbd}>⌫</span> undo · <span class={kbd}>esc</span> clear
                </li>
              </ul>
            </section>
          </div>
        </aside>

        {/* center: stage */}
        <main class="flex min-h-0 min-w-0 flex-col gap-3">
          <div class="min-h-[320px] flex-1">
            <Stage
              src={stageSrc()}
              width={image()?.w ?? 1}
              height={image()?.h ?? 1}
              overlays={overlays()}
              points={mode() === "video" && frames().length ? [] : points()}
              boxes={mode() === "video" && frames().length ? [] : boxes()}
              negBoxes={mode() === "video" && frames().length ? [] : negBoxes()}
              busy={busy() || (tracking() ? `Tracking ${progress().done}/${progress().total}…` : "")}
              interactive={!tracking() && !useDetector() && !(mode() === "video" && frames().length > 0)}
              onPoint={addPoint}
              onBox={addBox}
              onFile={openFiles}
            />
          </div>
          <Show when={error()}>
            <div class={errorBox}>
              <Icon name="x" size={16} class="mt-0.5 flex-none text-rec" />
              <span class="min-w-0 flex-1 break-words">{error()}</span>
              <button class="cursor-pointer text-ink-400 hover:text-ink-100" onClick={() => setError("")}>
                <Icon name="x" size={14} />
              </button>
            </div>
          </Show>
          <Show when={mode() === "video" && frames().length > 0}>
            <div class={`${panel} flex-none flex-row items-center gap-3 px-3 py-2.5`}>
              <button class={`${btn} ${btnSmIcon}`} onClick={() => setPlaying((p) => !p)} title="Play / pause (space)">
                <Icon name={playing() ? "pause" : "play"} size={12} />
              </button>
              <input
                class="h-[18px] min-w-0 flex-1 cursor-pointer accent-[var(--color-accent)]"
                type="range"
                min={0}
                max={Math.max(0, frames().length - 1)}
                value={cur()}
                onInput={(e) => {
                  setPlaying(false);
                  setCur(+e.currentTarget.value);
                }}
              />
              <span class="flex-none font-mono text-[12px] text-ink-300">
                {cur() + 1} / {frames().length}
              </span>
              <button class={`${btn} ${btnSm}`} onClick={() => setFrames([])} title="Back to prompting on the first frame">
                Edit prompts
              </button>
            </div>
          </Show>
        </main>

        {/* right: results */}
        <aside class={`${panel} overflow-hidden`}>
          <div class={panelHead}>
            <span class={panelTitle}>{mode() === "video" ? "Tracking" : "Results"}</span>
            <Show when={mode() === "image" && session()}>
              <span class="ml-auto font-mono text-[11px] text-ink-400">
                encode {session()!.encode_ms.toFixed(0)} ms
                <Show when={resp()}> · decode {resp()!.decode_ms.toFixed(0)} ms</Show>
              </span>
            </Show>
          </div>
          <div class="min-h-0 flex-1 overflow-auto p-3">
            <Show
              when={mode() === "image"}
              fallback={
                <VideoPanel
                  source={source()}
                  fps={fps()}
                  setFps={setFps}
                  maxFrames={maxFrames()}
                  setMaxFrames={setMaxFrames}
                  detectEvery={detectEvery()}
                  setDetectEvery={setDetectEvery}
                  count={frameCount()}
                  hasPrompt={hasPrompt()}
                  isSam3={isSam3() || useDetector()}
                  tracking={tracking()}
                  progress={progress()}
                  frames={frames()}
                  cur={cur()}
                  onTrack={track}
                  onStop={stopTrack}
                  canVideo={!!caps()?.video}
                />
              }
            >
              <ImageResults resp={resp()} picked={picked()} setPicked={setPicked} hidden={hidden()} setHidden={setHidden} image={image()} hasImage={!!image()} hasPrompt={hasPrompt()} />
            </Show>
          </div>
        </aside>
      </div>
    </div>
  );
}

function ImageResults(props: {
  resp: SegmentResponse | null;
  picked: number | null;
  setPicked: (i: number | null) => void;
  hidden: number[];
  setHidden: (h: number[]) => void;
  image: Loaded | null;
  hasImage: boolean;
  hasPrompt: boolean;
}) {
  const toggle = (i: number) => props.setHidden(props.hidden.includes(i) ? props.hidden.filter((x) => x !== i) : [...props.hidden, i]);
  return (
    <Show
      when={props.resp}
      fallback={
        <div class="grid h-full place-items-center px-4 text-center leading-relaxed text-ink-400">
          {!props.hasImage ? "Open an image to begin." : props.hasPrompt ? "Waiting for the model…" : "Click the image, drag a box, or type a concept for SAM 3."}
        </div>
      }
    >
      {(r) => (
        <div class="flex flex-col gap-2">
          <div class="flex items-center justify-between text-[12px] text-ink-400">
            <span>
              {r().kind === "instances" ? `${r().results.length} instance${r().results.length === 1 ? "" : "s"}` : "Candidate masks"}
            </span>
            <Show when={r().object_score !== null}>
              <span class={r().object_score! > 0 ? "text-ok" : "text-warn"}>{r().object_score! > 0 ? "object present" : "no object"} · {r().object_score!.toFixed(1)}</span>
            </Show>
          </div>
          <Show when={r().results.length === 0}>
            <p class="rounded-lg border border-dashed border-ink-700 p-4 text-center text-ink-400">Nothing above the threshold. Lower it, or try another phrase.</p>
          </Show>
          <For each={r().results}>
            {(o, i) => {
              const idx = () => (typeof i === "function" ? (i as () => number)() : (i as number));
              const on = () => (r().kind === "masks" ? props.picked === idx() : !props.hidden.includes(idx()));
              return (
                <div
                  class="flex cursor-pointer items-center gap-2.5 rounded-xl border border-ink-750 bg-ink-850 px-3 py-2.5 transition-colors hover:border-ink-600 aria-pressed:border-accent/40"
                  aria-pressed={on() ? "true" : "false"}
                  onClick={() => (r().kind === "masks" ? props.setPicked(idx()) : toggle(idx()))}
                >
                  <span class="size-3 flex-none rounded-[4px]" style={{ background: colorOf(idx()), opacity: on() ? 1 : 0.3 }} />
                  <span class="min-w-0 flex-1">
                    <span class="flex items-baseline justify-between gap-2">
                      <span class="truncate text-ink-100">{o.label || (r().kind === "instances" ? `Instance ${idx() + 1}` : `Mask ${idx() + 1}`)}</span>
                      <span class="font-mono text-[12px] text-ink-200">{(o.score * 100).toFixed(1)}%</span>
                    </span>
                    <span class="mt-1.5 block h-1 overflow-hidden rounded-full bg-ink-700">
                      <span class="block h-full rounded-full" style={{ width: `${Math.max(2, Math.min(100, o.score * 100))}%`, background: colorOf(idx()) }} />
                    </span>
                    <span class="mt-1.5 block font-mono text-[11px] text-ink-400">{o.area.toLocaleString()} px{o.bbox ? ` · ${o.bbox.join(", ")}` : ""}</span>
                  </span>
                  <a
                    class="grid size-7 flex-none place-items-center rounded-md text-ink-400 hover:bg-ink-700 hover:text-ink-100"
                    href={`data:image/png;base64,${o.png}`}
                    download={`${(props.image?.name ?? "mask").replace(/\.[^.]+$/, "")}-mask-${idx() + 1}.png`}
                    title="Download the mask (PNG)"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Icon name="download" size={14} />
                  </a>
                </div>
              );
            }}
          </For>
        </div>
      )}
    </Show>
  );
}

function VideoPanel(props: {
  source: Source | null;
  fps: number;
  setFps: (v: number) => void;
  maxFrames: number;
  setMaxFrames: (v: number) => void;
  detectEvery: number;
  setDetectEvery: (v: number) => void;
  count: number;
  hasPrompt: boolean;
  isSam3: boolean;
  tracking: boolean;
  progress: { done: number; total: number };
  frames: TrackedFrame[];
  cur: number;
  onTrack: () => void;
  onStop: () => void;
  canVideo: boolean;
}) {
  const f = () => props.frames[props.cur];
  return (
    <Show when={props.source} fallback={<div class="grid h-full place-items-center px-4 text-center leading-relaxed text-ink-400">Drop a video clip, or select several frames, to track objects through it.</div>}>
      <div class="flex flex-col gap-3">
        <Show when={!props.canVideo}>
          <p class={notice}>The selected model has no video tracking. Pick a SAM 2 or SAM 3 model.</p>
        </Show>
        <Show when={props.source!.kind === "video"}>
          <Slider label="Frames per second" value={props.fps} min={1} max={30} step={1} digits={0} onChange={props.setFps} />
        </Show>
        <Slider label="Max frames" value={props.maxFrames} min={4} max={300} step={4} digits={0} onChange={props.setMaxFrames} />
        <Show when={props.isSam3}>
          <Slider label="Re-detect every N frames" value={props.detectEvery} min={1} max={30} step={1} digits={0} onChange={props.setDetectEvery} />
        </Show>
        <p class="text-[12px] leading-snug text-ink-400">
          {props.count} frame{props.count === 1 ? "" : "s"} will be tracked. Prompt on the first frame with points or a box
          {props.isSam3 ? ", or type a concept to follow every instance of it" : ""}.
        </p>
        <Show
          when={props.tracking}
          fallback={
            <button class={`${btnPrimary} !h-9`} disabled={!props.hasPrompt || !props.canVideo} onClick={props.onTrack}>
              <Icon name="play" size={12} />
              {props.frames.length ? "Track again" : "Track"}
            </button>
          }
        >
          <button class={`${btnStop} !h-9`} onClick={props.onStop}>
            <Icon name="stop" size={12} />
            Stop
          </button>
        </Show>
        <Show when={props.tracking || props.frames.length}>
          <div>
            <div class="mb-1.5 flex justify-between font-mono text-[11px] text-ink-400">
              <span class="flex items-center gap-1.5">
                <Show when={props.tracking}>
                  <Spinner size={11} />
                </Show>
                {props.progress.done} / {props.progress.total} frames
              </span>
              <Show when={f()}>
                <span>{f().ms.toFixed(0)} ms</span>
              </Show>
            </div>
            <div class="h-1.5 overflow-hidden rounded-full bg-ink-700">
              <div class="h-full rounded-full bg-gradient-to-r from-accent to-accent2 transition-[width]" style={{ width: `${props.progress.total ? (props.progress.done / props.progress.total) * 100 : 0}%` }} />
            </div>
          </div>
        </Show>
        <Show when={f()}>
          <div>
            <div class={sectionLabel}>Frame {props.cur + 1}</div>
            <Show when={f().objects.length === 0}>
              <p class="text-ink-400">No objects on this frame.</p>
            </Show>
            <div class="flex flex-col gap-1.5">
              <For each={f().objects}>
                {(o) => (
                  <div class="flex items-center gap-2.5 rounded-lg border border-ink-750 bg-ink-850 px-2.5 py-2">
                    <span class="size-3 flex-none rounded-[4px]" style={{ background: colorOf(o.id ?? 0), opacity: o.present ? 1 : 0.3 }} />
                    <span class="min-w-0 flex-1 truncate text-ink-100">{o.label ? `${o.label} #${(o.id ?? 0) + 1}` : `Object ${(o.id ?? 0) + 1}`}</span>
                    <span class={`font-mono text-[11.5px] ${o.present ? "text-ink-200" : "text-warn"}`}>{o.present ? `${o.area.toLocaleString()} px` : "absent"}</span>
                    <a
                      class="grid size-6 place-items-center rounded-md text-ink-400 hover:bg-ink-700 hover:text-ink-100"
                      href={`data:image/png;base64,${o.png}`}
                      download={`frame-${props.cur + 1}-object-${(o.id ?? 0) + 1}.png`}
                      title="Download the mask (PNG)"
                    >
                      <Icon name="download" size={13} />
                    </a>
                  </div>
                )}
              </For>
            </div>
          </div>
        </Show>
      </div>
    </Show>
  );
}
