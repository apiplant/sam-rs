/* The image canvas: shows the image, tints masks over it, and turns clicks and drags into prompts. */
import { For, Show, createEffect, createSignal, onSettled } from "solid-js";
import type { Box, Point } from "./api";
import { Icon } from "./ui";
import { btnPrimary } from "./styles";

export interface Overlay {
  png: string;
  color: string;
  label?: string;
  bbox?: Box | null;
}

const tinted = new Map<string, Promise<HTMLCanvasElement>>();

/** The mask PNG recoloured: `color` where the mask is on, transparent elsewhere. Cached per (png, color). */
function tint(png: string, color: string): Promise<HTMLCanvasElement> {
  const key = `${color}:${png.length}:${png.slice(-48)}:${png.slice(60, 120)}`;
  let p = tinted.get(key);
  if (!p) {
    p = (async () => {
      const img = new Image();
      img.src = `data:image/png;base64,${png}`;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const g = c.getContext("2d", { willReadFrequently: true })!;
      g.drawImage(img, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height);
      const r = parseInt(color.slice(1, 3), 16), gr = parseInt(color.slice(3, 5), 16), b = parseInt(color.slice(5, 7), 16);
      for (let i = 0; i < d.data.length; i += 4) {
        const on = d.data[i] > 127;
        d.data[i] = r;
        d.data[i + 1] = gr;
        d.data[i + 2] = b;
        d.data[i + 3] = on ? 120 : 0;
      }
      g.putImageData(d, 0, 0);
      return c;
    })();
    tinted.set(key, p);
    if (tinted.size > 200) tinted.delete(tinted.keys().next().value!);
  }
  return p;
}

export function Stage(props: {
  src: string | null;
  width: number;
  height: number;
  overlays: Overlay[];
  points: Point[];
  boxes: Box[];
  negBoxes: Box[];
  busy?: string;
  interactive: boolean;
  onPoint: (p: Point) => void;
  onBox: (b: Box, negative: boolean) => void;
  onFile: (files: File[]) => void;
}) {
  let host!: HTMLDivElement;
  let canvas!: HTMLCanvasElement;
  let fileInput!: HTMLInputElement;
  const [size, setSize] = createSignal({ w: 1, h: 1 });
  const [drag, setDrag] = createSignal<{ x0: number; y0: number; x1: number; y1: number; neg: boolean } | null>(null);
  const [over, setOver] = createSignal(false);

  onSettled(() => {
    const ro = new ResizeObserver(() => setSize({ w: host.clientWidth, h: host.clientHeight }));
    ro.observe(host);
    setSize({ w: host.clientWidth, h: host.clientHeight });
    return () => ro.disconnect();
  });

  // displayed size of the image, fitted into the host
  const fit = () => {
    const { w, h } = size();
    const s = Math.min((w - 24) / props.width, (h - 24) / props.height, 4);
    const k = Math.max(0.05, s);
    return { w: props.width * k, h: props.height * k, k };
  };

  let token = 0;
  createEffect(
    () => ({ overlays: props.overlays, w: props.width, h: props.height }),
    ({ overlays, w, h }) => {
      const mine = ++token;
      void (async () => {
        const layers = await Promise.all(overlays.map((o) => tint(o.png, o.color)));
        if (mine !== token || !canvas) return;
        canvas.width = w;
        canvas.height = h;
        const g = canvas.getContext("2d")!;
        g.clearRect(0, 0, w, h);
        for (const l of layers) g.drawImage(l, 0, 0, w, h);
      })();
    },
  );

  const toImage = (e: PointerEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const clamp = (v: number, m: number) => Math.min(m, Math.max(0, v));
    return { x: clamp(((e.clientX - r.left) / r.width) * props.width, props.width), y: clamp(((e.clientY - r.top) / r.height) * props.height, props.height) };
  };

  let down: { x: number; y: number; sx: number; sy: number; button: number; shift: boolean; alt: boolean } | null = null;

  const onDown = (e: PointerEvent) => {
    if (!props.interactive) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const p = toImage(e);
    down = { ...p, sx: e.clientX, sy: e.clientY, button: e.button, shift: e.shiftKey, alt: e.altKey };
  };
  const onMove = (e: PointerEvent) => {
    if (!down) return;
    if (Math.hypot(e.clientX - down.sx, e.clientY - down.sy) < 5 && !drag()) return;
    const p = toImage(e);
    setDrag({ x0: down.x, y0: down.y, x1: p.x, y1: p.y, neg: down.shift });
  };
  const onUp = (e: PointerEvent) => {
    if (!down) return;
    const d = drag();
    const was = down;
    down = null;
    setDrag(null);
    if (d) {
      const b: Box = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
      if (b[2] - b[0] > 3 && b[3] - b[1] > 3) props.onBox(b, d.neg);
      return;
    }
    const p = toImage(e);
    props.onPoint({ x: p.x, y: p.y, label: was.button === 2 || was.alt ? 0 : 1 });
  };

  const pick = (e: Event) => {
    const input = e.currentTarget as HTMLInputElement;
    if (input.files?.length) props.onFile([...input.files]);
    input.value = "";
  };

  const k = () => 1 / fit().k; // image px per screen px: keeps markers a constant size on screen

  return (
    <div
      ref={host}
      class="checker relative flex size-full items-center justify-center overflow-hidden rounded-xl border border-ink-700"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const files = [...(e.dataTransfer?.files ?? [])];
        if (files.length) props.onFile(files);
      }}
    >
      <input ref={fileInput} type="file" accept="image/*,video/*" multiple class="hidden" onChange={pick} />
      <Show
        when={props.src}
        fallback={
          <div class="flex max-w-sm flex-col items-center gap-3 px-6 text-center">
            <div class="grid size-14 place-items-center rounded-2xl border border-ink-700 bg-ink-850 text-accent">
              <Icon name="image" size={26} />
            </div>
            <p class="text-base text-ink-100">Drop an image to segment</p>
            <p class="leading-normal text-ink-400">Drag a file here, paste from the clipboard, or choose one. In Video mode, drop a clip or a set of frames.</p>
            <button class={`${btnPrimary} !h-9 !px-4`} onClick={() => fileInput.click()}>
              <Icon name="upload" size={14} />
              Choose a file
            </button>
          </div>
        }
      >
        <div
          class={`relative select-none touch-none ${props.interactive ? "cursor-crosshair" : ""}`}
          style={{ width: `${fit().w}px`, height: `${fit().h}px` }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onContextMenu={(e) => e.preventDefault()}
        >
          <img src={props.src!} alt="" class="pointer-events-none absolute inset-0 size-full rounded-sm" draggable={false} />
          <canvas ref={canvas} class="pointer-events-none absolute inset-0 size-full rounded-sm" />
          <svg class="pointer-events-none absolute inset-0 size-full overflow-visible" viewBox={`0 0 ${props.width} ${props.height}`}>
            <For each={props.overlays}>
              {(o) => (
                <Show when={o.bbox}>
                  <g>
                    <rect x={o.bbox![0]} y={o.bbox![1]} width={o.bbox![2] - o.bbox![0]} height={o.bbox![3] - o.bbox![1]} fill="none" stroke={o.color} stroke-width={1.5 * k()} />
                    <Show when={o.label}>
                      <g transform={`translate(${o.bbox![0]} ${Math.max(o.bbox![1] - 18 * k(), 0)})`}>
                        <rect width={(o.label!.length * 6.6 + 10) * k()} height={17 * k()} rx={3 * k()} fill={o.color} />
                        <text x={5 * k()} y={12.5 * k()} font-size={`${11 * k()}`} font-family="ui-monospace, monospace" fill="#07080b" font-weight="600">
                          {o.label}
                        </text>
                      </g>
                    </Show>
                  </g>
                </Show>
              )}
            </For>
            <For each={props.boxes}>
              {(b) => <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="#73e0c91a" stroke="#73e0c9" stroke-width={2 * k()} stroke-dasharray={`${6 * k()} ${4 * k()}`} />}
            </For>
            <For each={props.negBoxes}>
              {(b) => <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="#ff5d731a" stroke="#ff5d73" stroke-width={2 * k()} stroke-dasharray={`${6 * k()} ${4 * k()}`} />}
            </For>
            <Show when={drag()}>
              {(d) => (
                <rect
                  x={Math.min(d().x0, d().x1)}
                  y={Math.min(d().y0, d().y1)}
                  width={Math.abs(d().x1 - d().x0)}
                  height={Math.abs(d().y1 - d().y0)}
                  fill="none"
                  stroke={d().neg ? "#ff5d73" : "#ffffff"}
                  stroke-width={1.5 * k()}
                  stroke-dasharray={`${5 * k()} ${3 * k()}`}
                />
              )}
            </Show>
            <For each={props.points}>
              {(p) => (
                <g>
                  <circle cx={p.x} cy={p.y} r={8 * k()} fill={p.label ? "#3ddc97" : "#ff5d73"} stroke="#07080b" stroke-width={2 * k()} />
                  <path
                    d={p.label ? `M${p.x - 3.5 * k()} ${p.y}h${7 * k()}M${p.x} ${p.y - 3.5 * k()}v${7 * k()}` : `M${p.x - 3.5 * k()} ${p.y}h${7 * k()}`}
                    stroke="#07080b"
                    stroke-width={2 * k()}
                    stroke-linecap="round"
                  />
                </g>
              )}
            </For>
          </svg>
        </div>
        <Show when={props.busy}>
          <div class="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
            <span class="inline-flex items-center gap-2 rounded-full border border-ink-600 bg-ink-900/90 px-3 py-1.5 text-ink-100 shadow-lg backdrop-blur">
              <span class="size-2 animate-pulse-soft rounded-full bg-accent" />
              {props.busy}
            </span>
          </div>
        </Show>
        <button
          class="absolute right-3 top-3 grid size-8 cursor-pointer place-items-center rounded-lg border border-ink-600 bg-ink-900/85 text-ink-300 backdrop-blur hover:text-ink-100"
          title="Open another file"
          onClick={() => fileInput.click()}
        >
          <Icon name="upload" size={14} />
        </button>
      </Show>
      <Show when={over()}>
        <div class="pointer-events-none absolute inset-2 grid place-items-center rounded-lg border-2 border-dashed border-accent bg-accent/10 text-base text-accent">Drop to open</div>
      </Show>
    </div>
  );
}
