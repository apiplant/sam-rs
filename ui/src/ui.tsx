/* Small shared building blocks: icons, a toggle, a slider and a segmented control. */
import { For, Show } from "solid-js";
import { slider, switchCls } from "./styles";

const PATHS: Record<string, string> = {
  image: "M4 5h16v14H4zM4 16l4.5-4.5 4 4 3-3L20 16M9 9.5h.01",
  video: "M4 6h11a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4zM16 10l5-3v10l-5-3",
  upload: "M12 16V5m0 0L7.5 9.5M12 5l4.5 4.5M5 19h14",
  download: "M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14",
  x: "M6 6l12 12M18 6L6 18",
  check: "M5 12.5l4.5 4.5L19 7.5",
  play: "M7 4.5v15l12-7.5z",
  pause: "M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z",
  stop: "M6 6h12v12H6z",
  trash: "M5 7h14M10 7V5h4v2m-6 0l.7 12h6.6L16 7",
  undo: "M9 7L4 12l5 5M4 12h10a6 6 0 0 1 0 12",
  point: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v3M12 19v3M2 12h3M19 12h3",
  box: "M5 5h14v14H5zM3 3h4v4H3zM17 3h4v4h-4zM3 17h4v4H3zM17 17h4v4h-4z",
  text: "M5 6V4h14v2M12 4v16M9 20h6",
  wand: "M15 4l1 2.5L18.5 7.5 16 8.5 15 11l-1-2.5-2.5-1L14 6.5zM5 19l9-9m-9 9l2-2",
  cpu: "M8 8h8v8H8zM4 9h2M4 15h2M18 9h2M18 15h2M9 4v2M15 4v2M9 18v2M15 18v2",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  copy: "M9 9h10v11H9zM5 15V4h10",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
  eyeoff: "M3 3l18 18M10.6 6.1A9.6 9.6 0 0 1 12 6c6 0 10 6 10 6a17 17 0 0 1-3.1 3.7M6.6 7.6A17 17 0 0 0 2 12s4 7 10 7a9.7 9.7 0 0 0 4.2-1",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  book: "M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h10",
  lock: "M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z",
  cloud: "M7 18a4 4 0 0 1-.5-8A6 6 0 0 1 18 9.5a4 4 0 0 1-.5 8.5z",
  refresh: "M20 11a8 8 0 0 0-14.5-4M4 4v4h4M4 13a8 8 0 0 0 14.5 4M20 20v-4h-4",
};

export function Icon(props: { name: string; size?: number; class?: string }) {
  const filled = () => props.name === "play" || props.name === "pause" || props.name === "stop";
  return (
    <svg
      class={props.class}
      width={props.size ?? 15}
      height={props.size ?? 15}
      viewBox="0 0 24 24"
      fill={filled() ? "currentColor" : "none"}
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[props.name] ?? ""} />
    </svg>
  );
}

export function Logo(props: { size?: number }) {
  const s = props.size ?? 24;
  return (
    <svg width={s} height={s} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#73e0c9" />
          <stop offset="1" stop-color="#7c8cff" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="#15171f" />
      <path d="M8 13V9a1 1 0 0 1 1-1h4M24 13V9a1 1 0 0 0-1-1h-4M8 19v4a1 1 0 0 0 1 1h4M24 19v4a1 1 0 0 1-1 1h-4" stroke="url(#lg)" stroke-width="2.2" fill="none" stroke-linecap="round" />
      <circle cx="16" cy="16" r="3.2" fill="url(#lg)" />
    </svg>
  );
}

export function Toggle(props: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <div class="flex items-center justify-between gap-3 py-1.5">
      <span class="min-w-0">
        <span class="block text-ink-100">{props.label}</span>
        <Show when={props.hint}>
          <span class="block text-[12px] leading-snug text-ink-400">{props.hint}</span>
        </Show>
      </span>
      <button class={switchCls} role="switch" aria-checked={props.checked ? "true" : "false"} aria-label={props.label} onClick={() => props.onChange(!props.checked)} />
    </div>
  );
}

export function Slider(props: { label: string; value: number; min: number; max: number; step: number; digits?: number; /** show an editable number box instead of the read-out */ numeric?: boolean; onChange: (v: number) => void }) {
  const fill = () => `${((props.value - props.min) / (props.max - props.min)) * 100}%`;
  return (
    <label class="block py-1.5">
      <span class="mb-0.5 flex justify-between text-ink-300">
        {props.label}
        <Show when={props.numeric} fallback={<span class="font-mono text-[12px] text-ink-100">{props.value.toFixed(props.digits ?? 2)}</span>}>
          <input
            class="h-6 w-[68px] rounded-md border border-ink-700 bg-ink-850 px-1.5 text-right font-mono text-[12px] text-ink-100 hover:border-ink-600 focus:border-accent focus:outline-none"
            type="number"
            min={props.min}
            max={props.max}
            step={props.step}
            value={props.value.toFixed(props.digits ?? 2)}
            aria-label={props.label}
            onChange={(e) => {
              // a typed value may be any number in range, not just a multiple of the slider's step
              const v = parseFloat(e.currentTarget.value);
              const shown = Number.isFinite(v) ? Math.min(props.max, Math.max(props.min, v)) : props.value;
              if (Number.isFinite(v)) props.onChange(shown);
              e.currentTarget.value = shown.toFixed(props.digits ?? 2);
            }}
          />
        </Show>
      </span>
      <input
        class={slider}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        style={{ "--fill": fill() }}
        onInput={(e) => props.onChange(+e.currentTarget.value)}
      />
    </label>
  );
}

export function Segmented<T extends string>(props: { value: T; options: { id: T; label: string; icon?: string }[]; onChange: (v: T) => void }) {
  return (
    <div class="grid auto-cols-fr grid-flow-col gap-1 rounded-[10px] border border-ink-700 bg-ink-850 p-1" role="tablist">
      <For each={props.options}>
        {(o) => (
          <button
            role="tab"
            aria-selected={props.value === o.id ? "true" : "false"}
            class="inline-flex h-7 cursor-pointer items-center justify-center gap-1.5 rounded-[7px] text-ink-300 transition-colors hover:text-ink-100 aria-selected:bg-ink-700 aria-selected:text-ink-100"
            onClick={() => props.onChange(o.id)}
          >
            <Show when={o.icon}>
              <Icon name={o.icon!} size={14} />
            </Show>
            {o.label}
          </button>
        )}
      </For>
    </div>
  );
}

export function Spinner(props: { size?: number }) {
  const s = props.size ?? 14;
  return <span class="inline-block animate-spin rounded-full border-2 border-ink-600 border-t-accent" style={{ width: `${s}px`, height: `${s}px` }} />;
}
