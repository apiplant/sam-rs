/** An illustration of what the models return: a click on a scene and the masks it selects, drawn in SVG so it
 * follows the site theme and ships no third-party image. It is a drawing, not model output. */

const MASKS = [
  { label: "wheel 0.94", hue: "#e6194b", shape: <><circle cx="306" cy="246" r="14" /><circle cx="368" cy="246" r="14" /></> },
  { label: "person 0.97", hue: "#3cb44b", shape: <><rect x="106" y="124" width="26" height="128" rx="10" /><rect x="179" y="136" width="24" height="116" rx="10" /></> },
];

export function Visual() {
  return (
    <div class="min-w-0 overflow-hidden rounded-xl border border-line bg-surface">
      <div class="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span class="font-mono text-xs text-faint">point → mask · illustration</span>
        <span class="font-mono text-xs text-faint">--point 340,225</span>
      </div>
      <svg viewBox="0 0 600 300" class="block w-full" role="img" aria-label="Illustration of a clicked point selecting a car, and concept masks over a street scene">
        {/* sky and ground */}
        <rect width="600" height="300" class="fill-surface-2" />
        <rect y="250" width="600" height="50" class="fill-surface-3" />
        {/* buildings */}
        <g class="fill-surface-3">
          <rect x="20" y="70" width="90" height="180" />
          <rect x="130" y="100" width="70" height="150" />
          <rect x="470" y="60" width="110" height="190" />
        </g>
        <g class="fill-surface-2">
          <rect x="34" y="86" width="14" height="18" />
          <rect x="60" y="86" width="14" height="18" />
          <rect x="34" y="120" width="14" height="18" />
          <rect x="60" y="120" width="14" height="18" />
          <rect x="486" y="76" width="16" height="20" />
          <rect x="516" y="76" width="16" height="20" />
          <rect x="546" y="76" width="16" height="20" />
        </g>
        {/* people */}
        <g class="fill-faint">
          <circle cx="119" cy="134" r="9" />
          <rect x="108" y="146" width="22" height="64" rx="8" />
          <rect x="110" y="208" width="8" height="42" />
          <rect x="121" y="208" width="8" height="42" />
          <circle cx="191" cy="144" r="8" />
          <rect x="181" y="154" width="20" height="58" rx="8" />
          <rect x="183" y="210" width="7" height="40" />
          <rect x="192" y="210" width="7" height="40" />
        </g>
        {/* car */}
        <g class="fill-muted">
          <rect x="272" y="206" width="130" height="38" rx="9" />
          <path d="M296 206l18-28h56l18 28z" />
        </g>
        <g class="fill-canvas">
          <circle cx="306" cy="246" r="13" />
          <circle cx="368" cy="246" r="13" />
        </g>
        {/* bicycle */}
        <g fill="none" stroke="currentColor" stroke-width="3" class="text-faint">
          <circle cx="462" cy="228" r="20" />
          <circle cx="510" cy="228" r="20" />
          <path d="M462 228l22-30h16l10 30M484 198l-6-8h-10" />
        </g>
        {/* the object selected by a click */}
        <g class="fill-accent" fill-opacity="0.35" stroke-width="2.5">
          <rect x="272" y="206" width="130" height="38" rx="9" class="stroke-accent" />
          <path d="M296 206l18-28h56l18 28z" class="stroke-accent" />
        </g>
        <g>
          <circle cx="340" cy="212" r="12" fill="none" class="stroke-ink" stroke-width="2" />
          <circle cx="340" cy="212" r="3.5" class="fill-ink" />
        </g>
        {/* concept masks, as SAM 3 returns for a text prompt */}
        {MASKS.map((m) => (
          <g fill={m.hue} fill-opacity="0.45" stroke={m.hue} stroke-width="2">
            {m.shape}
          </g>
        ))}
        <g font-size="11" font-family="ui-monospace, monospace">
          <rect x="92" y="104" width="86" height="18" rx="2" fill="#3cb44b" />
          <text x="98" y="117" fill="#fff">person 0.97</text>
          <rect x="268" y="262" width="76" height="18" rx="2" fill="#e6194b" />
          <text x="274" y="275" fill="#fff">wheel 0.94</text>
          <text x="404" y="170" class="fill-accent">click → mask 0.95</text>
        </g>
      </svg>
    </div>
  );
}
