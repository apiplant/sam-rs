/** An illustration of what the model returns: labelled boxes and a point over a scene, drawn in SVG so it
 * follows the site theme and ships no third-party image. It is a drawing, not model output. */

const BOXES = [
  { label: "person", x: 92, y: 118, w: 54, h: 132, hue: "#e6194b" },
  { label: "person", x: 168, y: 130, w: 46, h: 120, hue: "#e6194b" },
  { label: "car", x: 262, y: 168, w: 150, h: 82, hue: "#3cb44b" },
  { label: "bicycle", x: 440, y: 186, w: 88, h: 64, hue: "#0082c8" },
];

export function Visual() {
  return (
    <div class="min-w-0 overflow-hidden rounded-xl border border-line bg-surface">
      <div class="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span class="font-mono text-xs text-faint">Parallel Box Decoding · illustration</span>
        <span class="font-mono text-xs text-faint">&lt;ref&gt;car&lt;/ref&gt;&lt;box&gt;…</span>
      </div>
      <svg viewBox="0 0 600 300" class="block w-full" role="img" aria-label="Illustration of detection boxes drawn over a street scene">
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
        {/* detections */}
        {BOXES.map((b) => (
          <g>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={b.hue} stroke-width="2.5" stroke-dasharray="6 3" rx="2" />
            <rect x={b.x} y={b.y - 18} width={b.label.length * 7.4 + 12} height="18" fill={b.hue} rx="2" />
            <text x={b.x + 6} y={b.y - 5} font-size="11" font-family="ui-monospace, monospace" fill="#fff">
              {b.label}
            </text>
          </g>
        ))}
        {/* a point, as "point" mode returns */}
        <g>
          <circle cx="554" cy="130" r="14" fill="none" class="stroke-accent" stroke-width="2" />
          <circle cx="554" cy="130" r="4" class="fill-accent" />
          <text x="478" y="40" font-size="11" font-family="ui-monospace, monospace" class="fill-accent">
            point: the window
          </text>
        </g>
      </svg>
    </div>
  );
}
