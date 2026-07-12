// charts.jsx — hand-rolled SVG charts, no Chart.js dependency.
// Sparkline: last 60s of confidence, live during the interview.
// TrendChart: full-session area chart for the report.
// Radar: five-axis behavioral profile.

function pathFrom(points) {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`)
    .join(" ");
}

export function Sparkline({ data, width = 260, height = 54 }) {
  const d = data.slice(-60);
  if (d.length < 2)
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--cream-dim)",
          fontSize: 12,
        }}
      >
        Gathering signal…
      </div>
    );
  const pts = d.map((v, i) => [
    (i / (d.length - 1)) * width,
    height - (v / 100) * (height - 6) - 3,
  ]);
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      <path
        d={pathFrom(pts)}
        fill="none"
        stroke="var(--brass-bright)"
        strokeWidth="2"
        strokeLinejoin="round"
        opacity="0.9"
      />
      <circle
        cx={pts[pts.length - 1][0]}
        cy={pts[pts.length - 1][1]}
        r="3.2"
        fill="var(--brass-bright)"
      >
        <animate attributeName="opacity" values="1;0.35;1" dur="1.6s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

export function TrendChart({ data, width = 720, height = 220 }) {
  if (!data || data.length < 2) return null;
  const px = 8;
  const pts = data.map((v, i) => [
    px + (i / (data.length - 1)) * (width - px * 2),
    height - 16 - (v / 100) * (height - 40),
  ]);
  const line = pathFrom(pts);
  const area = `${line} L${pts[pts.length - 1][0]},${height - 14} L${pts[0][0]},${height - 14} Z`;
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      <defs>
        <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c79b5b" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#c79b5b" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[25, 50, 75].map((g) => {
        const y = height - 16 - (g / 100) * (height - 40);
        return (
          <g key={g}>
            <line x1={px} x2={width - px} y1={y} y2={y} stroke="rgba(255,245,228,0.06)" />
            <text x={px} y={y - 4} fontSize="9" fill="var(--cream-dim)" className="num">
              {g}
            </text>
          </g>
        );
      })}
      <path d={area} fill="url(#trendFill)" />
      <path d={line} fill="none" stroke="var(--brass-bright)" strokeWidth="2.4" strokeLinejoin="round" />
    </svg>
  );
}

export function Radar({ axes, size = 300 }) {
  // axes: [{ label, value 0–100 }]
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.36;
  const n = axes.length;
  const pt = (i, r) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  };
  const ring = (frac) =>
    pathFrom(axes.map((_, i) => pt(i, R * frac))) + " Z";
  const valuePath =
    pathFrom(axes.map((a, i) => pt(i, R * Math.max(0.04, a.value / 100)))) + " Z";

  return (
    <svg width="100%" viewBox={`0 0 ${size} ${size}`} style={{ display: "block" }}>
      {[0.33, 0.66, 1].map((f) => (
        <path key={f} d={ring(f)} fill="none" stroke="rgba(255,245,228,0.08)" />
      ))}
      {axes.map((_, i) => {
        const [x, y] = pt(i, R);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,245,228,0.06)" />;
      })}
      <path d={valuePath} fill="rgba(199,155,91,0.22)" stroke="var(--brass-bright)" strokeWidth="2" strokeLinejoin="round" />
      {axes.map((a, i) => {
        const [x, y] = pt(i, R * Math.max(0.04, a.value / 100));
        return <circle key={i} cx={x} cy={y} r="3" fill="var(--brass-bright)" />;
      })}
      {axes.map((a, i) => {
        const [x, y] = pt(i, R + 24);
        return (
          <text
            key={i}
            x={x}
            y={y}
            fontSize="10.5"
            fontWeight="600"
            fill="var(--cream-dim)"
            textAnchor="middle"
          >
            {a.label}
            <tspan x={x} dy="12" fill="var(--brass)" className="num">
              {a.value == null ? "—" : Math.round(a.value)}
            </tspan>
          </text>
        );
      })}
    </svg>
  );
}
