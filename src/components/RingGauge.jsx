// RingGauge.jsx — the signature element. An SVG ring whose stroke sweeps
// with the live confidence score and whose color breathes from ember →
// brass → sage as the score climbs. Replaces v1's water-fill circle.

const R = 52;
const CIRC = 2 * Math.PI * R;

function colorFor(score) {
  if (score >= 72) return ["#a8e0bd", "#7fc99a"];
  if (score >= 48) return ["#e8c98a", "#c79b5b"];
  return ["#f0937f", "#e06a5a"];
}

export default function RingGauge({ score, label = "Confidence", size = 168 }) {
  const s = Math.max(0, Math.min(100, score));
  const [c1, c2] = colorFor(s);
  const offset = CIRC * (1 - s / 100);

  return (
    <div
      style={{
        position: "relative",
        width: size,
        height: size,
        margin: "0 auto",
        filter: `drop-shadow(0 0 18px ${c2}44)`,
      }}
    >
      <svg viewBox="0 0 120 120" width={size} height={size}>
        <defs>
          <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={c1}>
              <animate
                attributeName="stop-color"
                values={`${c1};${c2};${c1}`}
                dur="4s"
                repeatCount="indefinite"
              />
            </stop>
            <stop offset="100%" stopColor={c2} />
          </linearGradient>
        </defs>
        {/* track */}
        <circle
          cx="60"
          cy="60"
          r={R}
          fill="none"
          stroke="rgba(255,245,228,0.08)"
          strokeWidth="7"
        />
        {/* live sweep */}
        <circle
          cx="60"
          cy="60"
          r={R}
          fill="none"
          stroke="url(#ringGrad)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
          style={{
            transition:
              "stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1)",
          }}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 2,
        }}
      >
        <span
          className="display num"
          style={{
            fontSize: size * 0.24,
            lineHeight: 1,
            color: c1,
            transition: "color 0.8s",
          }}
        >
          {Math.round(s)}
        </span>
        <span
          style={{
            fontSize: 10.5,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--cream-dim)",
            fontWeight: 600,
          }}
        >
          {label}
        </span>
      </div>
    </div>
  );
}
