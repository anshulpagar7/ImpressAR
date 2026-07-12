import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Topbar from "../components/Topbar";
import { TrendChart, Radar } from "../components/charts";
import { getLatestReport, getPreviousReport, updateLatestReport } from "../lib/store";
import { evaluateAnswers } from "../lib/evaluate";

function CountUp({ value, size = 68 }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf;
    const t0 = performance.now();
    const run = (t) => {
      const p = Math.min(1, (t - t0) / 1400);
      setV(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(run);
    };
    raf = requestAnimationFrame(run);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return (
    <span className="display num" style={{ fontSize: size, color: "var(--brass-bright)", lineHeight: 1 }}>
      {Math.round(v)}
    </span>
  );
}

const fade = (i) => ({
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, delay: 0.1 + i * 0.1, ease: [0.22, 1, 0.36, 1] },
});

export default function Report() {
  const nav = useNavigate();
  const [r, setR] = useState(getLatestReport());
  const prev = getPreviousReport();
  const [evalState, setEvalState] = useState(
    r?.evaluation ? "done" : r?.transcripts?.length ? "loading" : "none"
  );
  const [evalError, setEvalError] = useState(null);

  useEffect(() => {
    if (!r || r.evaluation || !r.transcripts?.length) return;
    let alive = true;
    evaluateAnswers(r.role, r.transcripts)
      .then((ev) => {
        if (!alive) return;
        setR(updateLatestReport({ evaluation: ev }));
        setEvalState("done");
      })
      .catch((e) => {
        if (!alive) return;
        setEvalError(e);
        setEvalState(e.code === "not_configured" ? "unconfigured" : "failed");
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!r) {
    return (
      <div className="page">
        <Topbar />
        <div className="glass" style={{ maxWidth: 480, margin: "80px auto", padding: 36, textAlign: "center" }}>
          <h2 style={{ marginBottom: 10 }}>No sessions yet</h2>
          <p style={{ color: "var(--cream-dim)", fontSize: 14, marginBottom: 22 }}>
            Your first report appears here after you complete an interview.
          </p>
          <button className="btn btn-brass" onClick={() => nav("/interview")}>Start your first session</button>
        </div>
      </div>
    );
  }

  const delta = prev ? Math.round((r.avg - prev.avg) * 10) / 10 : null;
  const axes = [
    { label: "Posture", value: r.posture },
    { label: "Eye contact", value: r.eye },
    { label: "Stillness", value: r.stillness },
    { label: "Calm hands", value: r.calm },
    { label: "Speech", value: r.speechScore ?? 50 },
  ];
  const verdict =
    r.avg > 80
      ? "Interview-ready. This is the composure panels remember."
      : r.avg > 60
        ? "Solid foundation — one more week of sessions sharpens the edges."
        : "Every session from here is measurable progress. Run another.";

  return (
    <div className="page">
      <Topbar />

      <div style={{ maxWidth: 1080, margin: "24px auto 0", display: "grid", gap: 22 }}>
        {/* hero score */}
        <motion.div {...fade(0)} className="glass" style={{ padding: "44px 36px", textAlign: "center" }}>
          <p className="eyebrow" style={{ marginBottom: 14 }}>Session report · {new Date(r.date).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</p>
          <CountUp value={r.avg} />
          <div style={{ marginTop: 10, display: "flex", justifyContent: "center", alignItems: "center", gap: 12 }}>
            <span style={{ color: "var(--cream-dim)", fontSize: 14 }}>{verdict}</span>
            {delta != null && (
              <span className={`pill ${delta >= 0 ? "good" : "bad"}`}>
                {delta >= 0 ? "▲" : "▼"} {Math.abs(delta)} vs last session
              </span>
            )}
          </div>
        </motion.div>

        {/* radar + trend */}
        <div style={{ display: "grid", gridTemplateColumns: "minmax(280px, 1fr) minmax(0, 1.7fr)", gap: 22 }}>
          <motion.div {...fade(1)} className="glass" style={{ padding: 24 }}>
            <p className="eyebrow" style={{ marginBottom: 10 }}>Behavioral profile</p>
            <Radar axes={axes} />
          </motion.div>
          <motion.div {...fade(2)} className="glass" style={{ padding: 24 }}>
            <p className="eyebrow" style={{ marginBottom: 10 }}>Confidence over the session</p>
            <TrendChart data={r.timeline} />
            {r.pace != null && (
              <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
                <div className="pill idle"><span className="num" style={{ color: "var(--brass-bright)", fontWeight: 700 }}>{r.pace}</span> words / min</div>
                <div className="pill idle"><span className="num" style={{ color: "var(--brass-bright)", fontWeight: 700 }}>{r.fillerCount}</span> filler words</div>
                <div className="pill idle"><span className="num" style={{ color: "var(--brass-bright)", fontWeight: 700 }}>{Math.floor(r.durationSec / 60)}m {r.durationSec % 60}s</span> duration</div>
              </div>
            )}
          </motion.div>
        </div>

        {/* per-question */}
        {r.questions?.length > 0 && (
          <motion.div {...fade(3)} className="glass" style={{ padding: 24 }}>
            <p className="eyebrow" style={{ marginBottom: 14 }}>Question by question</p>
            <div style={{ display: "grid", gap: 10 }}>
              {r.questions.map((q) => (
                <div
                  key={q.index}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 14,
                    alignItems: "center",
                    padding: "12px 16px",
                    borderRadius: 10,
                    background: "rgba(10,6,3,0.35)",
                    border: "1px solid var(--glass-line)",
                  }}
                >
                  <span style={{ fontSize: 13.5, color: "var(--cream)" }}>
                    <span style={{ color: "var(--brass)", fontWeight: 700, marginRight: 10 }} className="num">
                      Q{q.index + 1}
                    </span>
                    {q.text}
                  </span>
                  <span
                    className="num"
                    style={{
                      fontWeight: 700,
                      fontSize: 15,
                      color: q.avg == null ? "var(--cream-dim)" : q.avg >= 72 ? "var(--sage)" : q.avg >= 48 ? "var(--brass-bright)" : "var(--ember)",
                    }}
                  >
                    {q.avg == null ? "—" : Math.round(q.avg)}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* answer evaluation (LLM) */}
        {evalState !== "none" && (
          <motion.div {...fade(3.5)} className="glass" style={{ padding: 24 }}>
            <p className="eyebrow" style={{ marginBottom: 14 }}>Answer quality — AI evaluation</p>

            {evalState === "loading" && (
              <p style={{ color: "var(--cream-dim)", fontSize: 14 }}>
                <span style={{ color: "var(--brass-bright)" }}>●</span> Your coach is reading the transcripts…
              </p>
            )}

            {evalState === "unconfigured" && (
              <p style={{ color: "var(--cream-dim)", fontSize: 13.5, lineHeight: 1.6 }}>
                Answer evaluation is off — add a free Gemini API key (aistudio.google.com) to <code>.env</code> and run <code>npm run api</code>. Delivery analysis above works without it.
              </p>
            )}

            {evalState === "failed" && (
              <p style={{ color: "var(--cream-dim)", fontSize: 13.5 }}>
                Evaluation didn't complete ({String(evalError?.message || "network error")}). Your delivery report above is unaffected.
              </p>
            )}

            {evalState === "done" && r.evaluation && (
              <div style={{ display: "grid", gap: 14 }}>
                <p style={{ fontSize: 14, lineHeight: 1.65, color: "var(--cream)" }}>{r.evaluation.overall?.summary}</p>
                {r.evaluation.overall?.top_fixes?.length > 0 && (
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {r.evaluation.overall.top_fixes.map((f, i) => (
                      <span key={i} className="pill idle" style={{ fontWeight: 500 }}>{f}</span>
                    ))}
                  </div>
                )}
                <div style={{ display: "grid", gap: 10, marginTop: 4 }}>
                  {r.evaluation.answers?.map((a) => {
                    const q = r.transcripts?.find((t) => t.index === a.index);
                    const chip = (label, v) => (
                      <span className="pill idle" style={{ padding: "5px 10px", fontSize: 11.5 }}>
                        {label} <span className="num" style={{ fontWeight: 700, color: v >= 72 ? "var(--sage)" : v >= 48 ? "var(--brass-bright)" : "var(--ember)" }}>{v}</span>
                      </span>
                    );
                    return (
                      <div key={a.index} style={{ padding: "14px 16px", borderRadius: 10, background: "rgba(10,6,3,0.35)", border: "1px solid var(--glass-line)", display: "grid", gap: 8 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
                          <span style={{ fontSize: 13.5 }}>
                            <span className="num" style={{ color: "var(--brass)", fontWeight: 700, marginRight: 10 }}>Q{a.index + 1}</span>
                            {q?.question}
                          </span>
                          <span style={{ display: "flex", gap: 6 }}>
                            {chip("Content", a.content)}
                            {chip("Structure", a.structure)}
                            {chip("Detail", a.specificity)}
                          </span>
                        </div>
                        <p style={{ fontSize: 13, color: "var(--cream)", lineHeight: 1.55 }}>{a.verdict}</p>
                        <p style={{ fontSize: 13, color: "var(--cream-dim)", lineHeight: 1.55, paddingLeft: 14, borderLeft: "2px solid var(--brass-dim)" }}>{a.improve}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* suggestions */}
        <motion.div {...fade(4)} className="glass" style={{ padding: 24 }}>
          <p className="eyebrow" style={{ marginBottom: 14 }}>Coach's notes</p>
          <div style={{ display: "grid", gap: 10 }}>
            {r.suggestions?.map((s, i) => (
              <p key={i} style={{ fontSize: 14, lineHeight: 1.6, color: "var(--cream)", paddingLeft: 16, borderLeft: "2px solid var(--brass-dim)" }}>
                {s}
              </p>
            ))}
          </div>
        </motion.div>

        <motion.div {...fade(5)} style={{ display: "flex", gap: 12, justifyContent: "center", paddingTop: 6 }}>
          <button className="btn btn-brass" style={{ padding: "14px 30px" }} onClick={() => nav("/interview")}>
            Run it again
          </button>
          <button className="btn btn-ghost" onClick={() => window.print()}>
            Save as PDF
          </button>
        </motion.div>
      </div>
    </div>
  );
}
