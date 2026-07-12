import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Topbar from "../components/Topbar";
import { getProfile, getHistory } from "../lib/store";

const FEATURES = [
  ["Posture", "Shoulder alignment tracked at video rate, not once a second."],
  ["True gaze", "Eye-direction blendshapes — knows where you look, not just where you face."],
  ["Steady hands", "Wrist velocity flags fidgeting the moment it starts."],
  ["Speech coach", "Live filler-word counting and pace analysis while you answer."],
  ["Confidence trend", "A second-by-second score you can replay after the session."],
  ["Private by design", "Every frame is analyzed in your browser. Nothing is uploaded."],
];

const stagger = {
  hidden: { opacity: 0, y: 22 },
  show: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.35 + i * 0.07, duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  }),
};

export default function Home() {
  const nav = useNavigate();
  const profile = getProfile();
  const history = getHistory();
  const last = history[history.length - 1];

  return (
    <div className="page">
      <Topbar />

      <section style={{ textAlign: "center", padding: "72px 16px 56px" }}>
        <motion.p
          className="eyebrow"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          Welcome back{profile ? `, ${profile.name}` : ""}
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.75, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          style={{
            fontSize: "clamp(38px, 5.4vw, 62px)",
            lineHeight: 1.08,
            margin: "14px auto 18px",
            maxWidth: 760,
          }}
        >
          Walk in <em style={{ color: "var(--brass-bright)" }}>composed</em>.
          <br />
          Walk out hired.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          style={{
            color: "var(--cream-dim)",
            maxWidth: 520,
            margin: "0 auto 30px",
            fontSize: 15.5,
            lineHeight: 1.6,
          }}
        >
          ImpressAR watches the things interviewers notice — posture, gaze,
          stillness, and how you speak — and coaches you live, in your browser.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
          style={{ display: "flex", gap: 12, justifyContent: "center" }}
        >
          <button className="btn btn-brass" style={{ padding: "15px 30px", fontSize: 15 }} onClick={() => nav("/interview")}>
            Start a session
          </button>
          {last && (
            <button className="btn btn-ghost" style={{ padding: "15px 26px" }} onClick={() => nav("/report")}>
              Last report · <span className="num" style={{ color: "var(--brass-bright)" }}>{Math.round(last.avg)}</span>
            </button>
          )}
        </motion.div>
      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 22,
          maxWidth: 1120,
          margin: "0 auto",
        }}
      >
        {FEATURES.map(([title, body], i) => (
          <motion.div
            key={title}
            className="glass lift"
            custom={i}
            variants={stagger}
            initial="hidden"
            animate="show"
            style={{ padding: "26px 26px 24px" }}
          >
            <h3 style={{ fontSize: 19, marginBottom: 8, color: "var(--brass-bright)" }}>{title}</h3>
            <p style={{ fontSize: 13.5, color: "var(--cream-dim)", lineHeight: 1.6 }}>{body}</p>
          </motion.div>
        ))}
      </section>

      {history.length > 1 && (
        <section style={{ maxWidth: 1120, margin: "40px auto 0" }}>
          <p className="eyebrow" style={{ marginBottom: 12 }}>
            Recent sessions
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {history
              .slice(-6)
              .reverse()
              .map((h, i) => (
                <div key={i} className="pill idle" style={{ padding: "10px 16px" }}>
                  <span className="num" style={{ color: "var(--brass-bright)", fontWeight: 700 }}>
                    {Math.round(h.avg)}
                  </span>
                  <span style={{ opacity: 0.7 }}>
                    {new Date(h.date).toLocaleDateString(undefined, { day: "numeric", month: "short" })}
                  </span>
                </div>
              ))}
          </div>
        </section>
      )}
    </div>
  );
}
