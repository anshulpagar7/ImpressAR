import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Topbar from "../components/Topbar";
import RingGauge from "../components/RingGauge";
import { Sparkline } from "../components/charts";
import { initVision, analyzeFrame } from "../lib/visionEngine";
import { createScoring, beginQuestion, tick, summarize, buildSuggestions } from "../lib/scoring";
import { createSpeechTracker, transcriptsByQuestion } from "../lib/speech";
import { buildSession, saveReport, getProfile } from "../lib/store";

const DURATION = 180;

function Pill({ label, ok }) {
  const cls = ok == null ? "idle" : ok ? "good" : "bad";
  const text = ok == null ? "—" : ok ? "Good" : label.warn;
  return (
    <div className={`pill ${cls}`}>
      <span className="dot" />
      <span style={{ opacity: 0.75 }}>{label.name}</span>
      <span style={{ fontWeight: 700 }}>{text}</span>
    </div>
  );
}

export default function Interview() {
  const nav = useNavigate();
  const videoRef = useRef(null);
  const scoringRef = useRef(createScoring());
  const speechRef = useRef(null);
  const prevSigRef = useRef(null);
  const rafRef = useRef(null);
  const activeRef = useRef(false);
  const lastAnalysisRef = useRef(0);

  const [phase, setPhase] = useState("loading"); // loading | ready | live
  const [questions, setQuestions] = useState([]);
  const [qIndex, setQIndex] = useState(0);
  const [timeLeft, setTimeLeft] = useState(DURATION);
  const [score, setScore] = useState(70);
  const [spark, setSpark] = useState([]);
  const [signals, setSignals] = useState({});
  const [speechLive, setSpeechLive] = useState({ supported: true, wpm: 0, fillers: 0 });

  // camera + models
  useEffect(() => {
    let stream;
    (async () => {
      try {
        const [s] = await Promise.all([
          navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } }),
          initVision(),
        ]);
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          await videoRef.current.play().catch(() => {});
        }
        setPhase("ready");
      } catch (e) {
        console.error(e);
        setPhase("error");
      }
    })();
    return () => {
      activeRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      speechRef.current?.stop();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // countdown
  useEffect(() => {
    if (phase !== "live") return;
    const id = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          clearInterval(id);
          finish();
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  function loop(ts) {
    if (!activeRef.current) return;
    const v = videoRef.current;
    if (v && v.videoWidth && ts - lastAnalysisRef.current > 66) {
      // ~15 analyses/sec — v1 managed ~1.25/sec over HTTP
      lastAnalysisRef.current = ts;
      const sig = analyzeFrame(v, performance.now(), prevSigRef.current);
      prevSigRef.current = sig;
      const conf = tick(scoringRef.current, sig, performance.now());
      setScore(conf);
      setSignals({
        posture: sig.present ? sig.postureOk : null,
        eye: sig.present ? sig.eyeOk : null,
        still: sig.present ? sig.headSpeed < 0.011 : null,
        hands: sig.handsVisible ? sig.handSpeed < 0.02 : null,
      });
      setSpark([...scoringRef.current.timeline]);
      const sp = speechRef.current?.state;
      if (sp) setSpeechLive({ supported: sp.supported, wpm: sp.wpm, fillers: sp.fillers });
    }
    rafRef.current = requestAnimationFrame(loop);
  }

  function start() {
    scoringRef.current = createScoring();
    const qs = buildSession();
    setQuestions(qs);
    setQIndex(0);
    beginQuestion(scoringRef.current, 0, qs[0]);
    setTimeLeft(DURATION);
    setPhase("live");
    activeRef.current = true;
    speechRef.current = createSpeechTracker();
    speechRef.current.setQuestion(0);
    speechRef.current.start();
    rafRef.current = requestAnimationFrame(loop);
  }

  function next() {
    setQIndex((i) => {
      const n = Math.min(i + 1, questions.length - 1);
      if (n !== i) {
        beginQuestion(scoringRef.current, n, questions[n]);
        speechRef.current?.setQuestion(n);
      }
      return n;
    });
  }

  function finish() {
    activeRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    speechRef.current?.stop();
    const report = summarize(scoringRef.current, speechRef.current?.state);
    report.suggestions = buildSuggestions(report);
    report.transcripts = transcriptsByQuestion(speechRef.current?.state, questions);
    report.role = getProfile()?.role || "Student";
    saveReport(report);
    nav("/report");
  }

  const mm = String(Math.floor(timeLeft / 60)).padStart(1, "0");
  const ss = String(timeLeft % 60).padStart(2, "0");
  const live = phase === "live";

  return (
    <div className="page">
      <Topbar minimal />

      {phase === "error" && (
        <div className="glass" style={{ maxWidth: 520, margin: "80px auto", padding: 36, textAlign: "center" }}>
          <h2 style={{ marginBottom: 10 }}>Camera unavailable</h2>
          <p style={{ color: "var(--cream-dim)", fontSize: 14, lineHeight: 1.6 }}>
            Allow camera access in your browser, then reload this page. All analysis runs locally — no video is uploaded.
          </p>
        </div>
      )}

      {phase !== "error" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2.1fr) minmax(320px, 1fr)",
            gap: 24,
            alignItems: "start",
          }}
        >
          {/* ---------- CAMERA ---------- */}
          <motion.div
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="glass"
            style={{ padding: 10, overflow: "hidden" }}
          >
            <div style={{ position: "relative", borderRadius: 14, overflow: "hidden", background: "#0b0703" }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: "100%", display: "block", transform: "scaleX(-1)", aspectRatio: "16/10", objectFit: "cover" }}
              />
              {/* question teleprompter */}
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  padding: "22px 26px 40px",
                  background: "linear-gradient(rgba(8,5,2,0.78), transparent)",
                }}
              >
                <p className="eyebrow" style={{ marginBottom: 6 }}>
                  {live ? `Question ${qIndex + 1} of ${questions.length}` : phase === "loading" ? "Preparing studio…" : "Ready when you are"}
                </p>
                <AnimatePresence mode="wait">
                  <motion.h2
                    key={live ? qIndex : phase}
                    initial={{ opacity: 0, y: 10, filter: "blur(6px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    exit={{ opacity: 0, y: -10, filter: "blur(6px)" }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                    style={{ fontSize: "clamp(19px, 2.2vw, 27px)", maxWidth: 640, lineHeight: 1.3 }}
                  >
                    {live ? questions[qIndex] : "Sit comfortably, look at the lens, and press Begin."}
                  </motion.h2>
                </AnimatePresence>
              </div>
              {/* live speech readout */}
              {live && speechLive.supported && (
                <div
                  style={{
                    position: "absolute",
                    bottom: 14,
                    left: 14,
                    display: "flex",
                    gap: 8,
                  }}
                >
                  <div className="pill idle" style={{ background: "rgba(8,5,2,0.65)" }}>
                    <span className="num" style={{ color: "var(--brass-bright)", fontWeight: 700 }}>{speechLive.wpm}</span> wpm
                  </div>
                  <div className={`pill ${speechLive.fillers > 6 ? "bad" : "idle"}`} style={{ background: "rgba(8,5,2,0.65)" }}>
                    <span className="num" style={{ fontWeight: 700 }}>{speechLive.fillers}</span> fillers
                  </div>
                </div>
              )}
              {/* rec dot */}
              {live && (
                <div style={{ position: "absolute", bottom: 18, right: 18, display: "flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--ember)", fontWeight: 700 }}>
                  <span className="dot" style={{ width: 8, height: 8, borderRadius: "50%", background: "currentColor", boxShadow: "0 0 10px currentColor", animation: "ambient-drift 1.2s ease-in-out infinite alternate" }} />
                  LIVE · ON-DEVICE
                </div>
              )}
            </div>

            {/* controls */}
            <div style={{ display: "flex", gap: 10, padding: "14px 6px 6px" }}>
              {!live ? (
                <button className="btn btn-brass" style={{ flex: 1, padding: 15 }} onClick={start} disabled={phase !== "ready"}>
                  {phase === "loading" ? "Loading models…" : "Begin interview"}
                </button>
              ) : (
                <>
                  <button className="btn btn-ghost" style={{ flex: 1 }} onClick={next} disabled={qIndex >= questions.length - 1}>
                    Next question →
                  </button>
                  <button className="btn btn-brass" style={{ flex: 1 }} onClick={finish}>
                    Finish &amp; get report
                  </button>
                </>
              )}
            </div>
          </motion.div>

          {/* ---------- COACH PANEL ---------- */}
          <div style={{ display: "grid", gap: 18 }}>
            <motion.div
              initial={{ opacity: 0, x: 22 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
              className="glass"
              style={{ padding: "28px 22px 22px", textAlign: "center" }}
            >
              <RingGauge score={score} />
              <div style={{ marginTop: 18 }}>
                <Sparkline data={spark} />
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 22 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className="glass"
              style={{ padding: 22, display: "grid", gap: 10 }}
            >
              <p className="eyebrow" style={{ marginBottom: 4 }}>Live signals</p>
              <Pill label={{ name: "Posture", warn: "Sit straight" }} ok={signals.posture} />
              <Pill label={{ name: "Eye contact", warn: "Look at lens" }} ok={signals.eye} />
              <Pill label={{ name: "Head", warn: "Too much motion" }} ok={signals.still} />
              <Pill label={{ name: "Hands", warn: "Fidgeting" }} ok={signals.hands} />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 22 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.32, ease: [0.22, 1, 0.36, 1] }}
              className="glass"
              style={{ padding: "20px 22px", textAlign: "center" }}
            >
              <p className="eyebrow" style={{ marginBottom: 6 }}>Time remaining</p>
              <div className="display num" style={{ fontSize: 44, color: timeLeft < 30 && live ? "var(--ember)" : "var(--cream)", transition: "color 0.5s" }}>
                {mm}:{ss}
              </div>
            </motion.div>
          </div>
        </div>
      )}
    </div>
  );
}
