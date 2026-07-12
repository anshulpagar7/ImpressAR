import { useState } from "react";
import { motion } from "framer-motion";
import Topbar from "../components/Topbar";
import { addCustomQuestions, getCustomQuestions, CORE_QUESTIONS } from "../lib/store";

export default function Questions() {
  const [text, setText] = useState("");
  const [custom, setCustom] = useState(getCustomQuestions());
  const [saved, setSaved] = useState(0);

  function save() {
    const n = addCustomQuestions(text.split("\n"));
    setCustom(getCustomQuestions());
    setText("");
    setSaved(n);
    setTimeout(() => setSaved(0), 2500);
  }

  return (
    <div className="page">
      <Topbar />
      <div style={{ maxWidth: 920, margin: "36px auto 0", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22, alignItems: "start" }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="glass"
          style={{ padding: 28 }}
        >
          <h2 style={{ fontSize: 24, marginBottom: 8 }}>Add your own questions</h2>
          <p style={{ color: "var(--cream-dim)", fontSize: 13.5, marginBottom: 16, lineHeight: 1.55 }}>
            One question per line. They join the shuffle for every future session on this device.
          </p>
          <textarea
            rows={9}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={"Why do you want to work at this company?\nExplain a project from your resume in depth.\nWhat would your teammates say about you?"}
            style={{ resize: "vertical" }}
          />
          <button className="btn btn-brass" style={{ width: "100%", marginTop: 16 }} onClick={save} disabled={!text.trim()}>
            {saved ? `Added ${saved} question${saved > 1 ? "s" : ""} ✓` : "Add to my bank"}
          </button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
          className="glass"
          style={{ padding: 28 }}
        >
          <p className="eyebrow" style={{ marginBottom: 12 }}>
            Your bank · {CORE_QUESTIONS.length} built-in + {custom.length} custom
          </p>
          <div style={{ display: "grid", gap: 8, maxHeight: 380, overflowY: "auto", paddingRight: 6 }}>
            {custom.length === 0 && (
              <p style={{ color: "var(--cream-dim)", fontSize: 13.5 }}>
                No custom questions yet — add some on the left and they'll appear here.
              </p>
            )}
            {custom.map((q, i) => (
              <div key={i} style={{ fontSize: 13.5, padding: "10px 14px", borderRadius: 9, background: "rgba(10,6,3,0.35)", border: "1px solid var(--glass-line)" }}>
                {q}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
