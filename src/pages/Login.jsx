import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { setProfile } from "../lib/store";

export default function Login() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [role, setRole] = useState("Student");

  function enter(e) {
    if (e) e.preventDefault();
    if (!name.trim()) return;
    setProfile({ name: name.trim(), role, since: Date.now() });
    nav("/home");
  }

  return (
    <div
      className="page"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        paddingBottom: 0,
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 26, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="glass"
        style={{ width: 400, padding: "42px 36px", textAlign: "center" }}
      >
        <div
          className="logo"
          style={{ justifyContent: "center", fontSize: 30, marginBottom: 8 }}
        >
          <span className="mark" />
          Impress<em>AR</em>
        </div>
        <p className="eyebrow" style={{ marginBottom: 6 }}>
          AI interview studio
        </p>
        <p
          style={{
            color: "var(--cream-dim)",
            fontSize: 14,
            marginBottom: 28,
            lineHeight: 1.55,
          }}
        >
          Real-time body language, gaze, and speech coaching — entirely on your
          device.
        </p>

        <div style={{ display: "grid", gap: 4, textAlign: "left" }}>
          <input
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && enter()}
            autoFocus
          />
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option>Student</option>
            <option>Fresher</option>
            <option>Professional</option>
          </select>
        </div>

        <button
          className="btn btn-brass"
          style={{ width: "100%", marginTop: 22 }}
          onClick={enter}
          disabled={!name.trim()}
        >
          Enter the studio →
        </button>
      </motion.div>
    </div>
  );
}
