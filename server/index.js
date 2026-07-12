// Local dev API server — wraps api/evaluate.js on http://localhost:8787.
// Vite proxies /api/* here during `npm run dev`.
import fs from "fs";
import express from "express";

// tiny .env loader (no dotenv dep)
try {
  for (const line of fs.readFileSync(".env", "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.+)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
} catch { /* no .env — fine */ }
import evaluateHandler from "../api/evaluate.js";
import followupHandler from "../api/followup.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

app.post("/api/evaluate", (req, res) => evaluateHandler(req, res));
app.post("/api/followup", (req, res) => followupHandler(req, res));

const port = process.env.PORT || 8787;
app.listen(port, () => {
  console.log(`[impressar] eval API on http://localhost:${port}`);
  if (!process.env.GEMINI_API_KEY && !process.env.ANTHROPIC_API_KEY) {
    console.log("[impressar] No GEMINI_API_KEY / ANTHROPIC_API_KEY — evaluation disabled (app still works).");
  }
});