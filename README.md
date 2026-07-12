# ImpressAR v2 — AI Interview Studio

Full rebuild of ImpressAR: the Flask + base64-frame pipeline is gone. All
computer vision now runs **in the browser** via MediaPipe Tasks (WebAssembly),
so analysis runs at ~15 fps instead of ~1.25, video never leaves the device,
and the whole app deploys as a static site.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

Use **Chrome or Edge** for the full experience (Web Speech API powers the
live filler-word / pace coaching; other browsers skip it gracefully).
Allow camera + microphone when prompted.

```bash
npm run build && npm run preview   # production build
```

Deploys anywhere static (Vercel / Netlify / GitHub Pages) — no server.

## What changed from v1

| v1 (Flask) | v2 |
|---|---|
| Frames POSTed as base64 JPEG to `/analyze` every 800ms | MediaPipe Tasks in-browser, ~15 analyses/sec |
| `mp.solutions.*` (deprecated legacy API) | Pose + Face **Landmarker Tasks** (`pose_landmarker.task` bundled in `public/models/`) |
| "Eye contact" = head yaw (nose vs eye midpoint) | True gaze from eye-direction **blendshapes** |
| Separate Hands model for fidget | Wrist velocity from the pose model (one model fewer) |
| Global Python variables → single-user only | Per-session state client-side, history in localStorage |
| Score jumps by fixed deltas | EMA-smoothed confidence chasing a weighted target |
| Chart.js report | Hand-rolled SVG: live sparkline, trend, radar |
| No speech analysis | Live WPM + filler-word counting (Web Speech API) |
| Questions only | Per-answer LLM grading with coach feedback (free Gemini tier) |

## Structure

```
src/lib/visionEngine.js   MediaPipe setup + per-frame signal extraction
src/lib/scoring.js        EMA confidence engine, per-question stats, suggestions
src/lib/speech.js         Web Speech wrapper (fillers, WPM)
src/lib/store.js          localStorage persistence — swap for Supabase later
src/pages/                Login · Home · Interview · Report · Questions
src/components/           RingGauge (signature), charts, Topbar
src/styles/global.css     "Mahogany Glass" design system
```

## Answer evaluation (LLM) -- free

After a session, transcripts of your answers are graded (content / structure /
specificity per question + overall coaching) by an LLM through a tiny proxy
so no API key ever reaches the browser.

1. Get a **free** Gemini key: aistudio.google.com -> "Get API key" (no card).
2. `cp .env.example .env` and paste the key.
3. Run the proxy alongside vite: `npm run api` (second terminal).

Free-tier limits (~15 req/min, ~1500/day) are irrelevant here -- it's one
request per interview. Without a key the app works fully; the evaluation
card simply explains how to enable it. `ANTHROPIC_API_KEY` also works if
you ever have paid credits. Note: Google may use free-tier API data for
model improvement -- fine for practice answers, worth knowing.

## Next milestones

- Supabase auth + cloud session history (`store.js` is the only file to touch)
- LLM evaluation of transcribed answers (content scoring, not just delivery)
- Resume-driven question generation
