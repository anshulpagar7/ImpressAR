// api/evaluate.js — the ONE piece of backend in ImpressAR v2.
// Provider-agnostic answer evaluation proxy; the API key never reaches
// the browser. Works with either:
//   GEMINI_API_KEY     — Google AI Studio, FREE tier (default)
//   ANTHROPIC_API_KEY  — paid, used only if no Gemini key is set
// With neither key set, returns 501 and the app simply hides the section.
// Deploy as a Vercel/Netlify function as-is, or run locally: `npm run api`.

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";

const SYSTEM = `You are a senior interview coach evaluating a candidate's spoken answers from a mock interview. Transcripts come from speech recognition, so ignore transcription noise, punctuation, and casing — judge substance only.

Score each answer on three axes (0-100):
- content: does it actually answer the question with relevant substance?
- structure: is there a clear arc (situation → action → result, or point → evidence)?
- specificity: concrete examples, names, numbers, outcomes vs. vague generalities.

Be honest and calibrated: a rambling non-answer scores 20-40, a decent generic answer 55-70, a specific well-structured answer 75-90. Reserve 90+ for genuinely excellent responses.

Respond with ONLY valid JSON, no markdown fences, matching exactly:
{
  "answers": [
    {
      "index": <number, echo the input index>,
      "content": <0-100>,
      "structure": <0-100>,
      "specificity": <0-100>,
      "verdict": "<one blunt sentence on how this answer lands>",
      "improve": "<1-2 sentences: the single highest-leverage fix, concrete>"
    }
  ],
  "overall": {
    "summary": "<2-3 sentences on the candidate's answering pattern across the session>",
    "top_fixes": ["<fix 1>", "<fix 2>", "<fix 3>"]
  }
}`;

function extractJSON(text) {
  return JSON.parse(text.replace(/```json|```/g, "").trim());
}

async function callGemini(key, userText) {
  const r = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${key}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM }] },
        contents: [{ role: "user", parts: [{ text: userText }] }],
        generationConfig: {
          responseMimeType: "application/json",
          maxOutputTokens: 2500,
          temperature: 0.3,
        },
      }),
    }
  );
  if (!r.ok) throw new Error(`Gemini ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const data = await r.json();
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text).join("\n") || "";
  return extractJSON(text);
}

async function callAnthropic(key, userText) {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      max_tokens: 2500,
      system: SYSTEM,
      messages: [{ role: "user", content: userText }],
    }),
  });
  if (!r.ok) throw new Error(`Anthropic ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const data = await r.json();
  const text = (data.content || [])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  return extractJSON(text);
}

export default async function evaluateHandler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  const gemini = process.env.GEMINI_API_KEY;
  const anthropic = process.env.ANTHROPIC_API_KEY;
  if (!gemini && !anthropic) {
    res.status(501).json({
      error: "not_configured",
      message:
        "Set GEMINI_API_KEY (free — aistudio.google.com) or ANTHROPIC_API_KEY to enable answer evaluation.",
    });
    return;
  }

  const { role, answers } = req.body || {};
  if (!Array.isArray(answers) || !answers.length) {
    res.status(400).json({ error: "answers[] required" });
    return;
  }

  const payload = answers.slice(0, 12).map((a) => ({
    index: a.index,
    question: String(a.question).slice(0, 300),
    transcript: String(a.transcript).slice(0, 2400),
  }));
  const userText = `Candidate profile: ${role || "Student"} preparing for placement interviews in India.\n\nAnswers to evaluate:\n${JSON.stringify(payload, null, 2)}`;

  try {
    const result = gemini
      ? await callGemini(gemini, userText)
      : await callAnthropic(anthropic, userText);
    res.status(200).json(result);
  } catch (e) {
    res.status(502).json({ error: "evaluation_failed", detail: String(e.message || e).slice(0, 400) });
  }
}
