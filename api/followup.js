// api/followup.js — generates ONE adaptive follow-up question from the
// candidate's last answer, mid-interview. Same key handling as evaluate.js:
// GEMINI_API_KEY (free) preferred, ANTHROPIC_API_KEY fallback.

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";

const SYSTEM = `You are an experienced interviewer conducting a live mock interview. Given the question just asked and the candidate's spoken answer (raw speech-recognition transcript — ignore transcription noise), produce ONE sharp follow-up question that a real interviewer would ask next: probe a gap, a vague claim, or an interesting detail they mentioned. Keep it under 22 words, conversational, no preamble.

Respond with ONLY valid JSON: {"question": "<the follow-up>"}`;

function extractJSON(text) {
  return JSON.parse(text.replace(/```json|```/g, "").trim());
}

export default async function followupHandler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  const gemini = process.env.GEMINI_API_KEY;
  const anthropic = process.env.ANTHROPIC_API_KEY;
  if (!gemini && !anthropic) {
    res.status(501).json({ error: "not_configured" });
    return;
  }
  const { question, transcript } = req.body || {};
  if (!question || !transcript) {
    res.status(400).json({ error: "question and transcript required" });
    return;
  }
  const userText = `Question asked: ${String(question).slice(0, 300)}\n\nCandidate's answer (transcript): ${String(transcript).slice(0, 1800)}`;

  try {
    let result;
    if (gemini) {
      const r = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${gemini}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            systemInstruction: { parts: [{ text: SYSTEM }] },
            contents: [{ role: "user", parts: [{ text: userText }] }],
            generationConfig: {
              responseMimeType: "application/json",
              maxOutputTokens: 200,
              temperature: 0.7,
            },
          }),
        }
      );
      if (!r.ok) throw new Error(`Gemini ${r.status}`);
      const data = await r.json();
      result = extractJSON(
        data.candidates?.[0]?.content?.parts?.map((p) => p.text).join("\n") || ""
      );
    } else {
      const r = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": anthropic,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: ANTHROPIC_MODEL,
          max_tokens: 200,
          system: SYSTEM,
          messages: [{ role: "user", content: userText }],
        }),
      });
      if (!r.ok) throw new Error(`Anthropic ${r.status}`);
      const data = await r.json();
      result = extractJSON(
        (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("\n")
      );
    }
    if (!result?.question) throw new Error("no question in response");
    res.status(200).json(result);
  } catch (e) {
    res.status(502).json({ error: "followup_failed", detail: String(e.message || e).slice(0, 300) });
  }
}