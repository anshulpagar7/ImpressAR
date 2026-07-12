// evaluate.js — client side of answer evaluation. Talks only to our own
// /api/evaluate proxy; the Anthropic key never reaches the browser.

export async function evaluateAnswers(role, answers) {
  const r = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, answers }),
  });
  const data = await r.json().catch(() => null);
  if (!r.ok) {
    const err = new Error(data?.message || data?.error || `HTTP ${r.status}`);
    err.code = data?.error;
    throw err;
  }
  return data; // { answers: [...], overall: {...} }
}

// Generate one adaptive follow-up from the candidate's last answer.
// Returns the question string, or null on any failure (caller falls back
// to the question bank — the interview never stalls on the network).
export async function generateFollowup(question, transcript) {
  try {
    const r = await fetch("/api/followup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, transcript }),
    });
    if (!r.ok) return null;
    const data = await r.json();
    return data?.question || null;
  } catch {
    return null;
  }
}