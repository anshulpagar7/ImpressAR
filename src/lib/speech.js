// speech.js — live transcription via the Web Speech API.
// Counts filler words + pace, and captures a per-question transcript
// so answers can be evaluated by an LLM after the session.
// Best support: Chrome / Edge. Degrades silently elsewhere.

const FILLERS =
  /\b(um+|uh+|erm+|hmm+|like|basically|actually|you know|i mean|sort of|kind of)\b/gi;

export function createSpeechTracker() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const state = {
    supported: !!SR,
    running: false,
    words: 0,
    fillers: 0,
    wpm: 0,
    startedAt: null,
    segments: [], // { q: questionIndex, text }
    currentQ: 0,
  };
  if (!SR) return { state, start() {}, stop() {}, setQuestion() {} };

  const rec = new SR();
  rec.continuous = true;
  rec.interimResults = false;
  rec.lang = "en-IN";

  rec.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (!e.results[i].isFinal) continue;
      const text = e.results[i][0].transcript.trim();
      if (!text) continue;
      state.segments.push({ q: state.currentQ, text });
      state.words += text.split(/\s+/).length;
      state.fillers += (text.match(FILLERS) || []).length;
      const mins = (performance.now() - state.startedAt) / 60000;
      state.wpm = mins > 0.05 ? Math.round(state.words / mins) : 0;
    }
  };
  // Chrome stops recognition after silence — restart while active
  rec.onend = () => {
    if (state.running) {
      try {
        rec.start();
      } catch {
        /* already starting */
      }
    }
  };

  return {
    state,
    setQuestion(i) {
      state.currentQ = i;
    },
    start() {
      if (state.running) return;
      state.running = true;
      state.startedAt = performance.now();
      try {
        rec.start();
      } catch {
        /* ignore double-start */
      }
    },
    stop() {
      state.running = false;
      try {
        rec.stop();
      } catch {
        /* ignore */
      }
    },
  };
}

// Assemble { question, transcript } pairs for evaluation.
export function transcriptsByQuestion(state, questions) {
  if (!state?.segments?.length) return [];
  const map = new Map();
  for (const s of state.segments) {
    map.set(s.q, (map.get(s.q) || "") + " " + s.text);
  }
  return [...map.entries()]
    .map(([q, text]) => ({
      index: q,
      question: questions[q] ?? `Question ${q + 1}`,
      transcript: text.trim().slice(0, 2400),
    }))
    .filter((t) => t.transcript.split(/\s+/).length >= 8)
    .sort((a, b) => a.index - b.index);
}
