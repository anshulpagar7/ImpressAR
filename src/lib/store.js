// store.js — localStorage persistence for profile, question bank, and
// session history. (Swap this module for Supabase later — the page code
// only talks to these functions.)

const K = {
  profile: "impressar.profile",
  custom: "impressar.customQuestions",
  history: "impressar.history",
};

const read = (k, fb) => {
  try {
    return JSON.parse(localStorage.getItem(k)) ?? fb;
  } catch {
    return fb;
  }
};
const write = (k, v) => localStorage.setItem(k, JSON.stringify(v));

// ---- profile ----
export const getProfile = () => read(K.profile, null);
export const setProfile = (p) => write(K.profile, p);

// ---- question bank ----
export const INTRO_QUESTIONS = [
  "Introduce yourself in under a minute.",
  "What are you studying right now, and why did you choose it?",
  "Which of your skills are you most confident about?",
];

export const CORE_QUESTIONS = [
  "Tell me about yourself.",
  "Why should we hire you?",
  "What motivates you?",
  "Describe a difficult problem you solved.",
  "What is your biggest weakness?",
  "Where do you see yourself in five years?",
  "Tell me about a time you led a team.",
  "Describe your most challenging project.",
  "How do you handle pressure or tight deadlines?",
  "Tell me about a failure and what it taught you.",
  "How do you prioritize when everything is urgent?",
  "How do you pick up a technology you've never used?",
  "Tell me about a conflict in a team and how you handled it.",
  "What is your proudest achievement?",
  "What makes you different from other candidates?",
];

export const getCustomQuestions = () => read(K.custom, []);
export const addCustomQuestions = (lines) => {
  const cur = getCustomQuestions();
  const add = lines.map((l) => l.trim()).filter(Boolean);
  write(K.custom, [...new Set([...cur, ...add])]);
  return add.length;
};

export function buildSession() {
  const bank = [...CORE_QUESTIONS, ...getCustomQuestions()];
  const picked = [];
  const pool = [...bank];
  while (picked.length < Math.min(7, pool.length)) {
    picked.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  }
  return [...INTRO_QUESTIONS, ...picked];
}

// ---- history ----
export const getHistory = () => read(K.history, []);
export function saveReport(report) {
  const h = getHistory();
  h.push(report);
  write(K.history, h.slice(-25)); // keep last 25 sessions
  return h.length - 1;
}
export function updateLatestReport(patch) {
  const h = getHistory();
  if (!h.length) return null;
  h[h.length - 1] = { ...h[h.length - 1], ...patch };
  write(K.history, h);
  return h[h.length - 1];
}
export const getLatestReport = () => {
  const h = getHistory();
  return h.length ? h[h.length - 1] : null;
};
export const getPreviousReport = () => {
  const h = getHistory();
  return h.length > 1 ? h[h.length - 2] : null;
};
