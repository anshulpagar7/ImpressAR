// scoring.js — the confidence engine, now client-side and smooth.
// v1 nudged a global float by fixed deltas per frame. v2 computes a target
// score from weighted live signals and eases toward it with an EMA, so the
// gauge moves like a needle, not a slot machine.

const EMA_ALPHA = 0.055;

export function createScoring() {
  return {
    confidence: 70,
    startedAt: null,
    // frame tallies
    frames: 0,
    postureGood: 0,
    eyeGood: 0,
    stillHead: 0,
    calmHands: 0,
    // timeline: one confidence sample per second
    timeline: [],
    lastSampleAt: 0,
    // per-question tracking
    perQuestion: [],
    activeQ: null,
  };
}

export function beginQuestion(s, index, text) {
  s.activeQ = {
    index,
    text,
    frames: 0,
    postureGood: 0,
    eyeGood: 0,
    confSum: 0,
    startedAt: performance.now(),
  };
  s.perQuestion[index] = s.activeQ;
}

export function tick(s, sig, now) {
  if (!s.startedAt) s.startedAt = now;
  if (!sig.present) return s.confidence;

  s.frames += 1;
  const postureOk = sig.postureOk !== false;
  const eyeOk = sig.eyeOk !== false;
  const headStill = sig.headSpeed < 0.011;
  const handsCalm = !sig.handsVisible || sig.handSpeed < 0.02;

  if (postureOk) s.postureGood += 1;
  if (eyeOk) s.eyeGood += 1;
  if (headStill) s.stillHead += 1;
  if (handsCalm) s.calmHands += 1;

  // weighted target the EMA chases
  const target =
    28 +
    (postureOk ? 20 : 0) +
    (eyeOk ? 26 : 4) +
    (headStill ? 14 : 2) +
    (handsCalm ? 12 : 0);

  s.confidence += (target - s.confidence) * EMA_ALPHA;
  s.confidence = Math.max(0, Math.min(100, s.confidence));

  if (s.activeQ) {
    s.activeQ.frames += 1;
    if (postureOk) s.activeQ.postureGood += 1;
    if (eyeOk) s.activeQ.eyeGood += 1;
    s.activeQ.confSum += s.confidence;
  }

  if (now - s.lastSampleAt >= 1000) {
    s.timeline.push(Math.round(s.confidence * 10) / 10);
    s.lastSampleAt = now;
  }
  return s.confidence;
}

const pct = (n, d) => (d ? Math.round((n / d) * 1000) / 10 : 0);

export function summarize(s, speech) {
  const posture = pct(s.postureGood, s.frames);
  const eye = pct(s.eyeGood, s.frames);
  const stillness = pct(s.stillHead, s.frames);
  const calm = pct(s.calmHands, s.frames);
  const avg = s.timeline.length
    ? Math.round(
        (s.timeline.reduce((a, b) => a + b, 0) / s.timeline.length) * 10
      ) / 10
    : Math.round(s.confidence * 10) / 10;

  // speech scoring: ideal pace 110–160 wpm, penalize filler density
  let pace = null;
  let fillerRatio = null;
  let speechScore = null;
  if (speech && speech.words > 15) {
    pace = speech.wpm;
    fillerRatio = Math.round((speech.fillers / speech.words) * 1000) / 10;
    const paceScore =
      pace >= 110 && pace <= 160
        ? 100
        : Math.max(0, 100 - Math.abs(pace - 135) * 1.4);
    speechScore = Math.round(
      Math.max(0, paceScore - fillerRatio * 6)
    );
  }

  const questions = s.perQuestion
    .filter(Boolean)
    .map((q) => ({
      index: q.index,
      text: q.text,
      avg: q.frames ? Math.round((q.confSum / q.frames) * 10) / 10 : null,
      posture: pct(q.postureGood, q.frames),
      eye: pct(q.eyeGood, q.frames),
    }));

  return {
    date: new Date().toISOString(),
    durationSec: s.startedAt
      ? Math.round((performance.now() - s.startedAt) / 1000)
      : 0,
    avg,
    posture,
    eye,
    stillness,
    calm,
    pace,
    fillerRatio,
    fillerCount: speech?.fillers ?? null,
    speechScore,
    timeline: s.timeline,
    questions,
  };
}

export function buildSuggestions(r) {
  const out = [];
  if (r.posture < 80)
    out.push(
      "Level your shoulders and sit tall — your posture slipped in " +
        Math.round(100 - r.posture) +
        "% of frames."
    );
  if (r.eye < 75)
    out.push(
      "Anchor your gaze on the lens, not the screen. Glancing away reads as uncertainty."
    );
  if (r.stillness < 70)
    out.push(
      "Your head moved frequently while answering. Plant yourself and let your hands do the emphasis."
    );
  if (r.calm < 75)
    out.push(
      "Fidgeting spiked during answers. Rest your hands loosely on the desk between gestures."
    );
  if (r.pace != null && r.pace > 165)
    out.push(
      `You averaged ${r.pace} words per minute — breathe and slow to ~130 for clarity.`
    );
  if (r.pace != null && r.pace < 105 && r.pace > 0)
    out.push(
      `You averaged ${r.pace} words per minute — add a bit more energy and momentum.`
    );
  if (r.fillerRatio != null && r.fillerRatio > 4)
    out.push(
      `Filler words made up ${r.fillerRatio}% of your speech. Replace "um" with a silent pause — pauses read as composure.`
    );
  if (!out.length)
    out.push(
      "Strong session across every signal. Raise the difficulty: longer answers, harder questions."
    );
  return out;
}
