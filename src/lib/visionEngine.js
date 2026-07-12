// visionEngine.js — runs MediaPipe Tasks entirely in the browser.
// v1 shipped every frame to Flask as base64 JPEG (~1.25 fps).
// v2 analyzes locally at video rate; the camera feed never leaves the device.

import {
  FilesetResolver,
  PoseLandmarker,
  FaceLandmarker,
} from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Pose model is bundled locally (public/models/) — same .task file from v1's repo.
const POSE_MODEL = "/models/pose_landmarker.task";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let poseLandmarker = null;
let faceLandmarker = null;
let initPromise = null;

export function initVision() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    [poseLandmarker, faceLandmarker] = await Promise.all([
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
      }),
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        outputFaceBlendshapes: true,
        numFaces: 1,
      }),
    ]);
  })();
  return initPromise;
}

// Pull one blendshape score by name.
function shape(categories, name) {
  const c = categories.find((k) => k.categoryName === name);
  return c ? c.score : 0;
}

/**
 * Analyze a single video frame. Returns raw behavioral signals:
 * { present, postureOk, shoulderTilt, eyeOk, gazeDeviation,
 *   headSpeed, handSpeed, handsVisible }
 */
export function analyzeFrame(video, ts, prev) {
  const out = {
    present: false,
    postureOk: null,
    shoulderTilt: 0,
    eyeOk: null,
    gazeDeviation: 0,
    headSpeed: 0,
    handSpeed: 0,
    handsVisible: false,
    noseX: prev?.noseX ?? null,
    noseY: prev?.noseY ?? null,
    wristX: prev?.wristX ?? null,
    wristY: prev?.wristY ?? null,
  };
  if (!poseLandmarker || !faceLandmarker) return out;

  // ---- POSE: shoulders (11/12) + wrists (15/16) ----
  const pose = poseLandmarker.detectForVideo(video, ts);
  if (pose.landmarks && pose.landmarks.length) {
    const lm = pose.landmarks[0];
    const ls = lm[11];
    const rs = lm[12];
    out.present = true;
    out.shoulderTilt = Math.abs(ls.y - rs.y);
    out.postureOk = out.shoulderTilt < 0.045;

    // hand steadiness from wrist velocity (replaces v1's separate Hands model)
    const lw = lm[15];
    const rw = lm[16];
    const wrist =
      (lw.visibility ?? 0) > (rw.visibility ?? 0) ? lw : rw;
    if ((wrist.visibility ?? 0) > 0.4) {
      out.handsVisible = true;
      if (prev?.wristX != null) {
        out.handSpeed = Math.hypot(
          wrist.x - prev.wristX,
          wrist.y - prev.wristY
        );
      }
      out.wristX = wrist.x;
      out.wristY = wrist.y;
    }
  }

  // ---- FACE: gaze from blendshapes + head-motion from nose ----
  const face = faceLandmarker.detectForVideo(video, ts);
  if (face.faceLandmarks && face.faceLandmarks.length) {
    out.present = true;
    const nose = face.faceLandmarks[0][1];

    if (prev?.noseX != null) {
      out.headSpeed = Math.hypot(nose.x - prev.noseX, nose.y - prev.noseY);
    }
    out.noseX = nose.x;
    out.noseY = nose.y;

    if (face.faceBlendshapes && face.faceBlendshapes.length) {
      const c = face.faceBlendshapes[0].categories;
      // true gaze estimation — v1 could only detect head yaw
      const horiz = Math.max(
        (shape(c, "eyeLookOutLeft") + shape(c, "eyeLookInRight")) / 2,
        (shape(c, "eyeLookInLeft") + shape(c, "eyeLookOutRight")) / 2
      );
      const vert = Math.max(
        (shape(c, "eyeLookUpLeft") + shape(c, "eyeLookUpRight")) / 2,
        (shape(c, "eyeLookDownLeft") + shape(c, "eyeLookDownRight")) / 2
      );
      out.gazeDeviation = Math.max(horiz, vert * 0.8);
      out.eyeOk = out.gazeDeviation < 0.42;
    }
  }

  return out;
}
