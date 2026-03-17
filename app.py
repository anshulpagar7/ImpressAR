from flask import Flask, render_template, request, jsonify, redirect, session
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import random

app = Flask(__name__)
app.secret_key = "impressar_secret"  # for session

# ---------------- MEDIAPIPE ----------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

previous_hand_x = None
previous_hand_y = None
previous_nose_x = None

# ---------------- SESSION DATA ----------------

confidence_score = 70
session_scores = []

total_frames = 0
good_posture_frames = 0
good_eye_frames = 0
fidget_frames = 0
movement_total = 0
head_move_frames = 0

# ---------------- QUESTIONS ----------------

INTRO_QUESTIONS = [
    "What is your name?",
    "What are you currently studying?",
    "What are your strongest skills?"
]

RANDOM_QUESTIONS = [
    "Tell me about yourself",
    "Why should we hire you",
    "What motivates you",
    "Describe a challenge you solved",
    "What are your strengths",
    "What are your weaknesses",
    "Where do you see yourself in 5 years",
    "Tell me about a leadership experience",
    "Describe a difficult project",
    "How do you handle pressure",
    "Tell me about a failure",
    "How do you prioritize tasks",
    "How do you learn new technologies",
    "Tell me about teamwork",
    "What is your biggest achievement",
    "What makes you unique"
]

# ---------------- ROUTES ----------------

@app.route("/")
def login():
    return render_template("login.html")


# 🔥 HANDLE LOGIN (IMPORTANT FIX)
@app.route("/home", methods=["GET", "POST"])
def home():
    name = request.args.get("name")

    if name:
        session["name"] = name

    name = session.get("name", "User")

    return render_template("home.html", name=name)


@app.route("/interview")
def interview():
    name = session.get("name", "User")
    return render_template("interview.html", name=name)


@app.route("/add_questions")
def add_questions():
    return render_template("add_questions.html")


# ---------------- SAVE QUESTIONS ----------------

@app.route("/save_questions", methods=["POST"])
def save_questions():

    questions = request.form["questions"].split("\n")

    for q in questions:
        q = q.strip()
        if q:
            RANDOM_QUESTIONS.append(q)

    return redirect("/home")


# ---------------- GET QUESTIONS ----------------

@app.route("/questions")
def questions():
    selected_random = random.sample(RANDOM_QUESTIONS, min(7, len(RANDOM_QUESTIONS)))

    return jsonify({
        "questions": INTRO_QUESTIONS + selected_random
    })


# ---------------- ANALYZE ----------------

@app.route("/analyze", methods=["POST"])
def analyze():

    global previous_hand_x, previous_hand_y, previous_nose_x
    global confidence_score
    global total_frames, good_posture_frames
    global good_eye_frames, fidget_frames
    global movement_total, head_move_frames

    try:
        data = request.json["image"]

        image_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"feedback": "No frame", "score": confidence_score})

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        posture_feedback = "Good posture"
        eye_feedback = "Good eye contact"
        fidget_feedback = "Stable"

        total_frames += 1

        # POSTURE
        pose_results = pose.process(rgb)
        if pose_results.pose_landmarks:
            l = pose_results.pose_landmarks.landmark[11]
            r = pose_results.pose_landmarks.landmark[12]

            if abs(l.y - r.y) > 0.05:
                posture_feedback = "Sit straight"
            else:
                good_posture_frames += 1

        # FACE
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:

            face = face_results.multi_face_landmarks[0]
            nose = face.landmark[1]
            le = face.landmark[33]
            re = face.landmark[263]

            center = (le.x + re.x) / 2

            if abs(nose.x - center) > 0.05:
                eye_feedback = "Maintain eye contact"
            else:
                good_eye_frames += 1

            if previous_nose_x is not None:
                if abs(nose.x - previous_nose_x) > 0.015:
                    head_move_frames += 1

            previous_nose_x = nose.x

        # HAND
        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks:

            hand = hand_results.multi_hand_landmarks[0]
            wrist = hand.landmark[0]

            cx, cy = wrist.x, wrist.y

            if previous_hand_x is not None:
                movement = math.sqrt(
                    (cx - previous_hand_x) ** 2 +
                    (cy - previous_hand_y) ** 2
                )

                movement_total += movement

                if movement > 0.02:
                    fidget_feedback = "Don't fidget"
                    fidget_frames += 1

            previous_hand_x, previous_hand_y = cx, cy

        # SCORE LOGIC (slightly smoother)
        confidence_score += (
            (0.1 if posture_feedback == "Good posture" else -0.7) +
            (0.1 if eye_feedback == "Good eye contact" else -0.7) +
            (0.05 if fidget_feedback == "Stable" else -0.6)
        )

        confidence_score = max(0, min(100, confidence_score))

        session_scores.append(confidence_score)

        return jsonify({
            "feedback": f"{posture_feedback} | {eye_feedback} | {fidget_feedback}",
            "score": round(confidence_score, 2)
        })

    except Exception as e:
        return jsonify({"feedback": "Error", "score": confidence_score})


# ---------------- REPORT PAGE ----------------

@app.route("/report_page")
def report_page():

    if total_frames == 0:
        return redirect("/home")

    posture = (good_posture_frames / total_frames) * 100
    eye = (good_eye_frames / total_frames) * 100
    fidget = (fidget_frames / total_frames) * 100
    head = (head_move_frames / total_frames) * 100

    avg = sum(session_scores) / len(session_scores)

    suggestions = []

    if posture < 80:
        suggestions.append("Maintain a straight posture with relaxed shoulders.")

    if eye < 80:
        suggestions.append("Look towards the webcam to improve eye contact.")

    if fidget > 25:
        suggestions.append("Reduce unnecessary hand movements.")

    if head > 25:
        suggestions.append("Avoid excessive head movement.")

    if not suggestions:
        suggestions.append("Excellent performance. Keep practicing!")

    name = session.get("name", "User")

    return render_template(
        "report.html",
        name=name,
        avg=round(avg, 2),
        posture=round(posture, 2),
        eye=round(eye, 2),
        fidget=round(fidget, 2),
        head=round(head, 2),
        suggestions=suggestions,
        trend=session_scores
    )


# ---------------- RESET ----------------

@app.route("/reset")
def reset():

    global session_scores, total_frames
    global good_posture_frames, good_eye_frames
    global fidget_frames, movement_total
    global head_move_frames, confidence_score

    session_scores = []

    total_frames = 0
    good_posture_frames = 0
    good_eye_frames = 0
    fidget_frames = 0
    movement_total = 0
    head_move_frames = 0
    confidence_score = 70

    return jsonify({"status": "reset"})


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)