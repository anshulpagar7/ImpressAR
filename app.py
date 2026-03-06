from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import random

app = Flask(__name__)

# ---------------- MEDIAPIPE ----------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

previous_hand_x = None
previous_hand_y = None

# ---------------- CONFIDENCE SYSTEM ----------------

confidence_score = 70
session_scores = []

# ---------------- QUESTIONS ----------------

INTRO_QUESTIONS = [
    "What is your name?",
    "What are you currently studying or working on?",
    "What are your strongest technical skills?"
]

RANDOM_QUESTIONS = [

"Tell me about yourself.",
"Why should we hire you?",
"What motivates you?",
"Describe a challenge you solved.",
"What are your strengths?",
"What are your weaknesses?",
"Where do you see yourself in 5 years?",
"Tell me about a leadership experience.",
"Describe a difficult project you worked on.",
"How do you handle pressure?",
"Tell me about a time you failed.",
"How do you prioritize tasks?",
"What makes you unique?",
"Why did you choose your field?",
"Describe a conflict you resolved.",
"What is your greatest achievement?",
"How do you learn new technologies?",
"Tell me about teamwork experience.",
"What is a mistake you learned from?",
"How do you stay productive?",
"What is your dream job?",
"What do you know about our company?",
"Why should we hire you over others?",
"What skills are you currently improving?",
"How do you deal with criticism?"

]

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/questions')
def questions():

    selected_random = random.sample(RANDOM_QUESTIONS, 5)

    all_questions = INTRO_QUESTIONS + selected_random

    return jsonify({"questions": all_questions})


@app.route('/analyze', methods=['POST'])
def analyze():

    global previous_hand_x, previous_hand_y
    global confidence_score

    data = request.json['image']

    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    posture_feedback = "Good posture"
    eye_feedback = "Good eye contact"
    fidget_feedback = "Stable"

    # ---------- POSTURE DETECTION ----------

    pose_results = pose.process(rgb)

    if pose_results.pose_landmarks:

        left = pose_results.pose_landmarks.landmark[11]
        right = pose_results.pose_landmarks.landmark[12]

        if abs(left.y - right.y) > 0.05:
            posture_feedback = "Sit straight"

    # ---------- EYE CONTACT DETECTION ----------

    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:

        face = face_results.multi_face_landmarks[0]

        nose = face.landmark[1]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        center = (left_eye.x + right_eye.x) / 2

        if abs(nose.x - center) > 0.05:
            eye_feedback = "Maintain eye contact"

    # ---------- FIDGET DETECTION ----------

    hand_results = hands.process(rgb)

    if hand_results.multi_hand_landmarks:

        hand = hand_results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        cx = wrist.x
        cy = wrist.y

        if previous_hand_x is not None:

            movement = math.sqrt(
                (cx - previous_hand_x) ** 2 +
                (cy - previous_hand_y) ** 2
            )

            if movement > 0.02:
                fidget_feedback = "Don't fidget"

        previous_hand_x = cx
        previous_hand_y = cy

    # ---------- SLOW CONFIDENCE SYSTEM ----------

    if posture_feedback == "Good posture":
        confidence_score += 0.08
    else:
        confidence_score -= 0.6

    if eye_feedback == "Good eye contact":
        confidence_score += 0.08
    else:
        confidence_score -= 0.6

    if fidget_feedback == "Stable":
        confidence_score += 0.04
    else:
        confidence_score -= 0.6

    # clamp score
    confidence_score = max(0, min(100, confidence_score))
    confidence_score = round(confidence_score, 2)

    session_scores.append(confidence_score)

    feedback = f"{posture_feedback} | {eye_feedback} | {fidget_feedback}"

    return jsonify({
        "feedback": feedback,
        "score": confidence_score
    })


@app.route('/report')
def report():

    if len(session_scores) == 0:
        return jsonify({"error": "No session data"})

    avg_score = sum(session_scores) / len(session_scores)

    return jsonify({
        "average_score": round(avg_score, 2),
        "trend": session_scores
    })


if __name__ == "__main__":
    app.run(debug=True)