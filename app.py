from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import random

app = Flask(__name__)

# ---------- MEDIAPIPE INITIALIZATION ----------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

previous_hand_x = None
previous_hand_y = None

INTERVIEW_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want this job?",
    "What are your strengths and weaknesses?",
    "Describe a challenge you faced and how you solved it.",
    "Where do you see yourself in 5 years?",
    "Why should we hire you?",
    "Tell me about a time you worked in a team.",
    "Describe a situation where you showed leadership."
]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/question')
def question():
    q = random.choice(INTERVIEW_QUESTIONS)
    return jsonify({"question": q})

@app.route('/analyze', methods=['POST'])
def analyze():

    global previous_hand_x, previous_hand_y

    data = request.json['image']

    # ---------- IMAGE DECODING ----------
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    posture_feedback = "Good posture"
    eye_feedback = "Good eye contact"
    fidget_feedback = "Stable"

    posture_score = 40
    eye_score = 40
    fidget_score = 20

    # ---------- POSTURE DETECTION ----------
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[11]
        right_shoulder = pose_results.pose_landmarks.landmark[12]

        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            posture_feedback = "Sit straight"
            posture_score = 20

    # ---------- EYE CONTACT DETECTION ----------
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]

        nose = face.landmark[1]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        eye_center = (left_eye.x + right_eye.x) / 2

        if abs(nose.x - eye_center) > 0.05:
            eye_feedback = "Maintain eye contact"
            eye_score = 20

    # ---------- FIDGET DETECTION ----------
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        current_x = wrist.x
        current_y = wrist.y

        if previous_hand_x is not None:
            movement = math.sqrt(
                (current_x - previous_hand_x) ** 2 +
                (current_y - previous_hand_y) ** 2
            )

            if movement > 0.02:
                fidget_feedback = "Don't fidget"
                fidget_score = 10

        previous_hand_x = current_x
        previous_hand_y = current_y

    # ---------- CONFIDENCE SCORE ----------
    confidence_score = posture_score + eye_score + fidget_score

    feedback = f"{posture_feedback} | {eye_feedback} | {fidget_feedback}"

    return jsonify({
        "feedback": feedback,
        "score": confidence_score
    })


if __name__ == '__main__':
    app.run(debug=True)