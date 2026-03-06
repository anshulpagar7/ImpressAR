from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    data = request.json['image']

    # Decode base64 image
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    posture_feedback = "Good posture"
    eye_feedback = "Good eye contact"

    # -------- POSTURE DETECTION --------
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[11]
        right_shoulder = pose_results.pose_landmarks.landmark[12]

        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            posture_feedback = "Sit straight"

    # -------- EYE CONTACT DETECTION --------
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]

        nose = face.landmark[1]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        eye_center = (left_eye.x + right_eye.x) / 2

        if abs(nose.x - eye_center) > 0.05:
            eye_feedback = "Maintain eye contact"

    combined_feedback = posture_feedback + " | " + eye_feedback

    return jsonify({
        "feedback": combined_feedback
    })


if __name__ == '__main__':
    app.run(debug=True)