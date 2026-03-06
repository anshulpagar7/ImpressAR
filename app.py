from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']

    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    feedback = "Good posture"

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]

        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            feedback = "Sit straight"

    return jsonify({"feedback": feedback})


if __name__ == "__main__":
    app.run(debug=True)