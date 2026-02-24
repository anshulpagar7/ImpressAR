from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import base64

app = Flask(__name__)

# Create pose landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
pose = vision.PoseLandmarker.create_from_options(options)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']
    
    # Decode image
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Create mediapipe image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    results = pose.detect(image)

    feedback = "Good posture"

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks[0][11]
        right_shoulder = results.pose_landmarks[0][12]

        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            feedback = "Sit straight"

    return jsonify({"feedback": feedback})

if __name__ == '__main__':
    app.run(debug=True)