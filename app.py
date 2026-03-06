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

# ---------------- SESSION DATA ----------------

confidence_score = 70
session_scores = []

total_frames = 0
good_posture_frames = 0
good_eye_frames = 0
fidget_frames = 0
movement_total = 0

# ---------------- QUESTIONS ----------------

INTRO_QUESTIONS = [
"What is your name?",
"What are you currently studying or working on?",
"What are your strongest technical skills?"
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
"Describe a difficult project you worked on",
"How do you handle pressure",
"Tell me about a time you failed",
"How do you prioritize tasks",
"How do you learn new technologies",
"Tell me about teamwork experience",
"What is a mistake you learned from",
"How do you stay productive",
"What is your dream job",
"What do you know about our company",
"Why should we hire you over others",
"What skills are you improving",
"How do you deal with criticism",
"Describe a conflict you resolved",
"What is your biggest achievement",
"What makes you unique",
"Describe a difficult deadline you handled"
]

# ---------------- ROUTES ----------------


@app.route('/')
def home():
    return render_template("index.html")


# -------- QUESTION GENERATOR --------

@app.route('/questions')
def questions():

    selected_random = random.sample(RANDOM_QUESTIONS,7)

    return jsonify({
        "questions": INTRO_QUESTIONS + selected_random
    })


# -------- ANALYZE FRAME --------

@app.route('/analyze',methods=['POST'])
def analyze():

    global previous_hand_x,previous_hand_y
    global confidence_score

    global total_frames,good_posture_frames
    global good_eye_frames,fidget_frames,movement_total

    data=request.json['image']

    image_data=base64.b64decode(data.split(',')[1])
    np_arr=np.frombuffer(image_data,np.uint8)
    frame=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    posture_feedback="Good posture"
    eye_feedback="Good eye contact"
    fidget_feedback="Stable"

    total_frames+=1

    # -------- POSTURE --------

    pose_results=pose.process(rgb)

    if pose_results.pose_landmarks:

        left=pose_results.pose_landmarks.landmark[11]
        right=pose_results.pose_landmarks.landmark[12]

        if abs(left.y-right.y)>0.05:

            posture_feedback="Sit straight"

        else:

            good_posture_frames+=1


    # -------- EYE CONTACT --------

    face_results=face_mesh.process(rgb)

    if face_results.multi_face_landmarks:

        face=face_results.multi_face_landmarks[0]

        nose=face.landmark[1]
        left_eye=face.landmark[33]
        right_eye=face.landmark[263]

        center=(left_eye.x+right_eye.x)/2

        if abs(nose.x-center)>0.05:

            eye_feedback="Maintain eye contact"

        else:

            good_eye_frames+=1


    # -------- FIDGET --------

    hand_results=hands.process(rgb)

    if hand_results.multi_hand_landmarks:

        hand=hand_results.multi_hand_landmarks[0]

        wrist=hand.landmark[0]

        cx=wrist.x
        cy=wrist.y

        if previous_hand_x is not None:

            movement=math.sqrt(
            (cx-previous_hand_x)**2+
            (cy-previous_hand_y)**2
            )

            movement_total+=movement

            if movement>0.02:

                fidget_feedback="Don't fidget"
                fidget_frames+=1

        previous_hand_x=cx
        previous_hand_y=cy


    # -------- CONFIDENCE SYSTEM --------

    if posture_feedback=="Good posture":
        confidence_score+=0.08
    else:
        confidence_score-=0.6

    if eye_feedback=="Good eye contact":
        confidence_score+=0.08
    else:
        confidence_score-=0.6

    if fidget_feedback=="Stable":
        confidence_score+=0.04
    else:
        confidence_score-=0.6


    confidence_score=max(0,min(100,confidence_score))

    session_scores.append(confidence_score)

    feedback=f"{posture_feedback} | {eye_feedback} | {fidget_feedback}"

    return jsonify({

        "feedback":feedback,
        "score":round(confidence_score,2)

    })


# -------- REPORT GENERATOR --------

@app.route('/report')
def report():

    global total_frames,good_posture_frames
    global good_eye_frames,fidget_frames
    global movement_total,session_scores

    if total_frames==0:

        return jsonify({"error":"No session data"})

    posture_score=(good_posture_frames/total_frames)*100
    eye_score=(good_eye_frames/total_frames)*100
    fidget_ratio=(fidget_frames/total_frames)*100

    movement_avg=movement_total/total_frames

    suggestions=[]

    if posture_score<70:
        suggestions.append("Improve posture and keep shoulders straight")

    if eye_score<70:
        suggestions.append("Maintain stronger eye contact")

    if fidget_ratio>30:
        suggestions.append("Reduce hand movement while speaking")

    if movement_avg>0.02:
        suggestions.append("Try to stay centered and reduce body movement")

    if len(suggestions)==0:
        suggestions.append("Great job! Your interview body language was excellent.")

    avg_score=sum(session_scores)/len(session_scores)

    return jsonify({

        "average_score":round(avg_score,2),
        "posture_score":round(posture_score,2),
        "eye_score":round(eye_score,2),
        "fidget_ratio":round(fidget_ratio,2),
        "movement":round(movement_avg,4),
        "suggestions":suggestions,
        "trend":session_scores

    })


# -------- RESET SESSION --------

@app.route('/reset')
def reset():

    global session_scores,total_frames
    global good_posture_frames,good_eye_frames
    global fidget_frames,movement_total
    global confidence_score

    session_scores=[]
    total_frames=0
    good_posture_frames=0
    good_eye_frames=0
    fidget_frames=0
    movement_total=0

    confidence_score=70

    return jsonify({"status":"reset"})


# ---------------- RUN APP ----------------

if __name__=="__main__":
    app.run(debug=True)