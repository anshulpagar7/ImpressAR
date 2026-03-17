// ---------------- VARIABLES ----------------

let questions = []
let qIndex = 0

let interviewActive = false

let timeLeft = 180
let timerInterval = null

const video = document.getElementById("video")
const canvas = document.getElementById("hidden")
const ctx = canvas.getContext("2d")

const questionText = document.getElementById("question")
const timerText = document.getElementById("timer")
const scoreCircle = document.getElementById("scoreCircle")

const postureStatus = document.getElementById("postureStatus")
const eyeStatus = document.getElementById("eyeStatus")
const fidgetStatus = document.getElementById("fidgetStatus")

const progressTracker = document.getElementById("progressTracker")
const feedbackText = document.getElementById("feedback")

// ---------------- CAMERA ----------------

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream
})

// ---------------- START INTERVIEW ----------------

function startInterview(){

fetch("/reset")

fetch("/questions")
.then(r => r.json())
.then(data => {

questions = data.questions
qIndex = 0

questionText.innerText = questions[qIndex]

progressTracker.innerText =
"Question " + (qIndex+1) + " / " + questions.length

})

interviewActive = true
timeLeft = 180

timerInterval = setInterval(()=>{

if(!interviewActive) return

if(timeLeft <= 0){

finishInterview()
clearInterval(timerInterval)
return

}

timerText.innerText = timeLeft
timeLeft--

},1000)

}

// ---------------- NEXT QUESTION ----------------

function nextQuestion(){

if(qIndex < questions.length - 1){

qIndex++

questionText.innerText = questions[qIndex]

progressTracker.innerText =
"Question " + (qIndex+1) + " / " + questions.length

}

}

// ---------------- FINISH ----------------

function finishInterview(){

interviewActive = false

window.location.href = "/report_page"

}

// ---------------- STATUS COLORS ----------------

function colorStatus(element, text){

element.innerText = text

if(text.includes("Good")){
element.style.color = "#22c55e"
}
else{
element.style.color = "#ef4444"
}

}

// ---------------- FRAME ANALYSIS ----------------

setInterval(()=>{

if(!interviewActive) return
if(!video.videoWidth) return

canvas.width = video.videoWidth
canvas.height = video.videoHeight

ctx.drawImage(video, 0, 0)

const img = canvas.toDataURL("image/jpeg")

fetch("/analyze",{

method: "POST",

headers: {
"Content-Type": "application/json"
},

body: JSON.stringify({
image: img
})

})

.then(r => r.json())
.then(data => {

scoreCircle.innerText = data.score

let parts = data.feedback.split("|")

colorStatus(postureStatus, "Posture: " + parts[0])
colorStatus(eyeStatus, "Eye: " + parts[1])
colorStatus(fidgetStatus, "Fidget: " + parts[2])

feedbackText.innerText = data.feedback

})

},800)