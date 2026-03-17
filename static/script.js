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
const scoreCircleText = document.getElementById("scoreCircle")
const scoreCircle = document.querySelector(".score-circle")

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

updateQuestion()

})

interviewActive = true
timeLeft = 180

timerText.innerText = timeLeft

timerInterval = setInterval(()=>{

if(!interviewActive) return

if(timeLeft <= 0){
finishInterview()
clearInterval(timerInterval)
return
}

timeLeft--
timerText.innerText = timeLeft

},1000)

}

// ---------------- UPDATE QUESTION ----------------

function updateQuestion(){
questionText.style.opacity = 0

setTimeout(()=>{
    questionText.innerText = questions[qIndex]

    progressTracker.innerText =
    "Question " + (qIndex+1) + " / " + questions.length

    questionText.style.opacity = 1
},200)
}

// ---------------- NEXT QUESTION ----------------

function nextQuestion(){

if(qIndex < questions.length - 1){
qIndex++
updateQuestion()
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

// smooth glow effect 🔥
element.style.transition = "0.3s"
element.style.transform = "scale(1.05)"

setTimeout(()=>{
    element.style.transform = "scale(1)"
},150)

}

// ---------------- 💧 CONFIDENCE WATER ANIMATION ----------------

function updateScore(score){

scoreCircleText.innerText = score

let level = 100 - score

// update water level
scoreCircle.style.setProperty("--water-level", level + "%")

// directly control pseudo element
scoreCircle.style.setProperty("--top", level + "%")

// color shift based on score 🔥
if(score > 75){
    scoreCircle.style.background = "#022c22"
}
else if(score > 50){
    scoreCircle.style.background = "#3f2e05"
}
else{
    scoreCircle.style.background = "#2b0a0a"
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

updateScore(data.score)

// split feedback
let parts = data.feedback.split("|")

colorStatus(postureStatus, "Posture: " + parts[0])
colorStatus(eyeStatus, "Eye: " + parts[1])
colorStatus(fidgetStatus, "Fidget: " + parts[2])

// smooth text update
feedbackText.style.opacity = 0

setTimeout(()=>{
    feedbackText.innerText = data.feedback
    feedbackText.style.opacity = 1
},200)

})

},800)