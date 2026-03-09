let questions=[]
let qIndex=0

let interviewActive=false

let timeLeft=180
let timerInterval=null

let chartInstance=null

const video=document.getElementById("video")
const canvas=document.getElementById("hidden")
const ctx=canvas.getContext("2d")

const questionText=document.getElementById("question")
const timerText=document.getElementById("timer")
const scoreCircle=document.getElementById("scoreCircle")

const postureStatus=document.getElementById("postureStatus")
const eyeStatus=document.getElementById("eyeStatus")
const fidgetStatus=document.getElementById("fidgetStatus")

const progressTracker=document.getElementById("progressTracker")

navigator.mediaDevices.getUserMedia({video:true})
.then(stream=>{
video.srcObject=stream
})

function startInterview(){

fetch("/reset")

fetch("/questions")
.then(r=>r.json())
.then(data=>{

questions=data.questions
qIndex=0

questionText.innerText=questions[qIndex]

progressTracker.innerText="Question "+(qIndex+1)+" / "+questions.length

})

interviewActive=true
timeLeft=180

timerInterval=setInterval(()=>{

if(!interviewActive) return

if(timeLeft<=0){

finishInterview()
clearInterval(timerInterval)
return

}

timerText.innerText=timeLeft
timeLeft--

},1000)

}


function nextQuestion(){

if(qIndex<questions.length-1){

qIndex++

questionText.innerText=questions[qIndex]

progressTracker.innerText="Question "+(qIndex+1)+" / "+questions.length

}

}


function finishInterview(){

interviewActive=false

fetch("/report")
.then(r=>r.json())
.then(data=>{

document.getElementById("report").innerHTML=`

<h2>Interview Report</h2>

<p><b>Confidence:</b> ${data.average_score}</p>
<p><b>Posture:</b> ${data.posture_score}%</p>
<p><b>Eye Contact:</b> ${data.eye_score}%</p>
<p><b>Fidget:</b> ${data.fidget_ratio}%</p>

<h3>Suggestions</h3>

<ul>
${data.suggestions.map(s=>`<li>${s}</li>`).join("")}
</ul>

`

drawChart(data.trend)

})

}


function drawChart(scores){

const ctxChart=document.getElementById("chart").getContext("2d")

if(chartInstance){
chartInstance.destroy()
}

chartInstance=new Chart(ctxChart,{

type:"line",

data:{

labels:scores.map((_,i)=>i+1),

datasets:[{

label:"Confidence Trend",

data:scores,

borderColor:"#22c55e",

backgroundColor:"rgba(34,197,94,0.2)",

fill:true

}]

}

})

}


setInterval(()=>{

if(!interviewActive) return
if(!video.videoWidth) return

canvas.width=video.videoWidth
canvas.height=video.videoHeight

ctx.drawImage(video,0,0)

const img=canvas.toDataURL("image/jpeg")

fetch("/analyze",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
image:img
})

})

.then(r=>r.json())
.then(data=>{

scoreCircle.innerText=data.score

let parts=data.feedback.split("|")

postureStatus.innerText="Posture: "+parts[0]
eyeStatus.innerText="Eye Contact: "+parts[1]
fidgetStatus.innerText="Fidget: "+parts[2]

})

},800)