// Simplified version without Firebase dependency
import { useState, useEffect, useRef } from 'react';

const AIMockInterview = () => {
    const [skills, setSkills] = useState(['JavaScript', 'React', 'Node.js']); // Default skills or allow user to input
    const [question, setQuestion] = useState('');
    const [transcription, setTranscription] = useState('');
    const [feedback, setFeedback] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [countdown, setCountdown] = useState(0);
    const [recordingTimer, setRecordingTimer] = useState(0);
    const [timerInterval, setTimerInterval] = useState(null);
    const [showMicPopup, setShowMicPopup] = useState(false);
    const [loading, setLoading] = useState(false);
    const [audio, setAudio] = useState(null);
    const [responses, setResponses] = useState([]);
    const [interviewActive, setInterviewActive] = useState(false);
    const [finalAnalysis, setFinalAnalysis] = useState('');
    
    // For managing skills input
    const [newSkill, setNewSkill] = useState('');
    
    // Add a new skill
    const addSkill = () => {
        if (newSkill && !skills.includes(newSkill)) {
            setSkills([...skills, newSkill]);
            setNewSkill('');
        }
    };
    
    // Remove a skill
    const removeSkill = (skillToRemove) => {
        setSkills(skills.filter(skill => skill !== skillToRemove));
    };

    useEffect(() => {
        if (audio) {
            audio.play().catch(err => console.error("Playback failed:", err));
            return () => {
                audio.pause();
                audio.currentTime = 0;
            };
        }
    }, [audio]);
 
    // Clear all intervals when component unmounts
    useEffect(() => {
        return () => {
            if (timerInterval) clearInterval(timerInterval);
        };
    }, [timerInterval]);

    const generateQuestion = async () => {
        try {
            setTranscription("");
            setFeedback("");
           
            if (audio) {
                audio.pause();
                audio.currentTime = 0;
                setAudio(null);
            }
   
            setCountdown(5);
            setRecordingTimer(0);
            
            // Send skills directly in the request
            const response = await fetch("http://127.0.0.1:8000/generate_question", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ skills: skills.join(",") })
            });
            
            const data = await response.json();
            setQuestion(data.question);
           
            // Clear any existing interval
            if (timerInterval) clearInterval(timerInterval);
            
            // Set countdown interval with proper cleanup
            const countdownInt = setInterval(() => {
                setCountdown((currentCount) => {
                    if (currentCount <= 1) {
                        clearInterval(countdownInt);
                        recordAudio();
                        return 0;
                    }
                    return currentCount - 1;
                });
            }, 1000);
            
            setTimerInterval(countdownInt);
        } catch (error) {
            console.error("Error fetching question:", error);
        }
    };
 
    const endInterview = async () => {
        if (audio) {
            audio.pause();
            audio.currentTime = 0;
            setAudio(null);
        }
        
        // Clear any active timers
        if (timerInterval) {
            clearInterval(timerInterval);
            setTimerInterval(null);
        }
   
        setInterviewActive(false);
        try {
            const response = await fetch("http://127.0.0.1:8000/final_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    responses,
                    skills: skills.join(",") 
                })
            });
            const result = await response.json();
            setFinalAnalysis(result.analysis);
        } catch (error) {
            console.error("Error fetching final analysis:", error);
        }
    };
 
    const startInterview = async () => {
        setInterviewActive(true);
        setResponses([]);
        setFinalAnalysis('');
        await generateQuestion();
    };

    const recordAudio = async () => {
        setIsRecording(true);
        setShowMicPopup(true);
        setRecordingTimer(0);
        
        // Clear any existing interval
        if (timerInterval) {
            clearInterval(timerInterval);
        }
       
        // Create a new recording timer interval
        const recordingInt = setInterval(() => {
            setRecordingTimer((currentTimer) => currentTimer + 1);
        }, 1000);
        
        setTimerInterval(recordingInt);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            let chunks = [];

            mediaRecorder.ondataavailable = event => chunks.push(event.data);
            mediaRecorder.onstop = async () => {
                // Clear the recording timer
                if (timerInterval) {
                    clearInterval(timerInterval);
                    setTimerInterval(null);
                }
                
                setShowMicPopup(false);
                setLoading(true);
                const blob = new Blob(chunks, { type: "audio/wav" });
                const formData = new FormData();
                formData.append("file", blob, "response.wav");
                formData.append("skills", skills.join(","));
                formData.append("question", question);
               
                try {
                    const response = await fetch("http://127.0.0.1:8000/upload_audio/", {
                        method: "POST",
                        body: formData
                    });
                    const result = await response.json();

                    setTranscription("Transcription: " + result.transcription);
                    setFeedback("Feedback: " + result.feedback);
                    setResponses(prev => [...prev, { 
                        question, 
                        response: result.transcription, 
                        feedback: result.feedback 
                    }]);
                   
                    setTimeout(() => {
                        const newAudio = new Audio("http://127.0.0.1:8000/static/feedback.mp3");
                        setAudio(newAudio);
                    }, 1000);
                } catch (error) {
                    console.error("Error uploading audio:", error);
                }
                setLoading(false);
                setIsRecording(false);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            
            // Automatically stop recording after 15 seconds
            setTimeout(() => {
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
            }, 15000);
            
        } catch (error) {
            console.error("Error accessing microphone:", error);
            setIsRecording(false);
            setShowMicPopup(false);
            
            // Clear the interval if there's an error
            if (timerInterval) {
                clearInterval(timerInterval);
                setTimerInterval(null);
            }
        }
    };
    
    // JSX rendering...
    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-6">Technical Interview Simulator</h1>
            
            {!interviewActive && !finalAnalysis && (
                <div className="mb-8">
                    <h2 className="text-xl font-semibold mb-4">Your Technical Skills</h2>
                    <div className="flex flex-wrap gap-2 mb-4">
                        {skills.map((skill, index) => (
                            <div key={index} className="bg-blue-100 px-3 py-1 rounded-full flex items-center">
                                <span>{skill}</span>
                                <button 
                                    onClick={() => removeSkill(skill)}
                                    className="ml-2 text-red-500"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                    <div className="flex gap-2 mb-6">
                        <input
                            type="text"
                            value={newSkill}
                            onChange={(e) => setNewSkill(e.target.value)}
                            placeholder="Add a skill"
                            className="border p-2 rounded"
                        />
                        <button 
                            onClick={addSkill}
                            className="bg-blue-500 text-white px-4 py-2 rounded"
                        >
                            Add
                        </button>
                    </div>
                    <button 
                        onClick={startInterview}
                        disabled={skills.length === 0}
                        className="bg-green-500 text-white px-6 py-3 rounded-lg font-semibold disabled:bg-gray-300"
                    >
                        Start Interview
                    </button>
                </div>
            )}
            
            {interviewActive && (
                <div>
                    <div className="bg-gray-100 p-4 rounded-lg mb-6">
                        <h3 className="font-semibold mb-2">Question:</h3>
                        <p>{question}</p>
                    </div>
                    
                    {countdown > 0 && (
                        <div className="text-center mb-4">
                            <div className="text-2xl font-bold">Get ready in {countdown}...</div>
                        </div>
                    )}
                    
                    {isRecording && (
                        <div className="text-center mb-4">
                            <div className="text-xl font-bold text-red-500">Recording... {recordingTimer}s</div>
                        </div>
                    )}
                    
                    {transcription && (
                        <div className="bg-gray-100 p-4 rounded-lg mb-4">
                            <p className="mb-4">{transcription}</p>
                            <p>{feedback}</p>
                        </div>
                    )}
                    
                    <div className="flex gap-4 mt-6">
                        <button 
                            onClick={generateQuestion}
                            className="bg-blue-500 text-white px-4 py-2 rounded"
                            disabled={isRecording || countdown > 0}
                        >
                            Next Question
                        </button>
                        
                        <button 
                            onClick={endInterview}
                            className="bg-red-500 text-white px-4 py-2 rounded"
                            disabled={isRecording || countdown > 0 || responses.length === 0}
                        >
                            End Interview
                        </button>
                    </div>
                    
                    {loading && (
                        <div className="text-center mt-4">
                            <p>Processing your response...</p>
                        </div>
                    )}
                </div>
            )}
            
            {finalAnalysis && (
                <div className="mt-8">
                    <h2 className="text-xl font-semibold mb-4">Interview Analysis</h2>
                    <div className="bg-gray-100 p-6 rounded-lg whitespace-pre-line">
                        {finalAnalysis}
                    </div>
                    <button 
                        onClick={() => {
                            setFinalAnalysis('');
                            setInterviewActive(false);
                        }}
                        className="mt-6 bg-blue-500 text-white px-4 py-2 rounded"
                    >
                        Start New Interview
                    </button>
                </div>
            )}
            
            {showMicPopup && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
                    <div className="bg-white p-6 rounded-lg text-center">
                        <div className="text-xl mb-2">🎤 Recording your answer</div>
                        <div className="text-red-500 animate-pulse">{recordingTimer}s</div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIMockInterview;