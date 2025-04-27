import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const CourseTest = () => {
  const [testInProgress, setTestInProgress] = useState(false);
  const [testCompleted, setTestCompleted] = useState(false);
  const [showCertPopup, setShowCertPopup] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [testScore, setTestScore] = useState(0);
  const [courseInfo, setCourseInfo] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [certificationData, setCertificationData] = useState({
    name: "",
    issuer: "",
    date: new Date().toISOString().split('T')[0],
    imageUrl: "",
    skills: [""]
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageUploading, setImageUploading] = useState(false);
  
  const navigate = useNavigate();

  useEffect(() => {
    // Check if there's a recently viewed course
    const lastViewedCourse = sessionStorage.getItem('lastViewedCourse');
    
    if (lastViewedCourse) {
      const parsedCourse = JSON.parse(lastViewedCourse);
      const currentTime = Date.now();
      const viewedTime = parsedCourse.timeStamp;
      
      // Only show popup if the course was viewed within the last 5 minutes
      if (currentTime - viewedTime < 5 * 60 * 1000) {
        setCourseInfo(parsedCourse);
        setCertificationData({
          ...certificationData,
          name: parsedCourse.title || "",
          issuer: parsedCourse.provider || "",
          skills: [parsedCourse.skill || ""]
        });
        
        // Set a timeout to show the popup after 30 seconds
        const timer = setTimeout(() => {
          setShowPopup(true);
        }, 6000); // 30 seconds
        
        return () => clearTimeout(timer);
      }
    }
  }, []);

  // Test questions based on the course content
  const testQuestions = [
    {
      question: `What is the primary focus of the ${courseInfo?.title || "course"}?`,
      options: [
        `Learning ${courseInfo?.skill || "skills"} fundamentals`,
        `Advanced ${courseInfo?.skill || "skills"} techniques`,
        `${courseInfo?.skill || "Skills"} certification preparation`,
        `${courseInfo?.skill || "Skills"} in enterprise environments`
      ],
      correctAnswer: 0
    },
    {
      question: `Which of the following is NOT covered in the ${courseInfo?.title || "course"}?`,
      options: [
        "Fundamentals and basic concepts",
        "Advanced implementation strategies",
        "Enterprise-level architecture",
        "Machine learning algorithms"
      ],
      correctAnswer: 3
    },
    {
      question: "What is the recommended prerequisite knowledge for this course?",
      options: [
        "No prior knowledge required",
        "Basic understanding of the subject",
        "Intermediate level experience",
        "Expert level knowledge"
      ],
      correctAnswer: 1
    },
    {
      question: "How much time should you allocate for completing this course?",
      options: [
        "Less than 1 week",
        "1-2 weeks",
        "3-4 weeks",
        "More than 1 month"
      ],
      correctAnswer: 2
    },
    {
      question: `What is the most valuable skill you'll gain from the ${courseInfo?.title || "course"}?`,
      options: [
        "Theoretical knowledge",
        "Practical implementation skills",
        "Certification preparation",
        "Industry networking"
      ],
      correctAnswer: 1
    },
    {
      question: "Which learning approach is most emphasized in this course?",
      options: [
        "Reading and theory",
        "Video lectures",
        "Hands-on projects",
        "Group discussions"
      ],
      correctAnswer: 2
    },
    {
      question: "What type of assessment is primarily used throughout the course?",
      options: [
        "Multiple choice quizzes",
        "Practical projects",
        "Peer reviews",
        "Written essays"
      ],
      correctAnswer: 1
    },
    {
      question: "What is the career impact of completing this course?",
      options: [
        "Immediate job qualification",
        "Skill enhancement for current role",
        "Preparation for advanced certification",
        "Prerequisite for further courses"
      ],
      correctAnswer: 2
    },
    {
      question: "How does this course compare to similar offerings?",
      options: [
        "More theoretical focus",
        "More practical exercises",
        "More comprehensive coverage",
        "More specialized content"
      ],
      correctAnswer: 1
    },
    {
      question: "What is the most common feedback from previous learners?",
      options: [
        "Too basic for advanced learners",
        "Too difficult for beginners",
        "Excellent balance of theory and practice",
        "Needs more interactive content"
      ],
      correctAnswer: 2
    }
  ];

  const handleStartTest = () => {
    setTestInProgress(true);
    setShowPopup(false);
    sessionStorage.removeItem('lastViewedCourse');
  };

  // Handle selecting an answer
  const handleSelectAnswer = (questionIndex, answerIndex) => {
    setAnswers({
      ...answers,
      [questionIndex]: answerIndex
    });
  };

  // Handle navigating to the next question
  const handleNextQuestion = () => {
    if (currentQuestion < testQuestions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      // Calculate score if at the last question
      calculateScore();
    }
  };

  // Handle navigating to the previous question
  const handlePreviousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  // Calculate the test score
  const calculateScore = () => {
    let correct = 0;
    testQuestions.forEach((q, index) => {
      if (answers[index] === q.correctAnswer) {
        correct++;
      }
    });
    
    const percentage = (correct / testQuestions.length) * 100;
    setTestScore(percentage);
    setTestCompleted(true);
    
    if (percentage >= 70) {
      // Show certificate upload popup if score is 70% or higher (7 out of 10 questions)
      setShowCertPopup(true);
    }
  };

  // Handle image upload for certificate
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please upload a valid image file (JPEG, PNG, or GIF)');
      return;
    }

    setImageUploading(true);
    setError(null);

    try {
      // Mock image upload with setTimeout
      // In a real application, you would upload to your server/Cloudinary
      setTimeout(() => {
        // Create a dummy image URL (in a real app, this would be the uploaded image URL)
        const dummyImageUrl = URL.createObjectURL(file);
        
        // Update the certification image URL
        setCertificationData({
          ...certificationData,
          imageUrl: dummyImageUrl
        });
        
        setImageUploading(false);
      }, 1500);

    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Failed to upload image. Please try again.');
      setImageUploading(false);
    }
  };

  // Handle saving the certification to user profile
  const handleSaveCertification = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Mock saving data with setTimeout
      // In a real application, you would save to Firebase/your database
      setTimeout(() => {
        // Close the certification popup
        setShowCertPopup(false);
        setTestInProgress(false);
        setTestCompleted(false);
        
        // Show success message
        alert('Certification saved successfully!');
        navigate('/jobseeker-dashboard');
        
        setLoading(false);
      }, 1500);
      
    } catch (err) {
      console.error('Error saving certification:', err);
      setError('Failed to save certification. Please try again.');
      setLoading(false);
    }
  };

  // If no popup or test is active, don't render anything
  if (!showPopup && !testInProgress) return null;

  return (
    <>
      {/* Initial popup asking if course is complete */}
      {showPopup && !testInProgress && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Course Complete?</h3>
            <p className="text-gray-600 mb-4">
              Have you completed "{courseInfo?.title || 'the course'}"? Take the competency test and upload your certification to showcase your skills!
            </p>
            <div className="flex justify-end space-x-3">
              <button 
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                onClick={() => {
                  setShowPopup(false);
                  sessionStorage.removeItem('lastViewedCourse');
                }}
              >
                Not Yet
              </button>
              <button 
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                onClick={handleStartTest}
              >
                Take Test
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Test questions */}
      {testInProgress && !testCompleted && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-bold text-gray-800">
                Knowledge Assessment
              </h3>
              <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full font-medium">
                Question {currentQuestion + 1} of {testQuestions.length}
              </span>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-4">{testQuestions[currentQuestion].question}</h4>
              <div className="space-y-3">
                {testQuestions[currentQuestion].options.map((option, index) => (
                  <div 
                    key={index}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                      answers[currentQuestion] === index 
                        ? 'bg-blue-50 border-blue-500' 
                        : 'hover:bg-gray-50 border-gray-300'
                    }`}
                    onClick={() => handleSelectAnswer(currentQuestion, index)}
                  >
                    <div className="flex items-center">
                      <div className={`w-5 h-5 rounded-full border flex items-center justify-center mr-3 ${
                        answers[currentQuestion] === index 
                          ? 'border-blue-500 bg-blue-500' 
                          : 'border-gray-400'
                      }`}>
                        {answers[currentQuestion] === index && (
                          <div className="w-2 h-2 rounded-full bg-white"></div>
                        )}
                      </div>
                      <span>{option}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-between">
              <button 
                className={`px-4 py-2 border border-gray-300 rounded-md text-gray-700 ${
                  currentQuestion === 0 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-50'
                }`}
                onClick={handlePreviousQuestion}
                disabled={currentQuestion === 0}
              >
                Previous
              </button>
              <button 
                className={`px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 ${
                  answers[currentQuestion] === undefined ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                onClick={handleNextQuestion}
                disabled={answers[currentQuestion] === undefined}
              >
                {currentQuestion === testQuestions.length - 1 ? 'Submit' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Test result */}
      {testCompleted && !showCertPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <div className="text-center mb-6">
              <div className={`w-24 h-24 rounded-full mx-auto flex items-center justify-center ${
                testScore >= 70 ? 'bg-green-100' : 'bg-red-100'
              }`}>
                <span className={`text-2xl font-bold ${
                  testScore >= 70 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {Math.round(testScore)}%
                </span>
              </div>
              <h3 className="text-xl font-bold mt-4 mb-2">
                {testScore >= 70 ? 'Congratulations!' : 'Almost there!'}
              </h3>
              <p className="text-gray-600">
                {testScore >= 70 
                  ? 'You passed the assessment! You can now receive your certificate.' 
                  : 'You need to score at least 70% to pass. Review the material and try again.'}
              </p>
            </div>
            
            <div className="flex justify-center">
              {testScore >= 70 ? (
                <button 
                  className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  onClick={() => setShowCertPopup(true)}
                >
                  Get Certificate
                </button>
              ) : (
                <button 
                  className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  onClick={() => {
                    setTestCompleted(false);
                    setTestInProgress(false);
                    setCurrentQuestion(0);
                    setAnswers({});
                    navigate('/jobseeker-dashboard');
                  }}
                >
                  Review & Try Again
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Certificate upload popup */}
      {showCertPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-bold text-gray-800">
                Upload Your Certificate
              </h3>
              <button 
                onClick={() => setShowCertPopup(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                {error}
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Certificate Name</label>
                <input
                  type="text"
                  value={certificationData.name}
                  onChange={(e) => setCertificationData({...certificationData, name: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded"
                  placeholder="e.g., Web Development Fundamentals"
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Issuing Organization</label>
                  <input
                    type="text"
                    value={certificationData.issuer}
                    onChange={(e) => setCertificationData({...certificationData, issuer: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="e.g., Udemy"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Date Issued</label>
                  <input
                    type="date"
                    value={certificationData.date}
                    onChange={(e) => setCertificationData({...certificationData, date: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Skills Learned</label>
                <input
                  type="text"
                  value={certificationData.skills.join(", ")}
                  onChange={(e) => {
                    const skillsArray = e.target.value.split(",").map(skill => skill.trim()).filter(Boolean);
                    setCertificationData({...certificationData, skills: skillsArray});
                  }}
                  className="w-full p-2 border border-gray-300 rounded"
                  placeholder="e.g., React, JavaScript, Web Design (separate with commas)"
                />
                <p className="text-xs text-gray-500 mt-1">Separate skills with commas</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Certificate Image</label>
                <div className="mt-1 flex items-center">
                  {certificationData.imageUrl ? (
                    <div className="relative">
                      <img 
                        src={certificationData.imageUrl} 
                        alt={certificationData.name} 
                        className="h-32 w-auto object-contain border border-gray-300 rounded"
                      />
                      <button
                        onClick={() => setCertificationData({...certificationData, imageUrl: ""})}
                        className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center -mt-2 -mr-2"
                      >
                        ×
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <input
                        type="file"
                        id="certificate-image"
                        className="hidden"
                        accept="image/*"
                        onChange={handleImageUpload}
                      />
                      <label
                        htmlFor="certificate-image"
                        className="cursor-pointer px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                      >
                        {imageUploading ? "Uploading..." : "Upload Image"}
                      </label>
                      <p className="text-xs text-gray-500 mt-1">Upload your certification image</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={handleSaveCertification}
                disabled={loading || !certificationData.name || !certificationData.issuer}
                className={`px-4 py-2 bg-blue-600 text-white rounded-md ${
                  loading || !certificationData.name || !certificationData.issuer
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:bg-blue-700'
                }`}
              >
                {loading ? 'Saving...' : 'Save Certificate'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default CourseTest;