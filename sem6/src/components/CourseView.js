import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { doc, updateDoc, getDoc } from 'firebase/firestore';
import { db, auth } from '../firebase-config';

// A component to handle the course view and certificate flow
const CourseView = ({ course, role }) => {
  const navigate = useNavigate();
  const [showTestPopup, setShowTestPopup] = useState(false);
  const [showCertPopup, setShowCertPopup] = useState(false);
  const [testInProgress, setTestInProgress] = useState(false);
  const [testCompleted, setTestCompleted] = useState(false);
  const [testScore, setTestScore] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [certificationData, setCertificationData] = useState({
    name: course?.title || "",
    issuer: course?.provider || "",
    date: new Date().toISOString().split('T')[0],
    imageUrl: "",
    skills: [course?.skill || ""]
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageUploading, setImageUploading] = useState(false);

  // Handle the view course button click
  const handleViewCourse = () => {
    // Store course info in localStorage/sessionStorage before navigating
    sessionStorage.setItem('lastViewedCourse', JSON.stringify({
      courseId: course?.id,
      title: course?.title,
      provider: course?.provider,
      skill: course?.skill,
      timeStamp: Date.now() // Save the current timestamp
    }));
    
    // Redirect to jobseeker-dashboard
    navigate('/jobseeker-dashboard');
  };

  // Handle starting the test
  const handleStartTest = () => {
    setTestInProgress(true);
    setShowTestPopup(false);
  };
  // Test questions based on the course content
  const testQuestions = [
    {
      question: `What is the primary focus of the ${course?.title || "course"}?`,
      options: [
        `Learning ${course?.skill || "skills"} fundamentals`,
        `Advanced ${course?.skill || "skills"} techniques`,
        `${course?.skill || "Skills"} certification preparation`,
        `${course?.skill || "Skills"} in enterprise environments`
      ],
      correctAnswer: 0
    },
    {
      question: `Which of the following is NOT covered in the ${course?.title || "course"}?`,
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
      question: `What is the most valuable skill you'll gain from the ${course?.title || "course"}?`,
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
    
    if (percentage >= 60) {
      // Show certificate upload popup if score is 60% or higher
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
      const currentUser = auth.currentUser;
      if (!currentUser) {
        throw new Error('User not logged in');
      }

      // Create FormData to send to Cloudinary
      const formData = new FormData();
      formData.append('file', file);
      formData.append('upload_preset', 'profile_pictures');
      formData.append('folder', `user-profiles/${currentUser.uid}/certifications`);

      // Call Cloudinary API
      const response = await fetch(
        `https://api.cloudinary.com/v1_1/dihawgvdz/image/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error('Failed to upload image');
      }

      const data = await response.json();
      
      // Update the certification image URL
      setCertificationData({
        ...certificationData,
        imageUrl: data.secure_url
      });

    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Failed to upload image. Please try again.');
    } finally {
      setImageUploading(false);
    }
  };

  // Handle saving the certification to user profile
  const handleSaveCertification = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const currentUser = auth.currentUser;
      if (!currentUser) {
        throw new Error('User not logged in');
      }
      
      // Get the current user profile
      const userProfileRef = doc(db, "userProfiles", currentUser.uid);
      const userProfileSnap = await getDoc(userProfileRef);
      
      if (!userProfileSnap.exists()) {
        throw new Error('User profile not found');
      }
      
      const userProfile = userProfileSnap.data();
      
      // Add new certification to the array
      const updatedCertifications = [
        ...(userProfile.certifications || []),
        {
          ...certificationData,
          // Add course-specific details
          courseCompleted: course?.title || "Course",
          testScore: testScore,
          completionDate: new Date().toISOString().split('T')[0]
        }
      ];
      
      // Update the user profile
      await updateDoc(userProfileRef, {
        certifications: updatedCertifications
      });
      
      // Close the certification popup
      setShowCertPopup(false);
      
      // Show success message or redirect
      alert('Certification saved successfully!');
      navigate('/jobseeker-dashboard');
      
    } catch (err) {
      console.error('Error saving certification:', err);
      setError('Failed to save certification. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Course component with view button that redirects */}
      <div className="border-2 rounded-lg overflow-hidden shadow-md transition-all hover:shadow-lg bg-blue-50 border-blue-200">
        {/* Your existing course card JSX */}
        <div className="p-5">
          <div className="flex justify-between items-start">
            <div className="flex flex-col">
              <span className="inline-block px-3 py-1 text-xs font-medium rounded-full text-white mb-2 bg-blue-500">
                Beginner
              </span>
              {course?.skill && (
                <span className="inline-block px-3 py-1 text-xs font-medium rounded-full bg-gray-200 text-gray-700 mb-2">
                  {course.skill}
                </span>
              )}
            </div>
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span className="ml-1 text-gray-700">{course?.rating || "4.5"}</span>
            </div>
          </div>
          
          <h4 className="text-lg font-bold mb-2 text-gray-800">{course?.title || "Course Title"}</h4>
          
          <p className="text-gray-600 mb-4 text-sm line-clamp-2">{course?.description || "Course description"}</p>
          
          <div className="flex items-center text-sm text-gray-500 mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {course?.duration || "4 weeks"}
          </div>
          
          <button 
            className="block w-full bg-white border border-gray-300 hover:bg-gray-50 text-gray-800 font-medium py-2 px-4 rounded-lg transition-colors text-center"
            onClick={handleViewCourse}
          >
            View Course
          </button>
        </div>
      </div>

      {/* Pop-up for Course Completion Test */}
      {showTestPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Course Complete?</h3>
            <p className="text-gray-600 mb-4">
              Great progress on "{course?.title || 'the course'}"! Ready to test your knowledge? Take a quick assessment to upload your certificate.
            </p>
            <div className="flex justify-end space-x-3">
              <button 
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                onClick={() => setShowTestPopup(false)}
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
                testScore >= 60 ? 'bg-green-100' : 'bg-red-100'
              }`}>
                <span className={`text-2xl font-bold ${
                  testScore >= 60 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {Math.round(testScore)}%
                </span>
              </div>
              <h3 className="text-xl font-bold mt-4 mb-2">
                {testScore >= 60 ? 'Congratulations!' : 'Almost there!'}
              </h3>
              <p className="text-gray-600">
                {testScore >= 60 
                  ? 'You passed the assessment! You can now receive your certificate.' 
                  : 'You need to score at least 60% to pass. Review the material and try again.'}
              </p>
            </div>
            
            <div className="flex justify-center">
              {testScore >= 60 ? (
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

export default CourseView;