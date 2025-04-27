import React, { useState } from 'react';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const SkillsVerification = ({ technicalSkills, onSkillsVerified }) => {
  const [showVerificationModal, setShowVerificationModal] = useState(false);
  const [testInProgress, setTestInProgress] = useState(false);
  const [testCompleted, setTestCompleted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [testScore, setTestScore] = useState(0);
  const [selectedSkill, setSelectedSkill] = useState('');
  const [verifiedSkills, setVerifiedSkills] = useState([]);
  const [suggestedCourses, setSuggestedCourses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const auth = getAuth();
  const db = getFirestore();

  // Machine Learning test questions
  const mlTestQuestions = [
    { question: "What is supervised learning?", options: ["Learning without labeled data", "Learning with labeled data", "Learning through reinforcement", "Learning from unlabeled clusters"], correctAnswer: 1 },
    { question: "Which algorithm is NOT used for classification?", options: ["Logistic Regression", "Support Vector Machines", "K-means Clustering", "Random Forest"], correctAnswer: 2 },
    { question: "What is the goal of Principal Component Analysis (PCA)?", options: ["Predict target variables", "Cluster data points", "Dimensionality reduction", "Text generation"], correctAnswer: 2 },
    { question: "What does the term 'feature' refer to in machine learning?", options: ["A software product capability", "An input variable used for prediction", "A model evaluation metric", "A type of neural network"], correctAnswer: 1 },
    { question: "Which metric is most appropriate for evaluating imbalanced classification problems?", options: ["Accuracy", "ROC AUC", "Mean Squared Error", "R-squared"], correctAnswer: 1 },
    { question: "What is the purpose of the activation function in neural networks?", options: ["To initialize weights", "To introduce non-linearity", "To normalize input data", "To calculate loss"], correctAnswer: 1 },
    { question: "What is regularization in machine learning?", options: ["A technique to avoid overfitting", "A method for standardizing inputs", "A way to speed up training", "A procedure for balancing classes"], correctAnswer: 0 },
    { question: "What is the 'kernel trick' commonly used with?", options: ["Decision Trees", "Support Vector Machines", "Linear Regression", "K-Nearest Neighbors"], correctAnswer: 1 },
    { question: "Which of these is a type of unsupervised learning?", options: ["Reinforcement Learning", "Classification", "Regression", "Clustering"], correctAnswer: 3 },
    { question: "What does 'ensemble learning' refer to?", options: ["Using multiple learning algorithms together", "Training models on full datasets", "Using automatic feature selection", "Learning from streaming data"], correctAnswer: 0 },
  ];

  // Suggested courses based on score
  const mlCourses = [
    { title: "Machine Learning Fundamentals", provider: "Coursera", difficulty: "Beginner", url: "/courses/ml-fundamentals" },
    { title: "Advanced Machine Learning Specialization", provider: "Deeplearning.ai", difficulty: "Advanced", url: "/courses/advanced-ml" },
    { title: "Applied Machine Learning with Python", provider: "edX", difficulty: "Intermediate", url: "/courses/applied-ml-python" },
    { title: "Machine Learning for Data Analysis", provider: "Udacity", difficulty: "Intermediate", url: "/courses/ml-data-analysis" },
    { title: "Introduction to Machine Learning", provider: "Stanford Online", difficulty: "Beginner", url: "/courses/intro-ml" }
  ];

  const handleStartVerification = () => {
    setShowVerificationModal(true);
  };

  const handleSelectSkill = (skill) => {
    setSelectedSkill(skill);
    setTestInProgress(true);
    setShowVerificationModal(false);
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
    if (currentQuestion < mlTestQuestions.length - 1) {
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
    mlTestQuestions.forEach((q, index) => {
      if (answers[index] === q.correctAnswer) {
        correct++;
      }
    });
    
    const percentage = (correct / mlTestQuestions.length) * 100;
    setTestScore(percentage);
    setTestCompleted(true);
    
    // Generate course suggestions based on score
    if (percentage < 60) {
      // Suggest beginner courses if failed
      setSuggestedCourses(mlCourses.filter(course => course.difficulty === "Beginner"));
    } else if (percentage >= 60 && percentage < 80) {
      // Suggest intermediate courses if passed but not expert
      setSuggestedCourses(mlCourses.filter(course => course.difficulty === "Intermediate"));
    } else {
      // Suggest advanced courses if scored high
      setSuggestedCourses(mlCourses.filter(course => course.difficulty === "Advanced"));
    }
  };

  const handleVerifySkill = async () => {
    try {
      setLoading(true);
      setError(null);

      const currentUser = auth.currentUser;
      if (!currentUser) {
        setError('You must be logged in to verify skills.');
        setLoading(false);
        return;
      }

      // Reference to the user's profile document
      const userProfileRef = doc(db, 'userProfiles', currentUser.uid);
      const userProfileSnap = await getDoc(userProfileRef);

      if (!userProfileSnap.exists()) {
        setError('User profile not found.');
        setLoading(false);
        return;
      }

      // Get current verified skills
      const userProfileData = userProfileSnap.data();
      const existingVerifiedSkills = userProfileData.verifiedSkills || [];

      // Only add the skill if it's not already verified
      if (!existingVerifiedSkills.includes(selectedSkill)) {
        // Update the user profile with the verified skill
        await updateDoc(userProfileRef, {
          verifiedSkills: [...existingVerifiedSkills, selectedSkill]
        });
      }

      // Update local state
      setVerifiedSkills([...existingVerifiedSkills, selectedSkill]);
      
      // Call the callback to notify parent component
      if (onSkillsVerified) {
        onSkillsVerified([...existingVerifiedSkills, selectedSkill]);
      }

      // Close the test
      resetTest();
      setLoading(false);
    } catch (err) {
      console.error('Error verifying skill:', err);
      setError(`Failed to verify skill: ${err.message}`);
      setLoading(false);
    }
  };

  const resetTest = () => {
    setTestInProgress(false);
    setTestCompleted(false);
    setCurrentQuestion(0);
    setAnswers({});
    setSelectedSkill('');
  };

  return (
    <>
      {/* Verification Button */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold text-gray-800">Your Current Technical Skills</h3>
        <button 
          onClick={handleStartVerification}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          Verify Your Skills
        </button>
      </div>

      {/* Display Skills */}
      <div className="flex flex-wrap gap-2 mb-6">
        {technicalSkills.map((skill, index) => (
          <span key={index} className={`px-3 py-1 rounded-full text-sm font-medium ${
            verifiedSkills.includes(skill) 
              ? 'bg-green-100 text-green-800' 
              : 'bg-blue-100 text-blue-800'
          }`}>
            {skill}
            {verifiedSkills.includes(skill) && (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1 inline" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            )}
          </span>
        ))}
      </div>

      {/* Skill Selection Modal */}
      {showVerificationModal && !testInProgress && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Select a Skill to Verify</h3>
            <p className="text-gray-600 mb-4">
              Choose a skill to take a competency test and verify your knowledge.
              You'll need to score at least 60% to verify a skill.
            </p>
            <div className="space-y-2 mb-4">
              {technicalSkills.filter(skill => !verifiedSkills.includes(skill)).map((skill, index) => (
                <button
                  key={index}
                  className="w-full p-3 text-left border border-gray-300 rounded-lg hover:bg-gray-50"
                  onClick={() => handleSelectSkill(skill)}
                >
                  {skill}
                </button>
              ))}
            </div>
            <div className="flex justify-end">
              <button 
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                onClick={() => setShowVerificationModal(false)}
              >
                Cancel
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
                {selectedSkill} Knowledge Assessment
              </h3>
              <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full font-medium">
                Question {currentQuestion + 1} of {mlTestQuestions.length}
              </span>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-4">{mlTestQuestions[currentQuestion].question}</h4>
              <div className="space-y-3">
                {mlTestQuestions[currentQuestion].options.map((option, index) => (
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
                {currentQuestion === mlTestQuestions.length - 1 ? 'Submit' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Test result */}
      {testCompleted && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full p-6">
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
                {testScore >= 60 ? 'Skill Verified!' : 'Not Yet Verified'}
              </h3>
              <p className="text-gray-600">
                {testScore >= 60 
                  ? `Congratulations! You've demonstrated competency in ${selectedSkill}.` 
                  : `You need to score at least 60% to verify this skill. Keep learning and try again!`}
              </p>
            </div>
            
            {/* Suggested courses section */}
            <div className="mt-6 mb-6">
              <h4 className="text-lg font-semibold mb-3">Recommended Courses</h4>
              <div className="space-y-3">
                {suggestedCourses.map((course, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50">
                    <div className="flex justify-between items-start">
                      <div>
                        <h5 className="font-medium">{course.title}</h5>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        course.difficulty === "Beginner" ? "bg-green-100 text-green-800" :
                        course.difficulty === "Intermediate" ? "bg-blue-100 text-blue-800" :
                        "bg-purple-100 text-purple-800"
                      }`}>
                        {course.difficulty}
                      </span>
                    </div>
                    <div className="mt-2">
                      <a 
                        href="/jobseekers=dashboard" 
                        className="text-sm text-blue-600 hover:underline"
                      >
                        View Course
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="flex justify-end space-x-3">
              {testScore >= 60 && (
                <button 
                  onClick={handleVerifySkill}
                  disabled={loading}
                  className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                >
                  {loading ? 'Saving...' : 'Mark as Verified'}
                </button>
              )}
              <button 
                onClick={resetTest}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SkillsVerification;