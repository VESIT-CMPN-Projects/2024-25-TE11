import { useState, useEffect } from "react";
import { doc, setDoc } from "firebase/firestore";
import { db } from "../firebase-config";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import ContactInfoSection from "../components/profile/ContactInfo";
import ProfessionalSummarySection from "../components/profile/ProfessionalSummary";
import SkillsSection from "../components/profile/Skills";
import WorkExperienceSection from "../components/profile/WorkExperience";
import EducationSection from "../components/profile/Education";
import CertificationsSection from "../components/profile/Certifications";
import ProjectsSection from "../components/profile/Projects";
import AchievementsSection from "../components/profile/Achievements";


const CreateProfile = ({ userId: propUserId }) => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [showFinishPopup, setShowFinishPopup] = useState(false);
  const [processingState, setProcessingState] = useState("");
  const [userId, setUserId] = useState(propUserId || null);
  const [formData, setFormData] = useState({
    contactInformation: { fullName: "", phoneNumber: "", email: "", address: "" },
    professionalSummary: "",
    skills: [],
    workExperience: [],
    education: [],
    certifications: [],
    projects: [],
    achievements: [],
  });


  // Get userId from Firebase Auth if not provided as prop
  useEffect(() => {
    if (!userId) {
      const auth = getAuth();
      const unsubscribe = onAuthStateChanged(auth, (user) => {
        if (user) {
          console.log("User authenticated:", user.uid);
          setUserId(user.uid);
        } else {
          console.log("No user is signed in");
          setSaveError("You need to be logged in to create a profile. Please sign in first.");
        }
      });
     
      return () => unsubscribe();
    }
  }, [userId]);


  const steps = [
    { name: "Contact Information", component: <ContactInfoSection formData={formData} setFormData={setFormData} /> },
    { name: "Professional Summary", component: <ProfessionalSummarySection formData={formData} setFormData={setFormData} /> },
    { name: "Skills", component: <SkillsSection formData={formData} setFormData={setFormData} /> },
    { name: "Work Experience", component: <WorkExperienceSection formData={formData} setFormData={setFormData} /> },
    { name: "Education", component: <EducationSection formData={formData} setFormData={setFormData} /> },
    { name: "Certifications", component: <CertificationsSection formData={formData} setFormData={setFormData} /> },
    { name: "Projects", component: <ProjectsSection formData={formData} setFormData={setFormData} /> },
    { name: "Achievements", component: <AchievementsSection formData={formData} setFormData={setFormData} /> },
  ];


  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
      // Scroll to top when navigating to next step
      window.scrollTo(0, 0);
    }
  };


  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
      // Scroll to top when navigating to previous step
      window.scrollTo(0, 0);
    }
  };


  const handleSkip = () => {
    handleNext();
  };


  const handleSaveProgress = async () => {
    if (!userId) {
      console.error("Error: userId is not defined or null", { userId });
      setSaveError("User ID is missing. Please ensure you're logged in.");
      return false;
    }


    setSaving(true);
    setSaveError(null);
    setProcessingState("Saving profile data...");
    console.log("Attempting to save profile data for user:", userId);
    console.log("Form data being saved:", formData);


    try {
      const profileRef = doc(db, "userProfiles", userId);
     
      // Using setDoc with merge option to create the document if it doesn't exist
      setProcessingState("Creating/updating document in Firestore...");
      await setDoc(profileRef, formData, { merge: true });
      console.log("Profile saved successfully");
     
      setProcessingState("Save completed successfully");
      return true;
    } catch (err) {
      console.error("Error saving progress:", err);
      console.error("Error details:", {
        code: err.code,
        message: err.message,
        stack: err.stack,
        userId: userId,
        formDataKeys: Object.keys(formData)
      });
     
      let errorMessage = "Failed to save profile data. ";
     
      if (err.code === "permission-denied") {
        errorMessage += "You don't have permission to update this profile.";
      } else {
        errorMessage += err.message || "Unknown error occurred.";
      }
     
      setSaveError(errorMessage);
      setProcessingState("Error occurred during save");
      return false;
    } finally {
      setSaving(false);
    }
  };


  const handleFinish = async () => {
    console.log("Finish button clicked");
   
    if (!userId) {
      console.error("Cannot finish: userId is not defined");
      setSaveError("You need to be logged in to save your profile. Please sign in and try again.");
      return;
    }
   
    try {
      setProcessingState("Starting finish process");
      console.log("Current user ID:", userId);
     
      console.log("Attempting to save data before showing popup");
      const saveSuccessful = await handleSaveProgress();
      console.log("Save result:", saveSuccessful);
     
      if (saveSuccessful) {
        console.log("Setting popup to visible");
        setShowFinishPopup(true);
      } else {
        console.error("Save was not successful, not showing popup");
      }
    } catch (err) {
      console.error("Error in handleFinish function:", err);
      setSaveError(`An unexpected error occurred: ${err.message}`);
      setProcessingState("Error in finish process");
    }
  };


  // For demonstration purposes if no user is logged in
  const handleFinishDemo = () => {
    console.log("Demo finish button clicked");
    setShowFinishPopup(true);
  };


  const navigateToDashboard = () => {
    console.log("Navigating to dashboard");
    navigate("/jobseeker-dashboard");
  };


  const navigateToResume = () => {
    console.log("Navigating to resume view");
    navigate("/jobseeker-dashboard");
  };


  const progressPercentage = ((currentStep + 1) / steps.length) * 100;


  return (
    <div className="max-w-4xl mx-auto p-8 bg-white rounded-xl shadow-lg border border-gray-100">
      <h1 className="text-3xl font-bold mb-6 text-gray-800 text-center">Build Your Professional Profile</h1>
     
      {/* Step indicators */}
      <div className="mb-8 overflow-x-auto">
        <div className="flex justify-between min-w-max">
          {steps.map((step, index) => (
            <div key={index} className="flex flex-col items-center mx-2">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 transition-all duration-300
                  ${index < currentStep ? 'bg-green-500 text-white' :
                    index === currentStep ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'}`}
              >
                {index < currentStep ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  index + 1
                )}
              </div>
              <span className={`text-xs text-center whitespace-nowrap ${index === currentStep ? 'font-bold text-blue-600' : 'text-gray-500'}`}>
                {step.name}
              </span>
            </div>
          ))}
        </div>
      </div>


      {/* Debug info - remove in production
      {process.env.NODE_ENV !== "production" && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-xs font-mono">Debug Info:</p>
          <p className="text-xs font-mono">User ID: {userId || "not set"}</p>
          <p className="text-xs font-mono">Current State: {processingState}</p>
          <p className="text-xs font-mono">Current Step: {currentStep + 1}/{steps.length}</p>
          <p className="text-xs font-mono">Popup Visible: {showFinishPopup ? "Yes" : "No"}</p>
        </div>
      )} */}


      {/* User not logged in warning */}
      {!userId && (
        <div className="mb-6 p-4 bg-orange-50 border-l-4 border-orange-400 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-orange-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="font-medium text-orange-800">You are not logged in</p>
              <p className="text-sm text-orange-700 mt-1">You need to be logged in to save your profile. You can continue filling out the form, but your progress won't be saved.</p>
            </div>
          </div>
        </div>
      )}


      {/* Progress bar */}
      <div className="w-full bg-gray-100 rounded-full h-2.5 mb-6 overflow-hidden">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 ease-in-out"
          style={{ width: `${progressPercentage}%` }}
        ></div>
      </div>


      {/* Step indicator */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-700">
          {steps[currentStep].name}
        </h2>
        <button
          onClick={handleSaveProgress}
          className={`flex items-center px-3 py-1.5 rounded-lg transition-colors
            ${!userId ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-blue-50 text-blue-600 hover:bg-blue-100'}`}
          disabled={saving || !userId}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          {saving ? "Saving..." : "Save Progress"}
        </button>
      </div>


      {saveError && (
        <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-400 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="font-medium text-red-800">Error</p>
              <p className="text-sm text-red-700 mt-1">{saveError}</p>
            </div>
          </div>
        </div>
      )}


      {/* Current step component with card styling */}
      <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mb-8 transition-all duration-300">
        {steps[currentStep].component}
      </div>


      {/* Navigation buttons */}
      <div className="flex justify-between items-center mt-8">
        <button
          onClick={handlePrevious}
          disabled={currentStep === 0}
          className={`flex items-center px-5 py-2.5 rounded-lg transition-colors
            ${currentStep === 0 ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "bg-gray-200 text-gray-700 hover:bg-gray-300"}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Previous
        </button>


        <button
          onClick={handleSkip}
          className="px-4 py-2 text-gray-500 hover:text-gray-700 font-medium transition-colors"
        >
          Skip for now
        </button>


        {currentStep < steps.length - 1 ? (
          <button
            onClick={handleNext}
            className="flex items-center px-5 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Next
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        ) : (
          <button
            onClick={userId ? handleFinish : handleFinishDemo}
            className="flex items-center px-5 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            disabled={saving}
          >
            {saving ? (
              <>
                <svg className="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Saving...
              </>
            ) : (
              <>
                Complete Profile
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </>
            )}
          </button>
        )}
      </div>


      {/* Finish Popup with improved modal design */}
      {showFinishPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all animate-fade-in-up">
            <div className="flex justify-center mb-4">
              <div className="rounded-full bg-green-100 p-3">
                <svg className="h-10 w-10 text-green-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <h2 className="text-2xl font-bold text-center text-gray-800 mb-2">Profile Created Successfully!</h2>
            <p className="text-center text-gray-600 mb-6">
              {userId
                ? "Your professional profile has been created and saved. What would you like to do next?"
                : "Demo mode: Your profile would have been created if you were logged in. What would you like to do next?"}
            </p>
            <div className="flex flex-col space-y-3">
              <button
                onClick={navigateToDashboard}
                className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                Go to Dashboard
              </button>
              <button
                onClick={navigateToResume}
                className="w-full px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                View My Resume
              </button>
              <button
                onClick={() => setShowFinishPopup(false)}
                className="w-full px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors mt-2"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};


export default CreateProfile;