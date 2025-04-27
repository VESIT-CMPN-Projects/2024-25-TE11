import { useState, useEffect, useCallback, memo } from 'react';
import { analyzeCareerPaths, getCourseRecommendations } from '../services/genaiService';
import { getJobRolesSkills, getUserProfile } from '../services/firebaseService';  
import { auth } from '../firebase-config'; 
import CourseView from './CourseView';
import SkillsVerification from './SkillsVerification';

// Memoized loading state component
const LoadingState = memo(() => (
  <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
    <div className="text-center p-10 bg-white rounded-2xl shadow-xl max-w-md">
      <div className="animate-spin w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-6"></div>
      <h3 className="text-2xl font-bold text-gray-800 mb-2">Analyzing Your Skills</h3>
      <p className="text-xl font-medium text-gray-700 mb-4">Creating your personalized career roadmap...</p>
      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
        <div className="bg-blue-600 h-2.5 rounded-full w-3/4 animate-pulse"></div>
      </div>
      <p className="text-gray-500 text-sm">This may take a moment</p>
    </div>
  </div>
));

// Memoized error state component
const ErrorState = memo(({ error, onRetry }) => (
  <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-red-50 to-pink-50">
    <div className="bg-white border-l-4 border-red-500 p-8 rounded-xl shadow-xl max-w-2xl">
      <div className="flex items-center">
        <div className="text-red-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div className="ml-4">
          <h3 className="text-xl font-bold text-red-800">Error Analyzing Career Paths</h3>
          <p className="text-red-700 mt-2">{error}</p>
          <p className="mt-4 text-gray-600">We're having trouble analyzing your skills at the moment.</p>
          <button 
            onClick={onRetry} 
            className="mt-4 bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
            Try Again
          </button>
        </div>
      </div>
    </div>
  </div>
));

// Memoized no analysis state component
const NoAnalysisState = memo(({ onRetry }) => (
  <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-orange-50 to-yellow-50">
    <div className="bg-white border-l-4 border-orange-500 p-8 rounded-xl shadow-xl max-w-2xl">
      <div className="flex items-center">
        <div className="text-orange-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div className="ml-4">
          <h3 className="text-xl font-bold text-orange-800">Analysis Unavailable</h3>
          <p className="text-orange-700 mt-2">We couldn't generate your career analysis at this time.</p>
          <button 
            onClick={onRetry} 
            className="mt-4 bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-lg font-medium transition-colors">
            Try Again
          </button>
        </div>
      </div>
    </div>
  </div>
));

// Memoized course card component
const CourseCard = memo(({ course, role }) => <CourseView course={course} role={role} />);

// Memoized skill badge component
const SkillBadge = memo(({ skill, isVerified }) => (
  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
    isVerified ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
  } flex items-center`}>
    {skill}
    {isVerified && (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
    )}
  </span>
));

// Memoized missing skill badge component
const MissingSkillBadge = memo(({ skill }) => (
  <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
    {skill}
  </span>
));

// Memoized career path card component
const CareerPathCard = memo(({ path, verifiedSkills, onFetchCourses }) => {

  return (
    <div className="bg-white rounded-xl shadow-md hover:shadow-xl overflow-hidden transition-all duration-300 border border-gray-100">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 border-b relative overflow-hidden">
        <div className="relative z-10 flex justify-between items-center">
          <h3 className="text-2xl font-bold text-gray-800">{path.role}</h3>
          <div className="flex items-center justify-center w-20 h-20 bg-white bg-opacity-90 rounded-full shadow-md">
            <div className="text-2xl font-bold" style={{ 
              color: path.matchPercentage > 70 ? '#10b981' : path.matchPercentage > 40 ? '#3b82f6' : '#ef4444'
            }}>
              {path.matchPercentage}%
            </div>
          </div>
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${matchColor}"></div>
      </div>
          
      <div className="p-6">
        <div className="mb-6">
          <h4 className="flex items-center font-semibold text-green-700 mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Skills You Already Have ({path.matchedSkills.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {path.matchedSkills.map((skill, idx) => (
              <SkillBadge 
                key={idx} 
                skill={skill} 
                isVerified={verifiedSkills.includes(skill)} 
              />
            ))}
          </div>
        </div>
        
        <div className="mb-6">
          <h4 className="flex items-center font-semibold text-red-700 mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Skills To Develop ({path.missingSkills.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {path.missingSkills.map((skill, idx) => (
              <MissingSkillBadge key={idx} skill={skill} />
            ))}
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={() => onFetchCourses(path.role, path.missingSkills)}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-4 py-3 rounded-lg transition-colors flex items-center justify-center shadow-md transform hover:translate-y-px"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            Course Recommendations
          </button>
        </div>
      </div>
    </div>
  );
});

// Memoized course section component for the modal
const CourseSection = memo(({ title, courses, color, icon, role }) => {
  if (courses.length === 0) {
    return (
      <div className="mb-10">
        <h3 className={`text-2xl font-bold text-${color}-800 mb-4 pb-2 border-b flex items-center`}>
          <span className={`inline-block mr-2 w-3 h-3 rounded-full bg-${color}-500`}></span>
          {title}
        </h3>
        <p className="text-gray-500 italic py-4">No courses found for this category.</p>
      </div>
    );
  }

  return (
    <div className="mb-10">
      <h3 className={`text-2xl font-bold text-${color}-800 mb-4 pb-2 border-b flex items-center`}>
        <span className={`inline-block mr-2 w-3 h-3 rounded-full bg-${color}-500`}></span>
        {title}
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map((course, index) => (
          <CourseCard key={`${color}-${index}`} course={course} role={role} />
        ))}
      </div>
    </div>
  );
});

// Memoized course recommendations overlay
const CourseRecommendationsOverlay = memo(({ isOpen, onClose, courses, currentRole, courseLoading }) => {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 z-50 overflow-y-auto backdrop-blur-sm">
      <div className="min-h-screen p-4">
        <div className="max-w-6xl mx-auto bg-white rounded-2xl shadow-2xl transform transition-all duration-300">
          <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-6 flex justify-between items-center rounded-t-2xl z-10">
            <h2 className="text-2xl font-bold">
              Courses for {currentRole} Career Path
            </h2>
            <button 
              onClick={onClose}
              className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white p-2 rounded-full transition-all hover:rotate-90 duration-300"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="p-8">
            {courseLoading ? (
              <div className="py-16 text-center">
                <div className="animate-spin w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-6"></div>
                <p className="text-xl text-gray-700">Finding the perfect courses to advance your career...</p>
              </div>
            ) : (
              <div className="space-y-6">
                <CourseSection title="Foundation Courses" courses={courses.beginner} color="blue" role={currentRole} />
                <CourseSection title="Skill-Building Courses" courses={courses.intermediate} color="green" role={currentRole} />
                <CourseSection title="Expert-Level Specializations" courses={courses.advanced} color="purple" role={currentRole} />
              </div>
            )}
          </div>
          
          <div className="p-6 border-t bg-gray-50 rounded-b-2xl flex justify-between items-center">
            <p className="text-gray-600 italic text-sm">
              Courses are personalized based on your skill gap analysis
            </p>
            <button 
              onClick={onClose} 
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-lg text-lg font-medium transition-colors shadow-md"
            >
              Back to Career Paths
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

const SkillAnalyzer = () => {
  const [loading, setLoading] = useState(true);
  const [careerAnalysis, setCareerAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [technicalSkills, setTechnicalSkills] = useState([]);
  const [verifiedSkills, setVerifiedSkills] = useState([]);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [currentRole, setCurrentRole] = useState(null);
  const [courseLoading, setCourseLoading] = useState(false);
  const [courses, setCourses] = useState({
    beginner: [],
    intermediate: [],
    advanced: []
  });

  const fetchUserSkills = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const currentUser = auth.currentUser; 
      
      if (!currentUser) {
        setError("User not logged in");
        return;
      }

      const userProfile = await getUserProfile(currentUser.uid); 
      
      if (!userProfile || !userProfile.skills || !userProfile.skills.technical) {
        setError("No skills found in user profile");
        return;
      }

      const skills = userProfile.skills.technical;
      setTechnicalSkills(skills);
      
      if (userProfile.verifiedSkills) {
        setVerifiedSkills(userProfile.verifiedSkills);
      }

      const jobRolesSkills = await getJobRolesSkills();
      const analysis = await analyzeCareerPaths(skills, jobRolesSkills);
      setCareerAnalysis(analysis);
    } catch (err) {
      setError(err.message);
      console.error('Error analyzing career paths:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUserSkills();
  }, [fetchUserSkills]);

  const handleSkillsVerified = useCallback((newVerifiedSkills) => {
    setVerifiedSkills(newVerifiedSkills);
  }, []);

  const fetchCourseRecommendations = useCallback(async (role, missingSkills) => {
    setCourseLoading(true);
    setCurrentRole(role);
    setShowRecommendations(true);
    
    try {
      const courseRecommendations = await getCourseRecommendations(role, missingSkills);
      setCourses(courseRecommendations);
    } catch (err) {
      console.error('Error fetching course recommendations:', err);
      setCourses({
        beginner: [],
        intermediate: [],
        advanced: []
      });
    } finally {
      setCourseLoading(false);
    }
  }, []);

  const handleCloseRecommendations = useCallback(() => {
    setShowRecommendations(false);
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} onRetry={fetchUserSkills} />;
  if (!careerAnalysis) return <NoAnalysisState onRetry={fetchUserSkills} />;

  return (
    <>
      <div className="container mx-auto p-4 max-w-7xl">    
        <div className="bg-white p-6 rounded-xl shadow-lg mb-8 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
          <SkillsVerification 
            technicalSkills={technicalSkills} 
            onSkillsVerified={handleSkillsVerified} 
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {careerAnalysis.careerPaths?.map((path, index) => (
            <CareerPathCard 
              key={index} 
              path={path} 
              verifiedSkills={verifiedSkills} 
              onFetchCourses={fetchCourseRecommendations} 
            />
          ))}
        </div>
      </div>

      <CourseRecommendationsOverlay 
        isOpen={showRecommendations}
        onClose={handleCloseRecommendations}
        courses={courses}
        currentRole={currentRole}
        courseLoading={courseLoading}
      />
    </>
  );
};

export default SkillAnalyzer;