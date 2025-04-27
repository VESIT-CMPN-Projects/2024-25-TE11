import React, { useState, useEffect } from "react";
import { collection, getDocs, query, where } from "firebase/firestore";
import { db } from "../firebase-config";
import JobCard from "./JobCard";

const JobsListingPage = ({ jobs: propJobs, loading: propLoading, userProfile }) => {
  const [localJobs, setLocalJobs] = useState([]);
  const [loading, setLoading] = useState(propLoading !== undefined ? propLoading : true);
  const [debugInfo, setDebugInfo] = useState({ 
    userSkillsFound: false,
    skillsCount: 0,
    jobsWithSkillsCount: 0,
    matchingAttempted: false,
    error: null
  });
  
  // Use either the props jobs or local jobs
  const jobsToDisplay = propJobs || localJobs;

  useEffect(() => {
    if (propJobs && userProfile) {
      // Process the jobs prop with user skills
      processJobsWithUserSkills(propJobs, userProfile);
      return;
    }
    
    if (!propJobs) {
      // If no jobs prop, fetch jobs and process them
      fetchJobsAndProcess();
    }
  }, [propJobs, userProfile]);

  const extractUserSkills = (profile) => {
    if (!profile) return [];
    
    let extractedSkills = [];
    let debugData = {
      userSkillsFound: false,
      skillsCount: 0,
      jobsWithSkillsCount: 0,
      matchingAttempted: false,
      error: null
    };
    
    // Check if skills object exists
    if (!profile.skills) {
      console.error("DEBUG: User profile missing skills object");
      debugData.error = "User profile missing skills object";
    } 
    // Check if technical skills array exists
    else if (!profile.skills.technical) {
      console.error("DEBUG: User profile missing technical skills array");
      debugData.error = "User profile missing technical skills array";
    }
    // Extract skills if they exist
    else if (profile.skills.technical) {
      extractedSkills = profile.skills.technical;
      console.log("DEBUG: Found technical skills in userProfile data:", extractedSkills);
      debugData.userSkillsFound = true;
    }
    
    // Normalize skills to lowercase
    const normalizedSkills = extractedSkills.map(skill => {
      if (typeof skill === 'string') {
        return skill.toLowerCase();
      } else if (skill && typeof skill === 'object' && skill.name) {
        return skill.name.toLowerCase();
      }
      console.log("DEBUG: Unexpected skill format:", skill);
      return '';
    }).filter(Boolean);
    
    console.log("DEBUG: Normalized user skills:", normalizedSkills);
    debugData.skillsCount = normalizedSkills.length;
    
    setDebugInfo(prev => ({...prev, ...debugData}));
    return normalizedSkills;
  };

  const processJobsWithUserSkills = (jobs, profile) => {
    try {
      setLoading(true);
      const userSkills = extractUserSkills(profile);
      
      if (!userSkills.length) {
        console.log("DEBUG: No user skills found, skipping matching");
        setLocalJobs(jobs);
        setLoading(false);
        return;
      }
      
      // Count jobs with required skills
      const jobsWithSkills = jobs.filter(job => 
        job.requiredSkills && Array.isArray(job.requiredSkills) && job.requiredSkills.length > 0
      );
      
      console.log(`DEBUG: ${jobsWithSkills.length} jobs have required skills`);
      setDebugInfo(prev => ({
        ...prev, 
        jobsWithSkillsCount: jobsWithSkills.length,
        matchingAttempted: true
      }));
      
      // Calculate match percentages
      const jobsWithMatch = jobs.map(job => {
        // Check if job has requiredSkills and it's an array
        if (!job.requiredSkills || !Array.isArray(job.requiredSkills)) {
          console.log(`DEBUG: Job "${job.jobTitle}" has no valid requiredSkills`);
          return { ...job, matchPercentage: 0 };
        }
        
        const jobSkills = job.requiredSkills.map(skill => 
          typeof skill === 'string' ? skill.toLowerCase() : ''
        ).filter(Boolean);
        
        console.log(`DEBUG: Job "${job.jobTitle}" skills:`, jobSkills);
        
        let matchCount = 0;
        let matchDetails = [];
        
        for (const jobSkill of jobSkills) {
          let matched = false;
          for (const userSkill of userSkills) {
            if (jobSkill === userSkill || 
                jobSkill.includes(userSkill) || 
                userSkill.includes(jobSkill)) {
              matched = true;
              matchDetails.push(`"${jobSkill}" matched with "${userSkill}"`);
              break;
            }
          }
          if (matched) matchCount++;
        }
        
        const matchPercentage = jobSkills.length ? Math.round((matchCount / jobSkills.length) * 100) : 0;
        
        console.log(`DEBUG: Job "${job.jobTitle}" match: ${matchCount}/${jobSkills.length} = ${matchPercentage}%`);
        if (matchDetails.length > 0) {
          console.log(`DEBUG: Match details for "${job.jobTitle}":`, matchDetails);
        }
        
        return { ...job, matchPercentage };
      });
      
      // Sort by match percentage
      jobsWithMatch.sort((a, b) => b.matchPercentage - a.matchPercentage);
      setLocalJobs(jobsWithMatch);
    } catch (error) {
      console.error("DEBUG: Error in job matching:", error);
      setDebugInfo(prev => ({
        ...prev,
        error: error.message
      }));
      setLocalJobs(jobs);
    } finally {
      setLoading(false);
    }
  };

  const fetchJobsAndProcess = async () => {
    setLoading(true);
    try {
      // Fetch jobs
      const jobsRef = collection(db, "jobs");
      const jobsQuery = query(jobsRef, where("status", "==", "active"));
      const jobsSnapshot = await getDocs(jobsQuery);
      
      const jobsList = jobsSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      
      console.log(`DEBUG: Found ${jobsList.length} jobs`);
      
      // Process jobs with user profile if available
      if (userProfile) {
        processJobsWithUserSkills(jobsList, userProfile);
      } else {
        setLocalJobs(jobsList);
        setLoading(false);
      }
    } catch (error) {
      console.error("DEBUG: Error fetching jobs:", error);
      setDebugInfo(prev => ({
        ...prev,
        error: error.message
      }));
      setLocalJobs([]);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      
      {/* Debug information panel - only show in development */}
      {/* {process.env.NODE_ENV !== 'production' && (
        <div className="bg-gray-100 p-4 mb-6 rounded-lg text-sm">
          <h3 className="font-bold mb-2">Debug Information:</h3>
          <ul className="list-disc pl-5">
            <li>User profile available: {userProfile ? 'Yes' : 'No'}</li>
            <li>User skills found: {debugInfo.userSkillsFound ? 'Yes' : 'No'}</li>
            <li>Number of user skills: {debugInfo.skillsCount}</li>
            <li>Jobs with required skills: {debugInfo.jobsWithSkillsCount}</li>
            <li>Matching algorithm ran: {debugInfo.matchingAttempted ? 'Yes' : 'No'}</li>
            {debugInfo.error && <li className="text-red-600">Error: {debugInfo.error}</li>}
          </ul>
          <p className="mt-2 text-gray-600">Check browser console for detailed logs</p>
        </div>
      )} */}
      
      <div className="space-y-6">
        {jobsToDisplay.length > 0 ? (
          jobsToDisplay.map(job => (
            <JobCard 
              key={job.id} 
              job={job} 
              matchPercentage={job.matchPercentage || 0}
            />
          ))
        ) : (
          <p className="text-gray-600 text-center py-8">No jobs found. Check back later for new opportunities.</p>
        )}
      </div>
    </div>
  );
};

export default JobsListingPage;