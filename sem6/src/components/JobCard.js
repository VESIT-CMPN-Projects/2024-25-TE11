import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { doc, getDoc, collection, query, where, getDocs } from "firebase/firestore";
import { db, auth } from "../firebase-config";

import { 
  MapPinIcon, 
  CheckIcon, 
  ChevronDownIcon, 
  DocumentCheckIcon, 
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';

// Export an object with named icon components
const Icons = {
  MapPin: (props) => <MapPinIcon {...props} />,
  Check: (props) => <CheckIcon {...props} />,
  ChevronDown: (props) => <ChevronDownIcon {...props} />,
  FileCheck: (props) => <DocumentCheckIcon {...props} />,
  AlertTriangle: (props) => <ExclamationTriangleIcon {...props} />
};

const JobCard = ({ job, matchPercentage = 0 }) => {
  const [expanded, setExpanded] = useState(false);
  const [companyData, setCompanyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasApplied, setHasApplied] = useState(false);
  const [testStatus, setTestStatus] = useState(null);
  const navigate = useNavigate();

  // Check if user has already applied for this job
  useEffect(() => {
    const checkApplicationStatus = async () => {
      const user = auth.currentUser;
      if (!user) return;

      try {
        // Check user's profile for applied jobs
        const userDoc = await getDoc(doc(db, "candidates", user.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          const appliedJobs = userData.appliedJobs || [];
          const application = appliedJobs.find(app => app.jobId === job.id);
          
          if (application) {
            setHasApplied(true);
          }
        }

        // Check if user has taken the test
        const testsRef = collection(db, "tests");
        const q = query(
          testsRef, 
          where("userId", "==", user.uid),
          where("jobId", "==", job.id)
        );
        
        const testSnapshot = await getDocs(q);
        if (!testSnapshot.empty) {
          // Get the most recent test
          const tests = testSnapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data()
          }));
          
          // Sort by timestamp (most recent first)
          tests.sort((a, b) => b.timestamp.seconds - a.timestamp.seconds);
          
          const latestTest = tests[0];
          setTestStatus({
            id: latestTest.id,
            score: latestTest.percentage,
            pass: latestTest.pass
          });
        }
      } catch (error) {
        console.error("Error checking application status:", error);
      }
    };

    if (job.id) {
      checkApplicationStatus();
    }
  }, [job.id]);

  // Fetch company data when card is expanded
  useEffect(() => {
    if (!expanded) {
      setCompanyData(null);
    } else if (job.userId) {
      const fetchCompanyData = async () => {
        if (expanded && job.userId) {
          try {
            setLoading(true);
            const companyDocRef = doc(db, "employers", job.userId);
            const companyDoc = await getDoc(companyDocRef);
            
            if (companyDoc.exists()) {
              setCompanyData(companyDoc.data());
            } else {
              setCompanyData(null);
            }
          } catch (error) {
            console.error("Error fetching company data:", error);
            setCompanyData(null); // Clear data on error
          } finally {
            setLoading(false);
          }
        }
      };
    
      fetchCompanyData();
    }
  }, [expanded, job.userId]);
  
  const handleCompetencyTest = () => {
    // Navigate to competency test page with job ID
    navigate(`/competency-test/${job.id}`);
  };

  const handleViewResults = () => {
    // Navigate to test results with the test ID and job ID
    navigate(`/test-results/${testStatus.id}/${job.id}`);
  };

  // Format salary to include commas and currency symbol
  const formatSalary = (salary) => {
    return salary ? `₹${salary.toLocaleString()}` : "Not disclosed";
  };

  // Get color based on match percentage
  // const getMatchColor = () => {
  //   if (matchPercentage >= 80) return "bg-green-100 text-green-800";
  //   if (matchPercentage >= 60) return "bg-blue-100 text-blue-800";
  //   if (matchPercentage >= 40) return "bg-yellow-100 text-yellow-800";
  //   return "bg-orange-100 text-orange-800";
  // };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 transition-all duration-200 hover:shadow-lg">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-xl font-semibold text-gray-800">{job.jobTitle}</h3>
          <p className="text-gray-600 font-medium">{job.companyName}</p>
          <div className="flex flex-wrap gap-2 mt-3">
            {job.requiredSkills && job.requiredSkills.map((skill, index) => (
              <span
                key={index}
                className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-sm"
              >
                {skill}
              </span>
            ))}
          </div>
        </div>
        <div className="flex flex-col items-end">
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium mb-2">
            {job.jobType}
          </span>
          
          {/* Show match percentage */}
          {/* <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getMatchColor()}`}>
            {matchPercentage}% Match
          </span> */}
          
          {/* Show application status badge if applied */}
          {hasApplied && (
            <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium mt-2 flex items-center">
              <Icons.Check className="w-3 h-3 mr-1" />
              Applied
            </span>
          )}
          
          <span className="text-gray-500 text-sm mt-2">
            {job.createdAt ? new Date(job.createdAt.toDate()).toLocaleDateString() : "Recently posted"}
          </span>
        </div>
      </div>

      <div className="mt-4">
        <div className="mt-4 flex justify-between items-center">
          <div className="text-gray-700 font-medium">
            {formatSalary(job.salary)} <span className="text-gray-500 font-normal">per year</span>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-blue-600 hover:text-blue-800 flex items-center text-sm font-medium"
          >
            {expanded ? "Show less" : "Show more details"}
            <Icons.ChevronDown className={`ml-1 w-4 h-4 transition-transform ${expanded ? "transform rotate-180" : ""}`} />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-6 border-t pt-4 animate-fadeIn">
          <h4 className="font-medium text-gray-800 mb-2">Job Description</h4>
          <p className="text-gray-700 mb-4 whitespace-pre-line">{job.jobDescription}</p>
          
          {/* Company information section */}
          {loading ? (
            <div className="flex justify-center items-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-700"></div>
            </div>
          ) : companyData ? (
            <div className="mb-4">
              <h4 className="font-medium text-gray-800 mb-2">About {job.companyName}</h4>
              <div className="text-gray-700">
                {companyData.industry && <p><span className="font-medium">Industry:</span> {companyData.industry}</p>}
                {companyData.companySize && <p><span className="font-medium">Company Size:</span> {companyData.companySize}</p>}
                {companyData.headquarters && <p><span className="font-medium">Location:</span> {companyData.headquarters}</p>}
                {companyData.about && <p><span className="font-medium">About Company:</span> {companyData.about}</p>}
                {companyData.website && (
                  <p>
                    <span className="font-medium">Website:</span>{" "}
                    <a 
                      href={companyData.website.startsWith("http") ? companyData.website : `https://${companyData.website}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800"
                    >
                      {companyData.website}
                    </a>
                  </p>
                )}
              </div>
            </div>
          ) : null}
          
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4 pt-3 border-t">
            {hasApplied ? (
              <button
                className="bg-green-600 hover:bg-green-700 text-white py-2 px-6 rounded-md flex items-center justify-center cursor-default"
              >
                <Icons.Check className="w-4 h-4 mr-2" />
                Application Submitted
              </button>
            ) : testStatus ? (
              testStatus.pass ? (
                <button
                  onClick={handleViewResults}
                  className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-md flex items-center justify-center"
                >
                  <Icons.FileCheck className="w-4 h-4 mr-2" />
                  View Test Results & Apply
                </button>
              ) : (
                <button
                  onClick={handleViewResults}
                  className="bg-orange-600 hover:bg-orange-700 text-white py-2 px-6 rounded-md flex items-center justify-center"
                >
                  <Icons.AlertTriangle className="w-4 h-4 mr-2" />
                  View Test Results ({testStatus.score}%)
                </button>
              )
            ) : (
              <button
                onClick={handleCompetencyTest}
                className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-md flex items-center justify-center"
              >
                <Icons.FileCheck className="w-4 h-4 mr-2" />
                Take Competency Test
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default JobCard;