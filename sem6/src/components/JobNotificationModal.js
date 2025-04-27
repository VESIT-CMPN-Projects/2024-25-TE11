import React, { useState, useEffect } from "react";
import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebase-config";
import { useNavigate } from "react-router-dom";

// Import icons as needed
import { XMarkIcon } from '@heroicons/react/24/outline';

const JobNotificationModal = ({ isOpen, onClose, jobId }) => {
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchJobDetails = async () => {
      if (!jobId || !isOpen) return;
      
      setLoading(true);
      try {
        const jobRef = doc(db, "jobs", jobId);
        const jobSnap = await getDoc(jobRef);
        
        if (jobSnap.exists()) {
          setJob({
            id: jobSnap.id,
            ...jobSnap.data()
          });
        } else {
          console.error("Job not found");
        }
      } catch (error) {
        console.error("Error fetching job details:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchJobDetails();
  }, [jobId, isOpen]);

  const handleCompetencyTest = () => {
    // Navigate to competency test page with job ID
    navigate(`/competency-test/${jobId}`);
    onClose(); // Close the modal after navigation
  };

  // Format salary to include commas and currency symbol
  const formatSalary = (salary) => {
    return salary ? `₹${salary.toLocaleString()}` : "Not disclosed";
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black bg-opacity-60 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        {/* Header with close button */}
        <div className="flex justify-between items-center border-b p-4">
          <h3 className="text-lg font-semibold text-gray-900">Job Details</h3>
          <button
            className="text-gray-400 hover:text-gray-500 focus:outline-none"
            onClick={onClose}
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {loading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
            </div>
          ) : job ? (
            <div>
              <div className="mb-6">
                <h2 className="text-xl font-bold text-gray-800 mb-1">{job.jobTitle}</h2>
                <p className="text-gray-600 font-medium">{job.companyName}</p>
                <div className="flex items-center mt-2 gap-4">
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                    {job.jobType}
                  </span>
                  <span className="text-gray-500 text-sm">
                    Posted: {job.createdAt ? new Date(job.createdAt.toDate()).toLocaleDateString() : "Recently"}
                  </span>
                </div>
                <p className="mt-4 text-gray-700 font-medium">
                  {formatSalary(job.salary)} <span className="text-gray-500 font-normal">per year</span>
                </p>
              </div>

              <div className="mb-6">
                <h4 className="font-medium text-gray-800 mb-2">Required Skills</h4>
                <div className="flex flex-wrap gap-2">
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

              <div className="mb-6">
                <h4 className="font-medium text-gray-800 mb-2">Job Description</h4>
                <p className="text-gray-700 whitespace-pre-line">{job.jobDescription}</p>
              </div>

              {/* Apply button */}
              <div className="pt-4 border-t">
                <button
                  onClick={handleCompetencyTest}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-md flex items-center justify-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Take Competency Test
                </button>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center">
              <p className="text-gray-500">Job details not found.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default JobNotificationModal;