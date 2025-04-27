import React, { useEffect, useState } from "react";
import { auth, db } from "../firebase-config"; // Adjust path if needed
import { collection, query, where, getDocs, doc, getDoc } from "firebase/firestore";
import ResumeViewer from "./ResumeViewer"; // Adjust path if needed

const JobListings = () => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedApplicant, setSelectedApplicant] = useState(null);
  const [resumeData, setResumeData] = useState(null);
  const [loadingResume, setLoadingResume] = useState(false);

  useEffect(() => {
    const fetchJobsWithApplicants = async () => {
      if (!auth.currentUser) {
        setError("You must be logged in as an employer to view applicants.");
        setLoading(false);
        return;
      }

      try {
        const employerId = auth.currentUser.uid;
        const jobsRef = collection(db, "jobs");
        const q = query(jobsRef, where("userId", "==", employerId));
        const jobSnapshots = await getDocs(q);

        const jobsData = [];
        jobSnapshots.forEach((doc) => {
          const job = doc.data();
          jobsData.push({ id: doc.id, ...job });
        });

        setJobs(jobsData);
      } catch (error) {
        console.error("Error fetching jobs:", error);
        setError("Failed to fetch jobs. Please try again.");
      }
      setLoading(false);
    };

    fetchJobsWithApplicants();
  }, []);

  const fetchApplicantResume = async (userId) => {
    if (!userId) return;
    
    setLoadingResume(true);
    setResumeData(null);
    
    try {
      // Using the userId as the document ID to fetch the user's profile
      const userProfileRef = doc(db, "userProfiles", userId);
      const profileSnapshot = await getDoc(userProfileRef);
      
      if (profileSnapshot.exists()) {
        setResumeData(profileSnapshot.data());
        setSelectedApplicant(userId);
      } else {
        console.error("No resume found for this applicant");
        setError("No resume found for this applicant");
      }
    } catch (error) {
      console.error("Error fetching resume:", error);
      setError("Failed to load resume data");
    } finally {
      setLoadingResume(false);
    }
  };

  const closeResume = () => {
    setSelectedApplicant(null);
    setResumeData(null);
  };

  if (loading) return <p>Loading candidates...</p>;
  if (error && !selectedApplicant) return <p className="text-red-500">{error}</p>;

  return (
    <div className="p-6">
      {selectedApplicant ? (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg w-full max-w-4xl max-h-screen overflow-y-auto p-6 relative">
            <button 
              onClick={closeResume} 
              className="absolute top-4 right-4 bg-gray-200 rounded-full p-2 hover:bg-gray-300 transition"
            >
              ✕
            </button>
            
            {loadingResume ? (
              <p className="text-center py-8">Loading resume...</p>
            ) : resumeData ? (
              <ResumeViewer resumeData={resumeData} />
            ) : (
              <p className="text-red-500 text-center py-8">Failed to load resume</p>
            )}
          </div>
        </div>
      ) : null}
      
      <h2 className="text-2xl font-bold mb-4">Applied Candidates</h2>
      {jobs.length === 0 ? (
        <p>No job applications found.</p>
      ) : (
        jobs.map((job) => (
          <div key={job.id} className="border p-4 rounded-lg shadow-md mb-4">
            <h3 className="text-xl font-semibold">{job.jobTitle}</h3>
            <p className="text-gray-600">{job.companyName}</p>

            {job.applicants && job.applicants.length > 0 ? (
              <ul className="mt-2">
                {job.applicants.map((applicant, index) => (
                  <li key={index} className="p-2 border-b">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{applicant.name}</p>
                        <p className="text-gray-500">{applicant.email}</p>
                        {applicant.testScore !== undefined && (
                          <p className="text-blue-500">Test Score: {applicant.testScore}%</p>
                        )}
                      </div>
                      <button
                        onClick={() => fetchApplicantResume(applicant.userId)}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition"
                      >
                        View Resume
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500 mt-2">No applicants for this job yet.</p>
            )}
          </div>
        ))
      )}
    </div>
  );
};

export default JobListings;