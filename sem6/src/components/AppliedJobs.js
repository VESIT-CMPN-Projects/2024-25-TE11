import React, { useState, useEffect } from "react";
import { Box, Typography, CircularProgress, Paper, Divider, Button, Chip } from "@mui/material";
import { auth, db } from "../firebase-config";
import { doc, getDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

const AppliedJobsPage = () => {
  const [appliedJobs, setAppliedJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchAppliedJobs = async () => {
      const user = auth.currentUser;
      if (!user) {
        navigate("/login");
        return;
      }
      try {
        setLoading(true);
        
        // Get user's profile to find applied jobs
        const userDoc = await getDoc(doc(db, "candidates", user.uid));
        
        if (!userDoc.exists()) {
          setAppliedJobs([]);
          return;
        }
        
        const userData = userDoc.data();
        const appliedJobsData = userData.appliedJobs || [];
        
        // Fetch full job details for each applied job
        const jobDetails = await Promise.all(
          appliedJobsData.map(async (application) => {
            try {
              const jobDoc = await getDoc(doc(db, "jobs", application.jobId));
              const jobData = jobDoc.exists() ? jobDoc.data() : null;

              // Get test details if available
              let testData = null;
              if (application.testId) {
                const testDoc = await getDoc(doc(db, "tests", application.testId));
                testData = testDoc.exists() ? testDoc.data() : null;
              }
              
              return {
                ...application,
                jobDetails: jobData,
                testDetails: testData,
                appliedDate: application.appliedAt.toDate(),
                applicationId: application.jobId + "-" + user.uid, // Unique application ID
                status: application.status || "submitted", // Default status if missing
              };
            } catch (err) {
              console.error(`Error fetching job ${application.jobId}:`, err);
              return {
                ...application,
                jobDetails: null,
                testDetails: null,
                error: true
              };
            }
          })
        );
        
        setAppliedJobs(jobDetails);
      } catch (err) {
        console.error("Error fetching applied jobs:", err);
        setError("Failed to load your applications. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchAppliedJobs();
  }, [navigate]);

  const getStatusColor = (status) => {
    switch (status) {
      case "submitted":
        return "#3f51b5"; // Indigo
      case "testing":
        return "#ff9800"; // Orange
      case "shortlisted":
        return "#2196f3"; // Blue
      case "interview":
        return "#9c27b0"; // Purple
      case "rejected":
        return "#f44336"; // Red
      case "accepted":
        return "#4caf50"; // Green
      default:
        return "#757575"; // Grey
    }
  };

  const viewJobDetails = (jobId) => {
    navigate(`/jobs/${jobId}`);
  };

  const continueTest = (testId, applicationId) => {
    navigate(`/test/${testId}?application=${applicationId}`);
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ mt: 4, p: 2 }}>
        <Typography color="error">{error}</Typography>
        <Button 
          variant="contained" 
          onClick={() => window.location.reload()} 
          sx={{ mt: 2 }}
        >
          Try Again
        </Button>
      </Box>
    );
  }

  if (appliedJobs.length === 0) {
    return (
      <Box sx={{ mt: 4, p: 2 }}>
        <Typography variant="h5" gutterBottom>Applied Jobs</Typography>
        <Paper sx={{ p: 3, textAlign: "center" }}>
          <Typography variant="body1">
            You haven't applied to any jobs yet.
          </Typography>
          <Button 
            variant="contained" 
            onClick={() => navigate("/jobseeker-dashboard")} 
            sx={{ mt: 2 }}
          >
            Browse Jobs
          </Button>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ mt: 4, p: 2 }}>
      <Typography variant="h5" gutterBottom>Applied Jobs</Typography>
      
      {appliedJobs.map((job) => (
        <Paper key={job.applicationId} sx={{ p: 3, mb: 2 }}>
          {job.error ? (
            <Typography color="error">
              There was an error loading this application. The job may have been removed.
            </Typography>
          ) : (
            <>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                <Typography variant="h6">{job.jobDetails?.jobTitle || "Unknown Job"}</Typography>
                <Chip 
                  label={job.status.charAt(0).toUpperCase() + job.status.slice(1)} 
                  sx={{ 
                    backgroundColor: getStatusColor(job.status),
                    color: "white"
                  }}
                />
              </Box>
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {job.jobDetails?.companyName || "Unknown Company"} • Applied on {job.appliedDate.toLocaleDateString()}
              </Typography>
              
              <Typography variant="body1" sx={{ mt: 2, mb: 2 }}>
                {job.jobDetails?.jobDescription || "No description available"}
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <Button 
                  variant="outlined" 
                  onClick={() => viewJobDetails(job.jobId)}
                >
                  View Job
                </Button>
                
                {job.status === "testing" && job.testDetails && (
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={() => continueTest(job.testId, job.applicationId)}
                  >
                    Continue Test
                  </Button>
                )}
              </Box>
            </>
          )}
        </Paper>
      ))}
    </Box>
  );
};

export default AppliedJobsPage;
