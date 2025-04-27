import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Box, Paper, Typography, Button, Divider, CircularProgress, List, ListItem, ListItemText, ListItemIcon } from "@mui/material";
import { doc, getDoc, updateDoc, arrayUnion, Timestamp, setDoc } from "firebase/firestore";
import { db, auth } from "../firebase-config";

const ResultsPage = () => {
  const [result, setResult] = useState(null);
  const [job, setJob] = useState(null);
  const [employer, setEmployer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const { testId, jobId } = useParams();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        const resultDoc = await getDoc(doc(db, "tests", testId));
        if (resultDoc.exists()) {
          setResult(resultDoc.data());
        } else {
          setError("Test result not found");
          return;
        }
        
        const jobDoc = await getDoc(doc(db, "jobs", jobId));
        if (jobDoc.exists()) {
          const jobData = jobDoc.data();
          setJob(jobData);

          if (jobData.userId) {
            const employerDoc = await getDoc(doc(db, "employers", jobData.userId));
            if (employerDoc.exists()) {
              const employerData = employerDoc.data();
              setEmployer(employerData);

              setJob(prev => ({
                ...prev,
                companyName: employerData.companyName || "Unknown Company"
              }));
            }
          }
          
          if (auth.currentUser) {
            const candidateDoc = await getDoc(doc(db, "candidates", auth.currentUser.uid));
            if (candidateDoc.exists()) {
              const appliedJobs = candidateDoc.data().appliedJobs || [];
              if (appliedJobs.some(job => job.jobId === jobId)) {
                setApplied(true);
              }
            }
          }
        } else {
          setError("Job not found");
          return;
        }
      } catch (err) {
        console.error("Error fetching data:", err);
        setError("Failed to load test results");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [testId, jobId]);

  const handleApplyForJob = async () => {
    if (!auth.currentUser || !job) {
      setError("You must be logged in to apply");
      return;
    }
  
    try {
      setApplying(true);
  
      const candidateId = auth.currentUser.uid;
      const employerId = job.userId;
      const companyName = employer?.companyName || job.companyName || "Unknown Company";
  
      // Fetch user's profile from 'userProfiles' collection
      const userProfileRef = doc(db, "userProfiles", candidateId);
      const userProfileSnap = await getDoc(userProfileRef);
  
      let candidateName = auth.currentUser.displayName || auth.currentUser.email; // Fallback name
      if (userProfileSnap.exists()) {
        const userData = userProfileSnap.data();
        candidateName = userData.fullName || candidateName; // Use fullName if available
      }
  
      const applicationData = {
        jobId,
        jobTitle: job.jobTitle,
        companyName,
        employerId,
        appliedAt: Timestamp.now(),
        testId,
        testScore: result.percentage,
        name: candidateName, // Store fetched full name
        email: auth.currentUser.email,
      };
  
      const candidateDocRef = doc(db, "candidates", candidateId);
      const candidateDoc = await getDoc(candidateDocRef);
  
      if (!candidateDoc.exists()) {
        await setDoc(candidateDocRef, {
          name: candidateName,
          email: auth.currentUser.email,
          appliedJobs: [applicationData],
        });
      } else {
        await updateDoc(candidateDocRef, {
          appliedJobs: arrayUnion(applicationData),
        });
      }
  
      const jobDocRef = doc(db, "jobs", jobId);
      await updateDoc(jobDocRef, {
        applicants: arrayUnion({
          userId: candidateId,
          name: candidateName,
          email: auth.currentUser.email,
          appliedAt: Timestamp.now(),
          testId,
          testScore: result.percentage,
        }),
      });
  
      setApplied(true);
      setApplying(false);
  
      // Redirect to dashboard after 3 seconds
      setTimeout(() => {
        navigate("/jobseeker-dashboard");
      }, 3000);
    } catch (error) {
      console.error("Error applying for job:", error);
      setError("Failed to apply for job. Please try again.");
      setApplying(false);
    }
  };   

  const recommendedCourses = [
    { title: "Machine Learning Fundamentals", provider: "Coursera", url: "https://coursera.org/ml-fundamentals" },
    { title: "Python for Data Science", provider: "edX", url: "https://edx.org/python-data-science" },
    { title: "Deep Learning Specialization", provider: "Udacity", url: "https://udacity.com/deep-learning" },
    { title: "AI Algorithms and Implementation", provider: "MIT OpenCourseWare", url: "https://ocw.mit.edu/ai-algorithms" }
  ];

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <Paper sx={{ p: 4, maxWidth: 600, textAlign: "center" }}>
          <Typography variant="h5" color="error" gutterBottom>{error}</Typography>
          <Button variant="contained" onClick={() => navigate("/jobseeker-dashboard")}>
            Back to Dashboard
          </Button>
        </Paper>
      </Box>
    );
  }

  const isPassing = result && parseFloat(result.percentage) >= 60;

  return (
    <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh", backgroundColor: "#f4f6f9", p: 2 }}>
      <Paper sx={{ padding: 4, maxWidth: 700, width: "100%", borderRadius: 3, boxShadow: 3 }}>
        {applied ? (
          <Box sx={{ textAlign: "center", py: 4 }}>
            <Typography variant="h4" gutterBottom color="success.main">
              Application Successful!
            </Typography>
            <Typography variant="body1" sx={{ mb: 3 }}>
              Your application for {job?.jobTitle} at {job?.companyName} has been submitted.
            </Typography>
            <Button variant="contained" onClick={() => navigate("/jobseeker-dashboard")}>
            Back to Dashboard
            </Button>
          </Box>
        ) : (
          <>
            <Typography variant="h4" fontWeight="bold" gutterBottom color="primary">
              Test Results
            </Typography>
            <Box sx={{ textAlign: "center", my: 3 }}>
              <Box
                sx={{
                  position: "relative",
                  display: "inline-flex",
                  mb: 2
                }}
              >
                <CircularProgress
                  variant="determinate"
                  value={parseFloat(result?.percentage)}
                  size={120}
                  thickness={5}
                  sx={{ color: isPassing ? "success.main" : "error.main" }}
                />
                <Box
                  sx={{
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    position: "absolute",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Typography variant="h4" component="div" color="text.secondary">
                    {result?.percentage}%
                  </Typography>
                </Box>
              </Box>
              
              <Typography variant="h5" fontWeight="bold" color={isPassing ? "success.main" : "error.main"}>
                {isPassing ? "Congratulations!" : "Not Qualified"}
              </Typography>
              
              <Typography variant="body1" sx={{ mt: 1 }}>
                {result?.score} correct out of {result?.totalQuestions} questions
              </Typography>
            </Box>
            <Divider sx={{ my: 2 }} />
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Job Details
              </Typography>
              <Typography variant="body1">
                <strong>Position:</strong> {job?.jobTitle}
              </Typography>
              <Typography variant="body1">
                <strong>Company:</strong> {job?.companyName || employer?.companyName || "Unknown Company"}
              </Typography>
              <Typography variant="body1">
                <strong>Required Score:</strong> 60%
              </Typography>
            </Box>
            {isPassing ? (
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="body1" sx={{ mb: 3 }}>
                  You've passed the competency test and are eligible to apply for this position.
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={applying || applied}
                  onClick={handleApplyForJob}
                  sx={{ py: 1.5, px: 4 }}
                >
                  {applying ? "Applying..." : applied ? "Already Applied" : "Apply for this Job"}
                </Button>
              </Box>
            ) : (
              <Box>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  You didn't meet the minimum score requirement for this position. We recommend enhancing your skills with these courses:
                </Typography>
                
                <List>
                  {recommendedCourses.map((course, index) => (
                    <ListItem key={index} disablePadding sx={{ mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 36 }}>
                      </ListItemIcon>
                      <ListItemText 
                        primary={course.title} 
                        secondary={`Provider: ${course.provider}`}
                        primaryTypographyProps={{ fontWeight: "medium" }}
                      />
                    </ListItem>
                  ))}
                </List>
                
                <Typography variant="body1" sx={{ mt: 2 }}>
                  You can retake the test after improving your skills to become eligible for this position.
                </Typography>
                
                <Box sx={{ display: "flex", justifyContent: "space-between", mt: 3 }}>
                  <Button 
                    variant="outlined" 
                    onClick={() => navigate("/jobseeker-dashboard")}
                  >
                    Back to Dashboard
                  </Button>
                </Box>
              </Box>
            )}
          </>
        )}
      </Paper>
    </Box>
  );
};

export default ResultsPage;