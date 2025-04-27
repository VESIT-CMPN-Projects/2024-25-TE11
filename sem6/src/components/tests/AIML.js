import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Box,
  Paper,
  Typography,
  Button,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  LinearProgress,
  Divider,
  Snackbar,
  Alert,
  CircularProgress,
} from "@mui/material";
import { auth, db } from "../../firebase-config";
import { doc, setDoc, Timestamp, getDoc } from "firebase/firestore";
import TestProctoring from "../TestProctoring"; // Import the new component

// Questions array remains the same
const questions = [
  { question: "What is the primary function of a machine learning model?", options: ["Storing data", "Making predictions", "Rendering graphics", "Managing databases"], answer: "Making predictions" },
  { question: "Which algorithm is commonly used for classification problems?", options: ["Linear Regression", "K-Means Clustering", "Decision Trees", "Apriori"], answer: "Decision Trees" },
  { question: "Which programming language is most commonly used for AI/ML?", options: ["Java", "Python", "C++", "JavaScript"], answer: "Python" },
  { question: "What does 'overfitting' mean in machine learning?", options: ["Model performs poorly on training data", "Model performs well on training but poorly on new data", "Model is too simple", "Model ignores outliers"], answer: "Model performs well on training but poorly on new data" },
  { question: "Which of the following is a supervised learning algorithm?", options: ["K-Means", "Random Forest", "PCA", "Apriori"], answer: "Random Forest" },
];

const AIMLtest = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  
  const [user, setUser] = useState(null);
  const [job, setJob] = useState(null);
  const [companyName, setCompanyName] = useState(null);
  const [jobLoading, setJobLoading] = useState(true);
  const [userAnswers, setUserAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [showError, setShowError] = useState(false);
  
  // New state for verification
  const [verificationComplete, setVerificationComplete] = useState(false);
  const [showProctoring, setShowProctoring] = useState(true);

  useEffect(() => {
    const fetchJobAndCompany = async () => {
      try {
        // Fetch job details
        const jobDoc = await getDoc(doc(db, "jobs", jobId));
        
        if (jobDoc.exists()) {
          const jobData = jobDoc.data();
          setJob(jobData);
          
          // If job has employerId, fetch the company name
          if (jobData.employerId) {
            try {
              const employerDoc = await getDoc(doc(db, "employers", jobData.employerId));
              if (employerDoc.exists()) {
                setCompanyName(employerDoc.data().companyName);
              }
            } catch (err) {
              console.error("Error fetching employer:", err);
            }
          }
        } else {
          setError("Job not found");
          setShowError(true);
        }
      } catch (err) {
        console.error("Error fetching job:", err);
        setError("Error loading job details");
        setShowError(true);
      } finally {
        setJobLoading(false);
      }
    };

    fetchJobAndCompany();
  }, [jobId]);

  useEffect(() => {
    const currentUser = auth.currentUser;
    if (currentUser) {
      setUser({
        uid: currentUser.uid,
        email: currentUser.email,
        name: currentUser.displayName || currentUser.email || "User",
      });
    } else {
      // Redirect if not logged in
      navigate("/login", { state: { redirect: `/competency-test/${jobId}` } });
    }
  }, [jobId, navigate]);

  const handleAnswerChange = (index, value) => {
    setUserAnswers({ ...userAnswers, [index]: value });
  };

  const handleSubmit = async () => {
    if (!user) return;
    
    try {
      setSubmitting(true);
      
      // Calculate score
      let score = 0;
      questions.forEach((q, index) => {
        if (userAnswers[index] === q.answer) {
          score++;
        }
      });
      
      // Create test result document
      const testResultId = `test_${Date.now()}`;
      const testData = {
        testType: job ? `${job.jobTitle || 'Competency'} Test` : "AI-ML Engineer Test",
        jobId: jobId || null,
        jobTitle: job?.jobTitle || null,
        companyName: companyName || null, // Use the separately fetched company name
        userId: user.uid,
        userEmail: user.email,
        userName: user.name,
        totalQuestions: questions.length,
        score: score,
        percentage: ((score / questions.length) * 100).toFixed(2),
        pass: score >= (questions.length * 0.6),
        answers: userAnswers,
        timestamp: Timestamp.now(),
        proctored: true, // Add this field to indicate the test was proctored
      };

      // Save to Firestore
      await setDoc(doc(db, "tests", testResultId), testData);
      
      // Navigate to results page
      navigate(`/test-results/${testResultId}/${jobId}`);
      
    } catch (error) {
      console.error("Error submitting test:", error);
      setError("There was an error submitting your test. Please try again.");
      setShowError(true);
      setSubmitting(false);
    }
  };

  // Handler for verification completion
  const handleVerificationComplete = (success) => {
    if (success) {
      setVerificationComplete(true);
      setShowProctoring(false);
    }
  };

  if (jobLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <CircularProgress />
      </Box>
    );
  }

  // Show proctoring component if verification is not complete
  if (showProctoring && !verificationComplete) {
    return <TestProctoring onVerificationComplete={handleVerificationComplete} jobId={jobId} />;
  }

  return (
    <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh", backgroundColor: "#f4f6f9", p: 2 }}>
      <Paper sx={{ padding: 4, maxWidth: 700, width: "100%", borderRadius: 3, boxShadow: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom color="primary">
          {job ? `${job.jobTitle || 'Competency'} Test` : "AI-ML Engineer Test"}
        </Typography>
        
        {job && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6">{companyName || 'Company'}</Typography>
            <Typography variant="body2" color="text.secondary">
              Complete this test to determine if you qualify for this position.
              You need to score at least 60% to be eligible to apply.
            </Typography>
          </Box>
        )}
        
        {user && <Typography variant="subtitle1">Candidate: {user.name}</Typography>}
        
        <Alert severity="info" sx={{ my: 2 }}>
          This test is being proctored. Please keep your camera on and remain in view.
        </Alert>
        
        <LinearProgress 
          variant="determinate" 
          value={(Object.keys(userAnswers).length / questions.length) * 100} 
          sx={{ mt: 2 }} 
        />
        
        {questions.map((q, index) => (
          <Box key={index} sx={{ mt: 3, p: 2, borderRadius: 2, boxShadow: 1, backgroundColor: "#fafafa" }}>
            <Typography variant="h6">{index + 1}. {q.question}</Typography>
            <FormControl component="fieldset">
              <RadioGroup 
                value={userAnswers[index] || ""} 
                onChange={(e) => handleAnswerChange(index, e.target.value)}
              >
                {q.options.map((option, i) => (
                  <FormControlLabel key={i} value={option} control={<Radio />} label={option} />
                ))}
              </RadioGroup>
            </FormControl>
          </Box>
        ))}
        
        <Divider sx={{ my: 3 }} />
        
        <Button 
          variant="contained" 
          color="primary" 
          fullWidth 
          sx={{ py: 1.5 }} 
          onClick={handleSubmit} 
          disabled={submitting || Object.keys(userAnswers).length !== questions.length}
        >
          {submitting ? "Submitting..." : "Submit Test"}
        </Button>
        
        {Object.keys(userAnswers).length !== questions.length && (
          <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ mt: 1 }}>
            Please answer all questions before submitting
          </Typography>
        )}
        
        <Snackbar open={showError} autoHideDuration={6000} onClose={() => setShowError(false)}>
          <Alert onClose={() => setShowError(false)} severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
      </Paper>
    </Box>
  );
};

export default AIMLtest;