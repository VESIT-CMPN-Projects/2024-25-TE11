// import React, { useState, useEffect, useRef } from "react";
// import { useParams, useNavigate } from "react-router-dom";
// import { auth, db } from "../firebase-config";
// import { doc, getDoc, updateDoc } from "firebase/firestore";
// import {
//   Box,
//   Paper,
//   Typography,
//   Button,
//   CircularProgress,
//   Dialog,
//   Grid,
//   Alert,
//   Stepper,
//   Step,
//   StepLabel,
// } from "@mui/material";

// // API configuration
// const API_URL = "http://127.0.0.1:8000";
// const VERIFICATION_ENDPOINT = "/compare_faces/";

// const TestProctoring = ({ onVerificationComplete, jobId }) => {
//   const navigate = useNavigate();
//   const { jobId: urlJobId } = useParams();
//   const actualJobId = jobId || urlJobId;
  
//   const [user, setUser] = useState(null);
//   const [profilePicture, setProfilePicture] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [step, setStep] = useState(0);
//   const [error, setError] = useState(null);
//   const [verificationStatus, setVerificationStatus] = useState("pending"); // pending, processing, success, failed
//   const [similarity, setSimilarity] = useState(null);
//   const [apiAvailable, setApiAvailable] = useState(null); // null=unknown, true=available, false=unavailable
//   const [debugInfo, setDebugInfo] = useState([]); // Store debug information
//   const [showDebug, setShowDebug] = useState(false);
  
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const mediaStreamRef = useRef(null);
  
//   const steps = ["Identity Check", "Environment Check", "Rules Acceptance"];

//   // Add debug log function
//   const addDebugLog = (message, type = "info") => {
//     console.log(`[${type.toUpperCase()}] ${message}`);
//     setDebugInfo(prev => [...prev, { 
//       timestamp: new Date().toISOString(),
//       type, 
//       message 
//     }]);
//   };

//   // Check if API is available
//   useEffect(() => {
//     const checkApiAvailability = async () => {
//       try {
//         addDebugLog("Checking API availability...");
//         const response = await fetch(`${API_URL}/`, { 
//           method: 'GET',
//           // Set timeout to avoid long waiting
//           signal: AbortSignal.timeout(5000)
//         });
        
//         if (response.ok) {
//           addDebugLog("API is available", "success");
//           setApiAvailable(true);
//         } else {
//           addDebugLog(`API returned status ${response.status}`, "error");
//           setApiAvailable(false);
//           setError("Verification service is not responding correctly. Please try again later.");
//         }
//       } catch (err) {
//         addDebugLog(`API connectivity error: ${err.message}`, "error");
//         setApiAvailable(false);
//         setError("Cannot connect to verification service. Please ensure the service is running.");
//       }
//     };
    
//     checkApiAvailability();
//   }, []);

//   // Load user data and profile picture
//   useEffect(() => {
//     const fetchUserProfile = async () => {
//       try {
//         addDebugLog("Fetching user profile...");
//         const currentUser = auth.currentUser;
//         if (!currentUser) {
//           addDebugLog("No authenticated user found", "warning");
//           navigate("/login", { state: { redirect: `/competency-test/${actualJobId}` } });
//           return;
//         }
        
//         setUser({
//           uid: currentUser.uid,
//           email: currentUser.email,
//           name: currentUser.displayName || currentUser.email,
//         });
        
//         // Fetch user profile picture from Firestore
//         addDebugLog(`Fetching profile picture for user: ${currentUser.uid}`);
//         const userProfileDoc = await getDoc(doc(db, "userProfiles", currentUser.uid));
//         if (userProfileDoc.exists() && userProfileDoc.data().profilePicture) {
//           addDebugLog("Profile picture found", "success");
//           setProfilePicture(userProfileDoc.data().profilePicture);
//         } else {
//           addDebugLog("No profile picture found", "error");
//           setError("Profile picture not found. Please upload a profile picture before taking the test.");
//         }
//       } catch (err) {
//         addDebugLog(`Error fetching user profile: ${err.message}`, "error");
//         setError(`Failed to load user profile: ${err.message}. Please try again.`);
//       } finally {
//         setLoading(false);
//       }
//     };
    
//     fetchUserProfile();
//   }, [actualJobId, navigate]);
  
//   // Start camera when ready for verification
//   const startCamera = async () => {
//     try {
//       addDebugLog("Attempting to start camera...");
//       if (videoRef.current) {
//         const constraints = { 
//           video: { 
//             width: 640, 
//             height: 480,
//             facingMode: "user"
//           } 
//         };
        
//         addDebugLog("Requesting camera with constraints: " + JSON.stringify(constraints));
//         const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
//         videoRef.current.srcObject = stream;
//         mediaStreamRef.current = stream;
        
//         // Ensure video is actually playing
//         videoRef.current.onloadedmetadata = () => {
//           addDebugLog("Camera started successfully", "success");
//           videoRef.current.play().catch(playErr => {
//             addDebugLog(`Error playing video: ${playErr.message}`, "error");
//             setError(`Camera started but couldn't play: ${playErr.message}`);
//           });
//         };
//       } else {
//         addDebugLog("Video reference not found", "error");
//         setError("Camera initialization failed. Please refresh and try again.");
//       }
//     } catch (err) {
//       addDebugLog(`Error accessing camera: ${err.message}`, "error");
      
//       let errorMsg = "Camera access is required for identity verification.";
      
//       // More specific error messages based on the error
//       if (err.name === "NotAllowedError") {
//         errorMsg += " You denied permission to use the camera. Please allow camera access and try again.";
//       } else if (err.name === "NotFoundError") {
//         errorMsg += " No camera was found on your device.";
//       } else if (err.name === "NotReadableError") {
//         errorMsg += " Your camera might be in use by another application.";
//       } else {
//         errorMsg += ` Error: ${err.message}`;
//       }
      
//       setError(errorMsg);
//     }
//   };
  
//   // Check if video is actually receiving data
//   const checkVideoStream = () => {
//     if (!videoRef.current || !mediaStreamRef.current) {
//       addDebugLog("No video stream to check", "warning");
//       return false;
//     }
    
//     const tracks = mediaStreamRef.current.getVideoTracks();
//     if (!tracks || tracks.length === 0) {
//       addDebugLog("No video tracks found in media stream", "error");
//       return false;
//     }
    
//     const isActive = tracks[0].enabled && tracks[0].readyState === 'live';
//     addDebugLog(`Video stream active: ${isActive}`);
    
//     return isActive;
//   };
  
//   // Stop camera
//   const stopCamera = () => {
//     addDebugLog("Stopping camera...");
//     if (mediaStreamRef.current) {
//       mediaStreamRef.current.getTracks().forEach(track => {
//         track.stop();
//         addDebugLog(`Stopped track: ${track.kind}`);
//       });
//     } else {
//       addDebugLog("No media stream to stop", "warning");
//     }
//   };
  
//   // Clean up on unmount
//   useEffect(() => {
//     return () => {
//       stopCamera();
//     };
//   }, []);

//   // Capture current frame from webcam as a blob
//   const captureImage = () => {
//     return new Promise((resolve, reject) => {
//       if (!videoRef.current) {
//         const error = new Error("Video stream not available");
//         addDebugLog(`Capture failed: ${error.message}`, "error");
//         reject(error);
//         return;
//       }
      
//       // Check if video is actually streaming
//       if (!checkVideoStream()) {
//         const error = new Error("Video stream is not active");
//         addDebugLog(`Capture failed: ${error.message}`, "error");
//         reject(error);
//         return;
//       }
      
//       // Make sure video has loaded metadata
//       if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
//         const error = new Error("Video dimensions not available yet");
//         addDebugLog(`Capture failed: ${error.message}`, "error");
//         reject(error);
//         return;
//       }
      
//       const canvas = document.createElement('canvas');
//       canvas.width = videoRef.current.videoWidth;
//       canvas.height = videoRef.current.videoHeight;
      
//       addDebugLog(`Capturing frame at ${canvas.width}x${canvas.height}`);
      
//       const ctx = canvas.getContext('2d');
//       ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
//       canvas.toBlob(blob => {
//         if (blob) {
//           addDebugLog(`Captured image blob size: ${blob.size} bytes`, "success");
//           resolve(blob);
//         } else {
//           const error = new Error("Failed to capture image");
//           addDebugLog(`Capture failed: ${error.message}`, "error");
//           reject(error);
//         }
//       }, 'image/jpeg', 0.95);
//     });
//   };
  
//   // Convert profile picture URL to a blob
//   const getProfileImageBlob = async (url) => {
//     try {
//       addDebugLog(`Fetching profile image from URL: ${url.substring(0, 50)}...`);
//       const response = await fetch(url);
      
//       if (!response.ok) {
//         throw new Error(`Failed to fetch profile image: ${response.status} ${response.statusText}`);
//       }
      
//       const blob = await response.blob();
//       addDebugLog(`Profile image blob size: ${blob.size} bytes`, "success");
//       return blob;
//     } catch (err) {
//       addDebugLog(`Error fetching profile image: ${err.message}`, "error");
//       throw new Error(`Failed to fetch profile image: ${err.message}`);
//     }
//   };
  
//   // Perform face verification using the API
//   const verifyIdentity = async () => {
//     if (!videoRef.current || !profilePicture) {
//       setError("Verification system not ready. Please try again.");
//       return;
//     }

//     if (!apiAvailable) {
//       setError("Verification service is not available. Please try again later.");
//       return;
//     }

//     addDebugLog("Starting verification process...");
//     setVerificationStatus("processing");
    
//     try {
//       // Capture current frame from webcam
//       addDebugLog("Capturing webcam image...");
//       const webcamImageBlob = await captureImage();
      
//       // Get profile image as blob
//       addDebugLog("Retrieving profile image...");
//       const profileImageBlob = await getProfileImageBlob(profilePicture);
      
//       // Create form data for API request
//       const formData = new FormData();
//       formData.append("image1", profileImageBlob, "profile.jpg");
//       formData.append("image2", webcamImageBlob, "webcam.jpg");
      
//       // Send request to backend API
//       addDebugLog(`Sending comparison request to ${API_URL}${VERIFICATION_ENDPOINT}`);
      
//       const controller = new AbortController();
//       const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
//       try {
//         const response = await fetch(`${API_URL}${VERIFICATION_ENDPOINT}`, {
//           method: "POST",
//           body: formData,
//           signal: controller.signal
//         });
        
//         clearTimeout(timeoutId);
        
//         // Check for HTTP errors
//         if (!response.ok) {
//           const errorText = await response.text();
//           throw new Error(`API request failed with status ${response.status}: ${errorText}`);
//         }
        
//         // Parse response
//         const data = await response.json();
//         addDebugLog(`Face comparison result: ${JSON.stringify(data)}`);
        
//         // Set similarity score
//         setSimilarity(data.similarity_score);
        
//         // Check for API-reported errors
//         if ("error" in data) {
//           throw new Error(data.error || "API reported an error");
//         }
        
//         // When comparison completes
//         if (data.match) {
//           addDebugLog("Verification SUCCESS - faces match!", "success");
//           setVerificationStatus("success");
//           // Move to next step after a short delay
//           setTimeout(() => {
//             setStep(1);
//             // Reset verification status for next step
//             setVerificationStatus("pending");
//           }, 1500);
//         } else {
//           addDebugLog(`Verification FAILED - faces don't match. Similarity: ${data.similarity_score}`, "warning");
//           setError(`Identity verification failed. Similarity score: ${data.similarity_score.toFixed(2)}`);
//           setVerificationStatus("failed");
//         }
//       } catch (fetchErr) {
//         clearTimeout(timeoutId);
        
//         if (fetchErr.name === 'AbortError') {
//           addDebugLog("API request timed out", "error");
//           throw new Error("Verification request timed out. Please try again.");
//         } else {
//           throw fetchErr;
//         }
//       }
//     } catch (err) {
//       addDebugLog(`Verification error: ${err.message}`, "error");
//       setError(`Verification failed: ${err.message}. Please try again in better lighting conditions.`);
//       setVerificationStatus("failed");
//     }
//   };

//   // Verify environment
//   const verifyEnvironment = async () => {
//     addDebugLog("Starting environment verification...");
//     setVerificationStatus("processing");
    
//     try {
//       // First check if video is still active
//       if (!checkVideoStream()) {
//         addDebugLog("Video stream not active for environment check", "error");
//         throw new Error("Camera feed is not active. Please restart the camera.");
//       }
      
//       // Simulate environment check (you could implement more sophisticated checks)
//       addDebugLog("Performing environment checks...");
//       await new Promise(resolve => setTimeout(resolve, 2000));
      
//       // Simulated check passed
//       addDebugLog("Environment verification completed successfully", "success");
//       setVerificationStatus("success");
      
//       // Move to next step after successful verification
//       setTimeout(() => {
//         setStep(2);
//         // Reset verification status for next step
//         setVerificationStatus("pending");
//       }, 1500);
//     } catch (err) {
//       addDebugLog(`Environment check failed: ${err.message}`, "error");
//       setError(`Environment check failed: ${err.message}. Please ensure you're in a quiet, well-lit room.`);
//       setVerificationStatus("failed");
//     }
//   };
  
//   // Accept rules and start test
//   const acceptRulesAndStart = () => {
//     // Stop camera before proceeding to test
//     addDebugLog("User accepted rules, stopping camera and proceeding to test", "success");
//     stopCamera();
    
//     // Call the completion callback to start the test
//     if (onVerificationComplete) {
//       onVerificationComplete(true);
//     } else {
//       // If used as standalone component, navigate to the test
//       navigate(`/competency-test/${actualJobId}`);
//     }
//   };
  
//   // Reset verification if it fails
//   const resetVerification = () => {
//     addDebugLog("Resetting verification state", "info");
//     setVerificationStatus("pending");
//     setError(null);
    
//     // Check if we need to restart camera
//     if (!checkVideoStream()) {
//       addDebugLog("Restarting camera during reset", "info");
//       startCamera();
//     }
//   };
  
//   // Debug panel component
//   const DebugPanel = () => (
//     <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1, maxHeight: 200, overflow: 'auto' }}>
//       <Typography variant="subtitle2">Debug Log:</Typography>
//       {debugInfo.map((log, index) => (
//         <Typography key={index} variant="caption" sx={{ 
//           display: 'block', 
//           color: log.type === 'error' ? 'error.main' : 
//                  log.type === 'warning' ? 'warning.main' : 
//                  log.type === 'success' ? 'success.main' : 'text.secondary'
//         }}>
//           [{log.timestamp.split('T')[1].split('.')[0]}] {log.message}
//         </Typography>
//       ))}
//     </Box>
//   );
  
//   // Render different steps
//   const renderStep = () => {
//     switch (step) {
//       case 0: // Identity verification
//         return (
//           <Box sx={{ mt: 2, textAlign: "center" }}>
//             <Typography variant="h6" gutterBottom>Identity Verification</Typography>
//             <Typography variant="body2" sx={{ mb: 2 }}>
//               Please look directly at the camera and position your face within the frame.
//             </Typography>
            
//             {!apiAvailable && apiAvailable !== null && (
//               <Alert severity="error" sx={{ mb: 3 }}>
//                 {error || "Face verification service is not available. Please make sure the service is running."}
//               </Alert>
//             )}
            
//             <Grid container spacing={2} sx={{ mb: 3 }}>
//               <Grid item xs={6}>
//                 <Box sx={{ border: "1px solid #ddd", borderRadius: 2, overflow: "hidden", height: 240 }}>
//                   <Typography variant="subtitle2" sx={{ p: 1, bgcolor: "#f5f5f5" }}>Profile Picture</Typography>
//                   {profilePicture ? (
//                     <img 
//                       src={profilePicture} 
//                       alt="Profile" 
//                       style={{ width: "100%", height: 200, objectFit: "cover" }} 
//                       onError={() => {
//                         addDebugLog("Failed to load profile picture", "error");
//                         setError("Could not load your profile picture. Please check your profile.");
//                       }}
//                     />
//                   ) : (
//                     <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
//                       <Typography variant="body2" color="text.secondary">No profile picture found</Typography>
//                     </Box>
//                   )}
//                 </Box>
//               </Grid>
//               <Grid item xs={6}>
//                 <Box sx={{ border: "1px solid #ddd", borderRadius: 2, overflow: "hidden", height: 240 }}>
//                   <Typography variant="subtitle2" sx={{ p: 1, bgcolor: "#f5f5f5" }}>Camera Feed</Typography>
//                   <video 
//                     ref={videoRef} 
//                     autoPlay 
//                     style={{ width: "100%", height: 200, objectFit: "cover" }} 
//                     onError={() => {
//                       addDebugLog("Video element error", "error");
//                       setError("Camera feed error. Please refresh and try again.");
//                     }}
//                   />
//                 </Box>
//               </Grid>
//             </Grid>
            
//             {!videoRef.current?.srcObject && (
//               <Button
//                 variant="contained"
//                 color="primary"
//                 onClick={startCamera}
//                 sx={{ mb: 2 }}
//                 disabled={!apiAvailable && apiAvailable !== null}
//               >
//                 Start Camera
//               </Button>
//             )}
            
//             {videoRef.current?.srcObject && verificationStatus === "pending" && (
//               <Button
//                 variant="contained"
//                 color="primary"
//                 onClick={verifyIdentity}
//                 sx={{ mb: 2 }}
//                 disabled={!apiAvailable && apiAvailable !== null}
//               >
//                 Verify Identity
//               </Button>
//             )}
            
//             {verificationStatus === "processing" && (
//               <Box sx={{ textAlign: "center", my: 2 }}>
//                 <CircularProgress size={24} sx={{ mr: 1 }} />
//                 <Typography>Verifying your identity...</Typography>
//                 <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
//                   Comparing facial features...
//                 </Typography>
//               </Box>
//             )}
            
//             {verificationStatus === "success" && (
//               <Alert severity="success" sx={{ mb: 2 }}>
//                 <Typography variant="body2">Identity verified successfully!</Typography>
//                 <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
//                   Face match confirmed (Similarity: {similarity?.toFixed(2)}). Proceeding to next step...
//                 </Typography>
//               </Alert>
//             )}
            
//             {verificationStatus === "failed" && (
//               <Box>
//                 <Alert severity="error" sx={{ mb: 2 }}>
//                   <Typography variant="body2">{error || "Verification failed. Please try again."}</Typography>
//                   <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
//                     {similarity !== null && `Similarity score: ${similarity.toFixed(2)}`}
//                   </Typography>
//                   <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
//                     Make sure your face is clearly visible and well-lit.
//                   </Typography>
//                 </Alert>
//                 <Button variant="outlined" onClick={resetVerification}>
//                   Try Again
//                 </Button>
//               </Box>
//             )}
            
//             <canvas ref={canvasRef} style={{ display: "none" }} />
//           </Box>
//         );
        
//       case 1: // Environment check
//         return (
//           <Box sx={{ mt: 2, textAlign: "center" }}>
//             <Typography variant="h6" gutterBottom>Environment Check</Typography>
//             <Typography variant="body2" sx={{ mb: 3 }}>
//               Please ensure you are in a quiet, well-lit room with no one else present.
//               No additional devices should be accessible during the test.
//             </Typography>
            
//             <Box sx={{ border: "1px solid #ddd", borderRadius: 2, overflow: "hidden", mb: 3, mx: "auto", maxWidth: 480 }}>
//               <video 
//                 ref={videoRef} 
//                 autoPlay 
//                 style={{ width: "100%", height: 320, objectFit: "cover" }} 
//               />
//             </Box>
            
//             {verificationStatus === "pending" && (
//               <Button
//                 variant="contained"
//                 color="primary"
//                 onClick={verifyEnvironment}
//               >
//                 Verify Environment
//               </Button>
//             )}
            
//             {verificationStatus === "processing" && (
//               <Box sx={{ textAlign: "center", my: 2 }}>
//                 <CircularProgress size={24} sx={{ mr: 1 }} />
//                 <Typography>Verifying your environment...</Typography>
//               </Box>
//             )}
            
//             {verificationStatus === "success" && (
//               <Alert severity="success" sx={{ mb: 2 }}>
//                 <Typography variant="body2">Environment verified successfully!</Typography>
//                 <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
//                   Proceeding to next step...
//                 </Typography>
//               </Alert>
//             )}
            
//             {verificationStatus === "failed" && (
//               <Box>
//                 <Alert severity="error" sx={{ mb: 2 }}>
//                   <Typography variant="body2">{error || "Environment check failed. Please try again."}</Typography>
//                 </Alert>
//                 <Button variant="outlined" onClick={resetVerification}>
//                   Try Again
//                 </Button>
//               </Box>
//             )}
//           </Box>
//         );
        
//       case 2: // Rules acceptance
//         return (
//           <Box sx={{ mt: 2 }}>
//             <Typography variant="h6" gutterBottom>Test Rules</Typography>
//             <Alert severity="info" sx={{ mb: 3 }}>
//               By proceeding, you agree to the following rules:
//             </Alert>
            
//             <Box sx={{ bgcolor: "#f8f9fa", p: 2, borderRadius: 2, mb: 3 }}>
//               <Typography component="div" variant="body2">
//                 <ul>
//                   <li>You must remain visible on camera throughout the test</li>
//                   <li>No additional screens or devices are allowed</li>
//                   <li>No other individuals may be present in the room</li>
//                   <li>You may not browse other websites or use search engines</li>
//                   <li>You may not use notes, books, or other resources</li>
//                   <li>Leaving the test window will be flagged and may result in disqualification</li>
//                 </ul>
//               </Typography>
//             </Box>
            
//             <Button
//               variant="contained"
//               color="primary"
//               fullWidth
//               onClick={acceptRulesAndStart}
//             >
//               I Accept & Start Test
//             </Button>
//           </Box>
//         );
        
//       default:
//         return null;
//     }
//   };
  
//   if (loading) {
//     return (
//       <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
//         <CircularProgress />
//       </Box>
//     );
//   }
  
//   if (error && !profilePicture) {
//     return (
//       <Box sx={{ p: 3 }}>
//         <Alert severity="error" sx={{ mb: 2 }}>
//           {error}
//         </Alert>
//         <Button variant="contained" onClick={() => navigate("/jobseeker-dashboard")}>
//           Go to Profile
//         </Button>
//       </Box>
//     );
//   };

//   return (
//     <Paper sx={{ p: 3, maxWidth: 800, mx: "auto", borderRadius: 2, boxShadow: 2 }}>
//       <Typography variant="h5" gutterBottom>
//         Test Verification
//       </Typography>
      
//       <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
//         Before you can take the test, we need to verify your identity and ensure test security.
//       </Typography>
      
//       <Button 
//         size="small" 
//         variant="outlined" 
//         color="secondary" 
//         sx={{ mb: 2, position: 'absolute', top: 10, right: 10 }}
//         onClick={() => setShowDebug(!showDebug)}
//       >
//         {showDebug ? "Hide Debug" : "Debug"}
//       </Button>
      
//       <Stepper activeStep={step} sx={{ mb: 4 }}>
//         {steps.map((label) => (
//           <Step key={label}>
//             <StepLabel>{label}</StepLabel>
//           </Step>
//         ))}
//       </Stepper>
      
//       {renderStep()}
      
//       {showDebug && <DebugPanel />}
//     </Paper>
//   );
// };

// export default TestProctoring;

import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { auth, db } from "../firebase-config";
import { doc, getDoc, updateDoc } from "firebase/firestore";
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Dialog,
  Grid,
  Alert,
  Stepper,
  Step,
  StepLabel,
} from "@mui/material";

// API configuration
const API_URL = "http://127.0.0.1:8000";
const VERIFICATION_ENDPOINT = "/compare_faces/";

const TestProctoring = ({ onVerificationComplete, jobId }) => {
  const navigate = useNavigate();
  const { jobId: urlJobId } = useParams();
  const actualJobId = jobId || urlJobId;
  
  const [user, setUser] = useState(null);
  const [profilePicture, setProfilePicture] = useState(null);
  const [loading, setLoading] = useState(true);
  const [step, setStep] = useState(0);
  const [error, setError] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState("pending"); // pending, processing, success, failed
  const [similarity, setSimilarity] = useState(null);
  const [apiAvailable, setApiAvailable] = useState(null); // null=unknown, true=available, false=unavailable
  const [debugInfo, setDebugInfo] = useState([]); // Store debug information
  const [showDebug, setShowDebug] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);
  
  const steps = ["Identity Verification", "Rules Acceptance"];

  // Add debug log function
  const addDebugLog = (message, type = "info") => {
    console.log(`[${type.toUpperCase()}] ${message}`);
    setDebugInfo(prev => [...prev, { 
      timestamp: new Date().toISOString(),
      type, 
      message 
    }]);
  };

  // Check if API is available
  useEffect(() => {
    const checkApiAvailability = async () => {
      try {
        addDebugLog("Checking API availability...");
        const response = await fetch(`${API_URL}/`, { 
          method: 'GET',
          // Set timeout to avoid long waiting
          signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
          addDebugLog("API is available", "success");
          setApiAvailable(true);
        } else {
          addDebugLog(`API returned status ${response.status}`, "error");
          setApiAvailable(false);
          setError("Verification service is not responding correctly. Please try again later.");
        }
      } catch (err) {
        addDebugLog(`API connectivity error: ${err.message}`, "error");
        setApiAvailable(false);
        setError("Cannot connect to verification service. Please ensure the service is running.");
      }
    };
    
    checkApiAvailability();
  }, []);

  // Load user data and profile picture
  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        addDebugLog("Fetching user profile...");
        const currentUser = auth.currentUser;
        if (!currentUser) {
          addDebugLog("No authenticated user found", "warning");
          navigate("/login", { state: { redirect: `/competency-test/${actualJobId}` } });
          return;
        }
        
        setUser({
          uid: currentUser.uid,
          email: currentUser.email,
          name: currentUser.displayName || currentUser.email,
        });
        
        // Fetch user profile picture from Firestore
        addDebugLog(`Fetching profile picture for user: ${currentUser.uid}`);
        const userProfileDoc = await getDoc(doc(db, "userProfiles", currentUser.uid));
        if (userProfileDoc.exists() && userProfileDoc.data().profilePicture) {
          addDebugLog("Profile picture found", "success");
          setProfilePicture(userProfileDoc.data().profilePicture);
        } else {
          addDebugLog("No profile picture found", "error");
          setError("Profile picture not found. Please upload a profile picture before taking the test.");
        }
      } catch (err) {
        addDebugLog(`Error fetching user profile: ${err.message}`, "error");
        setError(`Failed to load user profile: ${err.message}. Please try again.`);
      } finally {
        setLoading(false);
      }
    };
    
    fetchUserProfile();
  }, [actualJobId, navigate]);
  
  // Start camera when ready for verification
  const startCamera = async () => {
    try {
      addDebugLog("Attempting to start camera...");
      if (videoRef.current) {
        const constraints = { 
          video: { 
            width: 640, 
            height: 480,
            facingMode: "user"
          } 
        };
        
        addDebugLog("Requesting camera with constraints: " + JSON.stringify(constraints));
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        videoRef.current.srcObject = stream;
        mediaStreamRef.current = stream;
        
        // Ensure video is actually playing
        videoRef.current.onloadedmetadata = () => {
          addDebugLog("Camera started successfully", "success");
          videoRef.current.play().catch(playErr => {
            addDebugLog(`Error playing video: ${playErr.message}`, "error");
            setError(`Camera started but couldn't play: ${playErr.message}`);
          });
        };
      } else {
        addDebugLog("Video reference not found", "error");
        setError("Camera initialization failed. Please refresh and try again.");
      }
    } catch (err) {
      addDebugLog(`Error accessing camera: ${err.message}`, "error");
      
      let errorMsg = "Camera access is required for identity verification.";
      
      // More specific error messages based on the error
      if (err.name === "NotAllowedError") {
        errorMsg += " You denied permission to use the camera. Please allow camera access and try again.";
      } else if (err.name === "NotFoundError") {
        errorMsg += " No camera was found on your device.";
      } else if (err.name === "NotReadableError") {
        errorMsg += " Your camera might be in use by another application.";
      } else {
        errorMsg += ` Error: ${err.message}`;
      }
      
      setError(errorMsg);
    }
  };
  
  // Check if video is actually receiving data
  const checkVideoStream = () => {
    if (!videoRef.current || !mediaStreamRef.current) {
      addDebugLog("No video stream to check", "warning");
      return false;
    }
    
    const tracks = mediaStreamRef.current.getVideoTracks();
    if (!tracks || tracks.length === 0) {
      addDebugLog("No video tracks found in media stream", "error");
      return false;
    }
    
    const isActive = tracks[0].enabled && tracks[0].readyState === 'live';
    addDebugLog(`Video stream active: ${isActive}`);
    
    return isActive;
  };
  
  // Stop camera
  const stopCamera = () => {
    addDebugLog("Stopping camera...");
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => {
        track.stop();
        addDebugLog(`Stopped track: ${track.kind}`);
      });
    } else {
      addDebugLog("No media stream to stop", "warning");
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Capture current frame from webcam as a blob
  const captureImage = () => {
    return new Promise((resolve, reject) => {
      if (!videoRef.current) {
        const error = new Error("Video stream not available");
        addDebugLog(`Capture failed: ${error.message}`, "error");
        reject(error);
        return;
      }
      
      // Check if video is actually streaming
      if (!checkVideoStream()) {
        const error = new Error("Video stream is not active");
        addDebugLog(`Capture failed: ${error.message}`, "error");
        reject(error);
        return;
      }
      
      // Make sure video has loaded metadata
      if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
        const error = new Error("Video dimensions not available yet");
        addDebugLog(`Capture failed: ${error.message}`, "error");
        reject(error);
        return;
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      addDebugLog(`Capturing frame at ${canvas.width}x${canvas.height}`);
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(blob => {
        if (blob) {
          addDebugLog(`Captured image blob size: ${blob.size} bytes`, "success");
          resolve(blob);
        } else {
          const error = new Error("Failed to capture image");
          addDebugLog(`Capture failed: ${error.message}`, "error");
          reject(error);
        }
      }, 'image/jpeg', 0.95);
    });
  };
  
  // Convert profile picture URL to a blob
  const getProfileImageBlob = async (url) => {
    try {
      addDebugLog(`Fetching profile image from URL: ${url.substring(0, 50)}...`);
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch profile image: ${response.status} ${response.statusText}`);
      }
      
      const blob = await response.blob();
      addDebugLog(`Profile image blob size: ${blob.size} bytes`, "success");
      return blob;
    } catch (err) {
      addDebugLog(`Error fetching profile image: ${err.message}`, "error");
      throw new Error(`Failed to fetch profile image: ${err.message}`);
    }
  };
  
  // Perform face verification using the API
  const verifyIdentity = async () => {
    if (!videoRef.current || !profilePicture) {
      setError("Verification system not ready. Please try again.");
      return;
    }

    if (!apiAvailable) {
      setError("Verification service is not available. Please try again later.");
      return;
    }

    addDebugLog("Starting verification process...");
    setVerificationStatus("processing");
    
    try {
      // Capture current frame from webcam
      addDebugLog("Capturing webcam image...");
      const webcamImageBlob = await captureImage();
      
      // Get profile image as blob
      addDebugLog("Retrieving profile image...");
      const profileImageBlob = await getProfileImageBlob(profilePicture);
      
      // Create form data for API request
      const formData = new FormData();
      formData.append("image1", profileImageBlob, "profile.jpg");
      formData.append("image2", webcamImageBlob, "webcam.jpg");
      
      // Send request to backend API
      addDebugLog(`Sending comparison request to ${API_URL}${VERIFICATION_ENDPOINT}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      try {
        const response = await fetch(`${API_URL}${VERIFICATION_ENDPOINT}`, {
          method: "POST",
          body: formData,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Check for HTTP errors
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`API request failed with status ${response.status}: ${errorText}`);
        }
        
        // Parse response
        const data = await response.json();
        addDebugLog(`Face comparison result: ${JSON.stringify(data)}`);
        
        // Set similarity score
        setSimilarity(data.similarity_score);
        
        // Check for API-reported errors
        if ("error" in data) {
          throw new Error(data.error || "API reported an error");
        }
        
        // When comparison completes
        if (data.match) {
          addDebugLog("Verification SUCCESS - faces match!", "success");
          setVerificationStatus("success");
          // Move directly to rules step after a short delay
          setTimeout(() => {
            setStep(1);
            // Reset verification status for next step
            setVerificationStatus("pending");
          }, 1500);
        } else {
          addDebugLog(`Verification FAILED - faces don't match. Similarity: ${data.similarity_score}`, "warning");
          setError(`Identity verification failed. Similarity score: ${data.similarity_score.toFixed(2)}`);
          setVerificationStatus("failed");
        }
      } catch (fetchErr) {
        clearTimeout(timeoutId);
        
        if (fetchErr.name === 'AbortError') {
          addDebugLog("API request timed out", "error");
          throw new Error("Verification request timed out. Please try again.");
        } else {
          throw fetchErr;
        }
      }
    } catch (err) {
      addDebugLog(`Verification error: ${err.message}`, "error");
      setError(`Verification failed: ${err.message}. Please try again in better lighting conditions.`);
      setVerificationStatus("failed");
    }
  };
  
  // Accept rules and start test
  const acceptRulesAndStart = () => {
    // Stop camera before proceeding to test
    addDebugLog("User accepted rules, stopping camera and proceeding to test", "success");
    stopCamera();
    
    // Call the completion callback to start the test
    if (onVerificationComplete) {
      onVerificationComplete(true);
    } else {
      // If used as standalone component, navigate to the test
      navigate(`/competency-test/${actualJobId}`);
    }
  };
  
  // Reset verification if it fails
  const resetVerification = () => {
    addDebugLog("Resetting verification state", "info");
    setVerificationStatus("pending");
    setError(null);
    
    // Check if we need to restart camera
    if (!checkVideoStream()) {
      addDebugLog("Restarting camera during reset", "info");
      startCamera();
    }
  };
  
  // Debug panel component
  const DebugPanel = () => (
    <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1, maxHeight: 200, overflow: 'auto' }}>
      <Typography variant="subtitle2">Debug Log:</Typography>
      {debugInfo.map((log, index) => (
        <Typography key={index} variant="caption" sx={{ 
          display: 'block', 
          color: log.type === 'error' ? 'error.main' : 
                 log.type === 'warning' ? 'warning.main' : 
                 log.type === 'success' ? 'success.main' : 'text.secondary'
        }}>
          [{log.timestamp.split('T')[1].split('.')[0]}] {log.message}
        </Typography>
      ))}
    </Box>
  );
  
  // Render different steps
  const renderStep = () => {
    switch (step) {
      case 0: // Identity verification
        return (
          <Box sx={{ mt: 2, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>Identity Verification</Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Please look directly at the camera and position your face within the frame.
            </Typography>
            
            {!apiAvailable && apiAvailable !== null && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error || "Face verification service is not available. Please make sure the service is running."}
              </Alert>
            )}
            
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={6}>
                <Box sx={{ border: "1px solid #ddd", borderRadius: 2, overflow: "hidden", height: 240 }}>
                  <Typography variant="subtitle2" sx={{ p: 1, bgcolor: "#f5f5f5" }}>Profile Picture</Typography>
                  {profilePicture ? (
                    <img 
                      src={profilePicture} 
                      alt="Profile" 
                      style={{ width: "100%", height: 200, objectFit: "cover" }} 
                      onError={() => {
                        addDebugLog("Failed to load profile picture", "error");
                        setError("Could not load your profile picture. Please check your profile.");
                      }}
                    />
                  ) : (
                    <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">No profile picture found</Typography>
                    </Box>
                  )}
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ border: "1px solid #ddd", borderRadius: 2, overflow: "hidden", height: 240 }}>
                  <Typography variant="subtitle2" sx={{ p: 1, bgcolor: "#f5f5f5" }}>Camera Feed</Typography>
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    style={{ width: "100%", height: 200, objectFit: "cover" }} 
                    onError={() => {
                      addDebugLog("Video element error", "error");
                      setError("Camera feed error. Please refresh and try again.");
                    }}
                  />
                </Box>
              </Grid>
            </Grid>
            
            {!videoRef.current?.srcObject && (
              <Button
                variant="contained"
                color="primary"
                onClick={startCamera}
                sx={{ mb: 2 }}
                disabled={!apiAvailable && apiAvailable !== null}
              >
                Start Camera
              </Button>
            )}
            
            {videoRef.current?.srcObject && verificationStatus === "pending" && (
              <Button
                variant="contained"
                color="primary"
                onClick={verifyIdentity}
                sx={{ mb: 2 }}
                disabled={!apiAvailable && apiAvailable !== null}
              >
                Verify Identity
              </Button>
            )}
            
            {verificationStatus === "processing" && (
              <Box sx={{ textAlign: "center", my: 2 }}>
                <CircularProgress size={24} sx={{ mr: 1 }} />
                <Typography>Verifying your identity...</Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                  Comparing facial features...
                </Typography>
              </Box>
            )}
            
            {verificationStatus === "success" && (
              <Alert severity="success" sx={{ mb: 2 }}>
                <Typography variant="body2">Identity verified successfully!</Typography>
                <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
                  Face match confirmed (Similarity: {similarity?.toFixed(2)}). Proceeding to next step...
                </Typography>
              </Alert>
            )}
            
            {verificationStatus === "failed" && (
              <Box>
                <Alert severity="error" sx={{ mb: 2 }}>
                  <Typography variant="body2">{error || "Verification failed. Please try again."}</Typography>
                  <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
                    {similarity !== null && `Similarity score: ${similarity.toFixed(2)}`}
                  </Typography>
                  <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
                    Make sure your face is clearly visible and well-lit.
                  </Typography>
                </Alert>
                <Button variant="outlined" onClick={resetVerification}>
                  Try Again
                </Button>
              </Box>
            )}
            
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </Box>
        );
        
      case 1: // Rules acceptance
        return (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>Test Rules</Typography>
            <Alert severity="info" sx={{ mb: 3 }}>
              By proceeding, you agree to the following rules:
            </Alert>
            
            <Box sx={{ bgcolor: "#f8f9fa", p: 2, borderRadius: 2, mb: 3 }}>
              <Typography component="div" variant="body2">
                <ul>
                  <li>You must remain visible on camera throughout the test</li>
                  <li>No additional screens or devices are allowed</li>
                  <li>No other individuals may be present in the room</li>
                  <li>You may not browse other websites or use search engines</li>
                  <li>You may not use notes, books, or other resources</li>
                  <li>Leaving the test window will be flagged and may result in disqualification</li>
                </ul>
              </Typography>
            </Box>
            
            <Button
              variant="contained"
              color="primary"
              fullWidth
              onClick={acceptRulesAndStart}
            >
              I Accept & Start Test
            </Button>
          </Box>
        );
        
      default:
        return null;
    }
  };
  
  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error && !profilePicture) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={() => navigate("/jobseeker-dashboard")}>
          Go to Profile
        </Button>
      </Box>
    );
  };

  return (
    <Paper sx={{ p: 3, maxWidth: 800, mx: "auto", borderRadius: 2, boxShadow: 2 }}>
      <Typography variant="h5" gutterBottom>
        Test Verification
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Before you can take the test, we need to verify your identity and ensure test security.
      </Typography>
      
      <Button 
        size="small" 
        variant="outlined" 
        color="secondary" 
        sx={{ mb: 2, position: 'absolute', top: 10, right: 10 }}
        onClick={() => setShowDebug(!showDebug)}
      >
        {showDebug ? "Hide Debug" : "Debug"}
      </Button>
      
      <Stepper activeStep={step} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      {renderStep()}
      
      {showDebug && <DebugPanel />}
    </Paper>
  );
};

export default TestProctoring;