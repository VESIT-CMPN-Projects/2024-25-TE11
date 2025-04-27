// import React, { useState, useEffect } from 'react';
// import {
//     Box,
//     Button,
//     FormControl,
//     InputLabel,
//     MenuItem,
//     Select,
//     TextField,
//     Typography,
//     Paper,
//     Divider,
//     InputAdornment,
//     IconButton,
//     Fade,
//     Container,
// } from '@mui/material';
// import { Link, useNavigate } from 'react-router-dom';
// import { auth } from '../firebase-config'; 
// import { createUserWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
// import GoogleIcon from '@mui/icons-material/Google';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
// import EmailIcon from '@mui/icons-material/Email';
// import PersonIcon from '@mui/icons-material/Person';
// import LockIcon from '@mui/icons-material/Lock';
// import ArrowBackIcon from '@mui/icons-material/ArrowBack';

// const SignupPage = () => {
//     const [email, setEmail] = useState('');
//     const [password, setPassword] = useState('');
//     const [confirmPassword, setConfirmPassword] = useState('');
//     const [role, setRole] = useState('');
//     const [error, setError] = useState('');
//     const [showPassword, setShowPassword] = useState(false);
//     const [showConfirmPassword, setShowConfirmPassword] = useState(false);
//     const [fadeIn, setFadeIn] = useState(false);
//     const navigate = useNavigate();

//     // Animation effect on page load
//     useEffect(() => {
//         setFadeIn(true);
//     }, []);

//     // Google sign-in provider
//     const provider = new GoogleAuthProvider();

//     const handleGoogleSignIn = async () => {
//         try {
//             const result = await signInWithPopup(auth, provider);
//             const user = result.user;
//             console.log("Google Sign-In successful:", user);
//             if (role === 'jobseeker') {
//                 navigate('/jobseeker-dashboard');
//             } else if (role === 'employer') {
//                 navigate('/employer-dashboard');
//             } 
//         } catch (error) {
//             console.error('Google Sign-In error:', error);
//             setError('Failed to sign in with Google.');
//         }
//     };

//     const handleRegister = async (e) => {
//         e.preventDefault();
//         if (password !== confirmPassword) {
//             setError("Passwords do not match");
//             return;
//         }

//         try {
//             await createUserWithEmailAndPassword(auth, email, password);
//             if (role === 'jobseeker') {
//                 navigate('/get-started');
//             } else if (role === 'employer') {
//                 navigate('/employer-dashboard');
//             }
//         } catch (error) {
//             setError('Failed to register. Please check your details.');
//             console.error('Registration error:', error);
//         }
//     };

//     const togglePasswordVisibility = () => {
//         setShowPassword(!showPassword);
//     };

//     const toggleConfirmPasswordVisibility = () => {
//         setShowConfirmPassword(!showConfirmPassword);
//     };

//     return (
//         <Box 
//             sx={{ 
//                 minHeight: '100vh', 
//                 display: 'flex', 
//                 justifyContent: 'center', 
//                 alignItems: 'center', 
//                 backgroundImage: 'linear-gradient(135deg, #2979ff 0%, #1565c0 100%)',
//                 backgroundSize: 'cover',
//                 backgroundPosition: 'center',
//                 backgroundRepeat: 'no-repeat',
//                 padding: { xs: 2, sm: 4 },
//             }}
//         >
//             <Fade in={fadeIn} timeout={1000}>
//                 <Container maxWidth="xs">
//                     <Paper
//                         elevation={24}
//                         sx={{
//                             borderRadius: '24px',
//                             overflow: 'hidden',
//                             backgroundColor: 'rgba(255, 255, 255, 0.9)',
//                             backdropFilter: 'blur(10px)',
//                             boxShadow: '0 10px 40px rgba(0, 0, 0, 0.2)',
//                             transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
//                             '&:hover': {
//                                 transform: 'translateY(-5px)',
//                                 boxShadow: '0 15px 50px rgba(0, 0, 0, 0.3)',
//                             },
//                         }}
//                     >
//                         <Box 
//                             sx={{ 
//                                 width: '100%', 
//                                 padding: 4,
//                                 display: 'flex',
//                                 flexDirection: 'column',
//                                 alignItems: 'center',
//                             }}
//                         >
//                             <Typography 
//                                 variant="h4" 
//                                 fontWeight="700" 
//                                 gutterBottom 
//                                 align="center"
//                                 sx={{ 
//                                     marginBottom: 3,
//                                     fontFamily: '"Poppins", sans-serif',
//                                     letterSpacing: '0.5px',
//                                     background: 'linear-gradient(to right, #2979ff, #1565c0)',
//                                     WebkitBackgroundClip: 'text',
//                                     WebkitTextFillColor: 'transparent',
//                                 }}
//                             >
//                                 Create Account
//                             </Typography>
                            
//                             {error && (
//                                 <Box 
//                                     sx={{ 
//                                         width: '100%', 
//                                         backgroundColor: 'rgba(255, 0, 0, 0.1)', 
//                                         p: 2, 
//                                         borderRadius: 2,
//                                         mb: 2
//                                     }}
//                                 >
//                                     <Typography color="error" fontSize="0.875rem" align="center">
//                                         {error}
//                                     </Typography>
//                                 </Box>
//                             )}
                            
//                             <form onSubmit={handleRegister} style={{ width: '100%' }}>
//                                 <FormControl fullWidth margin="normal" variant="outlined">
//                                     <InputLabel id="role-label" sx={{ color: '#1976d2' }}>Select Role</InputLabel>
//                                     <Select
//                                         labelId="role-label"
//                                         id="role"
//                                         value={role}
//                                         onChange={(e) => setRole(e.target.value)}
//                                         required
//                                         startAdornment={
//                                             <InputAdornment position="start">
//                                                 <PersonIcon sx={{ color: '#1976d2' }} />
//                                             </InputAdornment>
//                                         }
//                                         sx={{ 
//                                             borderRadius: '12px',
//                                             '& .MuiOutlinedInput-notchedOutline': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                             '&:hover .MuiOutlinedInput-notchedOutline': {
//                                                 borderColor: '#1565c0',
//                                             },
//                                             '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
//                                                 borderColor: '#1976d2',
//                                             }
//                                         }} 
//                                     >
//                                         <MenuItem value="">
//                                             <em>Select Role</em>
//                                         </MenuItem>
//                                         <MenuItem value="jobseeker">Jobseeker</MenuItem>
//                                         <MenuItem value="employer">Employer</MenuItem>
//                                     </Select>
//                                 </FormControl>
                                
//                                 <TextField
//                                     fullWidth
//                                     margin="normal"
//                                     id="email"
//                                     label="Email ID / Username"
//                                     variant="outlined"
//                                     value={email}
//                                     onChange={(e) => setEmail(e.target.value)}
//                                     required
//                                     InputProps={{
//                                         startAdornment: (
//                                             <InputAdornment position="start">
//                                                 <EmailIcon sx={{ color: '#1976d2' }} />
//                                             </InputAdornment>
//                                         ),
//                                     }}
//                                     sx={{ 
//                                         '& .MuiOutlinedInput-root': {
//                                             borderRadius: '12px',
//                                             '& fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                             '&:hover fieldset': {
//                                                 borderColor: '#1565c0',
//                                             },
//                                             '&.Mui-focused fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                         },
//                                         '& .MuiInputLabel-root': {
//                                             color: '#1976d2',
//                                         }
//                                     }}
//                                 />
                                
//                                 <TextField
//                                     fullWidth
//                                     margin="normal"
//                                     id="password"
//                                     label="Password"
//                                     type={showPassword ? "text" : "password"}
//                                     variant="outlined"
//                                     value={password}
//                                     onChange={(e) => setPassword(e.target.value)}
//                                     required
//                                     InputProps={{
//                                         startAdornment: (
//                                             <InputAdornment position="start">
//                                                 <LockIcon sx={{ color: '#1976d2' }} />
//                                             </InputAdornment>
//                                         ),
//                                         endAdornment: (
//                                             <InputAdornment position="end">
//                                                 <IconButton
//                                                     aria-label="toggle password visibility"
//                                                     onClick={togglePasswordVisibility}
//                                                     edge="end"
//                                                 >
//                                                     {showPassword ? 
//                                                         <VisibilityOffIcon sx={{ color: '#1976d2' }} /> : 
//                                                         <VisibilityIcon sx={{ color: '#1976d2' }} />
//                                                     }
//                                                 </IconButton>
//                                             </InputAdornment>
//                                         )
//                                     }}
//                                     sx={{ 
//                                         '& .MuiOutlinedInput-root': {
//                                             borderRadius: '12px',
//                                             '& fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                             '&:hover fieldset': {
//                                                 borderColor: '#1565c0',
//                                             },
//                                             '&.Mui-focused fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                         },
//                                         '& .MuiInputLabel-root': {
//                                             color: '#1976d2',
//                                         }
//                                     }}
//                                 />
                                
//                                 <TextField
//                                     fullWidth
//                                     margin="normal"
//                                     id="confirm-password"
//                                     label="Confirm Password"
//                                     type={showConfirmPassword ? "text" : "password"}
//                                     variant="outlined"
//                                     value={confirmPassword}
//                                     onChange={(e) => setConfirmPassword(e.target.value)}
//                                     required
//                                     InputProps={{
//                                         startAdornment: (
//                                             <InputAdornment position="start">
//                                                 <LockIcon sx={{ color: '#1976d2' }} />
//                                             </InputAdornment>
//                                         ),
//                                         endAdornment: (
//                                             <InputAdornment position="end">
//                                                 <IconButton
//                                                     aria-label="toggle confirm password visibility"
//                                                     onClick={toggleConfirmPasswordVisibility}
//                                                     edge="end"
//                                                 >
//                                                     {showConfirmPassword ? 
//                                                         <VisibilityOffIcon sx={{ color: '#1976d2' }} /> : 
//                                                         <VisibilityIcon sx={{ color: '#1976d2' }} />
//                                                     }
//                                                 </IconButton>
//                                             </InputAdornment>
//                                         )
//                                     }}
//                                     sx={{ 
//                                         '& .MuiOutlinedInput-root': {
//                                             borderRadius: '12px',
//                                             '& fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                             '&:hover fieldset': {
//                                                 borderColor: '#1565c0',
//                                             },
//                                             '&.Mui-focused fieldset': {
//                                                 borderColor: '#1976d2',
//                                             },
//                                         },
//                                         '& .MuiInputLabel-root': {
//                                             color: '#1976d2',
//                                         }
//                                     }}
//                                 />
                                
//                                 <Divider sx={{ my: 3, color: '#1976d2', opacity: 0.7 }}>
//                                     <Typography variant="body2" sx={{ px: 1, color: '#666' }}>
//                                         or continue with
//                                     </Typography>
//                                 </Divider>
                                
//                                 <Button
//                                     fullWidth
//                                     variant="contained"
//                                     size="large"
//                                     onClick={handleGoogleSignIn}
//                                     startIcon={<GoogleIcon />}
//                                     sx={{
//                                         backgroundColor: '#fff',
//                                         color: '#1976d2',
//                                         borderRadius: '12px',
//                                         mb: 3,
//                                         py: 1.5,
//                                         textTransform: 'none',
//                                         fontSize: '1rem',
//                                         fontWeight: 600,
//                                         border: '1px solid #1976d2',
//                                         boxShadow: '0 4px 12px rgba(25, 118, 210, 0.15)',
//                                         '&:hover': {
//                                             backgroundColor: 'rgba(25, 118, 210, 0.04)',
//                                             boxShadow: '0 6px 15px rgba(25, 118, 210, 0.2)',
//                                         },
//                                         transition: 'all 0.3s ease',
//                                     }}
//                                 >
//                                     Sign up with Google
//                                 </Button>
                                
//                                 <Button 
//                                     type="submit" 
//                                     fullWidth 
//                                     variant="contained" 
//                                     size="large"
//                                     sx={{ 
//                                         backgroundColor: '#1976d2',
//                                         borderRadius: '12px',
//                                         padding: '12px',
//                                         textTransform: 'none',
//                                         fontSize: '1rem',
//                                         fontWeight: 600,
//                                         letterSpacing: '0.5px',
//                                         boxShadow: '0 6px 20px rgba(25, 118, 210, 0.4)',
//                                         transition: 'all 0.3s ease',
//                                         '&:hover': {
//                                             backgroundColor: '#1565c0',
//                                             boxShadow: '0 8px 25px rgba(25, 118, 210, 0.6)',
//                                             transform: 'translateY(-2px)',
//                                         },
//                                     }}
//                                 >
//                                     Create Account
//                                 </Button>
                                
//                                 <Box mt={4} textAlign="center">
//                                     <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
//                                         Already have an account?{' '}
//                                         <Link 
//                                             to="/login" 
//                                             style={{ 
//                                                 color: '#1976d2',
//                                                 textDecoration: 'none',
//                                                 fontWeight: 600
//                                             }}
//                                         >
//                                             Log in
//                                         </Link>
//                                     </Typography>
                                    
//                                     <Link to="/" style={{ textDecoration: 'none' }}>
//                                         <Button 
//                                             variant="text"
//                                             startIcon={<ArrowBackIcon />}
//                                             sx={{
//                                                 color: '#1976d2',
//                                                 textTransform: 'none',
//                                                 fontSize: '0.9rem',
//                                                 fontWeight: 500,
//                                                 transition: 'all 0.2s ease',
//                                                 '&:hover': {
//                                                     backgroundColor: 'rgba(25, 118, 210, 0.08)',
//                                                 },
//                                             }}
//                                         >
//                                             Back to home
//                                         </Button>
//                                     </Link>
//                                 </Box>
//                             </form>
//                         </Box>
//                     </Paper>
//                 </Container>
//             </Fade>
//         </Box>
//     );
// };

// export default SignupPage;

import React, { useState, useEffect } from 'react';
import {
    Box,
    Button,
    TextField,
    Typography,
    Link,
    Paper,
    Container,
    Divider,
    InputAdornment,
    IconButton,
    Fade,
    ToggleButtonGroup,
    ToggleButton,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { auth } from '../firebase-config';
import { createUserWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';

// Import icons
import EmailIcon from '@mui/icons-material/Email';
import LockIcon from '@mui/icons-material/Lock';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import GoogleIcon from '@mui/icons-material/Google';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

// Import images
import jobseekerImg from '../assets/jobseeker.png';
import interviewerImg from '../assets/interviewer.png';
import bgImg from '../assets/bg.jpg';

const SignupPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [role, setRole] = useState('jobseeker');
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [fadeIn, setFadeIn] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        setFadeIn(true);
    }, []);

    const handleRegister = async (e) => {
        e.preventDefault();
        setError("");

        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            return;
        }

        try {
            await createUserWithEmailAndPassword(auth, email, password);
            if (role === 'jobseeker') {
                navigate('/get-started');
            } else if (role === 'employer') {
                navigate('/employer-dashboard');
            }
        } catch (error) {
            console.error('Registration error:', error.message);
            setError(error.message);
        }
    };

    const handleGoogleSignIn = async () => {
        const provider = new GoogleAuthProvider();
        
        try {
            const result = await signInWithPopup(auth, provider);
            const user = result.user;
            console.log("Google Sign-In successful:", user);
            if (role === 'jobseeker') {
                navigate('/get-started');
            } else if (role === 'employer') {
                navigate('/employer-dashboard');
            }
        } catch (error) {
            console.error('Google Sign-In error:', error);
            setError('Failed to sign in with Google.');
        }
    };

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    const toggleConfirmPasswordVisibility = () => {
        setShowConfirmPassword(!showConfirmPassword);
    };

    const handleRoleChange = (event, newRole) => {
        if (newRole !== null) {
            setRole(newRole);
        }
    };

    const getRoleImage = () => (role === 'jobseeker' ? jobseekerImg : interviewerImg);

    return (
        <Box 
            sx={{ 
                minHeight: '100vh', 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                backgroundImage: `url(${bgImg})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                padding: { xs: 2, sm: 4 },
                position: 'relative',
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.4)',
                    zIndex: 0
                }
            }}
        >
            <Fade in={fadeIn} timeout={1000}>
                <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
                    <Paper
                        elevation={5}
                        sx={{
                            display: 'flex',
                            borderRadius: '8px',
                            overflow: 'hidden',
                            backgroundColor: 'white',
                            maxHeight: '90vh',
                            boxShadow: '0 4px 20px rgba(0,0,0,0.15)'
                        }}
                    >
                        {/* Left Side */}
                        <Box 
                            sx={{ 
                                width: '45%',
                                backgroundColor: '#0066CC',
                                display: { xs: 'none', md: 'flex' },
                                flexDirection: 'column',
                                padding: 3,
                                justifyContent: 'center',
                                alignItems: 'center',
                            }}
                        >
                            <Box 
                                component="img" 
                                src={getRoleImage()}
                                alt={role === 'jobseeker' ? "Candidate with laptop" : "Recruiter with laptop"}
                                sx={{ 
                                    width: '100%', 
                                    maxHeight: '60%',
                                    objectFit: 'contain',
                                }}
                            />
                        </Box>
                        
                        {/* Right Side - Signup Form */}
                        <Box 
                            sx={{ 
                                width: { xs: '100%', md: '55%' },
                                padding: { xs: 3, md: 4 },
                                display: 'flex',
                                flexDirection: 'column',
                                overflow: 'auto',
                            }}
                        >
                            <Typography variant="h4" fontWeight="700" gutterBottom>
                                Create Account
                            </Typography>
                            
                            {/* Role toggle */}
                            <ToggleButtonGroup
                                value={role}
                                exclusive
                                onChange={handleRoleChange}
                                aria-label="user role"
                                sx={{ 
                                    mb: 2.5,
                                    justifyContent: 'center',
                                    '& .MuiToggleButtonGroup-grouped': {
                                        border: 1,
                                        borderColor: '#e0e0e0',
                                        padding: '6px 24px',
                                        '&.Mui-selected': {
                                            backgroundColor: '#0066CC',
                                            color: 'white',
                                            '&:hover': {
                                                backgroundColor: '#004c99',
                                            }
                                        }
                                    }
                                }}
                            >
                                <ToggleButton value="jobseeker">As Jobseeker</ToggleButton>
                                <ToggleButton value="employer">As Recruiter</ToggleButton>
                            </ToggleButtonGroup>
                            
                            <Button
                                variant="outlined"
                                fullWidth
                                startIcon={<GoogleIcon />}
                                onClick={handleGoogleSignIn}
                                sx={{ mb: 3 }}
                            >
                                Continue with Google
                            </Button>
                            
                            <Divider sx={{ mb: 3 }}>
                                <Typography variant="body2" color="text.secondary">
                                    Or signup with email
                                </Typography>
                            </Divider>
                            
                            {error && (
                                <Typography color="error" align="center" sx={{ mb: 2 }}>
                                    {error}
                                </Typography>
                            )}
                            
                            <form onSubmit={handleRegister}>
                                <TextField
                                    fullWidth
                                    margin="normal"
                                    placeholder="Email Id"
                                    variant="outlined"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <EmailIcon />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                                
                                <TextField
                                    fullWidth
                                    margin="normal"
                                    placeholder="Enter Your Password"
                                    type={showPassword ? "text" : "password"}
                                    variant="outlined"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <LockIcon />
                                            </InputAdornment>
                                        ),
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <IconButton onClick={togglePasswordVisibility}>
                                                    {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                                                </IconButton>
                                            </InputAdornment>
                                        )
                                    }}
                                />
                                
                                <TextField
                                    fullWidth
                                    margin="normal"
                                    placeholder="Confirm Password"
                                    type={showConfirmPassword ? "text" : "password"}
                                    variant="outlined"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    required
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <LockIcon />
                                            </InputAdornment>
                                        ),
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <IconButton onClick={toggleConfirmPasswordVisibility}>
                                                    {showConfirmPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                                                </IconButton>
                                            </InputAdornment>
                                        )
                                    }}
                                />
                                
                                <Button type="submit" fullWidth variant="contained" sx={{ mt: 2 }}>
                                    Create Account
                                </Button>
                                
                                <Typography variant="body2" align="center" sx={{ mt: 2 }}>
                                    Already have an account? <Link href="/login">Log In</Link>
                                </Typography>
                            </form>

                            {/* Back to Home Page Button */}
                            <Button
                                variant="outlined"
                                fullWidth
                                startIcon={<ArrowBackIcon />}
                                sx={{ mt: 2 }}
                                onClick={() => navigate('/')}
                            >
                                Back to Home Page
                            </Button>
                        </Box>
                    </Paper>
                </Container>
            </Fade>
        </Box>
    );
};

export default SignupPage;