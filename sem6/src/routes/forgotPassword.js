import React, { useState } from 'react';
import {
    Box,
    Button,
    TextField,
    Typography,
    Link,
    Paper,
} from '@mui/material';
import { getAuth, sendPasswordResetEmail } from 'firebase/auth';

const ForgotPasswordPage = () => {
    const [email, setEmail] = useState('');
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    const handlePasswordReset = async (e) => {
        e.preventDefault();
        const auth = getAuth();

        try {
            await sendPasswordResetEmail(auth, email);
            setSuccessMessage('Password reset email sent! Please check your inbox.');
            setError('');
            // Clear the email field after successful submission
            setEmail('');
        } catch (error) {
            setError('Failed to send reset email. Please check if the email is correct.');
            setSuccessMessage('');
            console.error('Password reset error:', error);
        }
    };

    return (
        <Box 
            sx={{ 
                height: '100vh', 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                backgroundImage: 'url(frontend/src/assets/login-bg.jpg)',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
            }}
        >
            <Paper
                sx={{
                    display: 'flex',
                    maxWidth: '400px',
                    borderRadius: '20px',
                    overflow: 'hidden',
                    boxShadow: 10,
                    backgroundColor: '#ffffff',
                    opacity: 0.9,
                }}
            >
                <Box sx={{ width: '100%', padding: 4 }}>
                    <Typography variant="h4" fontWeight="bold" gutterBottom color="#1976d2">
                        Forgot Password
                    </Typography>
                    
                    {error && (
                        <Typography color="error" sx={{ mb: 2 }}>
                            {error}
                        </Typography>
                    )}
                    
                    {successMessage && (
                        <Typography color="success.main" sx={{ mb: 2 }}>
                            {successMessage}
                        </Typography>
                    )}

                    <Typography variant="body1" sx={{ mb: 3 }}>
                        Enter your email address below and we'll send you instructions to reset your password.
                    </Typography>

                    <form onSubmit={handlePasswordReset}>
                        <TextField
                            fullWidth
                            margin="normal"
                            id="email"
                            label="Email ID"
                            variant="outlined"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            sx={{ mb: 3 }}
                        />

                        <Button 
                            type="submit" 
                            fullWidth 
                            variant="contained" 
                            color="primary" 
                            size="large"
                            sx={{ 
                                backgroundColor: '#1976d2',
                                borderRadius: '30px',
                                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)',
                                '&:hover': {
                                    backgroundColor: '#115293',
                                    boxShadow: '0 6px 20px rgba(0, 0, 0, 0.4)',
                                },
                                mb: 2
                            }}
                        >
                            Send Reset Link
                        </Button>

                        <Box sx={{ height: '20px' }}></Box>

                        <Link href="/login">
                            <Button 
                                fullWidth 
                                variant="outlined" 
                                color="primary" 
                                size="large"
                                sx={{
                                    borderRadius: '30px',
                                    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)',
                                    '&:hover': {
                                        boxShadow: '0 6px 20px rgba(0, 0, 0, 0.4)',
                                    },
                                }}
                            >
                                Back to Login
                            </Button>
                        </Link>
                    </form>
                </Box>
            </Paper>
        </Box>
    );
};

export default ForgotPasswordPage;