import React, { useState } from 'react';
import { Button, Typography, Box } from '@mui/material';

const ResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [loading, setLoading] = useState(false); 
  const [error, setError] = useState(''); 

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
  };

  const handleFileUpload = async () => {
    if (file) {
      setLoading(true); 
      setError(''); 

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:5000/api/scan-resume', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Failed to scan the file'); 
        }

        const data = await response.json();
        setExtractedText(data.extractedText); 
      } catch (error) {
        setError('Error scanning the file. Please try again.'); 
        console.error('Error:', error);
      } finally {
        setLoading(false); 
      }
    }
  };

  return (
    <Box>
      <Typography variant="h6">Upload Your Resume (PDF/JPG/JPEG)</Typography>
      <input
        accept="application/pdf, image/jpeg, image/png"
        type="file"
        onChange={handleFileChange}
      />
      <Button
        variant="contained"
        color="primary"
        onClick={handleFileUpload}
        style={{ marginTop: '10px' }}
        disabled={loading}
      >
        {loading ? 'Uploading and Scanning...' : 'Upload and Scan'}
      </Button>
      {error && (
        <Box mt={4}>
          <Typography color="error">{error}</Typography> 
        </Box>
      )}
      {extractedText && (
        <Box mt={4}>
          <Typography variant="h6">Extracted Text from Resume:</Typography>
          <Typography>{extractedText}</Typography> 
        </Box>
      )}
    </Box>
  );
};

export default ResumeUpload;
