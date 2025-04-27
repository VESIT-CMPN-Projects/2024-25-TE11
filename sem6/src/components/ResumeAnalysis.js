import React, { useState, useEffect } from 'react';
import { 
  Container, Box, Typography, Select, MenuItem, Chip, 
  Button, TextField, Paper, Tab, Tabs, CircularProgress,
  FormControl, InputLabel, OutlinedInput, FormHelperText
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { styled } from '@mui/material/styles';

// API URL - Change this to your deployed API URL
const API_URL = 'http://localhost:5000/api';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const SkillBridgeAI = () => {
  // State variables
  const [jobRoles, setJobRoles] = useState([]);
  const [selectedRole, setSelectedRole] = useState('');
  const [availableSkills, setAvailableSkills] = useState([]);
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeBase64, setResumeBase64] = useState('');
  const [tabValue, setTabValue] = useState(0);
  const [analysisResults, setAnalysisResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch job roles on component mount
  useEffect(() => {
    fetchJobRoles();
  }, []);

  // Fetch available skills when selected role changes
  useEffect(() => {
    if (selectedRole) {
      fetchSkillsForRole(selectedRole);
    }
  }, [selectedRole]);

  // Fetch job roles from API
  const fetchJobRoles = async () => {
    try {
      const response = await fetch(`${API_URL}/job-roles`);
      const data = await response.json();
      setJobRoles(data.jobRoles || []);
    } catch (error) {
      console.error('Error fetching job roles:', error);
      setError('Failed to fetch job roles');
    }
  };

  // Fetch skills for selected role
  const fetchSkillsForRole = async (role) => {
    try {
      const response = await fetch(`${API_URL}/skills?role=${encodeURIComponent(role)}`);
      const data = await response.json();
      setAvailableSkills(data.skills || []);
      setSelectedSkills([]);
    } catch (error) {
      console.error('Error fetching skills:', error);
      setError('Failed to fetch skills for the selected role');
    }
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      setResumeFile(file);
      
      // Convert file to base64
      const reader = new FileReader();
      reader.onload = () => {
        setResumeBase64(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please upload a PDF file');
      setResumeFile(null);
      setResumeBase64('');
    }
  };

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setTabValue(newValue);
  };

  // Handle skill selection
  const handleSkillChange = (event) => {
    const { value } = event.target;
    setSelectedSkills(value);
  };

  // Run analysis
  const runAnalysis = async (analysisType) => {
    if (!selectedRole) {
      setError('Please select a job role');
      return;
    }

    if (selectedSkills.length === 0) {
      setError('Please select at least one skill');
      return;
    }

    if ((analysisType === 'resume_review' || analysisType === 'percentage_match') && !resumeFile) {
      setError('Please upload a resume PDF');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysisType,
          expectedRole: selectedRole,
          skills: selectedSkills,
          jobDescription,
          pdfBase64: resumeBase64,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze');
      }

      // Update results
      setAnalysisResults({
        ...analysisResults,
        [analysisType]: data,
      });
    } catch (error) {
      console.error('Analysis error:', error);
      setError(error.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  // Main render
  return (
    <Container maxWidth="lg" sx={{ my: 4 }}>
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          background: 'linear-gradient(to right, #121212, #1E1E1E)',
          color: '#E0E0E0',
          borderRadius: 2
        }}
      >
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom 
          sx={{ 
            textAlign: 'center', 
            background: 'linear-gradient(to right, #007BFF, #3a7bd5)',
            color: 'white',
            p: 2,
            borderRadius: 1,
            mb: 4
          }}
        >
          SkillBridgeAI
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4, mb: 4 }}>
          {/* Left Column */}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" color="#007BFF" gutterBottom>
              Your Information
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="role-select-label">Select Expected Job Role</InputLabel>
              <Select
                labelId="role-select-label"
                value={selectedRole}
                label="Select Expected Job Role"
                onChange={(e) => setSelectedRole(e.target.value)}
              >
                {jobRoles.map((role) => (
                  <MenuItem key={role} value={role}>{role}</MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="skills-select-label">Select Your Skills</InputLabel>
              <Select
                labelId="skills-select-label"
                multiple
                value={selectedSkills}
                onChange={handleSkillChange}
                input={<OutlinedInput label="Select Your Skills" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {availableSkills.map((skill) => (
                  <MenuItem key={skill} value={skill}>
                    {skill}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Button
              component="label"
              variant="outlined"
              startIcon={<UploadFileIcon />}
              sx={{ mb: 1, width: '100%' }}
            >
              Upload Resume (PDF)
              <VisuallyHiddenInput type="file" accept=".pdf" onChange={handleFileUpload} />
            </Button>

            {resumeFile && (
              <Typography variant="body2" color="#00cc66" sx={{ mb: 2 }}>
                ✅ {resumeFile.name} uploaded successfully
              </Typography>
            )}
          </Box>

          {/* Right Column */}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" color="#007BFF" gutterBottom>
              Job Details
            </Typography>
            <TextField
              label="Enter Job Description"
              multiline
              rows={12}
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="Paste the job description to get more accurate analysis"
            />
          </Box>
        </Box>

        {error && (
          <Box sx={{ mb: 2, p: 2, bgcolor: 'rgba(211, 47, 47, 0.1)', borderRadius: 1, color: '#f44336' }}>
            {error}
          </Box>
        )}

        <Typography variant="h6" color="#007BFF" gutterBottom>
          Analysis
        </Typography>

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Job Role Fit" id="tab-0" />
            <Tab label="Skill Gap Analysis" id="tab-1" />
            <Tab label="Course Recommendations" id="tab-2" />
            <Tab label="Resume Review" id="tab-3" />
            <Tab label="Percentage Match" id="tab-4" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => runAnalysis('job_fit')}
              disabled={loading}
            >
              {loading && tabValue === 0 ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              Predict Fit
            </Button>
          </Box>
          
          {analysisResults.job_fit && (
            <Paper sx={{ p: 3, bgcolor: '#1E1E1E', borderLeft: '4px solid #007BFF' }}>
              <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-line' }}>
                {analysisResults.job_fit.analysis}
              </Typography>
            </Paper>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => runAnalysis('skill_gap')}
              disabled={loading}
            >
              {loading && tabValue === 1 ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              Analyze Skill Gap
            </Button>
          </Box>
          
          {analysisResults.skill_gap && (
            <Paper sx={{ p: 3, bgcolor: '#1E1E1E', borderLeft: '4px solid #007BFF' }}>
              <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-line' }}>
                {analysisResults.skill_gap.analysis}
              </Typography>
            </Paper>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => runAnalysis('course_recommendations')}
              disabled={loading}
            >
              {loading && tabValue === 2 ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              Get Recommendations
            </Button>
          </Box>
          
          {analysisResults.course_recommendations && (
            <Paper sx={{ p: 3, bgcolor: '#1E1E1E', borderLeft: '4px solid #007BFF' }}>
              <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-line' }}>
                {analysisResults.course_recommendations.analysis}
              </Typography>
            </Paper>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => runAnalysis('resume_review')}
              disabled={loading || !resumeFile}
            >
              {loading && tabValue === 3 ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              Analyze Resume
            </Button>
            {!resumeFile && (
              <FormHelperText error>Resume required for this analysis</FormHelperText>
            )}
          </Box>
          
          {analysisResults.resume_review && (
            <Paper sx={{ p: 3, bgcolor: '#1E1E1E', borderLeft: '4px solid #007BFF' }}>
              <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-line' }}>
                {analysisResults.resume_review.analysis}
              </Typography>
            </Paper>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={() => runAnalysis('percentage_match')}
              disabled={loading || !resumeFile}
            >
              {loading && tabValue === 4 ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              Check Match
            </Button>
            {!resumeFile && (
              <FormHelperText error>Resume required for this analysis</FormHelperText>
            )}
          </Box>
          
          {analysisResults.percentage_match && (
            <Paper sx={{ p: 3, bgcolor: '#1E1E1E', borderLeft: '4px solid #007BFF' }}>
              <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-line' }}>
                {analysisResults.percentage_match.analysis}
              </Typography>
            </Paper>
          )}
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default SkillBridgeAI;