import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Grid,
  Paper,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Container,
  Snackbar,
  Alert,
  CircularProgress
} from "@mui/material";
import { 
  Work, 
  AttachMoney, 
  Edit, 
  Delete, 
  Category, 
  AccessTime,
  Search as SearchIcon,
  FilterList
} from "@mui/icons-material";
import { db } from "../firebase-config";
import { collection, getDocs, deleteDoc, doc, updateDoc, query, where } from "firebase/firestore";import { auth } from "../firebase-config";

const jobCategories = {
  "Software Engineer": "Software Development",
  "Full Stack Developer": "Software Development",
  "AI ML Engineer": "Data Science & AI",
  "CLoud And Devops Engineer": "IT And Cloud",
};

const jobRoles = Object.keys(jobCategories);

const ManageJobs = () => {
  const [jobs, setJobs] = useState([]);
  const [filteredJobs, setFilteredJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [currentJob, setCurrentJob] = useState(null);
  const [editFormData, setEditFormData] = useState({
    jobTitle: '',
    jobDescription: '',
    salary: '',
    jobType: '',
    requiredSkills: []
  });
  const [skillInput, setSkillInput] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // Fetch jobs on component mount
  useEffect(() => {
    fetchJobs();
  }, []);

  // Filter jobs when search term or category filter changes
  useEffect(() => {
    filterJobs();
  }, [jobs, searchTerm, categoryFilter]);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      // Get the current user ID
      const currentUser = auth.currentUser;
      
      if (!currentUser) {
        console.error("No user is logged in");
        setSnackbar({
          open: true,
          message: 'Please log in to view your jobs',
          severity: 'error'
        });
        setLoading(false);
        return;
      }
      
      const userId = currentUser.uid;
      
      // Query jobs collection with a filter for the current user's ID
      const querySnapshot = await getDocs(
        query(collection(db, "jobs"), where("userId", "==", userId))
      );
      
      const jobsData = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      
      setJobs(jobsData);
      setFilteredJobs(jobsData);
    } catch (error) {
      console.error("Error fetching jobs: ", error);
      setSnackbar({
        open: true,
        message: 'Error fetching jobs',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const filterJobs = () => {
    let result = [...jobs];
    
    // Apply search filter
    if (searchTerm) {
      result = result.filter(job => 
        job.jobTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
        job.jobDescription.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (job.requiredSkills && job.requiredSkills.some(skill => 
          skill.toLowerCase().includes(searchTerm.toLowerCase())
        ))
      );
    }
    
    // Apply category filter
    if (categoryFilter) {
      result = result.filter(job => job.category === categoryFilter);
    }
    
    setFilteredJobs(result);
  };

  const handleEdit = (job) => {
    setCurrentJob(job);
    setEditFormData({
      jobTitle: job.jobTitle,
      jobDescription: job.jobDescription,
      salary: job.salary,
      jobType: job.jobType,
      requiredSkills: job.requiredSkills || []
    });
    setOpenEditDialog(true);
  };

  const handleDelete = async (jobId) => {
    try {
      await deleteDoc(doc(db, "jobs", jobId));
      setJobs(jobs.filter(job => job.id !== jobId));
      setSnackbar({
        open: true,
        message: 'Job deleted successfully!',
        severity: 'success'
      });
    } catch (error) {
      console.error("Error deleting job: ", error);
      setSnackbar({
        open: true,
        message: 'Error deleting job',
        severity: 'error'
      });
    }
  };

  const handleAddSkill = () => {
    if (skillInput.trim() && !editFormData.requiredSkills.includes(skillInput.trim())) {
      setEditFormData({
        ...editFormData,
        requiredSkills: [...editFormData.requiredSkills, skillInput.trim()]
      });
      setSkillInput('');
    }
  };

  const handleRemoveSkill = (skillToRemove) => {
    setEditFormData({
      ...editFormData,
      requiredSkills: editFormData.requiredSkills.filter(skill => skill !== skillToRemove)
    });
  };

  const handleEditFormChange = (e) => {
    const { name, value } = e.target;
    setEditFormData({
      ...editFormData,
      [name]: value
    });
  };

  const handleSaveEdit = async () => {
    try {
      const updatedJob = {
        ...currentJob,
        jobTitle: editFormData.jobTitle,
        jobDescription: editFormData.jobDescription,
        category: jobCategories[editFormData.jobTitle] || currentJob.category,
        salary: editFormData.salary,
        jobType: editFormData.jobType,
        requiredSkills: editFormData.requiredSkills
      };

      await updateDoc(doc(db, "jobs", currentJob.id), updatedJob);
      
      // Update local state
      setJobs(jobs.map(job => 
        job.id === currentJob.id ? updatedJob : job
      ));
      
      setSnackbar({
        open: true,
        message: 'Job updated successfully!',
        severity: 'success'
      });
      
      setOpenEditDialog(false);
    } catch (error) {
      console.error("Error updating job: ", error);
      setSnackbar({
        open: true,
        message: 'Error updating job',
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight="bold" gutterBottom color="primary">
        Manage Your Posted Jobs
      </Typography>

      {/* Search and Filter Section */}
      <Paper sx={{ p: 3, mb: 4, borderRadius: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={5}>
            <TextField
              fullWidth
              placeholder="Search by title, description or skills"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
              }}
            />
          </Grid>
          <Grid item xs={12} md={5}>
            <FormControl fullWidth>
              <InputLabel>Filter by Category</InputLabel>
              <Select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                label="Filter by Category"
                startAdornment={<FilterList sx={{ color: 'text.secondary', mr: 1 }} />}
              >
                <MenuItem value="">All Categories</MenuItem>
                {Object.values(jobCategories).filter((value, index, self) => 
                  self.indexOf(value) === index
                ).map((category) => (
                  <MenuItem key={category} value={category}>{category}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button 
              fullWidth 
              variant="outlined" 
              onClick={() => {
                setSearchTerm('');
                setCategoryFilter('');
              }}
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Jobs List */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 8 }}>
          <CircularProgress />
        </Box>
      ) : filteredJobs.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center', borderRadius: 2 }}>
          <Typography variant="body1" color="text.secondary">
            {jobs.length === 0 
              ? "No jobs posted yet. Create a new job using the Post Job page!" 
              : "No jobs match your search criteria."}
          </Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {filteredJobs.map((job) => (
            <Grid item xs={12} md={6} lg={4} key={job.id}>
              <Paper 
                sx={{ 
                  p: 3, 
                  borderRadius: 2, 
                  boxShadow: 2,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Work sx={{ color: 'primary.main', mr: 1 }} />
                  <Typography variant="h6" fontWeight="bold">
                    {job.jobTitle}
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Category sx={{ color: 'text.secondary', fontSize: 18, mr: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    {job.category}
                  </Typography>
                </Box>
                
                {/* <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  sx={{ 
                    mb: 2,
                    display: '-webkit-box',
                    overflow: 'hidden',
                    WebkitBoxOrient: 'vertical',
                    WebkitLineClamp: 3,
                  }}
                >
                  {job.jobDescription}
                </Typography> */}
                
                <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <AttachMoney sx={{ color: 'success.main', fontSize: 20 }} />
                    <Typography variant="body1" fontWeight="medium" color="success.main">
                      ₹{job.salary}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <AccessTime sx={{ color: 'text.secondary', fontSize: 18, mr: 1 }} />
                    <Typography variant="body2" color="text.secondary">
                      {job.jobType}
                    </Typography>
                  </Box>
                </Box>
                
                <Box sx={{ mb: 2, flexGrow: 1 }}>
                  <Typography variant="body2" fontWeight="medium" sx={{ mb: 1 }}>
                    Skills:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {job.requiredSkills && job.requiredSkills.map((skill) => (
                      <Chip 
                        key={skill} 
                        label={skill} 
                        size="small" 
                        variant="outlined"
                        sx={{ borderRadius: 1 }}
                      />
                    ))}
                  </Box>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 'auto', pt: 2 }}>
                  <Button 
                    variant="outlined" 
                    size="small"
                    startIcon={<Edit />}
                    onClick={() => handleEdit(job)}
                  >
                    Edit
                  </Button>
                  <Button 
                    variant="outlined" 
                    color="error" 
                    size="small"
                    startIcon={<Delete />}
                    onClick={() => handleDelete(job.id)}
                  >
                    Delete
                  </Button>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Edit Job Dialog */}
      <Dialog 
        open={openEditDialog} 
        onClose={() => setOpenEditDialog(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>Edit Job</DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Job Title</InputLabel>
                <Select
                  name="jobTitle"
                  value={editFormData.jobTitle}
                  onChange={handleEditFormChange}
                  label="Job Title"
                >
                  {jobRoles.map((role) => (
                    <MenuItem key={role} value={role}>{role}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Category"
                value={jobCategories[editFormData.jobTitle] || ''}
                disabled
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                name="jobDescription"
                label="Job Description"
                multiline
                rows={4}
                value={editFormData.jobDescription}
                onChange={handleEditFormChange}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="salary"
                label="Salary (₹)"
                type="number"
                value={editFormData.salary}
                onChange={handleEditFormChange}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Job Type</InputLabel>
                <Select
                  name="jobType"
                  value={editFormData.jobType}
                  onChange={handleEditFormChange}
                  label="Job Type"
                >
                  <MenuItem value="Full-time">Full-time</MenuItem>
                  <MenuItem value="Part-time">Part-time</MenuItem>
                  <MenuItem value="Remote">Remote</MenuItem>
                  <MenuItem value="Internship">Internship</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <TextField
                  fullWidth
                  label="Add Skills"
                  value={skillInput}
                  onChange={(e) => setSkillInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddSkill())}
                />
                <Button
                  variant="contained"
                  onClick={handleAddSkill}
                  sx={{ height: 56 }}
                >
                  Add
                </Button>
              </Box>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {editFormData.requiredSkills.map((skill) => (
                  <Chip
                    key={skill}
                    label={skill}
                    onDelete={() => handleRemoveSkill(skill)}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveEdit} variant="contained" color="primary">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default ManageJobs;