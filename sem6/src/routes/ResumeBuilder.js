import React, { useState, useEffect } from 'react';
import { Formik, Form, Field, FieldArray } from 'formik';
import { TextField } from 'formik-material-ui';
import { Button,Select,MenuItem,FormControl,InputLabel,Container,Grid,Typography,IconButton } from '@mui/material';
import { Add, Remove } from '@mui/icons-material';
import ResumeViewer from '../components/ResumeViewer'; // Import ResumeViewer
import { useNavigate } from 'react-router-dom';

const ResumeBuilder = () => {
  const navigate = useNavigate();
  const [showPreview, setShowPreview] = useState(false);
  const [resumeData, setResumeData] = useState({
    contactInformation: {},
    skills: {},
    workExperience: [],
    education: [],
    certifications: [],
    projects: [],
    achievements: [],
  });  
  const [selectedTemplate, setSelectedTemplate] = useState('template1');
  const [formSubmitted] = useState(false);

  const handleTemplateChange = (event) => {
    setSelectedTemplate(event.target.value);
  };

  useEffect(() => {
    const savedData = JSON.parse(localStorage.getItem('resumeData'));
    const savedTemplate = localStorage.getItem('selectedTemplate');

    if (savedData) {
      setResumeData(savedData);  
    }
    if (savedTemplate) {
      setSelectedTemplate(savedTemplate); 
    }
  }, []);

  const handleSaveAndPreview = (e) => {
    e.preventDefault();
    localStorage.setItem('resumeData', JSON.stringify(resumeData));
    localStorage.setItem('selectedTemplate', selectedTemplate);
    navigate('/resume-viewer'); // Navigate to ResumeViewer page
  };

    return (
      <Container maxWidth="md" style={{ backgroundColor: '#f0f0f0' }}>
        <Typography variant="h4" style={{ textAlign: 'center', margin: '20px 20px' }}>
          Create Your Resume
        </Typography>
        {showPreview ? (
          <div style={{ backgroundColor: '#f0f0f0', padding: '20px', borderRadius: '10px' }}>
            <ResumeViewer data={resumeData} />
          </div>
        ) : (
          <Formik
            initialValues={{
              contactInformation: {
                fullName: '',
                phoneNumber: '',
                email: '',
                address: '',
              },
              professionalSummary: '',
              skills: {
                technical: [''],
                soft: [''],
              },
              workExperience: [
                {
                  companyName: '',
                  location: '',
                  jobTitle: '',
                  startDate: '',
                  endDate: '',
                  accomplishments: [''],
                },
              ],
              education: [
                {
                  degree: '',
                  institution: '',
                  location: '',
                  graduationYear: '',
                  grade: '',
                },
              ],
              certifications: [
                {
                  name: '',
                  issuer: '',
                  date: '',
                },
              ],
              projects: [
                {
                  name: '',
                  description: '',
                  role: '',
                  technologies: [''],
                },
              ],
              achievements: [
                {
                  name: '',
                  issuer: '',
                  date: '',
                },
              ],
            }}
            onSubmit={(values) => {
              setResumeData(values); // Store form data
              setShowPreview(true); // Show preview
            }}
          >
            {({ values, submitForm }) => {
              if (formSubmitted) {
                return <ResumeViewer data={values} selectedTemplate={selectedTemplate} />; // Render the resume preview
              }
              return (
                <Form onSubmit={handleSaveAndPreview}>
                  {/* Contact Information */}
                  <Typography variant="h5" style={{ marginBottom: '10px' }}>Contact Information</Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <Field component={TextField} name="contactInformation.fullName" label="Full Name" variant="outlined" fullWidth />
                    </Grid>
                    <Grid item xs={6}>
                      <Field component={TextField} name="contactInformation.phoneNumber" label="Phone Number" variant="outlined" fullWidth />
                    </Grid>
                    <Grid item xs={6}>
                      <Field component={TextField} name="contactInformation.email" label="Email" variant="outlined" fullWidth />
                    </Grid>
                    <Grid item xs={12}>
                      <Field component={TextField} name="contactInformation.address" label="Address" variant="outlined" fullWidth />
                    </Grid>
                  </Grid>
  
                  {/* Professional Summary */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Describe yourself in a line</Typography>
                  <Field
                    component={TextField}
                    name="professionalSummary"
                    label="Summary"
                    variant="outlined"
                    multiline
                    rows={4}
                    fullWidth
                  />
  
                  {/* Skills */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Skills</Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={6}>
                      <FieldArray name="skills.technical">
                        {({ push, remove }) => (
                          <div>
                            {values.skills.technical.map((skill, index) => (
                              <div key={index} style={{ display: 'flex', alignItems: 'center', marginTop: '20px' }}>
                                <Field
                                  component={TextField}
                                  name={`skills.technical[${index}]`}
                                  label="Technical Skill"
                                  variant="outlined"
                                  fullWidth
                                />
                                <IconButton onClick={() => remove(index)}><Remove /></IconButton>
                                <IconButton onClick={() => push('')}><Add /></IconButton>
                              </div>
                            ))}
                          </div>
                        )}
                      </FieldArray>
                    </Grid>
  
                    <Grid item xs={6}>
                      <FieldArray name="skills.soft">
                        {({ push, remove }) => (
                          <div>
                            {values.skills.soft.map((skill, index) => (
                              <div key={index} style={{ display: 'flex', alignItems: 'center' , marginTop: '20px'}}>
                                <Field
                                  component={TextField}
                                  name={`skills.soft[${index}]`}
                                  label="Soft Skill"
                                  variant="outlined"
                                  fullWidth
                                />
                                <IconButton onClick={() => remove(index)}><Remove /></IconButton>
                                <IconButton onClick={() => push('')}><Add /></IconButton>
                              </div>
                            ))}
                          </div>
                        )}
                      </FieldArray>
                    </Grid>
                  </Grid>
  
                  {/* Work Experience */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Work Experience</Typography>
                  <FieldArray name="workExperience">
                    {({ push, remove }) => (
                      <div>
                        {values.workExperience.map((experience, index) => (
                          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                            {/* <Typography>Previous Experience</Typography> */}
                            <Grid container spacing={3}>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`workExperience[${index}].companyName`}
                                  label="Company Name"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`workExperience[${index}].location`}
                                  label="Location"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`workExperience[${index}].jobTitle`}
                                  label="Job Title"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={3}>
                                <Field
                                  component={TextField}
                                  name={`workExperience[${index}].startDate`}
                                  label="Start Date"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={3}>
                                <Field
                                  component={TextField}
                                  name={`workExperience[${index}].endDate`}
                                  label="End Date"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                            </Grid>
                            {/* <Typography>Accomplishments</Typography> */}
                            <FieldArray name={`workExperience[${index}].accomplishments`}>
                              {({ push, remove }) => (
                                <div>
                                  {experience.accomplishments.map((accomplishment, i) => (
                                    <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' , marginTop: '20px'}}>
                                      <Field
                                        component={TextField}
                                        name={`workExperience[${index}].accomplishments[${i}]`}
                                        label="Description"
                                        variant="outlined"
                                        fullWidth
                                      />
                                      <IconButton onClick={() => remove(i)}><Remove /></IconButton>
                                      <IconButton onClick={() => push('')}><Add /></IconButton>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </FieldArray>
                            {/* Remove Experience Button */}
                            {values.workExperience.length > 1 && (
                              <Button
                                onClick={() => remove(index)}
                                variant="outlined"
                                color="error"
                                style={{ marginTop: '10px' }}
                              >
                                Remove Experience
                              </Button>
                            )}
                          </div>
                        ))}
                        <Button
                          onClick={() => push({ companyName: '', location: '', jobTitle: '', startDate: '', endDate: '', accomplishments: [''] })}
                          variant="contained"
                          color="primary"
                          style={{ marginTop: '10px' }}
                        >
                          Add Experience
                        </Button>
                      </div>
                    )}
                  </FieldArray>
  
                  {/* Education */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Education</Typography>
                  <FieldArray name="education">
                    {({ push, remove }) => (
                      <div>
                        {values.education.map((edu, index) => (
                          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                            <Typography></Typography>
                            <Grid container spacing={3}>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`education[${index}].degree`}
                                  label="Degree"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`education[${index}].institution`}
                                  label="Institution"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`education[${index}].location`}
                                  label="Location"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={3}>
                                <Field
                                  component={TextField}
                                  name={`education[${index}].graduationYear`}
                                  label="Graduation Year"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={3}>
                                <Field
                                  component={TextField}
                                  name={`education[${index}].grade`}
                                  label="Grade"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                            </Grid>
                            {/* Remove Education Button */}
                            {values.education.length > 1 && (
                              <Button
                                onClick={() => remove(index)}
                                variant="outlined"
                                color="error" // Red button for "Remove Education"
                                style={{ marginTop: '10px' }}
                              >
                                Remove Education
                              </Button>
                            )}
                          </div>
                        ))}
                        <Button
                          onClick={() => push({ degree: '', institution: '', location: '', graduationYear: '', grade: '' })}
                          variant="contained"
                          color="primary"
                          style={{ marginTop: '10px' }}
                        >
                          Add Education
                        </Button>
                      </div>
                    )}
                  </FieldArray>
  
                  {/* Certifications */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Certifications</Typography>
                  <FieldArray name="certifications">
                    {({ push, remove }) => (
                      <div>
                        {values.certifications.map((cert, index) => (
                          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                            {/* <Typography>Certification {index + 1}</Typography> */}
                            <Grid container spacing={3}>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`certifications[${index}].name`}
                                  label="Certification Name"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`certifications[${index}].issuer`}
                                  label="Issuer"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`certifications[${index}].date`}
                                  label="Date"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                            </Grid>
                            {/* Remove Certification Button */}
                            {values.certifications.length > 1 && (
                              <Button
                                onClick={() => remove(index)}
                                variant="outlined"
                                color="error" // Red button for "Remove Certification"
                                style={{ marginTop: '10px' }}
                              >
                                Remove Certification
                              </Button>
                            )}
                          </div>
                        ))}
                        <Button
                          onClick={() => push({ name: '', issuer: '', date: '' })}
                          variant="contained"
                          color="primary"
                          style={{ marginTop: '10px' }}
                        >
                          Add Certification
                        </Button>
                      </div>
                    )}
                  </FieldArray>
                  {/* Projects */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Projects</Typography>
                  <FieldArray name="projects">
                    {({ push, remove }) => (
                      <div>
                        {values.projects.map((proj, index) => (
                          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                            {/* <Typography>Project {index + 1}</Typography> */}
                            <Grid container spacing={3}>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`projects[${index}].name`}
                                  label="Project Name"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`projects[${index}].description`}
                                  label="Description"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`projects[${index}].role`}
                                  label="Role"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              {/* <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`projects[${index}].technologies`}
                                  label="Technologies"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid> */}
                            </Grid>
                            {/* Remove Project Button */}
                            {values.projects.length > 1 && (
                              <Button
                                onClick={() => remove(index)}
                                variant="outlined"
                                color="error" // Red button for "Remove Project"
                                style={{ marginTop: '10px' }}
                              >
                                Remove Project
                              </Button>
                            )}
                          </div>
                        ))}
                        <Button
                          onClick={() => push({ name: '', description: '', role: '', technologies: '' })}
                          variant="contained"
                          color="primary"
                          style={{ marginTop: '10px' }}
                        >
                          Add Project
                        </Button>
                      </div>
                    )}
                  </FieldArray>
                  {/* Achievements */}
                  <Typography variant="h5" style={{ marginTop: '20px' }}>Achievements</Typography>
                  <FieldArray name="achievements">
                    {({ push, remove }) => (
                      <div>
                        {values.achievements.map((ach, index) => (
                          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                            {/* <Typography>Achievement {index + 1}</Typography> */}
                            <Grid container spacing={3}>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`achievements[${index}].name`}
                                  label="Achievement Name"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`achievements[${index}].issuer`}
                                  label="Issuer"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Field
                                  component={TextField}
                                  name={`achievements[${index}].date`}
                                  label="Date"
                                  variant="outlined"
                                  fullWidth
                                />
                              </Grid>
                            </Grid>
                            {/* Remove Achievement Button */}
                            {values.achievements.length > 1 && (
                              <Button
                                onClick={() => remove(index)}
                                variant="outlined"
                                color="error" // Red button for "Remove Achievement"
                                style={{ marginTop: '10px' }}
                              >
                                Remove Achievement
                              </Button>
                            )}
                          </div>
                        ))}
                        <Button
                          onClick={() => push({ name: '', issuer: '', date: '' })}
                          variant="contained"
                          color="primary"
                          style={{ marginTop: '10px' }}
                        >
                          Add Achievement
                        </Button>
                      </div>
                    )}
                  </FieldArray>
                  {/* Template Selection */}
                  <Typography variant="h6" style={{ marginTop: '20px' }}>Select Resume Template</Typography>
                  <FormControl fullWidth style={{ marginTop: '10px' }}>
                    <InputLabel id="template-select-label">Template</InputLabel>
                    <Select
                      labelId="template-select-label"
                      value={selectedTemplate}
                      onChange={handleTemplateChange}
                      label="Template"
                    >
                      <MenuItem value="template1">Template 1</MenuItem>
                      <MenuItem value="template2">Template 2</MenuItem>
                      <MenuItem value="template3">Template 3</MenuItem>
                    </Select>
                  </FormControl>
                  <Grid container justifyContent="center" style={{ marginTop: '20px' }}>
                    <Button variant="contained" color="primary" onClick={submitForm}>
                      Submit and Preview
                    </Button>
                  </Grid>
                </Form>
              );
            }}
          </Formik>
        )}
      </Container>
    );
  };

export default ResumeBuilder;