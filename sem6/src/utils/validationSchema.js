import * as Yup from 'yup';

const validationSchema = Yup.object({
  contactInformation: Yup.object({
    fullName: Yup.string().required('Full Name is required'),
    phoneNumber: Yup.string().required('Phone Number is required'),
    email: Yup.string().email('Invalid email format').required('Email is required'),
    address: Yup.string(), // Address is optional
  }),

  professionalSummary: Yup.string().required('Professional Summary is required'),

  skills: Yup.object({
    technical: Yup.array().of(Yup.string().required('Technical skill is required')).min(1, 'At least one technical skill is required'),
    soft: Yup.array().of(Yup.string().required('Soft skill is required')).min(1, 'At least one soft skill is required'),
  }),

  workExperience: Yup.array().of(
    Yup.object({
      companyName: Yup.string().required('Company Name is required'),
      location: Yup.string().required('Location is required'),
      jobTitle: Yup.string().required('Job Title is required'),
      startDate: Yup.string().required('Start Date is required'),
      endDate: Yup.string(), // End date is optional
      accomplishments: Yup.array().of(Yup.string().required('Accomplishment is required')).min(1, 'At least one accomplishment is required'),
    })
  ).min(1, 'At least one work experience is required'),

  education: Yup.array().of(
    Yup.object({
      degree: Yup.string().required('Degree is required'),
      institution: Yup.string().required('Institution is required'),
      location: Yup.string().required('Location is required'),
      graduationYear: Yup.string().required('Graduation Year is required'),
      grade: Yup.string(), // Grade is optional
    })
  ).min(1, 'At least one education entry is required'),

  certifications: Yup.array().of(
    Yup.object({
      name: Yup.string().required('Certification Name is required'),
      issuer: Yup.string().required('Issuer is required'),
      date: Yup.string().required('Date is required'),
    })
  ).min(1, 'At least one certification is required'),

  projects: Yup.array().of(
    Yup.object({
      name: Yup.string().required('Project Name is required'),
      description: Yup.string().required('Description is required'),
      role: Yup.string().required('Role is required'),
      technologies: Yup.array().of(Yup.string().required('Technology is required')).min(1, 'At least one technology is required'),
    })
  ).min(1, 'At least one project is required'),

  achievements: Yup.array().of(
    Yup.object({
      name: Yup.string().required('Achievement Name is required'),
      issuer: Yup.string().required('Issuer is required'),
      date: Yup.string().required('Date is required'),
    })
  ),
});

export default validationSchema;
