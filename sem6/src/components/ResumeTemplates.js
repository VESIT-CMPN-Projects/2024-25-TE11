import React from 'react';
import PropTypes from 'prop-types';

const ResumeTemplate1 = ({ data }) => {
  return (
    <div className="max-w-5xl mx-auto p-8 bg-white shadow-lg">
      {/* Header Section */}
      <header className="border-b-2 border-gray-300 pb-6 mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">{data.contactInformation.fullName}</h1>
        <div className="flex flex-wrap gap-4 text-gray-600 mb-3">
          <p>{data.contactInformation.email}</p>
          <span>•</span>
          <p>{data.contactInformation.phoneNumber}</p>
          <span>•</span>
          <p>{data.contactInformation.address}</p>
        </div>
        <p className="text-lg text-gray-700 leading-relaxed">{data.professionalSummary}</p>
      </header>

      <div className="flex flex-col md:flex-row gap-8">
        {/* Left Sidebar - 1/3 width */}
        <div className="md:w-1/3 space-y-8">
          {/* Skills Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Skills
            </h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Technical Skills</h3>
                <p className="text-gray-600 leading-relaxed">
                  {data.skills.technical.join(', ')}
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Soft Skills</h3>
                <p className="text-gray-600 leading-relaxed">
                  {data.skills.soft.join(', ')}
                </p>
              </div>
            </div>
          </section>

          {/* Education Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Education
            </h2>
            <div className="space-y-4">
              {data.education.map((entry, index) => (
                <div key={index} className="text-gray-600">
                  Graduated with a {entry.degree} from {entry.institution} in {entry.graduationYear}.
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Main Content - 2/3 width */}
        <div className="md:w-2/3 space-y-8">
          {/* Work Experience Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Professional Experience
            </h2>
            <div className="space-y-6">
              {data.workExperience.map((exp, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between items-baseline">
                    <h3 className="text-lg font-semibold text-gray-800">{exp.jobTitle}</h3>
                    <p className="text-gray-500 text-sm">{exp.startDate} - {exp.endDate}</p>
                  </div>
                  <p className="text-gray-700 font-medium mb-2">{exp.companyName} • {exp.location}</p>
                  <div className="text-gray-600 space-y-2">
                    {exp.accomplishments.map((accomplishment, index) => (
                      <p key={index} className="leading-relaxed">
                        {accomplishment}
                      </p>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Certifications Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Professional Certifications
            </h2>
            <div className="space-y-2">
              {data.certifications.map((cert, index) => (
                <p key={index} className="text-gray-600 leading-relaxed">
                  Obtained {cert.name} certification from {cert.issuer} in {cert.date}.
                </p>
              ))}
            </div>
          </section>

          {/* Projects Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Key Projects
            </h2>
            <div className="space-y-4">
              {data.projects.map((project, index) => (
                <div key={index} className="text-gray-600 leading-relaxed">
                  <p>
                    Served as {project.role} for {project.name}, {project.description.toLowerCase()} 
                    Using {project.technologies}, successfully delivered project objectives and 
                    maintained high standards throughout the development lifecycle.
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* Achievements Section */}
          <section>
            <h2 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2 mb-4">
              Notable Achievements
            </h2>
            <div className="space-y-2">
              {data.achievements.map((achievement, index) => (
                <p key={index} className="text-gray-600 leading-relaxed">
                  Received {achievement.name} from {achievement.issuer} in {achievement.date} in recognition 
                  of outstanding contributions and exceptional performance.
                </p>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

// Prop Types Validation
ResumeTemplate1.propTypes = {
  data: PropTypes.shape({
    contactInformation: PropTypes.shape({
      fullName: PropTypes.string.isRequired,
      email: PropTypes.string.isRequired,
      phoneNumber: PropTypes.string.isRequired,
      address: PropTypes.string.isRequired,
    }).isRequired,
    professionalSummary: PropTypes.string.isRequired,
    skills: PropTypes.shape({
      technical: PropTypes.arrayOf(PropTypes.string).isRequired,
      soft: PropTypes.arrayOf(PropTypes.string).isRequired,
    }).isRequired,
    education: PropTypes.arrayOf(
      PropTypes.shape({
        degree: PropTypes.string.isRequired,
        institution: PropTypes.string.isRequired,
        graduationYear: PropTypes.string.isRequired,
      })
    ).isRequired,
    certifications: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        issuer: PropTypes.string.isRequired,
        date: PropTypes.string.isRequired,
      })
    ).isRequired,
    workExperience: PropTypes.arrayOf(
      PropTypes.shape({
        companyName: PropTypes.string.isRequired,
        jobTitle: PropTypes.string.isRequired,
        location: PropTypes.string.isRequired,
        startDate: PropTypes.string.isRequired,
        endDate: PropTypes.string.isRequired,
        accomplishments: PropTypes.arrayOf(PropTypes.string).isRequired,
      })
    ).isRequired,
    projects: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        description: PropTypes.string.isRequired,
        role: PropTypes.string.isRequired,
        technologies: PropTypes.arrayOf(PropTypes.string).isRequired,
      })
    ).isRequired,
    achievements: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        issuer: PropTypes.string.isRequired,
        date: PropTypes.string.isRequired,
      })
    ).isRequired,
  }).isRequired,
};

const ResumeTemplate2 = ({ data }) => {
  return (
    <div className="max-w-3xl mx-auto p-8 bg-gray-50 shadow-lg">
      {/* Header Section */}
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">{data.contactInformation.fullName}</h1>
        <p className="text-lg text-gray-600 mt-1">{data.professionalSummary}</p>
        <div className="mt-4">
          <p className="text-gray-600">Email: {data.contactInformation.email}</p>
          <p className="text-gray-600">Phone: {data.contactInformation.phoneNumber}</p>
          <p className="text-gray-600">Address: {data.contactInformation.address}</p>
        </div>
      </header>

      {/* Skills Section */}
      <section className="mb-8">
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Skills</h2>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="text-xl font-medium text-gray-700">Technical Skills</h3>
            <ul className="list-disc pl-6">
              {data.skills.technical.map((skill, index) => (
                <li key={index} className="text-gray-600">{skill}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-medium text-gray-700">Soft Skills</h3>
            <ul className="list-disc pl-6">
              {data.skills.soft.map((skill, index) => (
                <li key={index} className="text-gray-600">{skill}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* Work Experience Section */}
      <section className="mb-8">
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Work Experience</h2>
        {data.workExperience.map((exp, index) => (
          <div key={index} className="mb-6">
            <h3 className="text-2xl font-semibold text-gray-800">{exp.companyName} - {exp.jobTitle}</h3>
            <p className="text-gray-600 italic">{exp.location} | {exp.startDate} - {exp.endDate}</p>
            <ul className="list-disc pl-6 mt-2">
              {exp.accomplishments.map((accomplishment, index) => (
                <li key={index} className="text-gray-600">{accomplishment}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      {/* Education Section */}
      <section className="mb-8">
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Education</h2>
        <ul className="list-none">
          {data.education.map((entry, index) => (
            <li key={index} className="text-gray-600">
              {entry.degree} from {entry.institution} ({entry.graduationYear})
            </li>
          ))}
        </ul>
      </section>

      {/* Certifications Section */}
      <section className="mb-8">
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Certifications</h2>
        <ul className="list-none">
          {data.certifications.map((cert, index) => (
            <li key={index} className="text-gray-600">
              <strong>{cert.name}</strong> - {cert.issuer} ({cert.date})
            </li>
          ))}
        </ul>
      </section>

      {/* Projects Section */}
      <section className="mb-8">
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Projects</h2>
        {data.projects.map((project, index) => (
          <div key={index} className="mb-6">
            <h3 className="text-2xl font-semibold text-gray-800">{project.name}</h3>
            <p className="text-gray-600">{project.description}</p>
            <p className="text-gray-600"><strong>Role:</strong> {project.role}</p>
            <p className="text-gray-600"><strong>Technologies:</strong> {project.technologies.join(', ')}</p>
          </div>
        ))}
      </section>

      {/* Achievements Section */}
      <section>
        <h2 className="text-3xl font-semibold text-gray-700 mb-4">Achievements</h2>
        <ul className="list-none">
          {data.achievements.map((achievement, index) => (
            <li key={index} className="text-gray-600">
              <strong>{achievement.name}</strong> - {achievement.issuer} ({achievement.date})
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export default ResumeTemplate2;

const ResumeTemplate3 = ({ data }) => {
  return (
    <div className="max-w-3xl mx-auto p-8 bg-gray-50 shadow-xl">
      {/* Header Section */}
      <header className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-800">{data.contactInformation.fullName}</h1>
        <p className="text-lg text-gray-600 mt-2">{data.contactInformation.email} | {data.contactInformation.phoneNumber}</p>
        <p className="text-lg text-gray-600">{data.contactInformation.address}</p>
      </header>

      {/* Skills Section */}
      <section className="grid grid-cols-2 gap-12 mb-12">
        <div>
          <h2 className="text-3xl font-semibold text-gray-800 mb-4">Technical Skills</h2>
          <ul className="list-disc pl-6">
            {data.skills.technical.map((skill, index) => (
              <li key={index} className="text-gray-600">{skill}</li>
            ))}
          </ul>
        </div>
        <div>
          <h2 className="text-3xl font-semibold text-gray-800 mb-4">Soft Skills</h2>
          <ul className="list-disc pl-6">
            {data.skills.soft.map((skill, index) => (
              <li key={index} className="text-gray-600">{skill}</li>
            ))}
          </ul>
        </div>
      </section>

      {/* Work Experience Section */}
      <section className="mb-12">
        <h2 className="text-3xl font-semibold text-gray-800 mb-6">Work Experience</h2>
        {data.workExperience.map((exp, index) => (
          <div key={index} className="mb-8">
            <h3 className="text-2xl font-semibold text-gray-800">{exp.companyName} - {exp.jobTitle}</h3>
            <p className="text-gray-600">{exp.location} | {exp.startDate} - {exp.endDate}</p>
            <ul className="list-disc pl-6 mt-2">
              {exp.accomplishments.map((accomplishment, index) => (
                <li key={index} className="text-gray-600">{accomplishment}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      {/* Education Section */}
      <section className="mb-12">
        <h2 className="text-3xl font-semibold text-gray-800 mb-6">Education</h2>
        <ul className="list-none">
          {data.education.map((entry, index) => (
            <li key={index} className="text-gray-600">
              <strong>{entry.degree}</strong> - {entry.institution} ({entry.graduationYear})
            </li>
          ))}
        </ul>
      </section>

      {/* Certifications Section */}
      <section>
        <h2 className="text-3xl font-semibold text-gray-800 mb-6">Certifications</h2>
        <ul className="list-none">
          {data.certifications.map((cert, index) => (
            <li key={index} className="text-gray-600">
              <strong>{cert.name}</strong> - {cert.issuer} ({cert.date})
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export { ResumeTemplate1, ResumeTemplate2, ResumeTemplate3 };