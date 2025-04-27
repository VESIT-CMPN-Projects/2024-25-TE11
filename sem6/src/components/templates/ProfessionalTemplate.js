import React from 'react';
import { FaEnvelope, FaPhone, FaMapMarkerAlt } from "react-icons/fa";

const ProfessionalTemplate = ({ resumeData, font }) => (
  <div id="resume-content" className={`${font} max-w-4xl mx-auto bg-white shadow-lg p-8`}>
    <header className="text-center mb-8">
      <h1 className="text-4xl font-bold text-gray-800 mb-2">
        {resumeData?.contactInformation?.fullName}
      </h1>
      <p className="text-gray-600 mb-4">
        {resumeData?.professionalSummary}
      </p>
      <div className="flex justify-center gap-4 text-gray-600">
        <div className="flex items-center gap-2">
          <FaEnvelope />
          {resumeData?.contactInformation?.email}
        </div>
        <div className="flex items-center gap-2">
          <FaPhone />
          {resumeData?.contactInformation?.phoneNumber}
        </div>
        <div className="flex items-center gap-2">
          <FaMapMarkerAlt />
          {resumeData?.contactInformation?.address}
        </div>
      </div>
    </header>

    <section className="mb-8">
      <h2 className="text-2xl font-bold text-gray-800 border-b-2 border-gray-300 mb-4">
        Professional Experience
      </h2>
      {resumeData?.workExperience?.map((exp, index) => (
        <div key={index} className="mb-6">
          <h3 className="text-xl font-semibold text-gray-800">{exp.jobTitle}</h3>
          <div className="text-gray-600 mb-2">
            {exp.companyName} | {exp.location}
          </div>
          <div className="text-gray-500 italic mb-2">
            {exp.startDate} - {exp.endDate}
          </div>
          <p className="text-gray-700">{exp.description}</p>
        </div>
      ))}
    </section>

    <section className="mb-8">
      <h2 className="text-2xl font-bold text-gray-800 border-b-2 border-gray-300 mb-4">
        Skills
      </h2>
      <div className="flex flex-wrap gap-2">
        {resumeData?.skills?.technical?.map((skill, index) => (
          <span key={index} className="bg-gray-100 px-3 py-1 rounded-full text-gray-700">
            {skill}
          </span>
        ))}
      </div>
    </section>

    <section>
      <h2 className="text-2xl font-bold text-gray-800 border-b-2 border-gray-300 mb-4">
        Education
      </h2>
      {resumeData?.education?.map((edu, index) => (
        <div key={index} className="mb-4">
          <h3 className="text-xl font-semibold text-gray-800">{edu.institution}</h3>
          <div className="text-gray-600">{edu.degree}</div>
          <div className="text-gray-500">{edu.graduationYear}</div>
        </div>
      ))}
    </section>
  </div>
);

export default ProfessionalTemplate;