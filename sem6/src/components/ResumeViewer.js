import React from "react";
import { FaEnvelope, FaPhone, FaMapMarkerAlt } from "react-icons/fa";

const ResumeViewer = ({ resumeData }) => {
  const currentYear = new Date().getFullYear();

  if (!resumeData) {
    return <p className="text-red-500">No resume data available</p>;
  }

  return (
    <div id="resume-content" className="bg-white space-y-4 p-6">
      {/* Header/Contact Information with Profile Picture */}
      <div className="border-b border-gray-600 pb-2">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <h2 className="text-6xl font-bold text-gray-800 mb-2">
              {resumeData?.contactInformation?.fullName}
            </h2>
            {resumeData?.professionalSummary && (
              <p className="text-gray-600 mb-4 text-lg">
                {resumeData?.professionalSummary}
              </p>
            )}
          </div>
          
          {/* Profile Picture - only render if exists */}
          {resumeData?.profilePicture && (
            <div className="">
              <img 
                src={resumeData?.profilePicture} 
                alt="Profile" 
                className="w-32 h-32 flex-shrink-0 overflow-hidden border-2 border-gray-300 rounded-md"
              />
            </div>
          )}
        </div>
  
        {/* Contact information - only render if contact details exist */}
        {(resumeData?.contactInformation?.email || 
          resumeData?.contactInformation?.phoneNumber || 
          resumeData?.contactInformation?.address) && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-700 mt-4">
            {resumeData?.contactInformation?.email && (
              <p className="flex items-center">
                <FaEnvelope className="mr-2 text-blue-500" />
                {resumeData?.contactInformation?.email}
              </p>
            )}
            {resumeData?.contactInformation?.phoneNumber && (
              <p className="flex items-center">
                <FaPhone className="mr-2 text-blue-500" />
                {resumeData?.contactInformation?.phoneNumber}
              </p>
            )}
            {resumeData?.contactInformation?.address && (
              <p className="flex items-center">
                <FaMapMarkerAlt className="mr-2 text-blue-500" />
                {resumeData?.contactInformation?.address}
              </p>
            )}
          </div>
        )}
      </div>
  
      <div className="flex flex-row text-justify">
        <div className="flex w-full flex-col pr-3 border-r border-gray-600">
          {/* Education - only render if education array exists and has items */}
          {resumeData?.education && resumeData.education.length > 0 && (
            <div className="pb-2 border-b border-gray-600">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Education</h3>
              <div className="space-y-6">
                {resumeData.education.map((edu, index) => {
                  return (
                    <p key={index} className="text-gray-700">
                      <li>
                        <>
                          {edu.graduationYear > currentYear ? (
                            <>
                            Currently pursuing <strong>{edu.degree}</strong> from {edu.institution}, expected to graduate in {edu.graduationYear} with current grade of <strong>{edu.grade}</strong>.
                            </>
                          ) : edu.graduationYear < currentYear - 2 ? (
                            <>
                            Completed <strong>{edu.degree}</strong> from {edu.institution} in {edu.graduationYear} with a grade of <strong>{edu.grade}</strong>.
                            </>
                          ) : (
                            <>
                            Graduated with <strong>{edu.degree}</strong> from {edu.institution} in {edu.graduationYear} with a grade of <strong>{edu.grade}</strong>.
                            </>
                          )}
                        </>
                      </li>
                    </p>
                  );
                })}
              </div>
            </div>
          )}
  
          {/* Work Experience - only render if workExperience array exists and has items */}
          {resumeData?.workExperience && resumeData.workExperience.length > 0 && (
            <div className="py-2 border-b border-gray-600">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Work Experience</h3>
              <div className="space-y-6">
                {resumeData.workExperience.map((exp, index) => {
                  let sentence = "";
                  if (exp.endDate.toLowerCase() === "present") {
                    sentence = `${exp.jobTitle} at ${exp.companyName} (${exp.location}) since ${exp.startDate}.`;
                  } else {
                    sentence = `${exp.jobTitle} at ${exp.companyName} (${exp.location}) from ${exp.startDate} to ${exp.endDate}.`;
                  }
                  return (
                    <div key={`exp-${index}`}>
                      <p className="text-gray-700 font-bold"><li>{sentence}</li></p>
                      <p className="text-gray-700">{exp.description}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          
          {/* Certifications - only render if certifications array exists and has items */}
          {resumeData?.certifications && resumeData.certifications.length > 0 && (
            <div className="py-2">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Certifications</h3>
              <div className="space-y-6">
                {resumeData.certifications.map((cert, index) => (
                  <p key={index} className="text-gray-700">
                    <p className="font-bold">{cert.name}</p>{`issued by ${cert.issuer} on ${cert.date}.`}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="w-full flex flex-col pl-3">
          {/* Skills - only render if skills exist */}
          {resumeData?.skills && 
            ((resumeData.skills.technical && resumeData.skills.technical.length > 0) || 
             (resumeData.skills.soft && resumeData.skills.soft.length > 0)) && (
            <div className="py-2 border-b border-gray-600">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Skills</h3>
              <div className="grid grid-cols-2 gap-4">
                {resumeData.skills.technical && resumeData.skills.technical.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-700">Technical Skills</h4>
                    <ul className="list-disc list-inside text-gray-700">
                      {resumeData.skills.technical.map((skill, index) => (
                        <li key={index}>{skill}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {resumeData.skills.soft && resumeData.skills.soft.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-700">Soft Skills</h4>
                    <ul className="list-disc list-inside text-gray-700">
                      {resumeData.skills.soft.map((skill, index) => (
                        <li key={index}>{skill}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
  
          {/* Projects - only render if projects array exists and has items */}
          {resumeData?.projects && resumeData.projects.length > 0 && (
            <div className="py-2 border-b border-gray-600">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Projects</h3>
              <div className="space-y-6">
                {resumeData.projects.map((project, index) => (
                  <p key={index} className="text-gray-700">
                    <p className="font-bold">{project.name}:</p>{`${project.description}`}
                  </p>
                ))}
              </div>
            </div>
          )}
  
          {/* Achievements - only render if achievements array exists and has items */}
          {resumeData?.achievements && resumeData.achievements.length > 0 && (
            <div className="py-2">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Achievements</h3>
              <div className="space-y-6">
                {resumeData.achievements.map((ach, index) => (
                  <p key={index} className="text-gray-700">
                    <p className="font-bold">{ach.name}</p>{`issued by ${ach.issuer} on ${ach.date}.`}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResumeViewer;