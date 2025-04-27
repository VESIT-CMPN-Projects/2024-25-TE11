import React, { useState, useEffect, useRef } from "react";
import { doc, getDoc, onSnapshot } from "firebase/firestore";
import { db } from "../firebase-config";
import { FaEnvelope, FaPhone, FaMapMarkerAlt, FaPalette, FaFont, FaDownload } from "react-icons/fa";
import { toPng } from "html-to-image";
import jsPDF from "jspdf";

const ResumeBuilder = ({ userId }) => {
  const [resumeData, setResumeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedTemplate, setSelectedTemplate] = useState("modern");
  const [selectedFont, setSelectedFont] = useState("inter");
  const resumeRef = useRef(null);
  const [showComparison, setShowComparison] = useState(false);
  const [previousResumeData, setPreviousResumeData] = useState(null);

  const templates = {
    modern: {
      name: "Modern",
      className: "bg-white rounded-lg shadow-lg p-8 space-y-4"
    },
    minimal: {
      name: "Minimal",
      className: "bg-white rounded-lg shadow-lg p-8 space-y-4"
    },
    professional: {
      name: "Professional",
      className: "bg-white rounded-lg shadow-lg p-8 space-y-4"
    },
    compact: {
      name: "Compact",
      className: "bg-white rounded-lg shadow-lg p-8 space-y-4"
    }
  };

  const fonts = {
    inter: {
      name: "Inter",
      className: "font-sans"
    },
    georgia: {
      name: "Georgia",
      className: "font-serif"
    },
    monaco: {
      name: "Monaco",
      className: "font-mono"
    }
  };

  const handleCompareResume = () => {
    // Get previous resume data from local storage
    const storedPreviousResume = localStorage.getItem(`previous-resume-${userId}`);
    
    if (storedPreviousResume) {
      setPreviousResumeData(JSON.parse(storedPreviousResume));
      setShowComparison(true);
    } else {
      // Handle case where no previous resume exists
      alert("No previous version available for comparison");
    }
  };

  const handleFetchPreviousVersion = async () => {
    try {
      // Assuming you have a collection for storing previous versions
      const prevVersionRef = doc(db, "userProfileHistory", userId);
      const prevDocSnap = await getDoc(prevVersionRef);
      
      if (prevDocSnap.exists()) {
        setPreviousResumeData(prevDocSnap.data());
        setShowComparison(true);
      } else {
        alert("No previous version found");
      }
    } catch (error) {
      console.error("Error fetching previous version:", error);
    }
  };

  const currentYear = new Date().getFullYear();

  const ModernTemplate = ({ resumeData }) => {
    if (!resumeData) return <div>No resume data available</div>;
    
    return (
      <div 
        id="resume-content" 
        className="bg-white space-y-4"
      >
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
                              Currently pursuing <strong>{edu.degree}</strong> from {edu.institution}, expected to graduate in {edu.graduationYear}with current grade of <strong>{edu.grade}</strong> .
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

  const MinimalistTemplate = ( {resumeData} ) => (
    <div id="resume-content" className="bg-white">
      {/* Header Section */}
      <header className="mb-8">
        <div className="flex justify-between gap-6">
          {/* Header Text */}
          <div>
            <h1 className="text-4xl font-dark text-gray-900 mb-3">
              {resumeData?.contactInformation?.fullName}
            </h1>
            {resumeData?.professionalSummary && (
              <p className="text-gray-600 mb-4 text-lg font-light leading-relaxed">
                {resumeData?.professionalSummary}
              </p>
            )}
            <div className="flex flex-wrap gap-6 text-sm text-gray-600">
              {resumeData?.contactInformation?.email && (
                <span className="flex items-center">
                  <FaEnvelope className="mr-2" />
                  {resumeData.contactInformation.email}
                </span>
              )}
              {resumeData?.contactInformation?.phoneNumber && (
                <span className="flex items-center">
                  <FaPhone className="mr-2" />
                  {resumeData.contactInformation.phoneNumber}
                </span>
              )}
              {resumeData?.contactInformation?.address && (
                <span className="flex items-center">
                  <FaMapMarkerAlt className="mr-2" />
                  {resumeData.contactInformation.address}
                </span>
              )}
            </div>
          </div>
          {/* Profile Picture - only render if exists */}
          {resumeData?.profilePicture && (
            <div>
              <img
                src={resumeData.profilePicture}
                alt="Profile"
                className="w-32 h-32 flex-shrink-0 overflow-hidden border-2 border-gray-300 rounded-md"
              />
            </div>
          )}
        </div>
      </header>
  
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="md:col-span-2 space-y-6">
          {/* Work Experience Section */}
          {resumeData?.workExperience?.length > 0 && (
            <section>
              <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                Work Experience
              </h2>
              <div className="space-y-6">
                {resumeData.workExperience.map((exp, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex flex-row justify-between">
                      <h3 className="font-medium text-gray-900">{exp.jobTitle}</h3>
                      <p className="text-gray-600 text-sm">
                        {exp.startDate} - {exp.endDate}
                      </p>
                    </div>
                    <p className="text-gray-600 text-sm">
                      {exp.companyName} • {exp.location}
                    </p>
                    <p className="text-gray-700 text-sm leading-relaxed">
                      {exp.description}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}
  
          {/* Education Section */}
          {resumeData?.education?.length > 0 && (
            <section>
              <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                Education
              </h2>
              <div className="space-y-4">
                {resumeData.education.map((edu, index) => (
                  <div key={index} className="space-y-1">
                    <div className="flex flex-row justify-between">
                      <h3 className="font-medium text-gray-900">{edu.degree}</h3>
                      <p className="text-gray-600 text-sm">
                        Graduation Year: {edu.graduationYear}
                      </p>
                    </div>
                    <p className="text-gray-600 text-sm">{edu.institution}</p>
                    <p className="text-gray-700 text-sm">{edu.grade}</p>
                  </div>
                ))}
              </div>
            </section>
          )}
  
          {/* Projects Section */}
          {resumeData?.projects?.length > 0 && (
            <section>
              <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                Projects
              </h2>
              <div className="space-y-4">
                {resumeData.projects.map((project, index) => (
                  <div key={index} className="space-y-1">
                    <h3 className="font-medium text-gray-900">{project.name}</h3>
                    <p className="text-gray-700 text-sm">{project.description}</p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
  
        {/* Sidebar */}
        <div className="space-y-6">
          {/* Skills Section */}
          {resumeData?.skills &&
            (resumeData.skills.technical?.length > 0 || resumeData.skills.soft?.length > 0) && (
              <section>
                <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                  Skills
                </h2>
                <div className="space-y-4">
                  {resumeData.skills.technical?.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-900 mb-2">Technical</h3>
                      <div className="flex flex-wrap gap-2">
                        {resumeData.skills.technical.map((skill, index) => (
                          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-sm rounded">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {resumeData.skills.soft?.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-900 mb-2">Soft Skills</h3>
                      <div className="flex flex-wrap gap-2">
                        {resumeData.skills.soft.map((skill, index) => (
                          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-sm rounded">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </section>
            )}
  
          {/* Certifications Section */}
          {resumeData?.certifications?.length > 0 && (
            <section>
              <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                Certifications
              </h2>
              <div className="space-y-4">
                {resumeData.certifications.map((cert, index) => (
                  <div key={index} className="space-y-1">
                    <h3 className="font-medium text-gray-900">{cert.name}</h3>
                    <p className="text-gray-600 text-sm">
                      {cert.issuer} • {cert.date}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}
  
          {/* Achievements Section */}
          {resumeData?.achievements?.length > 0 && (
            <section>
              <h2 className="text-2xl font-dark text-gray-900 mb-4 pb-2 border-b border-gray-500">
                Achievements
              </h2>
              <div className="space-y-4">
                {resumeData.achievements.map((ach, index) => (
                  <div key={index} className="space-y-1">
                    <h3 className="font-medium text-gray-900">{ach.name}</h3>
                    <p className="text-gray-600 text-sm">
                      {ach.issuer} • {ach.date}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  ); 

  const CompactTemplate = ( {resumeData} ) => {
    return (
      <div className="mx-auto">
        {/* Header Section */}
        <header className="border-b-2 border-black pb-4 mb-6 flex justify-between items-start">
          {/* Contact Information */}
          <div>
            <h1 className="text-3xl font-bold">
              {resumeData?.contactInformation?.fullName}
            </h1>
            {resumeData?.professionalSummary && (
              <p className="text-sm mt-2 mb-3">{resumeData.professionalSummary}</p>
            )}
            <div className="flex gap-4 text-sm">
              {resumeData?.contactInformation?.email && <span>{resumeData.contactInformation.email}</span>}
              {resumeData?.contactInformation?.phoneNumber && <span>{resumeData.contactInformation.phoneNumber}</span>}
              {resumeData?.contactInformation?.address && <span>{resumeData.contactInformation.address}</span>}
            </div>
          </div>
  
          {/* Profile Picture */}
          {resumeData?.profilePicture && (
            <img
              src={resumeData.profilePicture}
              alt="Profile"
              className="w-24 h-24 border-2 border-gray-300 rounded-md"
            />
          )}
        </header>
  
        <main className="space-y-6">
          {/* Education Section */}
          {resumeData?.education?.length > 0 && (
            <section>
              <h2 className="text-xl font-bold mb-2">Education</h2>
              {resumeData.education.map((edu, index) => (
                <div key={index} className="flex justify-between mb-1">
                  <div className="flex flex-col">
                    <span className="font-semibold">• {edu.institution}</span>
                    <span>{edu.degree}</span>
                  </div>
                  <span>{edu.graduationYear} ({edu.grade})</span>
                </div>
              ))}
            </section>
          )}
  
          {/* Work Experience Section */}
          {resumeData?.workExperience?.length > 0 && (
            <section>
              <h2 className="text-xl font-bold mb-2">Experience</h2>
              {resumeData.workExperience.map((exp, index) => (
                <div key={index} className="mb-4">
                  <div className="flex justify-between">
                    <strong>{exp.jobTitle} at {exp.companyName}</strong>
                    <span>{exp.startDate} - {exp.endDate}</span>
                  </div>
                  <div className="text-sm text-gray-600 mb-1">{exp.location}</div>
                  <p className="text-sm">{exp.description}</p>
                </div>
              ))}
            </section>
          )}
  
          <div className="grid grid-cols-2 gap-6">
            {/* Technical Skills Section */}
            {resumeData?.skills?.technical?.length > 0 && (
              <section>
                <h2 className="text-xl font-bold mb-2">Technical Skills</h2>
                <div className="flex flex-wrap gap-2">
                  {resumeData.skills.technical.map((skill, index) => (
                    <span key={index} className="text-sm bg-gray-100 px-2 py-1 rounded">
                      {skill}
                    </span>
                  ))}
                </div>
              </section>
            )}
  
            {/* Soft Skills Section */}
            {resumeData?.skills?.soft?.length > 0 && (
              <section>
                <h2 className="text-xl font-bold mb-2">Soft Skills</h2>
                <div className="flex flex-wrap gap-2">
                  {resumeData.skills.soft.map((skill, index) => (
                    <span key={index} className="text-sm bg-gray-100 px-2 py-1 rounded">
                      {skill}
                    </span>
                  ))}
                </div>
              </section>
            )}
          </div>
  
          {/* Projects Section */}
          {resumeData?.projects?.length > 0 && (
            <section>
              <h2 className="text-xl font-bold mb-2">Projects</h2>
              {resumeData.projects.map((project, index) => (
                <div key={index} className="mb-3">
                  <div className="font-semibold">{project.name}</div>
                  <p className="text-sm">{project.description}</p>
                </div>
              ))}
            </section>
          )}
  
          <div className="grid grid-cols-2 gap-6">
            {/* Certifications Section */}
            {resumeData?.certifications?.length > 0 && (
              <section>
                <h2 className="text-xl font-bold mb-2">Certifications</h2>
                {resumeData.certifications.map((cert, index) => (
                  <div key={index} className="mb-2">
                    <div className="font-semibold">{cert.name}</div>
                    <div className="text-sm text-gray-600">
                      {cert.issuer} • {cert.date}
                    </div>
                  </div>
                ))}
              </section>
            )}
  
            {/* Achievements Section */}
            {resumeData?.achievements?.length > 0 && (
              <section>
                <h2 className="text-xl font-bold mb-2">Achievements</h2>
                {resumeData.achievements.map((achievement, index) => (
                  <div key={index} className="mb-2">
                    <div className="font-semibold">{achievement.name}</div>
                    <div className="text-sm text-gray-600">
                      {achievement.issuer} • {achievement.date}
                    </div>
                  </div>
                ))}
              </section>
            )}
          </div>
        </main>
      </div>
    );
  };  

  const ProfessionalTemplate = ( {resumeData} ) => (
    <div className="max-w-4xl mx-auto bg-white">
      {/* Header Section */}
      <header className="mb-8 bg-gray-800 text-white p-6">
        <h1 className="text-4xl font-bold mb-2">{resumeData?.contactInformation?.fullName}</h1>
        <p className="text-gray-300 mb-4">{resumeData?.professionalSummary}</p>
        <div className="flex flex-wrap gap-6 text-sm">
          <span className="flex items-center"><FaEnvelope className="mr-2" />{resumeData?.contactInformation?.email}</span>
          <span className="flex items-center"><FaPhone className="mr-2" />{resumeData?.contactInformation?.phoneNumber}</span>
          <span className="flex items-center"><FaMapMarkerAlt className="mr-2" />{resumeData?.contactInformation?.address}</span>
        </div>
      </header>
  
        <div className="flex w-full flex-col pr-3 border-r border-gray-600">
          {/* Education Section */}
          <div className="pb-2 border-b border-gray-600">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Education</h3>
            <div className="space-y-6">
              {resumeData?.education?.map((edu, index) => (
                <p key={index} className="text-gray-700">
                  <li>
                    {edu.graduationYear > currentYear ? (
                      `Currently pursuing ${edu.degree} from ${edu.institution}, expected to graduate in ${edu.graduationYear} with a current grade of ${edu.grade}.`
                    ) : edu.graduationYear < currentYear - 2 ? (
                      `Completed ${edu.degree} from ${edu.institution} in ${edu.graduationYear} with a grade of ${edu.grade}.`
                    ) : (
                      `Graduated with ${edu.degree} from ${edu.institution} in ${edu.graduationYear} with a grade of ${edu.grade}.`
                    )}
                  </li>
                </p>
              ))}
            </div>
          </div>
  
          {/* Work Experience Section */}
          <div className="py-2 border-b border-gray-600">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Work Experience</h3>
            <div className="space-y-6">
              {resumeData?.workExperience?.map((exp, index) => (
                <div key={index}>
                  <p className="text-gray-700 font-bold">
                    <li>
                      {exp.endDate.toLowerCase() === "present"
                        ? `${exp.jobTitle} at ${exp.companyName} (${exp.location}) since ${exp.startDate}.`
                        : `${exp.jobTitle} at ${exp.companyName} (${exp.location}) from ${exp.startDate} to ${exp.endDate}.`}
                    </li>
                  </p>
                  <p className="text-gray-700">{exp.description}</p>
                </div>
              ))}
            </div>
          </div>
  
          {/* Certifications Section */}
          <div className="py-2">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Certifications</h3>
            <div className="space-y-6">
              {resumeData?.certifications?.map((cert, index) => (
                <p key={index} className="text-gray-700">
                  <p className="font-bold">{cert.name}</p>{` issued by ${cert.issuer} on ${cert.date}.`}
                </p>
              ))}
            </div>
          </div>
  
        <div className="w-full flex flex-col pl-3">
          {/* Skills Section */}
          <div className="py-2 border-b border-gray-600">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Skills</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-gray-700">Technical Skills</h4>
                <ul className="list-disc list-inside text-gray-700">
                  {resumeData?.skills?.technical?.map((skill, index) => (
                    <li key={index}>{skill}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">Soft Skills</h4>
                <ul className="list-disc list-inside text-gray-700">
                  {resumeData?.skills?.soft?.map((skill, index) => (
                    <li key={index}>{skill}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
  
          {/* Projects Section */}
          <div className="py-2 border-b border-gray-600">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Projects</h3>
            <div className="space-y-6">
              {resumeData?.projects?.map((project, index) => (
                <p key={index} className="text-gray-700">
                  <p className="font-bold">{project.name}:</p>{` ${project.description}`}
                </p>
              ))}
            </div>
          </div>
  
          {/* Achievements Section */}
          <div className="py-2">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Achievements</h3>
            <div className="space-y-6">
              {resumeData?.achievements?.map((ach, index) => (
                <p key={index} className="text-gray-700">
                  <p className="font-bold">{ach.name}</p>{` issued by ${ach.issuer} on ${ach.date}.`}
                </p>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
  
  const renderTemplate = () => {
    switch (selectedTemplate) {
      case "modern":
        return <ModernTemplate resumeData={resumeData} />;
      case "minimal":
        return <MinimalistTemplate resumeData={resumeData} />;
      case "compact":
        return <CompactTemplate resumeData={resumeData} />;
      case "professional":
        return <ProfessionalTemplate resumeData={resumeData} />;
      default:
        return <ModernTemplate resumeData={resumeData} />;
    }
  };

  const renderTemplateWithData = (data) => {
    // Clone the components but pass in the specific data
    switch (selectedTemplate) {
      case "modern":
        return <ModernTemplate resumeData={data} />;
      case "minimal":
        return <MinimalistTemplate resumeData={data} />;
      case "compact":
        return <CompactTemplate resumeData={data} />;
      case "professional":
        return <ProfessionalTemplate resumeData={data} />;
      default:
        return <ModernTemplate resumeData={data} />;
    }
  };

  useEffect(() => {
    const fetchResumeData = async () => {
      try {
        const docRef = doc(db, "userProfiles", userId);
        
        // Set up real-time listener for resume data updates
        const unsubscribe = onSnapshot(docRef, (docSnap) => {
          if (docSnap.exists()) {
            const newData = docSnap.data();
            
            // Store the previous version in local storage before updating
            if (resumeData) {
              localStorage.setItem(`previous-resume-${userId}`, JSON.stringify(resumeData));
            }
            
            setResumeData(newData);
          }
        });
        
        // Initial fetch to avoid delay
        const initialDocSnap = await getDoc(docRef);
        if (initialDocSnap.exists()) {
          setResumeData(initialDocSnap.data());
        }
        
        // Clean up listener on component unmount
        return () => unsubscribe();
      } catch (error) {
        console.error("Error fetching resume data:", error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchResumeData();
  }, [userId]);

  const handleDownloadPDF = async () => {
    if (!resumeRef.current) return;
    
    try {
      const dataUrl = await toPng(resumeRef.current);
      const pdf = new jsPDF("p", "mm", "a4"); // Portrait, millimeters, A4 size
      const imgWidth = 210; // A4 width in mm
      const imgHeight = (resumeRef.current.offsetHeight * imgWidth) / resumeRef.current.offsetWidth; // Maintain aspect ratio

      pdf.addImage(dataUrl, "PNG", 0, 0, imgWidth, imgHeight);
      
      // Use the correct path for the name
      const fileName = resumeData?.contactInformation?.fullName || "Resume";
      pdf.save(`${fileName}.pdf`);
    } catch (error) {
      console.error("Error generating PDF:", error);
    }
  };

  if (loading) {
    return <div className="text-center mt-10">Loading...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8 flex flex-wrap gap-4">
        <div className="flex items-center">
          <FaPalette className="mr-2" />
          <select
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
            className="border rounded px-3 py-2"
          >
            {Object.entries(templates).map(([key, template]) => (
              <option key={key} value={key}>
                {template.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center">
          <FaFont className="mr-2" />
          <select
            value={selectedFont}
            onChange={(e) => setSelectedFont(e.target.value)}
            className="border rounded px-3 py-2"
          >
            {Object.entries(fonts).map(([key, font]) => (
              <option key={key} value={key}>
                {font.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center">
          <FaDownload className="mr-2" />
          <button
            onClick={handleDownloadPDF}
            className="border rounded px-3 py-2 bg-blue-500 text-white hover:bg-blue-600"
          >
            Download PDF
          </button>
        </div>
        
        {/* <div className="flex items-center">
          <button
            onClick={handleCompareResume}
            className="border rounded px-3 py-2 bg-purple-500 text-white hover:bg-purple-600"
          >
            Compare Previous Resume
          </button>
        </div> */}

        {/* Add toggle for comparison view if available */}
        {showComparison && (
          <div className="flex items-center">
            <button
              onClick={() => setShowComparison(false)}
              className="border rounded px-3 py-2 bg-gray-500 text-white hover:bg-gray-600"
            >
              Hide Comparison
            </button>
          </div>
        )}
      </div>
      
      {showComparison && previousResumeData ? (
        <div className="flex flex-col md:flex-row gap-4 w-full">
          {/* Previous Version */}
          <div className="md:w-1/2">
            <h3 className="text-xl font-bold mb-2 text-gray-700">Previous Version</h3>
            <div 
              className={`${templates[selectedTemplate].className} ${fonts[selectedFont].className} border border-gray-300 opacity-90`}
            >
              {renderTemplateWithData(previousResumeData)}
            </div>
          </div>
          
          {/* Current Version */}
          <div className="md:w-1/2">
            <h3 className="text-xl font-bold mb-2 text-gray-700 flex items-center">
              Current Version
              <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">Latest</span>
            </h3>
            <div 
              ref={resumeRef}
              className={`${templates[selectedTemplate].className} ${fonts[selectedFont].className} border border-green-300`}
            >
              {renderTemplate()}
            </div>
          </div>
        </div>
      ) : (
        <div 
          ref={resumeRef} 
          className={`${templates[selectedTemplate].className} ${fonts[selectedFont].className}`}
        >
          {renderTemplate()}
        </div>
      )}
    </div>
  );
};

export default ResumeBuilder;