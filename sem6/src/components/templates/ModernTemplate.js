import { FaEnvelope, FaPhone, FaMapMarkerAlt } from "react-icons/fa";

const ModernTemplate = ({ resumeData }) => {
    const currentYear = new Date().getFullYear();
    return (
        <div 
        id="resume-content" 
        className="bg-white rounded-lg shadow-lg p-8 space-y-4 border border-gray-200"
        >
        {/* Header/Contact Information */}
        <div className="border-b border-gray-600 pb-2">
          <h2 className="text-6xl font-bold text-gray-800 mb-2">
            {resumeData?.contactInformation?.fullName}
          </h2>
          <p className="text-gray-600 mb-4 text-lg">
            {resumeData?.professionalSummary}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-700">
            <p className="flex items-center">
              <FaEnvelope className="mr-2 text-blue-500" />
              {resumeData?.contactInformation?.email}
            </p>
            <p className="flex items-center">
              <FaPhone className="mr-2 text-blue-500" />
              {resumeData?.contactInformation?.phoneNumber}
            </p>
            <p className="flex items-center">
              <FaMapMarkerAlt className="mr-2 text-blue-500" />
              {resumeData?.contactInformation?.address}
            </p>
          </div>
        </div>

        {/* Education */}
        <div className="flex flex-row text-justify">
          <div className="flex w-full flex-col pr-3 border-r border-gray-600">
            <div className="pb-2 border-b border-gray-600">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Education</h3>
              <div className="space-y-6">
                {resumeData?.education?.map((edu, index) => {
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

        {/* Work Experience */}
        <div className="py-2 border-b border-gray-600">
          <h3 className="text-2xl font-bold text-gray-800 mb-2">Work Experience</h3>
          <div className="space-y-6">
            {resumeData?.workExperience?.map((exp, index) => {
              let sentence = "";
              if (exp.endDate.toLowerCase() === "present") {
                sentence = `${exp.jobTitle} at ${exp.companyName} (${exp.location}) since ${exp.startDate}.`;
              } else {
                sentence = `${exp.jobTitle} at ${exp.companyName} (${exp.location}) from ${exp.startDate} to ${exp.endDate}.`;
              }
              return (
                <div>
                <p key={index} className="text-gray-700 font-bold"><li>{sentence}</li></p>
                <p key={index} className="text-gray-700">{exp.description}</p>
                </div>
              );
            })}
          </div>
        </div>
        {/* Certifications */}
          <div className="py-2">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Certifications</h3>
            <div className="space-y-6">
              {resumeData?.certifications?.map((cert, index) => (
                <p key={index} className="text-gray-700">
                  <p className="font-bold">{cert.name}</p>{`issued by ${cert.issuer} on ${cert.date}.`}
                </p>
              ))}
            </div>
          </div>
        </div>
        <div className="w-full flex flex-col pl-3">
        {/* Skills */}
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

        {/* Projects */}
        <div className="py-2 border-b border-gray-600">
          <h3 className="text-2xl font-bold text-gray-800 mb-2">Projects</h3>
          <div className="space-y-6">
            {resumeData?.projects?.map((project, index) => (
              <p key={index} className="text-gray-700">
                <p className="font-bold">{project.name}:</p>{`${project.description}`}
              </p>
            ))}
          </div>
        </div>

        {/*Achievements*/}
        <div className="py-2">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Achievements</h3>
            <div className="space-y-6">
              {resumeData?.achievements?.map((ach, index) => (
                <p key={index} className="text-gray-700">
                  <p className="font-bold">{ach.name}</p>{`issued by ${ach.issuer} on ${ach.date}.`}
                </p>
              ))}
            </div>
        </div>
        
        </div>
        </div>
      </div>
    );
}

export default ModernTemplate;