const ClassicTemplate = ({ resumeData, font }) => (
    <div id="resume-content" className={`bg-white p-8 shadow-lg ${font} max-w-4xl mx-auto`}>
      <header className="text-center border-b-2 border-gray-300 pb-4">
        <h1 className="text-4xl font-bold mb-2">{resumeData?.contactInformation?.fullName}</h1>
        <div className="flex justify-center gap-4 text-gray-600">
          <span>{resumeData?.contactInformation?.email}</span>
          <span>{resumeData?.contactInformation?.phoneNumber}</span>
          <span>{resumeData?.contactInformation?.address}</span>
        </div>
      </header>
      
      <main className="mt-6 space-y-6">
        <section>
          <h2 className="text-2xl font-bold border-b border-gray-300 mb-3">Experience</h2>
          {resumeData?.workExperience?.map((exp, index) => (
            <div key={index} className="mb-4">
              <div className="flex justify-between">
                <h3 className="font-bold">{exp.jobTitle}</h3>
                <span className="text-gray-600">{exp.startDate} - {exp.endDate}</span>
              </div>
              <p className="text-gray-700">{exp.companyName}, {exp.location}</p>
              <p className="mt-2">{exp.description}</p>
            </div>
          ))}
        </section>
        
        <section>
          <h2 className="text-2xl font-bold border-b border-gray-300 mb-3">Education</h2>
          {resumeData?.education?.map((edu, index) => (
            <div key={index} className="mb-4">
              <h3 className="font-bold">{edu.institution}</h3>
              <p>{edu.degree} - {edu.graduationYear}</p>
              <p className="text-gray-600">Grade: {edu.grade}</p>
            </div>
          ))}
        </section>
      </main>
    </div>
);

export default ClassicTemplate;