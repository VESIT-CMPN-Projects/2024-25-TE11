const MinimalistTemplate = ({ resumeData, font }) => (
    <div id="resume-content" className={`bg-white p-8 ${font} max-w-4xl mx-auto`}>
      <header className="mb-8">
        <h1 className="text-3xl font-light mb-2">{resumeData?.contactInformation?.fullName}</h1>
        <div className="text-sm text-gray-600 space-y-1">
          <p>{resumeData?.contactInformation?.email}</p>
          <p>{resumeData?.contactInformation?.phoneNumber}</p>
          <p>{resumeData?.contactInformation?.address}</p>
        </div>
      </header>
  
      <main className="space-y-8">
        <section>
          <h2 className="text-xl font-light mb-4 uppercase tracking-wider">Experience</h2>
          {resumeData?.workExperience?.map((exp, index) => (
            <div key={index} className="mb-6">
              <div className="flex justify-between items-baseline">
                <h3 className="font-medium">{exp.jobTitle}</h3>
                <span className="text-sm text-gray-600">{exp.startDate} - {exp.endDate}</span>
              </div>
              <p className="text-sm text-gray-600 mb-2">{exp.companyName}, {exp.location}</p>
              <p className="text-sm">{exp.description}</p>
            </div>
          ))}
        </section>
  
        <section>
          <h2 className="text-xl font-light mb-4 uppercase tracking-wider">Skills</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-medium mb-2">Technical</h3>
              <div className="flex flex-wrap gap-2">
                {resumeData?.skills?.technical?.map((skill, index) => (
                  <span key={index} className="text-sm px-2 py-1 bg-gray-100 rounded">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium mb-2">Soft Skills</h3>
              <div className="flex flex-wrap gap-2">
                {resumeData?.skills?.soft?.map((skill, index) => (
                  <span key={index} className="text-sm px-2 py-1 bg-gray-100 rounded">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );

  export default MinimalistTemplate;