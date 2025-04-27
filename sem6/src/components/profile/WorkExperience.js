import { useState } from "react";

const WorkExperienceSection = ({ formData, setFormData }) => {
  const [expandedIndex, setExpandedIndex] = useState(null);

  const handleChange = (index, field, value) => {
    const updatedExperience = [...formData.workExperience];
    updatedExperience[index] = {
      ...updatedExperience[index],
      [field]: value,
    };
    setFormData({
      ...formData,
      workExperience: updatedExperience,
    });
  };

  const handleDescriptionChange = (expIndex, descIndex, value) => {
    const updatedExperience = [...formData.workExperience];
    const currentDesc = [...updatedExperience[expIndex].description];
    currentDesc[descIndex] = value;
    updatedExperience[expIndex] = {
      ...updatedExperience[expIndex],
      description: currentDesc,
    };
    setFormData({
      ...formData,
      workExperience: updatedExperience,
    });
  };

  const removeDescription = (expIndex, descIndex) => {
    const updatedExperience = [...formData.workExperience];
    const currentDesc = [...updatedExperience[expIndex].description];
    currentDesc.splice(descIndex, 1);
    updatedExperience[expIndex] = {
      ...updatedExperience[expIndex],
      description: currentDesc,
    };
    setFormData({
      ...formData,
      workExperience: updatedExperience,
    });
  };

  const addExperience = () => {
    setFormData({
      ...formData,
      workExperience: [
        ...formData.workExperience,
        {
          companyName: "",
          location: "",
          jobTitle: "",
          startDate: "",
          endDate: "",
          description: [""],
        },
      ],
    });
    setExpandedIndex(formData.workExperience.length);
  };

  const removeExperience = (index) => {
    const updatedExperience = [...formData.workExperience];
    updatedExperience.splice(index, 1);
    setFormData({
      ...formData,
      workExperience: updatedExperience,
    });
    setExpandedIndex(null);
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Work Experience</h2>
      <p className="mb-4 text-gray-600">Add your work history, starting with your most recent position.</p>
      
      <div className="space-y-6">
        {formData.workExperience.map((exp, index) => (
          <div key={index} className="border border-gray-300 rounded p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium">
                {exp.companyName ? exp.companyName : ``}
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                  className="text-blue-600 hover:text-blue-800"
                >
                  {expandedIndex === index ? "Collapse" : "Expand"}
                </button>
                <button
                  onClick={() => removeExperience(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  Remove
                </button>
              </div>
            </div>
            
            {(expandedIndex === index || formData.workExperience.length === 1) && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Company Name</label>
                    <input
                      type="text"
                      value={exp.companyName}
                      onChange={(e) => handleChange(index, "companyName", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                    <input
                      type="text"
                      value={exp.location}
                      onChange={(e) => handleChange(index, "location", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Job Title</label>
                  <input
                    type="text"
                    value={exp.jobTitle}
                    onChange={(e) => handleChange(index, "jobTitle", e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
                    <input
                      type="month"
                      value={exp.startDate}
                      onChange={(e) => handleChange(index, "startDate", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
                    <input
                      type="month"
                      value={exp.endDate}
                      onChange={(e) => handleChange(index, "endDate", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-sm font-medium text-gray-700">Description</label>
                  </div>
                  
                  {exp.description.map((desc, descIndex) => (
                    <div key={descIndex} className="flex items-center mb-2">
                      <input
                        type="text"
                        value={desc}
                        onChange={(e) => handleDescriptionChange(index, descIndex, e.target.value)}
                        className="flex-grow p-2 border border-gray-300 rounded"
                      />
                      {exp.description.length > 1 && (
                        <button
                          onClick={() => removeDescription(index, descIndex)}
                          className="ml-2 text-gray-500 hover:text-red-500"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
        
        <button
          onClick={addExperience}
          className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 w-full"
        >
          + Add Position
        </button>
      </div>
    </div>
  );
};

export default WorkExperienceSection;