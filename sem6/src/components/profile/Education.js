import { useState } from "react";

const EducationSection = ({ formData, setFormData }) => {
  const [expandedIndex, setExpandedIndex] = useState(null);

  const handleChange = (index, field, value) => {
    const updatedEducation = [...formData.education];
    updatedEducation[index] = {
      ...updatedEducation[index],
      [field]: value,
    };
    setFormData({
      ...formData,
      education: updatedEducation,
    });
  };

  const addEducation = () => {
    setFormData({
      ...formData,
      education: [
        ...formData.education,
        {
          degree: "",
          institution: "",
          location: "",
          graduationYear: "",
          grade: "",
        },
      ],
    });
    setExpandedIndex(formData.education.length);
  };

  const removeEducation = (index) => {
    const updatedEducation = [...formData.education];
    updatedEducation.splice(index, 1);
    setFormData({
      ...formData,
      education: updatedEducation,
    });
    setExpandedIndex(null);
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Education</h2>
      <p className="mb-4 text-gray-600">Add your educational background, starting with your highest level of education.</p>
      
      <div className="space-y-6">
        {formData.education.map((edu, index) => (
          <div key={index} className="border border-gray-300 rounded p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium">
                {edu.degree ? edu.degree : ``}
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                  className="text-blue-600 hover:text-blue-800"
                >
                  {expandedIndex === index ? "Collapse" : "Expand"}
                </button>
                <button
                  onClick={() => removeEducation(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  Remove
                </button>
              </div>
            </div>
            
            {(expandedIndex === index || formData.education.length === 1) && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Degree</label>
                  <input
                    type="text"
                    value={edu.degree}
                    onChange={(e) => handleChange(index, "degree", e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="e.g., Bachelor of Science in Computer Science"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Institution</label>
                    <input
                      type="text"
                      value={edu.institution}
                      onChange={(e) => handleChange(index, "institution", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                    <input
                      type="text"
                      value={edu.location}
                      onChange={(e) => handleChange(index, "location", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Graduation Year</label>
                    <input
                      type="text"
                      value={edu.graduationYear}
                      onChange={(e) => handleChange(index, "graduationYear", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      placeholder="e.g., 2023"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Grade/GPA/Percentage</label>
                    <input
                      type="text"
                      value={edu.grade}
                      onChange={(e) => handleChange(index, "grade", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      placeholder="e.g., 3.8/4.0"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
        
        <button
          onClick={addEducation}
          className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 w-full"
        >
          + Add Education
        </button>
      </div>
    </div>
  );
};

export default EducationSection;