import { useState } from "react";
 
const SkillsSection = ({ formData, setFormData }) => {
    const [newTechnicalSkill, setNewTechnicalSkill] = useState("");
    const [newSoftSkill, setNewSoftSkill] = useState("");
  
    // Ensure skills object and arrays exist
    const technicalSkills = formData.skills?.technical || [];
    const softSkills = formData.skills?.soft || [];
  
    const addTechnicalSkill = () => {
      if (newTechnicalSkill.trim() === "") return;
      setFormData((prev) => ({
        ...prev,
        skills: {
          technical: [...(prev.skills?.technical || []), newTechnicalSkill],
          soft: [...(prev.skills?.soft || [])], // Preserve soft skills
        },
      }));
      setNewTechnicalSkill("");
    };
  
    const addSoftSkill = () => {
      if (newSoftSkill.trim() === "") return;
      setFormData((prev) => ({
        ...prev,
        skills: {
          technical: [...(prev.skills?.technical || [])], // Preserve technical skills
          soft: [...(prev.skills?.soft || []), newSoftSkill],
        },
      }));
      setNewSoftSkill("");
    };
  
    const removeTechnicalSkill = (index) => {
      const updatedSkills = [...technicalSkills];
      updatedSkills.splice(index, 1);
      setFormData((prev) => ({
        ...prev,
        skills: {
          technical: updatedSkills,
          soft: [...(prev.skills?.soft || [])], // Preserve soft skills
        },
      }));
    };
  
    const removeSoftSkill = (index) => {
      const updatedSkills = [...softSkills];
      updatedSkills.splice(index, 1);
      setFormData((prev) => ({
        ...prev,
        skills: {
          technical: [...(prev.skills?.technical || [])], // Preserve technical skills
          soft: updatedSkills,
        },
      }));
    };
  
    return (
      <div>
        <h2 className="text-xl font-semibold mb-4">Skills</h2>
        <p className="mb-4 text-gray-600">Add skills that showcase your strengths to potential employers.</p>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium mb-2">Technical Skills</h3>
            <div className="flex mb-2">
              <input
                type="text"
                value={newTechnicalSkill}
                onChange={(e) => setNewTechnicalSkill(e.target.value)}
                className="flex-grow p-2 border border-gray-300 rounded-l"
                placeholder="e.g., JavaScript, Python, Adobe Photoshop"
              />
              <button
                onClick={addTechnicalSkill}
                className="bg-blue-600 text-white px-4 py-2 rounded-r hover:bg-blue-700"
              >
                Add
              </button>
            </div>
            
            <div className="flex flex-wrap gap-2 mt-2">
              {technicalSkills.map((skill, index) => (
                <div key={index} className="bg-gray-100 px-3 py-1 rounded-full flex items-center">
                  <span>{skill}</span>
                  <button 
                    onClick={() => removeTechnicalSkill(index)}
                    className="ml-2 text-gray-500 hover:text-red-500"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium mb-2">Soft Skills</h3>
            <div className="flex mb-2">
              <input
                type="text"
                value={newSoftSkill}
                onChange={(e) => setNewSoftSkill(e.target.value)}
                className="flex-grow p-2 border border-gray-300 rounded-l"
                placeholder="e.g., Leadership, Communication, Problem-solving"
              />
              <button
                onClick={addSoftSkill}
                className="bg-blue-600 text-white px-4 py-2 rounded-r hover:bg-blue-700"
              >
                Add
              </button>
            </div>
            
            <div className="flex flex-wrap gap-2 mt-2">
              {softSkills.map((skill, index) => (
                <div key={index} className="bg-gray-100 px-3 py-1 rounded-full flex items-center">
                  <span>{skill}</span>
                  <button 
                    onClick={() => removeSoftSkill(index)}
                    className="ml-2 text-gray-500 hover:text-red-500"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
};

export default SkillsSection;