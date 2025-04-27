import { useState } from "react";

const ProjectsSection = ({ formData, setFormData }) => {
    const [expandedIndex, setExpandedIndex] = useState(null);
  
    const handleChange = (index, field, value) => {
      const updatedProjects = [...formData.projects];
      updatedProjects[index] = {
        ...updatedProjects[index],
        [field]: value,
      };
      setFormData({
        ...formData,
        projects: updatedProjects,
      });
    };
  
    const addProject = () => {
      setFormData({
        ...formData,
        projects: [
          ...formData.projects,
          {
            name: "",
            description: "",
            role: "",
            technologies: [""],
          },
        ],
      });
      setExpandedIndex(formData.projects.length);
    };
  
    const removeProject = (index) => {
      const updatedProjects = [...formData.projects];
      updatedProjects.splice(index, 1);
      setFormData({
        ...formData,
        projects: updatedProjects,
      });
      setExpandedIndex(null);
    };
  
    return (
      <div>
        <h2 className="text-xl font-semibold mb-4">Projects</h2>
        <p className="mb-4 text-gray-600">Add notable projects that demonstrate your skills and experience.</p>
        
        <div className="space-y-6">
          {formData.projects.map((project, index) => (
            <div key={index} className="border border-gray-300 rounded p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">
                  {project.name ? project.name : ``}
                </h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    {expandedIndex === index ? "Collapse" : "Expand"}
                  </button>
                  <button
                    onClick={() => removeProject(index)}
                    className="text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                </div>
              </div>
              
              {(expandedIndex === index || formData.projects.length === 1) && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                    <input
                      type="text"
                      value={project.name}
                      onChange={(e) => handleChange(index, "name", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Your Role</label>
                    <input
                      type="text"
                      value={project.role}
                      onChange={(e) => handleChange(index, "role", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      placeholder="e.g., Lead Developer, UX Designer"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Project Description</label>
                    <textarea
                      value={project.description}
                      onChange={(e) => handleChange(index, "description", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      rows="3"
                      placeholder="Describe the project, your contributions, and its impact"
                    ></textarea>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          <button
            onClick={addProject}
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 w-full"
          >
            + Add Project
          </button>
        </div>
      </div>
    );
  };

export default ProjectsSection;