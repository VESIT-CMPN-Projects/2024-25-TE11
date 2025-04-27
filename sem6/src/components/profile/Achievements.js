import React, { useState } from "react";

const AchievementsSection = ({ formData, setFormData }) => {
    const [expandedIndex, setExpandedIndex] = useState(null);
  
    const handleChange = (index, field, value) => {
      const updatedAchievements = [...formData.achievements];
      updatedAchievements[index] = {
        ...updatedAchievements[index],
        [field]: value,
      };
      setFormData({
        ...formData,
        achievements: updatedAchievements,
      });
    };
  
    const addAchievement = () => {
      setFormData({
        ...formData,
        achievements: [
          ...formData.achievements,
          {
            name: "",
            issuer: "",
            date: "",
          },
        ],
      });
      setExpandedIndex(formData.achievements.length);
    };
  
    const removeAchievement = (index) => {
      const updatedAchievements = [...formData.achievements];
      updatedAchievements.splice(index, 1);
      setFormData({
        ...formData,
        achievements: updatedAchievements,
      });
      setExpandedIndex(null);
    };
  
    return (
      <div>
        <h2 className="text-xl font-semibold mb-4">Achievements</h2>
        <p className="mb-4 text-gray-600">Add awards, recognition, and other professional accomplishments.</p>
        
        <div className="space-y-6">
          {formData.achievements.map((achievement, index) => (
            <div key={index} className="border border-gray-300 rounded p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">
                  {achievement.name ? achievement.name : `Achievement ${index + 1}`}
                </h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    {expandedIndex === index ? "Collapse" : "Expand"}
                  </button>
                  <button
                    onClick={() => removeAchievement(index)}
                    className="text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                </div>
              </div>
              
              {(expandedIndex === index || formData.achievements.length === 1) && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Achievement Name</label>
                    <input
                      type="text"
                      value={achievement.name}
                      onChange={(e) => handleChange(index, "name", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      placeholder="e.g., Employee of the Year, Outstanding Innovation Award"
                    />
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Issuing Organization</label>
                      <input
                        type="text"
                        value={achievement.issuer}
                        onChange={(e) => handleChange(index, "issuer", e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder="e.g., Company Name, Professional Organization"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
                      <input
                        type="month"
                        value={achievement.date}
                        onChange={(e) => handleChange(index, "date", e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          <button
            onClick={addAchievement}
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 w-full"
          >
            + Add Another Achievement
          </button>
        </div>
      </div>
    );
  };

export default AchievementsSection;