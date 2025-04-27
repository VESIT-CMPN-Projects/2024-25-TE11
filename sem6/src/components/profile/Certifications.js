import React, { useState } from "react";

const CertificationsSection = ({ formData, setFormData, userId }) => {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [imageUploading, setImageUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (index, field, value) => {
    const updatedCertifications = [...formData.certifications];
    updatedCertifications[index] = {
      ...updatedCertifications[index],
      [field]: value,
    };
    setFormData({
      ...formData,
      certifications: updatedCertifications,
    });
  };

  const handleSkillsChange = (index, skills) => {
    // Split by commas and trim each skill
    const skillsArray = skills.split(",").map(skill => skill.trim()).filter(Boolean);
    
    const updatedCertifications = [...formData.certifications];
    updatedCertifications[index] = {
      ...updatedCertifications[index],
      skills: skillsArray,
    };
    setFormData({
      ...formData,
      certifications: updatedCertifications,
    });
  };

  const addCertification = () => {
    setFormData({
      ...formData,
      certifications: [
        ...formData.certifications,
        {
          name: "",
          issuer: "",
          date: "",
          imageUrl: "",
          skills: [],
        },
      ],
    });
    setExpandedIndex(formData.certifications.length);
  };

  const removeCertification = (index) => {
    const updatedCertifications = [...formData.certifications];
    updatedCertifications.splice(index, 1);
    setFormData({
      ...formData,
      certifications: updatedCertifications,
    });
    setExpandedIndex(null);
  };

  const handleImageUpload = async (index, e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please upload a valid image file (JPEG, PNG, or GIF)');
      return;
    }

    setImageUploading(true);
    setError(null);

    try {
      // Create FormData to send to Cloudinary
      const formData = new FormData();
      formData.append('file', file);
      formData.append('upload_preset', 'profile_pictures');
      formData.append('folder', `user-profiles/${userId}/certifications`);

      // Call Cloudinary API
      const response = await fetch(
        `https://api.cloudinary.com/v1_1/dihawgvdz/image/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error('Failed to upload image');
      }

      const data = await response.json();
      
      // Update the certification image URL
      handleChange(index, "imageUrl", data.secure_url);

    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Failed to upload image. Please try again.');
    } finally {
      setImageUploading(false);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Certifications</h2>
      <p className="mb-4 text-gray-600">Add professional certifications or licenses you've earned.</p>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      
      <div className="space-y-6">
        {formData.certifications.map((cert, index) => (
          <div key={index} className="border border-gray-300 rounded p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium">
                {cert.name ? cert.name : "New Certification"}
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                  className="text-blue-600 hover:text-blue-800"
                >
                  {expandedIndex === index ? "Collapse" : "Expand"}
                </button>
                <button
                  onClick={() => removeCertification(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  Remove
                </button>
              </div>
            </div>
            
            {(expandedIndex === index || formData.certifications.length === 1) && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Certification Name</label>
                  <input
                    type="text"
                    value={cert.name}
                    onChange={(e) => handleChange(index, "name", e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="e.g., AWS Certified Solutions Architect"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Issuing Organization</label>
                    <input
                      type="text"
                      value={cert.issuer}
                      onChange={(e) => handleChange(index, "issuer", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                      placeholder="e.g., Amazon Web Services"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Date Issued</label>
                    <input
                      type="month"
                      value={cert.date}
                      onChange={(e) => handleChange(index, "date", e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Skills Learned</label>
                  <input
                    type="text"
                    value={cert.skills ? cert.skills.join(", ") : ""}
                    onChange={(e) => handleSkillsChange(index, e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="e.g., Cloud Architecture, Security, Networking (separate with commas)"
                  />
                  <p className="text-xs text-gray-500 mt-1">Separate skills with commas</p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Certification Image</label>
                  <div className="mt-1 flex items-center">
                    {cert.imageUrl ? (
                      <div className="relative">
                        <img 
                          src={cert.imageUrl} 
                          alt={cert.name} 
                          className="h-32 w-auto object-contain border border-gray-300 rounded"
                        />
                        <button
                          onClick={() => handleChange(index, "imageUrl", "")}
                          className="absolute top-0 right-0 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center -mt-2 -mr-2"
                        >
                          ×
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center">
                        <input
                          type="file"
                          id={`certification-image-${index}`}
                          className="hidden"
                          accept="image/*"
                          onChange={(e) => handleImageUpload(index, e)}
                        />
                        <label
                          htmlFor={`certification-image-${index}`}
                          className="cursor-pointer px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                        >
                          {imageUploading ? "Uploading..." : "Upload Image"}
                        </label>
                        <p className="text-xs text-gray-500 mt-1">Upload your certification image</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
        
        <button
          onClick={addCertification}
          className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 w-full"
        >
          + Add Certification
        </button>
      </div>
    </div>
  );
};

export default CertificationsSection;