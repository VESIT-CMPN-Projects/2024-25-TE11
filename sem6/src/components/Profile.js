import React, { useState, useEffect } from "react";
import { doc, setDoc, getDoc } from "firebase/firestore";
import { db } from "../firebase-config";

const ProfilePicture = ({ formData, isEditing, imageUploading, handleImageUpload, error, setFormData }) => {
  return (
    <div className="mb-6">
      <h3 className="text-lg font-medium mb-2">Profile Picture</h3>
      <div className="flex items-center space-x-4">
        <div className="w-32 h-40 relative overflow-hidden border-2 border-gray-300">
          {formData.profilePicture ? (
            <img 
              src={formData.profilePicture} 
              alt="Profile" 
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-gray-200 flex items-center justify-center">
              <span className="text-gray-400">No Image</span>
            </div>
          )}
        </div>
        
        {isEditing && (
          <div className="flex flex-col">
            <label 
              htmlFor="profile-upload" 
              className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded cursor-pointer"
            >
              {imageUploading ? 'Uploading...' : 'Upload Photo'}
            </label>
            <input
              id="profile-upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              disabled={imageUploading}
            />
            {formData.profilePicture && (
              <button
                type="button"
                onClick={() => setFormData(prev => ({ ...prev, profilePicture: "" }))}
                className="mt-2 text-red-500 text-sm hover:text-red-700"
              >
                Remove Photo
              </button>
            )}
          </div>
        )}
      </div>
      {error && <p className="text-red-500 mt-2">{error}</p>}
    </div>
  );
};

const ProfilePage = ({ userId }) => {
  const educationorder = [
    "degree",
    "institution",
    "location",
    "graduationYear",
    "grade",
  ];
  const workExperienceorder = [
    "companyName",
    "jobTitle",
    "startDate",
    "endDate",
    "location",
    "description",
  ];
  const projectorder = [
    "name", 
    "description", 
    "role", 
    "technologies"
  ];
  const [formData, setFormData] = useState({
    profilePicture: "",
    contactInformation: {
      fullName: "",
      phoneNumber: "",
      email: "",
      address: "",
    },
    professionalSummary: "",
    skills: {
      technical: [""],
      soft: [""],
    },
    workExperience: [
      {
        companyName: "",
        location: "",
        jobTitle: "",
        startDate: "",
        endDate: "",
        description: [""],
      },
    ],
    education: [
      {
        degree: "",
        institution: "",
        location: "",
        graduationYear: "",
        grade: "",
      },
    ],
    certifications: [
      {
        name: "",
        issuer: "",
        date: "",
      },
    ],
    projects: [
      {
        name: "",
        description: "",
        role: "",
        technologies: [""],
      },
    ],
    achievements: [
      {
        name: "",
        issuer: "",
        date: "",
      },
    ],
  });

  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageUploading, setImageUploading] = useState(false);
  const [newTechnicalSkill, setNewTechnicalSkill] = useState("");
  const [newSoftSkill, setNewSoftSkill] = useState("");


  useEffect(() => {
    if (!userId) return;
  
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
  
      try {
        const profileRef = doc(db, "userProfiles", userId);
        const profileSnapshot = await getDoc(profileRef);
  
        if (profileSnapshot.exists()) {
          const data = profileSnapshot.data();
          setFormData((prevData) => ({
            ...prevData,
            ...data,
          }));
        }
      } catch (err) {
        console.error("Error fetching profile:", err);
        setError("Failed to load profile data");
      } finally {
        setLoading(false);
      }
    };
  
    fetchProfile();
  }, [userId]);   

  const handleSave = async () => {
    if (!userId) {
      setError("No user ID provided");
      return;
    }
  
    setLoading(true);
    setError(null);
  
    try {
      const profileRef = doc(db, "userProfiles", userId);
      await setDoc(
        profileRef,
        {
          ...formData, // Save the entire formData object
          updatedAt: new Date().toISOString(),
        },
        { merge: true }
      );
  
      setIsEditing(false);
    } catch (err) {
      console.error("Error saving profile:", err);
      setError("Failed to save profile data");
    } finally {
      setLoading(false);
    }
  };
  

  const handleInputChange = (section, field, value, index = null) => {
    setFormData((prev) => {
      const newData = structuredClone(prev);
  
      if (index !== null) {
        if (!Array.isArray(newData[section])) {
          newData[section] = [];
        }
        if (!newData[section][index]) {
          newData[section][index] = {};
        }
        newData[section][index][field] = value;
      } else if (section === "skills") {
        // Ensure value is assigned directly when it is already an array
        newData.skills[field] = Array.isArray(value) ? value : value.split(",").map((item) => item.trim()).filter(Boolean);
      } else if (section === "contactInformation") {
        newData.contactInformation[field] = value;
      } else {
        newData[section] = value;
      }
  
      return newData;
    });
  };
  

  const addItem = (section) => {
    setFormData((prev) => {
      const newData = { ...prev };
      const emptyItem = getEmptyItem(section);
      newData[section] = [...(prev[section] || []), emptyItem];
      return newData;
    });
  };

  const removeItem = (section, index) => {
    setFormData((prev) => {
      const newData = { ...prev };
      newData[section] = prev[section].filter((_, i) => i !== index);
      return newData;
    });
  };

  const getEmptyItem = (section) => {
    const emptyItems = {
      workExperience: {
        companyName: "",
        location: "",
        jobTitle: "",
        startDate: "",
        endDate: "",
        accomplishments: [""],
      },
      education: {
        degree: "",
        institution: "",
        location: "",
        graduationYear: "",
        grade: "",
      },
      certifications: {
        name: "",
        issuer: "",
        date: "",
      },
      projects: {
        name: "",
        description: "",
        role: "",
        technologies: [""],
      },
      achievements: {
        name: "",
        issuer: "",
        date: "",
      },
    };
    return emptyItems[section] || {};
  };

  // Add this function to handle certificate image uploads
  const handleCertificationUpload = async (index, e) => {
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
  
      // Add logging to track what's being sent
      console.log("Uploading to Cloudinary:", {
        fileName: file.name,
        fileType: file.type,
        fileSize: file.size,
        userId: userId,
        index: index
      });
  
      // Call Cloudinary API
      const response = await fetch(
        `https://api.cloudinary.com/v1_1/dihawgvdz/image/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );
  
      // Improved error handling
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Cloudinary API error response:", errorText);
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log("Cloudinary upload successful:", data.secure_url);
      
      // Update the certification image URL - fixed to use the state's formData not the local parameter
      setFormData((prevFormData) => {
        const updatedCertifications = [...prevFormData.certifications];
        updatedCertifications[index] = {
          ...updatedCertifications[index],
          imageUrl: data.secure_url,
        };
        
        return {
          ...prevFormData,
          certifications: updatedCertifications,
        };
      });
  
    } catch (err) {
      console.error('Error uploading certification image:', err);
      setError(`Failed to upload image: ${err.message}. Please try again.`);
    } finally {
      setImageUploading(false);
    }
  };

  // Cloudinary image upload handler
  const handleImageUpload = async (e) => {
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
      formData.append('folder', `user-profiles/${userId}`);

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
      
      // Update the profile picture URL in the form data
      setFormData((prev) => ({
        ...prev,
        profilePicture: data.secure_url,
      }));

    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Failed to upload image. Please try again.');
    } finally {
      setImageUploading(false);
    }
  };

  

  const addSkill = (category, skill) => {
    if (!skill.trim()) return;
  
    setFormData((prev) => ({
      ...prev,
      skills: {
        ...prev.skills,
        [category]: [...(prev.skills[category] || []), skill.trim()],
      },
    }));
  
    // Clear input after adding
    if (category === "technical") {
      setNewTechnicalSkill("");
    } else {
      setNewSoftSkill("");
    }
  };

  const removeSkill = (category, index) => {
    setFormData((prev) => {
      const updatedSkills = prev.skills[category].filter((_, i) => i !== index);
      return {
        ...prev,
        skills: {
          ...prev.skills,
          [category]: updatedSkills,
        },
      };
    });
  };
  

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Error Message */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-4 rounded-lg">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2">Loading...</p>
          </div>
        </div>
      )}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Profile</h1>
        <div>
          {isEditing ? (
            <div className="space-x-2">
              <button
                onClick={handleSave}
                className="bg-green-500 text-white px-4 py-2 rounded"
              >
                Save
              </button>
              <button
                onClick={() => setIsEditing(false)}
                className="bg-gray-500 text-white px-4 py-2 rounded"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setIsEditing(true)}
              className="bg-blue-500 text-white px-4 py-2 rounded"
            >
              Edit
            </button>
          )}
        </div>
      </div>

      <div className="mb-6">
        {/* Contact Information */}
        <section className="bg-white p-6 rounded-lg shadow">
        <ProfilePicture 
            formData={formData}
            isEditing={isEditing}
            imageUploading={imageUploading}
            handleImageUpload={handleImageUpload}
            error={error}
            setFormData={setFormData}
          />
        </section>
      </div>

      <div className="space-y-8">
        {/* Contact Information */}
        <section className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-2xl font-semibold mb-4">Contact Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(formData.contactInformation).map(([key, value]) => (
              <div key={key}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </label>
                {isEditing ? (
                  <input
                    type="text"
                    value={value}
                    onChange={(e) =>
                      handleInputChange(
                        "contactInformation",
                        key,
                        e.target.value
                      )
                    }
                    className="w-full p-2 border rounded"
                  />
                ) : (
                  <p className="text-gray-800">{value}</p>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Professional Summary */}
        <section className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-2xl font-semibold mb-4">
            Describe yourself in a line
          </h2>
          {isEditing ? (
            <textarea
              value={formData.professionalSummary}
              onChange={(e) =>
                handleInputChange("professionalSummary", null, e.target.value)
              }
              className="w-full p-2 border rounded h-32"
            />
          ) : (
            <p className="text-gray-800">{formData.professionalSummary}</p>
          )}
        </section>

        {/* Skills */}
        <section className="bg-white p-6 rounded-lg shadow">
  <h2 className="text-2xl font-semibold mb-4">Skills</h2>

  {/* Technical Skills */}
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-700 mb-1">
      Technical Skills
    </label>
    
    {formData.skills.technical && formData.skills.technical.length > 0 ? (
      <ul className="mb-2">
        {formData.skills.technical.map((skill, index) => (
          <li key={index} className="flex items-center justify-between bg-gray-100 p-2 rounded my-1">
            {skill}
            <button
              onClick={() => removeSkill("technical", index)}
              className="ml-2 text-red-600 hover:text-red-800"
            >
              ❌
            </button>
          </li>
        ))}
      </ul>
    ) : (
      <p className="text-gray-500">No technical skills listed</p>
    )}

    {isEditing && (
      <div className="flex gap-2">
        <input
          type="text"
          value={newTechnicalSkill}
          onChange={(e) => setNewTechnicalSkill(e.target.value)}
          className="w-full p-2 border rounded"
          placeholder="Enter a skill"
        />
        <button
          onClick={() => addSkill("technical", newTechnicalSkill)}
          className="bg-blue-500 text-white px-3 py-1 rounded"
        >
          Add
        </button>
      </div>
    )}
  </div>

  {/* Soft Skills */}
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">
      Soft Skills
    </label>

    {formData.skills.soft && formData.skills.soft.length > 0 ? (
      <ul className="mb-2">
        {formData.skills.soft.map((skill, index) => (
          <li key={index} className="flex items-center justify-between bg-gray-100 p-2 rounded my-1">
            {skill}
            <button
              onClick={() => removeSkill("soft", index)}
              className="ml-2 text-red-600 hover:text-red-800"
            >
              ❌
            </button>
          </li>
        ))}
      </ul>
    ) : (
      <p className="text-gray-500">No soft skills listed</p>
    )}

    {isEditing && (
      <div className="flex gap-2">
        <input
          type="text"
          value={newSoftSkill}
          onChange={(e) => setNewSoftSkill(e.target.value)}
          className="w-full p-2 border rounded"
          placeholder="Enter a skill"
        />
        <button
          onClick={() => addSkill("soft", newSoftSkill)}
          className="bg-blue-500 text-white px-3 py-1 rounded"
        >
          Add
        </button>
      </div>
    )}
  </div>
</section>

        {/* Education */}
        <section className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Education</h2>
            {isEditing && (
              <button
                onClick={() => addItem("education")}
                className="text-blue-500"
              >
                + Add Education
              </button>
            )}
          </div>
          <div className="space-y-6">
            {formData.education.map((edu, index) => (
              <div key={index} className="border-b pb-4 last:border-b-0">
                {isEditing && (
                  <button
                    onClick={() => removeItem("education", index)}
                    className="text-red-500 float-right"
                  >
                    Remove
                  </button>
                )}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {educationorder.map((key) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {key.charAt(0).toUpperCase() + key.slice(1)}
                      </label>
                      {isEditing ? (
                        <input
                          type="text"
                          value={edu[key] || ""} // Ensure it doesn't break if a key is missing
                          onChange={(e) =>
                            handleInputChange(
                              "education",
                              key,
                              e.target.value,
                              index
                            )
                          }
                          className="w-full p-2 border rounded"
                        />
                      ) : (
                        <p className="text-gray-800">{edu[key]}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Work Experience */}
        <section className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Work Experience</h2>
            {isEditing && (
              <button
                onClick={() => addItem("workExperience")}
                className="text-blue-500"
              >
                + Add Experience
              </button>
            )}
          </div>
          <div className="space-y-6">
            {formData.workExperience.map((exp, index) => (
              <div key={index} className="border-b pb-4 last:border-b-0">
                {isEditing && (
                  <button
                    onClick={() => removeItem("workExperience", index)}
                    className="text-red-500 float-right"
                  >
                    Remove
                  </button>
                )}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {workExperienceorder.map((key) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {key.charAt(0).toUpperCase() + key.slice(1)}
                      </label>
                      {isEditing ? (
                        <input
                          type="text"
                          value={exp[key] || ""} 
                          onChange={(e) =>
                            handleInputChange(
                              "workExperience",
                              key,
                              e.target.value,
                              index
                            )
                          }
                          className="w-full p-2 border rounded"
                        />
                      ) : (
                        <p className="text-gray-800">{exp[key]}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Projects */}
        <section className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Projects</h2>
            {isEditing && (
              <button
                onClick={() => addItem("projects")}
                className="text-blue-500"
              >
                + Add Project
              </button>
            )}
          </div>
          <div className="space-y-6">
            {formData.projects.map((project, index) => (
              <div key={index} className="border-b pb-4 last:border-b-0">
                {isEditing && (
                  <button
                    onClick={() => removeItem("projects", index)}
                    className="text-red-500 float-right"
                  >
                    Remove
                  </button>
                )}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {projectorder.map((key) => (
                    <div key={key}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        {key.charAt(0).toUpperCase() + key.slice(1)}
                    </label>
                    {isEditing ? (
                        key === "technologies" ? (
                        <input
                            type="text"
                            value={project[key]?.join(", ") || ""}
                            onChange={(e) =>
                            handleInputChange(
                                "projects",
                                key,
                                e.target.value.split(",").map((item) => item.trim()),
                                index
                            )
                            }
                            className="w-full p-2 border rounded"
                        />
                        ) : (
                        <input
                            type="text"
                            value={project[key] || ""}
                            onChange={(e) =>
                            handleInputChange("projects", key, e.target.value, index)
                            }
                            className="w-full p-2 border rounded"
                        />
                        )
                    ) : (
                        <p className="text-gray-800">
                        {Array.isArray(project[key]) ? project[key].join(", ") : project[key]}
                        </p>
                    )}
                    </div>
                ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Certifications */}
        <section className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Certifications</h2>
            {isEditing && (
              <button
                onClick={() => addItem("certifications")}
                className="text-blue-500"
              >
                + Add Certification
              </button>
            )}
          </div>
          <div className="space-y-6">
            {formData.certifications.map((cert, index) => (
              <div key={index} className="border-b pb-4 last:border-b-0">
                {isEditing && (
                  <button
                    onClick={() => removeItem("certifications", index)}
                    className="text-red-500 float-right"
                  >
                    Remove
                  </button>
                )}
                
                {/* Certification title */}
                <h3 className="text-lg font-medium mb-3">{cert.name}</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Left column - Text details */}
                  <div className="space-y-4">
                    {Object.entries(cert)
                      .filter(([key]) => !['imageUrl', 'skills'].includes(key)) // Exclude imageUrl and skills from this loop
                      .map(([key, value]) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            {key.charAt(0).toUpperCase() + key.slice(1)}
                          </label>
                          {isEditing ? (
                            <input
                              type="text"
                              value={value}
                              onChange={(e) =>
                                handleInputChange(
                                  "certifications",
                                  key,
                                  e.target.value,
                                  index
                                )
                              }
                              className="w-full p-2 border rounded"
                            />
                          ) : (
                            <p className="text-gray-800">{value}</p>
                          )}
                        </div>
                      ))}
                    
                    {/* Skills section with error handling */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Skills
                      </label>
                      {isEditing ? (
                        <input
                          type="text"
                          value={cert.skills && Array.isArray(cert.skills) ? cert.skills.join(", ") : ""}
                          onChange={(e) => {
                            const skillsArray = e.target.value.split(",").map(skill => skill.trim()).filter(Boolean);
                            handleInputChange(
                              "certifications",
                              "skills",
                              skillsArray,
                              index
                            );
                          }}
                          className="w-full p-2 border rounded"
                          placeholder="Separate skills with commas"
                        />
                      ) : (
                        <p className="text-gray-800">
                          {cert.skills && Array.isArray(cert.skills) && cert.skills.length > 0 
                            ? cert.skills.join(", ") 
                            : "No skills listed"}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  {/* Right column - Certification image */}
                  <div>
                    {isEditing ? (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Certification Image
                        </label>
                        {cert.imageUrl ? (
                          <div className="relative">
                            <img 
                              src={cert.imageUrl} 
                              alt={cert.name || "Certification"} 
                              className="w-full max-h-64 object-contain border border-gray-300 rounded"
                            />
                            <button
                              onClick={() => handleInputChange("certifications", "imageUrl", "", index)}
                              className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center"
                            >
                              ×
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center justify-center border border-dashed border-gray-300 rounded-md p-6">
                            <input
                              type="file"
                              id={`certification-image-${index}`}
                              className="hidden"
                              accept="image/*"
                              onChange={(e) => handleCertificationUpload(index, e)}
                            />
                            <label
                              htmlFor={`certification-image-${index}`}
                              className="cursor-pointer px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                            >
                              Upload Image
                            </label>
                          </div>
                        )}
                      </div>
                    ) : (
                      cert.imageUrl && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Certification Image
                          </label>
                          <img 
                            src={cert.imageUrl} 
                            alt={cert.name || "Certification"} 
                            className="w-full max-h-64 object-contain border border-gray-300 rounded"
                          />
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Achievements */}
        <section className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Achievements</h2>
            {isEditing && (
              <button
                onClick={() => addItem("achievements")}
                className="text-blue-500"
              >
                + Add Achievement
              </button>
            )}
          </div>
          <div className="space-y-6">
            {formData.achievements.map((achievement, index) => (
              <div key={index} className="border-b pb-4 last:border-b-0">
                {isEditing && (
                  <button
                    onClick={() => removeItem("achievements", index)}
                    className="text-red-500 float-right"
                  >
                    Remove
                  </button>
                )}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(achievement).map(([key, value]) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {key.charAt(0).toUpperCase() + key.slice(1)}
                      </label>
                      {isEditing ? (
                        <input
                          type="text"
                          value={value}
                          onChange={(e) =>
                            handleInputChange(
                              "achievements",
                              key,
                              e.target.value,
                              index
                            )
                          }
                          className="w-full p-2 border rounded"
                        />
                      ) : (
                        <p className="text-gray-800">{value}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Loading and Error States */}
      {isEditing && (
        <div className="fixed bottom-4 right-4 space-x-2">
          <button
            onClick={handleSave}
            className="bg-green-500 text-white px-6 py-2 rounded shadow hover:bg-green-600 transition-colors"
          >
            Save All Changes
          </button>
          <button
            onClick={() => setIsEditing(false)}
            className="bg-gray-500 text-white px-6 py-2 rounded shadow hover:bg-gray-600 transition-colors"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};

export default ProfilePage;
