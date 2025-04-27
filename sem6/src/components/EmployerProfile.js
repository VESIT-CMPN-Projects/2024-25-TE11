import React, { useState, useEffect, useCallback } from 'react';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, doc, getDoc, setDoc } from 'firebase/firestore';

const EmployerProfile = () => {
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [profileData, setProfileData] = useState({
    fullName: '',
    email: '',
    phone: '',
    bio: '',
    companyName: '',
    industry: '',
    companySize: '',
    headquarters: '',
    website: '',
    about: '',
    location: '',
    socialLinks: {
      linkedin: '',
      twitter: ''
    }
  });

  const auth = getAuth();
  const db = getFirestore();

  // Use useCallback to memoize the fetch function
  const fetchEmployerProfile = useCallback(async (userId) => {
    if (!userId) return;
    
    setLoading(true);
    try {
      const docRef = doc(db, 'employers', userId);
      const docSnap = await getDoc(docRef);

      if (docSnap.exists()) {
        setProfileData(docSnap.data());
      }
      setLoading(false);
    } catch (err) {
      console.error('Error fetching profile:', err);
      setError('Error fetching profile data');
      setLoading(false);
    }
  }, [db]);

  useEffect(() => {
    // Handle auth state changes properly
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        fetchEmployerProfile(user.uid);
      } else {
        setError('No authenticated user found');
        setLoading(false);
      }
    });

    // Clean up subscription
    return () => unsubscribe();
  }, [auth, fetchEmployerProfile]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSocialLinkChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      socialLinks: {
        ...prev.socialLinks,
        [name]: value
      }
    }));
  };

  const handleSubmit = async () => {
    try {
      const userId = auth.currentUser?.uid;
      if (!userId) {
        setError('No authenticated user found');
        return;
      }

      setLoading(true);
      const docRef = doc(db, 'employers', userId);
      await setDoc(docRef, profileData, { merge: true });
      setIsEditing(false);
      setLoading(false);
    } catch (err) {
      console.error('Error saving profile:', err);
      setError('Error saving profile data');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Company Profile</h1>
        <button
          onClick={() => isEditing ? handleSubmit() : setIsEditing(true)}
          className={`px-6 py-2 rounded-lg text-white font-medium transition-colors
            ${isEditing 
              ? 'bg-green-600 hover:bg-green-700' 
              : 'bg-blue-600 hover:bg-blue-700'
            }`}
          disabled={loading}
        >
          {isEditing ? 'Save Changes' : 'Edit Profile'}
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                Full Name
                </label>
                <input
                type="text"
                name="fullName"
                value={profileData.fullName}
                onChange={handleInputChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                Email
                </label>
                <input
                type="email"
                name="email"
                value={profileData.email}
                onChange={handleInputChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                Phone Number
                </label>
                <input
                type="text"
                name="phone"
                value={profileData.phone}
                onChange={handleInputChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                Bio
                </label>
                <input
                type="text"
                name="bio"
                value={profileData.bio}
                onChange={handleInputChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                />
            </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Company Name
            </label>
            <input
              type="text"
              name="companyName"
              value={profileData.companyName}
              onChange={handleInputChange}
              disabled={!isEditing}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Industry
            </label>
            <input
              type="text"
              name="industry"
              value={profileData.industry}
              onChange={handleInputChange}
              disabled={!isEditing}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Company Size
            </label>
            <input
              type="text"
              name="companySize"
              value={profileData.companySize}
              onChange={handleInputChange}
              disabled={!isEditing}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Location
            </label>
            <input
              type="text"
              name="headquarters"
              value={profileData.headquarters}
              onChange={handleInputChange}
              disabled={!isEditing}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Website
            </label>
            <input
              type="url"
              name="website"
              value={profileData.website}
              onChange={handleInputChange}
              disabled={!isEditing}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
            />
          </div>
        </div>

        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            About
          </label>
          <textarea
            name="about"
            value={profileData.about}
            onChange={handleInputChange}
            disabled={!isEditing}
            rows="5"
            className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
          />
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Social Links</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LinkedIn
              </label>
              <input
                type="url"
                name="linkedin"
                value={profileData.socialLinks.linkedin}
                onChange={handleSocialLinkChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Twitter
              </label>
              <input
                type="url"
                name="twitter"
                value={profileData.socialLinks.twitter}
                onChange={handleSocialLinkChange}
                disabled={!isEditing}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmployerProfile;