const ContactInfoSection = ({ formData, setFormData }) => {
    const contactInfo = formData?.contactInformation || {
        fullName: "",
        phoneNumber: "",
        email: "",
        address: "",
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prevFormData) => ({
          ...prevFormData,
          contactInformation: {
            ...prevFormData.contactInformation, 
            [name]: value,
          },
        }));
      };
      

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Contact Information</h2>
      <p className="mb-4 text-gray-600">Let employers know how they can reach you.</p>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
          <input
            type="text"
            name="fullName"
            value={contactInfo.fullName}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Phone Number</label>
          <input
            type="tel"
            name="phoneNumber"
            value={contactInfo.phoneNumber}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input
            type="email"
            name="email"
            value={contactInfo.email}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Address</label>
          <textarea
            name="address"
            value={contactInfo.address}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            rows="3"
          ></textarea>
        </div>
      </div>
    </div>
  );
};

export default ContactInfoSection;