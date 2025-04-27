const ProfessionalSummarySection = ({ formData, setFormData }) => {
    const handleChange = (e) => {
      setFormData({
        ...formData,
        professionalSummary: e.target.value,
      });
    };
  
    return (
      <div>
        <h2 className="text-xl font-semibold mb-4">Professional Summary</h2>
        <p className="mb-4 text-gray-600">Provide a brief overview of your professional background and goals.</p>
        
        <div>
          <textarea
            value={formData.professionalSummary}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            rows="6"
            placeholder="Examples: 'Dedicated software engineer with 5 years of experience...' or 'Recent graduate with a passion for...'"
          ></textarea>
        </div>
      </div>
    );
  };

export default ProfessionalSummarySection;