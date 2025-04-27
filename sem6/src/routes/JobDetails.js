import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, ChevronUp, ArrowLeft, Briefcase, Code, Database, Server, ChevronRight, ChevronLeft } from "lucide-react";

const jobData = {
  "full-stack-developer": {
    title: "Full Stack Developer",
    icon: <Code className="h-8 w-8" />,
    description: "Build scalable web applications by working on both frontend and backend technologies like React, Node.js, and databases.",
    responsibilities: [
      "Develop frontend with React, Angular, or Vue.js.",
      "Build backend using Node.js, Express.js, or Django.",
      "Manage databases like MongoDB, MySQL, or Firebase.",
      "Ensure responsiveness and optimize performance.",
      "Integrate APIs and third-party services."
    ],
    skills: [
      "HTML, CSS, Javascript, React, Angular",
      "Backend frameworks (Node.js, Express.js, Django)",
      "Database management (SQL, NoSQL, Firebase)",
      "Version control (Git, GitHub, GitLab)",
      "Cloud services (AWS, Google Cloud, Firebase)"
    ],
    color: "from-blue-600 to-indigo-600",
    accentColor: "blue"
  },
  "ai-ml-engineer": {
    title: "AI/ML Engineer",
    icon: <Database className="h-8 w-8" />,
    description: "Develop and deploy AI/ML models using Python, TensorFlow, PyTorch, and data science techniques.",
    responsibilities: [
      "Train and fine-tune AI models using Python.",
      "Implement deep learning solutions using TensorFlow and PyTorch.",
      "Optimize machine learning pipelines.",
      "Process and analyze large datasets.",
      "Deploy models using cloud services like AWS and GCP."
    ],
    skills: [
      "Machine Learning & Deep Learning",
      "Python, TensorFlow, PyTorch",
      "Big Data Processing (Hadoop, Spark)",
      "Cloud Computing & Deployment",
      "Data Science & Analytics"
    ],
    color: "from-purple-600 to-pink-600",
    accentColor: "purple"
  },
  "cloud-devops-engineer": {
    title: "Cloud & DevOps Engineer",
    icon: <Server className="h-8 w-8" />,
    description: "Manage and automate cloud infrastructure using AWS, Azure, Kubernetes, and Terraform.",
    responsibilities: [
      "Design and deploy scalable cloud infrastructure.",
      "Implement CI/CD pipelines using Jenkins, GitHub Actions.",
      "Monitor system performance and security.",
      "Manage Kubernetes clusters and containerized applications.",
      "Automate infrastructure with Terraform and Ansible."
    ],
    skills: [
      "Cloud Platforms (AWS, Azure, GCP)",
      "Infrastructure as Code (Terraform, Ansible)",
      "CI/CD Tools (Jenkins, GitHub Actions)",
      "Containerization (Docker, Kubernetes)",
      "Networking & Security"
    ],
    color: "from-cyan-600 to-teal-600",
    accentColor: "cyan"
  },
  "software-engineer": {
    title: "Software Engineer",
    icon: <Briefcase className="h-8 w-8" />,
    description: "Design, develop, and maintain software applications using various programming languages.",
    responsibilities: [
      "Write clean and maintainable code.",
      "Develop applications using Java, Python, or C++.",
      "Work with databases and backend APIs.",
      "Optimize application performance.",
      "Collaborate with cross-functional teams."
    ],
    skills: [
      "Programming (Java, Python, C++)",
      "Database Management (SQL, NoSQL)",
      "Software Development Life Cycle (SDLC)",
      "Version Control (Git, GitHub)",
      "Problem-Solving & Algorithms"
    ],
    color: "from-emerald-600 to-green-600",
    accentColor: "emerald"
  }
};

const JobDetails = () => {
  const [selectedJob, setSelectedJob] = useState("full-stack-developer");
  const [openSection, setOpenSection] = useState("description");
  const carouselRef = useRef(null);
  const navigate = useNavigate();

  const handleApplyNow = () => {
    navigate("/signup")
  }

  const toggleSection = (section) => {
    setOpenSection(openSection === section ? null : section);
  };

  const scrollCarousel = (direction) => {
    if (carouselRef.current) {
      const scrollAmount = direction === 'left' ? -280 : 280;
      carouselRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  const getAccentColorClass = (color, type) => {
    const colorMap = {
      blue: {
        bg: "bg-blue-600",
        hover: "hover:bg-blue-700",
        text: "text-blue-600",
        ring: "ring-blue-300",
        light: "bg-blue-50",
        lightText: "text-blue-700",
        gradient: "from-blue-600 to-indigo-600",
      },
      purple: {
        bg: "bg-purple-600",
        hover: "hover:bg-purple-700",
        text: "text-purple-600",
        ring: "ring-purple-300",
        light: "bg-purple-50",
        lightText: "text-purple-700",
        gradient: "from-purple-600 to-pink-600",
      },
      cyan: {
        bg: "bg-cyan-600",
        hover: "hover:bg-cyan-700",
        text: "text-cyan-600",
        ring: "ring-cyan-300",
        light: "bg-cyan-50",
        lightText: "text-cyan-700",
        gradient: "from-cyan-600 to-teal-600",
      },
      emerald: {
        bg: "bg-emerald-600",
        hover: "hover:bg-emerald-700",
        text: "text-emerald-600",
        ring: "ring-emerald-300",
        light: "bg-emerald-50",
        lightText: "text-emerald-700",
        gradient: "from-emerald-600 to-green-600",
      }
    };
    
    return colorMap[color][type];
  };

  const currentAccentColor = jobData[selectedJob].accentColor;

  return (
    <div className={`bg-gradient-to-br from-gray-50 to-${currentAccentColor}-50 min-h-screen flex flex-col items-center w-full`}>
      {/* Animated top wave decoration */}
      <div className="absolute top-0 left-0 w-full overflow-hidden h-40 pointer-events-none">
        <svg className="absolute top-0 w-full h-full" viewBox="0 0 1200 120" preserveAspectRatio="none">
          <path 
            d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" 
            className={`fill-${currentAccentColor}-100 opacity-30`}
          ></path>
          <path 
            d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z" 
            className={`fill-${currentAccentColor}-200 opacity-20`}
          ></path>
        </svg>
      </div>

      {/* Main Content Container - Make it take full width */}
      <div className="w-full max-w-screen-2xl px-4 sm:px-6 lg:px-8 z-10 mt-6">
        {/* Header with Back Button */}
        <div className="w-full mb-8 flex items-center justify-between">
          <button
            onClick={() => navigate("/")}
            className={`flex items-center px-4 py-2 ${getAccentColorClass(currentAccentColor, 'bg')} text-white rounded-lg shadow-md ${getAccentColorClass(currentAccentColor, 'hover')} transition-all duration-300 focus:outline-none focus:ring-2 ${getAccentColorClass(currentAccentColor, 'ring')}`}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </button>
          
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 text-center flex-grow">
            Explore Tech Career Paths
          </h1>
          
          <div className="w-32 hidden sm:block"></div> {/* Spacer for centering */}
        </div>

        {/* Main Content Flex Container */}
        <div className="flex flex-col lg:flex-row lg:space-x-6 w-full">
          {/* Side Navigation for Job Roles (Visible on larger screens) */}
          <div className="hidden lg:flex flex-col space-y-4 min-w-64 max-w-64 self-start sticky top-6">
            {Object.keys(jobData).map((jobKey) => (
              <div
                key={jobKey}
                className={`p-4 rounded-xl shadow-md cursor-pointer transition-all duration-300 flex items-center space-x-3 ${
                  selectedJob === jobKey
                    ? getAccentColorClass(jobData[jobKey].accentColor, 'bg') + " text-white"
                    : "bg-white hover:bg-gray-50 text-gray-800"
                }`}
                onClick={() => setSelectedJob(jobKey)}
              >
                <div className={`p-2 rounded-full ${
                  selectedJob === jobKey 
                    ? "bg-white bg-opacity-20" 
                    : getAccentColorClass(jobData[jobKey].accentColor, 'light')
                }`}>
                  <div className={
                    selectedJob === jobKey 
                      ? "text-white" 
                      : getAccentColorClass(jobData[jobKey].accentColor, 'text')
                  }>
                    {jobData[jobKey].icon}
                  </div>
                </div>
                <h3 className="font-semibold">{jobData[jobKey].title}</h3>
              </div>
            ))}
          </div>

          {/* Carousel for Job Role Cards (Visible on smaller screens) */}
          <div className="relative w-full lg:hidden mb-8">
            {/* Carousel Navigation Buttons */}
            <button
              onClick={() => scrollCarousel('left')}
              className={`absolute left-0 top-1/2 -translate-y-1/2 -ml-2 z-10 bg-white p-2 rounded-full shadow-md hover:bg-gray-100 focus:outline-none ${getAccentColorClass(currentAccentColor, 'text')}`}
              aria-label="Previous"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
           
            <button
              onClick={() => scrollCarousel('right')}
              className={`absolute right-0 top-1/2 -translate-y-1/2 -mr-2 z-10 bg-white p-2 rounded-full shadow-md hover:bg-gray-100 focus:outline-none ${getAccentColorClass(currentAccentColor, 'text')}`}
              aria-label="Next"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
           
            {/* Carousel Container */}
            <div
              ref={carouselRef}
              className="flex space-x-4 overflow-x-auto py-4 px-6 scrollbar-hide"
              style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
            >
              {Object.keys(jobData).map((jobKey) => (
                <div
                  key={jobKey}
                  className={`flex-shrink-0 w-64 p-5 rounded-xl shadow-md cursor-pointer transition-all duration-300 transform hover:-translate-y-1 ${
                    selectedJob === jobKey
                      ? getAccentColorClass(jobData[jobKey].accentColor, 'bg') + " text-white ring-4 " + getAccentColorClass(jobData[jobKey].accentColor, 'ring')
                      : "bg-white text-gray-800 hover:bg-gray-50"
                  }`}
                  onClick={() => setSelectedJob(jobKey)}
                >
                  <div className="flex items-center justify-center mb-3">
                    <div className={`p-3 rounded-full ${
                      selectedJob === jobKey 
                        ? "bg-white bg-opacity-20" 
                        : getAccentColorClass(jobData[jobKey].accentColor, 'light')
                    }`}>
                      <div className={
                        selectedJob === jobKey 
                          ? "text-white" 
                          : getAccentColorClass(jobData[jobKey].accentColor, 'text')
                      }>
                        {jobData[jobKey].icon}
                      </div>
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-center">{jobData[jobKey].title}</h3>
                </div>
              ))}
            </div>
          </div>

          {/* Job Details with Accordion */}
          <div className="bg-white rounded-xl shadow-lg w-full border border-gray-100 overflow-hidden flex-1 transform transition-all duration-300">
            {/* Job Title Header */}
            <div className={`bg-gradient-to-r ${jobData[selectedJob].color} p-8 text-white relative overflow-hidden`}>
              <div className="relative z-10">
                <h1 className="text-3xl md:text-4xl font-bold mb-4">{jobData[selectedJob].title}</h1>
                <p className="text-white text-opacity-90 text-lg max-w-3xl">
                  {jobData[selectedJob].description}
                </p>
              </div>
              
              {/* Decorative elements */}
              <div className="absolute top-0 right-0 transform translate-x-1/4 -translate-y-1/4 opacity-10">
                <div className="text-white">{jobData[selectedJob].icon}</div>
                <div style={{ fontSize: '200px' }}>{jobData[selectedJob].icon}</div>
              </div>
              
              <svg className="absolute bottom-0 left-0 w-full h-12 -mb-1" viewBox="0 0 1200 120" preserveAspectRatio="none">
                <path d="M985.66,92.83C906.67,72,823.78,31,743.84,14.19c-82.26-17.34-168.06-16.33-250.45.39-57.84,11.73-114,31.07-172,41.86A600.21,600.21,0,0,1,0,27.35V120H1200V95.8C1132.19,118.92,1055.71,111.31,985.66,92.83Z" className="fill-white"></path>
              </svg>
            </div>
           
            {/* Description Accordion Item */}
            <div className="border-b border-gray-200">
              <button
                className="w-full p-6 flex justify-between items-center focus:outline-none group"
                onClick={() => toggleSection("description")}
              >
                <h2 className={`text-xl font-semibold ${getAccentColorClass(currentAccentColor, 'text')} group-hover:text-gray-800 transition-colors duration-200`}>
                  Overview
                </h2>
                {openSection === "description" ? (
                  <ChevronUp className={`h-5 w-5 ${getAccentColorClass(currentAccentColor, 'text')}`} />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-500 group-hover:text-gray-800 transition-colors duration-200" />
                )}
              </button>
             
              {openSection === "description" && (
                <div className="px-6 pb-6 text-gray-700 animate-fadeIn">
                  <p className="text-lg leading-relaxed">{jobData[selectedJob].description}</p>
                </div>
              )}
            </div>

            {/* Responsibilities Accordion Item */}
            <div className="border-b border-gray-200">
              <button
                className="w-full p-6 flex justify-between items-center focus:outline-none group"
                onClick={() => toggleSection("responsibilities")}
              >
                <h2 className={`text-xl font-semibold ${getAccentColorClass(currentAccentColor, 'text')} group-hover:text-gray-800 transition-colors duration-200`}>
                  Key Responsibilities
                </h2>
                {openSection === "responsibilities" ? (
                  <ChevronUp className={`h-5 w-5 ${getAccentColorClass(currentAccentColor, 'text')}`} />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-500 group-hover:text-gray-800 transition-colors duration-200" />
                )}
              </button>
             
              {openSection === "responsibilities" && (
                <div className="px-6 pb-6 animate-fadeIn">
                  <ul className="space-y-4">
                    {jobData[selectedJob].responsibilities.map((item, index) => (
                      <li key={index} className="flex items-start group">
                        <div className={`flex-shrink-0 h-3 w-3 mt-2 rounded-full ${getAccentColorClass(currentAccentColor, 'bg')} mr-3 group-hover:scale-110 transition-transform duration-200`}></div>
                        <span className="text-gray-700 text-lg">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Skills Accordion Item */}
            <div>
              <button
                className="w-full p-6 flex justify-between items-center focus:outline-none group"
                onClick={() => toggleSection("skills")}
              >
                <h2 className={`text-xl font-semibold ${getAccentColorClass(currentAccentColor, 'text')} group-hover:text-gray-800 transition-colors duration-200`}>
                  Required Skills
                </h2>
                {openSection === "skills" ? (
                  <ChevronUp className={`h-5 w-5 ${getAccentColorClass(currentAccentColor, 'text')}`} />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-500 group-hover:text-gray-800 transition-colors duration-200" />
                )}
              </button>
             
              {openSection === "skills" && (
                <div className="px-6 pb-6 animate-fadeIn">
                  <div className="flex flex-wrap gap-3">
                    {jobData[selectedJob].skills.map((item, index) => (
                      <div 
                        key={index} 
                        className={`${getAccentColorClass(currentAccentColor, 'light')} ${getAccentColorClass(currentAccentColor, 'lightText')} px-4 py-2 rounded-full text-sm font-medium shadow-sm transform hover:scale-105 transition-transform duration-200`}
                      >
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            {/* Application CTA */}
            <div className="p-6 bg-gray-50 flex flex-col sm:flex-row items-center justify-between">
              <p className="text-gray-600 mb-4 sm:mb-0">Ready to start your journey as a {jobData[selectedJob].title}?</p>
              <button
                onClick={handleApplyNow} 
                className={`${getAccentColorClass(currentAccentColor, 'bg')} text-white px-6 py-3 rounded-lg font-medium shadow-md ${getAccentColorClass(currentAccentColor, 'hover')} transform hover:scale-105 transition-all duration-300`}
              >
                Apply Now
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer decoration */}
      <div className="w-full h-24 mt-16 bg-gradient-to-r from-gray-800 to-gray-900 flex items-center justify-center">
        <p className="text-white text-opacity-80">© 2025 SkillBridgeAI</p>
      </div>
      
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
};

export default JobDetails;