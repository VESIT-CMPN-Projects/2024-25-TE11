import React, { useState, useEffect } from "react";
import { doc, getDoc, collection, getDocs, setDoc } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";

const SkeletonLoader = () => {
  return (
    <div className="space-y-6">
      {["Beginner", "Intermediate", "Expert"].map((category) => (
        <div key={category} className="border border-gray-200 rounded-lg overflow-hidden animate-pulse">
          <div className="p-4 bg-gray-50 border-b border-gray-200">
            <div className="h-6 bg-gray-300 rounded w-1/3"></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-gray-200">
            {[1, 2].map((idx) => (
              <div key={idx} className="p-4">
                <div className="h-5 bg-gray-300 rounded w-3/4 mb-3"></div>
                <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-5/6 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-4/6 mb-4"></div>
                
                <div className="flex flex-wrap gap-2 mb-4">
                  {[1, 2, 3].map((skillIdx) => (
                    <div key={skillIdx} className="h-6 w-16 bg-gray-200 rounded-full"></div>
                  ))}
                </div>
                
                <div className="h-8 bg-gray-200 rounded w-full mt-4"></div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// Sample in-memory job data that would be used for context when suggesting projects
const IN_MEMORY_JOB_DATA = [
  {
    pageContent: "Looking for a skilled JavaScript developer with React experience to build responsive web applications. The ideal candidate should be proficient in HTML/CSS and have experience with modern frontend frameworks.",
    metadata: {
      title: "Frontend Developer",
      skills: "JavaScript, React, HTML/CSS"
    }
  },
  {
    pageContent: "We're seeking a Firebase specialist who can develop and maintain cloud-based applications. Knowledge of React and experience with NoSQL databases is essential.",
    metadata: {
      title: "Firebase Developer",
      skills: "Firebase, React, NoSQL, JavaScript"
    }
  },
  {
    pageContent: "Full-stack developer needed for building complete web applications. Must have strong skills in JavaScript, React for frontend, and Firebase for backend services. Experience with authentication flows and database design is a plus.",
    metadata: {
      title: "Full-Stack Web Developer",
      skills: "JavaScript, React, Firebase, HTML/CSS, Authentication"
    }
  },
  {
    pageContent: "UI/UX specialist with strong coding abilities required. Must be proficient in creating beautiful interfaces with HTML/CSS and implementing them with React. Experience with Firebase and user authentication flows is highly desirable.",
    metadata: {
      title: "UI Developer",
      skills: "HTML/CSS, React, JavaScript, UI/UX, Firebase"
    }
  },
  {
    pageContent: "Looking for a developer to build real-time applications. The candidate should have extensive knowledge of Firebase's real-time database, React for frontend, and JavaScript. Ability to optimize performance and implement complex data structures is essential.",
    metadata: {
      title: "Real-time Application Developer",
      skills: "Firebase, React, JavaScript, Real-time Data, Performance Optimization"
    }
  },
  {
    pageContent: "Backend developer needed with experience in Node.js, Express, and database design. Familiarity with RESTful API development and authentication systems is required.",
    metadata: {
      title: "Backend Developer",
      skills: "Node.js, Express, MongoDB, RESTful API, Authentication"
    }
  },
  {
    pageContent: "Mobile app developer with React Native experience needed. Should be comfortable with JavaScript/TypeScript and have knowledge of native device features.",
    metadata: {
      title: "Mobile Developer",
      skills: "React Native, JavaScript, TypeScript, Mobile Development"
    }
  },
  {
    pageContent: "DevOps engineer with CI/CD pipeline experience. Knowledge of Docker, Kubernetes, and cloud platforms like AWS or Firebase is essential.",
    metadata: {
      title: "DevOps Engineer",
      skills: "Docker, Kubernetes, CI/CD, AWS, Firebase"
    }
  }
];

const ProjectSuggester = () => {
    const [technicalSkills, setTechnicalSkills] = useState([]);
    const [projects, setProjects] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [debugInfo, setDebugInfo] = useState(null);
    const [relevantJobData, setRelevantJobData] = useState([]);
    const [selectedProject, setSelectedProject] = useState(null);
    const [savingProject, setSavingProject] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [cachedProjects, setCachedProjects] = useState({});
    const [refreshCounter, setRefreshCounter] = useState(0); // Add a refresh counter to force regeneration
    
    const auth = getAuth();
    const db = getFirestore();
  
    // Fetch user skills on component mount
    useEffect(() => {
      fetchUserSkills();
    }, []);
  
    // Automatically fetch project ideas when skills are loaded
    useEffect(() => {
      if (technicalSkills.length > 0) {
        // First retrieve relevant job data, then generate project ideas
        retrieveRelevantJobs().then(() => {
          fetchProjectIdeas();
        });
      }
    }, [technicalSkills, refreshCounter]); // Add refreshCounter to the dependency array

    // Function to fetch user skills from Firestore
    const fetchUserSkills = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Check if skills are already in state or cache
        const cachedSkills = localStorage.getItem('userSkills');
        if (cachedSkills) {
          setTechnicalSkills(JSON.parse(cachedSkills));
          setLoading(false);
          return;
        }
    
        const currentUser = auth.currentUser;
        if (!currentUser) {
          setError('You must be logged in to view project suggestions.');
          setLoading(false);
          return;
        }
    
        const userProfileRef = doc(db, 'userProfiles', currentUser.uid);
        const userProfileSnap = await getDoc(userProfileRef);
    
        if (userProfileSnap.exists()) {
          const userProfileData = userProfileSnap.data();
          let skills = userProfileData.skills?.technical || 
                       userProfileData.skills || 
                       userProfileData.userSkills || 
                       [];
    
          if (Array.isArray(skills) && skills.length > 0) {
            setTechnicalSkills(skills);
            localStorage.setItem('userSkills', JSON.stringify(skills));
            setLoading(false);
            return;
          }
        }
    
        // Use default skills if none found
        const mockSkills = ["JavaScript", "React", "HTML/CSS", "Firebase"];
        setTechnicalSkills(mockSkills);
        localStorage.setItem('userSkills', JSON.stringify(mockSkills));
        setLoading(false);
      } catch (err) {
        setError(`Failed to fetch skills: ${err.message}`);
        setLoading(false);
      }
    };   
    
    

    // Function to calculate cosine similarity between two lists of skills
    // This mimics the vector similarity search that would be done in ChromaDB
    const calculateSkillSimilarity = (userSkills, jobSkills) => {
      const userSkillsSet = new Set(userSkills.map(skill => skill.toLowerCase()));
      const jobSkillsArray = jobSkills.split(',').map(skill => skill.trim().toLowerCase());
      
      // Count matching skills
      let matchCount = 0;
      for (const skill of jobSkillsArray) {
        if (userSkillsSet.has(skill)) {
          matchCount++;
        }
      }
      
      // Simple similarity score based on matches
      // Higher score = more skills match
      if (jobSkillsArray.length === 0) return 0;
      return matchCount / jobSkillsArray.length;
    };

    const precomputedSkillLookup = {};
    IN_MEMORY_JOB_DATA.forEach(job => {
      job.metadata.skills.split(',').forEach(skill => {
        const trimmedSkill = skill.trim().toLowerCase();
        if (!precomputedSkillLookup[trimmedSkill]) {
          precomputedSkillLookup[trimmedSkill] = [];
        }
        precomputedSkillLookup[trimmedSkill].push(job);
      });
    });

const retrieveRelevantJobs = async () => {
  try {
    const matchedJobs = new Set();
    technicalSkills.forEach(skill => {
      const skillLower = skill.toLowerCase();
      if (precomputedSkillLookup[skillLower]) {
        precomputedSkillLookup[skillLower].forEach(job => matchedJobs.add(job));
      }
    });

    const formattedJobs = Array.from(matchedJobs).slice(0, 5).map(job => ({
      content: job.pageContent,
      metadata: job.metadata
    }));

    setRelevantJobData(formattedJobs);
  } catch (err) {
    setError(`Failed to retrieve job data: ${err.message}`);
  }
};


    // Define the output parser schema using Zod
    const createOutputParser = () => {
      const ProjectSchema = z.object({
        title: z.string().describe("The name of the project"),
        description: z.string().describe("A short description of the project"),
        steps: z.array(z.string()).describe("Implementation steps for the project"),
        skills: z.array(z.string()).describe("Skills used in this project"),
        learningObjectives: z.array(z.string()).describe("What you'll learn from this project"),
        estimatedTime: z.string().describe("Estimated time to complete the project"),
        resources: z.array(z.object({
          name: z.string().describe("Name of the resource"),
          url: z.string().url().optional().describe("URL of the resource (if applicable)")
        })).describe("Helpful resources for this project")
      });
    
      const CategorySchema = z.object({
        category: z.enum(["Beginner", "Intermediate", "Expert"]).describe("Difficulty level of the projects"),
        projects: z.array(ProjectSchema).describe("List of projects in this difficulty category")
      });
    
      const ResponseSchema = z.array(CategorySchema).describe("Categories of projects organized by difficulty");
    
      return StructuredOutputParser.fromZodSchema(ResponseSchema);
    };

    const fetchProjectIdeas = async () => {
      if (technicalSkills.length === 0) return;
      
      setLoading(true);
      const skillsString = technicalSkills.join(", ");
    
      const cacheKey = JSON.stringify({
        skills: skillsString,
        jobs: relevantJobData.map(job => job.metadata.title),
        refresh: refreshCounter
      });
    
      if (cachedProjects[cacheKey]) {
        setProjects(cachedProjects[cacheKey]);
        setLoading(false);
        return;
      }
    
      try {
        const model = new ChatGoogleGenerativeAI({
          model: "gemini-1.5-flash",
          temperature: 0.2,
          apiKey: process.env.REACT_APP_GOOGLE_API_KEY,
          maxRetries: 2,
          streaming: false, // Disable streaming for more reliable parsing
        });
    
        const outputParser = createOutputParser();
        const formatInstructions = outputParser.getFormatInstructions();
    
        let jobContext = "";
        if (relevantJobData.length > 0) {
          jobContext = `
          Relevant job market context to consider when suggesting projects:
          ${relevantJobData.map((job, index) => 
            `Job ${index + 1}: ${job.metadata.title || 'Unknown Position'}
             Required Skills: ${job.metadata.skills || 'Not specified'}
             Description: ${job.content.substring(0, 300)}...`
          ).join('\n\n')}
          `;
        }
    
        const projectIdeasPrompt = new PromptTemplate({
          template: `
          You are a career development advisor tasked with suggesting projects that will help a developer build their portfolio.
          
          User Skills: {skills}
          
          ${relevantJobData.length > 0 ? '{jobContext}' : ''}
          
          Categorize project ideas into Beginner, Intermediate, and Expert that would be valuable for someone with these skills.
          The projects should help the user develop practical experience that would make them competitive for jobs
          requiring these skills.
          
          Provide 2 ideas per category, and include:
          1. A descriptive title
          2. A concise description of what the project does and why it's valuable
          3. Implementation steps that are specific and actionable (minimum 5 steps, maximum 10)
          4. Skills used in the project
          5. Learning objectives (what they'll gain from completing it)
          6. Estimated time to complete (e.g. "2-3 days", "1-2 weeks")
          7. Helpful resources (tutorials, documentation, etc.)
        
          {format_instructions}
          `,
          inputVariables: ["skills", ...(relevantJobData.length > 0 ? ["jobContext"] : [])],
          partialVariables: { format_instructions: formatInstructions }
        });
    
        const chain = new LLMChain({
          llm: model,
          prompt: projectIdeasPrompt,
          outputParser: outputParser
        });
    
        const input = {
          skills: skillsString,
          ...(relevantJobData.length > 0 ? { jobContext: jobContext } : {})
        };
    
        // Add debug logging
        console.log("Sending request with input:", input);
        
        const result = await chain.call(input);
        
        // Add debug logging
        console.log("Received response:", result);
    
        if (!result.text) {
          throw new Error("Empty response from API");
        }
    
        setCachedProjects(prev => ({
          ...prev,
          [cacheKey]: result.text
        }));
    
        setProjects(result.text);
      } catch (apiError) {
        console.error("Error fetching projects:", apiError);
        
        // More detailed error handling
        let errorMessage = `API error: ${apiError.message}`;
        let fallbackProjects = null;
    
        if (apiError.message.includes("parsing") || apiError.message.includes("stream")) {
          errorMessage = "We encountered a formatting issue with the response. Showing simplified results.";
          
          // Try to extract any usable data from the error
          if (apiError.output) {
            try {
              // First try to parse as-is
              fallbackProjects = JSON.parse(apiError.output);
            } catch (e) {
              // If that fails, try to extract JSON from the string
              const jsonMatch = apiError.output.match(/```json\n([\s\S]*?)\n```/) || 
                              apiError.output.match(/\[\s*{[\s\S]*}\s*\]/);
              if (jsonMatch) {
                try {
                  fallbackProjects = JSON.parse(jsonMatch[1] || jsonMatch[0]);
                } catch (parseError) {
                  console.error("Secondary parsing failed:", parseError);
                }
              }
            }
          }
        }
    
        setError(errorMessage);
        if (fallbackProjects) {
          setProjects(fallbackProjects);
        } else {
          // Use mock data if parsing completely fails
          setProjects(getMockProjectData(technicalSkills));
        }
      } finally {
        setLoading(false);
      }
    };

    const getMockProjectData = (skills) => {
      return [
        {
          category: "Beginner",
          projects: [
            {
              title: "Personal Portfolio Website",
              description: "Build a responsive portfolio website to showcase your projects and skills.",
              steps: [
                "Plan your website structure",
                "Create HTML skeleton",
                "Add CSS styling",
                "Make it responsive",
                "Deploy to Firebase Hosting"
              ],
              skills: ["HTML", "CSS", "JavaScript"],
              learningObjectives: [
                "Basic web development",
                "Responsive design principles",
                "Deployment process"
              ],
              estimatedTime: "3-5 days",
              resources: [
                { name: "MDN Web Docs", url: "https://developer.mozilla.org" },
                { name: "Firebase Hosting Guide", url: "https://firebase.google.com/docs/hosting" }
              ]
            },
            // Add more mock projects as needed
          ]
        },
        // Add intermediate and expert categories
      ];
    };
    
    // Handle refresh to get new project ideas
    const handleRefresh = () => {
      setRefreshCounter(prev => prev + 1); // Increment refresh counter to force cache miss
    };

    // Handle selecting a project to view details
    const handleSelectProject = (category, project) => {
      setSelectedProject({
        ...project,
        difficulty: category
      });
    };

    // Handle going back to project list
    const handleBackToList = () => {
      setSelectedProject(null);
      setSaveSuccess(false);
    };

    // Handle starting project implementation (save to Firestore)
    const handleStartImplementing = async () => {
      try {
        setSavingProject(true);
        
        const currentUser = auth.currentUser;
        if (!currentUser) {
          setError('You must be logged in to save a project.');
          setSavingProject(false);
          return;
        }

        // Create a project object to save
        const projectToSave = {
          name: selectedProject.title,
          description: selectedProject.description,
          implementationSteps: selectedProject.steps,
          difficulty: selectedProject.difficulty,
          skills: selectedProject.skills,
          startDate: new Date().toISOString(),
          status: 'in-progress',
          estimatedTime: selectedProject.estimatedTime,
          learningObjectives: selectedProject.learningObjectives,
          resources: selectedProject.resources
        };

        // Reference to the user profile document
        const userProfileRef = doc(db, 'userProfiles', currentUser.uid);
        
        // Update the user profile with the current project
        await setDoc(userProfileRef, {
          currentProject: projectToSave
        }, { merge: true });
        
        setSaveSuccess(true);
        console.log("Project saved successfully!");
      } catch (err) {
        console.error("Error saving project:", err);
        setError(`Failed to save project: ${err.message}`);
      } finally {
        setSavingProject(false);
      }
    };

    // Map difficulty levels to appropriate colors
    const difficultyColors = {
      Beginner: "bg-green-100 text-green-800 border-green-200",
      Intermediate: "bg-yellow-100 text-yellow-800 border-yellow-200",
      Expert: "bg-red-100 text-red-800 border-red-200"
    };

    const difficultyBadgeColors = {
      Beginner: "bg-green-500 text-white",
      Intermediate: "bg-yellow-500 text-white",
      Expert: "bg-red-500 text-white"
    };

    return (
      <div className="container mx-auto p-4 bg-white rounded-lg shadow-md">
        {selectedProject ? (
          // Project Detail View
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            {/* Header with back button */}
            <div className="bg-gray-50 p-4 flex justify-between items-center border-b">
              <button 
                onClick={handleBackToList}
                className="flex items-center text-gray-600 hover:text-gray-900"
              >
                <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Projects
              </button>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${difficultyBadgeColors[selectedProject.difficulty]}`}>
                {selectedProject.difficulty}
              </span>
            </div>
            
            {/* Project content */}
            <div className="p-6">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">{selectedProject.title}</h2>
              
              {saveSuccess && (
                <div className="mb-6 p-4 bg-green-100 border border-green-300 text-green-700 rounded-md flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Project saved successfully! It's now available in your profile as your current project.
                </div>
              )}
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-2 text-gray-700">Description</h3>
                <p className="text-gray-600">{selectedProject.description}</p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-700">Skills Used</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedProject.skills.map((skill, index) => (
                      <span key={index} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-700">Estimated Time</h3>
                  <div className="flex items-center text-gray-600">
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {selectedProject.estimatedTime}
                  </div>
                </div>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-2 text-gray-700">Learning Objectives</h3>
                <ul className="list-disc pl-5 space-y-1 text-gray-600">
                  {selectedProject.learningObjectives.map((objective, index) => (
                    <li key={index}>{objective}</li>
                  ))}
                </ul>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-2 text-gray-700">Implementation Steps</h3>
                <div className="border border-gray-200 rounded-md overflow-hidden">
                  {selectedProject.steps.map((step, index) => (
                    <div 
                      key={index} 
                      className={`p-4 ${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'} ${index !== 0 ? 'border-t border-gray-200' : ''}`}
                    >
                      <div className="flex">
                        <div className="flex-shrink-0 mr-3">
                          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-bold">
                            {index + 1}
                          </div>
                        </div>
                        <div>
                          <p className="text-gray-700">{step}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-2 text-gray-700">Helpful Resources</h3>
                <div className="space-y-2">
                  {selectedProject.resources.map((resource, index) => (
                    <div key={index} className="flex items-center text-gray-600">
                      <svg className="w-5 h-5 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                      {resource.url ? (
                        <a 
                          href={resource.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {resource.name}
                        </a>
                      ) : (
                        <span>{resource.name}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="mt-8">
                <button 
                  onClick={handleStartImplementing}
                  disabled={savingProject || saveSuccess}
                  className={`px-6 py-3 rounded-md text-white font-semibold shadow-md w-full md:w-auto ${
                    saveSuccess 
                      ? 'bg-green-500 cursor-default' 
                      : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {savingProject ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Saving...
                    </span>
                  ) : saveSuccess ? (
                    <span className="flex items-center justify-center">
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Project Started
                    </span>
                  ) : (
                    'Start Implementing'
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          // Project List View
          <>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Project Suggestions Based on Your Skills</h2>
            
            {error && (
              <div className="p-3 mb-4 bg-red-100 text-red-700 rounded-md">
                {error}
              </div>
            )}
            
            {/* Skills display */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">Your Skills:</h3>
              <div className="flex flex-wrap gap-2">
                {technicalSkills.map((skill, index) => (
                  <span key={index} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
            
            {/* Job-based recommendations indicator */}
            {relevantJobData.length > 0 && (
              <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md">
                <p className="text-green-700">
                  <span className="font-medium">✓ Enhanced recommendations</span>: Project suggestions are based on {relevantJobData.length} relevant job postings matching your skills.
                </p>
              </div>
            )}
            
            <button 
              onClick={handleRefresh} 
              disabled={loading} 
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 mb-6"
            >
              {loading ? "Loading..." : "Refresh Project Ideas"}
            </button>

            {loading && <SkeletonLoader />}

            {projects && (
              <div className="space-y-6">
                {projects.map((category, index) => (
                  <div key={index} className={`border ${category.category === "Beginner" ? "border-green-200" : category.category === "Intermediate" ? "border-yellow-200" : "border-red-200"} rounded-lg overflow-hidden`}>
                    <div className={`p-4 ${category.category === "Beginner" ? "bg-green-50" : category.category === "Intermediate" ? "bg-yellow-50" : "bg-red-50"} border-b ${category.category === "Beginner" ? "border-green-200" : category.category === "Intermediate" ? "border-yellow-200" : "border-red-200"}`}>
                      <h3 className={`text-xl font-bold ${
                        category.category === "Beginner" 
                          ? "text-green-800" 
                          : category.category === "Intermediate" 
                            ? "text-yellow-800" 
                            : "text-red-800"
                      }`}>
                        {category.category} Projects
                      </h3>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-gray-200">
                      {category.projects.map((project, idx) => (
                        <div key={idx} className="p-4 hover:bg-gray-50 transition-colors">
                          <h4 className="font-semibold text-lg mb-2">{project.title}</h4>
                          <p className="text-gray-600 mb-4 line-clamp-3">{project.description}</p>
                          
                          <div className="flex flex-wrap gap-2 mb-4">
                            {project.skills.slice(0, 4).map((skill, skillIdx) => (
                              <span key={skillIdx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                                {skill}
                              </span>
                            ))}
                            {project.skills.length > 4 && (
                              <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs font-medium">
                              +{project.skills.length - 4} more
                              </span>
                              )}
                          </div>
                          
                          <div className="flex items-center text-gray-600 text-sm mb-4">
                            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            {project.estimatedTime}
                          </div>
                          
                          <button
                            onClick={() => handleSelectProject(category.category, project)}
                            className="mt-2 w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 hover:text-gray-900 hover:border-gray-400 transition-all font-medium text-sm flex items-center justify-center"
                          >
                            View Details
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
        
        {/* Debug panel - only shown in development */}
        {/* {process.env.NODE_ENV === 'development' && debugInfo && (
          <div className="mt-8 p-4 border border-gray-300 rounded-md bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Debug Info:</h3>
            <pre className="text-xs overflow-auto max-h-40">
              {JSON.stringify(debugInfo, null, 2)}
            </pre>
          </div>
        )} */}
      </div>
    );
};

export default ProjectSuggester;