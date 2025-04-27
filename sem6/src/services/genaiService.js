import { GoogleGenerativeAI } from "@google/generative-ai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";

// Utility function to initialize Google Generative AI
export const initializeGenAI = () => {
  const genAI = new GoogleGenerativeAI(process.env.REACT_APP_GOOGLE_API_KEY);
  return genAI;
};

// Create Langchain chat model wrapper
export const createChatModel = () => {
  return new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    temperature: 0.2,
    apiKey: process.env.REACT_APP_GOOGLE_API_KEY,
    maxRetries: 2, // Limit retries to avoid excessive requests
  });
};

// Prompt templates remain unchanged
export const careerPathPromptTemplate = new PromptTemplate({
  template: `Analyze career paths for a job seeker with the following technical skills: {userSkills}

Job roles and their required skills:
{jobRolesSkills}

Provide a detailed analysis:
1. Identify best-matching career paths
2. Calculate match percentage for each path
3. List matched and missing skills

Respond strictly in the following JSON format:
{{
  "careerPaths": [
    {{
      "role": "Job Role Name",
      "matchPercentage": 85,
      "matchedSkills": ["Skill1", "Skill2"],
      "missingSkills": ["Skill3", "Skill4"]
    }}
  ]
}}

Ensure results are sorted by matchPercentage in descending order.`,
  inputVariables: ["userSkills", "jobRolesSkills"]
});

export const courseRecommendationPromptTemplate = new PromptTemplate({
  template: `Recommend courses for learning these skills for a {role} career:
Missing skills: {missingSkills}

For each skill, provide courses at three levels:
1. Beginner
2. Intermediate
3. Advanced

Provide detailed course information in this JSON format:
{{
  "beginner": [
    {{
      "skill": "Skill Name",
      "title": "Course Title",
      "provider": "Provider Name",
      "duration": "X weeks",
      "rating": 4.5,
      "description": "Course description"
    }}
  ],
  "intermediate": [],
  "advanced": []
}}

Ensure courses are current, diverse, and from reputable providers.`,
  inputVariables: ["role", "missingSkills"]
});

// Cache for storing weighted skills to avoid redundant API calls
const skillWeightCache = new Map();

// OPTIMIZED: Career path analysis without excessive API calls
export const analyzeCareerPaths = async (userSkills, jobRolesSkills) => {
  try {
    // Use local calculation instead of API calls for skill weights
    const weightedJobSkills = calculateSkillWeights(jobRolesSkills);
    
    // Calculate career paths with weighted matching
    const careerPaths = [];
    
    for (const [role, weightedSkills] of Object.entries(weightedJobSkills)) {
      const matchedSkills = [];
      const missingSkills = [];
      let weightedMatchScore = 0;
      let totalWeight = 0;
      
      // Calculate total weight for normalization
      weightedSkills.forEach(skill => {
        totalWeight += skill.weight;
      });
      
      // Calculate matched and missing skills with their weights
      weightedSkills.forEach(skillObj => {
        const { skill, weight } = skillObj;
        if (userSkills.includes(skill)) {
          matchedSkills.push(skill);
          weightedMatchScore += weight;
        } else {
          missingSkills.push(skill);
        }
      });
      
      // Calculate match percentage (normalized to 100%)
      const matchPercentage = Math.round((weightedMatchScore / totalWeight) * 100);
      
      careerPaths.push({
        role,
        matchPercentage,
        matchedSkills,
        missingSkills,
        skillGap: missingSkills.length,
      });
    }
    
    // Sort career paths by match percentage (highest first)
    careerPaths.sort((a, b) => b.matchPercentage - a.matchPercentage);
    
    return {
      careerPaths,
      topMatch: careerPaths.length > 0 ? careerPaths[0] : null,
    };
  } catch (error) {
    console.error("Error analyzing career paths:", error);
    throw error;
  }
};

// OPTIMIZED: Function to calculate skill weights locally without API calls
const calculateSkillWeights = (jobRolesSkills) => {
  const weightedJobSkills = {};
  
  // Predefined weight templates based on common role patterns
  const weightTemplates = {
    technical: {
      primary: 25,     // Core technical skills
      secondary: 15,   // Supporting technical skills
      soft: 10,        // Soft skills
      tools: 12        // Tools and platforms
    },
    management: {
      primary: 20,     // Core management skills
      technical: 15,   // Technical understanding
      soft: 25,        // Leadership/soft skills
      tools: 10        // Tools and platforms
    }
  };
  
  for (const [role, skills] of Object.entries(jobRolesSkills)) {
    // Check if we have this role cached already
    if (skillWeightCache.has(role)) {
      weightedJobSkills[role] = skillWeightCache.get(role);
      continue;
    }
    
    // Determine if role is more technical or management focused
    const roleType = role.toLowerCase().includes('manager') || 
                     role.toLowerCase().includes('lead') ? 
                     'management' : 'technical';
    
    const template = weightTemplates[roleType];
    const weightedSkills = [];
    
    // Basic classification of skills into categories
    const skillCategories = {
      primary: skills.slice(0, Math.ceil(skills.length * 0.3)), // Top 30% are primary
      secondary: skills.slice(Math.ceil(skills.length * 0.3), Math.ceil(skills.length * 0.7)),
      soft: skills.filter(s => 
        ['communication', 'teamwork', 'leadership', 'problem solving', 'agile'].some(
          term => s.toLowerCase().includes(term)
        )
      ),
      tools: skills.filter(s => 
        ['tool', 'platform', 'framework', 'library'].some(
          term => s.toLowerCase().includes(term)
        )
      )
    };
    
    // Assign weights based on categories and ensure total is 100
    const categorizedSkills = new Set();
    let remainingWeight = 100;
    
    // First assign weights to categorized skills
    for (const [category, categorySkills] of Object.entries(skillCategories)) {
      categorySkills.forEach(skill => {
        if (!categorizedSkills.has(skill)) {
          weightedSkills.push({
            skill,
            weight: template[category]
          });
          categorizedSkills.add(skill);
          remainingWeight -= template[category];
        }
      });
    }
    
    // Distribute remaining weight to uncategorized skills
    const uncategorizedSkills = skills.filter(skill => !categorizedSkills.has(skill));
    const equalWeight = uncategorizedSkills.length > 0 ? 
                       Math.max(1, Math.floor(remainingWeight / uncategorizedSkills.length)) : 0;
    
    uncategorizedSkills.forEach(skill => {
      weightedSkills.push({
        skill,
        weight: equalWeight
      });
      remainingWeight -= equalWeight;
    });
    
    // If there's any remaining weight, add it to the first skill
    if (remainingWeight > 0 && weightedSkills.length > 0) {
      weightedSkills[0].weight += remainingWeight;
    }
    
    // Store in cache and return
    skillWeightCache.set(role, weightedSkills);
    weightedJobSkills[role] = weightedSkills;
  }
  
  return weightedJobSkills;
};

// OPTIMIZED: Batch course recommendations in a single API call for multiple skills
export const getCourseRecommendations = async (role, missingSkills) => {
  // If there are no missing skills, return empty results
  if (!missingSkills || missingSkills.length === 0) {
    return {
      beginner: [],
      intermediate: [],
      advanced: []
    };
  }
  
  // Limit the number of missing skills to request at once
  const limitedSkills = missingSkills.slice(0, 5); // Only get courses for top 5 missing skills
  
  try {
    const chatModel = createChatModel();
    
    // Single API call for multiple skills with batched approach
    const prompt = `
    As a technical career advisor, recommend courses to help someone become a "${role}".
    They need to learn these specific skills: ${JSON.stringify(limitedSkills)}
    
    For each skill (limited to the most important ones), recommend:
    1. One beginner-level course
    2. One intermediate-level course
    3. One advanced-level course
    
    Return ONLY a JSON object with this structure:
    {
      "beginner": [
        {"title": "Course Title", "platform": "Platform Name", "description": "Brief description", "skill": "Related skill", "duration": "X weeks", "url": "course-url"}
      ],
      "intermediate": [
        {"title": "Course Title", "platform": "Platform Name", "description": "Brief description", "skill": "Related skill", "duration": "X weeks", "url": "course-url"}
      ],
      "advanced": [
        {"title": "Course Title", "platform": "Platform Name", "description": "Brief description", "skill": "Related skill", "duration": "X weeks", "url": "course-url"}
      ]
    }
    `;
    
    // Call Gemini API once for all skills
    const result = await chatModel.invoke(prompt);
    const responseText = result.content;
    
    // Extract JSON from response
    const jsonMatch = responseText.match(/\{.*\}/s);
    if (jsonMatch) {
      const courseRecommendations = JSON.parse(jsonMatch[0]);
      return courseRecommendations;
    } else {
      console.log("Could not parse course recommendation response:", responseText);
      return {
        beginner: [],
        intermediate: [],
        advanced: []
      };
    }
  } catch (error) {
    console.error("Error getting course recommendations:", error);
    return {
      beginner: [],
      intermediate: [],
      advanced: []
    };
  }
};

// Optional: Add debounced chain execution to prevent rapid sequential requests
let debounceTimeout = null;
export const debouncedCareerAdvisorChain = (chatModel, input, callback, delay = 500) => {
  if (debounceTimeout) {
    clearTimeout(debounceTimeout);
  }
  
  debounceTimeout = setTimeout(async () => {
    try {
      const chain = new LLMChain({
        llm: chatModel,
        prompt: new PromptTemplate({
          template: "Analyze and advise on career development: {input}",
          inputVariables: ["input"]
        }),
        verbose: false // Disable verbose logging to reduce console noise
      });
      
      const result = await chain.call({ input });
      callback(null, result);
    } catch (error) {
      callback(error);
    }
  }, delay);
};