// useProjectAssistantContext.js
import { useState, useEffect } from 'react';
import { doc, getDoc } from 'firebase/firestore';
import { db, auth } from '../firebase/config'; // Adjust path as needed

export function useProjectAssistantContext() {
  const [projectContext, setProjectContext] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProjectContext = async () => {
      try {
        setLoading(true);
        const user = auth.currentUser;
        if (!user) {
          setError("User not authenticated");
          return;
        }

        const userProfileRef = doc(db, 'userProfiles', user.uid);
        const userProfileSnap = await getDoc(userProfileRef);
        
        if (!userProfileSnap.exists()) {
          setError("User profile not found");
          return;
        }
        
        const userData = userProfileSnap.data();
        if (!userData.currentProject) {
          setError("No active project found");
          return;
        }

        // Format project data into a context string for the AI
        const project = userData.currentProject;
        const contextString = `
Current Project:
Name: ${project.name}
Description: ${project.description}
Difficulty Level: ${project.difficulty}
Skills Required: ${project.skills.join(', ')}
Start Date: ${new Date(project.startDate).toLocaleDateString()}
Status: ${project.status}
Estimated Time: ${project.estimatedTime}

Implementation Steps:
${project.implementationSteps.map((step, idx) => `${idx + 1}. ${step}`).join('\n')}

Learning Objectives:
${project.learningObjectives.map((obj, idx) => `${idx + 1}. ${obj}`).join('\n')}

Recommended Resources:
${project.resources.map((res, idx) => `${idx + 1}. ${res}`).join('\n')}
        `.trim();

        setProjectContext({
          raw: project,
          formatted: contextString
        });
        setError(null);
      } catch (err) {
        console.error("Error fetching project context:", err);
        setError("Failed to load project data");
      } finally {
        setLoading(false);
      }
    };

    fetchProjectContext();
  }, []);

  return { projectContext, loading, error };
}