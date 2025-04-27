import React, { useState, useEffect, useRef } from 'react';
import { doc, getDoc } from 'firebase/firestore';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { db, auth } from '../firebase-config';
import { HumanMessage, SystemMessage, AIMessage } from '@langchain/core/messages';

const ProjectAssistant = () => {
  const [loading, setLoading] = useState(true);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [currentProject, setCurrentProject] = useState(null);
  const [suggestedPrompts, setSuggestedPrompts] = useState([]);
  const [responseLoading, setResponseLoading] = useState(false);
  const chatContainerRef = useRef(null);

  // Fetch current project data
  useEffect(() => {
    const fetchCurrentProject = async () => {
      try {
        setLoading(true);
        const user = auth.currentUser;
        if (!user) return;

        const userProfileRef = doc(db, 'userProfiles', user.uid);
        const userProfileSnap = await getDoc(userProfileRef);
        
        if (userProfileSnap.exists() && userProfileSnap.data().currentProject) {
          const projectData = userProfileSnap.data().currentProject;
          setCurrentProject(projectData);
          
          // Generate initial message and suggestions based on project
          generateInitialSuggestions(projectData);
        }
      } catch (error) {
        console.error("Error fetching project data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCurrentProject();
  }, []);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Format AI response for better readability
  const formatResponse = (text) => {
    // Handle code blocks
    text = text.replace(/```([a-z]*)([\s\S]*?)```/g, (match, language, code) => {
      return `<div class="code-block"><div class="code-header">${language || 'code'}</div><pre>${code}</pre></div>`;
    });
    
    // Handle lists
    text = text.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.+<\/li>)\s*(<li>.+<\/li>)/gs, '<ul>$1$2</ul>');
    
    // Handle numbered lists
    text = text.replace(/^\s*(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
    text = text.replace(/(<li>.+<\/li>)\s*(<li>.+<\/li>)/gs, '<ol>$1$2</ol>');
    
    // Handle headings
    text = text.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    text = text.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
    
    // Handle bold and italic
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    
    // Handle paragraphs
    text = text.replace(/\n\n/g, '</p><p>');
    text = `<p>${text}</p>`;
    
    return text;
  };

  // Generate initial suggestions based on project data
  const generateInitialSuggestions = async (projectData) => {
    try {
      const model = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0.2,
        apiKey: process.env.REACT_APP_GOOGLE_API_KEY,
        maxRetries: 2,
      });

      const systemPrompt = `You are a personalized project assistant for a learning project. 
      The user is working on the following project:
      Name: ${projectData.name}
      Description: ${projectData.description}
      Difficulty: ${projectData.difficulty}
      Skills involved: ${projectData.skills.join(', ')}
      Implementation steps: ${projectData.implementationSteps.join('\n')}
      Learning objectives: ${projectData.learningObjectives.join('\n')}
      
      Your goal is to provide helpful, personalized guidance. Generate:
      1. A welcome message to the user that's encouraging and acknowledges their project
      2. Three specific suggested prompts they might want to ask you about their project`;

      const response = await model.invoke([
        new SystemMessage(systemPrompt),
      ]);

      // Parse the response to extract welcome message and suggestions
      const responseText = response.content;
      
      // Better parsing logic
      const welcomeMessageRegex = /(.+?)(?=\d\.\s|\n\d\.\s|$)/s;
      const welcomeMatch = welcomeMessageRegex.exec(responseText);
      
      let welcomeMessage = "Hello! I'm your project assistant. How can I help you with your project today?";
      if (welcomeMatch && welcomeMatch[1]) {
        welcomeMessage = welcomeMatch[1].trim();
      }
      
      // Extract suggested prompts with regex
      const promptRegex = /\d\.\s+(.+?)(?=\d\.\s|\n\d\.\s|$)/gs;
      const promptMatches = [...responseText.matchAll(promptRegex)];
      
      const promptSuggestions = promptMatches.map(match => match[1].trim()).filter(Boolean);
      
      setMessages([{ type: 'assistant', content: welcomeMessage, formatted: formatResponse(welcomeMessage) }]);
      
      if (promptSuggestions.length > 0) {
        setSuggestedPrompts(promptSuggestions);
      } else {
        setSuggestedPrompts([
          "I'm stuck on the first step. Can you help?",
          "What resources do you recommend for this project?",
          "Can you break down the implementation steps further?"
        ]);
      }
    } catch (error) {
      console.error("Error generating initial suggestions:", error);
      setMessages([{ 
        type: 'assistant', 
        content: "Hello! I'm your project assistant. How can I help you with your project today?",
        formatted: formatResponse("Hello! I'm your project assistant. How can I help you with your project today?")
      }]);
      setSuggestedPrompts([
        "I'm stuck on the first step. Can you help?",
        "What resources do you recommend for this project?",
        "Can you break down the implementation steps further?"
      ]);
    }
  };

  // Handle sending a message
  const handleSendMessage = async (message = inputMessage) => {
    if (!message.trim()) return;

    // Add user message to chat
    const userMessage = { type: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setResponseLoading(true);

    try {
      const model = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0.2,
        apiKey: process.env.REACT_APP_GOOGLE_API_KEY,
        maxRetries: 2,
      });

      // Create system prompt with context for the AI
      const systemPrompt = `You are a personalized project assistant helping with a learning project.
      
      Current project details:
      Name: ${currentProject.name}
      Description: ${currentProject.description}
      Difficulty: ${currentProject.difficulty}
      Skills: ${currentProject.skills.join(', ')}
      Implementation steps: ${currentProject.implementationSteps.join('\n')}
      Current status: ${currentProject.status}
      Learning objectives: ${currentProject.learningObjectives.join('\n')}
      Resources: ${currentProject.resources.join('\n')}
      
      Based on these details, provide personalized, specific guidance to help the user. Be encouraging but practical.
      Structure your response with clear sections, using markdown formatting for readability.
      After answering, suggest 2-3 follow-up questions related to their current situation.
      Format these questions as a separate section titled "Follow-up Questions:"`;

      // Convert chat history to messages format
      const chatHistory = messages
        .filter(msg => !msg.isLoading)
        .map(msg => 
          msg.type === 'user' 
            ? new HumanMessage(msg.content) 
            : new AIMessage(msg.content)
        );

      // Add the new user message
      chatHistory.push(new HumanMessage(message));

      // Start with system message, then add chat history
      const allMessages = [new SystemMessage(systemPrompt), ...chatHistory];

      // Get response from AI
      const response = await model.invoke(allMessages);
      
      // Parse response for main content and suggestions
      const responseText = response.content;
      
      // Improved parsing with regex
      const followUpSectionRegex = /(?:Follow-up Questions:|Next steps:|Suggested questions:)([\s\S]+)$/i;
      const followUpMatch = responseText.match(followUpSectionRegex);
      
      let mainResponse = responseText;
      let followUpSection = '';
      let newSuggestions = [];
      
      if (followUpMatch) {
        mainResponse = responseText.substring(0, followUpMatch.index).trim();
        followUpSection = followUpMatch[1].trim();
        
        // Extract questions with regex
        const questionRegex = /(?:^\d+[\.\)]\s+|^[-*]\s+)(.+?)(?=$|^\d+[\.\)]\s+|^[-*]\s+)/gms;
        const questions = [...followUpSection.matchAll(questionRegex)];
        
        if (questions.length > 0) {
          newSuggestions = questions.map(q => q[1].trim()).filter(Boolean);
        } else {
          // Fallback if numbered/bulleted format not found
          newSuggestions = followUpSection
            .split(/\n+/)
            .map(s => s.trim())
            .filter(s => s.length > 0 && s.length < 100 && !s.match(/follow-up|questions|suggested|next steps/i));
        }
      }

      // Add formatted response
      const formattedResponse = formatResponse(mainResponse);
      
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: mainResponse,
        formatted: formattedResponse
      }]);
      
      // Update suggestions if we got new ones
      if (newSuggestions.length > 0) {
        setSuggestedPrompts(newSuggestions);
      }
    } catch (error) {
      console.error("Error getting AI response:", error);
      // Add error message
      setMessages(prev => [
        ...prev, 
        { 
          type: 'assistant', 
          content: "I'm sorry, I couldn't process your request. Please try again.",
          formatted: formatResponse("I'm sorry, I couldn't process your request. Please try again.")
        }
      ]);
    } finally {
      setResponseLoading(false);
    }
  };

  // Handle clicking a suggestion
  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
    handleSendMessage(suggestion);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="relative w-16 h-16">
          <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-300 rounded-full animate-pulse"></div>
          <div className="absolute top-2 left-2 w-12 h-12 border-4 border-blue-500 rounded-full animate-spin"></div>
          <div className="absolute top-4 left-4 w-8 h-8 border-4 border-blue-700 rounded-full animate-ping"></div>
        </div>
      </div>
    );
  }

  if (!currentProject) {
    return (
      <div className="p-4 text-center">
        <p>No active project found. Please select a project to get started.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[600px] border rounded-lg shadow-lg bg-gradient-to-b from-gray-50 to-white">
      {/* Project header */}
      <div className="border-b p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-t-lg">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
          <h2 className="text-lg font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            {currentProject.name}
          </h2>
        </div>
        <div className="flex items-center mt-1 text-sm text-gray-500">
          <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <span>AI Project Assistant</span>
        </div>
      </div>
      
      {/* Chat messages */}
      <div 
        ref={chatContainerRef}
        className="flex-1 p-4 overflow-y-auto space-y-4 bg-opacity-20 bg-blue-50"
      >
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`max-w-[80%] p-3 rounded-lg shadow-sm ${
                message.type === 'user' 
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-none' 
                  : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none'
              }`}
            >
              {message.type === 'assistant' && message.formatted ? (
                <div 
                  className="whitespace-pre-wrap assistant-message" 
                  dangerouslySetInnerHTML={{ __html: message.formatted }}
                />
              ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
              )}
            </div>
          </div>
        ))}
        
        {/* Robot typing animation when loading */}
        {responseLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 p-3 rounded-lg rounded-bl-none shadow-sm max-w-[80%]">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-200"></div>
                </div>
                <span className="text-xs text-gray-400 italic">AI Assistant is processing...</span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Suggested prompts */}
      {suggestedPrompts.length > 0 && (
        <div className="p-3 border-t flex flex-wrap gap-2 bg-gray-50">
          {suggestedPrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(prompt)}
              className="text-sm bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 
                border border-blue-200 rounded-full px-3 py-1 text-gray-700 transition-all 
                hover:shadow-md flex items-center space-x-1"
            >
              <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
              <span>{prompt}</span>
            </button>
          ))}
        </div>
      )}
      
      {/* Input area */}
      <div className="p-4 border-t bg-gradient-to-r from-gray-50 to-blue-50 rounded-b-lg">
        <div className="flex gap-2 items-center">
          <div className="relative flex-1">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask about your project..."
              className="w-full p-3 pl-10 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300 bg-white shadow-sm"
            />
            <svg className="w-5 h-5 text-gray-400 absolute left-3 top-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <button
            onClick={() => handleSendMessage()}
            disabled={!inputMessage.trim() || responseLoading}
            className={`p-3 rounded-lg transition-all ${
              !inputMessage.trim() || responseLoading
                ? 'bg-gray-300 text-gray-500'
                : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg'
            }`}
          >
            {responseLoading ? (
              <svg className="w-5 h-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            )}
          </button>
        </div>
      </div>
      
      {/* CSS for formatted messages */}
      <style jsx>{`
        .assistant-message h1 {
          font-size: 1.5rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
          color: #2563eb;
        }
        .assistant-message h2 {
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
          color: #3b82f6;
        }
        .assistant-message h3 {
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
          color: #60a5fa;
        }
        .assistant-message p {
          margin-bottom: 0.75rem;
        }
        .assistant-message ul, .assistant-message ol {
          margin-left: 1.5rem;
          margin-bottom: 0.75rem;
        }
        .assistant-message li {
          margin-bottom: 0.25rem;
        }
        .assistant-message .code-block {
          background-color: #f8fafc;
          border-radius: 0.375rem;
          margin-bottom: 0.75rem;
          overflow: hidden;
          border: 1px solid #e2e8f0;
        }
        .assistant-message .code-header {
          background-color: #e2e8f0;
          padding: 0.25rem 0.75rem;
          font-size: 0.75rem;
          color: #475569;
          font-family: monospace;
        }
        .assistant-message pre {
          padding: 0.75rem;
          overflow-x: auto;
          font-family: monospace;
          font-size: 0.875rem;
          line-height: 1.5;
        }
        .assistant-message strong {
          font-weight: 600;
          color: #1e40af;
        }
        .assistant-message em {
          font-style: italic;
          color: #4b5563;
        }
      `}</style>
    </div>
  );
};

export default ProjectAssistant;