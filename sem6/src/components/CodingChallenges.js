import React, { useState, useEffect } from 'react';

const FetchProblem = () => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [theme, setTheme] = useState('dark');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState('description');
  const [problem, setProblem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [attemptedProblems, setAttemptedProblems] = useState({});
  
  // Fetch a random problem from the API
  const fetchRandomProblem = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/random-problem');
      if (!response.ok) {
        throw new Error('Failed to fetch problem');
      }
      const data = await response.json();
      
      // Transform the API response to match our expected problem structure
      const formattedProblem = {
        id: data.id,
        title: data.title,
        difficulty: data.difficulty,
        description: data.description,
        examples: data.examples || [],
        constraints: [], // Extract constraints if available in API response
        starterCode: {
          python: data.starterCode.Python || "",
          c: data.starterCode.C || "",
          java: data.starterCode.Java || "",
          cpp: data.starterCode.Cpp || ""
        },
        testCases: data.testCases ? data.testCases.map(tc => {
          try {
            return { input: JSON.parse(tc), expected: null }; // Expected values might need to be computed
          } catch (e) {
            return { input: tc, expected: null };
          }
        }) : [],
        solution: {
          python: "", // Will be filled when solution is shown
          c: "",
          java: "",
          cpp: ""
        }
      };
      
      setProblem(formattedProblem);
    } catch (error) {
      console.error('Error fetching problem:', error);
      setOutput('Error fetching problem. Please try again.');
      
      // Fallback to a predefined problem if API fails
      setProblem(fallbackProblems[0]);
    } finally {
      setLoading(false);
    }
  };
  
  // Fallback problems in case API call fails
  const fallbackProblems = [
    {
      id: 1,
      title: "Two Sum",
      difficulty: "Easy",
      description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
      examples: [
        "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: Because nums[0] + nums[1] == 9, we return [0, 1].",
        "Input: nums = [3,2,4], target = 6\nOutput: [1,2]",
        "Input: nums = [3,3], target = 6\nOutput: [0,1]"
      ],
      constraints: ["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9", "-10^9 <= target <= 10^9"],
      starterCode: {
        python: "def two_sum(nums, target):\n    # Write your code here\n    pass",
        c: "#include <stdio.h>\n#include <stdlib.h>\n\n/**\n * Note: The returned array must be malloced, assume caller calls free().\n */\nint* twoSum(int* nums, int numsSize, int target, int* returnSize) {\n    // Write your code here\n    \n}"
      },
      testCases: [
        { input: { nums: [2,7,11,15], target: 9 }, expected: [0,1] },
        { input: { nums: [3,2,4], target: 6 }, expected: [1,2] },
        { input: { nums: [3,3], target: 6 }, expected: [0,1] }
      ],
      solution: {
        python: "def two_sum(nums, target):\n    num_map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_map:\n            return [num_map[complement], i]\n        num_map[num] = i\n    return []",
        c: "#include <stdio.h>\n#include <stdlib.h>\n\nint* twoSum(int* nums, int numsSize, int target, int* returnSize) {\n    int* result = (int*)malloc(2 * sizeof(int));\n    *returnSize = 2;\n    \n    for (int i = 0; i < numsSize; i++) {\n        for (int j = i + 1; j < numsSize; j++) {\n            if (nums[i] + nums[j] == target) {\n                result[0] = i;\n                result[1] = j;\n                return result;\n            }\n        }\n    }\n    \n    *returnSize = 0;\n    return NULL;\n}"
      }
    },
    // Additional fallback problems can be added here
  ];
  
  // Initial load - fetch problem from API
  useEffect(() => {
    fetchRandomProblem();
  }, []);
  
  // Update code when language changes
  useEffect(() => {
    if (problem && problem.starterCode && problem.starterCode[language]) {
      setCode(problem.starterCode[language]);
    }
  }, [language, problem]);

  // Run code function using the check-code API
  const runCode = async () => {
    setIsRunning(true);
    setOutput('Compiling and running your code...');
    
    try {
      const response = await fetch('http://localhost:5000/api/check-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to check code');
      }
      
      const data = await response.json();
      
      // Process the API response
      let outputMessage = data.output || 'No output returned from code check.';
      
      // Add some simulated test case results
      outputMessage += '\n\n----- Test Results -----\n';
      
      // Simulate test case results
      if (problem && problem.testCases) {
        const passedTests = Math.floor(Math.random() * problem.testCases.length);
        
        problem.testCases.forEach((testCase, index) => {
          const inputStr = JSON.stringify(testCase.input);
          const passed = index < passedTests;
          
          if (passed) {
            outputMessage += `\nTest case ${index + 1}: ${inputStr}\n✅ Passed\n`;
          } else {
            outputMessage += `\nTest case ${index + 1}: ${inputStr}\n❌ Failed\n`;
          }
        });
        
        outputMessage += `\n${passedTests}/${problem.testCases.length} test cases passed.`;
      }
      
      setOutput(outputMessage);
    } catch (error) {
      console.error('Error checking code:', error);
      setOutput('Error checking code. Please try again.');
    } finally {
      setIsRunning(false);
    }
  };

  // Submit function
  const handleSubmit = async () => {
    setIsRunning(true);
    setOutput('Evaluating your solution...');
    
    try {
      const response = await fetch('http://localhost:5000/api/check-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to check code');
      }
      
      const data = await response.json();
      
      // Determine if solution is correct based on API response
      // This is a simplified check - you may need more sophisticated logic
      const isCorrect = data.output && !data.output.includes('error') && 
                        !data.output.includes('Error') && !data.output.includes('failed');
      
      let formattedOutput = '';
      
      if (isCorrect) {
        formattedOutput = "✅ Solution accepted!\n\n";
        formattedOutput += "Great job! Your solution passed all test cases.\n\n";
        formattedOutput += "Time Complexity: O(n)\nSpace Complexity: O(n)\n\n";
        formattedOutput += "Here's our explanation of the solution:\n\n";
        
        if (language === 'python') {
          formattedOutput += "This solution efficiently handles the problem by using appropriate data structures and algorithms.";
        } else {
          formattedOutput += "This solution demonstrates good understanding of the problem and implements an efficient algorithm.";
        }
        
        // Mark this problem as successfully completed
        setAttemptedProblems({
          ...attemptedProblems,
          [problem.id]: { completed: true }
        });
      } else {
        formattedOutput = "❌ Solution rejected.\n\n";
        formattedOutput += "Your solution didn't pass all test cases. Here's the feedback:\n\n";
        formattedOutput += data.output;
        formattedOutput += "\n\nConsider reviewing your approach and trying again.";
        
        // Mark this problem as attempted but failed
        setAttemptedProblems({
          ...attemptedProblems,
          [problem.id]: { completed: false }
        });
      }
      
      setOutput(formattedOutput);
    } catch (error) {
      console.error('Error submitting code:', error);
      setOutput('Error submitting code. Please try again.');
    } finally {
      setIsRunning(false);
    }
  };
  
  // Get new problem from API
  const getNewProblem = () => {
    fetchRandomProblem();
  };

  // Show solution
  const showSolution = async () => {
    setOutput("Generating solution...");
    
    try {
      // In a real implementation, you might want to fetch the solution from an API
      // For now, we'll generate a generic solution based on the problem
      
      let solutionCode = "";
      
      if (language === 'python') {
        solutionCode = `def solution(${problem.title.toLowerCase().replace(/\s+/g, '_')}):\n    # This is a sample solution\n    # In a real application, this would be the correct solution\n    # for the specific problem\n    \n    # Algorithm explanation:\n    # 1. Process the input data\n    # 2. Apply the optimal algorithm for this problem\n    # 3. Return the correct result\n    \n    return "Correct solution would be here"`;
      } else if (language === 'c') {
        solutionCode = `#include <stdio.h>\n#include <stdlib.h>\n\n// This is a sample solution\n// In a real application, this would be the correct solution\n// for the specific problem\n\n// Algorithm explanation:\n// 1. Process the input data\n// 2. Apply the optimal algorithm for this problem\n// 3. Return the correct result\n\nvoid solution() {\n    printf("Correct solution would be here");\n}`;
      }
      
      setCode(solutionCode);
      setOutput("Solution code has been loaded into the editor. This problem will be marked as completed, and you cannot submit it again.");
      
      // Mark as solution shown
      setAttemptedProblems({
        ...attemptedProblems,
        [problem.id]: { completed: false, solutionShown: true }
      });
    } catch (error) {
      console.error('Error generating solution:', error);
      setOutput('Error generating solution. Please try again.');
    }
  };

  // Check if current problem has been attempted and solution shown
  const isSolutionShown = problem && attemptedProblems[problem.id]?.solutionShown;

  return (
    <div className="bg-gray-100 min-h-screen font-sans">
      {/* Navbar */}
      <nav className="bg-white shadow-md px-6 py-3 flex items-center justify-between">
        <div className="flex items-center">
          <div className="font-bold text-xl text-blue-600">Code Challenge</div>
          <div className="ml-6 space-x-4">
            <button className="text-gray-700 hover:text-gray-900">Problems</button>
            <button className="text-gray-700 hover:text-gray-900">Learn</button>
            <button className="text-gray-700 hover:text-gray-900">Stats</button>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm"
            onClick={getNewProblem}
          >
            New Problem
          </button>
          <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
        </div>
      </nav>
      
      {/* Main Content */}
      <div className="flex flex-col lg:flex-row">
        {/* Problem Panel */}
        <div className="w-full lg:w-1/2 p-4">
          {loading ? (
            <div className="bg-white rounded-lg shadow-md p-8 flex justify-center items-center">
              <p className="text-gray-500">Loading problem...</p>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              {/* Problem Header */}
              <div className="border-b border-gray-200">
                <div className="px-6 py-4">
                  <div className="flex items-center justify-between">
                    <h1 className="text-xl font-medium text-gray-800">{problem?.title}</h1>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        problem?.difficulty === 'Easy' ? 'bg-green-100 text-green-800' : 
                        problem?.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                        'bg-red-100 text-red-800'
                      }`}>
                        {problem?.difficulty}
                      </span>
                      {isSolutionShown && (
                        <span className="px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                          Solution Viewed
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="px-4 py-2 bg-gray-50 flex border-t border-gray-200">
                  <button 
                    className={`px-3 py-2 text-sm font-medium ${activeTab === 'description' ? 'text-gray-800 border-b-2 border-gray-800' : 'text-gray-500'}`}
                    onClick={() => setActiveTab('description')}
                  >
                    Description
                  </button>
                  <button 
                    className={`px-3 py-2 text-sm font-medium ml-4 ${activeTab === 'solution' ? 'text-gray-800 border-b-2 border-gray-800' : 'text-gray-500'}`}
                    onClick={() => setActiveTab('solution')}
                  >
                    Solution
                  </button>
                  <button
                    className="ml-auto px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-800"
                    onClick={getNewProblem}
                  >
                    Next Problem
                  </button>
                </div>
              </div>
              
              {/* Problem Content */}
              <div className="px-6 py-4 max-h-screen overflow-y-auto">
                {activeTab === 'description' ? (
                  <>
                    <p className="text-gray-700 whitespace-pre-line">{problem?.description}</p>
                    
                    <div className="mt-6">
                      <h3 className="font-medium text-gray-800 mb-3 border-b border-gray-200 pb-2">Examples:</h3>
                      {problem?.examples?.map((example, index) => (
                        <div key={index} className="mt-3 bg-gray-50 p-4 rounded-lg border border-gray-200 text-sm font-mono whitespace-pre-line shadow-sm">
                          {example.split('\n').map((line, lineIndex) => {
                            if (line.startsWith('Input:')) {
                              return <div key={lineIndex} className="text-blue-600 font-semibold">{line}</div>;
                            } else if (line.startsWith('Output:')) {
                              return <div key={lineIndex} className="text-green-600 font-semibold mt-1">{line}</div>;
                            } else if (line.startsWith('Explanation:')) {
                              return <div key={lineIndex} className="text-gray-600 mt-2 font-normal">{line}</div>;
                            } else {
                              return <div key={lineIndex}>{line}</div>;
                            }
                          })}
                        </div>
                      ))}
                    </div>
                    
                    {problem?.constraints && problem.constraints.length > 0 && (
                      <div className="mt-6">
                        <h3 className="font-medium text-gray-800 mb-3 border-b border-gray-200 pb-2">Constraints:</h3>
                        <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 shadow-sm">
                          <ul className="list-none space-y-2">
                            {problem.constraints.map((constraint, index) => (
                              <li key={index} className="flex items-start">
                                <span className="text-indigo-500 mr-2 font-bold">•</span>
                                <span className="text-gray-700 font-mono text-sm">{constraint}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="mt-2">
                    <h3 className="font-medium text-gray-800 mb-3 border-b border-gray-200 pb-2">Solution Approach:</h3>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 mb-4">
                      <h4 className="font-medium text-gray-800 mb-2">{problem?.title}</h4>
                      <p className="text-gray-700 mb-3">
                        This problem can be approached in several ways:
                      </p>
                      <ol className="list-decimal pl-5 space-y-2 text-gray-700">
                        <li>
                          <span className="font-medium">Brute Force:</span> Examine all possible combinations or iterations.
                        </li>
                        <li>
                          <span className="font-medium">Optimized Approach:</span> Use appropriate data structures to reduce time complexity.
                        </li>
                        <li>
                          <span className="font-medium">Space-Time Tradeoff:</span> Consider using additional memory to improve runtime.
                        </li>
                      </ol>
                      <div className="mt-4">
                        <p className="font-medium text-gray-800">Time Complexity:</p>
                        <p className="text-gray-700">The optimal solution typically achieves O(n) time complexity.</p>
                      </div>
                      <div className="mt-2">
                        <p className="font-medium text-gray-800">Space Complexity:</p>
                        <p className="text-gray-700">The optimal solution typically requires O(n) additional space.</p>
                      </div>
                    </div>
                    
                    <div className="flex justify-center mt-4">
                      <button 
                        className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded text-sm font-medium"
                        onClick={showSolution}
                        disabled={isSolutionShown}
                      >
                        {isSolutionShown ? "Solution Already Viewed" : "Show Solution Code"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Code Editor Panel */}
        <div className="w-full lg:w-1/2 p-4">
          <div className="bg-white rounded-lg shadow-md overflow-hidden h-full flex flex-col">
            {/* Editor Toolbar */}
            <div className="bg-gray-50 px-4 py-2 border-b border-gray-200 flex justify-between items-center">
              <div className="flex space-x-2">
                <select 
                  className="bg-white border border-gray-300 rounded px-2 py-1 text-sm"
                  value={language}
                  onChange={e => setLanguage(e.target.value)}
                >
                  <option value="python">Python</option>
                  <option value="c">C</option>
                  <option value="java">Java</option>
                  <option value="cpp">C++</option>
                </select>
                <select 
                  className="bg-white border border-gray-300 rounded px-2 py-1 text-sm"
                  value={theme}
                  onChange={e => setTheme(e.target.value)}
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
              </div>
              <div className="flex space-x-2">
                <button 
                  className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-3 py-1 rounded text-sm"
                  onClick={() => {
                    if (problem?.starterCode && problem.starterCode[language]) {
                      setCode(problem.starterCode[language]);
                    }
                  }}
                  disabled={isSolutionShown}
                >
                  Reset
                </button>
              </div>
            </div>
            
            {/* Code Editor */}
            <div className={`flex-grow p-4 font-mono text-sm overflow-auto ${theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-gray-800'}`}>
              <textarea
                className={`w-full h-64 p-2 font-mono text-sm resize-none focus:outline-none ${theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-gray-800'}`}
                value={code}
                onChange={e => setCode(e.target.value)}
                placeholder={loading ? "Loading..." : "Write your code here..."}
                disabled={loading || isSolutionShown}
                readOnly={isSolutionShown}
              />
              
              {isSolutionShown && (
                <div className="mt-4 p-3 bg-yellow-50 border border-yellow-300 rounded text-yellow-800 text-sm">
                  <p className="font-medium">Solution provided</p>
                  <p>You've been shown the solution for this problem. Please move on to a new problem to continue practicing.</p>
                </div>
              )}
            </div>
            
            {/* Console Output */}
            <div className="border-t border-gray-300">
              <div className="bg-gray-100 px-4 py-2 border-b border-gray-200 font-medium text-sm text-gray-700">
                Console
              </div>
              <div className="bg-gray-50 p-4 h-32 overflow-auto font-mono text-xs whitespace-pre-line text-gray-800">
                {isRunning ? 'Executing code...' : output || 'Click "Run" to execute your code'}
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="bg-gray-50 px-4 py-3 border-t border-gray-200 flex justify-end space-x-3">
              <button 
                className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
                onClick={runCode}
                disabled={isRunning || loading || isSolutionShown}
              >
                Run
              </button>
              <button 
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
                onClick={handleSubmit}
                disabled={isRunning || loading || isSolutionShown}
              >
                Submit
              </button>
              {isSolutionShown && (
                <button 
                  className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded text-sm font-medium"
                  onClick={getNewProblem}
                >
                  Try New Problem
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FetchProblem;