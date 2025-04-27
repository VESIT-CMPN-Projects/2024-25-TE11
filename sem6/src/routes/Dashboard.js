// src/routes/Jobseekers.js
import React from 'react';
import { Bar, Pie, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  ArcElement,
  LineElement,
  PointElement,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';

// Register the necessary components
ChartJS.register(
  CategoryScale,
  LinearScale,
  ArcElement,
  LineElement,
  PointElement,
  BarElement,
  Tooltip,
  Legend
);

function Dashboard() {
  // Dummy data for visualizations
  const skillData = {
    labels: ['Java', 'Python', 'JavaScript', 'C#', 'SQL'],
    datasets: [
      {
        label: 'Skills Proficiency',
        data: [90, 80, 70, 60, 50],
        backgroundColor: [
          'rgba(56, 189, 248, 0.6)', // Example: Light Blue
          'rgba(254, 215, 8, 0.6)',   // Example: Yellow
          'rgba(244, 63, 94, 0.6)',    // Example: Pink
          'rgba(167, 39, 160, 0.6)',   // Example: Purple
          'rgba(34, 197, 94, 0.6)',     // Example: Green
        ],
      },
    ],
  };

  const experienceData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Experience Over Time',
        data: [5, 7, 8, 6, 9, 10],
        fill: false,
        backgroundColor: 'rgba(56, 189, 248, 1)', // Example: Light Blue
        borderColor: 'rgba(56, 189, 248, 1)',
        borderWidth: 2,
      },
    ],
  };

  const jobOpportunitiesData = {
    labels: ['Data Science', 'Web Development', 'Cloud Engineering'],
    datasets: [
      {
        label: 'Job Opportunities',
        data: [20, 30, 50],
        backgroundColor: [
          'rgba(244, 63, 94, 0.6)', // Example: Pink
          'rgba(56, 189, 248, 0.6)', // Example: Light Blue
          'rgba(254, 215, 8, 0.6)',   // Example: Yellow
        ],
      },
    ],
  };

  return (
    <div className="bg-gray-100 h-screen w-full">
      <div className="container mx-auto py-2 px-6 md:px-8 lg:px-10">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">Jobseekers Dashboard</h1>
        <p className="text-lg md:text-xl text-gray-700 mb-8">
          Welcome! Here, you can find resources and tips to help you land your dream job.
        </p>

        {/* Skills Summary */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Skills Summary</h2>
          <ul className="list-disc pl-5 text-gray-700">
            <li>Proficient in Java and Python</li>
            <li>Experience with JavaScript frameworks</li>
            <li>Knowledge of SQL databases</li>
            <li>Familiar with cloud services and DevOps tools</li>
            <li>Strong problem-solving skills</li>
          </ul>
        </div>

        {/* Visualization Section with Grid Layout */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
            <h3 className="font-bold text-lg text-gray-800 mb-2">Skill Proficiency</h3>
            <div className="h-64 w-full">
              <Pie data={skillData} options={{ responsive: true }} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
            <h3 className="font-bold text-lg text-gray-800 mb-2">Experience Over Time</h3>
            <div className="h-64 w-full">
              <Line data={experienceData} options={{ responsive: true }} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
            <h3 className="font-bold text-lg text-gray-800 mb-2">Job Opportunities by Role</h3>
            <div className="h-64 w-full">
              <Bar data={jobOpportunitiesData} options={{ responsive: true }} />
            </div>
          </div>
        </div>

        {/* Job Search Tips */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Job Search Tips</h2>
          <ol className="list-decimal pl-5 text-gray-700">
            <li>Tailor your resume to highlight relevant skills.</li>
            <li>Utilize LinkedIn to network with professionals.</li>
            <li>Prepare for interviews by practicing common questions.</li>
            <li>Consider online courses to boost your skills.</li>
            <li>Stay updated with industry trends and news.</li>
          </ol>
        </div>

        {/* Job Search Progress Tracker */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Job Search Progress Tracker</h2>
          <table className="w-full text-left border-collapse">
            <thead>
              <tr>
                <th className="py-3 border-b-2 border-gray-300">Task</th>
                <th className="py-3 border-b-2 border-gray-300">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="py-4 border-b border-gray-200">Update Resume</td>
                <td className="py-4 border-b border-gray-200 text-green-600">✔ Completed</td>
              </tr>
              <tr>
                <td className="py-4 border-b border-gray-200">Submit Applications</td>
                <td className="py-4 border-b border-gray-200 text-yellow-600">✖ In Progress</td>
              </tr>
              <tr>
                <td className="py-4 border-b border-gray-200">Prepare for Interviews</td>
                <td className="py-4 border-b border-gray-200 text-gray-600">✖ Not Started</td>
              </tr>
              <tr>
                <td className="py-4 border-b border-gray-200">Follow Up with Employers</td>
                <td className="py-4 border-b border-gray-200 text-gray-600">✖ Not Started</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Resources Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Helpful Resources</h2>
          <ul className="list-disc pl-5 text-gray-700">
            {/* eslint-disable-next-line */}
            <li><a href="#" className="text-blue-600 hover:underline">LinkedIn Learning</a></li>
            {/* eslint-disable-next-line */}
            <li><a href="#" className="text-blue-600 hover:underline">Coursera</a></li>
            {/* eslint-disable-next-line */}
            <li><a href="#" className="text-blue-600 hover:underline">Glassdoor</a></li>
            {/* eslint-disable-next-line */}
            <li><a href="#" className="text-blue-600 hover:underline">Indeed Career Guide</a></li>
            {/* eslint-disable-next-line */}
            <li><a href="#" className="text-blue-600 hover:underline">Resume Templates</a></li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
