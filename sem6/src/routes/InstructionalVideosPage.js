import React, { useState, useEffect } from 'react';
import ReactPlayer from 'react-player/youtube';
import Navbar from '../components/Navbar';
import { Link } from 'react-router-dom';

const InstructionalVideos = () => {
  const [selectedCategory, setSelectedCategory] = useState('frontend');
  const [activeVideo, setActiveVideo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Function to extract YouTube video ID from URL
  const getYouTubeID = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  // Function to generate YouTube thumbnail URL
  const getYouTubeThumbnail = (url) => {
    const videoID = getYouTubeID(url);
    return videoID ? `https://img.youtube.com/vi/${videoID}/mqdefault.jpg` : null;
  };

  // Simulate loading state
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);
    return () => clearTimeout(timer);
  }, []);

  // Video data organized by categories
  const videoCategories = {
    frontend: [
      {
        id: 'fe1',
        title: 'HTML, CSS, and Javascript in 30 minutes',
        description: 'Learn the basics of HTML, CSS, and Javascript in 30 minutes',
        url: 'https://www.youtube.com/watch?v=_GTMOmRrqkU&ab_channel=devdojo',
        duration: '0:31:48',
        level: 'Beginner'
      },
      {
        
        id: 'fe2',
        title: 'Complete React Course for Beginners',
        description: 'Learn React from scratch with practical examples and projects.',
        url: 'https://www.youtube.com/watch?v=bMknfKXIFA8',
        duration: '12:08:40',
        level: 'Intermediate'
      },
      {
        id: 'fe3',
        title: 'JavaScript ES6+ Features Every Developer Should Know',
        description: 'Upgrade your JavaScript skills with the latest ES6+ features.',
        url: 'https://www.youtube.com/watch?v=NCwa_xi0Uuc',
        duration: '1:28:47',
        level: 'Advanced'
      }
    ],
    aiml: [
      {
        id: 'ml1',
        title: 'Machine Learning for JavaScript Developers',
        description: 'Integrate machine learning into your web applications with TensorFlow.js.',
        url: 'https://www.youtube.com/watch?v=WIHZ7kjJ35o',
        duration: '2:19:58',
        level: 'Intermediate'
      },
      {
        id: 'ml2',
        title: 'AI for Web Developers: Practical Applications',
        description: 'Learn how to implement AI features in your web applications.',
        url: 'https://www.youtube.com/watch?v=XB4MIexjvY0',
        duration: '1:45:32',
        level: 'Advanced'
      }
    ],
    devops: [
      {
        id: 'do1',
        title: 'Docker and Kubernetes for Developers',
        description: 'Learn container technologies to streamline your development workflow.',
        url: 'https://www.youtube.com/watch?v=Wf2eSG3owoA',
        duration: '3:11:28',
        level: 'Intermediate'
      },
      {
        id: 'do2',
        title: 'CI/CD Pipeline Tutorial using GitHub Actions',
        description: 'Set up automated testing and deployment for your web projects.',
        url: 'https://www.youtube.com/watch?v=R8_veQiYBjI',
        duration: '1:23:54',
        level: 'Intermediate'
      }
    ],
    software: [
      {
        id: 'se1',
        title: 'Data Structures and Algorithms in JavaScript',
        description: 'Master fundamental computer science concepts with JavaScript examples.',
        url: 'https://www.youtube.com/watch?v=t2CEgPsws3U',
        duration: '6:21:53',
        level: 'Advanced'
      },
      {
        id: 'se2',
        title: 'System Design for Web Developers',
        description: 'Learn how to design scalable systems for your web applications.',
        url: 'https://www.youtube.com/watch?v=FKA2KgkkcqY',
        duration: '1:56:09',
        level: 'Advanced'
      }
    ]
  };

  // Handle video selection
  const handleVideoSelect = (video) => {
    setActiveVideo(video);
    // Scroll to video player with offset for better UX
    const element = document.getElementById('video-player');
    const headerOffset = 80;
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
    
    window.scrollTo({
      top: offsetPosition,
      behavior: 'smooth'
    });
  };

  // Get category icon
  const getCategoryIcon = (category) => {
    switch(category) {
      case 'frontend':
        return '🎨';
      case 'aiml':
        return '🧠';
      case 'devops':
        return '🚀';
      case 'software':
        return '💻';
      default:
        return '📚';
    }
  };

  // Get level badge color
  const getLevelColor = (level) => {
    switch(level) {
      case 'Beginner':
        return 'bg-green-100 text-green-800';
      case 'Intermediate':
        return 'bg-yellow-100 text-yellow-800';
      case 'Advanced':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      {/* Navbar */}
      <Navbar />
      
      {/* Hero section with animated gradient */}
      <div className="bg-gradient-to-r from-blue-700 via-blue-600 to-blue-800 py-12 px-4 relative overflow-hidden">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmZmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzBoLTZWNmg2djI0em0tNiAwSDZWNmgyNHYyNHpNNiAwdjZoNlYwSDZ6Ii8+PC9nPjwvZz48L3N2Zz4=')] bg-repeat"></div>
        </div>
        <div className="container mx-auto relative z-10">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-2 tracking-tight">Developer Instructional Videos</h1>
          <p className="text-blue-100 text-xl max-w-2xl">
            Elevate your skills with our curated collection of high-quality tutorials
          </p>
        </div>
      </div>

      {/* Main content */}
      <div className="container mx-auto py-8 px-4">
        {/* Category navigation */}
        <div className="bg-white rounded-xl shadow-md p-4 mb-8">
          <div className="flex flex-wrap gap-3">
            {Object.keys(videoCategories).map(category => (
              <button 
                key={category}
                className={`px-5 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 ${
                  selectedCategory === category 
                    ? 'bg-blue-600 text-white shadow-md' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setSelectedCategory(category)}
              >
                <span>{getCategoryIcon(category)}</span>
                {category === 'frontend' && 'Frontend Development'}
                {category === 'aiml' && 'AI & Machine Learning'}
                {category === 'devops' && 'DevOps'}
                {category === 'software' && 'Software Engineering'}
              </button>
            ))}
          </div>
        </div>

        {/* Video player section */}
        <div id="video-player" className="mb-12 scroll-mt-20">
          {activeVideo ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <div className="bg-black rounded-2xl overflow-hidden shadow-xl">
                  <ReactPlayer
                    url={activeVideo.url}
                    controls
                    width="100%"
                    height="600px"
                    className="rounded-lg"
                  />
                </div>
              </div>
              <div className="bg-white p-8 rounded-2xl shadow-lg">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getLevelColor(activeVideo.level)}`}>
                    {activeVideo.level}
                  </span>
                  <span className="text-gray-500 text-sm">{activeVideo.duration}</span>
                </div>
                <h2 className="text-3xl font-bold mb-3 text-gray-800">{activeVideo.title}</h2>
                <p className="text-gray-600 mb-6 text-lg">{activeVideo.description}</p>
                
                <div className="mt-6 bg-blue-50 p-5 rounded-xl">
                  <h3 className="font-bold text-xl mb-3 text-blue-800">What you'll learn:</h3>
                  <ul className="space-y-3">
                    {['Fundamental concepts and best practices', 
                      'Practical implementation techniques', 
                      'Real-world problem solving', 
                      'Industry-standard workflows'].map((item, index) => (
                      <li key={index} className="flex items-start">
                        <span className="text-blue-600 mr-2">✓</span>
                        <span className="text-gray-700">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-24 bg-white rounded-2xl shadow-md border border-gray-100">
              <div className="max-w-md mx-auto">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="mt-2 text-xl font-medium text-gray-900">No video selected</h3>
                <p className="mt-1 text-gray-500">Choose a video from the collection below to start learning</p>
              </div>
            </div>
          )}
        </div>

        {/* Video selection grid with animation */}
        <div>
          <div className="flex items-center mb-6">
            <h2 className="text-3xl font-bold text-gray-800">
              {getCategoryIcon(selectedCategory)} {selectedCategory === 'frontend' && 'Frontend Development Videos'}
              {selectedCategory === 'aiml' && 'AI & Machine Learning Videos'}
              {selectedCategory === 'devops' && 'DevOps Videos'}
              {selectedCategory === 'software' && 'Software Engineering Videos'}
            </h2>
            <div className="ml-auto text-sm text-gray-500">
              {videoCategories[selectedCategory].length} videos
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {isLoading ? (
              // Skeleton loading states
              Array(3).fill(0).map((_, index) => (
                <div key={index} className="bg-white rounded-xl shadow-md overflow-hidden animate-pulse">
                  <div className="w-full h-48 bg-gray-300"></div>
                  <div className="p-5">
                    <div className="h-5 bg-gray-300 rounded w-3/4 mb-3"></div>
                    <div className="h-4 bg-gray-300 rounded w-full mb-2"></div>
                    <div className="h-4 bg-gray-300 rounded w-2/3"></div>
                  </div>
                </div>
              ))
            ) : (
              videoCategories[selectedCategory].map((video) => (
                <div 
                  key={video.id} 
                  className={`bg-white rounded-xl shadow-md overflow-hidden cursor-pointer transform transition-all duration-300 hover:shadow-xl hover:-translate-y-1 ${activeVideo?.id === video.id ? 'ring-2 ring-blue-500' : ''}`}
                  onClick={() => handleVideoSelect(video)}
                >
                  <div className="relative">
                    <img 
                      src={getYouTubeThumbnail(video.url)} 
                      alt={video.title} 
                      className="w-full h-48 object-cover"
                      loading="lazy"
                    />
                    <div className="absolute bottom-0 right-0 bg-black bg-opacity-70 text-white px-2 py-1 m-2 rounded text-sm">
                      {video.duration}
                    </div>
                    <div className="absolute inset-0 bg-black bg-opacity-30 opacity-0 hover:opacity-100 flex items-center justify-center transition-opacity">
                      <div className="w-16 h-16 rounded-full bg-blue-600 bg-opacity-90 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  </div>
                  <div className="p-5">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getLevelColor(video.level)}`}>
                        {video.level}
                      </span>
                    </div>
                    <h3 className="font-bold text-lg mb-2 text-gray-800 line-clamp-1">{video.title}</h3>
                    <p className="text-gray-600 text-sm mb-3 line-clamp-2">{video.description}</p>
                    <div className="flex justify-between items-center">
                      <span className="text-blue-600 text-sm font-medium">Watch now</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M12.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div className="bg-gray-900 text-white py-12 mt-16">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <h3 className="text-2xl font-bold mb-4">Continue Your Learning Journey</h3>
            <p className="text-gray-400 max-w-2xl mx-auto mb-8">
              Our curated collection is constantly updated with the latest tutorials and courses to help you stay ahead in your development career.
            </p>
            <div className="flex justify-center space-x-4">
              <Link to="/contactus">
              <button className="bg-transparent border border-gray-600 hover:border-gray-400 text-gray-300 hover:text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                Request a Topic
              </button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InstructionalVideos;