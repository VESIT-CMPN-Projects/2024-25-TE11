import React, { useState, useEffect } from 'react';
import ReactPlayer from 'react-player/youtube';
import Navbar from '../components/Navbar'; // Update this path to match your project structure

const SuccessStories = () => {
  const [selectedCategory, setSelectedCategory] = useState('career');
  const [activeVideo, setActiveVideo] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

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

  // Set first video as active on category change
  useEffect(() => {
    if (videoCategories[selectedCategory].length > 0) {
      // Set the first video of the category as active if none is selected
      if (!activeVideo || !videoCategories[selectedCategory].some(v => v.id === activeVideo.id)) {
        setActiveVideo(videoCategories[selectedCategory][0]);
      }
    }
  }, [selectedCategory]);

  // Success story videos organized by categories
  const videoCategories = {
    career: [
      {
        id: 'career1',
        title: 'From Junior to Senior Developer in 2 Years',
        description: 'Learn how Sarah transformed her career through consistent learning and strategic project choices.',
        url: 'https://www.youtube.com/watch?v=r_MbozD32eo',
        duration: '18:42',
        author: 'Sarah Chen'
      },
      {
        id: 'career2',
        title: 'How I Landed My Dream Job at a Tech Giant',
        description: 'Follow Mark\'s journey from bootcamp graduate to working at one of the top tech companies.',
        url: 'https://www.youtube.com/watch?v=4vKiJZNPhss',
        duration: '22:15',
        author: 'Mark Johnson'
      },
      {
        id: 'career3',
        title: 'Switching Careers: From Teacher to Developer',
        description: 'Jane shares how she successfully transitioned from education to tech with no prior coding experience.',
        url: 'https://www.youtube.com/watch?v=u7UyiLCQGas',
        duration: '24:56',
        author: 'Jane Williams'
      }
    ],
    freelance: [
      {
        id: 'freelance1',
        title: 'Building a Six-Figure Freelance Development Business',
        description: 'Dave explains his strategies for finding high-value clients and scaling his freelance operation.',
        url: 'https://www.youtube.com/watch?v=MJUJ4wbFm_A',
        duration: '32:18',
        author: 'Dave Rodriguez'
      },
      {
        id: 'freelance2',
        title: 'How I Got My First 10 Clients as a Freelance Developer',
        description: 'Practical tips on starting your freelance journey with no connections or portfolio.',
        url: 'https://www.youtube.com/watch?v=sDnPs_V87Dk',
        duration: '26:04',
        author: 'Priya Sharma'
      }
    ],
    startup: [
      {
        id: 'startup1',
        title: 'From Side Project to Funded Startup: My Journey',
        description: 'Learn how Alex turned a weekend project into a venture-backed company.',
        url: 'https://www.youtube.com/watch?v=RpT3BA_mK1I',
        duration: '28:45',
        author: 'Alex Torres'
      },
      {
        id: 'startup2',
        title: 'Bootstrapping Our SaaS to $50K MRR',
        description: 'The co-founders share their experience growing a profitable business without external funding.',
        url: 'https://www.youtube.com/watch?v=epHFUWmrAPg',
        duration: '34:22',
        author: 'Michael & Lisa Reynolds'
      }
    ],
    learning: [
      {
        id: 'learning1',
        title: 'How I Mastered 5 Programming Languages in 3 Years',
        description: 'Kevin shares his efficient learning system for rapidly acquiring new technical skills.',
        url: 'https://www.youtube.com/watch?v=5t9yk0HVzPI',
        duration: '19:36',
        author: 'Kevin Zhang'
      },
      {
        id: 'learning2',
        title: 'Self-Taught Developer: My Learning Path and Tips',
        description: 'A complete roadmap from zero knowledge to employment without a CS degree.',
        url: 'https://www.youtube.com/watch?v=qvn2JFX4lHs',
        duration: '25:18',
        author: 'Emma Wilson'
      }
    ]
  };

  // Handle video selection
  const handleVideoSelect = (video) => {
    setActiveVideo(video);
    setIsPlaying(false);
    // Scroll to video player
    document.getElementById('video-player').scrollIntoView({ behavior: 'smooth' });
    // Short delay to start playing
    setTimeout(() => setIsPlaying(true), 300);
  };

  return (
    <div className="bg-gray-50 min-h-screen font-sans">
      {/* Navbar */}
      <Navbar />
      
      {/* Header section */}
      <div className="bg-gradient-to-r from-green-600 to-green-800 py-12 px-4 shadow-lg">
        <div className="container mx-auto max-w-6xl">
          <h1 className="text-4xl font-bold text-white mb-3">Developer Success Stories</h1>
          <p className="text-green-100 text-lg max-w-2xl">
            Real journeys and inspirational stories from developers who've achieved their goals and transformed their careers
          </p>
        </div>
      </div>

      {/* Main content */}
      <div className="container mx-auto py-12 px-4 max-w-6xl">
        {/* Category navigation */}
        <div className="flex flex-wrap mb-10 gap-3 justify-center sm:justify-start">
          {Object.keys(videoCategories).map((category) => (
            <button 
              key={category}
              className={`px-5 py-2.5 rounded-full font-medium transition-all duration-300 shadow ${
                selectedCategory === category 
                  ? 'bg-green-600 text-white ring-2 ring-green-300 transform scale-105' 
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
              onClick={() => setSelectedCategory(category)}
            >
              {category === 'career' && 'Career Growth'}
              {category === 'freelance' && 'Freelance Success'}
              {category === 'startup' && 'Startup Stories'}
              {category === 'learning' && 'Learning Journeys'}
            </button>
          ))}
        </div>

        {/* Video player section */}
        <div id="video-player" className="mb-12">
          {activeVideo ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <div className="rounded-xl overflow-hidden shadow-2xl bg-black">
                  <ReactPlayer
                    url={activeVideo.url}
                    playing={isPlaying}
                    controls
                    width="100%"
                    height="500px"
                    className="rounded-lg overflow-hidden"
                  />
                </div>
                <div className="mt-4 flex items-center justify-between px-2">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                      </svg>
                    </div>
                    <span className="text-sm text-gray-600">{activeVideo.duration}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Watch in HD for best experience</span>
                  </div>
                </div>
              </div>
              <div className="bg-white p-8 rounded-xl shadow-lg border border-gray-100">
                <h2 className="text-2xl font-bold mb-3 text-gray-800">{activeVideo.title}</h2>
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center mr-3">
                    <span className="text-green-800 font-bold">{activeVideo.author.charAt(0)}</span>
                  </div>
                  <div>
                    <p className="text-gray-700 font-medium">{activeVideo.author}</p>
                    <p className="text-gray-500 text-sm">Developer & Creator</p>
                  </div>
                </div>
                <div className="h-0.5 w-full bg-gray-100 my-4"></div>
                <p className="text-gray-700 mb-6 leading-relaxed">{activeVideo.description}</p>
                <div className="mb-6">
                  <span className="inline-block bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full font-medium">
                    {selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)} Story
                  </span>
                </div>
                <div className="bg-green-50 p-5 rounded-lg border border-green-100">
                  <h3 className="font-bold text-lg mb-3 text-green-800">Key Takeaways:</h3>
                  <ul className="space-y-2">
                    {[
                      'Practical strategies that worked in real situations',
                      'Common obstacles and how they were overcome',
                      'Timeline of progress and milestones',
                      'Advice for developers on similar paths'
                    ].map((point, index) => (
                      <li key={index} className="flex items-start">
                        <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="text-gray-700">{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-20 bg-gray-100 rounded-xl">
              <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-gray-600 text-lg">Select a success story to start watching</p>
            </div>
          )}
        </div>

        {/* Video selection grid */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800">
              {selectedCategory === 'career' && 'Career Growth Stories'}
              {selectedCategory === 'freelance' && 'Freelance Success Stories'}
              {selectedCategory === 'startup' && 'Startup Journey Stories'}
              {selectedCategory === 'learning' && 'Learning Achievement Stories'}
            </h2>
            <div className="text-sm text-gray-500">
              {videoCategories[selectedCategory].length} stories
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videoCategories[selectedCategory].map((video) => (
              <div 
                key={video.id} 
                className={`bg-white rounded-xl shadow-md overflow-hidden cursor-pointer transition-all duration-300 hover:shadow-xl ${
                  activeVideo && activeVideo.id === video.id ? 'ring-2 ring-green-500 transform scale-102' : ''
                }`}
                onClick={() => handleVideoSelect(video)}
              >
                <div className="relative">
                  <img 
                    src={getYouTubeThumbnail(video.url)} 
                    alt={video.title} 
                    className="w-full h-48 object-cover"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300">
                    <div className="w-14 h-14 rounded-full bg-green-600 bg-opacity-90 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-6 h-6">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                      </svg>
                    </div>
                  </div>
                  <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                    {video.duration}
                  </div>
                </div>
                <div className="p-5">
                  <h3 className="font-bold text-lg mb-2 text-gray-800 line-clamp-1">{video.title}</h3>
                  <div className="flex items-center mb-2">
                    <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center mr-2">
                      <span className="text-green-800 text-xs font-bold">{video.author.charAt(0)}</span>
                    </div>
                    <p className="text-gray-600 text-sm">{video.author}</p>
                  </div>
                  <p className="text-gray-600 text-sm line-clamp-2">{video.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom call to action */}
      <div className="bg-gradient-to-r from-green-600 to-green-800 py-8 mt-12">
        <div className="container mx-auto text-center px-4">
          <h3 className="text-2xl font-bold text-white mb-4">Ready to share your success story?</h3>
          <p className="text-green-100 mb-6 max-w-2xl mx-auto">
            Your journey could inspire the next generation of developers. We'd love to hear about your experiences and achievements.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SuccessStories;