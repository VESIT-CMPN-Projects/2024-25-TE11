import React, { useState } from 'react';
import { Calendar, Clock } from 'lucide-react';

const ScrollingNews = () => {
  const [paused, setPaused] = useState(false);
  
  // Parse and format the news items for better display
  const formatNewsItems = (items) => {
    return items.map(item => {
      const parts = item.split('DATED');
      const event = parts[0].trim();
      const dateTime = parts[1] ? parts[1].trim() : '';
      
      return { event, dateTime };
    });
  };
  
  const newsItems = [
    "ONLINE JOB DRIVE NEAR MUMBAI SUBURBAN DISTRICTS DATED 22 AUGUST",
    "ONLINE TRAINING WORKSHOP ON DESIGN SKILLS DATED 24 AUGUST 8.00PM",
    "MINI JOB FAIR AT KARNATAKA DATED 23 AUGUST 12.00PM",
    "ONLINE WORKSHOP ON SPEAKING SKILLS DATED 25 AUGUST 7.00PM",
    "CAMPUS RECRUITMENT DRIVE FOR IT GRADUATES DATED 26 AUGUST",
    "TECHNICAL SKILLS ASSESSMENT WORKSHOP DATED 27 AUGUST 10.00AM",
    "CAREER COUNSELING SESSION FOR FRESHERS DATED 28 AUGUST 4.00PM",
    "RESUME BUILDING MASTERCLASS DATED 29 AUGUST 6.00PM"
  ];
  
  const formattedItems = formatNewsItems(newsItems);
  
  return (
    <div className="bg-gradient-to-br from-white to-blue-50 rounded-lg shadow-lg w-full md:w-full mb-8 md:mb-0 overflow-hidden border border-blue-100">
      <div className="bg-blue-600 px-6 py-4 flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Calendar className="mr-2" size={20} />
          Events & News
        </h2>
        <button 
          onClick={() => setPaused(!paused)} 
          className="text-xs bg-white bg-opacity-20 hover:bg-opacity-30 text-white py-1 px-3 rounded-full transition-all"
        >
          {paused ? 'Resume' : 'Pause'}
        </button>
      </div>
      
      <div 
        className="h-64 overflow-hidden relative"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        <ul className={`px-2 ${paused ? '' : 'animate-scroll'} space-y-0`}>
          {formattedItems.concat(formattedItems).map((item, index) => (
            <li key={index} className="py-3 px-4 border-b border-blue-100 hover:bg-blue-50 transition-colors">
              <div className="flex items-start">
                <div className="flex-shrink-0 w-2 h-2 rounded-full bg-blue-500 mt-2 mr-3"></div>
                <div className="flex-grow">
                  <p className="font-medium text-gray-800">{item.event}</p>
                  <div className="flex items-center mt-1 text-sm text-gray-500">
                    <Clock size={14} className="mr-1" />
                    <span>{item.dateTime}</span>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
      
      {/* Gradient overlays to indicate more content */}
      <div className="absolute top-16 left-0 right-0 h-8 bg-gradient-to-b from-white to-transparent pointer-events-none"></div>
      <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-white to-transparent pointer-events-none"></div>
    </div>
  );
};

export default ScrollingNews;