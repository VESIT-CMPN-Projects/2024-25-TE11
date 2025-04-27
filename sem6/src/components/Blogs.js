import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Blogs = () => {
  const [blogs, setBlogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const newData = {
    api_token: "2lJUueUSnMHDeDpUIKhllaZuxYLi8UTbczoR3IYw",
    q: "technology OR innovation OR AI OR machine learning OR robotics",
    country: "IN",
    language: "en",
    page: 5
  };
  
  useEffect(() => {
    setLoading(true);
    axios.post('https://api.thenewsapi.com/v1/news/all', newData)
      .then(response => {
        setBlogs(response.data.data);
        setLoading(false);
      })
      .catch(error => {
        console.error(error);
        setError("Failed to fetch news data. Please try again later.");
        setLoading(false);
      });
  }, []);
  
  if (loading) {
    return (
      <div className="w-full h-64 flex items-center justify-center">
        <p className="text-lg text-gray-600">Loading blogs...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="w-full h-64 flex items-center justify-center">
        <p className="text-lg text-red-500">{error}</p>
      </div>
    );
  }
  
  return (
    <div className="w-full h-96 border border-gray-200 rounded-lg shadow-md overflow-hidden">
      <div className="bg-gray-100 p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold">Latest Technology News</h2>
      </div>
      
      <div className="p-4 overflow-y-auto h-80">
        {blogs && blogs.length > 0 ? (
          blogs.map((blog, index) => (
            <div key={index} className="mb-6 pb-6 border-b border-gray-200 last:border-0">
              <div className="flex flex-col md:flex-row gap-4">
                {blog.image_url && (
                  <div className="w-full md:w-1/4 h-48 md:h-32 bg-gray-200 rounded-lg overflow-hidden flex-shrink-0">
                    <img 
                      src="/api/placeholder/300/200" 
                      alt={blog.title} 
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-2 line-clamp-2">{blog.title}</h3>
                  <p className="text-sm text-gray-600 mb-2">
                    {new Date(blog.published_at).toLocaleDateString()} • {blog.source}
                  </p>
                  <p className="text-gray-700 mb-3 line-clamp-3">{blog.description}</p>
                  <a 
                    href={blog.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  >
                    Read more →
                  </a>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-600">No articles found. Try changing your search criteria.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Blogs;