import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import jobseeker from '../assets/Jobseeker.png.jpg';
import employers from '../assets/employers.png.jpg';
import videoicon from '../assets/videoicon.png';
import Navbar from '../components/Navbar';
import tick from '../assets/tick.png';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Pagination, Navigation, Autoplay, EffectFade } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';
import 'swiper/css/autoplay';
import 'swiper/css/effect-fade';
import axios from 'axios';
import ScrollingNews from '../components/ScrollingNews';
import { FaBriefcase, FaGraduationCap, FaChartLine, FaHandshake } from 'react-icons/fa';
import Blogs from '../components/Blogs';
import Footer from '../components/Footer';
import ChatbotIcon from '../components/ChatbotIcon';

function Home() {
  const navigate = useNavigate();
  const [blogs, setBlogs] = useState([]);
  const [searchQuery] = useState('');
  const [testimonials] = useState([
    {
      name: "Rahul Sharma",
      role: "Software Engineer at Google",
      image: "/api/placeholder/100/100",
      quote: "SkillBridge AI transformed my job search. Within two weeks, I had multiple interviews and landed my dream role at Google."
    },
    {
      name: "Priya Patel",
      role: "Data Scientist at Amazon",
      image: "/api/placeholder/100/100",
      quote: "The AI-powered skill matching feature helped me identify exactly what employers were looking for. The results speak for themselves!"
    },
    {
      name: "Neha Gupta",
      role: "Product Manager at Flipkart",
      image: "/api/placeholder/100/100",
      quote: "As someone who changed careers, SkillBridge's targeted courses and resume guidance were invaluable in helping me transition smoothly."
    }
  ]);

  const newData = {
    api_token: "2lJUueUSnMHDeDpUIKhllaZuxYLi8UTbczoR3IYw",
    q: "technology OR innovation OR AI OR machine learning OR robotics",
    country: "IN",
    language: "en",
    page: 5
  };

  const handleVideoClick = () => {
    navigate('/instructional-videos');
  };

  useEffect(() => {
    axios.post('https://api.thenewsapi.com/v1/news/all', newData)
      .then(response => setBlogs(response.data.data))
      .catch(error => console.error(error));
  }, []);

  return (
    <div className="bg-gray-50 min-h-screen">
      <Navbar />
      
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-400 to-blue-600 text-white py-16">
        <div className="container mx-auto px-7 flex flex-col items-center">
          <h1 className="text-4xl md:text-6xl font-bold text-center mb-6">Your Career Journey Starts Here</h1>
          <p className="text-xl md:text-2xl text-center mb-8 max-w-3xl">
            Connect with top employers, discover opportunities, and upskill with India's leading career platform
          </p>
        </div>
      </div>
      
      <main className="container mx-auto py-12 px-7">
        {/* Value Propositions */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-10 text-gray-800">
            Why Choose <span className="text-blue-600">SkillBridge AI</span>
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white rounded-xl shadow-md p-8 hover:shadow-lg transition duration-300 border-t-4 border-blue-500">
              <div className="flex items-center mb-4">
                <div className="p-3 bg-blue-100 rounded-lg mr-4">
                  <FaChartLine className="text-blue-600 text-2xl" />
                </div>
                <h3 className="text-xl font-semibold">AI-Powered Matching</h3>
              </div>
              <p className="text-gray-600">Our intelligent algorithm matches your skills and experience with the perfect job opportunities, saving you time and increasing your success rate.</p>
            </div>
            
            <div className="bg-white rounded-xl shadow-md p-8 hover:shadow-lg transition duration-300 border-t-4 border-green-500">
              <div className="flex items-center mb-4">
                <div className="p-3 bg-green-100 rounded-lg mr-4">
                  <FaGraduationCap className="text-green-600 text-2xl" />
                </div>
                <h3 className="text-xl font-semibold">Skill Enhancement</h3>
              </div>
              <p className="text-gray-600">Access curated courses and workshops designed to bridge skill gaps and make you more competitive in today's job market.</p>
            </div>
            
            <div className="bg-white rounded-xl shadow-md p-8 hover:shadow-lg transition duration-300 border-t-4 border-purple-500">
              <div className="flex items-center mb-4">
                <div className="p-3 bg-purple-100 rounded-lg mr-4">
                  <FaHandshake className="text-purple-600 text-2xl" />
                </div>
                <h3 className="text-xl font-semibold">Direct Employer Connect</h3>
              </div>
              <p className="text-gray-600">Skip the middlemen and connect directly with hiring managers at top companies across India and beyond.</p>
            </div>
          </div>
        </section>

        {/* Trending Jobs Carousel */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-800">Trending Job Opportunities</h2>
          </div> 
          <Swiper
            pagination={{ clickable: true }}
            navigation={true}
            autoplay={{ delay: 5000, disableOnInteraction: false }}
            spaceBetween={20}
            slidesPerView={1}
            breakpoints={{
              640: { slidesPerView: 2 },
              1024: { slidesPerView: 3 }
            }}
            modules={[Pagination, Navigation, Autoplay]}
            className="mySwiper"
          >
            {/* Full Stack Developer Slide */}
            <SwiperSlide>
              <div className="flex flex-col h-80 bg-gradient-to-br from-blue-500 to-blue-700 text-white rounded-lg shadow-md p-6 transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center mb-4">
                  <FaBriefcase className="text-2xl mr-2" />
                  <h3 className="text-2xl font-semibold">Full Stack Developer</h3>
                </div>
                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm mb-4 inline-block w-fit">
                  ₹12-18 LPA
                </div>
                <p className="flex-grow">Build and manage both frontend and backend applications using modern web technologies like React, Node.js, and cloud services.</p>
                <div className="mt-4 flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">React</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Node.js</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">MongoDB</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">AWS</span>
                </div>
                <Link to="/job/full-stack-developer" className="px-4 py-2 bg-white text-blue-600 rounded-lg shadow hover:bg-blue-50 transition duration-200 text-center">
                  View Details
                </Link>
              </div>
            </SwiperSlide>

            {/* AI/ML Engineer Slide */}
            <SwiperSlide>
              <div className="flex flex-col h-80 bg-gradient-to-br from-green-500 to-green-700 text-white rounded-lg shadow-md p-6 transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center mb-4">
                  <FaBriefcase className="text-2xl mr-2" />
                  <h3 className="text-2xl font-semibold">AI/ML Engineer</h3>
                </div>
                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm mb-4 inline-block w-fit">
                  ₹18-25 LPA
                </div>
                <p className="flex-grow">Develop and deploy AI/ML models using Python, TensorFlow, PyTorch, and advanced data science techniques.</p>
                <div className="mt-4 flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Python</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">TensorFlow</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">PyTorch</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">NLP</span>
                </div>
                <Link to="/job/ai-ml-engineer" className="px-4 py-2 bg-white text-green-600 rounded-lg shadow hover:bg-green-50 transition duration-200 text-center">
                  View Details
                </Link>
              </div>
            </SwiperSlide>

            {/* Cloud and DevOps Engineer Slide */}
            <SwiperSlide>
              <div className="flex flex-col h-80 bg-gradient-to-br from-yellow-500 to-yellow-700 text-white rounded-lg shadow-md p-6 transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center mb-4">
                  <FaBriefcase className="text-2xl mr-2" />
                  <h3 className="text-2xl font-semibold">DevOps Engineer</h3>
                </div>
                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm mb-4 inline-block w-fit">
                  ₹15-22 LPA
                </div>
                <p className="flex-grow">Manage and automate cloud infrastructure using AWS, Azure, Kubernetes, Docker and Terraform.</p>
                <div className="mt-4 flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">AWS</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Docker</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Kubernetes</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">CI/CD</span>
                </div>
                <Link to="/job/cloud-devops-engineer" className="px-4 py-2 bg-white text-yellow-600 rounded-lg shadow hover:bg-yellow-50 transition duration-200 text-center">
                  View Details
                </Link>
              </div>
            </SwiperSlide>

            {/* Software Engineer Slide */}
            <SwiperSlide>
              <div className="flex flex-col h-80 bg-gradient-to-br from-orange-500 to-orange-700 text-white rounded-lg shadow-md p-6 transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center mb-4">
                  <FaBriefcase className="text-2xl mr-2" />
                  <h3 className="text-2xl font-semibold">Software Engineer</h3>
                </div>
                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm mb-4 inline-block w-fit">
                  ₹10-18 LPA
                </div>
                <p className="flex-grow">Design, develop, and maintain software applications using Java, Python, and other modern programming languages.</p>
                <div className="mt-4 flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Java</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Spring</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Python</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">SQL</span>
                </div>
                <Link to="/job/software-engineer" className="px-4 py-2 bg-white text-orange-600 rounded-lg shadow hover:bg-orange-50 transition duration-200 text-center">
                  View Details
                </Link>
              </div>
            </SwiperSlide>
            
            {/* UX/UI Designer Slide */}
            {/* <SwiperSlide>
              <div className="flex flex-col h-80 bg-gradient-to-br from-purple-500 to-purple-700 text-white rounded-lg shadow-md p-6 transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center mb-4">
                  <FaBriefcase className="text-2xl mr-2" />
                  <h3 className="text-2xl font-semibold">UX/UI Designer</h3>
                </div>
                <div className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm mb-4 inline-block w-fit">
                  ₹8-16 LPA
                </div>
                <p className="flex-grow">Create intuitive, engaging user experiences and interfaces for web and mobile applications.</p>
                <div className="mt-4 flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Figma</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Adobe XD</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Sketch</span>
                  <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Prototyping</span>
                </div>
                <Link to="/job/ux-ui-designer" className="px-4 py-2 bg-white text-purple-600 rounded-lg shadow hover:bg-purple-50 transition duration-200 text-center">
                  View Details
                </Link>
              </div>
            </SwiperSlide> */}
          </Swiper>
        </section>

        {/* User Type Cards - With better layout and hover effects */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-10 text-gray-800">Choose Your Path</h2>
          
          <div className="flex flex-row justify-evenly">
            <div className="bg-white rounded-xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-500 group">
              <div className="relative h-96 overflow-hidden">
                <img 
                  src={jobseeker} 
                  alt="Job Seekers" 
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-blue-900 to-transparent opacity-70"></div>
                <h2 className="absolute bottom-4 left-4 text-2xl font-bold text-white">FOR JOBSEEKERS</h2>
              </div>
              <div className="p-6">
                <Link 
                  to="/signup" 
                  className="block w-full py-3 text-center bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300"
                >
                  Find Your Dream Job
                </Link>
              </div>
            </div>
            
            <div className="bg-white rounded-xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-500 group">
              <div className="relative h-96 overflow-hidden">
                <img 
                  src={employers} 
                  alt="Employers" 
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-green-900 to-transparent opacity-70"></div>
                <h2 className="absolute bottom-4 left-4 text-2xl font-bold text-white">FOR EMPLOYERS</h2>
              </div>
              <div className="p-6">
                <Link 
                  to="/signup" 
                  className="block w-full py-3 text-center bg-green-600 text-white rounded-lg hover:bg-green-700 transition duration-300"
                >
                  Find Top Talent
                </Link>
              </div>
            </div>
            
            {/* <div className="bg-white rounded-xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-500 group">
              <div className="relative h-64 overflow-hidden">
                <img 
                  src={collaborators} 
                  alt="Collaborators" 
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-purple-900 to-transparent opacity-70"></div>
                <h2 className="absolute bottom-4 left-4 text-2xl font-bold text-white">FOR COLLABORATORS</h2>
              </div>
              <div className="p-6">
                <ul className="mb-6 space-y-2">
                  <li className="flex items-center"><FaStar className="text-purple-500 mr-2" /> Course creation platform</li>
                  <li className="flex items-center"><FaStar className="text-purple-500 mr-2" /> Revenue sharing model</li>
                  <li className="flex items-center"><FaStar className="text-purple-500 mr-2" /> Student engagement tools</li>
                  <li className="flex items-center"><FaStar className="text-purple-500 mr-2" /> Analytics dashboard</li>
                </ul>
                <Link 
                  to="/collaborator-signup" 
                  className="block w-full py-3 text-center bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition duration-300"
                >
                  Share Your Expertise
                </Link>
              </div>
            </div> */}
          </div>
        </section>

        {/* Resources and Videos section - Enhanced */}
        <section className="mb-16">
          <div className="flex flex-col md:flex-row md:space-x-8">
            <div className="w-full md:w-full mb-8 md:mb-0">
                <ScrollingNews className="w-full" />
            </div>
            
            <div className="w-full md:full">
              <h2 className="text-2xl font-bold mb-6 text-gray-800">Resources & Learning</h2>
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-xl p-6 hover:shadow-lg transition duration-300 cursor-pointer flex flex-col items-center"
                    onClick={handleVideoClick}
                  >
                    <img src={videoicon} alt="Instructional Video" className="h-16 mb-4" />
                    <h3 className="text-xl font-semibold mb-2">Instructional Videos</h3>
                    <p className="text-center text-blue-100">Learn essential career skills through our curated video library</p>
                  </div>
                  
                  <Link to="/success-stories" className="block">
                    <div className="bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-xl p-6 hover:shadow-lg transition duration-300 flex flex-col items-center h-full">
                      <img src={tick} alt="Success Story Video" className="h-16 mb-4" />
                      <h3 className="text-xl font-semibold mb-2">Success Stories</h3>
                      <p className="text-center text-green-100">Get inspired by the journeys of professionals who found success</p>
                    </div>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* <Blogs/> */}

        {/* Testimonials section
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-10 text-gray-800">Success Stories</h2>
          
          <Swiper
            effect="fade"
            pagination={{ clickable: true }}
            navigation={true}
            autoplay={{ delay: 5000, disableOnInteraction: false }}
            modules={[EffectFade, Pagination, Navigation, Autoplay]}
            className="mySwiper"
          >
            {testimonials.map((testimonial, index) => (
              <SwiperSlide key={index}>
                <div className="bg-white rounded-xl shadow-md p-8 md:p-12 max-w-4xl mx-auto text-center">
                  <div className="w-24 h-24 rounded-full overflow-hidden mx-auto mb-6 border-4 border-blue-100">
                    <img src={testimonial.image} alt={testimonial.name} className="w-full h-full object-cover" />
                  </div>
                  <p className="text-xl md:text-2xl italic mb-6 text-gray-600">"{testimonial.quote}"</p>
                  <h3 className="text-xl font-semibold">{testimonial.name}</h3>
                  <p className="text-blue-600">{testimonial.role}</p>
                </div>
              </SwiperSlide>
            ))}
          </Swiper>

        </section> */}
         
      </main>
      <Footer />
      <ChatbotIcon />
    </div>
  );
}

export default Home;