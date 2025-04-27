// src/routes/AboutUs.js
import React from 'react';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import Navbar from '../components/Navbar';

const AboutUs = () => {
  const teamMembers = [
    {
      name: "Sushanth Shetty",
      role: "Team Leader",
      description: "Passionate about leveraging technology to create impactful solutions.",
      image: "https://via.placeholder.com/150", // Replace with actual image URLs
    },
    {
      name: "Soham Parab",
      role: "Developer",
      description: "Skilled in full-stack development with a focus on front-end technologies.",
      image: "https://via.placeholder.com/150",
    },
    {
      name: "Vighnarth Nile",
      role: "Data Scientist",
      description: "Adept at machine learning and data analysis to drive insights.",
      image: "https://via.placeholder.com/150",
    },
    {
      name: "Atharva Sambhaji",
      role: "Designer",
      description: "Creative mind behind our UI/UX designs, ensuring a seamless user experience.",
      image: "https://via.placeholder.com/150",
    },
  ];

  // Testimonials data
  const testimonials = [
    {
      quote: "This platform helped me land my dream job! The resources are invaluable.",
      author: "Jane Doe",
    },
    {
      quote: "I found the perfect job thanks to the helpful tips and resources provided here.",
      author: "John Smith",
    },
    {
      quote: "A fantastic experience! The team really cares about helping job seekers.",
      author: "Emily Johnson",
    },
  ];

  // Slider settings
  const sliderSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    autoplay: true,
    autoplaySpeed: 3000,
    slidesToShow: 1,
    slidesToScroll: 1,
  };

  return (
    <div className="bg-gray-100 min-h-screen">
        <Navbar/>
      <div className="container mx-auto py-12 px-6">
        <h1 className="text-4xl font-bold text-blue-600 mb-6">About Us</h1>

        <p className="text-lg text-gray-800 mb-4">
          Welcome to our platform! We are dedicated to helping job seekers find their ideal roles
          and providing resources to enhance their careers. Our team is passionate about connecting
          talent with opportunities in a rapidly evolving job market.
        </p>

        <p className="text-lg text-gray-800 mb-4">
          We believe in the power of knowledge and strive to offer comprehensive tools and guides
          for every step of your job search journey. Whether you are looking to build your resume,
          improve your interview skills, or simply explore new job opportunities, we are here to
          support you.
        </p>

        <p className="text-lg text-gray-800 mb-4">
          Thank you for choosing us as your partner in your career development. We look forward to
          helping you achieve your goals!
        </p>

        {/* Team Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">Meet Our Team</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, index) => (
            <div key={index} className="bg-white rounded-lg shadow-lg p-6">
              <img src={member.image} alt={member.name} className="w-32 h-32 rounded-full mx-auto mb-4" />
              <h3 className="text-xl font-bold text-gray-800">{member.name}</h3>
              <p className="text-gray-600">{member.role}</p>
              <p className="mt-2 text-gray-700">{member.description}</p>
            </div>
          ))}
        </div>

        {/* Core Values Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">Our Core Values</h2>
        <ul className="list-disc pl-6 mb-4">
          <li className="text-lg text-gray-800">Integrity: We believe in transparency and honesty in all our interactions.</li>
          <li className="text-lg text-gray-800">Innovation: We strive to bring the best and latest solutions to job seekers.</li>
          <li className="text-lg text-gray-800">Collaboration: Working together to achieve common goals is at the heart of our mission.</li>
        </ul>

        {/* Testimonials Carousel Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">What Our Users Say</h2>
        <div className="mb-6">
          <Slider {...sliderSettings}>
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-white rounded-lg shadow-lg p-6 text-center">
                <p className="text-gray-600 italic">"{testimonial.quote}"</p>
                <p className="text-gray-800 font-bold">- {testimonial.author}</p>
              </div>
            ))}
          </Slider>
        </div>

        {/* Mission Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">Our Mission</h2>
        <p className="text-lg text-gray-800 mb-4">
          Our mission is to empower job seekers with the knowledge and tools necessary to navigate 
          the job market successfully. We aim to provide a supportive community where everyone can thrive.
        </p>

        {/* Vision Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">Our Vision</h2>
        <p className="text-lg text-gray-800 mb-4">
          We envision a world where every individual can easily access the resources they need to 
          achieve their career aspirations.
        </p>

        {/* Call to Action Section */}
        <h2 className="text-3xl font-semibold text-blue-600 mt-12 mb-4">Join Us Today!</h2>
        <p className="text-lg text-gray-800 mb-4">
          Sign up for our newsletter to stay updated on the latest job-seeking tips and resources.
        </p>
        <a href="/signup" className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700">Sign Up Now</a>
      </div>
    </div>
  );
};

export default AboutUs;
