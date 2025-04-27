import React from 'react';
import { Link } from 'react-router-dom';
import { FaFacebookSquare, FaTwitterSquare, FaLinkedin, FaInstagramSquare, FaPhoneAlt, FaEnvelope, FaMapMarkerAlt } from 'react-icons/fa';

function Footer() {
  return (
    <footer className="bg-blue-950 text-white pt-12 pb-6">
      <div className="container mx-auto px-7">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Company Info */}
          <div>
            <h3 className="text-xl font-bold mb-4">SkillBridge AI</h3>
            <p className="text-gray-300 mb-4">
              India's leading AI-powered career platform connecting talent with opportunities.
            </p>
            <div className="flex space-x-3">
              <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition duration-300">
                <FaFacebookSquare size={24} />
              </a>
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition duration-300">
                <FaTwitterSquare size={24} />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition duration-300">
                <FaLinkedin size={24} />
              </a>
              <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition duration-300">
                <FaInstagramSquare size={24} />
              </a>
            </div>
          </div>
          {/* Quick Links */}
          <div>
            <h3 className="text-xl font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/aboutus" className="text-gray-300 hover:text-blue-400 transition duration-300">About Us</Link>
              </li>
              <li>
                <Link to="/Login" className="text-gray-300 hover:text-blue-400 transition duration-300">Browse Jobs</Link>
              </li>
              <li>
                <Link to="/instructional-videos" className="text-gray-300 hover:text-blue-400 transition duration-300">Instructional Videos</Link>
              </li>
              <li>
                <Link to="/success-stories" className="text-gray-300 hover:text-blue-400 transition duration-300">Success Stories</Link>
              </li>
            </ul>
          </div>
          <div></div>

          {/* Contact Info */}
          <div>
            <h3 className="text-xl font-bold mb-4">Contact Us</h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <FaMapMarkerAlt className="text-blue-400 mt-1 mr-3" />
                <span className="text-gray-300">VESIT, Chembur 400074, Mumbai, India</span>
              </li>
              <li className="flex items-center">
                <FaPhoneAlt className="text-blue-400 mr-3" />
                <span className="text-gray-300">+91 9082664531</span>
              </li>
              <li className="flex items-center">
                <FaEnvelope className="text-blue-400 mr-3" />
                <span className="text-gray-300">psoham104@gmail.com</span>
              </li>
              <li className="mt-4">
                <Link 
                  to="/contactus" 
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 inline-block"
                >
                  Contact Us
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-700 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm mb-4 md:mb-0">
              &copy; {new Date().getFullYear()} SkillBridge AI. All Rights Reserved.
            </p>
            <div className="flex space-x-4">
              <Link to="/privacy-policy" className="text-gray-400 text-sm hover:text-blue-400 transition duration-300">
                Privacy Policy
              </Link>
              <Link to="/terms-of-service" className="text-gray-400 text-sm hover:text-blue-400 transition duration-300">
                Terms of Service
              </Link>
              <Link to="/cookies" className="text-gray-400 text-sm hover:text-blue-400 transition duration-300">
                Cookies
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;