import { Link } from 'react-router-dom';
import Logo from '../assets/logo.png';

const Navbar = () => {
  return (
    <div className="bg-white shadow-md py-4 sticky top-0 z-10">
      <div className="container mx-auto flex items-center justify-between px-7">
        <Link to="/" className="font-bold text-2xl text-blue-500 transition-transform hover:scale-105">
          <img src={Logo} alt="SkillBridge AI" className="w-40 h-auto" />
        </Link>

        <div className="flex space-x-4">
          <Link to="/login">
            <button className="text-white bg-blue-500 hover:bg-blue-700 focus:ring-2 focus:ring-blue-300 font-bold rounded-lg text-sm px-5 py-2.5 transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg">
              LOGIN
            </button>
          </Link>
          <Link to="/signup">
            <button className="text-white bg-blue-500 hover:bg-blue-700 focus:ring-2 focus:ring-blue-300 font-bold rounded-lg text-sm px-5 py-2.5 transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg">
              REGISTER
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Navbar;