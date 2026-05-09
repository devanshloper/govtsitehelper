import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Landmark, Search, User, Shield, BarChart3 } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const navLinks = [
    { name: 'Home', path: '/', icon: <Landmark className="w-5 h-5 mr-1" /> },
    { name: 'Search', path: '/search', icon: <Search className="w-5 h-5 mr-1" /> },
    { name: 'Analysis', path: '/analysis', icon: <BarChart3 className="w-5 h-5 mr-1" /> },
    { name: 'Dashboard', path: '/dashboard', icon: <User className="w-5 h-5 mr-1" /> },
    { name: 'Admin', path: '/admin', icon: <Shield className="w-5 h-5 mr-1" /> },
  ];

  return (
    <nav className="bg-white shadow-sm border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <div className="bg-primary-600 p-2 rounded-lg mr-2">
                <Landmark className="w-6 h-6 text-white" />
              </div>
              <span className="font-bold text-xl text-slate-900 tracking-tight">GovScheme Advisor</span>
            </Link>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                to={link.path}
                className={`flex items-center px-3 py-2 text-sm font-medium transition-colors duration-200 ${
                  location.pathname === link.path
                    ? 'text-primary-600 border-b-2 border-primary-600'
                    : 'text-slate-600 hover:text-primary-600'
                }`}
              >
                {link.icon}
                {link.name}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
