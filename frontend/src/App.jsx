import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Search from './pages/Search';
import Dashboard from './pages/Dashboard';
import Admin from './pages/Admin';
import Analysis from './pages/Analysis';
import Navbar from './components/Navbar';

function App() {
  return (
    <HashRouter>
      <div className="min-h-screen flex flex-col font-sans">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/search" element={<Search />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
        </main>
        <footer className="bg-slate-900 text-slate-400 py-8 text-center">
          <p>© {new Date().getFullYear()} GovScheme Advisor. This is an educational project.</p>
        </footer>
      </div>
    </HashRouter>
  );
}

export default App;
