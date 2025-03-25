<<<<<<< HEAD
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Discover from './pages/Discover';
import Profile from './pages/Profile';
import Navbar from './components/Navbar';
import './App.css';
=======
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import Discover from "./pages/Discover";
import Profile from "./pages/Profile";
>>>>>>> aa01dc8727fd281a5378793067920dc991a95a04

function App() {
  return (
    <Router>
<<<<<<< HEAD
      <Navbar />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/discover" element={<Discover />} />
          <Route path="/profile" element={<Profile />} />
        </Routes>
      </div>
=======
      <nav className="navbar">
        <Link to="/">Início</Link>
        <Link to="/discover">Descobrir</Link>
        <Link to="/profile">Perfil</Link>
      </nav>
      
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/discover" element={<Discover />} />
        <Route path="/profile" element={<Profile />} />
      </Routes>
>>>>>>> aa01dc8727fd281a5378793067920dc991a95a04
    </Router>
  );
}

export default App;
