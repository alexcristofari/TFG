import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/">Início</Link>
      <Link to="/discover">Descobrir</Link>
      <Link to="/profile">Perfil</Link>
    </nav>
  );
};

export default Navbar;
