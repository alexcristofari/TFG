import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import Discover from "./pages/Discover";
import Profile from "./pages/Profile";

function App() {
  return (
    <Router>
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
    </Router>
  );
}

export default App;
