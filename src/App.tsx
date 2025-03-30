import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import AuthPageForBalvidya from "./components/BalVidya/AuthPageForBalvidya";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";
import AuthPageForMadhyamik from "./components/Madhyamik/AuthPageForMadhyamik";

export default function App() {
  const [activeComponent, setActiveComponent] = useState("hero");

  return (
    <div>
      <Navbar setActiveComponent={setActiveComponent} />
      {activeComponent === "hero" && <HeroSection />}
      {activeComponent === "BalVidya" && <AuthPageForBalvidya />}
      {activeComponent === "Madhyamik" && <AuthPageForMadhyamik />}

      {activeComponent === "contact" && <Contact />}
    </div>
  );
}