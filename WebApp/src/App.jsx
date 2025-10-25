import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import UploadPage from "./UploadPage";
import LiveDemoPage from "./LiveDemoPage";

// Home Page Component
function HomePage() {
  return (
    <div className="app-container">
      <h1 className="app-title">PoseDetection</h1>

      <p className="app-description">
        Welcome to <strong>PoseDetection</strong> — a web tool that analyzes
        human body poses in videos or live camera feeds. You can upload a
        pre-recorded video or try a real-time demo using your webcam to see pose
        tracking in action!
      </p>

      <div className="button-container">
        <a href="/upload" className="upload-button">
          Upload a Video
        </a>

        <a href="/live" className="demo-button">
          Record a Live Demo
        </a>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/live" element={<LiveDemoPage />} />
      </Routes>
    </Router>
  );
}
