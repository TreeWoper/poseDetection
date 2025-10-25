import React from "react";
import { Link } from "react-router-dom";
import "./App.css";

export default function LiveDemoPage() {
  return (
    <div className="page-container">
      <h2 className="page-title">Record a Live Demo</h2>
      <p className="page-description">
        Here you’ll be able to record a live demo using your webcam.
        (Feature coming soon!)
      </p>

      <Link to="/" className="back-button">
        ⬅ Back to Home
      </Link>
    </div>
  );
}
