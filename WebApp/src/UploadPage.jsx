import React from "react";
import { Link } from "react-router-dom";
import "./App.css";

export default function UploadPage() {
  return (
    <div className="page-container">
      <h2 className="page-title">Upload a Video</h2>
      <p className="page-description">
        Here you’ll be able to upload your video for pose detection analysis.
        (Feature coming soon!)
      </p>

      <Link to="/" className="back-button">
        ⬅ Back to Home
      </Link>
    </div>
  );
}
