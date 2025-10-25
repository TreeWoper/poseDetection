import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./App.css";

export default function LiveDemoPage() {
  const [output, setOutput] = useState("");

  const runPython = async () => {
    try {
      const response = await fetch("http://localhost:5000/run-python");
      const text = await response.text();
      setOutput(text);
    } catch (err) {
      setOutput("Error calling backend.");
    }
  };

  return (
    <div className="page-container">
      <h2 className="page-title">Record a Live Demo</h2>
      <p className="page-description">
        Here you’ll be able to record a live demo using your webcam.
      </p>

      <button onClick={runPython} className="demo-button">
        Run Python Script
      </button>

      {output && (
        <pre style={{ marginTop: "20px", textAlign: "left", whiteSpace: "pre-wrap" }}>
          {output}
        </pre>
      )}

      <Link to="/" className="back-button" style={{ marginTop: "30px" }}>
        ⬅ Back to Home
      </Link>
    </div>
  );
}
