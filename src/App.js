import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [userInput, setUserInput] = useState("");
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    setError("");  // Clear previous errors
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", {
        review: userInput
      }, {
        headers: { "Content-Type": "application/json" }
      });
      
      setPrediction(response.data.sentiment);
    } catch (err) {
      console.error("Error:", err);
      setError("Failed to fetch prediction. Is the backend running?");
    }
  };

  return (
    <div className="App">
      <nav className="navbar">
        <h1>IMDB Sentiment Classifier</h1>
      </nav>
      
      <div className="landing-container">
        <div className="landing-content">
          <div className="landing-text">
            <h2>Analyze Movie Reviews with AI</h2>
            <p>Discover the sentiment behind movie reviews instantly using our advanced machine learning model.</p>
            <div className="feature-list">
              <div className="feature">
                <span className="feature-icon">🤖</span>
                <p>AI-Powered Analysis</p>
              </div>
              <div className="feature">
                <span className="feature-icon">⚡</span>
                <p>Instant Results</p>
              </div>
              <div className="feature">
                <span className="feature-icon">📊</span>
                <p>Accurate Sentiment</p>
              </div>
            </div>
          </div>
          
          <div className="classifier-section">
            <div className="content">
              <h2>Movie Review Sentiment Analysis</h2>
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="Enter your movie review..."
              />
              <button onClick={handleSubmit}>Predict Sentiment</button>
              {error && <p className="error">{error}</p>}
              {prediction && (
                <div className="sentiment-result">
                  <h3>Sentiment: {prediction}</h3>
                  <div className={`sentiment-indicator ${prediction.toLowerCase()}`}></div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <footer className="footer">
        <p>© 2025 IMDB Sentiment Classifier. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;