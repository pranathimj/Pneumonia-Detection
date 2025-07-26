import React, { useState } from 'react';
import './App.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';
import pneumoniaImage from './pneumonia.jpg';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showDetector, setShowDetector] = useState(false);

  const handleUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setToast(true);
      setPreviewUrl(URL.createObjectURL(uploadedFile));
      setTimeout(() => setToast(false), 3000);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setPreviewUrl(null);
    setLoading(false);
  };

  const handleShowDetector = () => {
    setShowDetector(true);
  };

  const handleBackToLanding = () => {
    setShowDetector(false);
    setFile(null);
    setResult(null);
    setPreviewUrl(null);
    setLoading(false);
  };

  const renderLandingPage = () => (
    <div className="landing-page">
      <h1 className="landing-title">Pneumonia Detection System</h1>
      
      <div className="landing-section">
        <h3>What is Pneumonia?</h3>
        <div className="pneumonia-content">
          <div className="pneumonia-text">
            <p>
              Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus, 
              causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including 
              bacteria, viruses, and fungi, can cause pneumonia.
            </p>
            <p>
              Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young 
              children, people older than age 65, and people with health problems or weakened immune systems.
            </p>
          </div>
          <div className="pneumonia-image">
            <img src={pneumoniaImage} alt="Pneumonia X-ray showing infected lungs" />
          </div>
        </div>
      </div>

      <div className="landing-section">
        <h3>Common Causes</h3>
        <ul>
          <li><strong>Bacterial pneumonia:</strong> Most commonly caused by Streptococcus pneumoniae</li>
          <li><strong>Viral pneumonia:</strong> Caused by various viruses including influenza and COVID-19</li>
          <li><strong>Mycoplasma pneumonia:</strong> Caused by Mycoplasma pneumoniae bacteria</li>
          <li><strong>Fungal pneumonia:</strong> More common in people with chronic health problems or weakened immune systems</li>
          <li><strong>Aspiration pneumonia:</strong> Occurs when you inhale food, drink, vomit, or saliva into your lungs</li>
        </ul>
      </div>

      <div className="landing-section">
        <h3>Prevention & Precautions</h3>
        <ul>
          <li><strong>Get vaccinated:</strong> Vaccines are available for pneumococcal pneumonia and flu</li>
          <li><strong>Practice good hygiene:</strong> Wash your hands regularly and use hand sanitizer</li>
          <li><strong>Don't smoke:</strong> Smoking damages your lungs' natural defenses against respiratory infections</li>
          <li><strong>Keep your immune system strong:</strong> Get adequate sleep, exercise regularly, and eat a healthy diet</li>
          <li><strong>Avoid sick people:</strong> Stay away from people who have colds, flu, or other respiratory infections when possible</li>
          <li><strong>Manage chronic conditions:</strong> Keep conditions like diabetes, heart disease, and lung disease under control</li>
        </ul>
      </div>

      <button className="detect-btn" onClick={handleShowDetector}>
        🔬 Detect with AI
      </button>
    </div>
  );

  const renderDetector = () => (
    <>
      <button className="back-btn" onClick={handleBackToLanding}>
        ← Back to Info
      </button>
      {toast && <div className="toast">Upload Success ✅</div>}

      {!result ? (
        <div className="card">
          <h2 className="title">AI Pneumonia Detector</h2>
          {!file ? (
            <>
              <label htmlFor="file-upload" className="upload-label">
                <FontAwesomeIcon icon={faUpload} className="upload-icon" />
                <span>Click to Upload X-ray Image</span>
              </label>
              <input
                type="file"
                id="file-upload"
                accept="image/*"
                onChange={handleUpload}
                style={{ display: 'none' }}
              />
            </>
          ) : (
            <div className="upload-preview">
              <div className="preview-image">
                <img src={previewUrl} alt="Uploaded X-ray preview" />
              </div>
              <div className="button-group">
                <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Analyzing...' : 'Analyze'}
                </button>
                <button className="reset-btn" onClick={handleReset} disabled={loading}>
                  Upload New Image
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card result-card">
          <h2 className="result-title">Prediction Result</h2>
          <div className="result-body">
            <div className="result-info">
              {result.diagnosis.toLowerCase() === "unknown" ? (
                <div style={{ color: 'red', fontWeight: 'bold' }}>
                  Unknown image. Please upload a valid X-ray image.
                </div>
              ) : (
                <>
                  <p><strong>Diagnosis:</strong> {result.diagnosis}</p>
                  <p><strong>Confidence:</strong> {result.confidence.toFixed(2)}%</p>
                </>
              )}
              <button className="reset-btn" onClick={handleReset} style={{ marginTop: '20px' }}>
                Try Another Image
              </button>
            </div>
            <div className="result-image">
              <img src={previewUrl} alt="Uploaded X-ray" />
            </div>
          </div>
        </div>
      )}
    </>
  );

  return (
    <div className="main-frame">
      {!showDetector ? renderLandingPage() : renderDetector()}
    </div>
  );
}

export default App;
