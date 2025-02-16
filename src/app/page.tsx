// src/app/page.tsx
"use client";

import { useState, ChangeEvent } from "react";

interface PredictionResult {
  prediction: string;
  probability: number;
  caption: string;
  code: string;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Handle file selection
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setPredictions(null);
      setError("");
    }
  };

  // Upload the image and get predictions
  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image.");
      return;
    }
    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "Failed to get prediction.");
        setLoading(false);
        return;
      }
      const data = await res.json();
      setPredictions(data.predictions);
    } catch (err) {
      console.error("Error during prediction:", err);
      setError("Error during prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Real-Time Violation Detection</h1>
      <p>Select an image to get predictions from the model.</p>

      <div style={{ marginBottom: "20px" }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleUpload} style={{ marginLeft: "10px" }} disabled={loading}>
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {predictions && (
        <div>
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((pred, index) => (
              <li key={index}>
                <strong>Prediction:</strong> {pred.prediction} (Confidence: {(pred.probability * 100).toFixed(2)}%)<br />
                <strong>Violation Code:</strong> {pred.code}<br />
                <strong>Caption:</strong> {pred.caption}
                <hr />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
