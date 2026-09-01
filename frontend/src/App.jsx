import { useState } from "react";

// Adjust if your backend runs somewhere other than localhost:8000.
const API_URL = "/documents/classify";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  function handleFileChange(e) {
    setFile(e.target.files[0] || null);
    setResult(null);
    setError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Contract Clause Classifier</h1>
        <p>Upload a contract (PDF or .txt). Each clause gets a CUAD label.</p>
      </header>

      <form className="upload-form" onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={handleFileChange}
          disabled={loading}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? "Classifying…" : "Classify contract"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {loading && (
        <p className="status">
          Running retrieval + local model over each clause — this can take a
          while for longer contracts.
        </p>
      )}

      {result && (
        <section className="results">
          <h2>
            {result.filename} — {result.total_clauses} clause
            {result.total_clauses === 1 ? "" : "s"}
          </h2>

          <ul className="clause-list">
            {result.clauses.map((clause) => (
              <li key={clause.clause_index} className="clause-card">
                <div className="clause-label">{clause.predicted_label}</div>
                <p className="clause-text">{clause.clause_text}</p>
                {clause.retrieved_labels.length > 0 && (
                  <details>
                    <summary>Retrieved evidence</summary>
                    <ul className="evidence-list">
                      {clause.retrieved_labels.map((label, i) => (
                        <li key={i}>
                          {label} — score {clause.retrieved_scores[i]}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}