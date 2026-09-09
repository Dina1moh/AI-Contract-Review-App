import { Component, useState } from "react";

const API_URL = "/documents/classify/stream";

class AppErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="page">
          <h1>Contract Clause Classifier</h1>
          <div className="error" role="alert">
            The interface failed to render. Refresh the page and try again.
            <details>
              <summary>Technical details</summary>
              <pre>{this.state.error.message}</pre>
            </details>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}

function ContractClassifier() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [trace, setTrace] = useState([]);

  function handleFileChange(e) {
    setFile(e.target.files[0] || null);
    setResult(null);
    setError(null);
    setTrace([]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setTrace([]);

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

      if (!response.body) {
        throw new Error("The server did not provide a progress stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let receivedResult = false;

      const handleEvent = (rawEvent) => {
        const dataLine = rawEvent
          .split("\n")
          .find((line) => line.startsWith("data:"));
        if (!dataLine) return;

        const event = JSON.parse(dataLine.slice(5).trim());
        if (event.type === "progress") {
          setTrace((items) => [...items, event]);
        } else if (event.type === "result") {
          if (!event.data || !Array.isArray(event.data.clauses)) {
            throw new Error("The server returned an invalid classification response.");
          }
          setResult(event.data);
          receivedResult = true;
        } else if (event.type === "error") {
          setTrace((items) => [
            ...items,
            { stage: "error", message: event.message || "Classification failed." },
          ]);
          throw new Error(event.message || "Classification failed.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        events.forEach(handleEvent);
        if (done) break;
      }
      if (buffer.trim()) handleEvent(buffer);
      if (!receivedResult) throw new Error("The classification stream ended without a result.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function riskStatusText(clause) {
    if (clause.risk) return "Potential Risk";
    const reason = String(clause.reason || "").toLowerCase();
    if (reason.includes("not configured")) {
      return "Risk Analysis Not Configured for This Clause";
    }
    return "No Risk Indicator Detected";
  }

  function isConfigured(clause) {
    return !String(clause.reason || "").toLowerCase().includes("not configured");
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
          {loading ? "Classifying..." : "Classify contract"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {loading && (
        <p className="status">
          Analysis is running. The live trace below shows the current pipeline stage.
        </p>
      )}

      {trace.length > 0 && (
        <section className="trace" aria-live="polite" aria-label="Analysis trace">
          <div className="trace-header">
            <h2>Live analysis trace</h2>
            <span>{loading ? "Running" : error ? "Failed" : "Finished"}</span>
          </div>
          <ol className="trace-list">
            {trace.map((event, index) => (
              <li key={`${event.stage}-${index}`} className={`trace-item trace-item--${event.stage}`}>
                <span className="trace-stage">{event.stage.replaceAll("_", " ")}</span>
                <span>{event.message}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {result && (
        <section className="results">
          <h2>
            {result.filename} - {result.total_clauses} clause
            {result.total_clauses === 1 ? "" : "s"}
          </h2>
          <div className="summary" aria-label="Risk analysis summary">
            <span>Total Clauses: {result.total_clauses}</span>
            <span>Potential Risks: {result.clauses.filter((c) => c?.risk).length}</span>
            <span>No Risk Indicator: {result.clauses.filter((c) => !c?.risk && isConfigured(c || {})).length}</span>
            <span>Supported Categories: {new Set(result.clauses.filter(isConfigured).map((c) => c.clause_type || c.predicted_label)).size}</span>
          </div>

          <ul className="clause-list">
            {result.clauses.map((clause) => (
              <li key={clause.clause_index} className="clause-card">
                <div className="clause-label">
                  {clause.clause_type || clause.predicted_label}
                </div>
                <div
                  className={`risk-badge ${
                    clause.risk ? "risk-badge--risk" : isConfigured(clause) ? "risk-badge--clear" : "risk-badge--unsupported"
                  }`}
                >
                  {riskStatusText(clause)}
                </div>
                {clause.risk_type && (
                  <div className="risk-type">{clause.risk_type}</div>
                )}
                <p className="clause-text">{clause.clause_text}</p>
                {clause.reason && (
                  <div className="risk-detail">
                    <strong>Reason:</strong> {clause.reason}
                  </div>
                )}
                {clause.evidence && (
                  <div className="risk-detail">
                    <strong>Evidence:</strong> {clause.evidence}
                  </div>
                )}
                {clause.source?.name && (
                  <div className="risk-detail">
                    <strong>Source:</strong>{" "}
                    {clause.source.url ? (
                      <a href={clause.source.url} target="_blank" rel="noreferrer">
                        {clause.source.name} - {clause.source.title}
                      </a>
                    ) : (
                      `${clause.source.name} - ${clause.source.title}`
                    )}
                  </div>
                )}
                {clause.legal_knowledge?.length > 0 && (
                  <details>
                    <summary>Retrieved legal guidance</summary>
                    <ul className="guidance-list">
                      {clause.legal_knowledge.map((item, i) => (
                        <li key={`${item.source_url}-${i}`}>
                          <strong>{item.source_name}</strong> — {item.title}
                          {item.relevance_score != null && ` (score ${item.relevance_score})`}
                          <div>{item.retrieved_text}</div>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
                {Array.isArray(clause.retrieved_labels) && clause.retrieved_labels.length > 0 && (
                  <details>
                    <summary>Retrieved evidence</summary>
                    <ul className="evidence-list">
                      {clause.retrieved_labels.map((label, i) => (
                        <li key={i}>
                          {label} - score {clause.retrieved_scores[i]}
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

export default function AppWithErrorBoundary() {
  return (
    <AppErrorBoundary>
      <ContractClassifier />
    </AppErrorBoundary>
  );
}
