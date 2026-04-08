import React from 'react';
import { ChatInterface } from '../chat/ChatInterface';

interface ResultViewProps {
  error: string | null;
  result: any;
}

export function ResultView({ error, result }: ResultViewProps) {
  if (!error && result === null) {
    return null;
  }

  return (
    <section className="panel">
      <h2>Result & AI Therapy Plan</h2>
      {error && <p className="err">{error}</p>}
      {result !== null && !error && (
        <div className="result">
          
          {result.therapy && result.therapy.therapy_plan && (
            <div className="therapy-plan" style={{ whiteSpace: "pre-wrap", background: "#f0fdf4", color: "#166534", padding: "1rem", borderRadius: "8px", border: "1px solid #bbf7d0", marginBottom: "1rem", direction: "rtl", textAlign: "right", fontFamily: "Traditional Arabic, Arial, sans-serif" }}>
              <h3 style={{ marginTop: 0, color: "#14532d" }}>خطة العلاج المقترحة:</h3>
              {result.therapy.therapy_plan}
            </div>
          )}
          
          <details>
            <summary style={{ cursor: "pointer", fontWeight: "bold" }}>View Raw JSON</summary>
            <pre style={{ marginTop: "1rem" }}>{JSON.stringify(result, null, 2)}</pre>
          </details>

          <hr style={{ margin: "2rem 0", borderColor: "#eee" }} />
          
          <ChatInterface />
        </div>
      )}
    </section>
  );
}
