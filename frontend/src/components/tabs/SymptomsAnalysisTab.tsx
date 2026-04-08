import React, { useState } from 'react';

interface SymptomsAnalysisTabProps {
  loading: boolean;
  onRun: (symptomText: string, useParser: boolean) => void;
}

export function SymptomsAnalysisTab({ loading, onRun }: SymptomsAnalysisTabProps) {
  const [symptomText, setSymptomText] = useState("Fatigue and increased thirst for two weeks.");
  const [useParser, setUseParser] = useState(true);

  return (
    <section className="panel">
      <h2>Symptoms → parser (optional) → diagnosis</h2>
      <div className="field">
        <label htmlFor="symptom-text">Symptom description</label>
        <textarea
          id="symptom-text"
          value={symptomText}
          onChange={(e) => setSymptomText(e.target.value)}
        />
      </div>
      <div className="field">
        <label>
          <input
            type="checkbox"
            checked={useParser}
            onChange={(e) => setUseParser(e.target.checked)}
          />{" "}
          Use symptom parser / validator
        </label>
      </div>
      <button
        type="button"
        className="btn"
        disabled={loading}
        onClick={() => onRun(symptomText, useParser)}
      >
        {loading ? "Running…" : "Run analysis"}
      </button>
    </section>
  );
}
