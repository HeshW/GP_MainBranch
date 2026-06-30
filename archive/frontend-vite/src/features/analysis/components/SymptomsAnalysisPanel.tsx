import { useState } from "react";

interface SymptomsAnalysisPanelProps {
  loading: boolean;
  onRun: (symptomText: string, useParser: boolean) => void;
}

export function SymptomsAnalysisPanel({ loading, onRun }: SymptomsAnalysisPanelProps) {
  const [symptomText, setSymptomText] = useState(
    "Fatigue and increased thirst for two weeks.",
  );
  const [useParser, setUseParser] = useState(true);

  return (
    <section className="panel">
      <h2>Symptoms to diagnosis workflow</h2>
      <div className="field">
        <label htmlFor="symptom-text">Symptom description</label>
        <textarea
          id="symptom-text"
          value={symptomText}
          onChange={(event) => setSymptomText(event.target.value)}
        />
      </div>

      <div className="field field--checkbox">
        <label>
          <input
            type="checkbox"
            checked={useParser}
            onChange={(event) => setUseParser(event.target.checked)}
          />
          Use symptom parser and validator
        </label>
      </div>

      <button
        type="button"
        className="btn"
        disabled={loading}
        onClick={() => onRun(symptomText, useParser)}
      >
        {loading ? "Running..." : "Run analysis"}
      </button>
    </section>
  );
}
