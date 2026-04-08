import React, { useState } from 'react';

interface LabAnalysisTabProps {
  loading: boolean;
  onRun: (labsJson: string, symptomsExtra: string) => void;
}

export function LabAnalysisTab({ loading, onRun }: LabAnalysisTabProps) {
  const [labsJson, setLabsJson] = useState('{\n  "glucose": 145,\n  "hemoglobin": 11.2\n}');
  const [symptomsExtra, setSymptomsExtra] = useState("");

  return (
    <section className="panel">
      <h2>Diagnosis from lab JSON</h2>
      <div className="field">
        <label htmlFor="labs-json">Lab values (JSON object)</label>
        <textarea
          id="labs-json"
          value={labsJson}
          onChange={(e) => setLabsJson(e.target.value)}
          spellCheck={false}
        />
        <p className="field-hint">
          Keys match OCR output (e.g. glucose, hemoglobin). Optional
          units: <code>{`{"glucose": {"value": 95, "unit": "mg/dL"}}`}</code>
        </p>
      </div>
      <div className="field">
        <label htmlFor="symptoms-extra">Optional symptoms (merged)</label>
        <input
          id="symptoms-extra"
          type="text"
          value={symptomsExtra}
          onChange={(e) => setSymptomsExtra(e.target.value)}
          placeholder="e.g. fatigue and thirst"
        />
      </div>
      <button
        type="button"
        className="btn"
        disabled={loading}
        onClick={() => onRun(labsJson, symptomsExtra)}
      >
        {loading ? "Running…" : "Run analysis"}
      </button>
    </section>
  );
}
