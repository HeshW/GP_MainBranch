import { useState } from "react";

interface LabAnalysisPanelProps {
  loading: boolean;
  onRun: (labsJson: string, symptomsExtra: string) => void;
}

const DEFAULT_LABS_JSON = `{
  "glucose": 145,
  "hemoglobin": 11.2
}`;

export function LabAnalysisPanel({ loading, onRun }: LabAnalysisPanelProps) {
  const [labsJson, setLabsJson] = useState(DEFAULT_LABS_JSON);
  const [symptomsExtra, setSymptomsExtra] = useState("");

  return (
    <section className="panel">
      <h2>Diagnosis from lab JSON</h2>
      <div className="field">
        <label htmlFor="labs-json">Lab values (JSON object)</label>
        <textarea
          id="labs-json"
          value={labsJson}
          onChange={(event) => setLabsJson(event.target.value)}
          spellCheck={false}
        />
        <p className="field-hint">
          Keys should match OCR output such as <code>glucose</code> and <code>hemoglobin</code>.
          You can also include units like <code>{`{"glucose": {"value": 95, "unit": "mg/dL"}}`}</code>.
        </p>
      </div>

      <div className="field">
        <label htmlFor="symptoms-extra">Optional symptoms</label>
        <input
          id="symptoms-extra"
          type="text"
          value={symptomsExtra}
          onChange={(event) => setSymptomsExtra(event.target.value)}
          placeholder="e.g. fatigue and thirst"
        />
      </div>

      <button
        type="button"
        className="btn"
        disabled={loading}
        onClick={() => onRun(labsJson, symptomsExtra)}
      >
        {loading ? "Running..." : "Run analysis"}
      </button>
    </section>
  );
}
