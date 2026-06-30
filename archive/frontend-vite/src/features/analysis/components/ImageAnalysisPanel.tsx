import { useState } from "react";

interface ImageAnalysisPanelProps {
  loading: boolean;
  onRun: (file: File | null) => void;
}

export function ImageAnalysisPanel({ loading, onRun }: ImageAnalysisPanelProps) {
  const [file, setFile] = useState<File | null>(null);

  return (
    <section className="panel">
      <h2>OCR and diagnosis from image</h2>
      <label className="dropzone">
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        {file ? (
          <span>
            Selected: <strong>{file.name}</strong>
          </span>
        ) : (
          <span>Click or drop a lab report image here</span>
        )}
      </label>

      <p className="field-hint field-hint--spaced">
        Requires PaddleOCR and backend runtime dependencies on the server.
      </p>

      <button
        type="button"
        className="btn btn--spaced"
        disabled={loading || !file}
        onClick={() => onRun(file)}
      >
        {loading ? "Running..." : "Run OCR and analysis"}
      </button>
    </section>
  );
}
