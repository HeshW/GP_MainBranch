import React, { useState } from 'react';

interface ImageAnalysisTabProps {
  loading: boolean;
  onRun: (file: File | null) => void;
}

export function ImageAnalysisTab({ loading, onRun }: ImageAnalysisTabProps) {
  const [file, setFile] = useState<File | null>(null);

  return (
    <section className="panel">
      <h2>OCR + diagnosis from image</h2>
      <label className="dropzone">
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <span>
            Selected: <strong>{file.name}</strong>
          </span>
        ) : (
          <span>Click or drop a lab report image here</span>
        )}
      </label>
      <p className="field-hint" style={{ marginTop: "1rem" }}>
        Requires PaddleOCR and runtime deps installed on the server.
      </p>
      <button
        type="button"
        className="btn"
        style={{ marginTop: "0.75rem" }}
        disabled={loading || !file}
        onClick={() => onRun(file)}
      >
        {loading ? "Running…" : "Run OCR + analysis"}
      </button>
    </section>
  );
}
