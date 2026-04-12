import { useMemo, useState } from "react";
import { ChatInterface } from "@/features/chat";
import { AnalysisResponse } from "@/shared/types";

type InputMode = "symptoms" | "labs" | "image";

interface UserInterfaceViewProps {
  loading: boolean;
  result: AnalysisResponse | null;
  error: string | null;
  runLabs: (labsJson: string, symptomsExtra: string) => Promise<void>;
  runImage: (file: File | null) => Promise<void>;
  runSymptoms: (symptomText: string, useParser: boolean) => Promise<void>;
}

const DEFAULT_LABS_JSON = `{
  "glucose": 145,
  "hemoglobin": 11.2
}`;

export function UserInterfaceView({
  loading,
  result,
  error,
  runLabs,
  runImage,
  runSymptoms,
}: UserInterfaceViewProps) {
  const [inputMode, setInputMode] = useState<InputMode>("symptoms");
  const [symptomText, setSymptomText] = useState(
    "Fatigue and increased thirst for two weeks.",
  );
  const [useParser, setUseParser] = useState(true);
  const [labsJson, setLabsJson] = useState(DEFAULT_LABS_JSON);
  const [symptomsExtra, setSymptomsExtra] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);

  const finalDiagnosis = result?.diagnosis?.final_diagnosis;
  const summaryText = result?.diagnosis?.summary;
  const therapyPlan = result?.therapy?.therapy_plan;
  const safetyReasons = result?.diagnosis?.safety?.reasons ?? [];
  const clarification = result?.diagnosis?.clarification;

  const submitDisabled = useMemo(() => {
    if (loading) return true;
    if (inputMode === "image") {
      return !imageFile;
    }
    if (inputMode === "labs") {
      return !labsJson.trim();
    }
    return !symptomText.trim();
  }, [imageFile, inputMode, labsJson, loading, symptomText]);

  const submitLabel = loading ? "Analyzing..." : "Analyze now";

  const handleSubmit = async () => {
    if (inputMode === "image") {
      await runImage(imageFile);
      return;
    }

    if (inputMode === "labs") {
      await runLabs(labsJson, symptomsExtra);
      return;
    }

    await runSymptoms(symptomText, useParser);
  };

  return (
    <section className="user-experience">
      <section className="user-hero">
        <div>
          <p className="user-hero__eyebrow">User Interface</p>
          <h2 className="user-hero__title">
            Guided medical analysis with a simplified, chat-first flow
          </h2>
          <p className="user-hero__subtitle">
            Enter symptoms, paste lab values, or upload a report image. This mode is designed for
            non-technical use and focuses on concise, actionable outputs.
          </p>
        </div>

        <div className="user-hero__status">
          <span className="user-status-pill">
            {result ? "Latest analysis is ready" : "Waiting for your first analysis"}
          </span>
          {finalDiagnosis?.diagnosis && (
            <p className="user-hero__diagnosis">
              Current likely diagnosis: <strong>{finalDiagnosis.diagnosis}</strong>
            </p>
          )}
        </div>
      </section>

      <div className="user-layout">
        <section className="user-panel">
          <h3>1. Choose what you want to analyze</h3>

          <div className="user-mode-picker" role="tablist" aria-label="Input mode">
            <button
              type="button"
              role="tab"
              aria-selected={inputMode === "symptoms"}
              className={inputMode === "symptoms" ? "is-active" : ""}
              onClick={() => setInputMode("symptoms")}
            >
              Symptoms text
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={inputMode === "labs"}
              className={inputMode === "labs" ? "is-active" : ""}
              onClick={() => setInputMode("labs")}
            >
              Lab JSON
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={inputMode === "image"}
              className={inputMode === "image" ? "is-active" : ""}
              onClick={() => setInputMode("image")}
            >
              Report image
            </button>
          </div>

          {inputMode === "symptoms" && (
            <div className="field">
              <label htmlFor="user-symptom-text">Describe symptoms in plain language</label>
              <textarea
                id="user-symptom-text"
                value={symptomText}
                onChange={(event) => setSymptomText(event.target.value)}
                placeholder="Example: Fever, headache, and sore throat for 3 days."
              />
              <label className="user-inline-check">
                <input
                  type="checkbox"
                  checked={useParser}
                  onChange={(event) => setUseParser(event.target.checked)}
                />
                Use advanced symptom parser
              </label>
            </div>
          )}

          {inputMode === "labs" && (
            <>
              <div className="field">
                <label htmlFor="user-labs-json">Paste lab values (JSON)</label>
                <textarea
                  id="user-labs-json"
                  value={labsJson}
                  onChange={(event) => setLabsJson(event.target.value)}
                  spellCheck={false}
                />
              </div>
              <div className="field">
                <label htmlFor="user-labs-symptoms">Optional symptoms</label>
                <input
                  id="user-labs-symptoms"
                  type="text"
                  value={symptomsExtra}
                  onChange={(event) => setSymptomsExtra(event.target.value)}
                  placeholder="Example: fatigue and thirst"
                />
              </div>
            </>
          )}

          {inputMode === "image" && (
            <div className="field">
              <label className="dropzone user-dropzone">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/bmp"
                  onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
                />
                {imageFile ? (
                  <span>
                    Selected image: <strong>{imageFile.name}</strong>
                  </span>
                ) : (
                  <span>Click to upload a lab report image</span>
                )}
              </label>
            </div>
          )}

          <button
            type="button"
            className="btn user-primary-btn"
            disabled={submitDisabled}
            onClick={() => void handleSubmit()}
          >
            {submitLabel}
          </button>
        </section>

        <section className="user-panel">
          <h3>2. Review your simplified summary</h3>

          {error && <p className="err">{error}</p>}

          {!result && !error && (
            <p className="user-placeholder">
              Run an analysis to receive a diagnosis summary, safety notes, and therapy guidance.
            </p>
          )}

          {result && !error && (
            <div className="user-summary-stack">
              <article className="user-summary-card">
                <p className="user-summary-label">Likely condition</p>
                <p className="user-summary-value">
                  {finalDiagnosis?.diagnosis ?? "No final diagnosis was produced."}
                </p>
                {finalDiagnosis?.confidence !== undefined && (
                  <p className="user-summary-meta">Confidence: {String(finalDiagnosis.confidence)}</p>
                )}
              </article>

              {summaryText && (
                <article className="user-summary-card">
                  <p className="user-summary-label">Clinical summary</p>
                  <p>{summaryText}</p>
                </article>
              )}

              {therapyPlan && (
                <article className="user-summary-card">
                  <p className="user-summary-label">Therapy guidance</p>
                  <p>{therapyPlan}</p>
                </article>
              )}

              {!!safetyReasons.length && (
                <article className="user-summary-card">
                  <p className="user-summary-label">Safety notes</p>
                  <ul className="flat-list">
                    {safetyReasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </article>
              )}

              {clarification?.needed && (
                <article className="user-summary-card">
                  <p className="user-summary-label">Follow-up questions recommended</p>
                  {clarification.questions?.length ? (
                    <ul className="flat-list">
                      {clarification.questions.map((item) => (
                        <li key={item.question}>{item.question}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>Additional clarification is recommended by the engine.</p>
                  )}
                </article>
              )}
            </div>
          )}
        </section>
      </div>

      <section className="user-panel user-panel--chat">
        <h3>3. Continue with medical Q&A</h3>
        <p className="user-chat-note">
          Use the chat to ask for explanation of findings, risks, and next steps.
        </p>
        <ChatInterface />
      </section>
    </section>
  );
}
