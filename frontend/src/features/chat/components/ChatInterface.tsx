import { useEffect, useRef, useState } from "react";
import { AnalysisTab, ChatTimelineMessage, ClarificationFlowState } from "@/shared/types";

interface ChatInterfaceProps {
  timeline: ChatTimelineMessage[];
  loading: boolean;
  error: string | null;
  clarificationState: ClarificationFlowState | null;
  onSubmitSymptoms: (symptomText: string, useParser: boolean) => Promise<void>;
  onSubmitLabs: (labsJson: string, symptomsExtra: string) => Promise<void>;
  onSubmitImage: (file: File | null) => Promise<void>;
  onSubmitClarificationAnswer: (answer: string) => Promise<void>;
}

const DEFAULT_LABS_JSON = `{
  "glucose": 145,
  "hemoglobin": 11.2
}`;

const MODE_LABELS: Record<AnalysisTab, string> = {
  symptoms: "Symptoms",
  labs: "Labs",
  image: "Image",
};

export function ChatInterface({
  timeline,
  loading,
  error,
  clarificationState,
  onSubmitSymptoms,
  onSubmitLabs,
  onSubmitImage,
  onSubmitClarificationAnswer,
}: ChatInterfaceProps) {
  const [mode, setMode] = useState<AnalysisTab>("symptoms");
  const [symptomDraft, setSymptomDraft] = useState("");
  const [labsDraft, setLabsDraft] = useState(DEFAULT_LABS_JSON);
  const [labsSymptomsDraft, setLabsSymptomsDraft] = useState("");
  const [clarificationDraft, setClarificationDraft] = useState("");
  const [useParser, setUseParser] = useState(true);
  const [imageFile, setImageFile] = useState<File | null>(null);

  const timelineEndRef = useRef<HTMLDivElement | null>(null);

  const clarificationActive =
    clarificationState !== null &&
    clarificationState.nextQuestionIndex < clarificationState.questions.length;

  useEffect(() => {
    timelineEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading, timeline]);

  const handleSubmit = async () => {
    if (loading) {
      return;
    }

    if (clarificationActive) {
      const answer = clarificationDraft.trim();
      if (!answer) {
        return;
      }
      setClarificationDraft("");
      await onSubmitClarificationAnswer(answer);
      return;
    }

    if (mode === "symptoms") {
      const text = symptomDraft.trim();
      if (!text) {
        return;
      }
      setSymptomDraft("");
      await onSubmitSymptoms(text, useParser);
      return;
    }

    if (mode === "labs") {
      await onSubmitLabs(labsDraft, labsSymptomsDraft);
      return;
    }

    await onSubmitImage(imageFile);
  };

  const submitLabel = clarificationActive
    ? "Send clarification answer"
    : mode === "symptoms"
      ? "Run symptom analysis"
      : mode === "labs"
        ? "Run labs analysis"
        : "Run image analysis";

  return (
    <section className="chat-surface">
      <header className="chat-surface__header">
        <div>
          <h2>Diagnostic conversation</h2>
          <p>
            Start with symptoms, labs, or an image. Diagnosis and follow-up prompts will stay
            in this thread.
          </p>
        </div>

        <div className="mode-switch" role="tablist" aria-label="Input mode">
          {(["symptoms", "labs", "image"] as AnalysisTab[]).map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              disabled={clarificationActive}
              aria-selected={mode === item}
              className={mode === item ? "mode-switch__button is-active" : "mode-switch__button"}
              onClick={() => setMode(item)}
            >
              {MODE_LABELS[item]}
            </button>
          ))}
        </div>
      </header>

      {error && <p className="chat-alert">{error}</p>}

      <div className="chat-timeline" aria-live="polite">
        {timeline.map((message) => (
          <article
            key={message.id}
            className={`chat-bubble chat-bubble--${message.role} chat-bubble--${message.kind}`}
          >
            <p className="chat-bubble__role">
              {message.role === "user" ? "You" : message.role === "assistant" ? "Assistant" : "System"}
            </p>
            <p className="chat-bubble__content">{message.content}</p>
          </article>
        ))}

        {loading && <p className="chat-loading">Working on your request...</p>}
        <div ref={timelineEndRef} />
      </div>

      <div className="chat-composer-panel">
        {clarificationActive ? (
          <div className="composer-block">
            <label htmlFor="clarification-answer">Your answer</label>
            <input
              id="clarification-answer"
              type="text"
              value={clarificationDraft}
              disabled={loading}
              placeholder="Write your answer to the follow-up question"
              onChange={(event) => setClarificationDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  void handleSubmit();
                }
              }}
            />
          </div>
        ) : (
          <>
            {mode === "symptoms" && (
              <div className="composer-block">
                <label htmlFor="symptom-input">Symptoms</label>
                <textarea
                  id="symptom-input"
                  rows={3}
                  value={symptomDraft}
                  disabled={loading}
                  placeholder="Describe symptoms in natural language"
                  onChange={(event) => setSymptomDraft(event.target.value)}
                />
                <label className="composer-checkbox">
                  <input
                    type="checkbox"
                    checked={useParser}
                    onChange={(event) => setUseParser(event.target.checked)}
                    disabled={loading}
                  />
                  Use symptom parser and validator
                </label>
              </div>
            )}

            {mode === "labs" && (
              <div className="composer-grid">
                <div className="composer-block">
                  <label htmlFor="labs-input">Lab JSON</label>
                  <textarea
                    id="labs-input"
                    rows={7}
                    value={labsDraft}
                    disabled={loading}
                    spellCheck={false}
                    onChange={(event) => setLabsDraft(event.target.value)}
                  />
                </div>
                <div className="composer-block">
                  <label htmlFor="labs-symptoms">Optional symptom context</label>
                  <input
                    id="labs-symptoms"
                    type="text"
                    value={labsSymptomsDraft}
                    disabled={loading}
                    placeholder="e.g. fatigue and thirst"
                    onChange={(event) => setLabsSymptomsDraft(event.target.value)}
                  />
                </div>
              </div>
            )}

            {mode === "image" && (
              <div className="composer-block">
                <label htmlFor="image-input">Report image</label>
                <input
                  id="image-input"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/bmp,image/tif,image/tiff"
                  disabled={loading}
                  onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
                />
                <p className="composer-hint">
                  {imageFile ? `Selected: ${imageFile.name}` : "Choose an image to run OCR + diagnosis."}
                </p>
              </div>
            )}
          </>
        )}

        <button type="button" className="btn" disabled={loading} onClick={() => void handleSubmit()}>
          {submitLabel}
        </button>
      </div>
    </section>
  );
}
