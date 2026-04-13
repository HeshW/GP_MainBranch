import { useEffect, useMemo, useRef, useState } from "react";
import { postChat, postChatStream } from "@/shared/api";
import { AnalysisResponse } from "@/shared/types";

type ComposerAction = "auto" | "symptoms" | "labs" | "image" | "chat";

type TimelineItem = {
  id: string;
  role: "user" | "assistant";
  kind: "text" | "analysis" | "error";
  content?: string;
  payload?: AnalysisResponse;
};

type ClarificationContext = {
  report: Record<string, unknown>;
  diagnosis: Record<string, unknown> | undefined;
  questions: string[];
};

interface UserInterfaceViewProps {
  loading: boolean;
  result: AnalysisResponse | null;
  error: string | null;
  runLabs: (labsJson: string, symptomsExtra: string) => Promise<void>;
  runImage: (file: File | null) => Promise<void>;
  runSymptoms: (symptomText: string, useParser: boolean) => Promise<void>;
  runClarification: (
    report: Record<string, unknown>,
    diagnosis: Record<string, unknown> | undefined,
    answers: string[],
  ) => Promise<void>;
}

const DEFAULT_LABS_JSON = `{
  "glucose": 145,
  "hemoglobin": 11.2
}`;

function createSessionId() {
  return `session-${Math.random().toString(36).slice(2, 10)}`;
}

function createMessageId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 12)}`;
}

function summarizeAnalysis(result: AnalysisResponse): string {
  const diagnosis = result.diagnosis?.final_diagnosis?.diagnosis ?? "No final diagnosis";
  const confidence = result.diagnosis?.final_diagnosis?.confidence;
  const confidenceText = confidence === undefined ? "n/a" : String(confidence);
  return `Latest analysis completed. Likely condition: ${diagnosis}. Confidence: ${confidenceText}.`;
}

function symptomHeuristic(text: string): boolean {
  const lowered = text.toLowerCase();
  return /pain|fever|cough|fatigue|dizziness|thirst|nausea|vomit|headache|chest|breath|حمى|الم|وجع|كحة|ضيق|دوخة|غثيان/.test(lowered);
}

function normalizeForCompare(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

export function UserInterfaceView({
  loading,
  result,
  error,
  runLabs,
  runImage,
  runSymptoms,
  runClarification,
}: UserInterfaceViewProps) {
  const [sessionId] = useState(createSessionId);
  const [timeline, setTimeline] = useState<TimelineItem[]>([
    {
      id: createMessageId("assistant"),
      role: "assistant",
      kind: "text",
      content:
        "Write your symptoms to start. You can also attach labs JSON or an image before sending, then continue asking follow-up questions in the same chat.",
    },
  ]);
  const [composerText, setComposerText] = useState("");
  const [composerAction, setComposerAction] = useState<ComposerAction>("auto");
  const [useParser, setUseParser] = useState(true);
  const [labsJson, setLabsJson] = useState(DEFAULT_LABS_JSON);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [streamingReply, setStreamingReply] = useState(false);
  const [clarificationContext, setClarificationContext] = useState<ClarificationContext | null>(null);
  const timelineEndRef = useRef<HTMLDivElement | null>(null);
  const lastResultRef = useRef<AnalysisResponse | null>(null);
  const lastErrorRef = useRef<string | null>(null);

  const finalDiagnosis = result?.diagnosis?.final_diagnosis;

  useEffect(() => {
    if (timelineEndRef.current && typeof timelineEndRef.current.scrollIntoView === "function") {
      timelineEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [timeline]);

  useEffect(() => {
    if (!result || result === lastResultRef.current) {
      return;
    }

    lastResultRef.current = result;
    setTimeline((current) => [
      ...current,
      {
        id: createMessageId("assistant"),
        role: "assistant",
        kind: "analysis",
        payload: result,
        content: summarizeAnalysis(result),
      },
    ]);

    const clarification = result.diagnosis?.clarification;
    if (clarification?.needed && result.report) {
      setClarificationContext({
        report: result.report as Record<string, unknown>,
        diagnosis: result.diagnosis as Record<string, unknown> | undefined,
        questions: (clarification.questions ?? [])
          .map((item) => item.question)
          .filter((item) => item.trim().length > 0),
      });
    } else {
      setClarificationContext(null);
    }
  }, [result]);

  useEffect(() => {
    if (!error || error === lastErrorRef.current) {
      return;
    }

    lastErrorRef.current = error;
    setTimeline((current) => [
      ...current,
      {
        id: createMessageId("assistant"),
        role: "assistant",
        kind: "error",
        content: error,
      },
    ]);
  }, [error]);

  const sendDisabled = useMemo(() => {
    if (loading || streamingReply) {
      return true;
    }
    if (composerAction === "image") {
      return !imageFile;
    }
    if (composerAction === "labs") {
      return !labsJson.trim();
    }
    return !composerText.trim();
  }, [composerAction, composerText, imageFile, labsJson, loading, streamingReply]);

  const sendLabel = loading || streamingReply ? "Sending..." : "Send";

  const addTimelineText = (role: "user" | "assistant", content: string, kind: TimelineItem["kind"] = "text") => {
    setTimeline((current) => [
      ...current,
      {
        id: createMessageId(role),
        role,
        kind,
        content,
      },
    ]);
  };

  const appendChunkToLastAssistant = (chunk: string) => {
    setTimeline((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant" && next[index].kind === "text") {
          next[index] = {
            ...next[index],
            content: `${next[index].content ?? ""}${chunk}`,
          };
          return next;
        }
      }
      return [
        ...next,
        {
          id: createMessageId("assistant"),
          role: "assistant",
          kind: "text",
          content: chunk,
        },
      ];
    });
  };

  const runChatTurn = async (text: string) => {
    setStreamingReply(true);
    addTimelineText("assistant", "");
    let streamed = false;

    try {
      const fallbackText = await postChatStream(
        { session_id: sessionId, message: text },
        (chunk) => {
          streamed = true;
          appendChunkToLastAssistant(chunk);
        },
      );

      if (!streamed && !fallbackText) {
        const fallback = await postChat({ session_id: sessionId, message: text });
        setTimeline((current) => {
          const next = [...current];
          for (let index = next.length - 1; index >= 0; index -= 1) {
            if (next[index].role === "assistant" && next[index].kind === "text") {
              next[index] = { ...next[index], content: fallback.response };
              return next;
            }
          }
          return current;
        });
      }
    } catch {
      if (!streamed) {
        try {
          const fallback = await postChat({ session_id: sessionId, message: text });
          setTimeline((current) => {
            const next = [...current];
            for (let index = next.length - 1; index >= 0; index -= 1) {
              if (next[index].role === "assistant" && next[index].kind === "text") {
                next[index] = { ...next[index], content: fallback.response };
                return next;
              }
            }
            return current;
          });
        } catch (chatError) {
          const content = chatError instanceof Error ? chatError.message : String(chatError);
          setTimeline((current) => [
            ...current,
            {
              id: createMessageId("assistant"),
              role: "assistant",
              kind: "error",
              content: `Chat failed: ${content}`,
            },
          ]);
        }
      }
    } finally {
      setStreamingReply(false);
    }
  };

  const routeAutoAction = (text: string): ComposerAction => {
    if (clarificationContext) {
      return "symptoms";
    }
    if (imageFile) {
      return "image";
    }
    if (composerAction === "labs") {
      return "labs";
    }
    const hasPriorAnalysis = timeline.some((item) => item.kind === "analysis");
    if (!hasPriorAnalysis || symptomHeuristic(text)) {
      return "symptoms";
    }
    return "chat";
  };

  const handleSend = async () => {
    const trimmed = composerText.trim();
    const action = composerAction === "auto" ? routeAutoAction(trimmed) : composerAction;

    if (action !== "image" && action !== "labs" && !trimmed) {
      return;
    }

    let userSummary = trimmed;
    if (!userSummary && action === "image" && imageFile) {
      userSummary = `Uploaded image: ${imageFile.name}`;
    }
    if (!userSummary && action === "labs") {
      userSummary = "Submitted lab values.";
    }

    addTimelineText("user", userSummary || "Submitted");
    setComposerText("");

    if (clarificationContext && trimmed && action !== "chat") {
      const answers = trimmed
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);
      await runClarification(
        clarificationContext.report,
        clarificationContext.diagnosis,
        answers.length ? answers : [trimmed],
      );
      setComposerAction("auto");
      return;
    }

    if (action === "image") {
      await runImage(imageFile);
      setImageFile(null);
      setComposerAction("auto");
      return;
    }

    if (action === "labs") {
      await runLabs(labsJson, trimmed);
      setComposerAction("auto");
      return;
    }

    if (action === "symptoms") {
      await runSymptoms(trimmed, useParser);
      setComposerAction("auto");
      return;
    }

    await runChatTurn(trimmed);
    setComposerAction("auto");
  };

  const renderAnalysisCard = (analysis: AnalysisResponse) => {
    const diagnosis = analysis.diagnosis?.final_diagnosis;
    const summary = analysis.diagnosis?.summary;
    const aiResponse = analysis.diagnosis?.ai_response ?? analysis.diagnosis?.gemini_response;
    const therapy = analysis.therapy?.therapy_plan;
    const safetyReasons = analysis.diagnosis?.safety?.reasons ?? [];
    const clarification = analysis.diagnosis?.clarification;
    const summaryText = summary?.trim();
    const aiResponseText = aiResponse?.trim();
    const normalizedAi = normalizeForCompare(aiResponseText);
    const showSummary = Boolean(summaryText) && !aiResponseText;
    const visibleSafetyReasons = safetyReasons.filter((item) => {
      const normalizedItem = normalizeForCompare(item);
      if (!normalizedItem) {
        return false;
      }
      return !normalizedAi.includes(normalizedItem);
    });

    return (
      <article className="chat-first-card">
        <header className="chat-first-card__header">
          <p className="chat-first-card__title">Diagnostic update</p>
          <p className="chat-first-card__meta">
            {diagnosis?.diagnosis ?? "No final diagnosis"}
            {diagnosis?.confidence !== undefined ? ` • confidence ${String(diagnosis.confidence)}` : ""}
          </p>
        </header>

        {showSummary && <p className="chat-first-card__body">{summaryText}</p>}
        {aiResponseText && <p className="chat-first-card__body">{aiResponseText}</p>}
        {therapy && <p className="chat-first-card__body">Therapy: {therapy}</p>}

        {!!visibleSafetyReasons.length && (
          <ul className="flat-list">
            {visibleSafetyReasons.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}

        {clarification?.needed && (
          <div className="chat-first-card__clarify">
            <p>Follow-up questions:</p>
            <ul className="flat-list">
              {(clarification.questions ?? []).map((item) => (
                <li key={item.question}>{item.question}</li>
              ))}
            </ul>
            <p className="chat-first-card__hint">Reply in the same composer to continue clarification.</p>
          </div>
        )}
      </article>
    );
  };

  return (
    <section className="chat-first" dir="rtl">
      <header className="chat-first__hero">
        <div>
          <p className="chat-first__eyebrow">User Interface</p>
          <h2 className="chat-first__title">One conversation for diagnosis, OCR, and follow-up</h2>
          <p className="chat-first__subtitle">
            Type symptoms and send immediately, or attach labs/image first. Follow-up questions are
            answered in the same chat box.
          </p>
        </div>
        <div className="chat-first__status">
          <span>{finalDiagnosis?.diagnosis ?? "No diagnosis yet"}</span>
        </div>
      </header>

      <section className="chat-first__history" aria-live="polite">
        {timeline.map((item) => (
          <article
            key={item.id}
            className={`chat-first-message chat-first-message--${item.role} chat-first-message--${item.kind}`}
          >
            {item.kind === "analysis" && item.payload ? (
              renderAnalysisCard(item.payload)
            ) : (
              <p>{item.content}</p>
            )}
          </article>
        ))}
        <div ref={timelineEndRef} />
      </section>

      {clarificationContext?.questions.length ? (
        <section className="chat-first__clarification">
          <p>Current follow-up questions:</p>
          <ul className="flat-list">
            {clarificationContext.questions.map((question) => (
              <li key={question}>{question}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="chat-first__composer">
        <div className="chat-first-actions" role="tablist" aria-label="Send mode">
          <button
            type="button"
            role="tab"
            aria-selected={composerAction === "auto"}
            className={composerAction === "auto" ? "is-active" : ""}
            onClick={() => setComposerAction("auto")}
          >
            Auto
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={composerAction === "symptoms"}
            className={composerAction === "symptoms" ? "is-active" : ""}
            onClick={() => setComposerAction("symptoms")}
          >
            Symptoms
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={composerAction === "labs"}
            className={composerAction === "labs" ? "is-active" : ""}
            onClick={() => setComposerAction("labs")}
          >
            Labs JSON
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={composerAction === "image"}
            className={composerAction === "image" ? "is-active" : ""}
            onClick={() => setComposerAction("image")}
          >
            Upload image
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={composerAction === "chat"}
            className={composerAction === "chat" ? "is-active" : ""}
            onClick={() => setComposerAction("chat")}
          >
            Ask question
          </button>
        </div>

        {(composerAction === "symptoms" || composerAction === "auto") && (
          <label className="chat-first__check">
            <input
              type="checkbox"
              checked={useParser}
              onChange={(event) => setUseParser(event.target.checked)}
            />
            Use advanced symptom parser
          </label>
        )}

        {composerAction === "labs" && (
          <div className="chat-first-attachment">
            <label htmlFor="chat-first-labs">Lab JSON attachment</label>
            <textarea
              id="chat-first-labs"
              value={labsJson}
              onChange={(event) => setLabsJson(event.target.value)}
              spellCheck={false}
            />
          </div>
        )}

        {composerAction === "image" && (
          <label className="dropzone chat-first-dropzone">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
            />
            {imageFile ? `Attached: ${imageFile.name}` : "Click to attach an image for OCR"}
          </label>
        )}

        <div className="chat-first__input-row">
          <input
            type="text"
            value={composerText}
            onChange={(event) => setComposerText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void handleSend();
              }
            }}
            placeholder={
              clarificationContext
                ? "Answer the follow-up questions here"
                : "Type symptoms or ask a follow-up question"
            }
            disabled={loading || streamingReply}
          />
          <button
            type="button"
            className="btn"
            disabled={sendDisabled}
            onClick={() => void handleSend()}
          >
            {sendLabel}
          </button>
        </div>
      </section>
    </section>
  );
}
