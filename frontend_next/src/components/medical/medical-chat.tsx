"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  postChat,
  postChatStream,
  postClarification,
  postImage,
  postLabs,
  postSymptoms,
} from "@/lib/api";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";
import type { AnalysisResponse, ChatMessage } from "@/lib/medical-types";

type SendMode = "auto" | "symptoms" | "labs" | "image" | "chat";

type ClarificationContext = {
  report: Record<string, unknown>;
  diagnosis: Record<string, unknown> | undefined;
  questions: string[];
};

type MedicalChatProps = {
  compact?: boolean;
  initialPrompt?: string;
  userName?: string;
};

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

function parseLabsJson(labsJson: string): Record<string, unknown> {
  try {
    return JSON.parse(labsJson) as Record<string, unknown>;
  } catch {
    throw new Error("Invalid JSON in lab values.");
  }
}

function symptomHeuristic(text: string): boolean {
  return /pain|fever|cough|fatigue|dizziness|thirst|nausea|vomit|headache|chest|breath|rash|sore|حمى|ألم|وجع|كحة|صداع|دوخة|غثيان/i.test(
    text,
  );
}

function summarizeAnalysis(result: AnalysisResponse): string {
  const diagnosis = result.diagnosis?.final_diagnosis?.diagnosis ?? "No final diagnosis";
  const confidence = result.diagnosis?.final_diagnosis?.confidence;
  const confidenceText = confidence === undefined ? "n/a" : String(confidence);
  return `Analysis completed. Likely condition: ${diagnosis}. Confidence: ${confidenceText}.`;
}

function normalizeForCompare(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

function getResponseText(analysis: AnalysisResponse): string | undefined {
  return (
    analysis.diagnosis?.ai_response?.trim() ||
    analysis.diagnosis?.gemini_response?.trim() ||
    analysis.diagnosis?.summary?.trim()
  );
}

function AnalysisCard({ analysis }: { analysis: AnalysisResponse }) {
  const diagnosis = analysis.diagnosis?.final_diagnosis;
  const responseText = getResponseText(analysis);
  const therapy = analysis.therapy?.therapy_plan;
  const safetyReasons = analysis.diagnosis?.safety?.reasons ?? [];
  const clarification = analysis.diagnosis?.clarification;
  const normalizedResponse = normalizeForCompare(responseText);
  const visibleSafetyReasons = safetyReasons.filter((item) => {
    const normalizedItem = normalizeForCompare(item);
    return normalizedItem && !normalizedResponse.includes(normalizedItem);
  });

  return (
    <div className="rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-4 text-[var(--brand-text)] shadow-[var(--brand-shadow)]">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--brand-border)] pb-3">
        <div>
          <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">Nabda analysis</p>
          <h3 className="mt-1 text-lg font-semibold text-[var(--brand-heading)]">
            {diagnosis?.diagnosis ?? "No final diagnosis"}
          </h3>
        </div>
        {diagnosis?.confidence !== undefined ? (
          <span className="rounded-2xl bg-[var(--brand-soft)] px-3 py-1 text-xs font-semibold text-[var(--brand-primary)]">
            Confidence {String(diagnosis.confidence)}
          </span>
        ) : null}
      </div>

      {responseText ? <p className="mt-4 whitespace-pre-wrap text-sm leading-6">{responseText}</p> : null}
      {therapy ? (
        <p className="mt-3 rounded-2xl bg-[var(--brand-soft)] px-3 py-2 text-sm leading-6 text-[var(--brand-text)]">
          Therapy: {therapy}
        </p>
      ) : null}

      {visibleSafetyReasons.length ? (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-3">
          <p className="text-xs font-semibold uppercase text-amber-900">Safety notes</p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-amber-950">
            {visibleSafetyReasons.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {clarification?.needed ? (
        <div className="mt-4 rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] p-3">
          <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">Follow-up questions</p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-[var(--brand-text)]">
            {(clarification.questions ?? []).map((item) => (
              <li key={item.question}>{item.question}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export function MedicalChat({ compact = false, initialPrompt, userName = "there" }: MedicalChatProps) {
  const { language } = usePreferences();
  const t = getCopy(language);
  const [sessionId] = useState(createSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [composerText, setComposerText] = useState("");
  const [sendMode, setSendMode] = useState<SendMode>("auto");
  const [actionPanelOpen, setActionPanelOpen] = useState(false);
  const [useParser, setUseParser] = useState(true);
  const [labsJson, setLabsJson] = useState(DEFAULT_LABS_JSON);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [clarificationContext, setClarificationContext] = useState<ClarificationContext | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const hasAnalysis = useMemo(() => messages.some((item) => item.kind === "analysis"), [messages]);
  const hasStarted = messages.length > 0 || loading;

  const sendDisabled = useMemo(() => {
    if (loading) return true;
    if (sendMode === "image") return !imageFile;
    if (sendMode === "labs") return !labsJson.trim();
    return !composerText.trim();
  }, [composerText, imageFile, labsJson, loading, sendMode]);

  const addMessage = (role: ChatMessage["role"], content: string, kind: ChatMessage["kind"] = "text") => {
    setMessages((current) => [
      ...current,
      {
        id: createMessageId(role),
        role,
        kind,
        content,
      },
    ]);
  };

  const addAnalysis = (analysis: AnalysisResponse) => {
    setMessages((current) => [
      ...current,
      {
        id: createMessageId("assistant"),
        role: "assistant",
        kind: "analysis",
        content: summarizeAnalysis(analysis),
        payload: analysis,
      },
    ]);

    const clarification = analysis.diagnosis?.clarification;
    if (clarification?.needed && analysis.report) {
      setClarificationContext({
        report: analysis.report,
        diagnosis: analysis.diagnosis as Record<string, unknown> | undefined,
        questions: (clarification.questions ?? [])
          .map((item) => item.question)
          .filter((item) => item.trim().length > 0),
      });
    } else {
      setClarificationContext(null);
    }
  };

  const setLastAssistantText = (content: string) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant" && next[index].kind === "text") {
          next[index] = { ...next[index], content };
          return next;
        }
      }
      return [...next, { id: createMessageId("assistant"), role: "assistant", kind: "text", content }];
    });
  };

  const appendChunkToLastAssistant = (chunk: string) => {
    setMessages((current) => {
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
        { id: createMessageId("assistant"), role: "assistant", kind: "text", content: chunk },
      ];
    });
  };

  const routeAutoAction = (text: string): SendMode => {
    if (clarificationContext) return "symptoms";
    if (imageFile) return "image";
    if (sendMode === "labs") return "labs";
    if (!hasAnalysis || symptomHeuristic(text)) return "symptoms";
    return "chat";
  };

  const chooseMode = (mode: SendMode) => {
    setSendMode(mode);
    setActionPanelOpen(false);
  };

  const runChatTurn = async (text: string) => {
    addMessage("assistant", "");
    let hasStreamedChunk = false;

    try {
      const streamedText = await postChatStream({ session_id: sessionId, message: text }, (chunk) => {
        hasStreamedChunk = true;
        appendChunkToLastAssistant(chunk);
      });

      if (!hasStreamedChunk && !streamedText) {
        const fallback = await postChat({ session_id: sessionId, message: text });
        setLastAssistantText(fallback.response);
      }
    } catch {
      if (!hasStreamedChunk) {
        try {
          const fallback = await postChat({ session_id: sessionId, message: text });
          setLastAssistantText(fallback.response);
        } catch (error) {
          const content = error instanceof Error ? error.message : String(error);
          setLastAssistantText(`Chat failed: ${content}`);
        }
      }
    }
  };

  const handleSend = async () => {
    const trimmed = composerText.trim();
    const action = sendMode === "auto" ? routeAutoAction(trimmed) : sendMode;

    if (action !== "image" && action !== "labs" && !trimmed) return;

    let userSummary = trimmed;
    if (!userSummary && action === "image" && imageFile) userSummary = `Uploaded image: ${imageFile.name}`;
    if (!userSummary && action === "labs") userSummary = "Submitted lab values.";

    addMessage("user", userSummary || "Submitted");
    setComposerText("");
    setLoading(true);

    try {
      if (clarificationContext && trimmed && action !== "chat") {
        const answers = trimmed
          .split("\n")
          .map((item) => item.trim())
          .filter(Boolean);
        const analysis = await postClarification({
          report: clarificationContext.report,
          diagnosis: clarificationContext.diagnosis,
          answers: answers.length ? answers : [trimmed],
        });
        addAnalysis(analysis);
      } else if (action === "image" && imageFile) {
        const analysis = await postImage(imageFile);
        addAnalysis(analysis);
        setImageFile(null);
      } else if (action === "labs") {
        const analysis = await postLabs({
          labs: parseLabsJson(labsJson),
          symptoms: trimmed || undefined,
        });
        addAnalysis(analysis);
      } else if (action === "symptoms") {
        const analysis = await postSymptoms({
          text: trimmed,
          use_symptom_parser: useParser,
        });
        addAnalysis(analysis);
      } else {
        await runChatTurn(trimmed);
      }
    } catch (error) {
      const content = error instanceof Error ? error.message : String(error);
      addMessage("assistant", content, "error");
    } finally {
      setLoading(false);
      setSendMode("auto");
      setActionPanelOpen(false);
    }
  };

  const modeLabel = {
    auto: "Instant",
    symptoms: t.chat.symptoms,
    labs: t.chat.labs,
    image: t.chat.image,
    chat: t.chat.chat,
  }[sendMode];

  const composer = (
    <div className="relative mx-auto w-full max-w-3xl">
      {actionPanelOpen ? (
        <div className="absolute bottom-full left-0 z-10 mb-3 w-full rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-3 shadow-[var(--brand-shadow)]">
          <div className="grid gap-2 sm:grid-cols-4">
            {[
              ["symptoms", t.chat.symptoms],
              ["chat", t.chat.chat],
              ["labs", t.chat.labs],
              ["image", t.chat.image],
            ].map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-3 py-2 text-sm font-semibold text-[var(--brand-primary)] transition hover:bg-[var(--brand-surface)] hover:shadow-sm"
                onClick={() => chooseMode(mode as SendMode)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="mt-3 flex items-center gap-2 text-sm text-[var(--brand-muted)]">
            <input
              type="checkbox"
              checked={useParser}
              onChange={(event) => setUseParser(event.target.checked)}
            />
            {t.chat.parser}
          </label>
        </div>
      ) : null}

      {sendMode === "labs" ? (
        <label className="mb-3 block text-sm font-medium text-[var(--brand-text)]">
          {t.chat.labLabel}
          <textarea
            className="mt-2 min-h-28 w-full resize-y rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-3 font-mono text-sm text-[var(--brand-text)] shadow-sm outline-none transition focus:border-[var(--brand-primary)] focus:ring-4 focus:ring-blue-500/10"
            value={labsJson}
            onChange={(event) => setLabsJson(event.target.value)}
            spellCheck={false}
          />
        </label>
      ) : null}

      {sendMode === "image" ? (
        <label className="mb-3 flex cursor-pointer items-center justify-center rounded-3xl border border-dashed border-[var(--brand-border-strong)] bg-[var(--brand-soft)] px-4 py-5 text-center text-sm font-medium text-[var(--brand-text)] transition hover:border-[var(--brand-primary)]">
          <input
            className="sr-only"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
          />
          {imageFile ? `Attached: ${imageFile.name}` : t.chat.imageLabel}
        </label>
      ) : null}

      <div className="flex items-center gap-2 rounded-[2rem] border border-[var(--brand-border)] bg-[var(--brand-surface-glass)] px-3 py-2 shadow-[0_18px_60px_rgba(15,73,128,0.14)] backdrop-blur-xl">
        <button
          type="button"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-2xl leading-none text-[var(--brand-primary)] transition hover:bg-[var(--brand-soft)]"
          onClick={() => setActionPanelOpen((current) => !current)}
          aria-label={t.chat.attach}
        >
          +
        </button>
        <textarea
          className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm text-[var(--brand-text)] outline-none placeholder:text-slate-400"
          value={composerText}
          onChange={(event) => setComposerText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void handleSend();
            }
          }}
          placeholder={clarificationContext ? "Answer the follow-up questions here" : t.chat.placeholder}
          disabled={loading}
        />
        <span className="hidden rounded-2xl bg-[var(--brand-soft)] px-3 py-2 text-xs font-semibold text-[var(--brand-primary)] sm:inline-flex">
          {modeLabel}
        </span>
        <button
          type="button"
          className="flex h-10 min-w-10 items-center justify-center rounded-2xl bg-[var(--brand-primary)] px-4 text-sm font-semibold text-white transition hover:bg-[var(--brand-primary-strong)] disabled:cursor-not-allowed disabled:opacity-60"
          disabled={sendDisabled}
          onClick={() => void handleSend()}
        >
          {loading ? t.chat.sending : t.chat.send}
        </button>
      </div>
    </div>
  );

  return (
    <section
      className={[
        "flex h-full min-h-0 flex-col overflow-hidden bg-[var(--brand-bg)]",
        compact ? "max-h-[78vh] rounded-3xl border border-[var(--brand-border)]" : "min-h-[calc(100svh-64px)]",
      ].join(" ")}
    >
      {!hasStarted ? (
        <div className="flex flex-1 flex-col items-center justify-end px-4 pb-10 pt-16 text-center sm:pb-14">
          <div className="mb-8 max-w-3xl">
            <p className="text-3xl font-semibold text-[var(--brand-heading)] sm:text-5xl">
              {initialPrompt ?? `Hey, ${userName} 👋`}
            </p>
            <p className="mt-4 text-2xl font-semibold text-[var(--brand-heading)] sm:text-4xl">
              How can Nabda help you today?
            </p>
          </div>
          {composer}
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {[t.chat.symptoms, t.chat.image, t.chat.labs].map((label) => (
              <button
                key={label}
                type="button"
                className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-2 text-sm font-medium text-[var(--brand-text)] shadow-sm transition hover:bg-[var(--brand-soft)]"
                onClick={() => {
                  const nextMode =
                    label === t.chat.image ? "image" : label === t.chat.labs ? "labs" : "symptoms";
                  chooseMode(nextMode);
                }}
              >
                {label}
              </button>
            ))}
            <Link
              href="/doctors"
              className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-2 text-sm font-medium text-[var(--brand-text)] shadow-sm transition hover:bg-[var(--brand-soft)]"
            >
              Find a doctor
            </Link>
          </div>
          <p className="mt-5 text-xs leading-5 text-[var(--brand-muted)]">{t.chat.disclaimer}</p>
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto px-4 py-6">
            <div className="mx-auto flex max-w-4xl flex-col gap-4">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={[
                    "max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm",
                    message.role === "user"
                      ? "ml-auto bg-[var(--brand-primary)] text-white"
                    : message.kind === "error"
                        ? "border border-rose-200 bg-rose-50 text-rose-900"
                        : "mr-auto bg-[var(--brand-surface)] text-[var(--brand-text)]",
                    message.kind === "analysis" ? "w-full max-w-full bg-transparent p-0 shadow-none" : "",
                  ].join(" ")}
                >
                  {message.kind === "analysis" && message.payload ? (
                    <AnalysisCard analysis={message.payload} />
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )}
                </article>
              ))}
              {loading ? (
                <div className="mr-auto rounded-2xl bg-[var(--brand-surface)] px-4 py-2 text-sm text-[var(--brand-muted)] shadow-sm">
                  {t.chat.working}
                </div>
              ) : null}
              <div ref={endRef} />
            </div>
          </div>

          {clarificationContext?.questions.length ? (
            <div className="border-t border-[var(--brand-border)] bg-[var(--brand-soft)] px-5 py-3 text-sm text-[var(--brand-text)]">
              <p className="font-semibold">Follow-up</p>
              <ul className="mt-2 space-y-1">
                {clarificationContext.questions.map((question) => (
                  <li key={question}>{question}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="border-t border-[var(--brand-border)] bg-[var(--brand-surface-glass)] p-4 backdrop-blur">
            {composer}
            <p className="mx-auto mt-3 max-w-3xl text-xs leading-5 text-[var(--brand-muted)]">
              {t.chat.disclaimer}
            </p>
          </div>
        </>
      )}
    </section>
  );
}
