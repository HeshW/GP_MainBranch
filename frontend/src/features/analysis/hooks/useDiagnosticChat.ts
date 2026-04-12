import { useCallback, useMemo, useRef, useState } from "react";
import { postClarification, postImage, postLabs, postSymptoms } from "@/shared/api";
import {
  AnalysisResponse,
  ChatTimelineMessage,
  ClarificationFlowState,
  ClarificationQuestion,
} from "@/shared/types";

interface ActiveDiagnosticContext {
  report: Record<string, unknown>;
  diagnosis?: Record<string, unknown>;
}

type ClarificationPayload = NonNullable<
  NonNullable<AnalysisResponse["diagnosis"]>["clarification"]
>;

export interface UseDiagnosticChatState {
  timeline: ChatTimelineMessage[];
  latestAnalysis: AnalysisResponse | null;
  loading: boolean;
  error: string | null;
  clarificationState: ClarificationFlowState | null;
  activeDiagnosticContext: ActiveDiagnosticContext | null;
  runSymptoms: (symptomText: string, useParser: boolean) => Promise<void>;
  runLabs: (labsJson: string, symptomsExtra: string) => Promise<void>;
  runImage: (file: File | null) => Promise<void>;
  submitClarificationAnswer: (answer: string) => Promise<void>;
  clearError: () => void;
}

function parseLabsJson(labsJson: string): Record<string, unknown> {
  try {
    return JSON.parse(labsJson) as Record<string, unknown>;
  } catch {
    throw new Error("Lab values must be valid JSON.");
  }
}

function formatConfidence(value: unknown): string {
  if (typeof value === "number") {
    return value.toFixed(2);
  }
  if (typeof value === "string") {
    return value;
  }
  return "n/a";
}

function buildDiagnosisSummary(result: AnalysisResponse): string {
  const lines: string[] = [];
  const finalDiagnosis = result.diagnosis?.final_diagnosis;

  if (finalDiagnosis?.diagnosis) {
    lines.push(`Likely diagnosis: ${finalDiagnosis.diagnosis}.`);
    lines.push(
      `Confidence: ${formatConfidence(finalDiagnosis.confidence)} | Source: ${finalDiagnosis.source ?? "unknown"}.`,
    );
    if (finalDiagnosis.reasoning) {
      lines.push(finalDiagnosis.reasoning);
    }
  } else if (result.diagnosis?.summary) {
    lines.push(result.diagnosis.summary);
  } else {
    lines.push("Analysis completed, but no final diagnosis label was returned.");
  }

  if (result.therapy?.therapy_plan) {
    lines.push(`Initial therapy guidance: ${result.therapy.therapy_plan}`);
  }

  if (result.warnings?.length) {
    lines.push(`Warnings: ${result.warnings.join(" | ")}`);
  }

  return lines.join("\n");
}

function isClarificationResolved(clarification: ClarificationPayload | undefined): boolean {
  if (!clarification) {
    return false;
  }

  if (clarification.completed) {
    return true;
  }

  if (clarification.applied || clarification.override_applied) {
    return true;
  }

  return (clarification.answers_used?.length ?? 0) > 0;
}

function extractClarificationFlow(result: AnalysisResponse): ClarificationFlowState | null {
  const report = result.report;
  const clarification = result.diagnosis?.clarification;
  const questions = clarification?.questions ?? [];

  if (
    !report
    || !clarification?.needed
    || questions.length === 0
    || isClarificationResolved(clarification)
  ) {
    return null;
  }

  const diagnosisContext = result.diagnosis
    ? (result.diagnosis as unknown as Record<string, unknown>)
    : undefined;

  return {
    report,
    diagnosis: diagnosisContext,
    questions,
    answers: [],
    nextQuestionIndex: 0,
  };
}

function formatClarificationPrompt(
  question: ClarificationQuestion,
  position: number,
  total: number,
): string {
  const targets = question.target_conditions?.length
    ? `\nTargets: ${question.target_conditions.join(", ")}`
    : "";
  const reason = question.reason ? `\nReason: ${question.reason}` : "";
  return `Follow-up question ${position}/${total}: ${question.question}${targets}${reason}`;
}

export function useDiagnosticChat(): UseDiagnosticChatState {
  const [timeline, setTimeline] = useState<ChatTimelineMessage[]>([
    {
      id: "msg-0",
      role: "assistant",
      kind: "note",
      content:
        "Describe symptoms, submit lab JSON, or upload an image. I will return a diagnosis summary here and ask follow-up questions when clarification is needed.",
    },
  ]);
  const [latestAnalysis, setLatestAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarificationState, setClarificationState] = useState<ClarificationFlowState | null>(
    null,
  );

  const messageCounterRef = useRef(0);

  const appendMessage = useCallback(
    (role: ChatTimelineMessage["role"], content: string, kind: ChatTimelineMessage["kind"]) => {
      messageCounterRef.current += 1;
      const nextMessage: ChatTimelineMessage = {
        id: `msg-${Date.now()}-${messageCounterRef.current}`,
        role,
        content,
        kind,
      };
      setTimeline((current) => [...current, nextMessage]);
    },
    [],
  );

  const addValidationError = useCallback(
    (message: string) => {
      setError(message);
      appendMessage("assistant", message, "error");
    },
    [appendMessage],
  );

  const captureRequestError = useCallback(
    (value: unknown) => {
      const message = value instanceof Error ? value.message : String(value);
      setError(message);
      appendMessage("assistant", `Request failed: ${message}`, "error");
    },
    [appendMessage],
  );

  const handleAnalysisResult = useCallback(
    (payload: AnalysisResponse) => {
      setLatestAnalysis(payload);
      appendMessage("assistant", buildDiagnosisSummary(payload), "diagnosis");

      const nextClarification = extractClarificationFlow(payload);
      if (!nextClarification) {
        setClarificationState(null);
        return;
      }

      setClarificationState(nextClarification);
      appendMessage(
        "assistant",
        formatClarificationPrompt(
          nextClarification.questions[0],
          1,
          nextClarification.questions.length,
        ),
        "clarification",
      );
    },
    [appendMessage],
  );

  const beginRequest = useCallback(() => {
    setLoading(true);
    setError(null);
  }, []);

  const runSymptoms = useCallback(
    async (symptomText: string, useParser: boolean) => {
      const trimmed = symptomText.trim();
      if (!trimmed) {
        addValidationError("Please enter symptom text before sending.");
        return;
      }

      appendMessage("user", trimmed, "note");
      beginRequest();

      try {
        const payload = await postSymptoms({
          text: trimmed,
          use_symptom_parser: useParser,
        });
        handleAnalysisResult(payload);
      } catch (value) {
        captureRequestError(value);
      } finally {
        setLoading(false);
      }
    },
    [addValidationError, appendMessage, beginRequest, captureRequestError, handleAnalysisResult],
  );

  const runLabs = useCallback(
    async (labsJson: string, symptomsExtra: string) => {
      let parsedLabs: Record<string, unknown>;
      try {
        parsedLabs = parseLabsJson(labsJson);
      } catch (value) {
        captureRequestError(value);
        return;
      }

      const symptoms = symptomsExtra.trim();
      appendMessage(
        "user",
        symptoms
          ? `Submitted manual labs with symptom context: ${symptoms}`
          : "Submitted manual labs for analysis.",
        "note",
      );
      beginRequest();

      try {
        const payload = await postLabs({
          labs: parsedLabs,
          symptoms: symptoms || undefined,
        });
        handleAnalysisResult(payload);
      } catch (value) {
        captureRequestError(value);
      } finally {
        setLoading(false);
      }
    },
    [appendMessage, beginRequest, captureRequestError, handleAnalysisResult],
  );

  const runImage = useCallback(
    async (file: File | null) => {
      if (!file) {
        addValidationError("Please choose an image before running analysis.");
        return;
      }

      appendMessage("user", `Uploaded report image: ${file.name}`, "note");
      beginRequest();

      try {
        const payload = await postImage(file);
        handleAnalysisResult(payload);
      } catch (value) {
        captureRequestError(value);
      } finally {
        setLoading(false);
      }
    },
    [addValidationError, appendMessage, beginRequest, captureRequestError, handleAnalysisResult],
  );

  const submitClarificationAnswer = useCallback(
    async (answer: string) => {
      const trimmed = answer.trim();
      if (!trimmed) {
        addValidationError("Please enter an answer before continuing clarification.");
        return;
      }

      const flow = clarificationState;
      if (!flow) {
        addValidationError("No clarification prompt is currently active.");
        return;
      }

      appendMessage("user", trimmed, "note");

      const answers = [...flow.answers, trimmed];
      if (answers.length < flow.questions.length) {
        const nextIndex = answers.length;
        setClarificationState({
          ...flow,
          answers,
          nextQuestionIndex: nextIndex,
        });
        appendMessage(
          "assistant",
          formatClarificationPrompt(
            flow.questions[nextIndex],
            nextIndex + 1,
            flow.questions.length,
          ),
          "clarification",
        );
        return;
      }

      setClarificationState({
        ...flow,
        answers,
        nextQuestionIndex: flow.questions.length,
      });
      appendMessage(
        "assistant",
        "Thanks. Re-running diagnosis with your clarification answers.",
        "note",
      );
      beginRequest();

      try {
        const payload = await postClarification({
          report: flow.report,
          diagnosis: flow.diagnosis,
          answers,
        });
        handleAnalysisResult(payload);
      } catch (value) {
        captureRequestError(value);
      } finally {
        setLoading(false);
      }
    },
    [
      addValidationError,
      appendMessage,
      beginRequest,
      captureRequestError,
      clarificationState,
      handleAnalysisResult,
    ],
  );

  const activeDiagnosticContext = useMemo<ActiveDiagnosticContext | null>(() => {
    if (clarificationState) {
      return {
        report: clarificationState.report,
        diagnosis: clarificationState.diagnosis,
      };
    }

    if (latestAnalysis?.report) {
      return {
        report: latestAnalysis.report,
        diagnosis: latestAnalysis.diagnosis
          ? (latestAnalysis.diagnosis as unknown as Record<string, unknown>)
          : undefined,
      };
    }

    return null;
  }, [clarificationState, latestAnalysis]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    timeline,
    latestAnalysis,
    loading,
    error,
    clarificationState,
    activeDiagnosticContext,
    runSymptoms,
    runLabs,
    runImage,
    submitClarificationAnswer,
    clearError,
  };
}
