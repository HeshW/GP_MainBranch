import { useState } from "react";
import { ChatInterface } from "@/features/chat";
import { AnalysisResponse } from "@/shared/types";

interface ResultViewProps {
  error: string | null;
  result: AnalysisResponse | null;
  onClarify: (
    report: Record<string, unknown>,
    diagnosis: Record<string, unknown> | undefined,
    answers: string[],
  ) => Promise<void>;
}

export function ResultView({ error, result, onClarify }: ResultViewProps) {
  const [clarificationDraft, setClarificationDraft] = useState("");
  if (!error && result === null) {
    return null;
  }

  const finalDiagnosis = result?.diagnosis?.final_diagnosis;
  const decisionFusion = result?.diagnosis?.decision_fusion;
  const classifierPrediction = result?.diagnosis?.classifier_prediction;
  const retrievedCases = result?.diagnosis?.retrieved_cases ?? [];
  const clarification = result?.diagnosis?.clarification;
  const assessmentState = result?.diagnosis?.assessment_state ?? (clarification?.needed ? "needs_clarification" : "final");
  const diagnosticCandidates = result?.diagnosis?.diagnostic_candidates ?? [];
  const aiResponse = result?.diagnosis?.ai_response ?? result?.diagnosis?.gemini_response;
  const aiMeta = result?.diagnosis?.ai_response_metadata ?? result?.diagnosis?.gemini_response_metadata;
  const therapy = result?.therapy;
  const clarificationQuestions = clarification?.questions ?? [];
  const showFinalDiagnosis = Boolean(finalDiagnosis && assessmentState !== "needs_clarification");
  const showAssessmentPending = Boolean(finalDiagnosis && assessmentState === "needs_clarification");
  const clarificationReady = Boolean(
    result?.report &&
      clarification?.needed &&
      clarificationDraft
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean).length > 0,
  );

  return (
    <section className="panel">
      <h2>Pipeline Result</h2>
      {error && <p className="err">{error}</p>}

      {result !== null && !error && (
        <div className="result-stack">
          {showFinalDiagnosis && (
            <section className="result-card">
              <h3>Final Diagnosis</h3>
              <p className="result-card__headline">
                {finalDiagnosis.diagnosis ?? "Unknown diagnosis"}
              </p>
              <p className="result-card__meta">
                Source: {finalDiagnosis.source ?? "unknown"} | Confidence: {finalDiagnosis.confidence ?? "n/a"}
              </p>
              {finalDiagnosis.reasoning && <p>{finalDiagnosis.reasoning}</p>}
              {!!finalDiagnosis.supporting_evidence?.length && (
                <ul className="flat-list">
                  {finalDiagnosis.supporting_evidence.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {showAssessmentPending && (
            <section className="result-card">
              <h3>Assessment Pending</h3>
              <p className="result-card__headline">
                Clarification is needed before a final diagnosis is shown.
              </p>
              <p className="result-card__meta">
                Interim leading label: {finalDiagnosis?.diagnosis ?? "Unknown assessment"}
                {finalDiagnosis?.confidence !== undefined ? ` • confidence ${String(finalDiagnosis.confidence)}` : ""}
              </p>
            </section>
          )}

          {(result.parsed?.symptoms?.length || result.validated?.symptoms?.length) && (
            <section className="result-card">
              <h3>Preprocessing</h3>
              {!!result.parsed?.symptoms?.length && (
                <p>
                  Parsed symptoms:{" "}
                  {result.parsed.symptoms.map((item) => item.symptom).join(", ")}
                </p>
              )}
              {!!result.validated?.symptoms?.length && (
                <p>
                  Validated symptoms: {result.validated.symptoms.join(", ")}
                </p>
              )}
            </section>
          )}

          {(decisionFusion || classifierPrediction) && (
            <section className="result-card">
              <h3>Decision Fusion</h3>
              {decisionFusion && (
                <>
                  <p>Primary source: {decisionFusion.primary_source ?? "unknown"}</p>
                  {!!decisionFusion.supporting_sources?.length && (
                    <p>
                      Supporting sources: {decisionFusion.supporting_sources.join(", ")}
                    </p>
                  )}
                  <p>
                    Rule validation: {decisionFusion.rule_validation_status ?? "n/a"}
                  </p>
                </>
              )}
              {classifierPrediction && (
                <p>
                  Classifier top label: {classifierPrediction.predicted_label ?? "n/a"} (
                  {classifierPrediction.confidence ?? "n/a"})
                </p>
              )}
            </section>
          )}

          {(clarification?.needed || diagnosticCandidates.length > 0) && (
            <section className="result-card">
              <h3>Clarification Mode</h3>
              {!!diagnosticCandidates.length && (
                <p>
                  Leading candidates:{" "}
                  {diagnosticCandidates
                    .slice(0, 3)
                    .map((item) => `${item.label} (${item.confidence ?? "n/a"})`)
                    .join(", ")}
                </p>
              )}
              {!!clarification?.reasons?.length && (
                <ul className="flat-list">
                  {clarification.reasons.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              )}
              {!!clarification?.questions?.length && (
                <div className="retrieval-list">
                  {clarification.questions.map((item) => (
                    <article key={item.question} className="retrieval-item">
                      <strong>{item.question}</strong>
                      {!!item.target_conditions?.length && (
                        <p className="result-card__meta">
                          Targets: {item.target_conditions.join(", ")}
                        </p>
                      )}
                      {item.reason && <p>{item.reason}</p>}
                    </article>
                  ))}
                </div>
              )}
              {clarification?.needed && result?.report && (
                <div className="retrieval-item">
                  <label htmlFor="clarification-answers">
                    Your follow-up answers
                  </label>
                  <textarea
                    id="clarification-answers"
                    rows={Math.max(4, clarificationQuestions.length + 1)}
                    value={clarificationDraft}
                    onChange={(event) => setClarificationDraft(event.target.value)}
                    placeholder="Write one answer per line, matching the follow-up questions above."
                  />
                  <button
                    type="button"
                    disabled={!clarificationReady}
                    onClick={() =>
                      onClarify(
                        result.report ?? {},
                        (result.diagnosis as Record<string, unknown> | undefined),
                        clarificationDraft
                          .split("\n")
                          .map((item) => item.trim())
                          .filter(Boolean),
                      )
                    }
                  >
                    Re-run with Clarification
                  </button>
                </div>
              )}
            </section>
          )}

          {(aiResponse || aiMeta) && (
            <section className="result-card">
              <h3>AI Clinical Response</h3>
              <p className="result-card__meta">
                Provider: {aiMeta?.provider_name ?? "unknown"}
                {aiMeta?.model_name ? ` (${aiMeta.model_name})` : ""}
                {" | "}Mode: {aiMeta?.mode ?? "unknown"}
                {aiMeta?.provider_status ? ` | Status: ${aiMeta.provider_status}` : ""}
              </p>
              <p>{aiResponse ?? "No AI response available."}</p>
            </section>
          )}

          {!!retrievedCases.length && (
            <section className="result-card">
              <h3>Top Retrieved Cases</h3>
              <div className="retrieval-list">
                {retrievedCases.slice(0, 3).map((item, index) => (
                  <article key={`${item.patient_id ?? "case"}-${index}`} className="retrieval-item">
                    <strong>{item.pathology ?? "Unknown pathology"}</strong>
                    <p className="result-card__meta">
                      Similarity: {item.similarity ?? "n/a"} | Case ID: {item.patient_id ?? "n/a"}
                    </p>
                    <p>{item.case_text ?? "No case text available."}</p>
                  </article>
                ))}
              </div>
            </section>
          )}

          {therapy?.therapy_plan && (
            <section className="therapy-plan">
              <h3>Therapy Plan</h3>
              <p className="result-card__meta">
                Mode: {therapy.metadata?.mode ?? "unknown"}
              </p>
              <p>{therapy.therapy_plan}</p>
            </section>
          )}

          <details className="raw-json">
            <summary>View raw JSON</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>

          <ChatInterface />
        </div>
      )}
    </section>
  );
}
