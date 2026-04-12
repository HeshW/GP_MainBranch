import { AnalysisResponse } from "@/shared/types";

interface ReviewerPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  result: AnalysisResponse | null;
  error: string | null;
}

function formatMetric(value: unknown): string {
  if (typeof value === "number") {
    return value.toFixed(2);
  }
  if (typeof value === "string") {
    return value;
  }
  return "n/a";
}

export function ReviewerPanel({ isOpen, onToggle, result, error }: ReviewerPanelProps) {
  const finalDiagnosis = result?.diagnosis?.final_diagnosis;
  const decisionFusion = result?.diagnosis?.decision_fusion;
  const classifierPrediction = result?.diagnosis?.classifier_prediction;
  const retrievedCases = result?.diagnosis?.retrieved_cases ?? [];
  const clarification = result?.diagnosis?.clarification;
  const diagnosticCandidates = result?.diagnosis?.diagnostic_candidates ?? [];
  const geminiResponse = result?.diagnosis?.gemini_response;
  const geminiMeta = result?.diagnosis?.gemini_response_metadata;
  const therapy = result?.therapy;

  return (
    <aside className={isOpen ? "reviewer-column is-open" : "reviewer-column is-collapsed"}>
      <section className="reviewer-panel">
        <button type="button" className="reviewer-toggle" onClick={onToggle}>
          {isOpen ? "Hide reviewer details" : "Reviewer details"}
        </button>

        {!isOpen && (
          <p className="reviewer-collapsed-note">
            Technical internals are hidden. Open this panel to inspect raw pipeline diagnostics.
          </p>
        )}

        {isOpen && (
          <div className="reviewer-content">
            {error && <p className="reviewer-error">{error}</p>}
            {!result && <p className="reviewer-empty">No analysis data yet.</p>}

            {result && (
              <>
                {finalDiagnosis && (
                  <section className="reviewer-card">
                    <h3>Final diagnosis</h3>
                    <p>
                      {finalDiagnosis.diagnosis ?? "Unknown"} | confidence: {formatMetric(finalDiagnosis.confidence)}
                    </p>
                    <p className="reviewer-meta">source: {finalDiagnosis.source ?? "unknown"}</p>
                    {finalDiagnosis.reasoning && <p>{finalDiagnosis.reasoning}</p>}
                  </section>
                )}

                {(result.parsed?.symptoms?.length || result.validated?.symptoms?.length) && (
                  <section className="reviewer-card">
                    <h3>Preprocessing</h3>
                    {!!result.parsed?.symptoms?.length && (
                      <p>
                        parsed symptoms: {result.parsed.symptoms.map((item) => item.symptom).join(", ")}
                      </p>
                    )}
                    {!!result.validated?.symptoms?.length && (
                      <p>validated symptoms: {result.validated.symptoms.join(", ")}</p>
                    )}
                  </section>
                )}

                {(decisionFusion || classifierPrediction) && (
                  <section className="reviewer-card">
                    <h3>Decision fusion</h3>
                    {decisionFusion && (
                      <>
                        <p>primary source: {decisionFusion.primary_source ?? "unknown"}</p>
                        <p>
                          supporting sources: {decisionFusion.supporting_sources?.join(", ") ?? "n/a"}
                        </p>
                        <p>rule validation: {decisionFusion.rule_validation_status ?? "n/a"}</p>
                      </>
                    )}
                    {classifierPrediction && (
                      <p>
                        classifier top label: {classifierPrediction.predicted_label ?? "n/a"} (
                        {formatMetric(classifierPrediction.confidence)})
                      </p>
                    )}
                  </section>
                )}

                {(
                  clarification?.needed
                  || clarification?.applied
                  || clarification?.completed
                  || diagnosticCandidates.length > 0
                ) && (
                  <section className="reviewer-card">
                    <h3>Clarification internals</h3>
                    {clarification && (
                      <p className="reviewer-meta">
                        status: {clarification.completed ? "completed" : clarification.needed ? "pending" : "inactive"}
                        {clarification.override_applied ? " | override applied" : ""}
                      </p>
                    )}
                    {diagnosticCandidates.length > 0 && (
                      <ul className="reviewer-list">
                        {diagnosticCandidates.slice(0, 5).map((item, index) => (
                          <li key={`${item.label}-${index}`}>
                            {item.label} ({formatMetric(item.confidence)})
                          </li>
                        ))}
                      </ul>
                    )}
                    {!!clarification?.reasons?.length && (
                      <ul className="reviewer-list">
                        {clarification.reasons.map((item, index) => (
                          <li key={`${item}-${index}`}>{item}</li>
                        ))}
                      </ul>
                    )}
                    {!!clarification?.questions?.length && (
                      <div className="reviewer-grid">
                        {clarification.questions.map((item, index) => (
                          <article key={`${item.question}-${index}`} className="reviewer-item">
                            <strong>{item.question}</strong>
                            {item.target_conditions?.length ? (
                              <p className="reviewer-meta">
                                targets: {item.target_conditions.join(", ")}
                              </p>
                            ) : null}
                            {item.reason && <p>{item.reason}</p>}
                          </article>
                        ))}
                      </div>
                    )}
                  </section>
                )}

                {(geminiResponse || geminiMeta) && (
                  <section className="reviewer-card">
                    <h3>Gemini response metadata</h3>
                    <p>mode: {geminiMeta?.mode ?? "unknown"}</p>
                    <p>{geminiResponse ?? "No Gemini response payload available."}</p>
                  </section>
                )}

                {retrievedCases.length > 0 && (
                  <section className="reviewer-card">
                    <h3>Retrieved cases</h3>
                    <div className="reviewer-grid">
                      {retrievedCases.slice(0, 5).map((item, index) => (
                        <article key={`${item.patient_id ?? "case"}-${index}`} className="reviewer-item">
                          <strong>{item.pathology ?? "Unknown pathology"}</strong>
                          <p className="reviewer-meta">
                            similarity: {formatMetric(item.similarity)} | case id: {item.patient_id ?? "n/a"}
                          </p>
                          <p>{item.case_text ?? "No case text available."}</p>
                        </article>
                      ))}
                    </div>
                  </section>
                )}

                {therapy?.therapy_plan && (
                  <section className="reviewer-card">
                    <h3>Therapy metadata</h3>
                    <p>mode: {therapy.metadata?.mode ?? "unknown"}</p>
                    <p>{therapy.therapy_plan}</p>
                  </section>
                )}

                <details className="reviewer-raw">
                  <summary>Raw JSON</summary>
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </details>
              </>
            )}
          </div>
        )}
      </section>
    </aside>
  );
}
