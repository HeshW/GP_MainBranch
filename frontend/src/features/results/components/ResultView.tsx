import { ChatInterface } from "@/features/chat";
import { AnalysisResponse } from "@/shared/types";

interface ResultViewProps {
  error: string | null;
  result: AnalysisResponse | null;
}

export function ResultView({ error, result }: ResultViewProps) {
  if (!error && result === null) {
    return null;
  }

  const finalDiagnosis = result?.diagnosis?.final_diagnosis;
  const decisionFusion = result?.diagnosis?.decision_fusion;
  const classifierPrediction = result?.diagnosis?.classifier_prediction;
  const retrievedCases = result?.diagnosis?.retrieved_cases ?? [];
  const geminiResponse = result?.diagnosis?.gemini_response;
  const geminiMeta = result?.diagnosis?.gemini_response_metadata;
  const therapy = result?.therapy;

  return (
    <section className="panel">
      <h2>Pipeline Result</h2>
      {error && <p className="err">{error}</p>}

      {result !== null && !error && (
        <div className="result-stack">
          {finalDiagnosis && (
            <section className="result-card">
              <h3>Final Diagnosis</h3>
              <p className="result-card__headline">
                {finalDiagnosis.diagnosis ?? "Unknown diagnosis"}
              </p>
              <p className="result-card__meta">
                Source: {finalDiagnosis.source ?? "unknown"} | Confidence:{" "}
                {finalDiagnosis.confidence ?? "n/a"}
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

          {(geminiResponse || geminiMeta) && (
            <section className="result-card">
              <h3>Gemini Clinical Response</h3>
              <p className="result-card__meta">
                Mode: {geminiMeta?.mode ?? "unknown"}
              </p>
              <p>{geminiResponse ?? "No Gemini response available."}</p>
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
