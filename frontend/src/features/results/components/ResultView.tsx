import { AnalysisResponse } from "@/shared/types";

interface ResultViewProps {
  error: string | null;
  result: AnalysisResponse | null;
}

export function ResultView({ error, result }: ResultViewProps) {
  if (!error && result === null) {
    return null;
  }

  return (
    <section className="panel">
      <h2>Legacy Result View</h2>
      <p>
        The chat-first flow is now the primary UX. Use Reviewer details in the main layout to
        inspect technical pipeline internals.
      </p>
      {error && <p className="reviewer-error">{error}</p>}
      {result && (
        <details className="reviewer-raw">
          <summary>Raw JSON snapshot</summary>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </details>
      )}
    </section>
  );
}
