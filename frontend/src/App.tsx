import { useState } from "react";
import { useDiagnosticChat } from "@/features/analysis";
import { ChatInterface } from "@/features/chat";
import { ReviewerPanel } from "@/features/results";
import { AppHeader, useMeta } from "@/shared";

export default function App() {
  const [reviewerOpen, setReviewerOpen] = useState(false);
  const { meta, metaErr } = useMeta();
  const {
    timeline,
    latestAnalysis,
    loading,
    error,
    clarificationState,
    runSymptoms,
    runLabs,
    runImage,
    submitClarificationAnswer,
  } = useDiagnosticChat();

  return (
    <div className="app-shell">
      <AppHeader meta={meta} metaErr={metaErr} />

      <main className="app-main chat-layout">
        <section className="chat-column">
          <ChatInterface
            timeline={timeline}
            loading={loading}
            error={error}
            clarificationState={clarificationState}
            onSubmitSymptoms={runSymptoms}
            onSubmitLabs={runLabs}
            onSubmitImage={runImage}
            onSubmitClarificationAnswer={submitClarificationAnswer}
          />
        </section>

        <ReviewerPanel
          isOpen={reviewerOpen}
          onToggle={() => setReviewerOpen((current) => !current)}
          result={latestAnalysis}
          error={error}
        />
      </main>

      <p className="footer-note">
        Educational prototype only, not for clinical use. Start the API with <code>cd backend &amp;&amp; uvicorn app.main:app --reload</code>, then run <code>npm run dev</code> inside <code>frontend/</code>.
      </p>
    </div>
  );
}
