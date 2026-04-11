import { useState } from "react";
import {
  ImageAnalysisPanel,
  LabAnalysisPanel,
  SymptomsAnalysisPanel,
  TabNavigation,
  useAnalysis,
} from "@/features/analysis";
import { ResultView } from "@/features/results";
import { AppHeader, useMeta, AnalysisTab } from "@/shared";

export default function App() {
  const [tab, setTab] = useState<AnalysisTab>("labs");
  const { meta, metaErr } = useMeta();
  const { loading, result, error, runLabs, runImage, runSymptoms, runClarification } = useAnalysis();

  return (
    <div className="app-shell">
      <AppHeader meta={meta} metaErr={metaErr} />

      <main className="app-main">
        <TabNavigation currentTab={tab} onTabChange={setTab} />

        {tab === "labs" && <LabAnalysisPanel loading={loading} onRun={runLabs} />}
        {tab === "image" && <ImageAnalysisPanel loading={loading} onRun={runImage} />}
        {tab === "symptoms" && (
          <SymptomsAnalysisPanel loading={loading} onRun={runSymptoms} />
        )}

        <ResultView error={error} result={result} onClarify={runClarification} />
      </main>

      <p className="footer-note">
        Educational prototype only, not for clinical use. Start the API with <code>cd backend &amp;&amp; uvicorn app.main:app --reload</code>, then run <code>npm run dev</code> inside <code>frontend/</code>.
      </p>
    </div>
  );
}
