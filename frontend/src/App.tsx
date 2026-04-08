import { useState } from "react";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useMeta } from "@/hooks/useMeta";
import { Tab } from "@/types";

import { AppHeader } from "@/components/layout/AppHeader";
import { TabNavigation } from "@/components/tabs/TabNavigation";
import { LabAnalysisTab } from "@/components/tabs/LabAnalysisTab";
import { ImageAnalysisTab } from "@/components/tabs/ImageAnalysisTab";
import { SymptomsAnalysisTab } from "@/components/tabs/SymptomsAnalysisTab";
import { ResultView } from "@/components/results/ResultView";

export default function App() {
  const [tab, setTab] = useState<Tab>("labs");
  const { meta, metaErr } = useMeta();
  const { loading, result, error, runLabs, runImage, runSymptoms } = useAnalysis();

  return (
    <div className="app-shell">
      <AppHeader meta={meta} metaErr={metaErr} />

      <main className="app-main">
        <TabNavigation currentTab={tab} onTabChange={setTab} />

        {tab === "labs" && (
          <LabAnalysisTab loading={loading} onRun={runLabs} />
        )}

        {tab === "image" && (
          <ImageAnalysisTab loading={loading} onRun={runImage} />
        )}

        {tab === "symptoms" && (
          <SymptomsAnalysisTab loading={loading} onRun={runSymptoms} />
        )}

        <ResultView error={error} result={result} />
      </main>

      <p className="footer-note">
        Educational prototype only — not for clinical use. Start the API with{" "}
        <code>cd backend &amp;&amp; uvicorn app.main:app --reload</code>, then{" "}
        <code>npm run dev</code> in <code>frontend/</code>.
      </p>
    </div>
  );
}
