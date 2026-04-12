import { useState } from "react";
import {
  ImageAnalysisPanel,
  LabAnalysisPanel,
  SymptomsAnalysisPanel,
  TabNavigation,
  useAnalysis,
} from "@/features/analysis";
import { UserInterfaceView } from "@/features/user";
import { ResultView } from "@/features/results";
import { AppHeader, useMeta, AnalysisTab, AppMode } from "@/shared";

export default function App() {
  const [mode, setMode] = useState<AppMode>("user");
  const [tab, setTab] = useState<AnalysisTab>("labs");
  const { meta, metaErr } = useMeta();
  const { loading, result, error, runLabs, runImage, runSymptoms, runClarification } = useAnalysis();

  return (
    <div className={`app-shell ${mode === "user" ? "app-shell--user" : "app-shell--workbench"}`}>
      <AppHeader meta={meta} metaErr={metaErr} mode={mode} onModeChange={setMode} />

      <main className="app-main">
        {mode === "workbench" ? (
          <>
            <TabNavigation currentTab={tab} onTabChange={setTab} />

            {tab === "labs" && <LabAnalysisPanel loading={loading} onRun={runLabs} />}
            {tab === "image" && <ImageAnalysisPanel loading={loading} onRun={runImage} />}
            {tab === "symptoms" && (
              <SymptomsAnalysisPanel loading={loading} onRun={runSymptoms} />
            )}

            <ResultView error={error} result={result} onClarify={runClarification} />
          </>
        ) : (
          <UserInterfaceView
            loading={loading}
            result={result}
            error={error}
            runLabs={runLabs}
            runImage={runImage}
            runSymptoms={runSymptoms}
          />
        )}
      </main>

      <p className="footer-note">
        Educational prototype only, not for clinical use. Use User Interface for streamlined
        patient-facing flow, or switch to Dev Workbench for full diagnostics and testing controls.
      </p>
    </div>
  );
}
