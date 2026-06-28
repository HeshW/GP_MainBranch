import { AppMode, MetaInfo } from "@/shared/types";

interface AppHeaderProps {
  meta: MetaInfo | null;
  metaErr: string | null;
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export function AppHeader({ meta, metaErr, mode, onModeChange }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div>
          <h1 className="app-title">GP Medical Report Analysis</h1>
          <p className="app-sub">
            Graduation demo: OCR lab extraction, rule-based findings, and
            optional RAG through one API and a cleaner web UI.
          </p>
        </div>

        <div className="header-controls">
          <div className="badge-row">
            {meta && (
              <>
                <span className="badge">API v{meta.api_version}</span>
                <span className={meta.rag_enabled ? "badge badge--on" : "badge"}>
                  RAG {meta.rag_enabled ? "on" : "off"}
                </span>
                <span className={meta.faiss_configured ? "badge badge--on" : "badge"}>
                  FAISS {meta.faiss_configured ? "ok" : "n/a"}
                </span>
              </>
            )}
            {metaErr && <span className="badge">API offline</span>}
          </div>

          <div className="mode-toggle" role="tablist" aria-label="Application mode">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "user"}
              className={mode === "user" ? "is-active" : ""}
              onClick={() => onModeChange("user")}
            >
              User interface
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "workbench"}
              className={mode === "workbench" ? "is-active" : ""}
              onClick={() => onModeChange("workbench")}
            >
              Dev workbench
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
