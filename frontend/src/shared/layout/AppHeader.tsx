import { MetaInfo } from "@/shared/types";

interface AppHeaderProps {
  meta: MetaInfo | null;
  metaErr: string | null;
}

export function AppHeader({ meta, metaErr }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div>
          <h1 className="app-title">GP Medical Report Analysis</h1>
          <p className="app-sub">
            Chat-first diagnostic workflow with inline clarification and a
            reviewer panel for technical pipeline details.
          </p>
        </div>

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
      </div>
    </header>
  );
}
