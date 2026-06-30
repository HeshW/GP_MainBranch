import { useEffect, useState } from "react";
import { fetchMeta } from "@/shared/api";
import { MetaInfo } from "@/shared/types";

export function useMeta() {
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta()
      .then((payload) => {
        setMeta({
          api_version: payload.api_version,
          rag_enabled: payload.rag_enabled,
          faiss_configured: payload.faiss_configured,
        });
        setMetaErr(null);
      })
      .catch((error: Error) => {
        setMeta(null);
        setMetaErr(error.message);
      });
  }, []);

  return { meta, metaErr };
}
