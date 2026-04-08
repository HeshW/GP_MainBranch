import { useState, useEffect } from 'react';
import { fetchMeta } from '@/api/client';
import { MetaInfo } from '@/types';

export function useMeta() {
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta()
      .then((m) => {
        setMeta({
          api_version: m.api_version,
          rag_enabled: m.rag_enabled,
          faiss_configured: m.faiss_configured,
        });
        setMetaErr(null);
      })
      .catch((e: Error) => {
        setMeta(null);
        setMetaErr(e.message);
      });
  }, []);

  return { meta, metaErr };
}
