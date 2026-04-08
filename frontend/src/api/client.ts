/**
 * All requests use relative `/api` so Vite dev proxy forwards to FastAPI.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function parseJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function fetchMeta(): Promise<{
  api_version: string;
  project: string;
  rag_enabled: boolean;
  faiss_configured: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/v1/meta`);
  const data = (await parseJson(res)) as Record<string, unknown>;
  if (!res.ok) throw new Error(String(data?.detail ?? res.statusText));
  return data as {
    api_version: string;
    project: string;
    rag_enabled: boolean;
    faiss_configured: boolean;
  };
}

export async function postLabs(body: {
  labs: Record<string, unknown>;
  symptoms?: string;
}): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/labs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data;
}

export async function postImage(file: File): Promise<unknown> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/pipeline/image`, {
    method: "POST",
    body: fd,
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data;
}

export async function postSymptoms(body: {
  text: string;
  use_symptom_parser: boolean;
}): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/symptoms`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data;
}

export async function postChat(body: {
  session_id: string;
  message: string;
}): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data;
}

function extractError(data: unknown, res: Response): string {
  if (data && typeof data === "object" && "detail" in data) {
    const d = (data as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    return JSON.stringify(d);
  }
  return res.statusText;
}
