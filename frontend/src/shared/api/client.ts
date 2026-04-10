import { AnalysisResponse } from "@/shared/types";

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

function extractError(data: unknown, res: Response): string {
  if (data && typeof data === "object" && "detail" in data) {
    const detail = (data as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return res.statusText;
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
}): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/labs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as AnalysisResponse;
}

export async function postImage(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/v1/pipeline/image`, {
    method: "POST",
    body: formData,
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as AnalysisResponse;
}

export async function postSymptoms(body: {
  text: string;
  use_symptom_parser: boolean;
}): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/symptoms`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as AnalysisResponse;
}

export async function postChat(body: {
  session_id: string;
  message: string;
}): Promise<{ response: string }> {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as { response: string };
}
