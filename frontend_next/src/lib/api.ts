import type { AnalysisResponse, MetaInfo } from "./medical-types";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const SERVICE_API_KEY = (process.env.NEXT_PUBLIC_API_KEY ?? "").trim();

type ChatRequest = {
  session_id: string;
  message: string;
};

function withServiceApiKey(headers?: HeadersInit): Headers {
  const merged = new Headers(headers);
  if (SERVICE_API_KEY && !merged.has("X-API-Key")) {
    merged.set("X-API-Key", SERVICE_API_KEY);
  }
  return merged;
}

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
  return res.statusText || "Request failed";
}

function consumeSseEventBlock(
  eventBlock: string,
  onChunk: (chunk: string) => void,
  currentText: string,
): string {
  let nextText = currentText;

  for (const line of eventBlock.split("\n")) {
    if (!line.startsWith("data:")) continue;
    const chunk = line.startsWith("data: ") ? line.slice(6) : line.slice(5);
    if (!chunk) continue;
    nextText += chunk;
    onChunk(chunk);
  }

  return nextText;
}

export async function fetchMeta(): Promise<MetaInfo> {
  const res = await fetch(`${API_BASE}/api/v1/meta`, {
    headers: withServiceApiKey(),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as MetaInfo;
}

export async function postLabs(body: {
  labs: Record<string, unknown>;
  symptoms?: string;
}): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/labs`, {
    method: "POST",
    headers: withServiceApiKey({ "Content-Type": "application/json" }),
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
    headers: withServiceApiKey(),
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
    headers: withServiceApiKey({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as AnalysisResponse;
}

export async function postClarification(body: {
  report: Record<string, unknown>;
  diagnosis?: Record<string, unknown>;
  answers: string[];
  low_confidence_threshold?: number;
}): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/v1/pipeline/diagnosis/clarify`, {
    method: "POST",
    headers: withServiceApiKey({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as AnalysisResponse;
}

export async function postChat(body: ChatRequest): Promise<{ response: string }> {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: withServiceApiKey({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await parseJson(res);
  if (!res.ok) throw new Error(extractError(data, res));
  return data as { response: string };
}

export async function postChatStream(
  body: ChatRequest,
  onChunk: (chunk: string) => void,
): Promise<string> {
  const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers: withServiceApiKey({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const data = await parseJson(res);
    throw new Error(extractError(data, res));
  }

  if (!res.body) return "";

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let delimiterIndex = buffer.indexOf("\n\n");

    while (delimiterIndex !== -1) {
      const eventBlock = buffer.slice(0, delimiterIndex);
      buffer = buffer.slice(delimiterIndex + 2);
      fullText = consumeSseEventBlock(eventBlock, onChunk, fullText);
      delimiterIndex = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode().replace(/\r\n/g, "\n");
  const trailingEvent = buffer.trim();
  if (trailingEvent) {
    fullText = consumeSseEventBlock(trailingEvent, onChunk, fullText);
  }

  return fullText;
}
