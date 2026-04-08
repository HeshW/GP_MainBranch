export type Tab = "labs" | "image" | "symptoms";

export interface MetaInfo {
  api_version: string;
  rag_enabled: boolean;
  faiss_configured: boolean;
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}
