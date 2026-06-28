import type { Metadata } from "next";
import { buildPageMetadata, indexableRoutes } from "@/lib/seo";

const route = indexableRoutes.find((item) => item.path === "/chatbot");

export const metadata: Metadata = buildPageMetadata({
  title: route?.title ?? "Nabda AI Medical Assistant",
  description:
    route?.description ??
    "Chat with Nabda for symptom triage, report OCR, lab analysis, and follow-up guidance.",
  path: "/chatbot",
});

export default function ChatbotLayout({ children }: { children: React.ReactNode }) {
  return children;
}
