import type { Metadata } from "next";
import { buildPageMetadata, indexableRoutes } from "@/lib/seo";

const route = indexableRoutes.find((item) => item.path === "/services");

export const metadata: Metadata = buildPageMetadata({
  title: route?.title ?? "Nabda Medical AI Services",
  description:
    route?.description ??
    "Explore Nabda services for symptom triage, report OCR, lab analysis, and doctor handoff.",
  path: "/services",
});

export default function ServicesLayout({ children }: { children: React.ReactNode }) {
  return children;
}
