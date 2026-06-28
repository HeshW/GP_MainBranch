import type { Metadata } from "next";
import { buildPageMetadata, indexableRoutes } from "@/lib/seo";

const route = indexableRoutes.find((item) => item.path === "/contact");

export const metadata: Metadata = buildPageMetadata({
  title: route?.title ?? "Contact Nabda Care Team",
  description:
    route?.description ??
    "Contact the Nabda care team for support and follow-up around the medical assistant workflow.",
  path: "/contact",
});

export default function ContactLayout({ children }: { children: React.ReactNode }) {
  return children;
}
