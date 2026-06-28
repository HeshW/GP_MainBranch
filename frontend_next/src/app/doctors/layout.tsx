import type { Metadata } from "next";
import { buildPageMetadata, indexableRoutes } from "@/lib/seo";

const route = indexableRoutes.find((item) => item.path === "/doctors");

export const metadata: Metadata = buildPageMetadata({
  title: route?.title ?? "Find Doctors Near You",
  description:
    route?.description ??
    "Use Nabda to search nearby doctors with Google Maps and move from AI guidance to real care.",
  path: "/doctors",
});

export default function DoctorsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
