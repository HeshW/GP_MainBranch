import type { MetadataRoute } from "next";
import { absoluteUrl, indexableRoutes } from "@/lib/seo";

export default function sitemap(): MetadataRoute.Sitemap {
  return indexableRoutes.map((route) => ({
    url: absoluteUrl(route.path),
    lastModified: new Date(),
    changeFrequency: route.changeFrequency,
    priority: route.priority,
    alternates: {
      languages: {
        en: `${absoluteUrl(route.path)}?lang=en`,
        ar: `${absoluteUrl(route.path)}?lang=ar`,
      },
    },
  }));
}
