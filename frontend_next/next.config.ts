import type { NextConfig } from "next";

const backendUrl =
  process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const enableBackendProxy =
  process.env.NODE_ENV !== "development" || process.env.ENABLE_NEXT_BACKEND_PROXY === "true";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "images.unsplash.com",
      },
    ],
  },
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    if (!enableBackendProxy) return [];

    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
