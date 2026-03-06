import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Environment variables exposed to Server Components and Route Handlers only
  env: {
    BACKEND_URL: process.env.BACKEND_URL || "http://localhost:8000",
  },
  // Allow cross-origin images from avatar providers
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "i.pravatar.cc",
      },
    ],
  },
};

export default nextConfig;
