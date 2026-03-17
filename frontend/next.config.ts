import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    BACKEND_URL: process.env.BACKEND_URL || "http://13.208.165.10:8000",
  },
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://13.208.165.10:8000/api/v1/:path*',
      },
    ];
  },
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