import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // self-contained server.js + minimal node_modules, keeps the Docker image small (web/Dockerfile)
  output: "standalone",
};

export default nextConfig;
