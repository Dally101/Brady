import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  server: {
    port: 8080, // Set the port explicitly for Render
  },
};

export default nextConfig;
