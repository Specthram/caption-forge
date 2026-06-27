import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API and the job WebSocket to the FastAPI backend
// (uvicorn on 7776), so the front-end talks to one logical origin. The
// production bundle is served by FastAPI itself, where these paths are
// same-origin and the proxy is irrelevant.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:7776",
      "/ws": {
        target: "ws://127.0.0.1:7776",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Split the third-party libraries out of the app chunk so neither
        // trips the 500 kB warning and the vendor bundle caches across app
        // rebuilds. React and TanStack Query are the heavy ones.
        manualChunks: {
          react: ["react", "react-dom"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
