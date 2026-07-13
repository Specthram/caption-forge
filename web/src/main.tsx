/** Front-end entry point: fonts, global CSS, query client, app mount. */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./design/global.css";

import App from "./App";

const queryClient = new QueryClient({
  defaultOptions: {
    // Refetch when the window/tab regains focus so data another surface (or
    // another app writing the same DB) changed is picked up without a manual
    // page reload — the common "stale until I refresh" case. Kept cheap by a
    // short staleTime so a quick tab-switch does not refetch everything.
    queries: { staleTime: 5000, refetchOnWindowFocus: true },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
