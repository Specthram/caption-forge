/** Global keyboard shortcuts: ⌘K opens search, Esc closes overlays. */

import { useEffect } from "react";
import { useUiStore } from "../store/uiStore";

export function useKeyboard() {
  const state = useUiStore.getState;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        state().toggleSearch(true);
        return;
      }
      if (event.key === "Escape") {
        const store = state();
        if (store.compare.open) store.closeCompare();
        else if (store.zoom.open) store.closeZoom();
        else if (store.searchOpen) store.toggleSearch(false);
        else if (store.grounding.open) store.closeGrounding();
        else if (store.jobsOpen) store.toggleJobs(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state]);
}
