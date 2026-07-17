/**
 * Caption-tab generation state (Zustand): the shared prompt/params the left
 * panel edits and the batch bar reuses, plus the client-side lock set.
 *
 * Lock is session-only (no schema change): locked media are excluded from
 * Generate-all and Generate-selected via the ``exclude_ids`` request field.
 */

import { create } from "zustand";

interface CaptionState {
  /**
   * Prompt text and seed are the only generation params kept here: the
   * rest (temperature, thinking, image res, max tokens) live on the model
   * profile (see /api/profiles). Seed "-1" = random each run.
   */
  prompt: string;
  seed: string;
  reviewAfter: boolean;
  /** Off = only caption media whose caption is still empty. */
  recaption: boolean;
  /** Chain a SigLIP grounding pass on every freshly written caption. */
  groundAfter: boolean;
  locked: Set<string>;

  set: (partial: Partial<Omit<CaptionState, "locked">>) => void;
  toggleLock: (key: string) => void;
  lockMany: (keys: string[], locked: boolean) => void;
}

export const useCaptionStore = create<CaptionState>((set) => ({
  prompt: "",
  seed: "-1",
  reviewAfter: false,
  recaption: true,
  groundAfter: false,
  locked: new Set(),

  set: (partial) => set(partial),
  toggleLock: (key) =>
    set((state) => {
      const next = new Set(state.locked);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return { locked: next };
    }),
  lockMany: (keys, locked) =>
    set((state) => {
      const next = new Set(state.locked);
      keys.forEach((key) => (locked ? next.add(key) : next.delete(key)));
      return { locked: next };
    }),
}));
