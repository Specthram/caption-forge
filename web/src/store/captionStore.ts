/**
 * Caption-tab generation state (Zustand): the shared prompt/params the left
 * panel edits and the batch bar reuses, plus the client-side lock set.
 *
 * Lock is session-only (no schema change): locked media are excluded from
 * Generate-all and Generate-selected via the ``exclude_ids`` request field.
 */

import { create } from "zustand";

interface CaptionState {
  model: string;
  prompt: string;
  temperature: number;
  seed: string;
  think: string;
  imgRes: number;
  reviewAfter: boolean;
  /** Judge model for the review-after pass ("" = reuse the captioner). */
  reviewJudge: string;
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
  model: "",
  prompt: "",
  temperature: 0.7,
  seed: "-1",
  think: "auto",
  imgRes: 1024,
  reviewAfter: false,
  reviewJudge: "",
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
