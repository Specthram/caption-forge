/** Multi-select state for grid batch actions (Zustand). */

import { create } from "zustand";

interface SelectionState {
  selected: Set<string>;
  toggle: (key: string) => void;
  set: (keys: string[]) => void;
  add: (keys: string[]) => void;
  remove: (keys: string[]) => void;
  clear: () => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selected: new Set(),
  toggle: (key) =>
    set((state) => {
      const next = new Set(state.selected);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return { selected: next };
    }),
  set: (keys) => set({ selected: new Set(keys) }),
  add: (keys) =>
    set((state) => {
      const next = new Set(state.selected);
      keys.forEach((key) => next.add(key));
      return { selected: next };
    }),
  remove: (keys) =>
    set((state) => {
      const next = new Set(state.selected);
      keys.forEach((key) => next.delete(key));
      return { selected: next };
    }),
  clear: () => set({ selected: new Set() }),
}));
