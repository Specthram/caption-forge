/**
 * Coverage arithmetic of the Index pipeline (see `src/index_steps.py`).
 *
 * A library's progress is the sum of the `done / total` of the steps
 * *enabled on this machine*: a disabled step never counts as missing, so a
 * GPU-less install still reads 100% once its thumbnails are cached.
 */

import type { IndexStep, StepCounts } from "../api/types";

export interface Coverage {
  done: number;
  total: number;
  missing: number;
}

export function overallCoverage(
  steps: IndexStep[],
  counts: StepCounts,
): Coverage {
  let done = 0;
  let total = 0;
  for (const step of steps) {
    if (!step.enabled) continue;
    done += counts[step.key]?.done ?? 0;
    total += counts[step.key]?.total ?? 0;
  }
  return { done, total, missing: total - done };
}

/** Percentage of a `done / total` pair; an empty scope reads as complete. */
export function pct(done: number, total: number): number {
  return total > 0 ? Math.round((done * 100) / total) : 100;
}
