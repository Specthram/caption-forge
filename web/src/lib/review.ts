/**
 * Shared helpers for the rule-based Review UI: the per-kind badge palette and
 * the word-level diff the queue and wizard render.
 *
 * The diff is computed at display time (never stored) from the finding's
 * `caption_before`/`caption_after`, so a caption edited since the run simply
 * re-diffs — a finding whose `stale` flag is set is skipped upstream.
 */

import { colors } from "../design/tokens";

export interface KindStyle {
  label: string;
  color: string;
  background: string;
  /** One-line cost/reliability hint shown in a tooltip. */
  hint: string;
}

/** Badge palette + tooltip by rule/finding kind (see the handoff palette). */
export const KIND_STYLES: Record<string, KindStyle> = {
  det: {
    label: "CHECK",
    color: colors.ok,
    background: "#1a2a1c",
    hint: "Deterministic string check — free, instant, always safe to accept.",
  },
  text: {
    label: "TEXT",
    color: colors.warn,
    background: "#2e2513",
    hint: "Judged by the LLM on text alone — fast, no image tokens.",
  },
  vlm: {
    label: "VISION",
    color: colors.info,
    background: "#16222e",
    hint: "Judged by the VLM against the image — slower, loads each image.",
  },
  integrity: {
    label: "INTEGRITY",
    color: colors.warn,
    background: "#2e2513",
    hint: "Built-in well-formedness heuristics — free, always safe to accept.",
  },
};

export function kindStyle(kind: string): KindStyle {
  return KIND_STYLES[kind] ?? KIND_STYLES.text;
}

/** The two finding kinds a human never has to judge (bulk "safe" accept). */
export const SAFE_KINDS = new Set(["det", "integrity"]);

export type DiffOp = "same" | "add" | "del";

export interface DiffSegment {
  op: DiffOp;
  text: string;
}

/**
 * Word-level diff of two captions (LCS), returned as ordered segments.
 *
 * Splits on whitespace, keeping it attached so the rebuilt text reads
 * naturally. `del` marks a word only in the original (struck through, red),
 * `add` a word only in the proposal (green), `same` an unchanged word.
 */
export function wordDiff(before: string, after: string): DiffSegment[] {
  const a = (before || "").split(/(\s+)/).filter((t) => t.length > 0);
  const b = (after || "").split(/(\s+)/).filter((t) => t.length > 0);
  const n = a.length;
  const m = b.length;
  // LCS length table.
  const lcs: number[][] = Array.from({ length: n + 1 }, () =>
    new Array(m + 1).fill(0),
  );
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      lcs[i][j] =
        a[i] === b[j]
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out: DiffSegment[] = [];
  const push = (op: DiffOp, text: string) => {
    const last = out[out.length - 1];
    if (last && last.op === op) last.text += text;
    else out.push({ op, text });
  };
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      push("same", a[i]);
      i += 1;
      j += 1;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      push("del", a[i]);
      i += 1;
    } else {
      push("add", b[j]);
      j += 1;
    }
  }
  while (i < n) {
    push("del", a[i]);
    i += 1;
  }
  while (j < m) {
    push("add", b[j]);
    j += 1;
  }
  return out;
}
