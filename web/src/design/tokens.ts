/**
 * Design tokens — the single source of truth for the Caption Forge look.
 *
 * Mirrors the redesign handoff (dark, dense, Lightroom/DaVinci vibe). Every
 * component reads colours, radii and type from here rather than hard-coding
 * hex values, so the palette stays consistent and tweakable in one place.
 */

export const colors = {
  // Backgrounds
  app: "#131418",
  panel: "#17181d",
  toolbar: "#15161b",
  card: "#1b1d22",
  raised: "#22242b",
  input: "#131418",

  // Borders
  border: "#24262d",
  borderControl: "#2c2f38",
  borderHover: "#3a3d47",

  // Text
  text: "#e6e7ea",
  textSecondary: "#d5d7dc",
  textSecondaryAlt: "#c9cbd1",
  textMuted: "#8b8e98",
  textMutedAlt: "#9a9ba2",
  textFaint: "#5c5f6a",

  // Accent (ember)
  accent: "#e8935a",
  accentHover: "#f0aa78",
  onAccent: "#1a1105",
  accentTint: "#26221d",
  accentTintAlt: "#241d12",
  accentBorder: "#4a3a22",

  // Status
  ok: "#6fbf73",
  warn: "#e0b356",
  danger: "#e06c5c",
  info: "#6fa8dc",
  fav: "#e06c9d",
  greenAlt: "#8bc48a",
  // Composition (Depth-Anything V2) — the re-skin signal fused into the
  // auto-builder's Proximity graph and the depth index step's dot.
  composition: "#5ac7c0",

  // Grounding (SigLIP) — its own cool-blue surface, so the grounding card
  // reads as a measurement panel rather than another ember-accented action.
  grounding: "#6fa8dc",
  groundingBg: "#141b22",
  groundingBorder: "#2e3a4a",
  groundingBtn: "#1b2530",
  groundingBtnHover: "#20303f",

  // Watermark Lab — a dedicated violet surface, so everything that touches
  // virtual patches reads apart from the ember actions and the blue grounding.
  watermark: "#a78bda",
  watermarkStrong: "#6d4fa8",
  watermarkBg: "#1c1826",
  watermarkBtn: "#1e1a2a",
  watermarkBorder: "rgba(167,139,218,0.4)",
  watermarkBorderSoft: "rgba(167,139,218,0.35)",
  watermarkZoneDetected: "rgba(224,108,92,0.85)",
  watermarkZonePatched: "rgba(167,139,218,0.75)",
  // "watermarked" (still-detected) status: amber. Patched/flattened ride the
  // violet accent above; clean is the ok green.
  watermarkAmber: "#e0b356",
  watermarkAmberBg: "#2a2312",

  // Tag categories
  catCharacter: "#e06c9d",
  catFraming: "#e0b356",
  catSetting: "#6fa8dc",
  catStyle: "#8bc48a",
  catGeneral: "#9a9ba2",
} as const;

export const radii = {
  card: 8,
  panel: 9,
  control: 6,
  modal: 12,
  chip: 10,
} as const;

export const font = {
  sans: "'IBM Plex Sans', sans-serif",
  mono: "'IBM Plex Mono', monospace",
  base: 13,
} as const;

export const shadow = {
  modal: "0 24px 80px rgba(0,0,0,0.6)",
} as const;

/** Quality badge colour by normalized score (0-100). */
export function qualityColor(score: number | null | undefined): string {
  if (score == null) return colors.textFaint;
  if (score >= 90) return colors.info;
  if (score >= 75) return colors.ok;
  if (score >= 60) return colors.warn;
  return colors.danger;
}

/**
 * Status colour of a grounded claim / tag against the validation threshold.
 *
 * The ramp of the handoff: at or above the threshold it is supported, within
 * 14 points below it is borderline, further down it is very likely absent.
 * A claim the user marked non-validated goes grey — it is out of the count.
 */
export const GROUNDING_AMBER_BAND = 14;

export function groundingColor(
  score: number,
  threshold: number,
  rejected = false,
): string {
  if (rejected) return colors.textFaint;
  if (score >= threshold) return colors.ok;
  if (score >= threshold - GROUNDING_AMBER_BAND) return colors.warn;
  return colors.danger;
}

/** Dot colour of an Index-pipeline step (src/index_steps.py keys). */
export function stepColor(key: string): string {
  switch (key) {
    case "thumbs":
      return colors.greenAlt;
    case "quality":
      return colors.info;
    case "embed":
      return colors.accent;
    case "depth":
      return colors.composition;
    case "wd14":
      return colors.fav;
    default:
      return colors.catGeneral;
  }
}

/** Toggle-chip dot colour of a Quality-report scorer. */
export function scorerColor(id: string): string {
  switch (id) {
    case "musiq":
      return colors.info;
    case "topiq_nr":
      return colors.greenAlt;
    case "laion_aes":
      return colors.fav;
    case "qalign":
      return colors.warn;
    case "dinov2":
      return colors.accent;
    default:
      return colors.catGeneral;
  }
}

/** Colour of a report row's tone. */
export function toneColor(tone: string): string {
  switch (tone) {
    case "ok":
      return colors.ok;
    case "warn":
      return colors.warn;
    case "danger":
      return colors.danger;
    case "info":
      return colors.info;
    default:
      return colors.textMuted;
  }
}

/** Cluster palette of the diversity map (wraps past six clusters). */
export const clusterColors = [
  colors.accent,
  colors.info,
  colors.greenAlt,
  colors.warn,
  colors.fav,
  colors.textMutedAlt,
] as const;

export function clusterColor(index: number): string {
  return clusterColors[index % clusterColors.length];
}

/** Tag chip colour by category name (falls back to general grey). */
export function categoryColor(name: string | null | undefined): string {
  switch ((name || "").toLowerCase()) {
    case "character":
      return colors.catCharacter;
    case "framing":
      return colors.catFraming;
    case "setting":
      return colors.catSetting;
    case "style":
      return colors.catStyle;
    default:
      return colors.catGeneral;
  }
}

/** Review badge colour + short label by verdict. */
export function reviewBadge(label: string): { text: string; color: string } {
  switch (label) {
    case "ok":
      return { text: "OK", color: colors.ok };
    case "warn":
      return { text: "WARN", color: colors.warn };
    case "fix":
      return { text: "FIX", color: colors.info };
    default:
      return { text: "—", color: colors.textFaint };
  }
}

/** Watermark aggregate-status colour + short label (v2). */
export function watermarkStatus(
  status: string | null | undefined,
): { color: string; label: string } {
  switch (status) {
    case "flattened":
      return { color: colors.watermark, label: "flattened" };
    case "patched":
      return { color: colors.watermark, label: "patched" };
    case "detected":
      return { color: colors.watermarkAmber, label: "watermark" };
    default:
      return { color: colors.ok, label: "clean" };
  }
}

/** Watermark confidence-score colour by band (≥80 ok, ≥65 amber, else red). */
export function watermarkScore(score: number | null | undefined): string {
  if (score == null) return colors.textFaint;
  if (score >= 80) return colors.ok;
  if (score >= 65) return colors.warn;
  return colors.danger;
}

/** Deploy dot colour by backend status string. */
export function deployColor(status: string | null | undefined): string {
  switch (status) {
    case "GREEN":
      return colors.ok;
    case "ORANGE":
      return colors.warn;
    case "RED":
      return colors.danger;
    default:
      return colors.textFaint;
  }
}
