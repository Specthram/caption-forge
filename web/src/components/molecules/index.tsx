/** Molecules — small composites built from atoms. */

import type { ReactNode } from "react";
import type { GpuInfo } from "../../api/types";
import { Badge, Dot } from "../atoms";
import {
  categoryColor,
  colors,
  font,
  qualityColor,
  reviewBadge,
} from "../../design/tokens";

export function QualityBadge({ score }: { score: number | null }) {
  if (score == null) return null;
  return (
    <Badge color={colors.onAccent} background={qualityColor(score)}>
      {Math.round(score)}
    </Badge>
  );
}

export function ReviewBadge({
  label,
  issues,
}: {
  label: string;
  issues: string[];
}) {
  const badge = reviewBadge(label);
  return (
    <span
      title={issues.join(", ")}
      style={{
        fontFamily: font.mono,
        fontSize: 9.5,
        fontWeight: 700,
        color: badge.color,
      }}
    >
      {badge.text}
    </span>
  );
}

export function TagChip({
  name,
  category,
  color,
  onRemove,
}: {
  name: string;
  category?: string;
  color?: string;
  onRemove?: () => void;
}) {
  const dotColor = color ?? categoryColor(category);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        padding: "2px 7px",
        borderRadius: 10,
        fontSize: 11,
        background: colors.raised,
        border: `1px solid ${colors.borderControl}`,
        color: colors.textSecondary,
      }}
    >
      <Dot color={dotColor} size={7} />
      {name}
      {onRemove && (
        <span
          onClick={onRemove}
          style={{ cursor: "pointer", color: colors.textMuted }}
        >
          ✕
        </span>
      )}
    </span>
  );
}

export function RepeatsStepper({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  const step = (delta: number) => onChange(Math.max(1, value + delta));
  const btn = {
    width: 22,
    height: 22,
    borderRadius: 5,
    border: `1px solid ${colors.borderControl}`,
    background: colors.raised,
    color: colors.text,
    cursor: "pointer",
  } as const;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <button style={btn} onClick={() => step(-1)}>
        −
      </button>
      <span style={{ fontFamily: font.mono, minWidth: 14, textAlign: "center" }}>
        {value}
      </span>
      <button style={btn} onClick={() => step(1)}>
        +
      </button>
    </span>
  );
}

export function NavItem({
  icon,
  label,
  count,
  active,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  count?: number | string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "6px 10px",
        borderRadius: 6,
        cursor: "pointer",
        fontSize: 13,
        fontWeight: 500,
        color: active ? colors.accent : colors.textSecondary,
        background: active ? colors.accentTint : "transparent",
      }}
    >
      <span
        style={{
          width: 15,
          textAlign: "center",
          display: "inline-flex",
          justifyContent: "center",
        }}
      >
        {icon}
      </span>
      <span style={{ flex: 1 }}>{label}</span>
      {count != null && (
        <span
          style={{
            fontSize: 10,
            color: colors.textFaint,
            fontFamily: font.mono,
          }}
        >
          {count}
        </span>
      )}
    </div>
  );
}

export function GpuChip({
  gpu,
  totalGb,
}: {
  gpu: GpuInfo | null;
  totalGb: number | null;
}) {
  // No CUDA device: fall back to the plain total (or "n/a" off-GPU).
  if (gpu == null) {
    return (
      <span
        style={{ fontFamily: font.mono, fontSize: 11, color: colors.textMuted }}
      >
        ● {totalGb == null ? "GPU n/a" : `${totalGb.toFixed(1)} GB`}
      </span>
    );
  }
  // Dot warms as the device fills — green under half, amber past it, red
  // when nearly full (a load about to OOM).
  const ratio = gpu.total_gb > 0 ? gpu.used_gb / gpu.total_gb : 0;
  const dot =
    ratio > 0.85 ? colors.danger : ratio > 0.5 ? colors.warn : colors.ok;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontFamily: font.mono,
        fontSize: 11,
        color: colors.textSecondary,
      }}
      title={`${gpu.free_gb.toFixed(1)} GB free of ${gpu.total_gb.toFixed(
        1,
      )} GB`}
    >
      <Dot color={dot} size={7} /> {gpu.name} ·{" "}
      <span style={{ color: dot }}>{gpu.used_gb.toFixed(1)}</span> /{" "}
      {gpu.total_gb.toFixed(0)} GB
    </span>
  );
}
