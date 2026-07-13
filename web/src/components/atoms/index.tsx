/**
 * Atoms — the smallest styled primitives. Every higher-level component is
 * composed from these, keeping colours/radii/type consistent with the
 * design tokens.
 */

import type {
  ButtonHTMLAttributes,
  CSSProperties,
  ReactNode,
} from "react";
import { colors, font, radii } from "../../design/tokens";

type ButtonVariant = "accent" | "ghost" | "danger" | "raised";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  block?: boolean;
  loading?: boolean;
}

const BUTTON_VARIANTS: Record<ButtonVariant, CSSProperties> = {
  accent: { background: colors.accent, color: colors.onAccent },
  raised: {
    background: colors.raised,
    color: colors.text,
    border: `1px solid ${colors.borderControl}`,
  },
  ghost: {
    background: "transparent",
    color: colors.textSecondary,
    border: `1px solid ${colors.borderControl}`,
  },
  danger: {
    background: "transparent",
    color: colors.danger,
    border: `1px solid ${colors.borderControl}`,
  },
};

export function Button({
  variant = "raised",
  block,
  style,
  disabled,
  loading,
  children,
  ...rest
}: ButtonProps) {
  const isDisabled = disabled || loading;
  return (
    <button
      {...rest}
      disabled={isDisabled}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        padding: "6px 12px",
        borderRadius: radii.control,
        border: "none",
        fontSize: 12,
        fontWeight: 600,
        cursor: isDisabled ? "not-allowed" : "pointer",
        opacity: isDisabled ? 0.5 : 1,
        width: block ? "100%" : undefined,
        ...BUTTON_VARIANTS[variant],
        ...style,
      }}
    >
      {loading && <Spinner size={11} />}
      {children}
    </button>
  );
}

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  title: string;
  children: ReactNode;
}

export function IconButton({ children, style, ...rest }: IconButtonProps) {
  return (
    <button
      {...rest}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 26,
        height: 26,
        borderRadius: radii.control,
        border: `1px solid ${colors.borderControl}`,
        background: "transparent",
        color: colors.textMuted,
        cursor: "pointer",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export interface SegmentedOption {
  value: string;
  label: string;
}

export function Segmented({
  options,
  value,
  onChange,
}: {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: colors.input,
        border: `1px solid ${colors.borderControl}`,
        borderRadius: radii.control,
        padding: 2,
        gap: 2,
      }}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              fontSize: 11.5,
              fontWeight: 600,
              cursor: "pointer",
              background: active ? colors.accentTint : "transparent",
              color: active ? colors.accent : colors.textMuted,
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({
  min,
  max,
  step,
  value,
  onChange,
}: {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      style={{ width: "100%" }}
    />
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: size,
        height: size,
        border: `2px solid ${colors.borderControl}`,
        borderTopColor: colors.accent,
        borderRadius: "50%",
        animation: "cfspin 0.8s linear infinite",
      }}
    />
  );
}

export function ProgressBar({
  pct,
  striped,
  height = 6,
  color = colors.accent,
}: {
  pct: number;
  striped?: boolean;
  height?: number;
  color?: string;
}) {
  return (
    <div
      style={{
        height,
        background: colors.border,
        borderRadius: Math.max(2, height / 2),
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.max(0, Math.min(100, pct))}%`,
          height: "100%",
          background: striped
            ? `repeating-linear-gradient(45deg, ${colors.accent} 0 8px, ${colors.accentHover} 8px 16px)`
            : color,
          backgroundSize: striped ? "24px 24px" : undefined,
          animation: striped ? "cfbar 0.6s linear infinite" : undefined,
          transition: "width 0.3s",
        }}
      />
    </div>
  );
}

/** Pill switch — the machine toggles of Settings. */
export function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      style={{
        width: 34,
        height: 18,
        flex: "none",
        padding: 0,
        border: "none",
        borderRadius: 9,
        position: "relative",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        background: checked ? colors.accent : colors.borderControl,
        transition: "background 0.2s",
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 2,
          left: checked ? 18 : 2,
          width: 14,
          height: 14,
          borderRadius: "50%",
          background: "#fff",
          transition: "left 0.2s",
        }}
      />
    </button>
  );
}

export function Badge({
  children,
  color,
  background,
}: {
  children: ReactNode;
  color: string;
  background?: string;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "1px 6px",
        borderRadius: radii.chip,
        fontSize: 9.5,
        fontWeight: 700,
        fontFamily: font.mono,
        color,
        background: background ?? "rgba(0,0,0,0.35)",
      }}
    >
      {children}
    </span>
  );
}

export function Dot({ color, size = 8 }: { color: string; size?: number }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "50%",
        background: color,
        flex: "none",
      }}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd
      style={{
        fontFamily: font.mono,
        fontSize: 10,
        padding: "1px 5px",
        borderRadius: 4,
        border: `1px solid ${colors.borderControl}`,
        color: colors.textMuted,
        background: colors.input,
      }}
    >
      {children}
    </kbd>
  );
}

export function Toast({
  kind,
  children,
}: {
  kind: "ok" | "danger";
  children: ReactNode;
}) {
  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        right: 20,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "10px 14px",
        borderRadius: radii.control,
        background: colors.raised,
        border: `1px solid ${kind === "ok" ? colors.ok : colors.danger}`,
        color: colors.text,
        fontSize: 12.5,
        boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
      }}
    >
      <Dot color={kind === "ok" ? colors.ok : colors.danger} />
      {children}
    </div>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        fontWeight: 600,
        color: colors.textMuted,
        marginBottom: 6,
      }}
    >
      {children}
    </div>
  );
}
