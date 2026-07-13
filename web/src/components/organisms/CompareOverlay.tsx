/**
 * Two-image comparator: a draggable wipe bar, or side by side.
 *
 * The wipe is the reason this component exists. The clip lives on a
 * *non-transformed* wrapper and the zoom/pan transform on an inner child,
 * so the orange bar stays where the cursor left it while the images move
 * underneath — a single transformed stage would drag the seam along with
 * the pixels. Both viewports share one ``translate/scale``, so the side by
 * side mode is zoomed and panned in lockstep.
 *
 * Wheel zooms (x1.2 per notch, clamped 1x-8x), left-drag pans, and the bar
 * is clamped to 4-96% so neither image ever disappears entirely.
 */

import { useEffect, useRef, useState } from "react";
import { colors, font, radii } from "../../design/tokens";

/** One side of the comparison. */
export interface CompareSide {
  src: string;
  name: string;
  /** The mono line under the name: resolution · framing · quality. */
  info?: string;
  infoColor?: string;
  /** The chip pinned in the viewport corner ("candidate" / "dataset"). */
  badge: string;
  badgeColor: string;
}

/** The two footer buttons of the composer's comparator. */
export interface CompareActions {
  dropLabel: string;
  keepLabel: string;
  onDrop: () => void;
  onKeep: () => void;
}

const MODES = [
  { value: "wipe", label: "Wipe" },
  { value: "side", label: "Side by side" },
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const MIN_POSITION = 4;
const MAX_POSITION = 96;

export function CompareOverlay({
  title,
  subtitle,
  candidate,
  reference,
  onClose,
  actions,
  zIndex = 90,
}: {
  title: string;
  subtitle: string;
  candidate: CompareSide;
  reference: CompareSide;
  onClose: () => void;
  actions?: CompareActions;
  zIndex?: number;
}) {
  const [mode, setMode] = useState("wipe");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [position, setPosition] = useState(50);
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Capture + stopImmediatePropagation: Escape closes the topmost overlay
    // only — the composer underneath must stay open.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopImmediatePropagation();
      onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const reset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setPosition(50);
  };

  const onWheel = (event: React.WheelEvent) => {
    const factor = event.deltaY < 0 ? 1.2 : 1 / 1.2;
    setZoom((current) =>
      Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, current * factor)),
    );
  };

  const onPanDown = (event: React.MouseEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX - pan.x;
    const startY = event.clientY - pan.y;
    const move = (moved: MouseEvent) =>
      setPan({ x: moved.clientX - startX, y: moved.clientY - startY });
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const onSliderDown = (event: React.MouseEvent) => {
    event.stopPropagation();
    event.preventDefault();
    const rect = viewport.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const move = (moved: MouseEvent) => {
      const percent = ((moved.clientX - rect.left) / rect.width) * 100;
      setPosition(
        Math.max(MIN_POSITION, Math.min(MAX_POSITION, percent)),
      );
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;

  return (
    <div
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      style={{ ...backdrop, zIndex }}
    >
      <div onClick={(event) => event.stopPropagation()} style={panel}>
        <div style={header}>
          <div>
            <div style={{ fontSize: 13.5, fontWeight: 700 }}>{title}</div>
            <div style={subtitleStyle}>{subtitle}</div>
          </div>
          <div style={{ flex: 1 }} />
          <div style={segmented}>
            {MODES.map((option) => (
              <span
                key={option.value}
                onClick={() => setMode(option.value)}
                style={{
                  padding: "4px 11px",
                  fontSize: 11,
                  cursor: "pointer",
                  color:
                    mode === option.value ? colors.text : colors.textMuted,
                  background:
                    mode === option.value
                      ? colors.borderControl
                      : "transparent",
                }}
              >
                {option.label}
              </span>
            ))}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <button
              onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.25))}
              style={zoomButton}
            >
              −
            </button>
            <span style={zoomLabel}>{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.25))}
              style={zoomButton}
            >
              +
            </button>
            <button onClick={reset} style={resetButton}>
              reset
            </button>
          </div>
          <span onClick={onClose} style={closeButton}>
            ✕
          </span>
        </div>

        <div style={body}>
          {mode === "wipe" ? (
            <div
              ref={viewport}
              onWheel={onWheel}
              onMouseDown={onPanDown}
              style={stage}
            >
              <Layer src={reference.src} transform={transform} />
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  clipPath: `inset(0 ${100 - position}% 0 0)`,
                }}
              >
                <Layer src={candidate.src} transform={transform} />
              </div>
              <div
                onMouseDown={onSliderDown}
                style={{ ...sliderHit, left: `calc(${position}% - 7px)` }}
              >
                <div style={sliderLine} />
                <div style={sliderKnob}>⇄</div>
              </div>
              <Chip side={candidate} corner="left" />
              <Chip side={reference} corner="right" />
            </div>
          ) : (
            <div style={{ display: "flex", gap: 10, ...stageBox }}>
              {[candidate, reference].map((side) => (
                <div
                  key={side.badge}
                  onWheel={onWheel}
                  onMouseDown={onPanDown}
                  style={{ ...stage, flex: 1, height: "100%" }}
                >
                  <Layer src={side.src} transform={transform} />
                  <Chip side={side} corner="left" />
                </div>
              ))}
            </div>
          )}
          <div style={hint}>
            wheel = zoom · drag = pan · pull the orange bar to wipe · zoom
            and position are synced
          </div>
        </div>

        <div style={footer}>
          <SideCard side={candidate} />
          <span style={divider} />
          <SideCard side={reference} />
          <div style={{ flex: 1 }} />
          <button onClick={onClose} style={ghostButton}>
            Close
          </button>
          {actions && (
            <>
              <button onClick={actions.onDrop} style={dangerButton}>
                {actions.dropLabel}
              </button>
              <button onClick={actions.onKeep} style={accentButton}>
                {actions.keepLabel}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Layer({ src, transform }: { src: string; transform: string }) {
  return (
    <div style={{ position: "absolute", inset: 0, transform }}>
      <img src={src} alt="" draggable={false} style={image} />
    </div>
  );
}

function Chip({ side, corner }: { side: CompareSide; corner: "left" | "right" }) {
  return (
    <span
      style={{
        ...chip,
        color: side.badgeColor,
        [corner]: 8,
      }}
    >
      {side.badge}
    </span>
  );
}

function SideCard({ side }: { side: CompareSide }) {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      <img src={side.src} alt="" style={swatch} />
      <div>
        <div style={{ fontSize: 11.5, fontWeight: 600 }}>
          {side.name}{" "}
          <span
            style={{
              fontSize: 9.5,
              color: side.badgeColor,
              fontFamily: font.mono,
            }}
          >
            {side.badge}
          </span>
        </div>
        {side.info && (
          <div
            style={{
              fontSize: 10,
              fontFamily: font.mono,
              color: side.infoColor ?? colors.textMuted,
            }}
          >
            {side.info}
          </div>
        )}
      </div>
    </div>
  );
}

const backdrop = {
  position: "fixed",
  inset: 0,
  background: "rgba(8,9,11,0.85)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 20,
} as const;

const panel = {
  width: "100%",
  maxWidth: 1060,
  maxHeight: "100%",
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: radii.modal,
  boxShadow: "0 24px 80px rgba(0,0,0,0.7)",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
} as const;

const header = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "12px 16px",
  borderBottom: `1px solid ${colors.border}`,
  flexWrap: "wrap",
} as const;

const subtitleStyle = {
  fontSize: 10.5,
  color: colors.textFaint,
  fontFamily: font.mono,
} as const;

const segmented = {
  display: "flex",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.control,
  overflow: "hidden",
} as const;

const zoomButton = {
  width: 24,
  height: 24,
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 5,
  background: colors.card,
  color: colors.textSecondary,
  fontSize: 13,
  cursor: "pointer",
} as const;

const zoomLabel = {
  width: 44,
  textAlign: "center",
  fontFamily: font.mono,
  fontSize: 10.5,
  color: colors.textMuted,
} as const;

const resetButton = {
  padding: "4px 9px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 5,
  background: colors.card,
  color: colors.textMuted,
  fontSize: 10.5,
  cursor: "pointer",
  fontFamily: font.mono,
} as const;

const closeButton = {
  cursor: "pointer",
  color: colors.textMuted,
  fontSize: 15,
  padding: "4px 8px",
} as const;

const body = {
  flex: 1,
  minHeight: 0,
  padding: 14,
  display: "flex",
  flexDirection: "column",
  gap: 8,
  overflowY: "auto",
} as const;

const stageBox = { height: "52vh", minHeight: 300 } as const;

const stage = {
  position: "relative",
  ...stageBox,
  overflow: "hidden",
  background: "#0f1013",
  cursor: "grab",
  borderRadius: radii.card,
  border: `1px solid ${colors.borderControl}`,
} as const;

const image = {
  width: "100%",
  height: "100%",
  objectFit: "contain",
  userSelect: "none",
  pointerEvents: "none",
} as const;

const sliderHit = {
  position: "absolute",
  top: 0,
  bottom: 0,
  width: 14,
  cursor: "ew-resize",
  display: "flex",
  justifyContent: "center",
  zIndex: 3,
} as const;

const sliderLine = {
  width: 2,
  height: "100%",
  background: colors.accent,
  boxShadow: "0 0 10px rgba(232,147,90,0.45)",
} as const;

const sliderKnob = {
  position: "absolute",
  top: "50%",
  marginTop: -11,
  width: 22,
  height: 22,
  borderRadius: "50%",
  background: colors.accent,
  color: colors.onAccent,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 11,
  fontWeight: 700,
  boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
} as const;

const chip = {
  position: "absolute",
  top: 8,
  fontSize: 9.5,
  fontFamily: font.mono,
  padding: "2px 6px",
  borderRadius: 4,
  background: "rgba(15,16,19,0.8)",
  zIndex: 2,
} as const;

const hint = {
  fontSize: 10,
  color: colors.textFaint,
  fontFamily: font.mono,
} as const;

const footer = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 14,
  padding: "12px 16px",
  borderTop: `1px solid ${colors.border}`,
  background: colors.toolbar,
  flexWrap: "wrap",
} as const;

const swatch = {
  width: 12,
  height: 12,
  borderRadius: 3,
  objectFit: "cover",
  border: `1px solid ${colors.borderControl}`,
  flex: "none",
} as const;

const divider = {
  width: 1,
  height: 26,
  background: colors.borderControl,
} as const;

const ghostButton = {
  padding: "8px 14px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: "transparent",
  color: colors.textMutedAlt,
  fontSize: 12,
  cursor: "pointer",
} as const;

const dangerButton = {
  padding: "8px 14px",
  border: "1px solid #3a2622",
  borderRadius: 7,
  background: "#241715",
  color: colors.danger,
  fontSize: 12,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const accentButton = {
  padding: "8px 16px",
  border: "none",
  borderRadius: 7,
  background: colors.accent,
  color: colors.onAccent,
  fontSize: 12,
  fontWeight: 700,
  cursor: "pointer",
} as const;
