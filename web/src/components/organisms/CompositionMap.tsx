/**
 * The Quality report's composition map (Depth-Anything V2).
 *
 * A second 2-D map that tells the opposite story to the diversity map: the
 * diversity map is a DINOv2 (style) projection, this is a Depth-Anything V2
 * (framing) projection. Nodes are positioned by their depth signature — so
 * images that share a framing cluster together even when their style differs
 * — and coloured by visual style, which makes the mixing legible. Dashed teal
 * links mark the re-skins (same framing, different style) DINOv2 rates as far
 * apart.
 *
 * Same hover/zoom idioms as the diversity map: hovering a dot floats a
 * thumbnail tooltip (clamped inside the card, flipped below near the top
 * edge); clicking opens the zoom lightbox on the original file.
 */

import { useMemo, useRef, useState } from "react";
import {
  colors,
  compositionStyleColors,
  font,
  radii,
  styleColor,
} from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import type { CompositionMapPoint, DatasetReport } from "../../api/types";

const cardStyle = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.card,
  padding: 14,
} as const;

const PLOT_HEIGHT = 260;
/** Margin keeping a dot (and its ring) fully inside the plot area. */
const MAP_INSET = 14;
const DOT = 8;
const DOT_RESKIN = 10;
const TOOLTIP_WIDTH = 132;
const TOOLTIP_HEIGHT = 150;

/** The five style buckets in a fixed legend order. */
const STYLE_ORDER = ["warm", "cool", "neutral", "green", "pink"] as const;

interface Hover {
  point: CompositionMapPoint;
  left: number;
  top: number;
}

function Tooltip({ hover }: { hover: Hover }) {
  const { point } = hover;
  return (
    <div
      style={{
        position: "absolute",
        left: hover.left,
        top: hover.top,
        width: TOOLTIP_WIDTH,
        pointerEvents: "none",
        background: colors.raised,
        border: `1px solid ${colors.borderHover}`,
        borderRadius: radii.control,
        padding: 6,
        zIndex: 6,
        display: "flex",
        flexDirection: "column",
        gap: 5,
      }}
    >
      <img
        src={`/api/media/${point.id}/thumb`}
        alt=""
        style={{
          width: "100%",
          height: 84,
          objectFit: "cover",
          borderRadius: 4,
          display: "block",
        }}
      />
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 9.5,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          color: colors.textSecondary,
        }}
      >
        {point.name}
      </div>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 9,
          color: colors.composition,
        }}
      >
        framing · {point.framing + 1}
      </div>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 9,
          color: point.reskin ? colors.composition : colors.textFaint,
        }}
      >
        {point.reskin ? "◇ re-skin — shares this framing" : "unique framing"}
      </div>
    </div>
  );
}

export function CompositionMap({ report }: { report: DatasetReport }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const openZoom = useUiStore((state) => state.openZoom);

  // Guard against reports stored before the composition fields existed: a
  // stale blob has no composition_map/links/framings/reskins, and reading
  // them unguarded would crash the whole tab.
  const points = useMemo(
    () => report.composition_map ?? [],
    [report.composition_map],
  );
  const links = report.composition_links ?? [];
  const framings = report.framings ?? 0;
  const reskins = report.reskins ?? 0;

  // Map id -> normalized position, so a re-skin link can find both endpoints.
  const byId = useMemo(() => {
    const table = new Map<number, CompositionMapPoint>();
    for (const point of points) table.set(point.id, point);
    return table;
  }, [points]);

  /** Anchor the tooltip on the hovered dot, measured against the card. */
  const show = (point: CompositionMapPoint, dot: HTMLDivElement) => {
    const box = boxRef.current;
    if (!box) return;
    const boxRect = box.getBoundingClientRect();
    const dotRect = dot.getBoundingClientRect();
    const x = dotRect.left - boxRect.left + dotRect.width / 2;
    const y = dotRect.top - boxRect.top + dotRect.height / 2;
    const flip = y < TOOLTIP_HEIGHT + 12;
    setHover({
      point,
      left: Math.max(
        0,
        Math.min(boxRect.width - TOOLTIP_WIDTH, x - TOOLTIP_WIDTH / 2),
      ),
      top: flip ? y + 14 : y - TOOLTIP_HEIGHT - 10,
    });
  };

  return (
    <div
      style={{ ...cardStyle, display: "flex", flexDirection: "column", gap: 10 }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: colors.composition,
            flex: "none",
          }}
        />
        <div style={{ fontSize: 12, fontWeight: 600, flex: 1 }}>
          Composition map{" "}
          <span style={{ color: colors.textFaint, fontWeight: 400 }}>
            — framing space, independent of style
          </span>
        </div>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textFaint,
          }}
        >
          Depth-Anything V2 · depth signatures
        </span>
      </div>

      <div
        ref={boxRef}
        style={{
          position: "relative",
          height: PLOT_HEIGHT,
          borderRadius: radii.control,
          background: colors.app,
          border: `1px solid ${colors.border}`,
          overflow: "hidden",
        }}
        onMouseLeave={() => setHover(null)}
      >
        {points.length === 0 ? (
          <div
            style={{
              height: "100%",
              display: "grid",
              placeItems: "center",
              fontSize: 11.5,
              color: colors.textFaint,
              padding: 20,
              textAlign: "center",
            }}
          >
            No depth signatures — run the Composition index in Libraries to map
            the framing space.
          </div>
        ) : (
          <div style={{ position: "absolute", inset: MAP_INSET }}>
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 1,
              }}
            >
              {links.map(([a, b]) => {
                const from = byId.get(a);
                const to = byId.get(b);
                if (!from || !to) return null;
                return (
                  <line
                    key={`${a}-${b}`}
                    x1={from.x * 100}
                    y1={(1 - from.y) * 100}
                    x2={to.x * 100}
                    y2={(1 - to.y) * 100}
                    stroke={colors.composition}
                    strokeWidth={0.45}
                    strokeDasharray="1.6 1.4"
                    opacity={0.75}
                  />
                );
              })}
            </svg>
            {points.map((point) => {
              const size = point.reskin ? DOT_RESKIN : DOT;
              return (
                <div
                  key={point.id}
                  title={point.name}
                  onMouseEnter={(event) => show(point, event.currentTarget)}
                  onClick={() =>
                    openZoom(`/api/media/${point.id}/file`, point.name)
                  }
                  style={{
                    position: "absolute",
                    left: `${point.x * 100}%`,
                    top: `${(1 - point.y) * 100}%`,
                    transform: "translate(-50%, -50%)",
                    width: size,
                    height: size,
                    borderRadius: "50%",
                    cursor: "zoom-in",
                    background: styleColor(point.style),
                    border: point.reskin
                      ? `2px solid ${colors.composition}`
                      : "1px solid rgba(0,0,0,0.4)",
                    zIndex: 2,
                  }}
                />
              );
            })}
          </div>
        )}
        {points.length > 0 && (
          <div
            style={{
              position: "absolute",
              left: 10,
              bottom: 9,
              fontFamily: font.mono,
              fontSize: 9,
              color: colors.composition,
              background: "rgba(15,16,19,0.8)",
              padding: "1px 6px",
              borderRadius: 4,
            }}
          >
            ◇ dashed link = same framing, different style
          </div>
        )}
        {hover && <Tooltip hover={hover} />}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          fontFamily: font.mono,
          fontSize: 10,
          color: colors.textMuted,
          flexWrap: "wrap",
        }}
      >
        <span style={{ color: colors.textFaint }}>node colour = visual style</span>
        {STYLE_ORDER.map((key) => (
          <span
            key={key}
            style={{ display: "flex", alignItems: "center", gap: 5 }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: compositionStyleColors[key],
              }}
            />
            {key}
          </span>
        ))}
        <span style={{ flex: 1 }} />
        <span>
          {framings} framings · {reskins} re-skins
        </span>
      </div>

      {reskins > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 10.5,
            color: colors.textMuted,
            borderTop: `1px solid ${colors.border}`,
            paddingTop: 9,
          }}
        >
          <span style={{ color: colors.composition }}>◇</span>
          <span>
            <strong style={{ color: colors.textSecondary }}>
              {reskins} composition re-skin{reskins === 1 ? "" : "s"}
            </strong>{" "}
            — same framing, different style. DINOv2 read them as separate
            images; the depth signal still teaches the LoRA the pose.
          </span>
        </div>
      )}
    </div>
  );
}
