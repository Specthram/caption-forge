/**
 * The Quality report's two charts: the per-media quality histogram and the
 * DINOv2 diversity map (a 2-D PCA projection, one dot per embedded media,
 * coloured by cluster, near-dup pairs ringed amber, outliers ringed blue).
 *
 * Hovering a dot floats a thumbnail tooltip — clamped inside the card and
 * flipped below the dot near the top edge; clicking it opens the zoom
 * lightbox on the original file.
 */

import { useRef, useState } from "react";
import {
  clusterColor,
  colors,
  font,
  qualityColor,
  radii,
} from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import type { DatasetReport, ReportMapPoint } from "../../api/types";

const cardStyle = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.card,
  padding: 14,
} as const;

const labelStyle = {
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: ".08em",
  fontWeight: 600,
  color: colors.textMuted,
} as const;

const MAP_HEIGHT = 300;
/** Margin keeping a dot (and its ring) fully inside the plot area. */
const MAP_INSET = 14;
const DOT = 9;
const TOOLTIP_WIDTH = 132;
const TOOLTIP_HEIGHT = 128;

/** The quality floor under which an image is flagged (mirrors the engine). */
const LOW_QUALITY_FLOOR = 70;

function Distribution({ report }: { report: DatasetReport }) {
  const max = Math.max(1, ...report.distribution.map((b) => b.count));
  const below = report.distribution
    .filter((bucket) => bucket.midpoint < LOW_QUALITY_FLOOR)
    .reduce((sum, bucket) => sum + bucket.count, 0);
  return (
    <div style={cardStyle}>
      <div style={labelStyle}>Quality distribution</div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${report.distribution.length}, 1fr)`,
          gap: 10,
          alignItems: "end",
          height: 168,
          marginTop: 14,
        }}
      >
        {report.distribution.map((bucket) => (
          <div
            key={bucket.label}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 5,
              height: "100%",
              justifyContent: "flex-end",
            }}
          >
            <span
              style={{
                fontFamily: font.mono,
                fontSize: 10.5,
                color: colors.textSecondary,
              }}
            >
              {bucket.count}
            </span>
            <div
              style={{
                width: "100%",
                height: `${(bucket.count / max) * 100}%`,
                minHeight: bucket.count ? 3 : 1,
                borderRadius: 3,
                background: bucket.count
                  ? qualityColor(bucket.midpoint)
                  : colors.border,
              }}
            />
            <span
              style={{
                fontFamily: font.mono,
                fontSize: 9.5,
                color: colors.textFaint,
              }}
            >
              {bucket.label}
            </span>
          </div>
        ))}
      </div>
      <div
        style={{
          marginTop: 12,
          padding: "7px 9px",
          borderRadius: radii.control,
          fontSize: 11,
          background: below ? colors.accentTintAlt : colors.raised,
          color: below ? colors.accent : colors.textFaint,
        }}
      >
        {below
          ? `${below} media below ${LOW_QUALITY_FLOOR} drag the floor — ` +
            "flagged in issues below"
          : `Nothing under the ${LOW_QUALITY_FLOOR} floor.`}
      </div>
    </div>
  );
}

interface Hover {
  point: ReportMapPoint;
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
        zIndex: 3,
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
          fontSize: 10,
          marginTop: 4,
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
          fontSize: 9.5,
          color: colors.textFaint,
        }}
      >
        {point.quality == null ? "—" : Math.round(point.quality)} ·{" "}
        {point.width ?? "?"}×{point.height ?? "?"}
      </div>
    </div>
  );
}

function DiversityMap({ report }: { report: DatasetReport }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const openZoom = useUiStore((state) => state.openZoom);

  /**
   * Anchor the tooltip on the hovered dot, measured against the card: the
   * dots are laid out in percentages (no width to read at render time), so
   * the pixel anchor can only come from the DOM on hover.
   */
  const show = (point: ReportMapPoint, dot: HTMLDivElement) => {
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

  const clusters = Array.from(
    new Set(report.map_points.map((point) => point.cluster)),
  ).sort((a, b) => a - b);

  return (
    <div style={cardStyle}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span style={labelStyle}>Diversity map — DINOv2</span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textFaint,
          }}
        >
          spread {report.spread.toFixed(2)} · {report.clusters} clusters
        </span>
      </div>

      <div
        ref={boxRef}
        style={{
          position: "relative",
          height: MAP_HEIGHT,
          marginTop: 12,
          borderRadius: radii.control,
          background: colors.app,
          border: `1px solid ${colors.border}`,
          overflow: "hidden",
        }}
        onMouseLeave={() => setHover(null)}
      >
        {report.map_points.length === 0 && (
          <div
            style={{
              height: "100%",
              display: "grid",
              placeItems: "center",
              fontSize: 11.5,
              color: colors.textFaint,
            }}
          >
            No embeddings — enable the DINOv2 scorer and re-run.
          </div>
        )}
        <div style={{ position: "absolute", inset: MAP_INSET }}>
          {report.map_points.map((point) => {
            const ring = point.near_dup
              ? colors.warn
              : point.outlier
                ? colors.info
                : null;
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
                  width: DOT,
                  height: DOT,
                  borderRadius: "50%",
                  cursor: "zoom-in",
                  background: clusterColor(point.cluster),
                  boxShadow: ring ? `0 0 0 2.5px ${ring}` : "none",
                }}
              />
            );
          })}
        </div>
        {hover && <Tooltip hover={hover} />}
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginTop: 10,
          fontSize: 10.5,
          color: colors.textMuted,
        }}
      >
        {clusters.map((index) => (
          <span
            key={index}
            style={{ display: "inline-flex", alignItems: "center", gap: 5 }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: clusterColor(index),
              }}
            />
            cluster {index + 1}
          </span>
        ))}
        <span style={{ color: colors.warn }}>◎ near-dup</span>
        <span style={{ color: colors.info }}>◎ outlier</span>
      </div>
    </div>
  );
}

export function QualityCharts({ report }: { report: DatasetReport }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 12 }}>
      <Distribution report={report} />
      <DiversityMap report={report} />
    </div>
  );
}
