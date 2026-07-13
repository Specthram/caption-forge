/**
 * The Studio's coverage map: a PCA projection of the DINOv2 vectors of the
 * pool, coloured by role — orange picks, grey candidates, blue example
 * seeds — over a density heatmap so a dense corpus stays readable. Grey
 * candidates are a viewport-culled, stride-sampled cap that grows as you
 * zoom (LOD); past ~2.8× points become mini-thumbnails. Wheel and
 * double-click zoom anchor on the cursor (max 10×); dragging with the left
 * or middle button pans 1:1. When a pick is selected for replacement the
 * grey points become click targets that swap it in.
 */

import { useEffect, useRef, useState } from "react";
import { colors } from "../../design/tokens";
import type { AutobuildMapPoint, AutobuildZone } from "../../api/types";

const MIN_ZOOM = 1;
const MAX_ZOOM = 10;
const WHEEL_IN = 1.25;
const WHEEL_OUT = 0.8;
const DBLCLICK_IN = 1.6;
const THUMB_ZOOM = 2.8;
const GREY_BASE_CAP = 170;
const THUMB_CAP = 90;
const HEAT_COLUMNS = 20;
const HEAT_ROWS = 15;

const ROLE_LABEL: Record<string, string> = {
  pick: "pick — chosen by the engine",
  candidate: "candidate — not selected",
  seed: "seed — your example",
};

const thumbUrl = (id: number) => `/api/media/${id}/thumb`;

interface CoverageMap {
  width: number;
  height: number;
  points: AutobuildMapPoint[];
  zones: AutobuildZone[];
}

export function AutoBuildCoverageMap({
  map,
  hovered,
  onHover,
  clusterColors,
  clusterHighlight,
  colorByCluster,
  selectedPick,
  onReplacePick,
}: {
  map: CoverageMap;
  hovered: number | null;
  onHover: (id: number | null) => void;
  clusterColors: string[];
  clusterHighlight: number | null;
  colorByCluster: boolean;
  selectedPick: number | null;
  onReplacePick: (inId: number) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [zoneHover, setZoneHover] = useState<number | null>(null);
  // Screen position of the cursor, for the large hover preview.
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{
    px: number;
    py: number;
    tx: number;
    ty: number;
  } | null>(null);

  const { width: W, height: H } = map;
  const replaceMode = selectedPick != null;

  // View transform: a point's on-screen (viewBox) position is x*zoom + tx.
  const zoomAt = (factor: number, mx: number, my: number) => {
    setZoom((z) => {
      const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * factor));
      const f = next / z;
      if (next <= MIN_ZOOM) {
        setTx(0);
        setTy(0);
      } else {
        setTx((t) => mx - (mx - t) * f);
        setTy((t) => my - (my - t) * f);
      }
      return next;
    });
  };

  const pointerToView = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { mx: W / 2, my: H / 2 };
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return { mx: W / 2, my: H / 2 };
    return {
      mx: ((clientX - rect.left) / rect.width) * W,
      my: ((clientY - rect.top) / rect.height) * H,
    };
  };

  // A native, non-passive wheel listener so preventDefault stops the page
  // scrolling while the map zooms, anchored on the cursor.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const { mx, my } = pointerToView(event.clientX, event.clientY);
      zoomAt(event.deltaY < 0 ? WHEEL_IN : WHEEL_OUT, mx, my);
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W, H]);

  const onPointerDown = (event: React.PointerEvent) => {
    if (zoom <= 1 || (event.button !== 0 && event.button !== 1)) return;
    drag.current = { px: event.clientX, py: event.clientY, tx, ty };
  };
  const onPointerMove = (event: React.PointerEvent) => {
    const state = drag.current;
    if (!state) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    setTx(state.tx + ((event.clientX - state.px) / rect.width) * W);
    setTy(state.ty + ((event.clientY - state.py) / rect.height) * H);
  };
  const endDrag = () => {
    drag.current = null;
  };
  const onDoubleClick = (event: React.MouseEvent) => {
    const { mx, my } = pointerToView(event.clientX, event.clientY);
    zoomAt(DBLCLICK_IN, mx, my);
  };

  const sx = (x: number) => x * zoom + tx;
  const sy = (y: number) => y * zoom + ty;
  const inView = (p: AutobuildMapPoint) => {
    if (!p.xy) return false;
    const x = sx(p.xy[0]);
    const y = sy(p.xy[1]);
    return x > -4 && x < W + 4 && y > -4 && y < H + 4;
  };
  const r = (base: number) => base / zoom;

  // --- density heatmap backdrop -------------------------------------------
  const hw = W / HEAT_COLUMNS;
  const hh = H / HEAT_ROWS;
  const heat = new Array(HEAT_COLUMNS * HEAT_ROWS).fill(0);
  for (const point of map.points) {
    if (!point.xy) continue;
    const col = Math.min(HEAT_COLUMNS - 1, Math.floor(point.xy[0] / hw));
    const row = Math.min(HEAT_ROWS - 1, Math.floor(point.xy[1] / hh));
    heat[row * HEAT_COLUMNS + col] += 1;
  }
  const heatMax = Math.max(1, ...heat);

  // --- grey candidates: cap + stride + LOD --------------------------------
  const thumbMode = zoom >= THUMB_ZOOM;
  const greyVisible = map.points.filter(
    (p) => p.role === "candidate" && inView(p),
  );
  const cap = Math.round(GREY_BASE_CAP * zoom);
  const stride = Math.max(1, Math.ceil(greyVisible.length / cap));
  const greyShown = greyVisible.filter(
    (p, index) => index % stride === 0 || p.id === hovered,
  );
  const picks = map.points.filter((p) => p.role === "pick");
  const seedPoints = map.points.filter((p) => p.role === "seed");

  const pickColor = (p: AutobuildMapPoint) =>
    colorByCluster && p.cluster != null
      ? (clusterColors[p.cluster] ?? colors.accent)
      : colors.accent;
  const pickDim = (p: AutobuildMapPoint) =>
    clusterHighlight == null || p.cluster === clusterHighlight ? 1 : 0.22;

  const hoveredPoint =
    hovered != null
      ? (map.points.find((point) => point.id === hovered) ?? null)
      : null;
  const selPoint =
    selectedPick != null
      ? (map.points.find((point) => point.id === selectedPick) ?? null)
      : null;

  const enter = (id: number) => onHover(id);
  const leave = (id: number) => onHover(hovered === id ? null : hovered);
  const hitProps = (p: AutobuildMapPoint) => ({
    onMouseEnter: () => enter(p.id),
    onMouseLeave: () => leave(p.id),
    onClick:
      replaceMode && p.role !== "pick"
        ? () => onReplacePick(p.id)
        : undefined,
    style: {
      cursor: replaceMode && p.role !== "pick" ? "crosshair" : "pointer",
    } as const,
  });

  const thumbRect = (
    p: AutobuildMapPoint,
    stroke: string,
    strokeW: number,
    opacity: number,
  ) => {
    if (!p.xy) return null;
    const w = r(10);
    const h = r(7.5);
    return (
      <g key={`t${p.id}`} opacity={opacity}>
        <image
          href={thumbUrl(p.id)}
          x={p.xy[0] - w / 2}
          y={p.xy[1] - h / 2}
          width={w}
          height={h}
          preserveAspectRatio="xMidYMid slice"
        />
        <rect
          x={p.xy[0] - w / 2}
          y={p.xy[1] - h / 2}
          width={w}
          height={h}
          rx={r(1.2)}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeW / zoom}
        />
      </g>
    );
  };

  return (
    <div style={{ position: "relative" }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        style={{
          width: "100%",
          display: "block",
          cursor: zoom > 1 ? "grab" : "default",
          touchAction: "none",
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onMouseMove={(event) =>
          setCursor({ x: event.clientX, y: event.clientY })
        }
        onMouseLeave={() => setCursor(null)}
        onDoubleClick={onDoubleClick}
      >
        <g transform={`translate(${tx} ${ty}) scale(${zoom})`}>
          {heat.map((count, index) =>
            count > 0 ? (
              <rect
                key={`hm${index}`}
                x={(index % HEAT_COLUMNS) * hw}
                y={Math.floor(index / HEAT_COLUMNS) * hh}
                width={hw}
                height={hh}
                fill="#3f434e"
                opacity={0.05 + 0.3 * (count / heatMax)}
              />
            ) : null,
          )}
          {map.zones.map((zone, index) => (
            <circle
              key={`z${index}`}
              cx={zone.x}
              cy={zone.y}
              r={zone.r}
              fill={
                zoneHover === index
                  ? "rgba(111,168,220,0.16)"
                  : "rgba(111,168,220,0.07)"
              }
              stroke={colors.info}
              strokeWidth={r(zoneHover === index ? 0.8 : 0.5)}
              strokeDasharray={`${r(2)} ${r(2)}`}
              onMouseEnter={() => setZoneHover(index)}
              onMouseLeave={() =>
                setZoneHover((current) =>
                  current === index ? null : current,
                )
              }
            />
          ))}
          {thumbMode
            ? greyShown
                .slice(0, THUMB_CAP)
                .map((p) =>
                  thumbRect(
                    p,
                    hovered === p.id ? colors.text : "#4a4d57",
                    hovered === p.id ? 0.6 : 0.3,
                    replaceMode ? 1 : 0.85,
                  ),
                )
            : greyShown.map((p) =>
                p.xy ? (
                  <circle
                    key={`c${p.id}`}
                    cx={p.xy[0]}
                    cy={p.xy[1]}
                    r={r(hovered === p.id ? 1.9 : replaceMode ? 1.35 : 1.1)}
                    fill={
                      hovered === p.id
                        ? "#8b8e98"
                        : replaceMode
                          ? "#565b68"
                          : "#3f434e"
                    }
                  />
                ) : null,
              )}
          {thumbMode
            ? picks
                .filter(inView)
                .slice(0, THUMB_CAP)
                .map((p) => thumbRect(p, pickColor(p), 0.7, pickDim(p)))
            : picks.map((p) =>
                p.xy ? (
                  <circle
                    key={`p${p.id}`}
                    cx={p.xy[0]}
                    cy={p.xy[1]}
                    r={r(hovered === p.id ? 2.8 : 2)}
                    fill={pickColor(p)}
                    opacity={pickDim(p)}
                    stroke={
                      clusterHighlight != null &&
                      p.cluster === clusterHighlight
                        ? colors.text
                        : "none"
                    }
                    strokeWidth={r(0.4)}
                  />
                ) : null,
              )}
          {seedPoints.map((p) =>
            p.xy ? (
              <circle
                key={`sd${p.id}`}
                cx={p.xy[0]}
                cy={p.xy[1]}
                r={r(hovered === p.id ? 3 : 2.4)}
                fill={colors.info}
                stroke="#9dc4e8"
                strokeWidth={r(0.5)}
              />
            ) : null,
          )}
          {selPoint?.xy && (
            <circle
              cx={selPoint.xy[0]}
              cy={selPoint.xy[1]}
              r={r(3.4)}
              fill="none"
              stroke={colors.accent}
              strokeWidth={r(0.8)}
              strokeDasharray={`${r(1.6)} ${r(1.2)}`}
            />
          )}
          {hoveredPoint?.xy && (
            <circle
              cx={hoveredPoint.xy[0]}
              cy={hoveredPoint.xy[1]}
              r={r(3.8)}
              fill="none"
              stroke={colors.accent}
              strokeWidth={r(0.7)}
            />
          )}
          {map.points.map((p) =>
            inView(p) && p.xy ? (
              <circle
                key={`hit${p.id}`}
                cx={p.xy[0]}
                cy={p.xy[1]}
                r={r(2.6)}
                fill="transparent"
                {...hitProps(p)}
              />
            ) : null,
          )}
        </g>
      </svg>

      {hoveredPoint?.xy && cursor && (
        <HoverPreview
          id={hoveredPoint.id}
          name={hoveredPoint.name}
          role={
            replaceMode && hoveredPoint.role !== "pick"
              ? "candidate — click to swap it in"
              : ROLE_LABEL[hoveredPoint.role]
          }
          roleColor={replaceMode ? colors.accent : colors.textFaint}
          cursor={cursor}
        />
      )}

      {zoneHover != null && map.zones[zoneHover] && (
        <div style={zoneTip}>
          <div style={{ color: colors.info }}>
            ◌ uncovered · {map.zones[zoneHover].count} candidate(s)
          </div>
          <div style={{ color: colors.textMutedAlt }}>
            {map.zones[zoneHover].top_tags.join(" · ") || "no dominant tag"}
          </div>
          <div style={{ color: colors.textFaint }}>
            {map.zones[zoneHover].why}
          </div>
        </div>
      )}

      <div style={controls}>
        <button
          onClick={() => zoomAt(WHEEL_OUT, W / 2, H / 2)}
          style={zoomButton}
        >
          −
        </button>
        <span style={{ fontSize: 9, color: colors.textFaint, minWidth: 26 }}>
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={() => zoomAt(WHEEL_IN, W / 2, H / 2)}
          style={zoomButton}
        >
          +
        </button>
        <button
          onClick={() => {
            setZoom(1);
            setTx(0);
            setTy(0);
          }}
          style={zoomButton}
        >
          ⟲
        </button>
      </div>
    </div>
  );
}

const PREVIEW_SIZE = 320;
const PREVIEW_GAP = 18;

/**
 * A large image preview of the hovered map point, pinned to the left of the
 * cursor (flips to the right near the left edge) as a fixed overlay so it can
 * spill out of the narrow map panel and stay big and readable.
 */
function HoverPreview({
  id,
  name,
  role,
  roleColor,
  cursor,
}: {
  id: number;
  name: string;
  role: string;
  roleColor: string;
  cursor: { x: number; y: number };
}) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let left = cursor.x - PREVIEW_SIZE - PREVIEW_GAP;
  if (left < 8) left = cursor.x + PREVIEW_GAP; // flip to the right near edge
  left = Math.max(8, Math.min(vw - PREVIEW_SIZE - 8, left));
  const top = Math.max(
    8,
    Math.min(vh - PREVIEW_SIZE - 40, cursor.y - PREVIEW_SIZE / 2),
  );
  return (
    <div style={{ ...hoverPreview, left, top, width: PREVIEW_SIZE }}>
      <img
        src={thumbUrl(id)}
        alt=""
        style={{
          width: "100%",
          maxHeight: PREVIEW_SIZE,
          objectFit: "contain",
          borderRadius: 6,
          background: colors.app,
          display: "block",
        }}
      />
      <div style={{ color: colors.text, fontSize: 11, marginTop: 5 }}>
        {name}
      </div>
      <div style={{ color: roleColor, fontSize: 10 }}>{role}</div>
    </div>
  );
}

const hoverPreview = {
  position: "fixed",
  zIndex: 999,
  padding: 6,
  borderRadius: 8,
  background: "rgba(15,16,19,0.96)",
  border: `1px solid ${colors.borderHover}`,
  boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
  fontFamily: "monospace",
  pointerEvents: "none",
} as const;

const zoneTip = {
  position: "absolute",
  top: 4,
  left: 4,
  maxWidth: 180,
  padding: "5px 7px",
  borderRadius: 6,
  background: "rgba(15,16,19,0.94)",
  border: "1px solid #2f4860",
  fontSize: 9,
  lineHeight: 1.4,
  pointerEvents: "none",
  zIndex: 3,
} as const;

const controls = {
  position: "absolute",
  top: 4,
  right: 4,
  display: "flex",
  alignItems: "center",
  gap: 3,
} as const;

const zoomButton = {
  width: 18,
  height: 18,
  borderRadius: 4,
  border: `1px solid ${colors.borderControl}`,
  background: "rgba(15,16,19,0.7)",
  color: colors.textMuted,
  fontSize: 11,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 0,
} as const;
