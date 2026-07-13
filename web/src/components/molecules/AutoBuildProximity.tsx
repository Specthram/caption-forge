/**
 * The Studio's Proximity view: the resemblance links between every image in
 * the proposed selection, drawn as an interactive similarity graph so the
 * user can spot redundant frames that a LoRA would memorise instead of
 * learning from. Nodes are the picks laid out from their DINOv2 projection
 * (lightly de-clumped), edges are real pairwise cosine links above a
 * threshold; red links/rings mark the app-wide near-duplicate line (0.92),
 * blue links plain resemblance.
 *
 * A third, fused signal — Depth-Anything V2 composition depth — catches
 * *re-skins*: pairs with the same composition/pose/framing but a different
 * style (photo ↔ anime, changed outfit) that DINOv2 rates as far apart yet
 * are still redundant (the model memorises the shared layout). Each edge
 * carries both cosines (`dinoS`, `depthS`); the fused link is
 * `max(dinoS, W·depthS)`. Rather than wire re-skins with long crossing teal
 * lines, the force-directed layout *pulls* strong re-skin pairs together, so
 * the teal links stay short and quiet, and light up on hover. DINOv2 still
 * anchors the layout, so this graph stays aligned with the coverage map.
 *
 * Hovering isolates a node's links; clicking selects it, then a grey point
 * on the coverage map swaps it out — the same replace flow as Grid/Clusters.
 */

import { useMemo, useState } from "react";
import { colors, font } from "../../design/tokens";
import type { AutobuildPick } from "../../api/types";

// The near-duplicate line, app-wide (src/dataset_quality.NEAR_DUP_COSINE).
const NEAR_DUP = 0.92;

// Composition fusion (mirrors src/autobuild_studio): the depth cosine is
// trusted a touch less than DINOv2, and a "strong re-skin" — the pair the
// layout pulls together — is a fixed, threshold-independent line so the
// layout never recomputes when the slider moves.
const COMP_W = 0.95;
const COMP_STRONG_DEPTH = 0.9;
const COMP_DINO_FAR = 0.85;

// Layout: anchor the nodes to their projection, de-clump exact overlaps, and
// — when composition is on — pull strong re-skin pairs together. The anchor
// pull softens and the relaxation runs longer when composition is on so the
// attraction has room to move nodes.
const LAYOUT_ITERATIONS = 70;
const LAYOUT_ITERATIONS_COMP = 90;
const ANCHOR_PULL = 0.09;
const ANCHOR_PULL_COMP = 0.055;
const REPULSE_RADIUS = 2.6;
const REPULSE_STRENGTH = 0.45;
const COMP_ATTRACT_TARGET = 2.2;
const COMP_ATTRACT_GAIN = 0.11;
const COMP_ATTRACT_CAP = 0.14;
const VIEW_W = 100;
const VIEW_H = 74;
const CLAMP_X: [number, number] = [4, 96];
const CLAMP_Y: [number, number] = [4, 70];

// Colours the handoff fixes outside the shared token set.
const NODE_STROKE = "#0f1013";
const RESEMBLE_LINK = "#5e7fa6";
const COMP_LINK = colors.composition; // #5ac7c0
const SEL_GLOW = "rgba(232,147,90,0.30)";
// Cycled colour of each redundant group's dot / member palette.
const GROUP_PALETTE = [
  "#e8935a",
  "#6fa8dc",
  "#8bc48a",
  "#e0b356",
  "#c58ad0",
  "#5ac7c0",
];
// Re-skin card accents (composition teal, kept apart from the ember cards).
const RESKIN_BORDER = "rgba(90,199,192,0.4)";
const RESKIN_BG = "#14201f";
const RESKIN_KEPT_BORDER = "#2a4a48";
const RESKIN_REPLACE_BORDER = "rgba(90,199,192,0.25)";
const RESKIN_BTN_BORDER = "rgba(90,199,192,0.45)";
const RESKIN_BTN_BG = "rgba(90,199,192,0.10)";

const thumbUrl = (id: number) => `/api/media/${id}/thumb`;

/** Quality-band colour: ≥85 green, ≥70 amber, else red (null → faint). */
function qColor(score: number | null): string {
  if (score == null) return colors.textFaint;
  if (score >= 85) return colors.ok;
  if (score >= 70) return colors.warn;
  return colors.danger;
}

type Nature = "near" | "resemble" | "composition";

interface Node {
  id: number;
  name: string;
  quality: number | null;
  ax: number;
  ay: number;
}

/** A raw pick pair with both fused cosines, before classification. */
interface Pair {
  a: number;
  b: number;
  dino: number;
  depth: number;
}

/** A pair shown at the current threshold, tagged by its carrying signal. */
interface ShownEdge extends Pair {
  nature: Nature;
}

interface Group {
  key: number;
  members: Node[];
  meanSim: number;
}

/**
 * Classify one pair at a threshold — appearance first, then composition.
 * DINOv2 near-dups and resemblance carry regardless of the composition
 * toggle; a composition (re-skin) edge is exactly a pair DINOv2 did *not*
 * consider resembling but the depth signal did, and only when the toggle is
 * on. Returns null when neither signal reaches the threshold.
 */
function classify(
  pair: Pair,
  sim: number,
  compOn: boolean,
): Nature | null {
  if (pair.dino >= NEAR_DUP) return "near";
  if (pair.dino >= sim) return "resemble";
  if (compOn && COMP_W * pair.depth >= sim) return "composition";
  return null;
}

export function AutoBuildProximity({
  picks,
  edges: allEdges,
  sim,
  onSimChange,
  hover,
  onHover,
  selId,
  onSelect,
  onAutoReplace,
}: {
  picks: AutobuildPick[];
  /** The fused edge list: `[a, b, dinoSim, depthSim]` from the payload. */
  edges: [number, number, number, number][];
  sim: number;
  onSimChange: (value: number) => void;
  hover: number | null;
  onHover: (id: number | null) => void;
  selId: number | null;
  onSelect: (id: number) => void;
  onAutoReplace: (ids: number[]) => void;
}) {
  // The composition signal is opt-in per view: off reverts to pure DINOv2
  // (no teal links, no re-skin groups, the layout un-pulled).
  const [compOn, setCompOn] = useState(true);

  // Nodes: the picks that carry a projection point. Their anchor is the same
  // DINOv2 2D position the coverage map plots, so the two maps stay aligned.
  const nodes = useMemo<Node[]>(
    () =>
      picks
        .filter((pick) => pick.xy != null)
        .map((pick) => ({
          id: pick.media_id,
          name: pick.name,
          quality: pick.quality,
          ax: (pick.xy as [number, number])[0],
          ay: (pick.xy as [number, number])[1],
        })),
    [picks],
  );
  const nodeById = useMemo(
    () => new Map(nodes.map((node) => [node.id, node])),
    [nodes],
  );
  const idsKey = useMemo(
    () =>
      nodes
        .map((node) => node.id)
        .sort((a, b) => a - b)
        .join(","),
    [nodes],
  );

  // Every payload pair whose endpoints are both present, parsed once.
  const pairs = useMemo<Pair[]>(
    () =>
      allEdges
        .filter(([a, b]) => nodeById.has(a) && nodeById.has(b))
        .map(([a, b, dino, depth]) => ({ a, b, dino, depth })),
    [allEdges, nodeById],
  );

  // Strong re-skin pairs feed the layout pull. Threshold-independent (raw
  // cosines), and only when composition is on — so the layout depends on the
  // picked set and the toggle alone, never on the slider.
  const strongPairs = useMemo(
    () =>
      compOn
        ? pairs.filter(
            (pair) =>
              pair.depth >= COMP_STRONG_DEPTH && pair.dino < COMP_DINO_FAR,
          )
        : [],
    [pairs, compOn],
  );

  // Layout is cached by the set of picked ids and the composition toggle — it
  // never recomputes on a threshold change or a hover.
  const layout = useMemo(
    () => relax(nodes, strongPairs, compOn),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [idsKey, compOn],
  );

  // Edges shown at the current threshold, tagged by their carrying signal.
  const shown = useMemo<ShownEdge[]>(
    () =>
      pairs.flatMap((pair) => {
        const nature = classify(pair, sim, compOn);
        return nature ? [{ ...pair, nature }] : [];
      }),
    [pairs, sim, compOn],
  );

  // Per-node degree, near-dup degree, re-skin degree and adjacency.
  const { degree, nearDupDegree, reskinDegree, adjacency } = useMemo(() => {
    const deg = new Map<number, number>();
    const near = new Map<number, number>();
    const reskin = new Map<number, number>();
    const adj = new Map<number, Set<number>>();
    for (const node of nodes) {
      deg.set(node.id, 0);
      near.set(node.id, 0);
      reskin.set(node.id, 0);
      adj.set(node.id, new Set());
    }
    for (const edge of shown) {
      deg.set(edge.a, (deg.get(edge.a) ?? 0) + 1);
      deg.set(edge.b, (deg.get(edge.b) ?? 0) + 1);
      adj.get(edge.a)?.add(edge.b);
      adj.get(edge.b)?.add(edge.a);
      if (edge.nature === "near") {
        near.set(edge.a, (near.get(edge.a) ?? 0) + 1);
        near.set(edge.b, (near.get(edge.b) ?? 0) + 1);
      }
      if (edge.nature === "composition") {
        reskin.set(edge.a, (reskin.get(edge.a) ?? 0) + 1);
        reskin.set(edge.b, (reskin.get(edge.b) ?? 0) + 1);
      }
    }
    return {
      degree: deg,
      nearDupDegree: near,
      reskinDegree: reskin,
      adjacency: adj,
    };
  }, [nodes, shown]);

  // Redundant groups, kept apart so a card never mislabels: appearance groups
  // over the near + resemble edges (mean DINOv2 similarity), re-skin groups
  // over the composition edges alone (mean depth similarity).
  const appearanceGroups = useMemo(
    () =>
      components(
        nodes,
        shown.filter((edge) => edge.nature !== "composition"),
        nodeById,
        (edge) => edge.dino,
      ),
    [nodes, shown, nodeById],
  );
  const reskinGroups = useMemo(
    () =>
      components(
        nodes,
        shown.filter((edge) => edge.nature === "composition"),
        nodeById,
        (edge) => edge.depth,
      ),
    [nodes, shown, nodeById],
  );

  const links = shown.length;
  const nearDupPairs = shown.filter((edge) => edge.nature === "near").length;
  const compPairs = shown.filter(
    (edge) => edge.nature === "composition",
  ).length;
  const groupsCount = appearanceGroups.length + reskinGroups.length;
  const redundant = appearanceGroups.reduce(
    (sum, group) => sum + group.members.length - 1,
    0,
  );

  const hoveredNeighbors = hover != null ? adjacency.get(hover) : undefined;

  return (
    <div style={wrap}>
      {/* ---- controls + stats ---- */}
      <div style={controlsBar}>
        <div style={{ minWidth: 190, flex: 1 }}>
          <div style={miniLabel}>Resemblance threshold</div>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <span style={endpoint}>0.70</span>
            <input
              type="range"
              min={0.7}
              max={0.98}
              step={0.01}
              value={sim}
              onChange={(event) => onSimChange(Number(event.target.value))}
              style={{ flex: 1, accentColor: colors.accent }}
            />
            <span style={endpoint}>0.98</span>
            <span style={simValue}>{sim.toFixed(2)}</span>
          </div>
        </div>
        <CompToggle on={compOn} onChange={setCompOn} />
        <div style={divider} />
        <Stat value={links} label="links" />
        <Stat
          value={nearDupPairs}
          label="near-dup pairs"
          color={nearDupPairs > 0 ? colors.danger : colors.ok}
        />
        <Stat value={groupsCount} label="redundant groups" />
        <Stat
          value={compPairs}
          label="composition pairs"
          color={compPairs > 0 ? COMP_LINK : colors.textMuted}
        />
      </div>

      {/* ---- advice ---- */}
      <div style={advice}>
        <span style={{ color: nearDupPairs > 0 ? colors.danger : colors.ok }}>
          ≈
        </span>{" "}
        {redundant > 0
          ? `Pruning ${redundant} redundant image(s) would sharpen the LoRA — near-identical frames get memorised instead of teaching the concept.`
          : "No redundant clusters at this threshold — the selection generalises well."}
        {compPairs > 0 && (
          <>
            {" "}
            <span style={{ color: COMP_LINK }}>
              {compPairs} of the links are composition re-skins DINOv2 alone
              would miss.
            </span>
          </>
        )}
      </div>

      {/* ---- graph (preview is a sibling so it can overflow) ---- */}
      <div style={{ position: "relative" }}>
        <div style={graphBox}>
          <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} style={{ display: "block" }}>
            {shown.map((edge) => {
              const pa = layout.get(edge.a);
              const pb = layout.get(edge.b);
              if (!pa || !pb) return null;
              const endpointHover =
                hover === edge.a || hover === edge.b;
              if (edge.nature === "composition") {
                const opacity =
                  hover == null ? 0.24 : endpointHover ? 0.95 : 0.05;
                return (
                  <line
                    key={`e${edge.a}-${edge.b}`}
                    x1={pa.x}
                    y1={pa.y}
                    x2={pb.x}
                    y2={pb.y}
                    stroke={COMP_LINK}
                    strokeWidth={0.5}
                    strokeDasharray="1.4 1.2"
                    opacity={opacity}
                  />
                );
              }
              const near = edge.nature === "near";
              const base = near ? 0.9 : 0.42;
              const opacity =
                hover == null ? base : endpointHover ? base : 0.05;
              return (
                <line
                  key={`e${edge.a}-${edge.b}`}
                  x1={pa.x}
                  y1={pa.y}
                  x2={pb.x}
                  y2={pb.y}
                  stroke={near ? colors.danger : RESEMBLE_LINK}
                  strokeWidth={near ? 0.85 : 0.45}
                  opacity={opacity}
                />
              );
            })}
            {nodes.map((node) => {
              const point = layout.get(node.id);
              if (!point) return null;
              const deg = degree.get(node.id) ?? 0;
              const active = selId === node.id;
              const hovered = hover === node.id;
              const isNearDup = (nearDupDegree.get(node.id) ?? 0) > 0;
              const isReskin = (reskinDegree.get(node.id) ?? 0) > 0;
              let radius = 1.5 + Math.min(2.4, deg * 0.4);
              if (active || hovered) radius += 0.8;
              const opacity =
                hover == null
                  ? deg > 0
                    ? 1
                    : 0.5
                  : hovered || hoveredNeighbors?.has(node.id)
                    ? 1
                    : 0.16;
              return (
                <g
                  key={`n${node.id}`}
                  style={{ cursor: "pointer" }}
                  opacity={opacity}
                  onMouseEnter={() => onHover(node.id)}
                  onMouseLeave={() =>
                    onHover(hover === node.id ? null : hover)
                  }
                  onClick={() => onSelect(node.id)}
                >
                  {/* Ring priority: selection > near-dup > re-skin. */}
                  {active ? (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={radius + 1.6}
                      fill="none"
                      stroke={colors.accent}
                      strokeWidth={0.6}
                      strokeDasharray="1.6 1.2"
                    />
                  ) : isNearDup ? (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={radius + 1.1}
                      fill="none"
                      stroke={colors.danger}
                      strokeWidth={0.5}
                    />
                  ) : (
                    isReskin && (
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r={radius + 1.1}
                        fill="none"
                        stroke={COMP_LINK}
                        strokeWidth={0.5}
                        strokeDasharray="0.9 0.9"
                      />
                    )
                  )}
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={radius}
                    fill={qColor(node.quality)}
                    stroke={active ? colors.accent : NODE_STROKE}
                    strokeWidth={active ? 0.7 : 0.4}
                  />
                </g>
              );
            })}
          </svg>
        </div>
        {hover != null &&
          (() => {
            const node = nodeById.get(hover);
            const point = layout.get(hover);
            if (!node || !point) return null;
            return (
              <HoverPreview
                node={node}
                degree={degree.get(hover) ?? 0}
                nearDup={(nearDupDegree.get(hover) ?? 0) > 0}
                reskins={reskinDegree.get(hover) ?? 0}
                active={selId === hover}
                point={point}
              />
            );
          })()}
      </div>

      {/* ---- legend ---- */}
      <div style={legend}>
        node size = number of look-alikes · red ring / red link =
        near-duplicate (≥ 0.92) · blue link = resemblance ·{" "}
        <span style={{ color: COMP_LINK }}>
          teal dashed link / ring = same composition, different style (re-skin
          — DINOv2 sees them far; the layout pulls them together)
        </span>{" "}
        · node fill = quality score · hover to preview &amp; isolate a
        node&apos;s links · click to select it, then swap it on the coverage
        map →
      </div>

      {/* ---- redundant groups ---- */}
      {appearanceGroups.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={miniLabel}>
              Redundant groups — keep one, replace the look-alikes
            </span>
            <span style={{ ...legend, flex: 1 }}>
              dataset size is fixed — replacing swaps a look-alike out for the
              next-best diverse pick, the count stays at target
            </span>
          </div>
          <div style={groupGrid}>
            {appearanceGroups.map((group, index) => (
              <GroupCard
                key={group.key}
                group={group}
                color={GROUP_PALETTE[index % GROUP_PALETTE.length]}
                selId={selId}
                onSelect={onSelect}
                onAutoReplace={onAutoReplace}
              />
            ))}
          </div>
        </div>
      )}

      {/* ---- composition re-skins (a separate section) ---- */}
      {reskinGroups.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={{ ...miniLabel, color: COMP_LINK }}>
              ◇ Composition re-skins — same framing, different style
            </span>
            <span style={{ ...legend, flex: 1 }}>
              Depth-Anything V2 · DINOv2 saw these as different — same
              pose/framing still teaches the LoRA to memorise the layout
            </span>
          </div>
          <div style={groupGrid}>
            {reskinGroups.map((group) => (
              <ReskinCard
                key={group.key}
                group={group}
                selId={selId}
                onSelect={onSelect}
                onAutoReplace={onAutoReplace}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** The `composition links` pill: teal when on, grey when off. */
function CompToggle({
  on,
  onChange,
}: {
  on: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      title="Fuse Depth-Anything V2 composition depth to surface re-skins"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        border: `1px solid ${on ? "rgba(90,199,192,0.5)" : colors.borderControl}`,
        background: on ? "rgba(90,199,192,0.10)" : "transparent",
        borderRadius: 7,
        padding: "6px 9px",
        cursor: "pointer",
      }}
    >
      <span
        style={{
          width: 22,
          height: 12,
          borderRadius: 6,
          background: on ? COMP_LINK : "#2c2f38",
          position: "relative",
          transition: "background 0.15s",
        }}
      >
        <span
          style={{
            position: "absolute",
            top: 1.5,
            left: on ? 11.5 : 1.5,
            width: 9,
            height: 9,
            borderRadius: 5,
            background: "#0f1013",
            transition: "left 0.15s",
          }}
        />
      </span>
      <span
        style={{
          fontSize: 10,
          fontWeight: 600,
          color: on ? COMP_LINK : colors.textMuted,
        }}
      >
        composition links
      </span>
    </button>
  );
}

/** One stat column: a big mono number over a small caption. */
function Stat({
  value,
  label,
  color,
}: {
  value: number;
  label: string;
  color?: string;
}) {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 16,
          fontWeight: 700,
          color: color ?? colors.text,
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: 9, color: colors.textMuted }}>{label}</div>
    </div>
  );
}

/** The large hover card, pinned to the side of the node so it can overflow. */
function HoverPreview({
  node,
  degree,
  nearDup,
  reskins,
  active,
  point,
}: {
  node: Node;
  degree: number;
  nearDup: boolean;
  reskins: number;
  active: boolean;
  point: { x: number; y: number };
}) {
  const left = point.x;
  const top = Math.max(16, Math.min(84, (point.y / VIEW_H) * 100));
  const toRight = point.x <= 52;
  const transform = toRight
    ? "translate(14px, -50%)"
    : "translate(calc(-100% - 14px), -50%)";
  const role = active
    ? {
        text: "◉ active — click a grey point on the coverage map to swap it in",
        color: colors.accent,
      }
    : nearDup
      ? {
          text: "⚠ near-duplicate — click to select, then swap it out on the map",
          color: colors.danger,
        }
      : {
          text: "click to select · swap it via the coverage map",
          color: colors.textMuted,
        };
  return (
    <div
      style={{
        ...previewCard,
        left: `${left}%`,
        top: `${top}%`,
        transform,
      }}
    >
      <div
        style={{
          aspectRatio: "4 / 3",
          borderRadius: 6,
          overflow: "hidden",
          background: colors.app,
        }}
      >
        <img
          src={thumbUrl(node.id)}
          alt=""
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </div>
      <div style={previewName}>{node.name}</div>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 10,
          color: qColor(node.quality),
        }}
      >
        Q {node.quality == null ? "—" : node.quality.toFixed(0)} · {degree}{" "}
        look-alike(s)
      </div>
      {reskins > 0 && (
        <div style={{ fontFamily: font.mono, fontSize: 9, color: COMP_LINK }}>
          ◇ {reskins} composition re-skin(s) — same framing, different style
        </div>
      )}
      <div style={{ fontFamily: font.mono, fontSize: 9.5, color: role.color }}>
        {role.text}
      </div>
    </div>
  );
}

/** One redundant-group card: keep the best, replace the look-alikes. */
function GroupCard({
  group,
  color,
  selId,
  onSelect,
  onAutoReplace,
}: {
  group: Group;
  color: string;
  selId: number | null;
  onSelect: (id: number) => void;
  onAutoReplace: (ids: number[]) => void;
}) {
  const count = group.members.length;
  const best = group.members[0];
  const replaceable = group.members.slice(1).map((member) => member.id);
  return (
    <div style={groupCard}>
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span
          style={{ width: 8, height: 8, borderRadius: 3, background: color }}
        />
        <span style={{ fontSize: 11.5, fontWeight: 700 }}>
          {count} near-identical
        </span>
      </div>
      <div style={{ fontFamily: font.mono, fontSize: 9, color: colors.textFaint }}>
        ~{group.meanSim.toFixed(2)} similarity · keep 1, drop {count - 1}
      </div>
      <MemberStrip
        members={group.members}
        best={best}
        selId={selId}
        keptBorder="#2a4a2e"
        replaceBorder="#3a3222"
        onSelect={onSelect}
      />
      <button
        type="button"
        onClick={() => onAutoReplace(replaceable)}
        style={replaceButton}
      >
        ⇄ Auto-replace {count - 1} look-alike(s)
      </button>
    </div>
  );
}

/** One composition re-skin card: teal-accented, kept apart from look-alikes. */
function ReskinCard({
  group,
  selId,
  onSelect,
  onAutoReplace,
}: {
  group: Group;
  selId: number | null;
  onSelect: (id: number) => void;
  onAutoReplace: (ids: number[]) => void;
}) {
  const count = group.members.length;
  const best = group.members[0];
  const replaceable = group.members.slice(1).map((member) => member.id);
  return (
    <div style={reskinCard}>
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 3,
            background: COMP_LINK,
          }}
        />
        <span style={{ fontSize: 11.5, fontWeight: 700 }}>
          {count} re-skins
        </span>
      </div>
      <div style={{ fontFamily: font.mono, fontSize: 9, color: colors.textFaint }}>
        ~{group.meanSim.toFixed(2)} composition · different style · keep 1,
        drop {count - 1}
      </div>
      <MemberStrip
        members={group.members}
        best={best}
        selId={selId}
        keptBorder={RESKIN_KEPT_BORDER}
        replaceBorder={RESKIN_REPLACE_BORDER}
        onSelect={onSelect}
      />
      <button
        type="button"
        onClick={() => onAutoReplace(replaceable)}
        style={reskinButton}
      >
        ⇄ Auto-replace {count - 1} re-skin(s)
      </button>
    </div>
  );
}

/** The keep/replace thumbnail strip shared by both group cards. */
function MemberStrip({
  members,
  best,
  selId,
  keptBorder,
  replaceBorder,
  onSelect,
}: {
  members: Node[];
  best: Node;
  selId: number | null;
  keptBorder: string;
  replaceBorder: string;
  onSelect: (id: number) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
      {members.map((member) => {
        const isBest = member.id === best.id;
        const selected = selId === member.id;
        return (
          <div
            key={member.id}
            onClick={() => onSelect(member.id)}
            title={member.name}
            style={{ width: 54, cursor: "pointer" }}
          >
            <div
              style={{
                position: "relative",
                height: 41,
                borderRadius: 5,
                overflow: "hidden",
                border: `1px solid ${
                  selected ? colors.accent : isBest ? keptBorder : replaceBorder
                }`,
                boxShadow: selected ? `0 0 0 2px ${SEL_GLOW}` : "none",
              }}
            >
              <img
                src={thumbUrl(member.id)}
                alt=""
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
              <span
                style={{ ...qualityBadge, color: qColor(member.quality) }}
              >
                {member.quality == null ? "—" : member.quality.toFixed(0)}
              </span>
            </div>
            <div
              style={{
                fontFamily: font.mono,
                fontSize: 7.5,
                textAlign: "center",
                marginTop: 2,
                color: isBest ? colors.ok : colors.warn,
              }}
            >
              {isBest ? "✓ keep" : "⇄ replace"}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// --- geometry -------------------------------------------------------------

/**
 * Relax the node layout: anchor spring + short-range repulsion, plus — when
 * composition is on — an attraction along strong re-skin pairs so they
 * cluster instead of being wired by long crossing lines. The anchor softens
 * when composition is on so the pull has room to move nodes.
 */
function relax(
  nodes: Node[],
  strongPairs: Pair[],
  compOn: boolean,
): Map<number, { x: number; y: number }> {
  const pos = nodes.map((node) => ({ x: node.ax, y: node.ay }));
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const anchor = compOn ? ANCHOR_PULL_COMP : ANCHOR_PULL;
  const iterations = compOn ? LAYOUT_ITERATIONS_COMP : LAYOUT_ITERATIONS;
  for (let iter = 0; iter < iterations; iter++) {
    const fx = new Array(nodes.length).fill(0);
    const fy = new Array(nodes.length).fill(0);
    for (let i = 0; i < nodes.length; i++) {
      fx[i] += (nodes[i].ax - pos[i].x) * anchor;
      fy[i] += (nodes[i].ay - pos[i].y) * anchor;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const d = Math.hypot(dx, dy) || 0.001;
        if (d < REPULSE_RADIUS) {
          const f = ((REPULSE_RADIUS - d) / d) * REPULSE_STRENGTH;
          fx[i] += dx * f;
          fy[i] += dy * f;
          fx[j] -= dx * f;
          fy[j] -= dy * f;
        }
      }
    }
    for (let i = 0; i < nodes.length; i++) {
      pos[i].x = clamp(pos[i].x + fx[i], CLAMP_X[0], CLAMP_X[1]);
      pos[i].y = clamp(pos[i].y + fy[i], CLAMP_Y[0], CLAMP_Y[1]);
    }
    // Pull each strong re-skin pair together while they sit far apart.
    for (const pair of strongPairs) {
      const ia = index.get(pair.a);
      const ib = index.get(pair.b);
      if (ia == null || ib == null) continue;
      const dx = pos[ib].x - pos[ia].x;
      const dy = pos[ib].y - pos[ia].y;
      const d = Math.hypot(dx, dy) || 0.001;
      if (d <= COMP_ATTRACT_TARGET) continue;
      const f = Math.min(
        COMP_ATTRACT_CAP,
        ((d - COMP_ATTRACT_TARGET) / d) * COMP_ATTRACT_GAIN,
      );
      pos[ia].x += dx * f;
      pos[ia].y += dy * f;
      pos[ib].x -= dx * f;
      pos[ib].y -= dy * f;
    }
  }
  const map = new Map<number, { x: number; y: number }>();
  nodes.forEach((node, i) => map.set(node.id, pos[i]));
  return map;
}

/**
 * Connected components of size ≥ 2 over one edge set, each sorted by quality
 * descending; `meanSim` is the mean of `valueOf` over the group's intra
 * edges (DINOv2 for appearance groups, depth for re-skin groups).
 */
function components(
  nodes: Node[],
  edges: ShownEdge[],
  nodeById: Map<number, Node>,
  valueOf: (edge: ShownEdge) => number,
): Group[] {
  const parent = new Map<number, number>();
  for (const node of nodes) parent.set(node.id, node.id);
  const find = (x: number): number => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root) as number;
    let cursor = x;
    while (parent.get(cursor) !== root) {
      const next = parent.get(cursor) as number;
      parent.set(cursor, root);
      cursor = next;
    }
    return root;
  };
  for (const edge of edges) {
    const ra = find(edge.a);
    const rb = find(edge.b);
    if (ra !== rb) parent.set(ra, rb);
  }
  const buckets = new Map<number, number[]>();
  for (const node of nodes) {
    const root = find(node.id);
    const bucket = buckets.get(root);
    if (bucket) bucket.push(node.id);
    else buckets.set(root, [node.id]);
  }
  const groups: Group[] = [];
  for (const [root, ids] of buckets) {
    if (ids.length < 2) continue;
    const members = ids
      .map((id) => nodeById.get(id))
      .filter((node): node is Node => Boolean(node))
      .sort((a, b) => (b.quality ?? -1) - (a.quality ?? -1));
    const idSet = new Set(ids);
    const intra = edges.filter(
      (edge) => idSet.has(edge.a) && idSet.has(edge.b),
    );
    const meanSim = intra.length
      ? intra.reduce((sum, edge) => sum + valueOf(edge), 0) / intra.length
      : 0;
    groups.push({ key: root, members, meanSim });
  }
  return groups.sort((a, b) => b.members.length - a.members.length);
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

// --- styles ---------------------------------------------------------------

const wrap = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
} as const;

const controlsBar = {
  display: "flex",
  alignItems: "center",
  gap: 16,
  flexWrap: "wrap",
  border: `1px solid ${colors.border}`,
  background: colors.toolbar,
  borderRadius: 8,
  padding: "9px 13px",
} as const;

const miniLabel = {
  fontSize: 9.5,
  letterSpacing: 0.4,
  textTransform: "uppercase",
  color: colors.textMuted,
  marginBottom: 4,
} as const;

const endpoint = {
  fontFamily: font.mono,
  fontSize: 9,
  color: colors.textFaint,
} as const;

const simValue = {
  fontFamily: font.mono,
  fontSize: 15,
  fontWeight: 700,
  color: colors.text,
  minWidth: 40,
  textAlign: "right",
} as const;

const divider = {
  width: 1,
  height: 30,
  background: colors.borderControl,
} as const;

const advice = {
  border: `1px solid ${colors.borderControl}`,
  background: colors.card,
  borderRadius: 7,
  padding: "8px 11px",
  fontSize: 11,
  color: colors.textMutedAlt,
  lineHeight: 1.45,
} as const;

const graphBox = {
  border: `1px solid ${colors.borderControl}`,
  background: colors.app,
  borderRadius: 8,
  overflow: "hidden",
} as const;

const legend = {
  fontFamily: font.mono,
  fontSize: 9,
  color: colors.textFaint,
  lineHeight: 1.5,
} as const;

const previewCard = {
  position: "absolute",
  width: 222,
  padding: 7,
  borderRadius: 9,
  background: "rgba(15,16,19,0.96)",
  border: "1px solid #3a3d47",
  boxShadow: "0 14px 44px rgba(0,0,0,0.62)",
  zIndex: 20,
  pointerEvents: "none",
} as const;

const previewName = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.text,
  marginTop: 5,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
} as const;

const groupGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(236px, 1fr))",
  gap: 9,
} as const;

const groupCard = {
  border: `1px solid ${colors.borderControl}`,
  background: colors.card,
  borderRadius: 8,
  padding: 10,
  display: "flex",
  flexDirection: "column",
  gap: 8,
} as const;

const reskinCard = {
  ...groupCard,
  border: `1px solid ${RESKIN_BORDER}`,
  background: RESKIN_BG,
} as const;

const qualityBadge = {
  position: "absolute",
  top: 2,
  right: 2,
  fontFamily: font.mono,
  fontSize: 8,
  padding: "0 3px",
  borderRadius: 3,
  background: "rgba(15,16,19,0.8)",
} as const;

const replaceButton = {
  width: "100%",
  border: `1px solid ${colors.accentBorder}`,
  background: colors.accentTintAlt,
  color: colors.accent,
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 10.5,
  cursor: "pointer",
} as const;

const reskinButton = {
  width: "100%",
  border: `1px solid ${RESKIN_BTN_BORDER}`,
  background: RESKIN_BTN_BG,
  color: COMP_LINK,
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 10.5,
  cursor: "pointer",
} as const;
