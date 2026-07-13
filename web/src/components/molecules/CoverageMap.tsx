/**
 * The composer's corpus map: a PCA projection of the DINOv2 vectors.
 *
 * Grey dots are the filtered candidates, lighter dots the dataset, orange
 * dots the current selection, and the dashed circles the regions of the
 * corpus the dataset does not cover yet. Hovering a candidate card rings
 * its dot, which is the whole point of the panel: it tells the user *where*
 * an image would land before they add it.
 */

import { colors } from "../../design/tokens";
import type { ComposeZone } from "../../api/types";

type Point = [number, number];

export function CoverageMap({
  width,
  height,
  candidates,
  dataset,
  selected,
  zones,
  zonesActive,
  hovered,
}: {
  width: number;
  height: number;
  candidates: Point[];
  dataset: Point[];
  selected: Point[];
  zones: ComposeZone[];
  /** The "Fill the gaps" mode paints the zones at full strength. */
  zonesActive: boolean;
  hovered: Point | null;
}) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", display: "block" }}
    >
      {zones.map((zone, index) => (
        <circle
          key={`z${index}`}
          cx={zone.x}
          cy={zone.y}
          r={zone.r}
          fill="rgba(111,168,220,0.06)"
          stroke={colors.info}
          strokeWidth={0.4}
          strokeDasharray="2 2"
          opacity={zonesActive ? 1 : 0.45}
        />
      ))}
      {candidates.map((point, index) => (
        <circle
          key={`c${index}`}
          cx={point[0]}
          cy={point[1]}
          r={1.1}
          fill="#3f434e"
        />
      ))}
      {dataset.map((point, index) => (
        <circle
          key={`d${index}`}
          cx={point[0]}
          cy={point[1]}
          r={1.5}
          fill={colors.textMuted}
        />
      ))}
      {selected.map((point, index) => (
        <circle
          key={`s${index}`}
          cx={point[0]}
          cy={point[1]}
          r={2}
          fill={colors.accent}
        />
      ))}
      {hovered && (
        <circle
          cx={hovered[0]}
          cy={hovered[1]}
          r={3.6}
          fill="none"
          stroke={colors.accent}
          strokeWidth={0.7}
        />
      )}
    </svg>
  );
}
