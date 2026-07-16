/** Renders a word-level caption diff (red struck deletions, green additions). */

import type { CSSProperties } from "react";
import { colors } from "../../design/tokens";
import { wordDiff } from "../../lib/review";

export function DiffText({
  before,
  after,
  clamp,
  style,
}: {
  before: string;
  after: string;
  /** Max lines before ellipsis (the queue clamps to 2; the wizard is free). */
  clamp?: number;
  style?: CSSProperties;
}) {
  const segments = wordDiff(before, after);
  const clampStyle: CSSProperties = clamp
    ? {
        display: "-webkit-box",
        WebkitLineClamp: clamp,
        WebkitBoxOrient: "vertical",
        overflow: "hidden",
      }
    : {};
  return (
    <span style={{ lineHeight: 1.55, ...clampStyle, ...style }}>
      {segments.map((seg, index) => {
        if (seg.op === "same") return <span key={index}>{seg.text}</span>;
        const isDel = seg.op === "del";
        return (
          <span
            key={index}
            style={{
              color: isDel ? colors.danger : colors.ok,
              background: isDel ? "#2a1715" : "#152a17",
              textDecoration: isDel ? "line-through" : "none",
              borderRadius: 3,
              padding: "0 1px",
            }}
          >
            {seg.text}
          </span>
        );
      })}
    </span>
  );
}
