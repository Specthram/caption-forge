/** The "◪ patched / review / watermark / excluded" chip on media cards. */

import type { WatermarkStatus } from "../../api/types";
import { font, watermarkStatus } from "../../design/tokens";

export function WatermarkBadge({
  status,
  compact = false,
}: {
  status: WatermarkStatus | null | undefined;
  compact?: boolean;
}) {
  if (!status) return null;
  const { color, label } = watermarkStatus(status);
  return (
    <span
      title={`Watermark — ${label}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 3,
        fontFamily: font.mono,
        fontSize: 9,
        lineHeight: 1.4,
        color,
        background: "rgba(15,16,19,0.78)",
        borderRadius: 4,
        padding: compact ? "0 3px" : "1px 4px",
      }}
    >
      ◪{compact ? "" : ` ${label}`}
    </span>
  );
}
