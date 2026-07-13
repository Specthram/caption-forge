/** Caption-grid media card: thumb, selection, badges, caption preview. */

import type { CaptionCard } from "../../api/types";
import { colors, deployColor, font } from "../../design/tokens";
import { Dot } from "../atoms";
import { QualityBadge, ReviewBadge } from "./index";

export function MediaCard({
  card,
  selected,
  focused,
  locked,
  onSelect,
  onFocus,
}: {
  card: CaptionCard;
  selected: boolean;
  focused: boolean;
  locked?: boolean;
  onSelect: () => void;
  onFocus: () => void;
}) {
  const ring = selected
    ? colors.accent
    : focused
      ? colors.borderHover
      : colors.border;
  return (
    <div
      onClick={onFocus}
      style={{
        display: "flex",
        flexDirection: "column",
        borderRadius: 8,
        border: `1px solid ${ring}`,
        boxShadow: selected ? `0 0 0 1px ${colors.accent}` : undefined,
        background: colors.card,
        overflow: "hidden",
        cursor: "pointer",
      }}
    >
      <div
        style={{
          position: "relative",
          aspectRatio: "4 / 3",
          background: colors.raised,
        }}
      >
        <img
          src={card.thumb}
          alt={card.name}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        <input
          type="checkbox"
          checked={selected}
          onChange={onSelect}
          onClick={(event) => event.stopPropagation()}
          style={{ position: "absolute", top: 6, left: 6, width: 17, height: 17 }}
        />
        <span style={{ position: "absolute", top: 6, right: 6 }}>
          <QualityBadge score={card.quality} />
        </span>
        {card.is_video && (
          <span
            style={{
              position: "absolute",
              bottom: 6,
              left: 6,
              fontSize: 10,
              fontFamily: font.mono,
              color: "#fff",
              background: "rgba(0,0,0,0.5)",
              padding: "1px 5px",
              borderRadius: 4,
            }}
          >
            ▶
          </span>
        )}
        {locked && (
          <span
            title="Locked — excluded from generation"
            style={{
              position: "absolute",
              bottom: 6,
              right: 6,
              fontSize: 11,
              color: colors.warn,
            }}
          >
            🔒
          </span>
        )}
      </div>
      <div style={{ padding: "6px 8px" }}>
        <div
          style={{
            fontSize: 11,
            lineHeight: 1.35,
            color: card.caption ? colors.textSecondary : colors.textFaint,
            fontStyle: card.caption ? "normal" : "italic",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
            minHeight: 30,
          }}
        >
          {card.caption || "No caption yet"}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginTop: 6,
            fontSize: 10,
            fontFamily: font.mono,
            color: colors.textFaint,
          }}
        >
          <ReviewBadge label={card.review} issues={card.review_issues} />
          <span>.{card.ext}</span>
          <span>r{card.revisions}</span>
          <span style={{ flex: 1 }} />
          <Dot color={deployColor(null)} size={7} />
        </div>
      </div>
    </div>
  );
}
