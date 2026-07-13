/** Media-tab library card: fav, quality, dims, zoom chip and tag chips. */

import type { LibraryCard } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { Dot } from "../atoms";
import { QualityBadge } from "./index";
import { WatermarkBadge } from "./WatermarkBadge";

export function LibraryMediaCard({
  card,
  focused,
  onFocus,
  onZoom,
  onToggleFav,
}: {
  card: LibraryCard;
  focused: boolean;
  onFocus: () => void;
  onZoom: () => void;
  onToggleFav: () => void;
}) {
  return (
    <div
      onClick={onFocus}
      style={{
        borderRadius: 8,
        overflow: "hidden",
        border: `1px solid ${focused ? colors.accent : colors.border}`,
        background: colors.card,
        cursor: "pointer",
      }}
    >
      <div
        style={{
          position: "relative",
          aspectRatio: "1 / 1",
          background: colors.raised,
        }}
      >
        <img
          src={card.thumb}
          alt={card.name}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        <span
          onClick={(event) => {
            event.stopPropagation();
            onToggleFav();
          }}
          style={{
            position: "absolute",
            top: 5,
            left: 6,
            cursor: "pointer",
            color: card.favorite ? colors.fav : "#fff",
            textShadow: "0 1px 2px rgba(0,0,0,0.6)",
          }}
        >
          {card.favorite ? "♥" : "♡"}
        </span>
        <span style={{ position: "absolute", top: 5, right: 5 }}>
          <QualityBadge score={card.quality} />
        </span>
        {card.wm_status && (
          <span style={{ position: "absolute", top: 24, left: 6 }}>
            <WatermarkBadge status={card.wm_status} />
          </span>
        )}
        <span
          onClick={(event) => {
            event.stopPropagation();
            onZoom();
          }}
          title="Zoom"
          style={{
            position: "absolute",
            bottom: 5,
            left: 6,
            cursor: "zoom-in",
            fontSize: 11,
            color: "#fff",
            background: "rgba(0,0,0,0.5)",
            borderRadius: 4,
            padding: "0 4px",
          }}
        >
          🔍
        </span>
        {card.width != null && (
          <span
            style={{
              position: "absolute",
              bottom: 5,
              right: 5,
              fontSize: 9.5,
              fontFamily: font.mono,
              color: "#fff",
              background: "rgba(0,0,0,0.5)",
              padding: "1px 4px",
              borderRadius: 4,
            }}
          >
            {card.width}×{card.height}
          </span>
        )}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          padding: "5px 6px",
          overflow: "hidden",
        }}
      >
        {card.tags.map((tag) => (
          <span
            key={tag.name}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 3,
              fontSize: 10,
              color: colors.textMuted,
              maxWidth: 60,
              overflow: "hidden",
              whiteSpace: "nowrap",
              textOverflow: "ellipsis",
            }}
          >
            <Dot color={tag.color} size={6} />
            {tag.name}
          </span>
        ))}
        {card.tag_count > 3 && (
          <span style={{ fontSize: 10, color: colors.accent }}>
            +{card.tag_count - 3}
          </span>
        )}
      </div>
    </div>
  );
}
