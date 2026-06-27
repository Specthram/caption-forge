/** Compact media card for the Datasets / picker grids. */

import type { MediaGridCard } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { QualityBadge } from "./index";

export function GridCard({
  card,
  selected,
  focused,
  onClick,
  onRemove,
  onCrop,
}: {
  card: MediaGridCard;
  selected?: boolean;
  focused?: boolean;
  onClick: () => void;
  /** When set, a ✕ appears on hover to unlink the media (the body click no
   *  longer removes — it inspects). */
  onRemove?: () => void;
  /** When set, a ⌗ opens the crop overlay — on a fresh frame, or on the
   *  card's own rectangle when it already is a crop. */
  onCrop?: () => void;
}) {
  const outline = focused ? colors.info : selected ? colors.accent : colors.border;
  return (
    <div
      onClick={onClick}
      className={onRemove ? "cf-card-removable" : undefined}
      style={{
        position: "relative",
        borderRadius: 8,
        overflow: "hidden",
        border: `1px solid ${outline}`,
        boxShadow:
          focused || selected ? `0 0 0 1px ${outline}` : undefined,
        cursor: "pointer",
        background: colors.card,
      }}
    >
      <div style={{ aspectRatio: "1 / 1", background: colors.raised }}>
        <img
          src={card.thumb}
          alt={card.name}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </div>
      {onRemove && (
        <button
          type="button"
          className="cf-remove-x"
          title="Remove from dataset (unlinks only, file kept)"
          onClick={(event) => {
            event.stopPropagation();
            onRemove();
          }}
          style={{
            position: "absolute",
            top: 5,
            left: 5,
            width: 22,
            height: 22,
            borderRadius: "50%",
            border: "none",
            background: "rgba(224,108,92,0.92)",
            color: "#fff",
            fontSize: 12,
            lineHeight: 1,
            cursor: "pointer",
          }}
        >
          ✕
        </button>
      )}
      <span style={{ position: "absolute", top: 5, right: 5 }}>
        <QualityBadge score={card.quality} />
      </span>
      {card.favorite && !onRemove && (
        <span
          style={{ position: "absolute", top: 5, left: 5, color: colors.fav }}
        >
          ♥
        </span>
      )}
      {card.width != null && (
        <span
          style={{
            position: "absolute",
            bottom: 5,
            right: onCrop ? 29 : 5,
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
      {card.crop && (
        <span
          title="Virtual crop — no file until deploy"
          style={{
            position: "absolute",
            bottom: 5,
            left: 5,
            fontSize: 8.5,
            fontWeight: 600,
            fontFamily: font.mono,
            color: colors.onAccent,
            background: "rgba(232,147,90,0.92)",
            padding: "1px 4px",
            borderRadius: 4,
          }}
        >
          ⌗ {card.crop.width}×{card.crop.height}
        </span>
      )}
      {onCrop && (
        <button
          type="button"
          title={card.crop ? "Edit this crop" : "Crop this image"}
          onClick={(event) => {
            event.stopPropagation();
            onCrop();
          }}
          style={{
            position: "absolute",
            bottom: 5,
            right: 5,
            width: 20,
            height: 20,
            borderRadius: 5,
            border: "none",
            background: "rgba(15,16,19,0.82)",
            color: colors.accent,
            fontSize: 11,
            lineHeight: 1,
            cursor: "pointer",
          }}
        >
          ⌗
        </button>
      )}
    </div>
  );
}
