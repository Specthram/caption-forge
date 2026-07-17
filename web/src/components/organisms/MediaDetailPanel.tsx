/** Media-tab right detail panel (330px): meta, quality, tags, captions. */

import { useEffect, useState } from "react";
import {
  useAddTag,
  useGroundingEnabled,
  useMediaFullDetail,
  useMediaInvalidator,
  useRemoveTag,
  useTagCategories,
  useToggleFavorite,
  useWatermarkMedia,
} from "../../api/hooks";
import {
  colors,
  font,
  qualityColor,
  watermarkStatus,
} from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { Button, Dot, IconButton, Label } from "../atoms";
import { TagChip } from "../molecules";
import { CropSection } from "./CropSection";
import { TagGroundingCard } from "./GroundingCard";
import { MediaTagScoreCard } from "./MediaTagScoreCard";

export function MediaDetailPanel({
  focusKey,
  onClose,
  datasetId,
  datasetName,
  onFocusChange,
}: {
  focusKey: string | null;
  /** Clears the focus so the panel closes; when absent, no ✕ is shown. */
  onClose?: () => void;
  /** Set by the Datasets tab: a crop only makes sense inside a dataset, so
   *  the crop section appears there and nowhere else. */
  datasetId?: number;
  datasetName?: string;
  /** Refocuses the entry a crop change left behind (the original, usually). */
  onFocusChange?: (key: string | null) => void;
}) {
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const openZoom = useUiStore((state) => state.openZoom);
  const setView = useUiStore((state) => state.setView);
  const setFocus = useUiStore((state) => state.setFocus);

  const detail = useMediaFullDetail(focusKey, qualityMetric);
  const toggleFav = useToggleFavorite();
  const addTag = useAddTag();
  const removeTag = useRemoveTag();
  const categories = useTagCategories();
  const groundingEnabled = useGroundingEnabled();
  const invalidateMedia = useMediaInvalidator();
  const [newTag, setNewTag] = useState("");

  // A focus restored from the last session can name a media that has since
  // been deleted: drop it rather than sit on "Loading…" forever.
  const stale = detail.isError;
  useEffect(() => {
    if (stale) onClose?.();
  }, [stale, onClose]);

  // Nothing selected → no panel at all, so the grid reclaims the width.
  if (focusKey == null) return null;

  const data = detail.data;
  if (!data) {
    return (
      <Aside onClose={onClose}>
        <div style={{ padding: 16, color: colors.textMuted }}>Loading…</div>
      </Aside>
    );
  }

  return (
    <Aside onClose={onClose}>
      <div style={{ padding: 14, overflowY: "auto" }}>
        <img
          src={data.thumb}
          alt={data.name}
          onClick={() => openZoom(data.file, data.name, data.is_video)}
          style={{ width: "100%", borderRadius: 8, cursor: "zoom-in" }}
        />
        <div style={{ marginTop: 10, fontSize: 12, fontWeight: 600 }}>
          {data.name}
        </div>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textFaint,
            marginTop: 4,
          }}
        >
          {data.meta.width}×{data.meta.height} · {data.meta.files} file(s)
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <Button
            block
            style={
              data.favorite
                ? { color: colors.fav, borderColor: colors.fav }
                : undefined
            }
            onClick={() => toggleFav.mutate(data.key)}
          >
            {data.favorite ? "♥ Favorite" : "♡ Favorite"}
          </Button>
          <a href={data.file} target="_blank" rel="noreferrer" style={{ flex: 1 }}>
            <Button block>Open file</Button>
          </a>
        </div>

        {Object.keys(data.quality_scores).length > 0 && (
          <div style={{ marginTop: 16 }}>
            <Label>Quality scores</Label>
            {Object.entries(data.quality_scores).map(([metric, score]) => (
              <div
                key={metric}
                style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}
              >
                <span
                  style={{ width: 72, fontSize: 10.5, color: colors.textMuted }}
                >
                  {metric}
                </span>
                <div
                  style={{
                    flex: 1,
                    height: 6,
                    background: colors.border,
                    borderRadius: 3,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${score ?? 0}%`,
                      height: "100%",
                      background: qualityColor(score),
                    }}
                  />
                </div>
                <span
                  style={{ width: 28, fontSize: 10.5, fontFamily: font.mono }}
                >
                  {score == null ? "—" : Math.round(score)}
                </span>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          {groundingEnabled && (
            <>
              <Label>Tag grounding</Label>
              <TagGroundingCard
                mediaKey={data.key}
                name={data.name}
                disabled={data.is_video}
              />
            </>
          )}
          <MediaTagScoreCard
            mediaKey={data.key}
            lines={data.tag_score}
            disabled={data.is_video}
          />
        </div>

        {datasetId != null && (
          <CropSection
            mediaKey={data.key}
            crop={data.crop}
            isVideo={data.is_video}
            datasetId={datasetId}
            datasetName={datasetName}
            onFocusChange={onFocusChange}
          />
        )}

        <WatermarkEncart mediaKey={data.key} />

        {data.datasets.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <Label>In datasets</Label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {data.datasets.map((name) => (
                <span
                  key={name}
                  style={{
                    fontSize: 11,
                    padding: "2px 7px",
                    borderRadius: 10,
                    background: "rgba(111,168,220,0.12)",
                    color: colors.info,
                  }}
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <Label>Tags</Label>
          {data.tags.map((group) => (
            <div key={group.category} style={{ marginBottom: 8 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 10.5,
                  color: colors.textMuted,
                  marginBottom: 4,
                }}
              >
                <Dot color={group.color} size={7} /> {group.category}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {group.tags.map((tag) => (
                  <TagChip
                    key={tag.id}
                    name={tag.name}
                    color={group.color}
                    onRemove={() =>
                      removeTag.mutate(
                        { key: data.key, tag_id: tag.id },
                        { onSuccess: invalidateMedia },
                      )
                    }
                  />
                ))}
              </div>
            </div>
          ))}
          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <input
              value={newTag}
              onChange={(event) => setNewTag(event.target.value)}
              placeholder="+ add tag"
              style={inputStyle}
            />
            <Button
              disabled={!newTag.trim() || !categories.data}
              onClick={() => {
                const category = categories.data?.categories[0];
                if (!category) return;
                addTag.mutate(
                  { key: data.key, name: newTag.trim(), category_id: category.id },
                  { onSuccess: invalidateMedia },
                );
                setNewTag("");
              }}
            >
              +
            </Button>
          </div>
        </div>

        {data.captions.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <Label>Captions</Label>
            {data.captions.map((caption) => (
              <div
                key={caption.type}
                style={{ display: "flex", gap: 8, marginBottom: 6, fontSize: 11 }}
              >
                <span
                  style={{
                    fontFamily: font.mono,
                    color: colors.textMuted,
                    minWidth: 40,
                  }}
                >
                  .{caption.type}
                </span>
                <span
                  style={{
                    flex: 1,
                    color: colors.textSecondary,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {caption.preview}
                </span>
              </div>
            ))}
            <a
              onClick={() => {
                setFocus(data.key);
                setView("caption");
              }}
              style={{ fontSize: 12, cursor: "pointer" }}
            >
              edit in Caption →
            </a>
          </div>
        )}
      </div>
    </Aside>
  );
}

function Aside({
  children,
  onClose,
}: {
  children: React.ReactNode;
  onClose?: () => void;
}) {
  return (
    <div
      style={{
        position: "relative",
        width: 330,
        flex: "none",
        borderLeft: `1px solid ${colors.border}`,
        background: colors.panel,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      {onClose && (
        <IconButton
          onClick={onClose}
          title="Close panel"
          aria-label="Close panel"
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            zIndex: 2,
            width: 22,
            height: 22,
            background: colors.raised,
          }}
        >
          ✕
        </IconButton>
      )}
      {children}
    </div>
  );
}

/** The "Watermark" encart of the Media side panel (violet card). */
function WatermarkEncart({ mediaKey }: { mediaKey: string }) {
  const media = useWatermarkMedia(mediaKey);
  const openWatermark = useUiStore((state) => state.openWatermark);
  const data = media.data;
  // No zones yet: still offer a one-click launch focused on this media.
  if (!data || data.zone_count === 0) {
    return (
      <div style={{ marginTop: 16 }}>
        <Label>Watermark</Label>
        <Button
          block
          onClick={() => openWatermark(mediaKey, "media")}
          style={{ background: colors.watermarkBtn, color: colors.watermark }}
        >
          ◪ Open in Watermark Lab
        </Button>
      </div>
    );
  }
  const tab = data.status === "detected" ? "watermarked" : "patched";
  const badge = data.flattened ? "flattened" : data.status;
  const { color, label } = watermarkStatus(badge);
  const line =
    data.status === "patched"
      ? data.flattened
        ? `${data.zone_count} watermark(s) flattened to disk`
        : `${data.zone_count} watermark(s) removed`
      : "Watermark detected — not patched yet";
  return (
    <div style={{ marginTop: 16 }}>
      <Label>Watermark</Label>
      <div
        style={{
          border: `1px solid ${colors.watermarkBorderSoft}`,
          background: colors.watermarkBg,
          borderRadius: 8,
          padding: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: color,
            }}
          />
          <span style={{ fontSize: 12, color: colors.textSecondary }}>
            {label} — {line}
          </span>
        </div>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textMuted,
            margin: "6px 0 10px",
          }}
        >
          {data.zone_count} zone(s)
          {data.detectors.length ? ` · ${data.detectors.join(" / ")}` : ""}
          {data.models.length ? ` · ${data.models.join(" + ")}` : ""}
          {data.score_min != null
            ? ` · min score ${Math.round(data.score_min)}%`
            : ""}
        </div>
        <Button
          block
          onClick={() => openWatermark(mediaKey, tab)}
          style={{ background: colors.watermarkBtn, color: colors.watermark }}
        >
          ◪ Open in Watermark Lab
        </Button>
      </div>
    </div>
  );
}

const inputStyle = {
  flex: 1,
  padding: "5px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;
