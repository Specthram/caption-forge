/**
 * "Crop · alias" section of the Caption and Datasets detail panels.
 *
 * A crop is a virtual alias: no image file is created, the pixels are
 * rendered on the fly and only materialized when the dataset is deployed. The
 * section therefore reads as a list of *framings* of one source image — the
 * focused media's own, or, when a crop is focused, its siblings.
 *
 * Focusing a crop swaps the header for its rectangle and offers Detach (the
 * entry reverts to the original) alongside Edit. Focusing an ordinary media
 * offers a new crop, plus its existing ones for reuse: clicking a row edits
 * it, the ⊕ places it in the dataset without touching the frame.
 */

import { useCrops, useDeleteCrop, usePlaceCrop } from "../../api/hooks";
import type { CropInfo } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { Label } from "../atoms";

export function CropSection({
  mediaKey,
  crop,
  isVideo,
  datasetId,
  datasetName,
  onFocusChange,
}: {
  /** The focused dataset entry: an ordinary media, or a crop. */
  mediaKey: string;
  crop: CropInfo | null;
  isVideo: boolean;
  datasetId: number;
  datasetName?: string;
  /** Called with the entry that should take the focus after a change. */
  onFocusChange?: (key: string | null) => void;
}) {
  const openCrop = useUiStore((state) => state.openCrop);
  const mediaId = Number(mediaKey);
  const sourceId = crop ? crop.parent_media_id : mediaId;

  const crops = useCrops(mediaId, datasetId);
  const placeCrop = usePlaceCrop();
  const deleteCrop = useDeleteCrop();

  // A video has no rectangle to frame: the crop is rendered from a still.
  if (isVideo) return null;

  const rows = crops.data?.crops ?? [];

  const edit = (id: number, info: { rect: CropInfo["rect"]; ratio: CropInfo["ratio"] }) =>
    openCrop(sourceId, { id, rect: info.rect, ratio: info.ratio });

  return (
    <div
      style={{
        marginTop: 14,
        paddingTop: 10,
        borderTop: `1px solid ${colors.border}`,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <Label>Crop · alias</Label>
        <span style={{ flex: 1 }} />
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 9.5,
            color: colors.textFaint,
          }}
        >
          virtual · {datasetName ?? `dataset #${datasetId}`}
        </span>
      </div>

      {crop ? (
        <div
          style={{
            marginTop: 8,
            padding: 10,
            borderRadius: 8,
            border: `1px solid ${colors.accentBorder}`,
            background: colors.accentTintAlt,
          }}
        >
          <div
            style={{
              fontFamily: font.mono,
              fontSize: 11,
              color: colors.accent,
            }}
          >
            {crop.ratio} · {crop.width}×{crop.height}
          </div>
          <div
            style={{ fontSize: 9.5, color: colors.textFaint, marginTop: 3 }}
          >
            Rendered on the fly — no file exists until the dataset is deployed.
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <SmallButton
              accent
              onClick={() => edit(mediaId, crop)}
              label="Edit the crop"
            />
            <SmallButton
              onClick={() =>
                deleteCrop.mutate(mediaId, {
                  onSuccess: () =>
                    onFocusChange?.(String(crop.parent_media_id)),
                })
              }
              label="Detach"
              title="Delete the crop; the original takes its place back"
            />
          </div>
        </div>
      ) : (
        <SmallButton
          accent
          block
          style={{ marginTop: 8 }}
          onClick={() => openCrop(mediaId)}
          label="⌗ Crop this image"
        />
      )}

      {rows.length > 0 && (
        <div style={{ marginTop: 10, display: "grid", gap: 5 }}>
          {rows.map((row) => {
            const focused = row.id === mediaId;
            return (
              <div
                key={row.id}
                onClick={() => edit(row.id, row)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "4px 6px",
                  borderRadius: 6,
                  cursor: "pointer",
                  border: `1px solid ${focused ? colors.accentBorder : "transparent"}`,
                  background: focused ? colors.accentTint : "transparent",
                }}
              >
                <img
                  src={row.thumb}
                  alt=""
                  loading="lazy"
                  style={{
                    width: 36,
                    height: 27,
                    objectFit: "cover",
                    borderRadius: 3,
                    background: colors.raised,
                  }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontFamily: font.mono,
                      fontSize: 10.5,
                      color: colors.text,
                    }}
                  >
                    {row.ratio} · {row.width}×{row.height}
                  </div>
                  <div
                    style={{
                      fontFamily: font.mono,
                      fontSize: 9,
                      color: colors.textFaint,
                    }}
                  >
                    {row.in_dataset ? "in this dataset" : "not placed"}
                  </div>
                </div>
                {!row.in_dataset && (
                  <IconAction
                    title="Add this crop to the dataset, beside the original"
                    color={colors.accent}
                    onClick={() =>
                      placeCrop.mutate({
                        id: row.id,
                        dataset_id: datasetId,
                        mode: "beside",
                      })
                    }
                  >
                    ⊕
                  </IconAction>
                )}
                <IconAction
                  title="Delete this crop"
                  color={colors.danger}
                  onClick={() =>
                    deleteCrop.mutate(row.id, {
                      onSuccess: () => {
                        if (focused) onFocusChange?.(String(sourceId));
                      },
                    })
                  }
                >
                  ✕
                </IconAction>
              </div>
            );
          })}
        </div>
      )}

      <div style={{ fontSize: 9.5, color: colors.textFaint, marginTop: 8 }}>
        Quality, auto-tag and grounding re-run on the cropped pixels.
      </div>
    </div>
  );
}

function SmallButton({
  label,
  onClick,
  accent,
  block,
  title,
  style,
}: {
  label: string;
  onClick: () => void;
  accent?: boolean;
  block?: boolean;
  title?: string;
  style?: React.CSSProperties;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      style={{
        padding: "6px 10px",
        borderRadius: 6,
        fontSize: 11.5,
        fontWeight: 600,
        cursor: "pointer",
        width: block ? "100%" : undefined,
        border: `1px solid ${accent ? colors.accentBorder : colors.borderControl}`,
        background: accent ? colors.accentTintAlt : colors.card,
        color: accent ? colors.accent : colors.textMutedAlt,
        ...style,
      }}
    >
      {label}
    </button>
  );
}

function IconAction({
  children,
  title,
  color,
  onClick,
}: {
  children: React.ReactNode;
  title: string;
  color: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      style={{
        width: 20,
        height: 20,
        border: "none",
        borderRadius: 5,
        background: "transparent",
        color,
        fontSize: 11,
        lineHeight: 1,
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
