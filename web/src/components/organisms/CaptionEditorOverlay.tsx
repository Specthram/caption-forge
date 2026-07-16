/**
 * Full-screen caption editor (the ⛶ on the detail panel's Caption header).
 *
 * Big media on the left, a wide editing column on the right: a full-height
 * textarea, a live char/word counter, "insert a tag" chips that append to the
 * caption, and ‹ › navigation across the current caption grid (Ctrl+← →).
 * Saving writes a new revision, exactly like the side panel. Esc closes.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useCaptionGrid, useMediaDetail, useSaveCaption } from "../../api/hooks";
import type { GridParams } from "../../api/hooks";
import { Button } from "../atoms";
import { colors, font, radii } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";

const NAV_LIMIT = 500;

export function CaptionEditorOverlay() {
  const editor = useUiStore((state) => state.captionEditor);
  const open = useUiStore((state) => state.openCaptionEditor);
  const close = useUiStore((state) => state.closeCaptionEditor);
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const reviewFilter = useUiStore((state) => state.reviewFilter);
  const qualityMetric = useUiStore((state) => state.qualityMetric);

  const gridParams: GridParams = {
    dataset_id: datasetId ?? 0,
    caption_type: captionType,
    review_filter: reviewFilter,
    offset: 0,
    limit: NAV_LIMIT,
    quality_metric: qualityMetric,
  };
  const grid = useCaptionGrid(
    gridParams,
    editor.open && datasetId != null,
  );
  const detail = useMediaDetail(
    editor.key,
    datasetId,
    captionType,
    qualityMetric,
  );
  const saveCaption = useSaveCaption();

  const [draft, setDraft] = useState("");
  const data = detail.data;
  // Reload the draft only when the shown media or its revision changes, never
  // on an unrelated refetch that would clobber the user's typing.
  useEffect(() => {
    if (data) setDraft(data.caption);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.key, data?.revision_value]);

  const keys = useMemo(
    () => (grid.data?.items ?? []).map((item) => item.key),
    [grid.data],
  );
  const index = editor.key ? keys.indexOf(editor.key) : -1;
  const goto = useCallback(
    (delta: number) => {
      const next = index + delta;
      if (next >= 0 && next < keys.length) open(keys[next]);
    },
    [index, keys, open],
  );

  useEffect(() => {
    if (!editor.open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
      else if (event.ctrlKey && event.key === "ArrowRight") goto(1);
      else if (event.ctrlKey && event.key === "ArrowLeft") goto(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editor.open, goto, close]);

  if (!editor.open || datasetId == null) return null;

  const dirty = data != null && draft !== data.caption;
  const save = () => {
    if (!data || !dirty) return;
    saveCaption.mutate({
      key: data.key,
      dataset_id: datasetId,
      caption_type: captionType,
      content: draft,
      scope: "type",
    });
  };
  const insertTag = (name: string) => {
    setDraft((prev) => (prev.trim() ? `${prev.trimEnd()} ${name}` : name));
  };

  const words = draft.trim() ? draft.trim().split(/\s+/).length : 0;

  return (
    <div style={overlay}>
      <div style={header}>
        <button style={navButton} onClick={() => goto(-1)}>
          ‹
        </button>
        <span style={{ fontSize: 12, color: colors.textMuted }}>
          {index >= 0 ? index + 1 : "—"} / {keys.length}
        </span>
        <button style={navButton} onClick={() => goto(1)}>
          ›
        </button>
        <b style={{ fontSize: 13, marginLeft: 8 }}>{data?.name ?? ""}</b>
        <span style={{ fontSize: 11, color: colors.textFaint }}>
          Ctrl+← → to navigate
        </span>
        <div style={{ flex: 1 }} />
        <button style={navButton} onClick={close} title="Close (Esc)">
          ✕
        </button>
      </div>
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <div style={imageWrap}>
          {data &&
            (data.is_video ? (
              <video src={data.file} controls style={media} />
            ) : (
              <img src={data.file} alt={data.name} style={media} />
            ))}
        </div>
        <div style={column}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            style={editArea}
            placeholder="Write the caption…"
          />
          <div style={counterRow}>
            <span>
              {draft.length} chars · {words} words
            </span>
            <span style={{ color: dirty ? colors.warn : colors.textFaint }}>
              {dirty ? "unsaved — saves as a new revision" : "saved"}
            </span>
          </div>
          {data && data.tags.length > 0 && (
            <div>
              <div style={tagLabel}>Insert a tag</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {data.tags.map((tag) => (
                  <button
                    key={tag.id}
                    onClick={() => insertTag(tag.name)}
                    style={{ ...tagChip, color: tag.color }}
                  >
                    {tag.name}
                  </button>
                ))}
              </div>
            </div>
          )}
          <Button
            variant="accent"
            onClick={save}
            disabled={!dirty || saveCaption.isPending}
          >
            Save revision
          </Button>
        </div>
      </div>
    </div>
  );
}

const overlay = {
  position: "fixed",
  inset: 0,
  zIndex: 850,
  background: colors.app,
  display: "flex",
  flexDirection: "column",
} as const;

const header = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "10px 16px",
  borderBottom: `1px solid ${colors.border}`,
  background: colors.toolbar,
} as const;

const navButton = {
  width: 28,
  height: 26,
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: "transparent",
  color: colors.text,
  cursor: "pointer",
} as const;

const imageWrap = {
  flex: 1,
  minWidth: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 24,
  background: colors.app,
} as const;

const media = {
  maxWidth: "100%",
  maxHeight: "100%",
  borderRadius: radii.card,
} as const;

const column = {
  width: 520,
  flex: "none",
  borderLeft: `1px solid ${colors.border}`,
  background: colors.panel,
  padding: 18,
  display: "flex",
  flexDirection: "column",
  gap: 12,
} as const;

const editArea = {
  flex: 1,
  minHeight: 200,
  padding: 12,
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 13.5,
  lineHeight: 1.7,
  fontFamily: font.sans,
  resize: "none",
} as const;

const counterRow = {
  display: "flex",
  justifyContent: "space-between",
  fontSize: 11,
  color: colors.textMuted,
} as const;

const tagLabel = {
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  fontWeight: 700,
  color: colors.textMuted,
  marginBottom: 6,
} as const;

const tagChip = {
  padding: "3px 8px",
  borderRadius: radii.chip,
  border: `1px solid ${colors.borderControl}`,
  background: colors.card,
  cursor: "pointer",
  fontSize: 11.5,
} as const;
