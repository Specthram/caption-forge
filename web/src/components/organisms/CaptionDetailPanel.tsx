/** Caption tab right panel: focused-media preview, editor, tags, deploy. */

import { useEffect, useRef, useState } from "react";
import {
  useAddTag,
  useAutosaveEnabled,
  useGroundingEnabled,
  useIntegrityReview,
  useMediaDetail,
  useRemoveTag,
  useSaveCaption,
  useSelectRevision,
  useSetRepeats,
  useTagCategories,
} from "../../api/hooks";
import { colors, deployColor, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useCaptionStore } from "../../store/captionStore";
import { Button, Dot, IconButton, Label } from "../atoms";
import { QualityBadge, RepeatsStepper, TagChip } from "../molecules";
import { CaptionGroundingCard } from "./GroundingCard";
import { CaptionScoreCard } from "./CaptionScoreCard";
import { CropSection } from "./CropSection";

export function CaptionDetailPanel() {
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const focusKey = useUiStore((state) => state.focusKey);
  const setFocus = useUiStore((state) => state.setFocus);
  const openZoom = useUiStore((state) => state.openZoom);
  const close = () => setFocus(null);

  const detail = useMediaDetail(
    focusKey,
    datasetId,
    captionType,
    qualityMetric,
  );
  const categories = useTagCategories();
  const saveCaption = useSaveCaption();
  const selectRevision = useSelectRevision();
  const setRepeats = useSetRepeats();
  const addTag = useAddTag();
  const removeTag = useRemoveTag();
  const integrity = useIntegrityReview();
  const groundingEnabled = useGroundingEnabled();
  const autosaveEnabled = useAutosaveEnabled();
  const locked = useCaptionStore((state) => state.locked);
  const toggleLock = useCaptionStore((state) => state.toggleLock);

  const [draft, setDraft] = useState("");
  const [newTag, setNewTag] = useState("");
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const loadedRef = useRef<string | null>(null);

  const data = detail.data;
  // Load the server caption into the draft when a *different* caption comes
  // into view (another media, type or pinned revision), or whenever the
  // editor is not focused. While the user is actively typing, the autosave
  // refetch must not reassign the textarea value — doing so snapped the
  // caret to the end on every save. Their own text is what autosave persisted
  // anyway, so skipping the reset loses nothing.
  useEffect(() => {
    if (!data) return;
    const id = `${data.key}|${captionType}|${data.revision_value}`;
    const switched = loadedRef.current !== id;
    loadedRef.current = id;
    const editing = document.activeElement === editorRef.current;
    if (switched || !editing) setDraft(data.caption);
  }, [data, captionType]);

  // A focus restored from the last session can name a media that has since
  // been deleted: drop it rather than sit on "Loading…" forever.
  const stale = detail.isError;
  useEffect(() => {
    if (stale) setFocus(null);
  }, [stale, setFocus]);

  // Autosave: after a pause in typing, persist a dirty draft so the Save
  // button becomes optional (toggle in Settings → Captioning). Tags have no
  // free-text editor, so they never autosave.
  useEffect(() => {
    if (!autosaveEnabled || !data || datasetId == null) return;
    if (captionType === "tags") return;
    if (draft === data.caption || saveCaption.isPending) return;
    const timer = setTimeout(() => {
      saveCaption.mutate({
        key: data.key,
        dataset_id: datasetId,
        caption_type: captionType,
        content: draft,
        scope: "type",
        // Autosave overwrites the current revision in place — no new version
        // per pause in typing. The Save button still snapshots a version.
        amend: true,
      });
    }, 800);
    return () => clearTimeout(timer);
  }, [autosaveEnabled, data, datasetId, captionType, draft, saveCaption]);

  // Nothing selected → no panel, so the grid reclaims the width.
  if (focusKey == null || datasetId == null) return null;
  if (!data) {
    return (
      <Aside onClose={close}>
        <div style={{ padding: 16, color: colors.textMuted }}>Loading…</div>
      </Aside>
    );
  }

  const dirty = draft !== data.caption;
  const isTags = captionType === "tags";

  const target = {
    key: data.key,
    dataset_id: datasetId,
    caption_type: captionType,
  };

  return (
    <Aside onClose={close}>
      <div style={{ padding: 14, overflowY: "auto" }}>
        <div style={{ position: "relative" }}>
          <img
            src={data.thumb}
            alt={data.name}
            onClick={() => openZoom(data.file, data.name, data.is_video)}
            style={{
              width: "100%",
              borderRadius: 8,
              cursor: "zoom-in",
              display: "block",
            }}
          />
          <span style={{ position: "absolute", top: 8, right: 8 }}>
            <QualityBadge score={data.quality} />
          </span>
        </div>

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
          {data.meta.width}×{data.meta.height} ·{" "}
          {(data.meta.size_bytes / 1024 / 1024).toFixed(2)} MB · in{" "}
          {data.meta.datasets} datasets
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <Button
            variant="ghost"
            block
            disabled={integrity.isPending}
            onClick={() => integrity.mutate(target)}
          >
            Review
          </Button>
          <Button
            block
            style={
              locked.has(data.key)
                ? { color: colors.warn, borderColor: colors.warn }
                : undefined
            }
            onClick={() => toggleLock(data.key)}
          >
            {locked.has(data.key) ? "🔒 Locked" : "🔓 Lock"}
          </Button>
        </div>

        {data.review === "warn" && data.review_issues.length > 0 && (
          <Box color={colors.warn} tint="rgba(224,179,86,0.08)">
            <b>Integrity issues</b>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
              {data.review_issues.map((code) => (
                <span
                  key={code}
                  style={{
                    fontFamily: font.mono,
                    fontSize: 10,
                    padding: "1px 6px",
                    borderRadius: 4,
                    background: "rgba(224,179,86,0.15)",
                    color: colors.warn,
                  }}
                >
                  {code}
                </span>
              ))}
            </div>
          </Box>
        )}

        {!isTags && groundingEnabled && (
          <CaptionGroundingCard
            mediaKey={data.key}
            name={data.name}
            datasetId={datasetId}
            captionType={captionType}
            summary={data.grounding}
            disabled={data.is_video || data.missing}
          />
        )}

        {!isTags && (
          <CaptionScoreCard
            mediaKey={data.key}
            datasetId={datasetId}
            captionType={captionType}
            lines={data.caption_score}
            disabled={data.is_video || data.missing}
          />
        )}

        {!isTags && (
          <div style={{ marginTop: 16 }}>
            <Label>Caption · {captionType}</Label>
            {data.revisions.length > 0 && (
              <select
                value={String(data.revision_value ?? "follow")}
                onChange={(event) =>
                  selectRevision.mutate({
                    ...target,
                    revision_id:
                      event.target.value === "follow"
                        ? null
                        : Number(event.target.value),
                  })
                }
                style={{
                  width: "100%",
                  marginBottom: 8,
                  padding: "5px 8px",
                  borderRadius: 6,
                  border: `1px solid ${colors.borderControl}`,
                  background: colors.input,
                  color: colors.text,
                  fontSize: 12,
                }}
              >
                {data.revisions.map((rev) => (
                  <option key={String(rev.value)} value={String(rev.value)}>
                    {rev.label}
                  </option>
                ))}
              </select>
            )}
            <textarea
              ref={editorRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={6}
              style={{
                width: "100%",
                padding: 8,
                borderRadius: 6,
                border: `1px solid ${colors.borderControl}`,
                background: colors.input,
                color: colors.text,
                fontSize: 12,
                lineHeight: 1.45,
                resize: "vertical",
                fontFamily: font.sans,
              }}
            />
            <div
              style={{
                display: "flex",
                alignItems: "center",
                marginTop: 6,
                fontSize: 10.5,
                color: colors.textFaint,
                fontFamily: font.mono,
              }}
            >
              {draft.length} chars · {draft.split(/\s+/).filter(Boolean).length}{" "}
              words
              <span style={{ flex: 1 }} />
              {autosaveEnabled && (
                <SaveState
                  saving={saveCaption.isPending}
                  dirty={dirty}
                />
              )}
              <Button
                variant="accent"
                disabled={
                  saveCaption.isPending ||
                  (autosaveEnabled ? !draft.trim() : !dirty)
                }
                onClick={() =>
                  saveCaption.mutate({
                    ...target,
                    content: draft,
                    scope: "type",
                    amend: false,
                  })
                }
              >
                {autosaveEnabled
                  ? "Save version"
                  : saveCaption.isPending
                    ? "Saving…"
                    : dirty
                      ? "Save"
                      : "Saved"}
              </Button>
            </div>
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <Label>Tags</Label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {data.tags.map((tag) => (
              <TagChip
                key={tag.id}
                name={tag.name}
                color={tag.color}
                onRemove={() =>
                  removeTag.mutate({ key: data.key, tag_id: tag.id })
                }
              />
            ))}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <input
              value={newTag}
              onChange={(event) => setNewTag(event.target.value)}
              placeholder="+ add tag"
              style={{
                flex: 1,
                padding: "5px 8px",
                borderRadius: 6,
                border: `1px solid ${colors.borderControl}`,
                background: colors.input,
                color: colors.text,
                fontSize: 12,
              }}
            />
            <Button
              disabled={!newTag.trim() || !categories.data}
              onClick={() => {
                const category = categories.data?.categories[0];
                if (!category) return;
                addTag.mutate({
                  key: data.key,
                  name: newTag.trim(),
                  category_id: category.id,
                });
                setNewTag("");
              }}
            >
              +
            </Button>
          </div>
        </div>

        <CropSection
          mediaKey={data.key}
          crop={data.crop}
          isVideo={data.is_video}
          datasetId={datasetId}
          onFocusChange={setFocus}
        />

        <div
          style={{
            marginTop: 16,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <Dot color={deployColor(data.deploy)} />
          <span style={{ fontSize: 11.5, color: colors.textMuted, flex: 1 }}>
            Deploy repeats
          </span>
          <RepeatsStepper
            value={data.repeats}
            onChange={(value) =>
              setRepeats.mutate({
                key: data.key,
                dataset_id: datasetId,
                repeats: value,
              })
            }
          />
        </div>
      </div>
    </Aside>
  );
}

/** The small autosave status beside Save: green ✓ saved / saving / unsaved. */
function SaveState({ saving, dirty }: { saving: boolean; dirty: boolean }) {
  const state = saving
    ? { text: "saving…", color: colors.textFaint }
    : dirty
      ? { text: "● unsaved", color: colors.warn }
      : { text: "✓ saved", color: colors.ok };
  return (
    <span
      style={{
        fontFamily: font.mono,
        fontSize: 10.5,
        color: state.color,
        marginRight: 8,
        whiteSpace: "nowrap",
      }}
    >
      {state.text}
    </span>
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
        width: 318,
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

function Box({
  color,
  tint,
  children,
}: {
  color: string;
  tint: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        marginTop: 12,
        padding: "10px 12px",
        borderRadius: 8,
        border: `1px solid ${color}55`,
        background: tint,
        fontSize: 11.5,
        color,
      }}
    >
      {children}
    </div>
  );
}
