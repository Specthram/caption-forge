/** Tags workspace: reorderable categories (master) + paged tag browser. */

import { useEffect, useMemo, useState } from "react";
import {
  useCreateCategory,
  useCreateTag,
  useDedupeTags,
  useDeleteTag,
  useMoveTag,
  useRenameTag,
  useReorderCategories,
  useTagCategories,
  useTagsList,
} from "../api/hooks";
import { colors, font } from "../design/tokens";
import { useUiStore } from "../store/uiStore";
import { Button, Dot, Label } from "../components/atoms";

const STEP = 96;

export function TagsView() {
  const categories = useTagCategories();
  const createCategory = useCreateCategory();
  const reorder = useReorderCategories();
  const createTag = useCreateTag();
  const deleteTag = useDeleteTag();
  const dedupe = useDedupeTags();
  const moveTag = useMoveTag();
  const renameTag = useRenameTag();

  // The selected category is restored after a refresh.
  const activeId = useUiStore((state) => state.tagsView.activeId);
  const setTagsView = useUiStore((state) => state.setTagsView);
  const setActiveId = (next: number | null) =>
    setTagsView({ activeId: next });
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(STEP);
  const [newCat, setNewCat] = useState("");
  const [newTag, setNewTag] = useState("");
  const [dedupeNote, setDedupeNote] = useState("");
  // The category currently hovered while dragging a tag onto it.
  const [dropCat, setDropCat] = useState<number | null>(null);
  const [dragId, setDragId] = useState<number | null>(null);
  // Inline rename: the tag being edited and its working text.
  const [editId, setEditId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  const commitRename = (id: number, name: string, original: string) => {
    const next = name.trim();
    if (!next || next === original) {
      setEditId(null);
      return;
    }
    renameTag.mutate(
      { id, name: next },
      {
        onSuccess: () => setEditId(null),
        // A name clash (409/400) keeps the field open so it can be fixed.
        onError: () => undefined,
      },
    );
  };

  const cats = useMemo(
    () => categories.data?.categories ?? [],
    [categories.data],
  );
  // Select the first category on a cold start, and fall back to it when the
  // one restored from the last session has since been deleted.
  useEffect(() => {
    if (!cats.length) return;
    if (activeId == null || !cats.some((cat) => cat.id === activeId)) {
      setActiveId(cats[0].id);
    }
  }, [activeId, cats]); // eslint-disable-line react-hooks/exhaustive-deps

  const active = cats.find((category) => category.id === activeId);
  const tags = useTagsList(activeId, query, limit);
  const total = tags.data?.total ?? 0;
  const shown = tags.data?.items.length ?? 0;

  const move = (index: number, delta: number) => {
    const order = cats.map((category) => category.id);
    const target = index + delta;
    if (target < 0 || target >= order.length) return;
    [order[index], order[target]] = [order[target], order[index]];
    reorder.mutate(order);
  };

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <div
        style={{
          width: 264,
          flex: "none",
          borderRight: `1px solid ${colors.border}`,
          background: colors.panel,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div style={{ padding: "12px 12px 6px" }}>
          <Label>Categories — in output order</Label>
          <div style={{ fontSize: 10.5, color: colors.textFaint }}>
            This order drives the tags caption type and Media/Caption lists.
          </div>
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: "4px 8px" }}>
          {cats.map((category, index) => (
            <div
              key={category.id}
              onClick={() => {
                setActiveId(category.id);
                setQuery("");
                setLimit(STEP);
              }}
              onDragOver={(event) => {
                if (dragId == null || category.id === activeId) return;
                event.preventDefault();
                setDropCat(category.id);
              }}
              onDragLeave={() => setDropCat((id) => (id === category.id ? null : id))}
              onDrop={(event) => {
                event.preventDefault();
                setDropCat(null);
                if (dragId != null && category.id !== activeId) {
                  moveTag.mutate({ id: dragId, category_id: category.id });
                }
                setDragId(null);
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 8px",
                borderRadius: 6,
                cursor: "pointer",
                marginBottom: 2,
                outline:
                  category.id === dropCat
                    ? `2px dashed ${colors.accent}`
                    : "none",
                background:
                  category.id === dropCat
                    ? colors.accentTintAlt
                    : category.id === activeId
                      ? colors.accentTint
                      : "transparent",
              }}
            >
              <span
                style={{ fontSize: 10, color: colors.textFaint, width: 14 }}
              >
                {index + 1}
              </span>
              <Dot color={category.color} size={9} />
              <span
                style={{
                  flex: 1,
                  fontSize: 12.5,
                  color:
                    category.id === activeId
                      ? colors.accent
                      : colors.textSecondary,
                }}
              >
                {category.name}
              </span>
              <span
                style={{
                  fontSize: 10,
                  fontFamily: font.mono,
                  color: colors.textFaint,
                }}
              >
                {category.count?.toLocaleString() ?? 0}
              </span>
              <span style={{ display: "flex", flexDirection: "column" }}>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    move(index, -1);
                  }}
                  style={arrowBtn}
                >
                  ▲
                </button>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    move(index, 1);
                  }}
                  style={arrowBtn}
                >
                  ▼
                </button>
              </span>
            </div>
          ))}
        </div>
        <div style={{ padding: 10, borderTop: `1px solid ${colors.border}` }}>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={newCat}
              onChange={(event) => setNewCat(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && newCat.trim()) {
                  createCategory.mutate({
                    name: newCat.trim(),
                    color: "#888888",
                  });
                  setNewCat("");
                }
              }}
              placeholder="New category…"
              style={inputStyle}
            />
            <Button
              onClick={() => {
                if (!newCat.trim()) return;
                createCategory.mutate({
                  name: newCat.trim(),
                  color: "#888888",
                });
                setNewCat("");
              }}
            >
              +
            </Button>
          </div>
          <Button
            block
            style={{ marginTop: 8 }}
            disabled={dedupe.isPending}
            title="Merge tags duplicated across categories (e.g. cloned into the WD14 category) back into their original one, re-pointing media."
            onClick={() =>
              dedupe.mutate(undefined, {
                onSuccess: (data) =>
                  setDedupeNote(
                    data.removed === 0
                      ? "No duplicates found."
                      : `Merged ${data.removed} duplicate(s) across ${data.names} name(s).`,
                  ),
              })
            }
          >
            {dedupe.isPending ? "Deduping…" : "⚗ Dedupe duplicate tags"}
          </Button>
          {dedupeNote && (
            <div
              style={{
                fontSize: 10.5,
                color: colors.ok,
                fontFamily: font.mono,
                marginTop: 6,
              }}
            >
              {dedupeNote}
            </div>
          )}
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        {active ? (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "12px 16px",
                borderBottom: `1px solid ${colors.border}`,
                background: colors.toolbar,
              }}
            >
              <Dot color={active.color} size={11} />
              <span style={{ fontSize: 15, fontWeight: 600 }}>
                {active.name}
              </span>
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: 11,
                  color: colors.textMuted,
                }}
              >
                {(active.count ?? 0).toLocaleString()} tags
              </span>
              <span style={{ flex: 1 }} />
              <input
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setLimit(STEP);
                }}
                placeholder={`Search in ${active.name}…`}
                style={{ ...inputStyle, width: 220 }}
              />
              <input
                value={newTag}
                onChange={(event) => setNewTag(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && newTag.trim()) {
                    createTag.mutate({
                      name: newTag.trim(),
                      category_id: active.id,
                    });
                    setNewTag("");
                  }
                }}
                placeholder="+ add tag"
                style={{ ...inputStyle, width: 110 }}
              />
            </div>

            {query && (
              <div
                style={{
                  padding: "6px 16px",
                  background: colors.accentTintAlt,
                  color: colors.accent,
                  fontSize: 11.5,
                }}
              >
                &ldquo;{query}&rdquo; — {total.toLocaleString()} matches across{" "}
                {(active.count ?? 0).toLocaleString()} tags (indexed search,
                server-side)
              </div>
            )}

            <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {tags.data?.items.map((tag) => (
                  <span
                    key={tag.id}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 7,
                      padding: "5px 10px",
                      borderRadius: 12,
                      background: colors.card,
                      border: `1px solid ${
                        editId === tag.id ? colors.accent : colors.border
                      }`,
                      fontSize: 12.5,
                      color: colors.textSecondary,
                      opacity: dragId === tag.id ? 0.4 : 1,
                    }}
                  >
                    <span
                      draggable
                      onDragStart={(event) => {
                        setDragId(tag.id);
                        event.dataTransfer.effectAllowed = "move";
                        event.dataTransfer.setData("text/plain", String(tag.id));
                      }}
                      onDragEnd={() => {
                        setDragId(null);
                        setDropCat(null);
                      }}
                      title="Drag onto a category to move it there"
                      style={{
                        color: colors.textFaint,
                        fontSize: 12,
                        lineHeight: 1,
                        letterSpacing: "-1px",
                        cursor: "grab",
                      }}
                    >
                      ⠿
                    </span>
                    {editId === tag.id ? (
                      <input
                        autoFocus
                        value={editText}
                        onChange={(event) => setEditText(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            commitRename(tag.id, editText, tag.name);
                          } else if (event.key === "Escape") {
                            setEditId(null);
                          }
                        }}
                        onBlur={() =>
                          commitRename(tag.id, editText, tag.name)
                        }
                        style={{
                          width: `${Math.max(4, editText.length + 1)}ch`,
                          padding: 0,
                          border: "none",
                          outline: "none",
                          background: "transparent",
                          color: colors.text,
                          fontSize: 12.5,
                          fontFamily: font.sans,
                        }}
                      />
                    ) : (
                      <span
                        onClick={() => {
                          setEditId(tag.id);
                          setEditText(tag.name);
                        }}
                        title="Click to rename"
                        style={{ cursor: "text" }}
                      >
                        {tag.name}
                      </span>
                    )}
                    <span
                      style={{
                        fontFamily: font.mono,
                        fontSize: 10,
                        color: colors.textFaint,
                      }}
                    >
                      {tag.usage_count.toLocaleString()}
                    </span>
                    <span
                      onClick={() => deleteTag.mutate(tag.id)}
                      style={{ cursor: "pointer", color: colors.textMuted }}
                    >
                      ✕
                    </span>
                  </span>
                ))}
              </div>
              <div
                style={{
                  marginTop: 14,
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <span style={{ fontSize: 11, color: colors.textFaint }}>
                  showing {shown.toLocaleString()} of {total.toLocaleString()}
                </span>
                {shown < total && (
                  <Button onClick={() => setLimit((value) => value + STEP)}>
                    Load more
                  </Button>
                )}
              </div>
            </div>
          </>
        ) : (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: colors.textFaint,
            }}
          >
            Create or select a category.
          </div>
        )}
      </div>
    </div>
  );
}

const arrowBtn = {
  border: "none",
  background: "transparent",
  color: colors.textFaint,
  cursor: "pointer",
  fontSize: 7,
  lineHeight: "8px",
  padding: 0,
} as const;

const inputStyle = {
  padding: "5px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;
