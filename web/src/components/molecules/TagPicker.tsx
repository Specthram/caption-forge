/**
 * Unified tag selector: search the catalogue, pick an existing tag, or
 * create a new one *in a chosen category*.
 *
 * The single source of truth for every "add a tag" control — the media and
 * caption detail panels attach the picked tag to their media, the library
 * bulk-tag and dataset-composer filters push it into their selection. The
 * parent only ever receives ``onPick({id, name})``; searching, the results
 * dropdown and category-scoped creation all live here.
 */

import { useState } from "react";
import {
  useCreateTag,
  useTagCategories,
  useTagSearch,
} from "../../api/hooks";
import { colors, radii } from "../../design/tokens";
import { Dot } from "../atoms";

export interface PickedTag {
  id: number;
  name: string;
}

export function TagPicker({
  placeholder,
  exclude,
  onPick,
  allowCreate = true,
}: {
  placeholder: string;
  /** Tag ids already chosen — hidden from the results. */
  exclude?: Set<number>;
  onPick: (tag: PickedTag) => void;
  /** Offer to create a missing name (category-scoped). Default true. */
  allowCreate?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const search = useTagSearch(query, query.trim().length > 0);
  const categories = useTagCategories();
  const createTag = useCreateTag();

  const trimmed = query.trim();
  const chosen = exclude ?? new Set<number>();
  const results = (search.data?.tags ?? []).filter(
    (tag) => !chosen.has(tag.id),
  );
  const exact = (search.data?.tags ?? []).some(
    (tag) => tag.name.toLowerCase() === trimmed.toLowerCase(),
  );
  const canCreate = allowCreate && trimmed.length > 0 && !exact;

  const reset = () => {
    setQuery("");
    setCreating(false);
  };

  const pick = (tag: PickedTag) => {
    onPick(tag);
    reset();
  };

  const create = (categoryId: number) => {
    const name = trimmed;
    createTag.mutate(
      { name, category_id: categoryId },
      { onSuccess: (data) => pick({ id: data.id, name }) },
    );
  };

  const open = query.length > 0 && (results.length > 0 || canCreate);

  return (
    <div style={{ position: "relative", minWidth: 150, flex: 1 }}>
      <input
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setCreating(false);
        }}
        placeholder={placeholder}
        style={{
          width: "100%",
          padding: "5px 8px",
          borderRadius: 6,
          border: `1px solid ${colors.borderControl}`,
          background: colors.input,
          color: colors.text,
          fontSize: 12,
        }}
      />
      {open && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 30,
            marginTop: 2,
            maxHeight: 240,
            overflowY: "auto",
            background: colors.panel,
            border: `1px solid ${colors.borderHover}`,
            borderRadius: 6,
            boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
          }}
        >
          {results.map((tag) => (
            <div
              key={tag.id}
              onClick={() => pick({ id: tag.id, name: tag.name })}
              style={rowStyle}
            >
              {tag.name}
            </div>
          ))}
          {canCreate && !creating && (
            <div
              onClick={() => setCreating(true)}
              style={{
                ...rowStyle,
                color: colors.accent,
                borderTop:
                  results.length > 0
                    ? `1px solid ${colors.border}`
                    : undefined,
              }}
            >
              ➕ Create “{trimmed}”…
            </div>
          )}
          {canCreate && creating && (
            <div
              style={{
                padding: "7px 9px",
                borderTop:
                  results.length > 0
                    ? `1px solid ${colors.border}`
                    : undefined,
              }}
            >
              <div
                style={{
                  fontSize: 10.5,
                  color: colors.textMuted,
                  marginBottom: 6,
                }}
              >
                Create “{trimmed}” in:
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {(categories.data?.categories ?? []).map((category) => (
                  <button
                    key={category.id}
                    disabled={createTag.isPending}
                    onClick={() => create(category.id)}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 5,
                      padding: "3px 8px",
                      borderRadius: radii.control,
                      border: `1px solid ${colors.borderControl}`,
                      background: colors.card,
                      color: colors.textSecondary,
                      fontSize: 11,
                      cursor: "pointer",
                    }}
                  >
                    <Dot color={category.color} size={7} /> {category.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const rowStyle = {
  padding: "5px 9px",
  fontSize: 12,
  cursor: "pointer",
  color: colors.textSecondary,
} as const;
