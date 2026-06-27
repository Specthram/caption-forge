/** Tag include/exclude filter: search + selected chips. */

import { useState } from "react";
import { useTagSearch } from "../../api/hooks";
import { colors } from "../../design/tokens";
import { TagChip } from "./index";

export interface SelectedTag {
  id: number;
  name: string;
}

export function TagFilter({
  label,
  selected,
  onAdd,
  onRemove,
  onCreate,
}: {
  label: string;
  selected: SelectedTag[];
  onAdd: (tag: SelectedTag) => void;
  onRemove: (id: number) => void;
  /** When set, typing a name with no exact match offers to create it. */
  onCreate?: (name: string) => void;
}) {
  const [query, setQuery] = useState("");
  const search = useTagSearch(query, query.length > 0);
  const chosen = new Set(selected.map((tag) => tag.id));
  const trimmed = query.trim();
  const results = (search.data?.tags ?? []).filter(
    (tag) => !chosen.has(tag.id),
  );
  const exact = (search.data?.tags ?? []).some(
    (tag) => tag.name.toLowerCase() === trimmed.toLowerCase(),
  );
  const canCreate = !!onCreate && trimmed.length > 0 && !exact;

  return (
    <div style={{ position: "relative", minWidth: 150 }}>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 4,
          alignItems: "center",
          padding: "3px 6px",
          borderRadius: 6,
          border: `1px solid ${colors.borderControl}`,
          background: colors.input,
        }}
      >
        {selected.map((tag) => (
          <TagChip
            key={tag.id}
            name={tag.name}
            color={colors.textMuted}
            onRemove={() => onRemove(tag.id)}
          />
        ))}
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={label}
          style={{
            flex: 1,
            minWidth: 70,
            border: "none",
            background: "transparent",
            color: colors.text,
            fontSize: 12,
            outline: "none",
          }}
        />
      </div>
      {query && (results.length > 0 || canCreate) && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 20,
            marginTop: 2,
            maxHeight: 200,
            overflowY: "auto",
            background: colors.panel,
            border: `1px solid ${colors.borderHover}`,
            borderRadius: 6,
          }}
        >
          {results.map((tag) => (
            <div
              key={tag.id}
              onClick={() => {
                onAdd({ id: tag.id, name: tag.name });
                setQuery("");
              }}
              style={{
                padding: "5px 9px",
                fontSize: 12,
                cursor: "pointer",
                color: colors.textSecondary,
              }}
            >
              {tag.name}
            </div>
          ))}
          {canCreate && (
            <div
              onClick={() => {
                onCreate?.(trimmed);
                setQuery("");
              }}
              style={{
                padding: "5px 9px",
                fontSize: 12,
                cursor: "pointer",
                color: colors.accent,
                borderTop:
                  results.length > 0 ? `1px solid ${colors.border}` : undefined,
              }}
            >
              ➕ Create “{trimmed}” · Uncategorized
            </div>
          )}
        </div>
      )}
    </div>
  );
}
