/** Tag include/exclude filter: selected chips + the shared tag picker. */

import { useMemo } from "react";
import { colors } from "../../design/tokens";
import { TagChip } from "./index";
import { TagPicker } from "./TagPicker";

export interface SelectedTag {
  id: number;
  name: string;
}

export function TagFilter({
  label,
  selected,
  onAdd,
  onRemove,
  allowCreate = false,
}: {
  label: string;
  selected: SelectedTag[];
  onAdd: (tag: SelectedTag) => void;
  onRemove: (id: number) => void;
  /** Offer to create a missing name (category-scoped) via the picker. */
  allowCreate?: boolean;
}) {
  const chosen = useMemo(
    () => new Set(selected.map((tag) => tag.id)),
    [selected],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      {selected.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {selected.map((tag) => (
            <TagChip
              key={tag.id}
              name={tag.name}
              color={colors.textMuted}
              onRemove={() => onRemove(tag.id)}
            />
          ))}
        </div>
      )}
      <TagPicker
        placeholder={label}
        exclude={chosen}
        allowCreate={allowCreate}
        onPick={onAdd}
      />
    </div>
  );
}
