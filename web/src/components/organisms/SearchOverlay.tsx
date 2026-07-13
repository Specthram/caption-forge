/** ⌘K command palette (Phase 1: navigation + focus, search wiring later). */

import { useState } from "react";
import { colors, font, radii, shadow } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import type { ViewId } from "../../api/types";

const ACTIONS: { label: string; view: ViewId }[] = [
  { label: "Go to Caption", view: "caption" },
  { label: "Go to Datasets", view: "datasets" },
  { label: "Go to Media", view: "media" },
  { label: "Go to Tags", view: "tags" },
  { label: "Go to Libraries", view: "libraries" },
  { label: "Go to Settings", view: "settings" },
  { label: "Go to System", view: "system" },
];

export function SearchOverlay() {
  const open = useUiStore((state) => state.searchOpen);
  const close = useUiStore((state) => state.toggleSearch);
  const setView = useUiStore((state) => state.setView);
  const [query, setQuery] = useState("");

  if (!open) return null;
  const results = ACTIONS.filter((action) =>
    action.label.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div
      onClick={() => close(false)}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8,9,11,0.6)",
        zIndex: 50,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        paddingTop: "12vh",
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 560,
          maxWidth: "90%",
          background: colors.panel,
          border: `1px solid ${colors.borderHover}`,
          borderRadius: radii.modal,
          boxShadow: shadow.modal,
          overflow: "hidden",
        }}
      >
        <input
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search media, tags, datasets…"
          style={{
            width: "100%",
            padding: "14px 16px",
            border: "none",
            borderBottom: `1px solid ${colors.border}`,
            background: "transparent",
            color: colors.text,
            fontSize: 14,
            fontFamily: font.sans,
          }}
        />
        <div style={{ maxHeight: 360, overflowY: "auto", padding: 6 }}>
          {results.map((action) => (
            <div
              key={action.label}
              onClick={() => {
                setView(action.view);
                close(false);
              }}
              style={{
                padding: "9px 12px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 13,
                color: colors.textSecondary,
              }}
            >
              {action.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
