/**
 * Server-side folder picker for the Libraries "Add folder" flow.
 *
 * The media live on the machine running the app, so choosing a library
 * folder means walking the *server* filesystem (a browser file input can
 * never hand back an absolute directory path). This navigates the folders
 * returned by `GET /api/system/browse` and returns the chosen absolute path.
 */

import { useState } from "react";
import { useBrowseFolder } from "../../api/hooks";
import { colors, font, radii, shadow } from "../../design/tokens";
import { Button } from "../atoms";

export function FolderBrowserModal({
  onClose,
  onSelect,
  initialPath = "",
  hint = "Add this folder as a library.",
  confirmLabel = "Use this folder",
}: {
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
  hint?: string;
  confirmLabel?: string;
}) {
  // "" is the drive-root level (This PC).
  const [cur, setCur] = useState(initialPath);
  const [filter, setFilter] = useState("");
  const listing = useBrowseFolder(cur, true);
  const data = listing.data;
  const canSelect = !!data && !data.is_root && !!data.path;
  const entries = (data?.entries ?? []).filter((entry) =>
    entry.name.toLowerCase().includes(filter.trim().toLowerCase()),
  );

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 520,
          maxWidth: "92vw",
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
          background: colors.panel,
          border: `1px solid ${colors.border}`,
          borderRadius: radii.modal,
          boxShadow: shadow.modal,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "12px 14px",
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>
            Choose a folder
          </span>
          <span
            onClick={onClose}
            style={{ cursor: "pointer", color: colors.textMuted }}
          >
            ✕
          </span>
        </div>

        <div
          style={{
            padding: "8px 14px",
            fontFamily: font.mono,
            fontSize: 11,
            color: colors.textMuted,
            borderBottom: `1px solid ${colors.border}`,
            wordBreak: "break-all",
          }}
        >
          {data?.is_root ? "This PC" : data?.path || "…"}
        </div>

        {data && !data.is_root && (data.entries.length ?? 0) > 0 && (
          <div
            style={{
              padding: "8px 14px",
              borderBottom: `1px solid ${colors.border}`,
            }}
          >
            <input
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter folders…"
              autoFocus
              style={{
                width: "100%",
                padding: "5px 8px",
                borderRadius: 6,
                border: `1px solid ${colors.borderControl}`,
                background: colors.input,
                color: colors.text,
                fontSize: 12,
                outline: "none",
              }}
            />
          </div>
        )}

        <div style={{ flex: 1, overflowY: "auto", padding: 6 }}>
          {data && data.parent !== null && (
            <Row
              icon="⬆"
              label=".."
              onClick={() => {
                setCur(data.parent ?? "");
                setFilter("");
              }}
            />
          )}
          {listing.isError && (
            <div style={{ padding: 12, fontSize: 12, color: colors.danger }}>
              {listing.error instanceof Error
                ? listing.error.message
                : "Cannot open this folder."}
            </div>
          )}
          {entries.map((entry) => (
            <Row
              key={entry.path}
              icon={data?.is_root ? "🖴" : "📁"}
              label={entry.name}
              onClick={() => {
                setCur(entry.path);
                setFilter("");
              }}
            />
          ))}
          {data && !data.is_root && entries.length === 0 && (
            <div style={{ padding: 12, fontSize: 12, color: colors.textFaint }}>
              {data.entries.length === 0
                ? "No sub-folders here."
                : "No folder matches the filter."}
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "10px 14px",
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          <span
            style={{
              flex: 1,
              fontSize: 11,
              color: colors.textFaint,
            }}
          >
            {canSelect ? hint : "Open a drive, then a folder."}
          </span>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            variant="accent"
            disabled={!canSelect}
            onClick={() => {
              if (data) {
                onSelect(data.path);
                onClose();
              }
            }}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}

function Row({
  icon,
  label,
  onClick,
}: {
  icon: string;
  label: string;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 9,
        padding: "7px 10px",
        borderRadius: 6,
        cursor: "pointer",
        fontSize: 12.5,
        color: colors.textSecondary,
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.background = colors.raised;
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = "transparent";
      }}
    >
      <span style={{ fontSize: 13 }}>{icon}</span>
      <span
        style={{
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
    </div>
  );
}
