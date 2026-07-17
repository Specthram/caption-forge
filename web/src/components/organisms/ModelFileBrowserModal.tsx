/**
 * Model file browser — picks a profile's weights or mmproj on the server.
 *
 * Navigates `GET /api/profiles/browse` (folders + .gguf/.safetensors files
 * with sizes). Files that cannot be picked for the current target — wrong
 * format for weights, non-mmproj when picking a projector and vice versa —
 * are dimmed and inert. Clicking a selectable file picks it and closes.
 */

import { useState } from "react";
import { useBrowseModelFiles } from "../../api/hooks";
import { colors, font, radii, shadow } from "../../design/tokens";

function isMmprojFile(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.startsWith("mmproj") && lower.endsWith(".gguf");
}

function formatSize(bytes: number | undefined): string {
  if (bytes == null) return "";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

export function ModelFileBrowserModal({
  target,
  format,
  initialPath,
  onClose,
  onPick,
}: {
  target: "model" | "mmproj";
  /** Weights format the editor is set to — other formats are dimmed. */
  format: "gguf" | "safetensors";
  initialPath: string;
  onClose: () => void;
  /** Receives the folder and the picked file name. */
  onPick: (dir: string, file: string) => void;
}) {
  const [cur, setCur] = useState(initialPath);
  const listing = useBrowseModelFiles(cur);
  const data = listing.data;

  const selectable = (name: string): boolean => {
    if (target === "mmproj") return isMmprojFile(name);
    if (isMmprojFile(name)) return false;
    return name.toLowerCase().endsWith(`.${format}`);
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,11,13,0.82)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 600,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 470,
          maxWidth: "92vw",
          maxHeight: "72vh",
          display: "flex",
          flexDirection: "column",
          background: colors.panel,
          border: `1px solid ${colors.borderHover}`,
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
          <span style={{ fontSize: 14, fontWeight: 700, flex: 1 }}>
            {target === "model"
              ? "Select model weights"
              : "Select mmproj (vision projector)"}
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
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "7px 14px",
            background: colors.toolbar,
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <button
            title="Parent folder"
            disabled={data?.parent === null}
            onClick={() => data && setCur(data.parent ?? "")}
            style={{
              border: `1px solid ${colors.borderControl}`,
              background: "transparent",
              borderRadius: radii.control,
              width: 24,
              height: 22,
              cursor: data?.parent === null ? "default" : "pointer",
              color:
                data?.parent === null ? colors.textFaint : colors.textMuted,
            }}
          >
            ↑
          </button>
          <span
            style={{
              flex: 1,
              fontFamily: font.mono,
              fontSize: 11,
              color: colors.textMuted,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {data?.is_root ? "This PC" : data?.path || "…"}
          </span>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: 6 }}>
          {(data?.entries ?? []).map((entry) => {
            if (entry.kind === "dir") {
              return (
                <BrowserRow
                  key={entry.path}
                  icon="📁"
                  name={entry.name}
                  meta=""
                  enabled
                  onClick={() => setCur(entry.path)}
                />
              );
            }
            const mmproj = isMmprojFile(entry.name);
            const enabled = selectable(entry.name);
            return (
              <BrowserRow
                key={entry.path}
                icon={mmproj ? "◈" : "▣"}
                name={entry.name}
                meta={
                  formatSize(entry.size) + (mmproj ? " · mmproj" : "")
                }
                enabled={enabled}
                onClick={() => {
                  if (!enabled || !data) return;
                  onPick(data.path, entry.name);
                  onClose();
                }}
              />
            );
          })}
          {data && !data.is_root && data.entries.length === 0 && (
            <div style={{ padding: 12, fontSize: 12, color: colors.textFaint }}>
              Nothing here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function BrowserRow({
  icon,
  name,
  meta,
  enabled,
  onClick,
}: {
  icon: string;
  name: string;
  meta: string;
  enabled: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 9,
        padding: "6px 10px",
        borderRadius: radii.control,
        cursor: enabled ? "pointer" : "default",
        color: enabled ? colors.textSecondary : "#4a4d57",
      }}
      onMouseEnter={(event) => {
        if (enabled) event.currentTarget.style.background = colors.raised;
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = "transparent";
      }}
    >
      <span style={{ fontSize: 12, flex: "none" }}>{icon}</span>
      <span
        style={{
          flex: 1,
          fontFamily: font.mono,
          fontSize: 11.5,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {name}
      </span>
      {meta && (
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textFaint,
            flex: "none",
          }}
        >
          {meta}
        </span>
      )}
    </div>
  );
}
