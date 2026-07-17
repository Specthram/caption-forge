/**
 * Model-profile selector — the shared dropdown of the Caption panel
 * (captioner) and the Review rail (judge).
 *
 * Trigger row: loaded dot, profile name + weights filename, an ✎ opening
 * the editor on the current profile, a caret. The menu lists every profile
 * (dot / name / file / type badge / per-row ✎) and ends with "+ New
 * profile". Selection binds the caption or judge slot server-side.
 */

import { useState } from "react";
import { useProfiles, useSelectProfile } from "../../api/hooks";
import type { ModelProfile } from "../../api/types";
import { colors, font, radii } from "../../design/tokens";
import { ProfileEditorModal, TypeBadge } from "./ProfileEditorModal";

export function ProfileSelector({ role }: { role: "caption" | "judge" }) {
  const profiles = useProfiles();
  const select = useSelectProfile();
  const [open, setOpen] = useState(false);
  const [editor, setEditor] = useState<
    { profile: ModelProfile | null } | null
  >(null);

  const data = profiles.data;
  const selectedId =
    role === "caption" ? data?.active_id : data?.judge_id;
  const selected = data?.profiles.find((p) => p.id === selectedId) ?? null;
  const families = data?.families ?? [];

  return (
    <div style={{ position: "relative" }}>
      <div
        onClick={() => setOpen((prev) => !prev)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "7px 9px",
          borderRadius: radii.control,
          border: `1px solid ${colors.borderControl}`,
          background: colors.input,
          cursor: "pointer",
        }}
      >
        <LoadedDot loaded={selected != null && selected.id === data?.loaded_id} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {selected?.name ?? "…"}
          </div>
          <div
            style={{
              fontFamily: font.mono,
              fontSize: 9.5,
              color: colors.textFaint,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {selected?.file || "no weights picked"}
          </div>
        </div>
        <EditIcon
          onClick={(event) => {
            event.stopPropagation();
            if (selected) setEditor({ profile: selected });
          }}
        />
        <span style={{ fontSize: 9, color: colors.textFaint }}>▾</span>
      </div>

      {open && (
        <>
          <div
            onClick={() => setOpen(false)}
            style={{ position: "fixed", inset: 0, zIndex: 55 }}
          />
          <div
            style={{
              position: "absolute",
              top: "calc(100% + 4px)",
              left: 0,
              right: 0,
              zIndex: 60,
              background: colors.card,
              border: `1px solid ${colors.borderHover}`,
              borderRadius: radii.card,
              boxShadow: "0 14px 40px rgba(0,0,0,0.55)",
              overflow: "hidden",
            }}
          >
            {(data?.profiles ?? []).map((profile) => {
              const isSelected = profile.id === selectedId;
              return (
                <div
                  key={profile.id}
                  onClick={() => {
                    select.mutate({ role, id: profile.id });
                    setOpen(false);
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "7px 9px",
                    cursor: "pointer",
                    background: isSelected ? "#1e2026" : "transparent",
                  }}
                  onMouseEnter={(event) => {
                    if (!isSelected)
                      event.currentTarget.style.background = colors.raised;
                  }}
                  onMouseLeave={(event) => {
                    event.currentTarget.style.background = isSelected
                      ? "#1e2026"
                      : "transparent";
                  }}
                >
                  <LoadedDot loaded={profile.id === data?.loaded_id} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 11.5,
                        fontWeight: 600,
                        color: isSelected ? colors.accent : colors.text,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {profile.name}
                    </div>
                    <div
                      style={{
                        fontFamily: font.mono,
                        fontSize: 9,
                        color: colors.textFaint,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {profile.file || "no weights picked"}
                    </div>
                  </div>
                  <TypeBadge type={profile.type} families={families} />
                  <EditIcon
                    onClick={(event) => {
                      event.stopPropagation();
                      setEditor({ profile });
                      setOpen(false);
                    }}
                  />
                </div>
              );
            })}
            <div
              onClick={() => {
                setEditor({ profile: null });
                setOpen(false);
              }}
              style={{
                padding: "7px 9px",
                borderTop: `1px solid ${colors.border}`,
                color: colors.accent,
                fontSize: 11.5,
                fontWeight: 600,
                cursor: "pointer",
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.background = colors.raised;
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.background = "transparent";
              }}
            >
              + New profile
            </div>
          </div>
        </>
      )}

      {editor && (
        <ProfileEditorModal
          profile={editor.profile}
          role={role}
          families={families}
          profileCount={data?.profiles.length ?? 1}
          onClose={() => setEditor(null)}
        />
      )}
    </div>
  );
}

function LoadedDot({ loaded }: { loaded: boolean }) {
  return (
    <span
      style={{
        width: 7,
        height: 7,
        borderRadius: "50%",
        flex: "none",
        background: loaded ? colors.ok : colors.borderHover,
      }}
    />
  );
}

function EditIcon({
  onClick,
}: {
  onClick: (event: React.MouseEvent) => void;
}) {
  return (
    <span
      title="Edit profile"
      onClick={onClick}
      style={{
        fontSize: 11,
        color: colors.textMuted,
        padding: "2px 4px",
        borderRadius: 4,
        cursor: "pointer",
        flex: "none",
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.color = colors.accent;
        event.currentTarget.style.background = colors.raised;
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.color = colors.textMuted;
        event.currentTarget.style.background = "transparent";
      }}
    >
      ✎
    </span>
  );
}
