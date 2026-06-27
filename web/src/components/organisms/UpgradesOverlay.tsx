/**
 * Living-dataset upgrades: a blue banner in the Datasets rail and the
 * overlay it opens (z-70). A dataset built by the Studio keeps its recipe;
 * replayed against the current library, it may find a stronger candidate
 * than one of its members. Each row is a swap — out (weaker, red) → in
 * (stronger, green) — applied one at a time or in bulk. The outgoing image
 * only leaves the dataset; it stays in the library.
 */

import { useState } from "react";
import {
  useApplyAutobuildUpgrades,
  useAutobuildUpgrades,
} from "../../api/hooks";
import type { AutobuildUpgrade } from "../../api/types";
import { colors, font, radii } from "../../design/tokens";

const thumbUrl = (id: number) => `/api/media/${id}/thumb`;

export function LivingDatasetUpgrades({ datasetId }: { datasetId: number }) {
  const upgrades = useAutobuildUpgrades(datasetId);
  const apply = useApplyAutobuildUpgrades(datasetId);
  const [open, setOpen] = useState(false);
  const [busyIn, setBusyIn] = useState<number | null>(null);

  const list = upgrades.data?.upgrades ?? [];
  if (!list.length) return null;

  const applyOne = (upgrade: AutobuildUpgrade) => {
    setBusyIn(upgrade.in_media_id);
    apply.mutate([upgrade], { onSettled: () => setBusyIn(null) });
  };
  const applyAll = () => {
    setBusyIn(-1);
    apply.mutate(list, {
      onSettled: () => {
        setBusyIn(null);
        setOpen(false);
      },
    });
  };

  return (
    <>
      <div onClick={() => setOpen(true)} style={banner}>
        <span style={{ fontSize: 12, color: colors.info }}>
          ◇ Living dataset — {list.length} upgrade
          {list.length > 1 ? "s" : ""}
        </span>
        <span style={{ fontSize: 11, color: colors.textFaint }}>
          the recipe found stronger candidates — review
        </span>
      </div>

      {open && (
        <div onClick={() => setOpen(false)} style={backdrop}>
          <div
            onClick={(event) => event.stopPropagation()}
            style={panel}
          >
            <div style={header}>
              <span style={{ fontSize: 14, fontWeight: 700 }}>
                Living dataset — {list.length} upgrade
                {list.length > 1 ? "s" : ""}
              </span>
              <div style={{ flex: 1 }} />
              <span onClick={() => setOpen(false)} style={closeX}>
                ✕
              </span>
            </div>

            <div style={body}>
              {list.map((upgrade) => {
                const busy =
                  busyIn === upgrade.in_media_id || busyIn === -1;
                return (
                  <div key={upgrade.in_media_id} style={row}>
                    <Side
                      id={upgrade.out_media_id}
                      name={upgrade.out_name}
                      quality={upgrade.out_quality}
                      color={colors.danger}
                      label="out"
                    />
                    <div style={arrow}>→</div>
                    <Side
                      id={upgrade.in_media_id}
                      name={upgrade.in_name}
                      quality={upgrade.in_quality}
                      color={colors.ok}
                      label="in"
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, color: colors.textMutedAlt }}>
                        {upgrade.reason}
                      </div>
                    </div>
                    <button
                      disabled={busy}
                      onClick={() => applyOne(upgrade)}
                      style={applyButton}
                    >
                      {busy ? "⟳ applying…" : "Apply"}
                    </button>
                  </div>
                );
              })}
            </div>

            <div style={footer}>
              <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                an upgrade = a swap in the dataset — the outgoing image stays
                in the library
              </span>
              <div style={{ flex: 1 }} />
              <button onClick={() => setOpen(false)} style={ghost}>
                Ignore
              </button>
              <button
                disabled={busyIn != null}
                onClick={applyAll}
                style={applyAllButton}
              >
                Apply all ({list.length})
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Side({
  id,
  name,
  quality,
  color,
  label,
}: {
  id: number;
  name: string;
  quality: number | null;
  color: string;
  label: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
      <img src={thumbUrl(id)} alt="" style={sideThumb} />
      <div>
        <div style={{ fontFamily: font.mono, fontSize: 9, color }}>
          {label} · Q {quality == null ? "—" : quality.toFixed(0)}
        </div>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 9,
            color: colors.textMuted,
            maxWidth: 90,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {name}
        </div>
      </div>
    </div>
  );
}

const banner = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 16px",
  background: "#15202c",
  borderBottom: "1px solid #2f4860",
  cursor: "pointer",
} as const;

const backdrop = {
  position: "fixed",
  inset: 0,
  zIndex: 70,
  background: "rgba(10,11,14,0.7)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 16,
} as const;

const panel = {
  width: 640,
  maxWidth: "94%",
  maxHeight: "86vh",
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: radii.modal,
  boxShadow: "0 24px 80px rgba(0,0,0,0.65)",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
} as const;

const header = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "12px 16px",
  borderBottom: `1px solid ${colors.border}`,
} as const;

const closeX = {
  cursor: "pointer",
  color: colors.textMuted,
  fontSize: 15,
  padding: "4px 8px",
} as const;

const body = {
  flex: 1,
  overflowY: "auto",
  padding: 12,
  display: "flex",
  flexDirection: "column",
  gap: 8,
} as const;

const row = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 10px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.card,
  background: colors.card,
} as const;

const arrow = { color: colors.textFaint, fontSize: 13 } as const;

const sideThumb = {
  width: 40,
  height: 30,
  objectFit: "cover",
  borderRadius: 4,
  border: `1px solid ${colors.borderControl}`,
} as const;

const applyButton = {
  flex: "none",
  padding: "6px 12px",
  border: "none",
  borderRadius: 6,
  background: colors.accent,
  color: colors.onAccent,
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const footer = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "11px 16px",
  borderTop: `1px solid ${colors.border}`,
  background: colors.toolbar,
} as const;

const ghost = {
  padding: "8px 14px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: "transparent",
  color: colors.textMutedAlt,
  fontSize: 12,
  cursor: "pointer",
} as const;

const applyAllButton = {
  padding: "8px 16px",
  border: "none",
  borderRadius: 7,
  background: colors.accent,
  color: colors.onAccent,
  fontSize: 12,
  fontWeight: 700,
  cursor: "pointer",
} as const;
