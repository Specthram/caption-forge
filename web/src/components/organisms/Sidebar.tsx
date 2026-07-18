/** Left navigation rail: logo, views, model status and the Jobs button. */

import {
  FolderOpen,
  Hash,
  Images,
  LayoutGrid,
  Pencil,
  Server,
  Settings as SettingsIcon,
} from "lucide-react";
import type { ReactNode } from "react";
import {
  useDatasets,
  useModelStatus,
  useNavCounts,
  useProfiles,
  useUnloadModel,
} from "../../api/hooks";
import type { ViewId } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useRunningCount } from "../../store/jobsStore";
import { Dot, Spinner } from "../atoms";
import { NavItem } from "../molecules";

interface NavDef {
  id: ViewId;
  label: string;
  icon: ReactNode;
}

const NAV: NavDef[] = [
  { id: "caption", label: "Caption", icon: <Pencil size={14} /> },
  { id: "datasets", label: "Datasets", icon: <LayoutGrid size={14} /> },
  { id: "media", label: "Media", icon: <Images size={14} /> },
  { id: "tags", label: "Tags", icon: <Hash size={14} /> },
  { id: "libraries", label: "Libraries", icon: <FolderOpen size={14} /> },
  { id: "settings", label: "Settings", icon: <SettingsIcon size={14} /> },
  { id: "system", label: "System", icon: <Server size={14} /> },
];

/** Eject glyph freeing VRAM — reachable from every view (persistent rail). */
function EjectButton({
  busy,
  onClick,
}: {
  busy: boolean;
  onClick: () => void;
}) {
  return (
    <button
      title="Unload model — free VRAM"
      disabled={busy}
      onClick={onClick}
      style={{
        flex: "none",
        width: 22,
        height: 22,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 5,
        border: "1px solid #2c2f38",
        background: "#1b1d22",
        color: "#9a9ba2",
        fontSize: 11,
        cursor: busy ? "default" : "pointer",
        opacity: busy ? 0.5 : 1,
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.color = "#e06c5c";
        event.currentTarget.style.borderColor = "#57302b";
        event.currentTarget.style.background = "#241715";
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.color = "#9a9ba2";
        event.currentTarget.style.borderColor = "#2c2f38";
        event.currentTarget.style.background = "#1b1d22";
      }}
    >
      ⏏
    </button>
  );
}

/** Abbreviate a total the way the rail's 10px mono column can hold it. */
function short(value: number | undefined): string | undefined {
  if (value == null) return undefined;
  if (value < 1000) return String(value);
  return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
}

export function Sidebar() {
  const view = useUiStore((state) => state.view);
  const setView = useUiStore((state) => state.setView);
  const datasetId = useUiStore((state) => state.datasetId);
  const toggleJobs = useUiStore((state) => state.toggleJobs);
  const datasets = useDatasets();
  const totals = useNavCounts();
  const status = useModelStatus();
  const profiles = useProfiles();
  const running = useRunningCount();
  const unloadModel = useUnloadModel();

  // The loaded model shown as its profile (name + weights file); a model
  // loaded outside profiles falls back to the loader's filename.
  const loaded = status.data?.loaded ?? false;
  const loadedProfile = profiles.data?.profiles.find(
    (p) => p.id === profiles.data.loaded_id,
  );
  // The "file" line shows the HF repo id for an HF profile, else the weights
  // filename (falling back to the loader's own name for a non-profile load).
  const loadedFile = loadedProfile
    ? loadedProfile.source === "hf"
      ? loadedProfile.repo
      : loadedProfile.file
    : (status.data?.name ?? "");

  // The Caption badge counts the dataset that tab is actually working on,
  // not the library: it is the size of the grid the user is about to see.
  const active = datasets.data?.datasets.find((row) => row.id === datasetId);
  const counts: Partial<Record<ViewId, string>> = {
    caption: short(active?.count),
    datasets: short(datasets.data?.datasets.length),
    media: short(totals.data?.media),
    tags: short(totals.data?.tags),
    libraries: short(totals.data?.libraries),
  };

  return (
    <div
      style={{
        width: 212,
        flex: "none",
        display: "flex",
        flexDirection: "column",
        background: colors.panel,
        borderRight: `1px solid ${colors.border}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          padding: "14px 14px 12px",
        }}
      >
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: 7,
            background: "linear-gradient(135deg,#e8935a,#b85c2e)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: 12,
            color: colors.onAccent,
          }}
        >
          CF
        </div>
        <div style={{ lineHeight: 1.15 }}>
          <div style={{ fontWeight: 600, fontSize: 13.5 }}>Caption Forge</div>
          <div
            style={{ fontSize: 10, color: colors.textFaint, fontFamily: font.mono }}
          >
            v2.0 · local
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          padding: "4px 8px",
        }}
      >
        {NAV.map((item) => (
          <NavItem
            key={item.id}
            icon={item.icon}
            label={item.label}
            count={counts[item.id]}
            active={view === item.id}
            onClick={() => setView(item.id)}
          />
        ))}
      </div>

      <div style={{ flex: 1 }} />

      <div style={{ padding: 12, borderTop: `1px solid ${colors.border}` }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 10,
          }}
          title={loaded ? loadedFile : ""}
        >
          <Dot color={loaded ? colors.ok : colors.textFaint} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: 11.5,
                color: colors.textSecondary,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {loaded
                ? (loadedProfile?.name ?? status.data?.name ?? "Model loaded")
                : "No model loaded"}
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
              {loaded ? `${loadedFile} · loaded` : "unloaded — memory purged"}
            </div>
          </div>
          {loaded && (
            <EjectButton
              busy={unloadModel.isPending}
              onClick={() => unloadModel.mutate()}
            />
          )}
        </div>
        <button
          onClick={() => toggleJobs()}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            padding: "7px 10px",
            borderRadius: 6,
            border: `1px solid ${colors.borderControl}`,
            background: colors.raised,
            color: colors.textSecondary,
            cursor: "pointer",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {running > 0 && <Spinner size={12} />}
          Jobs {running > 0 ? `· ${running}` : ""}
        </button>
      </div>
    </div>
  );
}
