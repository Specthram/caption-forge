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
import { useDatasets, useModelStatus, useNavCounts } from "../../api/hooks";
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
  const running = useRunningCount();

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
            fontSize: 11.5,
            color: colors.textSecondary,
            marginBottom: 10,
          }}
        >
          <Dot color={status.data?.loaded ? colors.ok : colors.textFaint} />
          <span
            style={{
              flex: 1,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {status.data?.loaded ? status.data.name : "No model loaded"}
          </span>
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
