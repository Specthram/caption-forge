/** Top bar: view title, global search field and the GPU chip. */

import { Search } from "lucide-react";
import { useModelStatus } from "../../api/hooks";
import type { ViewId } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { Kbd } from "../atoms";
import { GpuChip } from "../molecules";
import { PowerWidget } from "./PowerWidget";

const TITLES: Record<ViewId, string> = {
  caption: "Caption",
  datasets: "Datasets",
  media: "Media",
  tags: "Tags",
  libraries: "Libraries",
  settings: "Settings",
  system: "System",
};

export function Topbar() {
  const view = useUiStore((state) => state.view);
  const toggleSearch = useUiStore((state) => state.toggleSearch);
  const status = useModelStatus();

  return (
    <div
      style={{
        height: 46,
        flex: "none",
        display: "flex",
        alignItems: "center",
        gap: 16,
        padding: "0 16px",
        background: colors.panel,
        borderBottom: `1px solid ${colors.border}`,
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 14, minWidth: 90 }}>
        {TITLES[view]}
      </div>

      <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
        <button
          onClick={() => toggleSearch(true)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            width: 420,
            maxWidth: "60%",
            padding: "6px 10px",
            borderRadius: 6,
            border: `1px solid ${colors.borderControl}`,
            background: colors.input,
            color: colors.textMuted,
            cursor: "text",
            fontSize: 12,
          }}
        >
          <Search size={13} />
          <span style={{ flex: 1, textAlign: "left" }}>
            Search media, tags, datasets…
          </span>
          <Kbd>⌘K</Kbd>
        </button>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontFamily: font.mono,
        }}
      >
        <GpuChip
          gpu={status.data?.gpu ?? null}
          totalGb={status.data?.vram_total_gb ?? null}
        />
        <PowerWidget />
      </div>
    </div>
  );
}
