/** App shell: sidebar + topbar + the routed view, plus global overlays. */

import type { ReactNode } from "react";
import { colors, font } from "../../design/tokens";
import { useJobsSocket } from "../../hooks/useJobsSocket";
import { useKeyboard } from "../../hooks/useKeyboard";
import { CropOverlay } from "../organisms/CropOverlay";
import { GroundingModal } from "../organisms/GroundingModal";
import { JobsDrawer } from "../organisms/JobsDrawer";
import { SearchOverlay } from "../organisms/SearchOverlay";
import { Sidebar } from "../organisms/Sidebar";
import { Topbar } from "../organisms/Topbar";
import { ZoomLightbox } from "../organisms/ZoomLightbox";
import { LookalikeCompare } from "../organisms/LookalikeCompare";
import { WatermarkLab } from "../organisms/WatermarkLab";

export function AppShell({ children }: { children: ReactNode }) {
  useJobsSocket();
  useKeyboard();

  return (
    <div
      style={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: colors.app,
        color: colors.text,
        fontFamily: font.sans,
        fontSize: font.base,
      }}
    >
      <Sidebar />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <Topbar />
        <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
          {children}
        </div>
      </div>
      <JobsDrawer />
      <SearchOverlay />
      <ZoomLightbox />
      <LookalikeCompare />
      <GroundingModal />
      <CropOverlay />
      <WatermarkLab />
    </div>
  );
}
