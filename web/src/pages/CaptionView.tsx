/** Caption workspace — three-column layout wired to the real backend. */

import { useEffect } from "react";
import { useDatasets } from "../api/hooks";
import { useUiStore } from "../store/uiStore";
import { BatchBar } from "../components/organisms/BatchBar";
import { CaptionCenterGrid } from "../components/organisms/CaptionCenterGrid";
import { CaptionDetailPanel } from "../components/organisms/CaptionDetailPanel";
import { CaptionLeftPanel } from "../components/organisms/CaptionLeftPanel";

export function CaptionView() {
  const datasetId = useUiStore((state) => state.datasetId);
  const setDataset = useUiStore((state) => state.setDataset);
  const datasets = useDatasets();

  useEffect(() => {
    if (datasetId == null && datasets.data?.datasets.length) {
      setDataset(datasets.data.datasets[0].id);
    }
  }, [datasetId, datasets.data, setDataset]);

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <CaptionLeftPanel />
      <CaptionCenterGrid />
      <CaptionDetailPanel />
      <BatchBar />
    </div>
  );
}
