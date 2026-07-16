/** Caption workspace — the gallery and the rule-based Review sub-tab. */

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useDatasets, useJobResult, useReviewCounts } from "../api/hooks";
import { useJobsStore } from "../store/jobsStore";
import { useUiStore } from "../store/uiStore";
import { colors, font, radii } from "../design/tokens";
import { BatchBar } from "../components/organisms/BatchBar";
import { CaptionCenterGrid } from "../components/organisms/CaptionCenterGrid";
import { CaptionDetailPanel } from "../components/organisms/CaptionDetailPanel";
import { CaptionLeftPanel } from "../components/organisms/CaptionLeftPanel";
import { CaptionEditorOverlay } from "../components/organisms/CaptionEditorOverlay";
import { ReviewView } from "../components/organisms/ReviewView";
import { ReviewWizard } from "../components/organisms/ReviewWizard";

export function CaptionView() {
  const datasetId = useUiStore((state) => state.datasetId);
  const setDataset = useUiStore((state) => state.setDataset);
  const captionTab = useUiStore((state) => state.captionTab);
  const datasets = useDatasets();

  useEffect(() => {
    if (datasetId == null && datasets.data?.datasets.length) {
      setDataset(datasets.data.datasets[0].id);
    }
  }, [datasetId, datasets.data, setDataset]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <CaptionTabs />
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        {captionTab === "review" ? (
          <ReviewView />
        ) : (
          <>
            <CaptionLeftPanel />
            <CaptionCenterGrid />
            <CaptionDetailPanel />
            <BatchBar />
          </>
        )}
      </div>
      <ReviewWizard />
      <CaptionEditorOverlay />
      <ReviewJobWatcher />
    </div>
  );
}

/**
 * Watches a single-media review job (fired by the detail panel's Review
 * button) to completion — kept here, not in the panel, because switching to
 * the Review tab unmounts the panel mid-job. On success it refreshes the
 * queue and opens the wizard when the media produced findings.
 */
function ReviewJobWatcher() {
  const jobId = useUiStore((state) => state.pendingReviewJob);
  const setJob = useUiStore((state) => state.setPendingReviewJob);
  const openWizard = useUiStore((state) => state.openReviewWizard);
  const client = useQueryClient();
  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const done = job?.state === "done";
  const result = useJobResult<{ findings: number }>(jobId, done);

  useEffect(() => {
    if (!jobId || !job) return;
    if (job.state === "error" || job.state === "stopped") {
      setJob(null);
      return;
    }
    if (done && result.data) {
      client.invalidateQueries({ queryKey: ["review-findings"] });
      client.invalidateQueries({ queryKey: ["review-counts"] });
      if ((result.data.result?.findings ?? 0) > 0) openWizard(0);
      setJob(null);
    }
  }, [jobId, job, done, result.data, client, openWizard, setJob]);

  return null;
}

function CaptionTabs() {
  const captionTab = useUiStore((state) => state.captionTab);
  const setCaptionTab = useUiStore((state) => state.setCaptionTab);
  const datasetId = useUiStore((state) => state.datasetId);
  const counts = useReviewCounts(datasetId);
  const pending = counts.data?.pending ?? 0;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: "6px 12px",
        borderBottom: `1px solid ${colors.border}`,
        background: colors.toolbar,
      }}
    >
      <Tab
        active={captionTab === "caption"}
        onClick={() => setCaptionTab("caption")}
      >
        ✎ Caption
      </Tab>
      <Tab
        active={captionTab === "review"}
        onClick={() => setCaptionTab("review")}
      >
        ☑ Review
        {pending > 0 && (
          <span
            style={{
              marginLeft: 6,
              padding: "0 6px",
              borderRadius: 9,
              fontSize: 10,
              fontWeight: 700,
              fontFamily: font.mono,
              color: colors.onAccent,
              background: colors.info,
            }}
          >
            {pending}
          </span>
        )}
      </Tab>
      <div style={{ flex: 1 }} />
      <span style={{ fontSize: 11, color: colors.textFaint }}>
        Rule-based QA — judge model independent from the captioner
      </span>
    </div>
  );
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "6px 12px",
        borderRadius: radii.control,
        border: "none",
        background: active ? colors.accentTint : "transparent",
        color: active ? colors.accent : colors.textMuted,
        fontSize: 12.5,
        fontWeight: 600,
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
