/**
 * Datasets → Quality report: the whole tab.
 *
 * Owns the scorer toolbar, the running state (streamed from the jobs
 * WebSocket), and the resolution actions of the flagged-media card. The
 * report itself is server state: a run is a job, and when that job ends the
 * stored report is re-fetched rather than assembled here.
 */

import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useDatasetReport,
  useGenerate,
  useModelStatus,
  usePrompts,
  useRemoveDatasetMedia,
  useResolveIssue,
  useRunDatasetReport,
  useUnresolveIssue,
} from "../../api/hooks";
import { colors, font, radii, scorerColor } from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";
import { useUiStore } from "../../store/uiStore";
import { Button } from "../atoms";
import { CompositionMap } from "./CompositionMap";
import { FlaggedMedia } from "./FlaggedMedia";
import { QualityCharts } from "./QualityCharts";
import { QualityScoreRow } from "./QualityScoreRow";
import type {
  DatasetReport,
  ReportIssue,
  ReportRecommendation,
  ResolutionKind,
  ScorerChip,
} from "../../api/types";

const TARGET_TYPES = ["character", "style", "concept"];
const RECAPTION_IMAGE_SIZE = 1024;

const cardStyle = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.card,
  padding: 14,
} as const;

/** "1 h ago" / "4 min ago" / "just now" from an SQLite UTC timestamp. */
function sinceLabel(createdAt: string | null): string {
  if (!createdAt) return "never run";
  const stamp = Date.parse(`${createdAt.replace(" ", "T")}Z`);
  if (Number.isNaN(stamp)) return "never run";
  const minutes = Math.floor((Date.now() - stamp) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

function ScorerChips({
  catalogue,
  enabled,
  onToggle,
}: {
  catalogue: ScorerChip[];
  enabled: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <>
      {catalogue.map((chip) => {
        // A scorer whose index scan is disabled on this machine can never be
        // computed here: the chip is inert, not merely unselected.
        const off = !chip.available;
        const on = !off && enabled.has(chip.id);
        return (
          <button
            key={chip.id}
            disabled={off}
            title={off ? "Enable its scan in Settings → This machine" : ""}
            onClick={() => onToggle(chip.id)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              borderRadius: 20,
              cursor: off ? "default" : "pointer",
              fontSize: 11.5,
              background: on ? colors.raised : "transparent",
              border: `1px solid ${on ? colors.borderControl : colors.border}`,
              color: off
                ? "#4a4d58"
                : on
                  ? colors.textSecondary
                  : colors.textFaint,
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: on ? scorerColor(chip.id) : colors.textFaint,
              }}
            />
            {chip.label}
            {off && " · off on this machine"}
          </button>
        );
      })}
    </>
  );
}

function RunningCard({ pct, stage }: { pct: number; stage: string }) {
  return (
    <div
      style={{
        ...cardStyle,
        background: colors.accentTintAlt,
        border: `1px solid ${colors.accentBorder}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 10,
        }}
      >
        <span style={{ color: colors.accent }}>◉</span>
        <span style={{ fontSize: 12.5, color: colors.textSecondary, flex: 1 }}>
          {stage || "Preparing…"}
        </span>
        <span
          style={{ fontFamily: font.mono, fontSize: 11, color: colors.accent }}
        >
          {pct}%
        </span>
      </div>
      <div
        style={{
          height: 5,
          borderRadius: 3,
          background: colors.border,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: colors.accent,
            transition: "width .2s",
          }}
        />
      </div>
    </div>
  );
}

function Recommendations({ items }: { items: ReportRecommendation[] }) {
  if (items.length === 0) return null;
  return (
    <div
      style={{
        ...cardStyle,
        background: "#141b20",
        border: "1px solid #24313a",
      }}
    >
      <div
        style={{
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: ".08em",
          fontWeight: 600,
          color: colors.info,
          marginBottom: 8,
        }}
      >
        Recommendations
      </div>
      {items.map((item) => (
        <div
          key={item.head}
          style={{
            display: "flex",
            gap: 8,
            padding: "4px 0",
            fontSize: 11.5,
            color: colors.textMuted,
          }}
        >
          <span style={{ color: colors.info }}>→</span>
          <span>
            <strong style={{ color: colors.textSecondary }}>{item.head}</strong>{" "}
            — {item.body}
          </span>
        </div>
      ))}
    </div>
  );
}

export function QualityReportTab({ datasetId }: { datasetId: number }) {
  const client = useQueryClient();
  const captionType = useUiStore((state) => state.captionType);
  const setView = useUiStore((state) => state.setView);
  const setFocus = useUiStore((state) => state.setFocus);
  const setMediaFocus = useUiStore((state) => state.setMediaFocus);

  const report = useDatasetReport(datasetId);
  const runReport = useRunDatasetReport();
  const resolveIssue = useResolveIssue();
  const unresolveIssue = useUnresolveIssue();
  const removeMedia = useRemoveDatasetMedia();
  const generate = useGenerate();
  const modelStatus = useModelStatus();
  const prompts = usePrompts(modelStatus.data?.type ?? null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [force, setForce] = useState(false);
  const [targetType, setTargetType] = useState(TARGET_TYPES[0]);
  const [picked, setPicked] = useState<string[] | null>(null);

  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const running = job?.state === "queued" || job?.state === "running";

  useEffect(() => {
    if (!job || running) return;
    client.invalidateQueries({ queryKey: ["dataset-report", datasetId] });
    client.invalidateQueries({ queryKey: ["dataset-media", datasetId] });
    client.invalidateQueries({ queryKey: ["datasets"] });
    setJobId(null);
  }, [job, running, client, datasetId]);

  // Memoised: both are fresh arrays on every render otherwise, and a new
  // Set identity each render loops the effects reading `enabled`.
  const catalogue = useMemo(
    () => report.data?.scorer_catalogue ?? [],
    [report.data],
  );
  const enabled = useMemo(() => {
    const usable = new Set(
      catalogue.filter((chip) => chip.available).map((chip) => chip.id),
    );
    const last = report.data?.scorers ?? [];
    const wanted =
      picked ??
      (last.length
        ? last
        : catalogue.filter((chip) => chip.default).map((chip) => chip.id));
    // A previous run (or another machine) may have used a scorer whose scan
    // is off here: never re-select what cannot run.
    return new Set(wanted.filter((id) => usable.has(id)));
  }, [picked, report.data, catalogue]);

  const toggle = (id: string) => {
    const next = new Set(enabled);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setPicked([...next]);
  };

  const run = () => {
    runReport.mutate(
      {
        id: datasetId,
        scorers: [...enabled],
        caption_type: captionType,
        target_type: targetType,
        force,
      },
      { onSuccess: (data) => setJobId(data.job_id) },
    );
  };

  const stored: DatasetReport | null = report.data?.report ?? null;
  const resolutions = report.data?.resolutions ?? {};

  const promptText =
    prompts.data?.prompts.find((item) => item.title === prompts.data?.selected)
      ?.prompt ?? "";
  const recaptionDisabled = !modelStatus.data?.loaded
    ? "Load a model in the Caption tab first"
    : !promptText
      ? "Select a prompt preset in the Caption tab first"
      : null;

  const resolve = (issue: ReportIssue, resolution: ResolutionKind) =>
    resolveIssue.mutate({
      id: datasetId,
      issue_key: issue.key,
      resolution,
      fingerprint: issue.fingerprint,
    });

  const remove = (issue: ReportIssue, mediaIds: number[]) => {
    removeMedia.mutate(
      { id: datasetId, media_ids: mediaIds },
      { onSuccess: () => resolve(issue, "removed") },
    );
  };

  const recaption = (issue: ReportIssue, mediaId: number) => {
    if (recaptionDisabled) return;
    generate.mutate(
      {
        dataset_id: datasetId,
        caption_type: captionType,
        media_ids: [mediaId],
        exclude_ids: null,
        prompt: promptText,
        temperature: prompts.data?.temperature ?? 0.7,
        seed: null,
        think_mode: prompts.data?.think_mode ?? "auto",
        image_size: RECAPTION_IMAGE_SIZE,
        review_after: false,
        ground_after: false,
        // An explicit per-media re-caption always rewrites, filled or not.
        recaption: true,
      },
      { onSuccess: () => resolve(issue, "recaptioned") },
    );
  };

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          ...cardStyle,
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <span
          style={{
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: ".08em",
            fontWeight: 600,
            color: colors.textMuted,
          }}
        >
          Scorers
        </span>
        <ScorerChips catalogue={catalogue} enabled={enabled} onToggle={toggle} />

        <select
          value={targetType}
          onChange={(event) => setTargetType(event.target.value)}
          title="Drives the framing and size recommendations"
          style={{
            padding: "4px 8px",
            borderRadius: radii.control,
            border: `1px solid ${colors.borderControl}`,
            background: colors.input,
            color: colors.textSecondary,
            fontSize: 11.5,
          }}
        >
          {TARGET_TYPES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>

        <span style={{ flex: 1 }} />

        <label
          style={{
            fontSize: 11,
            color: colors.textMuted,
            display: "inline-flex",
            gap: 5,
            alignItems: "center",
          }}
        >
          <input
            type="checkbox"
            checked={force}
            onChange={(event) => setForce(event.target.checked)}
          />
          force
        </label>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textFaint,
          }}
        >
          last run {sinceLabel(report.data?.created_at ?? null)}
          {stored ? ` · ${stored.images} media` : ""}
          {report.data?.duration_s
            ? ` · ${report.data.duration_s.toFixed(0)} s`
            : ""}
        </span>
        <Button
          variant="accent"
          disabled={running || enabled.size === 0}
          onClick={run}
        >
          ↻ Re-run evaluation
        </Button>
      </div>

      {running && job && (
        <RunningCard pct={job.pct} stage={job.sub} />
      )}

      {!stored && !running && (
        <div
          style={{
            ...cardStyle,
            textAlign: "center",
            padding: 34,
            color: colors.textFaint,
            fontSize: 12,
          }}
        >
          ◌ This dataset has never been evaluated. Pick your scorers and run
          the evaluation.
        </div>
      )}

      {stored && (
        <>
          <QualityScoreRow report={stored} />
          <QualityCharts report={stored} />
          <CompositionMap report={stored} />
          <FlaggedMedia
            report={stored}
            resolutions={resolutions}
            onResolve={resolve}
            onReopen={(issue) =>
              unresolveIssue.mutate({ id: datasetId, issue_key: issue.key })
            }
            onRemove={remove}
            onRecaption={recaption}
            onOpenMedia={(mediaId) => {
              setMediaFocus(String(mediaId));
              setView("media");
            }}
            onOpenCaption={(mediaId) => {
              setFocus(String(mediaId));
              setView("caption");
            }}
            recaptionDisabled={recaptionDisabled}
          />
          <Recommendations items={stored.recommendations} />
        </>
      )}
    </div>
  );
}
