/**
 * Datasets → Caption score: reference-free caption quality across a dataset.
 *
 * The dataset-wide companion to the Caption tab's per-image score card. One
 * job scores every caption with the three encoders (each model loaded once
 * for the whole set), and this tab ranks the media worst-first so the
 * captions dragging the dataset average down surface at the top. Clicking a
 * row opens that caption in the Caption tab.
 */

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useCaptionScoreReport,
  useScoreDataset,
} from "../../api/hooks";
import type { CaptionScoreReportMedia } from "../../api/types";
import { colors, font, qualityColor, radii } from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";
import { useUiStore } from "../../store/uiStore";
import { Button } from "../atoms";

const cardStyle = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.card,
  padding: 14,
} as const;

// Shared column widths so the header labels line up with each row's cells.
const RANK_W = 22;
const THUMB_W = 40;
const KIND_W = 46;
const MEAN_W = 40;

export function CaptionScoreReportTab({ datasetId }: { datasetId: number }) {
  const client = useQueryClient();
  const captionType = useUiStore((state) => state.captionType);
  const setView = useUiStore((state) => state.setView);
  const setFocus = useUiStore((state) => state.setFocus);

  const report = useCaptionScoreReport(datasetId, captionType);
  const scoreDataset = useScoreDataset();
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const running = job?.state === "queued" || job?.state === "running";

  useEffect(() => {
    if (!jobId || !job || running) return;
    client.invalidateQueries({
      queryKey: ["caption-score-report", datasetId],
    });
    client.invalidateQueries({ queryKey: ["media-detail"] });
    setJobId(null);
  }, [jobId, job, running, client, datasetId]);

  const data = report.data;
  const kinds = data?.kinds ?? [];

  const run = () =>
    scoreDataset.mutate(
      { dataset_id: datasetId, caption_type: captionType },
      { onSuccess: (result) => setJobId(result.job_id) },
    );

  const openCaption = (key: string) => {
    setFocus(key);
    setView("caption");
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
          gap: 14,
          flexWrap: "wrap",
        }}
      >
        <div>
          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: ".08em",
              fontWeight: 600,
              color: colors.textMuted,
            }}
          >
            Dataset average · {captionType}
          </div>
          <div
            style={{
              fontFamily: font.mono,
              fontSize: 26,
              fontWeight: 700,
              lineHeight: 1.1,
              color:
                data?.overall == null
                  ? colors.textFaint
                  : qualityColor(data.overall),
            }}
          >
            {data?.overall == null ? "—" : Math.round(data.overall)}
          </div>
        </div>

        <div style={{ display: "flex", gap: 14 }}>
          {kinds.map((kind) => (
            <div key={kind.kind} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 10.5, color: colors.textMuted }}>
                {kind.label}
              </div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: 16,
                  fontWeight: 600,
                  color:
                    data?.averages[kind.kind] == null
                      ? colors.textFaint
                      : qualityColor(data.averages[kind.kind]!),
                }}
              >
                {data?.averages[kind.kind] == null
                  ? "—"
                  : Math.round(data.averages[kind.kind]!)}
              </div>
            </div>
          ))}
        </div>

        <span style={{ flex: 1 }} />

        <span
          style={{ fontFamily: font.mono, fontSize: 10, color: colors.textFaint }}
        >
          {data ? `${data.scored_media} / ${data.total_media} scored` : ""}
        </span>
        <Button variant="accent" disabled={running} onClick={run}>
          {running
            ? job?.sub || "Scoring…"
            : data && data.scored_media > 0
              ? "↻ Re-score all"
              : "◎ Score all captions"}
        </Button>
      </div>

      {running && job && (
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
              gap: 10,
              alignItems: "center",
              marginBottom: 10,
            }}
          >
            <span style={{ color: colors.accent }}>◉</span>
            <span
              style={{ fontSize: 12.5, color: colors.textSecondary, flex: 1 }}
            >
              {job.sub || "Preparing…"}
            </span>
            <span
              style={{
                fontFamily: font.mono,
                fontSize: 11,
                color: colors.accent,
              }}
            >
              {job.pct}%
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
                width: `${job.pct}%`,
                height: "100%",
                background: colors.accent,
                transition: "width .2s",
              }}
            />
          </div>
        </div>
      )}

      {data && data.scored_media === 0 && !running && (
        <div
          style={{
            ...cardStyle,
            textAlign: "center",
            padding: 34,
            color: colors.textFaint,
            fontSize: 12,
          }}
        >
          ◌ No caption scored yet. Run the caption score to rank this dataset’s
          captions and surface the weakest.
        </div>
      )}

      {data && data.scored_media > 0 && (
        <div style={{ ...cardStyle, padding: 0, overflow: "hidden" }}>
          <HeaderRow kinds={kinds} />
          {data.media.map((media, index) => (
            <Row
              key={media.key}
              rank={index + 1}
              media={media}
              kinds={kinds}
              onClick={() => openCaption(media.key)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function HeaderRow({ kinds }: { kinds: { kind: string; label: string }[] }) {
  const cell = {
    fontSize: 9.5,
    letterSpacing: ".05em",
    textTransform: "uppercase",
    fontWeight: 600,
    color: colors.textMuted,
  } as const;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "7px 12px",
        borderBottom: `1px solid ${colors.border}`,
        background: colors.toolbar,
        position: "sticky",
        top: 0,
        zIndex: 1,
      }}
    >
      <span style={{ ...cell, width: RANK_W, textAlign: "right" }}>#</span>
      <span style={{ width: THUMB_W, flex: "none" }} />
      <span style={{ ...cell, flex: 1, minWidth: 0 }}>Media</span>
      <div style={{ display: "flex", gap: 12 }}>
        {kinds.map((kind) => (
          <span
            key={kind.kind}
            title={kind.label}
            style={{
              ...cell,
              width: KIND_W,
              textAlign: "right",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {kind.label.split(" ")[0]}
          </span>
        ))}
      </div>
      <span style={{ ...cell, width: MEAN_W, textAlign: "right" }}>Avg</span>
    </div>
  );
}

function Row({
  rank,
  media,
  kinds,
  onClick,
}: {
  rank: number;
  media: CaptionScoreReportMedia;
  kinds: { kind: string; label: string }[];
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "8px 12px",
        borderBottom: `1px solid ${colors.border}`,
        cursor: "pointer",
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.background = colors.raised;
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = "transparent";
      }}
    >
      <span
        style={{
          fontFamily: font.mono,
          fontSize: 11,
          color: colors.textFaint,
          width: RANK_W,
          textAlign: "right",
        }}
      >
        {rank}
      </span>
      <img
        src={`/api/media/${media.key}/thumb`}
        alt={media.name}
        style={{
          width: THUMB_W,
          height: THUMB_W,
          borderRadius: 6,
          objectFit: "cover",
          flex: "none",
        }}
      />
      <span
        style={{
          fontSize: 12,
          color: colors.text,
          flex: 1,
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {media.name}
      </span>
      <div style={{ display: "flex", gap: 12 }}>
        {kinds.map((kind) => {
          const value = media.scores[kind.kind];
          return (
            <span
              key={kind.kind}
              title={`${kind.label}${media.stale[kind.kind] ? " · stale" : ""}`}
              style={{
                fontFamily: font.mono,
                fontSize: 12,
                width: KIND_W,
                textAlign: "right",
                opacity: media.stale[kind.kind] ? 0.5 : 1,
                color:
                  value == null ? colors.textFaint : qualityColor(value),
              }}
            >
              {value == null ? "—" : Math.round(value)}
            </span>
          );
        })}
      </div>
      <span
        style={{
          fontFamily: font.mono,
          fontSize: 16,
          fontWeight: 700,
          width: MEAN_W,
          textAlign: "right",
          color: qualityColor(media.mean),
        }}
      >
        {Math.round(media.mean)}
      </span>
    </div>
  );
}
