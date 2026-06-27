/**
 * "Tags Score" (Media tab, below Tag grounding).
 *
 * The tag counterpart of the Caption tab's score card: the media's tags,
 * joined into one comma-separated text, scored *whole* against the image by
 * the three reference-free encoders (SigLIP2 + CLIP + BLIP). Distinct from
 * Tag grounding, which scores each tag on its own — here it is the tag set
 * taken together. Like the caption card it never runs a model implicitly and
 * watches the background job to its end before refreshing the panel.
 */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useMediaInvalidator, useScoreMediaTags } from "../../api/hooks";
import type { CaptionScoreLine } from "../../api/types";
import { colors, font, qualityColor, radii } from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";

function useScoreJob(onDone: () => void) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const running = job?.state === "queued" || job?.state === "running";
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (!jobId || !job || running) return;
    if (job.state === "error") setError(job.error || "scoring failed");
    else doneRef.current();
    setJobId(null);
  }, [jobId, job, running]);

  return {
    running,
    error,
    sub: job?.sub ?? "",
    start: (id: string) => {
      setError(null);
      setJobId(id);
    },
  };
}

export function MediaTagScoreCard({
  mediaKey,
  lines,
  disabled,
}: {
  mediaKey: string;
  lines: CaptionScoreLine[];
  /** Videos have no tag score — an encoder scores a still, not a clip. */
  disabled: boolean;
}) {
  const score = useScoreMediaTags();
  const invalidate = useMediaInvalidator();
  const job = useScoreJob(invalidate);

  if (disabled) return null;
  const scored = lines.some((line) => line.score != null);

  const run = () =>
    score.mutate(
      { key: mediaKey },
      { onSuccess: (data) => job.start(data.job_id) },
    );

  return (
    <div
      style={{
        marginTop: 12,
        display: "flex",
        flexDirection: "column",
        gap: 9,
        padding: "10px 11px",
        borderRadius: radii.card,
        border: `1px solid ${colors.groundingBorder}`,
        background: colors.groundingBg,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color: colors.grounding }}>◎</span>
        <span
          style={{
            fontSize: 11.5,
            fontWeight: 600,
            color: colors.grounding,
            flex: 1,
          }}
        >
          Tags Score
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 9.5,
            color: colors.textFaint,
          }}
        >
          zero-reference
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {lines.map((line) => (
          <Line key={line.kind} line={line} />
        ))}
      </div>

      {!scored && (
        <div
          style={{ fontSize: 10.5, color: colors.textMuted, lineHeight: 1.45 }}
        >
          Never scored. The tags are joined into one text and rated against the
          image by each encoder — a cosine, no LLM.
        </div>
      )}

      <ScoreButton disabled={job.running || score.isPending} onClick={run}>
        {job.running
          ? job.sub || "Scoring…"
          : scored
            ? "↻ Re-score tags"
            : "◎ Score these tags"}
      </ScoreButton>
      {job.error && (
        <div
          style={{
            fontSize: 10.5,
            color: colors.danger,
            lineHeight: 1.4,
            wordBreak: "break-word",
          }}
        >
          ⚠ {job.error}
        </div>
      )}
    </div>
  );
}

function Line({ line }: { line: CaptionScoreLine }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0" }}>
      <span style={{ fontSize: 11.5, color: colors.text, flex: 1 }}>
        {line.label}
      </span>
      {line.stale && (
        <span
          style={{ fontFamily: font.mono, fontSize: 9, color: colors.warn }}
          title="Tags changed or another checkpoint — re-score to refresh"
        >
          stale
        </span>
      )}
      <span
        style={{
          fontFamily: font.mono,
          fontSize: 15,
          fontWeight: 600,
          minWidth: 34,
          textAlign: "right",
          color: line.score == null ? colors.textFaint : qualityColor(line.score),
        }}
      >
        {line.score == null ? "—" : Math.round(line.score)}
      </span>
    </div>
  );
}

function ScoreButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        width: "100%",
        padding: "6px 0",
        borderRadius: radii.control,
        border: `1px solid ${colors.groundingBorder}`,
        background: colors.groundingBtn,
        color: colors.grounding,
        fontSize: 11.5,
        fontWeight: 600,
        fontFamily: font.sans,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.55 : 1,
      }}
      onMouseEnter={(event) => {
        if (!disabled) {
          event.currentTarget.style.background = colors.groundingBtnHover;
        }
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.background = colors.groundingBtn;
      }}
    >
      {children}
    </button>
  );
}
