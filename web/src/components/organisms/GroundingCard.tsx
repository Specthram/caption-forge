/**
 * The two grounding entry points: one card per tab, one shared modal.
 *
 * `CaptionGroundingCard` (Caption tab) summarises the focused caption's
 * SigLIP grounding; `TagGroundingCard` (Media tab) does the same for the
 * focused media's tags. Both open the same modal, in their own mode.
 *
 * Neither card ever runs a model implicitly: a caption that was never
 * grounded shows a run button, not empty tiles. Grounding a caption also
 * needs a VLM loaded (it decomposes before scoring), which is why that
 * button and the tag one are worded differently.
 */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useGroundCaption,
  useGroundTags,
  useGroundingConfig,
  useModelStatus,
  useTagGrounding,
} from "../../api/hooks";
import type { GroundingSummary } from "../../api/types";
import { colors, font, groundingColor, radii } from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";
import { useUiStore } from "../../store/uiStore";

/**
 * Track a grounding job the card just submitted, to its end.
 *
 * A grounding run is a background job on the single-model queue, not a
 * request that resolves with the answer: the POST only returns a `job_id`.
 * Without this the card would fire the job and never react — the button
 * would flick back and, on success or failure alike, nothing would change.
 * So the job is watched through the jobs store (fed by the `/ws/jobs`
 * socket); on `done` the caller's queries are invalidated so the tiles
 * appear, and on `error` its message is surfaced instead of vanishing.
 */
function useGroundingJob(onDone: () => void) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const running = job?.state === "queued" || job?.state === "running";
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (!jobId || !job || running) return;
    if (job.state === "error") setError(job.error || "grounding failed");
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

interface CaptionGroundingCardProps {
  mediaKey: string;
  name: string;
  datasetId: number;
  captionType: string;
  summary: GroundingSummary | null;
  /** Videos have no grounding — SigLIP scores a picture, not a clip. */
  disabled: boolean;
}

export function CaptionGroundingCard({
  mediaKey,
  name,
  datasetId,
  captionType,
  summary,
  disabled,
}: CaptionGroundingCardProps) {
  const openGrounding = useUiStore((state) => state.openGrounding);
  const config = useGroundingConfig();
  const modelStatus = useModelStatus();
  const ground = useGroundCaption();
  const client = useQueryClient();
  const job = useGroundingJob(() => {
    client.invalidateQueries({ queryKey: ["caption-grid"] });
    client.invalidateQueries({ queryKey: ["media-detail"] });
    client.invalidateQueries({ queryKey: ["caption-grounding"] });
  });

  if (disabled) return null;
  const threshold = config.data?.threshold_caption ?? 55;
  // Grounding a caption decomposes it with a VLM before SigLIP scores the
  // claims. The job can run either against the model already loaded, or by
  // auto-loading the default claim model set in Settings — so grounding is
  // possible whenever one of the two exists.
  const vlmLoaded = modelStatus.data?.loaded ?? false;
  const claimModel = config.data?.claim_model ?? "";
  const canGround = vlmLoaded || claimModel !== "";

  const runGrounding = () =>
    ground.mutate(
      { key: mediaKey, dataset_id: datasetId, caption_type: captionType },
      { onSuccess: (data) => job.start(data.job_id) },
    );

  return (
    <Card>
      <Header meta="LLM claims → SigLIP">Grounding — SigLIP2</Header>

      {summary ? (
        <>
          <div style={{ display: "flex", gap: 6 }}>
            <Tile value={summary.validated} label="validated" tone={colors.ok} />
            <Tile
              value={summary.flagged}
              label="to check"
              tone={colors.danger}
            />
            <Tile
              value={`${summary.coverage}%`}
              label="covered"
              tone={colors.text}
            />
          </div>
          {summary.stale && (
            <div
              style={{
                fontFamily: font.mono,
                fontSize: 9.5,
                color: colors.warn,
              }}
            >
              scored by another checkpoint — re-run to compare
            </div>
          )}
          <GroundingButton
            onClick={() =>
              openGrounding("caption", mediaKey, name, threshold)
            }
          >
            ◈ Open grounding &amp; heatmap
          </GroundingButton>
          {canGround && (
            <GroundingButton
              subtle
              disabled={job.running || ground.isPending}
              onClick={runGrounding}
            >
              {job.running ? job.sub || "Grounding…" : "↻ Re-ground"}
            </GroundingButton>
          )}
          {job.error && <ErrorLine>{job.error}</ErrorLine>}
        </>
      ) : (
        <>
          <Hint>
            {vlmLoaded
              ? "Never grounded. The loaded VLM splits the caption into claims, then SigLIP scores each against the pixels."
              : claimModel
                ? `Never grounded. Splits the caption with ${claimModel} (auto-loaded), then SigLIP scores each claim.`
                : "Load a VLM, or set a claim-splitting model in Settings — grounding first decomposes the caption, then SigLIP scores it."}
          </Hint>
          <GroundingButton
            disabled={!canGround || job.running || ground.isPending}
            onClick={runGrounding}
          >
            {job.running
              ? job.sub || "Grounding…"
              : canGround
                ? "◈ Ground this caption"
                : "◈ VLM required"}
          </GroundingButton>
          {job.error && <ErrorLine>{job.error}</ErrorLine>}
        </>
      )}
    </Card>
  );
}

interface TagGroundingCardProps {
  mediaKey: string;
  name: string;
  /** Videos have no tag grounding either. */
  disabled: boolean;
}

export function TagGroundingCard({
  mediaKey,
  name,
  disabled,
}: TagGroundingCardProps) {
  const openGrounding = useUiStore((state) => state.openGrounding);
  const config = useGroundingConfig();
  const grounding = useTagGrounding(mediaKey, !disabled);
  const ground = useGroundTags();
  const client = useQueryClient();
  const job = useGroundingJob(() => {
    client.invalidateQueries({ queryKey: ["tag-grounding"] });
    client.invalidateQueries({ queryKey: ["media-full"] });
  });

  if (disabled) return null;

  const runGrounding = () =>
    ground.mutate(
      { media_ids: [Number(mediaKey)] },
      { onSuccess: (data) => job.start(data.job_id) },
    );

  const threshold = grounding.data?.threshold ?? config.data?.threshold_tags ?? 55;
  const scored = (grounding.data?.tags ?? []).filter(
    (tag) => tag.score != null,
  );
  const confirmed = scored.filter((tag) => (tag.score ?? 0) >= threshold);
  const hallucinated = scored.length - confirmed.length;
  // "How much of this media's tagging is backed by its pixels."
  const quality = scored.length
    ? Math.round((100 * confirmed.length) / scored.length)
    : 0;

  return (
    <Card>
      <Header meta="tag → SigLIP">Tag grounding — SigLIP2</Header>

      {scored.length > 0 ? (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: 20,
                  fontWeight: 600,
                  color: groundingColor(quality, threshold),
                  lineHeight: 1.1,
                }}
              >
                {quality}%
              </div>
              <div style={{ fontSize: 9.5, color: colors.textMuted }}>
                tags confirmed
              </div>
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
              <div
                style={{
                  height: 6,
                  borderRadius: 3,
                  background: colors.border,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${quality}%`,
                    background: groundingColor(quality, threshold),
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
              <div style={{ fontSize: 10.5, color: colors.textMuted }}>
                <b style={{ color: hallucinated ? colors.danger : colors.ok }}>
                  {hallucinated}
                </b>{" "}
                tag(s) likely hallucinated
              </div>
            </div>
          </div>
          <GroundingButton
            onClick={() => openGrounding("media", mediaKey, name, threshold)}
          >
            ◫ Verify tags on the image
          </GroundingButton>
          <GroundingButton
            subtle
            disabled={job.running || ground.isPending}
            onClick={runGrounding}
          >
            {job.running ? job.sub || "Grounding…" : "↻ Re-ground tags"}
          </GroundingButton>
          {job.error && <ErrorLine>{job.error}</ErrorLine>}
        </>
      ) : (
        <>
          <Hint>
            No tag scored yet. Each tag is checked on its own through a fixed
            pre-prompt — no LLM involved.
          </Hint>
          <GroundingButton
            disabled={job.running || ground.isPending}
            onClick={runGrounding}
          >
            {job.running ? job.sub || "Grounding…" : "◫ Ground these tags"}
          </GroundingButton>
          {job.error && <ErrorLine>{job.error}</ErrorLine>}
        </>
      )}
    </Card>
  );
}

// -- Shared chrome ------------------------------------------------------------

function Card({ children }: { children: ReactNode }) {
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
      {children}
    </div>
  );
}

function Header({ children, meta }: { children: ReactNode; meta: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ color: colors.grounding }}>◈</span>
      <span
        style={{
          fontSize: 11.5,
          fontWeight: 600,
          color: colors.grounding,
          flex: 1,
        }}
      >
        {children}
      </span>
      <span
        style={{
          fontFamily: font.mono,
          fontSize: 9.5,
          color: colors.textFaint,
        }}
      >
        {meta}
      </span>
    </div>
  );
}

function Tile({
  value,
  label,
  tone,
}: {
  value: number | string;
  label: string;
  tone: string;
}) {
  return (
    <div
      style={{
        flex: 1,
        padding: "6px 0",
        textAlign: "center",
        borderRadius: radii.control,
        border: `1px solid ${colors.border}`,
        background: colors.input,
      }}
    >
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 15,
          fontWeight: 600,
          color: tone,
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: 9, color: colors.textMuted }}>{label}</div>
    </div>
  );
}

function Hint({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: 10.5, color: colors.textMuted, lineHeight: 1.45 }}>
      {children}
    </div>
  );
}

/** A failed grounding job's message, shown on the card instead of vanishing. */
function ErrorLine({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10.5,
        color: colors.danger,
        lineHeight: 1.4,
        wordBreak: "break-word",
      }}
    >
      ⚠ {children}
    </div>
  );
}

function GroundingButton({
  children,
  onClick,
  disabled,
  subtle,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  /** A quieter secondary action (re-ground) beside the primary button. */
  subtle?: boolean;
}) {
  const idle = subtle ? "transparent" : colors.groundingBtn;
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
        background: idle,
        color: colors.grounding,
        fontSize: subtle ? 10.5 : 11.5,
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
        event.currentTarget.style.background = idle;
      }}
    >
      {children}
    </button>
  );
}
