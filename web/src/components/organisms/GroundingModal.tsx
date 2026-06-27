/**
 * The shared grounding modal: image + thermal overlay, element list, threshold.
 *
 * Opened from either grounding card. In `caption` mode its elements are the
 * caption's LLM-extracted claims; in `media` mode they are the media's tags,
 * each scored through a fixed pre-prompt. The layout is the same because the
 * question is: *which of these does the image actually support, and where?*
 *
 * Two data paths meet here. The **scores** are read from the database (a
 * previous grounding job wrote them) and re-thresholded live — moving the
 * slider never re-runs a model. The **heat maps** are not stored: opening the
 * modal enqueues one job that rebuilds every element's map in a single
 * forward pass, and its result arrives through the job queue. So the bars are
 * instant and the overlay fades in a moment later.
 */

import { useEffect, useMemo, useState } from "react";
import {
  useCaptionGrounding,
  useCaptionHeat,
  useGroundingConfig,
  useHeatResult,
  useRejectClaim,
  useRemoveGroundedTag,
  useTagGrounding,
  useTagHeat,
} from "../../api/hooks";
import type { ClaimKind, HeatElement } from "../../api/types";
import { UNRELIABLE_KINDS } from "../../api/types";
import { coveragePct, decodeHeat, unionGrid } from "../../design/heat";
import {
  colors,
  font,
  groundingColor,
  radii,
  shadow,
} from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";
import { useUiStore } from "../../store/uiStore";
import { HeatmapCanvas } from "../molecules/HeatmapCanvas";

/**
 * Floor on a hovered element's overlay opacity. A hallucinated claim scores
 * near zero, and painting it at `score/100` would show nothing at all — but
 * "nowhere in this image supports it" is exactly what the user came to see,
 * so its (weak) best-matching region is still drawn, just faintly.
 */
const MIN_HOVER_STRENGTH = 0.78;
const COVERAGE_STRENGTH = 0.85;

/** One row of the modal, whichever mode produced it. */
interface Element {
  id: number;
  label: string;
  kind: ClaimKind | null;
  score: number;
  rejected: boolean;
  grid: Uint8Array | null;
  side: number;
}

export function GroundingModal() {
  const grounding = useUiStore((state) => state.grounding);
  const setGrounding = useUiStore((state) => state.setGrounding);
  const close = useUiStore((state) => state.closeGrounding);
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);

  const { open, mode, key, name, threshold, hover, coverage } = grounding;
  const isCaption = mode === "caption";

  const config = useGroundingConfig();
  const captionData = useCaptionGrounding(
    key,
    datasetId,
    captionType,
    open && isCaption,
  );
  const tagData = useTagGrounding(key, open && !isCaption);

  const heat = useHeatMaps(open, mode, key, datasetId, captionType);
  const rejectClaim = useRejectClaim();
  const removeTag = useRemoveGroundedTag();

  // Escape is handled centrally in useKeyboard, which owns the overlay
  // dismissal chain (compare → zoom → search → grounding → jobs).

  const elements = useMemo<Element[]>(() => {
    const grids = new Map<number, HeatElement>(
      (heat.elements ?? []).map((element) => [element.id, element]),
    );
    const decode = (id: number) => {
      const element = grids.get(id);
      if (!element?.heat) return { grid: null, side: 0 };
      return { grid: decodeHeat(element.heat), side: element.side };
    };
    if (isCaption) {
      return (captionData.data?.grounding?.claims ?? []).map((claim) => ({
        id: claim.id,
        label: claim.text,
        kind: claim.kind,
        score: claim.score,
        rejected: claim.rejected,
        ...decode(claim.id),
      }));
    }
    return (tagData.data?.tags ?? []).map((tag) => ({
      id: tag.id,
      label: tag.name,
      kind: null,
      score: tag.score ?? grids.get(tag.id)?.score ?? 0,
      rejected: false,
      ...decode(tag.id),
    }));
  }, [isCaption, captionData.data, tagData.data, heat.elements]);

  const active = elements.filter((element) => !element.rejected);
  const validated = active.filter((element) => element.score >= threshold);
  const flagged = active.length - validated.length;

  const unionOfValidated = useMemo(
    () =>
      unionGrid(
        validated
          .map((element) => element.grid)
          .filter((grid): grid is Uint8Array => grid != null),
      ),
    [validated],
  );
  const coveredPct = coveragePct(unionOfValidated);

  const hovered = elements.find((element) => element.id === hover) ?? null;
  const shownGrid = hovered?.grid ?? (coverage ? unionOfValidated : null);
  const shownSide =
    hovered?.side ??
    validated.find((element) => element.side > 0)?.side ??
    0;
  const strength = hovered
    ? Math.max(MIN_HOVER_STRENGTH, hovered.score / 100)
    : COVERAGE_STRENGTH;

  if (!open || !key) return null;

  const modelId = config.data?.model_id ?? "";
  const chip = hovered
    ? `${hovered.label} · ${Math.round(hovered.score)}%`
    : coverage
      ? `Coverage ${coveredPct}%`
      : null;
  const chipColor = hovered
    ? groundingColor(hovered.score, threshold, hovered.rejected)
    : colors.ok;

  return (
    <div
      onClick={close}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 70,
        background: "rgba(10,11,14,0.66)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 980,
          maxWidth: "94vw",
          height: 640,
          maxHeight: "90vh",
          background: colors.panel,
          border: `1px solid ${colors.borderHover}`,
          borderRadius: radii.modal,
          boxShadow: shadow.modal,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <header
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "13px 16px",
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <span style={{ color: colors.grounding }}>◈</span>
          <span style={{ fontSize: 13.5, fontWeight: 700 }}>
            {isCaption ? "Caption grounding" : "Tag grounding"} — SigLIP2
          </span>
          <span
            style={{
              flex: 1,
              textAlign: "right",
              fontFamily: font.mono,
              fontSize: 10.5,
              color: colors.textFaint,
            }}
          >
            {name} · {modelId.replace("google/", "")}
          </span>
          <button
            type="button"
            onClick={close}
            style={{
              width: 22,
              height: 22,
              borderRadius: 5,
              border: "none",
              background: colors.raised,
              color: colors.textMutedAlt,
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        </header>

        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <ImagePane
            mediaKey={key}
            grid={shownGrid}
            side={shownSide}
            strength={strength}
            chip={chip}
            chipColor={chipColor}
            coverage={coverage}
            coveredPct={coveredPct}
            pending={heat.pending}
            onCoverage={(next) => setGrounding({ coverage: next })}
          />

          <aside
            style={{
              width: 392,
              flex: "none",
              borderLeft: `1px solid ${colors.border}`,
              background: colors.panel,
              overflowY: "auto",
            }}
          >
            <div
              style={{
                padding: "13px 15px",
                borderBottom: `1px solid ${colors.border}`,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {isCaption ? (
                <div
                  style={{
                    fontSize: 11,
                    color: colors.textMuted,
                    lineHeight: 1.45,
                  }}
                >
                  The LLM split the caption into <b>{elements.length}</b>{" "}
                  claims; SigLIP measures how well each matches the image.
                  Hover a row to locate its region.
                </div>
              ) : (
                <PromptChip template={config.data?.tag_prompt ?? ""} />
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 11, color: colors.textMuted }}>
                  Validation threshold
                </span>
                <span
                  style={{
                    fontFamily: font.mono,
                    fontSize: 12,
                    fontWeight: 600,
                    color: colors.warn,
                  }}
                >
                  {Math.round(threshold)}%
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={threshold}
                  onChange={(event) =>
                    setGrounding({ threshold: Number(event.target.value) })
                  }
                  style={{ flex: 1, accentColor: colors.accent }}
                />
              </div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: 10,
                  color: colors.textFaint,
                }}
              >
                {validated.length} validated · {flagged} to check
              </div>
            </div>

            <div
              style={{
                padding: "12px 15px",
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              {elements.map((element) => (
                <ElementRow
                  key={element.id}
                  element={element}
                  threshold={threshold}
                  isCaption={isCaption}
                  hovered={hover === element.id}
                  onHover={(on) =>
                    setGrounding({ hover: on ? element.id : null })
                  }
                  onAct={() => {
                    if (isCaption) {
                      rejectClaim.mutate({
                        claim_id: element.id,
                        rejected: !element.rejected,
                      });
                    } else {
                      removeTag.mutate({ key, tag_id: element.id });
                    }
                  }}
                />
              ))}
              {elements.length === 0 && (
                <div style={{ fontSize: 11.5, color: colors.textFaint }}>
                  Nothing grounded yet for this media.
                </div>
              )}
            </div>

            <div
              style={{
                margin: "0 15px 15px",
                padding: "9px 11px",
                borderRadius: radii.control,
                border: `1px dashed ${colors.borderControl}`,
                background: colors.input,
                fontSize: 10.5,
                color: colors.textMuted,
                lineHeight: 1.45,
              }}
            >
              SigLIP judges presence, not counting nor position — the amber
              rows (count / spatial) stay indicative.
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

// -- Heat maps ----------------------------------------------------------------

/**
 * Enqueue the heat-map job when the modal opens, then hand back its elements.
 *
 * The maps travel as the job's *result* rather than over the progress socket:
 * one grid per element is far too bulky to broadcast, and it is not progress,
 * it is the output. The job is watched through the jobs store (fed by the
 * WebSocket) and its payload fetched once it reports `done`.
 */
function useHeatMaps(
  open: boolean,
  mode: "caption" | "media",
  key: string | null,
  datasetId: number | null,
  captionType: string,
) {
  const [jobId, setJobId] = useState<string | null>(null);
  const captionHeat = useCaptionHeat();
  const tagHeat = useTagHeat();
  const state = useJobsStore((store) => (jobId ? store.jobs[jobId]?.state : undefined));
  const result = useHeatResult(jobId, state === "done");

  useEffect(() => {
    if (!open || !key) {
      setJobId(null);
      return;
    }
    const onDone = (response: { job_id: string }) => setJobId(response.job_id);
    if (mode === "caption") {
      if (datasetId == null) return;
      captionHeat.mutate(
        { key, dataset_id: datasetId, caption_type: captionType },
        { onSuccess: onDone },
      );
    } else {
      tagHeat.mutate({ media_ids: [Number(key)] }, { onSuccess: onDone });
    }
    // The heat job is fired once per (open, media, mode) — re-running it on
    // every mutation-object identity change would queue a GPU job per render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode, key, datasetId, captionType]);

  return {
    elements: result.data?.result.elements ?? null,
    pending: !!jobId && state !== "done",
  };
}

// -- Panes --------------------------------------------------------------------

function ImagePane({
  mediaKey,
  grid,
  side,
  strength,
  chip,
  chipColor,
  coverage,
  coveredPct,
  pending,
  onCoverage,
}: {
  mediaKey: string;
  grid: Uint8Array | null;
  side: number;
  strength: number;
  chip: string | null;
  chipColor: string;
  coverage: boolean;
  coveredPct: number;
  pending: boolean;
  onCoverage: (next: boolean) => void;
}) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        background: colors.app,
      }}
    >
      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 18,
        }}
      >
        {/*
          The wrapper shrinks to the rendered image, so the canvas laid over it
          covers exactly the pixels the patch grid describes. `contain`, never
          `cover`: SigLIP squashed the whole image into its square input, so a
          crop here would slide the heat off its subject.
        */}
        <div
          style={{
            position: "relative",
            display: "inline-block",
            maxWidth: "100%",
            maxHeight: "100%",
            borderRadius: 10,
            overflow: "hidden",
            border: `1px solid ${colors.border}`,
          }}
        >
          <img
            src={`/api/media/${mediaKey}/file`}
            alt=""
            style={{
              display: "block",
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
            }}
          />
          <HeatmapCanvas grid={grid} side={side} strength={strength} />
          {chip && (
            <div
              style={{
                position: "absolute",
                top: 10,
                left: 10,
                padding: "4px 9px",
                borderRadius: 7,
                background: "rgba(16,17,22,0.82)",
                backdropFilter: "blur(6px)",
                border: `1px solid ${colors.borderControl}`,
                fontFamily: font.mono,
                fontSize: 10.5,
                color: chipColor,
              }}
            >
              {chip}
            </div>
          )}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "11px 18px",
          borderTop: `1px solid ${colors.border}`,
          background: colors.toolbar,
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11,
            color: colors.textMuted,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={coverage}
            onChange={(event) => onCoverage(event.target.checked)}
            style={{ accentColor: colors.ok }}
          />
          Coverage
        </label>
        <div
          style={{
            flex: 1,
            height: 6,
            borderRadius: 3,
            background: colors.border,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${coveredPct}%`,
              background: `linear-gradient(90deg, ${colors.ok}, #4f9e77)`,
              transition: "width 0.35s ease",
            }}
          />
        </div>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 12,
            fontWeight: 600,
            color: colors.ok,
          }}
        >
          {coveredPct}%
        </span>
        <span style={{ fontSize: 10.5, color: colors.textFaint }}>
          {pending ? "building heatmap…" : "image covered"}
        </span>
      </div>
    </div>
  );
}

function PromptChip({ template }: { template: string }) {
  const [before, after] = template.split("{tag}");
  return (
    <div
      style={{
        fontFamily: font.mono,
        fontSize: 10.5,
        padding: "7px 9px",
        borderRadius: 7,
        border: `1px solid ${colors.border}`,
        background: colors.input,
        color: colors.textMuted,
      }}
    >
      <span style={{ color: colors.grounding }}>prompt</span> = &quot;{before}
      <span style={{ color: colors.warn }}>{"{tag}"}</span>
      {after}&quot;
    </div>
  );
}

function ElementRow({
  element,
  threshold,
  isCaption,
  hovered,
  onHover,
  onAct,
}: {
  element: Element;
  threshold: number;
  isCaption: boolean;
  hovered: boolean;
  onHover: (on: boolean) => void;
  onAct: () => void;
}) {
  const tone = groundingColor(element.score, threshold, element.rejected);
  const validated = !element.rejected && element.score >= threshold;
  const unreliable =
    element.kind != null && UNRELIABLE_KINDS.includes(element.kind);

  const status = element.rejected
    ? isCaption
      ? "⊘ rejected"
      : "⊘ removed"
    : validated
      ? isCaption
        ? "✓ validated"
        : "✓ present"
      : isCaption
        ? "! to check"
        : "! absent / hallucinated";

  return (
    <div
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 7,
        padding: "9px 11px",
        borderRadius: radii.panel,
        border: `1px solid ${hovered ? colors.borderHover : colors.border}`,
        background: hovered ? "#1e2027" : colors.card,
        opacity: element.rejected ? 0.55 : 1,
        transition: "background 0.15s ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            fontSize: 12,
            color: colors.text,
            flex: 1,
            textDecoration: element.rejected ? "line-through" : "none",
          }}
        >
          {element.label}
        </span>
        {element.kind && (
          <span
            title={
              unreliable
                ? "SigLIP scores counting and spatial relations poorly — indicative only."
                : "SigLIP judges presence reliably."
            }
            style={{
              fontFamily: font.mono,
              fontSize: 9.5,
              fontWeight: 600,
              padding: "1px 6px",
              borderRadius: 8,
              color: unreliable ? colors.warn : colors.textMutedAlt,
              background: unreliable
                ? "rgba(224,179,86,0.13)"
                : colors.raised,
            }}
          >
            {element.kind}
          </span>
        )}
      </div>

      <div
        style={{
          height: 8,
          borderRadius: 4,
          background: colors.raised,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${Math.min(100, element.score)}%`,
            background: tone,
            transition: "width 0.3s ease",
          }}
        />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 11,
            fontWeight: 600,
            color: tone,
          }}
        >
          {Math.round(element.score)}%
        </span>
        <span style={{ fontSize: 10.5, color: colors.textMuted, flex: 1 }}>
          {status}
        </span>
        <button
          type="button"
          onClick={onAct}
          style={{
            padding: "3px 8px",
            borderRadius: 5,
            border: `1px solid ${colors.borderControl}`,
            background: "transparent",
            color: colors.textMutedAlt,
            fontSize: 10.5,
            cursor: "pointer",
          }}
        >
          {isCaption
            ? element.rejected
              ? "Restore"
              : "Mark unsupported"
            : "Remove tag"}
        </button>
      </div>
    </div>
  );
}
