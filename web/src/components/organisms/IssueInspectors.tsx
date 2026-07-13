/**
 * The four expandable inspectors of the Quality report's flagged-media
 * card, one per finding kind.
 *
 * NEAR-DUP owns a comparator with four modes: side by side (the auto-picked
 * keeper badged ★ BEST), an A/B flicker at ~1 Hz, a difference composite
 * (``mix-blend-mode: difference`` — bright pixels are the ones that moved)
 * and a full-screen wipe, which reuses the existing ``LookalikeCompare``
 * overlay rather than duplicating its zoom/pan maths.
 */

import { useEffect, useState } from "react";
import { Button, Segmented } from "../atoms";
import { colors, font, qualityColor, radii } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import type {
  CaptionDetail,
  LowQualityDetail,
  NearDupDetail,
  OutlierDetail,
  PairMetrics,
  ReportIssue,
} from "../../api/types";

const FLICKER_MS = 900;

export interface IssueActions {
  remove: (mediaIds: number[]) => void;
  ignore: () => void;
  recaption: (mediaId: number) => void;
  openMedia: (mediaId: number) => void;
  openCaption: (mediaId: number) => void;
  recaptionDisabled: string | null;
}

const thumb = (id: number) => `/api/media/${id}/thumb`;
const original = (id: number) => `/api/media/${id}/file`;

const tileStyle = {
  width: "100%",
  aspectRatio: "4 / 3",
  objectFit: "cover",
  borderRadius: radii.control,
  display: "block",
} as const;

const noteStyle = {
  fontSize: 11,
  color: colors.textMuted,
  lineHeight: 1.5,
} as const;

/**
 * One labelled 0-100 bar. ``invert`` is for the metrics where *low* is
 * good (exposure clipping): the bar still shows the raw value, but the
 * colour reads it the right way round.
 */
function MetricBar({
  label,
  value,
  invert = false,
}: {
  label: string;
  value: number | null;
  invert?: boolean;
}) {
  const color = qualityColor(
    value == null ? null : invert ? 100 - value : value,
  );
  return (
    <div style={{ marginBottom: 7 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10.5,
          color: colors.textMuted,
          marginBottom: 3,
        }}
      >
        <span>{label}</span>
        <span style={{ fontFamily: font.mono, color }}>
          {value == null ? "—" : `${Math.round(value)}%`}
        </span>
      </div>
      <div
        style={{
          height: 4,
          borderRadius: 2,
          background: colors.border,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.max(0, Math.min(100, value ?? 0))}%`,
            height: "100%",
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function PairTile({ metrics, best }: { metrics: PairMetrics; best: boolean }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ position: "relative" }}>
        <img
          src={thumb(metrics.id)}
          alt={metrics.name}
          style={{
            ...tileStyle,
            border: `2px solid ${best ? colors.ok : "transparent"}`,
          }}
        />
        {best && (
          <span
            style={{
              position: "absolute",
              top: 6,
              left: 6,
              padding: "2px 6px",
              borderRadius: radii.chip,
              fontSize: 9.5,
              fontWeight: 700,
              background: colors.ok,
              color: colors.onAccent,
            }}
          >
            ★ BEST
          </span>
        )}
      </div>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 10,
          color: colors.textSecondary,
          margin: "6px 0 6px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {metrics.name} · {metrics.width ?? "?"}×{metrics.height ?? "?"}
      </div>
      <MetricBar label="quality" value={metrics.quality} />
      <MetricBar label="sharpness" value={metrics.sharpness} />
      <MetricBar label="exposure clipping" value={metrics.clipping} invert />
    </div>
  );
}

function Flicker({ ids }: { ids: number[] }) {
  const [first, setFirst] = useState(true);
  useEffect(() => {
    const timer = window.setInterval(
      () => setFirst((value) => !value),
      FLICKER_MS,
    );
    return () => window.clearInterval(timer);
  }, []);
  return (
    <img
      src={original(first ? ids[0] : ids[1])}
      alt=""
      style={{ ...tileStyle, aspectRatio: "16 / 10", objectFit: "contain" }}
    />
  );
}

function Difference({ ids }: { ids: number[] }) {
  return (
    <div
      style={{
        position: "relative",
        aspectRatio: "16 / 10",
        background: "#000",
        borderRadius: radii.control,
        overflow: "hidden",
      }}
    >
      {ids.map((id, index) => (
        <img
          key={id}
          src={original(id)}
          alt=""
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "contain",
            mixBlendMode: index ? "difference" : "normal",
          }}
        />
      ))}
    </div>
  );
}

function SimilarityBar({ detail }: { detail: NearDupDetail }) {
  return (
    <>
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 22,
          fontWeight: 700,
          color: colors.warn,
        }}
      >
        {detail.similarity.toFixed(3)}
      </div>
      <div style={{ ...noteStyle, marginBottom: 8 }}>
        {detail.source} cosine similarity
      </div>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: `linear-gradient(90deg, ${colors.ok}, ${colors.warn}, ${colors.danger})`,
          position: "relative",
          marginBottom: 6,
        }}
      >
        <span
          style={{
            position: "absolute",
            left: `${Math.max(0, Math.min(100, detail.similarity * 100))}%`,
            top: -3,
            width: 2,
            height: 12,
            background: colors.text,
          }}
        />
      </div>
      <div style={noteStyle}>
        Flagged above {detail.threshold.toFixed(2)}. Keeping both teaches the
        model the same picture twice.
      </div>
    </>
  );
}

function NearDupInspector({
  detail,
  actions,
}: {
  detail: NearDupDetail;
  actions: IssueActions;
}) {
  const [mode, setMode] = useState("side");
  const openCompare = useUiStore((state) => state.openCompare);
  const [best, loser] = [detail.best_id, detail.loser_id];
  const byId = new Map(detail.metrics.map((item) => [item.id, item]));
  const bestName = byId.get(best)?.name ?? String(best);
  const loserName = byId.get(loser)?.name ?? String(loser);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 16 }}>
      <div>
        <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
          <Segmented
            value={mode}
            onChange={setMode}
            options={[
              { value: "side", label: "Side by side" },
              { value: "flicker", label: "A/B flicker" },
              { value: "diff", label: "Difference" },
            ]}
          />
          <Button
            onClick={() =>
              openCompare(
                original(best),
                bestName,
                original(loser),
                loserName,
              )
            }
          >
            ⤢ Wipe compare
          </Button>
        </div>
        {mode === "side" && (
          <div style={{ display: "flex", gap: 12 }}>
            {detail.metrics.map((metrics) => (
              <PairTile
                key={metrics.id}
                metrics={metrics}
                best={metrics.id === best}
              />
            ))}
          </div>
        )}
        {mode === "flicker" && <Flicker ids={[best, loser]} />}
        {mode === "diff" && <Difference ids={[best, loser]} />}
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <SimilarityBar detail={detail} />
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Button variant="accent" onClick={() => actions.remove([loser])}>
            ★ Keep best — remove {loserName}
          </Button>
          <Button variant="danger" onClick={() => actions.remove([best])}>
            Remove {bestName} instead
          </Button>
          <Button variant="ghost" onClick={actions.ignore}>
            Ignore — not duplicates
          </Button>
        </div>
      </div>
    </div>
  );
}

function LowQualityInspector({
  mediaId,
  detail,
  actions,
}: {
  mediaId: number;
  detail: LowQualityDetail;
  actions: IssueActions;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <img src={thumb(mediaId)} alt="" style={tileStyle} />
      <div>
        {Object.entries(detail.scores).map(([metric, value]) => (
          <MetricBar key={metric} label={metric} value={value} />
        ))}
        <MetricBar label="sharpness" value={detail.sharpness} />
        <MetricBar label="noise / compression" value={detail.cleanliness} />
        <div style={{ ...noteStyle, margin: "10px 0" }}>
          One weak image carries ~{detail.gradient_share.toFixed(0)}% of the
          gradient steps in this set
          {detail.mean == null
            ? "."
            : `, and sits ${Math.round(detail.mean - detail.quality)} points below the dataset mean.`}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Button variant="danger" onClick={() => actions.remove([mediaId])}>
            Remove from dataset
          </Button>
          <Button variant="ghost" onClick={actions.ignore}>
            Ignore — accept quality
          </Button>
          <Button variant="ghost" onClick={() => actions.openMedia(mediaId)}>
            Open in Media ⧉
          </Button>
        </div>
      </div>
    </div>
  );
}

function OutlierInspector({
  mediaId,
  detail,
  actions,
}: {
  mediaId: number;
  detail: OutlierDetail;
  actions: IssueActions;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <img src={thumb(mediaId)} alt="" style={tileStyle} />
      <div>
        <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 8 }}>
          Nearest neighbors — all far
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {detail.neighbors.map((neighbor) => (
            <div key={neighbor.id} style={{ flex: 1, minWidth: 0 }}>
              <img src={thumb(neighbor.id)} alt="" style={tileStyle} />
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: 9.5,
                  color: colors.textFaint,
                  marginTop: 4,
                }}
              >
                d {neighbor.distance.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
        <div style={{ ...noteStyle, margin: "10px 0" }}>
          Nothing in the dataset looks like it (flagged above{" "}
          {detail.threshold.toFixed(2)}). Off-concept media teach the LoRA a
          subject you did not ask for.
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Button variant="danger" onClick={() => actions.remove([mediaId])}>
            Remove from dataset
          </Button>
          <Button variant="ghost" onClick={actions.ignore}>
            Ignore — intentional variety
          </Button>
          <Button variant="ghost" onClick={() => actions.openMedia(mediaId)}>
            Open in Media ⧉
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Render the caption, striking through every run of the looping phrase. */
function HighlightedCaption({ detail }: { detail: CaptionDetail }) {
  const phrase = detail.phrase;
  if (!phrase) {
    return <span>{detail.text}</span>;
  }
  const parts = detail.text.split(new RegExp(`(${escapeRegExp(phrase)})`, "gi"));
  return (
    <>
      {parts.map((part, index) =>
        part.toLowerCase() === phrase.toLowerCase() ? (
          <span
            key={index}
            style={{
              background: "#2a1715",
              color: colors.danger,
              textDecoration: "line-through",
            }}
          >
            {part}
          </span>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </>
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function CaptionInspector({
  mediaId,
  detail,
  actions,
}: {
  mediaId: number;
  detail: CaptionDetail;
  actions: IssueActions;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 16 }}>
      <img src={thumb(mediaId)} alt="" style={tileStyle} />
      <div>
        <div
          style={{
            padding: 10,
            borderRadius: radii.control,
            background: colors.input,
            border: `1px solid ${colors.border}`,
            fontSize: 11.5,
            lineHeight: 1.6,
            maxHeight: 140,
            overflowY: "auto",
          }}
        >
          <HighlightedCaption detail={detail} />
        </div>
        <div style={{ display: "flex", gap: 6, margin: "8px 0" }}>
          {detail.codes.map((code) => (
            <span
              key={code.code}
              title={code.detail}
              style={{
                fontFamily: font.mono,
                fontSize: 9.5,
                padding: "2px 6px",
                borderRadius: radii.chip,
                background: colors.accentTintAlt,
                color: colors.accent,
              }}
            >
              {code.code}
            </span>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Button
            variant="accent"
            disabled={!!actions.recaptionDisabled}
            title={actions.recaptionDisabled ?? undefined}
            onClick={() => actions.recaption(mediaId)}
          >
            ✦ Regenerate caption
          </Button>
          <Button variant="ghost" onClick={() => actions.openCaption(mediaId)}>
            Edit in Caption →
          </Button>
          <Button variant="ghost" onClick={actions.ignore}>
            Ignore
          </Button>
        </div>
      </div>
    </div>
  );
}

export function IssueInspector({
  issue,
  actions,
}: {
  issue: ReportIssue;
  actions: IssueActions;
}) {
  switch (issue.kind) {
    case "near_dup":
      return <NearDupInspector detail={issue.detail} actions={actions} />;
    case "low_quality":
      return (
        <LowQualityInspector
          mediaId={issue.media_ids[0]}
          detail={issue.detail}
          actions={actions}
        />
      );
    case "outlier":
      return (
        <OutlierInspector
          mediaId={issue.media_ids[0]}
          detail={issue.detail}
          actions={actions}
        />
      );
    case "caption":
      return (
        <CaptionInspector
          mediaId={issue.media_ids[0]}
          detail={issue.detail}
          actions={actions}
        />
      );
  }
}
