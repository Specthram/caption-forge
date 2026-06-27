/**
 * The Quality report's actionable core: every finding as an expandable row
 * (chevron, thumbnail(s), name, type badge, reason, metric) that opens the
 * matching inspector. A resolved row drops to 45% opacity, collapses and
 * shows how it was handled — clicking that status reopens the finding.
 */

import { useState } from "react";
import { colors, font, radii } from "../../design/tokens";
import type {
  DatasetReport,
  IssueKind,
  IssueResolution,
  ReportIssue,
  ResolutionKind,
} from "../../api/types";
import { IssueInspector } from "./IssueInspectors";
import type { IssueActions } from "./IssueInspectors";

const KIND_BADGE: Record<IssueKind, { label: string; color: string }> = {
  near_dup: { label: "NEAR-DUP", color: colors.warn },
  low_quality: { label: "LOW QUALITY", color: colors.danger },
  outlier: { label: "OUTLIER", color: colors.info },
  caption: { label: "CAPTION", color: colors.warn },
};

const RESOLUTION_LABEL: Record<ResolutionKind, { text: string; color: string }> =
  {
    removed: { text: "✕ removed from dataset", color: colors.danger },
    ignored: { text: "– ignored", color: colors.textFaint },
    recaptioned: { text: "✦ recaptioned", color: colors.ok },
  };

export interface FlaggedMediaProps {
  report: DatasetReport;
  resolutions: Record<string, IssueResolution>;
  onResolve: (issue: ReportIssue, resolution: ResolutionKind) => void;
  onReopen: (issue: ReportIssue) => void;
  onRemove: (issue: ReportIssue, mediaIds: number[]) => void;
  onRecaption: (issue: ReportIssue, mediaId: number) => void;
  onOpenMedia: (mediaId: number) => void;
  onOpenCaption: (mediaId: number) => void;
  recaptionDisabled: string | null;
}

function IssueRow({
  issue,
  resolution,
  expanded,
  onToggle,
  actions,
  onReopen,
}: {
  issue: ReportIssue;
  resolution: IssueResolution | undefined;
  expanded: boolean;
  onToggle: () => void;
  actions: IssueActions;
  onReopen: () => void;
}) {
  const badge = KIND_BADGE[issue.kind];
  const resolved = resolution
    ? RESOLUTION_LABEL[resolution.resolution]
    : null;
  return (
    <div
      style={{
        borderTop: `1px solid ${colors.border}`,
        opacity: resolved ? 0.45 : 1,
      }}
    >
      <div
        onClick={resolved ? undefined : onToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "9px 4px",
          cursor: resolved ? "default" : "pointer",
        }}
      >
        <span style={{ width: 12, color: colors.textFaint, fontSize: 10 }}>
          {resolved ? "" : expanded ? "▾" : "▸"}
        </span>
        <span style={{ display: "flex", gap: 3, flex: "none" }}>
          {issue.media_ids.slice(0, 2).map((mediaId) => (
            <img
              key={mediaId}
              src={`/api/media/${mediaId}/thumb`}
              alt=""
              style={{
                width: 40,
                height: 30,
                objectFit: "cover",
                borderRadius: 3,
              }}
            />
          ))}
        </span>
        <span
          title={issue.names.join(" ↔ ")}
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textSecondary,
            width: 150,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {issue.names.join(" ↔ ")}
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 9,
            fontWeight: 700,
            padding: "2px 6px",
            borderRadius: radii.chip,
            flex: "none",
            color: badge.color,
            border: `1px solid ${badge.color}`,
          }}
        >
          {badge.label}
        </span>
        <span
          style={{
            flex: 1,
            minWidth: 0,
            fontSize: 11.5,
            color: colors.textMuted,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {issue.reason}
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 11,
            color: badge.color,
            flex: "none",
          }}
        >
          {issue.metric}
        </span>
        {resolved ? (
          <span
            onClick={onReopen}
            title="Reopen this finding"
            style={{
              fontSize: 10.5,
              color: resolved.color,
              cursor: "pointer",
              flex: "none",
              width: 150,
              textAlign: "right",
            }}
          >
            {resolved.text}
          </span>
        ) : (
          <span
            style={{
              fontSize: 10.5,
              color: colors.textFaint,
              flex: "none",
              width: 150,
              textAlign: "right",
            }}
          >
            inspect
          </span>
        )}
      </div>
      {expanded && !resolved && (
        <div
          style={{
            padding: "10px 4px 16px 26px",
            borderTop: `1px dashed ${colors.border}`,
          }}
        >
          <IssueInspector issue={issue} actions={actions} />
        </div>
      )}
    </div>
  );
}

export function FlaggedMedia(props: FlaggedMediaProps) {
  const { report, resolutions } = props;
  const [expanded, setExpanded] = useState<string | null>(null);
  const open = report.issues.filter(
    (issue) => !resolutions[issue.key],
  ).length;

  return (
    <div
      style={{
        background: colors.card,
        border: `1px solid ${colors.border}`,
        borderRadius: radii.card,
        padding: "12px 14px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          paddingBottom: 8,
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
          Flagged media
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: open ? colors.accent : colors.ok,
          }}
        >
          {open} open
        </span>
      </div>

      {report.issues.length === 0 && (
        <div
          style={{
            padding: "18px 0",
            textAlign: "center",
            fontSize: 11.5,
            color: colors.textFaint,
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          ◌ Nothing flagged — the scorers found no actionable issue.
        </div>
      )}

      {report.issues.map((issue) => (
        <IssueRow
          key={issue.key}
          issue={issue}
          resolution={resolutions[issue.key]}
          expanded={expanded === issue.key}
          onToggle={() =>
            setExpanded(expanded === issue.key ? null : issue.key)
          }
          onReopen={() => props.onReopen(issue)}
          actions={{
            remove: (mediaIds) => props.onRemove(issue, mediaIds),
            ignore: () => props.onResolve(issue, "ignored"),
            recaption: (mediaId) => props.onRecaption(issue, mediaId),
            openMedia: props.onOpenMedia,
            openCaption: props.onOpenCaption,
            recaptionDisabled: props.recaptionDisabled,
          }}
        />
      ))}
    </div>
  );
}
