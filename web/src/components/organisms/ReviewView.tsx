/**
 * The Caption workspace's Review sub-tab: a rules rail and the findings queue.
 *
 * The rail picks the judge model (independent from the captioner), toggles the
 * per-dataset rules and launches a run over All / Selection / Flagged media.
 * The queue lists every finding with an inline word diff and per-row accept /
 * reject; the ⚡ wizard triages the pending ones full-screen. Nothing is
 * applied silently — "Accept all safe fixes" only ever touches the
 * deterministic and integrity findings.
 */

import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useCreateReviewRule,
  useClearReviewHistory,
  useDecideBulk,
  useDecideFinding,
  useRejectAll,
  useDeleteReviewRule,
  useProfiles,
  useReviewFindings,
  useReviewRules,
  useRunReview,
  useUpdateReviewRule,
} from "../../api/hooks";
import type { ReviewFinding, ReviewRule } from "../../api/types";
import { Badge, Button, ProgressBar } from "../atoms";
import { colors, font, radii } from "../../design/tokens";
import { kindStyle, SAFE_KINDS } from "../../lib/review";
import { DiffText } from "../molecules/DiffText";
import { useJobsStore } from "../../store/jobsStore";
import { useSelectionStore } from "../../store/selectionStore";
import { useUiStore } from "../../store/uiStore";
import { ProfileSelector } from "./ProfileSelector";

const RAIL_WIDTH = 284;
const thumbUrl = (id: number) => `/api/media/${id}/thumb`;

export function ReviewView() {
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const openWizard = useUiStore((state) => state.openReviewWizard);
  const selected = useSelectionStore((state) => state.selected);
  const client = useQueryClient();

  const rules = useReviewRules(datasetId);
  const findings = useReviewFindings(datasetId, null, datasetId != null);
  const runReview = useRunReview();
  const profiles = useProfiles();
  const [scope, setScope] = useState("all");
  const [unloadAfter, setUnloadAfter] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJobsStore((state) =>
    jobId ? state.jobs[jobId] : undefined,
  );
  const running = job?.state === "queued" || job?.state === "running";

  useEffect(() => {
    if (!jobId || !job || running) return;
    client.invalidateQueries({ queryKey: ["review-findings"] });
    client.invalidateQueries({ queryKey: ["review-counts"] });
    client.invalidateQueries({ queryKey: ["caption-grid"] });
    setJobId(null);
  }, [job, running, jobId, client]);

  const list = useMemo(
    () => findings.data?.findings ?? [],
    [findings.data],
  );
  const counts = findings.data?.counts ?? {
    pending: 0,
    accepted: 0,
    rejected: 0,
  };

  const selectedIds = useMemo(
    () => [...selected].map((key) => Number(key)),
    [selected],
  );
  const flaggedIds = useMemo(
    () =>
      [...new Set(list.filter((f) => f.status === "pending").map((f) => f.media_id))],
    [list],
  );

  const scopeCount =
    scope === "selection"
      ? selectedIds.length
      : scope === "flagged"
        ? flaggedIds.length
        : null;

  const run = () => {
    if (datasetId == null) return;
    const mediaIds =
      scope === "selection"
        ? selectedIds
        : scope === "flagged"
          ? flaggedIds
          : null;
    runReview.mutate(
      {
        dataset_id: datasetId,
        caption_type: captionType,
        media_ids: mediaIds,
        judge_profile_id: profiles.data?.judge_id ?? null,
        scope,
        unload_after: unloadAfter,
      },
      { onSuccess: (data) => setJobId(data.job_id) },
    );
  };

  if (datasetId == null) {
    return <EmptyPane text="Pick a dataset to review." />;
  }

  return (
    <div style={{ display: "flex", flex: 1, minWidth: 0, minHeight: 0 }}>
      <Rail
        datasetId={datasetId}
        rules={rules.data?.rules ?? []}
        scope={scope}
        onScope={setScope}
        scopeCount={scopeCount}
        running={running}
        job={job}
        onRun={run}
        unloadAfter={unloadAfter}
        onUnloadAfter={setUnloadAfter}
      />
      <Queue
        datasetId={datasetId}
        list={list}
        counts={counts}
        loading={findings.isLoading}
        rulesReady={(rules.data?.rules?.length ?? 0) > 0}
        onOpenWizard={openWizard}
      />
    </div>
  );
}

// -- Rail --------------------------------------------------------------------

interface JobLike {
  done: number;
  total: number;
  sub: string;
  pct: number;
}

function Rail({
  datasetId,
  rules,
  scope,
  onScope,
  scopeCount,
  running,
  job,
  onRun,
  unloadAfter,
  onUnloadAfter,
}: {
  datasetId: number;
  rules: ReviewRule[];
  scope: string;
  onScope: (value: string) => void;
  scopeCount: number | null;
  running: boolean;
  job: JobLike | undefined;
  onRun: () => void;
  unloadAfter: boolean;
  onUnloadAfter: (value: boolean) => void;
}) {
  const profiles = useProfiles();
  const createRule = useCreateReviewRule();
  const [text, setText] = useState("");
  const [needsImage, setNeedsImage] = useState(false);

  const judgeLabel =
    profiles.data?.profiles.find(
      (p) => p.id === profiles.data.judge_id,
    )?.name ?? "the judge profile";

  const add = () => {
    if (!text.trim()) return;
    createRule.mutate(
      { dataset_id: datasetId, text: text.trim(), needs_image: needsImage },
      {
        onSuccess: () => {
          setText("");
          setNeedsImage(false);
        },
      },
    );
  };

  return (
    <div
      style={{
        width: RAIL_WIDTH,
        flex: "none",
        borderRight: `1px solid ${colors.border}`,
        background: colors.panel,
        overflowY: "auto",
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <section>
        <SectionLabel>Judge profile</SectionLabel>
        <ProfileSelector role="judge" />
        <p style={hintStyle}>
          A model confirms its own mistakes poorly — pick a judge separate from
          the captioner. It is swapped into VRAM for the run only.
        </p>
      </section>

      <section>
        <SectionLabel>Rules · per dataset</SectionLabel>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {rules.map((rule) => (
            <RuleCard key={rule.id} rule={rule} datasetId={datasetId} />
          ))}
        </div>
        <div style={{ marginTop: 10 }}>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="New rule, plain language…"
            rows={2}
            style={{ ...selectStyle, resize: "vertical", height: "auto" }}
          />
          <label style={checkRow}>
            <input
              type="checkbox"
              checked={needsImage}
              onChange={(event) => setNeedsImage(event.target.checked)}
            />
            Judge sees the image
          </label>
          <Button
            variant="ghost"
            block
            onClick={add}
            loading={createRule.isPending}
            style={{ marginTop: 6 }}
          >
            + Add rule
          </Button>
        </div>
      </section>

      <section>
        <SectionLabel>Run</SectionLabel>
        <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
          {[
            { value: "all", label: "All" },
            { value: "selection", label: "Selection" },
            { value: "flagged", label: "Flagged" },
          ].map((option) => {
            const active = scope === option.value;
            const disabled =
              option.value !== "all" && (scopeCount ?? 0) === 0;
            return (
              <button
                key={option.value}
                disabled={disabled}
                onClick={() => onScope(option.value)}
                style={{
                  flex: 1,
                  padding: "5px 4px",
                  borderRadius: radii.control,
                  border: `1px solid ${active ? colors.accentBorder : colors.borderControl}`,
                  background: active ? colors.accentTint : "transparent",
                  color: active
                    ? colors.accent
                    : disabled
                      ? colors.textFaint
                      : colors.textMuted,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: disabled ? "default" : "pointer",
                }}
              >
                {option.label}
                {option.value !== "all" && scopeCount != null
                  ? ` · ${scopeCount}`
                  : ""}
              </button>
            );
          })}
        </div>
        {running ? (
          <div>
            <ProgressBar pct={job?.pct ?? 0} striped color={colors.info} />
            <p style={{ ...hintStyle, marginTop: 6 }}>
              {job?.done ?? 0} / {job?.total ?? 0} · {job?.sub ?? "…"}
            </p>
          </div>
        ) : (
          <Button variant="accent" block onClick={onRun}>
            ▶ Run review
          </Button>
        )}
        <label style={checkRow}>
          <input
            type="checkbox"
            checked={unloadAfter}
            onChange={(event) => onUnloadAfter(event.target.checked)}
          />
          Unload the model after the run
        </label>
        <p style={hintStyle}>
          Text-only rules run without loading images; vision rules load each
          one. Locked captions are skipped. Judge: {judgeLabel}.
        </p>
      </section>
    </div>
  );
}

function RuleCard({
  rule,
  datasetId,
}: {
  rule: ReviewRule;
  datasetId: number;
}) {
  const update = useUpdateReviewRule();
  const remove = useDeleteReviewRule();
  const style = kindStyle(rule.kind);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(rule.text);

  const save = () => {
    const text = draft.trim();
    if (text && text !== rule.text) {
      update.mutate({ id: rule.id, dataset_id: datasetId, text });
    }
    setEditing(false);
  };

  return (
    <div
      style={{
        border: `1px solid ${colors.border}`,
        borderRadius: radii.control,
        background: rule.enabled ? colors.card : "transparent",
        padding: 8,
        opacity: rule.enabled ? 1 : 0.6,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="checkbox"
          checked={rule.enabled}
          onChange={(event) =>
            update.mutate({
              id: rule.id,
              dataset_id: datasetId,
              enabled: event.target.checked,
            })
          }
        />
        <span title={style.hint}>
          <Badge color={style.color} background={style.background}>
            {style.label}
          </Badge>
        </span>
        {rule.needs_image && (
          <span title="The judge sees the image for this rule">
            <Badge color={colors.info}>◉ img</Badge>
          </span>
        )}
        <div style={{ flex: 1 }} />
        {!editing && (
          <button
            title="Edit rule"
            onClick={() => {
              setDraft(rule.text);
              setEditing(true);
            }}
            style={xButton}
          >
            ✎
          </button>
        )}
        {!rule.builtin && (
          <button
            title="Delete rule"
            onClick={() => remove.mutate({ id: rule.id, dataset_id: datasetId })}
            style={xButton}
          >
            ✕
          </button>
        )}
      </div>
      {editing ? (
        <div style={{ marginTop: 6 }}>
          <textarea
            autoFocus
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                save();
              } else if (event.key === "Escape") {
                setEditing(false);
              }
            }}
            rows={2}
            style={{ ...selectStyle, resize: "vertical", height: "auto" }}
          />
          <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
            <button style={editSaveButton} onClick={save}>
              Save
            </button>
            <button style={xButton} onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <p
          style={{
            margin: "6px 0 0",
            fontSize: 12,
            color: colors.textSecondary,
            lineHeight: 1.4,
          }}
        >
          {rule.text}
        </p>
      )}
    </div>
  );
}

// -- Queue -------------------------------------------------------------------

function Queue({
  datasetId,
  list,
  counts,
  loading,
  rulesReady,
  onOpenWizard,
}: {
  datasetId: number;
  list: ReviewFinding[];
  counts: { pending: number; accepted: number; rejected: number };
  loading: boolean;
  rulesReady: boolean;
  onOpenWizard: (index: number) => void;
}) {
  const decideBulk = useDecideBulk();
  const rejectAll = useRejectAll();
  const clearHistory = useClearReviewHistory();
  const decidedCount = counts.accepted + counts.rejected;
  const safeCount = list.filter(
    (f) => f.status === "pending" && SAFE_KINDS.has(f.rule_kind),
  ).length;
  const pendingIndexes = list
    .map((f, index) => ({ f, index }))
    .filter(({ f }) => f.status === "pending");

  if (loading) return <EmptyPane text="Loading the queue…" />;
  if (list.length === 0) {
    return (
      <EmptyPane
        text={
          rulesReady
            ? "All captions pass the enabled rules. Run a review to check again."
            : "Pick your rules on the left, then ▶ Run review."
        }
      />
    );
  }

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
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          borderBottom: `1px solid ${colors.border}`,
          background: colors.toolbar,
        }}
      >
        <CountChip n={counts.pending} label="pending" color={colors.info} />
        <CountChip n={counts.accepted} label="accepted" color={colors.ok} />
        <CountChip n={counts.rejected} label="rejected" color={colors.danger} />
        <div style={{ flex: 1 }} />
        {decidedCount > 0 && (
          <Button
            variant="ghost"
            onClick={() => {
              if (
                window.confirm(
                  `Clear the review history (${decidedCount} decided ` +
                    "findings)? Captions are not touched.",
                )
              ) {
                clearHistory.mutate({ dataset_id: datasetId });
              }
            }}
            loading={clearHistory.isPending}
          >
            🗑 Clear history · {decidedCount}
          </Button>
        )}
        {counts.pending > 0 && (
          <Button
            variant="ghost"
            style={{ color: colors.danger, borderColor: colors.danger }}
            onClick={() => {
              if (
                window.confirm(
                  `Reject all ${counts.pending} pending findings? ` +
                    "Captions are not touched.",
                )
              ) {
                rejectAll.mutate({ dataset_id: datasetId });
              }
            }}
            loading={rejectAll.isPending}
          >
            ✕ Reject all · {counts.pending}
          </Button>
        )}
        {safeCount > 0 && (
          <Button
            variant="ghost"
            onClick={() => decideBulk.mutate({ dataset_id: datasetId })}
            loading={decideBulk.isPending}
          >
            ✓ Accept all safe fixes · {safeCount}
          </Button>
        )}
        {pendingIndexes.length > 0 && (
          <Button
            variant="accent"
            onClick={() => onOpenWizard(pendingIndexes[0].index)}
          >
            ⚡ Review wizard
          </Button>
        )}
      </div>
      <div style={{ overflowY: "auto", flex: 1 }}>
        {list.map((finding, index) => (
          <FindingRow
            key={finding.id}
            finding={finding}
            onOpen={() => onOpenWizard(index)}
          />
        ))}
      </div>
    </div>
  );
}

function FindingRow({
  finding,
  onOpen,
}: {
  finding: ReviewFinding;
  onOpen: () => void;
}) {
  const decide = useDecideFinding();
  const style = kindStyle(finding.rule_kind);
  const pending = finding.status === "pending";
  return (
    <div
      onClick={onOpen}
      style={{
        display: "flex",
        gap: 10,
        padding: "10px 14px",
        borderBottom: `1px solid ${colors.border}`,
        cursor: "pointer",
        opacity: pending ? 1 : 0.55,
      }}
    >
      <img
        src={thumbUrl(finding.media_id)}
        alt=""
        style={{
          width: 76,
          height: 57,
          objectFit: "cover",
          borderRadius: radii.control,
          flex: "none",
          background: colors.card,
        }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 4,
          }}
        >
          <span title={style.hint}>
            <Badge color={style.color} background={style.background}>
              {style.label}
            </Badge>
          </span>
          {pending && finding.conflict && (
            <span title="Another accepted fix already changed this phrase — accepting takes this version of it.">
              <Badge color={colors.warn} background="rgba(224,179,86,0.14)">
                ⚠ conflict
              </Badge>
            </span>
          )}
          <span
            style={{
              fontSize: 11,
              color: colors.textMuted,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
            }}
          >
            {finding.rule_text ?? finding.note}
          </span>
          <span
            style={{
              fontSize: 9.5,
              fontWeight: 700,
              fontFamily: font.mono,
              color: pending
                ? colors.info
                : finding.status === "accepted"
                  ? colors.ok
                  : colors.danger,
            }}
          >
            {finding.status.toUpperCase()}
          </span>
        </div>
        <DiffText
          before={finding.caption_before}
          after={finding.caption_after}
          clamp={2}
          style={{ fontSize: 12, color: colors.textSecondary }}
        />
      </div>
      {pending && (
        <div
          style={{ display: "flex", gap: 4, alignItems: "flex-start" }}
          onClick={(event) => event.stopPropagation()}
        >
          <button
            title="Accept fix"
            onClick={() => decide.mutate({ id: finding.id, action: "accept" })}
            style={{ ...pillButton, color: colors.ok }}
          >
            ✓
          </button>
          <button
            title="Reject — keep original"
            onClick={() => decide.mutate({ id: finding.id, action: "reject" })}
            style={{ ...pillButton, color: colors.danger }}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}

// -- Small shared bits -------------------------------------------------------

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        fontWeight: 700,
        color: colors.textMuted,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

function CountChip({
  n,
  label,
  color,
}: {
  n: number;
  label: string;
  color: string;
}) {
  return (
    <span style={{ fontSize: 12, color: colors.textMuted }}>
      <b style={{ color }}>{n}</b> {label}
    </span>
  );
}

function EmptyPane({ text }: { text: string }) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: colors.textMuted,
        fontSize: 13,
        padding: 40,
        textAlign: "center",
      }}
    >
      {text}
    </div>
  );
}

const selectStyle = {
  width: "100%",
  padding: "7px 9px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
  fontFamily: font.sans,
} as const;

const hintStyle = {
  margin: "8px 0 0",
  fontSize: 10.5,
  lineHeight: 1.45,
  color: colors.textFaint,
} as const;

const checkRow = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  marginTop: 6,
  fontSize: 11.5,
  color: colors.textMuted,
} as const;

const xButton = {
  border: "none",
  background: "transparent",
  color: colors.textFaint,
  cursor: "pointer",
  fontSize: 12,
} as const;

const editSaveButton = {
  padding: "3px 10px",
  borderRadius: radii.control,
  border: "none",
  background: colors.accent,
  color: colors.onAccent,
  cursor: "pointer",
  fontSize: 11,
  fontWeight: 600,
} as const;

const pillButton = {
  width: 26,
  height: 26,
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.raised,
  cursor: "pointer",
  fontSize: 12,
} as const;
