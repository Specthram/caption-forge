/**
 * The Index pipeline panels of the Libraries view.
 *
 * "Index" is a chain of scans (thumbnails, quality scores, embeddings,
 * auto-tags — see `src/index_steps.py`). This file renders the two panels
 * that drive it: the global recap ("Index — all libraries", the only place
 * acting on every library at once) and the per-library card, whose buttons
 * are always scoped to the selected library. A step turned off in Settings
 * (per machine) is dimmed everywhere and its run buttons become a link back
 * to Settings.
 */

import type { IndexStep, StepCounts } from "../../api/types";
import { overallCoverage, pct } from "../../design/indexCoverage";
import { colors, font, stepColor } from "../../design/tokens";
import { Button, Dot, ProgressBar, Spinner } from "../atoms";

export interface IndexRun {
  (libraryId: number | null, steps: string[] | null): void;
}

/** Amber while scans are missing, green once the step is complete. */
function countColor(count: { done: number; total: number } | undefined) {
  const missing = (count?.total ?? 0) - (count?.done ?? 0);
  return missing > 0 ? colors.warn : colors.ok;
}

/** The step badges under a library card in the sources sidebar. */
export function LibraryStepBadges({
  steps,
  counts,
}: {
  steps: IndexStep[];
  counts: StepCounts;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {steps
        .filter((step) => step.enabled)
        .map((step) => {
          const count = counts[step.key];
          const missing = (count?.total ?? 0) - (count?.done ?? 0);
          const done = missing <= 0;
          return (
            <span
              key={step.key}
              title={
                done
                  ? `${step.label} — indexed`
                  : `${step.label} — ${missing} missing`
              }
              style={{
                fontFamily: font.mono,
                fontSize: 9,
                padding: "1px 5px",
                borderRadius: 4,
                color: done ? colors.ok : colors.warn,
                background: done ? "#152a17" : "#2a2312",
              }}
            >
              {done ? `✓ ${step.short}` : `${step.short} ${missing}`}
            </span>
          );
        })}
    </div>
  );
}

/** "Index — all libraries": global recap, one tile per step. */
export function IndexRecapPanel({
  steps,
  totals,
  libraries,
  onRun,
  busy,
  onRescanAll,
  rescanning,
  onReindexAll,
  reindexing,
  progressSub,
  progressPct,
}: {
  steps: IndexStep[];
  totals: StepCounts;
  libraries: number;
  onRun: IndexRun;
  busy?: boolean;
  onRescanAll?: () => void;
  rescanning?: boolean;
  onReindexAll?: () => void;
  reindexing?: boolean;
  progressSub?: string;
  progressPct?: number;
}) {
  const { missing } = overallCoverage(steps, totals);
  const anyBusy = busy || rescanning || reindexing;
  return (
    <div style={card}>
      <div style={headerRow}>
        <div style={{ fontSize: 12.5, fontWeight: 700 }}>
          Index — all libraries
        </div>
        <div style={{ marginLeft: "auto", ...mono, color: colors.textFaint }}>
          {missing > 0
            ? `${missing.toLocaleString()} scans missing across ${libraries} librar${libraries === 1 ? "y" : "ies"}`
            : "✓ everything indexed"}
        </div>
        {onRescanAll && (
          <button
            type="button"
            onClick={onRescanAll}
            disabled={anyBusy}
            title="Walk every library folder for new or removed files, then refresh the counters below."
            style={{
              ...outlineButton,
              cursor: anyBusy ? "not-allowed" : "pointer",
              opacity: anyBusy ? 0.6 : 1,
            }}
          >
            {rescanning ? (
              <>
                <Spinner size={11} /> Rescanning…
              </>
            ) : (
              "⟳ Rescan all libraries"
            )}
          </button>
        )}
        <button
          type="button"
          onClick={() => onRun(null, null)}
          disabled={anyBusy}
          style={{
            ...outlineButton,
            cursor: anyBusy ? "not-allowed" : "pointer",
            opacity: anyBusy ? 0.6 : 1,
          }}
        >
          {busy ? (
            <>
              <Spinner size={11} /> Indexing…
            </>
          ) : (
            "▶ Index everything missing"
          )}
        </button>
        {onReindexAll && (
          <button
            type="button"
            onClick={onReindexAll}
            disabled={anyBusy}
            title="Rescan every folder for new files, then run the whole index chain (thumbnails, quality, embeddings, WD14 auto-tags) — new media included."
            style={{
              ...outlineAccentButton,
              cursor: anyBusy ? "not-allowed" : "pointer",
              opacity: anyBusy ? 0.6 : 1,
            }}
          >
            {reindexing ? (
              <>
                <Spinner size={11} /> Rebuilding…
              </>
            ) : (
              "⚡ Rescan + re-index all"
            )}
          </button>
        )}
      </div>
      {(rescanning || reindexing) && (
        <div style={{ marginTop: 10 }}>
          <div
            style={{
              ...mono,
              fontSize: 9.5,
              color: colors.textMuted,
              marginBottom: 4,
            }}
          >
            {progressSub ||
              (reindexing ? "rebuilding…" : "scanning all libraries…")}
          </div>
          <ProgressBar
            height={3}
            color={colors.accent}
            pct={progressPct ?? 0}
          />
        </div>
      )}
      <div style={tileGrid}>
        {steps.map((step) => {
          const count = totals[step.key];
          const missingHere = (count?.total ?? 0) - (count?.done ?? 0);
          const color = step.enabled ? countColor(count) : colors.textFaint;
          return (
            <div
              key={step.key}
              style={{ ...tile, opacity: step.enabled ? 1 : 0.5 }}
              title={step.cost}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Dot color={stepColor(step.key)} size={7} />
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: colors.textSecondaryAlt,
                  }}
                >
                  {step.label}
                </span>
                <span
                  style={{
                    marginLeft: "auto",
                    ...mono,
                    fontSize: 9.5,
                    color,
                  }}
                >
                  {!step.enabled
                    ? "off"
                    : missingHere > 0
                      ? `${missingHere} missing`
                      : "✓ done"}
                </span>
              </div>
              <div style={{ margin: "7px 0 5px" }}>
                <ProgressBar
                  height={3}
                  color={color}
                  pct={pct(count?.done ?? 0, count?.total ?? 0)}
                />
              </div>
              <div style={{ ...mono, fontSize: 9.5, color: colors.textFaint }}>
                {count?.done ?? 0} / {count?.total ?? 0} media
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** One pipeline row: dot, name + models, progress, count, run button. */
function StepRow({
  step,
  count,
  libraryId,
  onRun,
  onSettings,
  busy,
}: {
  step: IndexStep;
  count: { done: number; total: number } | undefined;
  libraryId: number;
  onRun: IndexRun;
  onSettings: () => void;
  busy?: boolean;
}) {
  const done = count?.done ?? 0;
  const total = count?.total ?? 0;
  const missing = total - done;
  const color = step.enabled ? countColor(count) : colors.textFaint;
  return (
    <div style={{ ...stepRow, opacity: step.enabled ? 1 : 0.55 }}>
      <Dot color={stepColor(step.key)} size={7} />
      <div style={{ width: 190, flex: "none" }}>
        <div
          style={{ fontSize: 11.5, fontWeight: 600, color: colors.textSecondary }}
        >
          {step.label}
        </div>
        <div style={{ ...mono, fontSize: 9.5, color: colors.textFaint, ...ellipsis }}>
          {step.models}
        </div>
      </div>
      <div style={{ flex: 1, minWidth: 60 }}>
        <ProgressBar height={4} color={color} pct={pct(done, total)} />
      </div>
      <div
        style={{
          width: 92,
          flex: "none",
          textAlign: "right",
          ...mono,
          fontSize: 10,
          color,
        }}
      >
        {done} / {total}
      </div>
      <div style={{ width: 138, flex: "none", display: "flex", justifyContent: "flex-end" }}>
        {step.enabled ? (
          <Button loading={busy} onClick={() => onRun(libraryId, [step.key])}>
            {busy
              ? "Running…"
              : missing > 0
                ? `▶ Run — ${missing} missing`
                : "↻ Re-run"}
          </Button>
        ) : (
          <span onClick={onSettings} style={offLink} title="Enable it in Settings">
            off — Settings →
          </span>
        )}
      </div>
    </div>
  );
}

/** "Index — <library>": the chain, scoped to the selected library only. */
export function IndexLibraryCard({
  name,
  libraryId,
  steps,
  counts,
  force,
  onForce,
  onRun,
  onSettings,
  warning,
  busy,
  busyStep,
}: {
  name: string;
  libraryId: number;
  steps: IndexStep[];
  counts: StepCounts;
  force: boolean;
  onForce: (value: boolean) => void;
  onRun: IndexRun;
  onSettings: () => void;
  warning?: string;
  busy?: boolean;
  busyStep?: string | null;
}) {
  const enabled = steps.filter((step) => step.enabled);
  const { missing } = overallCoverage(steps, counts);
  return (
    <div style={{ ...card, marginTop: 14 }}>
      <div style={{ ...headerRow, flexWrap: "wrap" }}>
        <div style={{ fontSize: 12.5, fontWeight: 700 }}>Index — {name}</div>
        <div style={{ fontSize: 10.5, color: colors.textMuted }}>
          runs: {enabled.map((step) => step.short).join(" + ") || "nothing"}
        </div>
        <div style={{ marginLeft: "auto", ...mono, color: missing > 0 ? colors.warn : colors.ok }}>
          {missing > 0 ? `${missing.toLocaleString()} scans missing` : "✓ fully indexed"}
        </div>
        <Button
          variant="accent"
          loading={busy}
          onClick={() => onRun(libraryId, null)}
        >
          {busy
            ? "Indexing…"
            : missing > 0
              ? `▶ Index this library — ${missing} missing`
              : "↻ Re-index this library"}
        </Button>
        <label style={forceLabel}>
          <input
            type="checkbox"
            checked={force}
            onChange={(event) => onForce(event.target.checked)}
          />
          Force re-run
        </label>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 10 }}>
        {steps.map((step) => (
          <StepRow
            key={step.key}
            step={step}
            count={counts[step.key]}
            libraryId={libraryId}
            onRun={onRun}
            onSettings={onSettings}
            busy={busyStep === step.key}
          />
        ))}
      </div>

      {warning && (
        <div style={{ marginTop: 8, fontSize: 10.5, color: colors.warn }}>
          {warning}
        </div>
      )}
      <div style={{ marginTop: 8, fontSize: 10, color: colors.textFaint }}>
        “Index” = the scans above, chained on this library only. Grids, reports
        and Auto-build read these results — originals are never touched. Heavy
        models can be turned off per machine in Settings → This machine.
      </div>
    </div>
  );
}

const mono = { fontFamily: font.mono, fontSize: 10 } as const;

const ellipsis = {
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} as const;

const card = {
  border: `1px solid ${colors.border}`,
  borderRadius: 9,
  background: colors.panel,
  padding: 12,
} as const;

const headerRow = {
  display: "flex",
  alignItems: "center",
  gap: 10,
} as const;

const tileGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: 8,
  marginTop: 10,
} as const;

const tile = {
  border: `1px solid ${colors.border}`,
  borderRadius: 7,
  background: colors.app,
  padding: "8px 10px",
} as const;

const stepRow = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 11px",
  border: "1px solid #1e2026",
  borderRadius: 7,
  background: colors.app,
} as const;

const outlineAccentButton = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  border: `1px solid ${colors.accentBorder}`,
  background: colors.accentTintAlt,
  color: colors.accent,
  borderRadius: 6,
  padding: "5px 10px",
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const outlineButton = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.panel,
  color: colors.textSecondary,
  borderRadius: 6,
  padding: "5px 10px",
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const offLink = {
  ...mono,
  fontSize: 9.5,
  color: colors.textFaint,
  border: `1px dashed ${colors.borderControl}`,
  borderRadius: 6,
  padding: "4px 8px",
  cursor: "pointer",
} as const;

const forceLabel = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  fontSize: 10.5,
  color: colors.textMuted,
  cursor: "pointer",
} as const;
