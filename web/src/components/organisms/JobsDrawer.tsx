/** Right-hand jobs drawer: live job cards from the WebSocket store. */

import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { JobSnapshot } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useJobList, useJobsStore } from "../../store/jobsStore";
import { ProgressBar, Spinner } from "../atoms";

const STATE_GLYPH: Record<string, string> = {
  running: "◉",
  queued: "○",
  done: "✓",
  error: "✕",
  stopped: "■",
};

/** Render a second count as a compact ``2m 05s`` / ``1h 03m`` string. */
function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) {
    return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
  }
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

/**
 * The elapsed-time / time-left line under a running job.
 *
 * ``elapsed`` ticks off ``started_at`` and the ETA counts down from the
 * server's ``eta_seconds`` (measured over its recent-throughput window) less
 * the time since that estimate, so the two numbers stay consistent between
 * the sparse progress messages. The ETA hides until the server can measure a
 * rate, and while a phase reports no ``total`` (an indeterminate step).
 */
function JobTiming({ job, now }: { job: JobSnapshot; now: number }) {
  if (job.started_at == null) return null;
  const elapsed = now - job.started_at;
  const remaining =
    job.eta_seconds == null
      ? null
      : Math.max(0, job.eta_seconds - (now - job.updated_at));
  return (
    <div
      style={{
        fontFamily: font.mono,
        fontSize: 10,
        color: colors.textFaint,
        margin: "0 0 6px",
      }}
    >
      {formatDuration(elapsed)} elapsed
      {remaining != null && ` · ~${formatDuration(remaining)} left`}
    </div>
  );
}

/**
 * The media a run could not process, named one per line.
 *
 * A step that skips an image (an unreadable file, a file gone from disk)
 * keeps going, so the job ends green while the step still reports media
 * missing. Without this list the user has no way to learn *which* ones.
 */
function JobWarnings({ job }: { job: JobSnapshot }) {
  const [open, setOpen] = useState(false);
  const hidden = job.warning_count - job.warnings.length;
  return (
    <div style={{ marginTop: 6 }}>
      <span
        onClick={() => setOpen((current) => !current)}
        style={{
          color: colors.warn,
          fontSize: 11,
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        {open ? "▾" : "▸"} {job.warning_count} media skipped
      </span>
      {open && (
        <div
          style={{
            marginTop: 4,
            maxHeight: 160,
            overflowY: "auto",
            fontFamily: font.mono,
            fontSize: 10,
            lineHeight: 1.5,
            color: colors.textMutedAlt,
          }}
        >
          {job.warnings.map((line, index) => (
            <div key={index}>{line}</div>
          ))}
          {hidden > 0 && (
            <div style={{ color: colors.textFaint }}>
              … and {hidden} more (see the server log)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function JobsDrawer() {
  const open = useUiStore((state) => state.jobsOpen);
  const toggleJobs = useUiStore((state) => state.toggleJobs);
  const jobs = useJobList();
  const clearFinished = useJobsStore((state) => state.clearFinished);

  // A once-a-second clock so elapsed/ETA keep ticking between the sparse
  // progress messages. Runs only while a job is active — an idle drawer must
  // not re-render every second.
  const [now, setNow] = useState(() => Date.now() / 1000);
  const running = jobs.some((job) => job.state === "running");
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(
      () => setNow(Date.now() / 1000),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [running]);

  if (!open) return null;

  const stop = (id: string) => api.post(`/jobs/${id}/stop`).catch(() => {});
  const clear = () => {
    api.post("/jobs/clear").catch(() => {});
    clearFinished();
  };

  return (
    <div
      onClick={() => toggleJobs(false)}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8,9,11,0.5)",
        zIndex: 40,
        display: "flex",
        justifyContent: "flex-end",
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 360,
          height: "100%",
          background: colors.panel,
          borderLeft: `1px solid ${colors.border}`,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            padding: "12px 14px",
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <div style={{ fontWeight: 600, flex: 1 }}>Jobs</div>
          <button
            onClick={clear}
            style={{
              background: "transparent",
              border: "none",
              color: colors.textMuted,
              cursor: "pointer",
              fontSize: 11,
            }}
          >
            clear finished
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
          {jobs.length === 0 && (
            <div style={{ color: colors.textFaint, fontSize: 12 }}>
              No jobs yet.
            </div>
          )}
          {jobs.map((job) => (
            <div
              key={job.id}
              style={{
                background: colors.card,
                border: `1px solid ${colors.border}`,
                borderRadius: 8,
                padding: "10px 12px",
                marginBottom: 8,
              }}
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: 8 }}
              >
                <span
                  style={{
                    color:
                      job.state === "done"
                        ? colors.ok
                        : job.state === "error"
                          ? colors.danger
                          : colors.accent,
                  }}
                >
                  {STATE_GLYPH[job.state] ?? "○"}
                </span>
                <span style={{ fontWeight: 600, fontSize: 12, flex: 1 }}>
                  {job.name}
                </span>
                {job.state === "running" && <Spinner size={12} />}
              </div>
              {job.sub && (
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: 10.5,
                    color: colors.textMuted,
                    margin: "6px 0",
                  }}
                >
                  {job.sub}
                </div>
              )}
              {job.state === "running" && (
                <JobTiming job={job} now={now} />
              )}
              {job.state === "running" && (
                <div
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <div style={{ flex: 1 }}>
                    <ProgressBar pct={job.pct} striped />
                  </div>
                  <button
                    onClick={() => stop(job.id)}
                    style={{
                      background: "transparent",
                      border: `1px solid ${colors.borderControl}`,
                      borderRadius: 4,
                      color: colors.danger,
                      cursor: "pointer",
                      fontSize: 10,
                      padding: "2px 6px",
                    }}
                  >
                    stop
                  </button>
                </div>
              )}
              {job.state === "error" && (
                <div style={{ color: colors.danger, fontSize: 11 }}>
                  {job.error}
                </div>
              )}
              {job.warning_count > 0 && <JobWarnings job={job} />}
            </div>
          ))}
        </div>

        <div
          style={{
            padding: "10px 14px",
            borderTop: `1px solid ${colors.border}`,
            fontSize: 10.5,
            color: colors.textFaint,
          }}
        >
          Jobs run one at a time — models are chained, never co-loaded.
        </div>
      </div>
    </div>
  );
}
