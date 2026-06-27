/**
 * Server power controls, hidden under a flip-up safety cover (the topbar's
 * rightmost item). Two destructive actions — Restart and Shut down — each run
 * a 3 s cancellable countdown, then a presentational safe-teardown sequence,
 * then a terminal screen. The countdown lives client-side; the request only
 * fires when it expires. On restart the client polls ``/api/health`` and
 * reloads once the rebuilt server answers again.
 */

import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import { colors, font, radii, shadow } from "../../design/tokens";

type Mode = "restart" | "shutdown";
type Phase = "arm" | "down" | "off";

const STEP_MS = 650;
const HEALTH_POLL_MS = 1500;
const COVER_AUTOCLOSE_MS = 6000;

// Widget-only surfaces (not shared tokens): the amber/red button skins and the
// discreet diagonal-stripe cover. Kept inline because nothing else uses them.
const AMBER = {
  border: "#3d3524",
  bg: "#201b12",
  hoverBg: "#2c2416",
} as const;
const RED = {
  border: "#57302b",
  bg: "#241715",
  hoverBg: "#3a2220",
} as const;

function steps(mode: Mode): string[] {
  return [
    "Running jobs stopped cleanly",
    "Model unloaded · VRAM freed",
    "Database synced",
    mode === "restart" ? "Relaunching via run.bat" : "FastAPI server stopped",
  ];
}

export function PowerWidget() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("shutdown");
  const [phase, setPhase] = useState<Phase | null>(null);
  const [count, setCount] = useState(3);
  const [step, setStep] = useState(0);
  const [hover, setHover] = useState<string | null>(null);

  const closeTimer = useRef<number>();
  const seqTimers = useRef<number[]>([]);
  const healthInt = useRef<number>();
  const fired = useRef(false);

  const accent = mode === "restart" ? colors.warn : colors.danger;
  const icon = mode === "restart" ? "↻" : "⏻";

  // Countdown: one tick per second while arming, down to 0.
  useEffect(() => {
    if (phase !== "arm" || count <= 0) return;
    const t = window.setTimeout(() => setCount((c) => c - 1), 1000);
    return () => window.clearTimeout(t);
  }, [phase, count]);

  // At 0 the request fires (once) and the teardown sequence plays out.
  useEffect(() => {
    if (phase !== "arm" || count > 0 || fired.current) return;
    fired.current = true;
    setPhase("down");
    setStep(0);
    const path = mode === "restart" ? "/system/restart" : "/system/shutdown";
    // The server exits mid-response, so this never resolves — ignore it.
    api.post(path).catch(() => undefined);
    [1, 2, 3, 4].forEach((n, i) => {
      seqTimers.current.push(
        window.setTimeout(() => setStep(n), STEP_MS * (i + 1)),
      );
    });
    seqTimers.current.push(
      window.setTimeout(() => setPhase("off"), STEP_MS * 5),
    );
  }, [phase, count, mode]);

  // Restart terminal: poll health, reload when the rebuilt server answers.
  useEffect(() => {
    if (phase !== "off" || mode !== "restart") return;
    healthInt.current = window.setInterval(() => {
      fetch("/api/health", { cache: "no-store" })
        .then((r) => {
          if (r.ok) {
            window.clearInterval(healthInt.current);
            window.location.reload();
          }
        })
        .catch(() => undefined);
    }, HEALTH_POLL_MS);
    return () => window.clearInterval(healthInt.current);
  }, [phase, mode]);

  // Drop every pending timer on unmount.
  useEffect(
    () => () => {
      window.clearTimeout(closeTimer.current);
      window.clearInterval(healthInt.current);
      seqTimers.current.forEach(window.clearTimeout);
    },
    [],
  );

  function toggleCover() {
    if (phase) return;
    const next = !open;
    setOpen(next);
    window.clearTimeout(closeTimer.current);
    if (next) {
      closeTimer.current = window.setTimeout(
        () => setOpen(false),
        COVER_AUTOCLOSE_MS,
      );
    }
  }

  function arm(target: Mode) {
    if (phase) return;
    window.clearTimeout(closeTimer.current);
    fired.current = false;
    setMode(target);
    setCount(3);
    setStep(0);
    setOpen(false);
    setPhase("arm");
  }

  function cancel() {
    seqTimers.current.forEach(window.clearTimeout);
    seqTimers.current = [];
    fired.current = false;
    setPhase(null);
    setCount(5);
    setStep(0);
    setOpen(false);
  }

  return (
    <>
      <div style={{ position: "relative", width: 66, height: 32 }}>
        {/* Under the cover: the two real actions, side by side. */}
        <div
          style={{
            position: "absolute",
            left: 2,
            right: 2,
            top: 3,
            bottom: 3,
            display: "flex",
            gap: 3,
          }}
        >
          <button
            title="Restart the server"
            onClick={() => arm("restart")}
            onMouseEnter={() => setHover("restart")}
            onMouseLeave={() => setHover(null)}
            style={{
              flex: 1,
              borderRadius: 5,
              fontSize: 12,
              cursor: "pointer",
              color: colors.warn,
              border: `1px solid ${
                hover === "restart" ? colors.warn : AMBER.border
              }`,
              background: hover === "restart" ? AMBER.hoverBg : AMBER.bg,
            }}
          >
            ↻
          </button>
          <button
            title="Shut down the server"
            onClick={() => arm("shutdown")}
            onMouseEnter={() => setHover("shutdown")}
            onMouseLeave={() => setHover(null)}
            style={{
              flex: 1,
              borderRadius: 5,
              fontSize: 12,
              cursor: "pointer",
              color: colors.danger,
              border: `1px solid ${
                hover === "shutdown" ? colors.danger : RED.border
              }`,
              background: hover === "shutdown" ? RED.hoverBg : RED.bg,
            }}
          >
            ⏻
          </button>
        </div>

        {/* The safety cover, hinged at the top edge. */}
        <div
          title="Safety — lift to access the server controls"
          onClick={toggleCover}
          onMouseEnter={() => setHover("cover")}
          onMouseLeave={() => setHover(null)}
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: 6,
            cursor: "pointer",
            border: `1px solid ${hover === "cover" ? "#3a3d47" : "#2c2f38"}`,
            background:
              "repeating-linear-gradient(-45deg, #23252c 0 5px, #1b1d22 5px 11px)",
            transformOrigin: "50% -2px",
            transform: open
              ? "perspective(220px) rotateX(-112deg)"
              : "perspective(220px) rotateX(0deg)",
            transition: "transform 0.28s cubic-bezier(0.34, 1.3, 0.5, 1)",
            pointerEvents: open ? "none" : "auto",
          }}
        >
          <span
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              bottom: 2,
              textAlign: "center",
              fontFamily: font.mono,
              fontSize: 7,
              fontWeight: 600,
              letterSpacing: "0.14em",
              color: "#5c5f6a",
            }}
          >
            SAFE
          </span>
        </div>
      </div>

      {phase && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 400,
            background: "rgba(10,11,13,0.93)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: 380,
              background: colors.panel,
              border: `1px solid ${colors.borderHover}`,
              borderRadius: radii.modal,
              padding: "26px 28px",
              boxShadow: shadow.modal,
              display: "flex",
              flexDirection: "column",
              gap: 18,
            }}
          >
            {phase === "arm" && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <div
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <span style={{ fontSize: 17, color: accent }}>{icon}</span>
                  <span style={{ fontSize: 15, fontWeight: 700 }}>
                    {mode === "restart"
                      ? "Restarting the server"
                      : "Shutting down the server"}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 36,
                    fontWeight: 700,
                    fontFamily: font.mono,
                    color: accent,
                  }}
                >
                  {count}
                </div>
                <div style={{ fontSize: 11.5, color: colors.textMuted }}>
                  {mode === "restart" ? "Restart" : "Shutdown"} in {count} s
                  — running jobs will be stopped
                </div>
                <button
                  onClick={cancel}
                  onMouseEnter={() => setHover("cancel")}
                  onMouseLeave={() => setHover(null)}
                  style={{
                    padding: "7px 22px",
                    borderRadius: radii.control,
                    background:
                      hover === "cancel" ? colors.borderControl : colors.raised,
                    border: `1px solid ${
                      hover === "cancel"
                        ? colors.textFaint
                        : colors.borderHover
                    }`,
                    color: colors.text,
                    fontSize: 12.5,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Cancel
                </button>
              </div>
            )}

            {phase === "down" && (
              <div
                style={{ display: "flex", flexDirection: "column", gap: 14 }}
              >
                <div
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <span style={{ fontSize: 17, color: accent }}>{icon}</span>
                  <span style={{ fontSize: 15, fontWeight: 700 }}>
                    {mode === "restart"
                      ? "Safe restart in progress…"
                      : "Safe shutdown in progress…"}
                  </span>
                </div>
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 10 }}
                >
                  {steps(mode).map((label, i) => {
                    const done = step > i;
                    const active = step === i;
                    const mark = done ? "✓" : active ? "▸" : "·";
                    const col = done
                      ? colors.ok
                      : active
                        ? colors.warn
                        : colors.textFaint;
                    return (
                      <div
                        key={label}
                        style={{
                          display: "flex",
                          gap: 8,
                          fontSize: 12.5,
                          alignItems: "baseline",
                        }}
                      >
                        <span
                          style={{
                            width: 14,
                            fontFamily: font.mono,
                            color: col,
                          }}
                        >
                          {mark}
                        </span>
                        <span
                          style={{
                            color: done || active ? colors.text : colors.textMuted,
                          }}
                        >
                          {label}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: 10.5,
                    color: colors.textFaint,
                  }}
                >
                  POST /api/system/{mode === "restart" ? "restart" : "shutdown"}
                </div>
              </div>
            )}

            {phase === "off" && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 12,
                  textAlign: "center",
                }}
              >
                <span
                  style={{
                    fontSize: 34,
                    color: mode === "restart" ? colors.warn : colors.textFaint,
                  }}
                >
                  {icon}
                </span>
                <span style={{ fontSize: 16, fontWeight: 700 }}>
                  {mode === "restart"
                    ? "Server restarting"
                    : "Server shut down"}
                </span>
                <span style={{ fontSize: 12, color: colors.textMuted }}>
                  {mode === "restart"
                    ? "The page will reconnect automatically as soon as the server is ready again."
                    : "Caption Forge has shut down cleanly. You can close this tab."}
                </span>
                <span
                  style={{
                    fontFamily: font.mono,
                    fontSize: 10.5,
                    color: colors.textFaint,
                  }}
                >
                  {mode === "restart"
                    ? "run.bat — rebuild + relaunch"
                    : "run.bat to relaunch"}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
