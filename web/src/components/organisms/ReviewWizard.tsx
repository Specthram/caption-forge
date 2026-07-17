/**
 * The Review wizard: a full-screen, keyboard-first triage of the pending
 * findings. Left is the media (wheel-zoom + drag-pan, reset on every move);
 * right is the rule, the judge's verdict and the proposed caption as a big
 * word diff. A/R accept/reject, E edits inline before accepting, Z undoes the
 * last decision, ← → navigate, Esc closes. Each decision auto-advances.
 *
 * It works off the live findings query: accepting a finding flips its status,
 * the query refetches, and the same index lands on the next pending one.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useCaptionGrounding,
  useDecideBulk,
  useDecideFinding,
  useReviewFindings,
  useUndoFinding,
} from "../../api/hooks";
import type { GroundedClaim, ReviewFinding } from "../../api/types";
import { Badge, Button, Kbd, ProgressBar } from "../atoms";
import { colors, font, groundingColor, radii, shadow } from "../../design/tokens";
import { kindStyle } from "../../lib/review";
import { DiffText } from "../molecules/DiffText";
import { useUiStore } from "../../store/uiStore";

const MAX_UNDO = 40;
const fileUrl = (id: number) => `/api/media/${id}/file`;

interface Zoom {
  scale: number;
  tx: number;
  ty: number;
}
const RESET: Zoom = { scale: 1, tx: 0, ty: 0 };

export function ReviewWizard() {
  const wizard = useUiStore((state) => state.reviewWizard);
  const datasetId = useUiStore((state) => state.datasetId);
  const close = useUiStore((state) => state.closeReviewWizard);
  const setWizard = useUiStore((state) => state.setReviewWizard);

  const findings = useReviewFindings(datasetId, null, wizard.open);
  const decide = useDecideFinding();
  const decideBulk = useDecideBulk();
  const undo = useUndoFinding();

  const pending = useMemo(
    () =>
      (findings.data?.findings ?? []).filter((f) => f.status === "pending"),
    [findings.data],
  );
  const counts = findings.data?.counts ?? {
    pending: 0,
    accepted: 0,
    rejected: 0,
  };
  const total = counts.pending + counts.accepted + counts.rejected;
  const index = Math.min(wizard.index, Math.max(pending.length - 1, 0));
  const current: ReviewFinding | undefined = pending[index];

  const grounding = useCaptionGrounding(
    current?.key ?? null,
    datasetId,
    current?.caption_type ?? "txt",
    wizard.open && current != null,
  );

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [zoom, setZoom] = useState<Zoom>(RESET);
  const undoStack = useRef<number[]>([]);
  const panRef = useRef<{ x: number; y: number } | null>(null);

  // Reset zoom + edit state whenever the shown finding changes.
  useEffect(() => {
    setZoom(RESET);
    setEditing(false);
    setDraft(current?.caption_after ?? "");
  }, [current?.id, current?.caption_after]);

  const goto = useCallback(
    (next: number) => setWizard({ index: Math.max(0, next) }),
    [setWizard],
  );

  const accept = useCallback(() => {
    if (!current) return;
    undoStack.current = [current.id, ...undoStack.current].slice(0, MAX_UNDO);
    decide.mutate({
      id: current.id,
      action: "accept",
      caption: editing ? draft : null,
    });
  }, [current, decide, editing, draft]);

  const reject = useCallback(() => {
    if (!current) return;
    undoStack.current = [current.id, ...undoStack.current].slice(0, MAX_UNDO);
    decide.mutate({ id: current.id, action: "reject" });
  }, [current, decide]);

  const undoLast = useCallback(() => {
    const last = undoStack.current[0];
    if (last == null) return;
    undoStack.current = undoStack.current.slice(1);
    undo.mutate(last);
  }, [undo]);

  // Keyboard shortcuts (ignored while typing in the inline editor, bar Esc).
  useEffect(() => {
    if (!wizard.open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      const typing =
        event.target instanceof HTMLElement &&
        (event.target.tagName === "TEXTAREA" ||
          event.target.tagName === "INPUT");
      if (event.key === "Escape") {
        close();
        return;
      }
      if (typing) return;
      if (event.key === "a" || event.key === "A") accept();
      else if (event.key === "r" || event.key === "R") reject();
      else if (event.key === "e" || event.key === "E") setEditing(true);
      else if (event.key === "z" || event.key === "Z") undoLast();
      else if (event.key === "ArrowRight") goto(index + 1);
      else if (event.key === "ArrowLeft") goto(index - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [wizard.open, accept, reject, undoLast, goto, index, close]);

  if (!wizard.open) return null;

  const done = counts.accepted + counts.rejected;
  const clear = pending.length === 0;

  return (
    <div style={overlay}>
      <Header
        total={total}
        position={clear ? total : done + 1}
        pct={total ? (done / total) * 100 : 0}
        hasUndo={undoStack.current.length > 0}
        onUndo={undoLast}
        onClose={close}
      />
      {clear ? (
        <ClearScreen
          accepted={counts.accepted}
          rejected={counts.rejected}
          hasUndo={undoStack.current.length > 0}
          onUndo={undoLast}
          onClose={close}
        />
      ) : (
        current && (
          <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
            <div
              style={{
                flex: 1,
                minWidth: 0,
                display: "flex",
                flexDirection: "column",
              }}
            >
              <ImagePane
                mediaId={current.media_id}
                zoom={zoom}
                setZoom={setZoom}
                panRef={panRef}
              />
              <ClaimsStrip
                claims={grounding.data?.grounding?.claims ?? []}
                threshold={grounding.data?.threshold ?? 55}
              />
            </div>
            <DecisionPane
              finding={current}
              editing={editing}
              draft={draft}
              setDraft={setDraft}
              onEdit={() => setEditing(true)}
              onAccept={accept}
              onReject={reject}
              onPrev={() => goto(index - 1)}
              onNext={() => goto(index + 1)}
              samePending={pending.filter(
                (f) => f.rule_id != null && f.rule_id === current.rule_id,
              )}
              onAcceptRule={() =>
                current.rule_id != null &&
                datasetId != null &&
                decideBulk.mutate({
                  dataset_id: datasetId,
                  rule_id: current.rule_id,
                })
              }
            />
          </div>
        )
      )}
    </div>
  );
}

// -- Header ------------------------------------------------------------------

function Header({
  total,
  position,
  pct,
  hasUndo,
  onUndo,
  onClose,
}: {
  total: number;
  position: number;
  pct: number;
  hasUndo: boolean;
  onUndo: () => void;
  onClose: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "12px 16px",
        borderBottom: `1px solid ${colors.border}`,
        background: colors.toolbar,
      }}
    >
      <b style={{ fontSize: 13, color: colors.text }}>Review wizard</b>
      <span style={{ fontSize: 12, color: colors.textMuted }}>
        {position} / {total}
      </span>
      <div style={{ width: 160 }}>
        <ProgressBar pct={pct} color={colors.info} />
      </div>
      <div style={{ flex: 1 }} />
      <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <Kbd>A</Kbd> accept <Kbd>R</Kbd> reject <Kbd>E</Kbd> edit{" "}
        <Kbd>Z</Kbd> undo <Kbd>←→</Kbd> nav <Kbd>Esc</Kbd> close
      </span>
      {hasUndo && (
        <Button variant="ghost" onClick={onUndo}>
          ↩ Undo (Z)
        </Button>
      )}
      <button title="Close" onClick={onClose} style={closeButton}>
        ✕
      </button>
    </div>
  );
}

// -- Image pane --------------------------------------------------------------

function ImagePane({
  mediaId,
  zoom,
  setZoom,
  panRef,
}: {
  mediaId: number;
  zoom: Zoom;
  setZoom: (z: Zoom) => void;
  panRef: React.MutableRefObject<{ x: number; y: number } | null>;
}) {
  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const next = Math.min(
      8,
      Math.max(1, zoom.scale * (event.deltaY < 0 ? 1.15 : 0.87)),
    );
    setZoom(next === 1 ? RESET : { ...zoom, scale: next });
  };
  return (
    <div
      onWheel={onWheel}
      onDoubleClick={() =>
        setZoom(zoom.scale > 1 ? RESET : { scale: 2.4, tx: 0, ty: 0 })
      }
      onMouseDown={(event) => {
        if (zoom.scale <= 1) return;
        panRef.current = { x: event.clientX - zoom.tx, y: event.clientY - zoom.ty };
      }}
      onMouseMove={(event) => {
        if (!panRef.current) return;
        setZoom({
          ...zoom,
          tx: event.clientX - panRef.current.x,
          ty: event.clientY - panRef.current.y,
        });
      }}
      onMouseUp={() => (panRef.current = null)}
      onMouseLeave={() => (panRef.current = null)}
      style={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: colors.app,
        overflow: "hidden",
        cursor: zoom.scale > 1 ? "grab" : "default",
        position: "relative",
      }}
    >
      <img
        src={fileUrl(mediaId)}
        alt=""
        draggable={false}
        style={{
          maxWidth: "100%",
          maxHeight: "100%",
          transform: `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`,
          transition: panRef.current ? "none" : "transform 0.12s",
        }}
      />
      <div style={zoomBadge}>
        <button style={zoomBtn} onClick={() => setZoom(RESET)}>
          1:1
        </button>
        <span style={{ fontFamily: font.mono, fontSize: 11 }}>
          {zoom.scale.toFixed(1)}×
        </span>
      </div>
    </div>
  );
}

function ClaimsStrip({
  claims,
  threshold,
}: {
  claims: GroundedClaim[];
  threshold: number;
}) {
  if (claims.length === 0) return null;
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 6,
        padding: "10px 16px",
        borderTop: `1px solid ${colors.border}`,
        background: colors.panel,
        maxHeight: 88,
        overflowY: "auto",
      }}
    >
      <span
        style={{ fontSize: 10, color: colors.textFaint, alignSelf: "center" }}
      >
        SigLIP claims
      </span>
      {claims.map((claim) => {
        const color = groundingColor(claim.score, threshold, claim.rejected);
        return (
          <span
            key={claim.id}
            title={`${claim.kind} · ${claim.score.toFixed(1)}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              padding: "2px 8px",
              borderRadius: radii.chip,
              border: `1px solid ${color}`,
              color: colors.textSecondary,
              fontSize: 11,
            }}
          >
            {claim.text}
            <b style={{ color, fontFamily: font.mono, fontSize: 10 }}>
              {Math.round(claim.score)}
            </b>
          </span>
        );
      })}
    </div>
  );
}

// -- Decision pane -----------------------------------------------------------

function DecisionPane({
  finding,
  editing,
  draft,
  setDraft,
  onEdit,
  onAccept,
  onReject,
  onPrev,
  onNext,
  samePending,
  onAcceptRule,
}: {
  finding: ReviewFinding;
  editing: boolean;
  draft: string;
  setDraft: (value: string) => void;
  onEdit: () => void;
  onAccept: () => void;
  onReject: () => void;
  onPrev: () => void;
  onNext: () => void;
  samePending: ReviewFinding[];
  onAcceptRule: () => void;
}) {
  const style = kindStyle(finding.rule_kind);
  return (
    <div
      style={{
        width: 480,
        flex: "none",
        borderLeft: `1px solid ${colors.border}`,
        background: colors.panel,
        overflowY: "auto",
        padding: 18,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      {finding.stale && (
        <div style={staleBanner}>
          The caption changed since this run — the diff may not apply.
        </div>
      )}
      <Card title={<Badge color={style.color} background={style.background}>{style.label}</Badge>}>
        {finding.rule_text ?? "Built-in integrity check"}
      </Card>
      <Card
        title="Judge's finding"
        accent={colors.info}
        background={colors.groundingBg}
      >
        {finding.note || "—"}
      </Card>
      <div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: 6 }}>
          <span style={cardLabel}>Proposed caption</span>
          <div style={{ flex: 1 }} />
          {!editing && (
            <button style={linkButton} onClick={onEdit}>
              ✎ Edit inline (E)
            </button>
          )}
        </div>
        {editing ? (
          <>
            <textarea
              autoFocus
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={5}
              style={editArea}
            />
            <p style={{ fontSize: 10.5, color: colors.textFaint, margin: "4px 0 0" }}>
              Accept applies your edited text.
            </p>
          </>
        ) : (
          <div style={diffBox}>
            <DiffText
              before={finding.caption_before}
              after={finding.caption_after}
              style={{ fontSize: 14, color: colors.text }}
            />
          </div>
        )}
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <Button variant="ghost" onClick={onReject} style={{ flex: 1 }}>
          ✕ Reject — keep original
        </Button>
        <Button variant="accent" onClick={onAccept} style={{ flex: 1 }}>
          ✓ Accept fix
        </Button>
      </div>
      {samePending.length >= 2 && (
        <Button variant="ghost" block onClick={onAcceptRule}>
          ⚡ Accept all {samePending.length} pending fixes from this rule
        </Button>
      )}
      <div style={{ display: "flex", gap: 14, justifyContent: "center" }}>
        <button style={linkButton} onClick={onPrev}>
          ‹ previous
        </button>
        <button style={linkButton} onClick={onNext}>
          skip ›
        </button>
      </div>
    </div>
  );
}

function ClearScreen({
  accepted,
  rejected,
  hasUndo,
  onUndo,
  onClose,
}: {
  accepted: number;
  rejected: number;
  hasUndo: boolean;
  onUndo: () => void;
  onClose: () => void;
}) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 14,
        color: colors.textSecondary,
      }}
    >
      <div style={{ fontSize: 32, color: colors.ok }}>✓</div>
      <div style={{ fontSize: 15 }}>
        Queue clear · {accepted} accepted · {rejected} rejected
      </div>
      <p style={{ fontSize: 12, color: colors.textMuted }}>
        Accepted captions were saved as new revisions.
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        {hasUndo && (
          <Button variant="ghost" onClick={onUndo}>
            ↩ Undo last
          </Button>
        )}
        <Button variant="accent" onClick={onClose}>
          Back to queue
        </Button>
      </div>
    </div>
  );
}

// -- Small bits --------------------------------------------------------------

function Card({
  title,
  children,
  accent,
  background,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
  accent?: string;
  background?: string;
}) {
  return (
    <div
      style={{
        border: `1px solid ${accent ? colors.groundingBorder : colors.border}`,
        borderRadius: radii.control,
        background: background ?? colors.card,
        padding: 12,
      }}
    >
      <div style={cardLabel}>{title}</div>
      <div
        style={{
          marginTop: 6,
          fontSize: 13,
          color: accent ?? colors.textSecondary,
          lineHeight: 1.5,
        }}
      >
        {children}
      </div>
    </div>
  );
}

const overlay = {
  position: "fixed",
  inset: 0,
  zIndex: 900,
  background: colors.app,
  display: "flex",
  flexDirection: "column",
  boxShadow: shadow.modal,
} as const;

const cardLabel = {
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  fontWeight: 700,
  color: colors.textMuted,
} as const;

const diffBox = {
  border: `1px solid ${colors.border}`,
  borderRadius: radii.control,
  background: colors.card,
  padding: 12,
} as const;

const editArea = {
  width: "100%",
  padding: 10,
  borderRadius: radii.control,
  border: `1px solid ${colors.accentBorder}`,
  background: colors.input,
  color: colors.text,
  fontSize: 13.5,
  lineHeight: 1.6,
  fontFamily: font.sans,
  resize: "vertical",
} as const;

const linkButton = {
  border: "none",
  background: "transparent",
  color: colors.accent,
  cursor: "pointer",
  fontSize: 11.5,
  fontWeight: 600,
} as const;

const closeButton = {
  border: `1px solid ${colors.borderControl}`,
  background: "transparent",
  color: colors.textMuted,
  cursor: "pointer",
  borderRadius: radii.control,
  width: 28,
  height: 26,
} as const;

const zoomBadge = {
  position: "absolute",
  bottom: 12,
  right: 12,
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "4px 8px",
  borderRadius: radii.control,
  background: "rgba(0,0,0,0.55)",
  color: colors.text,
} as const;

const zoomBtn = {
  border: `1px solid ${colors.borderControl}`,
  background: "transparent",
  color: colors.text,
  cursor: "pointer",
  borderRadius: 4,
  fontSize: 10.5,
  padding: "2px 6px",
} as const;

const staleBanner = {
  padding: "7px 10px",
  borderRadius: radii.control,
  background: colors.watermarkAmberBg,
  color: colors.warn,
  fontSize: 11.5,
} as const;
