/**
 * Watermark Lab (v2) — the self-contained virtual-patch overlay.
 *
 * A big overlay (rail · inventory · review) that erases watermarks with FLUX.2
 * klein edit patches composed over the original — the source file is never
 * touched unless the user explicitly flattens a media to disk. The Lab ignores
 * the Media page: everything is chosen inside it. Three tabs (Medias /
 * Watermarked / Patched) each carry the Media page's own filters, pagination
 * and check-all; the workflow is filter → check → scan → review → patch.
 *
 * Detection defaults to OWLv2 zero-shot (editable text queries), YOLO the
 * exclusive secondary. Scan and patch are decoupled: scan only detects, patch
 * only erases already-detected zones, scan+patch chains both. The rail is
 * light — detection settings up top, the FLUX engine folded away.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useAddZone,
  useDeleteZone,
  useDismissMedia,
  useDismissSelection,
  useEditZone,
  useFlattenMedia,
  useJobResult,
  usePatchMedia,
  useRegenerateZone,
  useRevertMedia,
  useRevertSelection,
  useUpdateWatermarkConfig,
  useWatermarkConfig,
  useWatermarkInventory,
  useWatermarkMedia,
  useWatermarkPatch,
  useWatermarkScan,
  useWatermarkScanAndPatch,
  type WatermarkSelection,
} from "../../api/hooks";
import type {
  WatermarkBox,
  WatermarkConfig,
  WatermarkInventoryItem,
  WatermarkMedia,
  WatermarkPrefs,
  WatermarkTab,
  WatermarkZone,
} from "../../api/types";
import {
  colors,
  font,
  radii,
  shadow,
  watermarkScore,
  watermarkStatus,
} from "../../design/tokens";
import { useJobsStore } from "../../store/jobsStore";
import { useUiStore } from "../../store/uiStore";
import { Button, ProgressBar, Segmented, Slider, Spinner } from "../atoms";
import { api } from "../../api/client";
import { TagFilter, type SelectedTag } from "../molecules/TagFilter";
import { FilePickerModal } from "./FilePickerModal";

const MODEL_FILE_EXTS = "safetensors,gguf";
const PAGE_SIZE = 60;

const SORTS = [
  { value: "date_desc", label: "Newest" },
  { value: "quality_desc", label: "Quality ↓" },
  { value: "dimension_desc", label: "Largest" },
];

const TABS: { value: WatermarkTab; label: string }[] = [
  { value: "media", label: "Medias" },
  { value: "watermarked", label: "Watermarked" },
  { value: "patched", label: "Patched" },
];

const VIOLET_BTN = {
  background: colors.watermark,
  color: colors.onAccent,
} as const;

/** Ask the queue to stop a running job at its next checkpoint. */
function stopJob(jobId: string): void {
  api.post(`/jobs/${jobId}/stop`).catch(() => {});
}

/** Watch a submitted job to completion, firing `onDone` once. */
function useJobWatcher(jobId: string | null, onDone: () => void) {
  const jobs = useJobsStore((state) => state.jobs);
  const doneRef = useRef<string | null>(null);
  const job = jobId ? jobs[jobId] : undefined;
  useEffect(() => {
    if (!jobId || !job) return;
    if (
      (job.state === "done" ||
        job.state === "error" ||
        job.state === "stopped") &&
      doneRef.current !== jobId
    ) {
      doneRef.current = jobId;
      onDone();
    }
  }, [jobId, job, onDone]);
  return job;
}

export function WatermarkLab() {
  const state = useUiStore((s) => s.watermark);
  const close = useUiStore((s) => s.closeWatermark);
  const setWatermark = useUiStore((s) => s.setWatermark);
  const config = useWatermarkConfig();
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    if (!state.open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (state.focusKey) setWatermark({ focusKey: null });
        else close();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.open, state.focusKey, close, setWatermark]);

  if (!state.open) return null;

  return (
    <div
      onClick={close}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,11,14,0.72)",
        zIndex: 72,
        display: "flex",
        justifyContent: "center",
        padding: 0,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: "100%",
          maxWidth: 1560,
          height: "100%",
          background: colors.panel,
          border: `1px solid ${colors.borderHover}`,
          borderRadius: 0,
          boxShadow: shadow.modal,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <Header total={config.data?.media_total ?? 0} onClose={close} />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <Rail config={config.data} jobId={jobId} onStop={setJobId} />
          <Inventory
            config={config.data}
            jobId={jobId}
            onJob={setJobId}
            initialTab={state.initialTab}
            focusKey={state.focusKey}
            onFocus={(key) => setWatermark({ focusKey: key })}
          />
          {state.focusKey && (
            <ReviewPanel
              mediaKey={state.focusKey}
              defaultMode={config.data?.prefs.compare_mode ?? "slider"}
              onClose={() => setWatermark({ focusKey: null })}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function Header({ total, onClose }: { total: number; onClose: () => void }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "12px 16px",
        borderBottom: `1px solid ${colors.border}`,
      }}
    >
      <div
        style={{
          width: 26,
          height: 26,
          borderRadius: 7,
          display: "grid",
          placeItems: "center",
          background: `linear-gradient(135deg, ${colors.watermark}, ${colors.watermarkStrong})`,
          color: "#fff",
          fontSize: 14,
        }}
      >
        ◪
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 700, fontSize: 14 }}>
          Watermark Lab — virtual patches
        </div>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textMuted,
          }}
        >
          {total.toLocaleString()} media · OWLv2 zero-shot detection · virtual
          PNG patches — originals are never modified
        </div>
      </div>
      <Button onClick={onClose} style={{ background: colors.raised }}>
        ✕ Close
      </Button>
    </div>
  );
}

// -- Rail -------------------------------------------------------------------

function Section({
  title,
  right,
  children,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        padding: "12px 14px",
        borderTop: `1px solid ${colors.border}`,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: 10.5,
          fontWeight: 700,
          letterSpacing: 0.4,
          textTransform: "uppercase",
          color: colors.textMuted,
        }}
      >
        <span>{title}</span>
        {right}
      </div>
      {children}
    </div>
  );
}

function Check({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <label
      style={{
        display: "flex",
        gap: 7,
        alignItems: "flex-start",
        fontSize: 11.5,
        color: colors.textSecondary,
        cursor: "pointer",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        style={{ marginTop: 1 }}
      />
      {label}
    </label>
  );
}

/** A radio card for a model choice: bold title + one-line note. */
function RadioCard({
  active,
  title,
  note,
  disabled,
  onClick,
}: {
  active: boolean;
  title: string;
  note: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={disabled ? undefined : onClick}
      style={{
        padding: "7px 9px",
        borderRadius: radii.control,
        border: `1px solid ${active ? colors.watermark : colors.borderControl}`,
        background: active ? colors.watermarkBg : colors.input,
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      <div
        style={{
          fontSize: 11.5,
          fontWeight: 700,
          color: active ? colors.watermark : colors.text,
        }}
      >
        {title}
      </div>
      <div style={{ fontSize: 9.5, color: colors.textMuted, marginTop: 1 }}>
        {note}
      </div>
    </div>
  );
}

/** A mono one-line note, the rail's recurring caption style. */
function MonoNote({
  children,
  color,
}: {
  children: React.ReactNode;
  color?: string;
}) {
  return (
    <div
      style={{
        fontFamily: font.mono,
        fontSize: 9.5,
        lineHeight: 1.5,
        color: color ?? colors.textFaint,
      }}
    >
      {children}
    </div>
  );
}

type RailConfig = WatermarkConfig | undefined;

function Rail({
  config,
  jobId,
  onStop,
}: {
  config: RailConfig;
  jobId: string | null;
  onStop: (id: string | null) => void;
}) {
  const update = useUpdateWatermarkConfig();
  const [prefs, setPrefs] = useState<WatermarkPrefs | null>(null);
  const [queryInput, setQueryInput] = useState("");
  const [tagInput, setTagInput] = useState("");
  const [fluxOpen, setFluxOpen] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (config && !prefs) setPrefs(config.prefs);
  }, [config, prefs]);

  const job = useJobWatcher(jobId, () => {});
  const result = useJobResult<{
    scanned: number;
    media: number;
    detected: number;
    patched: number;
  }>(jobId, job?.state === "done");

  if (!prefs || !config) {
    return (
      <div style={railStyle}>
        <Spinner />
      </div>
    );
  }

  const patch = (partial: Partial<WatermarkPrefs>) => {
    const next = { ...prefs, ...partial };
    setPrefs(next);
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => update.mutate(partial), 300);
  };

  const running = !!jobId && (job?.state === "running" || job?.state === "queued");
  const isOwl = prefs.detector === "owlv2";
  const summary = result.data?.result;

  const addQuery = () => {
    const term = queryInput.trim();
    if (term && !prefs.owlv2_queries.includes(term)) {
      patch({ owlv2_queries: [...prefs.owlv2_queries, term] });
    }
    setQueryInput("");
  };

  return (
    <div style={railStyle}>
      {running && (
        <Section title="Progress">
          <ProgressBar pct={job?.pct ?? 0} color={colors.watermark} />
          <div style={runNote}>{job?.sub ?? "Queued…"}</div>
          <Button
            block
            onClick={() => jobId && stopJob(jobId)}
            style={{ background: colors.raised }}
          >
            ✕ Stop — keep what is already done
          </Button>
        </Section>
      )}
      {!running && job?.state === "done" && summary && (
        <Section title="Last run">
          <div style={{ ...runNote, color: colors.ok }}>
            ✓ {summary.patched} patched · {summary.detected} zone(s) over{" "}
            {summary.media} media · {summary.scanned} scanned
          </div>
          <Button
            block
            onClick={() => onStop(null)}
            style={{ background: colors.raised }}
          >
            Dismiss
          </Button>
        </Section>
      )}

      <Section title="Detection">
        <select
          value={prefs.detector}
          onChange={(event) =>
            patch({ detector: event.target.value as WatermarkPrefs["detector"] })
          }
          style={selectStyle}
        >
          <option value="owlv2">OWLv2 · zero-shot open-vocabulary</option>
          <option value="yolo">YOLO11-n watermark · .pt</option>
        </select>
        {isOwl ? (
          <>
            <select
              value={prefs.owlv2_model}
              onChange={(event) => patch({ owlv2_model: event.target.value })}
              style={selectStyle}
            >
              {config.owlv2_models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
            <MonoNote>
              Auto-downloaded & cached on first scan · large resolves smaller
              marks
            </MonoNote>
            <div style={{ fontSize: 10.5, color: colors.textMuted }}>
              Search terms
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {prefs.owlv2_queries.map((term) => (
                <span key={term} style={tagChip}>
                  {term}
                  <span
                    onClick={() =>
                      patch({
                        owlv2_queries: prefs.owlv2_queries.filter(
                          (t) => t !== term,
                        ),
                      })
                    }
                    style={{ cursor: "pointer", opacity: 0.7 }}
                  >
                    ✕
                  </span>
                </span>
              ))}
            </div>
            <input
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addQuery();
              }}
              placeholder="+ term ⏎"
              style={inputStyle}
            />
            <MonoNote>Sent as text queries; remembered between sessions.</MonoNote>
            <div style={{ fontSize: 11, color: colors.textMuted }}>
              Confidence ≥ {prefs.owlv2_confidence}%
            </div>
            <Slider
              min={5}
              max={90}
              step={5}
              value={prefs.owlv2_confidence}
              onChange={(value) => patch({ owlv2_confidence: value })}
            />
          </>
        ) : (
          <>
            <select
              value={prefs.yolo_model}
              onChange={(event) => patch({ yolo_model: event.target.value })}
              style={selectStyle}
            >
              <option value="">— none (drop a .pt in the models dir) —</option>
              {config.yolo_models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <MonoNote>Folder: {config.yolo_models_dir} — set in Settings</MonoNote>
            <div style={{ fontSize: 11, color: colors.textMuted }}>
              Confidence ≥ {prefs.confidence_min}%
            </div>
            <Slider
              min={10}
              max={95}
              step={5}
              value={prefs.confidence_min}
              onChange={(value) => patch({ confidence_min: value })}
            />
          </>
        )}
        <MonoNote>
          Zones scoring below the threshold are discarded at scan time.
        </MonoNote>
      </Section>

      <Section
        title="Patch engine — FLUX.2"
        right={
          <span
            onClick={() => setFluxOpen((open) => !open)}
            style={{ cursor: "pointer", color: colors.watermark }}
          >
            {fluxOpen ? "▾ hide" : "▸ edit"}
          </span>
        }
      >
        <MonoNote color={colors.textMuted}>{fluxSummary(prefs)}</MonoNote>
        {fluxOpen && (
          <EditingSection prefs={prefs} config={config} patch={patch} />
        )}
      </Section>

      <Section title="After patch">
        <Check
          checked={prefs.tag_cleanup}
          onChange={(value) => patch({ tag_cleanup: value })}
          label="Strip these tags once the media is fully patched"
        />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {prefs.tags_to_remove.map((tag) => (
            <span key={tag} style={tagChip}>
              {tag}
              <span
                onClick={() =>
                  patch({
                    tags_to_remove: prefs.tags_to_remove.filter(
                      (t) => t !== tag,
                    ),
                  })
                }
                style={{ cursor: "pointer", opacity: 0.7 }}
              >
                ✕
              </span>
            </span>
          ))}
        </div>
        <input
          value={tagInput}
          onChange={(event) => setTagInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && tagInput.trim()) {
              patch({
                tags_to_remove: [
                  ...prefs.tags_to_remove,
                  tagInput.trim().toLowerCase(),
                ],
              });
              setTagInput("");
            }
          }}
          placeholder="+ tag ⏎"
          style={inputStyle}
        />
      </Section>
    </div>
  );
}

/** The one-line résumé of the folded FLUX engine block. */
function fluxSummary(prefs: WatermarkPrefs): string {
  const kv = prefs.model === "9b" && prefs.precision !== "nvfp4" && prefs.kv;
  const label = `klein ${prefs.model === "9b" ? "9B" : "4B"}${
    kv ? " KV" : ""
  } ${prefs.precision}`;
  return `${label} · “${prefs.prompt}” · crops ≤ ${prefs.max_res} px · mask +${prefs.dilate_px} px`;
}

// -- Editing (FLUX.2 klein) rail section -------------------------------------

/** Whether the KV quick-edit variant exists for a model/precision combo. */
function kvUsable(prefs: WatermarkPrefs): boolean {
  return prefs.model === "9b" && prefs.precision !== "nvfp4";
}

/** A rough "~{vram} GB · ~{sec}s / patch" hint by model/precision/KV. */
function fluxEstimate(prefs: WatermarkPrefs): string {
  const vram =
    prefs.model === "9b"
      ? { std: 22, fp8: 13, nvfp4: 9 }[prefs.precision]
      : { std: 11, fp8: 7, nvfp4: 5 }[prefs.precision];
  const base = prefs.model === "9b" ? 7 : 4;
  const secs = kvUsable(prefs) && prefs.kv ? Math.round(base * 0.5) : base;
  const quick = kvUsable(prefs) && prefs.kv ? " · quick image edit" : "";
  return `~${vram} GB VRAM · ~${secs}s / patch 1024px${quick}`;
}

function EditingSection({
  prefs,
  config,
  patch,
}: {
  prefs: WatermarkPrefs;
  config: WatermarkConfig;
  patch: (partial: Partial<WatermarkPrefs>) => void;
}) {
  const encoder = prefs.text_encoder;
  const patchEncoder = (part: Partial<WatermarkPrefs["text_encoder"]>) =>
    patch({ text_encoder: { ...encoder, ...part } });
  const [picker, setPicker] = useState<null | "model" | "encoder">(null);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <RadioCard
        active={prefs.model === "9b"}
        title="FLUX.2 klein 9B"
        note="max quality · text & complex backgrounds"
        onClick={() => patch({ model: "9b" })}
      />
      <RadioCard
        active={prefs.model === "4b"}
        title="FLUX.2 klein 4B"
        note="light · small GPUs · simple watermarks"
        onClick={() => patch({ model: "4b" })}
      />
      <Segmented
        value={prefs.precision}
        onChange={(value) =>
          patch({ precision: value as WatermarkPrefs["precision"] })
        }
        options={[
          { value: "std", label: "standard" },
          { value: "fp8", label: "fp8" },
          { value: "nvfp4", label: "nvfp4" },
        ]}
      />
      <Check
        checked={kvUsable(prefs) && prefs.kv}
        onChange={(value) => patch({ kv: value })}
        label={
          kvUsable(prefs)
            ? "KV variant — quick image edit (ideal for watermarks)"
            : "KV variant — unavailable (9B standard or fp8 only)"
        }
      />
      <Segmented
        value={prefs.source}
        onChange={(value) =>
          patch({ source: value as WatermarkPrefs["source"] })
        }
        options={[
          { value: "hf", label: "Hugging Face" },
          { value: "local", label: "Local model" },
        ]}
      />
      {prefs.source === "hf" ? (
        <div style={monoPanel}>
          <div style={{ color: colors.watermark }}>
            {config.flux_repo ?? "— invalid combo —"}
          </div>
          <MonoNote>
            ✓ official repo · VAE included · downloaded & cached automatically
          </MonoNote>
        </div>
      ) : (
        <>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              value={prefs.local_model_path}
              onChange={(event) =>
                patch({ local_model_path: event.target.value })
              }
              placeholder="path to .safetensors / .gguf"
              style={{ ...inputStyle, flex: 1 }}
            />
            <Button onClick={() => setPicker("model")}>browse…</Button>
          </div>
          <MonoNote>
            ✓ VAE: ae.safetensors detected next to the model, when present
          </MonoNote>
        </>
      )}

      <div style={{ fontSize: 10.5, color: colors.textMuted, marginTop: 2 }}>
        Text encoder — Qwen3 (required)
      </div>
      <Segmented
        value={encoder.source}
        onChange={(value) =>
          patchEncoder({ source: value as WatermarkPrefs["source"] })
        }
        options={[
          { value: "hf", label: "Hugging Face" },
          { value: "local", label: "Local file" },
        ]}
      />
      {encoder.source === "hf" ? (
        <>
          <select
            value={encoder.version}
            onChange={(event) => patchEncoder({ version: event.target.value })}
            style={selectStyle}
          >
            <option value="">bundled with the model (recommended)</option>
            <option value="fp16">fp16</option>
            <option value="fp8">fp8 (8-bit)</option>
          </select>
          <MonoNote>{config.encoder_repo} · GGUF: use a local file</MonoNote>
        </>
      ) : (
        <>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              value={encoder.path}
              onChange={(event) => patchEncoder({ path: event.target.value })}
              placeholder="path to the Qwen3 encoder file"
              style={{ ...inputStyle, flex: 1 }}
            />
            <Button onClick={() => setPicker("encoder")}>browse…</Button>
          </div>
          <MonoNote>must match the model</MonoNote>
        </>
      )}

      <MonoNote color={colors.textMuted}>{fluxEstimate(prefs)}</MonoNote>

      <div style={{ fontSize: 10.5, color: colors.textMuted, marginTop: 2 }}>
        Edit instruction
      </div>
      <input
        value={prefs.prompt}
        onChange={(event) => patch({ prompt: event.target.value })}
        placeholder="remove any watermark, logo or brand"
        style={inputStyle}
      />

      <div style={{ fontSize: 11, color: colors.textMuted }}>
        Max res · {prefs.max_res} px
      </div>
      <Slider
        min={512}
        max={1536}
        step={64}
        value={prefs.max_res}
        onChange={(value) => patch({ max_res: value })}
      />
      <Segmented
        value={prefs.res_side}
        onChange={(value) =>
          patch({ res_side: value as WatermarkPrefs["res_side"] })
        }
        options={[
          { value: "long", label: "long side" },
          { value: "short", label: "short side" },
        ]}
      />
      <MonoNote>
        The detection box is cropped and sent to FLUX (ratio kept — 1536×1536
        hard limit); only the box itself is pasted back as a patch.
      </MonoNote>

      <div style={{ fontSize: 11, color: colors.textMuted }}>
        Context margin + {prefs.dilate_px} px
      </div>
      <Slider
        min={0}
        max={32}
        step={2}
        value={prefs.dilate_px}
        onChange={(value) => patch({ dilate_px: value })}
      />
      <MonoNote>
        Extra pixels sent around the box so FLUX sees clean background — never
        pasted back, so only the watermark's rectangle is ever replaced.
      </MonoNote>
      {picker && (
        <FilePickerModal
          exts={MODEL_FILE_EXTS}
          initialPath={
            picker === "model" ? prefs.local_model_path : encoder.path
          }
          hint={
            picker === "model"
              ? "Use this FLUX.2 transformer file."
              : "Use this Qwen3 encoder file."
          }
          onSelect={(path) =>
            picker === "model"
              ? patch({ local_model_path: path })
              : patchEncoder({ path })
          }
          onClose={() => setPicker(null)}
        />
      )}
    </div>
  );
}

// -- Inventory --------------------------------------------------------------

function Inventory({
  config,
  jobId,
  onJob,
  initialTab,
  focusKey,
  onFocus,
}: {
  config: RailConfig;
  jobId: string | null;
  onJob: (id: string) => void;
  initialTab: WatermarkTab | null;
  focusKey: string | null;
  onFocus: (key: string) => void;
}) {
  const [tab, setTab] = useState<WatermarkTab>(initialTab ?? "media");
  // The filter is persisted across sessions (like the Media page) so the Lab
  // reopens with the same tags, match, sort and favorites.
  const { include, exclude, match, sort, favOnly } = useUiStore(
    (s) => s.watermarkView,
  );
  const setView = useUiStore((s) => s.setWatermarkView);
  const setInclude = (next: SelectedTag[]) => setView({ include: next });
  const setExclude = (next: SelectedTag[]) => setView({ exclude: next });
  const setMatch = (next: string) => setView({ match: next });
  const setSort = (next: string) => setView({ sort: next });
  const setFavOnly = (next: boolean) => setView({ favOnly: next });
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [allPages, setAllPages] = useState(false);
  const anchor = useRef<number | null>(null);

  const scan = useWatermarkScan();
  const scanPatch = useWatermarkScanAndPatch();
  const patch = useWatermarkPatch();
  const dismiss = useDismissSelection();
  const revert = useRevertSelection();

  const filterKey = `${include.map((t) => t.id).join(",")}|${exclude
    .map((t) => t.id)
    .join(",")}|${match}|${favOnly}|${sort}`;
  // Reset page/selection whenever the filter, tab or sort changes.
  useEffect(() => {
    setPage(1);
    setSelected(new Set());
    setAllPages(false);
    anchor.current = null;
  }, [filterKey, tab]);

  const query = useMemo(
    () => ({
      tab,
      tag_ids: include.map((t) => t.id),
      exclude_tag_ids: exclude.map((t) => t.id),
      match,
      favorites_only: favOnly,
      sort,
      offset: (page - 1) * PAGE_SIZE,
      limit: PAGE_SIZE,
    }),
    [tab, include, exclude, match, favOnly, sort, page],
  );
  const inventory = useWatermarkInventory(query, !!config);
  const items = inventory.data?.items ?? [];
  const counts = inventory.data?.counts ?? { media: 0, watermarked: 0, patched: 0 };
  const total = inventory.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const selectionBody = (): WatermarkSelection =>
    allPages
      ? {
          select_all: true,
          tab,
          tag_ids: include.map((t) => t.id),
          exclude_tag_ids: exclude.map((t) => t.id),
          match,
          favorites_only: favOnly,
        }
      : { media_ids: [...selected], tab };

  const clearSelection = () => {
    setSelected(new Set());
    setAllPages(false);
    anchor.current = null;
  };
  const selCount = allPages ? total : selected.size;
  const busy =
    scan.isPending ||
    scanPatch.isPending ||
    patch.isPending ||
    dismiss.isPending ||
    revert.isPending ||
    !!jobId;

  const runJob = (hook: typeof scan) =>
    hook.mutate(selectionBody(), {
      onSuccess: (data) => {
        onJob(data.job_id);
        clearSelection();
      },
    });

  // Selection interactions (anchor set by plain/ctrl click and checkbox;
  // shift extends from the anchor across the current page's sort order).
  const toggle = (id: number) => {
    setAllPages(false);
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    anchor.current = id;
  };
  const rangeFrom = (id: number) => {
    const ids = items.map((m) => m.media_id);
    const from = anchor.current == null ? -1 : ids.indexOf(anchor.current);
    const to = ids.indexOf(id);
    if (from < 0 || to < 0) {
      toggle(id);
      return;
    }
    const [lo, hi] = from < to ? [from, to] : [to, from];
    setAllPages(false);
    setSelected((prev) => {
      const next = new Set(prev);
      for (let i = lo; i <= hi; i += 1) next.add(ids[i]);
      return next;
    });
  };
  const onCardClick = (item: WatermarkInventoryItem, event: React.MouseEvent) => {
    if (event.shiftKey) rangeFrom(item.media_id);
    else if (event.ctrlKey || event.metaKey) toggle(item.media_id);
    else {
      anchor.current = item.media_id;
      onFocus(String(item.media_id));
    }
  };

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
          gap: 8,
          padding: "10px 14px",
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <Segmented
          value={tab}
          onChange={(value) => setTab(value as WatermarkTab)}
          options={TABS.map((t) => ({
            value: t.value,
            label: `${t.label} ${counts[t.value] ?? 0}`,
          }))}
        />
        <span style={{ flex: 1 }} />
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textMuted,
          }}
        >
          red zone = detected · purple zone = patched
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 14px",
          borderBottom: `1px solid ${colors.border}`,
          background: colors.toolbar,
          flexWrap: "wrap",
        }}
      >
        <TagFilter
          label="+ include tag"
          selected={include}
          onAdd={(tag) => setInclude([...include, tag])}
          onRemove={(id) => setInclude(include.filter((t) => t.id !== id))}
        />
        <TagFilter
          label="− exclude tag"
          selected={exclude}
          onAdd={(tag) => setExclude([...exclude, tag])}
          onRemove={(id) => setExclude(exclude.filter((t) => t.id !== id))}
        />
        <Segmented
          value={match}
          onChange={setMatch}
          options={[
            { value: "all", label: "All" },
            { value: "any", label: "Any" },
          ]}
        />
        <select
          value={sort}
          onChange={(event) => setSort(event.target.value)}
          style={selectStyle2}
        >
          {SORTS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <label style={favLabel}>
          <input
            type="checkbox"
            checked={favOnly}
            onChange={(event) => setFavOnly(event.target.checked)}
          />
          ♥ Favorites
        </label>
        <span style={{ flex: 1 }} />
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 11,
            color: colors.textMuted,
          }}
        >
          {total.toLocaleString()} media match
        </span>
      </div>

      <SelectAllRow
        tab={tab}
        total={total}
        pageCount={pageCount}
        allPages={allPages}
        selectedCount={selected.size}
        onToggle={() => {
          setAllPages((value) => !value);
          setSelected(new Set());
          anchor.current = null;
        }}
      />

      <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
        {items.length === 0 ? (
          <div
            style={{
              color: colors.textMuted,
              fontSize: 12,
              padding: 20,
              lineHeight: 1.6,
            }}
          >
            {emptyText(tab)}
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
              gap: 10,
            }}
          >
            {items.map((item) => (
              <InventoryCard
                key={item.media_id}
                item={item}
                focused={focusKey === String(item.media_id)}
                selected={allPages || selected.has(item.media_id)}
                onClick={(event) => onCardClick(item, event)}
                onToggle={() => toggle(item.media_id)}
              />
            ))}
          </div>
        )}
      </div>

      {selCount > 0 && (
        <BatchBar
          tab={tab}
          count={selCount}
          busy={busy}
          onScan={() => runJob(scan)}
          onScanPatch={() => runJob(scanPatch)}
          onPatch={() => runJob(patch)}
          onDismiss={() =>
            dismiss.mutate(selectionBody(), { onSuccess: clearSelection })
          }
          onRevert={() =>
            revert.mutate(selectionBody(), { onSuccess: clearSelection })
          }
          onClearZones={() =>
            dismiss.mutate(selectionBody(), { onSuccess: clearSelection })
          }
        />
      )}

      {pageCount > 1 && (
        <div style={pager}>
          <Button disabled={page <= 1} onClick={() => setPage(page - 1)}>
            ‹ prev
          </Button>
          <span>
            page {page} / {pageCount}
          </span>
          <Button
            disabled={page >= pageCount}
            onClick={() => setPage(page + 1)}
          >
            next ›
          </Button>
          <span style={{ color: colors.textFaint }}>
            {total.toLocaleString()} media · {PAGE_SIZE} per page
          </span>
        </div>
      )}
    </div>
  );
}

function emptyText(tab: WatermarkTab): string {
  if (tab === "watermarked") {
    return "No watermarked media here — run a scan from the Medias tab.";
  }
  if (tab === "patched") {
    return "Nothing patched yet — patch zones from the Watermarked tab.";
  }
  return "No media matches this filter.";
}

function SelectAllRow({
  tab,
  total,
  pageCount,
  allPages,
  selectedCount,
  onToggle,
}: {
  tab: WatermarkTab;
  total: number;
  pageCount: number;
  allPages: boolean;
  selectedCount: number;
  onToggle: () => void;
}) {
  const hint =
    tab === "media"
      ? "the whole filtered result will be scanned"
      : tab === "watermarked"
        ? "acts on every watermarked media in the filter"
        : "acts on every patched media in the filter";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 14px",
        borderBottom: `1px solid ${colors.border}`,
        fontSize: 11,
        color: colors.textMuted,
      }}
    >
      <label style={{ display: "flex", gap: 6, cursor: "pointer" }}>
        <input type="checkbox" checked={allPages} onChange={onToggle} />
        Select all {total.toLocaleString()} media ({pageCount} pages)
      </label>
      <span style={{ color: colors.textFaint }}>— {hint}</span>
      {!allPages && selectedCount > 0 && (
        <span style={{ marginLeft: "auto", color: colors.watermark }}>
          {selectedCount} selected
        </span>
      )}
    </div>
  );
}

function BatchBar({
  tab,
  count,
  busy,
  onScan,
  onScanPatch,
  onPatch,
  onDismiss,
  onRevert,
  onClearZones,
}: {
  tab: WatermarkTab;
  count: number;
  busy: boolean;
  onScan: () => void;
  onScanPatch: () => void;
  onPatch: () => void;
  onDismiss: () => void;
  onRevert: () => void;
  onClearZones: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 14px",
        borderTop: `1px solid ${colors.border}`,
        background: colors.raised,
      }}
    >
      <span
        style={{ fontFamily: font.mono, fontSize: 11, color: colors.watermark }}
      >
        {count.toLocaleString()} selected
      </span>
      <span style={{ flex: 1 }} />
      {tab === "media" && (
        <>
          <Button disabled={busy} style={VIOLET_BTN} onClick={onScan}>
            ▶ Scan {count}
          </Button>
          <Button disabled={busy} style={VIOLET_BTN} onClick={onScanPatch}>
            ▶ Scan + patch {count}
          </Button>
          <Button
            disabled={busy}
            style={{ color: colors.danger }}
            onClick={onClearZones}
          >
            ✕ Clear zones
          </Button>
        </>
      )}
      {tab === "watermarked" && (
        <>
          <Button disabled={busy} style={VIOLET_BTN} onClick={onPatch}>
            ◪ Patch {count}
          </Button>
          <Button
            disabled={busy}
            style={{ color: colors.danger }}
            onClick={onDismiss}
          >
            ⊘ Dismiss detections {count}
          </Button>
        </>
      )}
      {tab === "patched" && (
        <Button disabled={busy} onClick={onRevert}>
          ↩ Revert {count} — back to Watermarked
        </Button>
      )}
    </div>
  );
}

function InventoryCard({
  item,
  focused,
  selected,
  onClick,
  onToggle,
}: {
  item: WatermarkInventoryItem;
  focused: boolean;
  selected: boolean;
  onClick: (event: React.MouseEvent) => void;
  onToggle: () => void;
}) {
  const badge = item.flattened ? "flattened" : item.status;
  const { color, label } = watermarkStatus(badge);
  const sub = cardSubline(item);
  return (
    <div
      onClick={onClick}
      style={{
        borderRadius: radii.card,
        overflow: "hidden",
        border: `1px solid ${
          selected || focused ? colors.watermark : colors.border
        }`,
        outline: selected ? `1px solid ${colors.watermark}` : "none",
        background: colors.card,
        cursor: "pointer",
      }}
    >
      <div style={{ position: "relative", aspectRatio: "4 / 3" }}>
        <img
          src={item.thumb}
          alt={item.name}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        <ZoneOverlay zones={item.zones} />
        <input
          type="checkbox"
          checked={selected}
          onClick={(event) => event.stopPropagation()}
          onChange={onToggle}
          title="Select"
          style={{
            position: "absolute",
            top: 6,
            left: 6,
            width: 16,
            height: 16,
            cursor: "pointer",
          }}
        />
        {item.score_min != null && (
          <span
            style={{
              position: "absolute",
              top: 5,
              right: 5,
              fontFamily: font.mono,
              fontSize: 10,
              padding: "1px 5px",
              borderRadius: 4,
              background: "rgba(15,16,19,0.8)",
              color: watermarkScore(item.score_min),
            }}
          >
            {Math.round(item.score_min)}
          </span>
        )}
        {item.status && (
          <span
            style={{
              position: "absolute",
              bottom: 5,
              left: 5,
              fontSize: 9.5,
              fontFamily: font.mono,
              padding: "1px 5px",
              borderRadius: 4,
              background: "rgba(15,16,19,0.82)",
              color,
            }}
          >
            ◪ {label}
          </span>
        )}
      </div>
      <div style={{ padding: "5px 7px" }}>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textSecondary,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {item.name}
        </div>
        <div style={{ fontSize: 9.5, color: colors.textMuted, marginTop: 1 }}>
          {sub}
        </div>
      </div>
    </div>
  );
}

/** The card's sub-line: dims/quality when neutral, zones/detector otherwise. */
function cardSubline(item: WatermarkInventoryItem): string {
  if (!item.status) {
    const dims = item.width && item.height ? `${item.width}×${item.height}` : "";
    const quality =
      item.quality != null ? ` · q ${Math.round(item.quality)}%` : "";
    return `${dims}${quality}` || "—";
  }
  const n = `${item.zone_count} zone${item.zone_count === 1 ? "" : "s"}`;
  if (item.status === "patched") {
    return `${n} · ${item.models.join(" + ") || "FLUX.2 klein"}`;
  }
  return `${n} · ${item.detectors.join(" / ") || "detected"}`;
}

/** Dashed rectangles over a thumbnail, one per zone (red detected / violet). */
function ZoneOverlay({ zones }: { zones: WatermarkZone[] }) {
  return (
    <>
      {zones.map((zone) => {
        const patched = zone.status !== "detected";
        return (
          <div
            key={zone.id}
            style={{
              position: "absolute",
              left: `${zone.box.x * 100}%`,
              top: `${zone.box.y * 100}%`,
              width: `${zone.box.w * 100}%`,
              height: `${zone.box.h * 100}%`,
              border: `1px dashed ${
                patched
                  ? colors.watermarkZonePatched
                  : colors.watermarkZoneDetected
              }`,
              background: patched ? "rgba(167,139,218,0.12)" : "transparent",
              pointerEvents: "none",
            }}
          />
        );
      })}
    </>
  );
}

// -- Review panel -----------------------------------------------------------

function ReviewPanel({
  mediaKey,
  defaultMode,
  onClose,
}: {
  mediaKey: string;
  defaultMode: string;
  onClose: () => void;
}) {
  const media = useWatermarkMedia(mediaKey);
  const [mode, setMode] = useState(defaultMode);
  const [activeZone, setActiveZone] = useState<number | null>(null);
  const [placer, setPlacer] = useState(false);
  const [zoom, setZoom] = useState(false);
  const [zonePrompt, setZonePrompt] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const regenerate = useRegenerateZone();
  const edit = useEditZone();
  const addZone = useAddZone();
  const del = useDeleteZone();
  const dismiss = useDismissMedia();
  const revert = useRevertMedia();
  const patchMedia = usePatchMedia();
  const flatten = useFlattenMedia();

  useJobWatcher(jobId, () => {
    setJobId(null);
    media.refetch();
  });

  const data = media.data;
  const zones = useMemo(() => data?.zones ?? [], [data]);
  const flattened = !!data?.flattened;
  const anyDetected = zones.some((zone) => zone.status === "detected");
  const allPatched = zones.length > 0 && !anyDetected;
  const neutral = zones.length === 0;
  const effectiveMode = neutral || anyDetected ? "zone" : mode;

  useEffect(() => {
    if (zones.length && activeZone == null) {
      const target = zones.find((z) => z.status === "detected") ?? zones[0];
      setActiveZone(target.id);
    }
  }, [zones, activeZone]);

  const version = useMemo(
    () => zones.map((z) => z.updated_at).join("|"),
    [zones],
  );
  const zone = zones.find((z) => z.id === activeZone) ?? null;
  const busy =
    regenerate.isPending ||
    edit.isPending ||
    !!jobId ||
    addZone.isPending ||
    revert.isPending ||
    dismiss.isPending ||
    flatten.isPending ||
    patchMedia.isPending;

  const zonePromptSeed = zone?.prompt ?? "";
  useEffect(() => {
    setZonePrompt(zonePromptSeed);
  }, [activeZone, zonePromptSeed]);

  const handlePatchResponse = (res: {
    job_id?: string;
    media?: WatermarkMedia;
  }) => {
    if (res.job_id) setJobId(res.job_id);
    else media.refetch();
  };

  const moveZone = (zoneId: number, box: WatermarkBox) =>
    edit.mutate(
      { zoneId, box },
      { onSuccess: (res) => handlePatchResponse(res) },
    );

  const createZone = (box: WatermarkBox) =>
    addZone.mutate(
      { mediaId: Number(mediaKey), box },
      {
        onSuccess: (res) => {
          setActiveZone(res.zone_id);
          regenerate.mutate(
            { zoneId: res.zone_id },
            { onSuccess: (r) => handlePatchResponse(r) },
          );
        },
      },
    );

  const badge = flattened ? "flattened" : data?.status;
  return (
    <div
      style={{
        width: 400,
        flexShrink: 0,
        borderLeft: `1px solid ${colors.border}`,
        background: colors.panel,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 12px",
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 11,
            flex: 1,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {data?.name ?? "…"}
        </span>
        <span style={{ fontSize: 10, color: watermarkStatus(badge).color }}>
          ◪ {watermarkStatus(badge).label}
        </span>
        <span
          onClick={onClose}
          style={{ cursor: "pointer", color: colors.textMuted }}
        >
          ✕
        </span>
      </div>

      <div style={{ padding: 12, overflowY: "auto", flex: 1 }}>
        {!neutral && (
          <Segmented
            value={effectiveMode}
            onChange={setMode}
            options={[
              { value: "slider", label: "Slider" },
              { value: "hover", label: "A/B hover" },
              { value: "zone", label: "Zone ✎" },
            ]}
          />
        )}

        {zones.length > 1 && (
          <div
            style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 8 }}
          >
            {zones.map((z, index) => (
              <span
                key={z.id}
                onClick={() => setActiveZone(z.id)}
                style={{
                  ...zoneChip,
                  borderColor:
                    z.id === activeZone ? colors.watermark : colors.border,
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: watermarkStatus(z.status).color,
                    display: "inline-block",
                  }}
                />
                zone {index + 1}
                {z.score != null ? ` · ${Math.round(z.score)}%` : ""}
              </span>
            ))}
          </div>
        )}

        <div style={{ marginTop: 10 }}>
          <Preview
            mediaKey={mediaKey}
            mode={effectiveMode}
            version={version}
            zones={zones}
            activeZone={activeZone}
            busy={busy}
            onSelectZone={setActiveZone}
            onMoveZone={moveZone}
            onZoom={() => setZoom(true)}
          />
        </div>

        <Button block onClick={() => setZoom(true)} style={{ marginTop: 8 }}>
          🔍 Enlarge & zoom — inspect the patch
        </Button>
        {effectiveMode === "zone" && (
          <Button block onClick={() => setPlacer(true)} style={{ marginTop: 8 }}>
            ⛶ Place zones full-size
          </Button>
        )}

        {zone && (
          <ZoneInfo
            zone={zone}
            index={zones.indexOf(zone) + 1}
            total={zones.length}
          />
        )}

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 6,
            marginTop: 12,
          }}
        >
          {neutral && (
            <Button
              block
              style={VIOLET_BTN}
              disabled={busy}
              onClick={() => setPlacer(true)}
            >
              ＋ Add manual zone
            </Button>
          )}
          {anyDetected && (
            <Button
              block
              style={VIOLET_BTN}
              disabled={busy}
              onClick={() =>
                patchMedia.mutate(Number(mediaKey), {
                  onSuccess: (res) => setJobId(res.job_id),
                })
              }
            >
              ◪ Patch this media — every detected zone
            </Button>
          )}
          {allPatched && (
            <Button
              block
              style={VIOLET_BTN}
              disabled={busy || flattened}
              onClick={() =>
                zone &&
                regenerate.mutate(
                  { zoneId: zone.id },
                  { onSuccess: (res) => handlePatchResponse(res) },
                )
              }
            >
              ↻ Re-patch — new seed
            </Button>
          )}

          {zone && (
            <>
              <div style={{ fontSize: 10.5, color: colors.textMuted }}>
                Instruction — this zone
              </div>
              <input
                value={zonePrompt}
                onChange={(event) => setZonePrompt(event.target.value)}
                onBlur={() => {
                  if ((zone.prompt ?? "") !== zonePrompt) {
                    edit.mutate({ zoneId: zone.id, prompt: zonePrompt });
                  }
                }}
                placeholder="remove any watermark, logo or brand"
                style={inputStyle}
              />
              <Button
                block
                disabled={busy}
                onClick={() => setPlacer(true)}
              >
                ＋ Add manual zone
              </Button>
              {zones.length > 1 && (
                <Button
                  block
                  style={{ color: colors.danger }}
                  disabled={busy}
                  onClick={() =>
                    del.mutate(zone.id, {
                      onSuccess: () => setActiveZone(null),
                    })
                  }
                >
                  ✕ Delete this zone
                </Button>
              )}
            </>
          )}

          {allPatched && (
            <Button
              block
              disabled={busy}
              onClick={() =>
                revert.mutate(Number(mediaKey), {
                  onSuccess: () => media.refetch(),
                })
              }
            >
              ↩ Revert patch — back to Watermarked
            </Button>
          )}
          {!neutral && anyDetected && (
            <Button
              block
              style={{ color: colors.danger }}
              disabled={busy}
              onClick={() =>
                dismiss.mutate(Number(mediaKey), {
                  onSuccess: () => onClose(),
                })
              }
            >
              ⊘ Dismiss detection — back to clean
            </Button>
          )}
          {allPatched && !flattened && (
            <Button
              block
              style={{ background: colors.danger, color: colors.onAccent }}
              disabled={busy}
              onClick={() =>
                flatten.mutate(Number(mediaKey), {
                  onSuccess: (res) => setJobId(res.job_id),
                })
              }
            >
              ⇩ Flatten to disk — edit the original file
            </Button>
          )}
          {flattened && (
            <div style={{ ...runNote, color: colors.watermark }}>
              ⇩ Flattened — patches baked into the source file. Revert to undo.
            </div>
          )}
        </div>

        <div
          style={{
            marginTop: 12,
            fontSize: 10,
            color: colors.textFaint,
            lineHeight: 1.5,
          }}
        >
          The patch is a PNG stored in the project, composited over the original
          at display time and on every deploy. The source file is never
          modified unless you flatten it.
        </div>
      </div>
      {placer && (
        <ZonePlacerModal
          mediaKey={mediaKey}
          zones={zones}
          activeZone={activeZone}
          busy={busy}
          onSelectZone={setActiveZone}
          onMoveZone={moveZone}
          onCreate={createZone}
          onClose={() => setPlacer(false)}
        />
      )}
      {zoom && (
        <ImageZoomModal
          before={`/api/media/${mediaKey}/original`}
          after={`/api/media/${mediaKey}/file?v=${encodeURIComponent(version)}`}
          hasPatch={allPatched || flattened}
          onClose={() => setZoom(false)}
        />
      )}
    </div>
  );
}

// -- Enlarge & zoom viewer --------------------------------------------------

/**
 * Full-screen inspector for a patch result: wheel to zoom, drag to pan, and a
 * toggle to flip between the composited "after" and the untouched "before".
 */
function ImageZoomModal({
  before,
  after,
  hasPatch,
  onClose,
}: {
  before: string;
  after: string;
  hasPatch: boolean;
  onClose: () => void;
}) {
  const [showBefore, setShowBefore] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  // Distinguishes a plain click (dismiss) from a pan drag, so releasing a pan
  // over empty space never closes the viewer.
  const start = useRef({ x: 0, y: 0 });
  const moved = useRef(false);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    setScale((s) => Math.min(12, Math.max(1, s * factor)));
  };

  const reset = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  const src = showBefore ? before : after;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8,9,12,0.92)",
        zIndex: 90,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          fontFamily: font.mono,
          fontSize: 11,
          color: colors.textSecondary,
        }}
      >
        <span style={{ color: colors.watermark }}>
          🔍 {showBefore ? "BEFORE — original" : "AFTER — patched"}
        </span>
        <span style={{ color: colors.textMuted }}>
          wheel = zoom · drag = pan · {Math.round(scale * 100)}%
        </span>
        <span style={{ flex: 1 }} />
        {hasPatch && (
          <Button
            onClick={() => setShowBefore((v) => !v)}
            style={{ background: colors.raised }}
          >
            ⇄ {showBefore ? "Show after" : "Show before"}
          </Button>
        )}
        <Button onClick={reset} style={{ background: colors.raised }}>
          ⟲ Reset
        </Button>
        <Button onClick={onClose} style={{ background: colors.raised }}>
          ✕ Close
        </Button>
      </div>
      <div
        onClick={(event) => {
          // A click that wasn't a pan closes the viewer (click beside the
          // image, or a plain tap on it).
          event.stopPropagation();
          if (!moved.current) onClose();
        }}
        onWheel={onWheel}
        onPointerDown={(event) => {
          start.current = { x: event.clientX, y: event.clientY };
          moved.current = false;
          drag.current = {
            x: event.clientX - offset.x,
            y: event.clientY - offset.y,
          };
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          if (
            Math.abs(event.clientX - start.current.x) > 4 ||
            Math.abs(event.clientY - start.current.y) > 4
          ) {
            moved.current = true;
          }
          setOffset({
            x: event.clientX - drag.current.x,
            y: event.clientY - drag.current.y,
          });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        style={{
          flex: 1,
          overflow: "hidden",
          display: "grid",
          placeItems: "center",
          cursor: scale > 1 ? "grab" : "zoom-out",
          touchAction: "none",
        }}
      >
        <img
          src={src}
          alt=""
          draggable={false}
          style={{
            maxWidth: "94vw",
            maxHeight: "82vh",
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transformOrigin: "center",
            userSelect: "none",
            imageRendering: scale > 3 ? "pixelated" : "auto",
          }}
        />
      </div>
    </div>
  );
}

function ZoneInfo({
  zone,
  index,
  total,
}: {
  zone: WatermarkZone;
  index: number;
  total: number;
}) {
  const source =
    zone.detector === "owlv2"
      ? `OWLv2 · “${zone.query ?? "watermark"}”`
      : zone.detector === "manual"
        ? "manual"
        : zone.detector === "yolo"
          ? "YOLO11-n"
          : (zone.detector ?? "—");
  const rows: [string, string][] = [
    ["zone", `${index} / ${total}`],
    ["status", zone.status],
    [
      "source",
      `${source}${zone.score != null ? ` · ${Math.round(zone.score)}%` : ""}`,
    ],
    ["model", `${zone.model ?? "—"}${zone.seed ? ` · seed ${zone.seed}` : ""}`],
    ["prompt", zone.prompt ? `“${zone.prompt}”` : "global"],
    ["mask", `+${zone.dilate_px} px`],
    ["edit", zone.edit_ms != null ? `${zone.edit_ms} ms` : "—"],
  ];
  return (
    <div style={{ marginTop: 10 }}>
      {zone.score != null && (
        <div style={{ marginBottom: 6 }}>
          <div
            style={{ fontSize: 10, color: colors.textMuted, marginBottom: 2 }}
          >
            {zone.status === "patched" ? "Patch confidence" : "Detection score"}
          </div>
          <ProgressBar pct={zone.score} color={watermarkScore(zone.score)} />
        </div>
      )}
      <div
        style={{
          fontFamily: font.mono,
          fontSize: 10.5,
          color: colors.textMuted,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        {rows.map(([key, value]) => (
          <div
            key={key}
            style={{ display: "flex", justifyContent: "space-between" }}
          >
            <span>{key}</span>
            <span style={{ color: colors.textSecondary }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// -- Preview (slider / hover / zone editor) ---------------------------------

function Preview({
  mediaKey,
  mode,
  version,
  zones,
  activeZone,
  busy,
  onSelectZone,
  onMoveZone,
  onZoom,
}: {
  mediaKey: string;
  mode: string;
  version: string;
  zones: WatermarkZone[];
  activeZone: number | null;
  busy: boolean;
  onSelectZone: (id: number) => void;
  onMoveZone: (id: number, box: WatermarkBox) => void;
  onZoom: () => void;
}) {
  const original = `/api/media/${mediaKey}/original`;
  const composed = `/api/media/${mediaKey}/file?v=${encodeURIComponent(version)}`;
  const [split, setSplit] = useState(50);
  const [hovering, setHovering] = useState(false);

  // The frame wraps the image at its natural aspect ratio (no letterbox), so a
  // zone positioned in source fractions lands exactly over the pixels it marks.
  const frame = {
    position: "relative" as const,
    borderRadius: radii.card,
    overflow: "hidden",
    background: colors.raised,
    userSelect: "none" as const,
    lineHeight: 0,
  };
  // The sizing image: block, full width, natural height — it defines the frame.
  const imgBase = {
    display: "block" as const,
    width: "100%",
    height: "auto",
  };
  // A stacked overlay image (same source dimensions, so it fills exactly).
  const imgFill = {
    position: "absolute" as const,
    inset: 0,
    width: "100%",
    height: "100%",
  };

  if (mode === "zone") {
    return (
      <ZoneEditor
        src={original}
        zones={zones}
        activeZone={activeZone}
        busy={busy}
        onSelectZone={onSelectZone}
        onMoveZone={onMoveZone}
        frame={frame}
        imgStyle={imgBase}
      />
    );
  }

  if (mode === "hover") {
    return (
      <div
        style={{ ...frame, cursor: "zoom-in" }}
        onClick={onZoom}
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
      >
        <img src={hovering ? original : composed} alt="" style={imgBase} />
        <span style={abLabel(hovering ? colors.danger : colors.watermark)}>
          {hovering ? "BEFORE — original" : "AFTER — hover to compare"}
        </span>
        {busy && <BusyVeil />}
      </div>
    );
  }

  // The split follows the cursor as it moves over the image (hover to scrub);
  // a plain click still opens the zoom viewer.
  const scrub = (event: React.MouseEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const pct = ((event.clientX - rect.left) / rect.width) * 100;
    setSplit(Math.max(0, Math.min(100, pct)));
  };
  return (
    <div
      style={{ ...frame, cursor: "ew-resize" }}
      onClick={onZoom}
      onMouseMove={scrub}
      title="Move to compare before / after · click to zoom"
    >
      <img src={original} alt="" style={imgBase} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 0 0 ${split}%)`,
        }}
      >
        <img src={composed} alt="" style={imgFill} />
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${split}%`,
          width: 2,
          background: colors.watermark,
          pointerEvents: "none",
        }}
      />
      <span style={abLabel(colors.danger, "left")}>BEFORE</span>
      <span style={abLabel(colors.watermark, "right")}>AFTER</span>
      {busy && <BusyVeil />}
    </div>
  );
}

function abLabel(color: string, side: "left" | "right" = "left") {
  return {
    position: "absolute" as const,
    bottom: 6,
    [side]: 6,
    fontFamily: font.mono,
    fontSize: 9.5,
    fontWeight: 700,
    color: "#fff",
    background: color,
    borderRadius: 3,
    padding: "1px 5px",
  };
}

function BusyVeil() {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "grid",
        placeItems: "center",
        background: "rgba(10,11,14,0.5)",
        gap: 6,
      }}
    >
      <Spinner />
      <span style={{ fontSize: 10, color: colors.watermark, marginTop: 24 }}>
        FLUX edit running…
      </span>
    </div>
  );
}

interface Drag {
  id: number;
  kind: "move" | "resize";
  startX: number;
  startY: number;
  box: WatermarkBox;
}

function ZoneEditor({
  src,
  zones,
  activeZone,
  busy,
  onSelectZone,
  onMoveZone,
  frame,
  imgStyle,
}: {
  src: string;
  zones: WatermarkZone[];
  activeZone: number | null;
  busy: boolean;
  onSelectZone: (id: number) => void;
  onMoveZone: (id: number, box: WatermarkBox) => void;
  frame: React.CSSProperties;
  imgStyle: React.CSSProperties;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [preview, setPreview] = useState<WatermarkBox | null>(null);

  useEffect(() => {
    if (!drag) return undefined;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return undefined;
    const onMove = (event: PointerEvent) => {
      const dx = (event.clientX - drag.startX) / rect.width;
      const dy = (event.clientY - drag.startY) / rect.height;
      const box = { ...drag.box };
      if (drag.kind === "move") {
        box.x = Math.max(0, Math.min(1 - box.w, drag.box.x + dx));
        box.y = Math.max(0, Math.min(1 - box.h, drag.box.y + dy));
      } else {
        box.w = Math.max(0.02, Math.min(1 - drag.box.x, drag.box.w + dx));
        box.h = Math.max(0.02, Math.min(1 - drag.box.y, drag.box.h + dy));
      }
      setPreview(box);
    };
    const onUp = () => {
      if (preview) onMoveZone(drag.id, preview);
      setDrag(null);
      setPreview(null);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, preview, onMoveZone]);

  return (
    <div>
      <div ref={ref} style={frame}>
        <img src={src} alt="" style={imgStyle} draggable={false} />
        {zones.map((zone) => {
          const active = zone.id === activeZone;
          const box = active && preview ? preview : zone.box;
          return (
            <div
              key={zone.id}
              onPointerDown={(event) => {
                event.stopPropagation();
                onSelectZone(zone.id);
                setDrag({
                  id: zone.id,
                  kind: "move",
                  startX: event.clientX,
                  startY: event.clientY,
                  box: zone.box,
                });
              }}
              style={{
                position: "absolute",
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.w * 100}%`,
                height: `${box.h * 100}%`,
                border: `${active ? 2 : 1}px ${active ? "solid" : "dashed"} ${
                  active ? colors.watermark : colors.watermarkZoneDetected
                }`,
                background: active ? "rgba(167,139,218,0.1)" : "transparent",
                cursor: "move",
              }}
            >
              {active && (
                <div
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setDrag({
                      id: zone.id,
                      kind: "resize",
                      startX: event.clientX,
                      startY: event.clientY,
                      box: zone.box,
                    });
                  }}
                  style={{
                    position: "absolute",
                    right: -5,
                    bottom: -5,
                    width: 10,
                    height: 10,
                    background: colors.watermark,
                    borderRadius: 2,
                    cursor: "nwse-resize",
                  }}
                />
              )}
            </div>
          );
        })}
        {busy && <BusyVeil />}
      </div>
      <div style={{ fontSize: 10, color: colors.textFaint, marginTop: 4 }}>
        Click a zone, drag / resize — auto-regenerates on release.
      </div>
    </div>
  );
}

// -- Big placement modal ----------------------------------------------------

function ZonePlacerModal({
  mediaKey,
  zones,
  activeZone,
  busy,
  onSelectZone,
  onMoveZone,
  onCreate,
  onClose,
}: {
  mediaKey: string;
  zones: WatermarkZone[];
  activeZone: number | null;
  busy: boolean;
  onSelectZone: (id: number) => void;
  onMoveZone: (id: number, box: WatermarkBox) => void;
  onCreate: (box: WatermarkBox) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [preview, setPreview] = useState<WatermarkBox | null>(null);
  const [draw, setDraw] = useState<WatermarkBox | null>(null);
  const drawStart = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (!drag) return undefined;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return undefined;
    const onMove = (event: PointerEvent) => {
      const dx = (event.clientX - drag.startX) / rect.width;
      const dy = (event.clientY - drag.startY) / rect.height;
      const box = { ...drag.box };
      if (drag.kind === "move") {
        box.x = Math.max(0, Math.min(1 - box.w, drag.box.x + dx));
        box.y = Math.max(0, Math.min(1 - box.h, drag.box.y + dy));
      } else {
        box.w = Math.max(0.02, Math.min(1 - drag.box.x, drag.box.w + dx));
        box.h = Math.max(0.02, Math.min(1 - drag.box.y, drag.box.h + dy));
      }
      setPreview(box);
    };
    const onUp = () => {
      if (preview) onMoveZone(drag.id, preview);
      setDrag(null);
      setPreview(null);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, preview, onMoveZone]);

  useEffect(() => {
    if (!draw) return undefined;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return undefined;
    const onMove = (event: PointerEvent) => {
      const start = drawStart.current;
      if (!start) return;
      const cx = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const cy = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
      setDraw({
        x: Math.min(start.x, cx),
        y: Math.min(start.y, cy),
        w: Math.abs(cx - start.x),
        h: Math.abs(cy - start.y),
      });
    };
    const onUp = () => {
      const box = draw;
      drawStart.current = null;
      setDraw(null);
      if (box && box.w > 0.02 && box.h > 0.02) onCreate(box);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [draw, onCreate]);

  const startDraw = (event: React.PointerEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    drawStart.current = { x, y };
    setDraw({ x, y, w: 0, h: 0 });
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,11,14,0.85)",
        zIndex: 80,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        padding: 20,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          fontFamily: font.mono,
          fontSize: 11,
          color: colors.textSecondary,
          display: "flex",
          gap: 14,
          alignItems: "center",
        }}
      >
        <span style={{ color: colors.watermark }}>◪ Zone placement</span>
        <span style={{ color: colors.textMuted }}>
          drag on empty canvas = create · move / resize a zone ·
          auto-regenerates on release
        </span>
        <Button onClick={onClose} style={{ background: colors.raised }}>
          ✓ Done
        </Button>
      </div>
      <div
        ref={ref}
        onClick={(event) => event.stopPropagation()}
        onPointerDown={(event) => {
          if (event.target === event.currentTarget) startDraw(event);
        }}
        style={{
          position: "relative",
          lineHeight: 0,
          cursor: "crosshair",
          maxWidth: "92vw",
          maxHeight: "82vh",
        }}
      >
        <img
          src={`/api/media/${mediaKey}/original`}
          alt=""
          draggable={false}
          style={{
            display: "block",
            maxWidth: "92vw",
            maxHeight: "82vh",
            userSelect: "none",
            pointerEvents: "none",
          }}
        />
        {zones.map((zone) => {
          const active = zone.id === activeZone;
          const box = active && preview ? preview : zone.box;
          return (
            <div
              key={zone.id}
              onPointerDown={(event) => {
                event.stopPropagation();
                onSelectZone(zone.id);
                setDrag({
                  id: zone.id,
                  kind: "move",
                  startX: event.clientX,
                  startY: event.clientY,
                  box: zone.box,
                });
              }}
              style={{
                position: "absolute",
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.w * 100}%`,
                height: `${box.h * 100}%`,
                border: `${active ? 2 : 1}px ${active ? "solid" : "dashed"} ${
                  active ? colors.watermark : colors.watermarkZoneDetected
                }`,
                background: active ? "rgba(167,139,218,0.12)" : "transparent",
                cursor: "move",
              }}
            >
              {active && (
                <div
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setDrag({
                      id: zone.id,
                      kind: "resize",
                      startX: event.clientX,
                      startY: event.clientY,
                      box: zone.box,
                    });
                  }}
                  style={{
                    position: "absolute",
                    right: -6,
                    bottom: -6,
                    width: 13,
                    height: 13,
                    background: colors.watermark,
                    borderRadius: 3,
                    cursor: "nwse-resize",
                  }}
                />
              )}
            </div>
          );
        })}
        {draw && (draw.w > 0 || draw.h > 0) && (
          <div
            style={{
              position: "absolute",
              left: `${draw.x * 100}%`,
              top: `${draw.y * 100}%`,
              width: `${draw.w * 100}%`,
              height: `${draw.h * 100}%`,
              border: `2px solid ${colors.watermark}`,
              background: "rgba(167,139,218,0.15)",
              pointerEvents: "none",
            }}
          />
        )}
        {busy && <BusyVeil />}
      </div>
    </div>
  );
}

// -- Styles -----------------------------------------------------------------

const runNote: React.CSSProperties = {
  fontFamily: font.mono,
  fontSize: 10,
  lineHeight: 1.5,
  color: colors.textMuted,
};

const railStyle: React.CSSProperties = {
  width: 272,
  flexShrink: 0,
  background: colors.toolbar,
  borderRight: `1px solid ${colors.border}`,
  overflowY: "auto",
};

const selectStyle: React.CSSProperties = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
};

const selectStyle2: React.CSSProperties = {
  padding: "5px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
};

const favLabel: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 5,
  fontSize: 11.5,
  color: colors.textSecondary,
  cursor: "pointer",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "5px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 11.5,
};

const monoPanel: React.CSSProperties = {
  fontFamily: font.mono,
  fontSize: 10,
  lineHeight: 1.5,
  padding: "6px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.watermarkBorderSoft}`,
  background: colors.watermarkBg,
  display: "flex",
  flexDirection: "column",
  gap: 3,
  wordBreak: "break-all",
};

const tagChip: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  fontSize: 10,
  color: colors.watermark,
  background: colors.watermarkBg,
  border: `1px solid ${colors.watermarkBorderSoft}`,
  borderRadius: 10,
  padding: "1px 6px",
};

const zoneChip: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  fontSize: 10.5,
  color: colors.textSecondary,
  border: `1px solid ${colors.border}`,
  borderRadius: 8,
  padding: "2px 7px",
  cursor: "pointer",
};

const pager: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 10,
  padding: "8px 14px",
  borderTop: `1px solid ${colors.border}`,
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textMuted,
};
