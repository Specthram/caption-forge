/** Settings workspace: directories, grounding, index scans, compute, video. */

import { useEffect, useState } from "react";
import {
  useIndexStatus,
  useModels,
  useProfiles,
  useSaveSettings,
  useSelectProfile,
  useSettings,
  useTaggerModels,
} from "../api/hooks";
import type { GroundingSize, IndexStep, ModelProfile } from "../api/types";
import { colors, font } from "../design/tokens";
import { Button, Label, Segmented, Toast, Toggle } from "../components/atoms";
import { FolderBrowserModal } from "../components/organisms/FolderBrowserModal";
import { ProfileEditorModal } from "../components/organisms/ProfileEditorModal";

const NCTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

type Draft = Record<string, unknown>;
type MachineSteps = Record<string, boolean>;

export function SettingsView() {
  const settings = useSettings();
  const models = useModels();
  const profiles = useProfiles();
  const selectProfile = useSelectProfile();
  const [editJudge, setEditJudge] = useState<ModelProfile | null>(null);
  const status = useIndexStatus();
  const taggers = useTaggerModels();
  const save = useSaveSettings();
  const [draft, setDraft] = useState<Draft>({});
  const [browse, setBrowse] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: "ok" | "danger"; message: string } | null>(null);

  useEffect(() => {
    if (settings.data) setDraft(settings.data);
  }, [settings.data]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(timer);
  }, [toast]);

  const handleSave = () => {
    save.mutate(draft, {
      onSuccess: () => setToast({ kind: "ok", message: "Settings saved" }),
      onError: (error) =>
        setToast({
          kind: "danger",
          message: error instanceof Error ? error.message : "Failed to save settings",
        }),
    });
  };

  const set = (key: string, value: unknown) =>
    setDraft((prev) => ({ ...prev, [key]: value }));

  const str = (key: string) => (draft[key] as string) ?? "";
  const num = (key: string, fallback: number) =>
    (draft[key] as number) ?? fallback;
  const extensions = (draft.caption_extensions as string[]) ?? [];
  const [newExt, setNewExt] = useState("");
  const device = str("device") || "cuda";

  const machineSteps = (draft.index_steps as MachineSteps) ?? {};
  const toggleStep = (key: string, value: boolean) =>
    set("index_steps", { ...machineSteps, [key]: value });

  // Which IQA models the "Quality scores" index step runs. The catalogue
  // ships with the settings payload; the selection is a machine setting.
  const qualityMetrics = (draft.index_quality_metrics as string[]) ?? [];
  const toggleQualityMetric = (id: string, on: boolean) =>
    set(
      "index_quality_metrics",
      on
        ? [...qualityMetrics, id]
        : qualityMetrics.filter((metric) => metric !== id),
    );
  const metricsCatalogue =
    (settings.data?.quality_metrics_catalogue as {
      id: string;
      label: string;
      vram: string;
    }[]) ?? [];

  // The SigLIP catalogue ships with the settings payload, so the selects
  // never hard-code a checkpoint list the backend could disagree with.
  const sizes =
    (settings.data?.grounding_sizes as Record<string, GroundingSize>) ?? {};
  const size = str("grounding_model_size") || "so400m";
  const resolution = num("grounding_resolution", 512);
  const groundingEnabled = (draft.grounding_enabled ?? true) as boolean;
  const autosaveEnabled = (draft.autosave_enabled ?? true) as boolean;

  // The CLIP / BLIP catalogue of the reference-free caption score (SigLIP2
  // reuses the grounding checkpoint above, so it has no selector of its own).
  const scoreCatalogue =
    (settings.data?.caption_score_catalogue as Record<
      string,
      {
        label: string;
        sizes: Record<
          string,
          { label: string; params: string; vram: string }
        >;
      }
    >) ?? {};

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: 24, overflowY: "auto", height: "100%" }}>
      <Card title="Directories">
        {(
          [
            ["Model directory", "model_dir"],
            ["Internal media directory", "internal_media_dir"],
            ["Deploy directory", "deploy_dir"],
            ["Watermark models directory (.pt)", "watermark_models_dir"],
          ] as [string, string][]
        ).map(([label, key]) => (
          <Field key={key} label={label}>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                style={input}
                value={str(key)}
                onChange={(e) => set(key, e.target.value)}
              />
              <Button onClick={() => setBrowse(key)}>Browse…</Button>
            </div>
          </Field>
        ))}
      </Card>

      <Card title="Model downloads">
        <Field label="Hugging Face token">
          <input
            style={input}
            type="password"
            autoComplete="off"
            placeholder="hf_… (optional)"
            value={str("hf_token")}
            onChange={(e) => set("hf_token", e.target.value)}
          />
          <div
            style={{
              fontSize: 10.5,
              color: colors.textMuted,
              marginTop: 6,
              lineHeight: 1.5,
            }}
          >
            Used for every model download (OWLv2, FLUX.2, SigLIP, WD14…). Needed
            for gated repos or to lift rate limits. Stored locally and exported
            to the process on save — no restart required. Leave empty for
            anonymous downloads.
          </div>
        </Field>
      </Card>

      <Card title="Beta features">
        <Field label="Grounding (SigLIP2)">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Toggle
              checked={groundingEnabled}
              onChange={(value) => set("grounding_enabled", value)}
            />
            <span style={{ fontSize: 11.5, color: colors.textMuted }}>
              {groundingEnabled ? "Shown in the app" : "Hidden across the app"}
            </span>
          </div>
          <div style={{ fontSize: 10.5, color: colors.textMuted, marginTop: 6, lineHeight: 1.5 }}>
            Off hides every grounding surface — the Caption &amp; Media
            grounding cards, the “weak grounding” filter and the “ground
            after” checkboxes. The reference-free Caption/Tags scores are
            separate and stay.
          </div>
        </Field>
      </Card>

      {groundingEnabled && (
      <Card title="Caption grounding — SigLIP2">
        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <Field label="Model size">
              <select
                style={input}
                value={size}
                onChange={(e) => {
                  // Giant ships no 512 checkpoint; keep the pair valid.
                  const next = e.target.value;
                  const offered = sizes[next]?.resolutions ?? [];
                  set("grounding_model_size", next);
                  if (!offered.includes(resolution)) {
                    set("grounding_resolution", Math.max(...offered));
                  }
                }}
              >
                {Object.entries(sizes).map(([key, spec]) => (
                  <option key={key} value={key}>
                    {spec.label} ({spec.params}) — {spec.vram}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div style={{ flex: 1 }}>
            <Field label="Input resolution">
              <select
                style={input}
                value={resolution}
                onChange={(e) => set("grounding_resolution", Number(e.target.value))}
              >
                {(sizes[size]?.resolutions ?? []).map((value) => (
                  <option key={value} value={value}>
                    {value} — heatmap {value / 16}×{value / 16}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        </div>
        <div
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textFaint,
            marginTop: -4,
            marginBottom: 12,
          }}
        >
          google/siglip2-{size}-patch16-{resolution}
          {settings.data?.grounding_model_id !== `google/siglip2-${size}-patch16-${resolution}` &&
            " · unsaved"}
        </div>

        <Field label="Model for splitting captions into claims">
          <select
            style={input}
            value={str("grounding_claim_model")}
            onChange={(e) => set("grounding_claim_model", e.target.value)}
          >
            <option value="">(use the loaded model)</option>
            {models.data?.models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </Field>

        <ThresholdField
          label="Caption claim validation threshold"
          value={num("grounding_threshold_caption", 55)}
          onChange={(value) => set("grounding_threshold_caption", value)}
        />
        <ThresholdField
          label="Tag validation threshold"
          value={num("grounding_threshold_tags", 55)}
          onChange={(value) => set("grounding_threshold_tags", value)}
        />
        <div style={{ fontSize: 10.5, color: colors.textMuted, lineHeight: 1.5 }}>
          The claim-splitting model is the VLM that decomposes a caption
          before SigLIP scores it. Leave it on the loaded model, or pick one
          to auto-load when grounding with nothing loaded. SigLIP scores each
          claim or tag independently (a sigmoid, never a distribution);
          higher resolution is a finer heatmap and more VRAM, and the
          thresholds apply on read, so moving them never invalidates a run.
        </div>
      </Card>
      )}

      <Card title="Caption score — zero-reference">
        <div style={{ display: "flex", gap: 12 }}>
          {[
            { kind: "clip", key: "caption_score_clip_size" },
            { kind: "blip", key: "caption_score_blip_size" },
          ].map(({ kind, key }) => {
            const family = scoreCatalogue[kind];
            if (!family) return null;
            return (
              <div key={kind} style={{ flex: 1 }}>
                <Field label={family.label}>
                  <select
                    style={input}
                    value={str(key) || "large"}
                    onChange={(e) => set(key, e.target.value)}
                  >
                    {Object.entries(family.sizes).map(([value, spec]) => (
                      <option key={value} value={value}>
                        {spec.label} ({spec.params}) — {spec.vram}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 10.5, color: colors.textMuted, lineHeight: 1.5 }}>
          The Caption tab’s score card rates a whole caption against the image
          with three encoders — SigLIP2 (the grounding checkpoint above), CLIP
          and BLIP — each a cosine calibrated against generic prompts, no LLM.
          CLIP and BLIP weights download on first score.
        </div>
      </Card>

      <Card title="Captions">
        <Field label="Caption types">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
            {extensions.map((ext) => (
              <Chip key={ext} onRemove={() => set("caption_extensions", extensions.filter((x) => x !== ext))}>
                .{ext}
              </Chip>
            ))}
            <Chip>tags (virtual)</Chip>
            <input
              value={newExt}
              onChange={(e) => setNewExt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && newExt.trim()) {
                  set("caption_extensions", [...extensions, newExt.trim().replace(/^\./, "")]);
                  setNewExt("");
                }
              }}
              placeholder="+ add"
              style={{ ...input, width: 80 }}
            />
          </div>
        </Field>
        <Field label="Auto-save caption edits">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Toggle
              checked={autosaveEnabled}
              onChange={(value) => set("autosave_enabled", value)}
            />
            <span style={{ fontSize: 11.5, color: colors.textMuted }}>
              {autosaveEnabled
                ? "Saves after a short typing pause"
                : "Manual Save button only"}
            </span>
          </div>
          <div style={{ fontSize: 10.5, color: colors.textMuted, marginTop: 6, lineHeight: 1.5 }}>
            On persists a caption a moment after you stop typing; the Save
            button then reads “Saved”. Off restores the classic manual save.
          </div>
        </Field>
      </Card>

      <Card title="Compute (GGUF)">
        <Field label="Device">
          <Segmented
            value={device}
            onChange={(value) => set("device", value)}
            options={[
              { value: "cuda", label: "cuda" },
              { value: "cpu", label: "cpu" },
            ]}
          />
          <div style={{ fontSize: 11, marginTop: 6, color: device === "cpu" ? colors.warn : colors.textMuted }}>
            {device === "cpu"
              ? "GGUF on CPU is very slow — choose cpu only if you have no GPU."
              : "Enforced device — never a silent CPU fallback."}
          </div>
        </Field>
        <Field label="Context (n_ctx)">
          <select style={input} value={num("gguf_n_ctx", 8192)} onChange={(e) => set("gguf_n_ctx", Number(e.target.value))}>
            {NCTX.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          <div style={{ fontSize: 11, marginTop: 6, color: colors.textMuted }}>
            4096 is plenty for images; video frames need more. Ignored by
            safetensors. Default for new model profiles — each profile
            overrides its own.
          </div>
        </Field>
      </Card>

      <Card title="Review judge">
        <Field label="Default judge profile">
          <select
            style={input}
            value={profiles.data?.judge_id ?? ""}
            onChange={(e) =>
              selectProfile.mutate({
                role: "judge",
                id: Number(e.target.value),
              })
            }
          >
            {(profiles.data?.profiles ?? []).map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
                {profile.file ? ` — ${profile.file}` : ""}
              </option>
            ))}
          </select>
          <div style={{ fontSize: 11, marginTop: 6 }}>
            <span
              onClick={() => {
                const judge = profiles.data?.profiles.find(
                  (p) => p.id === profiles.data.judge_id,
                );
                if (judge) setEditJudge(judge);
              }}
              style={{ color: colors.accent, cursor: "pointer" }}
            >
              edit profile
            </span>
            <span style={{ color: colors.textMuted }}>
              {" "}
              — the model reviewing captions (Review tab and “review after
              generation”). Applied immediately, not part of Save.
            </span>
          </div>
        </Field>
      </Card>
      {editJudge && (
        <ProfileEditorModal
          profile={editJudge}
          role="judge"
          families={profiles.data?.families ?? []}
          profileCount={profiles.data?.profiles.length ?? 1}
          onClose={() => setEditJudge(null)}
        />
      )}

      <Card
        title="This machine — index scans"
        note="saved per machine · applies app-wide"
      >
        <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 4 }}>
          “Index” chains the scans below. Turn one off if this machine can’t (or
          shouldn’t) run it — every button that needs it greys out, and
          libraries show that step as off instead of missing.
        </div>
        {(status.data?.steps ?? []).map((step: IndexStep) => {
          const on = machineSteps[step.key] ?? true;
          return (
            <div key={step.key} style={stepRow}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: on ? colors.text : colors.textMuted,
                  }}
                >
                  {step.label}
                </div>
                <div style={{ fontSize: 10.5, color: colors.textMuted }}>
                  {step.description}
                </div>
              </div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: 10,
                  color: colors.textFaint,
                }}
              >
                {step.cost}
              </div>
              <Toggle
                checked={on}
                onChange={(value) => toggleStep(step.key, value)}
              />
            </div>
          );
        })}
        <div
          style={{
            borderTop: `1px solid ${colors.border}`,
            marginTop: 12,
            paddingTop: 12,
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>
            Quality models scored
          </div>
          <div
            style={{ fontSize: 10.5, color: colors.textMuted, marginBottom: 8 }}
          >
            Which IQA models the “Quality scores” step runs during an index.
            Heavy VLM scorers (Q-Align) are opt-in. A media counts as scored
            once every checked model has run on it.
          </div>
          {metricsCatalogue.map((metric) => {
            const on = qualityMetrics.includes(metric.id);
            return (
              <div key={metric.id} style={stepRow}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: on ? colors.text : colors.textMuted,
                    }}
                  >
                    {metric.label}
                  </div>
                  <div style={{ fontSize: 10.5, color: colors.textMuted }}>
                    {metric.vram}
                  </div>
                </div>
                <Toggle
                  checked={on}
                  onChange={(value) => toggleQualityMetric(metric.id, value)}
                />
              </div>
            );
          })}
          {qualityMetrics.length === 0 && (
            <div style={{ fontSize: 10.5, color: colors.warn, marginTop: 4 }}>
              None selected — the default trio (MUSIQ · TOPIQ-NR · LAION-Aes)
              is used.
            </div>
          )}
        </div>
      </Card>

      <Card title="Auto-tagger (WD14)">
        <Field label="Tagger model">
          <select
            style={input}
            value={str("autotag_source")}
            onChange={(e) => set("autotag_source", e.target.value)}
          >
            {taggers.data?.models.map((model) => (
              <option key={model.source} value={model.source}>
                {model.label}
                {model.available ? " ✓" : ""}
              </option>
            ))}
          </select>
        </Field>
        <div style={{ fontSize: 10.5, color: colors.textMuted, marginBottom: 4 }}>
          New tags reuse an existing tag of the same name (in any category);
          a genuinely new name lands in an auto-managed “Uncategorized”
          category you can recategorise from — it disappears once empty.
        </div>
        <div style={{ display: "flex", gap: 16 }}>
          <Field label="General tag confidence ≥">
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              style={input}
              value={num("autotag_general", 0.35)}
              onChange={(e) => set("autotag_general", Number(e.target.value))}
            />
          </Field>
          <Field label="Character tag confidence ≥">
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              style={input}
              value={num("autotag_character", 0.85)}
              onChange={(e) => set("autotag_character", Number(e.target.value))}
            />
          </Field>
        </div>
      </Card>

      <Card title="Video captioning">
        <div style={{ display: "flex", gap: 16 }}>
          <Field label="fps">
            <input type="number" style={input} value={num("video_fps", 2)} onChange={(e) => set("video_fps", Number(e.target.value))} />
          </Field>
          <Field label="seconds">
            <input type="number" style={input} value={num("video_max_seconds", 5)} onChange={(e) => set("video_max_seconds", Number(e.target.value))} />
          </Field>
          <Field label="frame resolution">
            <input type="number" style={input} value={num("video_resolution", 256)} onChange={(e) => set("video_resolution", Number(e.target.value))} />
          </Field>
        </div>
        <Field label="Video prompt">
          <textarea
            rows={3}
            style={{ ...input, resize: "vertical", fontFamily: font.sans }}
            value={str("video_prompt")}
            onChange={(e) => set("video_prompt", e.target.value)}
          />
        </Field>
      </Card>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 8 }}>
        <Button variant="ghost" onClick={() => settings.data && setDraft(settings.data)}>
          Revert
        </Button>
        <Button variant="accent" disabled={save.isPending} onClick={handleSave}>
          Save settings
        </Button>
      </div>
      {toast && <Toast kind={toast.kind}>{toast.message}</Toast>}
      {browse && (
        <FolderBrowserModal
          initialPath={str(browse)}
          hint="Pick a folder on the machine running Caption Forge."
          confirmLabel="Use this folder"
          onClose={() => setBrowse(null)}
          onSelect={(path) => {
            set(browse, path);
            setBrowse(null);
          }}
        />
      )}
    </div>
  );
}

function Card({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ background: colors.card, border: `1px solid ${colors.border}`, borderRadius: 9, padding: 18, marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <div style={{ fontWeight: 600, fontSize: 14 }}>{title}</div>
        {note && (
          <div
            style={{
              marginLeft: "auto",
              fontFamily: font.mono,
              fontSize: 10,
              color: colors.textFaint,
            }}
          >
            {note}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}

/** A 0-100 grounding threshold: a slider with its live mono read-out. */
function ThresholdField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={label}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ flex: 1, accentColor: colors.accent }}
        />
        <span
          style={{
            width: 38,
            textAlign: "right",
            fontFamily: font.mono,
            fontSize: 12,
            fontWeight: 600,
            color: colors.warn,
          }}
        >
          {Math.round(value)}%
        </span>
      </div>
    </Field>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function Chip({ children, onRemove }: { children: React.ReactNode; onRemove?: () => void }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 8px", borderRadius: 10, background: colors.raised, border: `1px solid ${colors.borderControl}`, fontSize: 11.5, fontFamily: font.mono }}>
      {children}
      {onRemove && (
        <span onClick={onRemove} style={{ cursor: "pointer", color: colors.textMuted }}>
          ✕
        </span>
      )}
    </span>
  );
}

const stepRow = {
  display: "flex",
  alignItems: "center",
  gap: 14,
  paddingTop: 10,
  marginTop: 10,
  borderTop: `1px solid ${colors.border}`,
} as const;

const input = {
  width: "100%",
  padding: "7px 9px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12.5,
  fontFamily: font.mono,
} as const;
