/**
 * Model-profile editor — the modal owning everything a profile bundles:
 * weights file (via the server file browser), format, family (auto/manual),
 * mmproj (auto/manual, GGUF vision only) and the generation defaults
 * (temperature, image resolution, max tokens, n_ctx, thinking, default
 * prompt preset). Delete is a two-step arm; Duplicate reopens the copy as a
 * new profile.
 */

import { useEffect, useState } from "react";
import {
  useCreateProfile,
  useDeleteProfile,
  useDetectProfileFile,
  usePrompts,
  useUpdateProfile,
} from "../../api/hooks";
import type { ModelProfile, ProfileFamily } from "../../api/types";
import {
  colors,
  font,
  profileTypeColor,
  radii,
  shadow,
} from "../../design/tokens";
import { Button, Label, Segmented, Slider } from "../atoms";
import { ModelFileBrowserModal } from "./ModelFileBrowserModal";

const N_CTX_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

type Draft = Omit<ModelProfile, "id">;

const inputStyle = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;

function draftFrom(profile: ModelProfile | null): Draft {
  if (profile) {
    const rest = { ...profile } as Partial<ModelProfile>;
    delete rest.id;
    return rest as Draft;
  }
  return {
    name: "",
    file: "",
    dir: "",
    format: "gguf",
    type: "",
    type_mode: "auto",
    temp: 0.7,
    n_ctx: 8192,
    mmproj_mode: "auto",
    mmproj: null,
    think: "auto",
    max_tok: 512,
    img_res: 1024,
    prompt: "",
  };
}

export function ProfileEditorModal({
  profile,
  role,
  families,
  profileCount,
  onClose,
}: {
  /** null = create a new profile. */
  profile: ModelProfile | null;
  role: "caption" | "judge";
  families: ProfileFamily[];
  profileCount: number;
  onClose: () => void;
}) {
  const [isNew, setIsNew] = useState(profile === null);
  const [draft, setDraft] = useState<Draft>(() => draftFrom(profile));
  const [autoName, setAutoName] = useState(profile === null);
  const [armed, setArmed] = useState(false);
  const [browse, setBrowse] = useState<"model" | "mmproj" | null>(null);

  const create = useCreateProfile();
  const update = useUpdateProfile();
  const remove = useDeleteProfile();
  const detect = useDetectProfileFile();
  const prompts = usePrompts(draft.type || null);

  const set = (partial: Partial<Draft>) =>
    setDraft((prev) => ({ ...prev, ...partial }));

  const family = families.find((f) => f.key === draft.type);
  const isVisionGguf =
    draft.format === "gguf" && draft.type !== "" && draft.type !== "text";

  // An out-of-list default prompt falls back to the type's first preset.
  useEffect(() => {
    const titles = prompts.data?.prompts.map((p) => p.title) ?? [];
    if (titles.length === 0) return;
    setDraft((prev) =>
      titles.includes(prev.prompt)
        ? prev
        : { ...prev, prompt: titles[0] },
    );
  }, [prompts.data]);

  const applyDetection = (dir: string, file: string, manualType: boolean) => {
    detect.mutate(
      { dir, file },
      {
        onSuccess: (found) => {
          setDraft((prev) => ({
            ...prev,
            type: manualType ? prev.type : found.type,
            mmproj:
              prev.mmproj_mode === "auto" ? found.mmproj : prev.mmproj,
          }));
        },
      },
    );
  };

  const pickWeights = (dir: string, file: string) => {
    const format = file.toLowerCase().endsWith(".gguf")
      ? ("gguf" as const)
      : ("safetensors" as const);
    const stem = file.replace(/\.[^.]+$/, "");
    setDraft((prev) => ({
      ...prev,
      dir,
      file,
      format,
      name: autoName || !prev.name ? stem : prev.name,
    }));
    applyDetection(dir, file, draft.type_mode === "manual");
  };

  const switchFormat = (format: "gguf" | "safetensors") => {
    setDraft((prev) => {
      const keeps = prev.file.toLowerCase().endsWith(`.${format}`);
      return {
        ...prev,
        format,
        file: keeps ? prev.file : "",
        mmproj: keeps ? prev.mmproj : null,
        type: keeps ? prev.type : prev.type_mode === "manual" ? prev.type : "",
      };
    });
  };

  const save = () => {
    if (isNew) {
      create.mutate({ ...draft, role }, { onSuccess: onClose });
    } else if (profile) {
      update.mutate({ id: profile.id, ...draft }, { onSuccess: onClose });
    }
  };

  const canSave = !!draft.file && !!draft.name.trim();
  const detected = detect.data;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,11,13,0.82)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 500,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 560,
          maxWidth: "94vw",
          maxHeight: "88vh",
          display: "flex",
          flexDirection: "column",
          background: colors.panel,
          border: `1px solid ${colors.borderHover}`,
          borderRadius: radii.modal,
          boxShadow: shadow.modal,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "12px 16px",
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <span style={{ fontSize: 14, fontWeight: 700, flex: 1 }}>
            {isNew ? "New model profile" : "Edit model profile"}
          </span>
          <span
            onClick={onClose}
            style={{ cursor: "pointer", color: colors.textMuted }}
          >
            ✕
          </span>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
          <Label>Profile name</Label>
          <input
            value={draft.name}
            onChange={(event) => {
              setAutoName(false);
              set({ name: event.target.value });
            }}
            style={{ ...inputStyle, fontSize: 13, fontWeight: 600 }}
          />

          <div style={{ marginTop: 14 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 6,
              }}
            >
              <Label>Model weights</Label>
              <Segmented
                value={draft.format}
                onChange={(value) =>
                  switchFormat(value as "gguf" | "safetensors")
                }
                options={[
                  { value: "gguf", label: "gguf" },
                  { value: "safetensors", label: "safetensors" },
                ]}
              />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                readOnly
                value={draft.file ? `${draft.dir}\\${draft.file}` : ""}
                placeholder="No file picked"
                style={{
                  ...inputStyle,
                  fontFamily: font.mono,
                  fontSize: 11,
                  color: draft.file ? colors.text : colors.textFaint,
                }}
              />
              <Button onClick={() => setBrowse("model")}>Browse…</Button>
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginTop: 8,
              }}
            >
              <span style={{ fontSize: 11, color: colors.textMuted }}>
                Detected:{" "}
                {draft.file ? (
                  <TypeBadge
                    type={
                      draft.type_mode === "auto"
                        ? draft.type
                        : (detected?.type ?? "")
                    }
                    families={families}
                  />
                ) : (
                  <span style={{ color: colors.textFaint }}>—</span>
                )}
              </span>
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 11, color: colors.textMuted }}>
                Type
              </span>
              <select
                value={draft.type_mode === "auto" ? "auto" : draft.type}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === "auto") {
                    set({ type_mode: "auto" });
                    if (draft.file) {
                      applyDetection(draft.dir, draft.file, false);
                    } else {
                      set({ type_mode: "auto", type: "" });
                    }
                  } else {
                    set({ type_mode: "manual", type: value });
                  }
                }}
                style={{ ...inputStyle, width: 220 }}
              >
                <option value="auto">Auto — detect from filename</option>
                {families
                  .filter((f) => f.manual)
                  .map((f) => (
                    <option key={f.key} value={f.key}>
                      {f.label}
                    </option>
                  ))}
              </select>
            </div>
          </div>

          {isVisionGguf && (
            <div
              style={{
                marginTop: 14,
                border: `1px solid ${colors.border}`,
                background: colors.toolbar,
                borderRadius: radii.card,
                padding: "10px 12px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 8,
                }}
              >
                <Label>MMPROJ — vision projector</Label>
                <Segmented
                  value={draft.mmproj_mode}
                  onChange={(value) => {
                    if (value === "manual") {
                      set({ mmproj_mode: "manual" });
                      setBrowse("mmproj");
                    } else {
                      set({ mmproj_mode: "auto" });
                      if (draft.file) {
                        applyDetection(draft.dir, draft.file, true);
                      }
                    }
                  }}
                  options={[
                    { value: "auto", label: "Auto-detect" },
                    { value: "manual", label: "Manual" },
                  ]}
                />
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  fontFamily: font.mono,
                  fontSize: 10.5,
                }}
              >
                <span
                  style={{
                    flex: 1,
                    color: draft.mmproj ? colors.ok : colors.danger,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {draft.mmproj
                    ? `✓ ${draft.mmproj}`
                    : "✕ none found next to the model"}
                </span>
                <span
                  onClick={() => setBrowse("mmproj")}
                  style={{ color: colors.accent, cursor: "pointer" }}
                >
                  change
                </span>
              </div>
            </div>
          )}

          <div
            style={{
              marginTop: 16,
              paddingTop: 14,
              borderTop: `1px solid ${colors.border}`,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <Label>Generation defaults</Label>
            <FieldRow label={`Temperature · ${draft.temp.toFixed(2)}`}>
              <Slider
                min={0}
                max={2}
                step={0.05}
                value={draft.temp}
                onChange={(temp) => set({ temp })}
              />
            </FieldRow>
            <FieldRow label={`Image res. · ${draft.img_res}px`}>
              <Slider
                min={512}
                max={2048}
                step={128}
                value={draft.img_res}
                onChange={(img_res) => set({ img_res })}
              />
            </FieldRow>
            <FieldRow label="Max new tokens">
              <input
                value={String(draft.max_tok)}
                onChange={(event) =>
                  set({ max_tok: Number(event.target.value) || 0 })
                }
                style={{ ...inputStyle, width: 100, fontFamily: font.mono }}
              />
            </FieldRow>
            {draft.format === "gguf" && (
              <FieldRow label="Context (n_ctx)">
                <select
                  value={draft.n_ctx}
                  onChange={(event) =>
                    set({ n_ctx: Number(event.target.value) })
                  }
                  style={{ ...inputStyle, width: 130 }}
                >
                  {N_CTX_OPTIONS.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </FieldRow>
            )}
            {family?.think && (
              <FieldRow label="Thinking">
                <Segmented
                  value={draft.think}
                  onChange={(think) => set({ think })}
                  options={[
                    { value: "off", label: "Off" },
                    { value: "auto", label: "Auto" },
                    { value: "show", label: "On" },
                  ]}
                />
              </FieldRow>
            )}
            <FieldRow label="Default prompt">
              <select
                value={draft.prompt}
                onChange={(event) => set({ prompt: event.target.value })}
                style={{ ...inputStyle, maxWidth: 260 }}
              >
                {(prompts.data?.prompts ?? []).map((preset) => (
                  <option key={preset.title} value={preset.title}>
                    {preset.title}
                    {preset.builtin ? "" : " · user"}
                  </option>
                ))}
              </select>
            </FieldRow>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "12px 16px",
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          {!isNew && profileCount > 1 && profile && (
            <Button
              variant="danger"
              onClick={() => {
                if (!armed) {
                  setArmed(true);
                  return;
                }
                remove.mutate(profile.id, { onSuccess: onClose });
              }}
            >
              {armed ? "Really delete?" : "Delete"}
            </Button>
          )}
          {!isNew && (
            <Button
              onClick={() => {
                setIsNew(true);
                setArmed(false);
                set({ name: `${draft.name} (copy)` });
              }}
            >
              Duplicate
            </Button>
          )}
          <div style={{ flex: 1 }} />
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="accent"
            disabled={!canSave || create.isPending || update.isPending}
            onClick={save}
          >
            Save profile
          </Button>
        </div>
      </div>

      {browse && (
        <ModelFileBrowserModal
          target={browse}
          format={draft.format}
          initialPath={draft.dir}
          onClose={() => setBrowse(null)}
          onPick={(dir, file) => {
            if (browse === "model") pickWeights(dir, file);
            else set({ mmproj: file, mmproj_mode: "manual" });
          }}
        />
      )}
    </div>
  );
}

export function TypeBadge({
  type,
  families,
}: {
  type: string;
  families: ProfileFamily[];
}) {
  if (!type) {
    return (
      <span
        style={{
          fontFamily: font.mono,
          fontSize: 8.5,
          fontWeight: 700,
          color: colors.warn,
        }}
      >
        not recognized
      </span>
    );
  }
  const label = families.find((f) => f.key === type)?.label ?? type;
  return (
    <span
      title={label}
      style={{
        fontFamily: font.mono,
        fontSize: 8.5,
        fontWeight: 700,
        padding: "1px 5px",
        borderRadius: 4,
        background: colors.card,
        color: profileTypeColor(type),
      }}
    >
      {type}
    </span>
  );
}

function FieldRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span
        style={{
          width: 170,
          flex: "none",
          fontSize: 11.5,
          color: colors.textSecondary,
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}
