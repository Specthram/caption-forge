/**
 * Caption tab left panel: model load, prompt preset (save/delete), the
 * generation params (shared through the caption store so the batch bar
 * reuses them), Generate-all (streamed job, skips locked media) and deploy.
 */

import { useEffect, useMemo, useState } from "react";
import {
  useDeletePrompt,
  useDeploy,
  useGenerate,
  useGroundingEnabled,
  useLoadModel,
  useModelStatus,
  useModels,
  usePrompts,
  useSavePrompt,
  useUndeploy,
  useUnloadModel,
} from "../../api/hooks";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useCaptionStore } from "../../store/captionStore";
import { useJobList } from "../../store/jobsStore";
import { api } from "../../api/client";
import { Button, Label, Segmented, Slider, Spinner } from "../atoms";

const selectStyle = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;

export function CaptionLeftPanel() {
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const gen = useCaptionStore();

  const models = useModels();
  const status = useModelStatus();
  const loadModel = useLoadModel();
  const unloadModel = useUnloadModel();
  const generate = useGenerate();
  const groundingEnabled = useGroundingEnabled();
  const deploy = useDeploy();
  const undeploy = useUndeploy();
  const savePrompt = useSavePrompt();
  const deletePrompt = useDeletePrompt();
  const jobs = useJobList();

  const modelType = useMemo(() => {
    if (status.data?.loaded) return status.data.type;
    return models.data?.models.find((m) => m.name === gen.model)?.type ?? null;
  }, [status.data, models.data, gen.model]);

  const prompts = usePrompts(modelType);
  const [selectedTitle, setSelectedTitle] = useState("");
  const [modelOpen, setModelOpen] = useState(false);

  useEffect(() => {
    if (prompts.data) {
      const preset =
        prompts.data.prompts.find((p) => p.title === prompts.data.selected) ??
        prompts.data.prompts[0];
      gen.set({
        temperature: prompts.data.temperature,
        think: prompts.data.think_mode,
        prompt: preset?.prompt ?? "",
      });
      setSelectedTitle(preset?.title ?? "");
    }
    // Only re-sync when the fetched presets change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompts.data]);

  const selectedPreset = prompts.data?.prompts.find(
    (p) => p.title === selectedTitle,
  );
  const genJob = jobs.find(
    (job) => job.type === "generate" && job.state === "running",
  );
  // A load/unload is a queued job; the button reflects it as busy until the
  // model-status poll confirms the new loaded state.
  const modelJob = jobs.find(
    (job) =>
      (job.type === "load-model" || job.type === "unload-model") &&
      (job.state === "queued" || job.state === "running"),
  );
  const loaded = status.data?.loaded ?? false;
  const modelBusy =
    !!modelJob || loadModel.isPending || unloadModel.isPending;

  const startGenerate = () => {
    if (datasetId == null) return;
    generate.mutate({
      dataset_id: datasetId,
      caption_type: captionType,
      media_ids: null,
      exclude_ids: Array.from(gen.locked).map(Number),
      prompt: gen.prompt,
      temperature: gen.temperature,
      seed: gen.seed ? Number(gen.seed) : null,
      think_mode: gen.think,
      image_size: gen.imgRes,
      review_after: gen.reviewAfter,
      ground_after: gen.groundAfter,
    });
  };

  const saveCopy = () => {
    if (!modelType) return;
    const title = window.prompt("Save prompt as:", `${selectedTitle} copy`);
    if (title) {
      savePrompt.mutate({
        model_type: modelType,
        title,
        prompt: gen.prompt,
      });
      setSelectedTitle(title);
    }
  };

  const collapsedModelLabel = modelBusy
    ? modelJob?.sub || (loaded ? "Unloading…" : "Loading…")
    : loaded
      ? (status.data?.name ?? "Model loaded")
      : "No model loaded";

  const modelControls = (
    <>
      <select
        style={selectStyle}
        value={gen.model}
        onChange={(event) => gen.set({ model: event.target.value })}
      >
        <option value="">Select a model…</option>
        {models.data?.models.map((model) => (
          <option key={model.name} value={model.name}>
            {model.name}
          </option>
        ))}
      </select>
      <div style={{ marginTop: 10 }}>
        <Label>Image res. · {gen.imgRes}px</Label>
        <Slider
          min={512}
          max={2048}
          step={128}
          value={gen.imgRes}
          onChange={(imgRes) => gen.set({ imgRes })}
        />
      </div>
      <Button
        variant={loaded ? "ghost" : "accent"}
        block
        style={{ marginTop: 10 }}
        disabled={modelBusy || (!loaded && !gen.model)}
        onClick={() =>
          loaded ? unloadModel.mutate() : loadModel.mutate(gen.model)
        }
      >
        {modelBusy
          ? modelJob?.sub || (loaded ? "Unloading…" : "Loading…")
          : loaded
            ? "Unload model"
            : "Load model"}
      </Button>
      <div style={{ marginTop: 10, fontSize: 11, color: colors.textMuted }}>
        {loaded ? "● " : "○ "}
        {status.data?.status}
      </div>
    </>
  );

  return (
    <div
      style={{
        width: 252,
        flex: "none",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        height: "100%",
        borderRight: `1px solid ${colors.border}`,
        background: colors.panel,
      }}
    >
      <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
        <Section title="Prompt">
        <select
          style={selectStyle}
          value={selectedTitle}
          onChange={(event) => {
            const preset = prompts.data?.prompts.find(
              (p) => p.title === event.target.value,
            );
            setSelectedTitle(event.target.value);
            if (preset) gen.set({ prompt: preset.prompt });
          }}
        >
          {prompts.data?.prompts.map((preset) => (
            <option key={preset.title} value={preset.title}>
              {preset.title}
              {preset.builtin ? "" : " · user"}
            </option>
          ))}
        </select>
        <textarea
          value={gen.prompt}
          onChange={(event) => gen.set({ prompt: event.target.value })}
          rows={5}
          style={{
            ...selectStyle,
            marginTop: 8,
            resize: "vertical",
            fontFamily: font.sans,
            lineHeight: 1.4,
          }}
        />
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <Button onClick={saveCopy} disabled={!modelType}>
            Save a copy
          </Button>
          {selectedPreset && !selectedPreset.builtin && modelType && (
            <Button
              variant="danger"
              onClick={() =>
                deletePrompt.mutate({
                  model_type: modelType,
                  title: selectedTitle,
                })
              }
            >
              🗑 Delete
            </Button>
          )}
        </div>
        <div style={{ marginTop: 10 }}>
          <Label>Temperature · {gen.temperature.toFixed(2)}</Label>
          <Slider
            min={0}
            max={2}
            step={0.05}
            value={gen.temperature}
            onChange={(temperature) => gen.set({ temperature })}
          />
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          <input
            value={gen.seed}
            onChange={(event) => gen.set({ seed: event.target.value })}
            placeholder="seed"
            style={{ ...selectStyle, width: 90 }}
          />
          <Button
            onClick={() =>
              gen.set({ seed: String(Math.floor(Math.random() * 1e9)) })
            }
          >
            ⟳
          </Button>
        </div>
        <div style={{ marginTop: 10 }}>
          <Label>Thinking</Label>
          <Segmented
            value={gen.think}
            onChange={(think) => gen.set({ think })}
            options={[
              { value: "off", label: "Off" },
              { value: "auto", label: "Auto" },
              { value: "show", label: "On" },
            ]}
          />
        </div>
      </Section>

      <Section title="Generate">
        {genJob ? (
          <div>
            <div
              style={{ fontSize: 11, color: colors.textMuted, marginBottom: 6 }}
            >
              {genJob.sub}
            </div>
            <Button
              variant="danger"
              block
              onClick={() => api.post(`/jobs/${genJob.id}/stop`).catch(() => {})}
            >
              ■ Stop generation
            </Button>
          </div>
        ) : (
          <Button
            variant="accent"
            block
            disabled={datasetId == null || generate.isPending}
            onClick={startGenerate}
          >
            ✦ Generate all captions
          </Button>
        )}
        <label style={checkboxRow}>
          <input
            type="checkbox"
            checked={gen.reviewAfter}
            onChange={(event) => gen.set({ reviewAfter: event.target.checked })}
          />
          Review after generation
        </label>
        {groundingEnabled && (
          <label style={checkboxRow}>
            <input
              type="checkbox"
              checked={gen.groundAfter}
              onChange={(event) => gen.set({ groundAfter: event.target.checked })}
            />
            Ground after generation
          </label>
        )}
      </Section>

      <Section title="Deploy">
        <div style={{ display: "flex", gap: 8 }}>
          <Button
            block
            disabled={datasetId == null || deploy.isPending}
            onClick={() =>
              datasetId != null &&
              deploy.mutate({ dataset_id: datasetId, caption_type: captionType })
            }
          >
            {deploy.isPending ? <Spinner size={12} /> : "⇪ Deploy"}
          </Button>
          <Button
            variant="ghost"
            block
            disabled={datasetId == null}
            onClick={() => datasetId != null && undeploy.mutate(datasetId)}
          >
            Undeploy
          </Button>
        </div>
        {datasetId != null && (
          <a
            href={`/api/deploy/zip?dataset_id=${datasetId}`}
            style={{ display: "inline-block", marginTop: 10, fontSize: 12 }}
          >
            ↓ Download as .zip
          </a>
        )}
      </Section>
      </div>

      <div
        style={{
          flex: "none",
          borderTop: `1px solid ${colors.border}`,
          background: colors.panel,
        }}
      >
        {modelOpen && (
          <div
            style={{
              padding: 14,
              borderBottom: `1px solid ${colors.border}`,
              maxHeight: 340,
              overflowY: "auto",
            }}
          >
            {modelControls}
          </div>
        )}
        <div
          onClick={() => setModelOpen((open) => !open)}
          title={loaded ? status.data?.name ?? "" : "No model loaded"}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 14px",
            cursor: "pointer",
          }}
        >
          <span style={{ color: loaded ? colors.ok : colors.textFaint }}>
            {loaded ? "●" : "○"}
          </span>
          <span
            style={{
              flex: 1,
              fontSize: 12,
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              color: loaded ? colors.text : colors.textMuted,
            }}
          >
            {collapsedModelLabel}
          </span>
          {modelBusy && <Spinner size={11} />}
          <span style={{ color: colors.textMuted, fontSize: 11 }}>
            {modelOpen ? "▾" : "▴"}
          </span>
        </div>
      </div>
    </div>
  );
}

const checkboxRow = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: 11.5,
  color: colors.textSecondary,
  marginTop: 8,
  cursor: "pointer",
} as const;

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 22 }}>
      <Label>{title}</Label>
      {children}
    </div>
  );
}
