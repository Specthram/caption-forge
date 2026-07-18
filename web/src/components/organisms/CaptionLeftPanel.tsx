/**
 * Caption tab left panel: the captioner model profile (selector +
 * Load/Unload + status), the prompt preset (filtered by the profile's
 * family) with the seed dice, Generate-all (streamed job, skips locked
 * media) and deploy. Generation params (temperature, thinking, image res,
 * max tokens, n_ctx) all live on the profile — edited in its modal.
 */

import { useEffect, useMemo, useState } from "react";
import {
  useDeletePrompt,
  useDeploy,
  useGenerate,
  useGroundingEnabled,
  useLoadProfile,
  useModelStatus,
  useProfiles,
  usePrompts,
  useRememberPrompt,
  useSavePrompt,
  useUndeploy,
} from "../../api/hooks";
import type { JobSnapshot } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useCaptionStore } from "../../store/captionStore";
import { useJobList, useJobsStore } from "../../store/jobsStore";
import { api } from "../../api/client";
import { Button, Label, Spinner } from "../atoms";
import { ProfileSelector } from "./ProfileSelector";

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
  const setCaptionTab = useUiStore((state) => state.setCaptionTab);
  const gen = useCaptionStore();

  const profiles = useProfiles();
  const status = useModelStatus();
  const loadProfile = useLoadProfile();
  const generate = useGenerate();
  const groundingEnabled = useGroundingEnabled();
  const deploy = useDeploy();
  const undeploy = useUndeploy();
  const savePrompt = useSavePrompt();
  const deletePrompt = useDeletePrompt();
  const rememberPrompt = useRememberPrompt();
  const jobs = useJobList();

  const data = profiles.data;
  const active = useMemo(
    () => data?.profiles.find((p) => p.id === data.active_id) ?? null,
    [data],
  );
  const loadedProfile = useMemo(
    () => data?.profiles.find((p) => p.id === data.loaded_id) ?? null,
    [data],
  );
  const judgeProfile = useMemo(
    () => data?.profiles.find((p) => p.id === data.judge_id) ?? null,
    [data],
  );
  const modelType = active?.type || null;

  const prompts = usePrompts(modelType);
  const [selectedTitle, setSelectedTitle] = useState("");
  const [reviewJobId, setReviewJobId] = useState<string | null>(null);
  const reviewJob = useJobsStore((state) =>
    reviewJobId ? state.jobs[reviewJobId] : undefined,
  );

  // A generate-with-review job just finished → jump the user to the Review
  // sub-tab where the pending findings are waiting.
  useEffect(() => {
    if (!reviewJobId || !reviewJob) return;
    if (reviewJob.state === "done") {
      setCaptionTab("review");
      setReviewJobId(null);
    } else if (reviewJob.state === "error" || reviewJob.state === "stopped") {
      setReviewJobId(null);
    }
  }, [reviewJob, reviewJobId, setCaptionTab]);

  // Selecting a profile applies the preset it last used (auto-remembered) as
  // the panel's current prompt, falling back to the first preset.
  useEffect(() => {
    if (prompts.data) {
      const list = prompts.data.prompts;
      const preset =
        list.find((p) => p.title === active?.prompt) ?? list[0];
      gen.set({ prompt: preset?.prompt ?? "" });
      setSelectedTitle(preset?.title ?? "");
    }
    // Only re-sync when the presets or the active profile change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompts.data, active?.id, active?.prompt]);

  const selectedPreset = prompts.data?.prompts.find(
    (p) => p.title === selectedTitle,
  );
  const genJob = jobs.find(
    (job) => job.type === "generate" && job.state === "running",
  );
  // A load/unload is a queued job; the button reflects it as busy until the
  // profiles poll confirms the new loaded state.
  const modelJob = jobs.find(
    (job) =>
      (job.type === "load-model" || job.type === "unload-model") &&
      (job.state === "queued" || job.state === "running"),
  );
  const loaded = status.data?.loaded ?? false;
  const activeLoaded = active != null && active.id === data?.loaded_id;
  const modelBusy = !!modelJob || loadProfile.isPending;
  // A profile is loadable once it has weights: a picked file (local) or a
  // valid owner/repo id (HF).
  const activeLoadable =
    active != null &&
    (active.source === "hf" ? active.repo.includes("/") : !!active.file);
  // While an HF download runs, the load job streams byte progress with a
  // "downloading …" subtitle — surfaced as the inline progress card.
  const downloadJob =
    modelJob && modelJob.sub.startsWith("downloading") ? modelJob : null;

  const startGenerate = () => {
    if (datasetId == null || !data) return;
    generate.mutate(
      {
        dataset_id: datasetId,
        caption_type: captionType,
        media_ids: null,
        exclude_ids: Array.from(gen.locked).map(Number),
        prompt: gen.prompt,
        profile_id: data.active_id,
        seed: gen.seed ? Number(gen.seed) : null,
        review_after: gen.reviewAfter,
        review_judge_profile_id: gen.reviewAfter ? data.judge_id : null,
        ground_after: gen.groundAfter,
        recaption: gen.recaption,
        unload_after: gen.unloadAfter,
      },
      {
        onSuccess: (result) => {
          if (gen.reviewAfter) setReviewJobId(result.job_id);
        },
      },
    );
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
        <Section title="Model profile">
          <ProfileSelector role="caption" />
          <Button
            variant="accent"
            block
            style={{ marginTop: 8 }}
            disabled={modelBusy || !activeLoadable || activeLoaded}
            onClick={() => active && loadProfile.mutate(active.id)}
          >
            {modelBusy ? modelJob?.sub || "Working…" : "Load model"}
          </Button>
          {downloadJob && active && (
            <DownloadCard
              repo={active.repo}
              job={downloadJob}
              onCancel={() =>
                api.post(`/jobs/${downloadJob.id}/stop`).catch(() => {})
              }
            />
          )}
          <div
            style={{
              marginTop: 8,
              fontFamily: font.mono,
              fontSize: 10.5,
              color: activeLoaded
                ? colors.ok
                : loaded
                  ? colors.warn
                  : colors.textFaint,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            title={status.data?.status ?? ""}
          >
            {activeLoaded
              ? "● loaded"
              : loaded
                ? `● ${loadedProfile?.name ?? status.data?.name ?? "?"} in ` +
                  "VRAM — swaps on load or run"
                : "○ unloaded — loads on demand"}
          </div>
        </Section>

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
            // Remember it as this profile's last-used preset.
            if (active) {
              rememberPrompt.mutate({ id: active.id, title: event.target.value });
            }
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
        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          <input
            value={gen.seed}
            onChange={(event) => gen.set({ seed: event.target.value })}
            placeholder="seed"
            style={{ ...selectStyle, width: 90 }}
          />
          <Button
            title="Random each run (-1)"
            onClick={() => gen.set({ seed: "-1" })}
            style={{ fontSize: 14, color: colors.textMuted }}
          >
            ⚄
          </Button>
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
            checked={gen.recaption}
            onChange={(event) => gen.set({ recaption: event.target.checked })}
          />
          Re-caption existing captions
        </label>
        {!gen.recaption && (
          <div
            style={{
              margin: "0 0 6px 22px",
              fontSize: 10,
              color: colors.textFaint,
              lineHeight: 1.4,
            }}
          >
            Only media whose caption is still empty will be captioned.
          </div>
        )}
        <label style={checkboxRow}>
          <input
            type="checkbox"
            checked={gen.unloadAfter}
            onChange={(event) =>
              gen.set({ unloadAfter: event.target.checked })
            }
          />
          Unload the model after the job
        </label>
        <label style={checkboxRow}>
          <input
            type="checkbox"
            checked={gen.reviewAfter}
            onChange={(event) => gen.set({ reviewAfter: event.target.checked })}
          />
          Review after generation
        </label>
        {gen.reviewAfter && (
          <div
            style={{
              margin: "2px 0 6px 22px",
              fontSize: 10,
              color: colors.textFaint,
              lineHeight: 1.4,
            }}
          >
            After generating, the captioner is unloaded, the judge —{" "}
            <span style={{ color: colors.textSecondary }}>
              {judgeProfile?.name ?? "?"}
            </span>{" "}
            (Review tab) — reviews the new captions, and the app opens the
            Review tab. Fixes wait for you — nothing is applied automatically.
          </div>
        )}
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
    </div>
  );
}

/** Inline Hugging Face download progress + cancel, under the Load button. */
function DownloadCard({
  repo,
  job,
  onCancel,
}: {
  repo: string;
  job: JobSnapshot;
  onCancel: () => void;
}) {
  const gb = (bytes: number) => (bytes / 1e9).toFixed(1);
  const size =
    job.total > 0 ? `${gb(job.done)} / ${gb(job.total)} GB` : `${gb(job.done)} GB`;
  return (
    <div
      style={{
        marginTop: 8,
        display: "flex",
        flexDirection: "column",
        gap: 4,
        border: `1px solid ${colors.borderControl}`,
        background: colors.input,
        borderRadius: 6,
        padding: "7px 8px",
        fontFamily: font.mono,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            flex: 1,
            minWidth: 0,
            fontSize: 10,
            color: colors.warn,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          downloading {repo}
        </span>
        <span style={{ fontSize: 10, color: colors.warn }}>{job.pct}%</span>
      </div>
      <div
        style={{
          height: 4,
          borderRadius: 2,
          background: "#24262d",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${job.pct}%`,
            background: colors.warn,
            transition: "width 0.3s",
          }}
        />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ flex: 1, fontSize: 9.5, color: colors.textFaint }}>
          {size} · huggingface.co
        </span>
        <span
          onClick={onCancel}
          style={{ fontSize: 9.5, color: colors.danger, cursor: "pointer" }}
          onMouseEnter={(event) => {
            event.currentTarget.style.color = "#f08a7a";
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.color = colors.danger;
          }}
        >
          cancel
        </span>
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
