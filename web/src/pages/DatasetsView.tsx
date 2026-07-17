/**
 * Datasets workspace: list + New/Auto-build on the left; on the right the
 * dataset header and two tabs — the media grid, and the quality report.
 */

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useAddTriggerword,
  useAutobuildRecipe,
  useCaptionScoreReport,
  useCreateDataset,
  useDatasetMedia,
  useDatasetReport,
  useDatasets,
  useDeleteDataset,
  useRemoveDatasetMedia,
  useRemoveTriggerword,
  useTriggerwords,
  useUpdateDataset,
} from "../api/hooks";
import type { AutobuildRecipe } from "../api/hooks";
import type { MediaGridCard } from "../api/types";
import { colors, font, qualityColor } from "../design/tokens";
import { useUiStore } from "../store/uiStore";
import { useJobList } from "../store/jobsStore";
import { Button, Label } from "../components/atoms";
import { TagChip } from "../components/molecules";
import { GridCard } from "../components/molecules/GridCard";
import { AutoBuildStudio } from "../components/organisms/AutoBuildStudio";
import { DatasetComposerModal } from "../components/organisms/DatasetComposerModal";
import { MediaDetailPanel } from "../components/organisms/MediaDetailPanel";
import { QualityReportTab } from "../components/organisms/QualityReportTab";
import { CaptionScoreReportTab } from "../components/organisms/CaptionScoreReportTab";
import { LivingDatasetUpgrades } from "../components/organisms/UpgradesOverlay";

const PAGE = 60;

type DatasetTab = "media" | "quality" | "caption";

function TabStrip({
  tab,
  onChange,
  score,
  captionScore,
}: {
  tab: DatasetTab;
  onChange: (tab: DatasetTab) => void;
  score: number | null;
  captionScore: number | null;
}) {
  const item = (value: DatasetTab, label: string, badge: number | null) => {
    const active = tab === value;
    return (
      <button
        onClick={() => onChange(value)}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 7,
          padding: "9px 12px",
          background: "transparent",
          border: "none",
          borderBottom: `2px solid ${active ? colors.accent : "transparent"}`,
          color: active ? colors.accent : colors.textMuted,
          fontSize: 12.5,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        {label}
        {badge != null && (
          <span
            style={{
              fontFamily: font.mono,
              fontSize: 10,
              padding: "1px 6px",
              borderRadius: 10,
              color: colors.onAccent,
              background: qualityColor(badge),
            }}
          >
            {Math.round(badge)}
          </span>
        )}
      </button>
    );
  };
  return (
    <div
      style={{
        display: "flex",
        gap: 4,
        padding: "0 16px",
        borderBottom: `1px solid ${colors.border}`,
      }}
    >
      {item("media", "Media", null)}
      {item("quality", "Quality report", score)}
      {item("caption", "Caption score", captionScore)}
    </div>
  );
}

export function DatasetsView() {
  const activeId = useUiStore((state) => state.datasetId);
  const setActive = useUiStore((state) => state.setDataset);
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const captionType = useUiStore((state) => state.captionType);

  const datasets = useDatasets();
  const createDataset = useCreateDataset();
  const deleteDataset = useDeleteDataset();
  const updateDataset = useUpdateDataset();
  const removeMedia = useRemoveDatasetMedia();

  const [newName, setNewName] = useState("");
  // Inline rename of the active dataset (the ✎ next to the title).
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [autoOpen, setAutoOpen] = useState(false);
  // Once opened, the Studio stays mounted (hidden when closed) so an
  // incidental close — backdrop, ✕, Esc — keeps its in-progress recipe.
  // Only Cancel (or creating the dataset) resets it.
  const [autoMounted, setAutoMounted] = useState(false);
  // Re-editing a saved dataset in the Studio: the target it prefills from,
  // and a nonce the Studio watches to run its one-shot prefill each open.
  const [editTarget, setEditTarget] = useState<{
    id: number;
    recipe: AutobuildRecipe;
    name: string;
  } | null>(null);
  const [studioNonce, setStudioNonce] = useState(0);
  // Tab, page and focus are restored after a refresh, so they live in the
  // store. Switching dataset resets them (see uiStore.setDataset).
  const { tab, page, focusKey } = useUiStore((state) => state.datasetsView);
  const setDatasetsView = useUiStore((state) => state.setDatasetsView);
  const setTab = (next: DatasetTab) => setDatasetsView({ tab: next });
  const setPage = (next: number) => setDatasetsView({ page: next });
  const setFocusKey = (next: string | null) =>
    setDatasetsView({ focusKey: next });
  const openCropOverlay = useUiStore((state) => state.openCrop);
  const reportBadge = useDatasetReport(activeId);
  const captionScoreBadge = useCaptionScoreReport(activeId, captionType);

  const rows = datasets.data?.datasets;
  // Select the first dataset on a cold start, and fall back to it when the
  // one restored from the last session has since been deleted.
  useEffect(() => {
    if (!rows?.length) return;
    if (activeId == null || !rows.some((row) => row.id === activeId)) {
      setActive(rows[0].id);
    }
  }, [activeId, rows, setActive]);

  // A living dataset's upgrades depend on the library; when an index-type
  // job finishes it may have grown a stronger candidate, so re-fetch the
  // upgrade banner once such a job leaves the running set.
  const client = useQueryClient();
  const jobs = useJobList();
  const runningIndexJobs = useRef<Set<string>>(new Set());
  useEffect(() => {
    const indexTypes = new Set([
      "index",
      "scan",
      "scan-all",
      "reindex-all",
      "embeddings",
      "quality",
      "lookalike",
    ]);
    const runningNow = new Set(
      jobs
        .filter(
          (job) =>
            (job.state === "running" || job.state === "queued") &&
            indexTypes.has(job.type),
        )
        .map((job) => job.id),
    );
    let finished = false;
    runningIndexJobs.current.forEach((id) => {
      if (!runningNow.has(id)) finished = true;
    });
    runningIndexJobs.current = runningNow;
    if (finished) {
      client.invalidateQueries({ queryKey: ["autobuild-upgrades"] });
    }
  }, [jobs, client]);

  const active = rows?.find((d) => d.id === activeId);
  // The Studio recipe of the active dataset, if it was built there — the
  // "Edit in builder" button only shows when one exists to reopen.
  const activeRecipe = useAutobuildRecipe(active?.id ?? null);
  const editableRecipe = activeRecipe.data?.recipe ?? null;
  const media = useDatasetMedia(
    activeId,
    (page - 1) * PAGE,
    PAGE,
    qualityMetric,
  );

  // Open the Studio for a fresh build (null) or to re-edit a saved dataset.
  // Bumping the nonce drives the Studio's one-shot prefill on this open.
  const openStudio = (
    target: { id: number; recipe: AutobuildRecipe; name: string } | null,
  ) => {
    setEditTarget(target);
    setStudioNonce((value) => value + 1);
    setAutoMounted(true);
    setAutoOpen(true);
  };
  const total = media.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE));

  // The dataset shrank since the last session: a restored page past its end
  // would paint an empty grid with no way back but the pager.
  useEffect(() => {
    if (media.data && page > pageCount) setPage(1);
  }, [media.data, page, pageCount]); // eslint-disable-line react-hooks/exhaustive-deps

  // A crop is always framed on the image it aliases, never on itself: an
  // existing crop reopens the overlay on its parent, with its own rectangle.
  const openCrop = (card: MediaGridCard) => {
    if (card.crop) {
      openCropOverlay(card.crop.parent_media_id, {
        id: Number(card.key),
        rect: card.crop.rect,
        ratio: card.crop.ratio,
      });
    } else {
      openCropOverlay(Number(card.key));
    }
  };

  const create = () => {
    if (!newName.trim()) return;
    createDataset.mutate(newName.trim(), {
      onSuccess: (data) => setActive(data.id),
    });
    setNewName("");
  };

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <div
        style={{
          width: 240,
          flex: "none",
          borderRight: `1px solid ${colors.border}`,
          background: colors.panel,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div style={{ flex: 1, overflowY: "auto", padding: 10 }}>
          {datasets.data?.datasets.map((dataset) => (
            <div
              key={dataset.id}
              onClick={() => {
                setActive(dataset.id);
                setPage(1);
                setTab("media");
              }}
              style={{
                padding: "8px 10px",
                borderRadius: 6,
                cursor: "pointer",
                marginBottom: 4,
                background:
                  dataset.id === activeId ? colors.accentTint : "transparent",
                color:
                  dataset.id === activeId ? colors.accent : colors.textSecondary,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 600 }}>
                {dataset.name}
              </div>
              <div
                style={{
                  fontSize: 10.5,
                  fontFamily: font.mono,
                  color: colors.textFaint,
                }}
              >
                {dataset.count} media
              </div>
            </div>
          ))}
        </div>
        <div style={{ padding: 10, borderTop: `1px solid ${colors.border}` }}>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={newName}
              onChange={(event) => setNewName(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && create()}
              placeholder="New dataset…"
              style={inputStyle}
            />
            <Button onClick={create}>+</Button>
          </div>
          <Button
            block
            style={{
              marginTop: 8,
              background: colors.accentTintAlt,
              color: colors.accent,
            }}
            onClick={() => openStudio(null)}
          >
            ⚙ Auto-build a dataset…
          </Button>
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        {active ? (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "12px 16px",
                borderBottom: `1px solid ${colors.border}`,
                background: colors.toolbar,
              }}
            >
              <div style={{ flex: 1 }}>
                {renaming ? (
                  <input
                    autoFocus
                    value={nameDraft}
                    onChange={(event) => setNameDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        const name = nameDraft.trim();
                        if (name && name !== active.name) {
                          updateDataset.mutate({ id: active.id, name });
                        }
                        setRenaming(false);
                      } else if (event.key === "Escape") {
                        setRenaming(false);
                      }
                    }}
                    onBlur={() => setRenaming(false)}
                    style={{
                      fontSize: 15,
                      fontWeight: 600,
                      padding: "2px 6px",
                      borderRadius: 6,
                      border: `1px solid ${colors.borderControl}`,
                      background: colors.input,
                      color: colors.text,
                      width: 260,
                    }}
                  />
                ) : (
                  <div
                    style={{
                      fontSize: 15,
                      fontWeight: 600,
                      display: "flex",
                      alignItems: "center",
                      gap: 7,
                    }}
                  >
                    {active.name}
                    <span
                      title="Rename dataset"
                      onClick={() => {
                        setNameDraft(active.name);
                        setRenaming(true);
                      }}
                      style={{
                        fontSize: 12,
                        color: colors.textMuted,
                        cursor: "pointer",
                      }}
                    >
                      ✎
                    </span>
                  </div>
                )}
                <div
                  style={{
                    fontSize: 11,
                    color: colors.textMuted,
                    fontFamily: font.mono,
                  }}
                >
                  {active.count} media · linked, never copied
                </div>
              </div>
              {editableRecipe && (
                <Button
                  onClick={() =>
                    openStudio({
                      id: active.id,
                      recipe: editableRecipe,
                      name: active.name,
                    })
                  }
                >
                  ✎ Edit in builder
                </Button>
              )}
              <Button variant="accent" onClick={() => setAddOpen(true)}>
                + Add media from library
              </Button>
              <Button
                variant="danger"
                onClick={() => {
                  if (window.confirm(`Delete dataset "${active.name}"?`)) {
                    deleteDataset.mutate(active.id);
                    setActive(null);
                  }
                }}
              >
                Delete
              </Button>
            </div>

            <LivingDatasetUpgrades datasetId={active.id} />

            <TabStrip
              tab={tab}
              onChange={setTab}
              score={reportBadge.data?.report?.overall ?? null}
              captionScore={captionScoreBadge.data?.overall ?? null}
            />

            {tab === "quality" && <QualityReportTab datasetId={active.id} />}

            {tab === "caption" && (
              <CaptionScoreReportTab datasetId={active.id} />
            )}

            {tab === "media" && (
              <DatasetMeta
                datasetId={active.id}
                deployName={active.deploy_name}
                deployResolution={active.deploy_resolution}
              />
            )}

            {tab === "media" && (
              <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
                <div
                  style={{
                    flex: 1,
                    minWidth: 0,
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
                    <div
                      style={{
                        fontSize: 11,
                        color: colors.textFaint,
                        marginBottom: 8,
                      }}
                    >
                      Click a card to inspect it · hover for the ✕ to unlink
                      (the file is kept).
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fill, minmax(122px, 1fr))",
                        gap: 8,
                      }}
                    >
                      {media.data?.items.map((card) => (
                        <GridCard
                          key={card.key}
                          card={card}
                          focused={focusKey === card.key}
                          onClick={() => setFocusKey(card.key)}
                          onRemove={() => {
                            removeMedia.mutate({
                              id: active.id,
                              media_ids: [Number(card.key)],
                            });
                            if (focusKey === card.key) setFocusKey(null);
                          }}
                          onCrop={
                            card.is_video ? undefined : () => openCrop(card)
                          }
                        />
                      ))}
                    </div>
                  </div>

                  {pageCount > 1 && (
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        gap: 10,
                        padding: 10,
                        borderTop: `1px solid ${colors.border}`,
                        fontFamily: font.mono,
                        fontSize: 12,
                      }}
                    >
                      <Button
                        disabled={page <= 1}
                        onClick={() => setPage(page - 1)}
                      >
                        ‹
                      </Button>
                      <span>
                        {page} / {pageCount}
                      </span>
                      <Button
                        disabled={page >= pageCount}
                        onClick={() => setPage(page + 1)}
                      >
                        ›
                      </Button>
                    </div>
                  )}
                </div>

                <MediaDetailPanel
                  focusKey={focusKey}
                  onClose={() => setFocusKey(null)}
                  datasetId={active.id}
                  datasetName={active.name}
                  onFocusChange={setFocusKey}
                />
              </div>
            )}
          </>
        ) : (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: colors.textFaint,
            }}
          >
            Create or select a dataset.
          </div>
        )}
      </div>

      {addOpen && active && (
        <DatasetComposerModal
          datasetId={active.id}
          datasetName={active.name}
          onClose={() => setAddOpen(false)}
        />
      )}
      {autoMounted && (
        <AutoBuildStudio
          open={autoOpen}
          onClose={() => setAutoOpen(false)}
          editId={editTarget?.id ?? null}
          initialRecipe={editTarget?.recipe ?? null}
          initialName={editTarget?.name ?? ""}
          nonce={studioNonce}
        />
      )}
    </div>
  );
}

function DatasetMeta({
  datasetId,
  deployName,
  deployResolution,
}: {
  datasetId: number;
  deployName: string;
  deployResolution: number;
}) {
  const update = useUpdateDataset();
  const triggerwords = useTriggerwords(datasetId);
  const addTrigger = useAddTriggerword();
  const removeTrigger = useRemoveTriggerword();
  const [deploy, setDeploy] = useState(deployName);
  const [resolution, setResolution] = useState(String(deployResolution));
  const [trigger, setTrigger] = useState("");

  useEffect(() => setDeploy(deployName), [deployName]);
  useEffect(() => setResolution(String(deployResolution)), [deployResolution]);

  return (
    <div
      style={{
        display: "flex",
        gap: 20,
        padding: "10px 16px",
        borderBottom: `1px solid ${colors.border}`,
        alignItems: "flex-end",
      }}
    >
      <div>
        <Label>Deploy folder name</Label>
        <span style={{ display: "flex", gap: 6 }}>
          <input
            value={deploy}
            onChange={(event) => setDeploy(event.target.value)}
            placeholder="(dataset name)"
            style={{ ...inputStyle, width: 160 }}
          />
          <Button
            onClick={() =>
              update.mutate({
                id: datasetId,
                deploy_name: deploy,
                deploy_resolution: Math.max(0, Number(resolution) || 0),
              })
            }
          >
            Save
          </Button>
        </span>
      </div>
      <div>
        <Label>Deploy size (min side, px — 0 = off)</Label>
        <input
          type="number"
          min={0}
          step={64}
          value={resolution}
          onChange={(event) => setResolution(event.target.value)}
          placeholder="1280"
          title="Shortest image side on deploy (Lanczos, downscale only, PNG). 0 keeps originals."
          style={{ ...inputStyle, width: 110 }}
        />
      </div>
      <div style={{ flex: 1 }}>
        <Label>Trigger words</Label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {triggerwords.data?.triggerwords.map((word) => (
            <TagChip
              key={word.id}
              name={word.name}
              onRemove={() =>
                removeTrigger.mutate({
                  id: datasetId,
                  triggerword_id: word.id,
                })
              }
            />
          ))}
          <input
            value={trigger}
            onChange={(event) => setTrigger(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && trigger.trim()) {
                addTrigger.mutate({ id: datasetId, name: trigger.trim() });
                setTrigger("");
              }
            }}
            placeholder="+ add"
            style={{ ...inputStyle, width: 90 }}
          />
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  padding: "5px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;
