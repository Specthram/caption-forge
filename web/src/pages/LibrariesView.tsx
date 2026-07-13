/** Libraries workspace: sources, the Index pipeline, near-duplicates. */

import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useBulkTags,
  useCreateUncategorizedTag,
  useMissingMedia,
  usePurgeMedia,
  useDeleteLibrary,
  useIndexRun,
  useIndexStatus,
  useLibraries,
  useLibraryMedia,
  useLookalikeDetect,
  useLookalikeDiscard,
  useLookalikeDismiss,
  useLookalikeKeepBest,
  useLookalikeResetDismissed,
  useReindexAllLibraries,
  useScanAllLibraries,
  useScanLibrary,
  useSetLibraryPath,
  useSetRecursive,
} from "../api/hooks";
import type {
  IndexStep,
  LibrarySource,
  LookalikeMember,
  LookalikeResult,
  StepCounts,
} from "../api/types";
import { overallCoverage, pct } from "../design/indexCoverage";
import { colors, font } from "../design/tokens";
import { useJobsStore } from "../store/jobsStore";
import { useUiStore } from "../store/uiStore";
import { Button, Label, ProgressBar, Slider } from "../components/atoms";
import {
  IndexLibraryCard,
  IndexRecapPanel,
  LibraryStepBadges,
} from "../components/organisms/IndexPipeline";
import { TagFilter } from "../components/molecules/TagFilter";
import type { SelectedTag } from "../components/molecules/TagFilter";
import { GridCard } from "../components/molecules/GridCard";
import { FolderBrowserModal } from "../components/organisms/FolderBrowserModal";
import { SubfolderMappingModal } from "../components/organisms/SubfolderMappingModal";

const PAGE = 60;
const EMPTY_COUNTS: StepCounts = {};

export function LibrariesView() {
  const client = useQueryClient();
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const openZoom = useUiStore((state) => state.openZoom);
  const setView = useUiStore((state) => state.setView);

  const libraries = useLibraries();
  const status = useIndexStatus();
  const deleteLibrary = useDeleteLibrary();
  const setRecursive = useSetRecursive();
  const setLibraryPath = useSetLibraryPath();
  const scan = useScanLibrary();
  const scanAll = useScanAllLibraries();
  const reindexAll = useReindexAllLibraries();
  const indexRun = useIndexRun();
  const bulkTags = useBulkTags();
  const createTag = useCreateUncategorizedTag();

  // The selected library and its page are restored after a refresh.
  const { activeId, page } = useUiStore((state) => state.librariesView);
  const setLibrariesView = useUiStore((state) => state.setLibrariesView);
  const setActiveId = (next: number | null) =>
    setLibrariesView({ activeId: next, page: 1 });
  const setPage = (next: number) => setLibrariesView({ page: next });
  const [editPathOpen, setEditPathOpen] = useState(false);
  // The subfolder-mapping wizard. `mappingTarget` is what is being mapped
  // (`add` = a new folder at step 1, a number = an existing library at step
  // 2); `mappingOpen` toggles visibility. A backdrop misclick only hides the
  // modal — the target stays, so reopening the SAME target restores the draft
  // exactly (the modal instance is kept mounted). ✕/Cancel/Apply clear it.
  const [mappingTarget, setMappingTarget] = useState<"add" | number | null>(
    null,
  );
  const [mappingOpen, setMappingOpen] = useState(false);
  const openMapping = (target: "add" | number) => {
    setMappingTarget((prev) => (prev === target ? prev : target));
    setMappingOpen(true);
  };
  const closeMapping = () => {
    setMappingOpen(false);
    setMappingTarget(null);
  };
  const [force, setForce] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [pendingRun, setPendingRun] = useState<{
    libraryId: number | null;
    steps: string[] | null;
  } | null>(null);
  const [addTags, setAddTags] = useState<SelectedTag[]>([]);
  const [removeTags, setRemoveTags] = useState<SelectedTag[]>([]);

  const libs = useMemo(
    () => libraries.data?.libraries ?? [],
    [libraries.data],
  );
  // Select the first library on a cold start, and fall back to it when the
  // one restored from the last session has since been deleted.
  useEffect(() => {
    if (!libs.length) return;
    if (activeId == null || !libs.some((lib) => lib.id === activeId)) {
      setActiveId(libs[0].id);
    }
  }, [activeId, libs]); // eslint-disable-line react-hooks/exhaustive-deps

  // An Index job rewrites every counter it touches: refresh them on the
  // job's last state change rather than polling the status endpoint.
  const job = useJobsStore((state) => (jobId ? state.jobs[jobId] : undefined));
  const running = job?.state === "queued" || job?.state === "running";
  useEffect(() => {
    if (!job || running) return;
    client.invalidateQueries({ queryKey: ["libraries"] });
    client.invalidateQueries({ queryKey: ["index-status"] });
    client.invalidateQueries({ queryKey: ["library-media"] });
    // A "Refresh scan" is what re-detects the files that left the disk:
    // the warning banner is the scan's own report, not a separate action.
    client.invalidateQueries({ queryKey: ["library-missing"] });
    setJobId(null);
    setPendingRun(null);
  }, [job, running, client]);
  const scanning = running && job?.type === "scan";
  const rescanningAll = running && job?.type === "scan-all";
  const reindexingAll = running && job?.type === "reindex-all";

  const steps: IndexStep[] = useMemo(
    () => status.data?.steps ?? [],
    [status.data],
  );
  const perLibrary = useMemo(() => {
    const map = new Map<number, StepCounts>();
    for (const library of status.data?.libraries ?? []) {
      map.set(library.id, library.steps);
    }
    return map;
  }, [status.data]);

  const active = libs.find((library) => library.id === activeId);
  const media = useLibraryMedia(activeId, (page - 1) * PAGE, PAGE, qualityMetric);
  const total = media.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE));

  // The library shrank since the last session: a restored page past its end
  // would paint an empty grid.
  useEffect(() => {
    if (media.data && page > pageCount) setPage(1);
  }, [media.data, page, pageCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const runIndex = (libraryId: number | null, picked: string[] | null) => {
    setPendingRun({ libraryId, steps: picked });
    indexRun.mutate(
      { library_id: libraryId, steps: picked, force },
      {
        onSuccess: (data) => setJobId(data.job_id),
        onError: () => setPendingRun(null),
      },
    );
  };

  const runScan = (libraryId: number) => {
    scan.mutate(libraryId, { onSuccess: (data) => setJobId(data.job_id) });
  };

  const runScanAll = () => {
    scanAll.mutate(undefined, {
      onSuccess: (data) => setJobId(data.job_id),
    });
  };

  const runReindexAll = () => {
    reindexAll.mutate(undefined, {
      onSuccess: (data) => setJobId(data.job_id),
    });
  };

  const busyRun = indexRun.isPending || running;
  const isPendingRun = (libraryId: number | null, steps: string[] | null) =>
    busyRun &&
    pendingRun !== null &&
    pendingRun.libraryId === libraryId &&
    JSON.stringify(pendingRun.steps) === JSON.stringify(steps);

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <div style={sourcesPanel}>
        <div style={{ flex: 1, overflowY: "auto", padding: 10 }}>
          {libs
            .filter((library) => library.parent_library_id == null)
            .map((library) => {
              const subs = libs.filter(
                (child) => child.parent_library_id === library.id,
              );
              if (library.mapped) {
                return (
                  <LibraryGroupCard
                    key={library.id}
                    library={library}
                    subs={subs}
                    activeId={activeId}
                    steps={steps}
                    perLibrary={perLibrary}
                    onSelect={setActiveId}
                    onEdit={() => openMapping(library.id)}
                  />
                );
              }
              return (
                <LibraryFlatRow
                  key={library.id}
                  library={library}
                  selected={library.id === activeId}
                  steps={steps}
                  counts={perLibrary.get(library.id) ?? EMPTY_COUNTS}
                  onSelect={() => setActiveId(library.id)}
                />
              );
            })}
        </div>
        <div style={{ padding: 10, borderTop: `1px solid ${colors.border}` }}>
          <Button block variant="accent" onClick={() => openMapping("add")}>
            + Add folder
          </Button>
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 0, overflowY: "auto" }}>
        {status.data && (
          <div style={{ padding: "14px 16px 0" }}>
            <IndexRecapPanel
              steps={steps}
              totals={status.data.totals}
              libraries={libs.length}
              onRun={runIndex}
              busy={isPendingRun(null, null)}
              onRescanAll={runScanAll}
              rescanning={rescanningAll}
              onReindexAll={runReindexAll}
              reindexing={reindexingAll}
              progressSub={
                rescanningAll || reindexingAll ? job?.sub : undefined
              }
              progressPct={
                rescanningAll || reindexingAll
                  ? pct(job?.done ?? 0, job?.total ?? 0)
                  : undefined
              }
            />
          </div>
        )}
        {active && (
          <div style={{ padding: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 16, fontWeight: 600 }}>
                  {active.name}
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 11,
                    fontFamily: font.mono,
                    color: colors.textMuted,
                  }}
                >
                  <span>
                    {active.path} · {active.count} media
                  </span>
                  {!active.internal && (
                    <button
                      onClick={() => setEditPathOpen(true)}
                      title="Change folder"
                      style={editPathBtn}
                    >
                      ✎
                    </button>
                  )}
                </div>
              </div>
              {!active.internal && active.parent_library_id == null && (
                <Button onClick={() => openMapping(active.id)}>
                  ⊞ Subfolder mapping…
                </Button>
              )}
              {!active.internal && (
                <label style={recursiveLabel}>
                  <input
                    type="checkbox"
                    checked={active.recursive}
                    onChange={(event) =>
                      setRecursive.mutate({
                        id: active.id,
                        recursive: event.target.checked,
                      })
                    }
                  />
                  Recursive
                </label>
              )}
              <Button
                variant="accent"
                loading={scanning}
                onClick={() => runScan(active.id)}
              >
                {scanning ? "Scanning…" : "⟳ Refresh scan"}
              </Button>
              {!active.internal && (
                <Button
                  variant="danger"
                  onClick={() => {
                    if (window.confirm(`Delete library "${active.name}"?`)) {
                      deleteLibrary.mutate(active.id);
                      setActiveId(null);
                    }
                  }}
                >
                  Delete
                </Button>
              )}
            </div>

            {scanning && (
              <div style={{ marginTop: 12 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 11,
                    fontFamily: font.mono,
                    color: colors.textMuted,
                    marginBottom: 4,
                  }}
                >
                  <span>Scanning folder…</span>
                  <span>{job?.sub}</span>
                </div>
                <ProgressBar
                  height={4}
                  color={colors.accent}
                  pct={pct(job?.done ?? 0, job?.total ?? 0)}
                />
              </div>
            )}

            <MissingFilesBanner libraryId={active.id} />

            <IndexLibraryCard
              name={active.name}
              libraryId={active.id}
              steps={steps}
              counts={perLibrary.get(active.id) ?? EMPTY_COUNTS}
              force={force}
              onForce={setForce}
              onRun={runIndex}
              onSettings={() => setView("settings")}
              busy={isPendingRun(active.id, null)}
              busyStep={
                pendingRun?.libraryId === active.id &&
                busyRun &&
                pendingRun.steps?.length === 1
                  ? pendingRun.steps[0]
                  : null
              }
            />

            <ActionCard title="Bulk tags" hint="add/remove on every media">
              <TagFilter
                label="+ add tag"
                selected={addTags}
                onAdd={(tag) => setAddTags((prev) => [...prev, tag])}
                onRemove={(id) =>
                  setAddTags((prev) => prev.filter((tag) => tag.id !== id))
                }
                onCreate={(name) =>
                  createTag.mutate(name, {
                    onSuccess: (data) =>
                      setAddTags((prev) =>
                        prev.some((tag) => tag.id === data.id)
                          ? prev
                          : [...prev, { id: data.id, name: data.name }],
                      ),
                  })
                }
              />
              <div style={{ height: 6 }} />
              <TagFilter
                label="− remove tag"
                selected={removeTags}
                onAdd={(tag) => setRemoveTags((prev) => [...prev, tag])}
                onRemove={(id) =>
                  setRemoveTags((prev) => prev.filter((tag) => tag.id !== id))
                }
              />
              <div style={{ marginTop: 8 }}>
                <Button
                  disabled={addTags.length === 0 && removeTags.length === 0}
                  onClick={() =>
                    bulkTags.mutate({
                      library_id: active.id,
                      add_tag_ids: addTags.map((tag) => tag.id),
                      remove_tag_ids: removeTags.map((tag) => tag.id),
                    })
                  }
                >
                  Apply to library
                </Button>
              </div>
            </ActionCard>

            <NearDuplicates />

            <div style={{ marginTop: 16 }}>
              <Label>Library media</Label>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(122px, 1fr))",
                  gap: 8,
                }}
              >
                {media.data?.items.map((card) => (
                  <GridCard
                    key={card.key}
                    card={card}
                    onClick={() =>
                      openZoom(`/api/media/${card.key}/file`, card.name)
                    }
                  />
                ))}
              </div>
              {pageCount > 1 && (
                <div style={pager}>
                  <Button disabled={page <= 1} onClick={() => setPage(page - 1)}>
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
          </div>
        )}
      </div>

      {mappingTarget !== null && (
        <SubfolderMappingModal
          key={String(mappingTarget)}
          open={mappingOpen}
          libraryId={typeof mappingTarget === "number" ? mappingTarget : null}
          initialPath={
            typeof mappingTarget === "number"
              ? (libs.find((lib) => lib.id === mappingTarget)?.path ?? "")
              : ""
          }
          initialName={
            typeof mappingTarget === "number"
              ? (libs.find((lib) => lib.id === mappingTarget)?.name ?? "")
              : ""
          }
          existing={libs}
          onHide={() => setMappingOpen(false)}
          onDiscard={closeMapping}
          onApplied={(jobId) => {
            setJobId(jobId);
            closeMapping();
          }}
        />
      )}

      {editPathOpen && active && (
        <FolderBrowserModal
          initialPath={active.path}
          hint="Repoint this library at the chosen folder."
          confirmLabel="Set folder"
          onClose={() => setEditPathOpen(false)}
          onSelect={(path) =>
            setLibraryPath.mutate({ id: active.id, path })
          }
        />
      )}
    </div>
  );
}

const GROUPS_PER_PAGE = 8;
const DUP_THUMB = 128;

/** Stable client id for a group: its sorted member keys. */
function groupId(group: { members: LookalikeMember[] }): string {
  return group.members
    .map((member) => member.key)
    .sort()
    .join("-");
}

/**
 * Near-duplicates review: collapsible card, paginated groups, each group a
 * collapsible mini sub-section. Per group the user keep-selects members then
 * validates (discards the rest) or hides the group indefinitely; clicking a
 * thumbnail opens the slider comparator.
 */
function NearDuplicates() {
  const openCompare = useUiStore((state) => state.openCompare);
  const detect = useLookalikeDetect();
  const keepBest = useLookalikeKeepBest();
  const discard = useLookalikeDiscard();
  const dismiss = useLookalikeDismiss();
  const reset = useLookalikeResetDismissed();

  const [open, setOpen] = useState(true);
  const [similarity, setSimilarity] = useState(88);
  const [groups, setGroups] = useState<LookalikeResult | null>(null);
  const [page, setPage] = useState(1);
  const [closed, setClosed] = useState<Set<string>>(new Set());
  const [kept, setKept] = useState<Record<string, Set<string>>>({});

  const runDetect = () =>
    detect.mutate(similarity, {
      onSuccess: (result) => {
        setGroups(result);
        setPage(1);
        setKept({});
        setClosed(new Set());
      },
    });

  const all = groups?.groups ?? [];
  const pageCount = Math.max(1, Math.ceil(all.length / GROUPS_PER_PAGE));
  const shown = all.slice((page - 1) * GROUPS_PER_PAGE, page * GROUPS_PER_PAGE);

  const removeGroup = (id: string) => {
    setGroups((prev) =>
      prev
        ? { ...prev, groups: prev.groups.filter((g) => groupId(g) !== id) }
        : prev,
    );
    setKept((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setPage((current) => {
      const remaining = all.length - 1;
      const maxPage = Math.max(1, Math.ceil(remaining / GROUPS_PER_PAGE));
      return Math.min(current, maxPage);
    });
  };

  const toggleClosed = (id: string) =>
    setClosed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleKeep = (id: string, key: string) =>
    setKept((prev) => {
      const set = new Set(prev[id] ?? []);
      if (set.has(key)) set.delete(key);
      else set.add(key);
      return { ...prev, [id]: set };
    });

  const compareMember = (
    group: { members: LookalikeMember[] },
    member: LookalikeMember,
  ) => {
    const best = group.members.find((m) => m.is_best) ?? group.members[0];
    let other = best;
    if (member.key === best.key) {
      const idx = group.members.findIndex((m) => m.key === member.key);
      other = group.members[(idx + 1) % group.members.length];
    }
    openCompare(
      `/api/media/${member.key}/file`,
      member.name,
      `/api/media/${other.key}/file`,
      other.name,
    );
  };

  const validateGroup = (group: { members: LookalikeMember[] }) => {
    const id = groupId(group);
    const keepSet = kept[id] ?? new Set<string>();
    const discardIds = group.members
      .filter((member) => !keepSet.has(member.key))
      .map((member) => Number(member.key));
    if (discardIds.length === 0) {
      removeGroup(id);
      return;
    }
    discard.mutate(discardIds, { onSuccess: () => removeGroup(id) });
  };

  const hideGroup = (group: { members: LookalikeMember[] }) => {
    const id = groupId(group);
    dismiss.mutate(
      group.members.map((member) => Number(member.key)),
      { onSuccess: () => removeGroup(id) },
    );
  };

  return (
    <div style={dupCard}>
      <div
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: "pointer",
        }}
      >
        <span style={{ color: colors.textMuted, fontSize: 11 }}>
          {open ? "▾" : "▸"}
        </span>
        <div style={{ fontWeight: 600, fontSize: 13 }}>Near-duplicates</div>
        <div style={{ fontSize: 10.5, color: colors.textFaint }}>
          click a thumb to keep (green ring) · ★ best · ⤢ to compare
        </div>
        {groups && (
          <span
            style={{ marginLeft: "auto", fontSize: 11, color: colors.textMuted }}
          >
            {all.length} group(s) · {groups.hashed_count} hashed
          </span>
        )}
      </div>

      {open && (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginTop: 12,
              flexWrap: "wrap",
            }}
          >
            <div style={{ width: 200 }}>
              <Label>Similarity ≥ {similarity}</Label>
              <Slider
                min={70}
                max={100}
                step={1}
                value={similarity}
                onChange={setSimilarity}
              />
            </div>
            <Button onClick={runDetect}>Detect</Button>
            {all.length > 0 && (
              <Button
                variant="accent"
                onClick={() =>
                  keepBest.mutate(similarity, {
                    onSuccess: () => setGroups(null),
                  })
                }
              >
                Keep best of each
              </Button>
            )}
            <Button
              variant="ghost"
              onClick={() =>
                reset.mutate(undefined, { onSuccess: runDetect })
              }
            >
              Reset dismissed
            </Button>
          </div>

          {groups && all.length === 0 && (
            <div style={{ marginTop: 12, fontSize: 12, color: colors.textMuted }}>
              No near-duplicate group.
            </div>
          )}

          {shown.length > 0 && (
            <div
              style={{
                marginTop: 12,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {shown.map((group) => {
                const id = groupId(group);
                const isClosed = closed.has(id);
                const keepSet = kept[id] ?? new Set<string>();
                const bestQuality = group.members[0]?.quality;
                return (
                  <div key={id} style={dupGroup}>
                    <div style={dupGroupHeader}>
                      <span
                        onClick={() => toggleClosed(id)}
                        style={{
                          cursor: "pointer",
                          color: colors.textMuted,
                          fontSize: 11,
                        }}
                      >
                        {isClosed ? "▸" : "▾"} {group.members.length} images
                        {bestQuality != null &&
                          ` · best Q${Math.round(bestQuality)}`}
                      </span>
                      <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                        <Button
                          variant="accent"
                          disabled={keepSet.size === 0}
                          onClick={() => validateGroup(group)}
                        >
                          Validate
                        </Button>
                        <Button
                          variant="ghost"
                          onClick={() => hideGroup(group)}
                        >
                          Hide indefinitely
                        </Button>
                      </div>
                    </div>
                    {!isClosed && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                        {group.members.map((member) => {
                          const isKept = keepSet.has(member.key);
                          return (
                            <div
                              key={member.key}
                              style={{ position: "relative", width: DUP_THUMB }}
                            >
                              <img
                                src={member.thumb}
                                alt={member.name}
                                loading="lazy"
                                onClick={() => toggleKeep(id, member.key)}
                                title="click to keep / unkeep"
                                style={{
                                  width: DUP_THUMB,
                                  height: DUP_THUMB,
                                  objectFit: "cover",
                                  borderRadius: 8,
                                  cursor: "pointer",
                                  border: isKept
                                    ? `3px solid ${colors.ok}`
                                    : member.is_best
                                      ? `2px solid ${colors.accent}`
                                      : `1px solid ${colors.border}`,
                                }}
                              />
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  compareMember(group, member);
                                }}
                                title="compare with the best"
                                style={compareBtn}
                              >
                                ⤢
                              </button>
                              {isKept && <span style={keptBadge}>✓ keep</span>}
                              {member.is_best && (
                                <span style={bestBadge}>★</span>
                              )}
                              {member.quality != null && (
                                <span style={qualityBadge}>
                                  Q{Math.round(member.quality)}
                                </span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {pageCount > 1 && (
            <div style={pager}>
              <Button disabled={page <= 1} onClick={() => setPage(page - 1)}>
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
        </>
      )}
    </div>
  );
}

/**
 * Missing-files cleanup: scan the library for media whose source file was
 * deleted off disk (the app hits a read error), list them, and offer to
 * purge them from the app for good (row + tags + captions; no file touched).
 */
/**
 * The Libraries view's missing-files banner.
 *
 * Not a card and not an action: media whose source file left the disk are a
 * real inconsistency between the database and the folder, so the banner only
 * exists while there are any — zero missing files, zero UI. Detection rides
 * on the scan that already walks the disk, so there is nothing to press.
 *
 * Purging drops each row from the app — media, tags and captions — and never
 * touches the disk.
 */
function MissingFilesBanner({ libraryId }: { libraryId: number }) {
  const missing = useMissingMedia(libraryId);
  const purge = usePurgeMedia();
  const [expanded, setExpanded] = useState(false);

  // A library switch re-evaluates against the new library's result.
  useEffect(() => setExpanded(false), [libraryId]);

  const items = missing.data?.media ?? [];
  if (items.length === 0) return null;

  const runPurge = () => {
    if (
      !window.confirm(
        `Remove ${items.length} missing media from the app? ` +
          "Their tags and captions go too. Files on disk are not touched.",
      )
    )
      return;
    purge.mutate(
      items.map((item) => item.id),
      { onSuccess: () => setExpanded(false) },
    );
  };

  return (
    <div style={missingBanner}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 13, color: colors.warn, flex: "none" }}>
          ⚠
        </span>
        <span
          style={{
            fontSize: 12,
            color: colors.text,
            fontWeight: 600,
            flex: "none",
            whiteSpace: "nowrap",
          }}
        >
          {items.length} media missing on disk
        </span>
        <span style={{ fontSize: 10.5, color: colors.textMuted, flex: 1 }}>
          files deleted outside the app — captions and tags still in the
          database
        </span>
        <button
          onClick={() => setExpanded((current) => !current)}
          style={missingToggle}
        >
          {expanded ? "Hide files" : "Show files"}
        </button>
        <button onClick={runPurge} disabled={purge.isPending} style={missingPurge}>
          Remove {items.length} from app
        </button>
      </div>
      {expanded && (
        <div style={missingList}>
          {items.map((item) => (
            <div key={item.id} style={missingRow}>
              ⚠ {item.name}
            </div>
          ))}
          <div style={{ fontSize: 10, color: colors.textFaint, marginTop: 4 }}>
            Removing purges each row from the app (media + tags + captions).
            Nothing on disk is touched.
          </div>
        </div>
      )}
    </div>
  );
}

const missingBanner = {
  border: `1px solid ${colors.accentBorder}`,
  borderRadius: 9,
  background: "#211c10",
  padding: "10px 12px",
  marginTop: 12,
  display: "flex",
  flexDirection: "column",
} as const;

const missingToggle = {
  padding: "5px 11px",
  border: `1px solid ${colors.accentBorder}`,
  borderRadius: 6,
  background: "transparent",
  color: colors.warn,
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const missingPurge = {
  padding: "5px 11px",
  border: "1px solid #3a2622",
  borderRadius: 6,
  background: "#241715",
  color: colors.danger,
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const missingList = {
  marginTop: 9,
  borderTop: "1px solid #3a2f1a",
  paddingTop: 9,
  maxHeight: 150,
  overflowY: "auto",
  display: "flex",
  flexDirection: "column",
  gap: 3,
} as const;

const missingRow = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.warn,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} as const;

function ActionCard({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: colors.card,
        border: `1px solid ${colors.border}`,
        borderRadius: 9,
        padding: 14,
        marginTop: 14,
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 13 }}>{title}</div>
      {hint && (
        <div style={{ fontSize: 10.5, color: colors.textFaint, marginBottom: 10 }}>
          {hint}
        </div>
      )}
      {children}
    </div>
  );
}

/** A plain (un-mapped) library row in the Libraries sidebar. */
function LibraryFlatRow({
  library,
  selected,
  steps,
  counts,
  onSelect,
}: {
  library: LibrarySource;
  selected: boolean;
  steps: IndexStep[];
  counts: StepCounts;
  onSelect: () => void;
}) {
  const coverage = overallCoverage(steps, counts);
  const percent = pct(coverage.done, coverage.total);
  const color = coverage.missing > 0 ? colors.warn : colors.ok;
  return (
    <div
      onClick={onSelect}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 5,
        padding: "8px 10px",
        borderRadius: 7,
        cursor: "pointer",
        marginBottom: 4,
        border: `1px solid ${selected ? colors.accentBorder : "transparent"}`,
        background: selected ? colors.accentTint : "transparent",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            fontSize: 12.5,
            fontWeight: 600,
            color: selected ? colors.accent : colors.textSecondary,
          }}
        >
          {library.internal ? "◆ " : ""}
          {library.name}
        </span>
        <span
          style={{
            marginLeft: "auto",
            fontFamily: font.mono,
            fontSize: 10,
            color,
          }}
        >
          {steps.length ? `${percent}%` : "—"}
        </span>
      </div>
      <div
        style={{
          fontSize: 10,
          fontFamily: font.mono,
          color: colors.textFaint,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {library.path} · {library.count}
      </div>
      <ProgressBar height={3} color={color} pct={percent} />
      <LibraryStepBadges steps={steps} counts={counts} />
    </div>
  );
}

/**
 * A mapped library rendered as one bordered group: the parent (with a
 * `mapped` badge and a rule summary), its sub-libraries on tree rails, and a
 * footer that reopens the subfolder-mapping wizard.
 */
function LibraryGroupCard({
  library,
  subs,
  activeId,
  steps,
  perLibrary,
  onSelect,
  onEdit,
}: {
  library: LibrarySource;
  subs: LibrarySource[];
  activeId: number | null;
  steps: IndexStep[];
  perLibrary: Map<number, StepCounts>;
  onSelect: (id: number) => void;
  onEdit: () => void;
}) {
  const ruleLine = [
    `${library.rule_count} folder rule${library.rule_count === 1 ? "" : "s"}`,
    library.skipped_folders > 0 ? `${library.skipped_folders} skipped` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div style={groupCard}>
      <div
        onClick={() => onSelect(library.id)}
        style={{
          ...groupParent,
          background:
            library.id === activeId ? colors.accentTint : "transparent",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ fontSize: 11, color: colors.textMuted }}>▸</span>
          <span
            style={{
              flex: 1,
              fontSize: 12.5,
              fontWeight: 600,
              color: colors.text,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {library.name}
          </span>
          <span
            title="Subfolder mapping applied — rules persist for future files"
            style={mappedBadge}
          >
            mapped
          </span>
        </div>
        <div style={groupMeta}>
          {library.path} · {library.count}
        </div>
        <div style={{ ...groupMeta, color: colors.textMuted }}>{ruleLine}</div>
      </div>
      {subs.map((sub, index) => (
        <div
          key={sub.id}
          onClick={() => onSelect(sub.id)}
          title={`Sub-library — excluded from "${library.name}"; edit via the parent's mapping`}
          style={{
            ...groupSubRow,
            background:
              sub.id === activeId ? colors.accentTint : "transparent",
          }}
        >
          <span style={groupRail}>
            {index === subs.length - 1 ? "└" : "├"}
          </span>
          <span
            style={{
              flex: 1,
              fontSize: 11.5,
              fontWeight: 600,
              color: colors.accent,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {sub.name}
          </span>
          <span style={{ fontFamily: font.mono, fontSize: 9.5, color: colors.textFaint }}>
            {sub.count}
          </span>
          <span style={subBadge}>sub</span>
        </div>
      ))}
      <div style={groupFooter}>
        <span
          style={{
            flex: 1,
            fontFamily: font.mono,
            fontSize: 9.5,
            color: colors.textFaint,
          }}
        >
          {subs.length} sub-librar{subs.length === 1 ? "y" : "ies"}
        </span>
        <button onClick={onEdit} style={editMappingBtn}>
          ⊞ Edit mapping
        </button>
      </div>
      {/* The step badges keep the parent's index coverage visible in the card. */}
      <div style={{ padding: "0 10px 8px" }}>
        <LibraryStepBadges
          steps={steps}
          counts={perLibrary.get(library.id) ?? EMPTY_COUNTS}
        />
      </div>
    </div>
  );
}

const sourcesPanel = {
  width: 250,
  flex: "none",
  borderRight: `1px solid ${colors.border}`,
  background: colors.panel,
  display: "flex",
  flexDirection: "column",
} as const;

const groupCard = {
  display: "flex",
  flexDirection: "column",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 8,
  background: colors.app,
  overflow: "hidden",
  marginBottom: 4,
} as const;

const groupParent = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  padding: "8px 10px",
  cursor: "pointer",
} as const;

const groupMeta = {
  fontSize: 10,
  fontFamily: font.mono,
  color: colors.textFaint,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} as const;

const mappedBadge = {
  fontSize: 9,
  fontFamily: font.mono,
  padding: "1px 5px",
  borderRadius: 4,
  background: "#152a17",
  color: colors.ok,
} as const;

const groupSubRow = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  padding: "6px 10px 6px 16px",
  cursor: "pointer",
  borderTop: "1px solid #1e2026",
} as const;

const groupRail = {
  fontSize: 10,
  color: colors.accentBorder,
  fontFamily: font.mono,
} as const;

const subBadge = {
  fontSize: 9,
  fontFamily: font.mono,
  padding: "1px 5px",
  borderRadius: 4,
  background: "#2e2513",
  color: colors.accent,
} as const;

const groupFooter = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  padding: "7px 10px",
  borderTop: "1px solid #1e2026",
  background: colors.panel,
} as const;

const editMappingBtn = {
  padding: "4px 10px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 6,
  background: "transparent",
  color: colors.textSecondary,
  fontSize: 10.5,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const editPathBtn = {
  border: "none",
  background: "transparent",
  color: colors.textMuted,
  cursor: "pointer",
  fontSize: 12,
  padding: "0 2px",
  lineHeight: 1,
} as const;

const recursiveLabel = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  fontSize: 11.5,
  color: colors.textMuted,
  cursor: "pointer",
} as const;

const pager = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  gap: 10,
  padding: 10,
  fontFamily: font.mono,
  fontSize: 12,
} as const;

const dupCard = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: 9,
  padding: 14,
  marginTop: 14,
} as const;

const dupGroup = {
  border: `1px solid ${colors.border}`,
  borderRadius: 8,
  padding: 10,
  background: colors.panel,
} as const;

const dupGroupHeader = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 8,
} as const;

const compareBtn = {
  position: "absolute",
  top: 5,
  left: 5,
  border: "none",
  borderRadius: 5,
  padding: "2px 7px",
  fontSize: 13,
  lineHeight: 1,
  fontWeight: 700,
  cursor: "pointer",
  background: "rgba(0,0,0,0.6)",
  color: colors.text,
} as const;

const keptBadge = {
  position: "absolute",
  bottom: 5,
  left: 5,
  fontSize: 10,
  fontWeight: 700,
  padding: "1px 6px",
  borderRadius: 5,
  background: colors.ok,
  color: colors.onAccent,
} as const;

const bestBadge = {
  position: "absolute",
  top: 4,
  right: 6,
  color: colors.accent,
  fontSize: 14,
  textShadow: "0 1px 2px rgba(0,0,0,0.6)",
} as const;

const qualityBadge = {
  position: "absolute",
  bottom: 5,
  right: 5,
  fontFamily: font.mono,
  fontSize: 10,
  fontWeight: 700,
  padding: "1px 5px",
  borderRadius: 5,
  background: "rgba(0,0,0,0.6)",
  color: colors.text,
} as const;
