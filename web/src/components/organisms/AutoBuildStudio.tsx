/**
 * The Auto-build Studio: propose a whole training set from a library and
 * explain every pick.
 *
 * Three columns, like the composer, but the centre is the engine's own
 * proposal, not a hand-picked list: the left rail is the *recipe* (subject,
 * size, quality, framing), the centre shows the picks — each with the
 * reasons it was chosen and a ✕ to discard it (the engine re-picks the next
 * best) — and the right panel projects the grade, the framing split and the
 * coverage map of the proposal. Every recipe change or manual edit re-runs
 * the selection server-side (``POST /autobuild/preview``), debounced, so
 * dragging a slider costs one recomputation, not twenty.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useAutobuildConfig,
  useAutobuildCreate,
  useAutobuildNeighbors,
  useAutobuildPreviewStream,
  useAutobuildSuggestTags,
  useAutobuildUpdate,
  useLibraryGrid,
  useReleaseAutobuildModel,
} from "../../api/hooks";
import type { AutobuildRecipe } from "../../api/hooks";
import type {
  AutobuildPick,
  AutobuildStudioPreview,
  AutobuildSuggestedTag,
} from "../../api/types";
import { colors, font, radii } from "../../design/tokens";
import { AutoBuildCoverageMap } from "../molecules/AutoBuildCoverageMap";

const SUGGEST_DEBOUNCE_MS = 500;
const SIZE_PRESETS = [20, 50, 100, 500];
const TYPES = [
  { value: "img", label: "Images" },
  { value: "vid", label: "Videos" },
];
const FRAMING_HINT: Record<string, string> = {
  balanced: "A balanced spread of face, upper-body and full-body shots.",
  portraits: "Weighted toward faces and close-ups.",
  wide: "Weighted toward full-body and wide shots.",
  free: "No framing target — pure quality and diversity.",
};

const thumbUrl = (id: number) => `/api/media/${id}/thumb`;

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

function gradeColor(score: number | null): string {
  if (score == null) return colors.textFaint;
  if (score >= 83) return colors.ok;
  if (score >= 65) return colors.warn;
  return colors.danger;
}

function bandColor(score: number | null | undefined): string {
  if (score == null) return colors.textFaint;
  if (score >= 85) return colors.ok;
  if (score >= 70) return colors.warn;
  return colors.danger;
}

export function AutoBuildStudio({
  open,
  onClose,
  editId = null,
  initialRecipe = null,
  initialName = "",
  nonce = 0,
}: {
  open: boolean;
  onClose: () => void;
  /** Dataset being re-edited, or null for a fresh build. */
  editId?: number | null;
  /** Its stored recipe, prefilled into the knobs when re-editing. */
  initialRecipe?: AutobuildRecipe | null;
  /** Its name, prefilled for the overwrite/"save as new" actions. */
  initialName?: string;
  /** Bumped by the opener each time; drives a one-shot prefill on re-edit. */
  nonce?: number;
}) {
  const config = useAutobuildConfig();
  const preview = useAutobuildPreviewStream();
  const suggest = useAutobuildSuggestTags();
  const create = useAutobuildCreate();
  const update = useAutobuildUpdate();
  const release = useReleaseAutobuildModel();
  const releaseRef = useRef(release);
  releaseRef.current = release;

  const [mediaType, setMediaType] = useState("img");
  const [semanticQ, setSemanticQ] = useState("");
  const [lockedTags, setLockedTags] = useState<string[]>([]);
  const [excludeTags, setExcludeTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [excludeInput, setExcludeInput] = useState("");
  const [seeds, setSeeds] = useState<number[]>([]);
  const [size, setSize] = useState(50);
  const [metric, setMetric] = useState("");
  const [minScore, setMinScore] = useState(60);
  const [excludeBlur, setExcludeBlur] = useState(true);
  const [framing, setFraming] = useState("balanced");
  const [live, setLive] = useState(true);
  const [dropped, setDropped] = useState<number[]>([]);
  const [forced, setForced] = useState<number[]>([]);
  const [kept, setKept] = useState<number[]>([]);
  const [rebal, setRebal] = useState(false);
  const [view, setView] = useState<"grid" | "clusters">("grid");
  const [name, setName] = useState("");
  const [result, setResult] = useState<AutobuildStudioPreview | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [swapId, setSwapId] = useState<number | null>(null);
  const [swapQ, setSwapQ] = useState("");
  const [selId, setSelId] = useState<number | null>(null);
  const [clusterSel, setClusterSel] = useState<number | null>(null);
  const [previewId, setPreviewId] = useState<number | null>(null);
  const [flagHover, setFlagHover] = useState<number | null>(null);
  // The recipe knobs snapshot at the last Build — the amber "recipe changed"
  // note lights when the live knobs drift from it (edits do not count).
  const [builtSnapshot, setBuiltSnapshot] = useState<string | null>(null);
  const [triageOpen, setTriageOpen] = useState(false);
  // A re-edit prefills the knobs, then asks for a Build on the next commit
  // (once ``recipeRef`` reflects the loaded recipe). One-shot per prefill.
  const [pendingBuild, setPendingBuild] = useState(false);
  const neighbors = useAutobuildNeighbors();

  const recipe = useMemo<AutobuildRecipe>(
    () => ({
      media_type: mediaType,
      semantic_q: semanticQ,
      locked_tags: lockedTags,
      exclude_tags: excludeTags,
      seed_media_ids: seeds,
      size,
      metric: metric || null,
      min_score: minScore,
      exclude_blur: excludeBlur,
      framing_preset: framing,
      live,
      dropped,
      forced,
      kept,
      rebal,
    }),
    [
      mediaType,
      semanticQ,
      lockedTags,
      excludeTags,
      seeds,
      size,
      metric,
      minScore,
      excludeBlur,
      framing,
      live,
      dropped,
      forced,
      kept,
      rebal,
    ],
  );
  // The knobs only (no manual edits): the Build snapshot compares against
  // this, so discarding a pick or rebalancing never lights the amber note.
  const knobsKey = useMemo(
    () =>
      JSON.stringify({
        mediaType,
        semanticQ,
        lockedTags,
        excludeTags,
        seeds,
        size,
        metric,
        minScore,
        excludeBlur,
        framing,
        live,
      }),
    [
      mediaType,
      semanticQ,
      lockedTags,
      excludeTags,
      seeds,
      size,
      metric,
      minScore,
      excludeBlur,
      framing,
      live,
    ],
  );
  const dirty = builtSnapshot != null && builtSnapshot !== knobsKey;
  const debouncedQuery = useDebounced(semanticQ, SUGGEST_DEBOUNCE_MS);

  const previewRef = useRef(preview);
  previewRef.current = preview;
  // No auto-run: the recipe knobs are set freely, then a manual Build runs
  // the (slow) selection. ``recipeRef`` hands the current recipe to the
  // edit-triggered re-pick without re-running on every knob change.
  const recipeRef = useRef(recipe);
  recipeRef.current = recipe;
  const builtRef = useRef(false);
  const runBuild = () => {
    builtRef.current = true;
    setBuiltSnapshot(knobsKey);
    setClusterSel(null);
    setSelId(null);
    void previewRef.current.run(recipeRef.current, setResult);
  };

  // Reopen a saved dataset: prefill every knob from its stored recipe, then
  // ask for a Build. Guarded by the opener's nonce so it runs once per open
  // and never clobbers a fresh build's in-progress recipe.
  const appliedNonceRef = useRef<number | null>(null);
  useEffect(() => {
    if (!open || appliedNonceRef.current === nonce) return;
    appliedNonceRef.current = nonce;
    if (editId == null || !initialRecipe) return;
    const r = initialRecipe;
    setMediaType(r.media_type);
    setSemanticQ(r.semantic_q ?? "");
    setLockedTags(r.locked_tags ?? []);
    setExcludeTags(r.exclude_tags ?? []);
    setSeeds(r.seed_media_ids ?? []);
    setSize(r.size ?? 50);
    setMetric(r.metric ?? "");
    setMinScore(r.min_score ?? 60);
    setExcludeBlur(r.exclude_blur ?? true);
    setFraming(r.framing_preset ?? "balanced");
    setLive(r.live ?? true);
    setDropped(r.dropped ?? []);
    setForced(r.forced ?? []);
    setKept(r.kept ?? []);
    setRebal(r.rebal ?? false);
    setName(initialName ?? "");
    setView("grid");
    setResult(null);
    setSelId(null);
    setClusterSel(null);
    setPendingBuild(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, nonce]);

  // The prefill's deferred Build: ``recipeRef`` now mirrors the loaded knobs.
  useEffect(() => {
    if (!pendingBuild) return;
    setPendingBuild(false);
    runBuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingBuild]);
  // A pick-level edit (discard, swap, map-replace, rebalance) after the
  // first Build re-picks live — those are explicit actions on an existing
  // proposal, not passive knobs, so they do not light the amber note.
  useEffect(() => {
    if (!builtRef.current) return;
    // A pick edit leaves the recipe scope untouched: reuse the last Build's
    // pool and geometry so only the re-pick runs, not another library read.
    void previewRef.current.run(recipeRef.current, setResult, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dropped, forced, kept, rebal]);

  const suggestRef = useRef(suggest);
  suggestRef.current = suggest;
  useEffect(() => {
    if (debouncedQuery.trim()) suggestRef.current.mutate(debouncedQuery.trim());
  }, [debouncedQuery]);

  // Hand SigLIP's VRAM back whenever the Studio closes (it stays mounted so
  // an incidental close keeps the recipe; the checkpoint reloads on demand
  // when the semantic search runs again). Also release on final unmount.
  useEffect(() => {
    if (!open) releaseRef.current.mutate();
  }, [open]);
  useEffect(() => () => void releaseRef.current.mutate(), []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!open || event.key !== "Escape" || triageOpen) return;
      // Escape unwinds the transient modes first, the Studio last. Closing
      // this way keeps the recipe — only Cancel resets it.
      if (previewId != null) setPreviewId(null);
      else if (selId != null) setSelId(null);
      else if (swapId != null) setSwapId(null);
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, triageOpen, previewId, selId, swapId]);

  // Reset the whole recipe to its defaults — Cancel and a successful Create.
  const reset = () => {
    builtRef.current = false;
    setMediaType("img");
    setSemanticQ("");
    setLockedTags([]);
    setExcludeTags([]);
    setTagInput("");
    setExcludeInput("");
    setSeeds([]);
    setSize(50);
    setMetric("");
    setMinScore(60);
    setExcludeBlur(true);
    setFraming("balanced");
    setLive(true);
    setDropped([]);
    setForced([]);
    setKept([]);
    setRebal(false);
    setView("grid");
    setName("");
    setResult(null);
    setHovered(null);
    setSwapId(null);
    setSwapQ("");
    setSelId(null);
    setClusterSel(null);
    setPreviewId(null);
    setFlagHover(null);
    setBuiltSnapshot(null);
    setTriageOpen(false);
  };
  const cancel = () => {
    reset();
    onClose();
  };

  const metrics = config.data?.metrics ?? [];
  const effectiveMetric = metric || metrics[0]?.id || "";
  const picks = result?.picks ?? [];
  const suggested = suggest.data?.tags ?? [];
  const recomputing = preview.isPending;

  const addTag = (tagName: string) => {
    const clean = tagName.trim().toLowerCase();
    if (clean && !lockedTags.includes(clean)) {
      setLockedTags((prev) => [...prev, clean]);
    }
  };
  const discard = (id: number) => {
    setDropped((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setSelId((prev) => (prev === id ? null : prev));
    setSwapId((prev) => (prev === id ? null : prev));
  };
  const keepPick = (id: number) =>
    setKept((prev) => (prev.includes(id) ? prev : [...prev, id]));

  const flaggedPicks = picks.filter((pick) => pick.flag);

  const neighborsRef = useRef(neighbors);
  neighborsRef.current = neighbors;
  const runNeighbors = (id: number, query: string) =>
    neighborsRef.current.mutate({
      media_id: id,
      media_type: mediaType,
      metric: metric || null,
      exclude_ids: picks.map((pick) => pick.media_id),
      q: query,
    });
  const openSwap = (id: number) => {
    setSelId(null);
    setSwapQ("");
    setSwapId(id);
    runNeighbors(id, "");
  };
  const applySwap = (outId: number, inId: number) => {
    setForced((prev) => (prev.includes(inId) ? prev : [...prev, inId]));
    setDropped((prev) => (prev.includes(outId) ? prev : [...prev, outId]));
    setSwapId(null);
    setSwapQ("");
  };
  const debouncedSwapQ = useDebounced(swapQ, 300);
  useEffect(() => {
    if (swapId != null) runNeighbors(swapId, debouncedSwapQ.trim());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSwapQ]);

  // Selecting a pick card arms "Replace mode": clicking a grey map point
  // then swaps it in (forced in / dropped out — the sets the engine replays).
  const toggleSelect = (id: number) => {
    setSwapId(null);
    setSelId((prev) => (prev === id ? null : id));
  };
  const replaceFromMap = (inId: number) => {
    if (selId == null) return;
    setForced((prev) => (prev.includes(inId) ? prev : [...prev, inId]));
    setDropped((prev) => (prev.includes(selId) ? prev : [...prev, selId]));
    setSelId(null);
  };

  const cardHandlers: CardHandlers = {
    selId,
    flagHover,
    onDiscard: discard,
    onSwap: openSwap,
    onSelect: toggleSelect,
    onZoom: (id) => setPreviewId(id),
    onKeep: keepPick,
    onFlagEnter: (id) => setFlagHover(id),
    onFlagLeave: (id) =>
      setFlagHover((prev) => (prev === id ? null : prev)),
    onEnter: (id) => setHovered(id),
    onLeave: () => setHovered(null),
  };

  const placeholderName = result?.dominant_tag
    ? `${result.dominant_tag.name}_${picks.length}`
    : `dataset_${picks.length}`;

  const isEdit = editId != null;

  const runCreate = () => {
    let finalName = (name.trim() || placeholderName).trim();
    // "Save as new" while re-editing: never collide with the original name.
    if (isEdit && initialName && finalName === initialName.trim()) {
      finalName = `${finalName}_copy`;
    }
    if (!picks.length || !finalName) return;
    create.mutate(
      {
        name: finalName,
        selection: picks.map((pick) => pick.media_id),
        recipe: { ...recipe, dropped },
      },
      {
        onSuccess: () => {
          reset();
          onClose();
        },
      },
    );
  };

  const runOverwrite = () => {
    if (editId == null || !picks.length) return;
    update.mutate(
      {
        datasetId: editId,
        selection: picks.map((pick) => pick.media_id),
        recipe: { ...recipe, dropped },
      },
      {
        onSuccess: () => {
          reset();
          onClose();
        },
      },
    );
  };

  const indexWarn =
    (config.data?.unhashed ?? 0) > 0 || (config.data?.unembedded ?? 0) > 0;

  // Closed but still mounted: render nothing, keep every bit of state so the
  // next open restores the in-progress recipe (all hooks ran above).
  if (!open) return null;

  return (
    <div onClick={onClose} style={backdrop}>
      <div onClick={(event) => event.stopPropagation()} style={panel}>
        <div style={header}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700 }}>
              {isEdit
                ? `Edit dataset — ${initialName || "Studio"}`
                : "Auto-build a dataset — Studio"}
            </div>
            <div style={headerSub}>
              set the recipe · press Build to run the selection · linked,
              never copied
            </div>
          </div>
          <div style={{ flex: 1 }} />
          <span onClick={onClose} style={closeX}>
            ✕
          </span>
        </div>

        <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
          {/* ---- left rail: the recipe ---- */}
          <div style={rail}>
            <Section label="Type">
              <Seg value={mediaType} onChange={setMediaType} options={TYPES} />
            </Section>

            <Section label="Subject" divided>
              <div style={searchBox}>
                <span style={{ fontSize: 11, color: colors.textFaint }}>⌕</span>
                <input
                  value={semanticQ}
                  onChange={(event) => setSemanticQ(event.target.value)}
                  placeholder="Describe the subject (SigLIP2)…"
                  disabled={result ? !result.semantic_available : false}
                  style={searchInput}
                />
              </div>
              {semanticQ.trim() && result && result.semantic_available && (
                <div style={{ fontSize: 10.5, color: colors.info }}>
                  {suggest.isPending
                    ? "⌕ SigLIP2 · searching…"
                    : `${result.matched} media match the subject`}
                </div>
              )}
              {suggest.isPending ? (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                  {[0, 1, 2, 3].map((i) => (
                    <span
                      key={i}
                      style={{
                        ...pulseChip,
                        animationDelay: `${i * 0.15}s`,
                      }}
                    />
                  ))}
                </div>
              ) : (
                suggested.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {suggested
                      .filter((tag) => !lockedTags.includes(tag.name))
                      .slice(0, 8)
                      .map((tag) => (
                        <SuggestedChip
                          key={tag.name}
                          tag={tag}
                          onClick={() => addTag(tag.name)}
                        />
                      ))}
                  </div>
                )
              )}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {lockedTags.map((tag) => (
                  <span key={tag} style={lockedChip}>
                    # {tag}
                    <span
                      onClick={() =>
                        setLockedTags((prev) =>
                          prev.filter((item) => item !== tag),
                        )
                      }
                      style={chipX}
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
                    addTag(tagInput);
                    setTagInput("");
                  }
                }}
                placeholder="+ require a tag ⏎"
                style={input}
              />
              {lockedTags.length > 0 && (
                <div style={{ fontSize: 10.5, color: colors.textFaint }}>
                  Required: only media carrying every tag are considered.
                </div>
              )}
              {excludeTags.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                  {excludeTags.map((tag) => (
                    <span key={tag} style={excludeChip}>
                      − {tag}
                      <span
                        onClick={() =>
                          setExcludeTags((prev) =>
                            prev.filter((item) => item !== tag),
                          )
                        }
                        style={chipX}
                      >
                        ✕
                      </span>
                    </span>
                  ))}
                </div>
              )}
              <input
                value={excludeInput}
                onChange={(event) => setExcludeInput(event.target.value)}
                onKeyDown={(event) => {
                  const clean = excludeInput.trim().toLowerCase();
                  if (event.key === "Enter" && clean) {
                    setExcludeTags((prev) =>
                      prev.includes(clean) ? prev : [...prev, clean],
                    );
                    setExcludeInput("");
                  }
                }}
                placeholder="− exclude a tag ⏎"
                style={input}
              />
              {result &&
                (lockedTags.length > 0 || excludeTags.length > 0) && (
                  <div style={prefLine}>
                    tag pre-filter: {result.pref_before} → {result.pref_after}{" "}
                    candidates — speeds up the build
                  </div>
                )}
              <div style={sectionLabel}>Example seeds</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {seeds.map((id) => (
                  <div key={id} style={{ position: "relative" }}>
                    <img src={thumbUrl(id)} alt="" style={seedThumb} />
                    <span
                      onClick={() =>
                        setSeeds((prev) => prev.filter((x) => x !== id))
                      }
                      style={seedRemove}
                    >
                      ✕
                    </span>
                  </div>
                ))}
                <SeedPicker
                  seeds={seeds}
                  onAdd={(id) => setSeeds((prev) => [...prev, id])}
                />
              </div>
            </Section>

            <Section label="Size" divided>
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                {SIZE_PRESETS.map((value) => (
                  <button
                    key={value}
                    onClick={() => setSize(value)}
                    style={presetButton(size === value)}
                  >
                    {value}
                  </button>
                ))}
                <input
                  type="number"
                  min={1}
                  value={size}
                  onChange={(event) =>
                    setSize(Math.max(1, Number(event.target.value)))
                  }
                  style={{ ...input, width: 62 }}
                />
              </div>
              {result && (
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: 10.5,
                    color:
                      result.shortfall > 0 ? colors.warn : colors.textMuted,
                  }}
                >
                  {result.shortfall > 0
                    ? `${size} asked · ${result.eligible} pass subject + floor — build stops there`
                    : `${result.eligible} eligible candidate(s) — target reachable`}
                </div>
              )}
            </Section>

            <Section label="Quality" divided>
              <select
                value={effectiveMetric}
                onChange={(event) => setMetric(event.target.value)}
                style={select}
              >
                {metrics.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div style={inlineRow}>
                <span style={fieldLabel}>Min</span>
                <input
                  type="range"
                  min={0}
                  max={95}
                  step={1}
                  value={minScore}
                  onChange={(event) => setMinScore(Number(event.target.value))}
                  style={{ flex: 1 }}
                />
                <input
                  type="number"
                  min={0}
                  max={95}
                  value={minScore}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setMinScore(
                      Number.isNaN(value)
                        ? 0
                        : Math.max(0, Math.min(95, value)),
                    );
                  }}
                  style={{
                    ...input,
                    width: 46,
                    padding: "4px 6px",
                    fontFamily: font.mono,
                    color: minScore >= 70 ? colors.ok : colors.text,
                  }}
                />
              </div>
              <Check
                checked={excludeBlur}
                onChange={setExcludeBlur}
                label="Exclude detected blur"
              />
            </Section>

            <Section label="Framing" divided>
              <Seg
                value={framing}
                onChange={setFraming}
                options={(config.data?.framing_presets ?? []).map((p) => ({
                  value: p.key,
                  label: p.label,
                }))}
              />
              <div style={{ fontSize: 10.5, color: colors.textFaint }}>
                {FRAMING_HINT[framing]}
              </div>
            </Section>

            <Section label="Living dataset" divided>
              <Check
                checked={live}
                onChange={setLive}
                label="◇ Keep this dataset living"
              />
              <div style={{ fontSize: 10.5, color: colors.textFaint }}>
                The recipe is saved; after each index it can propose stronger
                swaps.
              </div>
            </Section>

            {indexWarn && (
              <div style={indexNote}>
                ⚠ {config.data?.unhashed ?? 0} not indexed ·{" "}
                {config.data?.unembedded ?? 0} without embedding — run the
                Libraries actions for full signal.
              </div>
            )}

            <div style={{ flex: 1 }} />
            {dirty && !recomputing && (
              <div style={dirtyNote}>
                ⚠ recipe changed since the last build — rebuild to apply
              </div>
            )}
            <button
              onClick={runBuild}
              disabled={recomputing}
              style={{
                ...buildButton,
                opacity: recomputing ? 0.5 : 1,
                cursor: recomputing ? "default" : "pointer",
                boxShadow: dirty ? "0 0 0 2px rgba(224,179,86,0.4)" : "none",
              }}
            >
              {recomputing
                ? "Building…"
                : builtRef.current
                  ? "⟳ Rebuild selection"
                  : "⟳ Build selection"}
            </button>
          </div>

          {/* ---- centre: the proposal ---- */}
          <div
            style={{
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              position: "relative",
            }}
          >
            <div style={toolbar}>
              <span style={{ fontSize: 12, fontWeight: 600 }}>
                Proposed selection
              </span>
              <span style={counterAccent}>
                {picks.length} / {size} asked
              </span>
              <div style={{ flex: 1 }} />
              {recomputing && (
                <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                  <span style={miniSpinner} />{" "}
                  {preview.stage ? `${preview.stage.label}…` : "recomputing…"}
                </span>
              )}
              <Seg
                value={view}
                onChange={(value) => setView(value as "grid" | "clusters")}
                options={[
                  { value: "grid", label: "Grid" },
                  { value: "clusters", label: "Clusters" },
                ]}
              />
              {dropped.length > 0 && (
                <span
                  onClick={() => setDropped([])}
                  style={{ ...clearLink, fontFamily: font.mono }}
                >
                  {dropped.length} discarded · reset
                </span>
              )}
            </div>

            {swapId != null && (
              <div style={swapStrip}>
                <span style={{ fontSize: 10.5, color: colors.textMuted }}>
                  {swapQ.trim()
                    ? "⇄ swap by tag / name"
                    : "⇄ swap for a visual neighbour"}
                </span>
                <div style={swapSearch}>
                  <span style={{ fontSize: 10, color: colors.textFaint }}>
                    ⌕
                  </span>
                  <input
                    value={swapQ}
                    onChange={(event) => setSwapQ(event.target.value)}
                    placeholder="search tag / text…"
                    style={swapSearchInput}
                  />
                </div>
                {neighbors.isPending ? (
                  <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                    <span style={miniSpinner} />{" "}
                    {swapQ.trim() ? "ranking (WD14)…" : "finding (DINOv2)…"}
                  </span>
                ) : (
                  (neighbors.data?.neighbors ?? []).map((n) => (
                    <div
                      key={n.media_id}
                      onClick={() => applySwap(swapId, n.media_id)}
                      style={swapOption}
                      title={`${n.name} · ${n.why} · cosine ${n.cosine}`}
                    >
                      <img src={thumbUrl(n.media_id)} alt="" style={swapThumb} />
                      <span style={swapMeta}>
                        Q {n.quality == null ? "—" : n.quality.toFixed(0)} ·{" "}
                        {n.why}
                      </span>
                    </div>
                  ))
                )}
                {!neighbors.isPending &&
                  !(neighbors.data?.neighbors ?? []).length && (
                    <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                      {swapQ.trim()
                        ? "no eligible candidate matches"
                        : "no neighbour available"}
                    </span>
                  )}
                <div style={{ flex: 1 }} />
                <span
                  onClick={() => {
                    setSwapId(null);
                    setSwapQ("");
                  }}
                  style={clearLink}
                >
                  close
                </span>
              </div>
            )}

            {selId != null && (
              <div style={replaceStrip}>
                <span style={{ fontSize: 10.5, color: colors.accent, fontWeight: 700 }}>
                  ◉ Replace mode
                </span>
                <span style={{ fontSize: 10.5, color: colors.textMuted }}>
                  <span style={{ fontFamily: font.mono, color: colors.text }}>
                    {picks.find((p) => p.media_id === selId)?.name ?? ""}
                  </span>{" "}
                  selected — click a grey point on the coverage map to swap it
                  in, or click the card again to cancel.
                </span>
                <div style={{ flex: 1 }} />
                <span onClick={() => setSelId(null)} style={clearLink}>
                  ✕ cancel
                </span>
              </div>
            )}

            <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
              {view === "grid" ? (
                <PickGrid picks={picks} handlers={cardHandlers} />
              ) : (
                <ClusterView
                  result={result}
                  handlers={cardHandlers}
                  clusterSel={clusterSel}
                  onClusterClick={(id) =>
                    setClusterSel((prev) => (prev === id ? null : id))
                  }
                />
              )}
              {!picks.length && !recomputing && (
                <div style={empty}>
                  {builtRef.current
                    ? "No pick — loosen the quality floor or the subject."
                    : "Set the recipe, then press ⟳ Build selection."}
                </div>
              )}
            </div>

            {recomputing && (
              <div style={recomputeOverlay}>
                <span style={bigSpinner} />
                <span style={{ fontSize: 11.5, color: colors.textMutedAlt }}>
                  {preview.stage
                    ? `${preview.stage.label}…`
                    : "the engine is recomposing the selection…"}
                </span>
                <div style={progressTrack}>
                  <div
                    style={{
                      ...progressFill,
                      width: preview.stage
                        ? `${
                            ((preview.stage.index + 1) / preview.stage.total) *
                            100
                          }%`
                        : "8%",
                    }}
                  />
                </div>
                {preview.stage && (
                  <span style={{ fontSize: 10, color: colors.textFaint }}>
                    step {preview.stage.index + 1} / {preview.stage.total}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* ---- right: composition ---- */}
          <CompositionPanel
            result={result}
            hovered={hovered}
            onHover={setHovered}
            recipe={recipe}
            recomputing={recomputing}
            view={view}
            clusterSel={clusterSel}
            onClusterClick={(id) =>
              setClusterSel((prev) => (prev === id ? null : id))
            }
            selId={selId}
            onReplacePick={replaceFromMap}
            rebal={rebal}
            onToggleRebal={() => {
              setRebal((prev) => !prev);
              setClusterSel(null);
            }}
          />
        </div>

        <div style={footer}>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder={placeholderName}
            style={{ ...input, width: 220 }}
          />
          {result?.dominant_tag && (
            <span style={{ fontSize: 10.5, color: colors.textFaint }}>
              dominant tag: {result.dominant_tag.name} (
              {result.dominant_tag.share}% of picks) — suggested trigger word
            </span>
          )}
          <div style={{ flex: 1 }} />
          <button onClick={cancel} style={ghost}>
            Cancel
          </button>
          <button
            disabled={!flaggedPicks.length}
            onClick={() => setTriageOpen(true)}
            style={{
              ...triageButton,
              opacity: flaggedPicks.length ? 1 : 0.4,
              cursor: flaggedPicks.length ? "pointer" : "default",
            }}
          >
            ⚑ Triage ({flaggedPicks.length})
          </button>
          {isEdit && (
            <button
              disabled={!picks.length || create.isPending || update.isPending}
              onClick={runCreate}
              style={{
                ...ghost,
                opacity: picks.length ? 1 : 0.5,
                cursor: picks.length ? "pointer" : "default",
              }}
            >
              {create.isPending
                ? "Saving…"
                : `＋ Save as new (${picks.length})`}
            </button>
          )}
          <button
            disabled={!picks.length || create.isPending || update.isPending}
            onClick={isEdit ? runOverwrite : runCreate}
            style={{
              ...confirmButton,
              background: picks.length
                ? colors.accent
                : colors.borderControl,
              color: picks.length ? colors.onAccent : colors.textFaint,
              cursor: picks.length ? "pointer" : "default",
            }}
          >
            {isEdit
              ? update.isPending
                ? "Overwriting…"
                : `⤳ Overwrite ${initialName} (${picks.length})`
              : create.isPending
                ? "Creating…"
                : `Create the dataset (${picks.length})`}
          </button>
        </div>
      </div>

      {triageOpen && (
        <TriageOverlay
          picks={flaggedPicks}
          mediaType={mediaType}
          metric={metric || null}
          onDiscard={discard}
          onKeep={keepPick}
          onClose={() => setTriageOpen(false)}
        />
      )}

      {previewId != null &&
        (() => {
          const pick = picks.find((p) => p.media_id === previewId);
          return pick ? (
            <PreviewLightbox
              pick={pick}
              onClose={() => setPreviewId(null)}
            />
          ) : null;
        })()}
    </div>
  );
}

/** A clickable suggested-tag chip (blue), with its match percentage. */
function SuggestedChip({
  tag,
  onClick,
}: {
  tag: AutobuildSuggestedTag;
  onClick: () => void;
}) {
  return (
    <span onClick={onClick} style={suggestChip} title="Lock this tag">
      {tag.name}
      <span style={{ color: colors.textFaint, marginLeft: 4 }}>
        {tag.pct}%
      </span>
    </span>
  );
}

/** The "＋" tile: browse the whole library for an example seed. */
function SeedPicker({
  seeds,
  onAdd,
}: {
  seeds: number[];
  onAdd: (id: number) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <button onClick={() => setOpen((value) => !value)} style={seedAdd}>
        ＋
      </button>
      {open && (
        <SeedPopover
          seeds={seeds}
          onAdd={(id) => {
            onAdd(id);
            setOpen(false);
          }}
        />
      )}
    </div>
  );
}

/** The seed browser popover — only mounted (and querying) while open. */
function SeedPopover({
  seeds,
  onAdd,
}: {
  seeds: number[];
  onAdd: (id: number) => void;
}) {
  const [query, setQuery] = useState("");
  const grid = useLibraryGrid({
    offset: 0,
    limit: 24,
    tag_ids: [],
    exclude_tag_ids: [],
    match: "any",
    favorites_only: false,
    sort: "date_desc",
    quality_metric: "",
  });
  const options = (grid.data?.items ?? [])
    .filter(
      (card) =>
        !card.is_video &&
        !seeds.includes(Number(card.key)) &&
        card.name.toLowerCase().includes(query.trim().toLowerCase()),
    )
    .slice(0, 12);
  return (
    <div style={seedPopover}>
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="filter by name…"
        style={{ ...input, marginBottom: 6 }}
      />
      <div style={seedGrid}>
        {options.map((card) => (
          <img
            key={card.key}
            src={thumbUrl(Number(card.key))}
            alt=""
            title={card.name}
            onClick={() => onAdd(Number(card.key))}
            style={seedThumb}
          />
        ))}
        {!options.length && (
          <span style={{ fontSize: 10, color: colors.textFaint }}>
            {grid.isLoading ? "loading…" : "no image"}
          </span>
        )}
      </div>
    </div>
  );
}

/** The handler bag threaded to every pick card (grid and cluster views). */
interface CardHandlers {
  selId: number | null;
  flagHover: number | null;
  onDiscard: (id: number) => void;
  onSwap: (id: number) => void;
  onSelect: (id: number) => void;
  onZoom: (id: number) => void;
  onKeep: (id: number) => void;
  onFlagEnter: (id: number) => void;
  onFlagLeave: (id: number) => void;
  onEnter: (id: number) => void;
  onLeave: () => void;
}

function PickGrid({
  picks,
  handlers,
}: {
  picks: AutobuildPick[];
  handlers: CardHandlers;
}) {
  return (
    <div style={grid}>
      {picks.map((pick) => (
        <PickCard key={pick.media_id} pick={pick} handlers={handlers} />
      ))}
    </div>
  );
}

function ClusterView({
  result,
  handlers,
  clusterSel,
  onClusterClick,
}: {
  result: AutobuildStudioPreview | null;
  handlers: CardHandlers;
  clusterSel: number | null;
  onClusterClick: (id: number) => void;
}) {
  if (!result) return null;
  const byId = new Map(result.picks.map((pick) => [pick.media_id, pick]));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {result.clusters.map((cluster) => {
        const members = cluster.media_ids
          .map((id) => byId.get(id))
          .filter((pick): pick is AutobuildPick => Boolean(pick));
        if (!members.length) return null;
        const active = clusterSel === cluster.id;
        return (
          <div key={cluster.id}>
            <div
              onClick={() => onClusterClick(cluster.id)}
              title="Highlight this cluster on the coverage map"
              style={{
                ...clusterHead,
                cursor: "pointer",
                border: `1px solid ${active ? cluster.color : "transparent"}`,
                background: active
                  ? "rgba(232,147,90,0.07)"
                  : "transparent",
              }}
            >
              <span
                style={{
                  width: 9,
                  height: 9,
                  borderRadius: 3,
                  background: cluster.color,
                }}
              />
              <span style={{ fontSize: 11.5, fontWeight: 700 }}>
                {cluster.label}
              </span>
              <span style={{ fontSize: 9.5, color: colors.textFaint }}>
                {cluster.count} picks · {cluster.pct}% · click to highlight
              </span>
            </div>
            <div style={grid}>
              {members.map((pick) => (
                <PickCard
                  key={pick.media_id}
                  pick={pick}
                  handlers={handlers}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PickCard({
  pick,
  handlers,
}: {
  pick: AutobuildPick;
  handlers: CardHandlers;
}) {
  const id = pick.media_id;
  const selected = handlers.selId === id;
  const flagOpen = handlers.flagHover === id;
  const stop = (event: React.MouseEvent) => event.stopPropagation();
  return (
    <div
      onClick={() => handlers.onSelect(id)}
      onMouseEnter={() => handlers.onEnter(id)}
      onMouseLeave={handlers.onLeave}
      title="Click to select — then click a grey map point to replace it"
      style={{
        ...pickCard,
        cursor: "pointer",
        border: `1.5px solid ${
          selected
            ? colors.accent
            : pick.flag
              ? "#5a4a22"
              : colors.borderControl
        }`,
        boxShadow: selected ? "0 0 0 2px rgba(232,147,90,0.22)" : "none",
      }}
    >
      <div style={{ position: "relative", aspectRatio: "4 / 3" }}>
        <img src={thumbUrl(id)} alt="" style={thumb} />
        <div style={cardActions}>
          <span
            onClick={(event) => {
              stop(event);
              handlers.onDiscard(id);
            }}
            title="Discard — the engine re-picks the next best"
            style={cardActionButton}
          >
            ✕
          </span>
          <span
            onClick={(event) => {
              stop(event);
              handlers.onSwap(id);
            }}
            title="See close replacements"
            style={cardActionButton}
          >
            ⇄
          </span>
          <span
            onClick={(event) => {
              stop(event);
              handlers.onZoom(id);
            }}
            title="Zoom — open a large preview"
            style={cardActionButton}
          >
            ⌕
          </span>
        </div>
        <span style={{ ...scoreBadge, color: bandColor(pick.quality) }}>
          {pick.quality == null ? "—" : pick.quality.toFixed(0)}
        </span>
        {pick.is_video ? (
          <span style={videoBadge}>▶</span>
        ) : (
          pick.favorite && <span style={favBadge}>♥</span>
        )}
        {pick.flag && (
          <span
            onMouseEnter={() => handlers.onFlagEnter(id)}
            style={flagBadge}
          >
            ⚑ borderline
          </span>
        )}
        {pick.flag && flagOpen && (
          <div
            onMouseLeave={() => handlers.onFlagLeave(id)}
            onClick={stop}
            style={flagPopover}
          >
            <div
              style={{
                fontFamily: font.mono,
                fontSize: 9,
                color: colors.warn,
                fontWeight: 700,
              }}
            >
              ⚑ Borderline pick
            </div>
            <div style={{ fontSize: 8.5, color: colors.textMuted, lineHeight: 1.4 }}>
              The engine kept it, but it barely cleared one of your rules:
            </div>
            <div style={{ fontSize: 9, color: colors.text, lineHeight: 1.45 }}>
              {pick.flag.why}
            </div>
            <div style={{ display: "flex", gap: 5, marginTop: 2 }}>
              <span
                onClick={(event) => {
                  stop(event);
                  handlers.onKeep(id);
                  handlers.onFlagLeave(id);
                }}
                title="Keep it and stop flagging it"
                style={flagKeepButton}
              >
                ✓ Keep
              </span>
              <span
                onClick={(event) => {
                  stop(event);
                  handlers.onDiscard(id);
                }}
                title="Discard — the engine re-picks the next best"
                style={flagReplaceButton}
              >
                ⇄ Replace
              </span>
            </div>
          </div>
        )}
        {selected && (
          <span style={selectedBanner}>◉ selected — click a grey map point</span>
        )}
      </div>
      <div style={metaRow}>
        <span style={metaName}>{pick.name}</span>
      </div>
      <div style={reasonRow}>
        {pick.reasons.map((reason, index) => (
          <span key={index} style={reasonChip} title={reason.title}>
            {reason.icon} {reason.label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** The card ⌕ zoom: a large preview of one pick (z-95). */
function PreviewLightbox({
  pick,
  onClose,
}: {
  pick: AutobuildPick;
  onClose: () => void;
}) {
  return (
    <div
      onClick={(event) => {
        // Stop the click bubbling to the Studio backdrop, which would close
        // the whole Studio — the outside click must close only the zoom.
        event.stopPropagation();
        onClose();
      }}
      style={lightboxBackdrop}
    >
      <div onClick={(event) => event.stopPropagation()} style={lightboxPanel}>
        <img src={fileUrl(pick.media_id)} alt="" style={lightboxImage} />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: font.mono, fontSize: 11 }}>
            {pick.name} · Q{" "}
            {pick.quality == null ? "—" : pick.quality.toFixed(0)}
          </span>
          <div style={{ flex: 1 }} />
          <span onClick={onClose} style={clearLink}>
            close (Esc)
          </span>
        </div>
      </div>
    </div>
  );
}

/**
 * Triage express — walk the borderline picks one by one (z-90). Each shows
 * why it was flagged and the closest visual alternative; the keyboard
 * drives it (E discard, G keep, Escape close). Discard drops the pick (the
 * grid re-picks the next best); Keep marks it no longer borderline.
 */
function TriageOverlay({
  picks,
  mediaType,
  metric,
  onDiscard,
  onKeep,
  onClose,
}: {
  picks: AutobuildPick[];
  mediaType: string;
  metric: string | null;
  onDiscard: (id: number) => void;
  onKeep: (id: number) => void;
  onClose: () => void;
}) {
  // Snapshot the flagged list at open so the walk does not shift under us
  // when a discard triggers a recompute upstream.
  const [queue] = useState(picks);
  const [index, setIndex] = useState(0);
  const neighbors = useAutobuildNeighbors();
  const current = queue[index] ?? null;

  const advance = () => setIndex((value) => value + 1);
  const discard = () => {
    if (current) onDiscard(current.media_id);
    advance();
  };
  const keep = () => {
    if (current) onKeep(current.media_id);
    advance();
  };

  const neighborsRef = useRef(neighbors);
  neighborsRef.current = neighbors;
  useEffect(() => {
    if (current) {
      neighborsRef.current.mutate({
        media_id: current.media_id,
        media_type: mediaType,
        metric,
        exclude_ids: queue.map((pick) => pick.media_id),
      });
    }
  }, [current, mediaType, metric, queue]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      event.stopPropagation();
      if (event.key === "Escape") onClose();
      else if (current && (event.key === "e" || event.key === "E")) discard();
      else if (current && (event.key === "g" || event.key === "G")) keep();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  const replacement = neighbors.data?.neighbors?.[0] ?? null;

  return (
    <div style={triageBackdrop}>
      <div style={triagePanel}>
        {current ? (
          <>
            <div style={triageCounter}>
              {index + 1} / {queue.length}
            </div>
            <img src={fileUrl(current.media_id)} alt="" style={triageImage} />
            <div style={{ fontFamily: font.mono, fontSize: 11 }}>
              {current.name} · Q{" "}
              {current.quality == null ? "—" : current.quality.toFixed(0)}
            </div>
            <div style={triageReason}>
              <div style={{ color: colors.warn, fontWeight: 600 }}>
                ⚑ {current.flag?.kind}
              </div>
              <div style={{ color: colors.textMutedAlt }}>
                {current.flag?.why}
              </div>
            </div>
            <div style={triageReplace}>
              {neighbors.isPending ? (
                <span style={{ color: colors.textFaint }}>
                  <span style={miniSpinner} /> finding a replacement…
                </span>
              ) : replacement ? (
                <>
                  <span style={{ color: colors.textMuted }}>
                    If discarded, closest alternative:
                  </span>
                  <img
                    src={thumbUrl(replacement.media_id)}
                    alt=""
                    style={seedThumb}
                  />
                  <span style={{ fontFamily: font.mono, fontSize: 10 }}>
                    {replacement.name} · Q{" "}
                    {replacement.quality == null
                      ? "—"
                      : replacement.quality.toFixed(0)}
                  </span>
                </>
              ) : (
                <span style={{ color: colors.textFaint }}>
                  the engine re-picks the next best
                </span>
              )}
            </div>
            <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
              <button onClick={discard} style={triageDiscard}>
                Discard (E)
              </button>
              <button onClick={keep} style={triageKeep}>
                Keep (G)
              </button>
            </div>
          </>
        ) : (
          <div style={{ textAlign: "center", padding: 30 }}>
            <div style={{ fontSize: 34, color: colors.ok }}>✓</div>
            <div style={{ fontSize: 13, marginTop: 8 }}>
              Triage complete — {queue.length} pick(s) reviewed.
            </div>
            <button
              onClick={onClose}
              style={{ ...triageKeep, marginTop: 16 }}
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

const fileUrl = (id: number) => `/api/media/${id}/file`;

function CompositionPanel({
  result,
  hovered,
  onHover,
  recipe,
  recomputing,
  view,
  clusterSel,
  onClusterClick,
  selId,
  onReplacePick,
  rebal,
  onToggleRebal,
}: {
  result: AutobuildStudioPreview | null;
  hovered: number | null;
  onHover: (id: number | null) => void;
  recipe: AutobuildRecipe;
  recomputing: boolean;
  view: "grid" | "clusters";
  clusterSel: number | null;
  onClusterClick: (id: number) => void;
  selId: number | null;
  onReplacePick: (inId: number) => void;
  rebal: boolean;
  onToggleRebal: () => void;
}) {
  if (!result) {
    return <div style={{ ...sidePanel, color: colors.textFaint }}>…</div>;
  }
  const { size, pillars } = result;
  const clusterColors = result.clusters.map((cluster) => cluster.color);
  const dominant = result.clusters[0] ?? null;
  return (
    <div style={{ ...sidePanel, opacity: recomputing ? 0.55 : 1 }}>
      <div style={gradeCard}>
        <div
          style={{
            fontSize: 29,
            fontWeight: 700,
            fontFamily: font.mono,
            color: gradeColor(result.score),
          }}
        >
          {result.grade}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10.5, color: colors.textMuted }}>
            Projected dataset score
          </div>
          <div style={{ fontFamily: font.mono, fontSize: 12.5 }}>
            {result.score == null ? "—" : result.score.toFixed(1)}
          </div>
        </div>
      </div>

      <div>
        <div style={{ display: "flex" }}>
          <span style={{ fontSize: 11, color: colors.textMuted, flex: 1 }}>
            Size
          </span>
          <span
            style={{
              fontFamily: font.mono,
              fontSize: 10.5,
              color: size?.over ? colors.warn : colors.ok,
            }}
          >
            {size?.total ?? 0} · target {size?.min ?? 0}–{size?.max ?? 0}
          </span>
        </div>
        <Bar percent={size?.percent ?? 0} color={colors.ok} height={5} />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        <Pillar
          label="Quality"
          percent={pillars.quality ?? 0}
          value={pillars.quality == null ? "—" : pillars.quality.toFixed(1)}
          color={bandColor(pillars.quality)}
        />
        <Pillar
          label="Diversity"
          percent={pillars.diversity ?? 0}
          value={
            pillars.diversity == null ? "—" : pillars.diversity.toFixed(0)
          }
          color={bandColor(pillars.diversity)}
        />
        <Pillar
          label="Duplicates"
          percent={pillars.hygiene}
          value={pillars.duplicates ? `${pillars.duplicates} ⚠` : "0 ✓"}
          color={pillars.duplicates ? colors.danger : colors.ok}
        />
      </div>

      {result.clusters.length > 0 && (
        <div style={dividedSection}>
          <div style={{ display: "flex", alignItems: "baseline" }}>
            <span style={{ ...sectionLabel, flex: 1 }}>Clusters — balance</span>
            <span style={legendMono}>
              {result.clusters.length} clusters · k auto
            </span>
          </div>
          <div style={balanceBar}>
            {result.clusters.map((cluster) => (
              <div
                key={cluster.id}
                title={`${cluster.label} — ${cluster.pct}%`}
                style={{
                  width: `${Math.max(2, cluster.pct)}%`,
                  background: cluster.color,
                  transition: "width 0.3s",
                }}
              />
            ))}
          </div>
          {result.clusters.map((cluster) => (
            <div
              key={cluster.id}
              onClick={() => onClusterClick(cluster.id)}
              title="Highlight on the map"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 7,
                cursor: "pointer",
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 3,
                  background: cluster.color,
                  flex: "none",
                }}
              />
              <span
                style={{
                  flex: 1,
                  fontFamily: font.mono,
                  fontSize: 9.5,
                  color:
                    clusterSel === cluster.id
                      ? colors.text
                      : colors.textMuted,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  minWidth: 0,
                }}
              >
                {cluster.label}
              </span>
              <span style={{ ...legendMono, flex: "none" }}>
                {cluster.count} · {cluster.pct}%
              </span>
            </div>
          ))}
          <button onClick={onToggleRebal} style={rebalanceButton(rebal)}>
            {rebal ? "✓ rebalanced — undo" : "⟲ Rebalance the selection"}
          </button>
          {dominant && (
            <div style={{ fontSize: 9, color: colors.textFaint, lineHeight: 1.45 }}>
              {rebal
                ? `Re-picked with a penalty on over-dense areas — dominant cluster is now "${dominant.label}" (${dominant.pct}%).`
                : `Dominant cluster: "${dominant.label}" (${dominant.pct}%). Rebalance re-picks while penalising over-represented areas.`}
            </div>
          )}
        </div>
      )}

      <div style={dividedSection}>
        <div style={sectionLabel}>Framing</div>
        {result.framing.map((row) => (
          <div key={row.bucket} style={framingRow}>
            <span
              style={{
                width: 70,
                fontSize: 10.5,
                color: row.under ? colors.warn : colors.textMuted,
              }}
            >
              {row.bucket.replace(/_/g, " ")}
            </span>
            <div style={framingTrack}>
              <div
                style={{
                  width: `${row.share}%`,
                  height: "100%",
                  background: colors.accent,
                  transition: "width 0.3s",
                }}
              />
            </div>
            <span style={framingCount}>
              {row.total} · {row.share.toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      <div style={mapCard}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <span style={{ ...sectionLabel, flex: 1 }}>Coverage</span>
          <span style={legendMono}>DINOv2 · 2D</span>
        </div>
        <AutoBuildCoverageMap
          map={result.map}
          hovered={hovered}
          onHover={onHover}
          clusterColors={clusterColors}
          clusterHighlight={clusterSel}
          colorByCluster={view === "clusters" || clusterSel != null}
          selectedPick={selId}
          onReplacePick={onReplacePick}
        />
        <div style={{ display: "flex", gap: 10, ...legendMono }}>
          <span>
            <span style={{ color: colors.accent }}>●</span> pick
          </span>
          <span>
            <span style={{ color: "#3f434e" }}>●</span> candidate
          </span>
          <span>
            <span style={{ color: colors.info }}>●</span> seed
          </span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {result.advice.map((row, index) => (
          <div key={index} style={adviceCard}>
            {row.text}
          </div>
        ))}
      </div>

      <div style={{ ...recipeCard }}>
        <div style={{ ...sectionLabel, color: colors.info }}>Recipe</div>
        <RecipeLine k="type" v={recipe.media_type} />
        <RecipeLine k="subject" v={recipe.semantic_q || "—"} />
        <RecipeLine
          k="tags"
          v={recipe.locked_tags.join(", ") || "—"}
        />
        <RecipeLine k="seeds" v={String(recipe.seed_media_ids.length)} />
        <RecipeLine k="size" v={String(recipe.size)} />
        <RecipeLine
          k="quality"
          v={`${recipe.metric || "avg"} ≥ ${recipe.min_score}`}
        />
        <RecipeLine k="framing" v={recipe.framing_preset} />
        <div
          style={{
            fontSize: 10,
            color: recipe.live ? colors.info : colors.textFaint,
            marginTop: 4,
          }}
        >
          ◇ living dataset {recipe.live ? "active" : "off"}
        </div>
      </div>
    </div>
  );
}

function RecipeLine({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: "flex", gap: 8, fontFamily: font.mono }}>
      <span style={{ width: 54, fontSize: 10, color: colors.textFaint }}>
        {k}
      </span>
      <span
        style={{
          flex: 1,
          fontSize: 10,
          color: colors.textMutedAlt,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {v}
      </span>
    </div>
  );
}

function Pillar({
  label,
  percent,
  value,
  color,
}: {
  label: string;
  percent: number;
  value: string;
  color: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ width: 64, fontSize: 11, color: colors.textMuted }}>
        {label}
      </span>
      <div style={{ flex: 1 }}>
        <Bar percent={percent} color={color} height={4} />
      </div>
      <span
        style={{
          width: 44,
          textAlign: "right",
          fontFamily: font.mono,
          fontSize: 10.5,
          color,
        }}
      >
        {value}
      </span>
    </div>
  );
}

function Bar({
  percent,
  color,
  height,
}: {
  percent: number;
  color: string;
  height: number;
}) {
  return (
    <div
      style={{
        height,
        borderRadius: height / 2,
        background: colors.border,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.max(0, Math.min(100, percent))}%`,
          height: "100%",
          background: color,
          transition: "width 0.3s",
        }}
      />
    </div>
  );
}

function Section({
  label,
  divided,
  children,
}: {
  label: string;
  divided?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        ...(divided
          ? { borderTop: `1px solid ${colors.border}`, paddingTop: 12 }
          : {}),
      }}
    >
      <div style={sectionLabel}>{label}</div>
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
    <label style={checkLabel}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      {label}
    </label>
  );
}

function Seg({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div style={segmented}>
      {options.map((option) => (
        <span
          key={option.value}
          onClick={() => onChange(option.value)}
          style={{
            padding: "3px 9px",
            fontSize: 10.5,
            cursor: "pointer",
            fontFamily: font.mono,
            color: value === option.value ? colors.text : colors.textMuted,
            background:
              value === option.value ? colors.borderControl : "transparent",
          }}
        >
          {option.label}
        </span>
      ))}
    </div>
  );
}

function presetButton(active: boolean) {
  return {
    padding: "5px 10px",
    border: `1px solid ${active ? colors.accent : colors.borderControl}`,
    borderRadius: radii.control,
    background: active ? "rgba(232,147,90,0.12)" : colors.card,
    color: active ? colors.accent : colors.textMutedAlt,
    fontSize: 11,
    fontFamily: font.mono,
    cursor: "pointer",
  } as const;
}

const backdrop = {
  position: "fixed",
  inset: 0,
  zIndex: 65,
  background: "rgba(10,11,14,0.7)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 16,
} as const;

const swapStrip = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "7px 12px",
  borderBottom: `1px solid ${colors.border}`,
  background: colors.card,
} as const;

const swapOption = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 2,
  cursor: "pointer",
} as const;

const swapThumb = {
  width: 44,
  height: 33,
  objectFit: "cover",
  borderRadius: 4,
  border: `1px solid ${colors.borderControl}`,
} as const;

const swapMeta = {
  fontFamily: font.mono,
  fontSize: 8.5,
  color: colors.textMuted,
} as const;

const triageButton = {
  padding: "8px 14px",
  border: `1px solid ${colors.warn}`,
  borderRadius: 7,
  background: "transparent",
  color: colors.warn,
  fontSize: 12,
  fontWeight: 600,
} as const;

const triageBackdrop = {
  position: "fixed",
  inset: 0,
  zIndex: 90,
  background: "rgba(10,11,14,0.82)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 16,
} as const;

const triagePanel = {
  width: 420,
  maxWidth: "94%",
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: radii.modal,
  boxShadow: "0 24px 80px rgba(0,0,0,0.7)",
  padding: 18,
  display: "flex",
  flexDirection: "column",
  gap: 10,
  alignItems: "center",
} as const;

const triageCounter = {
  fontFamily: font.mono,
  fontSize: 10.5,
  color: colors.textFaint,
} as const;

const triageImage = {
  width: "100%",
  maxHeight: 300,
  objectFit: "contain",
  borderRadius: 8,
  background: colors.app,
} as const;

const triageReason = {
  width: "100%",
  display: "flex",
  flexDirection: "column",
  gap: 3,
  padding: "8px 10px",
  borderRadius: 7,
  background: "#241715",
  border: "1px solid #3a2622",
  fontSize: 11,
  lineHeight: 1.4,
} as const;

const triageReplace = {
  width: "100%",
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: 10.5,
} as const;

const triageDiscard = {
  flex: 1,
  padding: "9px 0",
  border: "1px solid #3a2622",
  borderRadius: 7,
  background: "#241715",
  color: colors.danger,
  fontSize: 12,
  fontWeight: 600,
  cursor: "pointer",
} as const;

const triageKeep = {
  flex: 1,
  padding: "9px 0",
  border: "none",
  borderRadius: 7,
  background: colors.accent,
  color: colors.onAccent,
  fontSize: 12,
  fontWeight: 700,
  cursor: "pointer",
} as const;

const panel = {
  width: "100%",
  maxWidth: 1560,
  height: "100%",
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: radii.modal,
  boxShadow: "0 24px 80px rgba(0,0,0,0.65)",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
} as const;

const header = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "12px 16px",
  borderBottom: `1px solid ${colors.border}`,
} as const;

const headerSub = {
  fontSize: 10.5,
  color: colors.textFaint,
  fontFamily: font.mono,
} as const;

const closeX = {
  cursor: "pointer",
  color: colors.textMuted,
  fontSize: 15,
  padding: "4px 8px",
} as const;

const rail = {
  width: 262,
  flex: "none",
  borderRight: `1px solid ${colors.border}`,
  background: colors.toolbar,
  padding: 12,
  display: "flex",
  flexDirection: "column",
  gap: 14,
  overflowY: "auto",
} as const;

const sectionLabel = {
  fontSize: 10,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: colors.textMuted,
  fontWeight: 600,
} as const;

const inlineRow = { display: "flex", alignItems: "center", gap: 8 } as const;

const checkLabel = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  fontSize: 11.5,
  color: colors.textMutedAlt,
  cursor: "pointer",
  lineHeight: 1.4,
} as const;

const segmented = {
  display: "flex",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.control,
  overflow: "hidden",
} as const;

const fieldLabel = { fontSize: 11, color: colors.textMuted, width: 28 } as const;

const input = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: radii.control,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 11.5,
  fontFamily: font.sans,
} as const;

const select = {
  appearance: "none",
  width: "100%",
  background: colors.input,
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.control,
  color: colors.textSecondary,
  padding: "6px 9px",
  fontSize: 11.5,
  cursor: "pointer",
} as const;

const searchBox = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  padding: "6px 9px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: colors.input,
} as const;

const searchInput = {
  flex: 1,
  minWidth: 0,
  background: "transparent",
  border: "none",
  outline: "none",
  color: colors.text,
  fontSize: 11.5,
} as const;

const suggestChip = {
  fontSize: 10,
  padding: "3px 7px",
  borderRadius: 10,
  background: "#15202c",
  border: "1px solid #2f4860",
  color: colors.info,
  cursor: "pointer",
} as const;

const pulseChip = {
  width: 46,
  height: 18,
  borderRadius: 10,
  background: colors.border,
  animation: "cfpulse 1.1s ease-in-out infinite",
} as const;

const lockedChip = {
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  fontSize: 10,
  padding: "3px 7px",
  borderRadius: 10,
  background: "rgba(232,147,90,0.12)",
  border: `1px solid ${colors.accent}`,
  color: colors.accent,
} as const;

const excludeChip = {
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  fontSize: 10,
  padding: "3px 7px",
  borderRadius: 10,
  background: "#241715",
  border: `1px solid ${colors.danger}`,
  color: colors.danger,
} as const;

const chipX = { cursor: "pointer", fontSize: 9, opacity: 0.8 } as const;

const seedThumb = {
  width: 46,
  height: 34,
  objectFit: "cover",
  borderRadius: 4,
  cursor: "pointer",
  border: `1px solid ${colors.borderControl}`,
} as const;

const seedRemove = {
  position: "absolute",
  top: -5,
  right: -5,
  width: 15,
  height: 15,
  borderRadius: "50%",
  background: colors.danger,
  color: "#fff",
  fontSize: 9,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
} as const;

const seedAdd = {
  width: 46,
  height: 34,
  borderRadius: 4,
  border: `1px dashed ${colors.borderControl}`,
  background: "transparent",
  color: colors.textMuted,
  cursor: "pointer",
  fontSize: 14,
} as const;

const seedPopover = {
  position: "absolute",
  bottom: 40,
  left: 0,
  zIndex: 5,
  width: 220,
  padding: 8,
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: 8,
  boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
} as const;

const seedGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: 4,
  marginTop: 4,
} as const;

const indexNote = {
  fontSize: 10.5,
  color: colors.warn,
  lineHeight: 1.4,
  borderTop: `1px solid ${colors.border}`,
  paddingTop: 12,
} as const;

const toolbar = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 9,
  padding: "9px 12px",
  borderBottom: `1px solid ${colors.border}`,
  background: colors.toolbar,
} as const;

const counterAccent = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.accent,
} as const;

const grid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
  gap: 9,
} as const;

const pickCard = {
  position: "relative",
  borderRadius: radii.card,
  overflow: "hidden",
  background: colors.card,
} as const;

const thumb = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover",
} as const;

const cardActions = {
  position: "absolute",
  top: 6,
  left: 6,
  display: "flex",
  gap: 4,
} as const;

const cardActionButton = {
  width: 20,
  height: 20,
  borderRadius: 5,
  background: "rgba(15,16,19,0.72)",
  color: "#d5d7dc",
  fontSize: 11,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
} as const;

const scoreBadge = {
  position: "absolute",
  top: 6,
  right: 6,
  fontFamily: font.mono,
  fontSize: 9.5,
  fontWeight: 600,
  padding: "2px 5px",
  borderRadius: 4,
  background: "rgba(15,16,19,0.78)",
} as const;

const favBadge = {
  position: "absolute",
  bottom: 6,
  left: 6,
  fontSize: 10,
  color: colors.accent,
  textShadow: "0 1px 2px rgba(0,0,0,0.7)",
} as const;

const videoBadge = {
  position: "absolute",
  bottom: 6,
  left: 6,
  fontSize: 9,
  fontFamily: font.mono,
  padding: "2px 5px",
  borderRadius: 4,
  background: "rgba(15,16,19,0.75)",
  color: colors.info,
} as const;

const flagBadge = {
  position: "absolute",
  bottom: 6,
  right: 6,
  fontSize: 8.5,
  fontFamily: font.mono,
  padding: "2px 5px",
  borderRadius: 4,
  background: "rgba(40,30,12,0.9)",
  border: "1px solid #6b5626",
  color: colors.warn,
} as const;

const metaRow = { padding: "5px 7px 2px" } as const;

const metaName = {
  fontFamily: font.mono,
  fontSize: 9,
  color: colors.textMuted,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
  display: "block",
} as const;

const reasonRow = {
  display: "flex",
  flexWrap: "wrap",
  gap: 3,
  padding: "0 7px 7px",
} as const;

const reasonChip = {
  fontSize: 8.5,
  fontFamily: font.mono,
  padding: "1px 4px",
  borderRadius: 3,
  background: colors.toolbar,
  border: `1px solid ${colors.border}`,
  color: colors.textMutedAlt,
} as const;

const clusterHead = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  marginBottom: 7,
} as const;

const empty = {
  padding: 40,
  textAlign: "center",
  color: colors.textFaint,
  fontSize: 12,
} as const;

const recomputeOverlay = {
  position: "absolute",
  inset: 0,
  background: "rgba(19,20,24,0.45)",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  gap: 10,
  pointerEvents: "none",
} as const;

const bigSpinner = {
  width: 22,
  height: 22,
  borderRadius: "50%",
  border: `2px solid ${colors.borderHover}`,
  borderTopColor: colors.accent,
  animation: "cfspin 0.7s linear infinite",
} as const;

const progressTrack = {
  width: 180,
  height: 4,
  borderRadius: 999,
  background: colors.borderHover,
  overflow: "hidden",
} as const;

const progressFill = {
  height: "100%",
  background: colors.accent,
  borderRadius: "inherit",
  transition: "width 0.25s ease",
} as const;

const miniSpinner = {
  display: "inline-block",
  width: 10,
  height: 10,
  borderRadius: "50%",
  border: `2px solid ${colors.borderHover}`,
  borderTopColor: colors.accent,
  animation: "cfspin 0.7s linear infinite",
  marginRight: 4,
  verticalAlign: "middle",
} as const;

const sidePanel = {
  width: 308,
  flex: "none",
  borderLeft: `1px solid ${colors.border}`,
  background: colors.toolbar,
  padding: 12,
  display: "flex",
  flexDirection: "column",
  gap: 12,
  overflowY: "auto",
  transition: "opacity 0.25s",
} as const;

const gradeCard = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "10px 12px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.card,
  background: colors.card,
} as const;

const dividedSection = {
  display: "flex",
  flexDirection: "column",
  gap: 6,
  borderTop: `1px solid ${colors.border}`,
  paddingTop: 11,
} as const;

const framingRow = { display: "flex", alignItems: "center", gap: 8 } as const;

const framingTrack = {
  flex: 1,
  height: 9,
  background: colors.border,
  borderRadius: 3,
  overflow: "hidden",
} as const;

const framingCount = {
  width: 62,
  textAlign: "right",
  fontFamily: font.mono,
  fontSize: 9.5,
  color: colors.textMuted,
} as const;

const legendMono = {
  fontSize: 9.5,
  color: colors.textFaint,
  fontFamily: font.mono,
} as const;

const mapCard = {
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.card,
  background: colors.app,
  padding: 9,
  display: "flex",
  flexDirection: "column",
  gap: 6,
} as const;

const adviceCard = {
  padding: "7px 9px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: colors.card,
  fontSize: 10.5,
  lineHeight: 1.45,
  color: colors.textMutedAlt,
} as const;

const recipeCard = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  padding: "10px 11px",
  borderRadius: radii.card,
  background: "#15202c",
  border: "1px solid #2f4860",
} as const;

const clearLink = {
  fontSize: 10,
  color: colors.textFaint,
  cursor: "pointer",
} as const;

const footer = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "11px 16px",
  borderTop: `1px solid ${colors.border}`,
  background: colors.toolbar,
} as const;

const ghost = {
  padding: "8px 14px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: "transparent",
  color: colors.textMutedAlt,
  fontSize: 12,
  cursor: "pointer",
} as const;

const confirmButton = {
  padding: "8px 18px",
  border: "none",
  borderRadius: 7,
  fontSize: 12,
  fontWeight: 700,
} as const;

const buildButton = {
  position: "sticky",
  bottom: 0,
  padding: "10px 14px",
  border: "none",
  borderRadius: 7,
  background: colors.accent,
  color: colors.onAccent,
  fontSize: 12.5,
  fontWeight: 700,
} as const;

const dirtyNote = {
  fontSize: 10,
  color: colors.warn,
  lineHeight: 1.4,
  padding: "4px 2px",
} as const;

const prefLine = {
  fontFamily: font.mono,
  fontSize: 10,
  color: colors.ok,
  lineHeight: 1.4,
} as const;

const swapSearch = {
  display: "flex",
  alignItems: "center",
  gap: 5,
  padding: "3px 7px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 6,
  background: colors.input,
} as const;

const swapSearchInput = {
  width: 120,
  background: "transparent",
  border: "none",
  outline: "none",
  color: colors.text,
  fontSize: 10.5,
} as const;

const replaceStrip = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 12px",
  borderBottom: "1px solid #3a2d1d",
  background: "#241d12",
} as const;

const flagPopover = {
  position: "absolute",
  inset: 0,
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  gap: 4,
  background: "rgba(15,16,19,0.92)",
  padding: 8,
  cursor: "default",
} as const;

const flagKeepButton = {
  flex: 1,
  textAlign: "center",
  padding: "4px 0",
  borderRadius: 5,
  background: colors.card,
  border: "1px solid #2a4a2e",
  color: colors.ok,
  fontSize: 9,
  fontWeight: 700,
  cursor: "pointer",
} as const;

const flagReplaceButton = {
  flex: 1,
  textAlign: "center",
  padding: "4px 0",
  borderRadius: 5,
  background: "#241715",
  border: "1px solid #3a2622",
  color: colors.danger,
  fontSize: 9,
  fontWeight: 700,
  cursor: "pointer",
} as const;

const selectedBanner = {
  position: "absolute",
  left: 0,
  right: 0,
  bottom: 0,
  textAlign: "center",
  background: "rgba(232,147,90,0.92)",
  color: colors.onAccent,
  fontSize: 8.5,
  fontWeight: 700,
  fontFamily: font.mono,
  padding: "2px 4px",
} as const;

const balanceBar = {
  display: "flex",
  height: 7,
  borderRadius: 4,
  overflow: "hidden",
  gap: 1,
  background: colors.app,
} as const;

const lightboxBackdrop = {
  position: "fixed",
  inset: 0,
  zIndex: 95,
  background: "rgba(10,11,14,0.85)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 24,
} as const;

const lightboxPanel = {
  maxWidth: "90%",
  maxHeight: "90%",
  display: "flex",
  flexDirection: "column",
  gap: 8,
} as const;

const lightboxImage = {
  maxWidth: "100%",
  maxHeight: "80vh",
  objectFit: "contain",
  borderRadius: 8,
  background: colors.app,
} as const;

function rebalanceButton(active: boolean) {
  return {
    padding: "5px 0",
    border: `1px solid ${colors.borderControl}`,
    borderRadius: 6,
    background: colors.card,
    color: active ? colors.ok : colors.accent,
    fontSize: 10,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: font.mono,
  } as const;
}
