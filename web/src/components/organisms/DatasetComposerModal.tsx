/**
 * The dataset composer: pick library media *against* the dataset.
 *
 * Three columns. The left rail filters the corpus (tags, quality floor,
 * resolution, blur/noise, source); the centre ranks the survivors by
 * quality, by how much visual ground they add, or by name, and flags the
 * ones that duplicate what is already in; the right panel projects the
 * dataset's grade, size, pillars, framing split and coverage map for the
 * current selection, and says what is wrong with it.
 *
 * Nothing is computed here: every number comes from
 * ``GET /datasets/{id}/candidates`` and ``POST /datasets/{id}/compose/
 * preview`` (see :mod:`src.dataset_compose`). The selection is debounced
 * before it reaches either, so dragging through a dozen cards costs one
 * recomputation, not twelve.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useAddDatasetMedia,
  useComposePreview,
  useDatasetCandidates,
  useReleaseComposeModel,
} from "../../api/hooks";
import type { ComposeCandidate, ComposePreview } from "../../api/types";
import { colors, font, radii } from "../../design/tokens";
import { CoverageMap } from "../molecules/CoverageMap";
import { TagFilter } from "../molecules/TagFilter";
import type { SelectedTag } from "../molecules/TagFilter";
import { CompareOverlay } from "./CompareOverlay";

const PAGE = 60;
const DEBOUNCE_MS = 300;

const DANGER_BG = "#241715";
const DANGER_BORDER = "#3a2622";
const DATASET_GREY = "#3f434e";

const SORTS = [
  { value: "quality", label: "Quality" },
  { value: "gain", label: "Diversity gain" },
  { value: "name", label: "Name" },
];
const MATCHES = [
  { value: "all", label: "All" },
  { value: "any", label: "Any" },
];
const TYPES = [
  { value: "all", label: "All" },
  { value: "img", label: "IMG" },
  { value: "vid", label: "VID" },
];
const RESOLUTIONS = [
  { value: 0, label: "Any" },
  { value: 1024, label: "≥ 1024 px" },
  { value: 1536, label: "≥ 1536 px" },
  { value: 2048, label: "≥ 2048 px" },
];

/** Quality band colour of the composer (its own, tighter bands). */
function bandColor(score: number | null | undefined): string {
  if (score == null) return colors.textFaint;
  if (score >= 85) return colors.ok;
  if (score >= 70) return colors.warn;
  return colors.danger;
}

function spreadColor(value: number | null): string {
  if (value == null) return colors.textFaint;
  if (value >= 70) return colors.ok;
  if (value >= 50) return colors.warn;
  return colors.danger;
}

function gradeColor(score: number | null): string {
  if (score == null) return colors.textFaint;
  if (score >= 83) return colors.ok;
  if (score >= 65) return colors.warn;
  return colors.danger;
}

const TONE_ICON: Record<string, { icon: string; color: string }> = {
  danger: { icon: "⚠", color: colors.danger },
  warn: { icon: "▾", color: colors.warn },
  info: { icon: "◈", color: colors.info },
  ok: { icon: "✓", color: colors.ok },
};

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

const thumbUrl = (id: number) => `/api/media/${id}/thumb`;
const fileUrl = (id: number) => `/api/media/${id}/file`;

export function DatasetComposerModal({
  datasetId,
  datasetName,
  onClose,
}: {
  datasetId: number;
  datasetName: string;
  onClose: () => void;
}) {
  const [page, setPage] = useState(1);
  const [include, setInclude] = useState<SelectedTag[]>([]);
  const [exclude, setExclude] = useState<SelectedTag[]>([]);
  const [match, setMatch] = useState("all");
  const [metric, setMetric] = useState("");
  const [minScore, setMinScore] = useState(60);
  const [minSide, setMinSide] = useState(0);
  const [excludeBlur, setExcludeBlur] = useState(true);
  const [excludeNoise, setExcludeNoise] = useState(false);
  const [favOnly, setFavOnly] = useState(false);
  const [mediaType, setMediaType] = useState("all");
  const [hideNearDups, setHideNearDups] = useState(false);
  const [sort, setSort] = useState("quality");
  const [search, setSearch] = useState("");
  const [gapsMode, setGapsMode] = useState(false);
  const [similarMode, setSimilarMode] = useState(false);
  const [picked, setPicked] = useState<number[]>([]);
  const [hovered, setHovered] = useState<number | null>(null);
  const [compare, setCompare] = useState<ComposeCandidate | null>(null);

  const add = useAddDatasetMedia();
  const release = useReleaseComposeModel();
  const releaseRef = useRef(release);
  releaseRef.current = release;

  const debouncedSearch = useDebounced(search, DEBOUNCE_MS);
  const debouncedPicked = useDebounced(picked, DEBOUNCE_MS);

  // Reset the pager whenever the filtered set changes underneath it.
  useEffect(() => {
    setPage(1);
  }, [
    include,
    exclude,
    match,
    metric,
    minScore,
    minSide,
    excludeBlur,
    excludeNoise,
    favOnly,
    mediaType,
    hideNearDups,
    sort,
    debouncedSearch,
    gapsMode,
    similarMode,
  ]);

  // The semantic search keeps SigLIP resident between keystrokes; hand the
  // VRAM back when the composer goes away.
  useEffect(() => () => void releaseRef.current.mutate(), []);

  const candidates = useDatasetCandidates(
    datasetId,
    {
      offset: (page - 1) * PAGE,
      limit: PAGE,
      favorites_only: favOnly,
      tag_ids: include.map((tag) => tag.id),
      exclude_tag_ids: exclude.map((tag) => tag.id),
      match,
      metric: metric || undefined,
      min_score: minScore,
      min_side: minSide,
      exclude_blur: excludeBlur,
      exclude_noise: excludeNoise,
      media_type: mediaType,
      hide_near_dups: hideNearDups,
      gaps_only: gapsMode,
      similar_to_selection: similarMode,
      sort,
      semantic_q: debouncedSearch,
      selected_ids: debouncedPicked,
    },
    true,
  );

  const preview = useComposePreview(
    datasetId,
    debouncedPicked,
    metric || "",
    true,
  );

  const data = candidates.data;
  const metrics = data?.metrics ?? [];
  const effectiveMetric = metric || data?.items[0]?.metric || "";
  const items = useMemo(() => data?.items ?? [], [data]);
  const byId = useMemo(
    () => new Map(items.map((card) => [Number(card.key), card])),
    [items],
  );
  const hoveredPoint = hovered != null ? byId.get(hovered)?.xy ?? null : null;

  const toggle = (id: number) =>
    setPicked((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );

  const confirm = () => {
    if (picked.length === 0) return;
    add.mutate({ id: datasetId, media_ids: picked }, { onSuccess: onClose });
  };

  // Escape dismisses the topmost overlay: the comparator owns the key while
  // it is up, and only then does it reach the composer.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !compare) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, compare]);

  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE));
  const semanticOff = data ? !data.semantic_available : false;

  return (
    <div onClick={onClose} style={backdrop}>
      <div onClick={(event) => event.stopPropagation()} style={panel}>
        <div style={header}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700 }}>
              Compose the dataset — {datasetName}
            </div>
            <div style={headerSub}>
              {(data?.libraries ?? []).join(" + ") || "no library"} · quality
              &amp; embeddings indexed · linked, never copied
            </div>
          </div>
          <div style={{ flex: 1 }} />
          <span onClick={onClose} style={closeX}>
            ✕
          </span>
        </div>

        <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
          <div style={rail}>
            <Section label="Scope — tags">
              <TagFilter
                label="+ include tag"
                selected={include}
                onAdd={(tag) => setInclude((prev) => [...prev, tag])}
                onRemove={(id) =>
                  setInclude((prev) => prev.filter((tag) => tag.id !== id))
                }
              />
              <div style={inlineRow}>
                <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                  Match
                </span>
                <Seg value={match} onChange={setMatch} options={MATCHES} />
              </div>
              <div style={sectionLabel}>Exclude</div>
              <TagFilter
                label="− exclude tag"
                selected={exclude}
                onAdd={(tag) => setExclude((prev) => [...prev, tag])}
                onRemove={(id) =>
                  setExclude((prev) => prev.filter((tag) => tag.id !== id))
                }
              />
            </Section>

            <Section label="Quality thresholds" divided>
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
                <span style={fieldLabel}>Min score</span>
                <input
                  type="range"
                  min={0}
                  max={95}
                  step={5}
                  value={minScore}
                  onChange={(event) => setMinScore(Number(event.target.value))}
                  style={{ flex: 1 }}
                />
                <span
                  style={{
                    fontFamily: font.mono,
                    fontSize: 11,
                    width: 26,
                    textAlign: "right",
                    color:
                      minScore >= 70
                        ? colors.ok
                        : minScore >= 40
                          ? colors.warn
                          : colors.textMuted,
                  }}
                >
                  {minScore}
                </span>
              </div>
              <div style={inlineRow}>
                <span style={fieldLabel}>Resolution</span>
                <select
                  value={minSide}
                  onChange={(event) => setMinSide(Number(event.target.value))}
                  style={{ ...select, flex: 1 }}
                >
                  {RESOLUTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <Check
                checked={excludeBlur}
                onChange={setExcludeBlur}
                label="Exclude detected blur"
              />
              <Check
                checked={excludeNoise}
                onChange={setExcludeNoise}
                label="Exclude heavy noise"
              />
            </Section>

            <Section label="Source" divided>
              <Check
                checked={favOnly}
                onChange={setFavOnly}
                label="♥ Favorites only"
              />
              <div style={inlineRow}>
                <span style={{ fontSize: 11, color: colors.textMuted }}>
                  Type
                </span>
                <Seg
                  value={mediaType}
                  onChange={setMediaType}
                  options={TYPES}
                />
              </div>
              <Check
                checked={hideNearDups}
                onChange={setHideNearDups}
                label="Hide near-duplicates of the selection"
              />
            </Section>
          </div>

          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
            <div style={toolbar}>
              <div style={searchBox}>
                <span style={{ fontSize: 11, color: colors.textFaint }}>⌕</span>
                <input
                  type="text"
                  value={search}
                  disabled={semanticOff}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder={
                    semanticOff
                      ? "Semantic search — run the SigLIP index step"
                      : "Semantic search (SigLIP2)…"
                  }
                  style={searchInput}
                />
              </div>
              <span style={{ fontSize: 11, color: colors.textMuted }}>
                Sort
              </span>
              <Seg value={sort} onChange={setSort} options={SORTS} />
              <div style={{ flex: 1 }} />
              <span style={counter}>
                {total} / {data?.pool ?? 0} candidates
              </span>
            </div>

            <div style={suggestBar}>
              <button
                onClick={() => {
                  setGapsMode((on) => !on);
                  setSimilarMode(false);
                }}
                style={suggestButton(gapsMode, colors.info, true)}
              >
                ◈ Fill the gaps ({data?.gap_count ?? 0})
              </button>
              <button
                onClick={() => {
                  if (!picked.length && !similarMode) return;
                  setSimilarMode((on) => !on);
                  setGapsMode(false);
                }}
                style={suggestButton(
                  similarMode,
                  colors.accent,
                  picked.length > 0,
                )}
              >
                ≈ Similar to the selection
              </button>
              <span style={{ fontSize: 10.5, color: colors.textFaint }}>
                {gapsMode
                  ? "Candidates sitting in corpus regions with no dataset image."
                  : similarMode
                    ? "Candidates visually close to your selection (embeddings)."
                    : picked.length
                      ? ""
                      : 'Select images to enable "Similar".'}
              </span>
            </div>

            <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
              <div style={grid}>
                {items.map((card) => (
                  <CandidateCard
                    key={card.key}
                    card={card}
                    selected={picked.includes(Number(card.key))}
                    showGain={sort === "gain"}
                    gapsMode={gapsMode}
                    onToggle={() => toggle(Number(card.key))}
                    onCompare={() => setCompare(card)}
                    onEnter={() => setHovered(Number(card.key))}
                    onLeave={() =>
                      setHovered((current) =>
                        current === Number(card.key) ? null : current,
                      )
                    }
                  />
                ))}
              </div>
              {items.length === 0 && !candidates.isLoading && (
                <div style={empty}>
                  No candidate passes the current filters — loosen the
                  thresholds.
                </div>
              )}
              {pageCount > 1 && (
                <div style={pager}>
                  <button
                    disabled={page <= 1}
                    onClick={() => setPage(page - 1)}
                    style={pagerButton}
                  >
                    ‹
                  </button>
                  <span style={{ fontFamily: font.mono, fontSize: 11 }}>
                    {page} / {pageCount}
                  </span>
                  <button
                    disabled={page >= pageCount}
                    onClick={() => setPage(page + 1)}
                    style={pagerButton}
                  >
                    ›
                  </button>
                </div>
              )}
            </div>
          </div>

          <CompositionPanel
            preview={preview.data}
            picked={picked}
            candidates={data?.pool_points ?? []}
            hovered={hoveredPoint}
            gapsMode={gapsMode}
            onClear={() => setPicked([])}
            onRemove={(id) =>
              setPicked((prev) => prev.filter((x) => x !== id))
            }
          />
        </div>

        <div style={footer}>
          <span style={{ fontSize: 10.5, color: colors.textFaint, fontFamily: font.mono }}>
            Hover a thumbnail to locate it on the map
          </span>
          <div style={{ flex: 1 }} />
          <button onClick={onClose} style={ghost}>
            Cancel
          </button>
          <button
            onClick={confirm}
            disabled={picked.length === 0 || add.isPending}
            style={{
              ...confirmButton,
              background: picked.length ? colors.accent : colors.borderControl,
              color: picked.length ? colors.onAccent : colors.textFaint,
              cursor: picked.length ? "pointer" : "default",
            }}
          >
            {picked.length
              ? `Add ${picked.length} to the dataset`
              : "Add to the dataset"}
          </button>
        </div>
      </div>

      {compare && compare.near_dup && (
        <CompareOverlay
          title={`Near-duplicate — ${compare.name} ↔ ${compare.near_dup.name}`}
          subtitle={subtitleOf(compare)}
          candidate={{
            src: fileUrl(Number(compare.key)),
            name: compare.name,
            info: infoOf(compare),
            infoColor: bandColor(compare.score),
            badge: "candidate",
            badgeColor: colors.accent,
          }}
          reference={{
            src: fileUrl(compare.near_dup.media_id),
            name: compare.near_dup.name,
            info:
              compare.near_dup.kind === "hash"
                ? "probable duplicate (perceptual hash)"
                : "close neighbour (DINOv2)",
            badge: "dataset",
            badgeColor: colors.textMuted,
          }}
          onClose={() => setCompare(null)}
          actions={{
            dropLabel: picked.includes(Number(compare.key))
              ? "Remove from the selection"
              : "Do not add",
            keepLabel: picked.includes(Number(compare.key))
              ? "Keep in the selection"
              : "Add anyway",
            onDrop: () => {
              setPicked((prev) =>
                prev.filter((x) => x !== Number(compare.key)),
              );
              setCompare(null);
            },
            onKeep: () => {
              setPicked((prev) =>
                prev.includes(Number(compare.key))
                  ? prev
                  : [...prev, Number(compare.key)],
              );
              setCompare(null);
            },
          }}
        />
      )}
    </div>
  );
}

function subtitleOf(card: ComposeCandidate): string {
  const dup = card.near_dup;
  if (!dup) return "";
  return `hash similarity ${dup.similarity}% · DINOv2 cosine ${dup.cosine.toFixed(2)}`;
}

function infoOf(card: ComposeCandidate): string {
  const size = card.width && card.height ? `${card.width}×${card.height}` : "—";
  const score = card.score == null ? "—" : card.score.toFixed(0);
  return `${size} · quality ${score}`;
}

function CandidateCard({
  card,
  selected,
  showGain,
  gapsMode,
  onToggle,
  onCompare,
  onEnter,
  onLeave,
}: {
  card: ComposeCandidate;
  selected: boolean;
  showGain: boolean;
  gapsMode: boolean;
  onToggle: () => void;
  onCompare: () => void;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const warn = card.near_dup;
  return (
    <div
      onClick={onToggle}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      style={{
        position: "relative",
        border: `1.5px solid ${
          selected
            ? colors.accent
            : warn
              ? DANGER_BORDER
              : colors.borderControl
        }`,
        borderRadius: radii.card,
        overflow: "hidden",
        cursor: "pointer",
        background: colors.card,
        boxShadow: selected ? "0 0 0 1px rgba(232,147,90,0.4)" : "none",
      }}
    >
      <div style={{ position: "relative", aspectRatio: "4 / 3" }}>
        <img src={thumbUrl(Number(card.key))} alt="" style={thumb} />
        <div
          style={{
            ...checkbox,
            border: `1.5px solid ${
              selected ? colors.accent : "rgba(255,255,255,0.45)"
            }`,
            background: selected ? colors.accent : "rgba(15,16,19,0.45)",
          }}
        >
          {selected ? "✓" : ""}
        </div>
        <span style={{ ...scoreBadge, color: bandColor(card.score) }}>
          {card.score == null ? "—" : card.score.toFixed(0)}
        </span>
        {card.is_video ? (
          <span style={videoBadge}>▶</span>
        ) : (
          card.favorite && <span style={favBadge}>♥</span>
        )}
        {gapsMode && card.in_gap && !selected && (
          <span style={gapBadge}>◈ empty zone</span>
        )}
      </div>
      {showGain && (
        <div style={{ height: 3, background: colors.border }}>
          <div
            style={{
              width: `${Math.round(card.gain * 100)}%`,
              height: "100%",
              background: colors.info,
            }}
          />
        </div>
      )}
      <div style={metaRow}>
        <span style={metaName}>{card.name}</span>
        <span style={metaRes}>
          {card.width && card.height ? `${card.width}×${card.height}` : "—"}
        </span>
      </div>
      {warn && (
        <div
          onClick={(event) => {
            event.stopPropagation();
            onCompare();
          }}
          title="Compare side by side"
          style={warnBanner}
        >
          <span style={warnText}>
            {warn.kind === "hash"
              ? "≈ probable duplicate (hash)"
              : `≈ close to ${warn.name}`}
          </span>
          <span style={{ flex: "none", color: colors.accentHover }}>⇆</span>
        </div>
      )}
    </div>
  );
}

function CompositionPanel({
  preview,
  picked,
  candidates,
  hovered,
  gapsMode,
  onClear,
  onRemove,
}: {
  preview: ComposePreview | undefined;
  picked: number[];
  candidates: [number, number][];
  hovered: [number, number] | null;
  gapsMode: boolean;
  onClear: () => void;
  onRemove: (id: number) => void;
}) {
  if (!preview) {
    return <div style={{ ...sidePanel, color: colors.textFaint }}>…</div>;
  }
  const { size, pillars, map } = preview;
  const delta = preview.delta ?? 0;
  return (
    <div style={sidePanel}>
      <div style={gradeCard}>
        <div
          style={{
            fontSize: 29,
            fontWeight: 700,
            fontFamily: font.mono,
            color: gradeColor(preview.score),
          }}
        >
          {preview.grade}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10.5, color: colors.textMuted }}>
            Projected dataset score
          </div>
          <div
            style={{
              fontFamily: font.mono,
              fontSize: 12.5,
              color: colors.textSecondary,
            }}
          >
            {preview.score == null ? "—" : preview.score.toFixed(1)}{" "}
            <span
              style={{
                fontSize: 10.5,
                color:
                  delta > 0
                    ? colors.ok
                    : delta < 0
                      ? colors.danger
                      : colors.textFaint,
              }}
            >
              {delta >= 0 ? "+" : ""}
              {delta.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <span style={{ fontSize: 11, color: colors.textMuted, flex: 1 }}>
            Size
          </span>
          <span
            style={{
              fontFamily: font.mono,
              fontSize: 10.5,
              color: size.over ? colors.warn : colors.ok,
            }}
          >
            {size.base} + {size.picked} → {size.total} · target {size.min}–
            {size.max}
          </span>
        </div>
        <Bar
          percent={size.percent}
          color={size.over ? colors.warn : colors.ok}
          height={5}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        <Pillar
          label="Quality"
          percent={pillars.quality ?? 0}
          value={
            pillars.quality == null ? "—" : pillars.quality.toFixed(1)
          }
          color={bandColor(pillars.quality)}
        />
        <Pillar
          label="Diversity"
          percent={pillars.diversity ?? 0}
          value={
            pillars.diversity == null ? "—" : pillars.diversity.toFixed(0)
          }
          color={spreadColor(pillars.diversity)}
        />
        <Pillar
          label="Duplicates"
          percent={pillars.hygiene}
          value={pillars.duplicates ? `${pillars.duplicates} ⚠` : "0 ✓"}
          color={pillars.duplicates ? colors.danger : colors.ok}
        />
      </div>

      <div style={dividedSection}>
        <div style={sectionLabel}>Framing</div>
        {preview.framing.map((row) => (
          <div key={row.bucket} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                width: 70,
                fontSize: 10.5,
                color: row.under ? colors.warn : colors.textMuted,
              }}
            >
              {row.bucket.replace(/_/g, " ")}
            </span>
            <div style={stackedTrack}>
              <div
                style={{
                  width: `${(row.base / Math.max(1, size.total)) * 100}%`,
                  background: DATASET_GREY,
                }}
              />
              <div
                style={{
                  width: `${(row.added / Math.max(1, size.total)) * 100}%`,
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
        <div style={legendMono}>grey = dataset · orange = selection</div>
      </div>

      <div style={mapCard}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <span style={{ ...sectionLabel, flex: 1 }}>Corpus coverage</span>
          <span style={legendMono}>DINOv2 · 2D</span>
        </div>
        <CoverageMap
          width={map.width}
          height={map.height}
          candidates={candidates}
          dataset={map.dataset}
          selected={map.selected}
          zones={map.zones}
          zonesActive={gapsMode}
          hovered={hovered}
        />
        <div style={{ display: "flex", gap: 10, ...legendMono }}>
          <span>
            <span style={{ color: colors.textMuted }}>●</span> dataset
          </span>
          <span>
            <span style={{ color: colors.accent }}>●</span> selection
          </span>
          <span>
            <span style={{ color: colors.info }}>◌</span> empty zone
          </span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {preview.advice.map((row, index) => {
          const tone = TONE_ICON[row.tone] ?? TONE_ICON.ok;
          return (
            <div key={index} style={adviceCard}>
              <span style={{ color: tone.color, flex: "none" }}>
                {tone.icon}
              </span>
              <span>{row.text}</span>
            </div>
          );
        })}
      </div>

      <div style={dividedSection}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <span style={{ ...sectionLabel, flex: 1 }}>
            Selection ({picked.length})
          </span>
          <span onClick={onClear} style={clearLink}>
            clear all
          </span>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
          {picked.map((id) => (
            <img
              key={id}
              src={thumbUrl(id)}
              alt=""
              onClick={() => onRemove(id)}
              style={stripThumb}
            />
          ))}
        </div>
      </div>
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

function suggestButton(active: boolean, accent: string, enabled: boolean) {
  return {
    padding: "5px 11px",
    border: `1px solid ${active ? accent : colors.borderControl}`,
    borderRadius: radii.control,
    background: active
      ? accent === colors.info
        ? "rgba(111,168,220,0.14)"
        : "rgba(232,147,90,0.12)"
      : colors.card,
    color: active ? accent : enabled ? colors.textMutedAlt : colors.textFaint,
    fontSize: 11,
    fontWeight: 600,
    cursor: enabled || active ? "pointer" : "default",
  } as const;
}

const backdrop = {
  position: "fixed",
  inset: 0,
  zIndex: 66,
  background: "rgba(10,11,14,0.7)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 16,
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
  width: 230,
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

const inlineRow = {
  display: "flex",
  alignItems: "center",
  gap: 8,
} as const;

const fieldLabel = {
  fontSize: 11,
  color: colors.textMuted,
  width: 56,
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

const toolbar = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 9,
  padding: "9px 12px",
  borderBottom: `1px solid ${colors.border}`,
  background: colors.toolbar,
  flexWrap: "wrap",
} as const;

const searchBox = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  width: 280,
  padding: "6px 10px",
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

const counter = {
  fontSize: 11,
  color: colors.textFaint,
  fontFamily: font.mono,
} as const;

const suggestBar = {
  flex: "none",
  display: "flex",
  alignItems: "center",
  gap: 7,
  padding: "8px 12px",
  borderBottom: `1px solid ${colors.border}`,
} as const;

const grid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(142px, 1fr))",
  gap: 9,
} as const;

const thumb = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover",
} as const;

const checkbox = {
  position: "absolute",
  top: 6,
  left: 6,
  width: 17,
  height: 17,
  borderRadius: 5,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 11,
  color: colors.onAccent,
  fontWeight: 700,
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

const gapBadge = {
  position: "absolute",
  bottom: 6,
  right: 6,
  fontSize: 9,
  fontFamily: font.mono,
  padding: "2px 5px",
  borderRadius: 4,
  background: "rgba(21,32,44,0.85)",
  border: "1px solid #2f4860",
  color: colors.info,
} as const;

const metaRow = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "5px 7px",
} as const;

const metaName = {
  flex: 1,
  minWidth: 0,
  fontFamily: font.mono,
  fontSize: 9,
  color: colors.textMuted,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
} as const;

const metaRes = {
  fontFamily: font.mono,
  fontSize: 9,
  color: colors.textFaint,
} as const;

const warnBanner = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "3px 7px",
  background: DANGER_BG,
  borderTop: `1px solid ${DANGER_BORDER}`,
  fontSize: 9,
  color: colors.danger,
  fontFamily: font.mono,
  cursor: "pointer",
} as const;

const warnText = {
  flex: 1,
  minWidth: 0,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
} as const;

const empty = {
  padding: 40,
  textAlign: "center",
  color: colors.textFaint,
  fontSize: 12,
} as const;

const pager = {
  display: "flex",
  gap: 8,
  alignItems: "center",
  justifyContent: "center",
  paddingTop: 12,
} as const;

const pagerButton = {
  padding: "4px 10px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: radii.control,
  background: colors.card,
  color: colors.textSecondary,
  cursor: "pointer",
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

const stackedTrack = {
  flex: 1,
  height: 9,
  background: colors.border,
  borderRadius: 3,
  overflow: "hidden",
  display: "flex",
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
  display: "flex",
  gap: 7,
  padding: "7px 9px",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 7,
  background: colors.card,
  fontSize: 10.5,
  lineHeight: 1.45,
  color: colors.textMutedAlt,
} as const;

const clearLink = {
  fontSize: 10,
  color: colors.textFaint,
  cursor: "pointer",
} as const;

const stripThumb = {
  width: 38,
  height: 29,
  borderRadius: 4,
  objectFit: "cover",
  cursor: "pointer",
  border: `1px solid ${colors.borderControl}`,
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
