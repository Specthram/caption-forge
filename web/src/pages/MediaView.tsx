/** Media library workspace: filters, WD14 panel, grid and detail. */

import { useEffect, useMemo, useState } from "react";
import { useIndexStatus, useLibraryGrid } from "../api/hooks";
import { colors, font } from "../design/tokens";
import { useUiStore } from "../store/uiStore";
import { Button, Segmented } from "../components/atoms";
import { TagFilter } from "../components/molecules/TagFilter";
import type { SelectedTag } from "../components/molecules/TagFilter";
import { LibraryMediaCard } from "../components/molecules/LibraryMediaCard";
import { MediaDetailPanel } from "../components/organisms/MediaDetailPanel";
import { WD14Panel } from "../components/organisms/WD14Panel";
import { useToggleFavorite } from "../api/hooks";

const PAGE = 60;

const SORTS = [
  { value: "date_desc", label: "Newest" },
  { value: "quality_desc", label: "Quality ↓" },
  { value: "dimension_desc", label: "Largest" },
];

export function MediaView() {
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const openZoom = useUiStore((state) => state.openZoom);
  const openWatermark = useUiStore((state) => state.openWatermark);
  const toggleFav = useToggleFavorite();
  const status = useIndexStatus();
  // The auto-tagger is one of the index scans: a machine that turned it off
  // must not be able to launch it from here either.
  const wd14Off = (status.data?.steps ?? []).some(
    (step) => step.key === "wd14" && !step.enabled,
  );

  // Filters, sort and page survive a refresh, so they live in the store.
  const { include, exclude, match, favOnly, sort, page } = useUiStore(
    (state) => state.mediaView,
  );
  const setMediaView = useUiStore((state) => state.setMediaView);
  // Changing a filter re-slices the whole set, so the current page number no
  // longer means anything: every filter setter rewinds to the first page.
  const setInclude = (next: SelectedTag[]) =>
    setMediaView({ include: next, page: 1 });
  const setExclude = (next: SelectedTag[]) =>
    setMediaView({ exclude: next, page: 1 });
  const setMatch = (next: string) => setMediaView({ match: next, page: 1 });
  const setFavOnly = (next: boolean) => setMediaView({ favOnly: next, page: 1 });
  const setSort = (next: string) => setMediaView({ sort: next, page: 1 });
  const setPage = (next: number) => setMediaView({ page: next });
  const [wdOpen, setWdOpen] = useState(false);
  // Focus lives in the store, not here: the Quality report's "Open in
  // Media" navigates to this view with a card already focused.
  const focusKey = useUiStore((state) => state.mediaFocusKey);
  const setFocusKey = useUiStore((state) => state.setMediaFocus);

  const filters = {
    offset: (page - 1) * PAGE,
    limit: PAGE,
    tag_ids: include.map((tag) => tag.id),
    exclude_tag_ids: exclude.map((tag) => tag.id),
    match,
    favorites_only: favOnly,
    sort,
    quality_metric: qualityMetric,
  };
  const grid = useLibraryGrid(filters);
  const total = grid.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE));
  const pageKeys = useMemo(
    () => grid.data?.items.map((card) => card.key) ?? [],
    [grid.data],
  );

  // Media were removed since the last session: a restored page past the end
  // of the set would paint an empty grid.
  useEffect(() => {
    if (grid.data && page > pageCount) setPage(1);
  }, [grid.data, page, pageCount]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "10px 14px",
            borderBottom: `1px solid ${colors.border}`,
            background: colors.toolbar,
            flexWrap: "wrap",
          }}
        >
          <TagFilter
            label="+ include tag"
            selected={include}
            onAdd={(tag) => setInclude([...include, tag])}
            onRemove={(id) =>
              setInclude(include.filter((tag) => tag.id !== id))
            }
          />
          <TagFilter
            label="− exclude tag"
            selected={exclude}
            onAdd={(tag) => setExclude([...exclude, tag])}
            onRemove={(id) =>
              setExclude(exclude.filter((tag) => tag.id !== id))
            }
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
            style={selectStyle}
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
          <Button
            onClick={() => openWatermark()}
            style={{
              background: colors.watermarkBtn,
              color: colors.watermark,
              border: `1px solid ${colors.watermarkBorder}`,
            }}
          >
            ◪ Watermark Lab
          </Button>
          <Button
            disabled={wd14Off}
            title={
              wd14Off ? "Auto-tags is off on this machine — see Settings" : ""
            }
            style={
              wd14Off
                ? { color: colors.textFaint }
                : wdOpen
                  ? { background: colors.accentTint, color: colors.accent }
                  : undefined
            }
            onClick={() => setWdOpen((open) => !open)}
          >
            ✨ Auto-tag (WD14)
          </Button>
          <span style={{ flex: 1 }} />
          <span
            style={{ fontFamily: font.mono, fontSize: 11, color: colors.textMuted }}
          >
            {total} media
          </span>
        </div>

        {wdOpen && !wd14Off && (
          <WD14Panel
            filterTagIds={filters.tag_ids}
            excludeTagIds={filters.exclude_tag_ids}
            match={match}
            pageKeys={pageKeys}
            focusKey={focusKey}
          />
        )}

        <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              gap: 10,
            }}
          >
            {grid.data?.items.map((card) => (
              <LibraryMediaCard
                key={card.key}
                card={card}
                focused={focusKey === card.key}
                onFocus={() => setFocusKey(card.key)}
                onZoom={() => openZoom(`/api/media/${card.key}/file`, card.name)}
                onToggleFav={() => toggleFav.mutate(card.key)}
              />
            ))}
          </div>
        </div>

        {pageCount > 1 && (
          <div style={pager}>
            <Button disabled={page <= 1} onClick={() => setPage(page - 1)}>
              ‹
            </Button>
            <span>
              {page} / {pageCount}
            </span>
            <Button disabled={page >= pageCount} onClick={() => setPage(page + 1)}>
              ›
            </Button>
          </div>
        )}
      </div>

      <MediaDetailPanel
        focusKey={focusKey}
        onClose={() => setFocusKey(null)}
      />
    </div>
  );
}

const selectStyle = {
  padding: "5px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;

const favLabel = {
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
  borderTop: `1px solid ${colors.border}`,
  fontFamily: font.mono,
  fontSize: 12,
} as const;
