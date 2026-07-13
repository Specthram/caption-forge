/** Caption tab centre: dataset toolbar, paged media grid, pager. */

import { useEffect, useMemo } from "react";
import {
  useCaptionGrid,
  useCaptionTypes,
  useDatasets,
  useGroundingEnabled,
} from "../../api/hooks";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useSelectionStore } from "../../store/selectionStore";
import { useCaptionStore } from "../../store/captionStore";
import { Segmented } from "../atoms";
import { MediaCard } from "../molecules/MediaCard";

const PAGE_SIZE = 40;

const REVIEW_FILTERS = [
  { value: "all", label: "All" },
  { value: "to_review", label: "⚠ To review" },
  { value: "ungrounded", label: "◈ Weak grounding" },
];

const selectStyle = {
  padding: "5px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;

export function CaptionCenterGrid() {
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const reviewFilter = useUiStore((state) => state.reviewFilter);
  const qualityMetric = useUiStore((state) => state.qualityMetric);
  const page = useUiStore((state) => state.page);
  const focusKey = useUiStore((state) => state.focusKey);
  const setDataset = useUiStore((state) => state.setDataset);
  const setCaptionType = useUiStore((state) => state.setCaptionType);
  const setReviewFilter = useUiStore((state) => state.setReviewFilter);
  const setPage = useUiStore((state) => state.setPage);
  const setFocus = useUiStore((state) => state.setFocus);

  const selected = useSelectionStore((state) => state.selected);
  const toggle = useSelectionStore((state) => state.toggle);
  const addSelection = useSelectionStore((state) => state.add);
  const removeSelection = useSelectionStore((state) => state.remove);
  const locked = useCaptionStore((state) => state.locked);

  const datasets = useDatasets();
  const types = useCaptionTypes();
  const groundingEnabled = useGroundingEnabled();

  // The "weak grounding" filter only exists when grounding does. Drop it from
  // the options, and reset the filter if it was the one selected.
  const reviewFilters = groundingEnabled
    ? REVIEW_FILTERS
    : REVIEW_FILTERS.filter((filter) => filter.value !== "ungrounded");
  useEffect(() => {
    if (!groundingEnabled && reviewFilter === "ungrounded") {
      setReviewFilter("all");
    }
  }, [groundingEnabled, reviewFilter, setReviewFilter]);

  const grid = useCaptionGrid(
    {
      dataset_id: datasetId ?? 0,
      caption_type: captionType,
      review_filter: reviewFilter,
      offset: (page - 1) * PAGE_SIZE,
      limit: PAGE_SIZE,
      quality_metric: qualityMetric,
    },
    datasetId != null,
  );

  const total = grid.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const pageKeys = useMemo(
    () => grid.data?.items.map((card) => card.key) ?? [],
    [grid.data],
  );

  // The dataset shrank since the last session: a restored page past its end
  // would paint an empty grid.
  useEffect(() => {
    if (grid.data && page > pageCount) setPage(1);
  }, [grid.data, page, pageCount, setPage]);
  const pageSelected =
    pageKeys.length > 0 && pageKeys.every((key) => selected.has(key));
  const typeOptions =
    types.data?.types.map((type) => ({
      value: type,
      label: type === "tags" ? "tags" : `.${type}`,
    })) ?? [];

  // The caption type restored from the last session may have been removed
  // from Settings since: fall back to the first one still configured.
  const knownTypes = types.data?.types;
  useEffect(() => {
    if (knownTypes?.length && !knownTypes.includes(captionType)) {
      setCaptionType(knownTypes[0]);
    }
  }, [knownTypes, captionType, setCaptionType]);

  // Arrow keys move the focused card within the current page (ignored while
  // typing in an input or textarea).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (pageKeys.length === 0) return;
      const index = focusKey ? pageKeys.indexOf(focusKey) : -1;
      const delta = event.key === "ArrowRight" ? 1 : -1;
      const next = Math.max(
        0,
        Math.min(pageKeys.length - 1, index + delta),
      );
      setFocus(pageKeys[next]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focusKey, pageKeys, setFocus]);

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
          gap: 12,
          padding: "10px 14px",
          borderBottom: `1px solid ${colors.border}`,
          background: colors.toolbar,
        }}
      >
        <select
          style={selectStyle}
          value={datasetId ?? ""}
          onChange={(event) => setDataset(Number(event.target.value))}
        >
          {datasets.data?.datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name} · {dataset.count}
            </option>
          ))}
        </select>

        {typeOptions.length > 0 && (
          <Segmented
            value={captionType}
            onChange={setCaptionType}
            options={typeOptions}
          />
        )}

        <select
          style={selectStyle}
          value={reviewFilter}
          onChange={(event) => setReviewFilter(event.target.value)}
        >
          {reviewFilters.map((filter) => (
            <option key={filter.value} value={filter.value}>
              {filter.label}
            </option>
          ))}
        </select>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11.5,
            color: colors.textMuted,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={pageSelected}
            onChange={(event) =>
              event.target.checked
                ? addSelection(pageKeys)
                : removeSelection(pageKeys)
            }
          />
          Select page
        </label>

        <span style={{ flex: 1 }} />
        <span
          style={{ fontFamily: font.mono, fontSize: 11, color: colors.textMuted }}
        >
          {total} media
        </span>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
        {datasetId == null ? (
          <Empty text="Select a dataset to start." />
        ) : total === 0 ? (
          <Empty text="Nothing matches this filter." />
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(152px, 1fr))",
              gap: 10,
            }}
          >
            {grid.data?.items.map((card) => (
              <MediaCard
                key={card.key}
                card={card}
                selected={selected.has(card.key)}
                focused={focusKey === card.key}
                locked={locked.has(card.key)}
                onSelect={() => toggle(card.key)}
                onFocus={() => setFocus(card.key)}
              />
            ))}
          </div>
        )}
      </div>

      {pageCount > 1 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            padding: 10,
            borderTop: `1px solid ${colors.border}`,
            fontFamily: font.mono,
            fontSize: 12,
          }}
        >
          <button
            disabled={page <= 1}
            onClick={() => setPage(page - 1)}
            style={pagerButton}
          >
            ‹
          </button>
          <span>
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
  );
}

const pagerButton = {
  width: 26,
  height: 26,
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.raised,
  color: colors.text,
  cursor: "pointer",
} as const;

function Empty({ text }: { text: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "60%",
        color: colors.textFaint,
        fontSize: 13,
      }}
    >
      ◌ {text}
    </div>
  );
}
