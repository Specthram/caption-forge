/**
 * UI state store (Zustand): current view, focus, overlays, the shared grid
 * preferences and each workspace's own browsing state. Server data lives in
 * TanStack Query, not here — this store only holds interaction state.
 *
 * Everything a user *navigated to* is persisted to `localStorage` so a page
 * refresh lands exactly where they left off: the view, the active dataset and
 * caption type, the filters and sorts, the page of each grid, the focused
 * card. What is merely *open* is not: the overlays (zoom, crop, grounding,
 * compare), the jobs drawer and the search palette all reopen closed — a
 * refresh with a modal up should not trap the user back in it.
 *
 * A persisted id can go stale between sessions (the dataset, library or media
 * it names may have been deleted). Every view therefore validates what it
 * restores against the server's answer and falls back — see the effects in
 * `DatasetsView`, `LibrariesView`, `TagsView`, and the detail panels, which
 * drop a focus whose media no longer resolves.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  CropRatio,
  CropRect,
  ViewId,
  WatermarkTab,
} from "../api/types";
import type { SelectedTag } from "../components/molecules/TagFilter";

interface ZoomState {
  open: boolean;
  src: string | null;
  name: string;
  isVideo: boolean;
  scale: number;
  tx: number;
  ty: number;
}

interface CompareState {
  open: boolean;
  leftSrc: string;
  rightSrc: string;
  leftName: string;
  rightName: string;
  scale: number;
  tx: number;
  ty: number;
}

/**
 * The grounding modal's session state.
 *
 * `mode` decides everything the modal shows: `caption` scores the focused
 * caption's LLM-extracted claims, `media` scores the focused media's tags.
 * `threshold` starts from the Settings value and is a live slider — nothing
 * is re-scored when it moves, only re-read (see the backend note: scores are
 * stored raw, the threshold is applied on read).
 */
interface GroundingState {
  open: boolean;
  mode: "caption" | "media";
  /** The media key the modal was opened on. */
  key: string | null;
  name: string;
  threshold: number;
  /** Element id under the cursor, whose heat map is shown at full strength. */
  hover: number | null;
  /** Whether the union of every validated element's heat is shown instead. */
  coverage: boolean;
}

/**
 * The crop overlay's session state.
 *
 * `mediaId` is always the *source* image being framed, never a crop:
 * re-editing a crop opens the overlay on its parent with `editId` set. `rect`
 * is in percentages of the source; `zoom` pans/scales the viewport only and
 * never touches the rectangle; `confirm` switches to the replace/beside
 * dialog once a NEW rectangle is validated.
 */
interface CropState {
  open: boolean;
  mediaId: number | null;
  /** The crop being re-framed, or null when a new one is being drawn. */
  editId: number | null;
  rect: CropRect;
  ratio: CropRatio;
  confirm: boolean;
  zoom: { z: number; tx: number; ty: number };
}

/**
 * The Watermark Lab overlay's session state (v2 — fully self-contained).
 *
 * `open` mounts the Lab; `focusKey` is the media whose review panel is shown
 * (null = inventory only). `initialTab` is the tab to land on when an entry
 * point opens the Lab focused on a media (e.g. its side-panel encart opens it
 * on the media's current tab). Like every overlay it reopens closed after a
 * refresh — not persisted.
 */
interface WatermarkState {
  open: boolean;
  focusKey: string | null;
  initialTab: WatermarkTab | null;
}

/** The Datasets workspace's own browsing state. */
interface DatasetsViewState {
  tab: "media" | "quality" | "caption";
  page: number;
  focusKey: string | null;
}

/** The Media workspace's filters, sort and page. */
interface MediaViewState {
  include: SelectedTag[];
  exclude: SelectedTag[];
  match: string;
  favOnly: boolean;
  sort: string;
  page: number;
}

/** The Watermark Lab's persisted filter (tags/match/sort/favorites). */
interface WatermarkViewState {
  include: SelectedTag[];
  exclude: SelectedTag[];
  match: string;
  favOnly: boolean;
  sort: string;
}

/** The Libraries workspace's selected source and page. */
interface LibrariesViewState {
  activeId: number | null;
  page: number;
}

/** The Tags workspace's selected category. */
interface TagsViewState {
  activeId: number | null;
}

interface UiState {
  view: ViewId;
  datasetId: number | null;
  captionType: string;
  reviewFilter: string;
  qualityMetric: string;
  page: number;
  focusKey: string | null;
  /** The Media view's focused card — set from outside when a report
   *  inspector navigates to "Open in Media". */
  mediaFocusKey: string | null;

  datasetsView: DatasetsViewState;
  mediaView: MediaViewState;
  watermarkView: WatermarkViewState;
  librariesView: LibrariesViewState;
  tagsView: TagsViewState;

  jobsOpen: boolean;
  searchOpen: boolean;
  grounding: GroundingState;
  crop: CropState;
  zoom: ZoomState;
  compare: CompareState;
  watermark: WatermarkState;

  setView: (view: ViewId) => void;
  setDataset: (id: number | null) => void;
  setCaptionType: (type: string) => void;
  setReviewFilter: (filter: string) => void;
  setQualityMetric: (metric: string) => void;
  setPage: (page: number) => void;
  setFocus: (key: string | null) => void;
  setMediaFocus: (key: string | null) => void;

  setDatasetsView: (partial: Partial<DatasetsViewState>) => void;
  setMediaView: (partial: Partial<MediaViewState>) => void;
  setWatermarkView: (partial: Partial<WatermarkViewState>) => void;
  setLibrariesView: (partial: Partial<LibrariesViewState>) => void;
  setTagsView: (partial: Partial<TagsViewState>) => void;

  toggleJobs: (open?: boolean) => void;
  toggleSearch: (open?: boolean) => void;
  openGrounding: (
    mode: "caption" | "media",
    key: string,
    name: string,
    threshold: number,
  ) => void;
  closeGrounding: () => void;
  setGrounding: (partial: Partial<GroundingState>) => void;
  openCrop: (
    mediaId: number,
    edit?: { id: number; rect: CropRect; ratio: CropRatio },
  ) => void;
  closeCrop: () => void;
  setCrop: (partial: Partial<CropState>) => void;
  openZoom: (src: string, name: string, isVideo?: boolean) => void;
  closeZoom: () => void;
  setZoom: (partial: Partial<ZoomState>) => void;
  openCompare: (
    leftSrc: string,
    leftName: string,
    rightSrc: string,
    rightName: string,
  ) => void;
  closeCompare: () => void;
  setCompare: (partial: Partial<CompareState>) => void;
  openWatermark: (
    focusKey?: string | null,
    initialTab?: WatermarkTab | null,
  ) => void;
  closeWatermark: () => void;
  setWatermark: (partial: Partial<WatermarkState>) => void;
}

const CLOSED_ZOOM: ZoomState = {
  open: false,
  src: null,
  name: "",
  isVideo: false,
  scale: 1,
  tx: 0,
  ty: 0,
};

const CLOSED_GROUNDING: GroundingState = {
  open: false,
  mode: "caption",
  key: null,
  name: "",
  threshold: 55,
  hover: null,
  coverage: false,
};

/** A fresh rectangle: a centred frame the user drags into place. */
const DEFAULT_RECT: CropRect = { x: 14, y: 12, w: 58, h: 58 };

const CLOSED_CROP: CropState = {
  open: false,
  mediaId: null,
  editId: null,
  rect: DEFAULT_RECT,
  ratio: "free",
  confirm: false,
  zoom: { z: 1, tx: 0, ty: 0 },
};

const CLOSED_COMPARE: CompareState = {
  open: false,
  leftSrc: "",
  rightSrc: "",
  leftName: "",
  rightName: "",
  scale: 1,
  tx: 0,
  ty: 0,
};

const DEFAULT_DATASETS_VIEW: DatasetsViewState = {
  tab: "media",
  page: 1,
  focusKey: null,
};

const DEFAULT_MEDIA_VIEW: MediaViewState = {
  include: [],
  exclude: [],
  match: "all",
  favOnly: false,
  sort: "date_desc",
  page: 1,
};

const DEFAULT_WATERMARK_VIEW: WatermarkViewState = {
  include: [],
  exclude: [],
  match: "all",
  favOnly: false,
  sort: "date_desc",
};

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      view: "caption",
      datasetId: null,
      captionType: "txt",
      reviewFilter: "all",
      qualityMetric: "average",
      page: 1,
      focusKey: null,
      mediaFocusKey: null,

      datasetsView: DEFAULT_DATASETS_VIEW,
      mediaView: DEFAULT_MEDIA_VIEW,
      watermarkView: DEFAULT_WATERMARK_VIEW,
      librariesView: { activeId: null, page: 1 },
      tagsView: { activeId: null },

      jobsOpen: false,
      searchOpen: false,
      grounding: CLOSED_GROUNDING,
      crop: CLOSED_CROP,
      zoom: CLOSED_ZOOM,
      compare: CLOSED_COMPARE,
      watermark: { open: false, focusKey: null, initialTab: null },

      setView: (view) => set({ view }),
      // Switching dataset invalidates both grids that page through it.
      setDataset: (datasetId) =>
        set({
          datasetId,
          page: 1,
          focusKey: null,
          datasetsView: DEFAULT_DATASETS_VIEW,
        }),
      setCaptionType: (captionType) => set({ captionType }),
      setReviewFilter: (reviewFilter) => set({ reviewFilter, page: 1 }),
      setQualityMetric: (qualityMetric) => set({ qualityMetric }),
      setPage: (page) => set({ page }),
      setFocus: (focusKey) => set({ focusKey }),
      setMediaFocus: (mediaFocusKey) => set({ mediaFocusKey }),

      setDatasetsView: (partial) =>
        set((state) => ({
          datasetsView: { ...state.datasetsView, ...partial },
        })),
      setMediaView: (partial) =>
        set((state) => ({ mediaView: { ...state.mediaView, ...partial } })),
      setWatermarkView: (partial) =>
        set((state) => ({
          watermarkView: { ...state.watermarkView, ...partial },
        })),
      setLibrariesView: (partial) =>
        set((state) => ({
          librariesView: { ...state.librariesView, ...partial },
        })),
      setTagsView: (partial) =>
        set((state) => ({ tagsView: { ...state.tagsView, ...partial } })),

      toggleJobs: (open) =>
        set((state) => ({ jobsOpen: open ?? !state.jobsOpen })),
      toggleSearch: (open) =>
        set((state) => ({ searchOpen: open ?? !state.searchOpen })),
      openGrounding: (mode, key, name, threshold) =>
        set({
          grounding: {
            ...CLOSED_GROUNDING,
            open: true,
            mode,
            key,
            name,
            threshold,
          },
        }),
      closeGrounding: () => set({ grounding: CLOSED_GROUNDING }),
      setGrounding: (partial) =>
        set((state) => ({ grounding: { ...state.grounding, ...partial } })),
      openCrop: (mediaId, edit) =>
        set({
          crop: {
            ...CLOSED_CROP,
            open: true,
            mediaId,
            editId: edit?.id ?? null,
            rect: edit?.rect ?? DEFAULT_RECT,
            ratio: edit?.ratio ?? "free",
          },
        }),
      closeCrop: () => set({ crop: CLOSED_CROP }),
      setCrop: (partial) =>
        set((state) => ({ crop: { ...state.crop, ...partial } })),
      openZoom: (src, name, isVideo = false) =>
        set({ zoom: { ...CLOSED_ZOOM, open: true, src, name, isVideo } }),
      closeZoom: () => set({ zoom: CLOSED_ZOOM }),
      setZoom: (partial) =>
        set((state) => ({ zoom: { ...state.zoom, ...partial } })),
      openCompare: (leftSrc, leftName, rightSrc, rightName) =>
        set({
          compare: {
            ...CLOSED_COMPARE,
            open: true,
            leftSrc,
            leftName,
            rightSrc,
            rightName,
          },
        }),
      closeCompare: () => set({ compare: CLOSED_COMPARE }),
      setCompare: (partial) =>
        set((state) => ({ compare: { ...state.compare, ...partial } })),
      openWatermark: (focusKey, initialTab) =>
        set({
          watermark: {
            open: true,
            focusKey: focusKey ?? null,
            initialTab: initialTab ?? null,
          },
        }),
      closeWatermark: () =>
        set({
          watermark: {
            open: false,
            focusKey: null,
            initialTab: null,
          },
        }),
      setWatermark: (partial) =>
        set((state) => ({ watermark: { ...state.watermark, ...partial } })),
    }),
    {
      name: "cf-ui",
      version: 1,
      // Where the user *is*, never what happens to be *open*: an overlay,
      // the jobs drawer and the search palette all come back closed.
      partialize: (state) => ({
        view: state.view,
        datasetId: state.datasetId,
        captionType: state.captionType,
        reviewFilter: state.reviewFilter,
        qualityMetric: state.qualityMetric,
        page: state.page,
        focusKey: state.focusKey,
        mediaFocusKey: state.mediaFocusKey,
        datasetsView: state.datasetsView,
        mediaView: state.mediaView,
        watermarkView: state.watermarkView,
        librariesView: state.librariesView,
        tagsView: state.tagsView,
      }),
    },
  ),
);
