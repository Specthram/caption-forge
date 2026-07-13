/** Shared API payload types (mirrors the FastAPI JSON shapes). */

export type ViewId =
  | "caption"
  | "datasets"
  | "media"
  | "tags"
  | "libraries"
  | "settings"
  | "system";

export interface ModelInfo {
  name: string;
  type: string;
  format: string;
  hf_config: string | null;
  has_mmproj: boolean;
}

/** Live memory of the primary CUDA GPU (all figures in GB). */
export interface GpuInfo {
  name: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
}

export interface ModelStatus {
  loaded: boolean;
  name: string | null;
  type: string | null;
  format: string | null;
  status: string;
  vram_total_gb: number | null;
  gpu: GpuInfo | null;
  device: string;
}

export interface PromptPreset {
  title: string;
  prompt: string;
  builtin: boolean;
}

export interface PromptsResponse {
  prompts: PromptPreset[];
  selected: string | null;
  temperature: number;
  think_mode: string;
}

export interface DatasetInfo {
  id: number;
  name: string;
  count: number;
  deploy_name: string;
  deploy_resolution: number;
}

/** A crop rectangle, in percentages of the source image. */
export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type CropRatio =
  | "free"
  | "1:1"
  | "3:2"
  | "2:3"
  | "4:3"
  | "3:4"
  | "16:9"
  | "9:16";

/**
 * What a card or a detail carries when it *is* a crop: the image it frames,
 * the rectangle and the size it renders to. `null` on an ordinary media.
 */
export interface CropInfo {
  parent_media_id: number;
  rect: CropRect;
  ratio: CropRatio;
  width: number;
  height: number;
}

/** One crop of a media, as listed in the detail panels. */
export interface Crop {
  id: number;
  parent_media_id: number;
  rect: CropRect;
  ratio: CropRatio;
  width: number;
  height: number;
  thumb: string;
  dataset_ids: number[];
  /** Whether the crop already stands in the dataset being looked at. */
  in_dataset: boolean;
}

/** The source image the overlay frames (dimensions probed when unindexed). */
export interface CropSource {
  media_id: number;
  name: string;
  width: number;
  height: number;
  file: string;
}

export interface CaptionCard {
  key: string;
  name: string;
  is_video: boolean;
  missing: boolean;
  hidden: boolean;
  repeats: number;
  thumb: string;
  quality: number | null;
  caption: string;
  review: string;
  review_issues: string[];
  grounding: GroundingSummary | null;
  ext: string;
  revisions: number;
  revision_pinned: boolean;
  crop: CropInfo | null;
}

/**
 * The grounding tiles of a card: how many claims SigLIP supports, against
 * the configured threshold. `coverage` is the share of the *caption* those
 * validated claims account for — image-area coverage needs the heat maps,
 * which only the modal computes. `stale` means the scores were produced by
 * a different checkpoint and are not comparable to the current one's.
 */
export interface GroundingSummary {
  validated: number;
  flagged: number;
  total: number;
  coverage: number;
  stale: boolean;
}

export interface CaptionGrid {
  total: number;
  items: CaptionCard[];
}

export interface TagChipData {
  id: number;
  name: string;
  category: string;
  color: string;
}

export interface RevisionOption {
  label: string;
  value: number | string;
}

/** One folder listing from the server-side directory browser. */
export interface FolderListing {
  path: string;
  /** Where "up" goes: "" back to drive roots, null when already at the top. */
  parent: string | null;
  is_root: boolean;
  entries: { name: string; path: string }[];
}

export interface FileListing {
  path: string;
  parent: string | null;
  is_root: boolean;
  entries: { name: string; path: string; kind: "dir" | "file" }[];
}

export interface MediaMeta {
  width: number | null;
  height: number | null;
  size_bytes: number;
  sha256: string;
  datasets: number;
}

export interface MediaDetail {
  key: string;
  name: string;
  is_video: boolean;
  missing: boolean;
  favorite: boolean;
  hidden: boolean;
  repeats: number;
  file: string;
  thumb: string;
  caption: string;
  char_count: number;
  word_count: number;
  revisions: RevisionOption[];
  revision_value: number | string | null;
  tags: TagChipData[];
  meta: MediaMeta;
  quality: number | null;
  quality_metric: string | null;
  quality_scores: Record<string, number | null>;
  review: string;
  review_issues: string[];
  grounding: GroundingSummary | null;
  caption_score: CaptionScoreLine[];
  deploy: string;
  /** Set when the focused media is itself a virtual crop. */
  crop: CropInfo | null;
}

/**
 * One line of the reference-free caption-score card: an encoder family, the
 * checkpoint in effect and the 0-100 score it gave the whole caption (null
 * when never run). `stale` marks a score left by a checkpoint that is no
 * longer the configured one, so its number is not comparable to the others.
 */
export interface CaptionScoreLine {
  kind: string;
  label: string;
  model_id: string;
  score: number | null;
  stale: boolean;
}

/** One encoder column of the dataset caption-score report. */
export interface CaptionScoreReportKind {
  kind: string;
  label: string;
  model_id: string;
}

/** One media row of the dataset caption-score report. */
export interface CaptionScoreReportMedia {
  key: string;
  name: string;
  scores: Record<string, number>;
  stale: Record<string, boolean>;
  /** Mean across the encoders that scored it — the ranking key. */
  mean: number;
}

/**
 * The Datasets → Caption score report: every media's reference-free scores
 * aggregated. `media` is sorted worst-first so the captions dragging the
 * dataset average down sit at the top.
 */
export interface CaptionScoreReport {
  kinds: CaptionScoreReportKind[];
  averages: Record<string, number | null>;
  overall: number | null;
  scored_media: number;
  total_media: number;
  media: CaptionScoreReportMedia[];
}

export interface TagCategory {
  id: number;
  name: string;
  color: string;
  count?: number;
}

export interface TagItem {
  id: number;
  name: string;
  usage_count: number;
}

export interface TagListPage {
  total: number;
  items: TagItem[];
}

export interface MediaGridCard {
  key: string;
  name: string;
  thumb: string;
  quality: number | null;
  width: number | null;
  height: number | null;
  favorite: boolean;
  is_video: boolean;
  /** Non-null when the card is a virtual crop of another media. */
  crop?: CropInfo | null;
  /** Aggregate watermark status, when the media carries zones or is excluded. */
  wm_status?: WatermarkStatus | null;
}

export interface MediaGridPage {
  total: number;
  items: MediaGridCard[];
}

/** The sidebar badges (the Caption one is derived from the active dataset). */
export interface NavCounts {
  media: number;
  tags: number;
  libraries: number;
}

/** Why a candidate is flagged as a near-duplicate of the set. */
export interface NearDup {
  media_id: number;
  name: string;
  /** Perceptual-hash similarity, 0-100. */
  similarity: number;
  /** DINOv2 cosine, 0-1. */
  cosine: number;
  /** "hash" = probable duplicate, "cosine" = close neighbour. */
  kind: "hash" | "cosine";
}

/** A library media ranked against the dataset being composed. */
export interface ComposeCandidate extends MediaGridCard {
  /** Normalized 0-100 quality for the selected metric, null when unscored. */
  score: number | null;
  /** How much visual ground the media adds to the set, 0-1. */
  gain: number;
  near_dup: NearDup | null;
  in_gap: boolean;
  /** Map point in the coverage viewBox, null when never embedded. */
  xy: [number, number] | null;
  metric: string;
}

export interface ComposeCandidatesPage {
  /** How many candidates pass the filters (the grid pages through them). */
  total: number;
  /** How many candidates exist before any filter. */
  pool: number;
  items: ComposeCandidate[];
  /** Map point of every filtered candidate, not just the page. */
  pool_points: [number, number][];
  gap_count: number;
  semantic_available: boolean;
  metrics: { id: string; label: string }[];
  libraries: string[];
}

export interface ComposeZone {
  x: number;
  y: number;
  r: number;
}

export interface ComposeFramingRow {
  bucket: string;
  base: number;
  added: number;
  total: number;
  share: number;
  under: boolean;
}

export interface ComposePreview {
  score: number | null;
  base_score: number | null;
  delta: number | null;
  grade: string;
  pillars: {
    quality: number | null;
    diversity: number | null;
    hygiene: number;
    duplicates: number;
  };
  size: {
    base: number;
    picked: number;
    total: number;
    min: number;
    max: number;
    percent: number;
    over: boolean;
  };
  framing: ComposeFramingRow[];
  dup_alerts: { media_id: number; name: string; near_dup: NearDup }[];
  map: {
    dataset: [number, number][];
    selected: [number, number][];
    zones: ComposeZone[];
    width: number;
    height: number;
  };
  advice: { tone: "ok" | "warn" | "danger" | "info"; text: string }[];
}

export interface Triggerword {
  id: number;
  name: string;
}

export interface LibraryCard extends MediaGridCard {
  tags: { name: string; color: string }[];
  tag_count: number;
}

export interface LibraryGridPage {
  total: number;
  items: LibraryCard[];
}

export interface MediaFullDetail {
  key: string;
  name: string;
  is_video: boolean;
  favorite: boolean;
  /** Set when the focused media is itself a virtual crop. */
  crop: CropInfo | null;
  file: string;
  thumb: string;
  meta: {
    width: number | null;
    height: number | null;
    sha256: string;
    files: number;
    effective: string | null;
  };
  quality_scores: Record<string, number | null>;
  datasets: string[];
  tags: {
    category: string;
    color: string;
    tags: { id: number; name: string }[];
  }[];
  /** Reference-free "Tags Score" lines (tags scored as one joined text). */
  tag_score: CaptionScoreLine[];
  captions: { type: string; preview: string }[];
}

export interface TaggerModel {
  source: string;
  label: string;
  available: boolean;
}

export interface TaggerModels {
  models: TaggerModel[];
  default_source: string;
  general: number;
  character: number;
}

export interface LibrarySource {
  id: number;
  name: string;
  path: string;
  recursive: boolean;
  internal: boolean;
  count: number;
  /** Set on a sub-library: the parent folder library it was promoted from. */
  parent_library_id: number | null;
  /** A sub-library's folder path relative to its parent root (null on top). */
  rel_path: string | null;
  /** Whether the library carries a subfolder mapping (renders a group card). */
  mapped: boolean;
  /** How many folder rules the mapping stores. */
  rule_count: number;
  /** How many mapped folders are skipped (excluded). */
  skipped_folders: number;
  /** How many sub-libraries the library owns. */
  sub_count: number;
}

/** One sub-folder node of the mapping wizard's tree (see /folder-tree). */
export interface FolderTreeNode {
  rel_path: string;
  name: string;
  /** Media sitting directly in this folder. */
  own: number;
  /** Media in this folder and every descendant. */
  total: number;
  /** Up to three sample media file paths (unused by the placeholder minis). */
  samples: string[];
  children: FolderTreeNode[];
}

export interface FolderTree {
  path: string;
  name: string;
  own: number;
  total: number;
  samples: string[];
  children: FolderTreeNode[];
}

/** A persisted rule for one mapped sub-folder. */
export interface FolderRule {
  rel_path: string;
  mode: "keep" | "sublib" | "exclude";
  sub_library_id: number | null;
  tags: string[];
  removed: string[];
}

export interface FolderMapping {
  auto_tag_level: string;
  rules: FolderRule[];
}

/** One folder rule as the wizard sends it (carries the sub-library name). */
export interface FolderRuleInput {
  rel_path: string;
  mode: "keep" | "sublib" | "exclude";
  tags: string[];
  removed: string[];
  sub_name?: string | null;
}

export interface FolderMappingInput {
  auto_tag_level: string;
  rules: FolderRuleInput[];
  /** Optional rename of the parent library, applied before the mapping. */
  name?: string;
}

export interface LibraryCoverage {
  unhashed: number;
  unembedded: number;
  metrics_present: { id: string; count: number }[];
}

export interface QualityMetric {
  id: string;
  label: string;
}

/** One scan of the Index chain, as described by the backend catalogue. */
export interface IndexStep {
  key: string;
  label: string;
  short: string;
  models: string;
  cost: string;
  images_only: boolean;
  description: string;
  enabled: boolean;
}

export interface StepCount {
  done: number;
  total: number;
}

export type StepCounts = Record<string, StepCount>;

export interface IndexLibraryStatus {
  id: number;
  name: string;
  steps: StepCounts;
}

export interface IndexStatus {
  steps: IndexStep[];
  totals: StepCounts;
  libraries: IndexLibraryStatus[];
}

export interface LookalikeMember {
  key: string;
  name: string;
  thumb: string;
  quality: number | null;
  is_best: boolean;
}

export interface LookalikeResult {
  hashed_count: number;
  groups: { members: LookalikeMember[] }[];
}

export interface SystemDatabase {
  path: string;
  size_bytes: number;
  counts: Record<string, number>;
  backups: { filename: string; size_bytes: number }[];
}

export interface SystemRuntime {
  python: string;
  vram_total_gb: number | null;
  vram_used_gb: number | null;
  cuda: string | null;
  gpu: string | null;
  thumbnail_cache_bytes: number;
}

/** One Database-cleanup category's live orphan count and reclaimable size. */
export interface CleanupCount {
  count: number;
  bytes: number;
}

/** The four cleanup categories, keyed as the System view lists them. */
export interface CleanupReport {
  media: CleanupCount;
  captions: CleanupCount;
  patches: CleanupCount;
  thumbs: CleanupCount;
}

export type CleanupCategory = keyof CleanupReport;

/** The outcome of purging one cleanup category. */
export interface CleanupResult {
  purged: number;
  bytes: number;
  vacuumed: boolean;
}

export interface DbQueryResult {
  headers: string[];
  rows: (string | number | null)[][];
}

export interface AutobuildStudioConfig {
  framing_presets: { key: string; label: string }[];
  metrics: { id: string; label: string }[];
  libraries: { id: number; name: string }[];
  unhashed: number;
  unembedded: number;
}

/** One reason chip shown under a pick's name (icon + short value + tip). */
export interface AutobuildReason {
  icon: string;
  label: string;
  title: string;
}

/** A pick's "borderline" flag — the material of the triage queue. */
export interface AutobuildFlag {
  kind: string;
  why: string;
}

export interface AutobuildPick {
  media_id: number;
  name: string;
  is_video: boolean;
  favorite: boolean;
  width: number | null;
  height: number | null;
  quality: number | null;
  subject: number;
  gain: number;
  bucket: string;
  cluster: number | null;
  reasons: AutobuildReason[];
  flag: AutobuildFlag | null;
  xy: [number, number] | null;
}

export interface AutobuildCluster {
  id: number;
  color: string;
  label: string;
  top_tags: string[];
  count: number;
  pct: number;
  media_ids: number[];
}

export interface AutobuildZone {
  x: number;
  y: number;
  r: number;
  count: number;
  top_tags: string[];
  why: string;
}

export interface AutobuildMapPoint {
  id: number;
  name: string;
  xy: [number, number] | null;
  role: "pick" | "candidate" | "seed";
  cluster: number | null;
}

export interface AutobuildStudioPreview {
  picks: AutobuildPick[];
  eligible: number;
  pool_size: number;
  matched: number;
  requested: number;
  shortfall: number;
  pref_before: number;
  pref_after: number;
  clusters: AutobuildCluster[];
  zones: AutobuildZone[];
  map: {
    width: number;
    height: number;
    points: AutobuildMapPoint[];
    zones: AutobuildZone[];
  };
  dominant_tag: { name: string; share: number } | null;
  semantic_available: boolean;
  grade: string;
  score: number | null;
  pillars: {
    quality: number | null;
    diversity: number | null;
    hygiene: number;
    duplicates: number;
  };
  framing: {
    bucket: string;
    base: number;
    added: number;
    total: number;
    share: number;
    under: boolean;
  }[];
  size: {
    base: number;
    picked: number;
    total: number;
    min: number;
    max: number;
    percent: number;
    over: boolean;
  };
  advice: { tone: string; text: string }[];
}

export interface AutobuildSuggestedTag {
  name: string;
  pct: number;
}

export interface AutobuildNeighbor {
  media_id: number;
  name: string;
  cosine: number;
  quality: number | null;
  why: string;
}

export interface AutobuildUpgrade {
  out_media_id: number;
  out_name: string;
  out_quality: number | null;
  in_media_id: number;
  in_name: string;
  in_quality: number | null;
  reason: string;
  gain: number;
}

export interface AutobuildUpgrades {
  dataset_id: number;
  upgrades: AutobuildUpgrade[];
}

export interface JobSnapshot {
  id: string;
  type: string;
  name: string;
  sub: string;
  state: "queued" | "running" | "done" | "error" | "stopped";
  done: number;
  total: number;
  pct: number;
  error: string;
  /** Per-item problems the run survived, capped server-side. */
  warnings: string[];
  /** How many there really were (``warnings`` may be truncated). */
  warning_count: number;
  created_at: number;
  updated_at: number;
  /** Wall-clock start of the running phase (epoch seconds); null while queued. */
  started_at: number | null;
  /** Estimated seconds remaining, or null when not yet computable. */
  eta_seconds: number | null;
}

export interface ReviewVerdict {
  status: string;
  issues: { code: string; detail: string }[];
}

// -- SigLIP grounding ---------------------------------------------------------

/** The claim kinds the LLM tags its decomposition with. */
export type ClaimKind = "object" | "attribute" | "scene" | "count" | "spatial";

/**
 * The kinds SigLIP scores unreliably — it judges presence, not arithmetic
 * nor geometry. Their bars are shown amber and labelled indicative.
 * Mirrors `src.caption_claims.UNRELIABLE_KINDS`.
 */
export const UNRELIABLE_KINDS: ClaimKind[] = ["count", "spatial"];

/** One atomic caption claim with the score SigLIP gave it. */
export interface GroundedClaim {
  id: number;
  position: number;
  text: string;
  kind: ClaimKind;
  score: number;
  rejected: boolean;
}

/** A caption's stored grounding run. */
export interface CaptionGrounding {
  id: number;
  revision_id: number;
  model_id: string;
  created_at: string;
  claims: GroundedClaim[];
}

export interface CaptionGroundingResponse {
  grounding: CaptionGrounding | null;
  model_id: string;
  threshold: number;
}

/** A media tag with its SigLIP score (null when never grounded). */
export interface GroundedTag extends TagChipData {
  score: number | null;
}

export interface TagGroundingResponse {
  tags: GroundedTag[];
  model_id: string;
  threshold: number;
}

/**
 * One element of the modal, whichever mode it is in: a caption claim or a
 * tag, carrying the heat grid rebuilt for it. `heat` is base64 — one byte
 * per patch, row-major, `side * side` of them.
 */
export interface HeatElement {
  id: number;
  text?: string;
  name?: string;
  kind?: ClaimKind;
  score: number;
  rejected?: boolean;
  heat: string | null;
  side: number;
}

export interface HeatResponse {
  model_id: string;
  elements: HeatElement[];
}

export interface GroundingConfig {
  model_id: string;
  threshold_caption: number;
  threshold_tags: number;
  tag_prompt: string;
  /** Default claim-decomposition model; "" means "use the loaded VLM". */
  claim_model: string;
}

/** One SigLIP size tier as the Settings tab renders it. */
export interface GroundingSize {
  label: string;
  params: string;
  vram: string;
  resolutions: number[];
}

// -- Datasets → Quality report ------------------------------------------------

/** One labelled line inside a pillar card. */
export interface ReportRow {
  label: string;
  value: string;
  tone: "ok" | "warn" | "danger" | "info" | "muted";
}

/** One scored pillar of the report (quality / diversity / hygiene). */
export interface ReportPillar {
  key: string;
  label: string;
  score: number | null;
  detail: string;
  rows: ReportRow[];
}

/** One bar of the quality histogram. */
export interface ReportBucket {
  label: string;
  count: number;
  midpoint: number;
}

/** One dot of the diversity map. */
export interface ReportMapPoint {
  id: number;
  name: string;
  x: number;
  y: number;
  cluster: number;
  outlier: boolean;
  near_dup: boolean;
  quality: number | null;
  width: number | null;
  height: number | null;
}

/** One suggestion of the recommendations card. */
export interface ReportRecommendation {
  head: string;
  body: string;
}

/** The per-image metric block of a near-duplicate comparison. */
export interface PairMetrics {
  id: number;
  name: string;
  width: number | null;
  height: number | null;
  quality: number | null;
  scores: Record<string, number>;
  sharpness: number | null;
  clipping: number | null;
  cleanliness: number | null;
}

export interface NearDupDetail {
  similarity: number;
  threshold: number;
  source: string;
  best_id: number;
  loser_id: number;
  metrics: PairMetrics[];
}

export interface LowQualityDetail {
  quality: number;
  mean: number | null;
  floor: number;
  scores: Record<string, number>;
  sharpness: number | null;
  clipping: number | null;
  cleanliness: number | null;
  gradient_share: number;
}

export interface OutlierDetail {
  distance: number;
  threshold: number;
  neighbors: { id: number; name: string; distance: number }[];
}

export interface CaptionDetail {
  text: string;
  codes: { code: string; detail: string }[];
  phrase: string | null;
}

interface IssueBase {
  key: string;
  media_ids: number[];
  names: string[];
  reason: string;
  metric: string;
  value: number;
  impact: number;
  fingerprint: string;
}

/** One actionable finding; ``kind`` narrows the inspector payload. */
export type ReportIssue =
  | (IssueBase & { kind: "near_dup"; detail: NearDupDetail })
  | (IssueBase & { kind: "low_quality"; detail: LowQualityDetail })
  | (IssueBase & { kind: "outlier"; detail: OutlierDetail })
  | (IssueBase & { kind: "caption"; detail: CaptionDetail });

export type IssueKind = ReportIssue["kind"];

/** The stored evaluation of one dataset. */
export interface DatasetReport {
  total: number;
  images: number;
  videos: number;
  missing: number;
  favorites: number;
  overall: number | null;
  grade: string;
  verdict: string;
  summary: string;
  weights: Record<string, number>;
  scorers: string[];
  pillars: ReportPillar[];
  distribution: ReportBucket[];
  map_points: ReportMapPoint[];
  clusters: number;
  spread: number;
  issues: ReportIssue[];
  recommendations: ReportRecommendation[];
  framing: [string, number, number, number | null][];
}

/** One toggle chip of the report toolbar. */
export interface ScorerChip {
  id: string;
  label: string;
  kind: "iqa" | "embedding";
  default: boolean;
  /** False when the index step feeding this scorer is off on this machine. */
  available: boolean;
}

export type ResolutionKind = "removed" | "ignored" | "recaptioned";

export interface IssueResolution {
  resolution: ResolutionKind;
  fingerprint: string;
  note: string;
  resolved_at: string;
}

export interface DatasetReportResponse {
  scorer_catalogue: ScorerChip[];
  resolutions: Record<string, IssueResolution>;
  report: DatasetReport | null;
  created_at: string | null;
  duration_s: number;
  scorers: string[];
  caption_type: string;
}

// -- Watermark Lab ----------------------------------------------------------

/** A zone is either still-detected or erased by a patch (v2 — no review). */
export type WatermarkStatus = "detected" | "patched";

/** The three self-contained Lab tabs. */
export type WatermarkTab = "media" | "watermarked" | "patched";

export type WatermarkDetector = "owlv2" | "yolo";

export interface WatermarkBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WatermarkZone {
  id: number;
  media_id: number;
  box: WatermarkBox;
  status: WatermarkStatus;
  model: string | null;
  seed: number | null;
  score: number | null;
  patch_sha: string | null;
  detector: string | null;
  /** The OWLv2 text query that matched (null for YOLO/manual). */
  query: string | null;
  dilate_px: number;
  prompt: string | null;
  edit_ms: number | null;
  created_at: string;
  updated_at: string;
}

/** One media's watermark state, as the review panel and the /{id} route see it. */
export interface WatermarkMedia {
  media_id: number;
  status: WatermarkStatus | null;
  /** The patches were baked into the source file (reversible). */
  flattened: boolean;
  zones: WatermarkZone[];
  zone_count: number;
  score_min: number | null;
  models: string[];
  detectors: string[];
  name: string;
  thumb: string;
  is_video: boolean;
}

/** One inventory card: the media grid fields plus its watermark state. */
export interface WatermarkInventoryItem {
  media_id: number;
  key: string;
  name: string;
  thumb: string;
  width: number | null;
  height: number | null;
  quality: number | null;
  favorite: boolean;
  is_video: boolean;
  status: WatermarkStatus | null;
  flattened: boolean;
  zone_count: number;
  score_min: number | null;
  detectors: string[];
  models: string[];
  zones: WatermarkZone[];
}

export type WatermarkModel = "9b" | "4b";
export type WatermarkPrecision = "std" | "fp8" | "nvfp4";
export type WatermarkSource = "hf" | "local";
export type WatermarkResSide = "long" | "short";

export interface WatermarkTextEncoder {
  source: WatermarkSource;
  version: string;
  path: string;
}

export interface WatermarkPrefs {
  detector: WatermarkDetector;
  owlv2_model: string;
  owlv2_queries: string[];
  owlv2_confidence: number;
  confidence_min: number;
  model: WatermarkModel;
  precision: WatermarkPrecision;
  kv: boolean;
  source: WatermarkSource;
  local_model_path: string;
  text_encoder: WatermarkTextEncoder;
  prompt: string;
  max_res: number;
  res_side: WatermarkResSide;
  dilate_px: number;
  tag_cleanup: boolean;
  tags_to_remove: string[];
  compare_mode: "slider" | "hover" | "zone";
  yolo_model: string;
}

export interface WatermarkConfig {
  prefs: WatermarkPrefs;
  media_total: number;
  owlv2_model_id: string;
  owlv2_models: { id: string; label: string }[];
  yolo_available: boolean;
  yolo_models: string[];
  yolo_models_dir: string;
  flux_available: boolean;
  flux_repo: string | null;
  flux_label: string;
  encoder_repo: string;
}

export interface WatermarkInventory {
  items: WatermarkInventoryItem[];
  counts: Record<WatermarkTab, number>;
  total: number;
  tab: WatermarkTab;
}
