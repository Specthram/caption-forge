/**
 * TanStack Query hooks — the front-end's server-state layer.
 *
 * Queries cache and page the read routes; mutations post the write routes
 * and invalidate the affected caches. UI/interaction state lives in the
 * Zustand stores, never here.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";
import { api } from "./client";
import type {
  AutobuildNeighbor,
  AutobuildStudioConfig,
  AutobuildStudioPreview,
  AutobuildSuggestedTag,
  AutobuildUpgrade,
  AutobuildUpgrades,
  CaptionGrid,
  CaptionGroundingResponse,
  CaptionScoreReport,
  ComposeCandidatesPage,
  ComposePreview,
  Crop,
  CropRatio,
  CropRect,
  CropSource,
  DatasetInfo,
  FileListing,
  FolderListing,
  FolderMapping,
  FolderMappingInput,
  FolderTree,
  DatasetReportResponse,
  DbQueryResult,
  GroundingConfig,
  HeatResponse,
  IndexStatus,
  LibraryCoverage,
  HfRepoDetect,
  LibraryGridPage,
  LibrarySource,
  LookalikeResult,
  MediaDetail,
  MediaFullDetail,
  MediaGridPage,
  NavCounts,
  ModelInfo,
  ModelProfile,
  ModelStatus,
  ProfileDetect,
  ProfilesResponse,
  PromptsResponse,
  QualityMetric,
  ResolutionKind,
  ReviewCounts,
  ReviewFinding,
  ReviewFindingsResponse,
  ReviewRule,
  CleanupCategory,
  CleanupCount,
  CleanupReport,
  CleanupResult,
  SystemDatabase,
  SystemRuntime,
  TagCategory,
  TagGroundingResponse,
  TagListPage,
  TaggerModels,
  Triggerword,
  WatermarkBox,
  WatermarkConfig,
  WatermarkInventory,
  WatermarkMedia,
  WatermarkPrefs,
  WatermarkTab,
} from "./types";

export interface GridParams {
  dataset_id: number;
  caption_type: string;
  review_filter: string;
  offset: number;
  limit: number;
  quality_metric: string;
}

// -- Queries ----------------------------------------------------------------

export function useCaptionTypes() {
  return useQuery({
    queryKey: ["caption-types"],
    queryFn: () => api.get<{ types: string[] }>("/captions/types"),
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.get<{ datasets: DatasetInfo[] }>("/datasets"),
  });
}

/** The sidebar's per-section totals (three COUNT queries, no paging). */
export function useNavCounts() {
  return useQuery({
    queryKey: ["nav-counts"],
    queryFn: () => api.get<NavCounts>("/system/counts"),
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => api.get<{ models: ModelInfo[] }>("/models"),
  });
}

export function useModelStatus() {
  return useQuery({
    queryKey: ["model-status"],
    queryFn: () => api.get<ModelStatus>("/models/status"),
    refetchInterval: 4000,
  });
}

export function usePrompts(modelType: string | null) {
  return useQuery({
    queryKey: ["prompts", modelType],
    queryFn: () =>
      api.get<PromptsResponse>("/prompts", { model_type: modelType }),
    enabled: !!modelType,
  });
}

export function useSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: () => api.get<Record<string, unknown>>("/settings"),
  });
}

/**
 * Whether the SigLIP grounding feature is enabled (Settings → Beta features).
 * Defaults to true before settings load or when the key is absent, so the
 * feature only ever hides on an explicit opt-out.
 */
export function useGroundingEnabled(): boolean {
  const settings = useSettings();
  return (settings.data?.grounding_enabled ?? true) as boolean;
}

/**
 * Whether the caption editor auto-saves a dirty draft after a typing pause
 * (Settings → Captioning). Defaults to true before settings load or when the
 * key is absent, so autosave only ever stops on an explicit opt-out.
 */
export function useAutosaveEnabled(): boolean {
  const settings = useSettings();
  return (settings.data?.autosave_enabled ?? true) as boolean;
}

export function useSaveSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (settings: Record<string, unknown>) =>
      api.post("/settings", { settings }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["settings"] });
      client.invalidateQueries({ queryKey: ["caption-types"] });
      // The machine's index-scan toggles grey out steps and scorer chips
      // app-wide, so both catalogues are stale after a save.
      client.invalidateQueries({ queryKey: ["index-status"] });
      client.invalidateQueries({ queryKey: ["dataset-report"] });
      // A save can change the watermark models directory, so the Lab's YOLO
      // `.pt` list (and model availability) must be refetched too.
      client.invalidateQueries({ queryKey: ["watermark-config"] });
    },
  });
}

// -- System -----------------------------------------------------------------

export function useSystemDatabase() {
  return useQuery({
    queryKey: ["system-database"],
    queryFn: () => api.get<SystemDatabase>("/system/database"),
  });
}

export function useSystemRuntime() {
  return useQuery({
    queryKey: ["system-runtime"],
    queryFn: () => api.get<SystemRuntime>("/system/runtime"),
  });
}

export function useBackupNow() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<{ filename: string }>("/system/backup"),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["system-database"] }),
  });
}

export function useRestoreBackup() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (filename: string) =>
      api.post("/system/restore", { filename }),
    onSuccess: () => client.invalidateQueries(),
  });
}

export function usePurge() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/system/purge"),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["model-status"] }),
  });
}

export function useRestart() {
  return useMutation({
    mutationFn: () => api.post("/system/restart"),
  });
}

export function useCleanupReport() {
  return useQuery({
    queryKey: ["system-cleanup"],
    queryFn: () => api.get<CleanupReport>("/system/cleanup"),
  });
}

/**
 * One cleanup category's live ``{count, bytes}``. Each System-view row
 * fetches its own report so a slow scan (patch orphans, big cache trees)
 * shows its own loader instead of stalling the whole block.
 */
export function useCleanupCategory(category: CleanupCategory) {
  return useQuery({
    queryKey: ["system-cleanup", category],
    queryFn: () => api.get<CleanupCount>(`/system/cleanup/${category}`),
  });
}

export function useCleanupPurge() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (category: CleanupCategory) =>
      api.post<CleanupResult>(`/system/cleanup/${category}`),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["system-cleanup"] });
      client.invalidateQueries({ queryKey: ["system-database"] });
      client.invalidateQueries({ queryKey: ["system-runtime"] });
    },
  });
}

export function useDbTables() {
  return useQuery({
    queryKey: ["db-tables"],
    queryFn: () => api.get<{ tables: string[] }>("/system/db/tables"),
  });
}

export function useDbQuery() {
  return useMutation({
    mutationFn: (sql: string) =>
      api.post<DbQueryResult>("/system/db/query", { sql }),
  });
}

export function useDbDeleteRow() {
  return useMutation({
    mutationFn: (vars: { table: string; row_id: number }) =>
      api.post<{ message: string }>("/system/db/delete-row", vars),
  });
}

export function useTagCategories() {
  return useQuery({
    queryKey: ["tag-categories"],
    queryFn: () =>
      api.get<{ categories: TagCategory[] }>("/tags/categories"),
  });
}

// -- Tags admin -------------------------------------------------------------

export function useTagsList(
  categoryId: number | null,
  query: string,
  limit: number,
) {
  return useQuery({
    queryKey: ["tags-list", categoryId, query, limit],
    queryFn: () =>
      api.get<TagListPage>("/tags/list", {
        category_id: categoryId,
        query,
        limit,
        offset: 0,
      }),
    enabled: categoryId != null,
    placeholderData: (prev) => prev,
  });
}

function useTagsInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["tag-categories"] });
    client.invalidateQueries({ queryKey: ["tags-list"] });
    client.invalidateQueries({ queryKey: ["nav-counts"] });
  };
}

export function useCreateCategory() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (vars: { name: string; color: string }) =>
      api.post<{ id: number }>("/tags/categories", vars),
    onSuccess: invalidate,
  });
}

export function useReorderCategories() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (orderedIds: number[]) =>
      api.post("/tags/categories/reorder", { ordered_ids: orderedIds }),
    onSuccess: invalidate,
  });
}

export function useUpdateCategory() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; name?: string; color?: string }) =>
      api.post(`/tags/categories/${vars.id}`, {
        name: vars.name ?? null,
        color: vars.color ?? null,
      }),
    onSuccess: invalidate,
  });
}

export function useDeleteCategory() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del(`/tags/categories/${id}`),
    onSuccess: invalidate,
  });
}

export function useCreateTag() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (vars: { name: string; category_id: number }) =>
      api.post<{ id: number }>("/tags", vars),
    onSuccess: invalidate,
  });
}

/** Reuse a tag by name, else create it in the "Uncategorized" pen. */
export function useCreateUncategorizedTag() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (name: string) =>
      api.post<{ id: number; name: string }>("/tags/uncategorized", { name }),
    onSuccess: invalidate,
  });
}

export function useRenameTag() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; name: string }) =>
      api.post(`/tags/${vars.id}/rename`, { name: vars.name }),
    onSuccess: invalidate,
  });
}

export function useDeleteTag() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del(`/tags/${id}`),
    onSuccess: invalidate,
  });
}

/** Move a tag to another category (drag-and-drop; merges on name clash). */
export function useMoveTag() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; category_id: number }) =>
      api.post<{ id: number }>(`/tags/${vars.id}/move`, {
        category_id: vars.category_id,
      }),
    onSuccess: invalidate,
  });
}

/** Merge tags duplicated across categories into their real one. */
export function useDedupeTags() {
  const invalidate = useTagsInvalidator();
  return useMutation({
    mutationFn: () =>
      api.post<{ names: number; removed: number }>("/tags/dedupe"),
    onSuccess: invalidate,
  });
}

/** Which of the given tag names already exist (colours the wizard's auto tags). */
export function useExistingTagNames(names: string[], enabled: boolean) {
  const sorted = [...new Set(names)].sort();
  return useQuery({
    queryKey: ["existing-tags", sorted],
    queryFn: () =>
      api.post<{ existing: string[] }>("/tags/existing", { names: sorted }),
    enabled: enabled && sorted.length > 0,
    placeholderData: (prev) => prev,
  });
}

export function useTagSearch(query: string, enabled: boolean) {
  return useQuery({
    queryKey: ["tag-search", query],
    queryFn: () =>
      api.get<{
        tags: { id: number; name: string; category_id: number }[];
      }>("/tags/search", { query }),
    enabled,
  });
}

export function useCaptionGrid(params: GridParams, enabled: boolean) {
  return useQuery({
    queryKey: ["caption-grid", params],
    queryFn: () =>
      api.get<CaptionGrid>(
        "/captions/grid",
        params as unknown as Record<string, unknown>,
      ),
    enabled,
    placeholderData: (prev) => prev,
  });
}

export function useMediaDetail(
  key: string | null,
  datasetId: number | null,
  captionType: string,
  qualityMetric: string,
) {
  return useQuery({
    queryKey: ["media-detail", key, datasetId, captionType, qualityMetric],
    queryFn: () =>
      api.get<MediaDetail>(`/captions/media/${key}`, {
        dataset_id: datasetId,
        caption_type: captionType,
        quality_metric: qualityMetric,
      }),
    enabled: !!key && datasetId != null,
  });
}

// -- SigLIP grounding (reads) ------------------------------------------------

/** The configured checkpoint, thresholds and tag pre-prompt. */
export function useGroundingConfig() {
  return useQuery({
    queryKey: ["grounding-config"],
    queryFn: () => api.get<GroundingConfig>("/grounding/config"),
    staleTime: 60_000,
  });
}

/** A caption's stored claims and their scores (no GPU, no job). */
export function useCaptionGrounding(
  key: string | null,
  datasetId: number | null,
  captionType: string,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["caption-grounding", key, datasetId, captionType],
    queryFn: () =>
      api.get<CaptionGroundingResponse>("/grounding/caption", {
        key,
        dataset_id: datasetId,
        caption_type: captionType,
      }),
    enabled: enabled && !!key && datasetId != null,
  });
}

/** A media's tags with their stored scores (null for the ungrounded ones). */
export function useTagGrounding(key: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["tag-grounding", key],
    queryFn: () => api.get<TagGroundingResponse>("/grounding/tags", { key }),
    enabled: enabled && !!key,
  });
}

/**
 * A finished job's payload.
 *
 * The heat maps are the *output* of their job, not a side effect, and are
 * too bulky for the progress WebSocket. The caller passes the job id it got
 * back from a heat mutation plus whether the WebSocket has seen that job
 * reach `done`; only then is the result fetched.
 */
export function useJobResult<T>(jobId: string | null, finished: boolean) {
  return useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () =>
      api.get<{ state: string; result: T }>(`/jobs/${jobId}/result`),
    enabled: !!jobId && finished,
    staleTime: Infinity,
  });
}

// -- Mutation helpers -------------------------------------------------------

/** Invalidate every cache that a caption/media change can affect. */
function useCaptionInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["caption-grid"] });
    client.invalidateQueries({ queryKey: ["media-detail"] });
  };
}

// -- Caption mutations ------------------------------------------------------

interface TargetParams {
  key: string;
  dataset_id: number;
  caption_type: string;
}

export function useSaveCaption() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (
      vars: TargetParams & {
        content: string;
        scope: string;
        amend?: boolean;
      },
    ) =>
      api.post(
        `/captions/media/${vars.key}/caption`,
        { content: vars.content, scope: vars.scope, amend: vars.amend ?? false },
        { dataset_id: vars.dataset_id, caption_type: vars.caption_type },
      ),
    onSuccess: invalidate,
  });
}

export function useSelectRevision() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (
      vars: TargetParams & { revision_id: number | null },
    ) =>
      api.post(
        `/captions/media/${vars.key}/revision`,
        { revision_id: vars.revision_id },
        { dataset_id: vars.dataset_id, caption_type: vars.caption_type },
      ),
    onSuccess: invalidate,
  });
}

export function useSetRepeats() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (vars: { key: string; dataset_id: number; repeats: number }) =>
      api.post(
        `/captions/media/${vars.key}/repeats`,
        { repeats: vars.repeats },
        { dataset_id: vars.dataset_id },
      ),
    onSuccess: invalidate,
  });
}

export function useSetHidden() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (vars: { key: string; dataset_id: number; hidden: boolean }) =>
      api.post(
        `/captions/media/${vars.key}/hidden`,
        { hidden: vars.hidden },
        { dataset_id: vars.dataset_id },
      ),
    onSuccess: invalidate,
  });
}

export function useAddTag() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (vars: {
      key: string;
      tag_id?: number;
      name?: string;
      category_id?: number;
    }) =>
      api.post(`/captions/media/${vars.key}/tags/add`, {
        tag_id: vars.tag_id ?? null,
        name: vars.name ?? null,
        category_id: vars.category_id ?? null,
      }),
    onSuccess: invalidate,
  });
}

export function useRemoveTag() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (vars: { key: string; tag_id: number }) =>
      api.post(`/captions/media/${vars.key}/tags/remove`, {
        tag_id: vars.tag_id,
      }),
    onSuccess: invalidate,
  });
}

export function useRemoveFromDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: TargetParams) =>
      api.del(`/captions/media/${vars.key}`, {
        dataset_id: vars.dataset_id,
        caption_type: vars.caption_type,
      }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["caption-grid"] });
      client.invalidateQueries({ queryKey: ["datasets"] });
    },
  });
}

// -- Model mutations --------------------------------------------------------

export function useLoadModel() {
  return useMutation({
    mutationFn: (name: string) =>
      api.post<{ job_id: string }>("/models/load", { name }),
  });
}

export function useUnloadModel() {
  return useMutation({
    mutationFn: () => api.post<{ job_id: string }>("/models/unload"),
  });
}

// -- Model profiles -----------------------------------------------------------

export function useProfiles() {
  return useQuery({
    queryKey: ["profiles"],
    queryFn: () => api.get<ProfilesResponse>("/profiles"),
    // loaded_id follows load/unload/judge-swap jobs — poll like model-status.
    refetchInterval: 4000,
  });
}

/** Fields accepted by profile create/update (all optional server-side). */
export type ProfileFields = Partial<Omit<ModelProfile, "id">>;

function useProfilesInvalidator() {
  const client = useQueryClient();
  return useCallback(
    () => client.invalidateQueries({ queryKey: ["profiles"] }),
    [client],
  );
}

export function useCreateProfile() {
  const invalidate = useProfilesInvalidator();
  return useMutation({
    mutationFn: (vars: ProfileFields & { role?: "caption" | "judge" }) =>
      api.post<ModelProfile>("/profiles", vars),
    onSuccess: invalidate,
  });
}

export function useUpdateProfile() {
  const invalidate = useProfilesInvalidator();
  return useMutation({
    mutationFn: ({ id, ...fields }: ProfileFields & { id: number }) =>
      api.put<ModelProfile>(`/profiles/${id}`, fields),
    onSuccess: invalidate,
  });
}

export function useDeleteProfile() {
  const invalidate = useProfilesInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del<{ ok: boolean }>(`/profiles/${id}`),
    onSuccess: invalidate,
  });
}

export function useSelectProfile() {
  const invalidate = useProfilesInvalidator();
  return useMutation({
    mutationFn: (vars: { role: "caption" | "judge"; id: number }) =>
      api.post<{ ok: boolean }>("/profiles/select", vars),
    onSuccess: invalidate,
  });
}

/** Swap VRAM to a profile's weights (a job — progress via /ws/jobs). */
export function useLoadProfile() {
  return useMutation({
    mutationFn: (id: number) =>
      api.post<{ job_id: string }>(`/profiles/${id}/load`),
  });
}

/**
 * Remember the prompt preset last used with a profile. Fire-and-forget (no
 * query invalidation): the panel tracks the live selection locally, this only
 * persists it so the profile reopens on the same preset next session.
 */
export function useRememberPrompt() {
  return useMutation({
    mutationFn: (vars: { id: number; title: string }) =>
      api.post(`/profiles/${vars.id}/prompt`, { title: vars.title }),
  });
}

/** Re-run type/mmproj auto-detection for a picked weights file. */
export function useDetectProfileFile() {
  return useMutation({
    mutationFn: (vars: { dir: string; file: string }) =>
      api.post<ProfileDetect>("/profiles/detect", vars),
  });
}

/** Guess family/format/name from a Hugging Face repo id. */
export function useDetectHfRepo() {
  return useMutation({
    mutationFn: (repo: string) =>
      api.get<HfRepoDetect>("/profiles/detect-hf", { repo }),
  });
}

/** One folder listing of the profile editor's weights/mmproj picker. */
export function useBrowseModelFiles(path: string) {
  return useQuery({
    queryKey: ["profile-browse", path],
    queryFn: () => api.get<FileListing>("/profiles/browse", { path }),
  });
}

// -- Generation -------------------------------------------------------------

export interface GenerateVars {
  dataset_id: number;
  caption_type: string;
  media_ids: number[] | null;
  exclude_ids: number[] | null;
  prompt: string;
  /** Captioner profile: generation params come from it (lazy-swapped). */
  profile_id: number | null;
  seed: number | null;
  review_after: boolean;
  review_judge_profile_id?: number | null;
  ground_after: boolean;
  /** Off = only caption media whose caption is still empty. */
  recaption: boolean;
  /** On = free the VRAM when the job ends (default: model stays loaded). */
  unload_after?: boolean;
}

export function useGenerate() {
  return useMutation({
    mutationFn: (vars: GenerateVars) =>
      api.post<{ job_id: string }>("/captions/generate", vars),
  });
}

// -- Review -------------------------------------------------------------------

/** Run the integrity heuristics on one caption (pure text, no model, no job). */
export function useIntegrityReview() {
  const invalidate = useCaptionInvalidator();
  return useMutation({
    mutationFn: (vars: TargetParams) => api.post("/review/integrity", vars),
    onSuccess: invalidate,
  });
}

// -- Rule-based review (rules, run, findings queue) ---------------------------

/** Drop every review cache a rule/finding change can affect. */
function useReviewInvalidator() {
  const client = useQueryClient();
  return (datasetId?: number) => {
    client.invalidateQueries({ queryKey: ["review-findings"] });
    client.invalidateQueries({ queryKey: ["review-counts"] });
    if (datasetId != null) {
      client.invalidateQueries({ queryKey: ["review-rules", datasetId] });
    }
    // Accepting a finding writes a new revision → the grids/panels are stale.
    client.invalidateQueries({ queryKey: ["caption-grid"] });
    client.invalidateQueries({ queryKey: ["media-detail"] });
  };
}

/** A dataset's review rules (the builtin presets are seeded on first read). */
export function useReviewRules(datasetId: number | null) {
  return useQuery({
    queryKey: ["review-rules", datasetId],
    queryFn: () =>
      api.get<{ rules: ReviewRule[] }>("/review/rules", {
        dataset_id: datasetId,
      }),
    enabled: datasetId != null,
  });
}

/** The review queue for a dataset (optionally filtered by status). */
export function useReviewFindings(
  datasetId: number | null,
  status: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["review-findings", datasetId, status],
    queryFn: () =>
      api.get<ReviewFindingsResponse>("/review/findings", {
        dataset_id: datasetId,
        status,
      }),
    enabled: enabled && datasetId != null,
    placeholderData: (prev) => prev,
  });
}

/** The pending/accepted/rejected counts (drives the tab badge everywhere). */
export function useReviewCounts(datasetId: number | null) {
  return useQuery({
    queryKey: ["review-counts", datasetId],
    queryFn: () =>
      api.get<ReviewCounts>("/review/counts", { dataset_id: datasetId }),
    enabled: datasetId != null,
  });
}

export function useCreateReviewRule() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: {
      dataset_id: number;
      text: string;
      needs_image: boolean;
    }) => api.post<{ rule: ReviewRule }>("/review/rules", vars),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

export function useUpdateReviewRule() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: {
      id: number;
      dataset_id: number;
      enabled?: boolean;
      text?: string;
    }) =>
      api.patch<{ rule: ReviewRule }>(`/review/rules/${vars.id}`, {
        enabled: vars.enabled ?? null,
        text: vars.text ?? null,
      }),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

export function useDeleteReviewRule() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; dataset_id: number }) =>
      api.del(`/review/rules/${vars.id}`),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

/** Enqueue a review run; the caller watches the returned job id. */
export function useRunReview() {
  return useMutation({
    mutationFn: (vars: {
      dataset_id: number;
      caption_type: string;
      media_ids: number[] | null;
      judge_profile_id: number | null;
      scope: string;
      rule_ids?: number[] | null;
      seed?: number | null;
      unload_after?: boolean;
    }) => api.post<{ job_id: string }>("/review/run", vars),
  });
}

/** Accept (writes a new revision) or reject one finding. */
export function useDecideFinding() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: {
      id: number;
      action: "accept" | "reject";
      caption?: string | null;
    }) =>
      api.post<{ finding: ReviewFinding }>(
        `/review/findings/${vars.id}/decide`,
        { action: vars.action, caption: vars.caption ?? null },
      ),
    onSuccess: () => invalidate(),
  });
}

/** Undo a decision: restore the caption and reopen the finding. */
export function useUndoFinding() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (id: number) =>
      api.post<{ finding: ReviewFinding }>(`/review/findings/${id}/undo`),
    onSuccess: () => invalidate(),
  });
}

/** Reject every pending finding of the dataset (captions untouched). */
export function useRejectAll() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: { dataset_id: number }) =>
      api.post<{ rejected: number }>("/review/findings/reject_all", vars),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

/** Delete the decided findings (the history); pending ones stay. */
export function useClearReviewHistory() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: { dataset_id: number }) =>
      api.post<{ cleared: number }>("/review/findings/clear_history", vars),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

/** Accept every safe finding, or every pending finding of one rule. */
export function useDecideBulk() {
  const invalidate = useReviewInvalidator();
  return useMutation({
    mutationFn: (vars: { dataset_id: number; rule_id?: number | null }) =>
      api.post<{ accepted: number }>("/review/findings/decide_bulk", {
        dataset_id: vars.dataset_id,
        rule_id: vars.rule_id ?? null,
      }),
    onSuccess: (_data, vars) => invalidate(vars.dataset_id),
  });
}

// -- SigLIP grounding (jobs + verdicts) ---------------------------------------

/** Ground one caption. Needs a VLM loaded — it decomposes before scoring. */
export function useGroundCaption() {
  return useMutation({
    mutationFn: (vars: TargetParams) =>
      api.post<{ job_id: string }>("/grounding/caption", vars),
  });
}

/** Reference-free score one caption with every encoder (a background job). */
export function useScoreCaption() {
  return useMutation({
    mutationFn: (vars: TargetParams) =>
      api.post<{ job_id: string }>("/caption-score/caption", vars),
  });
}

/** Reference-free score one media's tags (a background job). */
export function useScoreMediaTags() {
  return useMutation({
    mutationFn: (vars: { key: string }) =>
      api.post<{ job_id: string }>("/caption-score/tags", vars),
  });
}

/** Reference-free score every caption of a dataset (a background job). */
export function useScoreDataset() {
  return useMutation({
    mutationFn: (vars: { dataset_id: number; caption_type: string }) =>
      api.post<{ job_id: string }>("/caption-score/dataset", vars),
  });
}

/** The dataset caption-score report (aggregated stored scores). */
export function useCaptionScoreReport(
  datasetId: number | null,
  captionType: string,
) {
  return useQuery({
    queryKey: ["caption-score-report", datasetId, captionType],
    queryFn: () =>
      api.get<CaptionScoreReport>("/caption-score/report", {
        dataset_id: datasetId,
        caption_type: captionType,
      }),
    enabled: datasetId != null,
  });
}

/** Ground every caption of a dataset in one two-phase job. */
export function useGroundDataset() {
  return useMutation({
    mutationFn: (vars: {
      dataset_id: number;
      caption_type: string;
      media_ids: number[] | null;
    }) => api.post<{ job_id: string }>("/grounding/dataset", vars),
  });
}

/** Score the tags of one or many media (SigLIP only, no LLM). */
export function useGroundTags() {
  return useMutation({
    mutationFn: (vars: { media_ids: number[] }) =>
      api.post<{ job_id: string }>("/grounding/tags", vars),
  });
}

/** Rebuild a caption's per-claim heat maps; read them back via useJobResult. */
export function useCaptionHeat() {
  return useMutation({
    mutationFn: (vars: TargetParams) =>
      api.post<{ job_id: string }>("/grounding/caption/heat", vars),
  });
}

/** Rebuild a media's per-tag heat maps. */
export function useTagHeat() {
  return useMutation({
    mutationFn: (vars: { media_ids: number[] }) =>
      api.post<{ job_id: string }>("/grounding/tags/heat", vars),
  });
}

/** Mark (or restore) a claim the user judged unsupported by the image. */
export function useRejectClaim() {
  const invalidate = useCaptionInvalidator();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { claim_id: number; rejected: boolean }) =>
      api.post("/grounding/claim/reject", vars),
    onSuccess: () => {
      invalidate();
      client.invalidateQueries({ queryKey: ["caption-grounding"] });
    },
  });
}

/** Detach a hallucinated tag from a media, dropping its scores with it. */
export function useRemoveGroundedTag() {
  const invalidate = useCaptionInvalidator();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { key: string; tag_id: number }) =>
      api.post<TagGroundingResponse>("/grounding/tags/remove", vars),
    onSuccess: () => {
      invalidate();
      client.invalidateQueries({ queryKey: ["tag-grounding"] });
      client.invalidateQueries({ queryKey: ["media-full"] });
    },
  });
}

/** Poll a heat job to completion, then hand back its elements. */
export function useHeatResult(jobId: string | null, finished: boolean) {
  return useJobResult<HeatResponse>(jobId, finished);
}

// -- Deploy -----------------------------------------------------------------

export function useDeploy() {
  return useMutation({
    mutationFn: (vars: { dataset_id: number; caption_type: string }) =>
      api.post<{ job_id: string }>(
        "/deploy/dataset",
        { dataset_id: vars.dataset_id },
        { caption_type: vars.caption_type },
      ),
  });
}

export function useUndeploy() {
  return useMutation({
    mutationFn: (datasetId: number) =>
      api.post("/deploy/undeploy", { dataset_id: datasetId }),
  });
}

export function useDeployMedia() {
  return useMutation({
    mutationFn: (vars: {
      dataset_id: number;
      keys: string[];
      caption_type: string;
    }) => api.post<{ written: number; count: number }>("/deploy/media", vars),
  });
}

// -- Prompt / settings mutations --------------------------------------------

export function useSaveGenSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: {
      model_type: string;
      temperature?: number;
      think_mode?: string;
      selected_prompt?: string;
    }) => api.post("/prompts/settings", vars),
    onSuccess: (_data, vars) =>
      client.invalidateQueries({
        queryKey: ["prompts", vars.model_type],
      }),
  });
}

export function useSavePrompt() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: {
      model_type: string;
      title: string;
      prompt: string;
    }) => api.post("/prompts", vars),
    onSuccess: (_data, vars) =>
      client.invalidateQueries({
        queryKey: ["prompts", vars.model_type],
      }),
  });
}

export function useDeletePrompt() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { model_type: string; title: string }) =>
      api.del("/prompts", vars),
    onSuccess: (_data, vars) =>
      client.invalidateQueries({
        queryKey: ["prompts", vars.model_type],
      }),
  });
}

// -- Datasets (management) --------------------------------------------------

function useDatasetsInvalidator() {
  const client = useQueryClient();
  return () => client.invalidateQueries({ queryKey: ["datasets"] });
}

export function useDatasetMedia(
  datasetId: number | null,
  offset: number,
  limit: number,
  qualityMetric: string,
) {
  return useQuery({
    queryKey: ["dataset-media", datasetId, offset, limit, qualityMetric],
    queryFn: () =>
      api.get<MediaGridPage>(`/datasets/${datasetId}/media`, {
        offset,
        limit,
        quality_metric: qualityMetric,
      }),
    enabled: datasetId != null,
    placeholderData: (prev) => prev,
  });
}

// -- Crops (virtual aliases) -------------------------------------------------

/**
 * A crop touches the dataset grid (a card appears or disappears), the caption
 * grid and every detail panel, so each crop mutation drops all of them.
 */
function useCropInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["dataset-media"] });
    client.invalidateQueries({ queryKey: ["caption-grid"] });
    client.invalidateQueries({ queryKey: ["media-detail"] });
    client.invalidateQueries({ queryKey: ["media-full"] });
    client.invalidateQueries({ queryKey: ["crops"] });
    client.invalidateQueries({ queryKey: ["datasets"] });
  };
}

/**
 * The crops of a media's *source*, marked against the dataset in view.
 *
 * Focus the original or one of its crops: the same list comes back, because a
 * crop is always re-framed from the image it aliases.
 */
export function useCrops(mediaId: number | null, datasetId: number | null) {
  return useQuery({
    queryKey: ["crops", mediaId, datasetId],
    queryFn: () =>
      api.get<{ crops: Crop[] }>("/crops", {
        media_id: mediaId,
        dataset_id: datasetId,
      }),
    enabled: mediaId != null,
  });
}

/** The source image the overlay frames; only fetched while it is open. */
export function useCropSource(mediaId: number | null) {
  return useQuery({
    queryKey: ["crop-source", mediaId],
    queryFn: () => api.get<CropSource>(`/crops/source/${mediaId}`),
    enabled: mediaId != null,
  });
}

export function useCreateCrop() {
  const invalidate = useCropInvalidator();
  return useMutation({
    mutationFn: (vars: {
      media_id: number;
      rect: CropRect;
      ratio: CropRatio;
      dataset_id: number;
      mode: "replace" | "beside";
    }) => api.post<Crop>("/crops", vars),
    onSuccess: invalidate,
  });
}

export function useUpdateCrop() {
  const invalidate = useCropInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; rect: CropRect; ratio: CropRatio }) =>
      api.post<Crop>(`/crops/${vars.id}`, {
        rect: vars.rect,
        ratio: vars.ratio,
      }),
    onSuccess: invalidate,
  });
}

export function usePlaceCrop() {
  const invalidate = useCropInvalidator();
  return useMutation({
    mutationFn: (vars: {
      id: number;
      dataset_id: number;
      mode: "replace" | "beside";
    }) =>
      api.post(`/crops/${vars.id}/place`, {
        dataset_id: vars.dataset_id,
        mode: vars.mode,
      }),
    onSuccess: invalidate,
  });
}

export function useDeleteCrop() {
  const invalidate = useCropInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del(`/crops/${id}`),
    onSuccess: invalidate,
  });
}

/** The composer's candidate query — every filter of the left rail. */
export interface ComposeParams {
  offset: number;
  limit: number;
  favorites_only: boolean;
  tag_ids?: number[];
  exclude_tag_ids?: number[];
  match?: string;
  metric?: string;
  min_score?: number;
  min_side?: number;
  exclude_blur?: boolean;
  exclude_noise?: boolean;
  media_type?: string;
  hide_near_dups?: boolean;
  gaps_only?: boolean;
  similar_to_selection?: boolean;
  sort?: string;
  semantic_q?: string;
  selected_ids?: number[];
}

export function useDatasetCandidates(
  datasetId: number | null,
  params: ComposeParams,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["dataset-candidates", datasetId, params],
    queryFn: () =>
      api.get<ComposeCandidatesPage>(`/datasets/${datasetId}/candidates`, {
        ...params,
      }),
    enabled: enabled && datasetId != null,
    placeholderData: (prev) => prev,
  });
}

/**
 * The live composition panel of a selection.
 *
 * Every projected number is recomputed server-side over
 * ``dataset + selection``, so the query key carries the ids — the caller
 * debounces the selection before handing it over.
 */
export function useComposePreview(
  datasetId: number | null,
  selectedIds: number[],
  metric: string,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["compose-preview", datasetId, selectedIds, metric],
    queryFn: () =>
      api.post<ComposePreview>(`/datasets/${datasetId}/compose/preview`, {
        selected_media_ids: selectedIds,
        metric,
      }),
    enabled: enabled && datasetId != null,
    placeholderData: (prev) => prev,
  });
}

/**
 * Free the SigLIP checkpoint the semantic search kept resident.
 *
 * The search encodes a query per keystroke, so the checkpoint stays loaded
 * while the composer is open; closing it must hand the VRAM back before
 * the next model job asks for it.
 */
export function useReleaseComposeModel() {
  return useMutation({
    mutationFn: () => api.post("/datasets/compose/release"),
  });
}

export function useCreateDataset() {
  const invalidate = useDatasetsInvalidator();
  return useMutation({
    mutationFn: (name: string) =>
      api.post<{ id: number }>("/datasets", { name }),
    onSuccess: invalidate,
  });
}

export function useUpdateDataset() {
  const invalidate = useDatasetsInvalidator();
  return useMutation({
    mutationFn: (vars: {
      id: number;
      name?: string;
      deploy_name?: string;
      deploy_resolution?: number;
    }) =>
      api.post(`/datasets/${vars.id}`, {
        name: vars.name ?? null,
        deploy_name: vars.deploy_name ?? null,
        deploy_resolution: vars.deploy_resolution ?? null,
      }),
    onSuccess: invalidate,
  });
}

export function useDeleteDataset() {
  const invalidate = useDatasetsInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del(`/datasets/${id}`),
    onSuccess: invalidate,
  });
}

export function useAddDatasetMedia() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { id: number; media_ids: number[] }) =>
      api.post(`/datasets/${vars.id}/media`, { media_ids: vars.media_ids }),
    onSuccess: (_data, vars) => {
      client.invalidateQueries({ queryKey: ["datasets"] });
      client.invalidateQueries({ queryKey: ["dataset-media", vars.id] });
      client.invalidateQueries({ queryKey: ["dataset-candidates", vars.id] });
    },
  });
}

export function useRemoveDatasetMedia() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { id: number; media_ids: number[] }) =>
      api.del(`/datasets/${vars.id}/media`, undefined, {
        media_ids: vars.media_ids,
      }),
    onSuccess: (_data, vars) => {
      client.invalidateQueries({ queryKey: ["datasets"] });
      client.invalidateQueries({ queryKey: ["dataset-media", vars.id] });
      client.invalidateQueries({ queryKey: ["dataset-candidates", vars.id] });
    },
  });
}

export function useTriggerwords(datasetId: number | null) {
  return useQuery({
    queryKey: ["triggerwords", datasetId],
    queryFn: () =>
      api.get<{ triggerwords: Triggerword[] }>(
        `/datasets/${datasetId}/triggerwords`,
      ),
    enabled: datasetId != null,
  });
}

// -- Datasets → Quality report ------------------------------------------------

export interface ReportRunVars {
  id: number;
  scorers: string[];
  caption_type: string;
  target_type: string;
  force: boolean;
}

/** The dataset's last stored report, its resolutions and the scorer chips. */
export function useDatasetReport(datasetId: number | null) {
  return useQuery({
    queryKey: ["dataset-report", datasetId],
    queryFn: () =>
      api.get<DatasetReportResponse>(`/datasets/${datasetId}/report`),
    enabled: datasetId != null,
  });
}

/** Enqueue an evaluation; the caller watches the returned job id. */
export function useRunDatasetReport() {
  return useMutation({
    mutationFn: ({ id, ...body }: ReportRunVars) =>
      api.post<{ job_id: string }>(`/datasets/${id}/report`, body),
  });
}

function useReportInvalidator() {
  const client = useQueryClient();
  return (datasetId: number) =>
    client.invalidateQueries({ queryKey: ["dataset-report", datasetId] });
}

/** Record how one flagged-media finding was handled. */
export function useResolveIssue() {
  const invalidate = useReportInvalidator();
  return useMutation({
    mutationFn: (vars: {
      id: number;
      issue_key: string;
      resolution: ResolutionKind;
      fingerprint: string;
      note?: string;
    }) =>
      api.post(
        `/datasets/${vars.id}/report/issues/${encodeURIComponent(
          vars.issue_key,
        )}`,
        {
          resolution: vars.resolution,
          fingerprint: vars.fingerprint,
          note: vars.note ?? "",
        },
      ),
    onSuccess: (_data, vars) => invalidate(vars.id),
  });
}

/** Reopen a finding the user had marked as handled. */
export function useUnresolveIssue() {
  const invalidate = useReportInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; issue_key: string }) =>
      api.del(
        `/datasets/${vars.id}/report/issues/${encodeURIComponent(
          vars.issue_key,
        )}`,
      ),
    onSuccess: (_data, vars) => invalidate(vars.id),
  });
}

export function useAddTriggerword() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { id: number; name: string }) =>
      api.post(`/datasets/${vars.id}/triggerwords`, { name: vars.name }),
    onSuccess: (_data, vars) =>
      client.invalidateQueries({ queryKey: ["triggerwords", vars.id] }),
  });
}

export function useRemoveTriggerword() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: { id: number; triggerword_id: number }) =>
      api.del(`/datasets/${vars.id}/triggerwords/${vars.triggerword_id}`),
    onSuccess: (_data, vars) =>
      client.invalidateQueries({ queryKey: ["triggerwords", vars.id] }),
  });
}

// -- Auto-build -------------------------------------------------------------

export function useAutobuildConfig() {
  return useQuery({
    queryKey: ["autobuild-config"],
    queryFn: () => api.get<AutobuildStudioConfig>("/autobuild/config"),
  });
}

/** The full Studio recipe plus the user's manual edits, sent on recompute. */
export interface AutobuildRecipe {
  media_type: string;
  semantic_q: string;
  locked_tags: string[];
  exclude_tags: string[];
  seed_media_ids: number[];
  size: number;
  metric: string | null;
  min_score: number;
  exclude_blur: boolean;
  framing_preset: string;
  live: boolean;
  dropped: number[];
  forced: number[];
  kept: number[];
  rebal: boolean;
}

/**
 * The Studio's live preview. Every recipe change re-runs the selection
 * server-side; the mutation's ``isPending`` drives the recompute overlay.
 */
export function useAutobuildPreview() {
  return useMutation({
    mutationFn: (recipe: AutobuildRecipe) =>
      api.post<AutobuildStudioPreview>("/autobuild/preview", recipe),
  });
}

/** One staged progress event of a streamed preview (the overlay bar). */
export interface AutobuildPreviewStage {
  stage: string;
  label: string;
  index: number;
  total: number;
}

/**
 * The Studio's live preview, streamed. Posts the recipe and reads an
 * NDJSON stream of stage events (``stage`` drives the recompute overlay's
 * progress bar), calling ``onResult`` with the final payload. Each run
 * aborts the one in flight, so a burst of keystrokes leaves only the last
 * stream updating state.
 */
export function useAutobuildPreviewStream() {
  const [stage, setStage] = useState<AutobuildPreviewStage | null>(null);
  const [isPending, setIsPending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (
      recipe: AutobuildRecipe,
      onResult: (result: AutobuildStudioPreview) => void,
      reusePool = false,
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setIsPending(true);
      setStage(null);
      try {
        const response = await fetch("/api/autobuild/preview-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...recipe, reuse_pool: reusePool }),
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(`preview stream failed: ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let newline = buffer.indexOf("\n");
          while (newline >= 0) {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            newline = buffer.indexOf("\n");
            if (!line) continue;
            const event = JSON.parse(line);
            if ("result" in event) {
              onResult(event.result as AutobuildStudioPreview);
            } else {
              setStage(event as AutobuildPreviewStage);
            }
          }
        }
      } catch {
        // A superseded run (AbortError) or a transient stream error: the
        // next run — or the finally below — restores the overlay state.
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setIsPending(false);
          setStage(null);
        }
      }
    },
    [],
  );

  return { run, stage, isPending };
}

export function useAutobuildSuggestTags() {
  return useMutation({
    mutationFn: (query: string) =>
      api.get<{ tags: AutobuildSuggestedTag[] }>("/autobuild/suggest-tags", {
        q: query,
      }),
  });
}

export function useAutobuildNeighbors() {
  return useMutation({
    mutationFn: (vars: {
      media_id: number;
      media_type: string;
      metric: string | null;
      exclude_ids: number[];
      q?: string;
    }) =>
      api.post<{ neighbors: AutobuildNeighbor[] }>(
        "/autobuild/neighbors",
        vars,
      ),
  });
}

export function useReleaseAutobuildModel() {
  return useMutation({
    mutationFn: () => api.post<{ ok: boolean }>("/autobuild/release", {}),
  });
}

/** Living-dataset upgrades for one dataset (replays its saved recipe). */
export function useAutobuildUpgrades(datasetId: number | null) {
  return useQuery({
    queryKey: ["autobuild-upgrades", datasetId],
    queryFn: () =>
      api.get<AutobuildUpgrades>(`/autobuild/upgrades/${datasetId}`),
    enabled: datasetId != null,
  });
}

export function useApplyAutobuildUpgrades(datasetId: number) {
  const client = useQueryClient();
  const invalidateDatasets = useDatasetsInvalidator();
  return useMutation({
    mutationFn: (swaps: AutobuildUpgrade[]) =>
      api.post<{ applied: number }>(`/autobuild/upgrades/${datasetId}/apply`, {
        swaps: swaps.map((swap) => ({
          out_media_id: swap.out_media_id,
          in_media_id: swap.in_media_id,
        })),
      }),
    onSuccess: () => {
      invalidateDatasets();
      client.invalidateQueries({ queryKey: ["autobuild-upgrades"] });
      client.invalidateQueries({ queryKey: ["dataset-media"] });
    },
  });
}

export function useAutobuildCreate() {
  const invalidate = useDatasetsInvalidator();
  return useMutation({
    mutationFn: (vars: {
      name: string;
      selection: number[];
      recipe: AutobuildRecipe;
    }) => api.post<{ id: number }>("/autobuild/create", vars),
    onSuccess: invalidate,
  });
}

/** A dataset's stored Studio recipe, or ``null`` when it was not built there. */
export interface StoredAutobuildRecipe {
  recipe: AutobuildRecipe | null;
  live: boolean;
}

/**
 * The Studio recipe a dataset was built from — the source for reopening it
 * in the Studio to edit again. ``recipe`` is null for hand-made datasets.
 */
export function useAutobuildRecipe(datasetId: number | null) {
  return useQuery({
    queryKey: ["autobuild-recipe", datasetId],
    queryFn: () =>
      api.get<StoredAutobuildRecipe>(`/autobuild/recipe/${datasetId}`),
    enabled: datasetId != null,
  });
}

/** Overwrite an existing dataset's media (and recipe) from a Studio edit. */
export function useAutobuildUpdate() {
  const invalidate = useDatasetsInvalidator();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (vars: {
      datasetId: number;
      selection: number[];
      recipe: AutobuildRecipe;
    }) =>
      api.post<{ id: number }>(`/autobuild/update/${vars.datasetId}`, {
        selection: vars.selection,
        recipe: vars.recipe,
      }),
    onSuccess: () => {
      invalidate();
      client.invalidateQueries({ queryKey: ["dataset-media"] });
      client.invalidateQueries({ queryKey: ["autobuild-upgrades"] });
      client.invalidateQueries({ queryKey: ["autobuild-recipe"] });
    },
  });
}

// -- Media library ----------------------------------------------------------

export interface LibraryFilters {
  offset: number;
  limit: number;
  tag_ids: number[];
  exclude_tag_ids: number[];
  match: string;
  favorites_only: boolean;
  sort: string;
  quality_metric: string;
}

export function useLibraryGrid(filters: LibraryFilters) {
  return useQuery({
    queryKey: ["library-grid", filters],
    queryFn: () =>
      api.get<LibraryGridPage>(
        "/medias/grid",
        filters as unknown as Record<string, unknown>,
      ),
    placeholderData: (prev) => prev,
  });
}

export function useMediaFullDetail(
  key: string | null,
  qualityMetric: string,
) {
  return useQuery({
    queryKey: ["media-full", key, qualityMetric],
    queryFn: () =>
      api.get<MediaFullDetail>(`/medias/${key}`, {
        quality_metric: qualityMetric,
      }),
    enabled: !!key,
  });
}

export function useToggleFavorite() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (key: string) =>
      api.post<{ favorite: boolean }>(`/medias/${key}/favorite`),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["library-grid"] });
      client.invalidateQueries({ queryKey: ["media-full"] });
    },
  });
}

// -- Libraries --------------------------------------------------------------

function useLibrariesInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["libraries"] });
    client.invalidateQueries({ queryKey: ["library-coverage"] });
    client.invalidateQueries({ queryKey: ["index-status"] });
    client.invalidateQueries({ queryKey: ["library-media"] });
    client.invalidateQueries({ queryKey: ["library-missing"] });
    client.invalidateQueries({ queryKey: ["nav-counts"] });
  };
}

export function useLibraries() {
  return useQuery({
    queryKey: ["libraries"],
    queryFn: () => api.get<{ libraries: LibrarySource[] }>("/libraries"),
  });
}

export function useLibraryCoverage() {
  return useQuery({
    queryKey: ["library-coverage"],
    queryFn: () => api.get<LibraryCoverage>("/libraries/coverage"),
  });
}

export function useQualityMetrics() {
  return useQuery({
    queryKey: ["quality-metrics"],
    queryFn: () =>
      api.get<{ metrics: QualityMetric[] }>("/libraries/quality-metrics"),
  });
}

/** Per-step index coverage of every library, plus this machine's toggles. */
export function useIndexStatus() {
  return useQuery({
    queryKey: ["index-status"],
    queryFn: () => api.get<IndexStatus>("/libraries/index-status"),
  });
}

export function useLibraryMedia(
  libraryId: number | null,
  offset: number,
  limit: number,
  qualityMetric: string,
) {
  return useQuery({
    queryKey: ["library-media", libraryId, offset, limit, qualityMetric],
    queryFn: () =>
      api.get<MediaGridPage>(`/libraries/${libraryId}/media`, {
        offset,
        limit,
        quality_metric: qualityMetric,
      }),
    enabled: libraryId != null,
    placeholderData: (prev) => prev,
  });
}

export function useCreateLibrary() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (vars: { name: string; path: string; recursive: boolean }) =>
      api.post<{ id: number }>("/libraries", vars),
    onSuccess: invalidate,
  });
}

/** List a folder's sub-folders for the "Add folder" picker (server-side). */
export function useBrowseFolder(path: string, enabled: boolean) {
  return useQuery({
    queryKey: ["browse", path],
    queryFn: () => api.get<FolderListing>("/system/browse", { path }),
    enabled,
  });
}

/** List sub-folders + files (suffix-filtered) for a server-side file picker. */
export function useBrowseFiles(path: string, exts: string, enabled: boolean) {
  return useQuery({
    queryKey: ["browse-files", path, exts],
    queryFn: () =>
      api.get<FileListing>("/system/browse-files", { path, exts }),
    enabled,
  });
}

/** The full sub-folder tree + media counts for the mapping wizard. */
export function useFolderTree(path: string, enabled: boolean) {
  return useQuery({
    queryKey: ["folder-tree", path],
    queryFn: () => api.get<FolderTree>("/libraries/folder-tree", { path }),
    enabled,
  });
}

/** A library's persisted subfolder mapping (auto-tag level + folder rules). */
export function useFolderRules(libraryId: number | null, enabled: boolean) {
  return useQuery({
    queryKey: ["folder-rules", libraryId],
    queryFn: () =>
      api.get<FolderMapping>(`/libraries/${libraryId}/folder-rules`),
    enabled: libraryId != null && enabled,
  });
}

/** Persist a subfolder mapping and queue the scan that applies it. */
export function usePutFolderRules() {
  const client = useQueryClient();
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (vars: { libraryId: number; mapping: FolderMappingInput }) =>
      api.put<{ job_id: string; sub_libraries: number; rules: number }>(
        `/libraries/${vars.libraryId}/folder-rules`,
        vars.mapping,
      ),
    onSuccess: (_data, vars) => {
      invalidate();
      client.invalidateQueries({
        queryKey: ["folder-rules", vars.libraryId],
      });
    },
  });
}

export function useDeleteLibrary() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (id: number) => api.del(`/libraries/${id}`),
    onSuccess: invalidate,
  });
}

export function useSetRecursive() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; recursive: boolean }) =>
      api.post(`/libraries/${vars.id}/recursive`, {
        recursive: vars.recursive,
      }),
    onSuccess: invalidate,
  });
}

export function useSetLibraryPath() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (vars: { id: number; path: string }) =>
      api.post(`/libraries/${vars.id}/path`, { path: vars.path }),
    onSuccess: invalidate,
  });
}

export function useScanLibrary() {
  return useMutation({
    mutationFn: (id: number) =>
      api.post<{ job_id: string }>(`/libraries/${id}/scan`),
  });
}

/** Rescan every library in one job (catch new files, refresh counters). */
export function useScanAllLibraries() {
  return useMutation({
    mutationFn: () => api.post<{ job_id: string }>("/libraries/scan-all"),
  });
}

/** Rescan + re-index every library in one chained job (auto-tags included). */
export function useReindexAllLibraries() {
  return useMutation({
    mutationFn: () => api.post<{ job_id: string }>("/libraries/reindex-all"),
  });
}

/** On-demand scan of a library for media whose file vanished from disk. */
/**
 * The library's media whose source file left the disk.
 *
 * A query, not an action: detection is a handful of ``stat`` calls, so the
 * warning banner surfaces on its own when a library is opened, and again
 * after a scan (which invalidates this key). There is no Detect button.
 */
export function useMissingMedia(libraryId: number | null) {
  return useQuery({
    queryKey: ["library-missing", libraryId],
    queryFn: () =>
      api.get<{ media: { id: number; name: string }[] }>(
        `/libraries/${libraryId}/missing`,
      ),
    enabled: libraryId != null,
  });
}

/** Hard-delete media from the app (source files deleted off disk). */
export function usePurgeMedia() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (mediaIds: number[]) =>
      api.post<{ removed: number }>("/libraries/purge-media", {
        media_ids: mediaIds,
      }),
    onSuccess: invalidate,
  });
}

/** Queue one chained Index job (null steps = every enabled step). */
export function useIndexRun() {
  return useMutation({
    mutationFn: (vars: {
      library_id: number | null;
      steps: string[] | null;
      force: boolean;
    }) => api.post<{ job_id: string }>("/libraries/index", vars),
  });
}

export function useBulkTags() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (vars: {
      library_id: number | null;
      add_tag_ids: number[];
      remove_tag_ids: number[];
    }) => api.post("/libraries/bulk-tags", vars),
    onSuccess: invalidate,
  });
}

export function useLookalikeDetect() {
  return useMutation({
    mutationFn: (similarity: number) =>
      api.post<LookalikeResult>("/libraries/lookalike/detect", { similarity }),
  });
}

export function useLookalikeKeepBest() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (similarity: number) =>
      api.post<{ discarded: number }>("/libraries/lookalike/keep-best", {
        similarity,
      }),
    onSuccess: invalidate,
  });
}

export function useLookalikeDiscard() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (media_ids: number[]) =>
      api.post<{ discarded: number }>("/libraries/lookalike/discard", {
        media_ids,
      }),
    onSuccess: invalidate,
  });
}

export function useLookalikeDismiss() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: (media_ids: number[]) =>
      api.post<{ dismissed: number }>("/libraries/lookalike/dismiss", {
        media_ids,
      }),
    onSuccess: invalidate,
  });
}

export function useLookalikeResetDismissed() {
  const invalidate = useLibrariesInvalidator();
  return useMutation({
    mutationFn: () =>
      api.post<{ ok: boolean }>("/libraries/lookalike/reset-dismissed", {}),
    onSuccess: invalidate,
  });
}

export function useTaggerModels() {
  return useQuery({
    queryKey: ["tagger-models"],
    queryFn: () => api.get<TaggerModels>("/tagger/models"),
  });
}

export interface TaggerRunVars {
  source: string;
  general: number;
  character: number;
  scope: string;
  media_ids: number[];
  filter_tag_ids: number[];
  exclude_tag_ids: number[];
  match: string;
  /** Chain a SigLIP grounding pass over the tags the run just attached. */
  ground_after: boolean;
}

export function useRunTagger() {
  return useMutation({
    mutationFn: (vars: TaggerRunVars) =>
      api.post<{ job_id: string; count: number }>("/tagger/run", vars),
  });
}

/** Invalidate the media grid + detail (after a tag edit). */
export function useMediaInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["library-grid"] });
    client.invalidateQueries({ queryKey: ["media-full"] });
    client.invalidateQueries({ queryKey: ["nav-counts"] });
  };
}

// -- Watermark Lab ----------------------------------------------------------

/** Invalidate everything a watermark change can affect (badges everywhere). */
export function useWatermarkInvalidator() {
  const client = useQueryClient();
  return () => {
    client.invalidateQueries({ queryKey: ["watermark-inventory"] });
    client.invalidateQueries({ queryKey: ["watermark-media"] });
    client.invalidateQueries({ queryKey: ["library-grid"] });
    client.invalidateQueries({ queryKey: ["media-detail"] });
    client.invalidateQueries({ queryKey: ["media-full"] });
    client.invalidateQueries({ queryKey: ["caption-grid"] });
  };
}

export function useWatermarkConfig() {
  return useQuery({
    queryKey: ["watermark-config"],
    queryFn: () => api.get<WatermarkConfig>("/watermarks/config"),
  });
}

export function useUpdateWatermarkConfig() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (prefs: Partial<WatermarkPrefs>) =>
      api.patch<{ prefs: WatermarkPrefs }>("/watermarks/config", prefs),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["watermark-config"] }),
  });
}

/** The inventory query params: a tab, the Media-page filter, sort and page. */
export interface WatermarkInventoryQuery {
  tab: WatermarkTab;
  tag_ids: number[];
  exclude_tag_ids: number[];
  match: string;
  favorites_only: boolean;
  sort: string;
  offset: number;
  limit: number;
}

export function useWatermarkInventory(
  query: WatermarkInventoryQuery,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["watermark-inventory", query],
    queryFn: () =>
      api.get<WatermarkInventory>("/watermarks", {
        tab: query.tab,
        tag_ids: query.tag_ids,
        exclude_tag_ids: query.exclude_tag_ids,
        match: query.match,
        favorites_only: query.favorites_only,
        sort: query.sort,
        offset: query.offset,
        limit: query.limit,
      }),
    enabled,
    placeholderData: (prev) => prev,
  });
}

export function useWatermarkMedia(key: string | null) {
  return useQuery({
    queryKey: ["watermark-media", key],
    queryFn: () => api.get<WatermarkMedia>(`/watermarks/${key}`),
    enabled: !!key,
  });
}

/** A batch selection: an explicit id list, or the whole filtered tab. */
export interface WatermarkSelection {
  media_ids?: number[];
  select_all?: boolean;
  tab: WatermarkTab;
  tag_ids?: number[];
  exclude_tag_ids?: number[];
  match?: string;
  favorites_only?: boolean;
}

function useWatermarkJob(path: string) {
  return useMutation({
    mutationFn: (sel: WatermarkSelection) =>
      api.post<{ job_id: string }>(`/watermarks/${path}`, sel),
  });
}

export function useWatermarkScan() {
  return useWatermarkJob("scan");
}

export function useWatermarkScanAndPatch() {
  return useWatermarkJob("scan_and_patch");
}

export function useWatermarkPatch() {
  return useWatermarkJob("patch");
}

export function useFlattenSelection() {
  return useWatermarkJob("flatten");
}

/** Patch every detected zone of one media (side panel / card). */
export function usePatchMedia() {
  return useMutation({
    mutationFn: (mediaId: number) =>
      api.post<{ job_id: string }>(`/watermarks/${mediaId}/patch`),
  });
}

export function useFlattenMedia() {
  return useMutation({
    mutationFn: (mediaId: number) =>
      api.post<{ job_id: string }>(`/watermarks/${mediaId}/flatten`),
  });
}

export function useDismissSelection() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (sel: WatermarkSelection) =>
      api.post<{ dismissed: number }>("/watermarks/dismiss", sel),
    onSuccess: invalidate,
  });
}

export function useDismissMedia() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (mediaId: number) =>
      api.del<{ media: WatermarkMedia }>(`/watermarks/media/${mediaId}`),
    onSuccess: invalidate,
  });
}

export function useRevertSelection() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (sel: WatermarkSelection) =>
      api.post<{ reverted: number }>("/watermarks/revert", sel),
    onSuccess: invalidate,
  });
}

export function useRevertMedia() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (mediaId: number) =>
      api.post<{ media: WatermarkMedia }>(`/watermarks/${mediaId}/revert`),
    onSuccess: invalidate,
  });
}

/** Response of a review action: a FLUX job, or the refreshed media (sync). */
type PatchResponse = { job_id?: string; media?: WatermarkMedia };

export function useRegenerateZone() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (vars: { zoneId: number; prompt?: string; seed?: number }) =>
      api.post<PatchResponse>(`/watermarks/${vars.zoneId}/regenerate`, {
        prompt: vars.prompt,
        seed: vars.seed,
      }),
    onSuccess: invalidate,
  });
}

export function useEditZone() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (vars: {
      zoneId: number;
      box?: WatermarkBox;
      status?: string;
      prompt?: string;
    }) =>
      api.patch<PatchResponse>(`/watermarks/${vars.zoneId}`, {
        box: vars.box,
        status: vars.status,
        prompt: vars.prompt,
      }),
    onSuccess: invalidate,
  });
}

export function useAddZone() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (vars: { mediaId: number; box: WatermarkBox }) =>
      api.post<{ zone_id: number; media: WatermarkMedia }>(
        `/watermarks/${vars.mediaId}/zones`,
        { box: vars.box },
      ),
    onSuccess: invalidate,
  });
}

export function useDeleteZone() {
  const invalidate = useWatermarkInvalidator();
  return useMutation({
    mutationFn: (zoneId: number) =>
      api.del<{ media: WatermarkMedia }>(`/watermarks/${zoneId}`),
    onSuccess: invalidate,
  });
}
