"""Pydantic request bodies for the routers.

Responses are plain JSON dicts assembled from the ``src`` engines; only
inbound bodies are typed here so FastAPI validates them and OpenAPI documents
them for the front-end.
"""

from pydantic import BaseModel

from src.quality import DEFAULT_METRIC


class LoadModelBody(BaseModel):
    """Body for ``POST /api/models/load`` — a scanned model name."""

    name: str


class ProfileBody(BaseModel):
    """Body for creating / updating a model profile.

    Every field is optional: only the provided ones are applied (see
    :func:`src.model_profiles.update_profile`); the rest keep their stored
    (or factory) values. ``role`` is create-only: ``"caption"``/``"judge"``
    selects the new profile for that slot.
    """

    name: str | None = None
    file: str | None = None
    dir: str | None = None
    format: str | None = None
    type: str | None = None
    type_mode: str | None = None
    temp: float | None = None
    n_ctx: int | None = None
    mmproj_mode: str | None = None
    mmproj: str | None = None
    think: str | None = None
    max_tok: int | None = None
    img_res: int | None = None
    prompt: str | None = None
    role: str | None = None


class ProfileSelectBody(BaseModel):
    """Body pointing the captioner or judge slot at a profile."""

    role: str  # "caption" | "judge"
    id: int


class ProfileDetectBody(BaseModel):
    """Body for re-running type/mmproj auto-detection on a picked file."""

    dir: str
    file: str


class SaveCaptionBody(BaseModel):
    """Body for saving a caption on a media in a dataset.

    ``amend`` (autosave) overwrites the current revision in place; the
    default appends a new revision (the explicit Save / version snapshot).
    """

    content: str
    scope: str = "type"  # "type" advances shared head, "dataset" branches
    amend: bool = False


class SelectRevisionBody(BaseModel):
    """Body for pinning/following a caption revision."""

    revision_id: int | None = None  # None follows the head


class RepeatsBody(BaseModel):
    """Body for the deploy-repeats stepper."""

    repeats: int


class HiddenBody(BaseModel):
    """Body for hiding/unhiding a media within a dataset."""

    hidden: bool


class DeployNameBody(BaseModel):
    """Body for renaming a dataset's deploy folder."""

    name: str


class SavePromptBody(BaseModel):
    """Body for saving a user prompt preset for a model type."""

    model_type: str
    title: str
    prompt: str


class GenModeBody(BaseModel):
    """Body for per-model-type generation settings (temp / thinking)."""

    model_type: str
    temperature: float | None = None
    think_mode: str | None = None
    selected_prompt: str | None = None


class GenerateBody(BaseModel):
    """Body for the Generate-all job."""

    dataset_id: int
    caption_type: str
    media_ids: list[int] | None = None  # None = the whole dataset
    exclude_ids: list[int] | None = None  # locked media, never captioned
    prompt: str
    # Captioner profile: generation params (temperature, thinking, image
    # size, max tokens) come from it, and the job lazy-swaps it into VRAM
    # when a different profile is resident. None = use the loaded model
    # with the legacy field values below.
    profile_id: int | None = None
    temperature: float = 0.7
    seed: int | None = None
    think_mode: str = "auto"
    image_size: int = 1024
    review_after: bool = False
    review_judge_profile_id: int | None = None  # judge for review_after
    ground_after: bool = False
    # Off = caption only media whose caption is still empty; on (default)
    # regenerates every targeted media, already-captioned ones included.
    recaption: bool = True
    # Free the VRAM when the job is done (default: keep the model resident).
    unload_after: bool = False


class DeployBody(BaseModel):
    """Body carrying a dataset id (deploy / undeploy dataset)."""

    dataset_id: int


class DeployMediaBody(BaseModel):
    """Body for deploying a selection of media (batch bar)."""

    dataset_id: int
    keys: list[str]
    caption_type: str


class SettingsBody(BaseModel):
    """Passthrough body for a settings save (validated by src.settings)."""

    settings: dict


class TagRefBody(BaseModel):
    """Body identifying a tag to attach: by id, or by name + category."""

    tag_id: int | None = None
    name: str | None = None
    category_id: int | None = None


class ReviewTargetBody(BaseModel):
    """Body identifying a caption to review/act on within a dataset."""

    dataset_id: int
    key: str
    caption_type: str


class GroundTargetBody(ReviewTargetBody):
    """Body identifying one caption to ground (or re-heat) in a dataset."""


class ReviewRuleCreateBody(BaseModel):
    """Body creating one custom review rule for a dataset."""

    dataset_id: int
    text: str
    needs_image: bool = False


class ReviewRuleUpdateBody(BaseModel):
    """Body toggling or rewriting one review rule."""

    enabled: bool | None = None
    text: str | None = None


class ReviewRunBody(BaseModel):
    """Body for a rule-based review run over a dataset (or a single media)."""

    dataset_id: int
    caption_type: str
    media_ids: list[int] | None = None  # None = the whole dataset
    judge_profile_id: int | None = None  # None = use the loaded model
    scope: str = "all"  # all | selection | flagged | single
    rule_ids: list[int] | None = None  # None = every enabled rule
    seed: int | None = None
    # Free the VRAM when the run is done (default: keep the model resident).
    unload_after: bool = False


class ReviewDecideBody(BaseModel):
    """Body recording a human decision on one finding."""

    action: str  # accept | reject
    caption: str | None = None  # inline-edited text applied on accept


class ReviewBulkDecideBody(BaseModel):
    """Body for the bulk-accept actions (safe fixes / all from a rule)."""

    dataset_id: int
    rule_id: int | None = None  # None = every safe (det + integrity) finding


class CaptionScoreBody(ReviewTargetBody):
    """Body identifying one caption to reference-free score in a dataset."""


class CaptionScoreDatasetBody(BaseModel):
    """Body for reference-free scoring every caption of a dataset."""

    dataset_id: int
    caption_type: str


class TagScoreBody(BaseModel):
    """Body for reference-free scoring one media's tags."""

    key: str


class GroundDatasetBody(BaseModel):
    """Body for grounding a whole dataset's captions."""

    dataset_id: int
    caption_type: str
    media_ids: list[int] | None = None  # None = the whole dataset


class GroundTagsBody(BaseModel):
    """Body for grounding the tags of one or many media."""

    media_ids: list[int]


class RejectClaimBody(BaseModel):
    """Body toggling a grounded claim's "marked non-validated" flag."""

    claim_id: int
    rejected: bool = True


class RemoveTagBody(BaseModel):
    """Body detaching a hallucinated tag from a media."""

    key: str
    tag_id: int


class DatasetCreateBody(BaseModel):
    """Body for creating a dataset."""

    name: str


class DatasetUpdateBody(BaseModel):
    """Body for renaming a dataset or its deploy folder / resize target."""

    name: str | None = None
    deploy_name: str | None = None
    deploy_resolution: int | None = None


class MediaIdsBody(BaseModel):
    """Body carrying a list of media ids (add / remove from dataset)."""

    media_ids: list[int]


class ComposePreviewBody(BaseModel):
    """Body of the dataset composer's live composition preview."""

    selected_media_ids: list[int] = []
    metric: str = DEFAULT_METRIC
    target_type: str = "character"


class DatasetReportBody(BaseModel):
    """Body for a Datasets → Quality report run (the dataset is the path)."""

    scorers: list[str] = []
    caption_type: str = "txt"
    target_type: str = "character"
    force: bool = False


class IssueResolutionBody(BaseModel):
    """Body recording how one report finding was handled."""

    resolution: str
    fingerprint: str = ""
    note: str = ""


class TriggerwordBody(BaseModel):
    """Body for adding a trigger word to a dataset."""

    name: str


class AutobuildPreviewBody(BaseModel):
    """Body for the Auto-build Studio's live selection preview.

    Full recipe plus manual edits (``dropped``/``forced``/``kept``), replayed
    on every recompute. ``locked_tags`` = WD14 subject tags,
    ``seed_media_ids`` = example images proximity is measured to,
    ``framing_preset`` = a ``server.runners.autobuild.FRAMING_PRESETS`` key.
    """

    media_type: str = "img"
    semantic_q: str = ""
    locked_tags: list[str] = []
    exclude_tags: list[str] = []
    seed_media_ids: list[int] = []
    library_ids: list[int] = []
    size: int = 50
    metric: str | None = None
    min_score: float = 60.0
    exclude_blur: bool = True
    framing_preset: str = "balanced"
    live: bool = True
    dropped: list[int] = []
    forced: list[int] = []
    kept: list[int] = []
    rebal: bool = False
    # A pick edit (drop/force/keep/rebalance) reuses the last Build's pool
    # and geometry instead of rereading the whole library; a fresh Build
    # leaves this false so index/quality changes are picked up.
    reuse_pool: bool = False


class AutobuildNeighborsBody(BaseModel):
    """Body for the Studio's ``⇄`` swap: the neighbours of one pick.

    An empty ``q`` returns the closest visual neighbours; a non-empty ``q``
    re-ranks the eligible candidates by tag/name match instead.
    """

    media_id: int
    media_type: str = "img"
    metric: str | None = None
    exclude_ids: list[int] = []
    q: str = ""


class AutobuildSwap(BaseModel):
    """One living-dataset upgrade: the media to drop and the one to add."""

    out_media_id: int
    in_media_id: int


class AutobuildApplyUpgradesBody(BaseModel):
    """Body for applying living-dataset upgrades (one or many swaps)."""

    swaps: list[AutobuildSwap]


class AutobuildCreateBody(BaseModel):
    """Body for creating a dataset from a Studio selection.

    ``recipe`` = the params the selection came from, stored with the dataset so
    a "living" dataset can be recomputed.
    """

    name: str
    selection: list[int]
    recipe: dict | None = None


class AutobuildUpdateBody(BaseModel):
    """Body for overwriting an existing dataset from a Studio selection.

    The re-edit's "overwrite" save: the dataset keeps its id and name, its
    media are replaced by ``selection`` and its stored ``recipe`` is
    refreshed so a later re-edit reopens the new state.
    """

    selection: list[int]
    recipe: dict | None = None


class SqlBody(BaseModel):
    """Body for a read-only SQL query (SQLite explorer)."""

    sql: str


class DeleteRowBody(BaseModel):
    """Body for the SQLite explorer's single-row delete escape hatch."""

    table: str
    row_id: int


class RestoreBody(BaseModel):
    """Body for restoring a database backup by file name."""

    filename: str


class LibraryCreateBody(BaseModel):
    """Body for registering a folder library."""

    name: str
    path: str
    recursive: bool = True


class MergeBody(BaseModel):
    """Body for merging folder libraries into a destination."""

    source_ids: list[int]
    dest_id: int


class RecursiveBody(BaseModel):
    """Body for toggling a library's recursive-scan flag."""

    recursive: bool


class LibraryPathBody(BaseModel):
    """Body for repointing a folder library at a new folder."""

    path: str


class FolderRuleItem(BaseModel):
    """One sub-folder's rule in a library's subfolder mapping.

    ``mode`` = ``keep``/``sublib``/``exclude``; ``tags``/``removed`` = tag
    *names* (manual tags, and inherited/auto tags overridden off); ``sub_name``
    = display name a ``sublib``'s promoted library takes (own folder if empty).
    """

    rel_path: str
    mode: str = "keep"
    tags: list[str] = []
    removed: list[str] = []
    sub_name: str | None = None


class FolderRulesBody(BaseModel):
    """Body for ``PUT /api/libraries/{id}/folder-rules`` (whole mapping)."""

    auto_tag_level: str = "0"
    rules: list[FolderRuleItem] = []
    # Optional rename of the parent library applied before the mapping;
    # blank/omitted keeps the current name.
    name: str | None = None


class IndexBody(BaseModel):
    """Body for an Index run: the chained scans of :mod:`src.index_steps`.

    ``steps`` restricts the run (per-step "Run" buttons); None chains every
    step enabled on this machine.
    """

    library_id: int | None = None
    steps: list[str] | None = None
    force: bool = False


class BulkTagsBody(BaseModel):
    """Body for adding/removing tags across a library's media."""

    library_id: int | None = None
    add_tag_ids: list[int] = []
    remove_tag_ids: list[int] = []


class LookalikeBody(BaseModel):
    """Body for near-duplicate detection / keep-best."""

    similarity: int = 88


class CategoryBody(BaseModel):
    """Body for creating a tag category."""

    name: str
    color: str = "#888888"


class CategoryUpdateBody(BaseModel):
    """Body for renaming / recolouring a tag category."""

    name: str | None = None
    color: str | None = None


class ReorderBody(BaseModel):
    """Body for reordering tag categories (full ordered id list)."""

    ordered_ids: list[int]


class TagCreateBody(BaseModel):
    """Body for creating a tag in a category."""

    name: str
    category_id: int


class TagNameBody(BaseModel):
    """Body for creating/reusing a tag by name in the Uncategorized pen."""

    name: str


class TagNamesBody(BaseModel):
    """Body for checking which of several tag names already exist."""

    names: list[str] = []


class RenameBody(BaseModel):
    """Body for renaming a tag."""

    name: str


class MoveTagBody(BaseModel):
    """Body for moving a tag to another category (Tags drag-and-drop)."""

    category_id: int


class TaggerRunBody(BaseModel):
    """Body for launching a WD14 auto-tag run."""

    source: str
    local_dir: str = ""
    general: float = 0.35
    character: float = 0.85
    replace_underscores: bool = True
    ground_after: bool = False  # chain a SigLIP tag-grounding pass
    scope: str = "media"  # "media" (explicit ids) or "filtered"
    media_ids: list[int] = []
    filter_tag_ids: list[int] = []
    exclude_tag_ids: list[int] = []
    match: str = "any"


class CropRect(BaseModel):
    """A crop rectangle, in percentages of the source image."""

    x: float
    y: float
    w: float
    h: float


class CropCreateBody(BaseModel):
    """Body for creating a crop and placing it in a dataset."""

    media_id: int
    rect: CropRect
    ratio: str = "free"
    dataset_id: int
    # "replace" swaps the parent's dataset entry, "beside" keeps both.
    mode: str = "replace"


class CropUpdateBody(BaseModel):
    """Body for re-framing an existing crop."""

    rect: CropRect
    ratio: str = "free"


class CropPlaceBody(BaseModel):
    """Body for adding an existing crop to a dataset."""

    dataset_id: int
    mode: str = "replace"


class WatermarkBox(BaseModel):
    """A watermark zone rectangle in fractions (0-1) of the source image."""

    x: float
    y: float
    w: float
    h: float


class WatermarkSelectionBody(BaseModel):
    """A Watermark Lab batch selection (scan / patch / dismiss / flatten…).

    Either an explicit ``media_ids`` list, or ``select_all`` = the whole
    filtered result of a tab (across pages), resolved server-side with the
    same tag/favorite filter the inventory uses. ``tab`` scopes that resolve
    to the tab's watermark membership. Rail prefs come from persisted
    Settings, so the body carries only the selection.
    """

    media_ids: list[int] = []
    select_all: bool = False
    tab: str = "media"
    tag_ids: list[int] = []
    exclude_tag_ids: list[int] = []
    match: str = "any"
    favorites_only: bool = False


class WatermarkRegenBody(BaseModel):
    """Body for regenerating a zone's patch (new seed and/or new prompt)."""

    prompt: str | None = None
    seed: int | None = None


class WatermarkZoneBody(BaseModel):
    """Body for editing a zone: move its box, set its status or its prompt."""

    box: WatermarkBox | None = None
    status: str | None = None
    prompt: str | None = None


class WatermarkCreateZoneBody(BaseModel):
    """Body for adding a manual zone to a media."""

    box: WatermarkBox
