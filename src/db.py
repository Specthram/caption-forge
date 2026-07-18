"""SQLite database bootstrap and connection management.

Backs the SQLite storage mode: datasets, media, captions and revisions in one
relational database. This module owns the low-level concerns only — where the
file lives, how a connection is configured, and the schema (created
idempotently). Repository logic lives elsewhere.

Schema uses ``CREATE TABLE IF NOT EXISTS``, so :func:`ensure_database` is safe
every launch. A one-row ``schema_version`` table records the version.
Pre-deployment there's no migration machinery: the schema below IS the
database, edited in place while the version stays 1. Post-deployment,
migrations will live under ``database/migrations/``.

:func:`connect` and :func:`ensure_database` also accept an SQLite ``file:``
URI — the test suite uses in-memory URIs so tests never touch a disk.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from src.constants import DB_PATH, DEFAULT_TAG_CATEGORIES, STORAGE_DIR

# Stored in the ``schema_version`` table. Pinned at 1 pre-deployment (schema
# changes edit ``_SCHEMA`` directly); post-deployment each change becomes a
# ``database/migrations/`` script and bumps this.
SCHEMA_VERSION = 1

# Full schema. Every statement is idempotent (``IF NOT EXISTS``) so it can be
# replayed on each launch. Foreign keys cascade where a child row is
# meaningless without its parent (a dataset's links, a caption's revisions).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL,
    file_extension TEXT NOT NULL,
    hidden INTEGER NOT NULL DEFAULT 0,
    -- User "favorite" flag: a quick heart toggle on the Medias grid, with a
    -- Medias "Favorites only" filter. Independent of hidden/discarded.
    favorite INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at TEXT,
    width INTEGER,
    height INTEGER,
    indexed_at TEXT,
    -- Perceptual hashes (foundation of the future "lookalike" near-duplicate
    -- detection): imagehash's 16-hex-char pHash/dHash of the media's original
    -- image (a video's first frame), filled by the Index action. TEXT for the
    -- same reason as sha256 -- to store the hex verbatim and dodge SQLite's
    -- signed-64-bit integer edge cases. NULL until indexed.
    phash TEXT,
    dhash TEXT,
    -- Model-free per-image statistics (see src.image_stats), written by the
    -- same cheap CPU probe of the Index chain that fills the dimensions and
    -- the hashes. The dataset quality report recomputes them over its own
    -- snapshot; these columns exist so the dataset composer can drop blurry
    -- or noisy candidates *before* paginating, without opening a file. NULL
    -- means "never analyzed" (a video, an undecodable image, or a media
    -- indexed before these columns existed) and never counts as flagged.
    sharpness REAL,
    clipping REAL,
    cleanliness REAL,
    -- Lookalike resolution: timestamp a media was "discarded" from the
    -- Lookalikes view (a near-duplicate the user set aside). NULL means
    -- active. Unlike `hidden` (which a re-scan clears) and `deleted_at`
    -- (which a re-scan would treat as new, losing tags), a discarded media
    -- is kept out of every user listing yet is never revived by ingest_file
    -- when its source file is re-seen -- the state survives re-scans and is
    -- reversible from the "Discarded media" restore section.
    discarded_at TEXT,
    -- Lookalike "hide indefinitely": timestamp a near-duplicate group was
    -- dismissed by the user without discarding any image. NULL means the
    -- media still participates in the Lookalikes view. Marked media stay
    -- live and hashed (they can still form NEW groups with future imports);
    -- only a group whose every member is reviewed is hidden. Like
    -- discarded_at, ingest_file never clears it, so the dismissal survives
    -- re-scans -- reversible with the "Reset dismissed" action.
    lookalike_reviewed_at TEXT,
    -- Virtual crop alias (see src.crops). A NON-NULL parent_media_id marks
    -- this row as a *crop*: not a file on disk but a rectangle of another
    -- media, materialized on the fly into a PNG cache and, at deploy time,
    -- into the deploy folder. A crop lives only inside a dataset and is
    -- excluded from every Media/Libraries listing, from the index and from
    -- the lookalike detection. Its sha256 is synthetic (the parent's hash
    -- combined with the rectangle, see src.crops.crop_sha256), so it names
    -- its deployed files like any other media and two identical crops of the
    -- same parent collapse onto one row. Deleting the parent cascades.
    parent_media_id INTEGER REFERENCES media (id) ON DELETE CASCADE,
    -- The crop rectangle as a JSON ``{"x","y","w","h"}`` object, each value a
    -- percentage of the *source* image (0-100). Percentages, not pixels, so
    -- the rectangle stays valid whatever the source is re-encoded to.
    crop_rect TEXT,
    -- The aspect ratio the rectangle was locked to ('free', '1:1', '16:9'
    -- ...): UI state, restored when the crop is reopened for editing.
    crop_ratio TEXT
);

-- Per-media, per-metric quality scores. A media carries one row per quality
-- metric it was scored with (see src.quality.QUALITY_METRICS), so switching
-- the Settings metric adds a score rather than overwriting the previous one
-- -- the grids can then display or sort by any stored metric, or their
-- normalized average. Replaces the single media.quality_score/quality_metric
-- columns (backfilled and dropped in ensure_database). ``score`` is the raw
-- value in the metric's native range; ``metric_id`` is the QUALITY_METRICS
-- key it was computed with (never the "average" pseudo-metric, which is
-- derived on read).
CREATE TABLE IF NOT EXISTS media_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    metric_id TEXT NOT NULL,
    score REAL NOT NULL,
    scored_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (media_id, metric_id)
);

-- The UNIQUE(media_id, metric_id) index leads with media_id, so it does not
-- help a lookup by metric alone (listing the metrics present across the
-- library, or the per-metric quality sort's join).
CREATE INDEX IF NOT EXISTS idx_media_quality_metric
    ON media_quality (metric_id);

-- Visual embedding vectors (foundation of the dataset auto-builder's
-- diversity selection): the float32 bytes of a model's L2-normalized
-- feature vector for the media's original image, filled by the Libraries
-- tab's on-demand "Embeddings" action (images only — videos are excluded,
-- like the lookalike detection). One row per (media, model), so a future
-- model can coexist with the current one.
CREATE TABLE IF NOT EXISTS media_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (media_id, model_id)
);

CREATE TABLE IF NOT EXISTS library (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL DEFAULT 'folder'
        CHECK (kind IN ('internal', 'folder')),
    recursive INTEGER NOT NULL DEFAULT 1,
    -- Subfolder mapping (see src.folder_rules). A *sub-library* is a real
    -- library row promoted from one of a parent folder library's
    -- sub-folders: parent_library_id points at that parent and rel_path is
    -- the sub-folder's path relative to the parent root. Both NULL on a
    -- normal top-level library. Deleting a parent SET-NULLs its children,
    -- promoting them to top-level rather than dropping their media.
    parent_library_id INTEGER REFERENCES library (id) ON DELETE SET NULL,
    rel_path TEXT,
    -- Auto-tag-folder-names rule of this library's mapping: '0' off, '1' the
    -- top level only, 'all' every depth. Read at scan to derive folder-name
    -- tags (see src.folder_rules.effective_tags).
    auto_tag_level TEXT NOT NULL DEFAULT '0',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS media_file (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    library_id INTEGER
        REFERENCES library (id) ON DELETE SET NULL,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per mapped sub-folder of a folder library (see src.folder_rules).
-- The mapping the "Subfolder mapping" wizard persists: how a sub-folder is
-- routed on scan (`keep` stays in the parent, `sublib` promotes it to its own
-- library, `exclude` skips it) and its own tag rule. ``rel_path`` is the
-- folder path relative to the library root ('' is the root itself). ``tags``
-- and ``removed`` are JSON arrays of tag *names*: the folder's manually added
-- tags, and the inherited/auto tags overridden off for it. ``sub_library_id``
-- links a `sublib` row to the library it created (SET NULL if that library is
-- later deleted). Rules are persistent: a later scan re-resolves every file
-- against them, so files added to a mapped folder inherit its tags and
-- library automatically.
CREATE TABLE IF NOT EXISTS library_folder_rule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id INTEGER NOT NULL
        REFERENCES library (id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'keep'
        CHECK (mode IN ('keep', 'sublib', 'exclude')),
    sub_library_id INTEGER REFERENCES library (id) ON DELETE SET NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    removed TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (library_id, rel_path)
);

CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    deploy_name TEXT,
    deploy_resolution INTEGER NOT NULL DEFAULT 1280,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    hidden INTEGER NOT NULL DEFAULT 0,
    repeats INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (dataset_id, media_id)
);

CREATE TABLE IF NOT EXISTS caption_type (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    file_extension TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS caption (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    caption_type_id INTEGER NOT NULL
        REFERENCES caption_type (id),
    head_revision_id INTEGER
        REFERENCES caption_revision (id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (media_id, caption_type_id)
);

CREATE TABLE IF NOT EXISTS caption_revision (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caption_id INTEGER NOT NULL
        REFERENCES caption (id) ON DELETE CASCADE,
    parent_revision_id INTEGER
        REFERENCES caption_revision (id),
    content TEXT NOT NULL DEFAULT '',
    message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_caption (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    caption_id INTEGER NOT NULL
        REFERENCES caption (id) ON DELETE CASCADE,
    revision_id INTEGER
        REFERENCES caption_revision (id),
    mode TEXT NOT NULL DEFAULT 'follow'
        CHECK (mode IN ('follow', 'pinned')),
    UNIQUE (dataset_id, caption_id)
);

-- Caption review (see src.caption_review). A review is attached to a
-- *revision*, so a new revision naturally supersedes the previous review.
-- ``status`` is 'ok' | 'integrity'; ``issues`` is a JSON list of
-- ``{code, detail}`` dicts. ``judge_model`` is a legacy column of the
-- retired veracity judge, always NULL for the integrity heuristics.
CREATE TABLE IF NOT EXISTS caption_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id INTEGER NOT NULL UNIQUE
        REFERENCES caption_revision (id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    issues TEXT NOT NULL,
    judge_model TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- SigLIP grounding of a caption (see src.siglip_grounding). Like a review
-- it hangs off a *revision*, so editing the caption drops its scores
-- instead of showing numbers for text that no longer exists. ``model_id``
-- is the checkpoint that produced them: read back under a different one,
-- the UI marks the run stale rather than trusting incomparable scores.
CREATE TABLE IF NOT EXISTS caption_grounding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id INTEGER NOT NULL UNIQUE
        REFERENCES caption_revision (id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One atomic claim of a grounded caption: the LLM's decomposition
-- (``text`` + ``kind``, see src.caption_claims) carrying the SigLIP score
-- it earned. ``position`` preserves the reading order of the caption.
-- ``rejected`` is the user's "marked non-validated" flag: a claim they
-- judged unsupported, greyed out and excluded from the coverage union.
CREATE TABLE IF NOT EXISTS caption_grounding_claim (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grounding_id INTEGER NOT NULL
        REFERENCES caption_grounding (id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    score REAL NOT NULL,
    rejected INTEGER NOT NULL DEFAULT 0,
    UNIQUE (grounding_id, position)
);

-- SigLIP grounding of a media's tags: each tag scored through the fixed
-- pre-prompt (no LLM). Keyed on the media, not a dataset — the same
-- picture keeps its verdict everywhere it is used. ``model_id`` is part of
-- the key, so switching checkpoint re-scores rather than silently reusing
-- stale numbers. A tag detached from the media leaves its row orphaned;
-- every reader joins ``media_tag``, which hides it.
CREATE TABLE IF NOT EXISTS media_tag_grounding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tag (id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (media_id, tag_id, model_id)
);

-- Reference-free caption score (see src.caption_score). Like a grounding it
-- hangs off a *revision*, so editing the caption drops its scores. One row
-- per (revision, encoder family): a re-run of that family upserts.
-- ``model_id`` is the checkpoint that produced the score — read it back
-- under a different one, the UI marks the line stale instead of trusting
-- an incomparable number.
CREATE TABLE IF NOT EXISTS caption_score (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id INTEGER NOT NULL
        REFERENCES caption_revision (id) ON DELETE CASCADE,
    model_kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (revision_id, model_kind)
);

-- Reference-free score of a media's *tags* (see src.caption_score), the Media
-- tab's "Tags Score" card. The tags are scored as one comma-joined text, so
-- the row hangs off the media (not a caption revision) with one row per
-- encoder family. ``scored_text`` is the exact tag string that produced the
-- score: when the media's tags change, the stored text no longer matches and
-- the UI flags the line stale rather than showing a number for tags that are
-- gone. ``model_id`` is the checkpoint, flagged stale the same way.
CREATE TABLE IF NOT EXISTS media_tag_score (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    model_kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    score REAL NOT NULL,
    scored_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (media_id, model_kind)
);

CREATE TABLE IF NOT EXISTS triggerword (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dataset_triggerword (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    triggerword_id INTEGER NOT NULL
        REFERENCES triggerword (id) ON DELETE CASCADE,
    UNIQUE (dataset_id, triggerword_id)
);

-- Rule-based caption review (see src.caption_judge). A judge model, chosen
-- independently from the captioner, checks each caption against rules written
-- in plain language and scoped per dataset. ``kind`` selects the engine:
-- 'det' is deterministic (regex / substring, no model), 'text' a text-only
-- LLM pass, 'vlm' a vision judge. ``needs_image`` gates whether the image is
-- loaded for the rule. ``builtin`` marks a shipped preset. Rules are never
-- applied silently: they only ever produce ``review_finding`` rows.
CREATE TABLE IF NOT EXISTS review_rule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('det', 'text', 'vlm')),
    needs_image INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    builtin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One review run over a dataset. ``scope`` records what was targeted;
-- ``judge_model`` the checkpoint that judged. A run over the whole dataset
-- replaces its findings; a single-media run merges into the existing queue
-- (see src.sqlite_store.review_queue).
CREATE TABLE IF NOT EXISTS review_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    judge_model TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'all'
        CHECK (scope IN ('all', 'selection', 'flagged', 'single')),
    total INTEGER NOT NULL DEFAULT 0,
    findings_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);

-- One proposed correction awaiting human validation. ``rule_id`` NULL marks
-- an integrity finding (the src.caption_review heuristics), tagged with
-- ``rule_kind = 'integrity'`` so the queue badges it without a join. ``note``
-- is the judge's one-sentence verdict; ``caption_before`` / ``caption_after``
-- feed the word diff shown in the queue and wizard. ``status`` is the human
-- decision; ``applied_caption`` holds the final text when the user edited it
-- before accepting.
CREATE TABLE IF NOT EXISTS review_finding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL
        REFERENCES review_run (id) ON DELETE CASCADE,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    caption_type_id INTEGER NOT NULL
        REFERENCES caption_type (id) ON DELETE CASCADE,
    rule_id INTEGER
        REFERENCES review_rule (id) ON DELETE SET NULL,
    rule_kind TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    caption_before TEXT NOT NULL DEFAULT '',
    caption_after TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'rejected')),
    applied_caption TEXT,
    -- The live caption at accept time, so an undo restores exactly the text
    -- the accept replaced (never the run-time original, which other accepts
    -- may have already moved past).
    undo_caption TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT
);

-- The last quality report of a dataset (Datasets → Quality report tab): a
-- JSON blob of the whole evaluation, so the tab renders instantly on open
-- instead of re-running the scorers. One row per dataset — a run replaces
-- the previous report. ``duration_s`` and ``created_at`` back the toolbar's
-- "last run 1 h ago · 16 media · 42 s" note.
CREATE TABLE IF NOT EXISTS dataset_report (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL UNIQUE
        REFERENCES dataset (id) ON DELETE CASCADE,
    payload TEXT NOT NULL,
    scorers TEXT NOT NULL DEFAULT '',
    caption_type TEXT NOT NULL DEFAULT '',
    duration_s REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- How the user handled one finding of a report. ``issue_key`` is the
-- finding's stable id (see :class:`src.dataset_issues.Issue`) and
-- ``fingerprint`` a digest of the measurement behind it: a re-run keeps the
-- resolution while the fingerprint matches, and drops it once the finding
-- has genuinely changed (a recaptioned image, a new similarity).
-- The Auto-build Studio recipe a dataset was created from: the full set
-- of selection parameters (subject, quality, framing, size, manual edits),
-- as one JSON blob. ``live`` marks a "living" dataset whose recipe is
-- replayed after each index to propose upgrades.
CREATE TABLE IF NOT EXISTS autobuild_recipe (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL UNIQUE
        REFERENCES dataset (id) ON DELETE CASCADE,
    payload TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_issue_resolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL
        REFERENCES dataset (id) ON DELETE CASCADE,
    issue_key TEXT NOT NULL,
    resolution TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    resolved_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (dataset_id, issue_key)
);

CREATE TABLE IF NOT EXISTS tag_category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL DEFAULT '#888888',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tag (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL
        REFERENCES tag_category (id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (category_id, name)
);

CREATE TABLE IF NOT EXISTS media_tag (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL
        REFERENCES media (id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL
        REFERENCES tag (id) ON DELETE CASCADE,
    -- Provenance of the link, so the "Auto-tags" index step counts a media
    -- as tagged only on a real per-media tag. 'library' marks a tag pushed
    -- by the Libraries "Bulk tags" action (add to every media of a library);
    -- NULL is any genuine per-media tag (a WD14 auto-tag from the index or
    -- the Media tab, or a manual add). A non-library add promotes an
    -- existing 'library' link to NULL (see add_tag_to_media).
    source TEXT,
    UNIQUE (media_id, tag_id)
);

-- The UNIQUE(media_id, tag_id) index above has media_id as its leading
-- column, so it does not help a lookup by tag_id alone (e.g. filtering media
-- by tag). With a large tag catalogue (an imported danbooru/e621-style tag
-- list runs into the hundreds of thousands of tags) that reverse lookup
-- would otherwise be a full table scan.
CREATE INDEX IF NOT EXISTS idx_media_tag_tag_id ON media_tag (tag_id);

-- Watermark zones (see src.watermark): one row per detected/patched
-- filigrane on a media. Non-destructive — the source file is never touched.
-- A zone records its bounding box in fractions (0-1) of the SOURCE image so
-- it survives a re-encode, plus the FLUX edit patch that virtually erases it.
-- The patch is a small PNG (the dilated bbox) stored under src.constants
-- PATCHES_DIR and composed over the original at display and deploy time, so a
-- media can carry N independent zones. ``status`` is per-zone
-- (detected|patched); a media's flatten flag lives on media.wm_flattened.
CREATE TABLE IF NOT EXISTS watermark_zone (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL REFERENCES media (id) ON DELETE CASCADE,
    -- Bounding box of the watermark, fractions (0-1) of the source image.
    x REAL NOT NULL,
    y REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL,
    -- detected (found, no patch) | patched (erased). No "review" in v2: a box
    -- under the confidence floor is never created rather than flagged.
    status TEXT NOT NULL DEFAULT 'detected',
    -- Label of the FLUX.2 klein engine that produced the current patch, e.g.
    -- 'klein 9B KV fp8' | 'klein 4B' | NULL until a patch exists.
    model TEXT,
    -- Diffusion seed of the current patch (a fresh seed varies the generative
    -- fill so "regenerate — new seed" gives a different erase to iterate on).
    seed INTEGER,
    -- Detection confidence 0-100 (a manual/caption zone has none).
    score REAL,
    -- sha256 of the current patch PNG bytes, the composition cache key: any
    -- regeneration changes it, invalidating the composited image. NULL until
    -- a patch is generated.
    patch_sha TEXT,
    -- Where the box came from: 'owlv2' | 'yolo' | 'manual'.
    detector TEXT,
    -- The OWLv2 text query that matched (e.g. 'copyright text'), NULL for a
    -- YOLO or manual zone.
    query TEXT,
    -- Mask dilation (px on the source) grown around the box before the crop.
    dilate_px INTEGER NOT NULL DEFAULT 8,
    -- Edit instruction of the current patch, NULL when the zone rode the
    -- global prompt (surchargeable per zone in the Lab).
    prompt TEXT,
    -- Wall-clock FLUX edit time of the current patch, milliseconds.
    edit_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_watermark_zone_media
    ON watermark_zone (media_id);

-- One media per content hash (among the non-deleted ones): the same bytes
-- reached through several files must map to a single media row.
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_sha256_live
    ON media (sha256) WHERE deleted_at IS NULL;
"""


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""
    return DB_PATH


def _is_uri(target) -> bool:
    """Return whether ``target`` is an SQLite URI rather than a file path."""
    return isinstance(target, str) and target.startswith("file:")


def connect(db_path=None) -> sqlite3.Connection:
    """Open a configured connection to the database.

    Foreign keys and WAL are on. The caller owns and must close it (per-call
    connections are the safe pattern under worker threads). ``db_path`` is a
    file or SQLite ``file:`` URI (tests pass in-memory URIs), default
    :func:`get_db_path`. Rows are ``sqlite3.Row``.
    """
    target = db_path if db_path is not None else get_db_path()
    conn = sqlite3.connect(str(target), uri=_is_uri(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def checkpoint_wal(db_path=None) -> None:
    """Flush the write-ahead log back into the main database file.

    Best effort: called on an intentional shutdown so the ``.db`` file is
    self-contained and the ``-wal``/``-shm`` side files hold nothing pending.
    ``TRUNCATE`` checkpoints then shrinks the WAL to zero. Any engine error
    (locked, missing) is swallowed — a failed flush must never block exit.
    """
    try:
        with closing(connect(db_path)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass


def connect_readonly(db_path=None) -> sqlite3.Connection:
    """Open a read-only connection to the database.

    Writes raise at the engine level (safe for the developer explorer). An
    in-memory URI can't carry ``mode=ro`` (``mode=memory`` holds the slot), so
    it's opened normally and locked with ``PRAGMA query_only``, enforced the
    same way. ``db_path`` default :func:`get_db_path`.
    """
    target = db_path if db_path is not None else get_db_path()
    if _is_uri(target):
        conn = sqlite3.connect(str(target), uri=True)
        conn.execute("PRAGMA query_only = ON")
    else:
        conn = sqlite3.connect(f"{Path(target).as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """Add a column to an existing table when it is missing (idempotent).

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a new
    column on a shipped table needs this additive step for older development
    databases. ``table``/``column``/``definition`` are code constants (safe to
    interpolate).
    """
    existing = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names of an existing table.

    ``table`` is a code constant (safe to interpolate).
    """
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _drop_column_if_exists(
    conn: sqlite3.Connection, table: str, column: str
) -> None:
    """Drop a column from an existing table when present (idempotent).

    The counterpart of :func:`_ensure_column`, for a retired column an older
    development database still carries. ``table``/``column`` are code
    constants (safe to interpolate).
    """
    if column in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


def _migrate_media_quality(conn: sqlite3.Connection) -> None:
    """Move the old media quality columns into the media_quality table.

    Before per-metric scoring, a media carried one ``quality_score``/
    ``quality_metric`` on ``media`` itself; now it's one :data:`media_quality`
    row per metric. Old development databases migrate in place: each non-null
    score copies into ``media_quality`` (null metric = MUSIQ) and the columns
    drop. Idempotent — a no-op on a fresh database.
    """
    columns = _table_columns(conn, "media")
    if "quality_score" not in columns:
        return
    conn.execute(
        "INSERT OR IGNORE INTO media_quality (media_id, metric_id, score) "
        "SELECT id, COALESCE(quality_metric, 'musiq'), quality_score "
        "FROM media WHERE quality_score IS NOT NULL"
    )
    conn.execute("ALTER TABLE media DROP COLUMN quality_score")
    if "quality_metric" in columns:
        conn.execute("ALTER TABLE media DROP COLUMN quality_metric")


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Insert or update the single ``schema_version`` row."""
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version,)
        )
    else:
        conn.execute("UPDATE schema_version SET version = ?", (version,))


def get_schema_version(db_path=None) -> int | None:
    """Return the stored schema version, or None if the table is empty."""
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
    return row["version"] if row else None


def seed_default_tag_categories(db_path=None) -> None:
    """Seed the default tag categories, but only when none exist yet.

    Seeding only into an empty ``tag_category`` table means a deleted default
    is never recreated (unlike the additive caption types). Each is inserted
    with its factory color.
    """
    with closing(connect(db_path)) as conn:
        with conn:
            existing = conn.execute(
                "SELECT COUNT(*) AS n FROM tag_category"
            ).fetchone()["n"]
            if existing:
                return
            for name, color in DEFAULT_TAG_CATEGORIES:
                conn.execute(
                    "INSERT OR IGNORE INTO tag_category (name, color) "
                    "VALUES (?, ?)",
                    (name, color),
                )


def seed_internal_library(db_path=None) -> None:
    """Create the single internal upload library if it does not exist yet.

    The internal library is the one folder the app writes to: drag-drop uploads
    land in :data:`~src.constants.STORAGE_DIR`. Seeded only when no
    ``internal`` library exists, so a renamed one is never duplicated.
    """
    with closing(connect(db_path)) as conn:
        with conn:
            existing = conn.execute(
                "SELECT 1 FROM library WHERE kind = 'internal' LIMIT 1"
            ).fetchone()
            if existing:
                return
            conn.execute(
                "INSERT OR IGNORE INTO library (name, path, kind, recursive) "
                "VALUES (?, ?, 'internal', 0)",
                ("Internal (uploads)", str(STORAGE_DIR)),
            )


def seed_caption_types(extensions, db_path=None) -> None:
    """Insert a ``caption_type`` row for each extension if absent.

    Idempotent (``INSERT OR IGNORE`` on the unique name). Each type's ``name``
    and ``file_extension`` are both the extension; the file extension names the
    caption file on export.
    """
    with closing(connect(db_path)) as conn:
        with conn:
            for ext in extensions:
                conn.execute(
                    "INSERT OR IGNORE INTO caption_type "
                    "(name, file_extension) VALUES (?, ?)",
                    (ext, ext),
                )


def ensure_database(db_path=None):
    """Create the database folder, file and schema if they do not exist.

    Idempotent. Creates the parent folder, opens (creates) the file, applies
    the schema and records :data:`SCHEMA_VERSION`. Post-deployment, the
    ``database/migrations/`` scripts will apply here too. ``db_path`` may be an
    SQLite ``file:`` URI (in-memory skips folder creation). Returns the ready
    path or URI.
    """
    target = db_path if db_path is not None else get_db_path()
    if not _is_uri(target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(target)) as conn:
        with conn:
            conn.executescript(_SCHEMA)
            # Additive columns for development databases created before them.
            _ensure_column(
                conn,
                "tag_category",
                "position",
                "position INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                conn,
                "dataset",
                "deploy_resolution",
                "deploy_resolution INTEGER NOT NULL DEFAULT 1280",
            )
            for column, definition in (
                ("favorite", "favorite INTEGER NOT NULL DEFAULT 0"),
                ("width", "width INTEGER"),
                ("height", "height INTEGER"),
                ("indexed_at", "indexed_at TEXT"),
                ("phash", "phash TEXT"),
                ("dhash", "dhash TEXT"),
                ("sharpness", "sharpness REAL"),
                ("clipping", "clipping REAL"),
                ("cleanliness", "cleanliness REAL"),
                ("discarded_at", "discarded_at TEXT"),
                ("lookalike_reviewed_at", "lookalike_reviewed_at TEXT"),
                (
                    "parent_media_id",
                    "parent_media_id INTEGER REFERENCES media (id) "
                    "ON DELETE CASCADE",
                ),
                ("crop_rect", "crop_rect TEXT"),
                ("crop_ratio", "crop_ratio TEXT"),
                # Watermark "flatten to disk": the media's virtual patches were
                # baked into its source file. wm_flat_sha keeps the pre-flatten
                # content sha so a revert restores both the bytes (from the
                # backup) and the media's content identity.
                ("wm_flattened", "wm_flattened INTEGER NOT NULL DEFAULT 0"),
                ("wm_flat_sha", "wm_flat_sha TEXT"),
            ):
                _ensure_column(conn, "media", column, definition)
            # Watermark Lab v2 retired the media-level exclusion flag: an
            # irrecoverable filigrane is now just left un-patched (the media
            # stays neutral, simply hard to patch), so the column is dropped.
            _drop_column_if_exists(conn, "media", "wm_excluded")
            # Indexed after the additive columns, never inside _SCHEMA: an
            # existing database runs the script against a media table that
            # does not have parent_media_id yet. A crop is looked up by its
            # parent (the panels list a media's crops) and the column is
            # sparse, so the partial index stays tiny.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_parent "
                "ON media (parent_media_id) "
                "WHERE parent_media_id IS NOT NULL"
            )
            _ensure_column(conn, "media_tag", "source", "source TEXT")
            # Subfolder mapping columns on the library table (see
            # src.folder_rules): back-filled for databases created before it.
            for column, definition in (
                (
                    "parent_library_id",
                    "parent_library_id INTEGER "
                    "REFERENCES library (id) ON DELETE SET NULL",
                ),
                ("rel_path", "rel_path TEXT"),
                (
                    "auto_tag_level",
                    "auto_tag_level TEXT NOT NULL DEFAULT '0'",
                ),
            ):
                _ensure_column(conn, "library", column, definition)
            # Watermark zones moved from LaMa/OpenCV inpaint to FLUX.2 klein
            # generative edits: the engine label, the per-zone edit prompt and
            # the edit timing replace the old method/inpaint_ms columns (left
            # behind, unused, on databases created before the switch).
            for column, definition in (
                ("model", "model TEXT"),
                ("prompt", "prompt TEXT"),
                ("edit_ms", "edit_ms INTEGER"),
                # OWLv2 (v2 default) records which text query matched, so the
                # review panel can show `OWLv2 · "copyright text"`.
                ("query", "query TEXT"),
            ):
                _ensure_column(conn, "watermark_zone", column, definition)
            # Undo snapshot for review accepts (see review_queue): databases
            # created before conflict-aware accepts lack the column.
            _ensure_column(
                conn, "review_finding", "undo_caption", "undo_caption TEXT"
            )
            # v2 retired the per-zone "review" status: any pre-v2 zone still
            # carrying it becomes a plain detected zone (idempotent).
            conn.execute(
                "UPDATE watermark_zone SET status = 'detected' "
                "WHERE status NOT IN ('detected', 'patched')"
            )
            # Per-metric quality scores now live in media_quality: backfill
            # from the old single-score columns, then drop them.
            _migrate_media_quality(conn)
            _set_schema_version(conn, SCHEMA_VERSION)
    seed_default_tag_categories(target)
    seed_internal_library(target)
    return target
