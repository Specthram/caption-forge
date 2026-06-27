"""Dataset deployment for the SQLite storage mode.

Deploying mirrors a dataset to disk for training: every *visible* media is
copied into ``<deploy_root>/<deploy_name>/`` (folder name defaults to the
dataset name) beside a ``<stem>.txt`` caption, and stale files (hidden/removed
media, leftover captions) are pruned. A ``repeats`` count above one copies the
media that many times (``<hash>``, ``<hash>_1``…, each with its caption).

Deployment is *differential*: a copy already present with the wanted caption is
left alone, so a redeploy writes only what changed. Scope: one media
(:func:`deploy_media` / :func:`undeploy_media`) or the whole dataset
(:func:`undeploy_dataset`).

The caption is the media's *active* type (per-image override, else the
gallery-wide type) at its effective revision, trigger-word prefix prepended —
always ``.txt`` regardless of the caption type's extension.

Sync state is derived live by diffing the folder against the DB, so it's
correct after a refresh/restart with no stored state. Each media (and the
dataset) carries one of:

* ``RED`` — not deployed at all (no image file in the folder);
* ``ORANGE`` — deployed but out of date (caption differs from what is saved);
* ``GREEN`` — deployed and up to date;
* ``WHITE`` — hidden (excluded from deployment);
* ``NONE`` — no deploy folder configured, so there is nothing to compare.
"""

import os
import shutil
from pathlib import Path

from src import storage
from src import sqlite_store as store
from src.settings import get_deploy_dir

# Synchronization states. The badge a media or dataset shows is the disc in
# ``_BADGE_EMOJI`` for its state.
RED = "red"
ORANGE = "orange"
GREEN = "green"
WHITE = "white"
NONE = "none"

_BADGE_EMOJI = {
    RED: "🔴",
    ORANGE: "🟠",
    GREEN: "🟢",
    WHITE: "⚪",
    NONE: "",
}

# Characters not allowed in a Windows path component, replaced by "_" so a
# dataset name always maps to a valid sub-folder.
_UNSAFE_NAME_CHARS = '<>:"/\\|?*'


def _sanitize_name(name: str) -> str:
    """Return a dataset name reduced to a safe single path component."""
    cleaned = "".join(
        "_" if ch in _UNSAFE_NAME_CHARS else ch for ch in (name or "").strip()
    )
    return cleaned.rstrip(". ") or "dataset"


def deploy_root() -> Path | None:
    """Return the configured deploy root directory, or None when unset."""
    return get_deploy_dir()


def dataset_deploy_dir(dataset_id: int) -> Path | None:
    """Return a dataset's deploy folder, or None when unavailable.

    The folder is ``<deploy_root>/<deploy_name>/`` where ``deploy_name`` is
    the dataset's configured deploy folder name, falling back to the dataset
    name when unset; None is returned when no deploy root is configured or
    the dataset does not exist.
    """
    root = deploy_root()
    if root is None:
        return None
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        return None
    return root / _sanitize_name(dataset["deploy_name"] or dataset["name"])


def dataset_deploy_resolution(dataset_id: int) -> int:
    """Return a dataset's deploy resize target (shortest image side, px).

    ``0`` means resizing is disabled and media are deployed verbatim.
    """
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        return 0
    try:
        return max(0, int(dataset["deploy_resolution"]))
    except (TypeError, ValueError, KeyError, IndexError):
        return 0


def deployed_ext(file_ext: str, is_video: bool, resolution: int) -> str:
    """Return the on-disk extension a media deploys under.

    With a resize resolution set, an image is re-encoded to lossless PNG so
    its deployed extension becomes ``png``; videos are never resized and keep
    their original extension, as does every media when resizing is off.
    """
    if resolution and resolution > 0 and not is_video:
        return "png"
    return file_ext


def _copy_stem(sha256: str, index: int) -> str:
    """Return the filename stem of a media's ``index``-th deployed copy.

    The first copy keeps the plain hash stem; each extra repeat gets a
    ``_<index>`` suffix (``<hash>_1``, ``<hash>_2``...), so a repeat count of
    3 deploys ``<hash>``, ``<hash>_1`` and ``<hash>_2``.
    """
    return sha256 if index == 0 else f"{sha256}_{index}"


def _image_name(sha256: str, file_ext: str, index: int = 0) -> str:
    """Return a deployed image filename (``<hash>[_<index>]<ext>``).

    Files are named by their content hash so two media that share a basename
    never collide in the deploy folder.
    """
    suffix = (file_ext or "").lower().lstrip(".")
    suffix = f".{suffix}" if suffix else ""
    return f"{_copy_stem(sha256, index)}{suffix}"


def _caption_name(sha256: str, index: int = 0) -> str:
    """Return a deployed caption filename (``<hash>[_<index>].txt``)."""
    return f"{_copy_stem(sha256, index)}.txt"


def _repeats(item_repeats) -> int:
    """Return a sanitized repeat count (always at least one copy)."""
    try:
        return max(1, int(item_repeats or 1))
    except (TypeError, ValueError):
        return 1


def _wanted_caption(dataset_id: int, key: str, ext: str) -> str:
    """Return a media's deploy caption text (trigger-word prefixed)."""
    prefix = store.triggerword_prefix(dataset_id)
    text = storage.read_caption(dataset_id, key, ext)
    return f"{prefix}{text}"


def _read_deployed_caption(caption_path: Path) -> str | None:
    """Return a deployed caption file's text, or None when it is absent."""
    if not caption_path.is_file():
        return None
    try:
        return caption_path.read_text(encoding="utf-8")
    except OSError:
        return None


def image_status(
    dataset_id: int,
    key: str,
    ext: str,
    hidden: bool,
    sha256: str | None,
    file_ext: str = "",
    repeats: int = 1,
    wanted: str | None = None,
) -> str:
    """Return the deployment state of one media.

    ``repeats`` copies must all be present + up to date (no extra) to be
    :data:`GREEN`. ``wanted`` is the pre-resolved caption (trigger prefix
    included) a page painter passes from a bulk read so 40 cards don't each
    re-query; None resolves it here. Returns :data:`NONE`/:data:`WHITE`/
    :data:`RED`/:data:`ORANGE`/:data:`GREEN`.
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None:
        return NONE
    if hidden:
        return WHITE
    if not (deploy_dir / _image_name(sha256, file_ext)).exists():
        return RED
    if wanted is None:
        wanted = _wanted_caption(dataset_id, key, ext)
    in_sync = _copies_in_sync(
        deploy_dir, wanted, sha256, file_ext, _repeats(repeats)
    )
    return GREEN if in_sync else ORANGE


def _copies_in_sync(
    deploy_dir: Path,
    wanted: str,
    sha256: str,
    file_ext: str,
    repeats: int,
    present=None,
) -> bool:
    """Return whether every deployed copy matches the wanted state.

    Every copy up to ``repeats`` must exist with the wanted caption text, and
    no copy beyond that count may linger (repeats was lowered since the last
    deploy). Deploys write contiguous indices and the prune removes
    everything beyond, so checking the one next index is enough.

    ``present`` is an optional pre-listed set of the folder's file names
    (one ``os.listdir`` shared across a whole dataset check) used for the
    existence tests instead of a disk stat per copy.
    """

    def exists(name: str) -> bool:
        if present is not None:
            return name in present
        return (deploy_dir / name).exists()

    for index in range(repeats):
        if not exists(_image_name(sha256, file_ext, index)):
            return False
        deployed = _read_deployed_caption(
            deploy_dir / _caption_name(sha256, index)
        )
        if deployed != wanted:
            return False
    return not exists(_image_name(sha256, file_ext, repeats))


def dataset_status(dataset_id: int, items: list, captions=None) -> str:
    """Return the aggregate deployment state of a dataset.

    ``items`` is one dict per media (``key``/``ext``/``hidden``/``sha256``/
    ``file_ext``/``missing``/``repeats``). ``captions`` optionally pre-resolves
    caption texts (``{key: text}``, no prefix) from a bulk read; a missing key
    resolves per media. Returns :data:`NONE` (no folder), :data:`RED` (nothing
    deployed), :data:`GREEN` (all visible up to date, no stray), else
    :data:`ORANGE` on the first out-of-sync media (no full walk).
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None:
        return NONE
    # One folder listing for every per-item existence test below (a stat per
    # copy adds up on a large dataset). A missing folder means nothing is
    # deployed, same as an empty one.
    try:
        present = set(os.listdir(deploy_dir))
    except OSError:
        present = set()
    if not present:
        return RED

    prefix = store.triggerword_prefix(dataset_id)
    for item in items:
        image_name = _image_name(item["sha256"], item.get("file_ext", ""))
        if item["hidden"] or item.get("missing"):
            # An excluded media (hidden or missing) must be absent to be in
            # sync.
            if image_name in present:
                return ORANGE
            continue
        if image_name not in present:
            return ORANGE
        if captions is not None and item["key"] in captions:
            wanted = f"{prefix}{captions[item['key']]}"
        else:
            wanted = _wanted_caption(dataset_id, item["key"], item["ext"])
        if not _copies_in_sync(
            deploy_dir,
            wanted,
            item["sha256"],
            item.get("file_ext", ""),
            _repeats(item.get("repeats", 1)),
            present,
        ):
            return ORANGE
    return GREEN


def badge_html(status: str) -> str:
    """Return the inline HTML disc for a status (empty for :data:`NONE`)."""
    emoji = _BADGE_EMOJI.get(status, "")
    if not emoji:
        return ""
    return f'<span class="cf-deploy-badge" title="{status}">{emoji}</span>'


def deploy_dataset(dataset_id: int, items: list) -> dict:
    """Mirror a dataset's visible media into its deploy folder.

    Copies every visible media (``repeats`` copies each) with a ``.txt``
    caption, then prunes anything not belonging to the visible set (hidden,
    removed, stale captions, copies beyond the repeat count). ``items`` is one
    dict per media (``key``/``path``/``ext``/``hidden``/``sha256``/
    ``file_ext``/``missing``/``repeats``); deployed under their hash so
    basenames never collide; hidden/missing skipped. Returns ``{copied,
    written, removed,
    folder}`` — ``copied`` counts media, ``written`` counts files actually
    touched (differential). Raises ``ValueError`` when no folder resolves.
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None:
        raise ValueError(
            "No deploy folder configured — set a deploy directory in Settings."
        )
    deploy_dir.mkdir(parents=True, exist_ok=True)

    keep = set()
    copied = 0
    written = 0
    for item in items:
        if item["hidden"] or item.get("missing") or not item.get("path"):
            continue
        caption = _wanted_caption(dataset_id, item["key"], item["ext"])
        for index in range(_repeats(item.get("repeats", 1))):
            image_name = _image_name(
                item["sha256"], item.get("file_ext", ""), index
            )
            caption_name = _caption_name(item["sha256"], index)
            keep.add(image_name)
            keep.add(caption_name)
            written += _sync_copy(
                deploy_dir, item, image_name, caption_name, caption
            )
        copied += 1

    removed = _prune(deploy_dir, keep)
    return {
        "copied": copied,
        "written": written,
        "removed": removed,
        "folder": str(deploy_dir),
    }


def deploy_media(dataset_id: int, item: dict) -> dict:
    """Deploy a single media into the dataset's folder (differential, scoped).

    Writes just this media's copies (image + caption, honoring repeats) and
    prunes only its own leftover extras, leaving other media untouched.
    Hidden/missing writes nothing. ``item`` uses the same keys as
    :func:`deploy_dataset`. Returns ``{written, removed, folder, deployed}``
    (``deployed`` false when skipped). Raises ``ValueError`` when no folder
    resolves.
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None:
        raise ValueError(
            "No deploy folder configured — set a deploy directory in Settings."
        )
    if item["hidden"] or item.get("missing") or not item.get("path"):
        return {
            "written": 0,
            "removed": 0,
            "folder": str(deploy_dir),
            "deployed": False,
        }
    deploy_dir.mkdir(parents=True, exist_ok=True)
    caption = _wanted_caption(dataset_id, item["key"], item["ext"])
    repeats = _repeats(item.get("repeats", 1))
    written = 0
    for index in range(repeats):
        image_name = _image_name(
            item["sha256"], item.get("file_ext", ""), index
        )
        caption_name = _caption_name(item["sha256"], index)
        written += _sync_copy(
            deploy_dir, item, image_name, caption_name, caption
        )
    removed = _prune_media_extras(
        deploy_dir, item["sha256"], item.get("file_ext", ""), repeats
    )
    return {
        "written": written,
        "removed": removed,
        "folder": str(deploy_dir),
        "deployed": True,
    }


def undeploy_media(dataset_id: int, sha256: str, file_ext: str = "") -> dict:
    """Remove every deployed copy of one media from the dataset's folder.

    Deletes the media's image + caption files at all copy indices, leaving the
    rest untouched. Returns ``{removed, folder}``.
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None or not deploy_dir.is_dir():
        folder = str(deploy_dir) if deploy_dir is not None else ""
        return {"removed": 0, "folder": folder}
    removed = _prune_media_extras(deploy_dir, sha256, file_ext, 0)
    return {"removed": removed, "folder": str(deploy_dir)}


def undeploy_dataset(dataset_id: int) -> dict:
    """Remove every deployed file from the dataset's folder.

    Clears all images + captions so the dataset is fully undeployed; the
    now-empty folder is kept. Returns ``{removed, folder}``. Raises
    ``ValueError`` when no folder resolves.
    """
    deploy_dir = dataset_deploy_dir(dataset_id)
    if deploy_dir is None:
        raise ValueError(
            "No deploy folder configured — set a deploy directory in Settings."
        )
    if not deploy_dir.is_dir():
        return {"removed": 0, "folder": str(deploy_dir)}
    removed = _prune(deploy_dir, set())
    return {"removed": removed, "folder": str(deploy_dir)}


def _target_size(width: int, height: int, resolution: int) -> tuple:
    """Return the downscaled ``(w, h)`` for a shortest-side ``resolution``.

    Downscale only: an image whose shortest side is already at or below the
    target keeps its original size (never upscaled). Otherwise it is scaled so
    the shortest side lands exactly on ``resolution`` and the aspect ratio is
    preserved (the long side rounded to the nearest pixel, at least 1).
    """
    short = min(width, height)
    if resolution <= 0 or short <= resolution:
        return width, height
    if width <= height:
        return resolution, max(1, round(height * resolution / width))
    return max(1, round(width * resolution / height)), resolution


def _oriented_size(img) -> tuple:
    """Return an image's ``(w, h)`` after applying its EXIF orientation.

    Reads only the orientation tag (no full decode); orientations 5-8 swap the
    width and height, matching what :func:`_write_resized_png` bakes in.
    """
    orientation = img.getexif().get(0x0112, 1)
    if orientation in {5, 6, 7, 8}:
        return img.height, img.width
    return img.width, img.height


def _write_resized_png(source: Path, dest: Path, resolution: int) -> None:
    """Write ``source`` to ``dest`` as lossless PNG, downscaled to fit.

    The EXIF orientation is baked into the pixels (PNG carries no orientation
    tag) and the shortest side is reduced to ``resolution`` with a Lanczos
    filter when the image is larger than the target; smaller images are
    re-encoded at their original size.
    """
    # pylint: disable=import-outside-toplevel  # Pillow is only needed here.
    from PIL import Image, ImageOps

    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        target = _target_size(img.width, img.height, resolution)
        if target != img.size:
            img = img.resize(target, Image.Resampling.LANCZOS)
        img.save(dest, format="PNG")


def _needs_resize_write(source: Path, dest: Path, resolution: int) -> bool:
    """Return whether a resized copy must be (re)written for ``source``.

    True when the destination is absent or its dimensions no longer match the
    target for the current ``resolution`` (the setting was changed since the
    last deploy). Only image headers are read, so this stays cheap.
    """
    if not dest.exists():
        return True
    # pylint: disable=import-outside-toplevel  # Pillow is only needed here.
    from PIL import Image

    try:
        with Image.open(source) as src:
            width, height = _oriented_size(src)
        expected = _target_size(width, height, resolution)
        with Image.open(dest) as out:
            return out.size != expected
    except OSError:
        return True


def _sync_image(
    source: Path, image_dst: Path, resize: bool, resolution: int
) -> int:
    """Write one copy's image when needed; return 1 if a file was written.

    A resized image is (re)written only when missing or stale for the current
    target; an unresized image is copied only when absent.
    """
    if resize:
        if _needs_resize_write(source, image_dst, resolution):
            _write_resized_png(source, image_dst, resolution)
            return 1
        return 0
    if source.is_file() and _needs_copy(source, image_dst):
        shutil.copy2(source, image_dst)
        return 1
    return 0


def _needs_copy(source: Path, image_dst: Path) -> bool:
    """Return whether an unresized image must be (re)copied.

    Missing destination, or a source newer than what was deployed. The source
    can change without its name doing so when it is a virtual composite (a
    watermark-patched media, see :mod:`src.wm_compose`): a regenerated patch
    yields a fresh cache file whose mtime beats the stale deployed copy, so a
    redeploy picks it up. ``shutil.copy2`` preserves the mtime, so an
    unchanged source never re-copies.
    """
    if not image_dst.exists():
        return True
    try:
        return source.stat().st_mtime > image_dst.stat().st_mtime
    except OSError:
        return True


def _sync_copy(
    deploy_dir: Path,
    item: dict,
    image_name: str,
    caption_name: str,
    caption: str,
) -> int:
    """Copy one media copy's image (if needed) and write its caption file.

    Differential: unresized image copied only when absent; resized image
    (re)written only when missing/stale; caption (re)written only when content
    differs — an up-to-date copy touches disk not at all. Returns files written
    (0, 1 or 2).
    """
    written = 0
    image_dst = deploy_dir / image_name
    source = Path(item["path"])
    resolution = item.get("resolution") or 0
    resize = resolution > 0 and not item.get("is_video") and source.is_file()
    written += _sync_image(source, image_dst, resize, resolution)
    caption_dst = deploy_dir / caption_name
    if _read_deployed_caption(caption_dst) != caption:
        caption_dst.write_text(caption, encoding="utf-8")
        written += 1
    return written


def _prune_media_extras(
    deploy_dir: Path, sha256: str, file_ext: str, from_index: int
) -> int:
    """Delete a media's deployed copies from ``from_index`` upward.

    Deploys write contiguous indices, so removing image/caption pairs from
    ``from_index`` until a fully-absent index clears exactly this media's
    leftover copies (``from_index=0`` removes the media entirely). Returns how
    many files were deleted.
    """
    removed = 0
    index = from_index
    while True:
        found = False
        for path in (
            deploy_dir / _image_name(sha256, file_ext, index),
            deploy_dir / _caption_name(sha256, index),
        ):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                    found = True
                except OSError:
                    pass
        if not found:
            return removed
        index += 1


def _prune(deploy_dir: Path, keep: set) -> int:
    """Delete files in the deploy folder not in ``keep``; return the count."""
    removed = 0
    for path in deploy_dir.iterdir():
        if path.is_file() and path.name not in keep:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
