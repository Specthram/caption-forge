"""Library repository: sources, scanning, merging and listings.

A library is a scanned source folder (plus the one ``internal``
upload library). Also holds the paged media listings the grids
consume and the sidecar caption import a scan performs.
"""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from src import db
from src import folder_rules as fr
from src import media
from src.sqlite_store.base import (
    SORT_DATE_DESC,
    _query_all,
    _query_one,
    _sort_order_by,
    _write,
)
from src.sqlite_store.captions import (
    _insert_revision,
    get_caption_type,
    get_or_create_caption,
    get_revision,
    head_revision_id,
)
from src.sqlite_store.folder_rules import folder_rules_map
from src.sqlite_store.media import _media_dicts, ingest_file
from src.sqlite_store.tags import add_tag_to_media, get_or_create_tag_reuse


def _watermark_tab_sql(wm_tab) -> str:
    """Return the ``AND m.id (NOT) IN (...)`` clause for a Watermark Lab tab.

    ``watermarked`` keeps media with at least one still-detected zone;
    ``patched`` keeps media that carry zones but none detected (every
    watermark erased); anything else (``media``/None) adds no constraint. No
    bound params — the sub-selects are constant SQL.
    """
    if wm_tab == "watermarked":
        return (
            " AND m.id IN (SELECT media_id FROM watermark_zone "
            "WHERE status = 'detected')"
        )
    if wm_tab == "patched":
        return (
            " AND m.id IN (SELECT media_id FROM watermark_zone) "
            "AND m.id NOT IN (SELECT media_id FROM watermark_zone "
            "WHERE status = 'detected')"
        )
    return ""


def _library_media_query(
    tag_ids,
    match: str,
    favorites_only: bool = False,
    exclude_tag_ids=None,
    in_dataset_id=None,
    not_in_dataset_id=None,
    wm_tab=None,
):
    """Return ``(sql_body, params)`` for the filtered library media set.

    ``sql_body`` is everything *after* ``FROM media m`` up to (not
    including) the ``ORDER BY``: the extra joins, WHERE and GROUP BY
    selecting the non-deleted, non-archived library media that match the tag
    filter. The caller writes the ``FROM media m`` itself (and may splice a
    sort-specific join between it and this body — see
    :func:`src.sqlite_store.base._sort_order_by`). The library is every such
    media, whatever its source: media uploaded into the internal library and
    media referenced from scanned folders.

    Parameters
    ----------
    tag_ids : list of int or None
        The tag ids to filter on. Empty/None selects every library media.
    match : str
        ``"any"`` keeps media carrying at least one of the tags; ``"all"``
        keeps only media carrying every one of them.
    favorites_only : bool, optional
        When true, only media flagged as favorite are kept (the Medias tab's
        "Favorites only" filter).
    exclude_tag_ids : list of int or None, optional
        Tag ids to *exclude*: any media carrying at least one of them is
        dropped, whatever the include filter matched (the Medias/Datasets
        "Exclude tags" field).
    in_dataset_id : int, optional
        When given, only media linked to this dataset are kept (the
        Datasets tab's default view: the dataset's own content).
    not_in_dataset_id : int, optional
        When given, media linked to this dataset are dropped (the Datasets
        tab's "Add images" picker: only media not yet in the dataset).
    """
    tag_ids = [int(t) for t in (tag_ids or [])]
    exclude_tag_ids = [int(t) for t in (exclude_tag_ids or [])]
    fav = " AND m.favorite = 1" if favorites_only else ""
    if exclude_tag_ids:
        excl_ph = ", ".join("?" for _ in exclude_tag_ids)
        excl = (
            " AND m.id NOT IN (SELECT media_id FROM media_tag "
            f"WHERE tag_id IN ({excl_ph}))"
        )
    else:
        excl = ""
    dataset_sql = ""
    dataset_params = []
    if in_dataset_id is not None:
        dataset_sql += (
            " AND m.id IN (SELECT media_id FROM dataset_media "
            "WHERE dataset_id = ?)"
        )
        dataset_params.append(int(in_dataset_id))
    if not_in_dataset_id is not None:
        dataset_sql += (
            " AND m.id NOT IN (SELECT media_id FROM dataset_media "
            "WHERE dataset_id = ?)"
        )
        dataset_params.append(int(not_in_dataset_id))
    wm_sql = _watermark_tab_sql(wm_tab)
    # A virtual crop (see src.crops) is not a library media: it owns no file
    # on disk and lives inside a dataset only, so it appears neither in the
    # Medias grid nor in the Datasets "add images" picker.
    live = (
        "WHERE m.deleted_at IS NULL AND m.hidden = 0 "
        "AND m.discarded_at IS NULL AND m.parent_media_id IS NULL"
    )
    if not tag_ids:
        return (
            live + fav + excl + dataset_sql + wm_sql,
            list(exclude_tag_ids) + dataset_params,
        )
    placeholders = ", ".join("?" for _ in tag_ids)
    if match == "all":
        having = f"HAVING COUNT(DISTINCT mt.tag_id) = {len(tag_ids)}"
    else:
        having = ""
    return (
        "JOIN media_tag mt ON mt.media_id = m.id " + live + fav + " "
        f"AND mt.tag_id IN ({placeholders})"
        + excl
        + dataset_sql
        + wm_sql
        + " "
        + f"GROUP BY m.id {having}",
        tag_ids + list(exclude_tag_ids) + dataset_params,
    )


def count_library_media(
    tag_ids=None,
    match: str = "any",
    favorites_only: bool = False,
    exclude_tag_ids=None,
    in_dataset_id=None,
    not_in_dataset_id=None,
    wm_tab=None,
) -> int:
    """Return how many library media match a tag filter (no disk access).

    A cheap ``COUNT`` used to size the Medias/Datasets grid pagers without
    materializing (and stat-ing) every media like
    :func:`library_media_filtered`. ``in_dataset_id`` /
    ``not_in_dataset_id`` optionally scope the count to a dataset's
    membership (see :func:`_library_media_query`).
    """
    body, params = _library_media_query(
        tag_ids,
        match,
        favorites_only,
        exclude_tag_ids,
        in_dataset_id=in_dataset_id,
        not_in_dataset_id=not_in_dataset_id,
        wm_tab=wm_tab,
    )
    row = _query_one(
        f"SELECT COUNT(*) AS n FROM (SELECT m.id FROM media m {body})", params
    )
    return row["n"] if row else 0


def library_media_page(
    tag_ids,
    match: str,
    sort: str,
    offset: int,
    limit: int,
    quality_metric_selected=None,
    favorites_only: bool = False,
    exclude_tag_ids=None,
    in_dataset_id=None,
    not_in_dataset_id=None,
    wm_tab=None,
) -> list:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Return one page of the filtered library media as dicts.

    Only the returned rows have their effective file resolved (a disk stat
    per media), so paging a large library never stats every file — unlike
    :func:`library_media_filtered`, which materializes the whole set. Backs
    the Medias grid and the Datasets "add media" picker.

    Parameters
    ----------
    tag_ids : list of int or None
        The tag ids to filter on. Empty/None selects every library media.
    match : str
        ``"any"`` or ``"all"`` (see :func:`_library_media_query`).
    sort : str
        One of the ``SORT_*`` constants above.
    offset : int
        How many media to skip (``(page - 1) * page_size``).
    limit : int
        The page size.
    quality_metric_selected : str, optional
        The metric to rank a quality sort by and to display on each card's
        badge (see :func:`src.sqlite_store.base._sort_order_by` and
        :func:`src.sqlite_store.media._media_dicts`).
    favorites_only : bool, optional
        When true, only media flagged as favorite are returned (the Medias
        tab's "Favorites only" filter).
    exclude_tag_ids : list of int or None, optional
        Tag ids to exclude (see :func:`_library_media_query`).
    in_dataset_id : int, optional
        Only media linked to this dataset (the Datasets tab's own view).
    not_in_dataset_id : int, optional
        Only media *not* linked to this dataset (the "Add images" picker).
    wm_tab : str, optional
        Scope to a Watermark Lab tab's membership (``watermarked`` /
        ``patched``); ``None``/``media`` adds no watermark constraint.
    """
    body, params = _library_media_query(
        tag_ids,
        match,
        favorites_only,
        exclude_tag_ids,
        in_dataset_id=in_dataset_id,
        not_in_dataset_id=not_in_dataset_id,
        wm_tab=wm_tab,
    )
    join, order, sort_params = _sort_order_by(sort, quality_metric_selected)
    rows = _query_all(
        f"SELECT m.* FROM media m {join} {body} ORDER BY {order} "
        "LIMIT ? OFFSET ?",
        sort_params + params + [int(limit), int(offset)],
    )
    return _media_dicts(rows, quality_metric_selected=quality_metric_selected)


def library_media_ids(
    tag_ids=None,
    match: str = "any",
    exclude_tag_ids=None,
    in_dataset_id=None,
    not_in_dataset_id=None,
    favorites_only: bool = False,
    wm_tab=None,
) -> list:
    """Return the ids of the library media matching a tag filter.

    No disk access: backs bulk membership operations (the Datasets tab's
    "Select all"/"Remove all") and the Watermark Lab's "select all filtered"
    batch, which only need identities, not files. ``favorites_only`` mirrors
    the Media grid's favorites filter; ``wm_tab`` scopes the ids to a
    Watermark Lab tab's membership; ``in_dataset_id`` / ``not_in_dataset_id``
    optionally scope to a dataset's membership (see
    :func:`_library_media_query`).
    """
    body, params = _library_media_query(
        tag_ids,
        match,
        favorites_only,
        exclude_tag_ids=exclude_tag_ids,
        in_dataset_id=in_dataset_id,
        not_in_dataset_id=not_in_dataset_id,
        wm_tab=wm_tab,
    )
    return [
        row["id"]
        for row in _query_all(f"SELECT m.id FROM media m {body}", params)
    ]


def library_media_set(
    tag_ids=None,
    match: str = "any",
    favorites_only: bool = False,
    exclude_tag_ids=None,
    not_in_dataset_id=None,
    quality_metric_selected=None,
) -> list:
    """Return every library media matching the full picker filter.

    The whole matching set, not a page: the dataset composer ranks its
    candidates against each other (diversity gain, near-duplicates, the
    coverage map, the gap zones), which no single page can answer. It is
    therefore a batch reader like :func:`library_media_filtered` — a disk
    stat per media — and never a grid hot path; those keep paging with
    :func:`count_library_media` + :func:`library_media_page`.

    Parameters
    ----------
    tag_ids : list of int or None, optional
        The tag ids to filter on; empty/None selects every library media.
    match : str, optional
        ``"any"`` or ``"all"`` (see :func:`_library_media_query`).
    favorites_only : bool, optional
        Keep only the favorites.
    exclude_tag_ids : list of int or None, optional
        Drop media carrying any of these tags.
    not_in_dataset_id : int, optional
        Drop media already linked to this dataset.
    quality_metric_selected : str, optional
        The metric each card's badge shows.

    Returns
    -------
    list of dict
        The matching media as ``_media_dict`` dicts, newest first.
    """
    body, params = _library_media_query(
        tag_ids,
        match,
        favorites_only,
        exclude_tag_ids,
        not_in_dataset_id=not_in_dataset_id,
    )
    rows = _query_all(
        f"SELECT m.* FROM media m {body} ORDER BY m.id DESC", params
    )
    return _media_dicts(rows, quality_metric_selected=quality_metric_selected)


def list_library_media(
    sort: str = SORT_DATE_DESC, quality_metric_selected=None
) -> list:
    """Return every media in the library as dicts.

    Each media is returned as a :func:`_media_dict` (with its effective file
    and missing flag resolved live), so this materializes the whole library —
    a disk stat per media. Grid hot paths page with
    :func:`count_library_media` + :func:`library_media_page` instead.

    Parameters
    ----------
    sort : str, optional
        One of the ``SORT_*`` constants above; defaults to newest first.
    quality_metric_selected : str, optional
        The metric a quality sort ranks by and each card's badge shows.
    """
    return library_media_filtered(None, "any", sort, quality_metric_selected)


def library_media_filtered(
    tag_ids,
    match: str = "any",
    sort: str = SORT_DATE_DESC,
    quality_metric_selected=None,
) -> list:
    """Return library media filtered by tags.

    Materializes the whole matching set (a disk stat per media) — right for
    batch jobs that visit every file (auto-tagging), wrong for grid paints;
    those page with :func:`count_library_media` + :func:`library_media_page`.

    Parameters
    ----------
    tag_ids : list of int
        The tag ids to filter on. An empty list returns every library media.
    match : str, optional
        ``"any"`` keeps media carrying at least one of the tags; ``"all"``
        keeps only media carrying every one of them.
    sort : str, optional
        One of the ``SORT_*`` constants above; defaults to newest first.
    quality_metric_selected : str, optional
        The metric a quality sort ranks by and each card's badge shows.

    Returns
    -------
    list of dict
        The matching library media as :func:`_media_dict` dicts.
    """
    body, params = _library_media_query(tag_ids, match)
    join, order, sort_params = _sort_order_by(sort, quality_metric_selected)
    rows = _query_all(
        f"SELECT m.* FROM media m {join} {body} ORDER BY {order}",
        sort_params + params,
    )
    return _media_dicts(rows, quality_metric_selected=quality_metric_selected)


def create_library(name: str, path, recursive: bool = True) -> int:
    """Create a folder library and return its id.

    Parameters
    ----------
    name : str
        A display name; falls back to the folder's own name when empty.
    path : str or pathlib.Path
        The folder to scan.
    recursive : bool, optional
        Whether the scan descends into sub-folders (default true).

    Raises
    ------
    ValueError
        If a library already exists for this folder.
    """
    path = str(Path(path))
    name = (name or "").strip() or Path(path).name or "library"
    try:
        return _write(
            "INSERT INTO library (name, path, kind, recursive) "
            "VALUES (?, ?, 'folder', ?)",
            (name, path, int(bool(recursive))),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A library already exists for {path!r}.") from exc


def list_libraries() -> list:
    """Return all libraries, the internal one first then folders by name."""
    return _query_all("SELECT * FROM library ORDER BY kind = 'folder', name")


def get_library(library_id: int):
    """Return a library row by id, or None."""
    return _query_one("SELECT * FROM library WHERE id = ?", (library_id,))


def get_internal_library():
    """Return the internal upload library row, or None."""
    return _query_one("SELECT * FROM library WHERE kind = 'internal'")


def set_internal_library_path(path) -> None:
    """Point the internal upload library at a new folder.

    Only the library's folder changes; its existing ``media_file`` rows keep
    their recorded paths, so media already uploaded stay where they are and
    only new uploads land in the new folder.
    """
    _write(
        "UPDATE library SET path = ? WHERE kind = 'internal'",
        (str(Path(path)),),
    )


def delete_library(library_id: int) -> None:
    """Delete a folder library (the internal one cannot be deleted).

    A ``media_file`` is the *physical* trace of a media in a library, so
    removing the library from the index deletes its ``media_file`` rows. The
    ``media`` rows (the *virtual* entries, with their captions) are kept: a
    media that ends up with **no file at all** is purely virtual and gets
    **archived** (``hidden``) so it leaves the library view while keeping its
    captions — re-adding the same content later recreates a file, un-hides the
    media and recovers them. A media still backed by a file (another library or
    an external reference) stays visible. Datasets keep their links throughout.
    """
    library = get_library(library_id)
    if library is None or library["kind"] == "internal":
        return
    affected = [
        row["media_id"]
        for row in _query_all(
            "SELECT DISTINCT media_id FROM media_file WHERE library_id = ?",
            (library_id,),
        )
    ]
    with closing(db.connect()) as conn:
        with conn:
            # Drop this library's physical traces, then the library row.
            conn.execute(
                "DELETE FROM media_file WHERE library_id = ?", (library_id,)
            )
            conn.execute("DELETE FROM library WHERE id = ?", (library_id,))
    for media_id in affected:
        has_file = _query_one(
            "SELECT 1 FROM media_file WHERE media_id = ? LIMIT 1", (media_id,)
        )
        if has_file is None:
            _write("UPDATE media SET hidden = 1 WHERE id = ?", (media_id,))


def _dest_taken_stems(dest_dir: Path) -> set:
    """Return the lowercased file *stems* already used in a destination folder.

    Stems (not full names) are what must stay unique: two media sharing a
    stem with different extensions (``10.jpg`` / ``10.png``) would fight over
    the same ``10.txt`` sidecar caption, so a merge treats them as a
    collision. Combines the stems present on disk with those referenced by
    ``media_file`` rows pointing into ``dest_dir`` (a file may be recorded
    but currently missing on disk), so a merge never overwrites either.
    """
    taken = set()
    if dest_dir.exists():
        for entry in dest_dir.iterdir():
            taken.add(entry.stem.lower())
    for row in _query_all("SELECT path FROM media_file"):
        candidate = Path(row["path"])
        if candidate.parent == dest_dir:
            taken.add(candidate.stem.lower())
    return taken


def _next_free_stem(taken: set, stem: str) -> str:
    """Return ``stem`` (or a ``stem_N`` variant, N from 2) not in ``taken``.

    ``taken`` holds lowercased stems; the caller adds the returned stem to
    it. Numbering starts at ``_2`` — the original holder of the stem is
    implicitly the first copy.
    """
    if stem.lower() not in taken:
        return stem
    index = 2
    while f"{stem}_{index}".lower() in taken:
        index += 1
    return f"{stem}_{index}"


def merge_libraries(source_ids, dest_id: int) -> dict:
    """Merge folder libraries into one, moving their files into its folder.

    Every media reachable through a source library is re-attached to the
    destination: each of its files is physically moved into the destination
    folder and its ``media_file`` row is re-pointed at the destination
    library and its new path. A same-stem ``.txt`` sidecar caption sitting
    next to the file moves (and renames) along with it. Media identity is
    untouched, so captions, tags and dataset links follow automatically. The
    emptied source libraries (every selected one except the destination) are
    then removed.

    File *stems* stay unique in the destination: a stem already taken —
    even by a different extension (``10.jpg`` vs ``10.png``), which would
    otherwise fight over the same ``10.txt`` sidecar — gets a ``_2``/``_3``…
    suffix instead.

    A file already inside the destination folder, or missing on disk, is only
    re-pointed in the database (there is nothing to move). The internal library
    cannot take part: only folder libraries can be merged.

    Parameters
    ----------
    source_ids : iterable of int
        The folder libraries to merge (the destination may be one of them).
    dest_id : int
        The folder library that receives every file.

    Returns
    -------
    dict
        ``{"moved": int, "captions": int, "removed": int, "dest": str}`` —
        files re-attached, sidecar captions moved with them, source
        libraries removed and the destination's display name.

    Raises
    ------
    ValueError
        If the destination or any source is not a folder library.
    """
    dest = get_library(dest_id)
    if dest is None or dest["kind"] != "folder":
        raise ValueError("The destination must be a folder library.")
    move_ids = [sid for sid in source_ids if sid != dest_id]
    for lib_id in move_ids:
        library = get_library(lib_id)
        if library is None or library["kind"] != "folder":
            raise ValueError("Only folder libraries can be merged.")
    dest_dir = Path(dest["path"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    taken = _dest_taken_stems(dest_dir)
    moved = 0
    captions = 0
    with closing(db.connect()) as conn:
        for lib_id in move_ids:
            rows = conn.execute(
                "SELECT id, path FROM media_file WHERE library_id = ?",
                (lib_id,),
            ).fetchall()
            for row in rows:
                source = Path(row["path"])
                new_path = source
                if source.exists() and source.parent != dest_dir:
                    stem = _next_free_stem(taken, source.stem)
                    taken.add(stem.lower())
                    new_path = dest_dir / f"{stem}{source.suffix}"
                    shutil.move(str(source), str(new_path))
                    sidecar = media.caption_path(str(source))
                    if sidecar.exists():
                        shutil.move(
                            str(sidecar), str(dest_dir / f"{stem}.txt")
                        )
                        captions += 1
                with conn:
                    conn.execute(
                        "UPDATE media_file SET library_id = ?, path = ? "
                        "WHERE id = ?",
                        (dest_id, str(new_path), row["id"]),
                    )
                moved += 1
        with conn:
            for lib_id in move_ids:
                conn.execute("DELETE FROM library WHERE id = ?", (lib_id,))
    return {
        "moved": moved,
        "captions": captions,
        "removed": len(move_ids),
        "dest": dest["name"],
    }


def set_library_recursive(library_id: int, recursive: bool) -> None:
    """Set whether a library's scan descends into sub-folders."""
    _write(
        "UPDATE library SET recursive = ? WHERE id = ?",
        (int(bool(recursive)), library_id),
    )


def set_library_name(library_id: int, name: str) -> None:
    """Rename a library. A blank name is ignored (the old one is kept)."""
    clean = (name or "").strip()
    if not clean:
        return
    _write("UPDATE library SET name = ? WHERE id = ?", (clean, library_id))


def set_library_path(library_id: int, path) -> None:
    """Point a folder library at a new source folder.

    Only the library's scanned folder changes; its existing ``media_file``
    rows keep their recorded paths, so media already ingested stay attached
    and only the *next* scan reads from the new folder. The internal upload
    library is repointed with :func:`set_internal_library_path` instead.

    Parameters
    ----------
    library_id : int
        The folder library to repoint.
    path : str or pathlib.Path
        The new folder to scan.

    Raises
    ------
    ValueError
        If the library is not a folder library, or another library already
        uses that folder.
    """
    library = get_library(library_id)
    if library is None or library["kind"] != "folder":
        raise ValueError("Only folder libraries can be repointed.")
    path = str(Path(path))
    try:
        _write("UPDATE library SET path = ? WHERE id = ?", (path, library_id))
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A library already exists for {path!r}.") from exc


def library_paths() -> list:
    """Return every library's folder path."""
    return [row["path"] for row in _query_all("SELECT path FROM library")]


def media_file_dirs() -> list:
    """Return the distinct parent directories of all media files.

    Used to allow Gradio to serve media that live outside the working
    directory (media are referenced in place at their original location).
    """
    dirs = set()
    for row in _query_all("SELECT DISTINCT path FROM media_file"):
        dirs.add(str(Path(row["path"]).parent))
    return sorted(dirs)


def media_in_library(library_id: int) -> list:
    """Return the media that have a file in a library, as dicts (newest first).

    A media belongs to a library when at least one of its files was discovered
    there. Returned as :func:`_media_dict` dicts (effective file + missing flag
    resolved live). Materializes the whole library (a disk stat per media);
    use :func:`count_media_in_library` + :func:`media_in_library_page` to page
    a large library without stat-ing every file.
    """
    rows = _query_all(
        "SELECT DISTINCT m.* FROM media m "
        "JOIN media_file mf ON mf.media_id = m.id "
        "WHERE mf.library_id = ? AND m.deleted_at IS NULL "
        "ORDER BY m.id DESC",
        (library_id,),
    )
    return _media_dicts(rows)


def count_media_in_library(library_id: int) -> int:
    """Return how many media have a file in a library (no disk access).

    A cheap ``COUNT`` used to size the library grid's pager without
    materializing (and stat-ing) every media like :func:`media_in_library`.
    """
    row = _query_one(
        "SELECT COUNT(DISTINCT mf.media_id) AS n FROM media_file mf "
        "JOIN media m ON m.id = mf.media_id "
        "WHERE mf.library_id = ? AND m.deleted_at IS NULL",
        (library_id,),
    )
    return row["n"] if row else 0


def media_in_library_page(
    library_id: int, offset: int, limit: int, quality_metric_selected=None
) -> list:
    """Return one page of a library's media as dicts (newest first).

    Only the returned rows have their effective file resolved (a disk stat per
    media), so paging a large library never stats every file — unlike
    :func:`media_in_library`, which materializes the whole library.

    Parameters
    ----------
    library_id : int
        The library to page.
    offset : int
        How many media to skip (``(page - 1) * page_size``).
    limit : int
        The page size.
    quality_metric_selected : str, optional
        The metric each card's quality badge should reflect (see
        :func:`src.sqlite_store.media._media_dicts`).
    """
    rows = _query_all(
        "SELECT DISTINCT m.* FROM media m "
        "JOIN media_file mf ON mf.media_id = m.id "
        "WHERE mf.library_id = ? AND m.deleted_at IS NULL "
        "ORDER BY m.id DESC LIMIT ? OFFSET ?",
        (library_id, int(limit), int(offset)),
    )
    return _media_dicts(rows, quality_metric_selected=quality_metric_selected)


def _import_sidecar_caption(
    media_id: int, media_path, extension: str, caption_type_id: int
) -> bool:
    """Import a sidecar caption file as the newest revision.

    Reads the caption file next to ``media_path`` (same stem, ``extension``);
    a missing/empty file, or content identical to the caption's current head,
    changes nothing. Otherwise the text becomes a new revision on top of any
    existing history, so a media already captioned keeps its older revisions
    and just gains this one as the most recent.

    Returns
    -------
    bool
        Whether a new revision was added.
    """
    text = media.read_caption(str(media_path), extension)
    if not text:
        return False
    caption_id = get_or_create_caption(media_id, caption_type_id)
    head_id = head_revision_id(caption_id)
    if head_id:
        head = get_revision(head_id)
        if head and head["content"].strip() == text.strip():
            return False
    with closing(db.connect()) as conn:
        with conn:
            revision = _insert_revision(
                conn, caption_id, text, "Imported from library scan", head_id
            )
            conn.execute(
                "UPDATE caption SET head_revision_id = ? WHERE id = ?",
                (revision, caption_id),
            )
    return True


def _apply_folder_tags(media_id: int, tag_names) -> None:
    """Attach a mapped folder's effective tags to a scanned media.

    Each tag name is resolved to a tag (reusing an existing one, else created
    in the "Uncategorized" pen — see :func:`get_or_create_tag_reuse`) and
    linked with ``source='library'``: rule-driven like the "Bulk tags" push,
    so it never downgrades a genuine per-media tag and a re-scan is additive
    (``INSERT OR IGNORE``), never clobbering a tag the user edited by hand.
    """
    for name in tag_names:
        add_tag_to_media(media_id, get_or_create_tag_reuse(name), "library")


def scan_library(
    library_id: int,
    import_captions: bool = False,
    caption_type_id=None,
    progress=None,
) -> dict:
    """Scan a library's folder, ingesting and routing its media files.

    Every media file in the folder (recursively when the library is recursive)
    is ingested. A media counts as **fresh** when it is brand-new content, or
    one that was archived or missing (its library had been deleted, or its
    file had simply gone from disk) and is now re-added.

    When the library carries a **subfolder mapping** (see
    :mod:`src.folder_rules`), the scan resolves each file's folder chain
    against the rules: a file under an ``exclude`` folder is skipped, a file
    under a ``sublib`` folder is attached to that sub-library instead of the
    parent (re-pointing an already-known file — :func:`ingest_file` updates
    its ``library_id``), and the folder's effective tags (auto ∪ manual ∪
    inherited − removed) are applied. This is what makes the mapping
    persistent: files added to a mapped folder later are routed and tagged on
    the next scan. A library with no rules behaves exactly as before — every
    file ingested into it, no tags. A **sub-library** is scanned through its
    parent, so the whole tree is re-resolved in one authoritative pass.

    Parameters
    ----------
    library_id : int
        The library to scan.
    import_captions : bool, optional
        When true, a sidecar caption file next to each scanned media (same
        stem, ``caption_type_id``'s extension) is imported as a caption
        revision (see :func:`_import_sidecar_caption`).
    caption_type_id : int, optional
        The caption type the sidecar files are read as; required when
        ``import_captions`` is set.
    progress : callable, optional
        Called ``progress(done, total)`` after each file is ingested, so a
        job body can stream a determinate progress bar. The file list is
        materialised up front (the discovery walk) to know ``total``.

    Returns
    -------
    dict
        ``{"new_media", "existing", "captions_imported", "skipped"}`` counts,
        where ``new_media`` is the number of fresh (new or recovered) media
        and ``skipped`` the files an ``exclude`` rule kept out.
    """
    library = get_library(library_id)
    summary = {
        "new_media": 0,
        "existing": 0,
        "captions_imported": 0,
        "skipped": 0,
    }
    if library is None:
        return summary
    # A sub-library re-resolves through its parent's full tree walk.
    if library["parent_library_id"] is not None:
        return scan_library(
            library["parent_library_id"],
            import_captions,
            caption_type_id,
            progress,
        )
    folder = Path(library["path"])
    if not folder.is_dir():
        return summary
    caption_type = (
        get_caption_type(caption_type_id) if import_captions else None
    )
    auto_level, rules = folder_rules_map(library_id)
    files = list(
        media.get_media_files(folder, recursive=bool(library["recursive"]))
    )
    total = len(files)
    if progress is not None:
        progress(0, total)
    for done, path in enumerate(files, start=1):
        resolved = fr.resolve_file(
            fr.rel_folder(path, folder), rules, auto_level
        )
        if resolved is None:
            summary["skipped"] += 1
            if progress is not None:
                progress(done, total)
            continue
        owner_id, tag_names = resolved
        media_id, is_fresh = ingest_file(path, owner_id or library_id)
        summary["new_media" if is_fresh else "existing"] += 1
        _apply_folder_tags(media_id, tag_names)
        if caption_type is not None and _import_sidecar_caption(
            media_id, path, caption_type["file_extension"], caption_type["id"]
        ):
            summary["captions_imported"] += 1
        if progress is not None:
            progress(done, total)
    return summary
