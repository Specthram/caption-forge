"""Server-side folder browser for the Libraries "Add folder" picker.

The app runs on the machine that holds the media, so picking a library
folder is a *server* filesystem walk, not a browser upload: a browser file
input never yields an absolute directory path. This lists the sub-folders of
a directory (or the drive roots when none is given) so the front-end can
navigate to a folder and hand its absolute path to the scan.

Directories only — files never matter to a library folder choice — and every
per-entry error (a permission-denied system folder, a vanished mount) is
swallowed so one unreadable child never breaks the listing.
"""

import os
from pathlib import Path

from src.media import MEDIA_EXTENSIONS


def list_drives() -> list[str]:
    r"""Return existing drive roots (Windows ``C:\`` …); empty elsewhere."""
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives


def _entries(path: Path) -> list[dict]:
    """Return a directory's sub-folders as ``{name, path}``, sorted, safe."""
    found = []
    try:
        children = list(os.scandir(path))
    except OSError:
        return []
    for child in sorted(children, key=lambda entry: entry.name.lower()):
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir(follow_symlinks=False):
                found.append(
                    {"name": child.name, "path": str(Path(child.path))}
                )
        except OSError:
            continue
    return found


def _resolve_dir(path: str):
    """Return the nearest existing directory for ``path``, or None.

    Climbs to the closest existing ancestor (a moved/deleted folder still opens
    the picker somewhere real); None means not even the root exists.
    """
    resolved = Path(path)
    while not resolved.is_dir() and resolved.parent != resolved:
        resolved = resolved.parent
    return resolved if resolved.is_dir() else None


def browse(path: str = "") -> dict:
    """Return one folder listing for the picker.

    ``path`` empty (default) lists the drive roots (Windows) or ``/``. A
    non-existent ``path`` (a moved/deleted library — what the repoint picker
    seeds) isn't an error: the walk climbs to the nearest existing ancestor,
    falling back to the drive roots, so the picker always opens on a real
    folder. Returns ``{"path", "parent", "is_root", "entries"}`` — ``path`` the
    resolved dir (``""`` at drive-root level), ``parent`` where "up" goes
    (``""`` = drive roots, ``None`` = already top), ``is_root``, ``entries``.
    """
    if not path:
        drives = list_drives()
        if drives:
            return {
                "path": "",
                "parent": None,
                "is_root": True,
                "entries": [{"name": d, "path": d} for d in drives],
            }
        path = "/"
    resolved = _resolve_dir(path)
    if resolved is None:
        # No existing ancestor at all — send the picker to the drive roots.
        return browse("")
    parent = resolved.parent
    # A drive root (or "/") is its own parent — send "up" to the drive list.
    parent_path = "" if parent == resolved else str(parent)
    return {
        "path": str(resolved),
        "parent": parent_path,
        "is_root": False,
        "entries": _entries(resolved),
    }


def _file_entries(path: Path, exts: tuple) -> list[dict]:
    """Return a directory's sub-folders and matching files, kind-tagged.

    Sub-folders (``kind": "dir"``) come first, then files whose suffix is in
    ``exts`` (empty ``exts`` = every file), each ``kind": "file"`` with its
    byte ``size``; both sorted case-insensitively. Dot-entries and per-entry
    OS errors are skipped.
    """
    dirs: list = []
    files: list = []
    try:
        children = list(os.scandir(path))
    except OSError:
        return []
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir(follow_symlinks=False):
                dirs.append(
                    {
                        "name": child.name,
                        "path": str(Path(child.path)),
                        "kind": "dir",
                    }
                )
            elif not exts or Path(child.name).suffix.lower() in exts:
                files.append(
                    {
                        "name": child.name,
                        "path": str(Path(child.path)),
                        "kind": "file",
                        "size": child.stat(follow_symlinks=False).st_size,
                    }
                )
        except OSError:
            continue
    dirs.sort(key=lambda entry: entry["name"].lower())
    files.sort(key=lambda entry: entry["name"].lower())
    return dirs + files


def browse_files(path: str = "", exts=()) -> dict:
    """Return one folder listing that also includes files, for a file picker.

    Same navigation contract as :func:`browse` (drive roots at ``""``, climbs
    to a real ancestor), but each entry carries a ``kind`` (``dir``/``file``)
    and files matching ``exts`` (suffixes like ``.safetensors``) are listed so
    the picker can select one. ``exts`` empty lists every file.
    """
    wanted = tuple(f".{e.lower().lstrip('.')}" for e in exts if e)
    if not path:
        drives = list_drives()
        if drives:
            return {
                "path": "",
                "parent": None,
                "is_root": True,
                "entries": [
                    {"name": d, "path": d, "kind": "dir"} for d in drives
                ],
            }
        path = "/"
    resolved = _resolve_dir(path)
    if resolved is None:
        return browse_files("", exts)
    parent = resolved.parent
    parent_path = "" if parent == resolved else str(parent)
    return {
        "path": str(resolved),
        "parent": parent_path,
        "is_root": False,
        "entries": _file_entries(resolved, wanted),
    }


def _is_media(name: str) -> bool:
    """Return whether a file name has a recognized media extension."""
    return Path(name).suffix.lower() in MEDIA_EXTENSIONS


def _scan_folder(abs_path: str, rel_path: str) -> dict:
    """Recursively describe one folder for the mapping wizard.

    Walks ``abs_path`` once, counting media directly in it (``own``) and
    recursively beneath (``total``), keeping up to three sample paths per
    folder. Dot-folders and per-entry OS errors are skipped. ``rel_path`` is
    the ``/``-separated path from the scanned root (``""`` = root) — the folder
    rule key. Returns ``{"rel_path", "name", "own", "total", "samples",
    "children"}``.
    """
    own = 0
    samples: list = []
    subdirs = []
    try:
        entries = list(os.scandir(abs_path))
    except OSError:
        entries = []
    for entry in sorted(entries, key=lambda item: item.name.lower()):
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            subdirs.append(entry)
        elif _is_media(entry.name):
            own += 1
            if len(samples) < 3:
                samples.append(str(Path(entry.path)))
    children = []
    total = own
    for sub in subdirs:
        child_rel = f"{rel_path}/{sub.name}" if rel_path else sub.name
        node = _scan_folder(str(Path(sub.path)), child_rel)
        children.append(node)
        total += node["total"]
    return {
        "rel_path": rel_path,
        "name": Path(abs_path).name,
        "own": own,
        "total": total,
        "samples": samples,
        "children": children,
    }


def folder_tree(path: str) -> dict:
    """Return a folder's full sub-folder tree with recursive media counts.

    Backs the "Subfolder mapping" wizard: one walk of ``path`` yielding the
    nested tree (each node ``own``/``total``/samples). A non-existent path
    returns an empty tree. Returns ``{"path", "name", "own", "total",
    "children"}`` — ``children`` the top-level :func:`_scan_folder` nodes.
    """
    root = Path(path)
    if not root.is_dir():
        return {
            "path": str(root),
            "name": root.name,
            "own": 0,
            "total": 0,
            "children": [],
        }
    node = _scan_folder(str(root), "")
    return {
        "path": str(root),
        "name": node["name"],
        "own": node["own"],
        "total": node["total"],
        "samples": node["samples"],
        "children": node["children"],
    }
