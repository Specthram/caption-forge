"""Pure logic for the Libraries "Subfolder mapping" feature.

A folder library can map each of its sub-folders to one of three modes —
``keep`` (media stays in the parent), ``sublib`` (the folder becomes its own
library, excluded from the parent) or ``exclude`` (never scanned) — and carry
tag rules: auto tags derived from folder names, tags inherited down the tree,
and manual tags, with per-folder overrides. This module is the single source
of truth for resolving those rules; it holds no database access so both the
scan (which applies the rules to real files) and the wizard preview can reuse
it.

A *rule* is a plain dict ``{"mode", "tags", "removed", "sub_library_id"}``
keyed by a folder's ``rel_path`` (its path relative to the library root, with
``/`` separators; ``""`` is the root itself). ``tags`` and ``removed`` are
lists of tag *names*: the folder's manually added tags, and the inherited/auto
tags overridden off for it. A missing rule means an untouched ``keep`` folder.
"""

import re
from pathlib import Path

MODE_KEEP = "keep"
MODE_SUBLIB = "sublib"
MODE_EXCLUDE = "exclude"

AUTO_OFF = "0"
AUTO_TOP = "1"
AUTO_ALL = "all"


def slugify(name: str) -> str:
    """Return a tag-safe slug of a folder name (whitespace collapsed to ``_``).

    Mirrors the wizard prototype: the name is trimmed and every run of
    whitespace becomes a single underscore. Case is preserved (tags are
    matched verbatim elsewhere in the app).
    """
    return re.sub(r"\s+", "_", str(name).strip())


def _chain(rel_path: str) -> list:
    """Return a folder's ancestor rel_paths, root first, itself last.

    ``"a/b/c"`` yields ``["a", "a/b", "a/b/c"]``; the root ``""`` yields the
    empty list (it carries no rule of its own).
    """
    parts = [part for part in rel_path.split("/") if part]
    return ["/".join(parts[: index + 1]) for index in range(len(parts))]


def rel_folder(file_path, root) -> str:
    """Return the ``/``-joined folder rel_path of a file under ``root``.

    ``""`` when the file sits directly in the library root. A file outside
    ``root`` (which a normal scan never yields) also returns ``""``.
    """
    try:
        relative = Path(file_path).parent.relative_to(Path(root))
    except ValueError:
        return ""
    return "/".join(part for part in relative.parts if part not in (".", ""))


def is_excluded(rel_path: str, rules: dict) -> bool:
    """Return whether a folder or any of its ancestors is an ``exclude``."""
    for ancestor in _chain(rel_path):
        rule = rules.get(ancestor)
        if rule and rule.get("mode") == MODE_EXCLUDE:
            return True
    return False


def owning_sub_library(rel_path: str, rules: dict):
    """Return the id of the sub-library that owns a folder, or ``None``.

    The owner is the deepest ``sublib`` ancestor (the folder itself counts),
    so nested sub-libraries route a file to the closest one; ``None`` means
    the file stays in the parent library. A ``sublib`` rule with no
    ``sub_library_id`` yet (not applied) is ignored.
    """
    owner = None
    for ancestor in _chain(rel_path):
        rule = rules.get(ancestor)
        if rule and rule.get("mode") == MODE_SUBLIB:
            sub_id = rule.get("sub_library_id")
            if sub_id is not None:
                owner = sub_id
    return owner


def _auto_covers(depth: int, auto_tag_level: str) -> bool:
    """Return whether the auto-tag rule tags a folder at ``depth``.

    ``depth`` is 1-based (the top-level folder is depth 1).
    """
    if auto_tag_level == AUTO_ALL:
        return True
    return auto_tag_level == AUTO_TOP and depth == 1


def effective_tags(rel_path: str, rules: dict, auto_tag_level: str) -> list:
    """Return the effective tag names of a folder, inherited chain included.

    Walks the ancestor chain root-down, accumulating each level's tags — the
    auto tag (slugified name when ``auto_tag_level`` covers that depth), manual
    tags, and everything inherited — minus each level's ``removed`` list.
    Returns the ordered, de-duplicated names (inherited first), or empty when
    the folder or an ancestor is excluded. ``rules`` is ``rel_path -> rule``
    (see module docstring); ``""`` is the root, seeding every sub-folder.
    """
    parts = [part for part in rel_path.split("/") if part]
    # The root rule (rel_path "") tags every file and is inherited by all
    # sub-folders; a child can drop one via its own ``removed`` list.
    root_tags = list((rules.get("") or {}).get("tags") or [])
    inherited: list = list(root_tags)
    current: list = list(root_tags)
    for depth, ancestor in enumerate(_chain(rel_path), start=1):
        rule = rules.get(ancestor) or {}
        if rule.get("mode") == MODE_EXCLUDE:
            return []
        removed = set(rule.get("removed") or [])
        slug = slugify(parts[depth - 1])
        auto = (
            [slug]
            if _auto_covers(depth, auto_tag_level) and slug not in removed
            else []
        )
        manual = [
            tag
            for tag in (rule.get("tags") or [])
            if tag not in removed and tag not in auto
        ]
        inherited_here = [
            tag
            for tag in inherited
            if tag not in removed and tag not in auto and tag not in manual
        ]
        current = inherited_here + auto + manual
        inherited = current
    return current


def resolve_file(rel_path: str, rules: dict, auto_tag_level: str):
    """Return a file's ``(owning_sub_library_id, tag_names)``, or ``None``.

    The one call the scanner makes per file: ``None`` when the file's folder
    is excluded (the file is skipped), otherwise the id of the sub-library
    that owns it (``None`` = the parent library) and the effective tag names
    to attach.
    """
    if is_excluded(rel_path, rules):
        return None
    return (
        owning_sub_library(rel_path, rules),
        effective_tags(rel_path, rules, auto_tag_level),
    )
