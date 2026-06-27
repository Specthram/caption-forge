"""Media file helpers: extension sets, discovery and sidecar caption reads.

Used by the scanner (folder ingestion), the sidecar caption import and the
video/image dispatch throughout the UI.
"""

import os
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def get_media_files(input_dir: Path, recursive: bool = False) -> list[str]:
    """Return the sorted media file paths found in ``input_dir``.

    ``recursive`` descends into sub-folders; false (default) lists only the
    directory's own files.
    """
    input_dir = Path(input_dir)
    if recursive:
        files = [
            str(path)
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
        ]
    else:
        files = [
            str(input_dir / f)
            for f in os.listdir(input_dir)
            if Path(f).suffix.lower() in MEDIA_EXTENSIONS
        ]
    return sorted(files)


def is_video_file(path: str) -> bool:
    """Return whether ``path`` has a recognized video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def caption_path(media_path: str, ext: str = "txt") -> Path:
    """Return the caption file path for a media and a given extension.

    Uses ``with_name`` (not ``with_suffix``) so media names containing dots
    keep their full stem, e.g. ``a.b.png`` -> ``a.b.txt`` rather than
    ``a.txt``.
    """
    path = Path(media_path)
    ext = str(ext).lstrip(".")
    return path.with_name(f"{path.stem}.{ext}")


def read_caption(media_path: str, ext: str = "txt") -> str:
    """Return the caption text for a media, or "" if no caption file exists.

    Used by the Libraries tab's sidecar caption import (see
    :func:`src.sqlite_store.scan_library`).
    """
    cap = caption_path(media_path, ext)
    if cap.exists():
        return cap.read_text(encoding="utf-8").strip()
    return ""
