"""Native OS folder dialogs and folder opening for the local app.

Caption Forge runs on the user's own machine (the Gradio server and the
browser are the same computer), so a server-side native folder picker and an
"open in the file manager" action are legitimate here — they act on the user's
own filesystem, not a remote one.

Both helpers degrade quietly: on a headless server (no display, no tkinter)
the picker returns an empty string and the opener is a no-op, so the textbox
the user can still type into by hand is never blocked.
"""

import os
import subprocess
import sys
from pathlib import Path


def resolve_within(root: Path, candidate: Path) -> Path | None:
    """Return ``candidate`` if it resolves inside ``root``, else None.

    Guards path traversal: a candidate that escapes ``root`` (via ``..``) or
    fails to resolve yields None. Existence is not checked.
    """
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    if resolved != root_resolved and root_resolved not in resolved.parents:
        return None
    return candidate


def pick_folder(initial: str | None = None) -> str:
    """Open a native folder-picker dialog and return the chosen path.

    ``initial`` is where the dialog opens (ignored when it doesn't exist).
    Returns the selected absolute path, or "" when cancelled or no graphical
    dialog is available (headless server).
    """
    try:
        # Imported lazily so a headless server without tkinter still imports
        # this module (the picker simply becomes unavailable).
        import tkinter as tk  # pylint: disable=import-outside-toplevel
        from tkinter import (  # pylint: disable=import-outside-toplevel
            filedialog,
        )
    except ImportError:
        return ""

    start = initial if initial and Path(initial).is_dir() else None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=start)
        root.destroy()
        return chosen or ""
    except Exception:  # pylint: disable=broad-exception-caught
        # Any Tk failure (no display, threading quirk) leaves the field as-is.
        return ""


def open_folder(path: str) -> None:
    """Open ``path`` in the OS file manager (no-op when it is missing)."""
    if not path:
        return
    target = Path(path)
    if not target.exists():
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target)], check=False)
        else:
            subprocess.run(["xdg-open", str(target)], check=False)
    except OSError:
        pass


def reveal_file(path: str) -> bool:
    """Open ``path``'s containing folder in the OS file manager, selected.

    Unlike :func:`open_folder` (which opens a directory), this targets a single
    file and asks the file manager to select it, not launch it. Returns whether
    the file was found and a reveal attempted; false for a missing path.
    """
    if not path:
        return False
    target = Path(path)
    if not target.exists():
        return False
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", f"/select,{target}"], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(target)], check=False)
        else:
            # No universal "select in file manager" on Linux; open the
            # containing folder instead.
            subprocess.run(["xdg-open", str(target.parent)], check=False)
    except OSError:
        pass
    return True
