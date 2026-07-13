"""Dataset export: bundle the captioned media folder into a zip archive."""

import tempfile
import zipfile
from pathlib import Path

# Archive name; kept stable so the browser download is named predictably. It
# lives in the system temp dir (not the project root) to avoid polluting the
# working directory, and is overwritten on each export.
_ARCHIVE_NAME = "collection.zip"


def create_zip_archive(input_folder: str | Path) -> str:
    """Create a flat zip archive of every file in ``input_folder``.

    Non-recursive. Returns the path to the ``collection.zip`` in the temp dir.
    """
    zip_path = Path(tempfile.gettempdir()) / _ARCHIVE_NAME
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in Path(input_folder).iterdir():
            if path.is_file():
                archive.write(path, arcname=path.name)
    return str(zip_path)
