"""Optional runtime redirects for isolated dev / verification runs.

Production reads the database, deploy folder and the two pixel caches from
their default locations. For an isolated end-to-end run (never the user's
real data) four environment variables repoint them *before* any engine
touches the disk:

* ``CF_DB_PATH`` — SQLite database file (patches
  :func:`src.db.get_db_path`);
* ``CF_DEPLOY_DIR`` — deploy root (patches
  :func:`src.settings.get_deploy_dir` as seen by :mod:`src.deploy`);
* ``CF_THUMBNAILS_DIR`` — thumbnail cache (patches
  :func:`src.thumbnails.get_thumbnails_dir`);
* ``CF_CROPS_DIR`` — rendered-crop cache (patches
  :func:`src.crops.get_crops_dir`).

:func:`apply_redirects` is a no-op when none are set, so importing it in
production is harmless.
"""

import os
from pathlib import Path


def apply_redirects() -> None:
    """Repoint DB / deploy / cache paths from the environment.

    Called once at import time in :mod:`server.main`, before the routers
    issue any query. Each redirect is applied only when its variable is
    present, leaving the shipped defaults intact otherwise. The engine
    imports are deferred so this stays cheap when no redirect is set.
    """
    # pylint: disable=import-outside-toplevel
    db_path = os.environ.get("CF_DB_PATH")
    if db_path:
        from src import db

        db.get_db_path = lambda: Path(db_path)

    deploy_dir = os.environ.get("CF_DEPLOY_DIR")
    if deploy_dir:
        from src import deploy

        deploy.get_deploy_dir = lambda: Path(deploy_dir)

    thumbs_dir = os.environ.get("CF_THUMBNAILS_DIR")
    if thumbs_dir:
        from src import thumbnails

        thumbnails.get_thumbnails_dir = lambda: Path(thumbs_dir)

    crops_dir = os.environ.get("CF_CROPS_DIR")
    if crops_dir:
        from src import crops

        crops.get_crops_dir = lambda: Path(crops_dir)
