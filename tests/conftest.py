"""Shared fixtures for the Caption Forge test suite."""

import sqlite3
import uuid

import pytest

from src import config
from src import crops
from src import db
from src import settings
from src import thumbnails


@pytest.fixture(autouse=True)
def _isolated_user_config(monkeypatch, tmp_path):
    """Redirect the user config layer to a throwaway per-test folder.

    The suite must never read the developer's real ``config/user/``
    overrides (a test asserting a factory default would fail whenever the
    live app has another value selected — e.g. the quality metric) nor
    write into them. The shipped ``config/default/`` layer stays visible,
    matching a fresh install. The settings cache is dropped on both sides
    so no test sees values merged from another layer.
    """
    user_dir = tmp_path / "user_config"
    monkeypatch.setattr(config, "USER_CONFIG_DIR", user_dir)
    monkeypatch.setattr(config, "USER_MODELS_DIR", user_dir / "models")
    settings._invalidate_cache()  # pylint: disable=protected-access
    yield
    settings._invalidate_cache()  # pylint: disable=protected-access


@pytest.fixture(name="store_db")
def _store_db(monkeypatch):
    """Point :mod:`src.db` at a fresh in-memory database with the schema.

    A shared-cache ``file:...?mode=memory`` URI keeps one database visible
    across the per-call connections the app opens, with zero disk I/O; the
    keeper connection holds the database alive for the test's duration. The
    name is unique per test so parallel/successive tests never share state.
    """
    uri = f"file:cf_test_{uuid.uuid4().hex}?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    monkeypatch.setattr(db, "get_db_path", lambda: uri)
    db.ensure_database()
    yield uri
    keeper.close()


@pytest.fixture(name="thumb_cache_dir")
def _thumb_cache_dir(monkeypatch, tmp_path):
    """Redirect the thumbnail cache to a throwaway per-test folder.

    Keeps generated thumbnails out of the project's own ``cache/`` folder,
    the same isolation :func:`src.deploy.deploy_root` gets from tests that
    patch ``get_deploy_dir``.
    """
    cache_dir = tmp_path / "thumb_cache"
    monkeypatch.setattr(thumbnails, "get_thumbnails_dir", lambda: cache_dir)
    return cache_dir


@pytest.fixture(name="crop_cache_dir", autouse=True)
def _crop_cache_dir(monkeypatch, tmp_path):
    """Redirect the rendered-crop cache to a throwaway per-test folder.

    Autouse: resolving a crop's effective file renders its PNG, so any test
    that merely *lists* a dataset holding a crop would otherwise write into
    the project's own ``cache/crops/``.
    """
    cache_dir = tmp_path / "crop_cache"
    monkeypatch.setattr(crops, "get_crops_dir", lambda: cache_dir)
    return cache_dir
