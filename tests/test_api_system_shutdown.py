"""Test the safe-shutdown route without ever letting it exit the process.

``POST /api/system/shutdown`` frees models, checkpoints the WAL and schedules
``os._exit(0)`` on a timer. The timer, the model teardown and the checkpoint
are all patched out so the assertion runs in-process: the route wired the
teardown in the right order and returned ``{"ok": True}``.
"""

import pytest
from fastapi.testclient import TestClient

from server.routers import system
from src import db


@pytest.fixture(name="client")
def _client(store_db):
    """Return a TestClient over the app (throwaway sandboxed database)."""
    # pylint: disable=unused-argument
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client


def test_shutdown_tears_down_then_schedules_exit(client, monkeypatch):
    """The route frees models, flushes the WAL and arms the exit timer."""
    calls: dict = {}

    class FakeTimer:  # pylint: disable=too-few-public-methods
        """Record the scheduled exit instead of ever firing it."""

        def __init__(self, delay, func):
            calls["timer_delay"] = delay
            calls["timer_func"] = func

        def start(self):
            calls["timer_started"] = True

    monkeypatch.setattr(
        system, "_free_models", lambda: calls.setdefault("freed", True)
    )
    monkeypatch.setattr(
        db, "checkpoint_wal", lambda: calls.setdefault("wal", True)
    )
    monkeypatch.setattr(system.threading, "Timer", FakeTimer)

    response = client.post("/api/system/shutdown")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls["freed"] is True
    assert calls["wal"] is True
    assert calls["timer_started"] is True
    assert calls["timer_delay"] == 0.4
