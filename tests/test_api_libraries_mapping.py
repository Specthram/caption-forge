"""API tests for the Libraries subfolder-mapping routes."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(name="client")
def _client(store_db, thumb_cache_dir):
    # pylint: disable=unused-argument
    """Return a TestClient bound to a fresh in-memory database."""
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client


def _seed_tree(root):
    """Seed a small nested folder of media files under ``root``."""
    (root / "a.png").write_bytes(b"x")
    robe = root / "robe"
    robe.mkdir()
    (robe / "b.jpg").write_bytes(b"x")
    (robe / "ete").mkdir()
    (robe / "ete" / "c.webp").write_bytes(b"x")


def test_folder_tree_route(client, tmp_path):
    """GET /folder-tree returns the nested tree with recursive counts."""
    _seed_tree(tmp_path)
    body = client.get(
        "/api/libraries/folder-tree", params={"path": str(tmp_path)}
    ).json()
    assert body["own"] == 1
    assert body["total"] == 3
    robe = body["children"][0]
    assert robe["rel_path"] == "robe"
    assert robe["total"] == 2


def test_get_folder_rules_defaults(client, tmp_path):
    """An unmapped library reports the default level and no rules."""
    (tmp_path / "robe").mkdir()
    library_id = client.post(
        "/api/libraries", json={"name": "vet", "path": str(tmp_path)}
    ).json()["id"]
    body = client.get(f"/api/libraries/{library_id}/folder-rules").json()
    assert body["auto_tag_level"] == "0"
    assert body["rules"] == []


def test_put_folder_rules_creates_sublibraries(client, tmp_path):
    """PUT persists the mapping, promotes a folder and queues a scan."""
    (tmp_path / "robe").mkdir()
    library_id = client.post(
        "/api/libraries", json={"name": "vet", "path": str(tmp_path)}
    ).json()["id"]
    resp = client.put(
        f"/api/libraries/{library_id}/folder-rules",
        json={
            "auto_tag_level": "all",
            "rules": [
                {"rel_path": "robe", "mode": "sublib", "sub_name": "Robe"}
            ],
        },
    ).json()
    assert "job_id" in resp
    assert resp["sub_libraries"] == 1
    names = [
        row["name"] for row in client.get("/api/libraries").json()["libraries"]
    ]
    assert "Robe" in names


def test_put_folder_rules_renames_parent(client, tmp_path):
    """A ``name`` in the payload renames the parent library on apply."""
    (tmp_path / "robe").mkdir()
    library_id = client.post(
        "/api/libraries", json={"name": "vet", "path": str(tmp_path)}
    ).json()["id"]
    client.put(
        f"/api/libraries/{library_id}/folder-rules",
        json={"auto_tag_level": "0", "rules": [], "name": "Clothing 2024"},
    )
    libs = client.get("/api/libraries").json()["libraries"]
    renamed = next(lib for lib in libs if lib["id"] == library_id)
    assert renamed["name"] == "Clothing 2024"
