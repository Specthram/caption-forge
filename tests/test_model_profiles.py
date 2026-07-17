"""Model profiles: store CRUD, detection, mmproj pairing and the API.

Everything runs against the sandboxed user config layer (see
``tests/conftest.py``) — never the real ``config/user/``.
"""

import pytest
from fastapi.testclient import TestClient

from src import model_profiles
from src.model_registry import hf_config_for


@pytest.fixture(autouse=True)
def _fresh_loaded_marker():
    """Reset the process-level loaded-profile marker around each test."""
    model_profiles.set_loaded_id(None)
    yield
    model_profiles.set_loaded_id(None)


@pytest.fixture(name="client")
def _client(store_db):
    """Return a TestClient (profiles routes need no seeded media)."""
    # pylint: disable=unused-argument,import-outside-toplevel
    from server.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestDetection:
    """Filename type/format detection and the text fallback."""

    def test_vision_rule_wins(self):
        """A known vision filename maps to its family."""
        assert model_profiles.detect_type("Qwen3-VL-8B-Q8_0.gguf") == "qwen3"

    def test_unmatched_gguf_falls_back_to_text(self):
        """A GGUF no rule matches is loadable as text-only."""
        assert model_profiles.detect_type("random-llm-7b.gguf") == "text"

    def test_unmatched_safetensors_is_unrecognized(self):
        """A safetensors no rule matches has no family."""
        assert model_profiles.detect_type("mystery.safetensors") == ""

    def test_format_detection(self):
        """Format follows the file extension; unknown suffix is None."""
        assert model_profiles.detect_format("a.gguf") == "gguf"
        assert model_profiles.detect_format("a.safetensors") == ("safetensors")
        assert model_profiles.detect_format("a.bin") is None

    def test_hf_config_for_sized_family(self):
        """A manually forced family resolves its repo, size-aware."""
        assert "8B" in hf_config_for("qwen3", "anything-8b.gguf")
        assert hf_config_for("nope", "x.gguf") is None


class TestMmproj:
    """Auto-detection of the vision projector next to the weights."""

    def test_finds_matching_projector(self, tmp_path):
        """A same-family, same-size mmproj in the folder is paired."""
        (tmp_path / "gemma-3-12b-it-Q8_0.gguf").touch()
        (tmp_path / "mmproj-gemma-3-12b-it-f16.gguf").touch()
        found = model_profiles.auto_mmproj(
            str(tmp_path), "gemma-3-12b-it-Q8_0.gguf", "gemma3"
        )
        assert found == "mmproj-gemma-3-12b-it-f16.gguf"

    def test_text_family_never_pairs(self, tmp_path):
        """A text-only model gets no projector."""
        (tmp_path / "mmproj-gemma-3-12b-it-f16.gguf").touch()
        assert (
            model_profiles.auto_mmproj(str(tmp_path), "llm.gguf", "text")
            is None
        )

    def test_missing_folder_is_none(self):
        """A profile whose folder vanished detects nothing."""
        assert (
            model_profiles.auto_mmproj("Z:/nope", "a.gguf", "gemma3") is None
        )


class TestStore:
    """CRUD, seeding, clamps and the selection fallbacks."""

    def test_first_read_seeds_default(self):
        """An empty store seeds one profile named Default, selected twice."""
        data = model_profiles.list_profiles()
        assert [p["name"] for p in data["profiles"]] == ["Default"]
        first = data["profiles"][0]["id"]
        assert data["active_id"] == first
        assert data["judge_id"] == first
        assert data["loaded_id"] is None

    def test_create_clamps_and_autonames(self, tmp_path):
        """Out-of-range numbers clamp; the name comes from the filename."""
        profile = model_profiles.create_profile(
            {
                "file": "Qwen3-VL-8B-Q8_0.gguf",
                "dir": str(tmp_path),
                "temp": 9.0,
                "img_res": 1000,
                "max_tok": -5,
                "n_ctx": 4096,
                "think": "banana",
            }
        )
        assert profile["name"] == "Qwen3-VL-8B-Q8_0"
        assert profile["temp"] == 2.0
        assert profile["img_res"] == 1024  # snapped to the 128 step
        assert profile["max_tok"] == 16
        assert profile["type"] == "qwen3"  # auto-detected
        assert profile["format"] == "gguf"

    def test_manual_type_survives_update(self, tmp_path):
        """typeMode manual keeps the forced family; auto re-derives."""
        profile = model_profiles.create_profile(
            {"file": "weird-name.gguf", "dir": str(tmp_path)}
        )
        assert profile["type"] == "text"
        forced = model_profiles.update_profile(
            profile["id"], {"type": "llava", "type_mode": "manual"}
        )
        assert forced["type"] == "llava"
        back = model_profiles.update_profile(
            profile["id"], {"type_mode": "auto"}
        )
        assert back["type"] == "text"

    def test_safetensors_clears_mmproj(self, tmp_path):
        """A non-GGUF profile never keeps a projector."""
        (tmp_path / "mmproj-gemma-3-12b-it-f16.gguf").touch()
        profile = model_profiles.create_profile(
            {
                "file": "gemma-3-12b-it.safetensors",
                "dir": str(tmp_path),
                "mmproj": "mmproj-gemma-3-12b-it-f16.gguf",
                "mmproj_mode": "manual",
            }
        )
        assert profile["mmproj"] is None
        assert profile["mmproj_mode"] == "auto"

    def test_role_selection_on_create(self):
        """Creating from the judge picker selects the new profile there."""
        profile = model_profiles.create_profile({"name": "J"}, role="judge")
        data = model_profiles.list_profiles()
        assert data["judge_id"] == profile["id"]
        assert data["active_id"] != profile["id"]

    def test_delete_falls_back_and_protects_last(self):
        """Deleting the selected profile falls back; the last is refused."""
        data = model_profiles.list_profiles()
        first = data["profiles"][0]["id"]
        second = model_profiles.create_profile({"name": "B"})["id"]
        model_profiles.select_profile("caption", second)
        assert model_profiles.delete_profile(second) is True
        data = model_profiles.list_profiles()
        assert data["active_id"] == first
        assert model_profiles.delete_profile(first) is False

    def test_select_unknown_is_refused(self):
        """Selecting a nonexistent profile changes nothing."""
        assert model_profiles.select_profile("caption", 999) is False


class TestLoadCfg:
    """Loader config assembly from a profile."""

    def test_no_file_is_none(self):
        """A profile without weights cannot be loaded."""
        profile = model_profiles.create_profile({"name": "empty"})
        assert model_profiles.load_cfg(profile) is None

    def test_paths_and_nctx(self, tmp_path):
        """Weights/mmproj paths join the profile dir; n_ctx rides along."""
        (tmp_path / "mmproj-gemma-3-12b-it-f16.gguf").touch()
        profile = model_profiles.create_profile(
            {
                "file": "gemma-3-12b-it-Q8_0.gguf",
                "dir": str(tmp_path),
                "n_ctx": 8192,
            }
        )
        cfg = model_profiles.load_cfg(profile)
        assert cfg["local_path"].name == "gemma-3-12b-it-Q8_0.gguf"
        assert cfg["mmproj_path"].name == "mmproj-gemma-3-12b-it-f16.gguf"
        assert cfg["n_ctx"] == 8192
        assert cfg["type"] == "gemma3"
        assert cfg["hf_config"]

    def test_manual_family_gets_a_repo(self, tmp_path):
        """A forced family on an unmatched filename still resolves a repo."""
        profile = model_profiles.create_profile(
            {
                "file": "weird.gguf",
                "dir": str(tmp_path),
                "type": "llava",
                "type_mode": "manual",
            }
        )
        cfg = model_profiles.load_cfg(profile)
        assert cfg["hf_config"] == (
            "fancyfeast/llama-joycaption-beta-one-hf-llava"
        )


class TestProfilesApi:
    """The /api/profiles routes end to end."""

    def test_list_seeds_and_reports_families(self, client):
        """GET returns the seeded Default and the family table."""
        body = client.get("/api/profiles").json()
        assert [p["name"] for p in body["profiles"]] == ["Default"]
        keys = {f["key"] for f in body["families"]}
        assert {"qwen3", "qwen3.6", "gemma4", "llava", "text"} <= keys

    def test_crud_roundtrip(self, client, tmp_path):
        """Create → update → delete through the API."""
        created = client.post(
            "/api/profiles",
            json={
                "file": "Qwen3-VL-8B-Q8_0.gguf",
                "dir": str(tmp_path),
                "role": "caption",
            },
        ).json()
        assert created["type"] == "qwen3"
        assert client.get("/api/profiles").json()["active_id"] == (
            created["id"]
        )
        updated = client.put(
            f"/api/profiles/{created['id']}", json={"name": "Mine"}
        ).json()
        assert updated["name"] == "Mine"
        assert (
            client.delete(f"/api/profiles/{created['id']}").status_code == 200
        )

    def test_delete_last_is_409(self, client):
        """The last remaining profile cannot be deleted."""
        only = client.get("/api/profiles").json()["profiles"][0]["id"]
        assert client.delete(f"/api/profiles/{only}").status_code == 409

    def test_select_judge(self, client):
        """POST /select moves the judge slot."""
        created = client.post("/api/profiles", json={"name": "J"}).json()
        ok = client.post(
            "/api/profiles/select",
            json={"role": "judge", "id": created["id"]},
        )
        assert ok.status_code == 200
        assert client.get("/api/profiles").json()["judge_id"] == (
            created["id"]
        )
        missing = client.post(
            "/api/profiles/select", json={"role": "judge", "id": 999}
        )
        assert missing.status_code == 404

    def test_browse_lists_model_files(self, client, tmp_path):
        """The picker lists folders and weight files with sizes."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "model.gguf").write_bytes(b"x" * 10)
        (tmp_path / "notes.txt").touch()
        body = client.get(
            "/api/profiles/browse", params={"path": str(tmp_path)}
        ).json()
        kinds = {e["name"]: e["kind"] for e in body["entries"]}
        assert kinds == {"sub": "dir", "model.gguf": "file"}
        file_entry = next(e for e in body["entries"] if e["kind"] == "file")
        assert file_entry["size"] == 10

    def test_detect_route(self, client, tmp_path):
        """Detection reports family, format, projector and auto-name."""
        (tmp_path / "mmproj-gemma-3-12b-it-f16.gguf").touch()
        body = client.post(
            "/api/profiles/detect",
            json={"dir": str(tmp_path), "file": "gemma-3-12b-it-Q8.gguf"},
        ).json()
        assert body["type"] == "gemma3"
        assert body["format"] == "gguf"
        assert body["mmproj"] == "mmproj-gemma-3-12b-it-f16.gguf"
        assert body["name"] == "gemma-3-12b-it-Q8"

    def test_load_without_file_is_409(self, client):
        """Loading a profile with no picked weights is refused."""
        only = client.get("/api/profiles").json()["profiles"][0]["id"]
        assert client.post(f"/api/profiles/{only}/load").status_code == 409
        assert client.post("/api/profiles/999/load").status_code == 404
