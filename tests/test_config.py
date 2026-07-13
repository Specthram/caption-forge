"""Tests for :mod:`src.config` (merge, per-type config, migration)."""

import json

import pytest

from src import config
from src.config import deep_merge


def test_override_wins_on_scalar():
    """A scalar in the override replaces the base value."""
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_missing_keys_are_added():
    """Keys only in the override are added to the result."""
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_nested_dicts_merge_recursively():
    """Two nested dicts merge key by key instead of replacing."""
    base = {"x": {"a": 1, "b": 2}}
    override = {"x": {"b": 3, "c": 4}}
    assert deep_merge(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}


def test_inputs_are_not_mutated():
    """``deep_merge`` returns a new dict and leaves inputs untouched."""
    base = {"x": {"a": 1}}
    override = {"x": {"b": 2}}
    deep_merge(base, override)
    assert base == {"x": {"a": 1}}
    assert override == {"x": {"b": 2}}


def test_dict_replaces_scalar_when_types_differ():
    """A dict override replaces a scalar base (no recursion possible)."""
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


def _write_json(path, data):
    """Write ``data`` as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(name="model_layers")
def _model_layers(tmp_path, monkeypatch):
    """Redirect the default/user model + settings dirs to a temp tree."""
    default_dir = tmp_path / "default" / "models"
    user_dir = tmp_path / "user" / "models"
    user_config = tmp_path / "user"
    monkeypatch.setattr(config, "DEFAULT_MODELS_DIR", default_dir)
    monkeypatch.setattr(config, "USER_MODELS_DIR", user_dir)
    monkeypatch.setattr(config, "USER_CONFIG_DIR", user_config)
    return default_dir, user_dir, user_config


class TestLoadModelConfig:
    """Tests for :func:`config.load_model_config`."""

    def test_merges_prompts_and_settings(self, model_layers):
        """User layer overlays default, missing keys are inherited."""
        default_dir, user_dir, _ = model_layers
        _write_json(
            default_dir / "qwen3.json",
            {
                "prompts": {"A": {"prompt": "a", "default": True}},
                "settings": {"think_mode": "auto", "temperature": 0.7},
            },
        )
        _write_json(
            user_dir / "qwen3.json",
            {"settings": {"think_mode": "off"}},
        )
        merged = config.load_model_config("qwen3")
        assert merged["prompts"] == {"A": {"prompt": "a", "default": True}}
        assert merged["settings"] == {"think_mode": "off", "temperature": 0.7}

    def test_missing_type_returns_empty(self, model_layers):
        """An unknown model type yields an empty config."""
        assert config.load_model_config("nope") == {}


class TestSaveUserPrompt:
    """Tests for :func:`config.save_user_prompt`."""

    def test_preserves_existing_settings(self, model_layers):
        """Adding a prompt leaves the user ``settings`` block intact."""
        _, user_dir, _ = model_layers
        _write_json(
            user_dir / "qwen3.json",
            {"settings": {"think_mode": "off"}},
        )
        config.save_user_prompt("qwen3", "My", "text")
        data = json.loads(
            (user_dir / "qwen3.json").read_text(encoding="utf-8")
        )
        assert data["prompts"]["My"] == {"prompt": "text"}
        assert data["settings"] == {"think_mode": "off"}


class TestSetUserModelSetting:
    """Tests for :func:`config.set_user_model_setting`."""

    def test_preserves_existing_prompts(self, model_layers):
        """Writing a setting leaves the user ``prompts`` block intact."""
        _, user_dir, _ = model_layers
        _write_json(
            user_dir / "qwen3.json",
            {"prompts": {"My": {"prompt": "text"}}},
        )
        config.set_user_model_setting("qwen3", "temperature", 1.2)
        data = json.loads(
            (user_dir / "qwen3.json").read_text(encoding="utf-8")
        )
        assert data["settings"] == {"temperature": 1.2}
        assert data["prompts"] == {"My": {"prompt": "text"}}


class TestMigrateLegacyLayout:
    """Tests for :func:`config.migrate_legacy_layout`."""

    def test_moves_selected_prompts_into_model_files(self, model_layers):
        """``selected_prompts`` moves to each type and leaves settings."""
        _, user_dir, user_config = model_layers
        _write_json(
            user_config / "settings.json",
            {"device": "cuda", "selected_prompts": {"qwen3": "AI"}},
        )
        config.migrate_legacy_layout()

        settings_data = json.loads(
            (user_config / "settings.json").read_text(encoding="utf-8")
        )
        assert "selected_prompts" not in settings_data
        assert settings_data["device"] == "cuda"

        model_data = json.loads(
            (user_dir / "qwen3.json").read_text(encoding="utf-8")
        )
        assert model_data["settings"]["selected_prompt"] == "AI"

    def test_folds_legacy_user_prompt_files(self, model_layers):
        """Legacy ``user/prompts/*.json`` are folded into the models layer."""
        _, user_dir, user_config = model_layers
        _write_json(
            user_config / "prompts" / "qwen3.json",
            {"My": {"prompt": "text"}},
        )
        config.migrate_legacy_layout()

        assert not (user_config / "prompts").exists()
        model_data = json.loads(
            (user_dir / "qwen3.json").read_text(encoding="utf-8")
        )
        assert model_data["prompts"]["My"] == {"prompt": "text"}

    def test_is_idempotent(self, model_layers):
        """A second run with nothing legacy left is a no-op."""
        _, _, user_config = model_layers
        _write_json(user_config / "settings.json", {"device": "cuda"})
        config.migrate_legacy_layout()
        config.migrate_legacy_layout()
        settings_data = json.loads(
            (user_config / "settings.json").read_text(encoding="utf-8")
        )
        assert settings_data == {"device": "cuda"}
