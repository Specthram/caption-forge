"""Tests for the rule-based review pipeline: repo, storage and a det run.

Covers everything reachable without a model: the ``review_rule`` /
``review_run`` / ``review_finding`` repository, the storage bridge (rules,
findings, accept / reject / undo), and a full deterministic (trigger-word)
run driven synchronously through the job body — no judge model required.
"""

import pytest
from PIL import Image

from server.runners.review_run import review_run_body
from server.schemas import ReviewRunBody
from src import sqlite_store as store
from src import storage


class _Progress:
    """Minimal stand-in for :class:`server.jobs.Progress`."""

    def __call__(self, done=None, total=None, sub=None):
        """Accept and ignore progress updates."""

    def warn(self, message):
        """Accept and ignore a non-fatal warning."""


@pytest.fixture(name="dataset")
def _dataset(store_db, tmp_path):
    """Seed a dataset with one captioned image missing its trigger word."""
    # pylint: disable=unused-argument
    image = tmp_path / "red.png"
    Image.new("RGB", (32, 32), (219, 68, 55)).save(image)
    library_id = store.create_library("fixtures", str(tmp_path))
    store.scan_library(library_id)
    media = store.list_library_media()
    dataset_id = store.create_dataset("shapes_all")
    for row in media:
        store.add_media_to_dataset(dataset_id, row["id"])
    key = str(media[0]["id"])
    storage.write_caption(dataset_id, key, "txt", "a red ball on the grass.")
    store.add_triggerword_to_dataset(dataset_id, "ryn")
    return {"dataset_id": dataset_id, "key": key}


def _det_rule(dataset_id):
    """Return the seeded deterministic (trigger-word) rule of a dataset."""
    rules = storage.review_rules(dataset_id)
    return next(rule for rule in rules if rule["kind"] == "det")


def _run_det(dataset_id, scope="all", media_ids=None):
    """Run a deterministic-only review synchronously; return findings count."""
    det = _det_rule(dataset_id)
    body = ReviewRunBody(
        dataset_id=dataset_id,
        caption_type="txt",
        media_ids=media_ids,
        judge_profile_id=None,
        scope=scope,
        rule_ids=[det["id"]],
    )
    return review_run_body(body)(_Progress())


class TestRules:
    """Rule presets and CRUD through the storage bridge."""

    def test_presets_seeded_once(self, dataset):
        """First read seeds the builtin presets, embedding the trigger word."""
        rules = storage.review_rules(dataset["dataset_id"])
        kinds = {rule["kind"] for rule in rules}
        assert kinds == {"det", "vlm"}
        det = next(rule for rule in rules if rule["kind"] == "det")
        assert '"ryn"' in det["text"]
        assert det["builtin"] is True

    def test_seed_is_idempotent(self, dataset):
        """Reading rules twice does not duplicate the presets."""
        first = storage.review_rules(dataset["dataset_id"])
        second = storage.review_rules(dataset["dataset_id"])
        assert len(first) == len(second)

    def test_create_custom_rule_kind_from_image(self, dataset):
        """A custom rule needing the image is a vision rule; else text."""
        text_rule = storage.create_review_rule(
            dataset["dataset_id"], "no subjective words", needs_image=False
        )
        vision_rule = storage.create_review_rule(
            dataset["dataset_id"], "clothes match", needs_image=True
        )
        assert text_rule["kind"] == "text"
        assert vision_rule["kind"] == "vlm"
        assert vision_rule["needs_image"] is True

    def test_toggle_and_delete(self, dataset):
        """A custom rule can be toggled off and deleted."""
        rule = storage.create_review_rule(
            dataset["dataset_id"], "x", needs_image=False
        )
        store.set_rule_enabled(rule["id"], False)
        assert store.get_rule(rule["id"])["enabled"] is False
        store.delete_rule(rule["id"])
        assert store.get_rule(rule["id"]) is None


class TestDetRun:
    """A deterministic run produces safe, acceptable findings."""

    def test_run_flags_missing_trigger(self, dataset):
        """The missing trigger word yields one pending det finding."""
        result = _run_det(dataset["dataset_id"])
        assert result["findings"] == 1
        findings = storage.review_findings(dataset["dataset_id"])
        assert len(findings) == 1
        finding = findings[0]
        assert finding["rule_kind"] == "det"
        assert finding["status"] == "pending"
        assert finding["caption_after"].startswith("ryn, ")

    def test_accept_writes_revision(self, dataset):
        """Accepting a finding rewrites the caption and marks it accepted."""
        _run_det(dataset["dataset_id"])
        finding = storage.review_findings(dataset["dataset_id"])[0]
        storage.decide_review_finding(finding["id"], "accept")
        caption = storage.read_caption(
            dataset["dataset_id"], dataset["key"], "txt"
        )
        assert caption.startswith("ryn, ")
        assert store.get_finding(finding["id"])["status"] == "accepted"

    def test_second_accept_merges_with_the_first(self, dataset):
        """Accepting two findings keeps both fixes (diff-merge on accept)."""
        base = "a red ball on the grass."
        run_id = storage.open_review_run(dataset["dataset_id"], "", "all", 1)
        media_id = int(dataset["key"])
        first = storage.record_review_finding(
            run_id,
            media_id,
            "txt",
            "color",
            base,
            "a crimson ball on the grass.",
        )
        second = storage.record_review_finding(
            run_id,
            media_id,
            "txt",
            "ground",
            base,
            "a red ball on the lawn.",
        )
        storage.close_review_run(run_id, 2)
        storage.decide_review_finding(first, "accept")
        # The sibling is rebased: its "original" now shows the first accept
        # applied, and its proposal is the same fix on top of it.
        sibling = store.get_finding(second)
        assert sibling["caption_before"] == "a crimson ball on the grass."
        assert sibling["caption_after"] == "a crimson ball on the lawn."
        storage.decide_review_finding(second, "accept")
        caption = storage.read_caption(
            dataset["dataset_id"], dataset["key"], "txt"
        )
        assert caption == "a crimson ball on the lawn."

    def test_undo_restores_caption(self, dataset):
        """Undo restores the original caption and reopens the finding."""
        _run_det(dataset["dataset_id"])
        finding = storage.review_findings(dataset["dataset_id"])[0]
        storage.decide_review_finding(finding["id"], "accept")
        storage.undo_review_finding(finding["id"])
        caption = storage.read_caption(
            dataset["dataset_id"], dataset["key"], "txt"
        )
        assert caption == "a red ball on the grass."
        assert store.get_finding(finding["id"])["status"] == "pending"

    def test_reject_leaves_caption(self, dataset):
        """Rejecting a finding does not touch the caption."""
        _run_det(dataset["dataset_id"])
        finding = storage.review_findings(dataset["dataset_id"])[0]
        storage.decide_review_finding(finding["id"], "reject")
        caption = storage.read_caption(
            dataset["dataset_id"], dataset["key"], "txt"
        )
        assert caption == "a red ball on the grass."
        assert store.get_finding(finding["id"])["status"] == "rejected"

    def test_clean_caption_no_finding(self, dataset):
        """A caption already carrying the trigger word yields no finding."""
        storage.write_caption(
            dataset["dataset_id"], dataset["key"], "txt", "ryn, a red ball."
        )
        assert _run_det(dataset["dataset_id"])["findings"] == 0

    def test_accept_safe_fixes_bulk(self, dataset):
        """Bulk safe-accept applies the deterministic finding."""
        _run_det(dataset["dataset_id"])
        applied = storage.accept_safe_fixes(dataset["dataset_id"])
        assert applied == 1
        counts = storage.review_counts(dataset["dataset_id"])
        assert counts["accepted"] == 1
        assert counts["pending"] == 0


class TestMergeSemantics:
    """Whole-dataset runs replace the queue; single-media runs merge."""

    def test_global_run_replaces(self, dataset):
        """A second global run replaces the first run's findings."""
        _run_det(dataset["dataset_id"])
        _run_det(dataset["dataset_id"])
        assert len(storage.review_findings(dataset["dataset_id"])) == 1

    def test_single_run_clears_only_that_media(self, dataset):
        """A single-media run clears only that media's findings."""
        _run_det(dataset["dataset_id"])
        before = storage.review_findings(dataset["dataset_id"])[0]
        _run_det(
            dataset["dataset_id"],
            scope="single",
            media_ids=[int(dataset["key"])],
        )
        after = storage.review_findings(dataset["dataset_id"])
        assert len(after) == 1
        # A fresh finding row (the old one was cleared for this media).
        assert after[0]["id"] != before["id"]
