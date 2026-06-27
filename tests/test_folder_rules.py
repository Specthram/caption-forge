"""Tests for the subfolder-mapping engine and its repository."""

from src import folder_rules as fr
from src import sqlite_store as store


class TestSlugAndChain:
    """slugify / rel_folder path helpers."""

    def test_slugify_collapses_whitespace(self):
        """Spaces (any run) become a single underscore; ends trimmed."""
        assert fr.slugify("  hugo   boss ") == "hugo_boss"

    def test_slugify_preserves_case(self):
        """Case is kept verbatim (tags match exactly elsewhere)."""
        assert fr.slugify("Café Menu") == "Café_Menu"

    def test_rel_folder_under_root(self):
        """A file's folder is returned relative to the root, ``/``-joined."""
        rel = fr.rel_folder(r"C:\d\vet\robe\a.png", r"C:\d\vet")
        assert rel == "robe"

    def test_rel_folder_root_level(self):
        """A file directly in the root yields the empty rel_path."""
        assert fr.rel_folder(r"C:\d\vet\a.png", r"C:\d\vet") == ""

    def test_rel_folder_outside_root(self):
        """A file outside the root degrades to the root, never raises."""
        assert fr.rel_folder(r"D:\other\a.png", r"C:\d\vet") == ""


class TestExcludeAndRouting:
    """is_excluded / owning_sub_library."""

    def test_exclude_cascades_to_descendants(self):
        """An excluded folder excludes everything beneath it."""
        rules = {"sys": {"mode": fr.MODE_EXCLUDE}}
        assert fr.is_excluded("sys", rules)
        assert fr.is_excluded("sys/logs", rules)
        assert not fr.is_excluded("other", rules)

    def test_owning_sub_library_takes_the_deepest(self):
        """The closest sublib ancestor owns the file; else the parent."""
        rules = {
            "a": {"mode": fr.MODE_SUBLIB, "sub_library_id": 5},
            "a/b": {"mode": fr.MODE_SUBLIB, "sub_library_id": 7},
        }
        assert fr.owning_sub_library("a/b/c", rules) == 7
        assert fr.owning_sub_library("a/x", rules) == 5
        assert fr.owning_sub_library("z", rules) is None

    def test_sublib_without_id_is_ignored(self):
        """A sublib rule not yet applied (no id) routes to the parent."""
        rules = {"a": {"mode": fr.MODE_SUBLIB, "sub_library_id": None}}
        assert fr.owning_sub_library("a/b", rules) is None


class TestEffectiveTags:
    """effective_tags across auto levels, inheritance and overrides."""

    def test_manual_tags_inherit_down_the_chain(self):
        """A child sees its ancestors' manual tags plus its own."""
        rules = {
            "a": {"mode": fr.MODE_KEEP, "tags": ["red"], "removed": []},
            "a/b": {"mode": fr.MODE_KEEP, "tags": ["blue"], "removed": []},
        }
        assert fr.effective_tags("a", rules, fr.AUTO_OFF) == ["red"]
        assert fr.effective_tags("a/b", rules, fr.AUTO_OFF) == ["red", "blue"]

    def test_removed_overrides_only_below(self):
        """Removing an inherited tag on a child drops it there only."""
        rules = {
            "a": {"mode": fr.MODE_KEEP, "tags": ["red"], "removed": []},
            "a/b": {
                "mode": fr.MODE_KEEP,
                "tags": ["blue"],
                "removed": ["red"],
            },
        }
        assert fr.effective_tags("a", rules, fr.AUTO_OFF) == ["red"]
        assert fr.effective_tags("a/b", rules, fr.AUTO_OFF) == ["blue"]

    def test_auto_all_tags_every_depth(self):
        """AUTO_ALL adds each folder's slug and inherits it downward."""
        tags = fr.effective_tags("robe/ete", {}, fr.AUTO_ALL)
        assert tags == ["robe", "ete"]

    def test_auto_top_tags_only_the_top_level(self):
        """AUTO_TOP tags depth 1 only; deeper folders just inherit it."""
        assert fr.effective_tags("robe", {}, fr.AUTO_TOP) == ["robe"]
        assert fr.effective_tags("robe/ete", {}, fr.AUTO_TOP) == ["robe"]

    def test_auto_tag_can_be_removed(self):
        """A folder's ``removed`` list can drop its own auto tag."""
        rules = {"robe": {"mode": fr.MODE_KEEP, "removed": ["robe"]}}
        assert fr.effective_tags("robe", rules, fr.AUTO_ALL) == []

    def test_root_tags_apply_to_root_and_cascade(self):
        """Root ("") tags tag root files and every sub-folder inherits them."""
        rules = {
            "": {"mode": fr.MODE_KEEP, "tags": ["vetement"], "removed": []},
            "robe": {"mode": fr.MODE_KEEP, "tags": ["red"], "removed": []},
        }
        assert fr.effective_tags("", rules, fr.AUTO_OFF) == ["vetement"]
        assert fr.effective_tags("robe", rules, fr.AUTO_OFF) == [
            "vetement",
            "red",
        ]

    def test_root_tag_can_be_removed_on_a_child(self):
        """A child's ``removed`` list drops an inherited root tag locally."""
        rules = {
            "": {"mode": fr.MODE_KEEP, "tags": ["vetement"]},
            "robe": {"mode": fr.MODE_KEEP, "removed": ["vetement"]},
        }
        assert fr.effective_tags("robe", rules, fr.AUTO_OFF) == []
        assert fr.effective_tags("autre", rules, fr.AUTO_OFF) == ["vetement"]

    def test_excluded_folder_has_no_tags(self):
        """An excluded folder resolves to no tags at all."""
        rules = {"sys": {"mode": fr.MODE_EXCLUDE}}
        assert fr.effective_tags("sys/x", rules, fr.AUTO_ALL) == []

    def test_no_duplicate_when_manual_repeats_auto(self):
        """A manual tag equal to the auto tag appears once."""
        rules = {"robe": {"mode": fr.MODE_KEEP, "tags": ["robe"]}}
        assert fr.effective_tags("robe", rules, fr.AUTO_ALL) == ["robe"]

    def test_deep_chain_accumulates_every_level(self):
        """A 4-level folder inherits every ancestor's auto tag, in order."""
        tags = fr.effective_tags(
            "bijoux/rollex/daytona/macro", {}, fr.AUTO_ALL
        )
        assert tags == ["bijoux", "rollex", "daytona", "macro"]

    def test_deep_chain_manual_and_override_midway(self):
        """A manual tag high up flows down; a mid-chain removal stops there."""
        rules = {
            "bijoux": {"mode": fr.MODE_KEEP, "tags": ["luxe"]},
            # Drop the inherited "luxe" from daytona and everything below it,
            # while daytona's own auto tag still applies.
            "bijoux/rollex/daytona": {
                "mode": fr.MODE_KEEP,
                "removed": ["luxe"],
            },
        }
        assert fr.effective_tags("bijoux/rollex", rules, fr.AUTO_ALL) == [
            "bijoux",
            "luxe",
            "rollex",
        ]
        assert fr.effective_tags(
            "bijoux/rollex/daytona/macro", rules, fr.AUTO_ALL
        ) == ["bijoux", "rollex", "daytona", "macro"]


class TestResolveFile:
    """resolve_file combines routing and tags in one call."""

    def test_keep_folder_routes_to_parent_with_tags(self):
        """A kept folder returns (None, tags)."""
        rules = {"robe": {"mode": fr.MODE_KEEP, "tags": ["red"]}}
        assert fr.resolve_file("robe", rules, fr.AUTO_ALL) == (
            None,
            ["robe", "red"],
        )

    def test_sublib_folder_routes_to_its_library(self):
        """A sublib folder returns (sub_library_id, tags)."""
        rules = {"robe": {"mode": fr.MODE_SUBLIB, "sub_library_id": 9}}
        owner, tags = fr.resolve_file("robe", rules, fr.AUTO_ALL)
        assert owner == 9 and tags == ["robe"]

    def test_excluded_file_resolves_to_none(self):
        """An excluded folder yields None so the scanner skips the file."""
        rules = {"sys": {"mode": fr.MODE_EXCLUDE}}
        assert fr.resolve_file("sys", rules, fr.AUTO_OFF) is None


class TestFolderRulesRepo:
    """The sqlite_store folder-rules primitives."""

    def _parent(self):
        """Create and return a parent folder library id."""
        return store.create_library("vet", r"C:\d\vet", recursive=True)

    def test_auto_tag_level_round_trip(self, store_db):
        # pylint: disable=unused-argument
        """The level defaults to off and persists once set."""
        parent = self._parent()
        assert store.get_folder_mapping(parent)["auto_tag_level"] == "0"
        store.set_library_auto_tag_level(parent, "all")
        assert store.get_folder_mapping(parent)["auto_tag_level"] == "all"

    def test_get_or_create_sub_library_creates_then_reuses(self, store_db):
        # pylint: disable=unused-argument
        """A second call on the same path reuses the row and refreshes it."""
        parent = self._parent()
        first = store.get_or_create_sub_library(
            parent, "hugo", r"C:\d\vet\hugo", "hugo"
        )
        again = store.get_or_create_sub_library(
            parent, "hugo boss", r"C:\d\vet\hugo", "hugo"
        )
        assert first == again
        subs = store.list_sub_libraries(parent)
        assert [row["name"] for row in subs] == ["hugo boss"]
        assert subs[0]["parent_library_id"] == parent
        assert subs[0]["rel_path"] == "hugo"

    def test_replace_folder_rules_round_trip(self, store_db):
        # pylint: disable=unused-argument
        """Rules read back as parsed dicts and feed the resolver map."""
        parent = self._parent()
        sub = store.get_or_create_sub_library(
            parent, "hugo", r"C:\d\vet\hugo", "hugo"
        )
        store.replace_folder_rules(
            parent,
            [
                {
                    "rel_path": "hugo",
                    "mode": "sublib",
                    "sub_library_id": sub,
                    "tags": ["brand"],
                    "removed": [],
                },
                {
                    "rel_path": "sys",
                    "mode": "exclude",
                    "sub_library_id": None,
                    "tags": [],
                    "removed": [],
                },
            ],
        )
        mapping = store.get_folder_mapping(parent)
        assert {rule["rel_path"] for rule in mapping["rules"]} == {
            "hugo",
            "sys",
        }
        level, rules_map = store.folder_rules_map(parent)
        assert level == "0"
        assert rules_map["hugo"]["sub_library_id"] == sub
        assert rules_map["hugo"]["tags"] == ["brand"]
        assert rules_map["sys"]["mode"] == "exclude"

    def test_replace_folder_rules_is_authoritative(self, store_db):
        # pylint: disable=unused-argument
        """A second replace wipes the previous rule set."""
        parent = self._parent()
        store.replace_folder_rules(parent, [{"rel_path": "a", "mode": "keep"}])
        store.replace_folder_rules(
            parent, [{"rel_path": "b", "mode": "exclude"}]
        )
        rels = [
            r["rel_path"] for r in store.get_folder_mapping(parent)["rules"]
        ]
        assert rels == ["b"]

    def test_mapping_stats(self, store_db):
        # pylint: disable=unused-argument
        """Stats report mapped state, rule/skip/sub counts for the sidebar."""
        parent = self._parent()
        assert store.library_mapping_stats(parent)["mapped"] is False
        sub = store.get_or_create_sub_library(
            parent, "hugo", r"C:\d\vet\hugo", "hugo"
        )
        store.replace_folder_rules(
            parent,
            [
                {"rel_path": "hugo", "mode": "sublib", "sub_library_id": sub},
                {"rel_path": "sys", "mode": "exclude"},
            ],
        )
        stats = store.library_mapping_stats(parent)
        assert stats["mapped"] is True
        assert stats["rule_count"] == 2
        assert stats["skipped_folders"] == 1
        assert stats["sub_count"] == 1

    def test_apply_prunes_demoted_sub_libraries(self, store_db):
        # pylint: disable=unused-argument
        """Re-applying without a sublib rule removes its sub-library."""
        parent = self._parent()
        store.apply_folder_mapping(
            parent,
            "0",
            [{"rel_path": "hugo", "mode": "sublib", "sub_name": "Hugo"}],
        )
        assert len(store.list_sub_libraries(parent)) == 1
        store.apply_folder_mapping(
            parent, "0", [{"rel_path": "hugo", "mode": "keep"}]
        )
        assert store.list_sub_libraries(parent) == []

    def test_delete_parent_promotes_children(self, store_db):
        # pylint: disable=unused-argument
        """Deleting a parent orphans its rules but keeps its sub-libraries."""
        parent = self._parent()
        sub = store.get_or_create_sub_library(
            parent, "hugo", r"C:\d\vet\hugo", "hugo"
        )
        store.replace_folder_rules(
            parent,
            [{"rel_path": "hugo", "mode": "sublib", "sub_library_id": sub}],
        )
        store.delete_library(parent)
        child = store.get_library(sub)
        assert child is not None
        assert child["parent_library_id"] is None
        assert store.get_folder_mapping(parent)["rules"] == []
