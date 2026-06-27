"""Tests for :mod:`src.deploy` (SQLite dataset deployment + sync state)."""

# Testing private helpers and a deliberate one-media item builder.
# pylint: disable=protected-access

import pytest

from src import deploy
from src import sqlite_store as store
from src import storage


@pytest.fixture(name="deployed")
def _deployed(tmp_path, monkeypatch, store_db):
    # pylint: disable=unused-argument
    """Set up a SQLite dataset with one media and a configured deploy root.

    Returns ``(dataset_id, key, image, deploy_root)`` where ``key`` is the
    media's database id, ``image`` its file path and ``deploy_root`` the deploy
    directory.
    """
    deploy_root = tmp_path / "deployed"
    monkeypatch.setattr(deploy, "get_deploy_dir", lambda: deploy_root)

    image = tmp_path / "pics" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image-bytes")
    dataset_id = store.create_dataset("my set")
    media_id, _ = store.ingest_file(str(image))
    store.add_media_to_dataset(dataset_id, media_id)
    storage.write_caption(dataset_id, str(media_id), "txt", "a cat")
    return dataset_id, str(media_id), str(image), deploy_root


def _items(key, image, hidden=False, ext="txt", missing=False, repeats=1):
    """Return a one-media deploy item list (hash from the file content)."""
    return [
        {
            "key": key,
            "path": image,
            "hidden": hidden,
            "ext": ext,
            "sha256": store.compute_sha256(image),
            "file_ext": "png",
            "missing": missing,
            "repeats": repeats,
        }
    ]


def _sha(image):
    """Return the content hash of an image file."""
    return store.compute_sha256(image)


class TestSanitizeName:
    """The dataset name maps to a safe single path component."""

    def test_replaces_unsafe_chars(self):
        """Reserved characters become underscores."""
        assert deploy._sanitize_name("a/b:c?") == "a_b_c_"

    def test_blank_falls_back(self):
        """A blank name falls back to a default."""
        assert deploy._sanitize_name("   ") == "dataset"


class TestStatusBeforeDeploy:
    """Nothing is deployed yet."""

    def test_dataset_red(self, deployed):
        """An undeployed dataset reports RED."""
        dataset_id, key, image, _ = deployed
        assert (
            deploy.dataset_status(dataset_id, _items(key, image)) == deploy.RED
        )

    def test_image_red(self, deployed):
        """An undeployed visible media reports RED."""
        dataset_id, key, image, _ = deployed
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png"
            )
            == deploy.RED
        )

    def test_no_root_is_none(self, deployed, monkeypatch):
        """With no deploy root configured the state is NONE."""
        dataset_id, key, image, _ = deployed
        monkeypatch.setattr(deploy, "get_deploy_dir", lambda: None)
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png"
            )
            == deploy.NONE
        )


class TestDeploy:
    """Deploying mirrors the dataset and turns it green."""

    def test_writes_hash_named_image_and_caption(self, deployed):
        """Files are named by content hash to avoid basename collisions."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        result = deploy.deploy_dataset(dataset_id, _items(key, image))
        folder = root / "my set"
        assert (folder / f"{sha}.png").is_file()
        assert (folder / f"{sha}.txt").read_text(encoding="utf-8") == "a cat"
        # The original basename is never used as the deployed name.
        assert not (folder / "a.png").exists()
        assert result["copied"] == 1

    def test_triggerword_prefix_applied(self, deployed):
        """Deployed captions carry the dataset's trigger-word prefix."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        deploy.deploy_dataset(dataset_id, _items(key, image))
        caption = (root / "my set" / f"{sha}.txt").read_text(encoding="utf-8")
        assert caption == "xbl1. a cat"

    def test_green_after_deploy(self, deployed):
        """Both image and dataset are GREEN once deployed."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png"
            )
            == deploy.GREEN
        )
        assert (
            deploy.dataset_status(dataset_id, _items(key, image))
            == deploy.GREEN
        )

    def test_edit_makes_orange(self, deployed):
        """Editing the caption after deploy makes the media ORANGE."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        storage.write_caption(dataset_id, key, "txt", "a dog")
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png"
            )
            == deploy.ORANGE
        )
        assert (
            deploy.dataset_status(dataset_id, _items(key, image))
            == deploy.ORANGE
        )


class TestHidden:
    """A hidden media is white and pruned from the folder on deploy."""

    def test_hidden_image_white(self, deployed):
        """A hidden media always reports WHITE."""
        dataset_id, key, image, _ = deployed
        assert (
            deploy.image_status(
                dataset_id, key, "txt", True, _sha(image), "png"
            )
            == deploy.WHITE
        )

    def test_deploy_prunes_hidden(self, deployed):
        """Deploying after hiding removes the media from the folder."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_dataset(dataset_id, _items(key, image))
        # Hide it, then redeploy: the file must be pruned.
        result = deploy.deploy_dataset(
            dataset_id, _items(key, image, hidden=True)
        )
        folder = root / "my set"
        assert not (folder / f"{sha}.png").exists()
        assert not (folder / f"{sha}.txt").exists()
        assert result["removed"] == 2

    def test_hidden_still_present_makes_orange(self, deployed):
        """A hidden media left in the folder makes the dataset ORANGE."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        # Hide without redeploying: the stale file remains, so out of sync.
        assert (
            deploy.dataset_status(dataset_id, _items(key, image, hidden=True))
            == deploy.ORANGE
        )


class TestDeployName:
    """The deploy sub-folder honors the dataset's configured deploy name."""

    def test_deploy_name_overrides_folder(self, deployed):
        """A configured deploy name replaces the dataset-name folder."""
        dataset_id, key, image, root = deployed
        store.update_dataset(dataset_id, deploy_name="custom folder")
        deploy.deploy_dataset(dataset_id, _items(key, image))
        assert (root / "custom folder" / f"{_sha(image)}.png").is_file()
        assert not (root / "my set").exists()

    def test_blank_deploy_name_falls_back(self, deployed):
        """A blank deploy name is stored as unset (dataset name is used)."""
        dataset_id, key, image, root = deployed
        store.update_dataset(dataset_id, deploy_name="   ")
        deploy.deploy_dataset(dataset_id, _items(key, image))
        assert (root / "my set" / f"{_sha(image)}.png").is_file()


class TestDifferentialDeploy:
    """A redeploy only rewrites what actually changed."""

    def test_first_deploy_writes_both_files(self, deployed):
        """The initial deploy writes the image and the caption (2 files)."""
        dataset_id, key, image, _ = deployed
        result = deploy.deploy_dataset(dataset_id, _items(key, image))
        assert result["written"] == 2

    def test_redeploy_unchanged_writes_nothing(self, deployed):
        """Re-deploying an up-to-date dataset touches no file."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        result = deploy.deploy_dataset(dataset_id, _items(key, image))
        assert result["written"] == 0

    def test_redeploy_after_edit_writes_only_caption(self, deployed):
        """Editing the caption then redeploying rewrites just the caption."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        storage.write_caption(dataset_id, key, "txt", "a dog")
        result = deploy.deploy_dataset(dataset_id, _items(key, image))
        assert result["written"] == 1


class TestDeployMedia:
    """Deploying a single media, scoped and differential."""

    def test_deploys_one_media(self, deployed):
        """A single media's files are written and it turns GREEN."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        result = deploy.deploy_media(dataset_id, _items(key, image)[0])
        assert result["deployed"] is True and result["written"] == 2
        assert (root / "my set" / f"{sha}.png").is_file()
        assert (
            deploy.image_status(dataset_id, key, "txt", False, sha, "png")
            == deploy.GREEN
        )

    def test_second_deploy_is_differential(self, deployed):
        """Re-deploying the same media writes nothing."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_media(dataset_id, _items(key, image)[0])
        result = deploy.deploy_media(dataset_id, _items(key, image)[0])
        assert result["written"] == 0

    def test_hidden_media_is_not_deployed(self, deployed):
        """A hidden media writes nothing and reports not deployed."""
        dataset_id, key, image, root = deployed
        result = deploy.deploy_media(
            dataset_id, _items(key, image, hidden=True)[0]
        )
        assert result["deployed"] is False and result["written"] == 0
        assert not (root / "my set" / f"{_sha(image)}.png").exists()

    def test_leaves_other_media_untouched(self, deployed):
        """Deploying one media never prunes another media's files."""
        dataset_id, key, image, root = deployed
        folder = root / "my set"
        folder.mkdir(parents=True)
        # A stray file belonging to some other (already deployed) media.
        (folder / "otherhash.png").write_bytes(b"x")
        deploy.deploy_media(dataset_id, _items(key, image)[0])
        assert (folder / "otherhash.png").exists()

    def test_lowering_repeats_prunes_only_own_extras(self, deployed):
        """Deploying with fewer repeats prunes just this media's extras."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_media(dataset_id, _items(key, image, repeats=3)[0])
        result = deploy.deploy_media(dataset_id, _items(key, image)[0])
        folder = root / "my set"
        assert not (folder / f"{sha}_1.png").exists()
        assert not (folder / f"{sha}_2.txt").exists()
        assert result["removed"] == 4  # _1 and _2 image + caption


class TestUndeploy:
    """Removing deployed files, per media and per dataset."""

    def test_undeploy_media_removes_its_files(self, deployed):
        """Undeploying one media deletes its files and it turns RED."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_dataset(dataset_id, _items(key, image))
        result = deploy.undeploy_media(dataset_id, sha, "png")
        folder = root / "my set"
        assert not (folder / f"{sha}.png").exists()
        assert not (folder / f"{sha}.txt").exists()
        assert result["removed"] == 2
        assert (
            deploy.image_status(dataset_id, key, "txt", False, sha, "png")
            == deploy.RED
        )

    def test_undeploy_media_removes_all_repeats(self, deployed):
        """Every deployed copy of a media is removed, not just the first."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_dataset(dataset_id, _items(key, image, repeats=3))
        result = deploy.undeploy_media(dataset_id, sha, "png")
        folder = root / "my set"
        assert not any(folder.iterdir())
        assert result["removed"] == 6  # 3 images + 3 captions

    def test_undeploy_dataset_clears_folder(self, deployed):
        """Undeploying the dataset removes every file and turns it RED."""
        dataset_id, key, image, root = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image, repeats=2))
        result = deploy.undeploy_dataset(dataset_id)
        folder = root / "my set"
        assert folder.is_dir() and not any(folder.iterdir())
        assert result["removed"] == 4
        assert (
            deploy.dataset_status(dataset_id, _items(key, image)) == deploy.RED
        )

    def test_undeploy_media_before_deploy_is_noop(self, deployed):
        """Undeploying a media that was never deployed removes nothing."""
        dataset_id, _, image, _ = deployed
        result = deploy.undeploy_media(dataset_id, _sha(image), "png")
        assert result["removed"] == 0


def _real_image(path, size):
    """Write a real solid-colour PNG at ``path``; return ``(path, sha)``."""
    from PIL import Image

    Image.new("RGB", size, (120, 30, 200)).save(str(path), format="PNG")
    return str(path), store.compute_sha256(str(path))


def _resize_items(key, image, sha, resolution, is_video=False, src_ext="png"):
    """Return a one-media deploy item carrying a resize ``resolution``."""
    return [
        {
            "key": key,
            "path": image,
            "hidden": False,
            "ext": "txt",
            "sha256": sha,
            "file_ext": deploy.deployed_ext(src_ext, is_video, resolution),
            "is_video": is_video,
            "resolution": resolution,
            "missing": False,
            "repeats": 1,
        }
    ]


class TestDeployedExt:
    """The on-disk extension follows the resize setting and media kind."""

    def test_image_becomes_png_when_resizing(self):
        """A resized image deploys as lossless PNG."""
        assert deploy.deployed_ext("jpg", False, 1280) == "png"

    def test_video_keeps_extension(self):
        """A video is never resized and keeps its extension."""
        assert deploy.deployed_ext("mp4", True, 1280) == "mp4"

    def test_no_resolution_keeps_extension(self):
        """With resizing off every media keeps its extension."""
        assert deploy.deployed_ext("jpg", False, 0) == "jpg"


class TestTargetSize:
    """Shortest-side downscaling preserves the aspect ratio, downscale only."""

    def test_downscales_to_shortest_side(self):
        """A landscape image lands its shortest side on the target."""
        assert deploy._target_size(3000, 2000, 1280) == (1920, 1280)

    def test_smaller_image_is_untouched(self):
        """An image already at or below the target is not upscaled."""
        assert deploy._target_size(800, 600, 1280) == (800, 600)

    def test_zero_resolution_is_untouched(self):
        """A zero target leaves the size unchanged."""
        assert deploy._target_size(3000, 2000, 0) == (3000, 2000)


class TestResizeDeploy:
    """Deploying under a resolution re-encodes images to fitted PNGs."""

    def test_downscales_large_image(self, deployed):
        """A large image is written at the shortest-side target as PNG."""
        from PIL import Image

        dataset_id, key, _, root = deployed
        big = root.parent / "big.png"
        image, sha = _real_image(big, (3000, 2000))
        deploy.deploy_dataset(dataset_id, _resize_items(key, image, sha, 1280))
        with Image.open(root / "my set" / f"{sha}.png") as out:
            assert min(out.size) == 1280
            assert out.size == (1920, 1280)

    def test_small_image_kept_at_original_size(self, deployed):
        """A small image is re-encoded to PNG but never upscaled."""
        from PIL import Image

        dataset_id, key, _, root = deployed
        small = root.parent / "small.jpg"
        image, sha = _real_image(small, (640, 480))
        deploy.deploy_dataset(
            dataset_id, _resize_items(key, image, sha, 1280, src_ext="jpg")
        )
        with Image.open(root / "my set" / f"{sha}.png") as out:
            assert out.size == (640, 480)

    def test_redeploy_same_resolution_writes_nothing(self, deployed):
        """An up-to-date resized copy is left untouched on redeploy."""
        dataset_id, key, _, root = deployed
        image, sha = _real_image(root.parent / "big.png", (3000, 2000))
        deploy.deploy_dataset(dataset_id, _resize_items(key, image, sha, 1280))
        result = deploy.deploy_dataset(
            dataset_id, _resize_items(key, image, sha, 1280)
        )
        assert result["written"] == 0

    def test_changing_resolution_rewrites_image(self, deployed):
        """Lowering the target rewrites the image at the new size."""
        from PIL import Image

        dataset_id, key, _, root = deployed
        image, sha = _real_image(root.parent / "big.png", (3000, 2000))
        deploy.deploy_dataset(dataset_id, _resize_items(key, image, sha, 1280))
        result = deploy.deploy_dataset(
            dataset_id, _resize_items(key, image, sha, 768)
        )
        assert result["written"] == 1
        with Image.open(root / "my set" / f"{sha}.png") as out:
            assert min(out.size) == 768


class TestTagsCaptionType:
    """Deploying on the virtual "tags" type writes the media's gallery tags."""

    def _tag(self, media_id, *names):
        """Attach one gallery tag per name (single dedicated category)."""
        existing = {
            row["name"]: row["id"] for row in store.list_tag_categories()
        }
        category = existing.get("z_deploy") or store.create_tag_category(
            "z_deploy"
        )
        for name in names:
            store.add_tag_to_media(
                media_id, store.get_or_create_tag(name, category)
            )

    def test_deploys_comma_joined_tags(self, deployed):
        """The .txt holds the media's tags, comma-separated."""
        dataset_id, key, image, root = deployed
        self._tag(int(key), "blue", "circle")
        deploy.deploy_dataset(dataset_id, _items(key, image, ext="tags"))
        txt = (root / "my set" / f"{_sha(image)}.txt").read_text(
            encoding="utf-8"
        )
        assert txt == "blue, circle"

    def test_triggerword_prefix_applied(self, deployed):
        """The trigger word prefixes the comma-joined tags, as elsewhere."""
        dataset_id, key, image, root = deployed
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        self._tag(int(key), "blue")
        deploy.deploy_dataset(dataset_id, _items(key, image, ext="tags"))
        txt = (root / "my set" / f"{_sha(image)}.txt").read_text(
            encoding="utf-8"
        )
        assert txt == "xbl1. blue"

    def test_editing_tags_makes_orange(self, deployed):
        """Adding a tag after deploy puts the media out of sync."""
        dataset_id, key, image, _ = deployed
        self._tag(int(key), "blue")
        deploy.deploy_dataset(dataset_id, _items(key, image, ext="tags"))
        assert (
            deploy.image_status(
                dataset_id, key, "tags", False, _sha(image), "png"
            )
            == deploy.GREEN
        )
        self._tag(int(key), "circle")
        assert (
            deploy.image_status(
                dataset_id, key, "tags", False, _sha(image), "png"
            )
            == deploy.ORANGE
        )


class TestRepeats:
    """A repeat count above one deploys extra suffixed copies."""

    def test_deploys_extra_copies(self, deployed):
        """Repeats write ``<hash>``, ``<hash>_1``... with their captions."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_dataset(dataset_id, _items(key, image, repeats=3))
        folder = root / "my set"
        for stem in (sha, f"{sha}_1", f"{sha}_2"):
            assert (folder / f"{stem}.png").is_file()
            text = (folder / f"{stem}.txt").read_text(encoding="utf-8")
            assert text == "a cat"
        assert not (folder / f"{sha}_3.png").exists()

    def test_green_with_repeats(self, deployed):
        """Every copy present and up to date reports GREEN."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image, repeats=3))
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png", 3
            )
            == deploy.GREEN
        )

    def test_raising_repeats_makes_orange(self, deployed):
        """Wanting more copies than deployed reports ORANGE."""
        dataset_id, key, image, _ = deployed
        deploy.deploy_dataset(dataset_id, _items(key, image))
        assert (
            deploy.image_status(
                dataset_id, key, "txt", False, _sha(image), "png", 2
            )
            == deploy.ORANGE
        )

    def test_lowering_repeats_prunes_extras(self, deployed):
        """A leftover extra copy is ORANGE, then pruned on redeploy."""
        dataset_id, key, image, root = deployed
        sha = _sha(image)
        deploy.deploy_dataset(dataset_id, _items(key, image, repeats=3))
        assert (
            deploy.image_status(dataset_id, key, "txt", False, sha, "png", 1)
            == deploy.ORANGE
        )
        deploy.deploy_dataset(dataset_id, _items(key, image))
        folder = root / "my set"
        assert not (folder / f"{sha}_1.png").exists()
        assert not (folder / f"{sha}_2.txt").exists()
        assert (
            deploy.image_status(dataset_id, key, "txt", False, sha, "png", 1)
            == deploy.GREEN
        )

    def test_repeats_persisted_per_dataset_media(self, deployed):
        """The stepper value round-trips through the store (clamped >= 1)."""
        dataset_id, key, _, _ = deployed
        media_id = int(key)
        store.set_media_repeats(dataset_id, media_id, 4)
        assert store.get_media_repeats(dataset_id, media_id) == 4
        assert store.media_in_dataset(dataset_id)[0]["repeats"] == 4
        store.set_media_repeats(dataset_id, media_id, 0)
        assert store.get_media_repeats(dataset_id, media_id) == 1
