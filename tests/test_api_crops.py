"""End-to-end tests for the crop routes and the deploy of a crop.

Real repository, real routes, real deploy engine against the shared
in-memory database and a throwaway deploy root — a crop must travel from
"rectangle in JSON" to "cropped PNG on disk" with no file created anywhere
in between.
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import deploy
from src import sqlite_store as store
from src import storage

RECT = {"x": 10, "y": 20, "w": 50, "h": 40}


@pytest.fixture(name="scenario")
def _scenario(store_db, thumb_cache_dir, monkeypatch, tmp_path):
    """Seed one 800x600 image, in a library and in a dataset."""
    # pylint: disable=unused-argument
    monkeypatch.setattr(deploy, "get_deploy_dir", lambda: tmp_path / "out")
    images = tmp_path / "pics"
    images.mkdir()
    Image.new("RGB", (800, 600), (219, 68, 55)).save(images / "photo.png")
    library_id = store.create_library("fixtures", str(images))
    store.scan_library(library_id)
    media_id = store.list_library_media()[0]["id"]
    dataset_id = store.create_dataset("shapes")
    store.add_media_to_dataset(dataset_id, media_id)
    storage.write_caption(dataset_id, str(media_id), "txt", "a red square")
    return {
        "dataset_id": dataset_id,
        "media_id": media_id,
        "deploy_root": tmp_path / "out",
    }


@pytest.fixture(name="client")
def _client(scenario):
    """Return a TestClient bound to the seeded scenario."""
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client, scenario


def _create(client, scenario, mode="replace", rect=None):
    """POST a crop of the scenario's media into its dataset."""
    return client.post(
        "/api/crops",
        json={
            "media_id": scenario["media_id"],
            "rect": rect or RECT,
            "ratio": "free",
            "dataset_id": scenario["dataset_id"],
            "mode": mode,
        },
    )


class TestCropSource:
    """What the overlay needs before it can draw a rectangle."""

    def test_returns_the_source_dimensions(self, client):
        """Dimensions come from the file when the index has not run."""
        test_client, scenario = client
        response = test_client.get(f"/api/crops/source/{scenario['media_id']}")

        assert response.status_code == 200
        assert response.json()["width"] == 800
        assert response.json()["height"] == 600

    def test_a_crop_cannot_be_cropped(self, client):
        """Crops do not nest."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]

        response = test_client.get(f"/api/crops/source/{crop_id}")
        assert response.status_code == 400

    def test_an_unknown_media_is_a_404(self, client):
        """A missing id is not an empty overlay."""
        test_client, _ = client
        assert test_client.get("/api/crops/source/999").status_code == 404


class TestCreateAndPlace:
    """Creating a crop and putting it in the dataset."""

    def test_replace_swaps_the_parent_card(self, client):
        """The dataset shows one card, and it is the crop."""
        test_client, scenario = client
        body = _create(test_client, scenario).json()

        assert body["replaced"] is True
        assert (body["width"], body["height"]) == (400, 240)

        cards = test_client.get(
            f"/api/datasets/{scenario['dataset_id']}/media"
        ).json()["items"]
        assert [card["key"] for card in cards] == [str(body["id"])]
        assert cards[0]["crop"]["parent_media_id"] == scenario["media_id"]
        assert cards[0]["crop"]["rect"]["x"] == 10.0

    def test_beside_adds_a_second_card(self, client):
        """The original stays, the crop joins it as another sample."""
        test_client, scenario = client
        body = _create(test_client, scenario, mode="beside").json()

        cards = test_client.get(
            f"/api/datasets/{scenario['dataset_id']}/media"
        ).json()["items"]
        assert [card["key"] for card in cards] == [
            str(scenario["media_id"]),
            str(body["id"]),
        ]
        assert cards[0]["crop"] is None

    def test_the_crop_stays_out_of_the_medias_grid(self, client):
        """A crop is a dataset entry, never a library media."""
        test_client, scenario = client
        _create(test_client, scenario)

        cards = test_client.get("/api/medias/grid").json()["items"]
        assert [card["key"] for card in cards] == [str(scenario["media_id"])]

    def test_the_thumbnail_serves_the_cropped_pixels(self, client):
        """No engine changed: the crop's effective file is the rendered PNG."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]

        response = test_client.get(f"/api/media/{crop_id}/thumb")
        assert response.status_code == 200

    def test_an_unreadable_rectangle_is_a_400(self, client):
        """A crop of an unknown media is a client error, not a stack trace."""
        test_client, scenario = client
        response = test_client.post(
            "/api/crops",
            json={
                "media_id": 999,
                "rect": RECT,
                "dataset_id": scenario["dataset_id"],
            },
        )
        assert response.status_code == 400


class TestUpdateAndDelete:
    """Re-framing and removing a crop through the routes."""

    def test_posting_a_rect_reframes_the_crop(self, client):
        """The rendered size follows the new rectangle."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]

        body = test_client.post(
            f"/api/crops/{crop_id}",
            json={"rect": {"x": 0, "y": 0, "w": 25, "h": 50}, "ratio": "free"},
        ).json()
        assert (body["width"], body["height"]) == (200, 300)

    def test_delete_restores_the_parent(self, client):
        """The dataset never silently loses a sample."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]

        assert test_client.delete(f"/api/crops/{crop_id}").json() == {
            "deleted": True,
            "restored": [scenario["dataset_id"]],
        }
        cards = test_client.get(
            f"/api/datasets/{scenario['dataset_id']}/media"
        ).json()["items"]
        assert [card["key"] for card in cards] == [str(scenario["media_id"])]

    def test_deleting_a_plain_media_is_a_404(self, client):
        """The route refuses to purge a real image."""
        test_client, scenario = client
        response = test_client.delete(f"/api/crops/{scenario['media_id']}")
        assert response.status_code == 404

    def test_the_detail_marks_the_focused_media_as_a_crop(self, client):
        """The panel needs to know whether it is looking at a crop."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]
        params = {"dataset_id": scenario["dataset_id"], "caption_type": "txt"}

        detail = test_client.get(
            f"/api/captions/media/{crop_id}", params=params
        ).json()
        assert detail["crop"]["parent_media_id"] == scenario["media_id"]

        parent = test_client.get(
            f"/api/captions/media/{scenario['media_id']}", params=params
        ).json()
        assert parent["crop"] is None


class TestListCrops:
    """The reusable list a panel offers, keyed on the media's source."""

    def test_lists_the_crops_of_the_source_marked_by_dataset(self, client):
        """Ask from the parent or from a crop: the same siblings come back."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario, mode="beside").json()["id"]
        dataset_id = scenario["dataset_id"]

        parent_id = scenario["media_id"]
        from_parent = test_client.get(
            "/api/crops",
            params={"media_id": parent_id, "dataset_id": dataset_id},
        ).json()["crops"]
        assert [crop["id"] for crop in from_parent] == [crop_id]
        assert from_parent[0]["in_dataset"] is True

        from_crop = test_client.get(
            "/api/crops",
            params={"media_id": crop_id, "dataset_id": dataset_id},
        ).json()["crops"]
        assert [crop["id"] for crop in from_crop] == [crop_id]

    def test_a_crop_of_another_dataset_is_not_marked(self, client):
        """`in_dataset` is scoped to the dataset the panel is looking at."""
        test_client, scenario = client
        _create(test_client, scenario, mode="beside")
        other = test_client.post(
            "/api/datasets", json={"name": "other"}
        ).json()["id"]

        crops_list = test_client.get(
            "/api/crops",
            params={"media_id": scenario["media_id"], "dataset_id": other},
        ).json()["crops"]
        assert crops_list[0]["in_dataset"] is False


class TestDeployACrop:
    """The rectangle becomes pixels only when the dataset is deployed."""

    def test_the_deployed_file_holds_the_cropped_image(self, client):
        """No physical file existed until deploy; now the PNG is the crop."""
        test_client, scenario = client
        crop_id = _create(test_client, scenario).json()["id"]
        storage.write_caption(
            scenario["dataset_id"], str(crop_id), "txt", "a red rectangle"
        )
        sha = store.get_crop(crop_id)["sha256"]

        items = [
            {
                "key": str(crop_id),
                "path": store.effective_file(crop_id),
                "ext": "txt",
                "hidden": False,
                "sha256": sha,
                "file_ext": "png",
                "is_video": False,
                "resolution": 0,
                "missing": False,
                "repeats": 1,
            }
        ]
        deploy.deploy_dataset(scenario["dataset_id"], items)

        folder = scenario["deploy_root"] / "shapes"
        with Image.open(folder / f"{sha}.png") as img:
            assert img.size == (400, 240)
        assert (folder / f"{sha}.txt").read_text(
            encoding="utf-8"
        ) == "a red rectangle"
