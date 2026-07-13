"""Feature tests: end-to-end dataset/media/caption scenarios.

Only the scenarios not already covered piecewise by the unit suites live
here (see ``test_sqlite_store`` for CRUD, revisions, trigger words and
composition). Each test reads back the database directly (via the
``query`` / ``count`` fixtures) so the assertions describe the real stored
state, not just the return values of the repository functions.
"""

from src import sqlite_store as store


def test_multiple_caption_types_per_media(count, make_media_file):
    """A media can hold captions of several types independently."""
    dataset_id = store.create_dataset("d")
    image = make_media_file("c.png")
    media_id = store.ingest_file(image)[0]
    store.add_media_to_dataset(dataset_id, media_id)
    txt = store.get_or_create_caption_type("txt")
    booru = store.get_or_create_caption_type("booru")

    store.save_caption(dataset_id, media_id, txt, "a natural caption")
    store.save_caption(dataset_id, media_id, booru, "1girl, solo")

    assert count("caption") == 2
    assert store.read_caption(dataset_id, media_id, txt) == "a natural caption"
    assert store.read_caption(dataset_id, media_id, booru) == "1girl, solo"


def test_deleting_dataset_cascades_links(count, make_media_file):
    """Deleting a dataset removes its links but keeps shared media."""
    dataset_id = store.create_dataset("d")
    image = make_media_file("d.png")
    media_id = store.ingest_file(image)[0]
    store.add_media_to_dataset(dataset_id, media_id)
    txt = store.get_or_create_caption_type("txt")
    store.save_caption(dataset_id, media_id, txt, "cap")
    store.add_triggerword_to_dataset(dataset_id, "xbl1")

    store.delete_dataset(dataset_id)

    assert count("dataset") == 0
    assert count("dataset_media") == 0
    assert count("dataset_triggerword") == 0
    # Media and the caption history survive the dataset deletion.
    assert count("media") == 1
    assert count("caption_revision") == 1
