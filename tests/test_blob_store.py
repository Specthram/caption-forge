"""Tests for :mod:`src.blob_store` (content hashing)."""

import hashlib

from src import blob_store


def test_compute_sha256_matches_hashlib(tmp_path):
    """The digest equals a direct hashlib computation of the bytes."""
    path = tmp_path / "f.bin"
    path.write_bytes(b"some-image-bytes")
    assert (
        blob_store.compute_sha256(path)
        == hashlib.sha256(b"some-image-bytes").hexdigest()
    )


def test_same_bytes_same_hash(tmp_path):
    """Two files with identical content hash to the same digest."""
    first = tmp_path / "a.png"
    first.write_bytes(b"xy")
    second = tmp_path / "b.png"
    second.write_bytes(b"xy")
    assert blob_store.compute_sha256(first) == blob_store.compute_sha256(
        second
    )
