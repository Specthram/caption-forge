"""Content hashing for media de-duplication.

A media's identity is the SHA-256 of its file contents: the same bytes reached
through several files on disk map to one media (see :mod:`src.sqlite_store`).
This module owns the hashing only — media now reference their files in place
(there is no managed content-addressable copy anymore).
"""

import hashlib

# Number of bytes read at a time when hashing a media file.
_HASH_CHUNK_SIZE = 1 << 20


def compute_sha256(path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
