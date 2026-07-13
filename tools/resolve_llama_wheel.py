"""Resolve a prebuilt llama-cpp-python wheel from the JamePeng releases.

Given a CUDA tag (e.g. ``cu128``) and an optional version, query the GitHub
releases API and print the best matching Windows / CPython 3.12 wheel as a
single space-separated line of three tokens::

    <download_url> <sha256> <filename>

``<sha256>`` is ``-`` when the API exposes no digest (so the token positions
stay fixed for the batch parser). ``<filename>`` is the decoded wheel name to
save as (with a literal ``+``, which pip requires). The installer
(``install.bat``) captures this output to download and verify the wheel. Exits
non-zero (with a message on stderr) when nothing matches.

Usage:
======
    py -3.12 tools/resolve_llama_wheel.py cu128 [version]

    cu128:   the CUDA tag to match (cu124 / cu126 / cu128 ...)
    version: optional, e.g. 0.3.40; empty selects the latest available
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

RELEASES_URL = (
    "https://api.github.com/repos/JamePeng/llama-cpp-python"
    "/releases?per_page=100"
)

# This script targets the project's pinned interpreter and platform.
PYTHON_TAG = "cp312"
PLATFORM_TAG = "win_amd64"


def _fetch_releases():
    """Return the parsed list of releases from the GitHub API."""
    request = urllib.request.Request(
        RELEASES_URL, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _is_candidate(tag: str, cuda: str, version: str) -> bool:
    """Return whether a release tag matches the wanted CUDA and version.

    Parameters
    ----------
    tag : str
        The release ``tag_name`` (e.g. ``v0.3.40-cu128-win-20260608``).
    cuda : str
        The CUDA tag to match, such as ``cu128``.
    version : str
        A specific version to require (e.g. ``0.3.40``), or ``""`` for any.
    """
    tag_lower = tag.lower()
    if f"-{cuda}-" not in tag_lower or "-win-" not in tag_lower:
        return False
    if version and not tag_lower.startswith(f"v{version.lower()}-"):
        return False
    return True


def _pick_asset(assets: list) -> dict | None:
    """Return the cp312 win_amd64 wheel asset from a release, or None."""
    for asset in assets:
        name = asset.get("name", "")
        if (
            name.endswith(".whl")
            and PYTHON_TAG in name
            and PLATFORM_TAG in name
        ):
            return asset
    return None


def resolve(cuda: str, version: str) -> tuple[str, str, str] | None:
    """Return ``(download_url, sha256, filename)`` for the best wheel.

    Releases are considered newest first (by publish date). Full builds are
    preferred over "Basic" variants. ``sha256`` is an empty string when the
    API exposes no digest for the asset.
    """
    releases = _fetch_releases()
    releases.sort(key=lambda rel: rel.get("published_at", ""), reverse=True)

    candidates = [
        rel
        for rel in releases
        if _is_candidate(rel.get("tag_name", ""), cuda, version)
    ]
    # Prefer full builds; fall back to "Basic" variants only if needed.
    full = [
        rel
        for rel in candidates
        if "basic" not in rel.get("tag_name", "").lower()
    ]
    for rel in full or candidates:
        asset = _pick_asset(rel.get("assets", []))
        if asset is not None:
            digest = (asset.get("digest") or "").removeprefix("sha256:")
            return asset["browser_download_url"], digest, asset["name"]
    return None


def main() -> int:
    """Parse arguments, resolve the wheel and print the result line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda", help="CUDA tag to match, e.g. cu128.")
    parser.add_argument(
        "version", nargs="?", default="", help="Version, e.g. 0.3.40."
    )
    args = parser.parse_args()

    try:
        result = resolve(args.cuda, args.version)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(f"Failed to query GitHub releases: {exc}", file=sys.stderr)
        return 2

    if result is None:
        print(
            f"No {PYTHON_TAG} {PLATFORM_TAG} wheel found for {args.cuda}"
            + (f" {args.version}" if args.version else ""),
            file=sys.stderr,
        )
        return 1

    url, sha256, filename = result
    print(f"{url} {sha256 or '-'} {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
