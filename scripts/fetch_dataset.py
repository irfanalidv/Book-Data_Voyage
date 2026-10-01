"""Fetch the full TalentLens dataset from the book's GitHub release.

This script is what ``make fetch-dataset`` invokes. It:

1. Reads ``talentlens/data_versions.json`` to find the current dataset
   version and its release URL.
2. Downloads the gzipped CSV to ``data/clean/jobs_clean.csv.gz``.
3. Verifies the SHA256 hash against the manifest.
4. Decompresses to ``data/clean/jobs_clean.large.csv``.
5. Leaves the small bundled ``data/clean/jobs_clean.csv`` untouched so
   ``make test`` still has a deterministic, network-free fallback.

Run from the repo root:

    python scripts/fetch_dataset.py
    # or:
    make fetch-dataset

Pass ``--version vX.Y.Z-dataset-N`` to fetch a specific historical
version. Pass ``--force`` to re-download even if the file already exists
with a matching hash.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_RELEASE_URL = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/releases/download/(?P<tag>[^/]+)/(?P<filename>[^/]+)$"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "talentlens" / "data_versions.json"
TARGET_DIR = REPO_ROOT / "data" / "clean"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST_PATH}. Run from the repo root.")
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _github_token() -> str | None:
    """Token for private-repo release assets (GITHUB_TOKEN or GH_TOKEN)."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _auth_headers() -> dict[str, str]:
    if token := _github_token():
        return {"Authorization": f"Bearer {token}"}
    return {}


def _stream_to_file(req: urllib.request.Request, dest: Path) -> None:
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _release_asset_api_url(browser_url: str) -> str | None:
    """Resolve a releases/download URL to the GitHub API asset endpoint."""
    match = _RELEASE_URL.match(browser_url)
    if not match or not _github_token():
        return None
    owner, repo, tag, filename = (
        match["owner"],
        match["repo"],
        match["tag"],
        match["filename"],
    )
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(
        api,
        headers={**_auth_headers(), "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        release = json.load(resp)
    for asset in release.get("assets", []):
        if asset.get("name") == filename:
            return asset["url"]
    return None


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    print(f"  →           {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = _auth_headers()
    req = urllib.request.Request(url, headers=headers)
    try:
        _stream_to_file(req, dest)
        return
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
        if not headers:
            raise SystemExit(
                "Release download returned 404. If the repository is private, set "
                "GITHUB_TOKEN or GH_TOKEN with repo scope and retry."
            ) from err
        not_found = err
    api_url = _release_asset_api_url(url)
    if not api_url:
        raise SystemExit(
            "Release download returned 404. For a private repository, set "
            "GITHUB_TOKEN or GH_TOKEN with repo scope and retry."
        ) from not_found
    print("  (private repo — fetching via GitHub API asset URL)")
    api_req = urllib.request.Request(
        api_url,
        headers={
            **_auth_headers(),
            "Accept": "application/octet-stream",
        },
    )
    _stream_to_file(api_req, dest)


def _decompress(gz_path: Path, csv_path: Path) -> None:
    print(f"  Decompressing → {csv_path}")
    with gzip.open(gz_path, "rb") as gz, csv_path.open("wb") as out:
        shutil.copyfileobj(gz, out)


def _ensure_publishable(entry: dict, version: str) -> None:
    sha = entry.get("sha256", "")
    if not sha or sha == "TBD":
        print(
            "No published dataset is available yet. The book ships with a small "
            "bundled dataset at data/clean/jobs_clean.csv (576 rows) which is "
            "enough to run every chapter. A larger production dataset may be "
            "published as a future GitHub release; track:\n\n"
            "  https://github.com/irfanalidv/Book-Data_Voyage/releases\n\n"
            'See "The dataset" section of the project README for context.\n'
        )
        raise SystemExit(0)


def fetch(version: str | None = None, force: bool = False) -> Path:
    manifest = _load_manifest()
    chosen = version or manifest["current"]
    if chosen not in manifest["versions"]:
        available = ", ".join(sorted(manifest["versions"]))
        raise SystemExit(f"Dataset version {chosen!r} not in manifest. Available: {available}")

    entry = manifest["versions"][chosen]
    _ensure_publishable(entry, chosen)

    gz_path = TARGET_DIR / entry["filename"]
    csv_path = TARGET_DIR / "jobs_clean.large.csv"

    if gz_path.exists() and not force:
        actual = _sha256(gz_path)
        if actual == entry["sha256"]:
            print(f"  Cached {gz_path.name} already matches manifest SHA256.")
            if not csv_path.exists():
                _decompress(gz_path, csv_path)
            print(f"  Ready: {csv_path}")
            return csv_path
        print(
            f"  Cached file SHA mismatch (have {actual[:12]}, "
            f"want {entry['sha256'][:12]}). Re-downloading."
        )

    _download(entry["url"], gz_path)
    actual = _sha256(gz_path)
    if actual != entry["sha256"]:
        gz_path.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA256 mismatch after download: got {actual}, "
            f"expected {entry['sha256']}. "
            "File deleted. Check your network or open an issue if the "
            "manifest is out of date."
        )
    _decompress(gz_path, csv_path)
    print(f"  Ready: {csv_path}")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="Dataset version (default: current).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cache matches.",
    )
    args = parser.parse_args()
    try:
        fetch(version=args.version, force=args.force)
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            return 0
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
