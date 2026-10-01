"""Publish a new TalentLens dataset version (author-only).

Runs Chapter 5's live collector and Chapter 6's cleaner against real
APIs, produces the canonical jobs_clean.csv, gzips it, hashes it, and
updates ``talentlens/data_versions.json``. The actual GitHub release
upload is intentionally manual - run ``gh release create`` against the
output file once you've reviewed the manifest diff.

Required environment for a real run:

    ADZUNA_APP_ID, ADZUNA_API_KEY   - see ch05/README.md

Without those, this script will refuse to run (will not silently
publish demo data as if it were production data).

Usage:

    python scripts/publish_dataset.py --version v1.0.0-dataset-2 \\
        --description "Refresh: October 2026 collection cycle"
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "talentlens" / "data_versions.json"
CLEAN_CSV = REPO_ROOT / "data" / "clean" / "jobs_clean.csv"
RELEASE_GZ = REPO_ROOT / "data" / "clean" / "jobs_clean.csv.gz"

# The small bundled jobs_clean.csv is the reproducibility anchor.
# publish_dataset.py snapshots it, runs ch05+ch06 (which overwrites
# it), then restores the snapshot. If this constant ever fails to
# match the on-disk file, decide whether you meant to change the
# anchor - do NOT just bump the number.
EXPECTED_SMALL_CSV_BYTES = 164_536


def _require_live_keys() -> None:
    missing = [k for k in ("ADZUNA_APP_ID", "ADZUNA_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"Refusing to publish: missing env vars {missing}. "
            "This script must run against live collectors so the released "
            "dataset reflects real data, not demo data."
        )


def _run_collection() -> None:
    print("  Running ch05 (live) ...")
    subprocess.run(
        [sys.executable, "book/ch05/ch05_data_collection.py", "--live"],
        check=True,
        cwd=REPO_ROOT,
    )
    print("  Running ch06 ...")
    subprocess.run(
        [sys.executable, "book/ch06/ch06_data_cleaning_preprocessing.py"],
        check=True,
        cwd=REPO_ROOT,
    )


def _gzip_and_hash(src: Path, dst: Path) -> tuple[str, int, int]:
    print(f"  Gzipping {src} → {dst}")
    with src.open("rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
    h = hashlib.sha256()
    with dst.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), src.stat().st_size, dst.stat().st_size


def _git_head() -> str:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    return out.decode().strip()


def _snapshot_bundled_csv() -> bytes | None:
    """Return a byte copy of the small bundled CSV if it exists."""
    if CLEAN_CSV.exists():
        return CLEAN_CSV.read_bytes()
    return None


def _restore_bundled_csv(snapshot: bytes | None) -> None:
    """Put back the reproducibility anchor after ch06 overwrote it."""
    if snapshot is None:
        return
    CLEAN_CSV.write_bytes(snapshot)
    if CLEAN_CSV.read_bytes() != snapshot:
        raise SystemExit(
            f"Failed to restore bundled {CLEAN_CSV.name} — "
            "do not commit until the small file is intact."
        )
    print(f"  Restored bundled anchor: {CLEAN_CSV} ({len(snapshot):,} bytes)")


def main() -> int:
    actual = CLEAN_CSV.stat().st_size
    if actual != EXPECTED_SMALL_CSV_BYTES:
        raise SystemExit(
            f"Bundled jobs_clean.csv is {actual} bytes; expected "
            f"{EXPECTED_SMALL_CSV_BYTES}. The small file is the "
            "reproducibility anchor — if you intentionally changed it, "
            "update EXPECTED_SMALL_CSV_BYTES in this script in the same "
            "commit. Refusing to publish until this is reconciled."
        )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        required=True,
        help="Version tag, e.g. v1.0.0-dataset-2.",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="One-line description for the manifest.",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip ch05/ch06 — use existing jobs_clean.csv.",
    )
    args = parser.parse_args()

    bundled_snapshot = _snapshot_bundled_csv()
    if bundled_snapshot is not None:
        print(f"  Bundled anchor snapshot: {CLEAN_CSV} ({len(bundled_snapshot):,} bytes)")

    if not args.skip_collect:
        _require_live_keys()
        _run_collection()

    if not CLEAN_CSV.exists():
        raise SystemExit(f"Expected {CLEAN_CSV} after pipeline; not found.")

    df = pd.read_csv(CLEAN_CSV)
    sha, uncomp, comp = _gzip_and_hash(CLEAN_CSV, RELEASE_GZ)
    _restore_bundled_csv(bundled_snapshot)

    manifest = json.loads(MANIFEST_PATH.read_text())
    if args.version in manifest["versions"]:
        raise SystemExit(
            f"Version {args.version} already in manifest. "
            "Pick a new version tag — manifest entries are immutable."
        )

    manifest["versions"][args.version] = {
        "description": args.description,
        "rows": len(df),
        "schema_commit": _git_head(),
        "url": (
            "https://github.com/irfanalidv/Book-Data_Voyage/releases/"
            f"download/{args.version}/jobs_clean.csv.gz"
        ),
        "sha256": sha,
        "filename": "jobs_clean.csv.gz",
        "compressed_bytes": comp,
        "uncompressed_bytes": uncomp,
    }
    manifest["current"] = args.version
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print("=" * 60)
    print(f"  Dataset {args.version} published (manifest updated).")
    print("=" * 60)
    print(f"  Rows:         {len(df):,}")
    print(f"  Uncompressed: {uncomp / 1e6:.1f} MB")
    print(f"  Compressed:   {comp / 1e6:.1f} MB")
    print(f"  SHA256:       {sha}")
    print(f"  Release file: {RELEASE_GZ}")
    print()
    print("  Next step (manual):")
    print(f"    gh release create {args.version} {RELEASE_GZ} \\")
    print(f'        --title "TalentLens dataset {args.version}" \\')
    print(f'        --notes "{args.description}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
