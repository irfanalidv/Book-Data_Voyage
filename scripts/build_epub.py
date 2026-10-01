"""Build a store-ready EPUB from manuscript/ for Amazon KDP, Google Play, Apple Books, Kobo.

Usage:
    make manuscript                      # refresh manuscript/ from book/ first
    python scripts/build_epub.py         # writes dist/data-voyage.epub
    python scripts/build_epub.py --cover assets/cover-1600x2400.jpg

Leanpub builds its own PDF and EPUB from manuscript/Book.txt, so this script is
only for the other storefronts. It reads the same files in the same order, drops
the Leanpub section markers, and shrinks figures to at most 1600 px wide. The
print-resolution originals stay untouched; smaller figures keep the EPUB small,
which matters on Amazon because the 70% royalty tier charges delivery per MB.

Requires pandoc (https://pandoc.org). Validate the result with EPUBCheck:
    java -jar epubcheck.jar dist/data-voyage.epub
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "manuscript"
DIST = ROOT / "dist"
MAX_WIDTH = 1600
SECTION_MARKERS = {"{frontmatter}", "{mainmatter}", "{backmatter}"}

METADATA = """---
title: "Data Voyage"
subtitle: "Building Real AI Systems from Data to Deployment"
author: "Irfan Ali"
lang: en-IN
date: "2026-10-01"
rights: "Copyright © 2026 Irfan Ali. All rights reserved."
publisher: "DataCortex IQ"
subject: ["Artificial intelligence", "Machine learning", "Software engineering"]
description: >-
  Production AI engineering taught through one running project, TalentLens,
  from raw job postings to a deployed, tested, and packaged API.
identifier:
  - scheme: UUID
    text: urn:uuid:6f1c2b0e-9d7a-4c3e-8a51-2f0b7c9e4d10
---
"""


def combined_markdown() -> str:
    """Concatenate manuscript files in Book.txt order, without Leanpub markers."""
    parts = []
    for name in (MANUSCRIPT / "Book.txt").read_text(encoding="utf-8").split():
        lines = (MANUSCRIPT / name).read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        part_heading = False
        for ln in lines:
            if ln.strip() in SECTION_MARKERS:
                continue
            if ln.strip() == "{class: part}":  # Markua; pandoc wants a heading attribute
                part_heading = True
                continue
            if part_heading and ln.startswith("# "):
                ln = f"{ln} {{.part}}"
                part_heading = False
            kept.append(ln)
        parts.append("\n".join(kept))
    return "\n\n".join(parts) + "\n"


def shrink_images(src: Path, dest: Path) -> None:
    """Copy images, downscaling anything wider than MAX_WIDTH."""
    dest.mkdir(parents=True)
    for img_path in src.iterdir():
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(img_path) as im:
            if im.width > MAX_WIDTH:
                im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
            if img_path.suffix.lower() == ".png":
                # Charts use a few flat colours; a 256-colour palette is visually
                # identical and several times smaller.
                im = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                im.save(dest / img_path.name, optimize=True)
            else:
                im.convert("RGB").save(dest / img_path.name, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cover", type=Path, default=ROOT / "assets" / "cover-kdp.jpg")
    ap.add_argument("--out", type=Path, default=DIST / "data-voyage.epub")
    args = ap.parse_args()

    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found: install it from https://pandoc.org")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shrink_images(MANUSCRIPT / "images", work / "images")
        (work / "metadata.yaml").write_text(METADATA, encoding="utf-8")
        (work / "book.md").write_text(combined_markdown(), encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "pandoc",
                "metadata.yaml",
                "book.md",
                "-o",
                str(args.out.resolve()),
                "--toc",
                "--toc-depth=2",
                "--split-level=1",
                f"--epub-cover-image={args.cover.resolve()}",
            ],
            cwd=work,
            check=True,
        )
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
