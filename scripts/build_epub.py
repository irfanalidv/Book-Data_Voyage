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

# E-reader stylesheet. pandoc's default scrolls wide code blocks, and most
# e-readers (Kindle included) cannot scroll inside a block, so long lines would
# be cut off. Body fonts are left to the reader's own settings.
EPUB_CSS = """
body { margin: 0 4%; line-height: 1.5; text-align: left; }
h1 { font-size: 1.6em; line-height: 1.25; margin: 0.6em 0 0.8em; page-break-before: always; }
h1.part { text-align: center; font-size: 1.9em; margin-top: 30%; }
h1.part + p { text-align: center; font-style: italic; }
h2 { font-size: 1.3em; margin: 1.4em 0 0.5em; page-break-after: avoid; }
h3 { font-size: 1.1em; margin: 1.2em 0 0.4em; page-break-after: avoid; }
h4 { font-size: 1em; margin: 1em 0 0.3em; page-break-after: avoid; }
p { margin: 0 0 0.7em; }
a { color: inherit; }
code { font-family: Menlo, Monaco, Consolas, "Courier New", monospace; font-size: 0.85em; }
pre { font-size: 0.78em; line-height: 1.35; white-space: pre-wrap; word-wrap: break-word;
      overflow-wrap: anywhere; border: 1px solid #ccc; padding: 0.5em; margin: 0.8em 0; }
pre code { font-size: 1em; white-space: pre-wrap; }
/* pandoc's highlighting styles set white-space: pre on these exact selectors
   and only switch wrapping on for print, so override them for e-readers. */
pre > code.sourceCode { white-space: pre-wrap !important; }
div.sourceCode, .sourceCode { overflow: visible !important; }
pre > code.sourceCode > span { text-indent: -2em; padding-left: 2em; }
blockquote { margin: 1em 0; padding: 0.3em 0.8em; border-left: 3px solid #e2262c; }
table { border-collapse: collapse; width: 100%; font-size: 0.8em; margin: 0.8em 0; }
th, td { border-bottom: 1px solid #ccc; padding: 0.25em 0.4em; text-align: left; vertical-align: top; }
img { max-width: 100%; height: auto; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 0.85em; font-style: italic; }
ul, ol { margin: 0 0 0.7em; padding-left: 1.3em; }
"""

METADATA = """---
title: "Data Voyage"
subtitle: "Building Real AI Systems from Data to Deployment"
author: "Irfan Ali"
lang: en-IN
date: "2026-10-02"
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


def shrink_images(src: Path, dest: Path, max_width: int = MAX_WIDTH) -> None:
    """Copy images, downscaling anything wider than *max_width*."""
    dest.mkdir(parents=True)
    for img_path in src.iterdir():
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(img_path) as im:
            if im.width > max_width:
                im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
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
        (work / "epub.css").write_text(EPUB_CSS, encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "pandoc",
                # GitHub and Leanpub allow a list right after a line of text;
                # plain pandoc Markdown needs a blank line, so turn that on.
                "--from=markdown+lists_without_preceding_blankline",
                "metadata.yaml",
                "book.md",
                "-o",
                str(args.out.resolve()),
                "--toc",
                "--toc-depth=2",
                "--split-level=1",
                f"--epub-cover-image={args.cover.resolve()}",
                "--css=epub.css",
            ],
            cwd=work,
            check=True,
        )
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
