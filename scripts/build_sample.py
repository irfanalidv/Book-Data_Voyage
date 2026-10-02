"""Build the free Leanpub sample from the full PDF: front matter, contents, Chapters 1-2.

Usage:
    make pdf                          # dist/data-voyage.pdf first
    python scripts/build_sample.py    # writes dist/data-voyage-sample.pdf

The sample keeps every page up to the start of Chapter 3, its bookmarks, and
adds a closing page that links to the full book. Requires PyMuPDF
(make install-dev).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "dist" / "data-voyage.pdf"
SAMPLE = ROOT / "dist" / "data-voyage-sample.pdf"
STOP_AT = "Chapter 3:"  # the sample ends on the page before this chapter

CLOSING_PAGE = """<div style="font-family: serif; text-align: center; color: #1b1b1b;">
<p style="font-size: 22pt; font-weight: bold; margin: 0;">End of the free sample</p>
<p style="margin: 6pt auto 18pt; color: #e2262c; font-size: 14pt;">———</p>
<p style="font-size: 11.5pt; line-height: 1.5;">The complete book has 24 chapters in seven
parts: data collection and cleaning, classical machine learning, NLP and time series, vector
search, LLM generation, agents, FastAPI, Docker, CI/CD, packaging, four case studies from
shipped products, and a career playbook for the Indian AI and ML market.</p>
<p style="font-size: 12.5pt; font-weight: bold; margin-top: 18pt;">Get the full PDF and EPUB at
<a href="https://leanpub.com/datavoyage" style="color: #1a5fb4;">leanpub.com/datavoyage</a></p>
<p style="font-size: 10.5pt; color: #555; margin-top: 10pt;">Companion code:
<a href="https://github.com/irfanalidv/Book-Data_Voyage"
style="color: #1a5fb4;">github.com/irfanalidv/Book-Data_Voyage</a></p>
</div>"""


def main() -> None:
    if not FULL.exists():
        raise SystemExit(f"{FULL.relative_to(ROOT)} not found: run make pdf first")
    full = pymupdf.open(FULL)
    toc = full.get_toc()
    starts = [page for level, title, page in toc if level == 1 and title.startswith(STOP_AT)]
    if not starts:
        raise SystemExit(f"No bookmark starting with {STOP_AT!r} in the full PDF")
    last = starts[0] - 1  # 1-based number of the last page to keep

    sample = pymupdf.open()
    sample.insert_pdf(full, from_page=0, to_page=last - 1)
    width, height = full[0].rect.width, full[0].rect.height
    page = sample.new_page(width=width, height=height)
    page.insert_htmlbox(pymupdf.Rect(60, 2.4 * 72, width - 60, height - 72), CLOSING_PAGE)

    kept = [entry for entry in toc if entry[2] <= last]
    sample.set_toc(kept + [[1, "End of the free sample", sample.page_count]])
    sample.set_metadata({**full.metadata, "title": "Data Voyage (free sample)"})
    sample.save(SAMPLE, garbage=3, deflate=True)
    size_mb = SAMPLE.stat().st_size / 1e6
    print(f"Wrote {SAMPLE.relative_to(ROOT)} ({sample.page_count} pages, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
