"""Build a typeset PDF of the book from manuscript/, for Leanpub upload mode and direct sales.

Usage:
    make manuscript                  # refresh manuscript/ from book/ first
    python scripts/build_pdf.py      # writes dist/data-voyage.pdf

pandoc renders the manuscript to one styled HTML file, and headless Google
Chrome prints it to PDF: a 7 x 9.25 in trim, the cover as page one, a linked
table of contents, each chapter on a new page, and page numbers. Figures are
reused from build_epub.py (downscaled and palette-reduced) to keep the file
small. Requires pandoc and Google Chrome (or set CHROME to another Chromium).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from build_epub import METADATA, ROOT, combined_markdown, shrink_images

DIST = ROOT / "dist"
CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page {
  size: 7in 9.25in;
  margin: 0.8in 0.75in 0.85in 0.75in;
  @bottom-center { content: counter(page); font: 9pt "Avenir Next", Georgia, Menlo, sans-serif; color: #666; }
}
@page :first { margin: 0; @bottom-center { content: none; } }
@page front { @bottom-center { content: none; } }

html { font-size: 10.5pt; }
body { font-family: Charter, Georgia, Menlo, serif; line-height: 1.5; color: #1b1b1b; margin: 0; }

.cover { height: 9.25in; break-after: page; overflow: hidden; }
.cover img { width: 7in; height: 9.25in; max-height: none; object-fit: cover; display: block; }

header#title-block-header { page: front; break-after: page; padding-top: 2.4in; text-align: center; }
header#title-block-header .title { font: 700 34pt "Avenir Next", Georgia, Menlo, sans-serif; letter-spacing: 0.04em; margin: 0; padding: 0; break-before: avoid; }
header#title-block-header .title::after { margin: 0.4em auto 0; }
header#title-block-header p { text-align: center; }
header#title-block-header .subtitle { font: 500 14pt "Avenir Next", Georgia, Menlo, sans-serif; color: #444; margin-top: 0.6em; }
header#title-block-header .author { font: 600 13pt "Avenir Next", Georgia, Menlo, sans-serif; margin-top: 2.2in; letter-spacing: 0.12em; text-transform: uppercase; }
header#title-block-header .date { display: none; }

nav#TOC { page: front; break-after: page; }
nav#TOC::before { content: "Contents"; display: block; font: 700 22pt "Avenir Next", Georgia, Menlo, sans-serif; margin-bottom: 0.8em; }
nav#TOC ul { list-style: none; padding-left: 0; margin: 0; }
nav#TOC > ul > li { margin-top: 0.35em; font: 600 10.5pt "Avenir Next", Georgia, Menlo, sans-serif; }
nav#TOC > ul > li > ul { padding-left: 1.2em; }
nav#TOC > ul > li > ul > li { font: 400 9.5pt Charter, Georgia, Menlo, serif; margin: 0.1em 0; }
nav#TOC a { color: inherit; text-decoration: none; }

h1, h2, h3, h4 { font-family: "Avenir Next", Georgia, Menlo, sans-serif; line-height: 1.25; break-after: avoid; }
h1 { font-size: 22pt; font-weight: 700; break-before: page; margin: 0 0 0.9em; padding-top: 0.6in; }
h1::after { content: ""; display: block; width: 0.9in; height: 3px; background: #e2262c; margin-top: 0.35em; }
h1.part { text-align: center; font-size: 28pt; padding-top: 2.6in; }
h1.part::after { margin: 0.4em auto 0; }
h1.part + p { text-align: center; font-style: italic; color: #444; max-width: 4.3in; margin: 1em auto; }
h2 { font-size: 14pt; font-weight: 700; margin: 1.6em 0 0.5em; }
h3 { font-size: 11.5pt; font-weight: 600; margin: 1.3em 0 0.4em; }
h4 { font-size: 10.5pt; font-weight: 600; }
p { margin: 0 0 0.7em; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
a { color: #1a5fb4; text-decoration: none; }
blockquote { margin: 1em 0; padding: 0.5em 0.9em; border-left: 3px solid #e2262c; background: #f7f7f7; }
blockquote p { text-align: left; }

code { font-family: Menlo, Georgia, monospace; font-size: 8.6pt; background: #f2f2f2; padding: 0.05em 0.25em; border-radius: 2px; }
pre { font-family: Menlo, Georgia, monospace; font-size: 8pt; line-height: 1.4; background: #f6f6f6;
      border: 1px solid #e3e3e3; border-radius: 3px; padding: 0.6em 0.75em;
      white-space: pre-wrap; overflow-wrap: anywhere; break-inside: auto; }
pre code { background: none; padding: 0; font-size: inherit; }

table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 0.8em 0 1.1em; break-inside: auto; }
th, td { border-bottom: 1px solid #ddd; padding: 0.3em 0.45em; text-align: left; vertical-align: top; }
th { font-family: "Avenir Next", Georgia, Menlo, sans-serif; font-weight: 600; border-bottom: 2px solid #999; }
tr { break-inside: avoid; }

figure { margin: 1em 0; text-align: center; break-inside: avoid; }
img { max-width: 100%; max-height: 4.6in; }
figcaption { font: italic 8.8pt Charter, Georgia, Menlo, serif; color: #555; margin-top: 0.3em; }
ul, ol { margin: 0 0 0.7em; padding-left: 1.4em; }
li { margin: 0.15em 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
.keep { break-inside: avoid; }
"""

# Chrome does not reliably honour break-after: avoid on headings, so a heading
# can end up alone at the foot of a page. Before printing, wrap each h2-h4 with
# the block that follows it (when that block is short enough to move) in a
# div that must not be split.
KEEP_WITH_NEXT_JS = """<script>
document.addEventListener("DOMContentLoaded", () => {
  const movable = new Set(["P", "UL", "OL", "BLOCKQUOTE", "FIGURE", "TABLE", "PRE", "DIV"]);
  for (const h of document.querySelectorAll("h2, h3, h4")) {
    const next = h.nextElementSibling;
    if (!next || !movable.has(next.tagName)) continue;
    const lines = (next.innerText.match(/\\n/g) || []).length;
    if ((next.tagName === "PRE" || next.tagName === "DIV") && lines > 14) continue;
    const wrap = document.createElement("div");
    wrap.className = "keep";
    h.before(wrap);
    const lead = next.tagName === "P" && next.innerText.trim().endsWith(":") ? next.nextElementSibling : null;
    wrap.append(h, next);
    if (lead && movable.has(lead.tagName) && (lead.innerText.match(/\\n/g) || []).length <= 14) wrap.append(lead);
  }
});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cover", type=Path, default=ROOT / "assets" / "cover-1600x2400.jpg")
    ap.add_argument("--out", type=Path, default=DIST / "data-voyage.pdf")
    args = ap.parse_args()

    chrome = os.environ.get("CHROME", CHROME_DEFAULT)
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found: install it from https://pandoc.org")
    if not Path(chrome).exists() and shutil.which(chrome) is None:
        raise SystemExit(f"Chrome not found at {chrome}; set CHROME to a Chromium binary")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # 1200 px across a 5.5 in text block is ~220 dpi: sharp on screen, and
        # keeps the PDF under the 10 MB many storefront uploaders accept.
        shrink_images(ROOT / "manuscript" / "images", work / "images", max_width=1200)
        shutil.copy(args.cover, work / "cover.jpg")
        (work / "style.css").write_text(CSS, encoding="utf-8")
        (work / "keep.html").write_text(KEEP_WITH_NEXT_JS, encoding="utf-8")
        (work / "cover.html").write_text(
            '<div class="cover"><img src="cover.jpg" alt="Cover"></div>\n', encoding="utf-8"
        )
        (work / "metadata.yaml").write_text(METADATA, encoding="utf-8")
        (work / "book.md").write_text(combined_markdown(), encoding="utf-8")
        subprocess.run(
            [
                "pandoc",
                "metadata.yaml",
                "book.md",
                "-s",
                "-o",
                "book.html",
                "--toc",
                "--toc-depth=1",
                "--css=style.css",
                "--include-before-body=cover.html",
                "--include-in-header=keep.html",
                "--embed-resources",
                "--metadata=title-prefix:",
            ],
            cwd=work,
            check=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--generate-pdf-document-outline",
                f"--print-to-pdf={args.out.resolve()}",
                "--virtual-time-budget=20000",
                (work / "book.html").as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
