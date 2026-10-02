"""Build Leanpub ``manuscript/`` from ``book/`` sources.

Leanpub GitHub mode expects ``manuscript/Book.txt`` listing chapter files
in reading order, plus supporting assets under ``manuscript/``. Authoritative
order matches ``book/README.md`` (front matter → ch01–ch24 → back matter).

Run from repo root::

    python scripts/build_manuscript.py
    make manuscript

The script is idempotent: it wipes ``manuscript/`` and rebuilds. Image embeds
are copied into ``manuscript/images/`` with rewritten paths. Repo-only
cross-links are rewritten to plain text for PDF/EPUB. Transformations are
reported, not silent.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
OUT = ROOT / "manuscript"
IMAGES = OUT / "images"

# Canonical reading order - mirrors book/README.md TOC.
# (section, dest_filename, source_relative_to_ROOT)
PART_PAGES: dict[str, tuple[str, str]] = {
    "part1.md": (
        "Part I — Foundations",
        "The field we're entering, the working engineer's Python, the statistics you "
        "actually need, and which data sources TalentLens is allowed to use.",
    ),
    "part2.md": (
        "Part II — Data to Insight",
        "From an empty folder to a clean, understood dataset: collection, cleaning, "
        "exploratory analysis, and the statistical tests that decide which patterns "
        "are real.",
    ),
    "part3.md": (
        "Part III — Classical ML",
        "A role classifier, the features that did and didn't improve it, the "
        "structure clustering finds without labels, and when a neural network earns "
        "its cost.",
    ),
    "part4.md": (
        "Part IV — Specialist AI",
        "Three specialisms TalentLens needs: extracting skills from text, reading "
        "trends over time, and scaling Python past one comfortable DataFrame.",
    ),
    "part5.md": (
        "Part V — The GenAI Stack",
        "Semantic search, LLM-generated explanations, and an agent that chooses its "
        "own tools — with every failure mode measured.",
    ),
    "part6.md": (
        "Part VI — Shipping",
        "An API, a container, a CI/CD pipeline, and a package: the work that turns "
        "scripts into something other people can use.",
    ),
    "part7.md": (
        "Part VII — Career",
        "Three production systems and what broke in them, then the playbook for "
        "turning what you've built into offers.",
    ),
}

# Canonical reading order - mirrors book/README.md TOC.
# (section, dest_filename, source_relative_to_ROOT); source None = generated.
MANIFEST: list[tuple[str, str, str | None]] = [
    ("frontmatter", "00-copyright.md", "book/COPYRIGHT.md"),
    ("frontmatter", "00-dedication.md", "book/DEDICATION.md"),
    ("frontmatter", "01-foreword.md", "book/FOREWORD.md"),
    ("frontmatter", "02-preface.md", "book/PREFACE.md"),
    ("frontmatter", "03-acknowledgements.md", "book/ACKNOWLEDGEMENTS.md"),
    ("mainmatter", "04-prologue.md", "book/PROLOGUE.md"),
]
_PART_BEFORE = {
    1: "part1.md",
    5: "part2.md",
    9: "part3.md",
    13: "part4.md",
    16: "part5.md",
    19: "part6.md",
    23: "part7.md",
}
for _n in range(1, 25):
    if _n in _PART_BEFORE:
        MANIFEST.append(("mainmatter", _PART_BEFORE[_n], None))
    MANIFEST.append(("mainmatter", f"ch{_n:02d}.md", f"book/ch{_n:02d}/README.md"))
MANIFEST += [
    ("backmatter", "zz-glossary.md", "book/GLOSSARY.md"),
    ("backmatter", "zz-references.md", "book/REFERENCES.md"),
    ("backmatter", "zz-about-the-author.md", "book/ABOUT_THE_AUTHOR.md"),
]

# Emoji and pictographs that PDF fonts often lack. Signpost emoji are dropped
# (the bold label carries the meaning); status marks become words.
PRINT_REPLACEMENTS: list[tuple[str, str]] = [
    ("📑 ", ""),
    ("⏭️ ", ""),
    ("🔍 ", ""),
    ("📝 ", ""),
    ("✅", "OK"),
    ("❌", "FAIL"),
    ("⏭️", "SKIP"),
    ("⚠️", "WARN"),
    ("✓", "ok"),
    ("\ufe0f", ""),
]


def _print_safe(text: str) -> str:
    for old, new in PRINT_REPLACEMENTS:
        text = text.replace(old, new)
    return text


# Markdown image: ![alt](path) or ![alt](path "title")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# Relative .md links: [text](path.md) or [text](path.md#anchor)
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+\.md)(?:#[^)\s]*)?\)")
# Angle-bracket tokens that look like HTML/XML tags
TAG_RE = re.compile(r"</?[A-Za-z_][^>\n]{0,80}>")


@dataclass
class Report:
    files_written: list[str] = field(default_factory=list)
    images_copied: list[str] = field(default_factory=list)
    broken_images: list[str] = field(default_factory=list)
    rewritten_images: int = 0
    code_links: int = 0
    link_rewrites: list[str] = field(default_factory=list)
    leftover_md_links: list[str] = field(default_factory=list)
    placeholders_safe: list[str] = field(default_factory=list)
    placeholders_bare: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _slug_image_name(dest_stem: str, source_path: Path) -> str:
    """Unique, stable image name inside manuscript/images/."""
    return f"{dest_stem}__{source_path.name}"


def _link_target_name(href: str) -> str:
    return Path(href.split("#", 1)[0]).name


def _rewrite_md_links(text: str, src: Path, report: Report) -> str:
    def repl(match: re.Match[str]) -> str:
        link_text, href = match.group(1), match.group(2)
        if href.startswith(("http://", "https://")):
            return match.group(0)
        name = _link_target_name(href)
        lower = name.lower()
        # Mid-sentence-safe phrases (avoid "live in see the Glossary…").
        if lower == "glossary.md":
            plain = "the Glossary at the end of this book"
            report.link_rewrites.append(
                f"{src.relative_to(ROOT)}: [{link_text}]({href}) → {plain!r}"
            )
            return plain
        if lower == "references.md":
            plain = "the References at the end of this book"
            report.link_rewrites.append(
                f"{src.relative_to(ROOT)}: [{link_text}]({href}) → {plain!r}"
            )
            return plain
        if lower in ("scope.md", "roadmap.md", "open_threads.md"):
            plain = "the companion repository's SCOPE.md"
            report.link_rewrites.append(
                f"{src.relative_to(ROOT)}: [{link_text}]({href}) → {plain!r}"
            )
            return plain
        # Unknown relative .md link - leave and flag
        report.leftover_md_links.append(f"{src.relative_to(ROOT)} → {href}")
        return match.group(0)

    return MD_LINK_RE.sub(repl, text)


def _classify_placeholders(text: str, src: Path, report: Report) -> None:
    """Report angle-bracket tokens as safe (code) or bare (source defect)."""
    fence_spans = [(m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?```", text)]
    inline_spans = [(m.start(), m.end()) for m in re.finditer(r"`[^`\n]+`", text)]

    def in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
        return any(a <= pos < b for a, b in spans)

    seen: set[str] = set()
    for m in TAG_RE.finditer(text):
        tag = m.group(0)
        key = f"{src}:{m.start()}:{tag}"
        if key in seen:
            continue
        seen.add(key)
        line = text.count("\n", 0, m.start()) + 1
        loc = f"{src.relative_to(ROOT)}:{line} {tag[:70]}"
        if in_spans(m.start(), fence_spans) or in_spans(m.start(), inline_spans):
            report.placeholders_safe.append(loc)
        else:
            report.placeholders_bare.append(loc)


GITHUB_BLOB = "https://github.com/irfanalidv/Book-Data_Voyage/blob/main/"
GITHUB_TREE = "https://github.com/irfanalidv/Book-Data_Voyage/tree/main/"
_ROOT_FILES = {"Dockerfile", "Makefile", "pyproject.toml", "render.yaml", "LICENSE", "README.md"}
_CODE_PATH = re.compile(r"(?<![\[`])`(\.?/?[A-Za-z0-9_.\-/]+?)(?::(\d+))?`(?!\])")


def _tracked_paths() -> tuple[set[str], set[str]]:
    """Files git tracks, and every directory that contains one."""
    import subprocess

    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    files = set(out.stdout.split())
    dirs = {"/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)}
    return files, dirs


try:
    _TRACKED = _tracked_paths()
except Exception:  # no git checkout: leave code spans as plain text
    _TRACKED = (set(), set())


def _link_code_paths(text: str, src: Path, report: "Report") -> str:
    """Turn `book/ch05/x.py`-style code spans into links to the file on GitHub.

    Only spans that resolve to a file or folder git tracks are linked (paths in
    a chapter folder resolve relative to it too); API routes, generated files
    and placeholders stay plain. Fenced code, headings, and spans already
    inside a link are left alone.
    """
    files, dirs = _TRACKED
    chapter_dir = src.parent.relative_to(ROOT).as_posix()

    def resolve(raw: str) -> str | None:
        p = raw.removeprefix("./").rstrip("/")
        if not p or ("/" not in p and p not in _ROOT_FILES and not p.startswith("requirements")):
            return None
        for cand in (p, f"{chapter_dir}/{p}", f"{chapter_dir}/reports/{p}"):
            if cand in files or cand in dirs:
                return cand
        return None

    def repl(m: re.Match) -> str:
        target = resolve(m.group(1))
        if target is None:
            return m.group(0)
        report.code_links += 1
        if target in files:
            url = GITHUB_BLOB + target + (f"#L{m.group(2)}" if m.group(2) else "")
        else:
            url = GITHUB_TREE + target
        return f"[{m.group(0)}]({url})"

    out = []
    for i, part in enumerate(re.split(r"(```.*?```|~~~.*?~~~)", text, flags=re.S)):
        if i % 2:
            out.append(part)
            continue
        lines = [
            ln if ln.lstrip().startswith("#") else _CODE_PATH.sub(repl, ln)
            for ln in part.split("\n")
        ]
        out.append("\n".join(lines))
    return "".join(out)


def _rewrite_images(text: str, src: Path, dest_stem: str, report: Report) -> str:
    def replace_image(match: re.Match[str]) -> str:
        alt, rel = match.group(1), match.group(2)
        if rel.startswith(("http://", "https://", "data:")):
            report.notes.append(f"{src}: left remote/data image unchanged: {rel}")
            return match.group(0)
        target = (src.parent / rel).resolve()
        if not target.is_file():
            report.broken_images.append(f"{src.relative_to(ROOT)} → {rel} (resolved {target})")
            return match.group(0)
        name = _slug_image_name(dest_stem, target)
        dest_img = IMAGES / name
        if not dest_img.exists():
            shutil.copy2(target, dest_img)
            report.images_copied.append(name)
        report.rewritten_images += 1
        return f"![{alt}](images/{name})"

    return IMAGE_RE.sub(replace_image, text)


def _rewrite_content(text: str, src: Path, dest_stem: str, report: Report) -> str:
    # Classify placeholders on source text (before any rewrite) for accurate locations
    _classify_placeholders(text, src, report)
    new_text = _rewrite_images(text, src, dest_stem, report)
    new_text = _rewrite_md_links(new_text, src, report)
    new_text = _link_code_paths(new_text, src, report)
    return new_text


def _write_book_txt(entries: list[tuple[str, str]]) -> None:
    """Write Leanpub Book.txt (one filename per line, nothing else) and mark sections.

    Leanpub reads Book.txt as a plain file list. Front, main, and back matter
    are declared with Markua directives at the top of the first file of each
    section, so this function prepends ``{frontmatter}`` and friends there.
    """
    current: str | None = None
    for section, filename in entries:
        if section != current:
            path = OUT / filename
            path.write_text(
                f"{{{section}}}\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            current = section
    (OUT / "Book.txt").write_text("\n".join(f for _, f in entries) + "\n", encoding="utf-8")


def build() -> Report:
    report = Report()

    if not BOOK.is_dir():
        raise SystemExit(f"Missing book/ at {BOOK}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    IMAGES.mkdir()

    report.notes.append(
        "Wiped manuscript/ and rebuilt from book/ (idempotent). "
        "Images → images/<dest>__<basename>; "
        "GLOSSARY/REFERENCES/SCOPE links → plain text; "
        "Part pages generated; emoji replaced for print."
    )

    book_entries: list[tuple[str, str]] = []

    for section, dest_name, src_rel in MANIFEST:
        if src_rel is None:
            title, blurb = PART_PAGES[dest_name]
            text = f"{{class: part}}\n# {title}\n\n{blurb}\n"
            (OUT / dest_name).write_text(text, encoding="utf-8")
            report.files_written.append(dest_name)
            book_entries.append((section, dest_name))
            continue
        src = ROOT / src_rel
        if not src.is_file():
            raise SystemExit(f"Missing source file: {src_rel}")
        dest_stem = Path(dest_name).stem
        text = src.read_text(encoding="utf-8")
        rewritten = _print_safe(_rewrite_content(text, src, dest_stem, report))

        dest = OUT / dest_name
        dest.write_text(rewritten, encoding="utf-8")
        report.files_written.append(dest_name)
        book_entries.append((section, dest_name))

    _write_book_txt(book_entries)
    # Leanpub uses images/title_page.jpg as the book's title page (make_cover.py renders it).
    cover = ROOT / "assets" / "cover-1600x2400.jpg"
    if cover.is_file():
        shutil.copy2(cover, IMAGES / "title_page.jpg")

    # Deduplicate leftover links
    seen: set[str] = set()
    unique: list[str] = []
    for item in report.leftover_md_links:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    report.leftover_md_links = unique

    return report


def print_report(report: Report) -> int:
    print("=== manuscript build report ===")
    print(f"files in manuscript/ (excl. images/): {len(report.files_written)}")
    print(f"images copied: {len(report.images_copied)}")
    print(f"image embeds rewritten: {report.rewritten_images}")
    print(f"code paths linked to GitHub: {report.code_links}")
    print(f"broken image references: {len(report.broken_images)}")
    for row in report.broken_images:
        print(f"  - {row}")
    print()

    print(f"link rewrites made: {len(report.link_rewrites)}")
    for row in report.link_rewrites:
        print(f"  - {row}")
    print()

    print(f"leftover relative .md links (unhandled): {len(report.leftover_md_links)}")
    for row in report.leftover_md_links:
        print(f"  - {row}")
    print()

    print(f"placeholder tags SAFE (code span/fence): {len(report.placeholders_safe)}")
    for row in report.placeholders_safe:
        print(f"  - {row}")
    print(f"placeholder tags BARE (source defects — fix in book/): {len(report.placeholders_bare)}")
    for row in report.placeholders_bare:
        print(f"  - {row}")
    print()

    for note in report.notes:
        print(f"NOTE: {note}")
    print()
    print("Git: COMMIT manuscript/ (Leanpub GitHub sync). Regenerate with `make manuscript`.")
    print(f"Output: {OUT}")

    exit_bad = bool(report.broken_images or report.placeholders_bare)
    return 1 if exit_bad else 0


def main() -> None:
    report = build()
    code = print_report(report)
    sys.exit(code)


if __name__ == "__main__":
    main()
