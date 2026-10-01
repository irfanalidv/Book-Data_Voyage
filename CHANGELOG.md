# Changelog

All notable changes to *Data Voyage* and its companion code are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com), and versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- DataCortex IQ is named as the book's publisher (copyright page, EPUB
  metadata, README, citation). Copyright stays with the author.

### Added

- `make pdf`: a typeset 7 x 9.25 in PDF (cover, linked contents, bookmarks,
  page numbers) built with pandoc and headless Chrome.

### Fixed

- Part pages showed the raw Leanpub marker `{class: part}` in the EPUB; both
  builds now turn it into a styled heading.
- `make type-check` always errored (it passed `book/` to mypy, which
  `pyproject.toml` excludes). It now checks `talentlens/` in strict mode,
  which passes after adding the missing type hints.
- `make verify-api-deps` installed the package with all its dependencies, so
  it could never catch a gap in `requirements-api.txt`. It now uses
  `--no-deps`, like the Dockerfile.

### Security

- sentence-transformers 5.6 (critical advisory), urllib3 2.8, and, for
  development, virtualenv 21.14 and black 26.3. Chapter 13 and 16 outputs
  are unchanged and the stored embeddings are bit-identical.

## [2.1.0] - 2026-10-01

First public release of the companion repository.

### Added

- Book cover, rendered from the author's photograph by `scripts/make_cover.py`,
  and used as the Leanpub title page.
- `make epub` builds a store-ready EPUB (3.7 MB, passes EPUBCheck 5.4 with no
  errors or warnings) for Amazon KDP, Google Play, and Apple Books, with a
  1600 x 2560 Amazon cover.
- Copyright page and About the Author page in the book.
- `LICENSE-BOOK.md`: the book text is © Irfan Ali, all rights reserved. Code
  stays under the MIT License, which now names the author as copyright holder.
- `ROADMAP.md`, a reader-facing list of what the book leaves out and what is
  planned next.

### Changed

- README rewritten around a quick start, a map of the repository, and a
  troubleshooting section, with the cover, status badges, and citation details
  (`CITATION.cff`).
- Copy edit across all 24 chapters and the front and back matter: lighter
  punctuation, plainer section headings, and fewer filler words.
- `make collect-dataset` documentation now matches the script (up to 400
  postings per role).
- Uvicorn reference link updated to its current documentation site.

### Fixed

- Chapter 17's latency chart was a stretched 2655 x 23565 image of the stub
  run; it now shows the measured OpenAI timings the chapter describes.
- Chapter 10's two figures had no generating code and showed an old feature
  set and an empty chart; the script now draws both from the current run.

### Removed

- Internal drafting notes and an editor workspace file.

### Security

- anyio 4.14, starlette 1.7, pillow 12.3, pyarrow 23.0, setuptools 84 in the
  runtime lockfile; jupyter-server, jupyterlab, tornado, bleach, mistune, and
  soupsieve updated in the development lockfile. None of these change a
  number quoted in the book. The transformers 5 and torch 2.9 upgrades are
  tracked in `ROADMAP.md` because they need a re-run of Chapters 12, 13,
  and 16.

## [2.0.0] - 2026-09-27

Publication pass. Every chapter re-run from a clean, lockfile-built
environment; every number in the prose now comes from that run.

### Changed: data

- **Bundled dataset rebuilt.** The previous `jobs_clean.csv` was degenerate
  (`salary_min` all null, salaries stuck near ₹2.1L, 3 of 5 roles, one-line
  descriptions). The Chapter 5 demo generator now produces fictional
  employers, role × seniority salaries with log-normal noise, overlapping
  duties between neighbouring roles, seniority-linked remote work, and about
  20% hidden salaries. `jobs_raw.csv` / `jobs_clean.csv` (576 rows) are its
  output and reproduce byte for byte (`tests/test_bundled_dataset.py`).
- Real company names removed from all synthetic generators.

### Changed: chapters

- All invented example outputs replaced with measured ones (Ch 5, 7, 8, 9,
  15, 16, 17, 19, 21).
- Ch 8 now demonstrates a real confound: a ₹2.45L remote gap (p = 3e-7) that
  vanishes within seniority bands. Ch 24's report says so instead of calling
  the raw gap a premium.
- Ch 9 selects models with the one-standard-error rule.
- Ch 10 reports a measured negative result and documents salary-imputation
  leakage on the Adzuna dataset (salary-only F1 0.677 on all rows vs 0.236
  on disclosed rows); v2 keeps only features whose lift beats noise.
- Ch 11 rewritten as a standalone clustering chapter; Ch 12 is now a full
  neural-network chapter with new seeded PyTorch code and tests.
- Ch 14 decomposition uses `seasonal_decompose` on daily data (period 7);
  hand-drawn components removed.
- Ch 15 benchmarks `apply` vs vectorised and uses realistic text columns.
- Ch 16 describes the implemented SQLite/NumPy store and weighted fusion.
- Ch 17/18: retired Groq models replaced and env-overridable; `--live`
  honours `LLM_PROVIDER`; Ch 18 replays its committed recorded run offline.
- Ch 22: `publish.yml` trusted-publishing workflow added; package
  dependencies bounded; the checkout requirement stated plainly.
- The chapters no longer promise a Kaggle collector or Colab notebooks
  that don't exist.

### Changed: book

- Leanpub manuscript: plain `Book.txt`, `{frontmatter}`/`{mainmatter}`/
  `{backmatter}` directives, generated Part pages, print-safe emoji
  replacement, Dedication first, Prologue opens the main matter.
- Glossary, References, chapter index, and running-project guide rewritten
  to match the code.

### Security

- torch 2.4 → 2.8 (CVE-2025-32434), transformers 4.46 → 4.57,
  sentence-transformers 3.3 → 5.1, requests 2.32 → 2.33, python-dotenv
  1.0 → 1.2. Unused dependencies removed (beautifulsoup4, lxml, SQLAlchemy,
  prophet, explicit tqdm).

### Fixed

- Figures render the ₹ sign (DejaVu Sans pinned after the seaborn style).
- `role_classifier_path()` and `talentlens_core` fall back to Chapter 9's
  real model path.
- Ch 19 band heuristic no longer maps "15 L" to ₹30L+ or matches bare "40".
- CI Docker job now waits for lint as well as tests.

## Earlier releases

Versions 1.0.0 to 1.2.0 (May to July 2026) were pre-publication drafts. They
built the 24-chapter structure, the TalentLens package and tests, the Docker
and CI/CD setup, the glossary and references, and the first Leanpub
manuscript pipeline. Version 2.0.0 replaced their dataset and rewrote the
chapters against measured output, so their detailed notes no longer describe
the current book.
