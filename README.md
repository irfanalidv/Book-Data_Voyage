<p align="center">
  <img src="assets/cover-web.jpg" alt="Data Voyage book cover: a hill town lit up at night inside the fog" width="340">
</p>

<h1 align="center">Data Voyage</h1>

<p align="center"><strong>Building Real AI Systems from Data to Deployment</strong><br>
Irfan Ali · Published by DataCortex IQ</p>

<p align="center">
  <a href="https://leanpub.com/datavoyage"><img src="https://img.shields.io/badge/buy%20the%20ebook-Leanpub-1A5FB4" alt="Buy on Leanpub"></a>
  <a href="https://github.com/irfanalidv/Book-Data_Voyage/actions/workflows/ci.yml"><img src="https://github.com/irfanalidv/Book-Data_Voyage/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-green" alt="Code license: MIT"></a>
  <a href="LICENSE-BOOK.md"><img src="https://img.shields.io/badge/book%20text-%C2%A9%20all%20rights%20reserved-lightgrey" alt="Book text: all rights reserved"></a>
  <img src="https://img.shields.io/badge/code%20style-black-000000" alt="Code style: black">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64" alt="Lint: ruff">
  <img src="https://img.shields.io/badge/chapters-24-E2262C" alt="24 chapters">
</p>

<p align="center"><em>One project. 24 chapters. From raw job postings to a deployed, tested, packaged AI service.</em></p>

---

This is the companion repository for the book *Data Voyage*. The book teaches production AI engineering through one running project, **TalentLens**, a job market intelligence platform that grows from raw data to a deployed API. Every chapter lives in `book/chNN/` and has two things: a `README.md` with the chapter text, and a Python script you can run.

## Get the book

**Read it free right here on GitHub**, or get the typeset ebook (PDF and EPUB, readable offline, with free updates) on **[Leanpub](https://leanpub.com/datavoyage)**. Buying a copy is the best way to support the book.

Students in India: use [this link for the student price](https://leanpub.com/datavoyage/c/INDIASTUDENTS).

---

**Contents:** [Get the book](#get-the-book) · [Quick start](#quick-start) · [How to read along](#how-to-read-along) · [What you will build](#what-you-will-build) · [Repository layout](#whats-in-this-repository) · [The dataset](#the-dataset) · [Troubleshooting](#troubleshooting) · [About the author](#about-the-author) · [Cite this book](#cite-this-book) · [License](#license)

---

## Quick start

You need Python 3.11 and about 1.5 GB of disk space (most of it is PyTorch).

```bash
git clone https://github.com/irfanalidv/Book-Data_Voyage
cd Book-Data_Voyage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # exact versions from the lockfile + spaCy model
python book/ch01/ch01_data_science_landscape.py
```

If the last command prints an environment check with ticks, you are ready for Chapter 1.

No `make` on your machine? Run the three steps it wraps:

```bash
pip install -r requirements-lock.txt
pip install -e .
python -m spacy download en_core_web_sm
```

**Linux without an NVIDIA GPU:** add the CPU-only PyTorch index, or pip will download several GB of CUDA libraries the book never uses:

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu make install
```

---

## How to read along

1. Open `book/chNN/README.md` and read the chapter.
2. Run that chapter's script from the repository root, for example `python book/ch05/ch05_data_collection.py`.
3. Compare your output with the numbers quoted in the chapter. On the bundled dataset they match exactly.

Chapters build on each other, but every chapter also runs on its own, because the repository ships the data and models the early chapters produce. `book/README.md` is the full chapter index.

| Part | Chapters | Topics |
|---|---|---|
| I. Foundations | 1–4 | The field, working Python, statistics, data sources and ethics |
| II. Data to Insight | 5–8 | Collection, cleaning, exploratory analysis, inference |
| III. Classical ML | 9–12 | Classification, feature engineering, clustering, neural networks |
| IV. Specialist AI | 13–15 | NLP, time series, scaling Python |
| V. The GenAI Stack | 16–18 | Vector search, LLM generation, agents |
| VI. Shipping | 19–22 | FastAPI, Docker and Render, CI/CD, packaging |
| VII. Career | 23–24 | Case studies, the India job-market playbook |

---

## What you will build

- A collection pipeline for live job postings, plus a reproducible 576-row corpus the whole book runs on
- A role classifier for five data and AI roles, with leakage checks and a linear baseline to beat
- A skill extractor benchmarked three ways (regex, spaCy, sentence embeddings) on a labelled evaluation set
- Semantic job search with sentence embeddings, and the pgvector pattern for production
- An LLM layer that parses CVs and explains job matches, with validation and graceful fallback
- A tool-calling agent written from scratch, with its failure modes traced and named
- A FastAPI service in a 462 MB Docker image, deployed to Render
- A GitHub Actions pipeline that tests, builds, and deploys on every push
- `talentlens-core`, a Python package with a trusted-publishing release workflow
- A career playbook for the Indian AI and ML job market that separates measured findings from judgement

---

## What's in this repository

```
book/            One folder per chapter (text + script + figures), plus front and back matter
talentlens/      Shared package the chapters import (paths, settings, features)
data/            The bundled dataset: raw, clean, and feature tables
models/          The Chapter 10 classifier (other chapters keep models in their own folder)
tests/           One test file per chapter; run with `make test`
scripts/         Helpers: real-data collection, evaluation-set labelling, manuscript build
manuscript/      The ebook edition, generated from book/ by `make manuscript`
assets/          Book cover (rendered from the author's photograph by scripts/make_cover.py)
```

`make help` lists every command. The ones you will use most:

```bash
make install        # runtime dependencies (enough for every chapter)
make install-dev    # adds pytest, ruff, black for tests and linting
make test           # full test suite
make run            # start the Chapter 19 API on http://localhost:8000
make docker         # build the Chapter 20 production image
make epub           # build the ebook in dist/ (needs pandoc)
make pdf            # build the typeset PDF in dist/ (needs pandoc + Chrome)
```

---

## The dataset

The repository ships `data/clean/jobs_clean.csv`: 576 cleaned job postings from fictional employers. Chapter 5's demo mode generates them with a fixed seed and Chapter 6 cleans them. The data is synthetic but behaves like real postings: salaries depend on role and seniority and are right-skewed, neighbouring roles share duties, and about one posting in five hides its salary.

Every chapter runs on it with no network access and no API keys, and every number in the book reproduces from a fresh clone. `tests/test_bundled_dataset.py` checks that the pipeline regenerates the file byte for byte.

**Using real data.** Register a free key at [developer.adzuna.com](https://developer.adzuna.com), copy `.env.example` to `.env`, fill in `ADZUNA_APP_ID` and `ADZUNA_API_KEY`, then run:

```bash
make collect-dataset
```

This requests up to 400 postings for each of the five roles, cleans them with Chapter 6's pipeline, and writes `data/clean/jobs_clean.large.csv`. The author's run kept 1,190 postings after cleaning. Every chapter picks up that file automatically; delete it to go back to the bundled corpus. Your numbers will differ from the book's, and Chapter 10 shows what changes.

Real-data results in the book come from The Adzuna API. The repository does not redistribute collected Adzuna datasets; their terms ask each user to collect under their own key. The one exception is Chapter 13's evaluation set: 200 listing excerpts (a title and up to 500 characters of description), each labelled "Jobs by Adzuna" with a link to the original ad and included only to reproduce that chapter's benchmark. [`book/ch13/data/README.md`](book/ch13/data/README.md) explains the source, the rights, and how to request removal.

**Optional LLM keys.** Chapters 17 and 18 run offline by default (a deterministic stub and a recorded agent run). Add `GROQ_API_KEY` or `OPENAI_API_KEY` to `.env` to call a live model.

---

## Troubleshooting

- **`No module named ruff` or `pytest`:** run `make install-dev`. The runtime install leaves out development tools on purpose.
- **`No module named talentlens`:** run `pip install -e .` from the repository root (`make install` does this for you).
- **A chapter can't find its input file:** run the scripts from the repository root, not from inside `book/chNN/`.
- **Different numbers from the book:** check whether `data/clean/jobs_clean.large.csv` exists. If it does, you are running on your own collected data.

To change dependencies, edit `requirements.txt`, run `pip install uv && make lock`, and reinstall from the new lockfile.

---

## Known limits and plans

[ROADMAP.md](./ROADMAP.md) lists what the book deliberately leaves out and what is planned for the next edition. [CHANGELOG.md](./CHANGELOG.md) records what changed between releases.

---

## About the author

Irfan Ali is a senior AI engineer with seven years of production experience and the founder of DataCortex IQ. He has fine-tuned LLMs at a Schneider Electric subsidiary, built the AI layer for a Hong Kong startup, shipped a voice-first wellness app, and built inventory software for a Nepal-based FMCG business. He has published eleven libraries on PyPI and two peer-reviewed papers (IJAINN, 2025), and holds a Master's in Data Science and AI from IISER Tirupati.

[github.com/irfanalidv](https://github.com/irfanalidv) · irfan@datacortex.in

---

## Cite this book

```bibtex
@book{ali2026datavoyage,
  author    = {Irfan Ali},
  publisher = {DataCortex IQ},
  title     = {Data Voyage: Building Real AI Systems from Data to Deployment},
  year      = {2026},
  edition   = {2.1},
  url       = {https://github.com/irfanalidv/Book-Data_Voyage}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff) and offers a "Cite this repository" button.

---

## License

- **Code** (Python files, configuration, Dockerfile, workflows, data-generation scripts): [MIT](./LICENSE). Use it, change it, ship it.
- **Book text** (the chapter `README.md` files and the other Markdown under `book/` and `manuscript/`): © 2026 Irfan Ali, all rights reserved. See [LICENSE-BOOK.md](./LICENSE-BOOK.md). You are welcome to read it here and quote short passages with attribution.
- **Adzuna listing excerpts** in `book/ch13/data/`: owned by Adzuna and the original advertisers, not covered by either license above, and included only to reproduce Chapter 13's benchmark.

---

## Contributing

Bug reports and code fixes are welcome as GitHub issues or pull requests. So are notes from your own production experience that contradict something in a chapter.
