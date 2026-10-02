# Chapter 4: Data — Types, Sources, Ethics

> **TalentLens milestone:** Decide **legally and practically** which sources feed TalentLens before Chapter 5 writes `data/raw/jobs_raw.csv`. You will compare exactly three allowed sources, map them to one canonical schema, and understand why scraping LinkedIn is the wrong first move.

---

## The problem we're solving

You finished Chapter 2. Paths resolve from `REPO_ROOT`. You open a new file, call it `scrape_linkedin.py`, because TalentLens needs job postings and LinkedIn obviously has them.

Ten minutes later you are in a tab maze: login walls, infinite scroll, JSON buried inside script tags, and a Terms of Service page that says automated access is not permitted. Your scraper works on your laptop Tuesday night. Wednesday morning LinkedIn changes a CSS class; Thursday your IP is throttled; Friday a teammate asks whether this repo is legal to ship in a portfolio.

**That failure is the point of this chapter.** TalentLens does not train you to win a cat-and-mouse game with a front-end you do not control. It trains you to build datasets the way teams that still have jobs in twelve months build them: **documented APIs**, **archives with clear licences**, **rate limits**, **schema normalisation**, and **ethics you can explain in an interview**.

Chapter 4 is the contract Chapter 5 executes. Only three sources are allowed into the pipeline. Everything else (LinkedIn, Naukri, Indeed) appears in a deliberate **what we do not do** section so you stop reinventing a blocked path.

---

## Why this practice, and why now

Chapter 1 gave you the **five roles** and showed why keyword search on titles fails. Chapter 2 installed `talentlens` and `DATA_DIR` so collectors know where files land.

If we jumped straight to Chapter 5 without Chapter 4, you would copy the first Stack Overflow snippet that paginates a job board, hardcode cookies, and discover data-quality problems only after you had committed to an illegal source. **Source choice is architecture.**

| Choice | TalentLens stance |
|--------|-------------------|
| Adzuna API | Allowed: registered key, JSON, India endpoint |
| RemoteOK API | Allowed: public JSON, remote tech roles |
| GitHub Jobs archive (Kaggle) | Allowed: static CSV, historical breadth |
| LinkedIn / Naukri / Indeed scraping | **Not in scope** (see below) |

**What Chapters 5 and 6 will show on real pulls:**

- Adzuna returns rich descriptions but **`salary_min` / `salary_max` are often null**. Expect roughly **a third or more** of salary fields to be missing on Indian tech pulls; that is a source property, not a bug in your code. Chapter 6 handles coverage and imputation; Chapter 7 visualises it.
- RemoteOK fills gaps for remote USD roles but overlaps Adzuna on many titles, so dedup in Chapter 5 matters.
- A historical archive would add title variety for classifiers even though its postings are stale; Chapter 5 leaves that collector as an exercise.

**Looking ahead to Chapter 24:** the India playbook measures what it can from this same pipeline (the remote vs on-site comparison) and labels everything else as the author's market read. It never leans on scraped CVs or DMs.

---

## The methods

### The three sources (and only these three)

Chapter 4 prints this table from `source_comparison_table()`. Chapter 5 implements collectors for the two live APIs; the archive is allowed but left as an exercise.

> **📑 Reference: Allowed data sources**

| Source | Auth | Format | Rate limit | Sweet spot |
|--------|------|--------|------------|------------|
| **Adzuna API** | `ADZUNA_APP_ID` + `ADZUNA_API_KEY` (free tier) | JSON | ~250 calls/day (free tier) | Current Indian + global postings; salary fields when employer supplied them |
| **RemoteOK API** | None | JSON | Be polite; no published hard cap | Live remote tech roles; fast confidence check without keys |
| **GitHub Jobs archive (Kaggle)** | Kaggle account for download | CSV | N/A (static file) | Historical postings (GitHub Jobs closed in 2021); title variety for classifiers; check the dataset's licence before use |

**Adzuna in one sentence:** best free programmatic option for **current** India-market postings with structured fields. If you set keys in `.env`, `ch04_data_sources.py` can demo a three-title fetch; without keys, the script tells you to skip until Chapter 5.

**RemoteOK in one sentence:** zero-setup JSON list; normalise five live rows in the default CLI run (unless `--offline`).

**Kaggle archive in one sentence:** not for "what opened today" but for **title and description variety** across years. The companion code does not ship a collector for it; writing a `KaggleArchiveCollector` on Chapter 5's `BaseCollector` pattern is a good first exercise.

### Storage formats: JSON, CSV, Parquet (brief)

> **📑 Reference: Storage formats**

| Format | Typical use in TalentLens | Tradeoff |
|--------|---------------------------|----------|
| **JSON** | API responses (Adzuna, RemoteOK) | Nested, schema-flexible; normalise immediately |
| **CSV** | `jobs_raw.csv`, Kaggle archive | Human-readable; no types enforced; fine to ~low millions of rows |
| **Parquet** | Optional later in pipeline (Chapter 6+) | Columnar, typed, smaller on disk; better when data grows past comfortable pandas CSV |

Rule: **collect JSON → normalise to canonical rows → persist CSV** for the book pipeline. Parquet is an optimisation, not a requirement to start.

### Canonical schema: every source maps to this

> **📑 Reference: TalentLens canonical schema**

The executable authority is Chapter 5's `SCHEMA` dict in `ch05_data_collection.py`; this table is the reading copy. Chapter 4 demonstrates normalisation with `normalize_remoteok_posting()`. Every row, regardless of origin, must align to:

| Field | Type (logical) | Notes |
|-------|----------------|-------|
| `job_id` | string | Prefix by source, e.g. `adzuna_123`, `remoteok_456` |
| `source` | string | `adzuna` \| `remoteok` \| `kaggle` \| `demo` |
| `title` | string | Required |
| `company` | string | Required |
| `city` | string | City or `"Remote"` |
| `country` | string | ISO or `"Remote"` |
| `description` | string | Required full text |
| `skills_raw` | string | Comma-separated tags if available |
| `salary_min` | float or null | Annual, local currency |
| `salary_max` | float or null | Annual, local currency |
| `currency` | string | `INR`, `USD`, etc. |
| `is_remote` | bool | |
| `posted_date` | string | ISO date string when known |
| `url` | string | Link back to original posting |

**Required for validation:** `title`, `company`, `description` non-empty. Optional fields may be `None`, especially salary. Do not drop rows with missing pay; keep the nulls for Chapter 6.

Example normalised RemoteOK-shaped record (from the chapter script):

```
source: remoteok
title: Senior ML Engineer
company: Example Remote Co
salary_min: 80000.0
is_remote: True
```

### Ethics and compliance: what to do, what to refuse

**robots.txt and site terms:** If a site disallows automated access in robots.txt *and* in Terms of Service, treat that as a stop sign for production pipelines. Educational scraping of career pages is still a common mistake in portfolios, and interviewers notice.

**hiQ Labs v. LinkedIn (US, 2017–2022):** hiQ scraped public LinkedIn profiles; LinkedIn tried to block it. Appeals rulings suggested that scraping public pages was probably not "unauthorised access" under US computer-crime law, but in 2022 a district court found hiQ had breached LinkedIn's User Agreement, and the case settled with hiQ agreeing to stop. The litigation does not hand you a licence to scrape every job board worldwide: it turned on specific facts and US law, and contract terms still bit. **Practical lesson for TalentLens:** prefer APIs and datasets the provider intends you to use; if you cannot cite permission, do not ship the collector.

**Rate limiting and attribution:** Sleep between calls; honour `429` and `Retry-After`; set an honest `User-Agent` (the book uses `TalentLens/1.0 (Data Voyage educational project)`). Keep `url` and `source` on every row so downstream reports can attribute postings. Some providers make attribution a condition of use. RemoteOK's API terms ask you to link back to the original posting and credit Remote OK as the source; Adzuna's developer terms likewise require visible attribution. Read the terms page of every API you add.

**GDPR / India DPDPA angle:** Job **postings** are generally business communications about roles, with lower sensitivity than **candidate CVs** or employee records. TalentLens stores posting text, not applicant PII. Do not collect names, emails, or phone numbers from profiles. If you later add user accounts (Chapter 19+), that is a separate consent and retention design, not part of job ingestion.

**API keys:** `ADZUNA_*` in `.env` only; never commit. Chapter 2's `SecretStr` pattern applies.

### What we do not do: LinkedIn, Naukri, Indeed

This section exists so you do not waste Chapter 5 on the wrong work.

| Source | Why TalentLens skips it |
|--------|-------------------------|
| **LinkedIn** | ToS restrict automated scraping; technical anti-bot; legal risk for redistributing content; no stable public job API for your use case |
| **Naukri** | ToS and technical barriers similar in spirit; no first-class API in this curriculum |
| **Indeed** | Publisher programme exists for partners; ad-hoc HTML scraping is fragile and often disallowed |

You may still **read** postings as a human job seeker. You should not build TalentLens on scraped HTML from these sites. When Chapter 24 measures anything, it measures **your** pipeline's output, not a scraper you cannot show in a compliance review.

---

## The code

Runnable entrypoint:

```bash
python book/ch04/ch04_data_sources.py
```

**Offline / CI mode**: skips live RemoteOK HTTP (tests and air-gapped runs):

```bash
python book/ch04/ch04_data_sources.py --offline
```

**Design decision 1: three rows, asserted in tests.** `assert len(rows) == 3` in `print_comparison_table()` so a fourth "convenience" source does not creep in without updating Chapter 5.

**Design decision 2: normalisation before persistence.** `normalize_remoteok_posting()` returns a flat dict matching Chapter 5's `coerce_to_schema()` expectations: same field names, typed floats for salary when parseable.

**Design decision 3: Adzuna fails open.** `try_adzuna_sample()` returns a skip message without credentials instead of raising. That mirrors how CI runs while still teaching the env vars Chapter 5 needs.

**Tests:**

```bash
pytest book/ch04/tests/test_ch04.py -q
```

---

## Interpreting the output

Successful run (online) prints:

1. **Source comparison**: three blocks with auth, format, rate limit, sweet spot.
2. **Schema normalisation**: selected keys from a fabricated RemoteOK JSON blob.
3. **Adzuna line**: either sample titles or "skipped: set ADZUNA_APP_ID…".
4. **Live RemoteOK sample**: five titles/companies unless `--offline`.
5. **Repo root**: confirms `talentlens.paths` resolution from Chapter 2.

**What "skipped Adzuna" means:** not an error. Register free keys at Adzuna's developer portal, add to `.env`, re-run before Chapter 5 collection day.

**What live RemoteOK proves:** the internet is reachable, JSON parses, and your normaliser produces canonical rows. It does **not** mean your final dataset is representative. Five rows are a wiring test.

**What you should not see:** LinkedIn cookies, Selenium drivers, or Naukri URLs in this chapter's code path. If you added them locally, keep them out of git.

**Re-run confidence check:** execute the script twice ten minutes apart. RemoteOK titles should not be byte-identical, because the feed updates. Identical output across runs on different days means you are reading cached or hardcoded data, not the API.

---

## Common mistakes I've seen (and made)

**Mistake: Scraping LinkedIn because "everyone does it"**

What happens: brittle selectors, account warnings, unusable dataset for sharing, and an interview answer you cannot defend.

How to catch it: grep the repo for `linkedin.com` in collectors; restrict to the three-source table. Use APIs in Chapter 5.

---

**Mistake: Mixing salary units and currencies in one column**

What happens: RemoteOK USD doubles beside Adzuna INR lakhs; means and medians in Chapter 3 become nonsense.

How to catch it: always set `currency`; never compare `salary_min` across rows without checking units. Chapter 6 adds cleaning rules.

---

**Mistake: Ignoring rate limits until IP ban**

What happens: 429 storms during a demo; empty `jobs_raw.csv`; blame on "the API."

How to catch it: sleep between Adzuna calls in Chapter 5; exponential backoff on 429; track daily quota (~250 free).

---

**Mistake: Storing only normalised CSV and throwing away source JSON**

What happens: when Adzuna adds a field, you re-collect everything instead of re-parsing.

How to catch it: optional raw JSON snapshots under `data/raw/` per source in Chapter 5; at minimum keep `source` and `url` on each row for audit.

---

**Mistake: Treating job postings like personal data you can monetise**

What happens: privacy review blocks your side project; GDPR/DPDPA questions you cannot answer.

How to catch it: postings only; no applicant CVs; document purpose (labour market analytics); attribution via `url`; retention policy before production (Chapter 22–24).

---

## Interview questions

**Q1: Why would you choose Adzuna + RemoteOK + a Kaggle archive instead of scraping LinkedIn?**

Template answer: "LinkedIn scraping violates ToS, breaks often, and creates legal and redistribution risk. Adzuna gives a documented JSON API with an India endpoint; RemoteOK gives current remote tech roles without auth; the GitHub Jobs archive adds historical title diversity for ML. The trio is sustainable, explainable in compliance review, and good enough for labour-market analytics when I document missing salary rates and dedup across sources."

**Q2: How do you normalise heterogeneous API responses into one schema?**

Template answer: "I define a canonical row with required fields title, company, description, and optional salary and location fields. Each collector maps source-specific keys (e.g. RemoteOK position → title) into that schema, coerces types, uses None for missing optionals, and prefixes job_id with the source name. Validation rejects empty required fields; I do not silently invent salaries."

**Q3: What is your approach to robots.txt and terms of service?**

Template answer: "I read both. If automated access is prohibited, I do not build a production scraper. I use official APIs or licensed datasets. For educational projects I still avoid prohibited scraping because portfolio code becomes production code. Rate limits and honest User-Agent strings are part of being a good client."

**Q4: How do missing Adzuna salaries affect downstream analysis?**

Template answer: "A third or more of salary fields being null is expected for some pulls, because employers omit pay. I track field coverage at collection time, do not drop rows solely for missing salary, and handle imputation or missing indicators in cleaning. EDA in Chapter 7 reports coverage by source; models in Chapter 9 may use log salary where present and separate models or features where absent."

**Q5: CSV vs Parquet for the TalentLens pipeline?**

Template answer: "CSV for raw and early clean stages in this project: simple, inspectable, works with pandas across chapters. I move to Parquet when size or type stability matters: columnar storage, less disk, faster reads. Regardless of format, the logical contract is the canonical schema, not the file extension."

---

## What's next

**Chapter 5** runs collectors (Adzuna and RemoteOK live, or the seeded demo generator offline) into `data/raw/jobs_raw.csv` under `DATA_DIR` from Chapter 2.

**Chapter 6** cleans: salary parsing, dedup, and a missing-value strategy for postings that hide pay.

**Chapter 7** EDA on the cleaned TalentLens postings: field coverage charts, geographic and remote splits.

**Chapter 24** turns what TalentLens can measure into career positioning, and separates it clearly from market judgement.

The schema normaliser in `ch04_data_sources.py` is not throwaway demo code. Chapter 5's `RemoteOKCollector._parse_job` returns the same keys. If you change field names here, update Chapter 5 before you collect.

---

## TalentLens checkpoint

You should have:

- [ ] Run `python book/ch04/ch04_data_sources.py` (or `--offline` in CI)
- [ ] Named all three allowed sources and none beyond them
- [ ] Memorised canonical schema fields and required trio (`title`, `company`, `description`)
- [ ] Understood why LinkedIn/Naukri/Indeed are out of scope
- [ ] Copied `.env.example` → `.env` if testing Adzuna live
- [ ] `pytest book/ch04/tests/test_ch04.py -q` passing

**Concepts you own:**

- Source choice is architecture: APIs and archives before scraping blocked front-ends
- One canonical row schema across heterogeneous JSON/CSV sources
- Ethics you can explain in an interview (what we collect, what we deliberately skip)

```bash
make test-ch04
python book/ch04/ch04_data_sources.py
python book/ch04/ch04_data_sources.py --offline
pytest book/ch04/tests/test_ch04.py -q
```

Part I (Chapters 1–4) is now complete: landscape, scaffold, statistics vocabulary, and data-source ethics.

Before Chapter 5: register Adzuna keys, confirm `DATA_DIR` exists, and delete any experimental `scrape_linkedin.py` from your machine (not from this repo's spine).
