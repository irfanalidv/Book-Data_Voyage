"""
Chapter 5: Data Collection - Building the TalentLens Dataset
Data Voyage - Building TalentLens

TalentLens milestone: go from zero data to a raw dataset of real job
postings collected from public APIs and curated open datasets.
Output feeds every downstream chapter.

Run (demo - no API keys needed):
    python book/ch05/ch05_data_collection.py

Run (live - requires ADZUNA_APP_ID + ADZUNA_API_KEY in .env):
    python book/ch05/ch05_data_collection.py --live

Outputs:
    data/raw/jobs_raw.demo.csv   (demo mode - does not touch bundled jobs_raw.csv)
    data/raw/jobs_raw.csv        (--live mode only)
    book/ch05/reports/figures/ch05_collection_funnel.png
    book/ch05/reports/figures/ch05_source_breakdown.png
    book/ch05/reports/figures/ch05_field_coverage.png
    book/ch05/reports/collection_summary.md
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent


def _display_path(path: Path | str) -> str:
    """Show *path* relative to the repository root, so reports carry no home directory."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Output paths - demo mode writes *.demo.csv so bundled raw/clean stay immutable.
    raw_data_path: Path = _REPO_ROOT / "data" / "raw" / "jobs_raw.csv"
    raw_demo_data_path: Path = _REPO_ROOT / "data" / "raw" / "jobs_raw.demo.csv"
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"

    # Collection targets
    adzuna_target: int = 500  # postings to collect from Adzuna
    remoteok_target: int = 300  # postings from RemoteOK
    demo_target: int = 600  # synthetic records in demo mode

    # API settings
    adzuna_app_id: str = field(default_factory=lambda: os.getenv("ADZUNA_APP_ID", ""))
    adzuna_api_key: str = field(default_factory=lambda: os.getenv("ADZUNA_API_KEY", ""))
    adzuna_results_per_page: int = 50
    request_timeout: int = 10
    requests_per_minute: int = 30  # conservative rate limit

    # Demo mode
    demo_mode: bool = True


# ---------------------------------------------------------------------------
# Canonical schema - every source maps to this
# ---------------------------------------------------------------------------

# All collected jobs conform to this structure.
# None = field is optional; str = required.
SCHEMA: dict[str, type] = {
    "job_id": str,  # source-specific ID
    "source": str,  # "adzuna" | "remoteok" | "kaggle" | "demo"
    "title": str,  # job title (required)
    "company": str,  # company name (required)
    "city": str,  # city or "Remote"
    "country": str,  # ISO country code
    "description": str,  # full job description (required)
    "skills_raw": str,  # comma-separated skills if provided by source
    "salary_min": float,  # annual, local currency
    "salary_max": float,  # annual, local currency
    "currency": str,  # "INR" | "USD" | "GBP" | etc.
    "is_remote": bool,
    "posted_date": str,  # ISO date string
    "url": str,  # original posting URL
}

REQUIRED_FIELDS = {"title", "description", "company"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_job(job: dict) -> bool:
    """Return True if job record has all required non-empty fields.

    Args:
        job: Raw job dict from any collector.

    Returns:
        True if all REQUIRED_FIELDS are present and non-empty.
    """
    return all(job.get(f) and str(job[f]).strip() for f in REQUIRED_FIELDS)


def coerce_to_schema(job: dict) -> dict:
    """Coerce a raw job dict to the canonical schema with correct types.

    Missing optional fields are filled with None. Types are coerced where
    safe (e.g., int salary to float). Values that can't be coerced are
    set to None rather than raising.

    Args:
        job: Raw job dict (may have extra or missing keys).

    Returns:
        Dict conforming to SCHEMA (all keys present, correct types or None).
    """
    result: dict = {}
    for key, typ in SCHEMA.items():
        val = job.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            result[key] = None
            continue
        try:
            if typ is bool:
                result[key] = bool(val)
            elif typ is float:
                result[key] = float(val)
            else:
                result[key] = str(val).strip()
        except (ValueError, TypeError):
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# Base collector
# ---------------------------------------------------------------------------


class BaseCollector:
    """Abstract base for all job data collectors.

    Subclasses implement `_fetch_page` and `collect`.
    """

    SOURCE_NAME: str = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "TalentLens/1.0 (educational data science project; github.com/irfanalidv)",
                "Accept": "application/json",
            }
        )
        self._request_interval = 60 / max(cfg.requests_per_minute, 1)

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        """Make a rate-limited GET request with retry on 429 and 5xx.

        Args:
            url: Full URL to request.
            params: Query parameters dict.

        Returns:
            Response object, or None if all retries failed.
        """
        for attempt in range(3):
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.request_timeout)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = 2**attempt
                    logger.warning(f"Server error {resp.status_code} — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(self._request_interval)
                return resp
            except requests.RequestException as exc:
                wait = 2**attempt
                logger.warning(f"Request failed ({exc}) — retrying in {wait}s")
                time.sleep(wait)
        logger.error(f"All retries failed for {url}")
        return None

    def collect(self, target: int) -> list[dict]:
        """Collect up to `target` job postings from this source.

        Args:
            target: Maximum number of postings to collect.

        Returns:
            List of job dicts conforming to SCHEMA.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Adzuna collector
# ---------------------------------------------------------------------------


class AdzunaCollector(BaseCollector):
    """Collect job postings from the Adzuna Jobs API.

    Free tier: 250 calls/day. Register at developer.adzuna.com.
    Focuses on India tech jobs (endpoint: /api/v1/jobs/in/search).

    Args:
        cfg: Config with adzuna_app_id, adzuna_api_key.
    """

    SOURCE_NAME = "adzuna"
    BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search"
    CATEGORIES = ["it-jobs", "engineering-jobs", "scientific-qa-jobs"]

    def collect(self, target: int) -> list[dict]:
        if not self.cfg.adzuna_app_id or not self.cfg.adzuna_api_key:
            logger.warning(
                "Adzuna API keys not set — skipping (set ADZUNA_APP_ID and ADZUNA_API_KEY)"
            )
            return []

        collected: list[dict] = []
        per_category = max(1, target // len(self.CATEGORIES))

        for category in self.CATEGORIES:
            if len(collected) >= target:
                break
            category_collected = self._collect_category(category, per_category)
            collected.extend(category_collected)
            logger.info(f"Adzuna [{category}]: {len(category_collected)} postings")

        return collected[:target]

    def collect_by_keyword(
        self,
        keyword: str,
        target_rows: int,
        category: str = "it-jobs",
    ) -> list[dict]:
        """Collect postings matching a keyword (Adzuna ``what`` query param).

        Paginates until ``target_rows`` reached or no more results. Returns
        validated records in the canonical schema.

        Args:
            keyword: Adzuna ``what`` search string.
            target_rows: Maximum postings to return.
            category: Adzuna category filter (default ``it-jobs``).

        Returns:
            List of job dicts conforming to SCHEMA.
        """
        if not self.cfg.adzuna_app_id or not self.cfg.adzuna_api_key:
            logger.warning("Adzuna API keys not set — skipping collect_by_keyword")
            return []

        results: list[dict] = []
        page = 1
        per_page = self.cfg.adzuna_results_per_page
        max_pages = (target_rows // per_page) + 2

        while len(results) < target_rows and page <= max_pages:
            resp = self._get(
                f"{self.BASE_URL}/{page}",
                params={
                    "app_id": self.cfg.adzuna_app_id,
                    "app_key": self.cfg.adzuna_api_key,
                    "results_per_page": per_page,
                    "what": keyword,
                    "category": category,
                    "content-type": "application/json",
                },
            )
            if resp is None:
                logger.warning(f"  Adzuna keyword={keyword!r} page {page}: no response, stopping")
                break

            data = resp.json()
            page_results = data.get("results", [])
            if not page_results:
                logger.info(f"  Adzuna keyword={keyword!r} page {page}: no more results")
                break

            before = len(results)
            for raw in page_results:
                parsed = self._parse_job(raw)
                if validate_job(parsed):
                    results.append(coerce_to_schema(parsed))
                if len(results) >= target_rows:
                    break

            logger.info(
                f"  Adzuna keyword={keyword!r} page {page}: "
                f"+{len(results) - before} (total {len(results)})"
            )
            page += 1

        return results[:target_rows]

    def _collect_category(self, category: str, target: int) -> list[dict]:
        results: list[dict] = []
        page = 1
        max_pages = (target // self.cfg.adzuna_results_per_page) + 2

        while len(results) < target and page <= max_pages:
            resp = self._get(
                f"{self.BASE_URL}/{page}",
                params={
                    "app_id": self.cfg.adzuna_app_id,
                    "app_key": self.cfg.adzuna_api_key,
                    "results_per_page": self.cfg.adzuna_results_per_page,
                    "category": category,
                    "content-type": "application/json",
                },
            )
            if resp is None:
                break

            data = resp.json()
            jobs = data.get("results", [])
            if not jobs:
                break

            for job in jobs:
                parsed = self._parse_job(job)
                if validate_job(parsed):
                    results.append(coerce_to_schema(parsed))

            page += 1

        return results

    def _parse_job(self, raw: dict) -> dict:
        """Map Adzuna API response to canonical schema."""
        salary_min = raw.get("salary_min")
        salary_max = raw.get("salary_max")
        return {
            "job_id": f"adzuna_{raw.get('id', '')}",
            "source": "adzuna",
            "title": raw.get("title", ""),
            "company": (
                raw.get("company", {}).get("display_name", "")
                if isinstance(raw.get("company"), dict)
                else str(raw.get("company", ""))
            ),
            "city": (
                raw.get("location", {}).get("display_name", "")
                if isinstance(raw.get("location"), dict)
                else ""
            ),
            "country": "IN",
            "description": raw.get("description", ""),
            "skills_raw": "",
            "salary_min": float(salary_min) if salary_min else None,
            "salary_max": float(salary_max) if salary_max else None,
            "currency": "INR",
            "is_remote": "remote" in str(raw.get("title", "")).lower()
            or "remote" in str(raw.get("description", "")).lower(),
            "posted_date": raw.get("created", ""),
            "url": raw.get("redirect_url", ""),
        }


# ---------------------------------------------------------------------------
# RemoteOK collector
# ---------------------------------------------------------------------------


class RemoteOKCollector(BaseCollector):
    """Collect remote tech job postings from remoteok.com public API.

    No authentication required. Returns JSON array.
    Endpoint: https://remoteok.com/api

    Args:
        cfg: Config.
    """

    SOURCE_NAME = "remoteok"
    API_URL = "https://remoteok.com/api"

    def collect(self, target: int) -> list[dict]:
        logger.info("RemoteOK: fetching all postings (single endpoint)...")
        resp = self._get(self.API_URL)
        if resp is None:
            logger.error("RemoteOK: fetch failed")
            return []

        try:
            raw_list = resp.json()
        except Exception:
            logger.error("RemoteOK: JSON parse failed")
            return []

        # First element is metadata, skip it
        jobs_raw = [j for j in raw_list if isinstance(j, dict) and j.get("id")]

        results: list[dict] = []
        for raw in jobs_raw[: target * 2]:  # over-fetch to account for invalid
            parsed = self._parse_job(raw)
            if validate_job(parsed):
                results.append(coerce_to_schema(parsed))
            if len(results) >= target:
                break

        logger.info(f"RemoteOK: {len(results)} valid postings collected")
        return results

    def _parse_job(self, raw: dict) -> dict:
        """Map RemoteOK API response to canonical schema."""
        tags = raw.get("tags", [])
        sal_min = raw.get("salary_min") or raw.get("salary")
        sal_max = raw.get("salary_max")

        # RemoteOK salaries are in USD
        try:
            sal_min_f = float(str(sal_min).replace(",", "")) if sal_min else None
            sal_max_f = float(str(sal_max).replace(",", "")) if sal_max else None
        except (ValueError, TypeError):
            sal_min_f = sal_max_f = None

        return {
            "job_id": f"remoteok_{raw.get('id', '')}",
            "source": "remoteok",
            "title": raw.get("position", ""),
            "company": raw.get("company", ""),
            "city": "Remote",
            "country": "Remote",
            "description": raw.get("description", "") or raw.get("text", ""),
            "skills_raw": ",".join(str(t) for t in tags) if tags else "",
            "salary_min": sal_min_f,
            "salary_max": sal_max_f,
            "currency": "USD",
            "is_remote": True,
            "posted_date": raw.get("date", ""),
            "url": raw.get("url", ""),
        }


# ---------------------------------------------------------------------------
# Demo data generator (no API keys needed)
# ---------------------------------------------------------------------------


class DemoCollector(BaseCollector):
    """Generate realistic synthetic job postings for demo mode.

    Produces data that matches the canonical schema and behaves like a real
    scrape in the ways the later chapters depend on:

    - Titles vary within a role ("ML Engineer", "Machine Learning Engineer",
      "MLOps Engineer") and carry a seniority prefix.
    - Salary depends on role *and* seniority, with log-normal noise - so pay
      is right-skewed and overlaps across roles.
    - Descriptions and skills are sampled from role phrase banks with
      deliberate overlap (ML and AI Engineers both "deploy models"; Data
      Scientists sometimes mention Spark), so role classification is
      learnable but not trivial.
    - Senior roles are more likely to be remote - the confound Chapter 8
      tests for.
    - About one posting in five hides its salary.

    Every draw comes from one seeded generator, so a fresh clone reproduces
    the bundled ``data/raw/jobs_raw.csv`` byte for byte.

    DEMO DATA - the employers are fictional; replace with the live
    collectors for real analysis.
    """

    SOURCE_NAME = "demo"

    # role -> title variants, mid-level annual salary (INR), core skills,
    # responsibility phrases, and the share of postings for this role.
    _ROLES = {
        "AI Engineer": {
            "titles": ["AI Engineer", "GenAI Engineer", "LLM Engineer", "Applied AI Engineer"],
            "base": 2_000_000,
            "weight": 0.16,
            "skills": [
                "Python",
                "LLMs",
                "RAG",
                "FastAPI",
                "PostgreSQL",
                "Docker",
                "NLP",
                "PyTorch",
            ],
            "phrases": [
                "build retrieval-augmented generation pipelines over internal documents",
                "ship LLM features behind well-tested APIs",
                "evaluate prompt and retrieval quality with offline test sets",
                "design agent workflows that call internal tools",
                "own latency and token cost for our generative features",
                "integrate vector search with our product database",
            ],
        },
        "ML Engineer": {
            "titles": ["ML Engineer", "Machine Learning Engineer", "MLOps Engineer"],
            "base": 1_800_000,
            "weight": 0.22,
            "skills": [
                "Python",
                "scikit-learn",
                "PyTorch",
                "MLflow",
                "Docker",
                "Kubernetes",
                "Airflow",
                "Machine Learning",
            ],
            "phrases": [
                "train and deploy models to production",
                "build feature pipelines and model monitoring",
                "own the model registry and experiment tracking",
                "reduce inference latency for ranking models",
                "set up CI/CD for model training and release",
                "detect drift and retrain models on schedule",
            ],
        },
        "Data Scientist": {
            "titles": ["Data Scientist", "Research Scientist", "Data Scientist (Product)"],
            "base": 1_600_000,
            "weight": 0.24,
            "skills": [
                "Python",
                "SQL",
                "Statistics",
                "pandas",
                "scikit-learn",
                "Machine Learning",
                "NumPy",
            ],
            "phrases": [
                "design and analyse A/B tests",
                "build statistical models for pricing and churn",
                "translate business questions into experiments",
                "present findings to product and leadership",
                "develop forecasting models for demand planning",
                "explore customer data to find growth levers",
            ],
        },
        "Data Engineer": {
            "titles": ["Data Engineer", "Analytics Engineer", "ETL Developer"],
            "base": 1_600_000,
            "weight": 0.16,
            "skills": [
                "Python",
                "SQL",
                "Spark",
                "Airflow",
                "dbt",
                "PostgreSQL",
                "Cloud (AWS)",
                "Cloud (GCP)",
            ],
            "phrases": [
                "build and maintain batch and streaming data pipelines",
                "model the warehouse for analytics teams",
                "own data quality checks and pipeline alerting",
                "migrate legacy ETL jobs to a modern orchestration stack",
                "optimise Spark jobs for cost and runtime",
                "manage schemas and late-arriving data",
            ],
        },
        "Data Analyst": {
            "titles": ["Data Analyst", "Business Analyst", "BI Analyst"],
            "base": 800_000,
            "weight": 0.12,
            "skills": ["SQL", "Statistics", "pandas", "Python"],
            "phrases": [
                "build dashboards in Tableau and Power BI",
                "answer ad-hoc business questions with SQL",
                "track weekly KPIs for the operations team",
                "clean spreadsheets and automate recurring reports",
                "support product managers with funnel analysis",
            ],
        },
        "Other": {
            "titles": ["NLP Engineer", "Computer Vision Engineer", "Backend Engineer"],
            "base": 1_500_000,
            "weight": 0.10,
            "skills": ["Python", "Docker", "PostgreSQL", "FastAPI", "Deep Learning", "NLP"],
            "phrases": [
                "build backend services in Python",
                "train deep learning models for text and images",
                "maintain REST APIs used by mobile apps",
                "work with the ML team on model serving",
            ],
        },
    }

    # Neighbouring roles whose work genuinely overlaps. Real job ads blur these
    # lines ("the title is marketing; the JD is the spec" - Chapter 1).
    _ADJACENT = {
        "AI Engineer": ["ML Engineer", "Other"],
        "ML Engineer": ["AI Engineer", "Data Scientist", "Data Engineer"],
        "Data Scientist": ["Data Analyst", "ML Engineer"],
        "Data Engineer": ["ML Engineer", "Data Analyst"],
        "Data Analyst": ["Data Scientist", "Data Engineer"],
        "Other": ["AI Engineer", "ML Engineer"],
    }
    _P_BODY_FROM_ADJACENT = 0.12  # whole description written like a neighbouring role
    _P_PHRASE_FROM_ADJACENT = 0.30  # any single duty borrowed from a neighbour

    # Cross-role phrases: the overlap that makes classification non-trivial.
    _SHARED_PHRASES = [
        "deploy models to production",
        "write clean, tested Python",
        "work closely with product managers",
        "document decisions for the wider team",
        "use SQL daily",
        "collaborate with data scientists and engineers",
        "work with large language models where they help",
        "run experiments and measure impact",
        "use Spark for large datasets",
    ]
    _SHARED_SKILLS = [
        "Python",
        "SQL",
        "Git",
        "Docker",
        "Cloud (AWS)",
        "Cloud (Azure)",
        "Cloud (GCP)",
        "Machine Learning",
        "Spark",
        "LLMs",
        "pandas",
    ]

    # seniority prefix, share of postings, salary multiplier, P(remote)
    _SENIORITY = [
        ("Junior", 0.20, 0.60, 0.15),
        ("", 0.35, 1.00, 0.30),
        ("Senior", 0.30, 1.45, 0.50),
        ("Lead", 0.15, 1.90, 0.60),
    ]

    # Fictional employers - the demo corpus is synthetic, so it must not attach
    # invented salaries to real companies.
    _COMPANIES = [
        "Nimbus Fintech",
        "Kestrel Commerce",
        "Monsoon Payments",
        "Banyan Health",
        "Indigo Logistics",
        "Saffron Credit",
        "Teal Analytics",
        "Deccan Mobility",
        "Lotus Insuretech",
        "Peacock Media",
        "Konark Robotics",
        "Coral Edtech",
        "Anchor Lending",
        "Orbit SaaS",
        "Vindhya Energy",
        "Himalaya Cloud",
        "Tamarind Foods",
        "Sahyadri Games",
        "Kaveri Retail",
        "Palash Travel",
        "Remote AI Startup",
        "Global ML Team",
        "Series B Fintech",
        "AI Consultancy",
    ]

    _CITIES = ["Bangalore", "Mumbai", "Hyderabad", "Delhi NCR", "Pune", "Chennai"]
    _CITY_WEIGHTS = [0.38, 0.17, 0.17, 0.15, 0.08, 0.05]

    def collect(self, target: int) -> list[dict]:
        rng = np.random.default_rng(seed=42)
        roles = list(self._ROLES)
        role_p = np.array([self._ROLES[r]["weight"] for r in roles])
        role_p = role_p / role_p.sum()
        sen_p = np.array([s[1] for s in self._SENIORITY])
        results: list[dict] = []

        for i in range(target):
            role = roles[rng.choice(len(roles), p=role_p)]
            spec = self._ROLES[role]
            prefix, _, sal_mult, p_remote = self._SENIORITY[
                rng.choice(len(self._SENIORITY), p=sen_p)
            ]
            base_title = spec["titles"][rng.integers(0, len(spec["titles"]))]
            title = f"{prefix} {base_title}".strip()
            company = self._COMPANIES[rng.integers(0, len(self._COMPANIES))]
            is_remote = bool(rng.random() < p_remote)
            city = (
                "Remote"
                if is_remote and rng.random() < 0.5
                else self._CITIES[rng.choice(len(self._CITIES), p=self._CITY_WEIGHTS)]
            )

            # Salary: role base x seniority x log-normal noise (right-skewed).
            noise = float(np.clip(rng.lognormal(mean=0.0, sigma=0.30), 0.5, 3.0))
            midpoint = spec["base"] * sal_mult * noise
            sal_min: float | None = round(midpoint * 0.85, -4)
            sal_max: float | None = round(midpoint * 1.15, -4)
            if rng.random() < 0.20:  # about one in five postings hides pay
                sal_min = sal_max = None

            # Some postings describe a neighbouring role's work under this title.
            neighbours = self._ADJACENT[role]
            body_role = role
            if rng.random() < self._P_BODY_FROM_ADJACENT:
                body_role = neighbours[rng.integers(0, len(neighbours))]
            body = self._ROLES[body_role]

            # Skills: 3-5 core skills for the body's role, plus 0-2 shared ones.
            n_core = int(rng.integers(3, min(6, len(body["skills"]) + 1)))
            core = list(rng.choice(body["skills"], size=n_core, replace=False))
            n_shared = int(rng.integers(0, 3))
            shared = list(rng.choice(self._SHARED_SKILLS, size=n_shared, replace=False))
            skills = list(dict.fromkeys(core + shared))

            # Description: 2 role phrases, 1-2 shared phrases, in shuffled order.
            duties = list(rng.choice(body["phrases"], size=2, replace=False))
            for j in range(len(duties)):
                if rng.random() < self._P_PHRASE_FROM_ADJACENT:
                    other = self._ROLES[neighbours[rng.integers(0, len(neighbours))]]
                    duties[j] = str(rng.choice(other["phrases"]))
            duties = list(dict.fromkeys(duties))
            duties += list(
                rng.choice(self._SHARED_PHRASES, size=int(rng.integers(1, 3)), replace=False)
            )
            duties = list(dict.fromkeys(duties))
            rng.shuffle(duties)
            location = "This role is remote-friendly." if is_remote else f"Based in {city}."
            article = "an" if title[0] in "AEIOU" else "a"
            desc = (
                f"We are hiring {article} {title} at {company}. "
                f"You will {duties[0]}, {duties[1]}"
                + (f", and {duties[2]}. " if len(duties) > 2 else ". ")
                + (f"Day to day you will also {duties[3]}. " if len(duties) > 3 else "")
                + f"Tools you will use: {', '.join(skills)}. "
                + f"{location} Competitive pay and learning budget."
            )

            record = {
                "job_id": f"demo_{i:05d}",
                "source": "demo",
                "title": title,
                "company": company,
                "city": city,
                "country": "IN",
                "description": desc,
                "skills_raw": ",".join(skills),
                "salary_min": sal_min,
                "salary_max": sal_max,
                "currency": "INR",
                "is_remote": is_remote,
                "posted_date": "2026-01-01",
                "url": f"https://example.com/jobs/{i}",
            }
            results.append(coerce_to_schema(record))

        logger.info(f"Demo: generated {len(results)} synthetic job postings")
        return results


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DataCollectionPipeline:
    """Orchestrates all collectors, merges, deduplicates, and saves.

    Args:
        cfg: Config.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.cfg.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self, live: bool = False) -> dict[str, int]:
        """Run the full collection pipeline.

        Args:
            live: If True, use real API collectors. If False, demo mode.

        Returns:
            Dict with collection statistics.
        """
        stats: dict[str, dict] = {}
        all_records: list[dict] = []

        if live:
            collectors_and_targets = [
                (AdzunaCollector(self.cfg), self.cfg.adzuna_target),
                (RemoteOKCollector(self.cfg), self.cfg.remoteok_target),
            ]
        else:
            collectors_and_targets = [
                (DemoCollector(self.cfg), self.cfg.demo_target),
            ]

        for collector, target in collectors_and_targets:
            source = collector.SOURCE_NAME
            logger.info(f"\n[Collecting] {source} (target: {target})...")
            records = collector.collect(target)
            valid = [r for r in records if validate_job(r)]
            stats[source] = {"collected": len(records), "valid": len(valid)}
            all_records.extend(valid)
            logger.info(f"  {source}: {len(records)} collected, {len(valid)} valid")

        # Deduplication: title + company + city fingerprint
        seen: set[str] = set()
        unique_records: list[dict] = []
        for rec in all_records:
            key = f"{str(rec.get('title','')).lower().strip()}|{str(rec.get('company','')).lower().strip()}|{str(rec.get('city','')).lower().strip()}"
            if key not in seen:
                seen.add(key)
                unique_records.append(rec)

        duplicates_removed = len(all_records) - len(unique_records)

        # Save to CSV - demo mode uses a separate file so bundled jobs_raw.csv is untouched.
        output_path = self.cfg.raw_data_path if live else self.cfg.raw_demo_data_path
        if unique_records:
            df = pd.DataFrame(unique_records)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"\nSaved: {output_path} ({file_size_mb:.1f}MB, {len(unique_records):,} rows)"
            )
            if not live:
                logger.info(
                    "Bundled jobs_raw.csv unchanged — demo output is for practice only; "
                    "downstream chapters read the committed clean dataset."
                )
        else:
            df = pd.DataFrame()
            logger.warning("No records collected.")

        return {
            "stats": stats,
            "total_raw": len(all_records),
            "total_valid": len(unique_records),
            "duplicates_removed": duplicates_removed,
            "output_path": output_path,
        }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_collection_funnel(results: dict, cfg: Config) -> Path:
    """Waterfall chart: collected → valid → after dedup."""
    total_raw = results["total_raw"]
    total_valid = results["total_valid"]
    dupes = results["duplicates_removed"]

    stages = ["Collected\n(raw)", "Valid\n(schema check)", "After\ndedup"]
    values = [total_raw, total_raw - (total_raw - total_valid), total_valid]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    subtitle = f"({dupes:,} duplicates removed)"

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", width=0.5, alpha=0.85)
    ax.set_title(f"TalentLens Data Collection Funnel\n{subtitle}", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:,}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Attrition arrows
    for i in range(len(stages) - 1):
        drop = values[i] - values[i + 1]
        if drop > 0:
            ax.annotate(
                f"−{drop:,}",
                xy=(i + 0.75, values[i + 1] + (values[i] - values[i + 1]) / 2),
                fontsize=9,
                color="#F44336",
                ha="center",
            )

    ax.set_ylabel("Number of job postings", fontsize=12)
    ax.set_title("TalentLens Data Collection Funnel", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    out = cfg.figures_dir / "ch05_collection_funnel.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_source_breakdown(results: dict, cfg: Config) -> Path:
    """Horizontal bars: valid postings per source, with each source's share."""
    stats = results["stats"]
    if not stats:
        logger.warning("No stats to plot source breakdown")
        return cfg.figures_dir / "ch05_source_breakdown.png"

    labels = sorted(stats, key=lambda s: stats[s]["valid"])
    sizes = [stats[s]["valid"] for s in labels]
    total = sum(sizes) or 1
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(labels) + 1.3))
    bars = ax.barh(labels, sizes, color=[colors[i % len(colors)] for i in range(len(labels))])
    for bar, n in zip(bars, sizes):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {n:,} ({n / total:.0%})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(sizes) * 1.25)
    ax.set_xlabel("Valid postings", fontsize=9)
    ax.set_title("Job postings by source", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = cfg.figures_dir / "ch05_source_breakdown.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_field_coverage(cfg: Config, raw_path: Path | None = None) -> Path:
    """Horizontal bar chart: % non-null per field in the collected dataset."""
    path = raw_path or cfg.raw_data_path
    if not path.exists():
        logger.warning("Raw data not found — skipping field coverage chart")
        return cfg.figures_dir / "ch05_field_coverage.png"

    df = pd.read_csv(path)
    coverage = pd.Series(
        {
            col: (df[col].notna() & (df[col].astype(str).str.strip() != "")).mean() * 100
            for col in df.columns
        }
    )

    # Only show fields in SCHEMA
    schema_fields = [f for f in SCHEMA if f in coverage.index]
    coverage = coverage[schema_fields].sort_values(ascending=True)

    colors = [
        "#4CAF50" if v >= 90 else "#FF9800" if v >= 60 else "#F44336" for v in coverage.values
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(coverage)), coverage.values, color=colors, edgecolor="white", alpha=0.85
    )
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage.index, fontsize=10)
    ax.set_xlabel("% of rows with non-null value", fontsize=12)
    ax.set_title("Field Coverage — TalentLens Raw Dataset", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, coverage.values):
        ax.text(
            val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%", va="center", fontsize=9
        )

    ax.axvline(90, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.axvline(60, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(91, -0.5, "90%\ntarget", fontsize=8, color="#4CAF50")

    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", alpha=0.7, label="≥90% — good coverage"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.7, label="60–89% — imputation needed"),
        mpatches.Patch(facecolor="#F44336", alpha=0.7, label="<60% — sparse field"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlim(0, 108)
    plt.tight_layout()
    out = cfg.figures_dir / "ch05_field_coverage.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_collection_summary(results: dict, cfg: Config) -> Path:
    """Write a Markdown collection summary report.

    Args:
        results: Output from DataCollectionPipeline.run().
        cfg: Config with reports_dir and raw_data_path.

    Returns:
        Path to saved report.
    """
    stats = results["stats"]
    total_raw = results["total_raw"]
    total_valid = results["total_valid"]
    dupes = results["duplicates_removed"]

    # Field coverage from saved CSV
    output_path = results.get("output_path", cfg.raw_data_path)
    coverage_section = ""
    if output_path.exists():
        df = pd.read_csv(output_path)
        coverage = pd.Series(
            {
                col: (df[col].notna() & (df[col].astype(str).str.strip() != "")).mean() * 100
                for col in df.columns
            }
        )
        coverage_section = "\n## Field coverage\n\n| Field | Coverage |\n|-------|----------|\n"
        for field, pct in coverage.items():
            if field in SCHEMA:
                icon = "✅" if pct >= 90 else "⚠️" if pct >= 60 else "❌"
                coverage_section += f"| `{field}` | {icon} {pct:.0f}% |\n"

    source_rows = "\n".join(
        f"| {src} | {s['collected']:,} | {s['valid']:,} | {s['valid']/max(s['collected'],1)*100:.1f}% |"
        for src, s in stats.items()
    )

    output_path = results.get("output_path", cfg.raw_data_path)
    file_size = ""
    if output_path.exists():
        mb = output_path.stat().st_size / (1024 * 1024)
        file_size = f"{mb:.1f}MB"

    report = f"""# TalentLens Data Collection Summary

## Overview
- **Total collected:** {total_raw:,}
- **Valid (schema check):** {total_valid + dupes:,}
- **After deduplication:** {total_valid:,}
- **Duplicates removed:** {dupes:,} ({dupes/max(total_raw,1)*100:.1f}%)
- **Output file:** `{_display_path(output_path)}` ({file_size})

## By source

| Source | Collected | Valid | Rate |
|--------|-----------|-------|------|
{source_rows}
| **Total** | **{total_raw:,}** | **{total_valid:,}** | **{total_valid/max(total_raw,1)*100:.1f}%** |

{coverage_section}

## Next step

Run Chapter 6 to clean this dataset:
```bash
python book/ch06/ch06_data_cleaning_preprocessing.py
```

Outputs: `data/clean/jobs_clean.csv` — ready for EDA (Ch7) and ML (Ch9).
"""

    out = cfg.reports_dir / "collection_summary.md"
    out.write_text(report, encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_block(title: str, lines: list[str]) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title}\n{sep}")
    for line in lines:
        logger.info(f"  {line}")


def _load_dotenv() -> None:
    """Load .env from repo root if present (does not override shell env)."""
    dotenv_path = _REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path, override=False)
    except ImportError:
        # Manual fallback
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _load_dotenv()

    live = "--live" in sys.argv
    cfg = Config(demo_mode=not live)

    logger.info("=" * 60)
    logger.info("  CHAPTER 5: DATA COLLECTION")
    logger.info("  TalentLens — Building the Job Postings Dataset")
    logger.info(f"  Mode: {'live (real APIs)' if live else 'demo (synthetic data)'}")
    logger.info("=" * 60)

    logger.info("\n[1/5] Running collection pipeline...")
    pipeline = DataCollectionPipeline(cfg)
    results = pipeline.run(live=live)

    stats = results["stats"]
    total_raw = results["total_raw"]
    total_valid = results["total_valid"]
    dupes = results["duplicates_removed"]

    _print_block(
        "COLLECTION SUMMARY",
        [
            f"{'Source':<20} {'Collected':>10} {'Valid':>8} {'Rate':>8}",
            "─" * 50,
            *[
                f"{src:<20} {s['collected']:>10,} {s['valid']:>8,} {s['valid']/max(s['collected'],1)*100:>7.1f}%"
                for src, s in stats.items()
            ],
            "─" * 50,
            f"{'Total raw':<20} {total_raw:>10,}",
            f"{'After dedup':<20} {total_valid:>10,} ({dupes:,} duplicates removed)",
        ],
    )

    logger.info("\n[2/5] Generating collection funnel chart...")
    plot_collection_funnel(results, cfg)

    logger.info("\n[3/5] Generating source breakdown chart...")
    plot_source_breakdown(results, cfg)

    logger.info("\n[4/5] Generating field coverage chart...")
    plot_field_coverage(cfg, raw_path=results.get("output_path"))

    logger.info("\n[5/5] Writing collection summary...")
    write_collection_summary(results, cfg)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 5 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Raw data: {results.get('output_path', cfg.raw_data_path)}")
    logger.info(f"  Figures:  {cfg.figures_dir}/")
    logger.info(f"  Report:   {cfg.reports_dir}/collection_summary.md")
    if not live:
        logger.info("\n  Running in demo mode — data is synthetic.")
        logger.info("  For real data: register at developer.adzuna.com (free)")
        logger.info("  Then: export ADZUNA_APP_ID=... ADZUNA_API_KEY=...")
        logger.info("        python book/ch05/ch05_data_collection.py --live")
    logger.info("\nNext: Chapter 6 — Data Cleaning")
    logger.info("  python book/ch06/ch06_data_cleaning_preprocessing.py")


if __name__ == "__main__":
    main()
