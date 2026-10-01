#!/usr/bin/env python3
"""Chapter 4: TalentLens data sources - comparison, ethics, live RemoteOK sample."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

from talentlens.config import get_settings
from talentlens.paths import REPO_ROOT

logger = logging.getLogger(__name__)

REMOTEOK_API = "https://remoteok.com/api"


@dataclass(frozen=True)
class SourceSpec:
    """One row in the source comparison table."""

    name: str
    auth_required: str
    format: str
    rate_limit: str
    sweet_spot: str


def source_comparison_table() -> list[SourceSpec]:
    """Exactly the three sources Chapter 5 collects from."""
    return [
        SourceSpec(
            name="Adzuna API",
            auth_required="ADZUNA_APP_ID + ADZUNA_API_KEY (free tier)",
            format="JSON",
            rate_limit="~250 calls/day (free tier)",
            sweet_spot="Current Indian + global postings with salary fields",
        ),
        SourceSpec(
            name="RemoteOK API",
            auth_required="None",
            format="JSON",
            rate_limit="Be polite; no published hard cap",
            sweet_spot="Live remote tech roles, fast confidence check",
        ),
        SourceSpec(
            name="GitHub Jobs archive (Kaggle)",
            auth_required="Kaggle account for download",
            format="CSV",
            rate_limit="N/A (static file)",
            sweet_spot="~40k historical postings for classifier training",
        ),
    ]


def print_comparison_table() -> None:
    """Print the three-source comparison."""
    rows = source_comparison_table()
    print("\n--- TalentLens data sources (Chapter 5 will collect from these) ---")
    for row in rows:
        print(f"\n{row.name}")
        print(f"  Auth:        {row.auth_required}")
        print(f"  Format:      {row.format}")
        print(f"  Rate limit:  {row.rate_limit}")
        print(f"  Sweet spot:  {row.sweet_spot}")
    assert len(rows) == 3


def normalize_remoteok_posting(raw: dict[str, Any]) -> dict[str, Any]:
    """Map RemoteOK JSON to the canonical schema Chapter 5 uses."""
    tags = raw.get("tags", [])
    sal_min = raw.get("salary_min") or raw.get("salary")
    sal_max = raw.get("salary_max")
    try:
        sal_min_f = float(str(sal_min).replace(",", "")) if sal_min else None
        sal_max_f = float(str(sal_max).replace(",", "")) if sal_max else None
    except (ValueError, TypeError):
        sal_min_f = sal_max_f = None

    return {
        "job_id": f"remoteok_{raw.get('id', '')}",
        "source": "remoteok",
        "title": raw.get("position", "") or raw.get("title", ""),
        "company": raw.get("company", ""),
        "city": "Remote",
        "country": "Remote",
        "description": raw.get("description", "") or raw.get("text", ""),
        "skills_raw": ",".join(str(t) for t in tags) if tags else "",
        "salary_min": sal_min_f,
        "salary_max": sal_max_f,
        "currency": "USD",
        "is_remote": True,
        "posted_date": str(raw.get("date", "")),
        "url": raw.get("url", ""),
    }


def fetch_remoteok_sample(limit: int = 5) -> list[dict[str, Any]]:
    """Live fetch from RemoteOK - used by CLI, not by default in pytest."""
    headers = {"User-Agent": "TalentLens/1.0 (Data Voyage educational project)"}
    resp = requests.get(REMOTEOK_API, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    jobs = [j for j in data if isinstance(j, dict) and j.get("id")]
    return [normalize_remoteok_posting(j) for j in jobs[:limit]]


def try_adzuna_sample() -> str:
    """Optional Adzuna demo - only when credentials are set."""
    get_settings()  # load .env
    app_id = os.environ.get("ADZUNA_APP_ID")
    api_key = os.environ.get("ADZUNA_API_KEY")
    if not app_id or not api_key:
        return (
            "Adzuna live demo skipped — set ADZUNA_APP_ID and ADZUNA_API_KEY "
            "in .env to enable (Chapter 5 uses the same variables)."
        )
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": app_id,
        "app_key": api_key,
        "results_per_page": 3,
        "what": "machine learning",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    titles = [r.get("title", "?") for r in results[:3]]
    return f"Adzuna OK — sample titles: {titles}"


def main(argv: list[str] | None = None) -> int:
    """Run comparison table, optional live fetches, schema demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Chapter 4 data sources demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live RemoteOK fetch (use in CI or air-gapped runs)",
    )
    args = parser.parse_args(argv)

    print_comparison_table()

    print("\n--- Schema normalisation (RemoteOK-shaped record) ---")
    sample_raw = {
        "id": 999001,
        "position": "Senior ML Engineer",
        "company": "Example Remote Co",
        "tags": ["python", "pytorch"],
        "salary_min": 80000,
        "salary_max": 120000,
        "date": "2026-05-01",
        "url": "https://remoteok.com/example",
    }
    normalised = normalize_remoteok_posting(sample_raw)
    for key in ("source", "title", "company", "salary_min", "is_remote"):
        print(f"  {key}: {normalised[key]}")

    print(f"\n--- Adzuna ---\n{try_adzuna_sample()}")

    if not args.offline:
        logger.info("Fetching live RemoteOK sample (5 postings)...")
        try:
            postings = fetch_remoteok_sample(5)
        except requests.RequestException as exc:
            logger.error("RemoteOK fetch failed: %s", exc)
            return 1
        print("\n--- Live RemoteOK sample ---")
        for i, job in enumerate(postings, 1):
            print(f"{i}. {job['title']} @ {job['company']} ({job['city']})")
    else:
        print("\n(Offline mode — skipped live RemoteOK fetch.)")

    print(f"\nRepo root: {REPO_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
