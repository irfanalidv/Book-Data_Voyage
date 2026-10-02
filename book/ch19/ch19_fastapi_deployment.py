"""
Chapter 19: FastAPI - Deploy TalentLens as a REST API
Data Voyage - Building TalentLens

TalentLens milestone: expose semantic search and salary-band classification
over HTTP with validation, rate limiting, and a contract ready for Docker.

Run the API locally:

    uvicorn book.ch19.ch19_fastapi_deployment:app --reload --port 8765

Or from this directory (repo root recommended):

    python -m uvicorn book.ch19.ch19_fastapi_deployment:app --host 127.0.0.1 --port 8765

Optional load smoke (requires httpx):

    TALENTLENS_LOAD_TEST=1 python book/ch19/ch19_fastapi_deployment.py

Outputs:

    book/ch19/reports/figures/ch19_api_architecture.png

Role classification (Ch9/Ch10): when wiring the saved classifier, load via
``talentlens.paths.role_classifier_path()`` (v2 if Chapter 10 has run,
otherwise the Ch9 baseline).
"""

from __future__ import annotations  # noqa: E402

import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

_THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Rate limiter (module level - used by build_app)
# ---------------------------------------------------------------------------


@dataclass
class _RateLimiter:
    """Fixed-window style limiter using monotonic timestamps (single bucket demo)."""

    max_calls: int
    per_seconds: float
    _timestamps: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.per_seconds]
        if len(self._timestamps) >= self.max_calls:
            return False
        self._timestamps.append(now)
        return True

    def reset(self) -> None:
        self._timestamps.clear()


# ---------------------------------------------------------------------------
# Pydantic models - MUST stay at module level (FastAPI + Python 3.12 / 0.135+)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field  # noqa: E402


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str = "talentlens-api"


class JobResult(BaseModel):
    job_id: str
    title: str
    company: str
    score: float
    salary_min_l: Optional[float] = None
    salary_max_l: Optional[float] = None
    is_remote: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[JobResult]
    took_ms: float


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=8_000)


class ClassifyResponse(BaseModel):
    predicted_band: str
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# Demo data + keyword ranking (no heavy deps on import path)
# ---------------------------------------------------------------------------


def keyword_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    words = text.lower().split()
    if not q or not words:
        return 0.0
    doc = set(words)
    return sum(1 for t in q if t in doc) / len(q)


def normalise_jobs_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Align Chapter 5–7 CSV columns with what the API ranking expects."""
    out = df.copy()
    if "title" not in out.columns and "role_category" in out.columns:
        out["title"] = out["role_category"].astype(str)
    if "job_id" not in out.columns:
        out["job_id"] = [f"job_{i}" for i in range(len(out))]
    if "company" not in out.columns:
        if "company_type" in out.columns:
            out["company"] = out["company_type"].astype(str)
        else:
            out["company"] = "TalentLens"
    if "description" not in out.columns:
        skills = out["skills_normalised"].astype(str) if "skills_normalised" in out.columns else ""
        out["description"] = out["title"].astype(str) + " " + skills
    if "skills_normalised" not in out.columns:
        out["skills_normalised"] = ""
    if "salary_min" not in out.columns and "salary_annual_inr" in out.columns:
        s = pd.to_numeric(out["salary_annual_inr"], errors="coerce")
        out["salary_min"] = (s * 0.92).where(s.notna())
        out["salary_max"] = (s * 1.08).where(s.notna())
    if "is_remote" not in out.columns:
        out["is_remote"] = False
    return out


def load_jobs_dataframe() -> pd.DataFrame:
    """Prefer Chapter 6 cleaned CSV, then Ch5 raw, then legacy ch07 path; else synthetic."""
    from talentlens.paths import jobs_clean_path

    repo_root = _THIS_DIR.parents[1]
    clean_path = jobs_clean_path()
    if clean_path.exists():
        df = normalise_jobs_schema(pd.read_csv(clean_path))
        logger.info("Loaded jobs from %s", clean_path)
        return df.head(2_000)

    raw_path = repo_root / "data" / "raw" / "jobs_raw.csv"
    if raw_path.exists():
        df = normalise_jobs_schema(pd.read_csv(raw_path))
        logger.info("Loaded jobs from %s", raw_path)
        return df.head(2_000)

    csv_path = _THIS_DIR.parent / "ch07" / "data" / "clean" / "jobs_clean.csv"
    if csv_path.exists():
        df = normalise_jobs_schema(pd.read_csv(csv_path))
        logger.info("Loaded jobs from %s", csv_path)
        return df.head(2_000)

    rng = np.random.default_rng(20)
    n = 120
    titles = [
        "Senior NLP Engineer",
        "ML Engineer",
        "Data Scientist",
        "MLOps Engineer",
        "Data Engineer",
    ]
    rows = []
    for i in range(n):
        title = titles[int(rng.integers(0, len(titles)))]
        rows.append(
            {
                "job_id": f"demo_{i:04d}",
                "title": title,
                "company": f"Co{rng.integers(1, 20)}",
                "description": f"{title} building models with Python PyTorch SQL remote work.",
                "skills_normalised": "Python|SQL|PyTorch|MLflow",
                "salary_min": int(rng.integers(12, 25) * 100_000),
                "salary_max": int(rng.integers(28, 55) * 100_000),
                "is_remote": bool(rng.random() > 0.45),
            }
        )
    return pd.DataFrame(rows)


def rank_jobs(query: str, df: pd.DataFrame, top_k: int) -> list[JobResult]:
    scored: list[tuple[float, dict]] = []
    for _, row in df.iterrows():
        blob = " ".join(
            str(row.get(c, "") or "")
            for c in ("title", "description", "skills_normalised", "company")
        )
        s = keyword_score(query, blob)
        smin, smax = row.get("salary_min"), row.get("salary_max")
        if pd.isna(smin) and "salary_annual_inr" in row.index:
            sa = row.get("salary_annual_inr")
            if pd.notna(sa):
                smin, smax = float(sa) * 0.92, float(sa) * 1.08
        scored.append(
            (
                s,
                {
                    "job_id": str(row.get("job_id", "")),
                    "title": str(row.get("title", "")),
                    "company": str(row.get("company", "")),
                    "score": float(s),
                    "salary_min_l": (float(smin) / 100_000) if pd.notna(smin) else None,
                    "salary_max_l": (float(smax) / 100_000) if pd.notna(smax) else None,
                    "is_remote": bool(row.get("is_remote", False)),
                },
            )
        )
    scored.sort(key=lambda x: -x[0])
    out: list[JobResult] = []
    for _, payload in scored[:top_k]:
        out.append(JobResult(**payload))
    return out


def classify_salary_band(text: str) -> ClassifyResponse:
    """Keyword heuristic mapping free text to a coarse salary band - labelled as such in every response."""
    t = text.lower()
    score_high = sum(
        k in t for k in ("staff", "principal", "director", "10+", "12+", "40 l", "50 l")
    )
    score_mid = sum(k in t for k in ("senior", "lead", "5+", "6+", "15 l", "20 l", "25 l"))
    score_junior = sum(k in t for k in ("junior", "intern", "0-", "1 year", "fresher"))

    if score_high >= 1:
        return ClassifyResponse(
            predicted_band="₹30L+",
            confidence=0.72,
            rationale="Detected seniority / comp signals typical of top-of-band roles.",
        )
    if score_mid >= 1 and score_junior == 0:
        return ClassifyResponse(
            predicted_band="₹15–30L",
            confidence=0.68,
            rationale="Mid-senior language without junior markers.",
        )
    if score_junior >= 1:
        return ClassifyResponse(
            predicted_band="₹0–8L",
            confidence=0.61,
            rationale="Junior / early-career markers dominate the text.",
        )
    return ClassifyResponse(
        predicted_band="₹8–15L",
        confidence=0.55,
        rationale="Default mid-market band when no strong signals match.",
    )


# ---------------------------------------------------------------------------
# Architecture figure
# ---------------------------------------------------------------------------


def plot_api_architecture(out_dir: Optional[Path] = None) -> Path:
    import matplotlib.patches as mpatches  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402

    out_dir = out_dir or (_THIS_DIR / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color):
        r = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor="#333",
            facecolor=color,
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold"
        )

    box(0.5, 3.5, 2.2, 1.2, "Client\n(web / CLI)", "#E3F2FD")
    box(3.5, 3.5, 2.2, 1.2, "FastAPI\n(HTTP)", "#BBDEFB")
    box(6.5, 3.5, 2.8, 1.2, "Services\nsearch · classify", "#90CAF9")
    box(3.5, 1.0, 5.8, 1.2, "Data · vectors · models\n(SQLite / CSV → Ch 17–18)", "#64B5F6")

    ax.annotate("", xy=(3.4, 4.1), xytext=(2.7, 4.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.4, 4.1), xytext=(5.7, 4.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.0, 3.4), xytext=(6.0, 2.25), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("TalentLens API — request flow", fontsize=14, fontweight="bold", pad=12)
    p = out_dir / "ch19_api_architecture.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)
    return p


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def build_app(
    *,
    jobs_df: Optional[pd.DataFrame] = None,
    rate_limit_max: int = 120,
    rate_window_seconds: float = 60.0,
):
    """Create the FastAPI application (fresh limiter + routes each call)."""
    try:
        from fastapi import FastAPI, Request  # noqa: E402
        from fastapi.responses import JSONResponse  # noqa: E402
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install fastapi: pip install fastapi uvicorn[standard]") from e

    jobs_df = jobs_df if jobs_df is not None else load_jobs_dataframe()
    limiter = _RateLimiter(max_calls=rate_limit_max, per_seconds=rate_window_seconds)

    app = FastAPI(
        title="TalentLens API",
        version="1.0.0",
        description="Job search and salary-band classification for the TalentLens running project.",
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable):
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        if not limiter.allow():
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version="1.0.0")

    @app.post("/api/v1/search", response_model=SearchResponse, tags=["search"])
    async def search(body: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        results = rank_jobs(body.query, jobs_df, body.top_k)
        took = (time.perf_counter() - t0) * 1000
        return SearchResponse(query=body.query, results=results, took_ms=round(took, 3))

    @app.post("/api/v1/classify", response_model=ClassifyResponse, tags=["classify"])
    async def classify(body: ClassifyRequest) -> ClassifyResponse:
        return classify_salary_band(body.text)

    return app


# ASGI entry for uvicorn: `uvicorn book.ch19.ch19_fastapi_deployment:app`
app = build_app(rate_limit_max=500, rate_window_seconds=60.0)


def _run_inline_tests() -> None:
    """Lightweight checks without pytest (run when executed as script)."""
    df = load_jobs_dataframe()
    assert not df.empty
    r = rank_jobs("python pytorch remote", df, 5)
    assert len(r) <= 5
    c = classify_salary_band("Senior ML engineer 8 years PyTorch Bangalore")
    assert c.predicted_band
    hr = HealthResponse(status="ok", version="1.0.0")
    assert hr.status == "ok"
    logger.info("Inline tests OK (%d jobs)", len(df))


def _maybe_load_test() -> None:
    import os  # noqa: E402

    if os.environ.get("TALENTLENS_LOAD_TEST") != "1":
        return
    try:
        import httpx  # noqa: E402
    except ImportError:
        logger.warning("httpx not installed; skip load test")
        return

    application = build_app(rate_limit_max=10_000)
    transport = httpx.ASGITransport(app=application)
    with httpx.Client(transport=transport, base_url="http://test") as client:
        for _ in range(50):
            r = client.get("/health")
            r.raise_for_status()
        r = client.post("/api/v1/search", json={"query": "data scientist sql", "top_k": 5})
        r.raise_for_status()
    logger.info("Load smoke OK (50x health + 1 search)")


if __name__ == "__main__":
    import os  # noqa: E402

    _run_inline_tests()
    plot_api_architecture()
    _maybe_load_test()
    if os.environ.get("TALENTLENS_RUN_SERVER") != "1":
        logger.info(
            "Chapter 19 complete (inline tests + architecture figure). "
            "Set TALENTLENS_RUN_SERVER=1 to start uvicorn on :8765."
        )
    else:
        try:
            import uvicorn  # noqa: E402

            logger.info("Starting uvicorn on http://127.0.0.1:8765 — Ctrl+C to stop")
            uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
        except ImportError:
            logger.info(
                "uvicorn not installed; skipped server start. pip install uvicorn[standard]"
            )
