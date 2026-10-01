"""
Tests for Chapter 19: FastAPI - TalentLens API.

Run from repository root:

    pytest tests/test_ch19.py -v

Logic tests always run. HTTP tests skip automatically if `fastapi` is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch19"))

from ch19_fastapi_deployment import (  # noqa: E402
    ClassifyResponse,
    HealthResponse,
    JobResult,
    SearchRequest,
    SearchResponse,
    classify_salary_band,
    keyword_score,
    load_jobs_dataframe,
    normalise_jobs_schema,
    rank_jobs,
)


class TestKeywordScore:
    def test_full_overlap(self):
        assert keyword_score("a b", "a b c") == 1.0

    def test_zero_overlap(self):
        assert keyword_score("x y", "a b c") == 0.0

    def test_partial(self):
        s = keyword_score("python ml", "python engineer sql")
        assert 0 < s < 1


class TestNormaliseSchema:
    def test_adds_title_from_role(self):
        df = pd.DataFrame(
            {
                "role_category": ["Data Scientist"],
                "salary_annual_inr": [2_000_000],
                "skills_normalised": ["Python|SQL"],
                "is_remote": [True],
            }
        )
        out = normalise_jobs_schema(df)
        assert out["title"].iloc[0] == "Data Scientist"
        assert "salary_min" in out.columns

    def test_job_ids(self):
        df = pd.DataFrame({"role_category": ["A", "B"], "salary_annual_inr": [1, 2]})
        out = normalise_jobs_schema(df)
        assert out["job_id"].nunique() == 2


class TestLoadJobs:
    def test_non_empty(self):
        df = load_jobs_dataframe()
        assert len(df) >= 1


class TestRankJobs:
    def test_returns_job_results(self):
        df = load_jobs_dataframe()
        r = rank_jobs("python sql", df, 3)
        assert len(r) <= 3
        assert all(isinstance(x, JobResult) for x in r)

    def test_sorted_by_score(self):
        df = load_jobs_dataframe()
        r = rank_jobs("machine learning", df, 8)
        scores = [x.score for x in r]
        assert scores == sorted(scores, reverse=True)


class TestClassify:
    def test_senior_band(self):
        r = classify_salary_band("Staff ML engineer 12 years experience")
        assert isinstance(r, ClassifyResponse)
        assert r.predicted_band

    def test_junior_band(self):
        r = classify_salary_band("Junior analyst intern fresher")
        assert r.predicted_band


class TestPydanticModels:
    def test_health_response(self):
        h = HealthResponse(status="ok", version="1.0.0")
        assert h.model_dump()["status"] == "ok"

    def test_search_request_validation(self):
        s = SearchRequest(query="hello", top_k=5)
        assert s.top_k == 5

    def test_search_request_top_k_bounds(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchRequest(query="x", top_k=99)

    def test_job_result_optional_salary(self):
        j = JobResult(
            job_id="1",
            title="T",
            company="C",
            score=0.5,
            salary_min_l=None,
            salary_max_l=None,
            is_remote=False,
        )
        assert j.salary_min_l is None

    def test_search_response_roundtrip(self):
        jr = JobResult(
            job_id="a",
            title="t",
            company="c",
            score=0.2,
            is_remote=True,
        )
        sr = SearchResponse(query="q", results=[jr], took_ms=1.2)
        assert len(sr.results) == 1


@pytest.fixture(scope="module")
def fastapi_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from ch19_fastapi_deployment import build_app
    from fastapi.testclient import TestClient

    app = build_app(rate_limit_max=50_000, rate_window_seconds=60.0)
    with TestClient(app) as client:
        yield client


class TestHTTPHealth:
    def test_health_ok(self, fastapi_client):
        r = fastapi_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "talentlens-api"


class TestHTTPSearch:
    def test_search_ok(self, fastapi_client):
        r = fastapi_client.post(
            "/api/v1/search",
            json={"query": "python pytorch remote", "top_k": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert len(body["results"]) <= 5
        assert "took_ms" in body

    def test_search_query_required(self, fastapi_client):
        r = fastapi_client.post("/api/v1/search", json={"top_k": 3})
        assert r.status_code == 422

    def test_search_top_k_max(self, fastapi_client):
        r = fastapi_client.post(
            "/api/v1/search",
            json={"query": "sql", "top_k": 100},
        )
        assert r.status_code == 422

    def test_search_empty_query(self, fastapi_client):
        r = fastapi_client.post("/api/v1/search", json={"query": "", "top_k": 3})
        assert r.status_code == 422


class TestHTTPClassify:
    def test_classify_ok(self, fastapi_client):
        r = fastapi_client.post(
            "/api/v1/classify",
            json={"text": "Senior data scientist with 6 years in Bangalore"},
        )
        assert r.status_code == 200
        j = r.json()
        assert "predicted_band" in j
        assert "confidence" in j

    def test_classify_too_short(self, fastapi_client):
        r = fastapi_client.post("/api/v1/classify", json={"text": "ab"})
        assert r.status_code == 422


class TestOpenAPI:
    def test_openapi_contains_schemas(self, fastapi_client):
        r = fastapi_client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "SearchRequest" in spec["components"]["schemas"]

    def test_docs_available(self, fastapi_client):
        assert fastapi_client.get("/docs").status_code == 200


class TestRateLimit:
    def test_health_exempt_from_limit(self):
        pytest.importorskip("fastapi")
        from ch19_fastapi_deployment import build_app
        from fastapi.testclient import TestClient

        tight = build_app(rate_limit_max=3, rate_window_seconds=60.0)
        with TestClient(tight) as c:
            for _ in range(8):
                assert c.get("/health").status_code == 200

    def test_limit_triggers_429(self):
        pytest.importorskip("fastapi")
        from ch19_fastapi_deployment import build_app
        from fastapi.testclient import TestClient

        tight = build_app(rate_limit_max=2, rate_window_seconds=300.0)
        with TestClient(tight) as c:
            assert c.post("/api/v1/search", json={"query": "a", "top_k": 1}).status_code == 200
            assert c.post("/api/v1/search", json={"query": "b", "top_k": 1}).status_code == 200
            r = c.post("/api/v1/search", json={"query": "c", "top_k": 1})
            assert r.status_code == 429
