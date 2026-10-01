"""
Tests for Chapter 5: Data Collection
Run: pytest tests/test_ch05.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_THIS_FILE = Path(__file__).resolve()
for _candidate in [
    _THIS_FILE.parents[1] / "book" / "ch05",
    _THIS_FILE.parents[1],
]:
    if (_candidate / "ch05_data_collection.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from ch05_data_collection import (  # noqa: E402
    REQUIRED_FIELDS,
    SCHEMA,
    AdzunaCollector,
    Config,
    DataCollectionPipeline,
    DemoCollector,
    RemoteOKCollector,
    coerce_to_schema,
    plot_collection_funnel,
    plot_source_breakdown,
    validate_job,
    write_collection_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.raw_data_path = tmp_path / "data" / "raw" / "jobs_raw.csv"
    c.raw_demo_data_path = tmp_path / "data" / "raw" / "jobs_raw.demo.csv"
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path / "reports"
    c.demo_target = 50
    return c


@pytest.fixture
def valid_job():
    return {
        "job_id": "test_001",
        "source": "test",
        "title": "Senior ML Engineer",
        "company": "Test Corp",
        "city": "Bangalore",
        "country": "IN",
        "description": "We need a senior ML engineer with Python experience.",
        "skills_raw": "Python,PyTorch",
        "salary_min": 2_000_000.0,
        "salary_max": 3_000_000.0,
        "currency": "INR",
        "is_remote": False,
        "posted_date": "2026-01-15",
        "url": "https://example.com/jobs/1",
    }


@pytest.fixture
def minimal_job():
    return {
        "title": "Data Scientist",
        "company": "AI Startup",
        "description": "Building ML models for production.",
    }


# ---------------------------------------------------------------------------
# Schema and validation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_has_required_fields(self):
        for f in REQUIRED_FIELDS:
            assert f in SCHEMA, f"Required field '{f}' missing from SCHEMA"

    def test_schema_types_are_valid(self):
        valid_types = {str, float, bool}
        for field, typ in SCHEMA.items():
            assert typ in valid_types, f"Field '{field}' has invalid type {typ}"

    def test_required_fields_subset_of_schema(self):
        assert REQUIRED_FIELDS.issubset(set(SCHEMA.keys()))


class TestValidateJob:
    def test_valid_job_passes(self, valid_job):
        assert validate_job(valid_job) is True

    def test_minimal_job_passes(self, minimal_job):
        assert validate_job(minimal_job) is True

    def test_missing_title_fails(self, valid_job):
        valid_job["title"] = ""
        assert validate_job(valid_job) is False

    def test_none_title_fails(self):
        assert validate_job({"title": None, "company": "X", "description": "Y"}) is False

    def test_missing_description_fails(self, valid_job):
        del valid_job["description"]
        assert validate_job(valid_job) is False

    def test_whitespace_only_title_fails(self):
        assert validate_job({"title": "   ", "company": "X", "description": "Y"}) is False

    def test_empty_dict_fails(self):
        assert validate_job({}) is False


class TestCoerceToSchema:
    def test_returns_all_schema_keys(self, valid_job):
        result = coerce_to_schema(valid_job)
        for key in SCHEMA:
            assert key in result, f"Missing key: {key}"

    def test_string_coercion(self, minimal_job):
        result = coerce_to_schema(minimal_job)
        assert isinstance(result.get("title"), str)

    def test_float_coercion(self, valid_job):
        valid_job["salary_min"] = "2000000"
        result = coerce_to_schema(valid_job)
        assert isinstance(result["salary_min"], float)

    def test_bool_coercion(self, valid_job):
        valid_job["is_remote"] = "True"
        result = coerce_to_schema(valid_job)
        assert isinstance(result["is_remote"], bool)

    def test_invalid_float_becomes_none(self, valid_job):
        valid_job["salary_min"] = "not a number"
        result = coerce_to_schema(valid_job)
        assert result["salary_min"] is None

    def test_empty_string_becomes_none(self, valid_job):
        valid_job["city"] = "   "
        result = coerce_to_schema(valid_job)
        assert result["city"] is None

    def test_extra_keys_ignored(self, valid_job):
        valid_job["unexpected_field"] = "unexpected_value"
        result = coerce_to_schema(valid_job)
        assert "unexpected_field" not in result

    def test_strips_whitespace(self, valid_job):
        valid_job["title"] = "  Senior ML Engineer  "
        result = coerce_to_schema(valid_job)
        assert result["title"] == "Senior ML Engineer"


# ---------------------------------------------------------------------------
# DemoCollector
# ---------------------------------------------------------------------------


class TestDemoCollector:
    def test_collects_requested_count(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(50)
        assert len(records) == 50

    def test_all_records_valid(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(100)
        invalid = [r for r in records if not validate_job(r)]
        assert len(invalid) == 0, f"{len(invalid)} invalid records found"

    def test_all_records_have_schema_keys(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(20)
        for rec in records:
            for key in SCHEMA:
                assert key in rec, f"Record missing schema key: {key}"

    def test_source_field_is_demo(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(10)
        assert all(r["source"] == "demo" for r in records)

    def test_some_records_have_salary(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(100)
        with_salary = [r for r in records if r.get("salary_min") is not None]
        assert len(with_salary) > 50, "Expected majority of records to have salary"

    def test_some_records_are_remote(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(100)
        remote = [r for r in records if r.get("is_remote")]
        assert len(remote) > 0, "Expected some remote postings"

    def test_deterministic_output(self, cfg):
        collector = DemoCollector(cfg)
        r1 = collector.collect(20)
        r2 = collector.collect(20)
        assert [r["job_id"] for r in r1] == [r["job_id"] for r in r2]

    def test_salary_max_gte_min(self, cfg):
        collector = DemoCollector(cfg)
        records = collector.collect(100)
        for rec in records:
            if rec.get("salary_min") and rec.get("salary_max"):
                assert rec["salary_max"] >= rec["salary_min"]

    def test_collect_zero_returns_empty(self, cfg):
        collector = DemoCollector(cfg)
        assert collector.collect(0) == []


# ---------------------------------------------------------------------------
# AdzunaCollector (mocked)
# ---------------------------------------------------------------------------


class TestAdzunaCollector:
    def test_skips_when_no_api_keys(self, cfg):
        cfg.adzuna_app_id = ""
        cfg.adzuna_api_key = ""
        collector = AdzunaCollector(cfg)
        result = collector.collect(100)
        assert result == []

    def test_parse_job_maps_fields(self, cfg):
        cfg.adzuna_app_id = "test_id"
        cfg.adzuna_api_key = "test_key"
        collector = AdzunaCollector(cfg)
        raw = {
            "id": "12345",
            "title": "ML Engineer",
            "company": {"display_name": "Test Co"},
            "location": {"display_name": "Bangalore"},
            "description": "Python and ML experience required.",
            "salary_min": 2000000,
            "salary_max": 3000000,
            "created": "2026-01-01",
            "redirect_url": "https://adzuna.com/jobs/12345",
        }
        parsed = collector._parse_job(raw)
        assert parsed["title"] == "ML Engineer"
        assert parsed["company"] == "Test Co"
        assert parsed["salary_min"] == 2000000
        assert parsed["source"] == "adzuna"
        assert parsed["job_id"].startswith("adzuna_")

    def test_parse_handles_missing_company(self, cfg):
        cfg.adzuna_app_id = "x"
        cfg.adzuna_api_key = "y"
        collector = AdzunaCollector(cfg)
        raw = {
            "id": "1",
            "title": "Engineer",
            "description": "Some description here.",
        }
        parsed = collector._parse_job(raw)
        assert parsed["title"] == "Engineer"
        assert parsed["company"] is not None  # might be empty string


# ---------------------------------------------------------------------------
# RemoteOKCollector (mocked)
# ---------------------------------------------------------------------------


class TestRemoteOKCollector:
    def test_parse_job_maps_fields(self, cfg):
        collector = RemoteOKCollector(cfg)
        raw = {
            "id": "99",
            "position": "Senior Python Engineer",
            "company": "Remote AI",
            "description": "Build AI systems remotely.",
            "tags": ["python", "pytorch", "rag"],
            "salary_min": 80000,
            "salary_max": 120000,
            "date": "2026-02-01",
            "url": "https://remoteok.com/jobs/99",
        }
        parsed = collector._parse_job(raw)
        assert parsed["title"] == "Senior Python Engineer"
        assert parsed["is_remote"] is True
        assert parsed["source"] == "remoteok"
        assert parsed["currency"] == "USD"
        assert "python" in parsed["skills_raw"]

    def test_parse_handles_missing_salary(self, cfg):
        collector = RemoteOKCollector(cfg)
        raw = {
            "id": "1",
            "position": "Engineer",
            "company": "Corp",
            "description": "Great job description.",
        }
        parsed = collector._parse_job(raw)
        assert parsed["salary_min"] is None

    def test_collect_returns_empty_on_network_error(self, cfg):
        collector = RemoteOKCollector(cfg)
        # Simulate network failure by pointing to an invalid host
        collector.API_URL = "http://localhost:1"
        collector.cfg.request_timeout = 1  # fast timeout
        result = collector.collect(10)
        assert result == []


# ---------------------------------------------------------------------------
# DataCollectionPipeline
# ---------------------------------------------------------------------------


class TestDataCollectionPipeline:
    def test_demo_run_creates_csv(self, cfg):
        pipeline = DataCollectionPipeline(cfg)
        pipeline.run(live=False)
        assert cfg.raw_demo_data_path.exists()
        assert not cfg.raw_data_path.exists()

    def test_demo_csv_has_records(self, cfg):
        pipeline = DataCollectionPipeline(cfg)
        pipeline.run(live=False)
        df = pd.read_csv(cfg.raw_demo_data_path)
        assert len(df) > 0

    def test_demo_csv_has_schema_columns(self, cfg):
        pipeline = DataCollectionPipeline(cfg)
        pipeline.run(live=False)
        df = pd.read_csv(cfg.raw_demo_data_path)
        for col in REQUIRED_FIELDS:
            assert col in df.columns

    def test_pipeline_returns_stats(self, cfg):
        pipeline = DataCollectionPipeline(cfg)
        results = pipeline.run(live=False)
        assert "stats" in results
        assert "total_raw" in results
        assert "total_valid" in results
        assert "duplicates_removed" in results
        assert results["output_path"] == cfg.raw_demo_data_path

    def test_deduplication_removes_duplicates(self, cfg):
        """Pipeline should not output exact duplicate job_id + title combos."""
        pipeline = DataCollectionPipeline(cfg)
        pipeline.run(live=False)
        df = pd.read_csv(cfg.raw_demo_data_path)
        # No exact duplicate title+company+city fingerprints (matches pipeline logic)
        fingerprint = (
            df["title"].str.lower().fillna("")
            + "|"
            + df["company"].str.lower().fillna("")
            + "|"
            + df["city"].str.lower().fillna("")
        )
        assert (
            fingerprint.duplicated().sum() == 0
        ), f"{fingerprint.duplicated().sum()} duplicates found"

    def test_pipeline_counts_are_consistent(self, cfg):
        pipeline = DataCollectionPipeline(cfg)
        results = pipeline.run(live=False)
        assert results["total_raw"] >= results["total_valid"]
        assert results["duplicates_removed"] >= 0
        assert results["total_raw"] == results["total_valid"] + results["duplicates_removed"]


# ---------------------------------------------------------------------------
# Visualisations and reports
# ---------------------------------------------------------------------------


class TestVisualisations:
    @pytest.fixture
    def sample_results(self):
        return {
            "stats": {"demo": {"collected": 100, "valid": 97}},
            "total_raw": 100,
            "total_valid": 87,
            "duplicates_removed": 10,
        }

    def test_collection_funnel_creates_file(self, cfg, sample_results):
        cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        out = plot_collection_funnel(sample_results, cfg)
        assert out.exists()
        assert out.suffix == ".png"

    def test_source_breakdown_creates_file(self, cfg, sample_results):
        cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        out = plot_source_breakdown(sample_results, cfg)
        assert out.exists()

    def test_collection_summary_creates_md(self, cfg, sample_results):
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = write_collection_summary(sample_results, cfg)
        assert out.exists()
        assert out.suffix == ".md"

    def test_summary_has_required_sections(self, cfg, sample_results):
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = write_collection_summary(sample_results, cfg)
        content = out.read_text()
        for section in ["Overview", "By source", "Next step"]:
            assert section in content, f"Missing section: {section}"

    def test_summary_includes_dedup_stats(self, cfg, sample_results):
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = write_collection_summary(sample_results, cfg)
        content = out.read_text()
        assert "dedup" in content.lower() or "duplicat" in content.lower()
