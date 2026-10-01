"""
Tests for Chapter 17: LLM Generation Layer
Run: pytest tests/test_ch17.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_THIS_FILE = Path(__file__).resolve()
for _candidate in [
    _THIS_FILE.parents[1] / "book" / "ch17",
    _THIS_FILE.parents[1],
]:
    if (_candidate / "ch17_llm_generation.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from ch17_llm_generation import (  # noqa: E402
    DEMO_CV,
    DEMO_JOBS,
    Config,
    CVParser,
    LLMClient,
    MatchExplainer,
    TalentLensAdvisor,
    _extract_job_ids_from_prompt,
    _extract_skills_from_prompt,
    _format_salary,
    write_sample_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_cfg():
    return Config(provider="stub")


@pytest.fixture
def stub_client(stub_cfg):
    return LLMClient(stub_cfg)


@pytest.fixture
def sample_jobs():
    return DEMO_JOBS[:3]


@pytest.fixture
def minimal_cv_profile():
    """Minimal CV profile as a dict (works without Pydantic)."""
    return {
        "skills": ["Python", "NLP", "FastAPI"],
        "years_experience": 5,
        "current_role": "ML Engineer",
        "raw_text": "Senior ML engineer with Python and NLP experience.",
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_provider_is_stub_when_no_key(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Config()
        assert cfg.provider == "stub"

    def test_temperature_low_for_structured_output(self):
        cfg = Config()
        assert cfg.temperature <= 0.2, "Low temperature required for JSON output reliability"

    def test_max_jobs_in_prompt_reasonable(self):
        cfg = Config()
        assert 3 <= cfg.max_jobs_in_prompt <= 10

    def test_max_tokens_sufficient_for_response(self):
        cfg = Config()
        assert cfg.max_tokens >= 800, "Need at least 800 tokens for multi-job explanation"

    def test_max_retries_at_least_two(self):
        cfg = Config()
        assert cfg.max_retries >= 2


# ---------------------------------------------------------------------------
# LLMClient (stub mode)
# ---------------------------------------------------------------------------


class TestLLMClientStub:
    def test_complete_returns_tuple(self, stub_client):
        content, latency = stub_client.complete("system", "user message about jobs")
        assert isinstance(content, str)
        assert isinstance(latency, float)

    def test_stub_returns_valid_json(self, stub_client):
        content, _ = stub_client.complete("system", "analyse these jobs for me")
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_stub_output_has_required_keys(self, stub_client):
        content, _ = stub_client.complete("system", "job_id: job_001 skills: [Python]")
        data = json.loads(content)
        assert "cv_summary" in data
        assert "job_matches" in data

    def test_stub_is_deterministic(self, stub_client):
        msg = "consistent test message for determinism check"
        c1, _ = stub_client.complete("sys", msg)
        c2, _ = stub_client.complete("sys", msg)
        assert c1 == c2

    def test_stub_latency_positive(self, stub_client):
        _, latency = stub_client.complete("sys", "test")
        assert latency > 0

    def test_stub_latency_reasonable(self, stub_client):
        _, latency = stub_client.complete("sys", "test message")
        assert latency < 5.0, "Stub latency should be under 5 seconds"

    def test_provider_stub_never_calls_network(self, stub_client):
        """Stub must not make any network calls."""
        with patch("socket.create_connection") as mock_conn:
            stub_client.complete("sys", "test")
            mock_conn.assert_not_called()

    def test_model_resolves_correctly_for_stub(self, stub_cfg):
        client = LLMClient(stub_cfg)
        assert client._model == "stub"


# ---------------------------------------------------------------------------
# CVParser
# ---------------------------------------------------------------------------


class TestCVParser:
    def test_parse_returns_profile_with_skills(self, stub_client, stub_cfg):
        parser = CVParser(stub_client, stub_cfg)
        profile = parser.parse(DEMO_CV)
        skills = getattr(
            profile, "skills", profile.get("skills", []) if isinstance(profile, dict) else []
        )
        assert isinstance(skills, list)

    def test_parse_handles_empty_cv(self, stub_client, stub_cfg):
        parser = CVParser(stub_client, stub_cfg)
        profile = parser.parse("")
        # Should not crash even with empty input
        assert profile is not None

    def test_parse_handles_malformed_llm_output(self, stub_cfg):
        """Parser should gracefully handle non-JSON LLM output."""
        bad_client = LLMClient(stub_cfg)
        bad_client.complete = lambda s, u: ("not valid json!!!", 0.1)
        parser = CVParser(bad_client, stub_cfg)
        # Should not raise - returns empty/default profile
        profile = parser.parse("any cv text")
        assert profile is not None

    def test_parse_truncates_long_cv(self, stub_client, stub_cfg):
        """Parser must not send unbounded text to the LLM."""
        long_cv = "Python " * 5000  # ~35,000 chars
        calls = []
        orig_complete = stub_client.complete

        def tracking_complete(system, user):
            calls.append(len(user))
            return orig_complete(system, user)

        stub_client.complete = tracking_complete
        parser = CVParser(stub_client, stub_cfg)
        parser.parse(long_cv)
        assert calls[0] < stub_cfg.max_cv_tokens * 4 + 500, "CV should be truncated"

    def test_system_prompt_has_json_instruction(self):
        assert "JSON" in CVParser.SYSTEM_PROMPT.upper()


# ---------------------------------------------------------------------------
# MatchExplainer
# ---------------------------------------------------------------------------


class TestMatchExplainer:
    def test_explain_returns_tuple(self, stub_client, stub_cfg, minimal_cv_profile, sample_jobs):
        explainer = MatchExplainer(stub_client, stub_cfg)
        result, latency = explainer.explain(minimal_cv_profile, sample_jobs)
        assert isinstance(result, dict)
        assert isinstance(latency, float)

    def test_explain_output_has_job_matches(
        self, stub_client, stub_cfg, minimal_cv_profile, sample_jobs
    ):
        explainer = MatchExplainer(stub_client, stub_cfg)
        result, _ = explainer.explain(minimal_cv_profile, sample_jobs)
        assert "job_matches" in result
        assert isinstance(result["job_matches"], list)

    def test_explain_respects_max_jobs(self, stub_client, stub_cfg, minimal_cv_profile):
        many_jobs = DEMO_JOBS * 3  # 15 jobs
        explainer = MatchExplainer(stub_client, stub_cfg)
        result, _ = explainer.explain(minimal_cv_profile, many_jobs)
        # Should not include more than max_jobs_in_prompt
        assert len(result["job_matches"]) <= stub_cfg.max_jobs_in_prompt

    def test_explain_handles_empty_jobs(self, stub_client, stub_cfg, minimal_cv_profile):
        explainer = MatchExplainer(stub_client, stub_cfg)
        result, _ = explainer.explain(minimal_cv_profile, [])
        assert "job_matches" in result

    def test_fallback_on_json_error(self, stub_cfg, minimal_cv_profile, sample_jobs):
        bad_client = LLMClient(stub_cfg)
        bad_client.complete = lambda s, u: ("INVALID JSON {{{", 0.1)
        explainer = MatchExplainer(bad_client, stub_cfg)
        result, _ = explainer.explain(minimal_cv_profile, sample_jobs)
        # Should return fallback, not crash
        assert "job_matches" in result

    def test_prompt_contains_job_ids(self, stub_client, stub_cfg, minimal_cv_profile, sample_jobs):
        explainer = MatchExplainer(stub_client, stub_cfg)
        prompts = []
        orig = stub_client.complete

        def capture(s, u):
            prompts.append(u)
            return orig(s, u)

        stub_client.complete = capture
        explainer.explain(minimal_cv_profile, sample_jobs)
        assert any(job["job_id"] in prompts[0] for job in sample_jobs)

    def test_system_prompt_requires_json_only(self):
        assert "JSON" in MatchExplainer.SYSTEM_PROMPT
        assert "no markdown" in MatchExplainer.SYSTEM_PROMPT.lower()

    def test_system_prompt_defines_fit_levels(self):
        for level in ["STRONG MATCH", "GOOD MATCH", "PARTIAL MATCH", "NO MATCH"]:
            assert level in MatchExplainer.SYSTEM_PROMPT

    def test_system_prompt_defines_apply_options(self):
        for option in ["YES", "MAYBE", "NO"]:
            assert option in MatchExplainer.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TalentLensAdvisor (integration)
# ---------------------------------------------------------------------------


class TestTalentLensAdvisor:
    def test_advise_returns_expected_keys(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS)
        for key in ["cv_profile", "explanation", "provider", "model", "timings"]:
            assert key in result, f"Missing key: {key}"

    def test_advise_cv_profile_has_skills(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS)
        assert isinstance(result["cv_profile"]["skills"], list)

    def test_advise_explanation_has_job_matches(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS)
        assert "job_matches" in result["explanation"]

    def test_advise_timings_present(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS)
        for key in ["cv_parsing", "explanation"]:
            assert key in result["timings"]
            assert result["timings"][key] >= 0

    def test_advise_provider_matches_config(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS)
        assert result["provider"] == "stub"

    def test_advise_handles_empty_cv(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise("", DEMO_JOBS)
        assert result is not None

    def test_advise_handles_empty_jobs(self, stub_cfg):
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, [])
        assert result is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_job_ids(self):
        text = 'job_id: "job_001" ... job_id: "job_002"'
        ids = _extract_job_ids_from_prompt(text)
        assert "job_001" in ids or len(ids) > 0

    def test_extract_skills(self):
        text = 'skills: ["Python", "NLP", "FastAPI"]'
        skills = _extract_skills_from_prompt(text)
        assert "Python" in skills

    def test_format_salary_with_values(self):
        result = _format_salary(1_000_000, 2_000_000)
        assert "₹" in result or "L" in result

    def test_format_salary_none(self):
        result = _format_salary(None, None)
        assert "Not disclosed" in result

    def test_format_salary_partial(self):
        result = _format_salary(None, 2_000_000)
        assert "Not disclosed" in result


# ---------------------------------------------------------------------------
# Sample output writer
# ---------------------------------------------------------------------------


class TestWriteSampleOutput:
    def test_creates_markdown_file(self, stub_cfg, tmp_path):
        stub_cfg.reports_dir = tmp_path
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS[:2])
        out = write_sample_output(result, stub_cfg)
        assert out.exists()
        assert out.suffix == ".md"

    def test_markdown_has_required_sections(self, stub_cfg, tmp_path):
        stub_cfg.reports_dir = tmp_path
        advisor = TalentLensAdvisor(stub_cfg)
        result = advisor.advise(DEMO_CV, DEMO_JOBS[:2])
        out = write_sample_output(result, stub_cfg)
        content = out.read_text()
        for section in ["CV Summary", "Job Match Analysis", "Timing"]:
            assert section in content, f"Missing section: {section}"
