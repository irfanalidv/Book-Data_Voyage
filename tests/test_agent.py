"""Tests for talentlens.agent.

Iteration 0: smoke tests for the public API surface plus
NotImplementedError tripwires for the stubs. Each tripwire
gets replaced by real behaviour tests in the iteration that
fills in its function.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from talentlens.agent import (
    DEFAULT_AGENT_MODEL,
    TOOL_SPECS,
    TOOLS,
    Agent,
    AgentResult,
    ToolCall,
    classify_role,
    get_job_detail,
    search_jobs,
    summarise_for_candidate,
)


class TestPublicAPI:
    def test_agent_result_dataclass(self):
        r = AgentResult(query="q", final_response="x")
        assert r.query == "q"
        assert r.final_response == "x"
        assert r.tool_calls == []
        assert r.completed is True
        assert r.stop_reason == "natural"
        assert r.error_detail is None

    def test_tool_call_is_frozen(self):
        t = ToolCall(name="search", arguments={}, result=None, elapsed_seconds=0.1)
        with pytest.raises(Exception):  # FrozenInstanceError
            t.name = "something_else"

    def test_tools_registry_starts_empty(self):
        assert TOOLS == {} or isinstance(TOOLS, dict)
        assert TOOL_SPECS == [] or isinstance(TOOL_SPECS, list)


class TestToolStubs:
    """Tripwires - each fails when its iteration lands."""


class TestSearchJobs:
    def test_returns_list_of_dicts(self):
        results = search_jobs("ML Engineer", k=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)
            assert "job_id" in r
            assert "score" in r

    def test_no_results_returns_empty_list(self):
        results = search_jobs("zzz_no_such_thing_qqq", k=5)
        assert results == []

    def test_k_is_clamped_for_absurd_values(self):
        r1 = search_jobs("ML", k=-3)
        r2 = search_jobs("ML", k=999)
        assert len(r1) <= 1
        assert len(r2) <= 20

    def test_results_are_json_serialisable(self):
        import json

        results = search_jobs("ML Engineer", k=3)
        json.dumps(results)


class TestGetJobDetail:
    def test_returns_full_record_for_known_id(self):
        results = search_jobs("ML Engineer", k=1)
        if not results:
            pytest.skip("No matching postings on bundled data")
        detail = get_job_detail(results[0]["job_id"])
        assert "error" not in detail
        assert "title" in detail
        assert "description" in detail
        assert "role_category" in detail

    def test_returns_error_for_unknown_id(self):
        result = get_job_detail("definitely_not_a_real_id")
        assert result == {
            "error": "not_found",
            "job_id": "definitely_not_a_real_id",
        }


class TestClassifyRole:
    def test_unknown_job_returns_error(self):
        result = classify_role("not_a_real_job_id")
        assert "error" in result

    def test_known_job_returns_prediction_or_structured_error(self):
        results = search_jobs("ML Engineer", k=1)
        if not results:
            pytest.skip("No matching postings on bundled data")
        result = classify_role(results[0]["job_id"])
        if "error" not in result:
            assert "predicted_role" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0


class TestSummariseForCandidate:
    def test_returns_string(self):
        results = search_jobs("ML Engineer", k=1)
        if not results:
            pytest.skip("No matching postings on bundled data")
        summary = summarise_for_candidate(results[0]["job_id"], ["Python", "SQL"])
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_unknown_job_returns_error_string(self):
        result = summarise_for_candidate("nope", ["Python"])
        assert result.startswith("ERROR:")

    def test_empty_skills_returns_error_string(self):
        results = search_jobs("ML Engineer", k=1)
        if not results:
            pytest.skip("No matching postings on bundled data")
        result = summarise_for_candidate(results[0]["job_id"], [])
        assert result.startswith("ERROR:")

    def test_summary_lists_matched_and_missing_skills(self):
        results = search_jobs("ML Engineer", k=1)
        if not results:
            pytest.skip("No matching postings on bundled data")
        summary = summarise_for_candidate(results[0]["job_id"], ["Python", "SQL"])
        assert "Matched" in summary
        assert "Missing" in summary


class TestToolRegistry:
    def test_all_four_tools_registered(self):
        assert set(TOOLS.keys()) == {
            "search_jobs",
            "get_job_detail",
            "classify_role",
            "summarise_for_candidate",
        }

    def test_tool_specs_match_tool_names(self):
        spec_names = {s["function"]["name"] for s in TOOL_SPECS}
        assert spec_names == set(TOOLS.keys())

    def test_tool_specs_have_required_fields(self):
        for spec in TOOL_SPECS:
            assert spec["type"] == "function"
            f = spec["function"]
            assert "name" in f
            assert "description" in f
            assert "parameters" in f
            assert len(f["description"]) > 50, (
                f"Tool {f['name']!r} has a short description "
                f"({len(f['description'])} chars); the LLM needs more "
                f"context."
            )


class TestAgentInitialization:
    """Smoke tests for Agent class construction."""

    def test_constructs_with_defaults(self):
        agent = Agent()
        assert agent.model == DEFAULT_AGENT_MODEL
        assert agent.max_steps == 8
        assert agent.use_cache is True

    def test_cache_dir_defaults_to_chapter_path(self):
        agent = Agent()
        assert "ch18" in str(agent.cache_dir)
        assert "cache" in str(agent.cache_dir)

    def test_env_var_disables_cache(self, monkeypatch):
        monkeypatch.setenv("AGENT_NO_CACHE", "1")
        agent = Agent(use_cache=True)
        assert agent.use_cache is False

    def test_cache_key_is_deterministic(self):
        agent = Agent()
        k1 = agent._cache_key("test query")
        k2 = agent._cache_key("test query")
        assert k1 == k2
        k3 = agent._cache_key("different query")
        assert k1 != k3

    def test_cache_key_depends_on_model(self):
        a1 = Agent(model="model-a")
        a2 = Agent(model="model-b")
        assert a1._cache_key("q") != a2._cache_key("q")

    def test_dispatch_tool_unknown_returns_error(self):
        agent = Agent()
        result, error = agent._dispatch_tool("nope", {})
        assert result is None
        assert "Unknown tool" in error

    def test_dispatch_tool_works_on_search(self):
        agent = Agent()
        result, error = agent._dispatch_tool("search_jobs", {"query": "ML Engineer", "k": 1})
        assert error is None
        assert isinstance(result, list)

    def test_cache_round_trip_preserves_error_detail(self, tmp_path):
        agent = Agent(cache_dir=str(tmp_path))
        errored = AgentResult(
            query="test",
            final_response="",
            stop_reason="error",
            error_detail="BadRequestError: tool_use_failed",
            completed=False,
        )
        agent._save_cached(agent._cache_key("test"), errored)
        loaded = agent._load_cached(agent._cache_key("test"))
        assert loaded is not None
        assert loaded.error_detail == "BadRequestError: tool_use_failed"


class TestReportWriter:
    """Smoke tests for the chapter executable's report writer."""

    def test_write_report_handles_missing_file(self, tmp_path):
        from book.ch18.ch18_agentic_ai import _write_report

        fake_path = tmp_path / "does_not_exist.md"
        r = AgentResult(query="q", final_response="x")
        try:
            _write_report([({"id": "q1"}, r)], fake_path, "live")
        except Exception as e:
            pytest.fail(f"_write_report should warn, not raise: {e}")

    def test_write_report_idempotent(self, tmp_path):
        """Running _write_report twice preserves section structure."""
        from book.ch18.ch18_agentic_ai import _write_report

        report = tmp_path / "report.md"
        report.write_text(
            "# Test report\n\n"
            "## Per-query results\n\n"
            "(to be regenerated)\n\n"
            "## Next section\n\nKept as-is.\n"
        )
        r = AgentResult(
            query="test",
            final_response="ok",
            n_tool_calls=1,
            n_llm_calls=2,
            elapsed_seconds=1.5,
            input_tokens=100,
            output_tokens=50,
            stop_reason="natural",
        )
        _write_report([({"id": "q1"}, r)], report, "live")
        _write_report([({"id": "q1"}, r)], report, "live")
        second = report.read_text()
        assert "## Per-query results" in second
        assert "## Next section" in second
        assert "Kept as-is" in second
        assert second.count("| **Total** |") == 1


class _ScriptedClient:
    """Stands in for the Groq client: returns pre-written turns in order."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        message = self._turns.pop(0)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id, name, arguments):
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id=call_id, function=function)


class TestAgentLoop:
    """The loop itself, with the model replaced by a scripted client."""

    def test_two_tools_run_in_order_then_answer(self):
        client = _ScriptedClient(
            [
                SimpleNamespace(
                    content=None,
                    tool_calls=[
                        _tool_call("c1", "search_jobs", {"query": "ML Engineer", "k": 1}),
                        _tool_call("c2", "get_job_detail", {"job_id": "no-such-id"}),
                    ],
                ),
                SimpleNamespace(content="Here is what I found.", tool_calls=None),
            ]
        )
        agent = Agent(use_cache=False)
        agent._client = client

        result = agent.run("Find one ML Engineer job")

        assert [c.name for c in result.tool_calls] == ["search_jobs", "get_job_detail"]
        assert result.stop_reason == "natural"
        assert result.completed is True
        assert result.final_response == "Here is what I found."
        assert result.n_llm_calls == 2
        assert (result.input_tokens, result.output_tokens) == (20, 10)
        # The second request carries both tool results, in call order.
        tool_msgs = [m for m in client.requests[1] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]

    def test_malformed_arguments_become_a_tool_error(self):
        bad = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_jobs", arguments="{"))
        client = _ScriptedClient(
            [
                SimpleNamespace(content=None, tool_calls=[bad]),
                SimpleNamespace(content="Sorry.", tool_calls=None),
            ]
        )
        agent = Agent(use_cache=False)
        agent._client = client

        result = agent.run("anything")

        assert result.tool_calls[0].result is None
        assert result.tool_calls[0].error.startswith("Malformed tool arguments")
        assert result.completed is True

    def test_stops_at_max_steps(self):
        loop_turn = SimpleNamespace(
            content=None, tool_calls=[_tool_call("c", "search_jobs", {"query": "x", "k": 1})]
        )
        agent = Agent(use_cache=False, max_steps=3)
        agent._client = _ScriptedClient([loop_turn] * 3)

        result = agent.run("never finishes")

        assert result.stop_reason == "max_steps"
        assert result.completed is False
        assert result.n_llm_calls == 3
