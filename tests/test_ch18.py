"""Tests for Chapter 18: Agentic AI.

Reference: book/ch18/README.md
Source: book/ch18/ch18_agentic_ai.py, talentlens/agent.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from talentlens.agent import Agent


class TestToolDispatch:
    def test_unknown_tool_returns_error_string(self):
        agent = Agent(use_cache=False)
        result, error = agent._dispatch_tool("not_a_real_tool", {})
        assert result is None
        assert error is not None
        assert "Unknown tool" in error

    def test_malformed_arguments_surface_type_error(self):
        agent = Agent(use_cache=False)
        result, error = agent._dispatch_tool("search_jobs", {"k": "not-an-int"})
        assert result is None
        assert error is not None
        assert "search_jobs" in error


class TestAgentRunWithStubbedLlm:
    def test_tool_calls_follow_llm_order(self, tmp_path):
        agent = Agent(use_cache=False, cache_dir=tmp_path / "cache", max_steps=4)

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="search_jobs",
                                    arguments=json.dumps({"query": "ML Engineer", "k": 2}),
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        second_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Found two ML Engineer postings.",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=12),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        with patch.object(agent, "_ensure_client", return_value=mock_client):
            result = agent.run("Find ML Engineer postings.")

        assert result.completed is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_jobs"
        assert result.tool_calls[0].error is None
        assert isinstance(result.tool_calls[0].result, list)

    def test_tool_failure_populates_error_on_tool_call(self, tmp_path):
        agent = Agent(use_cache=False, cache_dir=tmp_path / "cache", max_steps=3)

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_bad",
                                function=SimpleNamespace(
                                    name="classify_role",
                                    arguments=json.dumps({}),
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )
        final = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Recovered after tool error.",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [response, final]

        with patch.object(agent, "_ensure_client", return_value=mock_client):
            result = agent.run("Classify a role without providing job_id.")

        assert result.tool_calls[0].name == "classify_role"
        assert result.tool_calls[0].error is not None
        assert "classify_role" in result.tool_calls[0].error


@pytest.mark.skip(reason="Requires GROQ_API_KEY or OPENAI_API_KEY for live Agent.run")
def test_live_llm_agent_run():
    pass
