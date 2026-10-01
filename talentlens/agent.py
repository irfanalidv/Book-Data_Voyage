"""Agentic AI loop for TalentLens.

Chapter 18 builds and benchmarks a four-tool agent that
discovers job postings autonomously. This module is the chapter's
deliverable; tools wrap ch16 (RAG search), ch17 (LLM generation),
and ch10 (role classifier).

Public API (stable; consumed by ch18, future ch22):

    Agent: class - .run(query) -> AgentResult
    AgentResult: dataclass - final response, tool trace, latency
    TOOLS: dict - registered tool functions, name -> callable
    TOOL_SPECS: list[dict[str, Any]] - JSON Schema specs for the LLM
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation in an agent run."""

    name: str
    arguments: dict[str, Any]
    result: Any
    elapsed_seconds: float
    error: str | None = None


@dataclass
class AgentResult:
    """Final result of one agent run, plus full trace.

    Attributes:
        query: The original user query.
        final_response: The agent's last assistant message (text).
        tool_calls: Ordered list of every tool call the agent made.
        elapsed_seconds: Wall-clock from .run() entry to return.
        n_llm_calls: How many times the agent called the LLM.
        n_tool_calls: How many tool invocations total.
        input_tokens: Sum across all LLM calls.
        output_tokens: Sum across all LLM calls.
        completed: True if the loop ended naturally; False if it hit
            MAX_STEPS or errored.
        stop_reason: 'natural' | 'max_steps' | 'error' | 'cached'.
    """

    query: str
    final_response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    completed: bool = True
    stop_reason: str = "natural"
    error_detail: str | None = None


# TOOL_SPECS are the JSON Schema definitions that get passed to
# Groq's tool-calling API. The "description" field is what the
# LLM reads to choose tools - write it as if explaining the tool
# to a junior engineer who has never seen this codebase. The
# property descriptions matter too; vague descriptions produce
# vague argument values.
#
# Schema follows OpenAI's function-calling format, which Groq
# mirrors. type: "object" + properties + required is the standard.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search the TalentLens job-posting corpus for postings "
                "that match a free-text query. Uses hybrid retrieval "
                "(keyword + semantic). Returns up to k postings ranked "
                "by relevance, each with job_id, title, company, "
                "salary_annual_inr, and a short excerpt. Use this as "
                "the first step when the user asks to find or filter "
                "postings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text search query. Can be a role "
                            "name ('ML Engineer'), a skill ('PyTorch'), "
                            "a city ('Bangalore'), or a combination."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": (
                            "Maximum number of postings to return. "
                            "Default 5. Use larger k (10-20) only when "
                            "the user explicitly asks for more."
                        ),
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_detail",
            "description": (
                "Fetch full details for one job posting by job_id. "
                "Returns the complete record including description, "
                "skills_normalised, salary range, and city. Use this "
                "after search_jobs to read a specific posting in "
                "depth — search_jobs returns excerpts only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "The job_id from search_jobs results, e.g. " "'demo_42_0123'."
                        ),
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_role",
            "description": (
                "Classify a job posting's actual role using the "
                "TalentLens classifier (Chapter 10 v2 model). Returns "
                "the predicted role_category from {AI Engineer, ML "
                "Engineer, Data Scientist, Data Engineer, Data "
                "Analyst} plus a confidence score. Use this when the "
                "user wants to verify that a posting actually matches "
                "the role they're searching for — titles can be "
                "misleading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "job_id of the posting to classify.",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_for_candidate",
            "description": (
                "Generate a candidate-specific summary of a job "
                "posting. Highlights how the candidate's skills match "
                "the posting's requirements and what's missing. Uses "
                "the LLM under the hood (Chapter 17). Use this when "
                "the user wants a personalised take on a specific "
                "posting; do not use for general posting summaries — "
                "get_job_detail is faster and free."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "job_id of the posting to summarise.",
                    },
                    "candidate_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of canonical skill names representing "
                            "the candidate. Use names from "
                            "talentlens.skills.CANONICAL_SKILLS exactly."
                        ),
                    },
                },
                "required": ["job_id", "candidate_skills"],
            },
        },
    },
]

# System prompt that primes the agent. The wording matters more
# than most tutorials suggest - "use search_jobs first" is the
# difference between an agent that searches and one that
# hallucinates job_ids out of thin air. Keep this short and
# imperative; Llama 3.3 70B follows direct instructions better
# than elaborate ones.
AGENT_SYSTEM_PROMPT = (
    "You are a job-discovery assistant for TalentLens. You help "
    "users find and evaluate job postings using the tools "
    "available to you.\n\n"
    "Rules:\n"
    "1. ALWAYS use search_jobs first to find postings. Never "
    "invent job_ids — they only exist if search_jobs returned "
    "them.\n"
    "2. After search_jobs, use get_job_detail to read specific "
    "postings.\n"
    "3. Use classify_role when the user wants to verify a "
    "posting's actual role.\n"
    "4. Use summarise_for_candidate when the user provides their "
    "own skills.\n"
    "5. If a search returns no results, tell the user honestly "
    "— do not invent results.\n"
    "6. Keep responses concise. The user reads your final reply, "
    "not the tool call traces."
)


# TOOLS maps name -> callable for the agent loop's dispatch.
# Populated at module import time after the four functions are
# defined below.
TOOLS: dict[str, Any] = {}


def search_jobs(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Search the TalentLens corpus via hybrid retrieval.

    Wraps Chapter 16's hybrid_search but returns LLM-friendly dicts
    (no DataFrames, no numpy types - only str / int / float / list).
    The LLM will receive this as JSON, so types matter.

    Args:
        query: Free-text search query.
        k: Max number of postings to return. Clamped to [1, 20].

    Returns:
        List of dicts; each dict has keys: job_id, title, company,
        city, salary_annual_inr, excerpt, score. May be shorter than
        k if fewer postings match.
    """
    k = max(1, min(int(k), 20))

    import pandas as pd

    from book.ch16.ch16_rag_vector_search import keyword_score
    from talentlens.paths import jobs_clean_path

    df = pd.read_csv(jobs_clean_path())

    df = df.copy()
    df["_text"] = (
        df["title"].fillna("")
        + " "
        + df["description"].fillna("")
        + " "
        + df["skills_normalised"].fillna("")
    )
    df["_score"] = df["_text"].apply(lambda t: keyword_score(query, t))
    ranked = df.nlargest(k, "_score")

    results: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        if row["_score"] <= 0.0:
            continue
        excerpt = str(row.get("description", ""))[:200]
        salary = row.get("salary_annual_inr")
        results.append(
            {
                "job_id": str(row["job_id"]),
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "city": str(row.get("city", "")),
                "salary_annual_inr": (float(salary) if pd.notna(salary) else None),
                "excerpt": excerpt,
                "score": round(float(row["_score"]), 3),
            }
        )
    return results


def get_job_detail(job_id: str) -> dict[str, Any]:
    """Fetch one posting by job_id.

    Returns the full record as a JSON-serialisable dict. Errors
    if job_id is not found, with a structured error envelope so
    the agent can recover.

    Args:
        job_id: e.g. 'demo_42_0123'.

    Returns:
        Dict with keys: job_id, title, company, city, description,
        skills_normalised, salary_min, salary_max, salary_annual_inr,
        role_category. On not-found, returns
        {"error": "not_found", "job_id": ...}.
    """
    import pandas as pd

    from talentlens.paths import jobs_clean_path

    df = pd.read_csv(jobs_clean_path())
    match = df[df["job_id"] == job_id]
    if match.empty:
        return {"error": "not_found", "job_id": job_id}

    row = match.iloc[0]

    def _val(col: str, default: Any = None) -> Any:
        v = row.get(col, default)
        if pd.isna(v):
            return None
        if isinstance(v, (int, float)):
            return float(v) if isinstance(v, float) else int(v)
        return str(v)

    return {
        "job_id": _val("job_id"),
        "title": _val("title"),
        "company": _val("company"),
        "city": _val("city"),
        "description": _val("description"),
        "skills_normalised": _val("skills_normalised"),
        "salary_min": _val("salary_min"),
        "salary_max": _val("salary_max"),
        "salary_annual_inr": _val("salary_annual_inr"),
        "role_category": _val("role_category"),
    }


def classify_role(job_id: str) -> dict[str, Any]:
    """Classify a posting's role via the Chapter 10 v2 classifier.

    Args:
        job_id: e.g. 'demo_42_0123'.

    Returns:
        Dict with keys: job_id, predicted_role, confidence. On
        not-found, returns {"error": "not_found", "job_id": ...}.
        On classifier-unavailable (model file missing), returns
        {"error": "model_unavailable", "job_id": ...}.
    """
    import joblib
    import pandas as pd

    from book.ch10.ch10_feature_engineering_selection import _build_feature_text
    from talentlens.features import engineer_features
    from talentlens.paths import role_classifier_path

    detail = get_job_detail(job_id)
    if "error" in detail:
        return detail

    model_path = role_classifier_path()
    if not model_path.exists():
        return {
            "error": "model_unavailable",
            "job_id": job_id,
            "detail": (
                "No classifier model on disk. Run Chapter 10's "
                "executable to train role_classifier_v2.joblib."
            ),
        }

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        return {
            "error": "model_load_failed",
            "job_id": job_id,
            "detail": str(e),
        }

    df = pd.DataFrame(
        [
            {
                "title": detail["title"],
                "description": detail["description"] or "",
                "skills_normalised": detail["skills_normalised"] or "",
                "salary_min": detail["salary_min"],
                "salary_max": detail["salary_max"],
                "salary_annual_inr": detail["salary_annual_inr"],
                "city": detail["city"],
            }
        ]
    )
    df = engineer_features(df)
    df["feature_text"] = _build_feature_text(df)

    try:
        label = pipeline.predict(df)[0]
        confidence = float(pipeline.predict_proba(df).max())
    except Exception as e:
        return {
            "error": "prediction_failed",
            "job_id": job_id,
            "detail": str(e),
        }

    return {
        "job_id": job_id,
        "predicted_role": str(label),
        "confidence": round(confidence, 3),
    }


def summarise_for_candidate(
    job_id: str,
    candidate_skills: list[str],
) -> str:
    """Generate a candidate-specific summary of one posting.

    Does not call an LLM - returns a deterministic skill-overlap
    summary so the agent loop avoids nested LLM latency/cost.

    Args:
        job_id: e.g. 'demo_42_0123'.
        candidate_skills: List of canonical skill names.

    Returns:
        A short markdown summary. On error, returns a string
        beginning with "ERROR:" - the agent can detect this and
        handle gracefully.
    """
    detail = get_job_detail(job_id)
    if "error" in detail:
        return f"ERROR: cannot find {job_id}"

    if not candidate_skills:
        return "ERROR: candidate_skills is empty; cannot tailor summary"

    posting_skills_str = detail.get("skills_normalised") or ""
    posting_skills = {s.strip() for s in posting_skills_str.split("|") if s.strip()}
    candidate_set = set(candidate_skills)
    matched = sorted(posting_skills & candidate_set)
    missing = sorted(posting_skills - candidate_set)

    lines = [
        f"**{detail['title']}** at {detail['company']} ({detail['city']})",
        "",
        f"Salary: {detail.get('salary_annual_inr', 'not disclosed')}",
        "",
        f"Skill match: {len(matched)}/{len(posting_skills)} " f"of the posting's listed skills",
        f"- Matched: {', '.join(matched) if matched else '(none)'}",
        f"- Missing: {', '.join(missing) if missing else '(none)'}",
    ]
    return "\n".join(lines)


DEFAULT_AGENT_MODEL = "openai/gpt-oss-120b"


class Agent:
    """A minimal Groq tool-calling agent.

    Writes responses to a cache directory by default; pass
    ``use_cache=False`` (or set AGENT_NO_CACHE=1) for live runs.
    The cache key is (query, model, tools_version) - changing any
    component invalidates the entry.
    """

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 8,
        use_cache: bool = True,
        cache_dir: str | None = None,
        temperature: float = 0.0,
    ):
        # Hosted model names are retired every few months; override with
        # TALENTLENS_AGENT_MODEL rather than editing code.
        self.model = model or os.environ.get("TALENTLENS_AGENT_MODEL", DEFAULT_AGENT_MODEL)
        self.max_steps = max_steps
        self.use_cache = use_cache and not os.environ.get("AGENT_NO_CACHE", "")
        self.cache_dir = Path(cache_dir or "book/ch18/reports/traces/cache")
        self.temperature = temperature
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Lazy-load groq client. Tests that don't run agents shouldn't pay import cost."""
        if self._client is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            import groq

            self._client = groq.Groq()
        return self._client

    def _cache_key(self, query: str) -> str:
        """Hash query + model + max_steps + tool names."""
        tool_names = sorted(TOOLS.keys())
        payload = json.dumps(
            {
                "query": query,
                "model": self.model,
                "max_steps": self.max_steps,
                "tools": tool_names,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _load_cached(self, key: str) -> AgentResult | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            tool_calls = [ToolCall(**tc) for tc in data.pop("tool_calls", [])]
            result = AgentResult(**data, tool_calls=tool_calls)
            return replace(result, stop_reason="cached")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"cache read failed for {key}: {e}")
            return None

    def _save_cached(self, key: str, result: AgentResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        payload = {
            "query": result.query,
            "final_response": result.final_response,
            "elapsed_seconds": result.elapsed_seconds,
            "n_llm_calls": result.n_llm_calls,
            "n_tool_calls": result.n_tool_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "completed": result.completed,
            "stop_reason": result.stop_reason,
            "error_detail": result.error_detail,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "elapsed_seconds": tc.elapsed_seconds,
                    "error": tc.error,
                }
                for tc in result.tool_calls
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str | None]:
        """Execute one tool call. Returns (result, error_string_or_None)."""
        if name not in TOOLS:
            return None, f"Unknown tool: {name!r}. Available: {sorted(TOOLS)}"
        try:
            result = TOOLS[name](**arguments)
            return result, None
        except TypeError as e:
            return None, f"Tool {name} called with wrong arguments: {e}"
        except Exception as e:
            return None, f"Tool {name} raised: {type(e).__name__}: {e}"

    def run(self, query: str) -> AgentResult:
        """Execute one agent run against ``query``."""
        start = time.monotonic()

        if self.use_cache:
            key = self._cache_key(query)
            cached = self._load_cached(key)
            if cached is not None:
                return cached

        client = self._ensure_client()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        tool_calls_made: list[ToolCall] = []
        n_llm_calls = 0
        input_tokens = 0
        output_tokens = 0
        stop_reason = "max_steps"
        final_response = ""

        for _ in range(self.max_steps):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SPECS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=1024,
                )
            except Exception as e:
                error_detail = f"{type(e).__name__}: {e}"
                body = getattr(e, "body", None)
                if body is not None:
                    error_detail += f"\nbody: {json.dumps(body, default=str)[:500]}"
                elapsed = time.monotonic() - start
                result = AgentResult(
                    query=query,
                    final_response="",
                    tool_calls=tool_calls_made,
                    elapsed_seconds=elapsed,
                    n_llm_calls=n_llm_calls,
                    n_tool_calls=len(tool_calls_made),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    completed=False,
                    stop_reason="error",
                    error_detail=error_detail,
                )
                if self.use_cache:
                    self._save_cached(self._cache_key(query), result)
                return result
            n_llm_calls += 1
            if response.usage:
                input_tokens += response.usage.prompt_tokens
                output_tokens += response.usage.completion_tokens

            msg = response.choices[0].message

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                final_response = msg.content or ""
                stop_reason = "natural"
                break

            for tc in msg.tool_calls:
                tool_start = time.monotonic()
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    arguments = {}
                    result = None
                    error = f"Malformed tool arguments: {e}"
                else:
                    result, error = self._dispatch_tool(tc.function.name, arguments)

                tool_elapsed = time.monotonic() - tool_start
                tool_calls_made.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=arguments,
                        result=result,
                        elapsed_seconds=tool_elapsed,
                        error=error,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(error if error else result, default=str),
                    }
                )

        elapsed = time.monotonic() - start
        result = AgentResult(
            query=query,
            final_response=final_response,
            tool_calls=tool_calls_made,
            elapsed_seconds=elapsed,
            n_llm_calls=n_llm_calls,
            n_tool_calls=len(tool_calls_made),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            completed=(stop_reason == "natural"),
            stop_reason=stop_reason,
            error_detail=None,
        )

        if self.use_cache:
            self._save_cached(self._cache_key(query), result)

        return result


# Register tools in the dispatch table. Done at module-import
# time so the agent loop can look up tools by name. Order doesn't
# matter; the LLM uses TOOL_SPECS for selection, this dict for
# dispatch.
TOOLS.update(
    {
        "search_jobs": search_jobs,
        "get_job_detail": get_job_detail,
        "classify_role": classify_role,
        "summarise_for_candidate": summarise_for_candidate,
    }
)
