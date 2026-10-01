"""Chapter 18: Agentic AI - agent runs against TalentLens corpus.

Loads the five test queries from query_viability.txt, runs each
through the Agent (from talentlens.agent), captures traces,
writes them to book/ch18/reports/traces/live/, and updates
reports/agent_report.md with measured numbers.

By default the script uses the LLM cache; pass --live to bypass
and make fresh calls. The chapter's committed numbers come from
a live run.

Run:
    python book/ch18/ch18_agentic_ai.py          # cached
    python book/ch18/ch18_agentic_ai.py --live   # live, ~$0.10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from talentlens.agent import Agent, AgentResult
from talentlens.paths import REPO_ROOT

logger = logging.getLogger(__name__)


TEST_QUERIES: list[dict[str, str]] = [
    {
        "id": "ml_engineer_simple",
        "query": "Find ML Engineer postings.",
    },
    {
        "id": "ml_engineer_classify_chain",
        "query": (
            "Find ML Engineer postings, then classify each to "
            "confirm the actual role matches the title."
        ),
    },
    {
        "id": "no_good_answer",
        "query": "Find postings for Quantum Engineer roles.",
    },
    {
        "id": "compare_top_three",
        "query": ("Compare the top 3 AI Engineer postings on salary " "and required skills."),
    },
    {
        "id": "ambiguous",
        "query": "Show me good jobs.",
    },
]


TRACES_LIVE_DIR = REPO_ROOT / "book" / "ch18" / "reports" / "traces" / "live"
REPORT_PATH = REPO_ROOT / "book" / "ch18" / "reports" / "agent_report.md"


def _save_trace(query_id: str, result: AgentResult) -> Path:
    """Write the full trace to the live directory."""
    TRACES_LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_LIVE_DIR / f"{query_id}.json"
    payload = {
        "query_id": query_id,
        "query": result.query,
        "final_response": result.final_response,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "error_detail": result.error_detail,
        "n_llm_calls": result.n_llm_calls,
        "n_tool_calls": result.n_tool_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "tool_calls": [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "elapsed_seconds": round(tc.elapsed_seconds, 3),
                "error": tc.error,
            }
            for tc in result.tool_calls
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD using Groq's published Llama 3.3 70B pricing."""
    input_cost = (input_tokens / 1_000_000) * 0.59
    output_cost = (output_tokens / 1_000_000) * 0.79
    return round(input_cost + output_cost, 5)


def _write_report(
    results: list[tuple[dict, AgentResult]],
    report_path: Path,
    mode: str,
) -> None:
    """Update the report's measured sections from a run.

    Regenerates only "Per-query results". Other narrative sections stay authored.
    """
    if not report_path.exists():
        logger.warning(f"Report not found: {report_path} — nothing to update.")
        return

    lines = report_path.read_text().splitlines()

    new_lines: list[str] = []
    skip_until_next_section = False
    for line in lines:
        if skip_until_next_section:
            if line.startswith("## "):
                skip_until_next_section = False
                new_lines.append(line)
            continue

        if line.strip() == "## Per-query results":
            new_lines.append(line)
            new_lines.append("")
            new_lines.append(
                f"_Numbers from the recorded live run (Groq, Llama 3.3 70B, May 2026; "
                f"{'regenerated from a new live run' if mode == 'live' else 'replayed from the committed cache'}). "
                f"Cost computed from token counts × Groq pricing for "
                f"Llama 3.3 70B as of Q4 2025 ($0.59/M input, $0.79/M "
                f"output). Wall-clock latency has high variance on "
                f"Groq's free tier due to rate-limit retries — cost "
                f"per query is stable across runs, time per query is "
                f"not._"
            )
            new_lines.append("")
            new_lines.append(
                "| Query ID | Tool calls | LLM calls | Latency (s) " "| Cost ($) | Stop reason |"
            )
            new_lines.append("|---|---|---|---|---|---|")

            total_tool_calls = 0
            total_llm_calls = 0
            total_latency = 0.0
            total_cost = 0.0
            for q, r in results:
                cost = _estimate_cost(r.input_tokens, r.output_tokens)
                total_cost += cost
                total_tool_calls += r.n_tool_calls
                total_llm_calls += r.n_llm_calls
                total_latency += r.elapsed_seconds
                new_lines.append(
                    f"| {q['id']} | {r.n_tool_calls} | {r.n_llm_calls} "
                    f"| {r.elapsed_seconds:.1f} | ${cost:.5f} "
                    f"| {r.stop_reason} |"
                )
            new_lines.append(
                f"| **Total** | **{total_tool_calls}** "
                f"| **{total_llm_calls}** | **{total_latency:.1f}** "
                f"| **${total_cost:.5f}** | — |"
            )
            new_lines.append("")
            new_lines.append(
                "Full traces are in `book/ch18/reports/traces/live/`. "
                "Each trace records every tool call's arguments, "
                "result, latency, and any error."
            )
            new_lines.append("")
            skip_until_next_section = True
            continue

        new_lines.append(line)

    report_path.write_text("\n".join(new_lines) + "\n")
    logger.info(f"  report updated: {report_path}")


# The model the committed traces were recorded with (Groq, May 2026). Groq has
# since retired it; replay mode reads the cache, so the name only keys the files.
RECORDED_MODEL = "llama-3.3-70b-versatile"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Bypass cache; make fresh LLM calls.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.live:
        agent = Agent(use_cache=False)
    else:
        # Replay the recorded run from the committed cache: no network, no key,
        # identical traces on every machine.
        agent = Agent(model=RECORDED_MODEL, use_cache=True)
        missing = [
            q["id"]
            for q in TEST_QUERIES
            if not (agent.cache_dir / f"{agent._cache_key(q['query'])}.json").exists()
        ]
        if missing:
            logger.error(
                f"No recorded run for {missing} in {agent.cache_dir}. "
                "Replay needs the committed cache; use --live with GROQ_API_KEY to record one."
            )
            return 1
    mode = "live" if args.live else "replay of recorded run"
    logger.info(f"Running {len(TEST_QUERIES)} test queries ({mode}).")

    results: list[tuple[dict, AgentResult]] = []
    for q in TEST_QUERIES:
        logger.info(f"  query: {q['id']}")
        result = agent.run(q["query"])
        results.append((q, result))
        _save_trace(q["id"], result)
        logger.info(
            f"    -> {result.n_tool_calls} tool calls, "
            f"{result.elapsed_seconds:.1f}s, "
            f"{result.stop_reason}"
        )

    logger.info("")
    logger.info(
        f"{'Query ID':<30} {'Tools':>6} {'LLM':>5} {'Time(s)':>9} " f"{'Cost($)':>9} {'Stop':>10}"
    )
    logger.info("-" * 80)
    total_cost = 0.0
    for q, r in results:
        cost = _estimate_cost(r.input_tokens, r.output_tokens)
        total_cost += cost
        logger.info(
            f"{q['id']:<30} {r.n_tool_calls:>6} {r.n_llm_calls:>5} "
            f"{r.elapsed_seconds:>9.1f} {cost:>9.5f} {r.stop_reason:>10}"
        )
    logger.info("-" * 80)
    logger.info(f"Total estimated cost: ${total_cost:.5f} ({mode})")
    _write_report(results, REPORT_PATH, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
