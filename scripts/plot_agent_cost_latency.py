"""Tiny script: plot cost vs latency across the five test queries.

Reads live traces and produces book/ch18/reports/figures/cost_latency.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

TRACES_DIR = Path("book/ch18/reports/traces/live")
FIG_PATH = Path("book/ch18/reports/figures/cost_latency.png")


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * 0.59 + (output_tokens / 1_000_000) * 0.79


def main() -> int:
    if not TRACES_DIR.exists():
        print(f"No traces at {TRACES_DIR}. Run the chapter executable first.")
        return 1

    rows = []
    for trace_path in sorted(TRACES_DIR.glob("*.json")):
        data = json.loads(trace_path.read_text())
        rows.append(
            {
                "query_id": data["query_id"],
                "latency": data["elapsed_seconds"],
                "cost": _estimate_cost(data["input_tokens"], data["output_tokens"]),
                "tool_calls": data["n_tool_calls"],
                "stop_reason": data["stop_reason"],
            }
        )

    if not rows:
        print("No trace files found in", TRACES_DIR)
        return 1

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    colours = {
        "natural": "tab:blue",
        "max_steps": "tab:orange",
        "error": "tab:red",
        "cached": "tab:gray",
    }
    for row in rows:
        ax.scatter(
            row["latency"],
            row["cost"],
            s=80 + row["tool_calls"] * 20,
            c=colours.get(row["stop_reason"], "black"),
            alpha=0.7,
        )
        ax.annotate(
            row["query_id"],
            (row["latency"], row["cost"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Wall-clock latency (s)")
    ax.set_ylabel("Estimated cost ($)")
    ax.set_title("Agent cost vs latency by query (point size = tool calls)")

    for reason, colour in colours.items():
        if any(r["stop_reason"] == reason for r in rows):
            ax.scatter([], [], c=colour, label=reason, s=80)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=120)
    print(f"Saved {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
