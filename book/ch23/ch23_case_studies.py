"""
Chapter 23: Real-World Case Studies
Data Voyage - Building TalentLens

TalentLens milestone: step back from TalentLens and look at three other
production systems I've built - Reflecta (voice AI wellness), Godam
(Nepal FMCG inventory), and RAGNav (open-source RAG library). What worked,
what broke, what you can borrow.

Run: python book/ch23/ch23_case_studies.py

Outputs:
    book/ch23/reports/figures/ch23_system_architectures.png
    book/ch23/reports/figures/ch23_lessons_matrix.png
    book/ch23/reports/case_studies_summary.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
_THIS_DIR = Path(__file__).resolve().parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


@dataclass
class Config:
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"


# ---------------------------------------------------------------------------
# Case study data
# ---------------------------------------------------------------------------

CASE_STUDIES = [
    {
        "name": "Reflecta",
        "tagline": "Voice-first AI wellness companion",
        "url": "getreflecta.com",
        "stack": [
            "Next.js 15",
            "Neon Postgres + pgvector",
            "Groq/Llama 3.3 70B",
            "Bolna AI (telephony)",
            "Twilio Verify",
            "Vercel",
        ],
        "what_it_does": (
            "Users call a phone number and talk to an AI wellness companion. "
            "The call is transcribed, analysed by an LLM, and summarised into "
            "a personal dashboard. The LLM tracks mood patterns over time using "
            "pgvector similarity search on past conversation embeddings."
        ),
        "key_lessons": [
            "Voice latency is unforgiving — anything over 800ms feels broken. "
            "Groq's inference speed (200-400ms) is what made this viable.",
            "Telephony adds a layer of failure modes that don't exist in web apps. "
            "Bolna handles retries and audio quality; don't build this yourself.",
            "The hardest part wasn't the AI — it was the security hardening: "
            "webhook signature verification, rate limiting per phone number, "
            "admin auth. Ship these before the first real user.",
            "pgvector for conversation history retrieval works well under 100k records. "
            "At larger scale, move the vector operations to a dedicated service.",
        ],
        "what_broke": [
            "The 'Call me now' backend endpoint silently failed for 3 days because "
            "Vercel logs weren't being monitored. Add Sentry or equivalent on day 1.",
            "LLM hallucinations in a wellness context are a real risk — the model "
            "occasionally 'remembered' things the user never said. "
            "Groundedness checks on every response are mandatory.",
        ],
        "metrics": {
            "stack_complexity": 7,
            "months_to_ship": 3,
            "lines_of_code": 4200,
            "prod_incidents": 3,
        },
    },
    {
        "name": "Godam",
        "tagline": "FMCG trade and inventory PWA for Nepal",
        "url": "getgodam.com",
        "stack": ["Next.js 15", "Supabase (PostgreSQL + JWT)", "jsPDF", "Vercel", "Namecheap DNS"],
        "what_it_does": (
            "Inventory management for a Nepal-based FMCG distribution business. "
            "Three user roles: admin, manager, field agent. Features include "
            "godown management, cheque tracking with photo uploads, daily sales "
            "reports as PDFs, and costing sheet generation. "
            "Runs as a PWA — works offline, installable on Android."
        ),
        "key_lessons": [
            "Nepali business requirements differ from Indian or Western defaults. "
            "Fiscal year follows Bikram Sambat calendar. Reports use Nepali "
            "number formatting. Build for the actual user, not an imaginary one.",
            "PWA offline-first is genuinely hard. IndexedDB sync with Supabase "
            "on reconnect took 3x longer to implement than estimated.",
            "Monthly subscription (NPR 16,000/month) sustains the product. "
            "Per-feature pricing would have created friction. One price, everything in.",
            "PDF generation with jsPDF is sufficient for business reports. "
            "Don't use a PDF service for something this simple.",
        ],
        "what_broke": [
            "Supabase row-level security policies were misconfigured at launch. "
            "Field agents could see each other's data for 6 hours. "
            "Always test RLS with a non-admin user before go-live.",
            "Photo uploads to Supabase storage had file size limits we hadn't "
            "tested on real mobile networks. Added client-side compression.",
        ],
        "metrics": {
            "stack_complexity": 4,
            "months_to_ship": 2,
            "lines_of_code": 8500,
            "prod_incidents": 2,
        },
    },
    {
        "name": "RAGNav",
        "tagline": "Open-source hybrid retrieval library",
        "url": "github.com/irfanalidv/RAGNav",
        "stack": [
            "Python",
            "BM25 (rank-bm25)",
            "NumPy",
            "SentenceTransformers",
            "PyMuPDF",
            "GitHub Actions",
        ],
        "what_it_does": (
            "A PyPI library (MIT) for hybrid BM25 + dense retrieval with "
            "structure-aware expansion, fully offline. On 500 SQuAD questions, "
            "recall@3 is 0.956 hybrid vs 0.932 BM25-only and 0.906 embedding-only, "
            "with results committed in the repository. Its companion library, "
            "ragfallback, adds a CI regression gate for RAG pipelines."
        ),
        "key_lessons": [
            "Lead the README with one reproducible number. The SQuAD benchmark "
            "and the script that regenerates it say more than any architecture "
            "description.",
            "Fuse rankings, not raw scores: BM25 and cosine similarity live on "
            "different scales, so Reciprocal Rank Fusion combines rank positions.",
            "Report the hard cases. On legal contracts (CUAD) block-level "
            "recall@3 is 0.047, and the README says so under Limitations.",
            "Publishing to PyPI is the easy part. Tests, CI, and documentation "
            "are the ongoing work.",
        ],
        "what_broke": [
            "Release 0.3.0 combined document-level and block-level access rules "
            "with OR, so a document's permissions could widen access to a "
            "restricted block. 0.4.0 changed it to AND, alongside new CI and "
            "test coverage of about 72%.",
            "The library shipped before it had CI; lint, tests, and coverage "
            "arrived only in 0.4.0.",
        ],
        "metrics": {
            "stack_complexity": 5,
            "months_to_ship": 1,
            "lines_of_code": 5742,
            "prod_incidents": 1,
        },
    },
]


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

RADAR_DIMENSIONS: tuple[str, ...] = (
    "Stack\ncomplexity",
    "Months\nto ship",
    "Code\nvolume (kLoC)",
    "Prod\nincidents",
    "Open\nsource",
)


def _normalise_radar_score(val: float, lo: float, hi: float) -> float:
    """Map a raw metric to the 0–10 radar axis used in plot_lessons_matrix."""
    return (val - lo) / (hi - lo) * 10


def build_radar_scores() -> dict[str, list[float]]:
    """Normalised radar scores for Reflecta, Godam, and RAGNav.

    Values are derived from each entry's ``metrics`` in CASE_STUDIES plus a
    fixed open-source flag (RAGNav only). Exported for tests and for
    plot_lessons_matrix so the chart and assertions cannot drift apart.
    """
    by_name = {study["name"]: study["metrics"] for study in CASE_STUDIES}
    return {
        "Reflecta": [
            _normalise_radar_score(by_name["Reflecta"]["stack_complexity"], 1, 10),
            _normalise_radar_score(by_name["Reflecta"]["months_to_ship"], 1, 6),
            _normalise_radar_score(by_name["Reflecta"]["lines_of_code"] / 1000, 1, 10),
            _normalise_radar_score(by_name["Reflecta"]["prod_incidents"], 0, 5),
            0.0,
        ],
        "Godam": [
            _normalise_radar_score(by_name["Godam"]["stack_complexity"], 1, 10),
            _normalise_radar_score(by_name["Godam"]["months_to_ship"], 1, 6),
            _normalise_radar_score(by_name["Godam"]["lines_of_code"] / 1000, 1, 10),
            _normalise_radar_score(by_name["Godam"]["prod_incidents"], 0, 5),
            0.0,
        ],
        "RAGNav": [
            _normalise_radar_score(by_name["RAGNav"]["stack_complexity"], 1, 10),
            _normalise_radar_score(by_name["RAGNav"]["months_to_ship"], 1, 6),
            _normalise_radar_score(by_name["RAGNav"]["lines_of_code"] / 1000, 1, 10),
            _normalise_radar_score(by_name["RAGNav"]["prod_incidents"], 0, 5),
            10.0,
        ],
    }


def plot_system_architectures(cfg: Config) -> Path:
    """Visual comparison of the three system stacks."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    colors = {
        "Frontend": "#2196F3",
        "Backend": "#4CAF50",
        "Database": "#FF9800",
        "AI/ML": "#9C27B0",
        "Infrastructure": "#607D8B",
        "Monitoring": "#F44336",
    }
    layer_map = {
        "Next.js 15": "Frontend",
        "Vercel": "Infrastructure",
        "Neon Postgres + pgvector": "Database",
        "Supabase (PostgreSQL + JWT)": "Database",
        "Groq/Llama 3.3 70B": "AI/ML",
        "Bolna AI (telephony)": "AI/ML",
        "Twilio Verify": "Backend",
        "jsPDF": "Backend",
        "Namecheap DNS": "Infrastructure",
        "Python": "Backend",
        "BM25 (rank-bm25)": "AI/ML",
        "NumPy": "Backend",
        "SentenceTransformers": "AI/ML",
        "PyMuPDF": "Backend",
        "GitHub Actions": "Infrastructure",
    }

    for ax, study in zip(axes, CASE_STUDIES):
        stack_items = study["stack"]
        y_positions = np.linspace(0.85, 0.1, len(stack_items))
        for tech, y in zip(stack_items, y_positions):
            layer = layer_map.get(tech, "Backend")
            color = colors[layer]
            ax.barh(y, 0.9, left=0.05, height=0.09, color=color, alpha=0.75, edgecolor="white")
            ax.text(
                0.5,
                y,
                tech,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{study['name']}\n{study['tagline']}", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.01, study["url"], ha="center", fontsize=8, color="gray")

    # Legend
    legend_elements = [mpatches.Patch(color=v, alpha=0.75, label=k) for k, v in colors.items()]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=6,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.suptitle("Three Production Systems — Stack Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.figures_dir / "ch23_system_architectures.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_lessons_matrix(cfg: Config) -> Path:
    """Radar / spider chart comparing the three systems on key dimensions."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    dimensions = list(RADAR_DIMENSIONS)
    n = len(dimensions)
    scores = build_radar_scores()

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = ["#2196F3", "#4CAF50", "#9C27B0"]
    for (name, vals), color in zip(scores.items(), colors_radar):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "o-", linewidth=2, color=color, label=name, markersize=6)
        ax.fill(angles, vals_closed, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7)
    ax.set_title("Case Study Comparison — Key Dimensions", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    out = cfg.figures_dir / "ch23_lessons_matrix.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_case_studies_summary(cfg: Config) -> Path:
    """Write the Markdown case studies document."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-World Case Studies — Three Production AI Systems",
        "",
        "Three systems I built and shipped. What worked, what broke, " "and what you can borrow.\n",
    ]
    for study in CASE_STUDIES:
        lines += [
            f"## {study['name']} — {study['tagline']}",
            f"*{study['url']}*\n",
            "**What it does:**",
            study["what_it_does"],
            "",
            f"**Stack:** {', '.join(study['stack'])}\n",
            "**Key lessons:**",
        ]
        for lesson in study["key_lessons"]:
            lines.append(f"- {lesson}")
        lines += ["", "**What broke (and why):**"]
        for broke in study["what_broke"]:
            lines.append(f"- {broke}")
        lines += ["", "---", ""]

    lines += [
        "## Cross-cutting lessons",
        "",
        "These patterns showed up across all three projects:\n",
        "**1. Ship monitoring before users.** Every incident above was caught "
        "late because I didn't have logging and alerting in place before the "
        "first real user. Add Sentry, structured logs, and a health endpoint "
        "before you tell anyone about the product.",
        "",
        "**2. The AI part is rarely the hard part.** The hard parts are: auth, "
        "rate limiting, mobile edge cases, payment flows, and keeping "
        "dependencies working 6 months later. The LLM call is 10 lines.",
        "",
        "**3. Charge from day one.** Godam's NPR 16,000/month subscription "
        "made it sustainable and gave the client skin in the game. Free tiers "
        "attract users who don't need what you built.",
        "",
        "**4. Constraints are clarifying.** Building for Nepal (different "
        "calendar, different number formatting, unreliable network) forced "
        "better engineering decisions than building for a generic 'global' user.",
    ]

    out = cfg.reports_dir / "case_studies_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  CHAPTER 23: REAL-WORLD CASE STUDIES")
    logger.info("  Reflecta | Godam | RAGNav")
    logger.info("=" * 60)

    for study in CASE_STUDIES:
        logger.info(f"\n  {study['name']}: {study['tagline']}")
        logger.info(f"  Stack: {', '.join(study['stack'][:3])}...")
        logger.info(
            f"  Lessons: {len(study['key_lessons'])} | "
            f"Incidents: {study['metrics']['prod_incidents']}"
        )

    logger.info("\n[1/3] Plotting system architectures...")
    plot_system_architectures(cfg)

    logger.info("\n[2/3] Plotting lessons matrix...")
    plot_lessons_matrix(cfg)

    logger.info("\n[3/3] Writing case studies summary...")
    write_case_studies_summary(cfg)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 23 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info(f"  Report:  {cfg.reports_dir}/case_studies_summary.md")
    logger.info("\nCross-cutting lesson: ship monitoring before users.")
    logger.info("Next: Chapter 24 — The India Playbook")


if __name__ == "__main__":
    main()
