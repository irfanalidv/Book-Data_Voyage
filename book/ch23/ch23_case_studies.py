"""
Chapter 23: Real-World Case Studies
Data Voyage - Building TalentLens

TalentLens milestone: step back from TalentLens and look at four other
systems I've built: Reflecta (voice AI wellness), Godam (Nepal FMCG
inventory), RAGNav (open-source RAG library), and StackSift (B2B product
intelligence API). What worked, what broke, what you can borrow.

Run: python book/ch23/ch23_case_studies.py

Outputs:
    book/ch23/reports/figures/ch23_system_architectures.png
    book/ch23/reports/figures/ch23_lessons_matrix.png
    book/ch23/reports/case_studies_summary.md
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
_THIS_DIR = Path(__file__).resolve().parent
SAVE_DPI = 300
plt.rcParams["font.family"] = "DejaVu Sans"


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
            "Voice latency is unforgiving: anything over 800ms feels broken. "
            "Groq's inference speed (200-400ms) is what made this viable.",
            "Telephony adds a layer of failure modes that don't exist in web apps. "
            "Bolna handles retries and audio quality; don't build this yourself.",
            "The hardest part wasn't the AI. It was the security hardening: "
            "webhook signature verification, rate limiting per phone number, "
            "admin auth. Ship these before the first real user.",
            "pgvector for conversation history retrieval works well under 100k records. "
            "At larger scale, move the vector operations to a dedicated service.",
        ],
        "what_broke": [
            "The 'Call me now' backend endpoint silently failed for 3 days because "
            "Vercel logs weren't being monitored. Add Sentry or equivalent on day 1.",
            "LLM hallucinations in a wellness context are a real risk: the model "
            "occasionally 'remembered' things the user never said. "
            "Groundedness checks on every response are mandatory.",
        ],
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
            "Runs as a PWA: works offline, installable on Android."
        ),
        "key_lessons": [
            "Nepali business requirements differ from Indian or Western defaults. "
            "Fiscal year follows Bikram Sambat calendar. Reports use Nepali "
            "number formatting. Build for the actual user, not an imaginary one.",
            "PWA offline-first is hard. IndexedDB sync with Supabase "
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
    },
    {
        "name": "StackSift",
        "tagline": "B2B product intelligence API",
        "url": "stacksift.in",
        "stack": [
            "FastAPI",
            "SQLite on Render disk",
            "OpenAI gpt-4.1 / mini",
            "Serper web search",
            "LangSmith tracing",
            "Docker on Render",
        ],
        "what_it_does": (
            "Given a company's domain, returns the software products that company "
            "actually sells, each with evidence URLs, as JSON. Five stages per "
            "domain: search, crawl, candidate extraction, de-duplication, and a "
            "final verdict pass. Ships as a REST API, an MCP server, and a live demo."
        ),
        "key_lessons": [
            "Treat the two kinds of error differently. A feature recorded as a "
            "product corrupts a customer's database; a missed product only lands "
            "in a review queue. Low-confidence results go to review.csv.",
            "Build the labelled evaluation set before tuning anything: 21 verified "
            "domains, macro F1 0.91 to 0.93, run-to-run noise about +/-0.03.",
            "Report cost per domain in every result (a few cents each), so cost "
            "stays an engineering number instead of a surprise.",
        ],
        "what_broke": [
            "An automatic prompt optimiser (DSPy MIPROv2) lowered macro F1 from "
            "0.919 to 0.859. The compiled prompt was reverted.",
            "Login failed in production behind Render's proxy: the app could not "
            "tell requests arrived over HTTPS, so secure cookies broke. Fixed by "
            "reading X-Forwarded-Proto and logging around sessions.",
            "A billing change wrote placeholder payment data onto existing accounts "
            "and had to be fixed.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Lessons matrix
# ---------------------------------------------------------------------------

# The six patterns from the chapter's "Patterns that transfer" section.
PATTERNS: tuple[str, ...] = (
    "The hard part is rarely the AI",
    "Observability before features",
    "Hardening is a phase",
    "Costs are easier to ignore than fix",
    "Build for the actual user",
    "Measure before you trust a change",
)

CENTRAL, SHOWN, ABSENT = 2, 1, 0

# How strongly each case study in the chapter demonstrates each pattern:
# CENTRAL = one of that system's main lessons, SHOWN = the case study shows
# it, ABSENT = the case study does not bear on it. Order follows PATTERNS.
LESSON_MATRIX: dict[str, tuple[int, ...]] = {
    "Reflecta": (CENTRAL, CENTRAL, SHOWN, CENTRAL, SHOWN, SHOWN),
    "Godam": (SHOWN, CENTRAL, SHOWN, ABSENT, CENTRAL, ABSENT),
    "RAGNav": (SHOWN, SHOWN, SHOWN, CENTRAL, ABSENT, CENTRAL),
    "StackSift": (SHOWN, SHOWN, SHOWN, SHOWN, SHOWN, CENTRAL),
}


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

LAYER_COLORS = {
    "Frontend": "#0969da",
    "Backend": "#1a7f37",
    "Data": "#bc4c00",
    "AI/ML": "#8250df",
    "Infrastructure": "#57606a",
}

LAYER_OF = {
    "Next.js 15": "Frontend",
    "Vercel": "Infrastructure",
    "Neon Postgres + pgvector": "Data",
    "Supabase (PostgreSQL + JWT)": "Data",
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
    "FastAPI": "Backend",
    "SQLite on Render disk": "Data",
    "OpenAI gpt-4.1 / mini": "AI/ML",
    "Serper web search": "Data",
    "LangSmith tracing": "Infrastructure",
    "Docker on Render": "Infrastructure",
}


def plot_system_architectures(cfg: Config) -> Path:
    """The four stacks side by side, coloured by layer, drawn at print size."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CASE_STUDIES), figsize=(6.6, 3.9))
    slots = max(len(study["stack"]) for study in CASE_STUDIES)
    step = 0.78 / slots

    for ax, study in zip(axes, CASE_STUDIES):
        for k, tech in enumerate(study["stack"]):
            y = 0.86 - k * step
            color = LAYER_COLORS[LAYER_OF.get(tech, "Backend")]
            ax.barh(y, 0.96, left=0.02, height=step * 0.86, color=color, edgecolor="white")
            ax.text(
                0.5,
                y,
                textwrap.fill(tech, 18, break_long_words=False),
                ha="center",
                va="center",
                fontsize=6.2,
                color="white",
                fontweight="bold",
                linespacing=1.1,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            0.5,
            1.07,
            study["name"],
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.06,
            textwrap.fill(study["tagline"], 20),
            ha="center",
            va="top",
            fontsize=6.4,
            color="#57606a",
            transform=ax.transAxes,
            linespacing=1.15,
        )
        ax.text(
            0.5,
            0.11 - step / 2,
            textwrap.fill(study["url"], 22),
            ha="center",
            va="top",
            fontsize=6,
            color="#57606a",
        )

    handles = [mpatches.Patch(color=c, label=k) for k, c in LAYER_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=6.8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Four case studies: stacks by layer", fontsize=9.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.07, wspace=0.08)
    out = cfg.figures_dir / "ch23_system_architectures.png"
    fig.savefig(out, dpi=SAVE_DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


def plot_lessons_matrix(cfg: Config) -> Path:
    """Which of the six patterns each case study demonstrates."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    systems = list(LESSON_MATRIX)
    fill = {CENTRAL: "#1a7f37", SHOWN: "#aceebb", ABSENT: "#f6f8fa"}
    label = {CENTRAL: "central", SHOWN: "shown", ABSENT: ""}

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for col, system in enumerate(systems):
        for row, level in enumerate(LESSON_MATRIX[system]):
            ax.add_patch(mpatches.Rectangle((col, row), 0.96, 0.92, color=fill[level]))
            ax.text(
                col + 0.48,
                row + 0.46,
                label[level],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if level == CENTRAL else "#1f2328",
            )
    ax.set_xlim(0, len(systems))
    ax.set_ylim(len(PATTERNS), 0)
    ax.set_xticks([c + 0.48 for c in range(len(systems))])
    ax.set_xticklabels(systems, fontsize=8, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks([r + 0.46 for r in range(len(PATTERNS))])
    ax.set_yticklabels([f"{i}. {p}" for i, p in enumerate(PATTERNS, 1)], fontsize=7.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    fig.suptitle("Which patterns each case study shows", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    out = cfg.figures_dir / "ch23_lessons_matrix.png"
    fig.savefig(out, dpi=SAVE_DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_case_studies_summary(cfg: Config) -> Path:
    """Write the Markdown case studies document."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-World Case Studies: Four Systems",
        "",
        "Four systems I built and shipped. What worked, what broke, and what you can borrow.\n",
    ]
    for study in CASE_STUDIES:
        lines += [
            f"## {study['name']}: {study['tagline']}",
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
        "These patterns showed up across the projects:\n",
        "**1. Ship monitoring before users.** Most incidents above were caught "
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
    logger.info("  Reflecta | Godam | RAGNav | StackSift")
    logger.info("=" * 60)

    for study in CASE_STUDIES:
        logger.info(f"\n  {study['name']}: {study['tagline']}")
        logger.info(f"  Stack: {', '.join(study['stack'][:3])}...")
        logger.info(
            f"  Lessons: {len(study['key_lessons'])} | "
            f"Incidents described: {len(study['what_broke'])}"
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
    logger.info("Next: Chapter 24, The India Playbook")


if __name__ == "__main__":
    main()
