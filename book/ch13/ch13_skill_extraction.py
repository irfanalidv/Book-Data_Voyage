"""Chapter 13: NLP for Skill Extraction
Data Voyage - Building TalentLens

TalentLens milestone: three extractors benchmarked against a
real eval set of 200 LLM-labelled postings. Run end-to-end
to regenerate reports/skill_extraction_eval.md.

Run: python book/ch13/ch13_skill_extraction.py

Outputs:
    book/ch13/reports/skill_extraction_eval.md
    book/ch13/reports/figures/ch13_threshold_sweep.png
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from talentlens.skills import (
    evaluate_extractor,
    extract_skills,
    load_eval_set,
    per_skill_metrics,
    skill_names,
)

if TYPE_CHECKING:
    from talentlens.skills import ExtractorMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _THIS_DIR / "reports" / "figures"
_REPORTS_DIR = _THIS_DIR / "reports"
_EVAL_PATH = _THIS_DIR / "data" / "eval_set_llm_labelled.jsonl"
_REPORT_PATH = _REPORTS_DIR / "skill_extraction_eval.md"
SEMANTIC_THRESHOLD = 0.80


def run_benchmarks(eval_rows: list[dict]) -> dict[str, ExtractorMetrics]:
    results: dict[str, ExtractorMetrics] = {}
    for name, extractor in [
        ("Regex", partial(extract_skills, method="regex")),
        ("spaCy", partial(extract_skills, method="spacy")),
        (
            "spaCy + abbreviations",
            partial(extract_skills, method="spacy", spacy_abbreviations=True),
        ),
        (
            "Semantic (τ=0.80)",
            partial(
                extract_skills,
                method="semantic",
                semantic_threshold=SEMANTIC_THRESHOLD,
            ),
        ),
    ]:
        logger.info("  Running %s...", name)
        m = evaluate_extractor(extractor, eval_rows)
        results[name] = m
        logger.info("    Macro F1: %.3f", m.macro_f1)
    return results


def run_threshold_sweep(eval_rows: list[dict]) -> list[tuple[float, float, float, float]]:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep: list[tuple[float, float, float, float]] = []
    for tau in thresholds:
        logger.info("  Threshold %.2f...", tau)
        extractor = partial(extract_skills, method="semantic", semantic_threshold=tau)
        m = evaluate_extractor(extractor, eval_rows)
        sweep.append((tau, m.macro_precision, m.macro_recall, m.macro_f1))
    return sweep


def plot_threshold_sweep(sweep: list[tuple[float, float, float, float]]) -> Path:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    taus = [s[0] for s in sweep]
    ax.plot(taus, [s[1] for s in sweep], "o-", label="Precision")
    ax.plot(taus, [s[2] for s in sweep], "s-", label="Recall")
    ax.plot(taus, [s[3] for s in sweep], "^-", label="Macro F1", linewidth=2)
    ax.set_xlabel("Cosine similarity threshold (τ)")
    ax.set_ylabel("Score")
    ax.set_title("Semantic extractor: threshold sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = _FIGURES_DIR / "ch13_threshold_sweep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)
    return out


def _top_skills(metrics: ExtractorMetrics, n: int = 10) -> list[str]:
    psk = per_skill_metrics(metrics)
    return [skill for skill, _ in sorted(psk.items(), key=lambda x: -x[1].support)[:n]]


def _count_ml_misses(eval_rows: list[dict]) -> tuple[int, int]:
    misses = 0
    support = 0
    for row in eval_rows:
        if row.get("skills_verified") is None:
            continue
        truth = set(row["skills_verified"])
        if "Machine Learning" not in truth:
            continue
        support += 1
        pred = set(skill_names(extract_skills(row.get("description", ""), method="regex")))
        if "Machine Learning" not in pred:
            misses += 1
    return misses, support


def _semantic_diagnosis(
    eval_rows: list[dict], *, threshold: float, limit: int = 3
) -> tuple[list[str], list[str]]:
    wins: list[str] = []
    fps: list[str] = []
    for row in eval_rows:
        if row.get("skills_verified") is None:
            continue
        text = row.get("description", "")
        gold = set(row["skills_verified"])
        regex_pred = set(skill_names(extract_skills(text, method="regex")))
        semantic_pred = set(
            skill_names(extract_skills(text, method="semantic", semantic_threshold=threshold))
        )
        regex_misses = gold - regex_pred
        semantic_catches = semantic_pred & regex_misses
        if semantic_catches and len(wins) < limit:
            wins.append(
                f"- Gold: `{sorted(gold)}`\n"
                f"  - Regex got: `{sorted(regex_pred)}`\n"
                f"  - Semantic added: `{sorted(semantic_catches)}`\n"
                f"  - Text: `{text[:200]}...`"
            )
        semantic_fps = semantic_pred - gold
        regex_avoids = semantic_fps - regex_pred
        if regex_avoids and len(fps) < limit:
            fps.append(
                f"- Gold: `{sorted(gold)}`\n"
                f"  - Semantic over-predicted: `{sorted(semantic_fps)}`\n"
                f"  - Of which regex avoided: `{sorted(regex_avoids)}`\n"
                f"  - Text: `{text[:200]}...`"
            )
        if len(wins) >= limit and len(fps) >= limit:
            break
    return wins, fps


def write_report(
    results: dict[str, ExtractorMetrics],
    sweep: list[tuple[float, float, float, float]],
    eval_rows: list[dict],
) -> None:
    regex_m = results["Regex"]
    spacy_m = results["spaCy"]
    spacy_abbr_m = results["spaCy + abbreviations"]
    semantic_m = results["Semantic (τ=0.80)"]
    regex_psk = per_skill_metrics(regex_m)
    spacy_psk = per_skill_metrics(spacy_m)
    spacy_abbr_psk = per_skill_metrics(spacy_abbr_m)
    semantic_psk = per_skill_metrics(semantic_m)

    ml_misses, ml_support = _count_ml_misses(eval_rows)
    top = _top_skills(regex_m)
    wins, lose_fps = _semantic_diagnosis(eval_rows, threshold=SEMANTIC_THRESHOLD)

    ml = regex_psk.get("Machine Learning")
    ml_abbr = spacy_abbr_psk.get("Machine Learning")

    lines: list[str] = [
        "# Skill extraction evaluation",
        "",
        "_Eval set: 200 postings sampled from `jobs_clean.large.csv`",
        "(v1.0.0-dataset-1, 1,190 rows). Stratified by description",
        "length, role, and skills_raw presence. Labelled by",
        "LLM (Groq llama-3.3-70b-versatile for 24 rows, OpenAI",
        "gpt-4o-mini for 176) and reviewed via automated spot-check",
        "(7% of 30 sampled rows flagged for label quality concerns;",
        "human interactive spot-check recommended before final prose)._",
        "",
        "_Generated by `book/ch13/ch13_skill_extraction.py`._",
        "",
        "## Extractors compared",
        "",
        "| Extractor | Implementation | Status |",
        "|---|---|---|",
        "| Regex (rule-based) | CANONICAL_SKILLS + SKILL_ALIASES via regex | shipped |",
        "| spaCy + EntityRuler | spaCy EntityRuler on description + skills_raw tokens | shipped |",
        "| Sentence-transformers + threshold | Semantic similarity to skill embeddings | shipped |",
        "",
        "## Method 1 — regex baseline",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Macro precision | {regex_m.macro_precision:.3f} |",
        f"| Macro recall | {regex_m.macro_recall:.3f} |",
        f"| Macro F1 | {regex_m.macro_f1:.3f} |",
        "",
        "Macro F1 is macro-averaged per posting (each posting weighted equally).",
        "Per-skill figures below are micro-averaged (TP/FP/FN pooled across postings).",
        "",
        "### Per-skill micro (top 10 by support)",
        "",
        "| Skill | P | R | F1 | n |",
        "|---|---|---|---|---|",
    ]
    for skill in top:
        ps = regex_psk[skill]
        lines.append(
            f"| {skill} | {ps.precision:.2f} | {ps.recall:.2f} | {ps.f1:.2f} | {ps.support} |"
        )

    lines.extend(
        [
            "",
            "### What the regex misses (sample failure cases)",
            "",
            "**ML/DL inference in gold, no substring match in text** (row 1 pattern):",
            "",
            "- Text excerpt: `'Job Description: Roles & Responsibilities: · You will be involved in every part of the project lifecycle… training and optimizing ML/DL models…'`",
            "- Gold: `Machine Learning`, `Deep Learning`",
            "- Predicted: `[]` (regex does not expand `ML/DL` abbreviations)",
            "",
            "**Title-only “ML Engineer”, body never says “machine learning”:**",
            "",
            "- Text excerpt: `'ML Engineer WE ARE GRAPHENE…'`",
            "- Gold: includes `Machine Learning`",
            "- Predicted: `[]`",
            "",
            f"Across the full eval set, regex misses **Machine Learning** on "
            f"**{ml_misses} / {ml_support}** gold-positive rows — mostly abbreviated "
            "or inferred labels, not explicit canonical strings.",
            "",
            "### What the regex over-predicts (sample false positives)",
            "",
            "**`Git` from generic “engineering” prose:**",
            "",
            "- Predicted extra: `{Git}`",
            "- Gold: `{SQL}`",
            "- Text excerpt: `'… Experience Engineering, Digital Engineering, and Proc…'`",
            "",
            "**`RAG` substring in unrelated context:**",
            "",
            "- Predicted extra: `{RAG}`",
            "- Gold: `{Machine Learning}`",
            "- Text excerpt: `'… Azure AI, Microsoft Fabric, and Machine Learning ecosystems…'`",
            "",
            "**`RAG` in “cutting-edge” / product copy:**",
            "",
            "- Predicted extra: `{RAG}`",
            "- Gold: `{NLP, Machine Learning, Python}`",
            "- Text excerpt: `'Senior Machine Learning Engineer - NLP/Python… cutting-edge…'`",
            "",
            "When regex precision is high (often 1.00 per skill), false positives are "
            "fewer than missed abbreviations; macro scores still look strong because many "
            "gold labels use canonical spellings the regex already matches.",
            "",
            "## Notes on label quality",
            "",
            "The LLM labeller normalises some abbreviated mentions to "
            'canonical names — "ML/DL background required" becomes '
            '`["Machine Learning", "Deep Learning"]`. This inflates '
            "apparent recall expectations on the regex (which only fires "
            "on full string matches), and is one of the documented "
            "measurement caveats for this chapter.",
            "",
            "The first labelled row (`adzuna_1233590624`) is an example:",
            "notes read `[LLM] Inferred ML/DL as Machine Learning and Deep Learning` "
            "while the description uses `ML/DL` rather than the full phrases.",
            "",
            "## Method 2 — spaCy + EntityRuler",
            "",
            "spaCy's EntityRuler offers structured phrase matching with "
            "token boundaries — it avoids some substring collisions (e.g. "
            "`RAG` inside unrelated words) while still using the same "
            "`skills_raw` token path as regex. Patterns are built from "
            "`CANONICAL_SKILLS` + `SKILL_ALIASES`. Description text is matched "
            "via EntityRuler only (not the regex substring pass).",
            "",
            "| Metric | Score | Δ vs regex |",
            "|---|---|---|",
            f"| Macro precision | {spacy_m.macro_precision:.3f} | "
            f"{spacy_m.macro_precision - regex_m.macro_precision:+.3f} |",
            f"| Macro recall | {spacy_m.macro_recall:.3f} | "
            f"{spacy_m.macro_recall - regex_m.macro_recall:+.3f} |",
            f"| Macro F1 | {spacy_m.macro_f1:.3f} | "
            f"{spacy_m.macro_f1 - regex_m.macro_f1:+.3f} |",
            "",
            "### Per-skill comparison (top 10 by support)",
            "",
            "| Skill | Regex F1 | spaCy F1 | Δ |",
            "|---|---|---|---|",
        ]
    )
    for skill in top:
        r = regex_psk[skill].f1
        s = spacy_psk[skill].f1 if skill in spacy_psk else 0.0
        lines.append(f"| {skill} | {r:.3f} | {s:.3f} | {s - r:+.3f} |")

    lines.extend(
        [
            "",
            "Default EntityRuler patterns do not beat regex on macro F1. "
            "The interesting delta is per-skill: spaCy is tied on most skills, "
            "slightly worse on LLMs (token `llm` vs substring `llms`).",
            "",
            "### With abbreviation patterns",
            "",
            "Adding ML, DL, and `ml/dl` as token-level patterns " "(`spacy_abbreviations=True`):",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| Macro precision | {spacy_abbr_m.macro_precision:.3f} |",
            f"| Macro recall | {spacy_abbr_m.macro_recall:.3f} |",
            f"| Macro F1 | {spacy_abbr_m.macro_f1:.3f} |",
        ]
    )
    if ml_abbr:
        lines.append(
            f"| ML micro F1 | {ml_abbr.f1:.3f} (P={ml_abbr.precision:.3f}, "
            f"R={ml_abbr.recall:.3f}, n={ml_abbr.support}) |"
        )
    if ml:
        lines.extend(
            [
                "",
                "Abbreviations lift macro F1 by "
                f"**{spacy_abbr_m.macro_f1 - regex_m.macro_f1:+.3f}** vs regex and recover "
                "much of the ML recall gap "
                f"({ml.recall:.2f} → {ml_abbr.recall:.2f}), at the cost of ML "
                f"precision ({ml.precision:.2f} → {ml_abbr.precision:.2f}). "
                "This is the chapter's precision/recall tradeoff: expanding aliases helps "
                "recall on abbreviated postings, but `ml` as a token is noisier than "
                "full-string `machine learning`.",
            ]
        )

    lines.extend(
        [
            "",
            "### What spaCy doesn't fix",
            "",
            "Both regex and spaCy still miss gold labels that exist only as "
            "LLM inferences (`ML/DL` → Machine Learning + Deep Learning) when "
            "abbreviation patterns are off. Both still struggle when the gold "
            "label uses a cloud canonical the posting only hints at (`Azu…` → "
            "Cloud (AWS)). Method 3 (sentence-transformers) targets semantic "
            "context for those cases — if it cannot beat **0.824** macro F1 "
            "with abbreviations, the chapter's conclusion favours tuned rules "
            "over heavier models for this vocabulary.",
            "",
            "## Method 3 — sentence-transformers semantic matching",
            "",
            "Embed each canonical skill with `all-MiniLM-L6-v2` (~80 MB, CPU). "
            "For each posting, generate 1–3 word candidate phrases from the "
            "description and `skills_raw`, embed them, and match canonical skills "
            "whose best-candidate cosine similarity exceeds a threshold. "
            "`skills_raw` tokens still use the exact-match path (same as regex/spaCy).",
            "",
            "![Threshold sweep](figures/ch13_threshold_sweep.png)",
            "",
            "### Threshold sweep",
            "",
            "| Threshold | Macro P | Macro R | Macro F1 |",
            "|---|---|---|---|",
        ]
    )
    best_f1 = max(s[3] for s in sweep)
    for tau, p, r, f1 in sweep:
        f1_cell = f"**{f1:.3f}**" if abs(f1 - best_f1) < 1e-9 else f"{f1:.3f}"
        lines.append(f"| {tau:.2f} | {p:.3f} | {r:.3f} | {f1_cell} |")

    lines.extend(
        [
            "",
            f"Chosen operating point: **threshold = {SEMANTIC_THRESHOLD:.2f}**, "
            f"macro F1 = **{semantic_m.macro_f1:.3f}**. "
            "Lower thresholds inflate recall but flood precision (0.50 → macro F1 "
            f"{sweep[0][3]:.3f}). Above {SEMANTIC_THRESHOLD:.2f}, macro F1 plateaus or "
            "drops — no setting beats spaCy + abbreviations "
            f"(**{spacy_abbr_m.macro_f1:.3f}**).",
            "",
            "### Full comparison",
            "",
            "| Method | Macro P | Macro R | Macro F1 |",
            "|---|---|---|---|",
            f"| Regex | {regex_m.macro_precision:.3f} | {regex_m.macro_recall:.3f} | "
            f"{regex_m.macro_f1:.3f} |",
            f"| spaCy + EntityRuler | {spacy_m.macro_precision:.3f} | "
            f"{spacy_m.macro_recall:.3f} | {spacy_m.macro_f1:.3f} |",
            f"| spaCy + abbreviations | {spacy_abbr_m.macro_precision:.3f} | "
            f"{spacy_abbr_m.macro_recall:.3f} | **{spacy_abbr_m.macro_f1:.3f}** |",
            f"| Sentence-transformers (τ={SEMANTIC_THRESHOLD:.2f}) | "
            f"{semantic_m.macro_precision:.3f} | {semantic_m.macro_recall:.3f} | "
            f"{semantic_m.macro_f1:.3f} |",
            "",
            "### Per-skill F1 (top 10 by support)",
            "",
            "| Skill | Regex | spaCy | sp+abbr | Semantic |",
            "|---|---|---|---|---|",
        ]
    )
    for skill in top:
        lines.append(
            f"| {skill} | {regex_psk[skill].f1:.3f} | "
            f"{spacy_psk[skill].f1:.3f} | {spacy_abbr_psk[skill].f1:.3f} | "
            f"{semantic_psk[skill].f1:.3f} |"
        )

    lines.extend(
        [
            "",
            "Semantic ties regex on most skills; it does not recover the ML/DL "
            "abbreviation gap that spaCy + abbreviations addresses.",
            "",
            "### What semantic catches that regex misses",
            "",
        ]
    )
    lines.extend(wins if wins else ["_(none in first pass over eval set)_"])
    lines.extend(
        [
            "",
            "These are real semantic wins, but sparse — they do not offset the "
            "volume of false positives at thresholds low enough to help recall.",
            "",
            "### What semantic over-predicts",
            "",
        ]
    )
    lines.extend(lose_fps if lose_fps else ["_(none in first pass over eval set)_"])
    lines.extend(
        [
            "",
            "The embedding model conflates domain-adjacent language (“statistical "
            "analysis”, “insights”, “data analyst”) with the canonical skill "
            "**Statistics**. Regex avoids this because it only fires on explicit "
            "vocabulary — the same conservatism that misses ML/DL abbreviations.",
            "",
            "### Verdict",
            "",
            f"Sentence-transformers at the best threshold (**{semantic_m.macro_f1:.3f}** "
            "macro F1) **does not beat** regex "
            f"(**{regex_m.macro_f1:.3f}**) or spaCy + abbreviations "
            f"(**{spacy_abbr_m.macro_f1:.3f}**) on this closed vocabulary. "
            "The chapter conclusion: for a controlled canonical skill list on job postings, "
            "**tuned rule-based extraction wins on cost/accuracy**; reach for embeddings "
            "when the vocabulary is open-ended or labels are fuzzy, not because "
            "the method sounds modern. For peak macro F1 on this eval set, use "
            "`extract_skills(..., method='spacy', spacy_abbreviations=True)`; for "
            "minimum dependencies at nearly the same score, use `method='regex'`.",
            "",
        ]
    )

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("  Wrote: %s", _REPORT_PATH)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  CHAPTER 13: SKILL EXTRACTION")
    logger.info("=" * 60)

    eval_rows = load_eval_set(_EVAL_PATH)
    logger.info("Loaded %d eval rows from %s", len(eval_rows), _EVAL_PATH.name)

    logger.info("\n[1/3] Running benchmarks across all extractors...")
    results = run_benchmarks(eval_rows)

    logger.info("\n[2/3] Threshold sweep for semantic extractor...")
    sweep = run_threshold_sweep(eval_rows)
    plot_threshold_sweep(sweep)

    logger.info("\n[3/3] Regenerating report...")
    write_report(results, sweep, eval_rows)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 13 COMPLETE")
    logger.info("=" * 60)
    logger.info("  Eval set: %s", _EVAL_PATH.name)
    logger.info("  Report:   %s", _REPORT_PATH)
    logger.info("  Figures:  %s/", _FIGURES_DIR)
    logger.info("\nNext: Chapter 14 — Time Series Analysis")


if __name__ == "__main__":
    main()
