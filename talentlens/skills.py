"""Skill vocabulary and extractor framework for TalentLens.

This module owns the canonical skill vocabulary used across the
book and exposes three pluggable skill extractors that Chapter 13
builds and benchmarks. Chapter 6 imports the vocabulary from here;
Chapter 6's extract_skills function optionally consumes
extract_skills() from this module with a regex fallback.

Public API (stable; consumed by ch06, ch13, ch16, future ch22):

    CANONICAL_SKILLS: tuple[str, ...]
    SKILL_ALIASES: dict[str, str]
    ExtractedSkill: dataclass
    extract_skills(text, *, skills_raw=None, method='default') -> list[ExtractedSkill]
    skill_names(extracted) -> list[str]
    evaluate_extractor(...) -> ExtractorMetrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# CANONICAL_SKILLS and SKILL_ALIASES were originally defined in
# book/ch06/ch06_data_cleaning_preprocessing.py. Relocating here
# so ch06 and ch13 (and future chapters) have one source of truth.
# Values are copied verbatim from the original ch06 definitions.
CANONICAL_SKILLS: tuple[str, ...] = (
    "Python",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "Machine Learning",
    "NLP",
    "LLMs",
    "RAG",
    "FastAPI",
    "PostgreSQL",
    "Docker",
    "Kubernetes",
    "Spark",
    "SQL",
    "Airflow",
    "dbt",
    "MLflow",
    "Cloud (AWS)",
    "Cloud (Azure)",
    "Cloud (GCP)",
    "Deep Learning",
    "Statistics",
    "pandas",
    "NumPy",
    "Git",
)

SKILL_ALIASES: dict[str, str] = {
    "python3": "Python",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language model": "LLMs",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "fastapi": "FastAPI",
    "postgres": "PostgreSQL",
    "k8s": "Kubernetes",
    "aws": "Cloud (AWS)",
    "amazon web services": "Cloud (AWS)",
    "gcp": "Cloud (GCP)",
    "google cloud": "Cloud (GCP)",
    "google cloud platform": "Cloud (GCP)",
    "azure": "Cloud (Azure)",
    "microsoft azure": "Cloud (Azure)",
}


@dataclass(frozen=True)
class ExtractedSkill:
    """A single extracted skill with optional metadata.

    Attributes:
        name: Canonical skill name from CANONICAL_SKILLS.
        confidence: Extractor confidence in [0.0, 1.0]. Regex and
            spaCy default to 1.0 (binary match). Sentence-transformer
            matching uses cosine similarity, typically in [0.3, 1.0].
        source_span: Character offsets (start, end) in the source
            text where the skill was found. None when the extractor
            doesn't operate on spans (sentence-transformer matching).
    """

    name: str
    confidence: float = 1.0
    source_span: tuple[int, int] | None = None


ExtractorMethod = Literal[
    "regex", "spacy", "semantic", "sentence_transformer", "default"
]


def extract_skills(
    text: str,
    *,
    skills_raw: str | None = None,
    method: ExtractorMethod = "default",
    spacy_abbreviations: bool = False,
    semantic_threshold: float = 0.6,
) -> list[ExtractedSkill]:
    """Extract canonical skills from a posting's text fields.

    The function takes both ``text`` (typically a free-text
    description) and an optional ``skills_raw`` (a comma- or
    pipe-separated structured field, as produced by Chapter 5).
    Each field is processed differently:

    - ``text`` is scanned for canonical skill names as substrings
      (case-insensitive). 'Python' in 'Pythonic' would match.
    - ``skills_raw`` is tokenised on [,|;\\s]+, each token lowered,
      then matched exactly against canonical names or aliases.
      'Pythonic' as a token would not match Python.

    The two-path treatment matches Chapter 6's incumbent extractor
    faithfully. Callers who only have one field (e.g. a live posting
    from a recruiter email) pass that field as ``text`` and leave
    ``skills_raw`` as None.

    See the Chapter 13 prose for which extractor to use when. The
    'default' method points at the chapter's recommended choice
    after measurement.

    Args:
        text: Free-text description. May be empty string.
        skills_raw: Optional structured skills field. May be None.
        method: Which extractor implementation to use.
        spacy_abbreviations: When method is ``spacy``, also match ML/DL/AI
            token patterns (may help recall, hurt precision).
        semantic_threshold: When method is ``semantic`` or
            ``sentence_transformer``, minimum cosine similarity between a
            candidate n-gram and a canonical skill embedding.

    Returns:
        List of ExtractedSkill, deduplicated by canonical name.
        Order is sorted alphabetically for deterministic output
        (matches Chapter 6's existing behaviour).
    """
    if method == "regex":
        return _extract_skills_regex(text, skills_raw)
    if method == "spacy":
        return _extract_skills_spacy(
            text, skills_raw, include_abbreviations=spacy_abbreviations
        )
    if method in ("semantic", "sentence_transformer"):
        return _extract_skills_semantic(
            text, skills_raw, threshold=semantic_threshold
        )
    if method == "default":
        raise NotImplementedError(
            "Default extractor is set in Chapter 13 iteration 6 "
            "based on measurement."
        )
    raise ValueError(f"Unknown method: {method!r}")


# Token separator pattern, ported verbatim from ch06's _skills.
# Splits on commas, pipes, semicolons, and any whitespace run.
_SKILLS_RAW_TOKEN_PATTERN = re.compile(r"[,|;\s]+")


def _normalize_skills_raw(skills_raw: str | float | None) -> str | None:
    """Coerce JSONL null/NaN/empty skills_raw to None for tokenisation."""
    if skills_raw is None:
        return None
    if isinstance(skills_raw, float):
        import math

        if math.isnan(skills_raw):
            return None
    s = str(skills_raw).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _extract_skills_regex(
    text: str,
    skills_raw: str | None,
) -> list[ExtractedSkill]:
    """Faithful port of Chapter 6's _skills inner function.

    Two paths, identical to ch06:
      1. Tokenise skills_raw; lower each token; exact-match against
         alias map first, then exact-match against canonical names.
      2. Lower the full description; substring-match each canonical
         name.

    Deduplicate via a set, sort alphabetically for deterministic
    output. confidence=1.0 for all (binary match). source_span=None
    for both paths to match ch06's no-span output; spans become
    meaningful in the spaCy and sentence-transformer paths.
    """
    found = _skills_from_raw(skills_raw)
    found |= _skills_from_description_regex(text)
    return [
        ExtractedSkill(name=name, confidence=1.0, source_span=None)
        for name in sorted(found)
    ]


def _skills_from_raw(skills_raw: str | None) -> set[str]:
    skills_raw = _normalize_skills_raw(skills_raw)
    alias_map = {a.lower(): c for a, c in SKILL_ALIASES.items()}
    found: set[str] = set()
    if not skills_raw:
        return found
    for token in _SKILLS_RAW_TOKEN_PATTERN.split(skills_raw):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok in alias_map:
            found.add(alias_map[tok])
        else:
            for canonical in CANONICAL_SKILLS:
                if tok == canonical.lower():
                    found.add(canonical)
    return found


def _skills_from_description_regex(text: str) -> set[str]:
    desc_lower = (text or "").lower()
    found: set[str] = set()
    for canonical in CANONICAL_SKILLS:
        if canonical.lower() in desc_lower:
            found.add(canonical)
    for alias, canonical in SKILL_ALIASES.items():
        if _alias_in_description(alias, desc_lower):
            found.add(canonical)
    return found


def _alias_in_description(alias: str, desc_lower: str) -> bool:
    """Word-boundary alias match in description (avoids 'python' in 'pythonic')."""
    if " " in alias:
        return alias in desc_lower
    return bool(re.search(r"\b" + re.escape(alias) + r"\b", desc_lower))


_spacy_cache: dict[bool, object] = {}


def _build_spacy_patterns(include_abbreviations: bool) -> list[dict]:
    patterns: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_pattern(pattern: str | list, canonical: str) -> None:
        key = (str(pattern), canonical)
        if key in seen:
            return
        seen.add(key)
        entry: dict = {"label": "SKILL", "pattern": pattern, "id": canonical}
        patterns.append(entry)

    for canonical in CANONICAL_SKILLS:
        add_pattern(canonical, canonical)
    for alias, canonical in SKILL_ALIASES.items():
        add_pattern(alias, canonical)

    if include_abbreviations:
        add_pattern([{"LOWER": "ml"}], "Machine Learning")
        add_pattern([{"LOWER": "dl"}], "Deep Learning")
        add_pattern([{"LOWER": "ml/dl"}], "Machine Learning")
        add_pattern([{"LOWER": "ml/dl"}], "Deep Learning")

    return patterns


def _get_spacy_nlp(include_abbreviations: bool = False):
    if include_abbreviations in _spacy_cache:
        return _spacy_cache[include_abbreviations]
    try:
        import spacy
    except ImportError as err:
        raise ImportError(
            "spaCy is required for method='spacy'. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        ) from err

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    ruler = nlp.add_pipe(
        "entity_ruler",
        config={"phrase_matcher_attr": "LOWER", "validate": True},
    )
    ruler.add_patterns(_build_spacy_patterns(include_abbreviations))
    _spacy_cache[include_abbreviations] = nlp
    return nlp


def _extract_skills_spacy(
    text: str,
    skills_raw: str | None,
    *,
    include_abbreviations: bool = False,
) -> list[ExtractedSkill]:
    """EntityRuler over canonical names + aliases (optional ML/DL tokens)."""
    nlp = _get_spacy_nlp(include_abbreviations)
    found: dict[str, tuple[int, int] | None] = {}

    doc = nlp(text or "")
    for ent in doc.ents:
        if ent.label_ != "SKILL":
            continue
        canonical = ent.ent_id_ or ent.text
        if canonical not in CANONICAL_SKILLS:
            continue
        span = (ent.start_char, ent.end_char)
        if canonical not in found:
            found[canonical] = span

    for name in _skills_from_raw(skills_raw):
        if name not in found:
            found[name] = None

    return [
        ExtractedSkill(name=name, confidence=1.0, source_span=span)
        for name, span in sorted(found.items())
    ]


_CANDIDATE_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9.+#\-]+")

_semantic_model = None
_semantic_skill_embeddings = None
_semantic_skill_names: list[str] | None = None


def _get_semantic_resources():
    global _semantic_model, _semantic_skill_embeddings, _semantic_skill_names
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for method='semantic'. "
                "Install with: pip install sentence-transformers"
            ) from err

        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        _semantic_skill_names = list(CANONICAL_SKILLS)
        _semantic_skill_embeddings = _semantic_model.encode(
            _semantic_skill_names,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    return _semantic_model, _semantic_skill_embeddings, _semantic_skill_names


def _generate_semantic_candidates(text: str) -> list[str]:
    """Extract 1–3 word phrases as embedding candidates."""
    tokens = [
        t.lower()
        for t in _CANDIDATE_TOKEN_PATTERN.findall(text or "")
        if 2 <= len(t) <= 30
    ]
    candidates: set[str] = set()
    for i, t in enumerate(tokens):
        candidates.add(t)
        if i + 1 < len(tokens):
            candidates.add(f"{t} {tokens[i + 1]}")
        if i + 2 < len(tokens):
            candidates.add(f"{t} {tokens[i + 1]} {tokens[i + 2]}")
    return sorted(candidates)


def _extract_skills_semantic(
    text: str,
    skills_raw: str | None,
    *,
    threshold: float = 0.6,
) -> list[ExtractedSkill]:
    """Match canonical skills via embedding similarity to text n-grams."""

    model, skill_embeddings, skill_names = _get_semantic_resources()
    skills_raw_norm = _normalize_skills_raw(skills_raw)
    raw_text = skills_raw_norm or ""
    full_text = f"{text or ''}\n{raw_text}".strip()

    found: dict[str, float] = {}
    for name in _skills_from_raw(skills_raw):
        found[name] = 1.0

    candidates = _generate_semantic_candidates(full_text)
    if candidates:
        candidate_embeddings = model.encode(
            candidates,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        similarities = candidate_embeddings @ skill_embeddings.T
        best_per_skill = similarities.max(axis=0)
        for i, score in enumerate(best_per_skill):
            if score >= threshold:
                name = skill_names[i]
                found[name] = max(found.get(name, 0.0), float(score))

    return [
        ExtractedSkill(name=name, confidence=conf, source_span=None)
        for name, conf in sorted(found.items())
    ]


def skill_names(extracted: list[ExtractedSkill]) -> list[str]:
    """Convenience: extract the canonical names only.

    ch06 uses this to recover a pipe-separated string when it
    doesn't care about confidence or spans.
    """
    return [s.name for s in extracted]


@dataclass(frozen=True)
class ExtractorMetrics:
    """Aggregate metrics for an extractor against an eval set.

    Attributes:
        precision: Macro-averaged across postings.
        recall: Macro-averaged across postings.
        per_skill: Mapping of skill name to (precision, recall, support).
        n_postings: Number of postings evaluated.
        n_predictions: Total predicted skill instances.
        n_truth: Total ground-truth skill instances.
    """

    precision: float
    recall: float
    per_skill: dict[str, tuple[float, float, int]]
    n_postings: int
    n_predictions: int
    n_truth: int

    @property
    def macro_precision(self) -> float:
        return self.precision

    @property
    def macro_recall(self) -> float:
        return self.recall

    @property
    def macro_f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass(frozen=True)
class PerSkillMetrics:
    """Micro-averaged precision/recall/F1 for one canonical skill."""

    precision: float
    recall: float
    support: int

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def per_skill_metrics(metrics: ExtractorMetrics) -> dict[str, PerSkillMetrics]:
    """Convert per_skill tuples to PerSkillMetrics for reporting."""
    return {
        skill: PerSkillMetrics(precision=p, recall=r, support=support)
        for skill, (p, r, support) in metrics.per_skill.items()
    }


def evaluate_extractor(
    extractor_fn,
    eval_set: list[dict],
) -> ExtractorMetrics:
    """Score an extractor against the hand-labelled eval set.

    For each posting in eval_set:
      1. Call extractor_fn(text, skills_raw=...) → list[ExtractedSkill]
      2. Compare the set of extracted skill names against the
         ground-truth set from skills_verified.
      3. Compute per-posting precision and recall:
          precision = |extracted ∩ truth| / |extracted|
          recall    = |extracted ∩ truth| / |truth|
         Edge case: if |extracted| == 0 and |truth| == 0, treat
         precision as 1.0 (vacuously correct) and recall as 1.0.
         If only one side is empty, the corresponding metric is 0.0.

    Macro-average precision and recall across postings - every
    posting weighted equally, regardless of how many skills it
    has. This matches the chapter's convention documented in
    book/ch13/reports/skill_extraction_eval.md.

    Per-skill metrics are micro-averaged: for each canonical skill
    that appears in any ground-truth label, compute how often the
    extractor predicted it correctly (TP), missed it (FN), and
    falsely predicted it (FP) across all postings. Returns
    (precision, recall, support) per skill.

    Postings with skills_verified == None (unlabelled) are
    SKIPPED with a warning, not counted toward metrics. The
    chapter's measurement requires labels.

    Args:
        extractor_fn: Callable taking (text, *, skills_raw=) →
            list[ExtractedSkill]. Pass a partial of extract_skills
            with the method bound:
                from functools import partial
                regex_ex = partial(extract_skills, method="regex")
                metrics = evaluate_extractor(regex_ex, eval_set)
        eval_set: List of dicts loaded from eval_set.jsonl. Each
            dict has at minimum 'description', 'skills_raw', and
            'skills_verified' (a list[str] or None).

    Returns:
        ExtractorMetrics with macro precision/recall, per-skill
        breakdown, and total counts.
    """
    import logging
    from collections import defaultdict

    _logger = logging.getLogger(__name__)

    labelled = [r for r in eval_set if r.get("skills_verified") is not None]
    n_skipped = len(eval_set) - len(labelled)
    if n_skipped:
        _logger.warning(
            f"evaluate_extractor: skipped {n_skipped} unlabelled rows "
            f"of {len(eval_set)} total"
        )
    if not labelled:
        raise ValueError(
            "No labelled rows in eval_set. Hand-label the postings "
            "in book/ch13/data/eval_set.jsonl per the annotation "
            "guide before running evaluation."
        )

    per_posting_precision: list[float] = []
    per_posting_recall: list[float] = []
    n_predictions = 0
    n_truth = 0

    per_skill_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for row in labelled:
        extracted = extractor_fn(
            row.get("description", ""),
            skills_raw=row.get("skills_raw", ""),
        )
        extracted_names = {s.name for s in extracted}
        truth_names = set(row["skills_verified"])

        n_predictions += len(extracted_names)
        n_truth += len(truth_names)

        tp_set = extracted_names & truth_names
        fp_set = extracted_names - truth_names
        fn_set = truth_names - extracted_names

        if not extracted_names and not truth_names:
            per_posting_precision.append(1.0)
            per_posting_recall.append(1.0)
        else:
            if extracted_names:
                per_posting_precision.append(len(tp_set) / len(extracted_names))
            else:
                per_posting_precision.append(0.0)
            if truth_names:
                per_posting_recall.append(len(tp_set) / len(truth_names))
            else:
                per_posting_recall.append(0.0)

        for skill in tp_set:
            per_skill_counts[skill]["tp"] += 1
        for skill in fp_set:
            per_skill_counts[skill]["fp"] += 1
        for skill in fn_set:
            per_skill_counts[skill]["fn"] += 1

    macro_precision = sum(per_posting_precision) / len(per_posting_precision)
    macro_recall = sum(per_posting_recall) / len(per_posting_recall)

    per_skill: dict[str, tuple[float, float, int]] = {}
    for skill, counts in per_skill_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        support = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_skill[skill] = (p, r, support)

    return ExtractorMetrics(
        precision=macro_precision,
        recall=macro_recall,
        per_skill=per_skill,
        n_postings=len(labelled),
        n_predictions=n_predictions,
        n_truth=n_truth,
    )


def load_eval_set(path) -> list[dict]:
    """Load the hand-labelled eval set from a JSONL file.

    Each line is a JSON object with at least 'description',
    'skills_raw', and 'skills_verified' (a list[str] or None).

    Args:
        path: Path or str pointing to eval_set.jsonl.

    Returns:
        List of dicts, one per row. Rows with skills_verified
        still None are included; evaluate_extractor handles
        skipping them.
    """
    import json
    from pathlib import Path

    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
