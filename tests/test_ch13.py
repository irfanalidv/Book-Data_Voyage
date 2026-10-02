"""Tests for Chapter 13: NLP for Skill Extraction.

Reference: book/ch13/README.md
Source: book/ch13/ch13_skill_extraction.py, talentlens/skills.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pytest

from talentlens.skills import evaluate_extractor, extract_skills, skill_names

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch13"))

from ch13_skill_extraction import run_threshold_sweep  # noqa: E402

_KNOWN_DESC = "We need strong Python and SQL skills. Experience with PyTorch and AWS is a plus."


def _semantic_stack_usable() -> bool:
    try:
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        import torch
        from sentence_transformers import SentenceTransformer  # noqa: F401

        _ = torch.tensor([1.0])
        return True
    except Exception:
        return False


_SEMANTIC_SKIP = pytest.mark.skipif(
    not _semantic_stack_usable(),
    reason=(
        "sentence-transformers/torch unavailable or broken in this environment "
        "(e.g. macOS arm64 torchvision::nms mismatch; see SCOPE.md)."
    ),
)


@pytest.fixture
def eval_rows() -> list[dict]:
    path = _REPO_ROOT / "book" / "ch13" / "data" / "eval_set_llm_labelled.jsonl"
    if not path.exists():
        pytest.skip(f"eval set not found: {path}")
    import json

    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    labelled = [r for r in rows if r.get("skills_verified") is not None]
    if len(labelled) < 10:
        pytest.skip("eval set has too few labelled rows")
    return labelled[:40]


class TestRegexExtractor:
    def test_python_and_sql_in_canonical_set(self):
        names = set(skill_names(extract_skills(_KNOWN_DESC, method="regex")))
        assert {"Python", "SQL"}.issubset(names)


class TestSpacyExtractor:
    def test_python_and_sql_in_canonical_set(self):
        names = set(skill_names(extract_skills(_KNOWN_DESC, method="spacy")))
        assert {"Python", "SQL"}.issubset(names)


class TestSemanticExtractor:
    @_SEMANTIC_SKIP
    def test_python_and_sql_above_threshold(self):
        names = set(
            skill_names(
                extract_skills(
                    _KNOWN_DESC,
                    method="semantic",
                    semantic_threshold=0.65,
                )
            )
        )
        assert "Python" in names
        assert "SQL" in names


@pytest.mark.skipif(
    not _semantic_stack_usable(),
    reason="threshold sweep uses semantic extractor (torch/sentence-transformers)",
)
class TestThresholdSweep:
    def test_sweep_rows_have_four_numeric_fields(self, eval_rows):
        sweep = run_threshold_sweep(eval_rows)
        assert len(sweep) >= 5
        for tau, prec, rec, f1 in sweep:
            assert 0.0 <= tau <= 1.0
            assert 0.0 <= prec <= 1.0
            assert 0.0 <= rec <= 1.0
            assert 0.0 <= f1 <= 1.0

    def test_higher_threshold_raises_precision_on_eval_subset(self, eval_rows):
        low = evaluate_extractor(
            partial(extract_skills, method="semantic", semantic_threshold=0.55),
            eval_rows,
        )
        high = evaluate_extractor(
            partial(extract_skills, method="semantic", semantic_threshold=0.85),
            eval_rows,
        )
        assert high.macro_precision >= low.macro_precision - 0.05


@pytest.mark.skip(
    reason="BERT NER and LLM-direct extractors are deferred in v1.0 (see ch13 What's deferred)."
)
def test_bert_and_llm_direct_not_implemented():
    pass
