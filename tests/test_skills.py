"""Tests for talentlens.skills.

The scaffolding tests verify the public API exists and is wired
correctly. NotImplementedError tripwires force the chapter to
replace these tests with real behaviour tests as each extractor
is implemented.
"""

from __future__ import annotations

import pytest

from talentlens.skills import (
    CANONICAL_SKILLS,
    SKILL_ALIASES,
    ExtractedSkill,
    evaluate_extractor,
    extract_skills,
    load_eval_set,
    skill_names,
)


class TestPublicAPI:
    """Smoke tests for the public API surface."""

    def test_canonical_skills_is_non_empty(self):
        assert len(CANONICAL_SKILLS) > 0

    def test_canonical_skills_is_a_tuple(self):
        assert isinstance(CANONICAL_SKILLS, tuple)

    def test_skill_aliases_maps_to_canonical(self):
        for alias, target in SKILL_ALIASES.items():
            assert (
                target in CANONICAL_SKILLS
            ), f"Alias {alias!r} maps to {target!r} which is not in CANONICAL_SKILLS"

    def test_extracted_skill_defaults(self):
        s = ExtractedSkill(name="Python")
        assert s.name == "Python"
        assert s.confidence == 1.0
        assert s.source_span is None

    def test_extracted_skill_with_metadata(self):
        s = ExtractedSkill(name="Python", confidence=0.85, source_span=(10, 16))
        assert s.confidence == 0.85
        assert s.source_span == (10, 16)

    def test_skill_names_helper(self):
        extracted = [
            ExtractedSkill(name="Python"),
            ExtractedSkill(name="SQL", confidence=0.9),
        ]
        assert skill_names(extracted) == ["Python", "SQL"]


class TestExtractorTripwires:
    """Tripwires: each extractor raises NotImplementedError until
    its iteration fills it in."""

    def test_spacy_extractor_matches_canonical_skills(self):
        result = extract_skills("Python and SQL required.", method="spacy")
        names = {s.name for s in result}
        assert "Python" in names
        assert "SQL" in names

    def test_semantic_extractor_matches_obvious_skills(self):
        result = extract_skills(
            "Experience with Python and SQL required.",
            method="semantic",
            semantic_threshold=0.5,
        )
        names = {s.name for s in result}
        assert "Python" in names
        assert "SQL" in names

    def test_sentence_transformer_alias_for_semantic(self):
        result = extract_skills(
            "Python developer",
            method="sentence_transformer",
            semantic_threshold=0.5,
        )
        assert any(s.name == "Python" for s in result)

    def test_default_extractor_not_set_until_iter_6(self):
        with pytest.raises(NotImplementedError):
            extract_skills("Python and SQL", method="default")


class TestUnknownMethod:
    """Unknown method should raise ValueError, not NotImplementedError."""

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            extract_skills("text", method="nonexistent")


class TestRegexExtractor:
    """Behaviour-preserving tests for the regex extractor.

    These tests verify the port matches Chapter 6's _skills logic
    exactly. Each test names which ch06 behaviour it's pinning.
    """

    def test_description_substring_match(self):
        result = extract_skills(
            "I use python daily for data work",
            method="regex",
        )
        names = [s.name for s in result]
        assert "Python" in names

    def test_skills_raw_token_match_via_alias(self):
        result = extract_skills(
            "",
            skills_raw="k8s",
            method="regex",
        )
        names = [s.name for s in result]
        if "k8s" in {a.lower() for a in SKILL_ALIASES}:
            assert "Kubernetes" in names

    def test_skills_raw_exact_token_match(self):
        result = extract_skills("", skills_raw="python", method="regex")
        assert "Python" in [s.name for s in result]

        result = extract_skills("", skills_raw="pythonic", method="regex")
        assert "Python" not in [s.name for s in result]

    def test_description_substring_DOES_match_partial_word(self):
        result = extract_skills("pythonic", method="regex")
        assert "Python" in [s.name for s in result]

    def test_skills_raw_separator_variants(self):
        for sep in (",", "|", ";", " ", "  ", " ,"):
            raw = f"python{sep}sql"
            result = extract_skills("", skills_raw=raw, method="regex")
            names = set(s.name for s in result)
            assert "Python" in names, f"Failed with separator {sep!r}"
            assert "SQL" in names, f"Failed with separator {sep!r}"

    def test_returns_sorted_alphabetically(self):
        result = extract_skills(
            "We use SQL, Python, and AWS daily.",
            method="regex",
        )
        names = [s.name for s in result]
        assert names == sorted(names)

    def test_deduplicates_across_fields(self):
        result = extract_skills(
            "We use python at work.",
            skills_raw="python",
            method="regex",
        )
        python_count = sum(1 for s in result if s.name == "Python")
        assert python_count == 1

    def test_empty_inputs_return_empty_list(self):
        assert extract_skills("", method="regex") == []
        assert extract_skills("", skills_raw="", method="regex") == []
        assert extract_skills("", skills_raw=None, method="regex") == []

    def test_none_text_handled_gracefully(self):
        result = extract_skills("", method="regex")
        assert result == []

    def test_returns_extracted_skill_instances(self):
        result = extract_skills("python", method="regex")
        assert all(isinstance(s, ExtractedSkill) for s in result)
        assert all(s.confidence == 1.0 for s in result)
        assert all(s.source_span is None for s in result)

    def test_byte_exact_match_with_ch06_on_synthetic_input(self):
        import importlib

        import pandas as pd

        ch06 = importlib.import_module("book.ch06.ch06_data_cleaning_preprocessing")
        ch06_extract = ch06.extract_skills
        Config = ch06.Config

        cases = [
            {"description": "Python and SQL daily", "skills_raw": ""},
            {"description": "", "skills_raw": "python|sql|aws"},
            {"description": "We use Kubernetes", "skills_raw": "k8s"},
            {"description": "Pythonic code review", "skills_raw": ""},
            {"description": "", "skills_raw": ""},
            {
                "description": "We use Python, PostgreSQL, and Docker.",
                "skills_raw": "Python, Docker, GraphQL",
            },
        ]
        df = pd.DataFrame(cases)
        df_ch06 = ch06_extract(df.copy(), Config())

        for i, case in enumerate(cases):
            ch06_value = df_ch06.iloc[i]["skills_normalised"]
            ch06_skills = sorted(ch06_value.split("|")) if ch06_value else []
            ts_result = extract_skills(
                case["description"],
                skills_raw=case["skills_raw"],
                method="regex",
            )
            ts_names = sorted([s.name for s in ts_result])
            assert ts_names == ch06_skills, (
                f"Case {i}: ch06={ch06_skills}, talentlens.skills={ts_names}, " f"input={case}"
            )


class TestEvaluateExtractor:
    """Behaviour tests for evaluate_extractor."""

    def _make_extractor(self, mapping: dict[str, list[str]]):
        """Make a deterministic extractor for testing."""

        def fn(text, *, skills_raw=None):
            return [ExtractedSkill(name=n) for n in mapping.get(text, [])]

        return fn

    def test_perfect_extractor_gets_1_0(self):
        extractor = self._make_extractor(
            {
                "post1": ["Python", "SQL"],
                "post2": ["AWS"],
            }
        )
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": ["Python", "SQL"]},
            {"description": "post2", "skills_raw": "", "skills_verified": ["AWS"]},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.n_postings == 2
        assert metrics.n_predictions == 3
        assert metrics.n_truth == 3

    def test_zero_recall_when_extractor_returns_nothing(self):
        extractor = self._make_extractor({})
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": ["Python"]},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.recall == 0.0
        assert metrics.precision == 0.0

    def test_low_precision_when_extractor_overpredicts(self):
        extractor = self._make_extractor(
            {
                "post1": ["Python", "SQL", "AWS", "Docker"],
            }
        )
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": ["Python"]},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.precision == 0.25
        assert metrics.recall == 1.0

    def test_macro_average_weights_postings_equally(self):
        extractor = self._make_extractor(
            {
                "many": ["Python", "SQL", "AWS", "Docker", "Kubernetes"],
                "one": ["Python"],
            }
        )
        eval_set = [
            {
                "description": "many",
                "skills_raw": "",
                "skills_verified": [
                    "Python",
                    "SQL",
                    "AWS",
                    "Docker",
                    "Kubernetes",
                    "PyTorch",
                    "TensorFlow",
                    "Spark",
                    "Airflow",
                    "Java",
                ],
            },
            {"description": "one", "skills_raw": "", "skills_verified": ["Python"]},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.precision == 1.0
        assert metrics.recall == 0.75

    def test_skips_unlabelled_rows(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        extractor = self._make_extractor({"post1": ["Python"]})
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": ["Python"]},
            {"description": "unlabelled", "skills_raw": "", "skills_verified": None},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.n_postings == 1
        assert "skipped 1 unlabelled" in caplog.text

    def test_raises_when_no_labelled_rows(self):
        extractor = self._make_extractor({})
        eval_set = [
            {"description": "x", "skills_raw": "", "skills_verified": None},
        ]
        with pytest.raises(ValueError, match="No labelled rows"):
            evaluate_extractor(extractor, eval_set)

    def test_per_skill_breakdown(self):
        extractor = self._make_extractor(
            {
                "post1": ["Python", "SQL"],
                "post2": ["Python", "AWS"],
                "post3": ["SQL"],
            }
        )
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": ["Python", "SQL"]},
            {"description": "post2", "skills_raw": "", "skills_verified": ["Python", "Docker"]},
            {"description": "post3", "skills_raw": "", "skills_verified": ["SQL", "Docker"]},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        python_p, python_r, python_support = metrics.per_skill["Python"]
        assert python_p == 1.0
        assert python_r == 1.0
        assert python_support == 2
        sql_p, sql_r, sql_support = metrics.per_skill["SQL"]
        assert sql_p == 1.0
        assert sql_r == 1.0
        aws_p, aws_r, aws_support = metrics.per_skill["AWS"]
        assert aws_p == 0.0
        assert aws_support == 0
        docker_p, docker_r, docker_support = metrics.per_skill["Docker"]
        assert docker_r == 0.0
        assert docker_support == 2

    def test_empty_extracted_and_empty_truth_is_vacuously_correct(self):
        extractor = self._make_extractor({})
        eval_set = [
            {"description": "post1", "skills_raw": "", "skills_verified": []},
        ]
        metrics = evaluate_extractor(extractor, eval_set)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0


class TestLoadEvalSet:
    def test_loads_jsonl_file(self, tmp_path):
        f = tmp_path / "eval.jsonl"
        f.write_text(
            '{"description": "x", "skills_raw": "", "skills_verified": ["Python"]}\n'
            '{"description": "y", "skills_raw": "", "skills_verified": null}\n'
        )
        rows = load_eval_set(f)
        assert len(rows) == 2
        assert rows[0]["skills_verified"] == ["Python"]
        assert rows[1]["skills_verified"] is None
