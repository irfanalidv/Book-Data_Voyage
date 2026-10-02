"""Tests for Chapter 23: Real-World Case Studies.

Reference: book/ch23/README.md
Source: book/ch23/ch23_case_studies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "book" / "ch23"))

from ch23_case_studies import (  # noqa: E402
    CASE_STUDIES,
    LAYER_OF,
    LESSON_MATRIX,
    PATTERNS,
    Config,
    plot_lessons_matrix,
    plot_system_architectures,
    write_case_studies_summary,
)

SYSTEMS = ["Reflecta", "Godam", "RAGNav", "StackSift"]


class TestCaseStudyData:
    def test_four_named_case_studies(self):
        assert [s["name"] for s in CASE_STUDIES] == SYSTEMS

    def test_every_study_has_lessons_and_incidents(self):
        for study in CASE_STUDIES:
            assert study["stack"] and study["key_lessons"] and study["what_broke"]

    def test_every_stack_item_has_a_layer(self):
        for study in CASE_STUDIES:
            for tech in study["stack"]:
                assert tech in LAYER_OF, tech


class TestLessonsMatrix:
    def test_matrix_covers_every_system_and_pattern(self):
        assert list(LESSON_MATRIX) == SYSTEMS
        assert len(PATTERNS) == 6
        for levels in LESSON_MATRIX.values():
            assert len(levels) == len(PATTERNS)
            assert set(levels) <= {0, 1, 2}

    def test_patterns_match_the_chapter_text(self):
        text = (_REPO_ROOT / "book" / "ch23" / "README.md").read_text(encoding="utf-8")
        assert "**The hard part is rarely the AI.**" in text
        assert "**Observability before features.**" in text
        assert "**Measure before you trust an improvement.**" in text

    def test_first_three_patterns_appear_in_every_system(self):
        for levels in LESSON_MATRIX.values():
            assert all(level > 0 for level in levels[:3])


class TestOutputs:
    def test_summary_markdown_lists_all_systems(self, tmp_path):
        cfg = Config(figures_dir=tmp_path / "figures", reports_dir=tmp_path / "reports")
        text = write_case_studies_summary(cfg).read_text(encoding="utf-8")
        for name in SYSTEMS:
            assert f"## {name}:" in text
        assert "Key lessons:" in text
        assert "What broke (and why):" in text

    def test_figures_are_written(self, tmp_path):
        cfg = Config(figures_dir=tmp_path / "figures", reports_dir=tmp_path / "reports")
        for out in (plot_lessons_matrix(cfg), plot_system_architectures(cfg)):
            assert out.parent == cfg.figures_dir
            assert out.stat().st_size > 1000
