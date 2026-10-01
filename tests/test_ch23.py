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
    RADAR_DIMENSIONS,
    Config,
    build_radar_scores,
    plot_lessons_matrix,
    write_case_studies_summary,
)


class TestCaseStudyData:
    def test_three_named_case_studies(self):
        names = [s["name"] for s in CASE_STUDIES]
        assert names == ["Reflecta", "Godam", "RAGNav"]

    def test_metrics_dict_has_expected_keys(self):
        for study in CASE_STUDIES:
            metrics = study["metrics"]
            assert set(metrics.keys()) == {
                "stack_complexity",
                "months_to_ship",
                "lines_of_code",
                "prod_incidents",
            }
            assert metrics["months_to_ship"] >= 1


class TestCaseStudiesReport:
    def test_summary_markdown_lists_all_systems(self, tmp_path):
        cfg = Config(
            figures_dir=tmp_path / "figures",
            reports_dir=tmp_path / "reports",
        )
        out = write_case_studies_summary(cfg)
        text = out.read_text(encoding="utf-8")
        assert "## Reflecta —" in text
        assert "## Godam —" in text
        assert "## RAGNav —" in text
        assert "Key lessons:" in text
        assert "What broke (and why):" in text

    def test_radar_scores_shape_and_plot_output(self, tmp_path):
        """Radar data comes from CASE_STUDIES metrics; plot writes the PNG."""
        scores = build_radar_scores()
        system_names = ["Reflecta", "Godam", "RAGNav"]
        assert list(scores.keys()) == system_names
        assert len(RADAR_DIMENSIONS) == 5

        dimension_sets = {name: set(RADAR_DIMENSIONS) for name in system_names}
        assert dimension_sets["Reflecta"] == dimension_sets["Godam"] == dimension_sets["RAGNav"]

        for name in system_names:
            vals = scores[name]
            assert len(vals) == 5
            assert all(0.0 <= v <= 10.0 for v in vals)

        assert scores["RAGNav"][-1] == 10.0
        assert scores["Reflecta"][-1] == scores["Godam"][-1] == 0.0

        cfg = Config(
            figures_dir=tmp_path / "figures",
            reports_dir=tmp_path / "reports",
        )
        out = plot_lessons_matrix(cfg)
        assert out == cfg.figures_dir / "ch23_lessons_matrix.png"
        assert out.exists()
        assert out.stat().st_size > 1000
