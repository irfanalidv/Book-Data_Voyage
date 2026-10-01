"""Smoke tests for the Chapter 10 scaffolding.

These tests verify that the public API surface exists and is wired
correctly. They do NOT test feature correctness - that's done by
the chapter's own tests once the builders are filled in.
"""

from __future__ import annotations

import importlib


def test_features_module_imports():
    mod = importlib.import_module("talentlens.features")
    assert hasattr(mod, "engineer_features")
    assert hasattr(mod, "select_features")
    assert hasattr(mod, "FEATURE_GROUPS")


def test_feature_groups_has_expected_keys():
    from talentlens.features import FEATURE_GROUPS

    assert set(FEATURE_GROUPS) >= {"title", "skills", "salary"}


def test_engineer_features_is_callable():
    from talentlens.features import engineer_features

    assert callable(engineer_features)


def test_role_classifier_path_helper_exists():
    from talentlens.paths import role_classifier_path

    p = role_classifier_path()
    assert p.name in ("role_classifier.joblib", "role_classifier_v2.joblib")
