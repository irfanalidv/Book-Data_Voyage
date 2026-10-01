"""Chapter 10: Feature Engineering and Selection - ablation experiment.

Loads the TalentLens cleaned dataset, engineers features via
``talentlens.features.engineer_features``, runs an ablation
measuring the marginal F1 contribution of each feature group, and
writes results into ``reports/feature_engineering_report.md``.

The chapter's hypothesis-result table is filled in by this script.
Feature selection (iteration 4) and the v2 model artefact (iteration 5)
are separate runs.

Run:
    python book/ch10/ch10_feature_engineering_selection.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from talentlens.features import (
    FEATURE_GROUPS,
    engineer_features,
    scrub_role_phrases_from_text,
)
from talentlens.paths import REPO_ROOT, jobs_clean_path

SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureSignalIssue:
    """Describes a feature that won't contribute to the model."""

    column: str
    kind: str  # "all_nan" or "constant"
    detail: str


@dataclass
class Config:
    clean_path: Path = field(default_factory=jobs_clean_path)
    v2_model_path: Path = REPO_ROOT / "models" / "role_classifier_v2.joblib"
    figures_dir: Path = REPO_ROOT / "book" / "ch10" / "reports" / "figures"
    report_path: Path = REPO_ROOT / "book" / "ch10" / "reports" / "feature_engineering_report.md"
    random_state: int = 42
    cv_folds: int = 5
    tfidf_max_features: int = 5_000  # smaller than Ch9 - bundled dataset is small


def _build_pipeline(numeric_features: list[str], cfg: Config) -> Pipeline:
    """Build a pipeline combining TF-IDF on text + numeric features.

    The text column is hard-coded to ``feature_text`` (description +
    skills_normalised, the same as Ch9). Numeric features are
    scaled. Logistic regression as the model (same as Ch9 baseline
    so the comparison is apples-to-apples).
    """
    text_pipe = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            ("text", text_pipe, "feature_text"),
            ("num", numeric_pipe, numeric_features),
        ]
    )
    return Pipeline(
        [
            ("features", column_transformer),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def _build_feature_text(df: pd.DataFrame) -> pd.Series:
    """Same text construction as Ch9 - description + skills, no title.

    The scrub is load-bearing on bundled demo data, where every
    description embeds the role name. On real scraped data the
    scrub is effectively a no-op. See talentlens.features for the
    function and Chapter 9's Common Mistakes for the story.
    """
    desc = df["description"].fillna("").astype(str).str[:800].apply(scrub_role_phrases_from_text)
    skills = df["skills_normalised"].fillna("").astype(str).str.replace("|", " ")
    return skills + " " + skills + " " + desc


def _check_features_have_signal(
    df: pd.DataFrame,
    numeric_features: list[str],
) -> list[FeatureSignalIssue]:
    """Return a list of features that won't contribute signal.

    Two distinct cases:
      - ``all_nan``: the column is entirely NaN. This is almost
        always an upstream bug (the column wasn't populated by an
        earlier chapter), and the caller should treat it as fatal.
      - ``constant``: the column has values but they're all the
        same. This is a measurement, not a bug: on the bundled
        ~540-row dataset, some features genuinely have no variance
        (e.g. ``is_remote_in_title`` is 0 for every demo row
        because no demo title contains "remote"). The caller
        should warn and continue, and the report should annotate
        the F1 delta with "constant on dataset".

    The distinction matters because the chapter's voice is to
    report honest measurements; collapsing both cases into a
    single error treats a real measurement as a bug.
    """
    issues: list[FeatureSignalIssue] = []
    for col in numeric_features:
        if col not in df.columns:
            continue
        n_non_null = df[col].notna().sum()
        if n_non_null == 0:
            issues.append(
                FeatureSignalIssue(
                    column=col,
                    kind="all_nan",
                    detail=(
                        f"Feature '{col}' is entirely NaN on the loaded "
                        f"dataset ({len(df):,} rows). This is almost "
                        f"certainly an upstream bug — re-check the "
                        f"column's producer (typically Chapter 6)."
                    ),
                )
            )
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique < 2:
            value_repr = repr(df[col].dropna().iloc[0])
            issues.append(
                FeatureSignalIssue(
                    column=col,
                    kind="constant",
                    detail=(
                        f"Feature '{col}' is constant on this dataset "
                        f"(all values = {value_repr}). The bundled "
                        f"sample is small and some features genuinely "
                        f"have no variance here; the ablation will "
                        f"include the feature but its contribution is "
                        f"structurally 0.000."
                    ),
                )
            )
    return issues


def _evaluate(
    df: pd.DataFrame,
    numeric_features: list[str],
    cfg: Config,
) -> tuple[float, float, list[FeatureSignalIssue]]:
    """Return (mean_f1, std_f1, weak_features).

    ``weak_features`` lists features that won't contribute signal
    on this dataset. all_nan features cause this function to raise
    ValueError - those are bugs, not measurements. constant
    features are returned as warnings; the caller decides how to
    report them in the final table.
    """
    issues = _check_features_have_signal(df, numeric_features)
    fatal = [i for i in issues if i.kind == "all_nan"]
    if fatal:
        raise ValueError(
            "Refusing to evaluate — features are entirely NaN:\n  "
            + "\n  ".join(i.detail for i in fatal)
        )

    for issue in issues:
        logger.warning(f"  weak feature: {issue.detail}")

    X = df[["feature_text"] + numeric_features].copy()
    y = df["role_category"]

    pipe = _build_pipeline(numeric_features, cfg)
    cv = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean()), float(scores.std()), issues


def _summarise_feature_variance(df: pd.DataFrame) -> list[dict]:
    """Per-feature variance summary across all FEATURE_GROUPS.

    For each engineered feature, report:
      - the group it belongs to
      - n_non_null
      - n_unique values (NaN excluded)
      - "kind": one of "varying" (>=2 unique), "constant" (exactly 1),
        or "all_nan" (0 non-null)

    The chapter's reader sees this as a top-level fact about the
    bundled dataset, separate from the ablation. The ablation
    measures the joint contribution; this summary explains why
    any given joint contribution can be misleading.
    """
    summary: list[dict] = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        for col in group_cols:
            if col not in df.columns:
                continue
            n_non_null = int(df[col].notna().sum())
            n_unique = int(df[col].nunique(dropna=True))
            if n_non_null == 0:
                kind = "all_nan"
            elif n_unique < 2:
                kind = "constant"
            else:
                kind = "varying"
            summary.append(
                {
                    "group": group_name,
                    "feature": col,
                    "n_non_null": n_non_null,
                    "n_unique": n_unique,
                    "kind": kind,
                }
            )
    return summary


def _run_ablation(df: pd.DataFrame, cfg: Config) -> dict[str, dict]:
    """Run the cumulative ablation.

    Returns a dict mapping configuration name to:
      {"mean": float, "std": float, "issues": list[FeatureSignalIssue]}
    """
    results: dict[str, dict] = {}

    logger.info("Evaluating baseline (text only)...")
    mean, std, issues = _evaluate(df, [], cfg)
    results["baseline (text only)"] = {
        "mean": mean,
        "std": std,
        "issues": issues,
    }

    cumulative: list[str] = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        cumulative.extend(group_cols)
        label = f"+ {group_name}"
        logger.info(f"Evaluating {label} (features so far: {len(cumulative)})...")
        mean, std, issues = _evaluate(df, list(cumulative), cfg)
        results[label] = {
            "mean": mean,
            "std": std,
            "issues": issues,
        }

    return results


def _run_per_feature_ablation(df: pd.DataFrame, cfg: Config) -> dict[str, dict]:
    """Measure each engineered feature's INDIVIDUAL contribution.

    For each feature X in FEATURE_GROUPS, train (baseline_text + X) and
    measure F1. Lift = F1(text + X) - F1(text only). The per-feature
    view is more informative than the cumulative one because the
    cumulative view conflates a group's effect with any one feature's
    effect.

    Constant features will show 0.000 lift; that's reported as a
    measurement, not skipped.

    Returns:
        dict mapping feature_name to
        {"group": str, "mean": float, "std": float, "lift": float,
         "is_constant": bool}.
    """
    baseline_mean, baseline_std, _ = _evaluate(df, [], cfg)
    results: dict[str, dict] = {}

    feature_to_group: dict[str, str] = {}
    for group, cols in FEATURE_GROUPS.items():
        for col in cols:
            feature_to_group[col] = group

    all_features = [c for cols in FEATURE_GROUPS.values() for c in cols]
    for feat in all_features:
        if feat not in df.columns:
            logger.warning(f"  per-feature: skipping missing column {feat}")
            continue
        is_constant = df[feat].nunique(dropna=True) < 2
        if is_constant:
            results[feat] = {
                "group": feature_to_group[feat],
                "mean": baseline_mean,
                "std": baseline_std,
                "lift": 0.0,
                "is_constant": True,
            }
            continue
        mean, std, _ = _evaluate(df, [feat], cfg)
        results[feat] = {
            "group": feature_to_group[feat],
            "mean": mean,
            "std": std,
            "lift": mean - baseline_mean,
            "is_constant": False,
        }
        logger.info(
            f"  per-feature: {feat:<30} "
            f"F1={mean:.3f} ± {std:.3f}  lift={mean - baseline_mean:+.3f}"
        )
    return results


def _run_selection_comparison(df: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Run all three selection methods on the engineered features.

    Returns:
        dict mapping method name to ranked list of features.
    """
    from talentlens.features import select_features

    all_features = [c for cols in FEATURE_GROUPS.values() for c in cols]
    X = df[all_features].copy()
    y = df["role_category"]

    results: dict[str, list[str]] = {}
    for method in ("mutual_info", "rfe", "l1"):
        logger.info(f"  selection: running {method}...")
        try:
            ranked = select_features(X, y, method=method, k=15)
        except Exception as e:
            logger.warning(f"    {method} failed: {e}")
            ranked = []
        results[method] = ranked
    return results


def _select_varying_features(variance: list[dict]) -> list[str]:
    """Return feature names that vary on the loaded dataset.

    A varying feature has at least 2 distinct non-NaN values.
    Constant and all-NaN features are excluded - they cannot
    contribute signal.
    """
    return [v["feature"] for v in variance if v["kind"] == "varying"]


def _train_final_model(
    df: pd.DataFrame,
    selected_features: list[str],
    cfg: Config,
) -> tuple[float, float]:
    """Train the final v2 model on the engineered + selected feature set.

    Uses the same pipeline shape as the ablation evaluator. Fits
    on the full dataset (no train/test split) and saves to disk.
    Returns (cv_mean_f1, cv_std_f1) - measured via cross-validation
    BEFORE fitting on the full data, so the reported number is
    out-of-sample.

    Args:
        df: DataFrame after engineer_features.
        selected_features: Numeric feature columns to include
            alongside feature_text. Empty list means text-only
            (Ch9 baseline shape).
        cfg: Config.

    Returns:
        Tuple of (mean macro F1, std macro F1) from CV.
    """
    import joblib

    cv_mean, cv_std, _ = _evaluate(df, selected_features, cfg)
    logger.info(
        f"  v2 CV macro F1: {cv_mean:.3f} ± {cv_std:.3f} "
        f"({len(selected_features)} numeric features + text)"
    )

    X = df[["feature_text"] + selected_features].copy()
    y = df["role_category"]
    pipe = _build_pipeline(selected_features, cfg)
    pipe.fit(X, y)

    cfg.v2_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg.v2_model_path)
    logger.info(f"  v2 model saved to {cfg.v2_model_path}")

    return cv_mean, cv_std


def _maybe_replace_hypothesis_row(
    line: str,
    per_feature: dict[str, dict],
    baseline_mean: float,
) -> str | None:
    """If `line` is a hypothesis-result row, fill in Result and Decision.

    Returns the new line, or None if `line` is not a row we replace.
    """
    # Rows are re-filled on every run so the table can never go stale.
    if not line.startswith("| ") or line.count("|") < 6:
        return None
    label_to_col = {
        "seniority_level (ordinal)": "seniority_level",
        "is_remote_in_title": "is_remote_in_title",
        "skill_count": "skill_count",
        "has_python / has_sql / etc.": "has_python",
        "log_salary_min": "log_salary_annual_inr",
        "log_salary_annual_inr": "log_salary_annual_inr",
        "salary_per_skill": "salary_per_skill",
        "salary_in_band_for_city": "salary_in_band_for_city",
    }
    matched_col = None
    for label, col in label_to_col.items():
        if label in line:
            matched_col = col
            break
    if matched_col is None or matched_col not in per_feature:
        return None

    result = per_feature[matched_col]
    lift = result["lift"]
    is_const = result["is_constant"]
    if is_const:
        result_str = "0.000 (constant on dataset)"
        decision = "Skipped — no variance on demo data"
    elif lift > 0.01:
        result_str = f"+{lift:.3f} F1"
        decision = "Keep"
    elif lift < -0.01:
        result_str = f"{lift:+.3f} F1"
        decision = "Drop"
    else:
        result_str = f"{lift:+.3f} F1 (within noise)"
        decision = "Keep for interpretability"

    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 7:
        return None
    parts[-3] = result_str
    parts[-2] = decision
    return "| " + " | ".join(parts[1:-1]) + " |"


def _salary_note(per_feature: dict[str, dict], baseline_mean: float) -> str:
    """Describe what the salary features did on this run, from the measurements."""
    lifts = {
        f: r["lift"]
        for f, r in per_feature.items()
        if r["group"] == "salary" and not r["is_constant"]
    }
    if not lifts:
        return "Salary features were constant or missing on this dataset."
    best = max(lifts, key=lifts.get)
    worst = min(lifts, key=lifts.get)
    if lifts[best] > 0.01:
        return (
            f"Salary features lifted F1 by up to {lifts[best]:+.3f} (`{best}`). Before "
            "keeping them, ask whether salary is partly a function of the label — on "
            "any dataset where pay bands are set per role, a salary feature is one hop "
            "from label leakage."
        )
    return (
        f"Salary features did not help: individual lifts ranged from {lifts[worst]:+.3f} "
        f"to {lifts[best]:+.3f}. Pay overlaps across roles once seniority varies "
        "(a Lead Data Analyst can out-earn a Junior ML Engineer), so salary adds noise "
        "to a role classifier rather than signal."
    )


def _select_by_lift(per_feature: dict[str, dict], min_lift: float = 0.01) -> list[str]:
    """Features whose individual lift over the text baseline exceeds ``min_lift``."""
    return [f for f, r in per_feature.items() if not r["is_constant"] and r["lift"] > min_lift]


def _write_report(
    ablation: dict[str, dict],
    variance: list[dict],
    per_feature: dict[str, dict],
    selection: dict[str, list[str]],
    cfg: Config,
    n_rows: int,
    v2_result: tuple[float, float, list[str]] | None = None,
) -> None:
    """Update the report with all measured sections.

    Sections emitted, in order:
      - Baseline row (in the existing Baseline table)
      - Hypothesis-result table (filled in from per_feature)
      - Final model (filled from v2_result when provided)
      - Feature selection comparison (filled in from selection)
      - Dataset notes
      - Ablation results
    """
    baseline_mean = ablation["baseline (text only)"]["mean"]
    baseline_std = ablation["baseline (text only)"]["std"]

    lines = cfg.report_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        if "| Ch9 baseline (TF-IDF + LR) |" in line:
            new_lines.append(
                f"| Ch9 baseline (TF-IDF + LR) | "
                f"{baseline_mean:.3f} ± {baseline_std:.3f} | "
                f"From Chapter 9, no engineered features |"
            )
            continue
        if v2_result is not None and "| Ch10 final" in line:
            v2_mean, v2_std, v2_features = v2_result
            new_lines.append(
                f"| Ch10 final (engineered + selected) | "
                f"{v2_mean:.3f} ± {v2_std:.3f} | "
                f"{len(v2_features)} engineered (lift > 0.01) + Ch9 text | "
                f"Saved to `models/role_classifier_v2.joblib` |"
            )
            continue
        replaced = _maybe_replace_hypothesis_row(line, per_feature, baseline_mean)
        new_lines.append(replaced if replaced else line)
    report_text = "\n".join(new_lines)

    for header in (
        "## Dataset notes",
        "## Ablation results",
        "## Feature selection comparison",
    ):
        if header in report_text:
            report_text = report_text.split(header)[0].rstrip()

    selection_lines = [
        "",
        "",
        "## Feature selection comparison",
        "",
        "Top-15 features chosen by each method on the engineered "
        "feature set. Methods are described in this chapter; the "
        "fact that they sometimes pick different features is the "
        "point — there is no single correct ranking.",
        "",
        "| Rank | Mutual Information | RFE (CV) | L1 (Logistic Regression) |",
        "|---|---|---|---|",
    ]
    max_len = max(
        len(selection.get("mutual_info", [])),
        len(selection.get("rfe", [])),
        len(selection.get("l1", [])),
        1,
    )
    for i in range(max_len):
        mi = selection.get("mutual_info", [])
        rfe = selection.get("rfe", [])
        l1 = selection.get("l1", [])
        mi_cell = mi[i] if i < len(mi) else "—"
        rfe_cell = rfe[i] if i < len(rfe) else "—"
        l1_cell = l1[i] if i < len(l1) else "—"
        selection_lines.append(f"| {i + 1} | `{mi_cell}` | `{rfe_cell}` | `{l1_cell}` |")
    report_text += "\n" + "\n".join(selection_lines) + "\n"

    n_const = sum(1 for v in variance if v["kind"] == "constant")
    n_nan = sum(1 for v in variance if v["kind"] == "all_nan")
    n_vary = sum(1 for v in variance if v["kind"] == "varying")

    on_release_data = n_rows >= 1000 or "large" in cfg.clean_path.name
    rows_blurb = (
        f"{n_rows:,} rows from Adzuna India collected across five " f"canonical TalentLens roles"
        if on_release_data
        else f"{n_rows:,} rows"
    )
    dataset_lines = [
        "",
        "",
        "## Dataset notes",
        "",
        f"_Variance of engineered features on the loaded dataset "
        f"(`{cfg.clean_path.name}`, {rows_blurb}): "
        f"**{n_vary}** features vary, **{n_const}** are constant, "
        f"**{n_nan}** are entirely NaN._",
        "",
        _salary_note(per_feature, baseline_mean),
        "",
        "| Group | Feature | n_non_null | n_unique | Kind |",
        "|---|---|---|---|---|",
    ]
    for v in variance:
        dataset_lines.append(
            f"| {v['group']} | {v['feature']} | {v['n_non_null']} | "
            f"{v['n_unique']} | {v['kind']} |"
        )
    report_text += "\n" + "\n".join(dataset_lines) + "\n"

    ablation_lines = [
        "",
        "",
        "## Ablation results",
        "",
        "_Cumulative feature groups added to the Ch9-shape baseline. "
        "Numbers are macro F1 ± standard deviation across 5-fold "
        "stratified CV._",
        "",
        "| Configuration | F1 ± std | Δ vs baseline | Notes |",
        "|---|---|---|---|",
    ]
    for name, result in ablation.items():
        mean, std = result["mean"], result["std"]
        issues = result["issues"]
        if name == "baseline (text only)":
            delta_str = "—"
        else:
            delta_str = f"{mean - baseline_mean:+.3f}"
        constant_cols = [i.column for i in issues if i.kind == "constant"]
        notes: list[str] = []
        if constant_cols:
            shown = ", ".join(constant_cols[:3])
            if len(constant_cols) > 3:
                shown += f" (+{len(constant_cols) - 3} more)"
            notes.append(f"Constant: {shown}")
        if name == "+ salary":
            notes.append("see Dataset notes on salary and role")
        note_str = "; ".join(notes)
        ablation_lines.append(f"| {name} | {mean:.3f} ± {std:.3f} | {delta_str} | {note_str} |")

    report_text += "\n" + "\n".join(ablation_lines) + "\n"
    cfg.report_path.write_text(report_text)
    logger.info(f"Report written to {cfg.report_path}")


def plot_selection_ranks(selection: dict[str, list[str]], cfg: Config) -> Path:
    """Grid of each engineered feature's rank under each selection method.

    Blank cells mean the method did not keep the feature in its top k.
    Where the columns disagree is the point of the figure.
    """
    methods = [("mutual_info", "Mutual information"), ("rfe", "RFE"), ("l1", "L1 (embedded)")]
    features = [c for cols in FEATURE_GROUPS.values() for c in cols]
    ranks = np.full((len(features), len(methods)), np.nan)
    for j, (key, _) in enumerate(methods):
        for r, feat in enumerate(selection.get(key, []), start=1):
            if feat in features:
                ranks[features.index(feat), j] = r

    order = np.argsort(np.nan_to_num(np.nanmean(ranks, axis=1), nan=99.0))
    ranks, features = ranks[order], [features[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(features) + 1.6))
    ax.imshow(
        np.nan_to_num(ranks, nan=len(features) + 4),
        cmap="Blues_r",
        aspect="auto",
        vmin=1,
        vmax=len(features) + 4,
    )
    for i in range(len(features)):
        for j in range(len(methods)):
            if not np.isnan(ranks[i, j]):
                ax.text(
                    j,
                    i,
                    f"{int(ranks[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if ranks[i, j] <= 5 else "#1a1a1a",
                )
    ax.set_xticks(range(len(methods)), [m[1] for m in methods])
    ax.set_yticks(range(len(features)), features)
    ax.grid(False)
    ax.set_title(
        "Feature rank by selection method (1 = most important)", fontsize=12, fontweight="bold"
    )
    out = cfg.figures_dir / "ch10_feature_importance.png"
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


def plot_skill_salary_premium(df: pd.DataFrame, cfg: Config) -> Path:
    """Median disclosed salary for postings with and without each skill flag.

    Uses employer-disclosed salaries only: imputed values are role medians,
    so including them would show the role, not the skill (see Common mistakes).
    """
    disclosed = df[df["salary_disclosed"].astype(bool)] if "salary_disclosed" in df else df
    pay = disclosed["salary_annual_inr"] / 1e5
    skills = [c for cols in FEATURE_GROUPS.values() for c in cols if c.startswith("has_")]
    rows = []
    for col in skills:
        has = disclosed[col].astype(bool)
        if has.sum() >= 5 and (~has).sum() >= 5:
            rows.append(
                (col.removeprefix("has_"), pay[has].median(), pay[~has].median(), int(has.sum()))
            )
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color="#4CAF50", label="Has skill")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color="#F44336", alpha=0.85, label="No skill")
    ax.set_xticks(x, [f"{r[0]}\n(n={r[3]})" for r in rows])
    ax.set_ylabel("Median disclosed salary (₹ lakhs)")
    ax.set_title(
        f"Median salary with and without each skill ({len(disclosed)} disclosed salaries)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    out = cfg.figures_dir / "ch10_skill_salary_premium.png"
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()

    logger.info(f"Loading: {cfg.clean_path}")
    df = pd.read_csv(cfg.clean_path)
    logger.info(f"Loaded {len(df):,} rows")

    logger.info("Engineering features...")
    df = engineer_features(df)
    df["feature_text"] = _build_feature_text(df)
    logger.info(f"Final shape: {df.shape}")

    logger.info("Summarising feature variance...")
    variance = _summarise_feature_variance(df)
    n_constant = sum(1 for v in variance if v["kind"] == "constant")
    n_nan = sum(1 for v in variance if v["kind"] == "all_nan")
    logger.info(
        f"  Features: {len(variance)} total, "
        f"{len(variance) - n_constant - n_nan} varying, "
        f"{n_constant} constant, {n_nan} all-NaN"
    )

    logger.info("Running cumulative ablation...")
    ablation = _run_ablation(df, cfg)

    logger.info("Running per-feature ablation...")
    per_feature = _run_per_feature_ablation(df, cfg)

    logger.info("Running selection-method comparison...")
    selection = _run_selection_comparison(df, cfg)

    plot_selection_ranks(selection, cfg)
    plot_skill_salary_premium(df, cfg)

    logger.info("Training final v2 model...")
    # v2 keeps only features that earned their place in the per-feature
    # ablation. If none did, v2 is the text-only baseline - never worse.
    varying_features = _select_by_lift(per_feature)
    logger.info(f"  features kept for v2: {varying_features or 'none (text only)'}")
    v2_mean, v2_std = _train_final_model(df, varying_features, cfg)

    logger.info("Results:")
    for name, result in ablation.items():
        mean, std = result["mean"], result["std"]
        annotation = ""
        if result["issues"]:
            constant_cols = [i.column for i in result["issues"] if i.kind == "constant"]
            if constant_cols:
                annotation = f"   (constant: {len(constant_cols)} features)"
        logger.info(f"  {name:<30} {mean:.3f} ± {std:.3f}{annotation}")
    logger.info(
        f"  v2 final                       {v2_mean:.3f} ± {v2_std:.3f}   "
        f"({len(varying_features)} engineered features)"
    )

    _write_report(
        ablation,
        variance,
        per_feature,
        selection,
        cfg,
        n_rows=len(df),
        v2_result=(v2_mean, v2_std, varying_features),
    )
    logger.info("Chapter 10 complete. Next: python book/ch11/ch11_unsupervised_learning.py")


if __name__ == "__main__":
    main()
