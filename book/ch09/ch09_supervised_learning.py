"""
Chapter 9: Supervised Learning - The TalentLens Role Classifier
Data Voyage - Building TalentLens

TalentLens milestone: train a job role classifier that predicts whether
a posting is AI Engineer, ML Engineer, Data Scientist, Data Engineer,
or Data Analyst - using description + skills_normalised as features
(labels from Chapter 6 ``role_category``, derived from title).

Saved model feeds Ch19 (FastAPI endpoint) and Ch22 (talentlens-core PyPI library).

Run:
    python book/ch09/ch09_supervised_learning.py

Outputs:
    book/ch09/models/role_classifier.joblib
    book/ch09/reports/figures/ch09_model_comparison.png
    book/ch09/reports/figures/ch09_confusion_matrix.png
    book/ch09/reports/figures/ch09_feature_importance.png
    book/ch09/reports/model_evaluation.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from talentlens.features import scrub_role_phrases_from_text
from talentlens.paths import jobs_clean_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Data
    clean_data_path: Path = field(default_factory=jobs_clean_path)
    fallback_data_path: Path = _REPO_ROOT / "data" / "raw" / "jobs_raw.csv"

    # Model output
    model_path: Path = _THIS_DIR / "models" / "role_classifier.joblib"
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"

    # TF-IDF
    tfidf_max_features: int = 15_000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.90

    # Training (fixed seed - report metrics must be reproducible on bundled CSV)
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    report_model_relpath: str = "book/ch09/models/role_classifier.joblib"

    # Role categories
    role_labels: list[str] = field(
        default_factory=lambda: [
            "AI Engineer",
            "ML Engineer",
            "Data Scientist",
            "Data Engineer",
            "Data Analyst",
        ]
    )


# ---------------------------------------------------------------------------
# Labels and features
# ---------------------------------------------------------------------------
#
# Labels come from Chapter 6's ``role_category`` column. Chapter 6 derives
# ``role_category`` from the job *title* via keyword patterns. To avoid
# the obvious leak - training a classifier on title to predict a label
# built from title - this chapter trains on ``description`` and
# ``skills_normalised`` only. Title is deliberately excluded from features.
#
# This is a meaningful task: in production, posting bodies arrive before
# titles are normalised (titles vary across sources), and a model that
# classifies a role from the body alone is genuinely useful.
#
# Residual leakage to be honest about: Chapter 6's ``extract_skills``
# keyword-scans ``description`` to build ``skills_normalised``. So a
# description that contains "PyTorch" populates a skill token the
# classifier sees. The role labels themselves are not derived from the
# skills field, so this is not circular - but the path is worth naming.
# The "Common mistakes" section of this chapter discusses both the
# original leak and this residual one.


def build_feature_text(row: pd.Series) -> str:
    """Build the classifier's input text from description and normalised skills only.

    Title is deliberately excluded. The labels in ``role_category`` come from
    Chapter 6's title-based heuristic; training on title would let the
    classifier reproduce the heuristic instead of learning anything new.

    Skills are repeated twice to weight them against the much longer
    description text. Description is truncated to 800 characters to keep
    TF-IDF vocabulary stable across short and long postings.

    Args:
        row: DataFrame row with ``description`` and ``skills_normalised``
            columns.

    Returns:
        Combined text string for vectorisation.
    """
    desc = scrub_role_phrases_from_text(str(row.get("description", ""))[:800])
    skills = str(row.get("skills_normalised", "")).replace("|", " ").strip()
    return f"{skills} {skills} {desc}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: Config) -> pd.DataFrame:
    """Load the TalentLens dataset and prepare labels and feature text.

    Labels are read from ``role_category`` (Chapter 6). Feature text uses
    description and skills only - never title. Falls back to demo data if
    no CSV is available.

    Args:
        cfg: Config with clean_data_path and fallback_data_path.

    Returns:
        DataFrame with feature_text and role_label columns added.
    """
    for path in [cfg.clean_data_path, cfg.fallback_data_path]:
        if path.exists():
            logger.info(f"Loading data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df):,} rows")
            break
    else:
        logger.warning("No data file found — generating demo dataset")
        df = _generate_demo_data(cfg)

    if "role_category" in df.columns:
        df["role_label"] = df["role_category"]
    elif "role_label" not in df.columns:
        raise KeyError(
            "jobs_clean.csv has neither 'role_category' nor 'role_label'. "
            "Re-run Chapter 6 to regenerate it with the canonical schema."
        )

    df["feature_text"] = df.apply(build_feature_text, axis=1)

    # Log label distribution
    dist = df["role_label"].value_counts()
    _print_block(
        "LABEL DISTRIBUTION",
        [f"  {role:<25} {count:>5,} ({count/len(df)*100:.1f}%)" for role, count in dist.items()],
    )

    return df


def _generate_demo_data(cfg: Config) -> pd.DataFrame:
    """Generate synthetic job postings matching the TalentLens schema.

    DEMO DATA - replace with real pipeline output from Ch05+Ch06.

    Args:
        cfg: Config with role_labels.

    Returns:
        DataFrame of 1,500 synthetic job postings with labels.
    """
    rng = np.random.default_rng(42)
    n_per_class = 300  # 5 classes × 300 = 1,500 total

    templates = {
        "AI Engineer": {
            "titles": ["AI Engineer", "Senior AI Engineer", "LLM Engineer", "GenAI Engineer"],
            "skills": "Python,LLMs,RAG,FastAPI,pgvector,LangChain,OpenAI,Groq,transformers",
            "desc": (
                "Build production LLM applications. Experience with RAG pipelines, "
                "vector databases (pgvector, Qdrant), and LLM APIs (OpenAI, Groq, Anthropic). "
                "FastAPI for API development. Strong Python fundamentals required."
            ),
        },
        "ML Engineer": {
            "titles": [
                "ML Engineer",
                "Senior ML Engineer",
                "Applied ML Engineer",
                "Machine Learning Engineer",
            ],
            "skills": "Python,scikit-learn,XGBoost,MLflow,Spark,feature engineering,model deployment",
            "desc": (
                "Design and deploy machine learning models. Experience with feature engineering, "
                "model training (XGBoost, scikit-learn), experiment tracking (MLflow), "
                "and model deployment. Spark for large-scale feature pipelines."
            ),
        },
        "Data Scientist": {
            "titles": [
                "Data Scientist",
                "Senior Data Scientist",
                "Applied Scientist",
                "Research Scientist",
            ],
            "skills": "Python,R,statistics,hypothesis testing,regression,scikit-learn,SQL,A/B testing",
            "desc": (
                "Analyse large datasets to generate insights. Strong statistics background, "
                "hypothesis testing, regression modelling. Present findings to stakeholders. "
                "Python and SQL required. Machine learning experience a plus."
            ),
        },
        "Data Engineer": {
            "titles": [
                "Data Engineer",
                "Senior Data Engineer",
                "Analytics Engineer",
                "Platform Engineer",
            ],
            "skills": "Python,SQL,Spark,dbt,Airflow,BigQuery,Kafka,data pipelines,ETL",
            "desc": (
                "Build and maintain data pipelines. Experience with Spark, dbt, Airflow, "
                "and cloud data warehouses (BigQuery, Snowflake). Design ETL processes, "
                "ensure data quality, manage data platform infrastructure."
            ),
        },
        "Data Analyst": {
            "titles": ["Data Analyst", "Business Analyst", "Analytics Analyst", "SQL Analyst"],
            "skills": "SQL,Excel,Tableau,Power BI,Python,business intelligence,reporting,KPI dashboards",
            "desc": (
                "Analyse business data to support decision making. Strong SQL and Excel skills. "
                "Build Tableau or Power BI dashboards. Present insights to business stakeholders. "
                "Python for data manipulation. Statistics background helpful."
            ),
        },
    }

    companies = [
        "Nimbus Fintech",
        "Kestrel Commerce",
        "Monsoon Payments",
        "Banyan Health",
        "Indigo Logistics",
        "Saffron Credit",
        "Teal Analytics",
        "Deccan Mobility",
        "Remote Startup",
        "Global AI Team",
        "Series B Fintech",
        "Lotus Insuretech",
        "Orbit SaaS",
        "Himalaya Cloud",
    ]

    rows = []
    for role, tpl in templates.items():
        for i in range(n_per_class):
            title = rng.choice(tpl["titles"])
            company = companies[rng.integers(0, len(companies))]
            # Add some noise to descriptions
            noise = rng.choice(
                ["", "Competitive salary. ", "Remote options available. ", "Hybrid role. "]
            )
            desc = tpl["desc"] + " " + noise

            # Salary variation
            base_min = {
                "AI Engineer": 2_800_000,
                "ML Engineer": 2_200_000,
                "Data Scientist": 1_800_000,
                "Data Engineer": 1_800_000,
                "Data Analyst": 900_000,
            }[role]
            scale = 0.7 + rng.random() * 0.6
            sal_min = int(base_min * scale)
            sal_max = int(sal_min * (1.3 + rng.random() * 0.4))

            rows.append(
                {
                    "title": title,
                    "company": company,
                    "description": desc,
                    "skills_normalised": tpl["skills"].replace(",", "|"),
                    "salary_min": sal_min if rng.random() > 0.2 else None,
                    "salary_max": sal_max if rng.random() > 0.2 else None,
                    "is_remote": rng.random() > 0.6,
                    "city": rng.choice(["Bangalore", "Mumbai", "Hyderabad", "Remote"]),
                    "role_label": role,
                    "role_category": role,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df):,} demo training examples")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def build_pipeline(model, cfg: Config) -> Pipeline:
    """Build a sklearn Pipeline with TF-IDF and a classifier.

    Args:
        model: Initialised sklearn classifier.
        cfg: Config with TF-IDF parameters.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.tfidf_max_features,
                    ngram_range=cfg.tfidf_ngram_range,
                    min_df=cfg.tfidf_min_df,
                    max_df=cfg.tfidf_max_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                ),
            ),
            ("clf", model),
        ]
    )


# Candidates in order of simplicity: when two models are within noise of each
# other, the earlier one wins.
MODEL_SIMPLICITY_ORDER = ("Logistic Regression", "SGD Classifier", "Random Forest")


def select_model(results: dict[str, dict]) -> str:
    """Pick a model with the one-standard-error rule.

    Take the model with the highest mean cross-validated F1, then accept any
    simpler model whose mean F1 is within one standard deviation of that top
    score. Differences smaller than the fold-to-fold noise are not evidence
    that the complex model is better.

    Args:
        results: ``{name: {"mean_f1": float, "std_f1": float, ...}}`` from
            :func:`compare_models`.

    Returns:
        Name of the selected model.
    """
    top = max(results, key=lambda k: results[k]["mean_f1"])
    threshold = results[top]["mean_f1"] - results[top]["std_f1"]
    for name in MODEL_SIMPLICITY_ORDER:
        if name in results and results[name]["mean_f1"] >= threshold:
            return name
    return top


def compare_models(
    X_train: pd.Series,
    y_train: pd.Series,
    cfg: Config,
) -> tuple[Pipeline, dict[str, dict]]:
    """Train and cross-validate multiple models, return the best.

    Args:
        X_train: Series of feature_text strings.
        y_train: Series of role_label strings.
        cfg: Config with cv_folds and random_state.

    Returns:
        Tuple of (best_pipeline, comparison_dict).
    """
    candidates = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=cfg.random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=cfg.random_state,
        ),
        "SGD Classifier": SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=100,
            random_state=cfg.random_state,
        ),
    }

    results: dict[str, dict] = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model, cfg)
        t0 = time.perf_counter()
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cfg.cv_folds,
            scoring="f1_macro",
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0

        results[name] = {
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
            "train_time": elapsed,
        }
        logger.info(
            f"  {name:<25} F1={scores.mean():.3f} ± {scores.std():.3f}  " f"({elapsed:.1f}s)"
        )

    best_name = select_model(results)
    logger.info(
        f"\n  Selected model: {best_name} (F1={results[best_name]['mean_f1']:.3f}) — "
        f"simplest model within one standard deviation of the top score"
    )
    best_pipeline = build_pipeline(candidates[best_name], cfg)

    # Fit best model on full training data
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    cfg: Config,
) -> dict:
    """Evaluate the trained pipeline on the holdout test set.

    Args:
        pipeline: Fitted Pipeline.
        X_test: Test feature texts.
        y_test: Test labels.
        cfg: Config with role_labels.

    Returns:
        Dict with predictions, probabilities, and evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)
        if hasattr(pipeline.named_steps["clf"], "predict_proba")
        else None
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=cfg.role_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=cfg.role_labels)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    _print_block(
        "HOLDOUT EVALUATION",
        [
            f"  Overall F1 (macro): {macro_f1:.3f}",
            "",
            classification_report(y_test, y_pred, labels=cfg.role_labels, zero_division=0),
        ],
    )

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_role(job: dict, pipeline: Pipeline) -> tuple[str, float]:
    """Predict the role category for a single job posting.

    Args:
        job: Dict with at least ``description``. Optional ``skills_normalised``
            (pipe-separated string, as written by Chapter 6) or ``skills`` /
            ``skills_raw`` (comma-separated) are accepted for convenience.
            ``title`` is accepted but ignored - see the chapter narrative for
            why title is not used at inference time.
        pipeline: Fitted classification Pipeline.

    Returns:
        Tuple of (predicted_role: str, confidence: float).
    """
    skills = job.get("skills_normalised") or str(
        job.get("skills", job.get("skills_raw", ""))
    ).replace(",", "|")
    row = pd.Series(
        {
            "description": job.get("description", ""),
            "skills_normalised": skills,
        }
    )
    text = build_feature_text(row)
    label = pipeline.predict([text])[0]

    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        confidence = float(proba[list(classes).index(label)])
    else:
        confidence = (
            0.85  # SGD doesn't have calibrated probabilities without CalibratedClassifierCV
        )

    return label, round(confidence, 3)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_model_comparison(comparison: dict[str, dict], cfg: Config) -> Path:
    """Bar chart comparing cross-validated F1 scores across models."""
    names = list(comparison.keys())
    means = [comparison[n]["mean_f1"] for n in names]
    stds = [comparison[n]["std_f1"] for n in names]
    times = [comparison[n]["train_time"] for n in names]
    ticks = [n.replace(" ", "\n", 1) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # F1 comparison: value printed inside each bar, error bar above it
    ax = axes[0]
    colors = ["#4CAF50" if m == max(means) else "#2196F3" for m in means]
    bars = ax.bar(ticks, means, color=colors, edgecolor="white", alpha=0.85, width=0.55)
    ax.errorbar(ticks, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.53,
            f"{m:.3f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )
    ax.set_ylabel("Macro F1 (5-fold CV)", fontsize=9)
    ax.set_title("Cross-validated F1 (± 1 std)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", labelsize=8)

    # Training time
    ax2 = axes[1]
    colors2 = ["#FF9800" if t == min(times) else "#607D8B" for t in times]
    bars2 = ax2.bar(ticks, times, color=colors2, edgecolor="white", alpha=0.85, width=0.55)
    for bar, t in zip(bars2, times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.03,
            f"{t:.1f}s",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_ylabel("Training time (seconds)", fontsize=9)
    ax2.set_title("Training time per model", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(times) * 1.25)
    ax2.tick_params(axis="x", labelsize=8)

    plt.suptitle("Role classifier: model selection", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out = cfg.figures_dir / "ch09_model_comparison.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_confusion_matrix(eval_results: dict, cfg: Config) -> Path:
    """Heatmap of the confusion matrix with counts and percentages."""
    cm = eval_results["confusion_matrix"]
    labels = cfg.role_labels

    # Normalise by row (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))

    for n, (ax, data, title, fmt) in enumerate(
        [
            (axes[0], cm, "Counts", "d"),
            (axes[1], cm_norm, "Row-normalised (recall)", ".2f"),
        ]
    ):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
        ax.set_yticklabels(labels if n == 0 else [], fontsize=7.5)
        ax.set_xlabel("Predicted", fontsize=8.5)
        if n == 0:
            ax.set_ylabel("True", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                text_color = "white" if val > data.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                    fontweight="bold",
                )

    plt.suptitle(
        f"Role classifier on the test set: macro F1 {eval_results['macro_f1']:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    plt.tight_layout()
    out = cfg.figures_dir / "ch09_confusion_matrix.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_feature_importance(pipeline: Pipeline, cfg: Config, top_n: int = 15) -> Path:
    """Top TF-IDF features by logistic regression coefficient per class."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        logger.warning("Model has no coef_ — skipping feature importance plot")
        return cfg.figures_dir / "ch09_feature_importance.png"

    vectoriser = pipeline.named_steps["tfidf"]
    feature_names = vectoriser.get_feature_names_out()
    classes = pipeline.classes_

    n_classes = len(classes)
    ncols = min(3, n_classes)
    nrows = -(-n_classes // ncols)  # ceiling division
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(7.0, 3.2 * nrows), squeeze=False)
    axes = list(axes_grid.flat)
    for spare in axes[n_classes:]:
        spare.axis("off")

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    for i, (cls, ax) in enumerate(zip(classes, axes)):
        coef = clf.coef_[i]
        top_idx = np.argsort(coef)[-top_n:]
        top_words = feature_names[top_idx]
        top_coef = coef[top_idx]

        ax.barh(range(top_n), top_coef, color=colors[i % len(colors)], alpha=0.8, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_words, fontsize=8)
        ax.set_title(cls, fontsize=10, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=9)

    plt.suptitle(
        "Top features by role: logistic regression coefficients", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = cfg.figures_dir / "ch09_feature_importance.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------


def _f1_band(f1: float) -> str:
    """Plain-English reading of macro F1, matching the scale in the chapter README."""
    if f1 >= 0.95:
        return "suspiciously high on real text — check for label leakage first"
    if f1 >= 0.90:
        return "excellent"
    if f1 >= 0.80:
        return "good — usable with confidence thresholding"
    if f1 >= 0.70:
        return "acceptable — more data or better features would help"
    return "investigate class imbalance, label noise, or data volume"


def write_model_evaluation(
    comparison: dict[str, dict],
    eval_results: dict,
    cfg: Config,
) -> Path:
    """Write a Markdown model evaluation report."""
    report_dict = eval_results["report"]
    best_model = select_model(comparison)

    lines = [
        "# TalentLens Role Classifier — Model Evaluation Report",
        "",
        "## Model Selection (5-fold cross-validation)",
        "",
        "| Model | F1 (macro) | Std |",
        "|-------|-----------|-----|",
    ]
    for name, stats in sorted(comparison.items(), key=lambda x: -x[1]["mean_f1"]):
        marker = " ← selected" if name == best_model else ""
        lines.append(f"| {name}{marker} | {stats['mean_f1']:.3f} | ±{stats['std_f1']:.3f} |")

    lines += [
        "",
        f"**Selected model:** {best_model} (simplest model within one standard deviation of the top CV score)",
        "",
        "## Holdout Evaluation",
        "",
        f"**Overall F1 (macro):** {eval_results['macro_f1']:.3f}",
        "",
        "| Role | Precision | Recall | F1 | Support |",
        "|------|-----------|--------|-----|---------|",
    ]
    for role in cfg.role_labels:
        if role in report_dict:
            r = report_dict[role]
            lines.append(
                f"| {role} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- Overall F1 of {eval_results['macro_f1']:.3f} — {_f1_band(eval_results['macro_f1'])}",
        "- See confusion matrix for class-level error patterns.",
        "- See feature importance chart for which words drive each prediction.",
        "",
        "## Model location",
        f"Saved to: `{cfg.report_model_relpath}`",
        "",
        "## Usage",
        "```python",
        "import joblib",
        f"pipeline = joblib.load('{cfg.report_model_relpath}')",
        "role, confidence = predict_role({'title': 'ML Engineer', 'description': '...'}, pipeline)",
        "```",
    ]

    out = cfg.reports_dir / "model_evaluation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_block(title: str, lines: list[str]) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title}\n{sep}")
    for line in lines:
        logger.info(f"  {line}")


def _ensure_dirs(cfg: Config) -> None:
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = Config()
    _ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("  CHAPTER 9: SUPERVISED LEARNING")
    logger.info("  TalentLens Role Classifier")
    logger.info("=" * 60)

    # 1. Load and prepare data
    logger.info("\n[1/6] Loading data and assigning labels...")
    df = load_data(cfg)

    X = df["feature_text"]
    y = df["role_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 2. Model comparison
    logger.info("\n[2/6] Comparing models (5-fold CV)...")
    _print_block("MODEL COMPARISON", ["  Model                     F1 ± Std     Time"])
    best_pipeline, comparison = compare_models(X_train, y_train, cfg)

    # 3. Evaluate on holdout
    logger.info("\n[3/6] Evaluating best model on holdout set...")
    eval_results = evaluate_model(best_pipeline, X_test, y_test, cfg)

    # 4. Save model
    logger.info(f"\n[4/6] Saving model to {cfg.model_path}...")
    joblib.dump(best_pipeline, cfg.model_path)
    size_mb = cfg.model_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Model saved ({size_mb:.1f}MB)")

    # 5. Visualisations
    logger.info("\n[5/6] Generating visualisations...")
    plot_model_comparison(comparison, cfg)
    plot_confusion_matrix(eval_results, cfg)
    plot_feature_importance(best_pipeline, cfg)

    # 6. Report
    logger.info("\n[6/6] Writing evaluation report...")
    write_model_evaluation(comparison, eval_results, cfg)

    # Quick inference demo
    demo_job = {
        "title": "Senior NLP Engineer",  # not used by the classifier; kept for the log line
        "skills_normalised": "Python|PyTorch|NLP|RAG|FastAPI|transformers",
        "description": "Build RAG pipelines and LLM applications. Production ML experience required.",
    }
    role, conf = predict_role(demo_job, best_pipeline)
    logger.info(f"\n  Demo prediction: '{demo_job['title']}' → {role} (confidence: {conf:.1%})")

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 9 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Model:   {cfg.model_path}")
    logger.info(f"  F1:      {eval_results['macro_f1']:.3f}")
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info("\nNext: Chapter 10 — Feature Engineering")
    logger.info("The classifier becomes the Ch19 FastAPI endpoint and Ch22 PyPI library.")


if __name__ == "__main__":
    main()
