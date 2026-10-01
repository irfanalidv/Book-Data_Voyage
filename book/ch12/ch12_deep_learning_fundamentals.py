"""
Chapter 12: Neural Networks - When Depth Pays for Itself
Data Voyage -- Building TalentLens

Four measured experiments, each answering one question:

1. Digits (classification)   - does a small MLP beat logistic regression on pixels?
2. California Housing (regression) - how much does input scaling matter to a network?
3. Diabetes (regression, 442 rows) - what does overfitting look like, and does
   early stopping rescue a network trained on too little data?
4. TalentLens role classifier - does an MLP beat Chapter 9's TF-IDF + logistic
   regression baseline on the bundled corpus?

Everything runs on CPU in about a minute. Digits and Diabetes ship with
scikit-learn; California Housing is downloaded once (about 400 KB) and cached.
Without network access that experiment is skipped and the report says so.

Run: python book/ch12/ch12_deep_learning_fundamentals.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing, load_diabetes, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from talentlens.features import scrub_role_phrases_from_text
from talentlens.paths import jobs_clean_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)


@dataclass
class Config:
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"
    clean_path: Path = field(default_factory=jobs_clean_path)
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Fix every RNG the chapter touches so reports are byte-stable across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_mlp(
    n_inputs: int,
    n_outputs: int,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a feedforward network: Linear -> ReLU (-> Dropout) per hidden layer.

    Args:
        n_inputs: Number of input features.
        n_outputs: Number of outputs (classes for classification, 1 for regression).
        hidden: Width of each hidden layer, in order.
        dropout: Dropout probability after each hidden activation (0 disables it).

    Returns:
        An untrained ``nn.Sequential``. The last layer has no activation - the
        loss function (cross-entropy or MSE) expects raw outputs.
    """
    layers: list[nn.Module] = []
    width_in = n_inputs
    for width in hidden:
        layers.append(nn.Linear(width_in, width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width_in = width
    layers.append(nn.Linear(width_in, n_outputs))
    return nn.Sequential(*layers)


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    stopped_early: bool


def train_network(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
    cfg: Config,
    max_epochs: int | None = None,
    early_stopping: bool = True,
    weight_decay: float = 0.0,
) -> TrainResult:
    """Train with Adam and mini-batches; optionally stop when validation loss stalls.

    Early stopping keeps a copy of the weights from the best validation epoch and
    restores them at the end - the model you get back is the best one seen, not
    the last one trained.

    Args:
        model: Network from :func:`build_mlp`.
        X_train, y_train: Training arrays (float features; int labels or float targets).
        X_val, y_val: Validation arrays, never used for gradient updates.
        task: ``"classification"`` (cross-entropy) or ``"regression"`` (MSE).
        cfg: Config with learning rate, batch size, patience.
        max_epochs: Override for ``cfg.max_epochs``.
        early_stopping: Stop after ``cfg.patience`` epochs without improvement.
        weight_decay: L2 penalty passed to Adam.

    Returns:
        :class:`TrainResult` with per-epoch train and validation losses.
    """
    epochs = max_epochs or cfg.max_epochs
    if task == "classification":
        loss_fn: nn.Module = nn.CrossEntropyLoss()
        yt = torch.tensor(y_train, dtype=torch.long)
        yv = torch.tensor(y_val, dtype=torch.long)
    elif task == "regression":
        loss_fn = nn.MSELoss()
        yt = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        yv = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    else:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(cfg.random_state)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=generator)
        running = 0.0
        for start in range(0, len(Xt), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            running += loss.item() * len(idx)
        train_losses.append(running / len(Xt))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif early_stopping and epoch - best_epoch >= cfg.patience:
            stopped_early = True
            break

    if early_stopping:
        model.load_state_dict(best_state)
    return TrainResult(model, train_losses, val_losses, best_epoch, stopped_early)


def predict(model: nn.Module, X: np.ndarray, task: str) -> np.ndarray:
    """Run inference in eval mode; argmax for classification, raw values for regression."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    if task == "classification":
        return out.argmax(dim=1).numpy()
    return out.numpy().ravel()


def _split(X, y, cfg: Config, stratify: bool):
    """Train / validation / test split. Validation is carved out of the training share."""
    strat = y if stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat
    )
    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=cfg.val_size, random_state=cfg.random_state, stratify=strat_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Experiment 1 - Digits
# ---------------------------------------------------------------------------


def run_digits(cfg: Config) -> dict:
    """Logistic regression vs a two-layer MLP on 8x8 handwritten digits."""
    seed_everything(cfg.random_state)
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel intensities are 0..16; scale to 0..1
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, cfg, stratify=True)

    logreg = LogisticRegression(max_iter=2000, random_state=cfg.random_state)
    logreg.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    acc_lr = accuracy_score(y_test, logreg.predict(X_test))

    mlp = build_mlp(X.shape[1], 10, hidden=(128, 64), dropout=0.2)
    res = train_network(mlp, X_train, y_train, X_val, y_val, task="classification", cfg=cfg)
    acc_mlp = accuracy_score(y_test, predict(res.model, X_test, "classification"))

    n_params = sum(p.numel() for p in mlp.parameters())
    logger.info(
        f"Digits: logistic regression {acc_lr:.3f} | MLP {acc_mlp:.3f} "
        f"(best epoch {res.best_epoch + 1}, {n_params:,} parameters)"
    )
    return {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "acc_logreg": acc_lr,
        "acc_mlp": acc_mlp,
        "best_epoch": res.best_epoch + 1,
        "epochs_run": len(res.train_losses),
        "n_params": n_params,
        "train_losses": res.train_losses,
        "val_losses": res.val_losses,
    }


# ---------------------------------------------------------------------------
# Experiment 2 - California Housing
# ---------------------------------------------------------------------------


def run_housing(cfg: Config) -> dict | None:
    """Linear regression vs MLP, with and without standardised inputs."""
    seed_everything(cfg.random_state)
    try:
        data = fetch_california_housing()
    except Exception as exc:  # network unavailable on first run
        logger.warning(f"California Housing unavailable ({exc}); skipping experiment 2")
        return None
    X, y = data.data, data.target  # target: median house value in $100k
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, cfg, stratify=False)

    lin = LinearRegression().fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    pred_lin = lin.predict(X_test)

    results: dict = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "linear": {"rmse": _rmse(y_test, pred_lin), "r2": r2_score(y_test, pred_lin)},
    }

    # Unscaled: raw features span 0.5 (income) to 35,000 (population).
    seed_everything(cfg.random_state)
    mlp_raw = build_mlp(X.shape[1], 1, hidden=(64, 32))
    res_raw = train_network(
        mlp_raw, X_train, y_train, X_val, y_val, task="regression", cfg=cfg, max_epochs=100
    )
    pred_raw = predict(res_raw.model, X_test, "regression")
    results["mlp_unscaled"] = {
        "rmse": _rmse(y_test, pred_raw),
        "r2": r2_score(y_test, pred_raw),
        "val_losses": res_raw.val_losses,
    }

    # Scaled: fit the scaler on training rows only - no test-set statistics leak in.
    scaler = StandardScaler().fit(X_train)
    seed_everything(cfg.random_state)
    mlp = build_mlp(X.shape[1], 1, hidden=(64, 32))
    res = train_network(
        mlp,
        scaler.transform(X_train),
        y_train,
        scaler.transform(X_val),
        y_val,
        task="regression",
        cfg=cfg,
        max_epochs=100,
    )
    pred_mlp = predict(res.model, scaler.transform(X_test), "regression")
    results["mlp_scaled"] = {
        "rmse": _rmse(y_test, pred_mlp),
        "r2": r2_score(y_test, pred_mlp),
        "val_losses": res.val_losses,
        "best_epoch": res.best_epoch + 1,
    }
    results["y_test"] = y_test
    results["pred_mlp"] = pred_mlp

    logger.info(
        f"Housing RMSE ($100k): linear {results['linear']['rmse']:.3f} | "
        f"MLP unscaled {results['mlp_unscaled']['rmse']:.3f} | "
        f"MLP scaled {results['mlp_scaled']['rmse']:.3f}"
    )
    return results


# ---------------------------------------------------------------------------
# Experiment 3 - Diabetes (small data)
# ---------------------------------------------------------------------------


def run_diabetes(cfg: Config) -> dict:
    """Show overfitting on 442 rows, then what early stopping and weight decay recover."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, cfg, stratify=False)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    lin = LinearRegression().fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    pred_lin = lin.predict(X_test)

    # Oversized network, no regularisation, trained for a fixed 500 epochs.
    seed_everything(cfg.random_state)
    big = build_mlp(X.shape[1], 1, hidden=(256, 256))
    res_big = train_network(
        big,
        X_train,
        y_train,
        X_val,
        y_val,
        task="regression",
        cfg=cfg,
        max_epochs=500,
        early_stopping=False,
    )
    pred_big = predict(res_big.model, X_test, "regression")

    # Same architecture, with early stopping and weight decay.
    seed_everything(cfg.random_state)
    reg = build_mlp(X.shape[1], 1, hidden=(256, 256), dropout=0.2)
    res_reg = train_network(
        reg,
        X_train,
        y_train,
        X_val,
        y_val,
        task="regression",
        cfg=cfg,
        max_epochs=500,
        early_stopping=True,
        weight_decay=1e-3,
    )
    pred_reg = predict(res_reg.model, X_test, "regression")

    min_val_epoch = int(np.argmin(res_big.val_losses)) + 1
    results = {
        "n_rows": len(X),
        "n_train": len(X_train),
        "linear": {"rmse": _rmse(y_test, pred_lin), "r2": r2_score(y_test, pred_lin)},
        "mlp_overfit": {
            "rmse": _rmse(y_test, pred_big),
            "r2": r2_score(y_test, pred_big),
            "train_losses": res_big.train_losses,
            "val_losses": res_big.val_losses,
            "min_val_epoch": min_val_epoch,
            "final_train_loss": res_big.train_losses[-1],
            "final_val_loss": res_big.val_losses[-1],
            "min_val_loss": min(res_big.val_losses),
        },
        "mlp_early_stop": {
            "rmse": _rmse(y_test, pred_reg),
            "r2": r2_score(y_test, pred_reg),
            "best_epoch": res_reg.best_epoch + 1,
            "epochs_run": len(res_reg.train_losses),
        },
    }
    logger.info(
        f"Diabetes test RMSE: linear {results['linear']['rmse']:.1f} | "
        f"MLP 500 epochs {results['mlp_overfit']['rmse']:.1f} | "
        f"MLP early-stopped {results['mlp_early_stop']['rmse']:.1f}"
    )
    return results


# ---------------------------------------------------------------------------
# Experiment 4 - TalentLens role classifier
# ---------------------------------------------------------------------------


def _talentlens_text(df: pd.DataFrame) -> pd.Series:
    """Same feature text as Chapter 9: scrubbed description plus skills, never title."""
    desc = df["description"].fillna("").astype(str).str[:800].map(scrub_role_phrases_from_text)
    skills = df.get("skills_normalised", pd.Series([""] * len(df), index=df.index))
    skills = skills.fillna("").astype(str).str.replace("|", " ", regex=False)
    return skills + " " + skills + " " + desc


def run_talentlens(cfg: Config) -> dict | None:
    """MLP on TF-IDF features vs Chapter 9's logistic regression, same split."""
    if not cfg.clean_path.exists():
        logger.warning(f"{cfg.clean_path} not found; skipping experiment 4")
        return None
    df = pd.read_csv(cfg.clean_path)
    df = df[
        df["role_category"].isin(
            ["AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer", "Data Analyst"]
        )
    ]
    text = _talentlens_text(df)
    labels = df["role_category"].astype("category")
    classes = list(labels.cat.categories)
    y = labels.cat.codes.to_numpy()

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        text, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    vec = TfidfVectorizer(
        max_features=5_000, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
    )
    X_train = vec.fit_transform(X_train_txt).toarray()
    X_test = vec.transform(X_test_txt).toarray()

    logreg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=cfg.random_state
    )
    logreg.fit(X_train, y_train)
    f1_lr = f1_score(y_test, logreg.predict(X_test), average="macro")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_train
    )
    seed_everything(cfg.random_state)
    mlp = build_mlp(X_train.shape[1], len(classes), hidden=(64,), dropout=0.3)
    res = train_network(mlp, X_tr, y_tr, X_val, y_val, task="classification", cfg=cfg)
    f1_mlp = f1_score(y_test, predict(res.model, X_test, "classification"), average="macro")

    logger.info(f"TalentLens macro F1: logistic regression {f1_lr:.3f} | MLP {f1_mlp:.3f}")
    return {
        "n_rows": len(df),
        "n_features": X_train.shape[1],
        "n_classes": len(classes),
        "f1_logreg": f1_lr,
        "f1_mlp": f1_mlp,
        "best_epoch": res.best_epoch + 1,
    }


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------


def plot_digits_curves(digits: dict, cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(digits["train_losses"]) + 1)
    ax.plot(epochs, digits["train_losses"], label="Training loss", color="#2196F3")
    ax.plot(epochs, digits["val_losses"], label="Validation loss", color="#FF9800")
    ax.axvline(
        digits["best_epoch"],
        color="#9E9E9E",
        ls="--",
        lw=1,
        label=f"Best validation epoch ({digits['best_epoch']})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(
        f"Digits MLP — test accuracy {digits['acc_mlp']:.1%} "
        f"(logistic regression {digits['acc_logreg']:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    out = cfg.figures_dir / "ch12_digits_training_curves.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    return out


def plot_housing(housing: dict, cfg: Config) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["Linear\nregression", "MLP\n(raw inputs)", "MLP\n(standardised)"]
    rmses = [
        housing["linear"]["rmse"],
        housing["mlp_unscaled"]["rmse"],
        housing["mlp_scaled"]["rmse"],
    ]
    bars = axes[0].bar(names, rmses, color=["#9E9E9E", "#F44336", "#4CAF50"], width=0.55)
    for bar, val in zip(bars, rmses):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    axes[0].set_ylabel("Test RMSE ($100k units, lower is better)")
    axes[0].set_title("Same network, different input scaling", fontweight="bold")

    y_test, pred = housing["y_test"], housing["pred_mlp"]
    axes[1].scatter(y_test, pred, s=4, alpha=0.3, color="#2196F3")
    lims = [0, max(y_test.max(), pred.max())]
    axes[1].plot(lims, lims, color="#F44336", lw=1, ls="--", label="Perfect prediction")
    axes[1].set_xlabel("Actual median house value ($100k)")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Standardised MLP — R² {housing['mlp_scaled']['r2']:.2f}", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    out = cfg.figures_dir / "ch12_housing_scaling.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    return out


def plot_diabetes(diabetes: dict, cfg: Config) -> Path:
    over = diabetes["mlp_overfit"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(over["train_losses"]) + 1)
    ax.plot(epochs, over["train_losses"], label="Training loss", color="#2196F3")
    ax.plot(epochs, over["val_losses"], label="Validation loss", color="#FF9800")
    ax.axvline(
        over["min_val_epoch"],
        color="#9E9E9E",
        ls="--",
        lw=1,
        label=f"Validation minimum (epoch {over['min_val_epoch']})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(
        f"Diabetes — {diabetes['n_train']} training rows, 68k-parameter network",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    out = cfg.figures_dir / "ch12_diabetes_overfitting.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    return out


def write_report(
    digits: dict, housing: dict | None, diabetes: dict, talent: dict | None, cfg: Config
) -> Path:
    lines = [
        "# Chapter 12 — Neural Network Report",
        "",
        "*Generated by `ch12_deep_learning_fundamentals.py`. Seeds are fixed; "
        "numbers reproduce on the same library versions.*",
        "",
        "## 1. Digits (1,797 images, 64 pixels, 10 classes)",
        "",
        "| Model | Test accuracy |",
        "|---|---|",
        f"| Logistic regression | {digits['acc_logreg']:.3f} |",
        f"| MLP 64→128→64→10, dropout 0.2 | {digits['acc_mlp']:.3f} |",
        "",
        f"Best validation epoch: {digits['best_epoch']} of {digits['epochs_run']} run "
        f"({digits['n_params']:,} parameters).",
        "",
    ]
    if housing is None:
        lines += [
            "## 2. California Housing",
            "",
            "Skipped — dataset could not be downloaded on this run.",
            "",
        ]
    else:
        lines += [
            f"## 2. California Housing ({housing['n_rows']:,} rows, 8 features)",
            "",
            "| Model | Test RMSE ($100k) | Test R² |",
            "|---|---|---|",
            f"| Linear regression | {housing['linear']['rmse']:.3f} | "
            f"{housing['linear']['r2']:.3f} |",
            f"| MLP, raw inputs | {housing['mlp_unscaled']['rmse']:.3f} | "
            f"{housing['mlp_unscaled']['r2']:.3f} |",
            f"| MLP, standardised inputs | {housing['mlp_scaled']['rmse']:.3f} | "
            f"{housing['mlp_scaled']['r2']:.3f} |",
            "",
        ]
    over, early = diabetes["mlp_overfit"], diabetes["mlp_early_stop"]
    lines += [
        f"## 3. Diabetes ({diabetes['n_rows']} rows; {diabetes['n_train']} used for training)",
        "",
        "| Model | Test RMSE | Test R² |",
        "|---|---|---|",
        f"| Linear regression | {diabetes['linear']['rmse']:.1f} | "
        f"{diabetes['linear']['r2']:.3f} |",
        f"| MLP 256→256, 500 epochs, no regularisation | {over['rmse']:.1f} | {over['r2']:.3f} |",
        f"| Same MLP, early stopping + dropout + weight decay | {early['rmse']:.1f} | "
        f"{early['r2']:.3f} |",
        "",
        f"Overfit run: validation loss bottomed at epoch {over['min_val_epoch']} "
        f"(MSE {over['min_val_loss']:,.0f}); after 500 epochs training MSE was "
        f"{over['final_train_loss']:,.0f} and validation MSE {over['final_val_loss']:,.0f}.",
        f"Early-stopped run kept epoch {early['best_epoch']} "
        f"(stopped after {early['epochs_run']}).",
        "",
    ]
    if talent is None:
        lines += ["## 4. TalentLens role classifier", "", "Skipped — no jobs_clean.csv.", ""]
    else:
        lines += [
            f"## 4. TalentLens role classifier ({talent['n_rows']} postings, "
            f"{talent['n_features']:,} TF-IDF features, {talent['n_classes']} roles)",
            "",
            "| Model | Test macro F1 |",
            "|---|---|",
            f"| Logistic regression (Chapter 9 baseline) | {talent['f1_logreg']:.3f} |",
            f"| MLP 1 hidden layer (64), dropout 0.3 | {talent['f1_mlp']:.3f} |",
            "",
        ]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / "neural_network_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("  CHAPTER 12: NEURAL NETWORKS")
    logger.info("=" * 60)

    digits = run_digits(cfg)
    housing = run_housing(cfg)
    diabetes = run_diabetes(cfg)
    talent = run_talentlens(cfg)

    plot_digits_curves(digits, cfg)
    if housing is not None:
        plot_housing(housing, cfg)
    plot_diabetes(diabetes, cfg)
    report = write_report(digits, housing, diabetes, talent, cfg)

    logger.info(f"Report: {report}")
    logger.info(f"Figures: {cfg.figures_dir}")
    logger.info("Next: python book/ch13/ch13_skill_extraction.py")


if __name__ == "__main__":
    main()
