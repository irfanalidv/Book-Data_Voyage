"""Tests for Chapter 12 - neural network building blocks."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from book.ch12.ch12_deep_learning_fundamentals import (  # noqa: E402
    Config,
    build_mlp,
    predict,
    seed_everything,
    train_network,
)


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.figures_dir = tmp_path / "figures"
    c.reports_dir = tmp_path
    c.max_epochs = 60
    c.patience = 10
    return c


def _blobs(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(-2, 1, size=(n // 2, 2))
    X1 = rng.normal(2, 1, size=(n // 2, 2))
    X = np.vstack([X0, X1]).astype("float32")
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


class TestBuildMlp:
    def test_output_shape(self):
        model = build_mlp(5, 3, hidden=(8, 4))
        out = model(torch.zeros(7, 5))
        assert out.shape == (7, 3)

    def test_dropout_layers_added(self):
        model = build_mlp(5, 1, hidden=(8, 4), dropout=0.5)
        assert sum(isinstance(m, torch.nn.Dropout) for m in model) == 2

    def test_no_activation_on_output(self):
        model = build_mlp(5, 2, hidden=(8,))
        assert isinstance(list(model)[-1], torch.nn.Linear)


class TestTrainNetwork:
    def test_classification_learns_separable_blobs(self, cfg):
        seed_everything(0)
        X, y = _blobs()
        model = build_mlp(2, 2, hidden=(8,))
        res = train_network(model, X[::2], y[::2], X[1::2], y[1::2], task="classification", cfg=cfg)
        acc = (predict(res.model, X[1::2], "classification") == y[1::2]).mean()
        assert acc > 0.95
        assert res.train_losses[-1] < res.train_losses[0]

    def test_regression_loss_decreases(self, cfg):
        seed_everything(0)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3)).astype("float32")
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 200)
        model = build_mlp(3, 1, hidden=(16,))
        res = train_network(model, X[:150], y[:150], X[150:], y[150:], task="regression", cfg=cfg)
        assert min(res.val_losses) < res.val_losses[0]

    def test_early_stopping_restores_best_epoch(self, cfg):
        seed_everything(0)
        X, y = _blobs()
        model = build_mlp(2, 2, hidden=(8,))
        res = train_network(model, X[::2], y[::2], X[1::2], y[1::2], task="classification", cfg=cfg)
        assert res.best_epoch == int(np.argmin(res.val_losses))

    def test_seeded_runs_are_identical(self, cfg):
        X, y = _blobs()
        losses = []
        for _ in range(2):
            seed_everything(42)
            model = build_mlp(2, 2, hidden=(8,))
            res = train_network(
                model,
                X[::2],
                y[::2],
                X[1::2],
                y[1::2],
                task="classification",
                cfg=cfg,
                max_epochs=5,
                early_stopping=False,
            )
            losses.append(res.train_losses)
        assert losses[0] == losses[1]

    def test_rejects_unknown_task(self, cfg):
        X, y = _blobs(20)
        with pytest.raises(ValueError):
            train_network(build_mlp(2, 2), X, y, X, y, task="ranking", cfg=cfg)
