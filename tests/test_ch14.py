"""Tests for Chapter 14: Time Series - Skill Trends.

Reference: book/ch14/README.md
Source: book/ch14/ch14_time_series_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


def _stationarity_dict(series: np.ndarray) -> dict:
    """Mirror the ADF/KPSS structure used in demonstrate_stationarity()."""
    adf = adfuller(series)
    kpss_result = kpss(series, regression="c", nlags="auto")
    return {
        "adf": {
            "test_statistic": float(adf[0]),
            "p_value": float(adf[1]),
            "critical_values": adf[4],
        },
        "kpss": {
            "test_statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": kpss_result[3],
        },
    }


def _forecast_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


class TestStationarity:
    def test_white_noise_adf_and_kpss_keys(self):
        rng = np.random.default_rng(14)
        series = rng.normal(0, 1, 500)
        result = _stationarity_dict(series)
        assert "test_statistic" in result["adf"]
        assert "p_value" in result["adf"]
        assert "critical_values" in result["adf"]
        assert "test_statistic" in result["kpss"]
        assert "p_value" in result["kpss"]

    def test_random_walk_more_nonstationary_than_white_noise(self):
        rng = np.random.default_rng(14)
        white = rng.normal(0, 1, 400)
        walk = np.cumsum(rng.normal(0, 1, 400))
        white_adf_p = _stationarity_dict(white)["adf"]["p_value"]
        walk_adf_p = _stationarity_dict(walk)["adf"]["p_value"]
        assert walk_adf_p > white_adf_p


class TestForecastingMetrics:
    def test_arima_metrics_positive_on_stationary_synthetic(self):
        rng = np.random.default_rng(14)
        n = 60
        series = pd.Series(100 + rng.normal(0, 2, n))
        train, test = series.iloc[:48], series.iloc[48:]
        model = ARIMA(train, order=(1, 0, 0)).fit()
        forecast = model.forecast(steps=len(test))
        metrics = _forecast_metrics(test, np.asarray(forecast))
        assert metrics["MAE"] > 0
        assert metrics["RMSE"] > 0
        assert metrics["MAPE"] > 0


class TestDecomposition:
    def test_additive_decomposition_on_long_monthly_series(self):
        rng = np.random.default_rng(14)
        n = 48
        idx = pd.date_range("2020-01-01", periods=n, freq="ME")
        trend = np.linspace(100, 160, n)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
        series = pd.Series(trend + seasonal + rng.normal(0, 1, n), index=idx)
        result = seasonal_decompose(series, model="additive", period=12)
        assert len(result.trend.dropna()) > 0
        assert len(result.seasonal) == n

    @pytest.mark.skip(
        reason=(
            "SCOPE.md: decomposition on the bundled COVID fetch is short/noisy; "
            "chapter uses network fallback. This test uses a long synthetic series instead."
        )
    )
    def test_covid_network_series_not_used_in_unit_tests(self):
        pass
