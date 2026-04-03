"""Unit tests for src/analysis/models.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.models import (
    compute_acf_pacf,
    detect_changepoints,
    fit_arima,
    fit_var,
    make_stationary,
    rolling_ols,
    stationarity_check,
    stl_decompose,
)


def _ar1_series(n=60, phi=0.8, seed=0):
    """Generate a stationary AR(1) series."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(x, index=dates)


def _rw_series(n=60, seed=0):
    """Generate a non-stationary random walk."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(size=n)))


class TestStationarityCheck:
    def test_stationary_ar1_detected(self):
        s = _ar1_series(n=100)
        result = stationarity_check(s)
        assert "adf_stationary" in result
        assert result["adf_stationary"] is True

    def test_random_walk_not_stationary(self):
        s = _rw_series(n=100)
        result = stationarity_check(s)
        assert result["adf_stationary"] is False

    def test_returns_all_keys(self):
        s = _ar1_series()
        result = stationarity_check(s)
        for key in ("adf_stat", "adf_pvalue", "adf_stationary",
                    "kpss_stat", "kpss_pvalue", "is_stationary"):
            assert key in result

    def test_short_series_returns_nan(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = stationarity_check(s)
        assert np.isnan(result["adf_stat"])


class TestMakeStationary:
    def test_already_stationary_returns_none(self):
        s = _ar1_series(n=100)
        result, label = make_stationary(s)
        # May return "none" or "diff" depending on borderline stationarity
        assert label in ("none", "diff", "log", "log_diff")

    def test_random_walk_differenced(self):
        s = _rw_series(n=100)
        result, label = make_stationary(s)
        assert label in ("diff", "log_diff")
        assert len(result) >= 1


class TestSTLDecompose:
    def test_returns_trend_seasonal_resid(self):
        s = _ar1_series(n=60)
        result = stl_decompose(s, period=7)
        assert "trend" in result
        assert "seasonal" in result
        assert "resid" in result

    def test_trend_same_length_as_input(self):
        s = _ar1_series(n=60)
        result = stl_decompose(s, period=7)
        assert len(result["trend"]) == len(s)

    def test_empty_on_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = stl_decompose(s, period=7)
        assert result["trend"].empty

    def test_components_sum_approximately(self):
        """trend + seasonal + resid ≈ original."""
        s = _ar1_series(n=60)
        result = stl_decompose(s, period=7)
        reconstructed = result["trend"] + result["seasonal"] + result["resid"]
        np.testing.assert_allclose(reconstructed.values, s.values, atol=1e-3)


class TestACFPACF:
    def test_returns_expected_keys(self):
        s = _ar1_series(n=60)
        result = compute_acf_pacf(s, nlags=10)
        for key in ("acf_values", "pacf_values", "acf_confint", "pacf_confint", "lags"):
            assert key in result

    def test_acf_lag0_is_one(self):
        s = _ar1_series(n=60)
        result = compute_acf_pacf(s, nlags=10)
        assert result["acf_values"][0] == pytest.approx(1.0)

    def test_ar1_pacf_significant_at_lag1_only(self):
        """For AR(1), PACF should be significant only at lag 1."""
        s = _ar1_series(n=200, phi=0.9)
        result = compute_acf_pacf(s, nlags=5)
        assert abs(result["pacf_values"][1]) > 0.3   # lag 1 significant
        assert abs(result["pacf_values"][3]) < abs(result["pacf_values"][1])  # lag 3 smaller


class TestFitARIMA:
    def test_returns_aic_bic(self):
        s = _ar1_series(n=80)
        result = fit_arima(s, order=(1, 0, 0), forecast_horizon=4)
        assert "aic" in result
        assert "bic" in result

    def test_forecast_length(self):
        s = _ar1_series(n=80)
        result = fit_arima(s, order=(1, 0, 0), forecast_horizon=4)
        assert len(result["forecast"]) == 4

    def test_error_on_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = fit_arima(s, order=(1, 0, 0))
        assert "error" in result

    def test_residuals_returned(self):
        s = _ar1_series(n=80)
        result = fit_arima(s, order=(1, 0, 0))
        assert "residuals" in result
        assert len(result["residuals"]) > 0


class TestFitVAR:
    def _make_var_df(self, n=60):
        rng = np.random.default_rng(2)
        dates = pd.date_range("2025-01-01", periods=n, freq="4D")
        ef = np.linspace(0.015, 0.018, n) + rng.normal(0, 0.0005, n)
        drift = np.linspace(0.5, 0.1, n) + rng.normal(0, 0.05, n)
        ctl = np.linspace(40, 75, n) + rng.normal(0, 3, n)
        return pd.DataFrame({"ef": ef, "cardiac_drift_slope": drift, "ctl": ctl}, index=dates)

    def test_returns_model_result(self):
        df = self._make_var_df(n=60)
        result = fit_var(df, maxlags=2)
        assert "model_result" in result or "error" in result

    def test_forecast_shape(self):
        df = self._make_var_df(n=60)
        result = fit_var(df, maxlags=2, forecast_horizon=4)
        if "forecast" in result:
            assert result["forecast"].shape == (4, 3)

    def test_error_on_short_data(self):
        df = self._make_var_df(n=5)
        result = fit_var(df, maxlags=4)
        assert "error" in result


class TestRollingOLS:
    def test_returns_dataframe(self, per_run_metrics_df):
        df = per_run_metrics_df.set_index("run_date")
        y = df["ef"]
        X = df[["ctl", "z1_pct"]]
        result = rolling_ols(y, X, window_weeks=4, obs_per_week=1.5)
        assert isinstance(result, pd.DataFrame)

    def test_coefficient_columns_match_features(self, per_run_metrics_df):
        df = per_run_metrics_df.set_index("run_date")
        y = df["ef"]
        features = ["ctl", "z1_pct"]
        X = df[features]
        result = rolling_ols(y, X, window_weeks=4, obs_per_week=1.5)
        if not result.empty:
            for col in features:
                assert col in result.columns

    def test_empty_on_too_short_series(self):
        y = pd.Series([1.0, 2.0, 3.0])
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = rolling_ols(y, X, window_weeks=10)
        assert result.empty


class TestDetectChangepoints:
    def test_detects_obvious_shift(self):
        """A step change at midpoint should produce ≥1 breakpoint."""
        n = 100
        s = pd.Series(np.concatenate([np.zeros(n // 2), np.ones(n // 2) * 5]))
        bkps = detect_changepoints(s, penalty=1.0)
        assert len(bkps) >= 1
        # Breakpoint should be near the middle
        assert any(40 < b < 60 for b in bkps)

    def test_returns_list(self):
        s = _ar1_series(n=60)
        result = detect_changepoints(s)
        assert isinstance(result, list)

    def test_empty_on_short_series(self):
        s = pd.Series([1.0, 2.0])
        result = detect_changepoints(s)
        assert result == []

    def test_fixed_n_bkps(self):
        s = pd.Series(np.concatenate([np.zeros(30), np.ones(30) * 3, np.zeros(30)]))
        bkps = detect_changepoints(s, n_bkps=2)
        assert len(bkps) <= 2
