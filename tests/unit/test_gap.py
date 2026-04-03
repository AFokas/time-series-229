"""Unit tests for src/analysis/gap.py (Minetti GAP)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.gap import add_gap_columns, compute_gradient, minetti_gap


class TestMinettiGAP:
    def test_flat_ground_returns_original_speed(self):
        speed = pd.Series([3.0, 4.0, 5.0])
        gradient = pd.Series([0.0, 0.0, 0.0])
        result = minetti_gap(speed, gradient)
        np.testing.assert_allclose(result.values, speed.values, rtol=1e-6)

    def test_uphill_increases_effective_pace(self):
        """Uphill → more metabolic cost → GAP speed > raw speed."""
        speed = pd.Series([3.3])
        gradient = pd.Series([0.10])   # 10% uphill
        result = minetti_gap(speed, gradient)
        assert result.iloc[0] > speed.iloc[0], "Uphill should increase GAP speed (harder effort)"

    def test_downhill_decreases_effective_pace(self):
        """Moderate downhill → less metabolic cost → GAP speed < raw speed."""
        speed = pd.Series([3.3])
        gradient = pd.Series([-0.08])  # 8% downhill
        result = minetti_gap(speed, gradient)
        assert result.iloc[0] < speed.iloc[0], "Moderate downhill should decrease GAP speed"

    def test_known_value_10pct_uphill(self):
        """Verify against manually computed Minetti value at 10% gradient.

        C_slope(0.1) = 155.4*0.1^5 - 30.4*0.1^4 - 43.3*0.1^3 + 46.3*0.1^2 + 19.5*0.1 + 3.6
                     = 0.001554 - 0.0304 - 0.0433 + 0.463 + 1.95 + 3.6 ≈ 5.9409
        ratio = 5.9409 / 3.6 ≈ 1.6503
        GAP @ 3.30 m/s = 3.30 * 1.6503 ≈ 5.446 m/s
        """
        g = 0.1
        C = 155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6
        expected = 3.30 * C / 3.6

        result = minetti_gap(pd.Series([3.30]), pd.Series([g]))
        np.testing.assert_allclose(result.iloc[0], expected, rtol=1e-4)

    def test_output_clipped_to_valid_range(self):
        """GAP speed must be within [0.5, 10.0] m/s."""
        speed = pd.Series([0.01, 100.0])
        gradient = pd.Series([0.0, 0.0])
        result = minetti_gap(speed, gradient)
        assert (result >= 0.5).all()
        assert (result <= 10.0).all()

    def test_numpy_array_input(self):
        """Accepts numpy arrays as well as Series."""
        result = minetti_gap(np.array([3.3]), np.array([0.0]))
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result.iloc[0], 3.3, rtol=1e-6)

    def test_series_index_preserved(self):
        idx = pd.Index([10, 20, 30])
        speed = pd.Series([3.0, 3.3, 3.6], index=idx)
        gradient = pd.Series([0.0, 0.0, 0.0], index=idx)
        result = minetti_gap(speed, gradient)
        assert list(result.index) == list(idx)


class TestComputeGradient:
    def test_flat_returns_zero(self):
        alt = pd.Series([50.0, 50.0, 50.0])
        dist = pd.Series([0.0, 10.0, 20.0])
        grad = compute_gradient(alt, dist)
        np.testing.assert_allclose(grad.values[1:], 0.0, atol=1e-9)

    def test_10pct_uphill(self):
        alt = pd.Series([0.0, 10.0])
        dist = pd.Series([0.0, 100.0])
        grad = compute_gradient(alt, dist)
        np.testing.assert_allclose(grad.iloc[1], 0.10, rtol=1e-6)

    def test_clipped_to_45_pct(self):
        alt = pd.Series([0.0, 500.0])
        dist = pd.Series([0.0, 1.0])
        grad = compute_gradient(alt, dist)
        assert grad.iloc[1] == pytest.approx(0.45)

    def test_no_movement_returns_zero(self):
        """Zero delta_distance should return 0.0 gradient."""
        alt = pd.Series([50.0, 60.0])
        dist = pd.Series([0.0, 0.0])
        grad = compute_gradient(alt, dist)
        assert grad.iloc[1] == pytest.approx(0.0)


class TestAddGAPColumns:
    def test_adds_required_columns(self, easy_run_df):
        df = easy_run_df.copy()
        df = add_gap_columns(df)
        for col in ("gradient", "gap_speed_ms", "gap_pace_min_per_km"):
            assert col in df.columns, f"Missing column: {col}"

    def test_gap_pace_positive(self, easy_run_df):
        df = add_gap_columns(easy_run_df.copy())
        assert (df["gap_pace_min_per_km"].dropna() > 0).all()

    def test_gap_speed_within_range(self, easy_run_df):
        df = add_gap_columns(easy_run_df.copy())
        assert df["gap_speed_ms"].between(0.5, 10.0).all()
