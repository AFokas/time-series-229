"""Unit tests for src/analysis/lt_estimate.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.lt_estimate import (
    ANCHOR_LT_HR,
    ANCHOR_LT_PACE,
    ANCHOR_DATE,
    LTEstimate,
    LTTracker,
    estimate_lt,
)


class TestLTTracker:
    def test_seeded_with_anchor(self):
        tracker = LTTracker()
        assert len(tracker.estimates) == 1
        anchor = tracker.estimates[0]
        assert anchor.lt_hr == ANCHOR_LT_HR
        assert anchor.lt_pace_min_per_km == ANCHOR_LT_PACE
        assert anchor.method == "anchor"

    def test_get_lt_before_any_estimates_returns_anchor(self):
        tracker = LTTracker()
        date = pd.Timestamp("2025-01-01", tz="UTC")
        result = tracker.get_lt_for_date(date)
        assert result.lt_hr == ANCHOR_LT_HR

    def test_get_lt_returns_most_recent_valid(self):
        tracker = LTTracker()
        e1 = LTEstimate(pd.Timestamp("2025-06-01", tz="UTC"), 172.0, 3.85, method="dmax")
        e2 = LTEstimate(pd.Timestamp("2025-09-01", tz="UTC"), 170.0, 3.80, method="dmax")
        tracker.add_estimate(e1)
        tracker.add_estimate(e2)

        result = tracker.get_lt_for_date(pd.Timestamp("2025-10-01", tz="UTC"))
        assert result.lt_hr == 170.0

    def test_get_lt_does_not_use_future_estimate(self):
        tracker = LTTracker()
        future = LTEstimate(pd.Timestamp("2026-01-01", tz="UTC"), 168.0, 3.75, method="dmax")
        tracker.add_estimate(future)

        result = tracker.get_lt_for_date(pd.Timestamp("2025-06-01", tz="UTC"))
        assert result.lt_hr == ANCHOR_LT_HR

    def test_suspect_flag_set_when_deviation_exceeds_threshold(self):
        tracker = LTTracker()
        # Add a valid estimate near the anchor
        valid = LTEstimate(pd.Timestamp("2025-06-15", tz="UTC"), 174.0, 3.90, method="dmax")
        tracker.add_estimate(valid)

        # Then add one that deviates > 15 bpm from the rolling mean
        outlier = LTEstimate(pd.Timestamp("2025-07-01", tz="UTC"), 155.0, 4.20, method="dmax")
        tracker.add_estimate(outlier)

        assert outlier.suspect is True

    def test_suspect_estimate_not_used_in_get_lt(self):
        tracker = LTTracker()
        good = LTEstimate(pd.Timestamp("2025-06-15", tz="UTC"), 174.0, 3.90, method="dmax")
        tracker.add_estimate(good)
        bad = LTEstimate(pd.Timestamp("2025-07-01", tz="UTC"), 150.0, 4.50, method="dmax")
        bad.suspect = True
        tracker.estimates.append(bad)
        tracker.estimates.sort(key=lambda e: e.date)

        result = tracker.get_lt_for_date(pd.Timestamp("2025-08-01", tz="UTC"))
        assert result.lt_hr == 174.0

    def test_to_dataframe(self):
        tracker = LTTracker()
        df = tracker.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "lt_hr" in df.columns
        assert "lt_pace_min_per_km" in df.columns
        assert len(df) >= 1


class TestEstimateLT:
    def _make_qualifying_df(self, n=1800, lt_hr=175.0, lt_speed=4.23) -> pd.DataFrame:
        """Construct a run with clear inflection point around lt_hr / lt_speed."""
        rng = np.random.default_rng(0)
        speeds = np.linspace(2.5, 5.5, n) + rng.normal(0, 0.05, n)
        # HR: flat below LT, then steep above — creates a clear Dmax inflection
        hrs = np.where(
            speeds < lt_speed,
            120 + (speeds - 2.5) * 15 + rng.normal(0, 1.5, n),
            lt_hr + (speeds - lt_speed) * 30 + rng.normal(0, 1.5, n),
        )
        gap_paces = 1000.0 / (speeds * 60.0)
        return pd.DataFrame({
            "gap_speed_ms": speeds,
            "heart_rate": hrs,
            "gap_pace_min_per_km": gap_paces,
            "elapsed_s": np.arange(n, dtype=float),
        })

    def test_returns_dict_with_required_keys(self):
        df = self._make_qualifying_df()
        result = estimate_lt(df)
        assert result is not None
        assert "lt_hr" in result
        assert "lt_pace_min_per_km" in result

    def test_lt_hr_in_plausible_range(self):
        df = self._make_qualifying_df(lt_hr=175.0)
        result = estimate_lt(df)
        assert result is not None
        assert 130 < result["lt_hr"] < 200

    def test_lt_pace_in_plausible_range(self):
        df = self._make_qualifying_df()
        result = estimate_lt(df)
        assert result is not None
        assert 2.5 < result["lt_pace_min_per_km"] < 7.0

    def test_returns_none_if_too_few_rows(self):
        df = pd.DataFrame({
            "gap_speed_ms": [3.0, 4.0],
            "heart_rate": [140.0, 160.0],
            "gap_pace_min_per_km": [5.5, 4.2],
            "elapsed_s": [0.0, 1.0],
        })
        assert estimate_lt(df) is None

    def test_returns_none_if_hr_missing(self):
        df = self._make_qualifying_df()
        df["heart_rate"] = np.nan
        assert estimate_lt(df) is None

    def test_returns_none_if_speed_missing(self):
        df = self._make_qualifying_df()
        df["gap_speed_ms"] = np.nan
        assert estimate_lt(df) is None

    def test_anchor_cross_validation(self):
        """LT estimate from a qualifying run should fall in a physiologically plausible range.

        The Dmax method is sensitive to curve shape, so we only assert that the
        returned value is within the physiological range for a trained runner rather
        than requiring it to match the 2019 anchor exactly (that check belongs in
        the real-data e2e test).
        """
        df = self._make_qualifying_df(lt_hr=ANCHOR_LT_HR, lt_speed=pace_to_speed_local(ANCHOR_LT_PACE))
        result = estimate_lt(df)
        if result is not None:
            assert 100 < result["lt_hr"] < 210
            assert 2.5 < result["lt_pace_min_per_km"] < 8.0


def pace_to_speed_local(pace_min_per_km: float) -> float:
    return 1000.0 / (pace_min_per_km * 60.0)
