"""Unit tests for src/pipeline/02_detect.py (run classification)."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline._02_detect import detect_intervals, is_long_run
from src.utils.zones import ZoneBoundaries

LT_HR = 175.0
LT_PACE = 3.95
ZONES = ZoneBoundaries.from_lt(LT_HR, LT_PACE)


class TestIsLongRun:
    def test_within_range(self):
        assert is_long_run(30.0) is True

    def test_at_min_boundary(self):
        assert is_long_run(25.0) is True

    def test_at_max_boundary(self):
        assert is_long_run(38.0) is True

    def test_below_min(self):
        assert is_long_run(24.9) is False

    def test_above_max(self):
        assert is_long_run(38.1) is False

    def test_custom_bounds(self):
        assert is_long_run(22.0, min_km=20.0, max_km=45.0) is True
        assert is_long_run(50.0, min_km=20.0, max_km=45.0) is False


class TestDetectIntervals:
    def test_detects_synthetic_intervals(self, interval_run_df):
        is_iv, intervals = detect_intervals(interval_run_df, ZONES)
        assert is_iv is True
        assert len(intervals) >= 3

    def test_interval_metadata_keys(self, interval_run_df):
        _, intervals = detect_intervals(interval_run_df, ZONES)
        if intervals:
            iv = intervals[0]
            for key in ("interval_idx", "interval_start_s", "interval_end_s",
                        "interval_duration_s", "interval_avg_gap_pace",
                        "interval_avg_hr", "recovery_duration_s"):
                assert key in iv, f"Missing key: {key}"

    def test_easy_run_not_interval(self, easy_run_df):
        is_iv, intervals = detect_intervals(easy_run_df, ZONES)
        assert is_iv is False

    def test_interval_indices_consecutive(self, interval_run_df):
        _, intervals = detect_intervals(interval_run_df, ZONES)
        for i, iv in enumerate(intervals, start=1):
            assert iv["interval_idx"] == i

    def test_interval_start_before_end(self, interval_run_df):
        _, intervals = detect_intervals(interval_run_df, ZONES)
        for iv in intervals:
            assert iv["interval_start_s"] < iv["interval_end_s"]

    def test_duration_matches_start_end(self, interval_run_df):
        _, intervals = detect_intervals(interval_run_df, ZONES)
        for iv in intervals:
            assert iv["interval_duration_s"] == pytest.approx(
                iv["interval_end_s"] - iv["interval_start_s"], abs=2.0
            )

    def test_recovery_duration_positive(self, interval_run_df):
        _, intervals = detect_intervals(interval_run_df, ZONES)
        for iv in intervals[:-1]:   # last has no recovery
            assert iv["recovery_duration_s"] > 0

    def test_missing_columns_returns_false(self):
        df = pd.DataFrame({"elapsed_s": [0.0, 1.0]})
        is_iv, intervals = detect_intervals(df, ZONES)
        assert is_iv is False
        assert intervals == []

    def test_fewer_than_3_intervals_returns_false(self):
        """A run with only 2 hard efforts should not be classified as interval."""
        from tests.fixtures.synthetic_data import make_run_df
        df = make_run_df(
            n_seconds=1800,
            base_speed_ms=3.0,
            interval_profile=[(300, 600, 1.6), (900, 1200, 1.6)],  # 2 intervals only
        )
        is_iv, intervals = detect_intervals(df, ZONES, min_intervals=3)
        assert is_iv is False

    def test_min_duration_filter(self, interval_run_df):
        """Setting min_duration_s very high should eliminate short intervals."""
        _, long_ivs = detect_intervals(interval_run_df, ZONES, min_duration_s=1000)
        # No 1000-second intervals exist in the synthetic session
        assert len(long_ivs) == 0

    def test_hard_fraction_threshold(self, easy_run_df):
        """Easy run with very short hard segments should not qualify."""
        is_iv, _ = detect_intervals(easy_run_df, ZONES, min_hard_fraction=0.15)
        assert is_iv is False
