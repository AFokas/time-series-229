"""Unit tests for src/analysis/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics import (
    aerobic_decoupling,
    cadence_pace_coupling,
    cardiac_drift_rate,
    compute_training_loads,
    efficiency_factor,
    hr_zone_distribution,
    interval_quality,
    pace_variability_index,
    recovery_rate,
    trimp,
)
from src.utils.zones import ZoneBoundaries


LT_HR = 175.0
LT_PACE = 3.95
ZONES = ZoneBoundaries.from_lt(LT_HR, LT_PACE)


class TestEfficiencyFactor:
    def test_basic_ef_computed(self, easy_run_df):
        result = efficiency_factor(easy_run_df, lt_hr=LT_HR)
        assert not np.isnan(result["ef"])
        # EF for easy run around 3.3 m/s / 140 bpm ≈ 0.0236
        assert 0.010 < result["ef"] < 0.040

    def test_all_halves_returned(self, easy_run_df):
        result = efficiency_factor(easy_run_df, lt_hr=LT_HR)
        for key in ("ef", "ef_first_half", "ef_second_half"):
            assert key in result

    def test_ef_nan_when_hr_above_ceiling(self, easy_run_df):
        df = easy_run_df.copy()
        df["heart_rate"] = LT_HR * 1.05   # above ef_zone_ceiling
        result = efficiency_factor(df, lt_hr=LT_HR, ef_zone_ceiling=0.95)
        assert np.isnan(result["ef"])

    def test_ef_nan_when_gap_missing(self):
        df = pd.DataFrame({"gap_speed_ms": [np.nan], "heart_rate": [140.0]})
        result = efficiency_factor(df, lt_hr=LT_HR)
        assert np.isnan(result["ef"])

    def test_ef_second_half_lower_on_drift_run(self):
        """When HR drifts upward, EF_second < EF_first."""
        n = 3600
        t = np.arange(n, dtype=float)
        speeds = np.full(n, 3.3)
        hrs = 130.0 + t * 0.02   # drift: +0.02 bpm/s
        df = pd.DataFrame({"gap_speed_ms": speeds, "heart_rate": hrs})
        result = efficiency_factor(df, lt_hr=LT_HR)
        assert result["ef_first_half"] > result["ef_second_half"]


class TestAerobicDecoupling:
    def test_positive_decoupling(self):
        result = aerobic_decoupling(0.020, 0.018)
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_zero_decoupling(self):
        assert aerobic_decoupling(0.020, 0.020) == pytest.approx(0.0)

    def test_nan_when_first_half_nan(self):
        assert np.isnan(aerobic_decoupling(np.nan, 0.018))

    def test_nan_when_first_half_zero(self):
        assert np.isnan(aerobic_decoupling(0.0, 0.018))


class TestCardiacDriftRate:
    def test_positive_slope_on_drifting_run(self):
        n = 1800
        t = np.arange(n, dtype=float)
        # Constant pace, rising HR
        df = pd.DataFrame({
            "gap_pace_min_per_km": np.full(n, 4.5),
            "heart_rate": 130.0 + t * 0.05,
            "elapsed_s": t,
        })
        result = cardiac_drift_rate(df, pace_tolerance=0.5)
        assert result["cardiac_drift_slope"] > 0

    def test_flat_slope_on_stable_run(self):
        n = 1800
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "gap_pace_min_per_km": np.full(n, 4.5),
            "heart_rate": 140.0 + rng.normal(0, 1.0, n),  # no trend
            "elapsed_s": np.arange(n, dtype=float),
        })
        result = cardiac_drift_rate(df, pace_tolerance=0.5)
        assert abs(result["cardiac_drift_slope"]) < 0.1   # near-flat

    def test_nan_when_insufficient_data(self):
        df = pd.DataFrame({
            "gap_pace_min_per_km": [4.5, 4.5],
            "heart_rate": [140.0, 141.0],
            "elapsed_s": [0.0, 1.0],
        })
        result = cardiac_drift_rate(df)
        assert np.isnan(result["cardiac_drift_slope"])

    def test_r2_between_0_and_1(self, easy_run_df):
        result = cardiac_drift_rate(easy_run_df)
        r2 = result["cardiac_drift_r2"]
        if not np.isnan(r2):
            assert 0.0 <= r2 <= 1.0


class TestRecoveryRate:
    def test_returns_required_keys(self):
        t = np.arange(120)
        hr = 180.0 * np.exp(-0.03 * t) + 60.0
        result = recovery_rate(180.0, hr)
        for key in ("hr_drop_60s", "hr_drop_90s", "decay_constant"):
            assert key in result

    def test_drop_60s_positive(self):
        t = np.arange(120)
        hr = 180.0 * np.exp(-0.03 * t) + 60.0
        result = recovery_rate(180.0, hr)
        assert result["hr_drop_60s"] > 0

    def test_drop_90s_geq_drop_60s(self):
        t = np.arange(120)
        hr = 180.0 * np.exp(-0.02 * t) + 60.0
        result = recovery_rate(180.0, hr)
        assert result["hr_drop_90s"] >= result["hr_drop_60s"]

    def test_nan_when_too_few_samples(self):
        result = recovery_rate(180.0, np.array([170.0, 165.0]))
        assert np.isnan(result["decay_constant"])

    def test_faster_decay_gives_larger_drop(self):
        t = np.arange(120)
        slow_hr = 180.0 * np.exp(-0.01 * t) + 60.0
        fast_hr = 180.0 * np.exp(-0.05 * t) + 60.0
        slow = recovery_rate(180.0, slow_hr)
        fast = recovery_rate(180.0, fast_hr)
        assert fast["hr_drop_60s"] > slow["hr_drop_60s"]


class TestTRIMP:
    def test_banister_known_value(self):
        """Hand-computed TRIMP for 1 hour at constant hr_ratio=0.5 (male):
        TRIMP = 3600 * (1/60) * 0.5 * 0.64 * exp(1.92 * 0.5)
              = 60 * 0.5 * 0.64 * exp(0.96)
              ≈ 60 * 0.5 * 0.64 * 2.6117 ≈ 50.14
        """
        hr_max, hr_rest = 200.0, 50.0
        hr_target = hr_rest + 0.5 * (hr_max - hr_rest)  # ratio 0.5 → 125 bpm
        hr_series = pd.Series(np.full(3600, hr_target))
        result = trimp(3600, hr_series, hr_max=hr_max, hr_rest=hr_rest)
        expected = 60 * 0.5 * 0.64 * np.exp(1.92 * 0.5)
        assert result == pytest.approx(expected, rel=0.01)

    def test_higher_hr_gives_higher_trimp(self):
        hr_max, hr_rest = 200.0, 50.0
        lo = trimp(1800, pd.Series(np.full(1800, 130.0)), hr_max, hr_rest)
        hi = trimp(1800, pd.Series(np.full(1800, 170.0)), hr_max, hr_rest)
        assert hi > lo

    def test_all_nan_hr_returns_zero(self):
        result = trimp(1800, pd.Series([np.nan, np.nan]), hr_max=200, hr_rest=50)
        assert result == 0.0

    def test_trimp_scales_with_duration(self):
        hr_max, hr_rest = 200.0, 50.0
        hr = pd.Series(np.full(1800, 150.0))
        t1 = trimp(1800, hr, hr_max, hr_rest)
        hr2 = pd.Series(np.full(3600, 150.0))
        t2 = trimp(3600, hr2, hr_max, hr_rest)
        assert t2 == pytest.approx(2 * t1, rel=0.01)


class TestComputeTrainingLoads:
    def test_returns_atl_ctl_tsb(self):
        dates = pd.date_range("2025-06-01", periods=60, freq="D")
        trimp_series = pd.Series(np.random.default_rng(0).uniform(50, 100, 60), index=dates)
        result = compute_training_loads(trimp_series, atl_days=7, ctl_days=42)
        for col in ("atl", "ctl", "tsb"):
            assert col in result.columns

    def test_ctl_increases_with_training(self):
        """CTL should converge upward when training load is consistently positive.

        Start from rest (first few days zero), then sustained load.
        """
        dates = pd.date_range("2025-06-01", periods=90, freq="D")
        trimp_values = np.concatenate([np.zeros(5), np.full(85, 80.0)])
        trimp_series = pd.Series(trimp_values, index=dates)
        result = compute_training_loads(trimp_series)
        # CTL at day 90 > CTL at day 6 (after ramp-up starts)
        assert result["ctl"].iloc[-1] > result["ctl"].iloc[5]

    def test_tsb_eq_ctl_minus_atl(self):
        dates = pd.date_range("2025-06-01", periods=30, freq="D")
        trimp_series = pd.Series(np.random.default_rng(1).uniform(40, 100, 30), index=dates)
        result = compute_training_loads(trimp_series)
        np.testing.assert_allclose(
            result["tsb"].values, (result["ctl"] - result["atl"]).values, rtol=1e-6
        )


class TestCadencePaceCoupling:
    def test_high_correlation_on_interval_run(self, interval_run_df):
        result = cadence_pace_coupling(interval_run_df)
        # Intervals vary both speed and cadence → positive correlation expected
        assert not np.isnan(result["cadence_pace_corr"])

    def test_returns_all_keys(self, easy_run_df):
        result = cadence_pace_coupling(easy_run_df)
        for key in ("cadence_pace_corr", "cadence_cv", "mean_cadence", "cadence_at_marathon_pace"):
            assert key in result

    def test_mean_cadence_in_range(self, easy_run_df):
        result = cadence_pace_coupling(easy_run_df)
        assert 100 < result["mean_cadence"] < 240

    def test_nan_when_no_cadence(self):
        df = pd.DataFrame({
            "cadence": [np.nan] * 100,
            "gap_speed_ms": np.full(100, 3.3),
            "gap_pace_min_per_km": np.full(100, 5.0),
            "elapsed_s": np.arange(100, dtype=float),
        })
        result = cadence_pace_coupling(df)
        assert np.isnan(result["mean_cadence"])


class TestPaceVariabilityIndex:
    def test_low_pvi_on_steady_pace(self):
        df = pd.DataFrame({
            "gap_pace_min_per_km": np.full(3600, 4.5) + np.random.default_rng(0).normal(0, 0.01, 3600)
        })
        result = pace_variability_index(df)
        assert result["pvi"] < 0.02

    def test_high_pvi_on_variable_pace(self, interval_run_df):
        result = pace_variability_index(interval_run_df)
        assert result["pvi"] > 0.05

    def test_nan_when_no_gap_pace(self):
        df = pd.DataFrame({"gap_pace_min_per_km": [np.nan] * 100})
        result = pace_variability_index(df)
        assert np.isnan(result["pvi"])


class TestHRZoneDistribution:
    def test_distributions_sum_to_100(self, easy_run_df):
        result = hr_zone_distribution(easy_run_df, ZONES)
        total = sum(result.values())
        assert total == pytest.approx(100.0, rel=0.01)

    def test_easy_run_dominated_by_z1_z2(self, easy_run_df):
        result = hr_zone_distribution(easy_run_df, ZONES)
        low_zone = result["z1_pct"] + result["z2_pct"]
        assert low_zone > 50.0


class TestIntervalQuality:
    def _make_intervals(self, n=5, pace=3.5, hr=170.0):
        return [
            {
                "interval_idx": i + 1,
                "interval_start_s": i * 480.0,
                "interval_end_s": i * 480.0 + 300,
                "interval_duration_s": 300.0,
                "interval_avg_gap_pace": pace + i * 0.02,
                "interval_avg_hr": hr + i * 1.5,
                "recovery_duration_s": 180.0,
                "recovery_avg_hr": 130.0,
            }
            for i in range(n)
        ]

    def test_returns_required_keys(self):
        result = interval_quality(self._make_intervals())
        for key in ("pace_consistency", "hr_progression", "pace_degradation",
                    "work_rest_ratio", "peak_hr_reached_bpm"):
            assert key in result

    def test_empty_returns_nan(self):
        result = interval_quality([])
        assert np.isnan(result["pace_consistency"])

    def test_positive_hr_progression_on_drifting_hr(self):
        ivs = self._make_intervals(hr=160.0)
        # HR increases by 1.5 bpm per interval → positive slope
        result = interval_quality(ivs)
        assert result["hr_progression"] > 0

    def test_work_rest_ratio(self):
        ivs = self._make_intervals()
        result = interval_quality(ivs)
        # 300s work / 180s rest ≈ 1.67
        assert result["work_rest_ratio"] == pytest.approx(300 / 180, rel=0.01)
