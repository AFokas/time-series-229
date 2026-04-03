"""Unit tests for src/utils/zones.py."""

import numpy as np
import pandas as pd
import pytest

from src.utils.zones import ZoneBoundaries, pace_to_speed, speed_to_pace, speed_to_pace_series


class TestPaceSpeedConversions:
    def test_pace_to_speed_known_value(self):
        # 4:00/km = 1000 / (4 * 60) = 4.1667 m/s
        result = pace_to_speed(4.0)
        assert result == pytest.approx(1000 / 240, rel=1e-4)

    def test_speed_to_pace_known_value(self):
        # 3.33 m/s = 1000 / (3.33 * 60) ≈ 5.0 min/km
        result = speed_to_pace(1000 / 300)
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_roundtrip(self):
        pace = 4.5
        assert speed_to_pace(pace_to_speed(pace)) == pytest.approx(pace, rel=1e-5)

    def test_zero_speed_returns_nan(self):
        assert np.isnan(speed_to_pace(0.0))

    def test_negative_speed_returns_nan(self):
        assert np.isnan(speed_to_pace(-1.0))

    def test_zero_pace_returns_zero(self):
        assert pace_to_speed(0.0) == 0.0

    def test_speed_to_pace_series_vectorised(self):
        speeds = pd.Series([0.0, 1.0, 1000 / 240])
        paces = speed_to_pace_series(speeds)
        assert np.isnan(paces.iloc[0])
        assert paces.iloc[2] == pytest.approx(4.0, rel=1e-4)


class TestZoneBoundaries:
    @pytest.fixture
    def zones(self):
        # LT: 175 bpm / 3:57/km (ground-truth anchor)
        return ZoneBoundaries.from_lt(lt_hr=175.0, lt_pace_min_per_km=3.95)

    # ---- HR zone classification ----

    def test_hr_zone1(self, zones):
        # < 75% * 175 = 131.25
        assert zones.hr_zone(120.0) == 1

    def test_hr_zone2(self, zones):
        # 75–88% * 175 = 131.25–154
        assert zones.hr_zone(140.0) == 2

    def test_hr_zone3(self, zones):
        # 88–100% * 175 = 154–175
        assert zones.hr_zone(165.0) == 3

    def test_hr_zone4(self, zones):
        # 100–110% * 175 = 175–192.5
        assert zones.hr_zone(180.0) == 4

    def test_hr_zone5(self, zones):
        # > 110% * 175 = 192.5
        assert zones.hr_zone(195.0) == 5

    def test_hr_zone_series(self, zones):
        hrs = pd.Series([120.0, 140.0, 165.0, 180.0, 195.0, np.nan])
        result = zones.hr_zone_series(hrs)
        assert list(result.iloc[:5]) == [1, 2, 3, 4, 5]
        assert result.iloc[5] == 0  # unknown

    def test_hr_zone_distribution_sums_to_100(self, zones):
        hrs = pd.Series(np.linspace(100, 200, 1000))
        dist = zones.hr_zone_distribution(hrs)
        total = sum(dist.values())
        assert total == pytest.approx(100.0, rel=1e-2)

    def test_hr_zone_distribution_all_nan(self, zones):
        hrs = pd.Series([np.nan, np.nan])
        dist = zones.hr_zone_distribution(hrs)
        assert all(np.isnan(v) for v in dist.values())

    # ---- Pace zone classification ----

    def test_pace_zone1_easy(self, zones):
        # > LT_pace + 1.0 min/km = 3.95 + 1.0 = 4.95
        assert zones.pace_zone(5.5) == 1

    def test_pace_zone2_aerobic(self, zones):
        # LT + 1/3 < pace <= LT + 1.0
        assert zones.pace_zone(4.5) == 2

    def test_pace_zone3_threshold(self, zones):
        # LT - 1/6 < pace <= LT + 1/3
        assert zones.pace_zone(3.95) == 3  # exactly at LT

    def test_pace_zone4_vo2max(self, zones):
        # <= LT - 1/6 ≈ 3.95 - 0.167 = 3.783
        assert zones.pace_zone(3.5) == 4

    def test_pace_zone_series(self, zones):
        paces = pd.Series([5.5, 4.5, 3.95, 3.5])
        result = zones.pace_zone_series(paces)
        assert list(result) == [1, 2, 3, 4]
