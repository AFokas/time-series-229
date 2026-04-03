"""HR and pace zone definitions.

Zones are defined relative to the current estimated LT HR and LT pace.
All pace values are in min/km; all HR values are in bpm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ZoneBoundaries:
    """HR and pace zone boundaries derived from LT estimates."""

    lt_hr: float          # bpm
    lt_pace: float        # min/km
    lt_speed: float       # m/s (derived from lt_pace)

    # HR zone upper bounds (lower bound of zone N = upper bound of zone N-1)
    # Relative multipliers applied to lt_hr
    z1_hr_upper: float = 0.75   # < 75% LT_HR
    z2_hr_upper: float = 0.88   # 75–88% LT_HR
    z3_hr_upper: float = 1.00   # 88–100% LT_HR
    z4_hr_upper: float = 1.10   # 100–110% LT_HR
    # Z5: > 110% LT_HR

    @classmethod
    def from_lt(cls, lt_hr: float, lt_pace_min_per_km: float) -> "ZoneBoundaries":
        lt_speed = pace_to_speed(lt_pace_min_per_km)
        return cls(lt_hr=lt_hr, lt_pace=lt_pace_min_per_km, lt_speed=lt_speed)

    # ---------- HR zone classification ----------

    def hr_zone(self, hr: float) -> int:
        """Return HR zone (1–5) for a single HR value."""
        if hr < self.lt_hr * self.z1_hr_upper:
            return 1
        if hr < self.lt_hr * self.z2_hr_upper:
            return 2
        if hr < self.lt_hr * self.z3_hr_upper:
            return 3
        if hr < self.lt_hr * self.z4_hr_upper:
            return 4
        return 5

    def hr_zone_series(self, hr_series: pd.Series) -> pd.Series:
        """Vectorised HR zone classification."""
        z = pd.Series(5, index=hr_series.index, dtype=int)
        z[hr_series < self.lt_hr * self.z4_hr_upper] = 4
        z[hr_series < self.lt_hr * self.z3_hr_upper] = 3
        z[hr_series < self.lt_hr * self.z2_hr_upper] = 2
        z[hr_series < self.lt_hr * self.z1_hr_upper] = 1
        z[hr_series.isna()] = 0  # unknown
        return z

    def hr_zone_distribution(self, hr_series: pd.Series) -> dict[str, float]:
        """Return % of time in each zone (only non-NaN rows counted)."""
        valid = hr_series.dropna()
        if valid.empty:
            return {f"z{i}_pct": float("nan") for i in range(1, 6)}
        zones = self.hr_zone_series(valid)
        total = len(valid)
        return {f"z{i}_pct": float((zones == i).sum() / total * 100) for i in range(1, 6)}

    # ---------- Pace zone classification (used in interval detection) ----------

    def pace_zone(self, gap_pace_min_per_km: float) -> int:
        """Return pace zone (1–4) for a single GAP pace value (min/km).

        Zone 1 (Easy):      GAP > LT_pace + 60s/km
        Zone 2 (Aerobic):   LT_pace + 20s/km < GAP <= LT_pace + 60s/km
        Zone 3 (Threshold): LT_pace - 10s/km < GAP <= LT_pace + 20s/km
        Zone 4 (VO2max):    GAP <= LT_pace - 10s/km
        (60s/km = 1.0 min/km, 20s/km ≈ 0.333 min/km, 10s/km ≈ 0.167 min/km)
        """
        if gap_pace_min_per_km <= self.lt_pace - 10 / 60:
            return 4
        if gap_pace_min_per_km <= self.lt_pace + 20 / 60:
            return 3
        if gap_pace_min_per_km <= self.lt_pace + 60 / 60:
            return 2
        return 1

    def pace_zone_series(self, gap_pace_series: pd.Series) -> pd.Series:
        """Vectorised pace zone classification (min/km input)."""
        z = pd.Series(1, index=gap_pace_series.index, dtype=int)
        z[gap_pace_series <= self.lt_pace + 60 / 60] = 2
        z[gap_pace_series <= self.lt_pace + 20 / 60] = 3
        z[gap_pace_series <= self.lt_pace - 10 / 60] = 4
        z[gap_pace_series.isna()] = 0
        return z


# ---------- Conversion helpers ----------

def pace_to_speed(pace_min_per_km: float) -> float:
    """Convert min/km to m/s."""
    if pace_min_per_km <= 0:
        return 0.0
    return 1000.0 / (pace_min_per_km * 60.0)


def speed_to_pace(speed_ms: float) -> float:
    """Convert m/s to min/km. Returns NaN for zero/negative speed."""
    if speed_ms <= 0:
        return float("nan")
    return 1000.0 / (speed_ms * 60.0)


def speed_to_pace_series(speed_series: pd.Series) -> pd.Series:
    """Vectorised m/s → min/km conversion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = 1000.0 / (speed_series * 60.0)
    pace[speed_series <= 0] = float("nan")
    return pace
