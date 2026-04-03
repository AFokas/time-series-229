"""Lactate threshold estimation using the Dmax method.

Reference: Cheng B et al. (1992) A new approach for the determination of ventilatory
and lactate thresholds. Int J Sports Med 13(6):518-522.

Ground-truth anchor: 175 bpm / 3:57 min/km (Apr 10 2019 lab test).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Ground-truth anchor from Apr 10 2019 lab test
ANCHOR_DATE = pd.Timestamp("2019-04-10", tz="UTC")
ANCHOR_LT_HR = 175.0   # bpm
ANCHOR_LT_PACE = 3.95  # min/km  (3:57 = 3 + 57/60)

# Maximum deviation from rolling mean before an estimate is considered suspect
MAX_HR_DEVIATION_BPM = 15.0


@dataclass
class LTEstimate:
    date: pd.Timestamp
    lt_hr: float
    lt_pace_min_per_km: float
    method: str = "dmax"
    suspect: bool = False
    activity_id: Optional[str] = None


@dataclass
class LTTracker:
    """Maintains a rolling series of LT estimates, seeded from the 2019 anchor."""

    estimates: list[LTEstimate] = field(default_factory=list)
    lookback_weeks: int = 4

    def __post_init__(self):
        # Seed with ground-truth anchor
        self.estimates.append(
            LTEstimate(
                date=ANCHOR_DATE,
                lt_hr=ANCHOR_LT_HR,
                lt_pace_min_per_km=ANCHOR_LT_PACE,
                method="anchor",
            )
        )

    def add_estimate(self, est: LTEstimate) -> None:
        """Add a new estimate, flagging it as suspect if it deviates too far."""
        rolling_mean_hr = self._rolling_mean_hr(est.date)
        if rolling_mean_hr is not None and abs(est.lt_hr - rolling_mean_hr) > MAX_HR_DEVIATION_BPM:
            est.suspect = True
        self.estimates.append(est)
        self.estimates.sort(key=lambda e: e.date)

    def get_lt_for_date(self, date: pd.Timestamp) -> LTEstimate:
        """Return the most recent non-suspect LT estimate on or before `date`."""
        valid = [e for e in self.estimates if e.date <= date and not e.suspect]
        if not valid:
            # Fall back to anchor
            return self.estimates[0]
        return max(valid, key=lambda e: e.date)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "date": e.date,
                "lt_hr": e.lt_hr,
                "lt_pace_min_per_km": e.lt_pace_min_per_km,
                "method": e.method,
                "suspect": e.suspect,
                "activity_id": e.activity_id,
            }
            for e in self.estimates
        ]
        return pd.DataFrame(rows)

    def _rolling_mean_hr(self, date: pd.Timestamp) -> Optional[float]:
        cutoff = date - pd.Timedelta(weeks=self.lookback_weeks)
        recent = [
            e.lt_hr
            for e in self.estimates
            if cutoff <= e.date < date and not e.suspect
        ]
        return float(np.mean(recent)) if recent else None


def estimate_lt(df: pd.DataFrame) -> Optional[dict]:
    """Estimate LT from a single run using the Dmax method.

    Qualifying conditions:
    - Run duration >= 20 minutes above Z2 HR (HR > 75% of some reference —
      here we use HR > 130 bpm as a loose proxy when LT is unknown)
    - Both heart_rate and gap_speed_ms present
    - At least 200 valid data points

    The Dmax method:
    1. Fit a 3rd-order polynomial to the HR vs GAP speed curve.
    2. Draw the chord connecting the first and last points of the curve.
    3. Find the point of maximum perpendicular distance from the chord.
    4. That point is the LT estimate.

    Returns dict with keys: lt_hr, lt_pace_min_per_km
    or None if qualification criteria are not met.
    """
    required = ["heart_rate", "gap_speed_ms", "gap_pace_min_per_km"]
    for col in required:
        if col not in df.columns or df[col].isna().all():
            return None

    sub = df[required].dropna()
    if len(sub) < 200:
        return None

    # Require at least 20 minutes of data above a moderate HR
    moderate_hr_threshold = 130.0
    time_above = (sub["heart_rate"] > moderate_hr_threshold).sum()
    if time_above < 20 * 60:
        return None

    hr = sub["heart_rate"].values
    speed = sub["gap_speed_ms"].values

    # Sort by speed for polynomial fitting
    order = np.argsort(speed)
    speed_s = speed[order]
    hr_s = hr[order]

    # Fit 3rd-order polynomial: HR = f(speed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(speed_s, hr_s, 3)
    except (np.linalg.LinAlgError, ValueError):
        return None

    poly = np.poly1d(coeffs)
    speed_eval = np.linspace(speed_s.min(), speed_s.max(), 500)
    hr_eval = poly(speed_eval)

    # Chord from first to last point
    x1, y1 = speed_eval[0], hr_eval[0]
    x2, y2 = speed_eval[-1], hr_eval[-1]

    # Perpendicular distance from each point to the chord
    chord_len = np.hypot(x2 - x1, y2 - y1)
    if chord_len < 1e-9:
        return None

    distances = np.abs(
        (y2 - y1) * speed_eval - (x2 - x1) * hr_eval + x2 * y1 - y2 * x1
    ) / chord_len

    dmax_idx = np.argmax(distances)
    lt_speed = float(speed_eval[dmax_idx])
    lt_hr = float(hr_eval[dmax_idx])

    if lt_speed <= 0:
        return None

    lt_pace = 1000.0 / (lt_speed * 60.0)

    return {"lt_hr": lt_hr, "lt_pace_min_per_km": lt_pace}
