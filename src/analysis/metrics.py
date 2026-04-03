"""Physiological metrics library.

All functions accept a single-run DataFrame (as produced by 01_extract.py)
and return a flat dict of metric values.  All pace-based metrics operate on
GAP (gap_speed_ms / gap_pace_min_per_km) unless explicitly noted.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from src.utils.zones import ZoneBoundaries, speed_to_pace_series


# ---------------------------------------------------------------------------
# 3.2  Efficiency Factor
# ---------------------------------------------------------------------------

def efficiency_factor(df: pd.DataFrame, lt_hr: float, ef_zone_ceiling: float = 0.95) -> dict:
    """EF = mean(GAP speed m/s) / mean(HR bpm).

    Computed for: full run, first half, second half.
    Only valid for runs where mean HR < ef_zone_ceiling * lt_hr.
    """
    required = ["gap_speed_ms", "heart_rate"]
    sub = df[required].dropna()
    if sub.empty:
        return {"ef": np.nan, "ef_first_half": np.nan, "ef_second_half": np.nan}

    mean_hr = sub["heart_rate"].mean()
    if mean_hr >= ef_zone_ceiling * lt_hr:
        return {"ef": np.nan, "ef_first_half": np.nan, "ef_second_half": np.nan}

    def _ef(rows: pd.DataFrame) -> float:
        if rows.empty or rows["heart_rate"].mean() == 0:
            return np.nan
        return float(rows["gap_speed_ms"].mean() / rows["heart_rate"].mean())

    mid = len(sub) // 2
    return {
        "ef": _ef(sub),
        "ef_first_half": _ef(sub.iloc[:mid]),
        "ef_second_half": _ef(sub.iloc[mid:]),
    }


# ---------------------------------------------------------------------------
# 3.3  Aerobic Decoupling (Pa:HR)
# ---------------------------------------------------------------------------

def aerobic_decoupling(ef_first_half: float, ef_second_half: float) -> float:
    """Pa:HR = (EF_first - EF_second) / EF_first * 100.

    Returns NaN if either half is NaN or EF_first is zero.
    """
    if np.isnan(ef_first_half) or np.isnan(ef_second_half) or ef_first_half == 0:
        return np.nan
    return float((ef_first_half - ef_second_half) / ef_first_half * 100.0)


# ---------------------------------------------------------------------------
# 3.4  Cardiac Drift Rate
# ---------------------------------------------------------------------------

def cardiac_drift_rate(
    df: pd.DataFrame,
    pace_tolerance: float = 0.25,
) -> dict:
    """OLS slope of HR over time at constant GAP pace.

    Filters to seconds where GAP pace is within ±pace_tolerance min/km of
    the run's median GAP pace, then fits HR ~ elapsed_time_minutes.

    Returns drift_slope (bpm/min) and drift_r2.
    Positive slope → HR rising over time → cardiac drift.
    """
    required = ["gap_pace_min_per_km", "heart_rate", "elapsed_s"]
    sub = df[required].dropna()
    if len(sub) < 60:
        return {"cardiac_drift_slope": np.nan, "cardiac_drift_r2": np.nan}

    median_pace = sub["gap_pace_min_per_km"].median()
    band = sub[
        (sub["gap_pace_min_per_km"] >= median_pace - pace_tolerance)
        & (sub["gap_pace_min_per_km"] <= median_pace + pace_tolerance)
    ]
    if len(band) < 60:
        return {"cardiac_drift_slope": np.nan, "cardiac_drift_r2": np.nan}

    t = band["elapsed_s"].values / 60.0  # convert to minutes
    hr = band["heart_rate"].values

    # OLS via numpy polyfit (degree 1)
    coeffs = np.polyfit(t, hr, 1)
    slope = float(coeffs[0])
    hr_pred = np.polyval(coeffs, t)
    ss_res = np.sum((hr - hr_pred) ** 2)
    ss_tot = np.sum((hr - hr.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"cardiac_drift_slope": slope, "cardiac_drift_r2": r2}


# ---------------------------------------------------------------------------
# 3.5  Recovery Rate Between Intervals
# ---------------------------------------------------------------------------

def recovery_rate(
    hr_at_end_of_interval: float,
    hr_series_during_recovery: np.ndarray | pd.Series,
) -> dict:
    """Fit exponential decay: HR(t) = (HR_end - HR_base) * exp(-lambda * t) + HR_base.

    Returns hr_drop_60s, hr_drop_90s, and decay_constant (lambda).
    Faster decay (larger lambda) → better recovery → better fitness.
    """
    hr = np.asarray(hr_series_during_recovery, dtype=float)
    n = len(hr)
    if n < 10:
        return {"hr_drop_60s": np.nan, "hr_drop_90s": np.nan, "decay_constant": np.nan}

    t = np.arange(n, dtype=float)

    def _exp_decay(t, lam, hr_base):
        return (hr_at_end_of_interval - hr_base) * np.exp(-lam * t) + hr_base

    try:
        popt, _ = curve_fit(
            _exp_decay,
            t,
            hr,
            p0=[0.02, hr[-1]],
            bounds=([0, 40], [1.0, hr_at_end_of_interval]),
            maxfev=2000,
        )
        lam, hr_base = popt
        drop_60 = float(hr_at_end_of_interval - _exp_decay(min(60, n - 1), lam, hr_base))
        drop_90 = float(hr_at_end_of_interval - _exp_decay(min(90, n - 1), lam, hr_base))
        return {
            "hr_drop_60s": drop_60,
            "hr_drop_90s": drop_90,
            "decay_constant": float(lam),
        }
    except (RuntimeError, ValueError):
        # Fall back to simple differences if curve fitting fails
        drop_60 = float(hr[0] - hr[min(60, n - 1)]) if n > 60 else float(hr[0] - hr[-1])
        drop_90 = float(hr[0] - hr[min(90, n - 1)]) if n > 90 else float(hr[0] - hr[-1])
        return {"hr_drop_60s": drop_60, "hr_drop_90s": drop_90, "decay_constant": np.nan}


# ---------------------------------------------------------------------------
# 3.6  TRIMP
# ---------------------------------------------------------------------------

def trimp(
    duration_s: float,
    hr_series: pd.Series | np.ndarray,
    hr_max: float,
    hr_rest: float,
) -> float:
    """Banister et al. (1991) TRIMP — male weighting factor.

    TRIMP = Σ(Δt * hr_ratio * 0.64 * e^(1.92 * hr_ratio))
    where hr_ratio = (HR - HR_rest) / (HR_max - HR_rest)
    Δt is in minutes (1 second = 1/60 minute).
    """
    hr = np.asarray(hr_series, dtype=float)
    hr = hr[~np.isnan(hr)]
    if len(hr) == 0 or hr_max <= hr_rest:
        return 0.0

    hr_ratio = (hr - hr_rest) / (hr_max - hr_rest)
    hr_ratio = np.clip(hr_ratio, 0, None)
    dt_min = 1.0 / 60.0  # each data point = 1 second
    score = np.sum(dt_min * hr_ratio * 0.64 * np.exp(1.92 * hr_ratio))
    return float(score)


def compute_training_loads(
    daily_trimp: pd.Series,
    atl_days: int = 7,
    ctl_days: int = 42,
) -> pd.DataFrame:
    """Compute ATL, CTL, and TSB from a daily TRIMP time series.

    ATL: exponentially weighted TRIMP over atl_days (fatigue proxy)
    CTL: exponentially weighted TRIMP over ctl_days (fitness proxy)
    TSB: CTL - ATL (freshness proxy)

    Parameters
    ----------
    daily_trimp: pd.Series indexed by date, with TRIMP values (0 on rest days)
    atl_days:   span for ATL EWM
    ctl_days:   span for CTL EWM

    Returns a DataFrame with columns: atl, ctl, tsb
    """
    atl = daily_trimp.ewm(span=atl_days, min_periods=1).mean()
    ctl = daily_trimp.ewm(span=ctl_days, min_periods=1).mean()
    tsb = ctl - atl
    return pd.DataFrame({"atl": atl, "ctl": ctl, "tsb": tsb})


# ---------------------------------------------------------------------------
# 3.7  Cadence-Pace Coupling
# ---------------------------------------------------------------------------

def cadence_pace_coupling(
    df: pd.DataFrame,
    marathon_pace_min_per_km: float = 3.55,  # 3:33/km
    mp_tolerance: float = 10.0 / 60.0,
) -> dict:
    """Pearson correlation between cadence and GAP speed, excluding warmup/cooldown.

    Also computes cadence_cv, mean_cadence, and cadence_at_marathon_pace.
    """
    required = ["cadence", "gap_speed_ms", "gap_pace_min_per_km", "elapsed_s"]
    sub = df[required].dropna()
    if sub.empty:
        return {
            "cadence_pace_corr": np.nan,
            "cadence_cv": np.nan,
            "mean_cadence": np.nan,
            "cadence_at_marathon_pace": np.nan,
        }

    # Exclude first and last 5 minutes
    t_max = sub["elapsed_s"].max()
    core = sub[(sub["elapsed_s"] >= 300) & (sub["elapsed_s"] <= t_max - 300)]
    if len(core) < 60:
        core = sub

    corr = np.nan
    if len(core) >= 2 and core["gap_speed_ms"].std() > 0 and core["cadence"].std() > 0:
        r, _ = pearsonr(core["gap_speed_ms"].values, core["cadence"].values)
        corr = float(r)

    mean_cad = float(core["cadence"].mean())
    cv = float(core["cadence"].std() / mean_cad) if mean_cad > 0 else np.nan

    at_mp = core[
        (core["gap_pace_min_per_km"] >= marathon_pace_min_per_km - mp_tolerance)
        & (core["gap_pace_min_per_km"] <= marathon_pace_min_per_km + mp_tolerance)
    ]["cadence"]
    cadence_mp = float(at_mp.mean()) if not at_mp.empty else np.nan

    return {
        "cadence_pace_corr": corr,
        "cadence_cv": cv,
        "mean_cadence": mean_cad,
        "cadence_at_marathon_pace": cadence_mp,
    }


# ---------------------------------------------------------------------------
# 3.8  Pace Variability Index
# ---------------------------------------------------------------------------

def pace_variability_index(df: pd.DataFrame) -> dict:
    """PVI = std(GAP pace) / mean(GAP pace).  For long runs only.

    Also computes rolling 5-minute PVI to detect late-run deterioration.
    """
    pace = df["gap_pace_min_per_km"].dropna()
    if pace.empty or pace.mean() == 0:
        return {"pvi": np.nan, "pvi_rolling_last_third": np.nan}

    pvi = float(pace.std() / pace.mean())

    # Rolling 5-min PVI: take the last third of the run
    n = len(pace)
    last_third = pace.iloc[2 * n // 3 :]
    rolling_pvi = (
        last_third.rolling(window=300, min_periods=60).std()
        / last_third.rolling(window=300, min_periods=60).mean()
    )
    pvi_last_third = float(rolling_pvi.mean()) if not rolling_pvi.isna().all() else np.nan

    return {"pvi": pvi, "pvi_rolling_last_third": pvi_last_third}


# ---------------------------------------------------------------------------
# 3.9  HR Zone Distribution
# ---------------------------------------------------------------------------

def hr_zone_distribution(df: pd.DataFrame, zones: ZoneBoundaries) -> dict:
    """Percentage of time spent in each of 5 HR zones."""
    if "heart_rate" not in df.columns:
        return {f"z{i}_pct": np.nan for i in range(1, 6)}
    return zones.hr_zone_distribution(df["heart_rate"])


# ---------------------------------------------------------------------------
# 3.10  Interval Quality Metrics
# ---------------------------------------------------------------------------

def interval_quality(intervals: list[dict]) -> dict:
    """Compute interval quality metrics from a list of per-interval dicts.

    Each dict must have: interval_avg_gap_pace, interval_avg_hr,
    interval_duration_s, recovery_duration_s.
    """
    if not intervals:
        return {
            "pace_consistency": np.nan,
            "hr_progression": np.nan,
            "pace_degradation": np.nan,
            "work_rest_ratio": np.nan,
            "peak_hr_reached_pct": np.nan,
        }

    paces = np.array([iv["interval_avg_gap_pace"] for iv in intervals])
    hrs = np.array([iv["interval_avg_hr"] for iv in intervals])
    durations = np.array([iv["interval_duration_s"] for iv in intervals])
    recoveries = np.array([iv.get("recovery_duration_s", np.nan) for iv in intervals])
    indices = np.arange(len(intervals), dtype=float)

    pace_consistency = float(np.nanstd(paces))

    # Linear slope of HR across interval indices
    if len(indices) >= 2 and not np.isnan(hrs).all():
        hr_slope = float(np.polyfit(indices[~np.isnan(hrs)], hrs[~np.isnan(hrs)], 1)[0])
    else:
        hr_slope = np.nan

    # Linear slope of pace across interval indices (negative = slowing)
    if len(indices) >= 2 and not np.isnan(paces).all():
        pace_slope = float(np.polyfit(indices[~np.isnan(paces)], paces[~np.isnan(paces)], 1)[0])
    else:
        pace_slope = np.nan

    valid_rec = recoveries[~np.isnan(recoveries)]
    wr_ratio = float(durations.mean() / valid_rec.mean()) if len(valid_rec) > 0 and valid_rec.mean() > 0 else np.nan

    peak_hr = float(np.nanmax(hrs)) if not np.isnan(hrs).all() else np.nan

    return {
        "pace_consistency": pace_consistency,
        "hr_progression": hr_slope,
        "pace_degradation": pace_slope,
        "work_rest_ratio": wr_ratio,
        "peak_hr_reached_bpm": peak_hr,
    }
