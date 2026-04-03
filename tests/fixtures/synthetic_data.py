"""Synthetic DataFrame fixtures that mirror pipeline intermediate outputs."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def make_run_df(
    n_seconds: int = 3600,
    start_time: datetime | None = None,
    base_speed_ms: float = 3.3,
    base_hr_bpm: float = 140.0,
    base_cadence_spm: float = 176.0,
    base_alt_m: float = 50.0,
    temperature_c: float = 15.0,
    hr_drift: float = 0.005,
    pace_variation: float = 0.05,
    interval_profile: list[tuple[int, int, float]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame matching the output of 01_extract.py.

    Columns: timestamp, elapsed_s, distance_m, altitude_m, gradient,
             speed_raw, speed, gap_speed_ms, gap_pace_min_per_km,
             hr_raw, heart_rate, cadence_raw, cadence, temperature,
             run_id, activity_id, run_date, is_run_boundary.
    """
    if start_time is None:
        start_time = datetime(2025, 8, 1, 7, 0, 0, tzinfo=timezone.utc)

    rng = np.random.default_rng(seed)
    times = pd.date_range(start_time, periods=n_seconds, freq="s")

    # Build interval lookup
    interval_zones: dict[int, float] = {}
    if interval_profile:
        for s0, s1, mult in interval_profile:
            for s in range(s0, s1):
                interval_zones[s] = mult

    speeds = []
    hrs = []
    cadences = []
    alts = []
    dists = [0.0]

    for i in range(n_seconds):
        mult = interval_zones.get(i, 1.0)
        spd = base_speed_ms * mult * (1 + rng.normal(0, pace_variation))
        spd = max(0.5, float(spd))
        speeds.append(spd)

        hr = base_hr_bpm + i * hr_drift + (mult - 1) * 25 + rng.normal(0, 1.5)
        hrs.append(float(np.clip(hr, 60, 200)))

        cad = base_cadence_spm * (0.9 + 0.2 * mult) + rng.normal(0, 2)
        cadences.append(float(np.clip(cad, 100, 220)))

        alt = base_alt_m + 0.5 * np.sin(i / 600) + rng.normal(0, 0.02)
        alts.append(float(alt))

        if i > 0:
            dists.append(dists[-1] + spd)

    speeds = np.array(speeds)
    hrs = np.array(hrs)
    cadences = np.array(cadences)
    alts = np.array(alts)
    dists = np.array(dists[:n_seconds])

    # Gradient
    delta_alt = np.diff(alts, prepend=alts[0])
    delta_dist = np.diff(dists, prepend=dists[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.where(delta_dist > 0.1, delta_alt / delta_dist, 0.0)
    grad = np.clip(grad, -0.45, 0.45)

    # GAP via Minetti
    C_slope = (155.4 * grad**5 - 30.4 * grad**4 - 43.3 * grad**3
               + 46.3 * grad**2 + 19.5 * grad + 3.6)
    gap_speed = np.clip(speeds * (C_slope / 3.6), 0.5, 10.0)
    gap_pace = 1000.0 / (gap_speed * 60.0)

    is_boundary = np.zeros(n_seconds, dtype=bool)
    is_boundary[0] = True
    is_boundary[-1] = True

    df = pd.DataFrame({
        "timestamp": times,
        "elapsed_s": np.arange(n_seconds, dtype=float),
        "distance_m": dists,
        "altitude_m": alts,
        "gradient": grad,
        "speed_raw": speeds,
        "speed": speeds,
        "gap_speed_ms": gap_speed,
        "gap_pace_min_per_km": gap_pace,
        "hr_raw": hrs,
        "heart_rate": hrs,
        "cadence_raw": cadences,
        "cadence": cadences,
        "temperature": float(temperature_c),
        "run_id": 1,
        "activity_id": "test_run_001",
        "run_date": start_time.date(),
        "is_run_boundary": is_boundary,
    })
    return df


def make_interval_run_df(
    n_warmup_s: int = 600,
    n_intervals: int = 5,
    interval_duration_s: int = 300,
    recovery_duration_s: int = 180,
    n_cooldown_s: int = 600,
    speed_easy_ms: float = 3.0,
    speed_hard_ms: float = 4.5,
    base_hr_bpm: float = 135.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame for a synthetic interval session."""
    profile = []
    t = n_warmup_s
    for _ in range(n_intervals):
        profile.append((t, t + interval_duration_s, speed_hard_ms / speed_easy_ms))
        t += interval_duration_s + recovery_duration_s
    total = t + n_cooldown_s

    return make_run_df(
        n_seconds=total,
        base_speed_ms=speed_easy_ms,
        base_hr_bpm=base_hr_bpm,
        interval_profile=profile,
        seed=seed,
    )


def make_long_run_df(
    distance_km: float = 30.0,
    pace_min_per_km: float = 4.5,
    base_hr_bpm: float = 145.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame for a synthetic long run (~30 km)."""
    speed_ms = 1000.0 / (pace_min_per_km * 60.0)
    n_seconds = int(distance_km * 1000 / speed_ms)
    return make_run_df(n_seconds=n_seconds, base_speed_ms=speed_ms,
                       base_hr_bpm=base_hr_bpm, seed=seed)


def make_per_run_metrics_df(n_runs: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic per_run_metrics.csv DataFrame for time-series model tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-06-01", periods=n_runs, freq="4D")

    ef_trend = np.linspace(0.0155, 0.0185, n_runs) + rng.normal(0, 0.0005, n_runs)
    pa_hr = np.clip(8.0 - np.linspace(0, 5, n_runs) + rng.normal(0, 1.0, n_runs), 0, 20)
    drift_slope = np.linspace(0.6, 0.1, n_runs) + rng.normal(0, 0.05, n_runs)
    ctl = np.linspace(40, 75, n_runs) + rng.normal(0, 3, n_runs)
    atl = ctl + rng.normal(0, 8, n_runs)
    tsb = ctl - atl
    trimp = rng.uniform(60, 130, n_runs)
    run_types = (["long"] * 3 + ["interval"] * 2) * (n_runs // 5 + 1)
    run_types = run_types[:n_runs]
    distances = [rng.uniform(26, 34) if t == "long" else rng.uniform(12, 18) for t in run_types]

    return pd.DataFrame({
        "run_date": dates,
        "activity_id": [f"act_{i:04d}" for i in range(n_runs)],
        "run_type": run_types,
        "distance_km": distances,
        "ef": ef_trend,
        "ef_first_half": ef_trend * rng.uniform(0.97, 1.03, n_runs),
        "ef_second_half": ef_trend * rng.uniform(0.92, 1.00, n_runs),
        "pa_hr_decoupling": pa_hr,
        "cardiac_drift_slope": drift_slope,
        "cardiac_drift_r2": rng.uniform(0.3, 0.8, n_runs),
        "recovery_hr_drop_60s": rng.uniform(15, 35, n_runs),
        "recovery_decay_constant": rng.uniform(0.01, 0.05, n_runs),
        "trimp": trimp,
        "atl": atl,
        "ctl": ctl,
        "tsb": tsb,
        "cadence_pace_corr": rng.uniform(0.3, 0.9, n_runs),
        "cadence_cv": rng.uniform(0.02, 0.07, n_runs),
        "mean_cadence": rng.uniform(165, 185, n_runs),
        "cadence_at_marathon_pace": rng.uniform(170, 185, n_runs),
        "pvi": rng.uniform(0.02, 0.08, n_runs),
        "pvi_rolling_last_third": rng.uniform(0.03, 0.10, n_runs),
        "z1_pct": rng.uniform(50, 75, n_runs),
        "z2_pct": rng.uniform(15, 30, n_runs),
        "z3_pct": rng.uniform(5, 20, n_runs),
        "z4_pct": rng.uniform(0, 10, n_runs),
        "z5_pct": rng.uniform(0, 3, n_runs),
        "estimated_lt_hr": 170 + rng.normal(0, 2, n_runs),
        "estimated_lt_pace": 3.95 + rng.normal(0, 0.1, n_runs),
        "n_intervals": [5 if t == "interval" else 0 for t in run_types],
        "pace_consistency": [rng.uniform(0.05, 0.15) if t == "interval" else np.nan for t in run_types],
        "hr_progression": [rng.uniform(1, 5) if t == "interval" else np.nan for t in run_types],
        "pace_degradation": [rng.uniform(-0.05, 0.05) if t == "interval" else np.nan for t in run_types],
    })
