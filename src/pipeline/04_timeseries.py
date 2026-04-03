"""Step 4: Metric time series assembly.

Reads outputs/timeseries/per_run_metrics.csv, appends ATL/CTL/TSB,
runs stationarity checks, and writes:
  - outputs/timeseries/per_run_metrics.csv   (with ATL/CTL/TSB columns added)
  - outputs/timeseries/stationarity.csv      (ADF + KPSS results per column)

Usage:
    python -m src.pipeline.04_timeseries \
        [--metrics-path outputs/timeseries/per_run_metrics.csv] \
        [--trimp-csv-path data/strava_runs_clean.csv] \
        [--atl-days 7] \
        [--ctl-days 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_training_loads, trimp as compute_trimp
from src.analysis.models import make_stationary, stationarity_check

# Columns to check for stationarity (excluding identifiers and categoricals)
METRIC_COLUMNS = [
    "ef", "ef_first_half", "ef_second_half", "pa_hr_decoupling",
    "cardiac_drift_slope", "cardiac_drift_r2",
    "recovery_hr_drop_60s", "recovery_hr_drop_90s", "recovery_decay_constant",
    "trimp", "atl", "ctl", "tsb",
    "cadence_pace_corr", "cadence_cv", "mean_cadence", "cadence_at_marathon_pace",
    "pvi", "pvi_rolling_last_third",
    "z1_pct", "z2_pct", "z3_pct", "z4_pct", "z5_pct",
    "estimated_lt_hr", "estimated_lt_pace",
    "pace_consistency", "hr_progression", "pace_degradation",
]


def assemble_timeseries(
    metrics_path: Path,
    output_dir: Path,
    atl_days: int = 7,
    ctl_days: int = 42,
    hr_max: float = 195.0,
    hr_rest: float = 42.0,
) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path, parse_dates=["run_date"])
    df = df.sort_values("run_date").reset_index(drop=True)

    # Build a daily TRIMP series spanning the block (including rest days as 0)
    if "trimp" in df.columns and "run_date" in df.columns:
        daily = df.groupby("run_date")["trimp"].sum()
        # Fill missing days with 0 (rest days)
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0.0)
        loads = compute_training_loads(daily, atl_days=atl_days, ctl_days=ctl_days)

        # Merge ATL/CTL/TSB back onto per-run df
        df["run_date_dt"] = pd.to_datetime(df["run_date"]).dt.normalize()
        loads.index = pd.to_datetime(loads.index).normalize()
        df = df.join(loads, on="run_date_dt", how="left")
        df.drop(columns=["run_date_dt"], inplace=True)

    # Flag runs with >30% missing signals
    signal_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    if signal_cols:
        missing_frac = df[signal_cols].isna().mean(axis=1)
        df["high_missing"] = missing_frac > 0.30

    # Save enriched metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "per_run_metrics.csv"
    df.to_csv(enriched_path, index=False)
    print(f"Enriched metrics saved → {enriched_path}")

    # Stationarity checks
    stat_rows = []
    for col in signal_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 5:
            continue
        result = stationarity_check(series)
        result["column"] = col
        # Also record recommended transformation
        _, transform = make_stationary(series)
        result["recommended_transform"] = transform
        stat_rows.append(result)

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows).set_index("column")
        stat_path = output_dir / "stationarity.csv"
        stat_df.to_csv(stat_path)
        print(f"Stationarity checks saved → {stat_path}")

        non_stationary = stat_df[~stat_df["is_stationary"].fillna(False)]
        if not non_stationary.empty:
            print(f"\nNon-stationary series ({len(non_stationary)}):")
            for col, row in non_stationary.iterrows():
                print(f"  {col}: recommended transform = {row['recommended_transform']}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Step 4: Time series assembly")
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/timeseries/per_run_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timeseries"))
    parser.add_argument("--atl-days", type=int, default=7)
    parser.add_argument("--ctl-days", type=int, default=42)
    parser.add_argument("--hr-max", type=float, default=195.0)
    parser.add_argument("--hr-rest", type=float, default=42.0)
    args = parser.parse_args()

    assemble_timeseries(
        metrics_path=args.metrics_path,
        output_dir=args.output_dir,
        atl_days=args.atl_days,
        ctl_days=args.ctl_days,
        hr_max=args.hr_max,
        hr_rest=args.hr_rest,
    )


if __name__ == "__main__":
    main()
