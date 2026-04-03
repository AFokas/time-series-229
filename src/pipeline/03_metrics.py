"""Step 3: Per-run metric computation.

For every run in the manifest (long + interval), calls all applicable
metric functions and writes enriched per-run Parquet files to
outputs/runs/metrics/{activity_id}.parquet.

Also exports per-interval segment data to outputs/runs/intervals/.

Usage:
    python -m src.pipeline.03_metrics \
        --runs-dir outputs/runs/raw/ \
        --manifest outputs/run_manifest.csv \
        [--hr-max 195] \
        [--hr-rest 42] \
        [--ef-zone-ceiling 0.95] \
        [--pace-tolerance 0.25]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.lt_estimate import LTTracker, estimate_lt, LTEstimate
from src.analysis.metrics import (
    aerobic_decoupling,
    cadence_pace_coupling,
    cardiac_drift_rate,
    efficiency_factor,
    hr_zone_distribution,
    interval_quality,
    pace_variability_index,
    trimp,
)
from src.pipeline.02_detect import detect_intervals
from src.utils.zones import ZoneBoundaries


def compute_run_metrics(
    df: pd.DataFrame,
    run_type: str,
    lt_hr: float,
    lt_pace: float,
    hr_max: float,
    hr_rest: float,
    ef_zone_ceiling: float = 0.95,
    pace_tolerance: float = 0.25,
) -> dict:
    """Compute all applicable metrics for one run. Returns a flat dict."""
    zones = ZoneBoundaries.from_lt(lt_hr, lt_pace)
    result: dict = {}

    # EF and Pa:HR (aerobic runs only; interval EF computed separately)
    ef_metrics = efficiency_factor(df, lt_hr=lt_hr, ef_zone_ceiling=ef_zone_ceiling)
    result.update(ef_metrics)
    result["pa_hr_decoupling"] = aerobic_decoupling(
        ef_metrics["ef_first_half"], ef_metrics["ef_second_half"]
    )

    # Cardiac drift
    result.update(cardiac_drift_rate(df, pace_tolerance=pace_tolerance))

    # TRIMP
    hr_series = df["heart_rate"].dropna() if "heart_rate" in df.columns else pd.Series(dtype=float)
    duration_s = float(df["elapsed_s"].max()) if "elapsed_s" in df.columns and df["elapsed_s"].notna().any() else 0.0
    result["trimp"] = trimp(duration_s, hr_series, hr_max=hr_max, hr_rest=hr_rest)

    # Cadence
    result.update(cadence_pace_coupling(df))

    # HR zone distribution
    result.update(hr_zone_distribution(df, zones))

    # Long-run specific
    if run_type == "long":
        result.update(pace_variability_index(df))
    else:
        result["pvi"] = np.nan
        result["pvi_rolling_last_third"] = np.nan

    # Interval-specific
    if run_type == "interval":
        is_iv, intervals = detect_intervals(df, zones)
        if is_iv and intervals:
            result.update(interval_quality(intervals))
            # Recovery metrics: average across all intervals
            drops_60 = []
            drops_90 = []
            decays = []
            for iv in intervals:
                rec_start = iv.get("interval_end_s")
                rec_end = iv.get("interval_end_s", 0) + iv.get("recovery_duration_s", 0)
                if rec_start and "elapsed_s" in df.columns and "heart_rate" in df.columns:
                    rec_df = df[
                        (df["elapsed_s"] >= rec_start) & (df["elapsed_s"] < rec_end)
                    ]["heart_rate"].dropna()
                    if len(rec_df) >= 10:
                        from src.analysis.metrics import recovery_rate as _recovery_rate
                        rr = _recovery_rate(iv["interval_avg_hr"], rec_df.values)
                        drops_60.append(rr["hr_drop_60s"])
                        drops_90.append(rr["hr_drop_90s"])
                        decays.append(rr["decay_constant"])
            result["recovery_hr_drop_60s"] = float(np.nanmean(drops_60)) if drops_60 else np.nan
            result["recovery_hr_drop_90s"] = float(np.nanmean(drops_90)) if drops_90 else np.nan
            result["recovery_decay_constant"] = float(np.nanmean(decays)) if decays else np.nan
            result["n_intervals"] = len(intervals)
        else:
            for k in ["pace_consistency", "hr_progression", "pace_degradation",
                      "work_rest_ratio", "peak_hr_reached_bpm",
                      "recovery_hr_drop_60s", "recovery_hr_drop_90s",
                      "recovery_decay_constant", "n_intervals"]:
                result[k] = np.nan
    else:
        for k in ["pace_consistency", "hr_progression", "pace_degradation",
                  "work_rest_ratio", "peak_hr_reached_bpm",
                  "recovery_hr_drop_60s", "recovery_hr_drop_90s",
                  "recovery_decay_constant", "n_intervals"]:
            result[k] = np.nan

    return result


def run_metrics(
    runs_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    hr_max: float = 195.0,
    hr_rest: float = 42.0,
    ef_zone_ceiling: float = 0.95,
    pace_tolerance: float = 0.25,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    qualifying = manifest[manifest["run_type"].isin(["long", "interval"])].copy()

    tracker = LTTracker()
    rows = []

    for _, row in qualifying.iterrows():
        activity_id = str(row["activity_id"])
        run_type = row["run_type"]
        lt_hr = float(row.get("estimated_lt_hr", 175.0))
        lt_pace = float(row.get("estimated_lt_pace", 3.95))

        pf = runs_dir / f"{activity_id}.parquet"
        if not pf.exists():
            print(f"  Missing parquet: {activity_id}")
            continue

        df = pd.read_parquet(pf)
        metrics = compute_run_metrics(
            df, run_type, lt_hr, lt_pace,
            hr_max=hr_max, hr_rest=hr_rest,
            ef_zone_ceiling=ef_zone_ceiling,
            pace_tolerance=pace_tolerance,
        )
        metrics["activity_id"] = activity_id
        metrics["run_date"] = row.get("run_date")
        metrics["run_type"] = run_type
        metrics["distance_km"] = row.get("distance_km")
        metrics["estimated_lt_hr"] = lt_hr
        metrics["estimated_lt_pace"] = lt_pace
        rows.append(metrics)
        print(f"  {activity_id}: {run_type} — trimp={metrics['trimp']:.1f}")

    result_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "per_run_metrics.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nPer-run metrics saved → {out_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Step 3: Per-run metric computation")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/run_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timeseries"))
    parser.add_argument("--hr-max", type=float, default=195.0)
    parser.add_argument("--hr-rest", type=float, default=42.0)
    parser.add_argument("--ef-zone-ceiling", type=float, default=0.95)
    parser.add_argument("--pace-tolerance", type=float, default=0.25)
    args = parser.parse_args()

    run_metrics(
        runs_dir=args.runs_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        hr_max=args.hr_max,
        hr_rest=args.hr_rest,
        ef_zone_ceiling=args.ef_zone_ceiling,
        pace_tolerance=args.pace_tolerance,
    )


if __name__ == "__main__":
    main()
