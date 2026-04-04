"""Step 2: Run classification.

Reads outputs/runs/raw/*.parquet and strava_runs_clean.csv to classify
each run as 'long', 'interval', or 'other', then saves outputs/run_manifest.csv.

Usage:
    python -m src.pipeline.02_detect \
        --runs-dir outputs/runs/raw/ \
        --csv-path data/strava_runs_clean.csv \
        --output outputs/run_manifest.csv \
        [--lt-tracker-path outputs/lt_tracker.csv] \
        [--interval-min-duration 60] \
        [--long-run-min-km 25] \
        [--long-run-max-km 38]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.lt_estimate import LTEstimate, LTTracker, estimate_lt
from src.utils.zones import ZoneBoundaries, speed_to_pace_series


# ---------------------------------------------------------------------------
# 2.1  Long run detection
# ---------------------------------------------------------------------------

def is_long_run(distance_km: float, min_km: float = 25.0, max_km: float = 38.0) -> bool:
    return min_km <= distance_km <= max_km


# ---------------------------------------------------------------------------
# 2.2  Interval run detection
# ---------------------------------------------------------------------------

def detect_intervals(
    df: pd.DataFrame,
    zones: ZoneBoundaries,
    min_duration_s: int = 60,
    min_recovery_s: int = 60,
    min_segment_gap_s: int = 30,
    min_hard_fraction: float = 0.15,
    min_intervals: int = 3,
) -> tuple[bool, list[dict]]:
    """State-machine interval detector.

    Returns (is_interval_run, list_of_interval_dicts).
    Each interval dict has: interval_idx, interval_start_s, interval_end_s,
    interval_duration_s, interval_avg_gap_pace, interval_avg_hr,
    recovery_duration_s, recovery_avg_hr.
    """
    required = ["gap_pace_min_per_km", "elapsed_s"]
    if not all(c in df.columns for c in required):
        return False, []

    sub = df[required + ["heart_rate", "gap_speed_ms"]].copy()
    sub = sub[sub["gap_pace_min_per_km"].notna() & sub["elapsed_s"].notna()]
    if sub.empty:
        return False, []

    pace_zone = zones.pace_zone_series(sub["gap_pace_min_per_km"])

    # State machine: track hard / recovery segments
    HARD = {3, 4}
    EASY = {1, 2}

    intervals: list[dict] = []
    state = "easy"
    seg_start = int(sub["elapsed_s"].iloc[0])
    seg_rows: list[int] = []

    elapsed = sub["elapsed_s"].values
    pz = pace_zone.values
    hr_vals = sub["heart_rate"].values if "heart_rate" in sub.columns else np.full(len(sub), np.nan)
    pace_vals = sub["gap_pace_min_per_km"].values

    for i in range(len(sub)):
        z = int(pz[i])
        t = float(elapsed[i])

        if state == "easy":
            if z in HARD:
                state = "hard"
                seg_start = t
                seg_rows = [i]
        elif state == "hard":
            if z in HARD:
                seg_rows.append(i)
            else:
                # End of hard segment
                duration = t - seg_start
                if duration >= min_duration_s:
                    avg_pace = float(np.nanmean(pace_vals[seg_rows]))
                    avg_hr = float(np.nanmean(hr_vals[seg_rows]))
                    intervals.append({
                        "interval_idx": len(intervals) + 1,
                        "interval_start_s": seg_start,
                        "interval_end_s": t,
                        "interval_duration_s": duration,
                        "interval_avg_gap_pace": avg_pace,
                        "interval_avg_hr": avg_hr,
                        "_end_row": i,
                    })
                state = "easy"
                seg_start = t
                seg_rows = []

    # Handle run ending in a hard segment
    if state == "hard" and seg_rows:
        t = float(elapsed[-1])
        duration = t - seg_start
        if duration >= min_duration_s:
            avg_pace = float(np.nanmean(pace_vals[seg_rows]))
            avg_hr = float(np.nanmean(hr_vals[seg_rows]))
            intervals.append({
                "interval_idx": len(intervals) + 1,
                "interval_start_s": seg_start,
                "interval_end_s": t,
                "interval_duration_s": duration,
                "interval_avg_gap_pace": avg_pace,
                "interval_avg_hr": avg_hr,
                "_end_row": len(sub) - 1,
            })

    if len(intervals) < min_intervals:
        return False, []

    # Check minimum gap between consecutive intervals
    filtered = [intervals[0]]
    for iv in intervals[1:]:
        gap = iv["interval_start_s"] - filtered[-1]["interval_end_s"]
        if gap >= min_segment_gap_s and (iv["interval_start_s"] - filtered[-1]["interval_end_s"]) >= min_recovery_s:
            filtered.append(iv)
    intervals = filtered
    # Re-index after filtering
    for idx, iv in enumerate(intervals, start=1):
        iv["interval_idx"] = idx

    if len(intervals) < min_intervals:
        return False, []

    # Check that hard segments represent >= 15% of total moving time
    total_time = float(elapsed[-1] - elapsed[0])
    total_hard = sum(iv["interval_duration_s"] for iv in intervals)
    if total_time > 0 and total_hard / total_time < min_hard_fraction:
        return False, []

    # Compute recovery durations
    for k in range(len(intervals)):
        if k < len(intervals) - 1:
            rec_start = intervals[k]["interval_end_s"]
            rec_end = intervals[k + 1]["interval_start_s"]
            rec_dur = rec_end - rec_start
            rec_rows = sub[
                (sub["elapsed_s"] >= rec_start) & (sub["elapsed_s"] < rec_end)
            ]
            intervals[k]["recovery_duration_s"] = float(rec_dur)
            intervals[k]["recovery_avg_hr"] = float(rec_rows["heart_rate"].mean()) if "heart_rate" in rec_rows.columns else np.nan
        else:
            intervals[k]["recovery_duration_s"] = np.nan
            intervals[k]["recovery_avg_hr"] = np.nan

    # Remove internal helper key
    for iv in intervals:
        iv.pop("_end_row", None)

    return True, intervals


# ---------------------------------------------------------------------------
# Main classification loop
# ---------------------------------------------------------------------------

def classify_runs(
    runs_dir: Path,
    output_path: Path,
    lt_tracker_path: Optional[Path] = None,
    interval_min_duration: int = 60,
    long_run_min_km: float = 25.0,
    long_run_max_km: float = 38.0,
) -> pd.DataFrame:
    tracker = LTTracker()
    manifest_rows = []

    parquet_files = sorted(runs_dir.glob("*.parquet"))
    print(f"Classifying {len(parquet_files)} runs...")

    for pf in parquet_files:
        activity_id = pf.stem
        df = pd.read_parquet(pf)

        if df.empty:
            continue

        run_date = pd.to_datetime(df["run_date"].iloc[0], utc=True) if "run_date" in df.columns else pd.NaT
        distance_km = float(df["distance_m"].max()) / 1000.0 if "distance_m" in df.columns else 0.0

        # Get current LT estimate for this date
        lt_est = tracker.get_lt_for_date(run_date if pd.notna(run_date) else pd.Timestamp.now(tz="UTC"))
        zones = ZoneBoundaries.from_lt(lt_est.lt_hr, lt_est.lt_pace_min_per_km)

        # Attempt LT estimation from this run
        lt_result = estimate_lt(df)
        if lt_result is not None:
            new_est = LTEstimate(
                date=run_date if pd.notna(run_date) else pd.Timestamp.now(tz="UTC"),
                lt_hr=lt_result["lt_hr"],
                lt_pace_min_per_km=lt_result["lt_pace_min_per_km"],
                activity_id=activity_id,
            )
            tracker.add_estimate(new_est)

        # Classification
        run_type = "other"
        n_intervals = 0
        total_interval_time_s = 0.0
        avg_interval_pace = np.nan

        if is_long_run(distance_km, long_run_min_km, long_run_max_km):
            run_type = "long"
        else:
            is_interval, intervals = detect_intervals(
                df, zones, min_duration_s=interval_min_duration
            )
            if is_interval:
                run_type = "interval"
                n_intervals = len(intervals)
                total_interval_time_s = sum(iv["interval_duration_s"] for iv in intervals)
                avg_interval_pace = float(np.nanmean([iv["interval_avg_gap_pace"] for iv in intervals]))

        manifest_rows.append({
            "activity_id": activity_id,
            "run_date": run_date.date() if pd.notna(run_date) else None,
            "run_type": run_type,
            "distance_km": round(distance_km, 2),
            "n_intervals": n_intervals,
            "total_interval_time_s": total_interval_time_s,
            "avg_interval_pace": avg_interval_pace,
            "estimated_lt_hr": lt_est.lt_hr,
            "estimated_lt_pace": lt_est.lt_pace_min_per_km,
        })

        print(f"  {activity_id}: {run_type} ({distance_km:.1f} km)")

    manifest = pd.DataFrame(manifest_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"\nManifest saved → {output_path}")

    if lt_tracker_path is not None:
        tracker.to_dataframe().to_csv(lt_tracker_path, index=False)
        print(f"LT tracker saved → {lt_tracker_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Step 2: Run classification")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run_manifest.csv"))
    parser.add_argument("--lt-tracker-path", type=Path, default=Path("outputs/lt_tracker.csv"))
    parser.add_argument("--interval-min-duration", type=int, default=60)
    parser.add_argument("--long-run-min-km", type=float, default=25.0)
    parser.add_argument("--long-run-max-km", type=float, default=38.0)
    args = parser.parse_args()

    classify_runs(
        runs_dir=args.runs_dir,
        output_path=args.output,
        lt_tracker_path=args.lt_tracker_path,
        interval_min_duration=args.interval_min_duration,
        long_run_min_km=args.long_run_min_km,
        long_run_max_km=args.long_run_max_km,
    )


if __name__ == "__main__":
    main()
