"""Step 1: FIT extraction pipeline.

For each qualifying run in strava_runs_clean.csv (Activity Type == Run,
date within training block), parse the .fit.gz file, apply smoothing,
compute GAP, and save a Parquet file to outputs/runs/raw/{activity_id}.parquet.

Usage:
    python -m src.pipeline.01_extract \
        --data-dir data/ \
        --output-dir outputs/runs/raw/ \
        [--smoothing-window 10] \
        [--workers 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.gap import add_gap_columns
from src.utils.fit_reader import (
    compute_elapsed_seconds,
    get_activity_id_from_filename,
    parse_fit_file,
)

TRAINING_BLOCK_START = pd.Timestamp("2025-06-01", tz="UTC")
TRAINING_BLOCK_END = pd.Timestamp("2025-12-21", tz="UTC")

OUTPUT_COLUMNS = [
    "timestamp", "elapsed_s", "distance_m", "altitude_m", "gradient",
    "speed_raw", "speed", "gap_speed_ms", "gap_pace_min_per_km",
    "hr_raw", "heart_rate", "cadence_raw", "cadence", "temperature",
    "run_id", "activity_id", "run_date", "is_run_boundary",
]


def _checksum(filepath: Path) -> str:
    """MD5 checksum of a file for cache invalidation."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _smooth(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Apply rolling mean to continuous signals; store raw originals."""
    smooth_cols = {
        "speed": "speed_raw",
        "heart_rate": "hr_raw",
        "cadence": "cadence_raw",
        "altitude": "altitude_raw",
    }
    for col, raw_col in smooth_cols.items():
        if col in df.columns:
            df[raw_col] = df[col].copy()
            df[col] = (
                df[col]
                .rolling(window=window, min_periods=5, center=True)
                .mean()
            )
    return df


def process_one_run(
    activity_id: str,
    fit_path: Path,
    run_date: pd.Timestamp,
    run_id: int,
    output_dir: Path,
    smoothing_window: int = 10,
    cache_dir: Path | None = None,
) -> str:
    """Extract, smooth, and save one run. Returns activity_id on success."""
    out_path = output_dir / f"{activity_id}.parquet"

    # Check cache: if output exists and checksum matches, skip
    if cache_dir is not None:
        cache_file = cache_dir / f"{activity_id}.md5"
        current_md5 = _checksum(fit_path)
        if out_path.exists() and cache_file.exists():
            if cache_file.read_text().strip() == current_md5:
                return activity_id  # already processed

    # Parse FIT
    df = parse_fit_file(fit_path)
    if df.empty:
        return f"EMPTY:{activity_id}"

    # Elapsed seconds
    df["elapsed_s"] = compute_elapsed_seconds(df)

    # Rename distance column
    if "distance" in df.columns:
        df.rename(columns={"distance": "distance_m"}, inplace=True)

    # Rename altitude
    if "altitude" in df.columns:
        df.rename(columns={"altitude": "altitude_m"}, inplace=True)

    # Smooth
    df = _smooth(df, window=smoothing_window)

    # GAP
    if "altitude_m" in df.columns and "distance_m" in df.columns and "speed" in df.columns:
        add_gap_columns(df)
    else:
        df["gradient"] = np.nan
        df["gap_speed_ms"] = np.nan
        df["gap_pace_min_per_km"] = np.nan

    # Metadata columns
    df["run_id"] = run_id
    df["activity_id"] = activity_id
    df["run_date"] = run_date.date()

    # Boundary flag
    df["is_run_boundary"] = False
    if len(df) > 0:
        df.iloc[0, df.columns.get_loc("is_run_boundary")] = True
        df.iloc[-1, df.columns.get_loc("is_run_boundary")] = True

    # Keep only specified columns (fill missing with NaN)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OUTPUT_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{activity_id}.md5").write_text(current_md5)

    return activity_id


def _worker(args):
    return process_one_run(*args)


def run_extraction(
    data_dir: Path,
    output_dir: Path,
    smoothing_window: int = 10,
    workers: int = 1,
) -> pd.DataFrame:
    """Main extraction function.

    Returns a DataFrame summarising processed runs.
    """
    csv_path = data_dir / "strava_runs_clean.csv"
    zip_path = data_dir / "activities.zip"
    cache_dir = output_dir / ".cache"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"activities.zip not found: {zip_path}")

    runs_df = pd.read_csv(csv_path)

    # Normalise column names
    runs_df.columns = [c.strip() for c in runs_df.columns]

    # Filter to runs only
    if "Activity Type" in runs_df.columns:
        runs_df = runs_df[runs_df["Activity Type"].str.strip().str.lower() == "run"].copy()

    # Parse date
    date_col = next((c for c in runs_df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found in strava_runs_clean.csv")
    runs_df["_date"] = pd.to_datetime(runs_df[date_col], utc=True)

    # Filter to training block
    mask = (runs_df["_date"] >= TRAINING_BLOCK_START) & (runs_df["_date"] <= TRAINING_BLOCK_END)
    runs_df = runs_df[mask].sort_values("_date").reset_index(drop=True)

    if runs_df.empty:
        print("No qualifying runs found in training block.")
        return pd.DataFrame()

    print(f"Found {len(runs_df)} qualifying runs.")

    # Extract zip to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"Extracting activities.zip → {tmp}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        # Build task list
        tasks = []
        for run_id, (_, row) in enumerate(runs_df.iterrows(), start=1):
            filename = str(row.get("Filename", ""))
            activity_id = get_activity_id_from_filename(filename)
            if activity_id is None:
                print(f"  Skipping row {run_id}: cannot parse activity_id from '{filename}'")
                continue
            fit_path = tmp / filename
            if not fit_path.exists():
                # Try just the basename
                fit_path = tmp / Path(filename).name
            if not fit_path.exists():
                print(f"  Skipping {activity_id}: file not found in zip ({filename})")
                continue

            tasks.append((
                activity_id,
                fit_path,
                row["_date"],
                run_id,
                output_dir,
                smoothing_window,
                cache_dir,
            ))

        # Process
        results = []
        if workers > 1:
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.map(_worker, tasks)
        else:
            for task in tasks:
                res = _worker(task)
                results.append(res)
                status = "SKIP" if res == task[0] else res
                print(f"  {status}")

    ok = [r for r in results if not r.startswith("EMPTY:")]
    print(f"\nExtracted {len(ok)} / {len(tasks)} runs.")
    return runs_df


def main():
    parser = argparse.ArgumentParser(description="Step 1: FIT extraction")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/raw"))
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_extraction(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        smoothing_window=args.smoothing_window,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
