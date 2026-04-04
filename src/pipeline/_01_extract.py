"""Step 1: FIT extraction pipeline.

Scans data/activities/ for *.fit.gz files, reads the first timestamp to
determine the run date, filters to the training block, smooths signals,
computes GAP, and saves a Parquet per run to outputs/runs/raw/.

Usage:
    python -m src.pipeline._01_extract \
        --data-dir data/ \
        --output-dir outputs/runs/raw/ \
        [--smoothing-window 10] \
        [--workers 4]
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.gap import add_gap_columns
from src.utils.fit_reader import compute_elapsed_seconds, get_first_timestamp, parse_fit_file

TRAINING_BLOCK_START = pd.Timestamp("2025-06-01", tz="UTC")
TRAINING_BLOCK_END = pd.Timestamp("2025-12-21", tz="UTC")

OUTPUT_COLUMNS = [
    "timestamp", "elapsed_s", "distance_m", "altitude_m", "gradient",
    "speed_raw", "speed", "gap_speed_ms", "gap_pace_min_per_km",
    "hr_raw", "heart_rate", "cadence_raw", "cadence", "temperature",
    "run_id", "activity_id", "run_date", "is_run_boundary",
]


def _activity_id_from_path(path: Path) -> str | None:
    """Return the numeric activity ID from a filename like 15643184146.fit.gz."""
    name = path.name
    for suffix in (".fit.gz", ".fit"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return stem if stem.isdigit() else stem  # keep non-numeric too
    return None


def _checksum(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _smooth(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
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

    if cache_dir is not None:
        cache_file = cache_dir / f"{activity_id}.md5"
        current_md5 = _checksum(fit_path)
        if out_path.exists() and cache_file.exists():
            if cache_file.read_text().strip() == current_md5:
                return activity_id

    df = parse_fit_file(fit_path)
    if df.empty:
        return f"EMPTY:{activity_id}"

    df["elapsed_s"] = compute_elapsed_seconds(df)

    if "distance" in df.columns:
        df.rename(columns={"distance": "distance_m"}, inplace=True)
    if "altitude" in df.columns:
        df.rename(columns={"altitude": "altitude_m"}, inplace=True)

    df = _smooth(df, window=smoothing_window)

    if "altitude_m" in df.columns and "distance_m" in df.columns and "speed" in df.columns:
        add_gap_columns(df)
    else:
        df["gradient"] = np.nan
        df["gap_speed_ms"] = np.nan
        df["gap_pace_min_per_km"] = np.nan

    df["run_id"] = run_id
    df["activity_id"] = activity_id
    df["run_date"] = run_date.date()

    df["is_run_boundary"] = False
    if len(df) > 0:
        df.iloc[0, df.columns.get_loc("is_run_boundary")] = True
        df.iloc[-1, df.columns.get_loc("is_run_boundary")] = True

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
    max_runs: int | None = None,
) -> pd.DataFrame:
    """Main extraction function.

    Scans data_dir/activities/ for *.fit.gz files, filters to the training
    block by reading each file's first timestamp, then processes each run.

    Args:
        max_runs: If set, stop after processing this many runs (useful in tests).

    Returns a summary DataFrame of processed runs.
    """
    activities_dir = data_dir / "activities"
    cache_dir = output_dir / ".cache"

    if not activities_dir.exists():
        raise FileNotFoundError(f"activities directory not found: {activities_dir}")

    fit_files = sorted(activities_dir.glob("*.fit.gz"))
    if not fit_files:
        raise FileNotFoundError(f"No *.fit.gz files found in {activities_dir}")

    print(f"Found {len(fit_files)} .fit.gz files; scanning timestamps...")

    # First pass: read only the first timestamp from each file (fast path)
    in_block: list[tuple[str, Path, pd.Timestamp]] = []
    for fit_path in fit_files:
        activity_id = _activity_id_from_path(fit_path)
        if activity_id is None:
            continue
        ts = get_first_timestamp(fit_path)
        if ts is None:
            continue
        if TRAINING_BLOCK_START <= ts <= TRAINING_BLOCK_END:
            in_block.append((activity_id, fit_path, ts))

    if not in_block:
        print("No qualifying runs found in training block.")
        return pd.DataFrame()

    in_block.sort(key=lambda x: x[2])
    if max_runs is not None:
        in_block = in_block[:max_runs]
    print(f"Found {len(in_block)} runs in training block.")

    tasks = [
        (activity_id, fit_path, run_date, run_id, output_dir, smoothing_window, cache_dir)
        for run_id, (activity_id, fit_path, run_date) in enumerate(in_block, start=1)
    ]

    results = []
    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        for task in tasks:
            res = _worker(task)
            results.append(res)

    ok = [r for r in results if not r.startswith("EMPTY:")]
    print(f"Extracted {len(ok)} / {len(tasks)} runs.")

    summary = pd.DataFrame([
        {"activity_id": aid, "run_date": ts.date()}
        for aid, _, ts in in_block
    ])
    return summary


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
