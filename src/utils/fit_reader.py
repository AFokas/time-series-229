"""FIT file parsing utilities."""

from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Try system fitparse first, then fall back to vendored copy
try:
    import fitparse
except ImportError:
    _vendor = Path(__file__).resolve().parents[2] / "vendor"
    if _vendor.exists() and str(_vendor) not in sys.path:
        sys.path.insert(0, str(_vendor))
    try:
        import fitparse
    except ImportError as e:
        raise ImportError(
            "fitparse is required. Install via 'pip install fitparse' or ensure "
            "vendor/fitparse exists in the repository root."
        ) from e

# Semicircles to degrees conversion factor (Garmin FIT format)
SEMICIRCLES_TO_DEGREES = 180.0 / 2**31

# Fields to extract from FIT record messages
RECORD_FIELDS = [
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "speed",
    "heart_rate",
    "cadence",
    "altitude",
    "enhanced_altitude",
    "temperature",
]


def parse_fit_file(filepath: Path | str) -> pd.DataFrame:
    """Parse a .fit or .fit.gz file and return a DataFrame of record messages.

    Handles both plain .fit and gzip-compressed .fit.gz files.
    Converts semicircles to degrees for lat/lon, multiplies cadence by 2
    for total SPM, and prefers enhanced_altitude over altitude when present.

    Returns a DataFrame with columns matching RECORD_FIELDS (NaN where absent).
    """
    filepath = Path(filepath)
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rb") as f:
            raw = io.BytesIO(f.read())
        fit = fitparse.FitFile(raw)
    else:
        fit = fitparse.FitFile(str(filepath))

    rows = []
    for record in fit.get_messages("record"):
        row: dict = {}
        data = {d.name: d.value for d in record}
        for field in RECORD_FIELDS:
            row[field] = data.get(field, None)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RECORD_FIELDS)

    df = pd.DataFrame(rows)

    # Convert semicircles to degrees
    for col in ("position_lat", "position_long"):
        if col in df.columns and df[col].notna().any():
            df[col] = df[col] * SEMICIRCLES_TO_DEGREES

    # Prefer enhanced_altitude over altitude
    if "enhanced_altitude" in df.columns and df["enhanced_altitude"].notna().any():
        df["altitude"] = df["enhanced_altitude"].combine_first(df["altitude"])
    df.drop(columns=["enhanced_altitude"], errors="ignore", inplace=True)

    # Total cadence: FIT stores per-foot strikes, multiply by 2
    if "cadence" in df.columns and df["cadence"].notna().any():
        df["cadence"] = df["cadence"] * 2

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def compute_elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    """Return elapsed seconds from the first timestamp."""
    if df["timestamp"].isna().all():
        return pd.Series(np.arange(len(df)), dtype=float)
    t0 = df["timestamp"].iloc[0]
    return (df["timestamp"] - t0).dt.total_seconds()


def get_activity_id_from_filename(filename: str) -> Optional[str]:
    """Extract numeric activity ID from a Strava filename like 'activities/1922813233.fit.gz'."""
    stem = Path(filename).stem
    # Remove .fit if it's a double extension like 1234.fit.gz → stem = 1234.fit
    if stem.endswith(".fit"):
        stem = stem[:-4]
    return stem if stem.isdigit() else None
