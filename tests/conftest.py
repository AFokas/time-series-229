"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure vendor/ is on path so fitparse can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

# Also ensure repo root is importable as a package root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.synthetic_data import (
    make_interval_run_df,
    make_long_run_df,
    make_per_run_metrics_df,
    make_run_df,
)
from tests.fixtures.fit_generator import (
    make_fit_gz,
    make_interval_run_fit_gz,
    make_long_run_fit_gz,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: requires data/activities/ directory with *.fit.gz files",
    )


def _real_data_present() -> bool:
    activities_dir = REPO_ROOT / "data" / "activities"
    return activities_dir.is_dir() and bool(list(activities_dir.glob("*.fit.gz")))


def pytest_collection_modifyitems(config, items):
    skip_real = pytest.mark.skip(
        reason="data/activities/ not present; add real data to run"
    )
    for item in items:
        if "real_data" in item.keywords and not _real_data_present():
            item.add_marker(skip_real)


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_run_df():
    """1-hour easy aerobic run DataFrame."""
    return make_run_df(n_seconds=3600, base_speed_ms=3.3, base_hr_bpm=140.0)


@pytest.fixture
def long_run_df():
    """30 km long run DataFrame."""
    return make_long_run_df(distance_km=30.0, pace_min_per_km=4.5)


@pytest.fixture
def interval_run_df():
    """5 × 5 min interval session DataFrame."""
    return make_interval_run_df(
        n_intervals=5, interval_duration_s=300, recovery_duration_s=180
    )


@pytest.fixture
def per_run_metrics_df():
    """20-run per_run_metrics table for modelling tests."""
    return make_per_run_metrics_df(n_runs=20)


# ---------------------------------------------------------------------------
# FIT file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_run_fit_gz(tmp_path):
    """Path to a synthetic easy-run .fit.gz file."""
    p = tmp_path / "easy_run.fit.gz"
    data = make_fit_gz(n_seconds=3600, base_speed_ms=3.3)
    p.write_bytes(data)
    return p


@pytest.fixture
def interval_run_fit_gz(tmp_path):
    """Path to a synthetic interval-session .fit.gz file."""
    p = tmp_path / "interval_run.fit.gz"
    data = make_interval_run_fit_gz()
    p.write_bytes(data)
    return p


@pytest.fixture
def long_run_fit_gz(tmp_path):
    """Path to a synthetic 30 km long run .fit.gz file."""
    p = tmp_path / "long_run.fit.gz"
    data = make_long_run_fit_gz(distance_km=30.0)
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Real data paths (for e2e tests marked @pytest.mark.real_data)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_data_dir():
    return REPO_ROOT / "data"


@pytest.fixture
def real_activities_dir():
    return REPO_ROOT / "data" / "activities"


@pytest.fixture
def real_strava_csv():
    """May be an LFS pointer if not downloaded; tests should handle gracefully."""
    return REPO_ROOT / "data" / "strava_runs_clean.csv"
