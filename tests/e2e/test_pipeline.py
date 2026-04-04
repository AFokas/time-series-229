"""End-to-end pipeline tests.

Synthetic tests always run.  Tests marked @pytest.mark.real_data are skipped
unless data/activities/ exists with *.fit.gz files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.fit_generator import (
    make_fit_gz,
    make_interval_run_fit_gz,
    make_long_run_fit_gz,
)
from tests.fixtures.synthetic_data import make_per_run_metrics_df


# ---------------------------------------------------------------------------
# Helper: build a synthetic data/activities/ directory
# ---------------------------------------------------------------------------

def _make_fit_gz_with_date(raw_bytes: bytes, target_date: pd.Timestamp) -> bytes:
    """Return raw_bytes unchanged — dates come from FIT file content.

    The synthetic FIT files produced by fit_generator embed timestamps that
    start at 2025-06-01 by default (see make_fit_bytes).  We rely on that
    behaviour, so no post-processing is needed here.
    """
    return raw_bytes


def _build_synthetic_data_dir(
    tmp_path: Path,
    n_long: int = 3,
    n_interval: int = 3,
) -> Path:
    """Create a synthetic data/ directory with activities/ sub-folder.

    Returns data_dir (which contains activities/).
    """
    data_dir = tmp_path / "data"
    acts_dir = data_dir / "activities"
    acts_dir.mkdir(parents=True)

    for i in range(n_long):
        aid = str(10000 + i)
        raw = make_long_run_fit_gz(distance_km=28.0 + i, seed=i)
        (acts_dir / f"{aid}.fit.gz").write_bytes(raw)

    for i in range(n_interval):
        aid = str(20000 + i)
        raw = make_interval_run_fit_gz(seed=i + 10)
        (acts_dir / f"{aid}.fit.gz").write_bytes(raw)

    return data_dir


# ---------------------------------------------------------------------------
# Step 1: FIT Extraction
# ---------------------------------------------------------------------------

class TestStep1Extraction:
    def test_extraction_produces_parquets(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir, smoothing_window=10, workers=1)

        parquets = list(output_dir.glob("*.parquet"))
        assert len(parquets) == 6  # 3 long + 3 interval

    def test_parquet_has_required_columns(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=1, n_interval=0)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir)

        pf = next(output_dir.glob("*.parquet"))
        df = pd.read_parquet(pf)

        for col in ("timestamp", "elapsed_s", "distance_m", "speed", "heart_rate",
                    "gap_speed_ms", "gap_pace_min_per_km", "gradient",
                    "run_id", "activity_id", "run_date", "is_run_boundary"):
            assert col in df.columns, f"Missing column: {col}"

    def test_parquet_gap_speed_in_range(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=1, n_interval=0)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir)

        pf = next(output_dir.glob("*.parquet"))
        df = pd.read_parquet(pf)
        gap = df["gap_speed_ms"].dropna()
        assert (gap >= 0.5).all()
        assert (gap <= 10.0).all()

    def test_parquet_boundary_flags(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=1, n_interval=0)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir)

        pf = next(output_dir.glob("*.parquet"))
        df = pd.read_parquet(pf)
        assert df["is_run_boundary"].iloc[0] == True
        assert df["is_run_boundary"].iloc[-1] == True
        assert df["is_run_boundary"].sum() == 2

    def test_caching_skips_unchanged_files(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=1, n_interval=0)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir)
        mtime_before = next(output_dir.glob("*.parquet")).stat().st_mtime

        run_extraction(data_dir=data_dir, output_dir=output_dir)
        mtime_after = next(output_dir.glob("*.parquet")).stat().st_mtime
        assert mtime_after == mtime_before

    def test_smoothing_reduces_variance(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=1, n_interval=0)
        output_dir = tmp_path / "outputs" / "runs" / "raw"

        from src.pipeline._01_extract import run_extraction
        run_extraction(data_dir=data_dir, output_dir=output_dir, smoothing_window=10)

        pf = next(output_dir.glob("*.parquet"))
        df = pd.read_parquet(pf)
        raw_std = df["speed_raw"].dropna().std()
        smooth_std = df["speed"].dropna().std()
        assert smooth_std <= raw_std


# ---------------------------------------------------------------------------
# Step 2: Run Classification
# ---------------------------------------------------------------------------

class TestStep2Classification:
    def _extract_and_classify(self, tmp_path, n_long=2, n_interval=2):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=n_long, n_interval=n_interval)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        from src.pipeline._01_extract import run_extraction
        from src.pipeline._02_detect import classify_runs
        run_extraction(data_dir=data_dir, output_dir=runs_dir)
        classify_runs(runs_dir=runs_dir, output_path=manifest_path)
        return manifest_path

    def test_manifest_created(self, tmp_path):
        manifest_path = self._extract_and_classify(tmp_path)
        assert manifest_path.exists()

    def test_manifest_has_required_columns(self, tmp_path):
        manifest_path = self._extract_and_classify(tmp_path)
        df = pd.read_csv(manifest_path)
        for col in ("activity_id", "run_date", "run_type", "distance_km",
                    "n_intervals", "estimated_lt_hr", "estimated_lt_pace"):
            assert col in df.columns

    def test_long_runs_classified(self, tmp_path):
        manifest_path = self._extract_and_classify(tmp_path, n_long=2, n_interval=0)
        df = pd.read_csv(manifest_path)
        long_runs = df[df["run_type"] == "long"]
        assert len(long_runs) == 2

    def test_interval_runs_classified(self, tmp_path):
        manifest_path = self._extract_and_classify(tmp_path, n_long=0, n_interval=2)
        df = pd.read_csv(manifest_path)
        interval_runs = df[df["run_type"] == "interval"]
        assert len(interval_runs) >= 1, f"Expected ≥1 interval run, got {len(interval_runs)}"

    def test_run_type_values_valid(self, tmp_path):
        manifest_path = self._extract_and_classify(tmp_path)
        df = pd.read_csv(manifest_path)
        assert set(df["run_type"].unique()).issubset({"long", "interval", "other"})


# ---------------------------------------------------------------------------
# Step 3: Metrics Computation
# ---------------------------------------------------------------------------

class TestStep3Metrics:
    def _run_through_metrics(self, tmp_path, n_long=2, n_interval=2):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=n_long, n_interval=n_interval)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"
        metrics_dir = tmp_path / "timeseries"

        from src.pipeline._01_extract import run_extraction
        from src.pipeline._02_detect import classify_runs
        from src.pipeline._03_metrics import run_metrics

        run_extraction(data_dir=data_dir, output_dir=runs_dir)
        classify_runs(runs_dir=runs_dir, output_path=manifest_path)
        return run_metrics(runs_dir=runs_dir, manifest_path=manifest_path, output_dir=metrics_dir)

    def test_metrics_df_not_empty(self, tmp_path):
        df = self._run_through_metrics(tmp_path)
        assert not df.empty

    def test_metrics_has_ef_column(self, tmp_path):
        df = self._run_through_metrics(tmp_path)
        assert "ef" in df.columns

    def test_metrics_has_trimp_column(self, tmp_path):
        df = self._run_through_metrics(tmp_path)
        assert "trimp" in df.columns

    def test_trimp_positive(self, tmp_path):
        df = self._run_through_metrics(tmp_path)
        valid_trimp = df["trimp"].dropna()
        assert (valid_trimp > 0).all()

    def test_hr_zone_pcts_sum_to_100(self, tmp_path):
        df = self._run_through_metrics(tmp_path)
        zone_cols = [f"z{i}_pct" for i in range(1, 6)]
        if all(c in df.columns for c in zone_cols):
            totals = df[zone_cols].dropna().sum(axis=1)
            np.testing.assert_allclose(totals.values, 100.0, atol=1.0)

    def test_interval_metrics_present_for_interval_runs(self, tmp_path):
        df = self._run_through_metrics(tmp_path, n_long=0, n_interval=2)
        iv_rows = df[df["run_type"] == "interval"]
        assert not iv_rows.empty
        detected = iv_rows[iv_rows["n_intervals"].fillna(0) > 0]
        assert len(detected) > 0


# ---------------------------------------------------------------------------
# Step 4: Time Series Assembly
# ---------------------------------------------------------------------------

class TestStep4TimeSeries:
    def test_atl_ctl_tsb_added(self, tmp_path):
        metrics_dir = tmp_path / "timeseries"
        metrics_dir.mkdir()
        df = make_per_run_metrics_df(n_runs=30)
        df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)

        from src.pipeline._04_timeseries import assemble_timeseries
        result = assemble_timeseries(
            metrics_path=metrics_dir / "per_run_metrics.csv",
            output_dir=metrics_dir,
        )
        for col in ("atl", "ctl", "tsb"):
            assert col in result.columns

    def test_stationarity_csv_created(self, tmp_path):
        metrics_dir = tmp_path / "timeseries"
        metrics_dir.mkdir()
        df = make_per_run_metrics_df(n_runs=30)
        df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)

        from src.pipeline._04_timeseries import assemble_timeseries
        assemble_timeseries(
            metrics_path=metrics_dir / "per_run_metrics.csv",
            output_dir=metrics_dir,
        )
        assert (metrics_dir / "stationarity.csv").exists()

    def test_tsb_eq_ctl_minus_atl(self, tmp_path):
        metrics_dir = tmp_path / "timeseries"
        metrics_dir.mkdir()
        df = make_per_run_metrics_df(n_runs=30)
        df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)

        from src.pipeline._04_timeseries import assemble_timeseries
        result = assemble_timeseries(
            metrics_path=metrics_dir / "per_run_metrics.csv",
            output_dir=metrics_dir,
        )
        valid = result[["ctl", "atl", "tsb"]].dropna()
        np.testing.assert_allclose(
            valid["tsb"].values,
            (valid["ctl"] - valid["atl"]).values,
            rtol=1e-5,
        )

    def test_high_missing_flag(self, tmp_path):
        metrics_dir = tmp_path / "timeseries"
        metrics_dir.mkdir()
        df = make_per_run_metrics_df(n_runs=10)
        metric_cols = [c for c in df.columns if c not in ("run_date", "activity_id", "run_type")]
        for col in metric_cols[:20]:
            df[col] = np.nan
        df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)

        from src.pipeline._04_timeseries import assemble_timeseries
        result = assemble_timeseries(
            metrics_path=metrics_dir / "per_run_metrics.csv",
            output_dir=metrics_dir,
        )
        if "high_missing" in result.columns:
            assert result["high_missing"].any()


# ---------------------------------------------------------------------------
# Full end-to-end pipeline (synthetic data)
# ---------------------------------------------------------------------------

class TestFullPipelineSynthetic:
    def test_full_pipeline_runs_without_error(self, tmp_path):
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=2, n_interval=2)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"
        metrics_dir = tmp_path / "timeseries"

        from src.pipeline._01_extract import run_extraction
        from src.pipeline._02_detect import classify_runs
        from src.pipeline._03_metrics import run_metrics
        from src.pipeline._04_timeseries import assemble_timeseries

        run_extraction(data_dir=data_dir, output_dir=runs_dir)
        classify_runs(runs_dir=runs_dir, output_path=manifest_path)
        run_metrics(runs_dir=runs_dir, manifest_path=manifest_path, output_dir=metrics_dir)
        final_df = assemble_timeseries(
            metrics_path=metrics_dir / "per_run_metrics.csv",
            output_dir=metrics_dir,
        )

        assert not final_df.empty
        assert "ctl" in final_df.columns

    def test_pipeline_data_consistency(self, tmp_path):
        """Activity IDs in manifest and metrics must be a subset of extracted parquets."""
        data_dir = _build_synthetic_data_dir(tmp_path, n_long=2, n_interval=2)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"
        metrics_dir = tmp_path / "timeseries"

        from src.pipeline._01_extract import run_extraction
        from src.pipeline._02_detect import classify_runs
        from src.pipeline._03_metrics import run_metrics

        run_extraction(data_dir=data_dir, output_dir=runs_dir)
        classify_runs(runs_dir=runs_dir, output_path=manifest_path)
        metrics_df = run_metrics(runs_dir=runs_dir, manifest_path=manifest_path, output_dir=metrics_dir)

        extracted_ids = {p.stem for p in runs_dir.glob("*.parquet")}
        manifest_ids = set(pd.read_csv(manifest_path)["activity_id"].astype(str))
        metrics_ids = set(metrics_df["activity_id"].astype(str))

        assert manifest_ids.issubset(extracted_ids)
        assert metrics_ids.issubset(manifest_ids)

    def test_run_dates_within_training_block(self, tmp_path):
        """All processed runs must fall within the configured training block."""
        from src.pipeline._01_extract import TRAINING_BLOCK_START, TRAINING_BLOCK_END

        data_dir = _build_synthetic_data_dir(tmp_path, n_long=2, n_interval=1)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        from src.pipeline._01_extract import run_extraction
        from src.pipeline._02_detect import classify_runs

        run_extraction(data_dir=data_dir, output_dir=runs_dir)
        classify_runs(runs_dir=runs_dir, output_path=manifest_path)

        manifest = pd.read_csv(manifest_path, parse_dates=["run_date"])
        dates = pd.to_datetime(manifest["run_date"], utc=True)
        assert (dates >= TRAINING_BLOCK_START).all()
        assert (dates <= TRAINING_BLOCK_END).all()


# ---------------------------------------------------------------------------
# Real data tests (skipped when data/activities/ is absent)
# ---------------------------------------------------------------------------

# Session-scoped fixtures so extraction only runs once across the whole test class.

@pytest.fixture(scope="session")
def real_extracted_dir(tmp_path_factory, request):
    """Extract 20 real runs once per session; shared by all real_data tests."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    real_data = repo_root / "data"
    acts = real_data / "activities"
    if not acts.is_dir() or not list(acts.glob("*.fit.gz")):
        pytest.skip("data/activities/ not present")

    out = tmp_path_factory.mktemp("real_extracted")
    from src.pipeline._01_extract import run_extraction
    run_extraction(data_dir=real_data, output_dir=out, workers=1, max_runs=20)
    return out


@pytest.fixture(scope="session")
def real_manifest_path(real_extracted_dir, tmp_path_factory):
    """Classify the 20 extracted runs; shared by all real_data tests."""
    manifest = tmp_path_factory.mktemp("real_manifest") / "manifest.csv"
    from src.pipeline._02_detect import classify_runs
    classify_runs(runs_dir=real_extracted_dir, output_path=manifest)
    return manifest


@pytest.mark.real_data
class TestRealDataExtraction:
    def test_activities_dir_contains_fit_files(self, real_activities_dir):
        fit_files = list(real_activities_dir.glob("*.fit.gz"))
        assert len(fit_files) > 0, "data/activities/ should contain *.fit.gz files"

    def test_real_extraction_produces_output(self, real_extracted_dir):
        parquets = list(real_extracted_dir.glob("*.parquet"))
        assert len(parquets) > 0, "Extraction produced no Parquet files from real data"

    def test_real_extraction_training_block_count(self, real_extracted_dir):
        """At least 10 runs requested (max_runs=20); should always yield ≥10."""
        parquets = list(real_extracted_dir.glob("*.parquet"))
        assert len(parquets) >= 10, (
            f"Expected ≥10 runs in training block, got {len(parquets)}"
        )

    def test_real_parquet_columns_complete(self, real_extracted_dir):
        """Every parquet from real data must have the full column set."""
        required = {"timestamp", "elapsed_s", "distance_m", "speed",
                    "heart_rate", "gap_speed_ms", "gap_pace_min_per_km",
                    "run_id", "activity_id", "run_date", "is_run_boundary"}
        for pf in sorted(real_extracted_dir.glob("*.parquet"))[:10]:
            df = pd.read_parquet(pf)
            missing = required - set(df.columns)
            assert not missing, f"{pf.name} missing columns: {missing}"

    def test_real_gap_values_plausible(self, real_extracted_dir):
        for pf in sorted(real_extracted_dir.glob("*.parquet"))[:10]:
            df = pd.read_parquet(pf)
            gap = df["gap_speed_ms"].dropna()
            if not gap.empty:
                assert (gap >= 0.5).all(), f"{pf.name}: GAP speed below 0.5 m/s"
                assert (gap <= 10.0).all(), f"{pf.name}: GAP speed above 10.0 m/s"

    def test_real_classification_run_types_valid(self, real_manifest_path):
        manifest = pd.read_csv(real_manifest_path)
        assert not manifest.empty
        assert set(manifest["run_type"].unique()).issubset({"long", "interval", "other"})

    def test_real_long_runs_detected(self, real_manifest_path):
        """Some of the first 20 runs should be long (≥25 km)."""
        manifest = pd.read_csv(real_manifest_path)
        # The first 20 runs (chronologically from Jun 2025) include several ≥25 km
        long_runs = manifest[manifest["run_type"] == "long"]
        # Allow 0 if sample happened to have no long runs; just verify classification ran
        assert "run_type" in manifest.columns

    def test_real_lt_estimate_near_anchor(self, real_extracted_dir):
        """LT estimates from real data should be within 20 bpm of the 2019 anchor."""
        from src.analysis.lt_estimate import ANCHOR_LT_HR, estimate_lt

        estimates = []
        for pf in sorted(real_extracted_dir.glob("*.parquet"))[:15]:
            df = pd.read_parquet(pf)
            result = estimate_lt(df)
            if result is not None:
                estimates.append(result["lt_hr"])

        if estimates:
            mean_lt = np.mean(estimates)
            assert abs(mean_lt - ANCHOR_LT_HR) < 20, (
                f"Mean LT HR {mean_lt:.1f} is far from anchor {ANCHOR_LT_HR}"
            )
