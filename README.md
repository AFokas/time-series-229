# time-series-229

Retrospective physiological analysis of the 8-month training block (Jun–Dec 21 2025) leading to a **2:30:52 marathon** (Feb 14 2026, 3:33/km).

See [`spec.md`](spec.md) for the full pipeline specification.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following in `data/`:
- `activities.zip` — Strava export of raw `.fit.gz` files (Git LFS)
- `strava_runs_clean.csv` — cleaned activity summary (Git LFS)

See [`DATA_GUIDE.md`](DATA_GUIDE.md) for instructions on adding data.

## Pipeline

Run steps in order:

```bash
# 1. Extract FIT files and compute GAP
python -m src.pipeline.01_extract

# 2. Classify runs (long / interval / other)
python -m src.pipeline.02_detect

# 3. Compute per-run physiological metrics
python -m src.pipeline.03_metrics

# 4. Assemble time series + stationarity checks
python -m src.pipeline.04_timeseries
```

Each step accepts `--help` for all options (smoothing window, LT lookback, etc.).

## Analysis

Open the interactive Marimo notebook:

```bash
marimo edit notebooks/analysis.py
```

All parameters (smoothing window, ARIMA order, forecast horizon, etc.) are exposed as sliders that propagate reactively through all sections.

## Repository layout

```
src/
  pipeline/   01_extract · 02_detect · 03_metrics · 04_timeseries
  analysis/   gap · lt_estimate · metrics · models
  utils/      fit_reader · zones
notebooks/    analysis.py  (Marimo)
outputs/
  runs/raw/         per-activity Parquet files
  timeseries/       per_run_metrics.csv · stationarity.csv
```
