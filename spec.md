# time-series-229: Pipeline Specification

**Goal**: Retrospective physiological analysis of the 8-month training block (Jun–Dec 21 2025) leading to a 2:30:52 marathon (Feb 14 2026). Identify which markers best explain the ability to sustain 3:33/km marathon pace, and which run types and training stimuli produced those adaptations — with a view to understanding the minimum effective dose to maintain that fitness going forward.

**Ground-truth anchor**: Lactate threshold test logged Apr 10 2019 — 3:57/km @ 175 bpm. All LT estimates in this pipeline should be cross-validated against this.

---

## Repository layout

```
time-series-229/
├── data/
│   ├── activities.zip          # Git LFS — raw .fit.gz files from Strava export
│   ├── strava_runs_clean.csv   # Git LFS — cleaned summary CSV
├── src/
│   ├── pipeline/
│   │   ├── 01_extract.py       # Step 1: FIT extraction + smoothing
│   │   ├── 02_detect.py        # Step 2: Run classification
│   │   ├── 03_metrics.py       # Step 3: Per-run metric computation
│   │   └── 04_timeseries.py    # Step 4: Metric time series assembly
│   ├── analysis/
│   │   ├── gap.py              # Grade Adjusted Pace (Minetti)
│   │   ├── lt_estimate.py      # Lactate threshold estimation
│   │   ├── metrics.py          # All physiological metrics library
│   │   └── models.py           # Time series modelling machinery
│   └── utils/
│       ├── fit_reader.py       # FIT file parsing utilities
│       └── zones.py            # HR and pace zone definitions
├── notebooks/
│   └── analysis.py             # Marimo notebook (Step 3 application)
├── outputs/
│   ├── runs/                   # Per-run metric CSVs
│   └── timeseries/             # Assembled metric time series
└── spec.md                     # This document
```

---

## Data scope

| Parameter | Value |
|---|---|
| Training block start | 2025-06-01 |
| Training block end | 2025-12-21 |
| Run types included | Interval runs + long runs (25–38 km) |
| Excluded | December altitude camp (Dec 22+), GPS-error runs |
| Source files | `activities.zip` → individual `.fit.gz` per activity |
| Summary metadata | `strava_runs_clean.csv` (activity_id, date, distance_km, etc.) |

---

## Step 1: FIT Extraction Pipeline (`src/pipeline/01_extract.py`)

### 1.1 File matching

Each row in `strava_runs_clean.csv` has a `Filename` column (e.g. `activities/1922813233.fit.gz`). Unzip `activities.zip` to a temp directory, then match each run in the training block by `activity_id` → filename. Only process runs where `Activity Type == Run` and `date` falls within the training block.

### 1.2 FIT parsing

Use `fitparse` (Python library) to parse each `.fit.gz` file. Extract the following fields from `record` messages at native resolution:

| FIT field | Description | Units |
|---|---|---|
| `timestamp` | UTC timestamp | datetime |
| `position_lat` / `position_long` | GPS coordinates | semicircles → degrees |
| `distance` | Cumulative distance | meters |
| `speed` | Instantaneous speed | m/s |
| `heart_rate` | HR | bpm |
| `cadence` | Running cadence (per foot) | spm → multiply by 2 for total |
| `altitude` | Elevation | meters |
| `enhanced_altitude` | Higher-res elevation if present | meters |
| `temperature` | Ambient temperature if recorded | °C |

If a field is absent for a given activity, record as `NaN` for that column. Do not drop the run — partial data is still usable for the metrics that are available.

### 1.3 Smoothing

Apply a **10-second rolling mean** to all continuous signals (speed, heart_rate, cadence, altitude). Use `pandas.DataFrame.rolling(window=10, min_periods=5, center=True).mean()`. This removes GPS noise and HRV jitter without destroying interval structure.

Store both raw and smoothed columns: `speed_raw`, `speed`, `hr_raw`, `heart_rate`, etc.

### 1.4 Gradient and Grade Adjusted Pace (GAP)

Compute instantaneous gradient from smoothed altitude and cumulative distance:

```python
gradient = delta_altitude / delta_distance  # dimensionless, e.g. 0.05 = 5%
gradient = gradient.clip(-0.45, 0.45)       # cap at ±45% — GPS noise artefact prevention
```

Apply the **Minetti et al. (2002)** metabolic cost function to convert raw pace to Grade Adjusted Pace.

### 1.5 Output format

Save each run as a Parquet file at `outputs/runs/raw/{activity_id}.parquet`.

---

## Step 2: Run Classification (`src/pipeline/02_detect.py`)

### 2.1 Long run detection

Flag any run with `distance_km >= 25.0` and `distance_km <= 38.0` as a long run.

### 2.2 Interval run detection

Detect via FIT-level pace zone analysis and hard/easy alternation detection (state machine, ≥3 intervals, each ≥60s in Z3/Z4, separated by ≥60s recovery).

### 2.3 Classification output

Save `outputs/run_manifest.csv`.

---

## Step 3: Metrics Library (`src/analysis/metrics.py`)

Metrics implemented:
- **LT estimation** (Dmax method, rolling estimate seeded from Apr 2019 anchor)
- **Efficiency Factor** (EF = GAP speed / HR)
- **Aerobic Decoupling** (Pa:HR)
- **Cardiac Drift Rate** (OLS slope of HR at constant GAP pace)
- **Recovery Rate Between Intervals** (exponential decay fit)
- **TRIMP + ATL/CTL/TSB** (Banister 1991)
- **Cadence-Pace Coupling** (Pearson correlation + CV)
- **Pace Variability Index** (long runs only)
- **HR Zone Distribution** (5 zones relative to LT HR)
- **Interval Quality Metrics** (pace consistency, HR progression, degradation)

---

## Step 4: Time Series Assembly (`src/pipeline/04_timeseries.py`)

Assemble all per-run metrics into `outputs/timeseries/per_run_metrics.csv`. Apply stationarity checks (ADF + KPSS) and flag non-stationary series.

---

## Analysis Machinery (`src/analysis/models.py`)

Models implemented:
- STL decomposition (EF and Pa:HR)
- ACF/PACF analysis
- ARIMA(p,d,q)
- VAR (EF, cardiac drift, CTL jointly)
- Rolling OLS feature importance
- Change point detection (ruptures, PELT)

---

## Marimo Notebook (`notebooks/analysis.py`)

8 interactive sections with configurable sliders:
0. Configuration
1. Training block overview (CTL/ATL/TSB, TID)
2. LT estimation tracker
3. EF and Pa:HR trend
4. Cardiac drift analysis
5. Interval session analysis
6. Cadence analysis
7. Multivariate modelling
8. Maintenance implications

---

## Research basis

| Metric | Source |
|---|---|
| Grade Adjusted Pace | Minetti et al. (2002) *J Appl Physiol* 93:1039 |
| Efficiency Factor / Pa:HR | Friel (2009) *The Triathlete's Training Bible* |
| TRIMP | Banister et al. (1991) *Fitness-Fatigue model* |
| Dmax LT estimation | Cheng et al. (1992) *Med Sci Sports Exerc* |
| Training intensity distribution | Seiler & Kjerland (2006) *Scand J Med Sci Sports* |
| Maintenance at reduced volume | Mujika & Padilla (2000) *Med Sci Sports Exerc* |
| Marathon training structure | Casado et al. (2023) *Int J Sports Physiol Perform* |
| Low-intensity training role | Muniz-Pumares et al. (2024), as reviewed in *Eur J Appl Physiol* (2025) |
