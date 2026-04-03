"""Marimo notebook: Retrospective physiological analysis.

Run with:
    marimo edit notebooks/analysis.py
"""

import marimo as mo
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# SECTION 0: Configuration sliders
# ---------------------------------------------------------------------------

_s0_header = mo.md("## Section 0 — Configuration")

smoothing_window = mo.ui.slider(5, 30, value=10, step=1, label="Smoothing window (s)")
lt_lookback_weeks = mo.ui.slider(2, 8, value=4, step=1, label="LT lookback (weeks)")
ef_zone_ceiling = mo.ui.slider(0.8, 1.0, value=0.95, step=0.01, label="EF zone ceiling (× LT HR)")
decoupling_threshold = mo.ui.slider(2.0, 15.0, value=5.0, step=0.5, label="Pa:HR threshold (%)")
trimp_atl_days = mo.ui.slider(5, 14, value=7, step=1, label="ATL window (days)")
trimp_ctl_days = mo.ui.slider(28, 56, value=42, step=1, label="CTL window (days)")
interval_min_duration = mo.ui.slider(30, 120, value=60, step=5, label="Min interval duration (s)")
gap_tolerance_ms = mo.ui.slider(0.1, 0.5, value=0.25, step=0.05, label="GAP pace tolerance (min/km)")
long_run_min_km = mo.ui.slider(20, 30, value=25, step=1, label="Long run min (km)")
long_run_max_km = mo.ui.slider(35, 45, value=38, step=1, label="Long run max (km)")
model_order_ar = mo.ui.slider(1, 6, value=2, step=1, label="ARIMA p (AR order)")
model_order_ma = mo.ui.slider(0, 4, value=1, step=1, label="ARIMA q (MA order)")
forecast_horizon = mo.ui.slider(1, 12, value=4, step=1, label="Forecast horizon (weeks)")

_config_panel = mo.vstack([
    mo.md("### Data & Signal"),
    mo.hstack([smoothing_window, lt_lookback_weeks, ef_zone_ceiling, decoupling_threshold]),
    mo.md("### Training Load"),
    mo.hstack([trimp_atl_days, trimp_ctl_days]),
    mo.md("### Run Classification"),
    mo.hstack([interval_min_duration, gap_tolerance_ms, long_run_min_km, long_run_max_km]),
    mo.md("### Modelling"),
    mo.hstack([model_order_ar, model_order_ma, forecast_horizon]),
])

# ---------------------------------------------------------------------------
# Data loading (reactive: re-runs whenever outputs change on disk)
# ---------------------------------------------------------------------------

METRICS_PATH = Path("outputs/timeseries/per_run_metrics.csv")
MANIFEST_PATH = Path("outputs/run_manifest.csv")
LT_TRACKER_PATH = Path("outputs/lt_tracker.csv")

def _load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_PATH, parse_dates=["run_date"])
    df = df.sort_values("run_date").reset_index(drop=True)
    return df

def _load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MANIFEST_PATH, parse_dates=["run_date"])

def _load_lt_tracker() -> pd.DataFrame:
    if not LT_TRACKER_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LT_TRACKER_PATH, parse_dates=["date"])

metrics = _load_metrics()
manifest = _load_manifest()
lt_tracker = _load_lt_tracker()

_no_data_msg = mo.callout(
    mo.md("**No data yet.** Run the pipeline steps first:\n"
          "```\npython -m src.pipeline.01_extract\n"
          "python -m src.pipeline.02_detect\n"
          "python -m src.pipeline.03_metrics\n"
          "python -m src.pipeline.04_timeseries\n```"),
    kind="warn",
) if metrics.empty else None

# ---------------------------------------------------------------------------
# SECTION 1: Training block overview
# ---------------------------------------------------------------------------

_s1_header = mo.md("## Section 1 — Training Block Overview")

def _section1(df: pd.DataFrame) -> mo.Html:
    if df.empty:
        return mo.md("_No data._")

    # Timeline coloured by run type
    color_map = {"long": "#2196F3", "interval": "#F44336", "other": "#9E9E9E"}
    fig1 = px.scatter(
        df,
        x="run_date",
        y="distance_km",
        color="run_type",
        color_discrete_map=color_map,
        size="trimp",
        hover_data=["activity_id", "trimp", "estimated_lt_hr"],
        title="Training block — runs by type and distance",
        labels={"run_date": "Date", "distance_km": "Distance (km)", "run_type": "Type"},
    )

    # CTL / ATL / TSB chart
    load_cols = [c for c in ["atl", "ctl", "tsb"] if c in df.columns]
    fig2 = go.Figure()
    if load_cols:
        colors = {"ctl": "#4CAF50", "atl": "#F44336", "tsb": "#2196F3"}
        for col in load_cols:
            fig2.add_trace(go.Scatter(
                x=df["run_date"], y=df[col], mode="lines",
                name=col.upper(), line=dict(color=colors.get(col, "grey"))
            ))
        fig2.update_layout(title="CTL / ATL / TSB", xaxis_title="Date", yaxis_title="Training load")

    # Weekly TID heatmap
    zone_cols = [c for c in ["z1_pct", "z2_pct", "z3_pct", "z4_pct", "z5_pct"] if c in df.columns]
    if zone_cols and "run_date" in df.columns:
        df2 = df.copy()
        df2["week"] = df2["run_date"].dt.to_period("W").astype(str)
        weekly_zones = df2.groupby("week")[zone_cols].mean().reset_index()
        fig3 = px.imshow(
            weekly_zones[zone_cols].T,
            x=weekly_zones["week"],
            y=["Z1", "Z2", "Z3", "Z4", "Z5"],
            color_continuous_scale="RdYlGn_r",
            title="Weekly training intensity distribution (% time in zone)",
            labels={"color": "% time"},
        )
    else:
        fig3 = go.Figure()

    return mo.vstack([
        mo.ui.plotly(fig1),
        mo.ui.plotly(fig2),
        mo.ui.plotly(fig3),
    ])

_section1_view = _section1(metrics)

# ---------------------------------------------------------------------------
# SECTION 2: LT estimation tracker
# ---------------------------------------------------------------------------

_s2_header = mo.md("## Section 2 — Lactate Threshold Estimation")

def _section2(lt_df: pd.DataFrame) -> mo.Html:
    if lt_df.empty:
        return mo.md("_No LT data._")

    fig = go.Figure()
    valid = lt_df[~lt_df["suspect"].fillna(False)]
    suspect = lt_df[lt_df["suspect"].fillna(False)]

    fig.add_trace(go.Scatter(
        x=valid["date"], y=valid["lt_hr"], mode="lines+markers",
        name="LT HR (valid)", line=dict(color="#4CAF50"),
        error_y=None,
    ))
    if not suspect.empty:
        fig.add_trace(go.Scatter(
            x=suspect["date"], y=suspect["lt_hr"], mode="markers",
            name="LT HR (suspect)", marker=dict(color="orange", symbol="x", size=10),
        ))
    # Ground-truth anchor
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp("2019-04-10")], y=[175.0], mode="markers",
        name="Anchor (Apr 2019 lab test)",
        marker=dict(color="red", symbol="star", size=14),
    ))
    fig.update_layout(title="Rolling LT HR estimate", xaxis_title="Date", yaxis_title="LT HR (bpm)")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=valid["date"], y=valid["lt_pace_min_per_km"], mode="lines+markers",
        name="LT pace (min/km)", line=dict(color="#2196F3"),
    ))
    fig2.add_trace(go.Scatter(
        x=[pd.Timestamp("2019-04-10")], y=[3.95], mode="markers",
        name="Anchor (3:57 min/km)",
        marker=dict(color="red", symbol="star", size=14),
    ))
    fig2.update_layout(
        title="Rolling LT pace estimate",
        xaxis_title="Date", yaxis_title="LT pace (min/km)",
        yaxis=dict(autorange="reversed"),
    )
    return mo.vstack([mo.ui.plotly(fig), mo.ui.plotly(fig2)])

_section2_view = _section2(lt_tracker)

# ---------------------------------------------------------------------------
# SECTION 3: EF and Pa:HR trend
# ---------------------------------------------------------------------------

_s3_header = mo.md("## Section 3 — Efficiency Factor & Aerobic Decoupling")

def _section3(df: pd.DataFrame, threshold: float) -> mo.Html:
    if df.empty or "ef" not in df.columns:
        return mo.md("_No EF data._")

    from src.analysis.models import stl_decompose

    long_runs = df[df["run_type"] == "long"].copy()
    iv_runs = df[df["run_type"] == "interval"].copy()

    fig = go.Figure()
    if not long_runs.empty and long_runs["ef"].notna().any():
        fig.add_trace(go.Scatter(
            x=long_runs["run_date"], y=long_runs["ef"], mode="markers",
            name="EF (long runs)", marker=dict(color="#2196F3"),
        ))
        # STL trend overlay
        ef_series = long_runs.set_index("run_date")["ef"].dropna()
        if len(ef_series) >= 7:
            decomp = stl_decompose(ef_series, period=min(7, len(ef_series) // 2))
            if not decomp["trend"].empty:
                fig.add_trace(go.Scatter(
                    x=decomp["trend"].index, y=decomp["trend"],
                    mode="lines", name="STL trend", line=dict(color="darkblue", dash="dash"),
                ))
    if not iv_runs.empty and iv_runs["ef"].notna().any():
        fig.add_trace(go.Scatter(
            x=iv_runs["run_date"], y=iv_runs["ef"], mode="markers",
            name="EF (interval sessions)", marker=dict(color="#F44336", symbol="diamond"),
        ))
    fig.update_layout(title="Efficiency Factor over training block", xaxis_title="Date", yaxis_title="EF (m/s per bpm)")

    fig2 = go.Figure()
    if "pa_hr_decoupling" in df.columns:
        fig2.add_trace(go.Scatter(
            x=df["run_date"], y=df["pa_hr_decoupling"], mode="lines+markers",
            name="Pa:HR (%)", line=dict(color="#FF9800"),
        ))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold ({threshold}%)")
    fig2.update_layout(title="Aerobic Decoupling (Pa:HR)", xaxis_title="Date", yaxis_title="Pa:HR (%)")

    return mo.vstack([mo.ui.plotly(fig), mo.ui.plotly(fig2)])

_section3_view = _section3(metrics, decoupling_threshold.value)

# ---------------------------------------------------------------------------
# SECTION 4: Cardiac drift analysis
# ---------------------------------------------------------------------------

_s4_header = mo.md("## Section 4 — Cardiac Drift")

def _section4(df: pd.DataFrame) -> mo.Html:
    if df.empty or "cardiac_drift_slope" not in df.columns:
        return mo.md("_No cardiac drift data._")

    fig = px.scatter(
        df[df["cardiac_drift_slope"].notna()],
        x="run_date", y="cardiac_drift_slope",
        color="temperature" if "temperature" in df.columns else None,
        color_continuous_scale="RdYlBu_r",
        title="Cardiac drift slope over training block",
        labels={"cardiac_drift_slope": "Drift slope (bpm/min)", "run_date": "Date"},
        hover_data=["run_type", "distance_km"],
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey")

    fig2 = px.histogram(
        df[df["cardiac_drift_slope"].notna()],
        x="cardiac_drift_slope",
        color="run_type",
        nbins=20,
        title="Drift slope distribution by run type",
        labels={"cardiac_drift_slope": "Drift slope (bpm/min)"},
    )
    return mo.vstack([mo.ui.plotly(fig), mo.ui.plotly(fig2)])

_section4_view = _section4(metrics)

# ---------------------------------------------------------------------------
# SECTION 5: Interval session analysis
# ---------------------------------------------------------------------------

_s5_header = mo.md("## Section 5 — Interval Session Analysis")

_interval_runs = manifest[manifest["run_type"] == "interval"]["activity_id"].tolist() if not manifest.empty else []

activity_selector = mo.ui.dropdown(
    options=_interval_runs,
    label="Select interval session",
) if _interval_runs else None

def _section5(activity_id: str | None) -> mo.Html:
    if not activity_id:
        return mo.md("_Select an interval session above._")
    pf = Path("outputs/runs/raw") / f"{activity_id}.parquet"
    if not pf.exists():
        return mo.md(f"_Parquet file not found for {activity_id}._")

    df = pd.read_parquet(pf)

    from src.analysis.lt_estimate import LTTracker
    from src.utils.zones import ZoneBoundaries
    from src.pipeline.02_detect import detect_intervals

    tracker = LTTracker()
    run_date = pd.to_datetime(df["run_date"].iloc[0], utc=True) if "run_date" in df.columns else pd.Timestamp.now(tz="UTC")
    lt = tracker.get_lt_for_date(run_date)
    zones = ZoneBoundaries.from_lt(lt.lt_hr, lt.lt_pace)
    _, intervals = detect_intervals(df, zones)

    fig = go.Figure()
    if "elapsed_s" in df.columns:
        if "gap_pace_min_per_km" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_s"] / 60, y=df["gap_pace_min_per_km"],
                mode="lines", name="GAP pace (min/km)", yaxis="y1",
                line=dict(color="#2196F3"),
            ))
        if "heart_rate" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_s"] / 60, y=df["heart_rate"],
                mode="lines", name="HR (bpm)", yaxis="y2",
                line=dict(color="#F44336"),
            ))
        # Highlight interval segments
        for iv in intervals:
            fig.add_vrect(
                x0=iv["interval_start_s"] / 60,
                x1=iv["interval_end_s"] / 60,
                fillcolor="rgba(255,152,0,0.2)",
                layer="below", line_width=0,
            )

    fig.update_layout(
        title=f"Interval session: {activity_id}",
        xaxis_title="Elapsed (min)",
        yaxis=dict(title="GAP pace (min/km)", autorange="reversed"),
        yaxis2=dict(title="HR (bpm)", overlaying="y", side="right"),
        legend=dict(x=0, y=1),
    )

    # Per-session summary table
    if intervals:
        iv_df = pd.DataFrame(intervals)
        iv_df["interval_avg_gap_pace"] = iv_df["interval_avg_gap_pace"].round(2)
        iv_df["interval_avg_hr"] = iv_df["interval_avg_hr"].round(1)
        return mo.vstack([mo.ui.plotly(fig), mo.ui.table(iv_df)])
    return mo.ui.plotly(fig)

_section5_view = mo.vstack([
    activity_selector or mo.md("_No interval runs found._"),
    _section5(activity_selector.value if activity_selector else None),
])

# ---------------------------------------------------------------------------
# SECTION 6: Cadence analysis
# ---------------------------------------------------------------------------

_s6_header = mo.md("## Section 6 — Cadence Analysis")

def _section6(df: pd.DataFrame) -> mo.Html:
    if df.empty or "cadence_pace_corr" not in df.columns:
        return mo.md("_No cadence data._")

    fig1 = px.scatter(
        df[df["cadence_pace_corr"].notna()],
        x="run_date", y="cadence_pace_corr", color="run_type",
        title="Cadence-pace coupling (Pearson r)",
        labels={"cadence_pace_corr": "Pearson r", "run_date": "Date"},
    )
    fig1.add_hline(y=0, line_dash="dot")

    fig2 = px.scatter(
        df[df["cadence_at_marathon_pace"].notna()],
        x="run_date", y="cadence_at_marathon_pace",
        title="Cadence at marathon pace (3:33/km GAP)",
        labels={"cadence_at_marathon_pace": "Cadence (spm)", "run_date": "Date"},
    )

    fig3 = px.scatter(
        df[df["cadence_cv"].notna()],
        x="run_date", y="cadence_cv", color="run_type",
        title="Cadence CV (fatigue proxy)",
        labels={"cadence_cv": "CV", "run_date": "Date"},
    )

    return mo.vstack([mo.ui.plotly(fig1), mo.ui.plotly(fig2), mo.ui.plotly(fig3)])

_section6_view = _section6(metrics)

# ---------------------------------------------------------------------------
# SECTION 7: Multivariate modelling
# ---------------------------------------------------------------------------

_s7_header = mo.md("## Section 7 — Multivariate Modelling")

def _section7(df: pd.DataFrame, ar: int, ma: int, horizon: int) -> mo.Html:
    if df.empty:
        return mo.md("_No data._")

    from src.analysis.models import (
        detect_changepoints,
        fit_arima,
        fit_var,
        rolling_ols,
    )

    panels = []

    # ARIMA on EF (long runs)
    long_ef = df[df["run_type"] == "long"].set_index("run_date")["ef"].dropna()
    if len(long_ef) >= 10:
        arima_result = fit_arima(long_ef, order=(ar, 0, ma), forecast_horizon=horizon)
        if "forecast" in arima_result:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=long_ef.index, y=long_ef, mode="lines+markers", name="EF (observed)"))
            fc_index = pd.date_range(long_ef.index[-1], periods=horizon + 1, freq="7D")[1:]
            fig.add_trace(go.Scatter(x=fc_index, y=arima_result["forecast"], mode="lines",
                                     name=f"ARIMA({ar},0,{ma}) forecast", line=dict(dash="dash")))
            ci = arima_result["conf_int"]
            fig.add_trace(go.Scatter(
                x=list(fc_index) + list(fc_index[::-1]),
                y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
                fill="toself", fillcolor="rgba(33,150,243,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="95% CI",
            ))
            fig.update_layout(title=f"EF forecast — ARIMA({ar},0,{ma})", xaxis_title="Date", yaxis_title="EF")
            panels.append(mo.ui.plotly(fig))
            panels.append(mo.md(f"AIC: **{arima_result['aic']:.1f}** | BIC: **{arima_result['bic']:.1f}**"))

    # VAR: EF + cardiac drift + CTL
    var_cols = [c for c in ["ef", "cardiac_drift_slope", "ctl"] if c in df.columns]
    if len(var_cols) == 3:
        var_df = df.set_index("run_date")[var_cols].dropna()
        if len(var_df) >= 15:
            var_result = fit_var(var_df, maxlags=4, forecast_horizon=horizon)
            if "irf" in var_result:
                irf = var_result["irf"]
                # Show impulse response of EF to a CTL shock
                try:
                    irf_df = pd.DataFrame(
                        irf.irfs[:, var_cols.index("ef"), var_cols.index("ctl")],
                        columns=["EF response to CTL shock"],
                    )
                    fig_irf = px.line(irf_df, title="VAR impulse response: EF response to CTL shock",
                                      labels={"index": "Weeks ahead", "value": "EF change"})
                    panels.append(mo.ui.plotly(fig_irf))
                except Exception:
                    pass

    # Rolling OLS coefficients
    feature_cols = [c for c in ["ctl", "cardiac_drift_slope", "recovery_decay_constant",
                                  "pvi", "z1_pct", "z3_pct"] if c in df.columns]
    if "ef" in df.columns and len(feature_cols) >= 2:
        y = df.set_index("run_date")["ef"]
        X = df.set_index("run_date")[feature_cols]
        rolling_coefs = rolling_ols(y, X, window_weeks=6)
        if not rolling_coefs.empty:
            fig_ols = px.line(rolling_coefs, title="Rolling OLS coefficients (EF ~ training variables)",
                               labels={"value": "Coefficient", "index": "Date", "variable": "Feature"})
            panels.append(mo.ui.plotly(fig_ols))

    # Change points on EF
    if "ef" in df.columns and df["ef"].notna().sum() >= 8:
        ef_series = df.set_index("run_date")["ef"]
        bkps = detect_changepoints(ef_series, penalty=10.0)
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(x=ef_series.index, y=ef_series, mode="lines+markers", name="EF"))
        for bkp in bkps:
            if bkp < len(ef_series):
                fig_cp.add_vline(x=ef_series.index[bkp], line_dash="dash", line_color="red",
                                  annotation_text="Change point")
        fig_cp.update_layout(title="EF series with change point detection", xaxis_title="Date", yaxis_title="EF")
        panels.append(mo.ui.plotly(fig_cp))

    return mo.vstack(panels) if panels else mo.md("_Insufficient data for modelling._")

_section7_view = _section7(metrics, model_order_ar.value, model_order_ma.value, forecast_horizon.value)

# ---------------------------------------------------------------------------
# SECTION 8: Maintenance implications
# ---------------------------------------------------------------------------

_s8_header = mo.md("## Section 8 — Maintenance Implications")

def _section8(df: pd.DataFrame) -> mo.Html:
    if df.empty or "ef" not in df.columns:
        return mo.md("_No data._")

    # Use final 6 weeks of data
    if "run_date" not in df.columns:
        return mo.md("_No run_date column._")
    final_6w_cutoff = df["run_date"].max() - pd.Timedelta(weeks=6)
    recent = df[df["run_date"] >= final_6w_cutoff]

    # Correlation of each metric with EF in the final 6 weeks
    metric_cols = [c for c in [
        "ctl", "cardiac_drift_slope", "recovery_decay_constant",
        "pvi", "z1_pct", "z3_pct", "cadence_pace_corr", "pa_hr_decoupling",
        "mean_cadence", "trimp",
    ] if c in recent.columns and recent[c].notna().sum() >= 3]

    corr_rows = []
    for col in metric_cols:
        combined = recent[["ef", col]].dropna()
        if len(combined) >= 3:
            r = combined.corr().loc["ef", col]
            corr_rows.append({"metric": col, "correlation_with_ef": round(r, 3)})

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows).sort_values("correlation_with_ef", ascending=False)
        top5 = corr_df.head(5)

        # Training pattern in weeks where all top metrics were highest (above median)
        top_metrics = top5["metric"].tolist()
        high_performance = recent.copy()
        for m in top_metrics:
            if m in high_performance.columns:
                med = high_performance[m].median()
                high_performance = high_performance[high_performance[m] >= med]

        maintenance_summary = {}
        if not high_performance.empty:
            if "run_type" in high_performance.columns:
                maintenance_summary["run_type_distribution"] = high_performance["run_type"].value_counts().to_dict()
            if "distance_km" in high_performance.columns:
                maintenance_summary["avg_distance_km"] = round(high_performance["distance_km"].mean(), 1)
            if "trimp" in high_performance.columns:
                maintenance_summary["avg_trimp"] = round(high_performance["trimp"].mean(), 1)
            for z in ["z1_pct", "z2_pct", "z3_pct"]:
                if z in high_performance.columns:
                    maintenance_summary[z] = round(high_performance[z].mean(), 1)

        return mo.vstack([
            mo.md("### Top 5 metrics correlated with EF (final 6 weeks)"),
            mo.ui.table(top5),
            mo.md("### Training profile during peak-EF weeks"),
            mo.ui.table(pd.DataFrame([maintenance_summary])),
            mo.md(
                "> **Minimum effective dose hypothesis**: maintain the run types and intensity "
                "distribution observed above to preserve current aerobic fitness.\n\n"
                "> Cross-reference with Mujika & Padilla (2000): frequency and intensity "
                "are more important than volume during maintenance phases."
            ),
        ])
    return mo.md("_Insufficient data for maintenance analysis._")

_section8_view = _section8(metrics)

# ---------------------------------------------------------------------------
# Notebook layout
# ---------------------------------------------------------------------------

mo.vstack([
    mo.md("# Time Series 229 — Physiological Analysis"),
    mo.md("> **Marathon**: 2:30:52 · Feb 14 2026 · 3:33/km · Training block Jun–Dec 21 2025"),
    _no_data_msg or mo.md(""),
    _config_panel,
    mo.divider(),
    _s1_header, _section1_view,
    mo.divider(),
    _s2_header, _section2_view,
    mo.divider(),
    _s3_header, _section3_view,
    mo.divider(),
    _s4_header, _section4_view,
    mo.divider(),
    _s5_header, _section5_view,
    mo.divider(),
    _s6_header, _section6_view,
    mo.divider(),
    _s7_header, _section7_view,
    mo.divider(),
    _s8_header, _section8_view,
])
