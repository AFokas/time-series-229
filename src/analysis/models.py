"""Time series modelling machinery.

Provides STL decomposition, ACF/PACF, ARIMA, VAR, rolling OLS, and
change point detection.  All functions are designed to be called from
the Marimo notebook with configurable parameters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import VAR
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
except ImportError as e:
    raise ImportError("statsmodels is required: pip install statsmodels") from e

try:
    import ruptures as rpt
except ImportError as e:
    raise ImportError("ruptures is required: pip install ruptures") from e


# ---------------------------------------------------------------------------
# Stationarity checks
# ---------------------------------------------------------------------------

def stationarity_check(series: pd.Series, alpha: float = 0.05) -> dict:
    """Run ADF and KPSS tests on a series.

    Returns a dict with: adf_stat, adf_pvalue, adf_stationary,
    kpss_stat, kpss_pvalue, kpss_stationary, is_stationary.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return {k: np.nan for k in [
            "adf_stat", "adf_pvalue", "adf_stationary",
            "kpss_stat", "kpss_pvalue", "kpss_stationary", "is_stationary"
        ]}

    adf_result = adfuller(clean, autolag="AIC")
    adf_stat, adf_p = float(adf_result[0]), float(adf_result[1])
    adf_stationary = adf_p < alpha

    try:
        kpss_result = kpss(clean, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_result[0]), float(kpss_result[1])
        kpss_stationary = kpss_p > alpha  # KPSS H0 = stationary
    except Exception:
        kpss_stat, kpss_p, kpss_stationary = np.nan, np.nan, None

    is_stationary = bool(adf_stationary and (kpss_stationary is True))

    return {
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_stationary": adf_stationary,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "kpss_stationary": kpss_stationary,
        "is_stationary": is_stationary,
    }


def make_stationary(series: pd.Series) -> tuple[pd.Series, str]:
    """Apply first-difference or log transform to make a series stationary.

    Returns (transformed_series, transformation_label).
    Tries: original → first diff → log → log + first diff.
    """
    result = stationarity_check(series)
    if result["is_stationary"]:
        return series, "none"

    diff1 = series.diff().dropna()
    r = stationarity_check(diff1)
    if r["is_stationary"]:
        return diff1, "diff"

    if (series > 0).all():
        log_s = np.log(series)
        r = stationarity_check(log_s)
        if r["is_stationary"]:
            return log_s, "log"
        log_diff = log_s.diff().dropna()
        r = stationarity_check(log_diff)
        if r["is_stationary"]:
            return log_diff, "log_diff"

    # Return first diff as best effort
    return diff1, "diff"


# ---------------------------------------------------------------------------
# STL Decomposition
# ---------------------------------------------------------------------------

def stl_decompose(series: pd.Series, period: int = 7) -> dict:
    """Apply STL decomposition to a time series.

    Parameters
    ----------
    series: pd.Series with datetime index (run dates)
    period: seasonal period in observations (default 7 = weekly cycle)

    Returns dict with trend, seasonal, resid components as pd.Series.
    """
    clean = series.dropna()
    if len(clean) < 2 * period + 1:
        return {"trend": pd.Series(dtype=float), "seasonal": pd.Series(dtype=float), "resid": pd.Series(dtype=float)}

    stl = STL(clean, period=period, robust=True)
    res = stl.fit()
    return {
        "trend": pd.Series(res.trend, index=clean.index),
        "seasonal": pd.Series(res.seasonal, index=clean.index),
        "resid": pd.Series(res.resid, index=clean.index),
    }


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def compute_acf_pacf(series: pd.Series, nlags: int = 30) -> dict:
    """Compute ACF and PACF for a series.

    Returns dict with acf_values, pacf_values, acf_confint, pacf_confint.
    """
    clean = series.dropna()
    if len(clean) < nlags + 2:
        nlags = max(1, len(clean) // 2 - 1)

    acf_vals, acf_ci = acf(clean, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_ci = pacf(clean, nlags=nlags, alpha=0.05)

    return {
        "acf_values": acf_vals,
        "pacf_values": pacf_vals,
        "acf_confint": acf_ci,
        "pacf_confint": pacf_ci,
        "lags": np.arange(len(acf_vals)),
    }


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def fit_arima(
    series: pd.Series,
    order: tuple[int, int, int] = (2, 0, 1),
    forecast_horizon: int = 4,
) -> dict:
    """Fit ARIMA(p,d,q) to a series and generate a forecast.

    Returns dict with: model_result, aic, bic, forecast, conf_int, residuals.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"error": "insufficient data"}

    try:
        model = ARIMA(clean, order=order)
        result = model.fit()
        forecast = result.forecast(steps=forecast_horizon)
        conf_int = result.get_forecast(steps=forecast_horizon).conf_int()
        return {
            "model_result": result,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "forecast": forecast,
            "conf_int": conf_int,
            "residuals": result.resid,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def fit_var(
    df: pd.DataFrame,
    maxlags: int = 4,
    forecast_horizon: int = 4,
) -> dict:
    """Fit a VAR model to a multivariate DataFrame.

    Parameters
    ----------
    df: DataFrame with columns as endogenous variables (no NaN rows).
    maxlags: maximum lag order to consider (AIC selection).

    Returns dict with: model_result, selected_lag, aic, irf (impulse response), forecast.
    """
    clean = df.dropna()
    if len(clean) < maxlags + 10:
        return {"error": "insufficient data"}

    try:
        model = VAR(clean)
        result = model.fit(maxlags=maxlags, ic="aic")
        lag = result.k_ar

        irf = result.irf(periods=forecast_horizon * 2)
        forecast = result.forecast(clean.values[-lag:], steps=forecast_horizon)

        return {
            "model_result": result,
            "selected_lag": lag,
            "aic": float(result.aic),
            "irf": irf,
            "forecast": pd.DataFrame(forecast, columns=clean.columns),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Rolling OLS feature importance
# ---------------------------------------------------------------------------

def rolling_ols(
    y: pd.Series,
    X: pd.DataFrame,
    window_weeks: int = 6,
    obs_per_week: float = 1.5,
) -> pd.DataFrame:
    """Rolling OLS: y ~ X over a sliding window.

    Parameters
    ----------
    y:             target series (e.g. EF)
    X:             feature DataFrame (e.g. CTL, drift slope, ...)
    window_weeks:  rolling window size in weeks
    obs_per_week:  approximate observations per week (used to convert weeks → rows)

    Returns a DataFrame of rolling coefficients indexed by date.
    """
    window = max(int(window_weeks * obs_per_week), 10)
    combined = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(combined) < window:
        return pd.DataFrame()

    rows = []
    for end in range(window, len(combined) + 1):
        chunk = combined.iloc[end - window: end]
        y_w = chunk["y"].values
        X_w = add_constant(chunk.drop(columns="y").values)
        try:
            res = OLS(y_w, X_w).fit()
            coef_dict = {"date": chunk.index[-1]}
            for i, col in enumerate(chunk.drop(columns="y").columns):
                coef_dict[col] = float(res.params[i + 1])  # skip intercept
            rows.append(coef_dict)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Change point detection
# ---------------------------------------------------------------------------

def detect_changepoints(
    series: pd.Series,
    n_bkps: Optional[int] = None,
    penalty: float = 10.0,
) -> list[int]:
    """Detect structural breaks using the PELT algorithm (ruptures library).

    Parameters
    ----------
    series:  time series (NaN values are forward-filled for detection only)
    n_bkps:  if provided, find exactly this many breakpoints; otherwise use penalty
    penalty: penalty for PELT (higher → fewer breakpoints); only used if n_bkps is None

    Returns a list of breakpoint indices (positions in the series).
    """
    clean = series.dropna().ffill().values.reshape(-1, 1)
    if len(clean) < 4:
        return []

    algo = rpt.Pelt(model="rbf").fit(clean)

    try:
        if n_bkps is not None:
            bkps = rpt.Binseg(model="rbf").fit(clean).predict(n_bkps=n_bkps)
        else:
            bkps = algo.predict(pen=penalty)
        # ruptures returns breakpoints as end-exclusive indices; exclude the last (len)
        return [b for b in bkps if b < len(clean)]
    except Exception:
        return []
