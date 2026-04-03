"""Grade Adjusted Pace using the Minetti et al. (2002) metabolic cost model.

Reference: Minetti AE, Moia C, Roi GS, Susta D, Ferretti G (2002).
Energy cost of walking and running at extreme uphill and downhill slopes.
J Appl Physiol 93:1039-1046.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.zones import speed_to_pace_series


def minetti_gap(speed_ms: pd.Series | np.ndarray, gradient: pd.Series | np.ndarray) -> pd.Series:
    """Convert running speed + gradient to metabolic-equivalent flat speed (GAP speed).

    The Minetti polynomial gives the energy cost of running (J/kg/m) as a
    function of gradient.  Dividing by the flat-ground cost gives a multiplier
    that is applied to speed to obtain the equivalent flat pace.

    Parameters
    ----------
    speed_ms:   instantaneous speed in m/s (smoothed)
    gradient:   dimensionless slope (delta_altitude / delta_distance),
                already clipped to ±0.45 before calling this function.

    Returns
    -------
    gap_speed:  pd.Series of GAP speed in m/s, clipped to [0.5, 10.0]
    """
    g = np.asarray(gradient, dtype=float)
    speed = np.asarray(speed_ms, dtype=float)

    # Minetti energy cost curve (J/kg/m)
    C_slope = (
        155.4 * g**5
        - 30.4 * g**4
        - 43.3 * g**3
        + 46.3 * g**2
        + 19.5 * g
        + 3.6
    )
    C_flat = 3.6  # energy cost on flat ground (J/kg/m)

    gap_speed = speed * (C_slope / C_flat)
    gap_speed = np.clip(gap_speed, 0.5, 10.0)

    if isinstance(speed_ms, pd.Series):
        return pd.Series(gap_speed, index=speed_ms.index)
    return pd.Series(gap_speed)


def compute_gradient(altitude: pd.Series, distance_m: pd.Series) -> pd.Series:
    """Compute instantaneous gradient from smoothed altitude and cumulative distance.

    gradient = delta_altitude / delta_distance, clipped to ±0.45.
    The first row is set to 0 (no previous point).
    """
    delta_alt = altitude.diff().fillna(0.0)
    delta_dist = distance_m.diff().fillna(0.0)

    # Avoid division by zero for stationary points
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.where(delta_dist > 0.1, delta_alt / delta_dist, 0.0)

    grad = np.clip(grad, -0.45, 0.45)
    return pd.Series(grad, index=altitude.index)


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gradient`, `gap_speed_ms`, and `gap_pace_min_per_km` columns to a run DataFrame.

    Expects smoothed `altitude` and `distance_m` columns, and smoothed `speed` (m/s).
    Operates in-place and returns the DataFrame.
    """
    df["gradient"] = compute_gradient(df["altitude_m"], df["distance_m"])
    df["gap_speed_ms"] = minetti_gap(df["speed"], df["gradient"])
    df["gap_pace_min_per_km"] = speed_to_pace_series(df["gap_speed_ms"])
    return df
