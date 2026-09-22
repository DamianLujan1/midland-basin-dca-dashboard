"""Arps decline-curve models, fitting, and EUR calculation.

Model: hyperbolic decline with a terminal exponential decline once the
instantaneous nominal decline rate falls to Dmin (standard industry practice
to keep late-life forecasts from declining unrealistically slowly).

    q(t) = qi / (1 + b * Di * t) ** (1 / b)          for D(t) > Dmin
    q(t) = q* * exp(-Dmin * (t - t*))                for t > t*

where D(t) = Di / (1 + b * Di * t) and t* is the switch time.

All rates are monthly volumes (bbl/month); Di and Dmin are nominal monthly
decline rates unless suffixed _annual.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# Rate functions
# ----------------------------------------------------------------------------
def hyperbolic_rate(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Pure hyperbolic Arps rate. t in months, Di nominal monthly."""
    t = np.asarray(t, dtype=float)
    return qi / (1.0 + b * Di * t) ** (1.0 / b)


def _switch_time(qi: float, Di: float, b: float, Dmin: float) -> float:
    """Month t* at which instantaneous decline D(t) drops to Dmin."""
    if Di <= Dmin:
        return 0.0
    return (Di / Dmin - 1.0) / (b * Di)


def terminal_rate(
    t: np.ndarray, qi: float, Di: float, b: float, Dmin: float
) -> np.ndarray:
    """Hyperbolic rate with terminal exponential decline at Dmin."""
    t = np.asarray(t, dtype=float)
    t_star = _switch_time(qi, Di, b, Dmin)
    q_star = hyperbolic_rate(np.array([t_star]), qi, Di, b)[0]
    q = np.empty_like(t)
    hyp = t <= t_star
    q[hyp] = hyperbolic_rate(t[hyp], qi, Di, b)
    q[~hyp] = q_star * np.exp(-Dmin * (t[~hyp] - t_star))
    return q


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_hyperbolic(
    t: np.ndarray, q: np.ndarray, Dmin: float = 0.05 / 12, qi_max: float | None = None
) -> dict:
    """Fit (qi, Di, b) to observed monthly rates via nonlinear least squares.

    qi_max optionally caps the initial rate at the observed peak rate --
    the fitted initial rate cannot exceed what the well actually produced.
    Returns dict with qi, Di (monthly nominal), b, Di_annual, r_squared,
    n_points, and a status flag. Falls back gracefully when the fit fails.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = q > 0
    t, q = t[mask], q[mask]
    result = {"status": "failed", "n_points": int(len(t))}

    if len(t) < 12:
        result["status"] = "insufficient_data"
        return result

    # Fit in log space: stabilizes the fit across orders of magnitude and
    # matches how engineers eyeball declines on semi-log plots.
    log_q = np.log(q)

    def model_log(tt, log_qi, Di, b):
        return np.log(hyperbolic_rate(tt, np.exp(log_qi), Di, b))

    p0 = [np.log(q.max()), 0.15, 1.0]
    qi_hi = np.log(qi_max) if qi_max else np.log(1e7)
    bounds = ([np.log(10.0), 0.005, 0.05], [qi_hi, 0.80, 2.0])
    try:
        popt, _ = curve_fit(
            model_log, t, log_q, p0=p0, bounds=bounds, maxfev=20000
        )
        qi, Di, b = float(np.exp(popt[0])), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return result

    q_hat = hyperbolic_rate(t, qi, Di, b)
    ss_res = float(np.sum((q - q_hat) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Nominal annual decline, the industry-standard quoting (Di_monthly * 12).
    Di_annual = float(Di * 12.0)
    return {
        "status": "ok",
        "n_points": int(len(t)),
        "qi": qi,
        "Di_monthly": Di,
        "Di_annual": Di_annual,
        "b": b,
        "r_squared": float(r2),
    }


# ----------------------------------------------------------------------------
# Forecasting / EUR
# ----------------------------------------------------------------------------
def forecast_to_limit(
    qi: float,
    Di: float,
    b: float,
    Dmin: float = 0.05 / 12,
    econ_limit_bpm: float = 150.0,
    max_months: int = 600,
    start_t: int = 1,
) -> np.ndarray:
    """Monthly forecast rates until the economic limit (bbl/month).

    start_t is the first forecast month in peak-relative time (1 = peak
    month). Pass the months-since-peak at end of history to forecast only
    the remaining life, not already-produced months.
    """
    rates = []
    t_star = _switch_time(qi, Di, b, Dmin)
    q_star = float(hyperbolic_rate(np.array([t_star]), qi, Di, b)[0])
    for m in range(int(start_t), int(start_t) + max_months):
        if m <= t_star:
            qm = float(hyperbolic_rate(np.array([m]), qi, Di, b)[0])
        else:
            qm = q_star * np.exp(-Dmin * (m - t_star))
        if qm < econ_limit_bpm:
            break
        rates.append(qm)
    return np.array(rates)


def eur_bbl(
    qi: float,
    Di: float,
    b: float,
    cum_produced_bbl: float,
    Dmin: float = 0.05 / 12,
    econ_limit_bpm: float = 150.0,
    start_t: int = 1,
) -> dict:
    """EUR = cumulative produced + forecast remaining to economic limit."""
    remaining = float(
        forecast_to_limit(qi, Di, b, Dmin, econ_limit_bpm, start_t=start_t).sum()
    )
    return {
        "eur_bbl": float(cum_produced_bbl + remaining),
        "remaining_bbl": remaining,
        "cum_produced_bbl": float(cum_produced_bbl),
    }
