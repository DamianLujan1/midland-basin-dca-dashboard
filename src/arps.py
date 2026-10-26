"""Arps decline-curve models, fitting, and EUR calculation.

Model: hyperbolic decline with a terminal exponential decline once the
instantaneous nominal decline rate falls to Dmin (standard industry practice
to keep late-life forecasts from declining unrealistically slowly).

    q(t) = qi / (1 + b * Di * t) ** (1 / b)          for D(t) > Dmin
    q(t) = q* * exp(-Dmin * (t - t*))                for t > t*

where D(t) = Di / (1 + b * Di * t) and t* is the switch time.

Time convention: t is calendar months since the peak-rate month, so the peak
month is t = 0 and qi is the modelled rate in the peak month.

All rates are monthly volumes (bbl/month); Di and Dmin are nominal monthly
decline rates unless suffixed _annual.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# Fit bounds: (log qi, Di monthly nominal, b). qi's upper bound is set per fit.
DI_BOUNDS = (0.005, 0.80)
B_BOUNDS = (0.05, 2.0)

# Robust loss for the log-space fit. A single partial month (a restart month
# with a few barrels, a curtailed month) is several log units off trend and
# would otherwise dominate an ordinary least-squares fit. soft_l1 behaves like
# least squares for residuals well under F_SCALE (in natural-log units, ~35%)
# and grows only linearly beyond it.
LOSS = "soft_l1"
F_SCALE = 0.3

MAX_FORECAST_MONTHS = 600  # 50-year cap on remaining life

# A month producing under 10% of the fitted rate is a partial month (restart,
# curtailment). The robust loss already discounts it in the fit; it is also
# left out of the log-space R² so it cannot decide the QC verdict alone.
PARTIAL_MONTH_RATIO = 0.10


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
    t: np.ndarray, q: np.ndarray, qi_max: float | None = None
) -> dict:
    """Fit (qi, Di, b) to observed monthly rates via robust nonlinear least squares.

    t is months since the peak month (peak = 0); gaps from excluded months are
    kept, not closed up. qi_max optionally caps the peak-month rate at the
    observed peak -- the fitted curve cannot start above what the well
    actually produced.
    Returns dict with qi, Di (monthly nominal), b, Di_annual, r_squared,
    r_squared_log, n_partial_months, n_points, bounds_hit, and a status flag. Falls back gracefully when the
    fit fails.
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
    bounds = ([np.log(10.0), DI_BOUNDS[0], B_BOUNDS[0]],
              [qi_hi, DI_BOUNDS[1], B_BOUNDS[1]])
    try:
        popt, _ = curve_fit(
            model_log, t, log_q, p0=p0, bounds=bounds, maxfev=20000,
            loss=LOSS, f_scale=F_SCALE,
        )
        qi, Di, b = float(np.exp(popt[0])), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return result

    # R² in linear space (all fitted points) and in the log space the fit works
    # in. Linear R² is dominated by the early high-rate months; log R² also
    # scores the late-life tail. Partial months are excluded from log R².
    q_hat = hyperbolic_rate(t, qi, Di, b)
    ss_res = float(np.sum((q - q_hat) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_log = log_q - np.log(q_hat)
    full_months = resid_log >= np.log(PARTIAL_MONTH_RATIO)
    ss_res_log = float(np.sum(resid_log[full_months] ** 2))
    ss_tot_log = float(np.sum((log_q[full_months] - log_q[full_months].mean()) ** 2))
    r2_log = 1.0 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 0.0

    bounds_hit = [
        name for name, val, (lo, hi) in (("Di", Di, DI_BOUNDS), ("b", b, B_BOUNDS))
        if np.isclose(val, lo, rtol=1e-4) or np.isclose(val, hi, rtol=1e-4)
    ]

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
        "r_squared_log": float(r2_log),
        "n_partial_months": int((~full_months).sum()),
        "bounds_hit": ",".join(bounds_hit),
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
    max_months: int = MAX_FORECAST_MONTHS,
    start_t: int = 1,
) -> np.ndarray:
    """Monthly forecast rates until the economic limit (bbl/month).

    start_t is the first forecast month in peak-relative time (0 = peak
    month). Pass the months-since-peak at end of history plus one to forecast
    only the remaining life, not already-produced months. The forecast stops
    after max_months even if the rate is still above the limit.
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
    max_months: int = MAX_FORECAST_MONTHS,
    start_t: int = 1,
) -> dict:
    """EUR = cumulative produced + forecast remaining to economic limit.

    forecast_capped is True when the forecast hit max_months while still above
    the economic limit, i.e. remaining_bbl is truncated by the time cap.
    """
    fc = forecast_to_limit(qi, Di, b, Dmin, econ_limit_bpm, max_months, start_t)
    remaining = float(fc.sum())
    return {
        "eur_bbl": float(cum_produced_bbl + remaining),
        "remaining_bbl": remaining,
        "cum_produced_bbl": float(cum_produced_bbl),
        "forecast_months": int(len(fc)),
        "forecast_capped": bool(len(fc) == max_months),
    }
