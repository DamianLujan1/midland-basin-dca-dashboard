"""Arps decline-curve models, fitting, and EUR calculation.

Model: hyperbolic decline with a terminal exponential decline once the
instantaneous nominal decline rate falls to Dmin (standard industry practice
to keep late-life forecasts from declining unrealistically slowly).

    q(t) = qi / (1 + b * Di * t) ** (1 / b)          for D(t) > Dmin
    q(t) = q* * exp(-Dmin * (t - t*))                for t > t*

where D(t) = Di / (1 + b * Di * t) and t* is the switch time.

Time convention: t is producing time in months since the peak-rate month, so
the peak month is t = 0 and qi is the modelled rate in the peak month. A well
does not deplete while it is shut in, so shut-in and curtailed months do not
advance t (see fit_decline); every forecast month counts as a full month.

All rates are monthly volumes (bbl/month); Di and Dmin are nominal monthly
decline rates unless suffixed _annual.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# Fit bounds: (log qi, Di monthly nominal, b). qi's upper bound is set per fit.
# b is capped at 1.5 by default -- an analog-based upper end for Midland Basin
# horizontals with a terminal decline, not a value fitted to this data; without
# a cap a dip-and-rebound history can drive b to ~2 and a decades-long flat tail.
DI_BOUNDS = (0.005, 0.80)
B_MIN = 0.05
B_CAP = 1.5
B_BOUNDS = (B_MIN, B_CAP)

# Robust loss for the log-space fit. A single partial month (a restart month
# with a few barrels, a curtailed month) is several log units off trend and
# would otherwise dominate an ordinary least-squares fit. soft_l1 behaves like
# least squares for residuals well under F_SCALE (in natural-log units, ~35%)
# and grows only linearly beyond it.
LOSS = "soft_l1"
F_SCALE = 0.3

MAX_FORECAST_MONTHS = 600  # default cap on forecast length

# A month producing under 10% of the fitted rate is a partial month (restart,
# curtailment). The robust loss already discounts it in the fit; it is also
# left out of the log-space R² so it cannot decide the QC verdict alone.
PARTIAL_MONTH_RATIO = 0.10

# Producing-time clock. Monthly volumes carry no days-on, so a month producing
# under CURTAIL_RATIO of the fitted trend is taken to have produced for only
# q / q_trend of the month (a zero month for none of it). Such a month's volume
# says nothing about the trend beyond that, so like a zero month it is left out
# of the fit. The fit and the clock are updated alternately until the clock
# moves by less than CLOCK_TOL months (at most MAX_PASSES passes).
CURTAIL_RATIO = 0.5
CLOCK_TOL = 1e-3
MAX_PASSES = 10


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
    t: np.ndarray, q: np.ndarray, qi_max: float | None = None, b_cap: float = B_CAP
) -> dict:
    """Fit (qi, Di, b) to observed monthly rates via robust nonlinear least squares.

    t is (producing) months since the peak month, peak = 0. qi_max optionally
    caps the peak-month rate at the observed peak -- the fitted curve cannot
    start above what the well actually produced.
    Returns dict with qi, Di (monthly nominal), b, Di_annual, r_squared,
    r_squared_log, n_partial_months, n_points, bounds_hit, b_at_cap, and a
    status flag. Falls back gracefully when the fit fails.
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

    p0 = [np.log(q.max()), 0.15, min(1.0, b_cap)]
    qi_hi = np.log(qi_max) if qi_max else np.log(1e7)
    bounds = ([np.log(10.0), DI_BOUNDS[0], B_MIN],
              [qi_hi, DI_BOUNDS[1], b_cap])
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

    # A parameter stuck on a bound means the data wanted more than the model
    # allows. b at its cap is the deliberate cap, reported separately.
    bounds_hit = [name for name, hit in (
        ("Di", np.isclose(Di, DI_BOUNDS[0], rtol=1e-4) or np.isclose(Di, DI_BOUNDS[1], rtol=1e-4)),
        ("b", np.isclose(b, B_MIN, rtol=1e-4)),
    ) if hit]

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
        "b_at_cap": bool(np.isclose(b, b_cap, rtol=1e-3)),
    }


def producing_clock(q: np.ndarray, q_trend: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimated producing fraction of each month, and producing time at the
    start of each month (months, first month = 0)."""
    q = np.asarray(q, dtype=float)
    ratio = np.where(q > 0, q / q_trend, 0.0)
    frac = np.where(ratio < CURTAIL_RATIO, np.clip(ratio, 0.0, 1.0), 1.0)
    return frac, np.r_[0.0, np.cumsum(frac)[:-1]]


def fit_decline(q: np.ndarray, qi_max: float | None = None,
                b_cap: float = B_CAP) -> tuple[dict, np.ndarray]:
    """Fit a well on producing time.

    q holds one monthly volume per calendar month from the peak month onward,
    zeros included (shut-in months). The first pass counts only zero months as
    downtime and fits every producing month. Each later pass re-estimates the
    producing fraction of every month from the previous fit (producing_clock),
    and fits only full months, until the clock settles. If the set of curtailed
    months cycles (a month sitting at the threshold flips back and forth), the
    state of the cycle with the most full months is kept, i.e. the borderline
    month counts as producing.

    The returned fit, clock and producing_months always come from the same
    pass. Returns the fit (fit_hyperbolic's dict plus producing_months --
    producing time at the first forecast month, i.e. its start_t --,
    downtime_months, curtailed_months and curtailed_months_at_end) and the
    producing time at the start of each history month.
    """
    q = np.asarray(q, dtype=float)
    producing = q > 0

    def fit_state(full, frac):
        clock = np.r_[0.0, np.cumsum(frac)[:-1]]
        return fit_hyperbolic(clock, np.where(full, q, 0.0), qi_max=qi_max, b_cap=b_cap), clock

    full, frac = producing.copy(), producing.astype(float)
    fit, clock = fit_state(full, frac)
    if fit["status"] != "ok":
        return fit, clock
    states = [(full, frac)]
    for _ in range(MAX_PASSES - 1):
        trend = hyperbolic_rate(clock, fit["qi"], fit["Di_monthly"], fit["b"])
        new_frac, _ = producing_clock(q, trend)
        if np.allclose(new_frac, frac, atol=CLOCK_TOL):
            break
        new_full = (new_frac >= 1.0) & producing
        seen = [i for i, (m, _) in enumerate(states[:-1]) if np.array_equal(m, new_full)]
        if seen and not np.array_equal(new_full, full):  # curtailed set cycling: settle it
            full, frac = max(states[seen[0]:] + [(new_full, new_frac)],
                             key=lambda s: int(s[0].sum()))
            new_fit, new_clock = fit_state(full, frac)
            if new_fit["status"] == "ok":
                fit, clock = new_fit, new_clock
            break
        new_fit, new_clock = fit_state(new_full, new_frac)
        if new_fit["status"] != "ok":  # too few full months left; keep the last fit
            break
        full, frac, fit, clock = new_full, new_frac, new_fit, new_clock
        states.append((full, frac))

    fit["producing_months"] = float(frac.sum())
    fit["downtime_months"] = float(len(q) - frac.sum())
    fit["curtailed_months"] = int((producing & (frac < 1.0)).sum())
    at_end = frac >= 1.0
    fit["curtailed_months_at_end"] = int(np.argmax(at_end[::-1])) if at_end.any() else len(q)
    return fit, clock


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
    start_t: float = 1.0,
) -> np.ndarray:
    """Monthly forecast rates until the economic limit (bbl/month).

    start_t is the decline time of the first forecast month (0 = peak month);
    pass the producing time elapsed at the end of history to forecast only the
    remaining life. Each later month adds one month. The forecast stops after
    max_months even if the rate is still above the limit.
    """
    rates = []
    t_star = _switch_time(qi, Di, b, Dmin)
    q_star = float(hyperbolic_rate(np.array([t_star]), qi, Di, b)[0])
    for i in range(int(max_months)):
        m = float(start_t) + i
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
    start_t: float = 1.0,
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
