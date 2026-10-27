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
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit

# Fit bounds: (log qi, Di monthly nominal, b). qi's upper bound is set per fit.
# b is capped at 1.2 by default. A cap is a judgement call -- without one a
# dip-and-rebound history can drive b to ~2 and a decades-long flat tail. 1.2
# was picked from 1.0 / 1.2 / 1.5 / 2.0 by hindcasting these wells
# (scripts/backtest.py): it beat 1.5 at 12 and 24 months (level at 6 and 18),
# while 1.0 under-forecast 24 months by ~12% (median).
DI_BOUNDS = (0.005, 0.80)
B_MIN = 0.05
B_CAP = 1.2
B_BOUNDS = (B_MIN, B_CAP)

# Robust loss for the log-space fit. A single partial month (a restart month
# with a few barrels, a curtailed month) is several log units off trend and
# would otherwise dominate an ordinary least-squares fit. soft_l1 behaves like
# least squares for residuals well under F_SCALE (in natural-log units, ~35%)
# and grows only linearly beyond it.
LOSS = "soft_l1"
F_SCALE = 0.3

# Solver tolerances (curve_fit's ftol, xtol and gtol). At the default 1e-8 the
# fit stops with parameters settled only to ~1e-4 along flat directions of the
# cost, so the same data could fit differently in the 4th digit on another
# CPU; at 1e-12 results agree to ~1e-6 or better.
FIT_TOL = 1e-12

MAX_FORECAST_MONTHS = 600  # default cap on forecast length

# A month producing under 10% of the fitted rate is a partial month (restart,
# curtailment). The robust loss already discounts it in the fit; it is also
# left out of the log-space R² so it cannot decide the QC verdict alone.
PARTIAL_MONTH_RATIO = 0.10

# Producing-time clock. Monthly volumes carry no days-on, so how much of a
# month a well produced is read from its ratio r = q / reference (see
# fit_decline). Under CURTAIL_RAMP[0] the month is taken to have produced at
# trend for r of the month, and its volume says nothing more about the trend,
# so it is left out of the fit; over CURTAIL_RAMP[1] it is a full month (the
# shortfall is noise); in between, its clock share rises linearly from
# CURTAIL_RAMP[0] to 1 and its fit weight from 0 to 1. No month flips from in
# to out at a threshold -- a month of a few barrels counts almost exactly like
# a zero month -- so a small change in one month's volume makes a small change
# in the fit. CURTAIL_RATIO, the ramp's midpoint, is the threshold the QC
# counts of curtailed months use.
CURTAIL_RAMP = (0.3, 0.7)
CURTAIL_RATIO = 0.5
# The first pass's reference: the median of the REF_HALF_WINDOW months either
# side (zeros included), but at least every other month's volume decayed by
# REF_DECAY per month away from it, and at least REF_FLOOR of the peak month.
# The decayed envelope matters only inside shut-ins, where the median is zero:
# it keeps a small test volume there reading as a small share of a month
# instead of a large share of the floor (which flipped the fit between two
# solutions).
REF_HALF_WINDOW = 3
REF_DECAY = 0.8
REF_FLOOR = 0.01


# ----------------------------------------------------------------------------
# Rate functions
# ----------------------------------------------------------------------------
def hyperbolic_rate(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Pure hyperbolic Arps rate. t in months, Di nominal monthly."""
    t = np.asarray(t, dtype=float)
    return qi / (1.0 + b * Di * t) ** (1.0 / b)


def _switch_time(qi, Di, b, Dmin):
    """Month t* at which instantaneous decline D(t) drops to Dmin (0 if Di <= Dmin).
    Works elementwise on arrays of parameters."""
    Di, b = np.asarray(Di, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Di > Dmin, (Di / Dmin - 1.0) / (b * Di), 0.0)


def terminal_rate(t: np.ndarray, qi, Di, b, Dmin: float) -> np.ndarray:
    """Hyperbolic rate with terminal exponential decline at Dmin.

    Parameters may be arrays that broadcast against t, e.g. shape (n, 1) for an
    ensemble of n curves evaluated at the months in t.
    """
    t = np.asarray(t, dtype=float)
    t_star = _switch_time(qi, Di, b, Dmin)
    q_star = hyperbolic_rate(t_star, qi, Di, b)
    tail = q_star * np.exp(-Dmin * np.maximum(t - t_star, 0.0))
    return np.where(t <= t_star, hyperbolic_rate(t, qi, Di, b), tail)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_hyperbolic(
    t: np.ndarray, q: np.ndarray, qi_max: float | None = None, b_cap: float = B_CAP,
    weights: np.ndarray | None = None, p0: np.ndarray | None = None,
) -> dict:
    """Fit (qi, Di, b) to observed monthly rates via robust nonlinear least squares.

    t is (producing) months since the peak month, peak = 0. qi_max optionally
    caps the peak-month rate at the observed peak -- the fitted curve cannot
    start above what the well actually produced. weights (0-1 per month)
    scale each month's influence; months with weight 0 or no volume are left
    out. p0 optionally starts the solver at (log qi, Di, b).
    Returns dict with qi, Di (monthly nominal), b, Di_annual, r_squared,
    r_squared_log, n_partial_months, n_points, bounds_hit, b_at_cap, and a
    status flag. Falls back gracefully when the fit fails.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    w = np.ones_like(q) if weights is None else np.asarray(weights, dtype=float)
    mask = (q > 0) & (w > 0)
    t, q, w = t[mask], q[mask], w[mask]
    result = {"status": "failed", "n_points": int(len(t))}

    if len(t) < 12:
        result["status"] = "insufficient_data"
        return result

    # Fit in log space: stabilizes the fit across orders of magnitude and
    # matches how engineers eyeball declines on semi-log plots.
    log_q = np.log(q)

    def model_log(tt, log_qi, Di, b):
        return np.log(hyperbolic_rate(tt, np.exp(log_qi), Di, b))

    qi_hi = np.log(qi_max) if qi_max else np.log(1e7)
    lower = np.array([np.log(10.0), DI_BOUNDS[0], B_MIN])
    upper = np.array([qi_hi, DI_BOUNDS[1], b_cap])
    if p0 is None:
        p0 = [np.log(q.max()), 0.15, min(1.0, b_cap)]
    else:  # keep a warm start strictly inside the bounds
        p0 = np.clip(p0, lower + 1e-9 * (upper - lower), upper - 1e-9 * (upper - lower))
    try:
        popt, _ = curve_fit(
            model_log, t, log_q, p0=p0, bounds=(lower, upper), maxfev=20000,
            loss=LOSS, f_scale=F_SCALE, ftol=FIT_TOL, xtol=FIT_TOL, gtol=FIT_TOL,
            sigma=None if np.all(w == 1.0) else 1.0 / np.sqrt(w),
        )
        qi, Di, b = float(np.exp(popt[0])), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return result

    # Weighted R² in linear space and in the log space the fit works in. Linear
    # R² is dominated by the early high-rate months; log R² also scores the
    # late-life tail. Partial months are excluded from log R².
    q_hat = hyperbolic_rate(t, qi, Di, b)
    ss_res = float(np.sum(w * (q - q_hat) ** 2))
    ss_tot = float(np.sum(w * (q - np.average(q, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_log = log_q - np.log(q_hat)
    full_months = resid_log >= np.log(PARTIAL_MONTH_RATIO)
    lw, lq = w[full_months], log_q[full_months]
    ss_res_log = float(np.sum(lw * resid_log[full_months] ** 2))
    ss_tot_log = float(np.sum(lw * (lq - np.average(lq, weights=lw)) ** 2)) if len(lq) else 0.0
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


def month_weights(q: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Producing fraction (clock share) and fit weight of each month from its
    ratio to a reference rate; see CURTAIL_RAMP."""
    q = np.asarray(q, dtype=float)
    lo, hi = CURTAIL_RAMP
    ratio = np.where(q > 0, q / q_ref, 0.0)
    weight = np.clip((ratio - lo) / (hi - lo), 0.0, 1.0)
    frac = np.where(ratio <= lo, ratio, lo + (1.0 - lo) * weight)
    return frac, weight


def local_reference(q: np.ndarray) -> np.ndarray:
    """Each month's first-pass reference rate: the median of its neighbouring
    months (REF_HALF_WINDOW either side, zeros included), but no less than any
    other month's volume times REF_DECAY per month between them, nor REF_FLOOR
    of the peak. It depends on no fit and moves continuously with every
    month's volume."""
    q = np.asarray(q, dtype=float)
    if len(q) < 2:
        return q.copy()
    w = REF_HALF_WINDOW
    win = sliding_window_view(np.pad(q, w, constant_values=np.nan), 2 * w + 1)
    ref = np.nanmedian(np.delete(win, w, axis=1), axis=1)
    k = np.arange(len(q))
    decay = REF_DECAY ** np.abs(k[:, None] - k[None, :]).astype(float)
    np.fill_diagonal(decay, 0.0)
    envelope = (q[None, :] * decay).max(axis=1)
    return np.maximum(np.maximum(ref, envelope), REF_FLOOR * q.max())


def _clock(frac: np.ndarray) -> np.ndarray:
    """Producing time at the start of each month (first month = 0)."""
    return np.r_[0.0, np.cumsum(frac)[:-1]]


def fit_decline(q: np.ndarray, qi_max: float | None = None,
                b_cap: float = B_CAP) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit a well on producing time, in two passes.

    q holds one monthly volume per calendar month from the peak month onward,
    zeros included (shut-in months). Each pass reads every month's clock share
    and fit weight from its ratio to a reference rate (month_weights) and fits
    on that clock with those weights. The first pass's reference is the median
    of the neighbouring months (local_reference), which needs no fit; the
    second pass's is the first fit's trend. Two fixed passes rather than
    iterating to a self-consistent clock: iterating can have two
    self-consistent answers for the same data, and which one it reached could
    flip on a small change in one month.

    Returns the fit (fit_hyperbolic's dict plus producing_months -- producing
    time at the first forecast month, i.e. its start_t --, downtime_months,
    curtailed_months and curtailed_months_at_end, the months shut in or under
    CURTAIL_RATIO of the first pass's trend, and downweighted_months, the
    months the second fit gives a weight under 1, shut-ins included), the
    producing time at the start of each history month, and each month's fit
    weight.
    """
    q = np.asarray(q, dtype=float)
    producing = q > 0
    frac, weight = month_weights(q, local_reference(q))
    clock = _clock(frac)
    first = fit_hyperbolic(clock, q, qi_max=qi_max, b_cap=b_cap, weights=weight)
    if first["status"] != "ok":
        return first, clock, weight

    trend = hyperbolic_rate(clock, first["qi"], first["Di_monthly"], first["b"])
    ratio = np.where(producing, q / trend, 0.0)
    new_frac, new_weight = month_weights(q, trend)
    new_clock = _clock(new_frac)
    fit = fit_hyperbolic(new_clock, q, qi_max=qi_max, b_cap=b_cap, weights=new_weight,
                         p0=[np.log(first["qi"]), first["Di_monthly"], first["b"]])
    if fit["status"] == "ok":
        frac, weight, clock = new_frac, new_weight, new_clock
    else:  # too few weighted months left; keep the first pass
        fit = first

    curtailed = ratio < CURTAIL_RATIO   # zero months included
    fit["producing_months"] = float(frac.sum())
    fit["downtime_months"] = float(len(q) - frac.sum())
    # Shut-in months count with the rest, so a month of a few barrels counts
    # the same as a zero month (both are almost all downtime at weight 0).
    fit["curtailed_months"] = int(curtailed.sum())
    fit["downweighted_months"] = int((weight < 1).sum())
    fit["curtailed_months_at_end"] = (int(np.argmin(curtailed[::-1])) if not curtailed.all()
                                      else len(q))
    return fit, clock, weight


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
    remaining life. Each later month adds one month. The forecast stops before
    the first month under the limit, and after max_months even if the rate is
    still above the limit.
    """
    if int(max_months) <= 0:
        return np.array([])
    q = terminal_rate(float(start_t) + np.arange(int(max_months)), qi, Di, b, Dmin)
    return q[:months_above(q, econ_limit_bpm)]


def months_above(q: np.ndarray, limit: float) -> np.ndarray:
    """Number of leading months at or above the limit (along the last axis)."""
    q = np.asarray(q, dtype=float)
    return np.cumprod(q >= limit, axis=-1).sum(axis=-1)


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
