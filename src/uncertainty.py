"""P90 / P50 / P10 forecast ranges from a calibrated ensemble of decline curves.

Every ensemble member is an Arps curve within the fit bounds, so every
scenario is a physically possible decline. A member combines:

1. Fit uncertainty: a block bootstrap of the well's own fit. The log
   residuals of the months the fit gives (nearly) full weight are resampled in
   blocks of BLOCK_MONTHS consecutive months, which keeps their month-to-month
   correlation, added back to the fitted curve at every month the fit uses,
   and refitted with the same clock, weights, bounds and robust loss.
   Down-weighted (curtailed) months are not in the pool: their shortfall is
   downtime, not noise.
2. Level uncertainty: the member's rate is scaled by exp(LEVEL_SIGMA * z) with
   z standard normal (stratified, so the factors are evenly spread). The
   bootstrap alone covers fewer than half of the held-out volumes in the
   hindcast: they miss the fitted trend for reasons one well's residuals do
   not capture (curtailment, offset frac hits, operating changes).

LEVEL_SIGMA is calibrated by scripts/calibrate_ranges.py: in a hindcast that
hides each well's last 6, 12 or 18 reported months and leaves the well out of
the calibration, the ensemble's 10-90% range covers 80% of the held-out
volumes. Over those 6-18 months the ensemble is no sharper than a band of
constant relative width; what it adds is a spread that grows with the
forecast horizon as the uncertainty in b and Di compounds. Ranges for
remaining reserves over decades are that extrapolation and cannot be checked
against this data: the level factor applies to the whole remaining life.

P50 is the deterministic fit. P90 and P10 are the ensemble's 10th and 90th
percentiles (each metric separately), and never cross P50. With BOOT_SAMPLES
members their Monte Carlo error is a few percent (see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.arps import fit_hyperbolic, hyperbolic_rate, terminal_rate
from src.economics import Economics, cash_flow, present_value

BOOT_SAMPLES = 1000
BLOCK_MONTHS = 3
SPARE_DRAWS = 64     # replacement draws per month for one outside the residual pool
# A month's residual joins the pool as its fit weight rises across POOL_WEIGHT
# (60-70% of trend, see src.arps.CURTAIL_RAMP): there, 2-3 standard deviations
# of full-month noise below trend, curtailment and noise overlap.
POOL_WEIGHT = (0.75, 1.0)
LEVEL_SIGMA = 0.225  # calibrated by scripts/calibrate_ranges.py (0.225)
LOW, HIGH = 10, 90   # percentiles reported as P90 (low) and P10 (high)
ENSEMBLE_COLS = ["api8", "member", "qi", "Di_monthly", "b"]


def block_indices(n: int, samples: int, rng: np.random.Generator,
                  block: int = BLOCK_MONTHS) -> np.ndarray:
    """(samples, n) indices of a moving-block bootstrap of a length-n series."""
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=(samples, -(-n // block)))
    return (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :n]


def resample_months(pool: np.ndarray, samples: int, rng: np.random.Generator,
                    block: int = BLOCK_MONTHS, spares: int = SPARE_DRAWS) -> np.ndarray:
    """(samples, n) source month for each of a well's n history months.

    pool (0-1 per month) is the probability that a draw landing on the month
    keeps it. Blocks of consecutive months are drawn over the whole history; a
    drawn month that is not kept is replaced by the first kept month among
    further uniform draws. Every draw has a fixed shape for a given n, so a
    small change in one month's pool weight changes only a few draws, not the
    whole sample.
    """
    pool = np.clip(np.asarray(pool, dtype=float), 0.0, 1.0)
    n = len(pool)
    src = block_indices(n, samples, rng, block)
    miss = rng.random((samples, n)) >= pool[src]
    for _ in range(spares):
        cand = rng.integers(0, n, size=(samples, n))
        take = miss & (rng.random((samples, n)) < pool[cand])
        src[take] = cand[take]
        miss &= ~take
    return src


def stratified_normal(m: int, rng: np.random.Generator) -> np.ndarray:
    """m standard-normal scores at evenly spaced quantiles, in random order."""
    return rng.permutation(norm.ppf((np.arange(m) + 0.5) / m))


def members(q: np.ndarray, clock: np.ndarray, weight: np.ndarray, fit: dict,
            qi_max: float, b_cap: float, seed, samples: int = BOOT_SAMPLES) -> tuple:
    """Bootstrap refits and their level-factor scores: ((m, 3) array of (qi,
    Di_monthly, b), (m,) z), m <= samples (a failed refit is dropped with its z).

    The residual pool is the months the fit gives (nearly) full weight; see
    POOL_WEIGHT. A down-weighted month sits below trend because of downtime, so
    its residual is an offset, not noise: resampled into other months it would
    read as a real dip and pull every member low. Each resampled series is the
    fitted curve plus pool residuals at every month the fit uses, refitted with
    the same clock and weights. (Keeping the down-weighted months' own
    residuals instead scored worse in the hindcast: its ensemble median sat
    above 58% of held-out volumes, against 52% for this.)
    """
    rng = np.random.default_rng(seed)
    q = np.asarray(q, dtype=float)
    used = (q > 0) & (weight > 0)
    lo, hi = POOL_WEIGHT
    pool = np.where(used, np.clip((weight - lo) / (hi - lo), 0.0, 1.0), 0.0)
    log_hat = np.log(hyperbolic_rate(clock, fit["qi"], fit["Di_monthly"], fit["b"]))
    resid = np.zeros(len(q))
    resid[used] = np.log(q[used]) - log_hat[used]
    src = resample_months(pool, samples, rng)
    z = stratified_normal(samples, rng)
    t, w, base = clock[used], weight[used], log_hat[used]
    p0 = [np.log(fit["qi"]), fit["Di_monthly"], fit["b"]]
    params, keep = [], []
    for i, s in enumerate(src[:, used]):
        f = fit_hyperbolic(t, np.exp(base + resid[s]), qi_max=qi_max, b_cap=b_cap,
                           weights=w, p0=p0)
        if f["status"] == "ok":
            params.append((f["qi"], f["Di_monthly"], f["b"]))
            keep.append(i)
    return np.array(params).reshape(-1, 3), z[keep]


def ensemble(q: np.ndarray, clock: np.ndarray, weight: np.ndarray, fit: dict,
             qi_max: float, b_cap: float, seed: int, samples: int = BOOT_SAMPLES,
             sigma: float = LEVEL_SIGMA) -> np.ndarray:
    """(m, 3) ensemble of (qi, Di_monthly, b); the level factor is folded into
    qi. Values are rounded to 10 significant digits, the precision
    results/forecast_ensemble.csv keeps, so a saved ensemble reproduces the run
    that wrote it exactly."""
    params, z = members(q, clock, weight, fit, qi_max, b_cap, seed, samples)
    params[:, 0] *= np.exp(sigma * z)
    return np.array([float(f"{v:.10g}") for v in params.ravel()]).reshape(params.shape)


def ensemble_rates(params: np.ndarray, start_t: float, months: int, dmin: float) -> np.ndarray:
    """(m, months) forecast rates of each member from producing time start_t."""
    t = float(start_t) + np.arange(int(months))
    return terminal_rate(t[None, :], params[:, :1], params[:, 1:2], params[:, 2:3], dmin)


def ranges(params: np.ndarray, start_t: float, months: int, dmin: float,
           econ_limit_bpm: float, econ: Economics, gor: float) -> dict:
    """Low (P90) and high (P10) remaining volume and PV over the ensemble.
    Each member stops at the economic limit or after `months` months."""
    if months <= 0 or len(params) == 0:
        return {"remaining": (0.0, 0.0), "pv": (0.0, 0.0)}
    rates = ensemble_rates(params, start_t, months, dmin)
    alive = np.cumprod(rates >= econ_limit_bpm, axis=1).astype(bool)
    rates = np.where(alive, rates, 0.0)
    pv = present_value(cash_flow(rates, econ, gor), econ.discount)
    return {"remaining": tuple(np.percentile(rates.sum(axis=1), [LOW, HIGH])),
            "pv": tuple(np.percentile(pv, [LOW, HIGH]))}


def band(params: np.ndarray, start_t: float, months: int, dmin: float) -> tuple:
    """Pointwise P90 (low) and P10 (high) monthly rate over the ensemble."""
    rates = ensemble_rates(params, start_t, months, dmin)
    lo, hi = np.percentile(rates, [LOW, HIGH], axis=0)
    return lo, hi


def to_frame(ensembles: dict) -> pd.DataFrame:
    """Long table of every well's ensemble, for results/forecast_ensemble.csv."""
    frames = [pd.DataFrame({"api8": int(api8), "member": np.arange(len(p)), "qi": p[:, 0],
                            "Di_monthly": p[:, 1], "b": p[:, 2]})
              for api8, p in ensembles.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ENSEMBLE_COLS)


def from_frame(df: pd.DataFrame) -> dict:
    """Inverse of to_frame: {api8: (m, 3) array}."""
    return {int(api8): g.sort_values("member")[["qi", "Di_monthly", "b"]].to_numpy(float)
            for api8, g in df.groupby("api8")}
