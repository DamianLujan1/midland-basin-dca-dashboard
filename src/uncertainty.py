"""P90 / P50 / P10 forecast ranges from a calibrated ensemble of decline curves.

Every ensemble member is an Arps curve within the fit bounds, so every
scenario is a physically possible decline. A member combines:

1. Fit uncertainty: a block bootstrap of the well's own fit. The second-pass
   fit's log residuals (standardized by the fit weights) are resampled in
   blocks of BLOCK_MONTHS consecutive months, which keeps their month-to-month
   correlation, added back to the fitted curve and refitted with the same
   clock, weights, bounds and robust loss.
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
percentiles (each metric separately), and never cross P50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.arps import fit_hyperbolic, hyperbolic_rate, terminal_rate
from src.economics import Economics, cash_flow, present_value

BOOT_SAMPLES = 200
BLOCK_MONTHS = 3
LEVEL_SIGMA = 0.21   # calibrated by scripts/calibrate_ranges.py (0.203, rounded up)
LOW, HIGH = 10, 90   # percentiles reported as P90 (low) and P10 (high)
ENSEMBLE_COLS = ["api8", "member", "qi", "Di_monthly", "b"]


def block_indices(n: int, samples: int, rng: np.random.Generator,
                  block: int = BLOCK_MONTHS) -> np.ndarray:
    """(samples, n) indices of a moving-block bootstrap of a length-n series."""
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=(samples, -(-n // block)))
    return (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :n]


def bootstrap_fits(q: np.ndarray, clock: np.ndarray, weight: np.ndarray, fit: dict,
                   qi_max: float, b_cap: float, rng: np.random.Generator,
                   samples: int = BOOT_SAMPLES) -> np.ndarray:
    """(m, 3) array of (qi, Di_monthly, b) refitted to block-resampled residuals;
    m <= samples (refits that fail are dropped)."""
    q = np.asarray(q, dtype=float)
    used = (q > 0) & (weight > 0)
    t, w = clock[used], weight[used]
    log_hat = np.log(hyperbolic_rate(t, fit["qi"], fit["Di_monthly"], fit["b"]))
    std_resid = (np.log(q[used]) - log_hat) * np.sqrt(w)
    p0 = [np.log(fit["qi"]), fit["Di_monthly"], fit["b"]]
    out = []
    for idx in block_indices(len(t), samples, rng):
        y = np.exp(log_hat + std_resid[idx] / np.sqrt(w))
        f = fit_hyperbolic(t, y, qi_max=qi_max, b_cap=b_cap, weights=w, p0=p0)
        if f["status"] == "ok":
            out.append((f["qi"], f["Di_monthly"], f["b"]))
    return np.array(out).reshape(-1, 3)


def level_factors(m: int, rng: np.random.Generator, sigma: float = LEVEL_SIGMA) -> np.ndarray:
    """m lognormal rate multipliers at evenly spaced quantiles, in random order."""
    return np.exp(sigma * rng.permutation(norm.ppf((np.arange(m) + 0.5) / m)))


def ensemble(q: np.ndarray, clock: np.ndarray, weight: np.ndarray, fit: dict,
             qi_max: float, b_cap: float, seed: int,
             samples: int = BOOT_SAMPLES) -> np.ndarray:
    """(m, 3) ensemble of (qi, Di_monthly, b); the level factor is folded into qi."""
    rng = np.random.default_rng(seed)
    params = bootstrap_fits(q, clock, weight, fit, qi_max, b_cap, rng, samples)
    params[:, 0] *= level_factors(len(params), rng)
    return params


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
