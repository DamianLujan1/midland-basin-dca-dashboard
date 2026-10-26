"""Hindcast backtest of the decline fit (prints a Markdown table).

For each holdout H, every well is refitted with its last H reported months
hidden, and the fitted curve's volume over those months is compared with what
the well actually produced (producing months only). The current method
(src.arps.fit_decline: producing-time clock, b <= 1.5) is compared with the
earlier one-curve fit in calendar time with b up to 2.0.

Run:  python scripts/backtest.py   (from the repo root)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arps import F_SCALE, LOSS, fit_decline, hyperbolic_rate  # noqa: E402
from src.data_loader import fitting_series, load_production  # noqa: E402
from src.pipeline import DATA  # noqa: E402

HOLDOUTS = (6, 12, 18)


def calendar_fit(series):
    """Earlier method: positive months on calendar time since peak, b <= 2.0.
    Returns (qi, Di, b, t of the first held-out month)."""
    pos = series[series["oil_bbl"] > 0]
    if len(pos) < 12:
        return None
    t, q = pos["t"].to_numpy(float), pos["oil_bbl"].to_numpy(float)

    def model(tt, log_qi, Di, b):
        return np.log(hyperbolic_rate(tt, np.exp(log_qi), Di, b))

    try:
        p, _ = curve_fit(model, t, np.log(q), p0=[np.log(q.max()), 0.15, 1.0],
                         bounds=([np.log(10), 0.005, 0.05], [np.log(q.max()), 0.80, 2.0]),
                         maxfev=20000, loss=LOSS, f_scale=F_SCALE)
    except (RuntimeError, ValueError):
        return None
    return float(np.exp(p[0])), float(p[1]), float(p[2]), float(series["t"].iloc[-1] + 1)


def current_fit(series):
    fit, _ = fit_decline(series["oil_bbl"].to_numpy(), qi_max=float(series["oil_bbl"].max()))
    if fit["status"] != "ok":
        return None
    return fit["qi"], fit["Di_monthly"], fit["b"], fit["producing_months"]


def errors(prod, method, holdout):
    """Held-out volume error per well: forecast / actual - 1."""
    out = []
    for api8, w in prod.groupby("api8"):
        cut = w["month_on_production"].max() - holdout
        series = fitting_series(w[w["month_on_production"] <= cut], api8)
        test = w[(w["month_on_production"] > cut) & (w["oil_bbl"] > 0)]
        fit = method(series) if len(series) else None
        if fit is None or test.empty:
            continue
        qi, Di, b, t_next = fit
        # Forecast months count as full producing months from the first held-out month.
        t = t_next + (test["month_on_production"] - (cut + 1)).to_numpy(float)
        out.append((api8, hyperbolic_rate(t, qi, Di, b).sum() / test["oil_bbl"].sum() - 1))
    return dict(out)


def main() -> None:
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    rng = np.random.default_rng(0)
    print("| Holdout | Wells | Mean abs. volume error, earlier → current | "
          "Paired change (90% CI) | Median bias, earlier → current |")
    print("|---|---|---|---|---|")
    for h in HOLDOUTS:
        old, new = errors(prod, calendar_fit, h), errors(prod, current_fit, h)
        wells = sorted(set(old) & set(new))
        e_old = np.array([old[w] for w in wells])
        e_new = np.array([new[w] for w in wells])
        d = np.abs(e_new) - np.abs(e_old)
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [5, 95])
        print(f"| {h} months | {len(wells)} | {np.abs(e_old).mean():.1%} → "
              f"{np.abs(e_new).mean():.1%} | {d.mean():+.1%} ({lo:+.1%} to {hi:+.1%}) | "
              f"{np.median(e_old):+.1%} → {np.median(e_new):+.1%} |")


if __name__ == "__main__":
    main()
