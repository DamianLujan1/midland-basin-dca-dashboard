"""Hindcast backtest of the decline fit (prints Markdown tables).

For each holdout H, every well is refitted with its last H reported months
hidden, and the fitted curve's volume over those months is compared with what
the well actually produced (producing months only). The variants separate the
two parts of the current method (src.arps.fit_decline) and show the b-cap
choice:

    earlier        one curve on calendar time, b up to 2.0 (the first version)
    calendar 1.2   calendar time, b <= 1.2
    clock 2.0      producing-time clock, b up to 2.0
    clock 1.5      producing-time clock, b <= 1.5
    clock 1.0      producing-time clock, b <= 1.0
    current        producing-time clock, b <= 1.2

Every method is scored on the same wells for each holdout: those every method
can fit (a fit needs 12 usable months, so a short or curtailed window drops
out). The variants and their settings were chosen on these same 30 wells, so
the confidence intervals are optimistic, and the windows overlap (only 14
wells have enough history for the 24-month holdout).

Run:  python scripts/backtest.py   (from the repo root)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arps import B_CAP, fit_decline, fit_hyperbolic, hyperbolic_rate  # noqa: E402
from src.data_loader import fitting_series, load_production  # noqa: E402
from src.pipeline import DATA  # noqa: E402

HOLDOUTS = (6, 12, 18, 24)


def calendar(b_cap):
    """One curve on calendar months since the peak, positive months only."""
    def method(series):
        fit = fit_hyperbolic(series["t"].to_numpy(float), series["oil_bbl"].to_numpy(),
                             qi_max=float(series["oil_bbl"].max()), b_cap=b_cap)
        if fit["status"] != "ok":
            return None
        return fit["qi"], fit["Di_monthly"], fit["b"], float(series["t"].iloc[-1] + 1)
    return method


def producing(b_cap):
    def method(series):
        fit, _, _ = fit_decline(series["oil_bbl"].to_numpy(),
                                qi_max=float(series["oil_bbl"].max()), b_cap=b_cap)
        if fit["status"] != "ok":
            return None
        return fit["qi"], fit["Di_monthly"], fit["b"], fit["producing_months"]
    return method


METHODS = {
    "earlier": calendar(2.0),
    "calendar 1.2": calendar(B_CAP),
    "clock 2.0": producing(2.0),
    "clock 1.5": producing(1.5),
    "clock 1.0": producing(1.0),
    "current": producing(B_CAP),
}


def errors(prod, method, holdout):
    """Held-out volume error per well: forecast / actual - 1."""
    out = {}
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
        out[api8] = hyperbolic_rate(t, qi, Di, b).sum() / test["oil_bbl"].sum() - 1
    return out


def paired(a, b, rng):
    """Mean change in absolute error b - a over common wells, with a 90% bootstrap CI."""
    wells = sorted(set(a) & set(b))
    d = np.abs([b[w] for w in wells]) - np.abs([a[w] for w in wells])
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [5, 95])
    return f"{d.mean():+.1%} ({lo:+.1%} to {hi:+.1%})"


def main() -> None:
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    errs = {}
    for h in HOLDOUTS:
        by_method = {name: errors(prod, m, h) for name, m in METHODS.items()}
        common = set.intersection(*(set(e) for e in by_method.values()))
        errs[h] = {name: {w: e[w] for w in common} for name, e in by_method.items()}
    rng = np.random.default_rng(0)

    print("Mean absolute held-out volume error (median bias):\n")
    print("| Holdout | Wells | " + " | ".join(METHODS) + " |")
    print("|---|---|" + "---|" * len(METHODS))
    for h in HOLDOUTS:
        cells = [f"{np.mean(np.abs(list(e.values()))):.1%} ({np.median(list(e.values())):+.1%})"
                 for e in errs[h].values()]
        print(f"| {h} months | {len(errs[h]['current'])} | " + " | ".join(cells) + " |")

    print("\nPaired change in absolute error, current minus each variant (90% CI):\n")
    others = [n for n in METHODS if n != "current"]
    print("| Holdout | " + " | ".join(f"vs {n}" for n in others) + " |")
    print("|---|" + "---|" * len(others))
    for h in HOLDOUTS:
        print(f"| {h} months | " + " | ".join(
            paired(errs[h][n], errs[h]["current"], rng) for n in others) + " |")


if __name__ == "__main__":
    main()
