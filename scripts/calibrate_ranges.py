"""Calibrate and check the P90-P10 forecast ranges on a hindcast (prints Markdown).

For each well and holdout H (6, 12, 18 months): hide the last H reported
months, fit the rest (src.arps.fit_decline), build the well's bootstrap
ensemble exactly as the pipeline does (src.uncertainty), and compute each
member's volume over the held-out producing months. The level factor
exp(sigma * z) scales the members; a window is covered when the actual volume
lies within [P90, P10] = [min(10th percentile, fit), max(90th percentile, fit)].

For each window, the smallest sigma that covers it is found on a fine grid.
LEVEL_SIGMA is the split-conformal 80% quantile of those values: the
ceil(0.8 (n + 1))-th smallest, so at least 80% of windows are covered. Coverage
is then checked leave-one-well-out: each well's windows are scored with the
sigma calibrated on the other wells only. A constant-width interval
(fit x exp(+/-k), k calibrated the same way) is the baseline; the interval
score (Gneiting & Raftery 2007, in log volume, lower is better) rewards
narrow intervals and penalizes misses.

Run:  python scripts/calibrate_ranges.py   (from the repo root, ~1 minute)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import uncertainty  # noqa: E402
from src.arps import B_CAP, fit_decline, terminal_rate  # noqa: E402
from src.data_loader import fitting_series, load_production  # noqa: E402
from src.pipeline import DATA, DMIN_ANNUAL  # noqa: E402

HOLDOUTS = (6, 12, 18)
LEVEL = 0.8
GRID = np.arange(0.0, 0.8001, 0.0025)


def well_windows(api8, w) -> list:
    """(api8, H, fit volume, member volumes, member z, actual volume) per holdout."""
    out = []
    for h in HOLDOUTS:
        cut = w["month_on_production"].max() - h
        series = fitting_series(w[w["month_on_production"] <= cut], api8)
        test = w[(w["month_on_production"] > cut) & (w["oil_bbl"] > 0)]
        if not len(series) or test.empty:
            continue
        q = series["oil_bbl"].to_numpy(float)
        qi_max = float(q.max())
        fit, clock, weight = fit_decline(q, qi_max=qi_max, b_cap=B_CAP)
        if fit["status"] != "ok":
            continue
        # The same members and level-factor scores uncertainty.ensemble uses.
        params, z = uncertainty.members(q, clock, weight, fit, qi_max, B_CAP,
                                        seed=[int(api8), h])
        t = fit["producing_months"] + (test["month_on_production"] - (cut + 1)).to_numpy(float)
        dmin = DMIN_ANNUAL / 12
        vol = terminal_rate(t[None, :], params[:, :1], params[:, 1:2], params[:, 2:3],
                            dmin).sum(axis=1)
        det = float(terminal_rate(t, fit["qi"], fit["Di_monthly"], fit["b"], dmin).sum())
        out.append((int(api8), h, det, vol, z, float(test["oil_bbl"].sum())))
    return out


def windows(prod) -> list:
    per_well = Parallel(n_jobs=-1)(delayed(well_windows)(a, w) for a, w in prod.groupby("api8"))
    return [win for wells in per_well for win in wells]


def interval(det, vol, z, sigma):
    lo, hi = np.percentile(vol * np.exp(sigma * z), [uncertainty.LOW, uncertainty.HIGH])
    return min(lo, det), max(hi, det)


def conformal(scores) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    return float(s[min(int(np.ceil(LEVEL * (len(s) + 1))) - 1, len(s) - 1)])


def interval_score(lo, hi, y, alpha=1 - LEVEL) -> float:
    lo, hi, y = np.log(lo), np.log(hi), np.log(y)
    return (hi - lo) + 2 / alpha * (max(lo - y, 0) + max(y - hi, 0))


def main() -> None:
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    win = windows(prod)
    api = np.array([w[0] for w in win])
    hs = np.array([w[1] for w in win])

    # Smallest sigma on the grid that covers each window.
    crit = []
    for _, _, det, vol, z, y in win:
        covered = [lo <= y <= hi for lo, hi in (interval(det, vol, z, s) for s in GRID)]
        crit.append(GRID[int(np.argmax(covered))] if any(covered) else np.inf)
    crit = np.array(crit)
    sigma = conformal(crit)
    loo_sigma = np.array([conformal(crit[api != a]) for a in api])
    # Baseline: constant relative width around the fit.
    dev = np.array([abs(np.log(y / det)) for _, _, det, _, _, y in win])
    loo_k = np.array([conformal(dev[api != a]) for a in api])

    rows = []
    for i, (_, h, det, vol, z, y) in enumerate(win):
        lo, hi = interval(det, vol, z, loo_sigma[i])
        lo0, hi0 = interval(det, vol, z, 0.0)
        blo, bhi = det * np.exp(-loo_k[i]), det * np.exp(loo_k[i])
        median = np.median(vol * np.exp(loo_sigma[i] * z))
        rows.append((h, lo <= y <= hi, hi / lo, interval_score(lo, hi, y), lo0 <= y <= hi0,
                     blo <= y <= bhi, interval_score(blo, bhi, y), y < det, y < median))
    rows = np.array(rows, dtype=float)

    print(f"Calibrated level sigma: {sigma:.3f} (src/uncertainty.py LEVEL_SIGMA = "
          f"{uncertainty.LEVEL_SIGMA}); constant-width baseline: fit x exp(+/-{conformal(dev):.3f}); "
          f"windows: {len(win)}\n")
    if abs(sigma - uncertainty.LEVEL_SIGMA) > 0.01:
        print(f"WARNING: LEVEL_SIGMA differs from the calibrated value by more than 0.01\n")
    print("| Holdout | Windows | Coverage of P90–P10, leave-one-well-out | Mean P10/P90 "
          "volume ratio | Coverage, bootstrap alone | Coverage, constant width "
          "| Interval score: ensemble / constant width |")
    print("|---|---|---|---|---|---|---|")
    for label, m in [(f"{h} months", hs == h) for h in HOLDOUTS] + [("All", hs > 0)]:
        r = rows[m]
        print(f"| {label} | {int(m.sum())} | {r[:, 1].mean():.0%} | {r[:, 2].mean():.2f} "
              f"| {r[:, 4].mean():.0%} | {r[:, 5].mean():.0%} "
              f"| {r[:, 3].mean():.3f} / {r[:, 6].mean():.3f} |")
    # Which is the better middle of the range: P50 (the fit) or the ensemble median?
    below_fit, below_median = int(rows[:, 7].sum()), int(rows[:, 8].sum())
    print(f"\nHeld-out volume below P50 (the fit) in {below_fit}/{len(rows)} windows "
          f"({below_fit / len(rows):.0%}), below the ensemble median in {below_median}/{len(rows)} "
          f"({below_median / len(rows):.0%}).")


if __name__ == "__main__":
    main()
