"""End-to-end pipeline: load -> clean -> fit Arps -> forecast/EUR -> economics
-> P90/P10 ranges -> cluster.

Run:  python -m src.pipeline   (from the repo root)
Writes: results/midland_basin_dca_results.csv, results/observed_type_curve.csv,
        results/forecast_ensemble.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import uncertainty
from src.arps import (B_CAP, CURTAIL_RATIO, eur_bbl, fit_decline, forecast_to_limit,
                      terminal_rate)
from src.clustering import (RETENTION_MONTHS, choose_k, cluster_stability, cluster_summary,
                            cluster_wells, retention)
from src.data_loader import fitting_series, load_production, load_well_metadata
from src.economics import (Economics, cash_flow, economic_limit, present_value,
                           shut_in_price, trailing_gor)
from src.type_curve import observed_type_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS, "midland_basin_dca_results.csv")
TYPE_CURVE_CSV = os.path.join(RESULTS, "observed_type_curve.csv")
ENSEMBLE_CSV = os.path.join(RESULTS, "forecast_ensemble.csv")

# Base-case forecast assumptions (see README "Forecast assumptions"). The
# economic limit is not an input: each well's comes from the price deck and
# operating costs (src.economics), unless run() is given a fixed one.
ECONOMICS = Economics()
DMIN_ANNUAL = 0.08       # 8%/yr nominal terminal decline (between the earlier 5% and the
                         # source dataset's own 9.6%)
MAX_LIFE_MONTHS = 600    # 50-year total well life, counted from first production
N_CLUSTERS = None         # None: choose k by silhouette (clustering.choose_k)
R2_LOW_CONFIDENCE = 0.6  # fits below this R² (linear or log) are low-confidence
# Wells whose last this-many reported months are shut in or curtailed are
# low-confidence: the forecast assumes they return to trend.
CURTAILED_AT_END_LOW_CONFIDENCE = 2
# Note wells with this many months of inferred downtime (shut-in and curtailed
# months taken off the producing clock and down-weighted in the fit), so a fit
# that discounted much of the history does not look clean.
DOWNTIME_NOTE_MONTHS = 3
# Flag forecasts whose first month is off the recent producing rate (median of
# the last 6 producing months, so one curtailed month does not trigger it) by
# more than this factor in either direction.
FORECAST_JUMP_FACTOR = 1.35
RECENT_MONTHS = 6

# RRC production-data DISTRICT_NO codes -> RRC district names (code 10 is
# District 08, which covers Martin County).
RRC_DISTRICT = {1: "01", 2: "02", 3: "03", 4: "04", 5: "05", 6: "06", 7: "6E",
                8: "7B", 9: "7C", 10: "08", 11: "8A", 12: "8B", 13: "09", 14: "10"}

FIT_COLS = {"qi": np.nan, "Di_monthly": np.nan, "Di_annual": np.nan, "b": np.nan,
            "r_squared": np.nan, "r_squared_log": np.nan, "n_partial_months": 0,
            "bounds_hit": "", "b_at_cap": False, "producing_months": np.nan,
            "downtime_months": np.nan, "curtailed_months": 0, "curtailed_months_at_end": 0,
            "downweighted_months": 0,
            **{f"retention_{m}mo": np.nan for m in RETENTION_MONTHS}}
NO_FORECAST = {"eur_bbl": np.nan, "remaining_bbl": np.nan, "forecast_months": 0,
               "forecast_capped": False, "forecast_vs_recent": np.nan, "pv10_usd": np.nan,
               "ncf_usd": np.nan, "shut_in_oil_price": np.nan}
RANGE_COLS = ["remaining_p90_bbl", "remaining_p10_bbl", "eur_p90_bbl", "eur_p10_bbl",
              "pv10_p90_usd", "pv10_p10_usd"]


def _qc_flags(row: dict, max_life_months: int) -> str:
    if row["status"] != "ok":
        return f"fit {row['status']}"
    flags = []
    if row["r_squared"] < R2_LOW_CONFIDENCE:
        flags.append(f"R² < {R2_LOW_CONFIDENCE}")
    if row["r_squared_log"] < R2_LOW_CONFIDENCE:
        flags.append(f"log-space R² {row['r_squared_log']:.2f}")
    flags += [f"{p} at fit bound" for p in row["bounds_hit"].split(",") if p]
    ratio = row["forecast_vs_recent"]
    if ratio > FORECAST_JUMP_FACTOR or ratio < 1 / FORECAST_JUMP_FACTOR:
        flags.append(f"forecast starts at {ratio:.2f}x recent rate")
    if row["b_at_cap"]:
        flags.append(f"b held at the {row['b_cap']} cap")
    if row["downtime_months"] >= DOWNTIME_NOTE_MONTHS:
        flags.append(f"{row['downtime_months']:.1f} months of downtime (shut-in and curtailed "
                     "months) taken off the producing clock and down-weighted in the fit")
    if row["curtailed_months_at_end"]:
        flags.append(f"last {row['curtailed_months_at_end']} reported mo shut in or under "
                     f"{CURTAIL_RATIO:.0%} of trend; forecast assumes a return to trend, "
                     "confirm status")
    if row["forecast_capped"]:
        flags.append(f"forecast capped at {max_life_months // 12}-yr well life")
    if row["forecast_months"] == 0:
        flags.append("rate already below the economic limit")
    return "; ".join(flags)


def _low_confidence(row: dict) -> bool:
    if row["status"] != "ok":
        return True
    ratio = row["forecast_vs_recent"]
    return bool(min(row["r_squared"], row["r_squared_log"]) < R2_LOW_CONFIDENCE
                or row["bounds_hit"]
                or ratio > FORECAST_JUMP_FACTOR or ratio < 1 / FORECAST_JUMP_FACTOR
                or row["curtailed_months_at_end"] >= CURTAILED_AT_END_LOW_CONFIDENCE)


def _ensemble(q: np.ndarray, b_cap: float, seed: int) -> np.ndarray | None:
    """One well's forecast ensemble (None if the well does not fit)."""
    qi_max = float(q.max()) if len(q) else None
    fit, clock, weight = fit_decline(q, qi_max=qi_max, b_cap=b_cap)
    if fit["status"] != "ok":
        return None
    return uncertainty.ensemble(q, clock, weight, fit, qi_max, b_cap, seed=seed)


def run(dmin_annual: float = DMIN_ANNUAL,
        max_life_months: int = MAX_LIFE_MONTHS,
        b_cap: float = B_CAP,
        economics: Economics = ECONOMICS,
        econ_limit_bpm: float | None = None,
        ensembles: dict | None = None,
        ranges: bool = True,
        write: bool = True) -> pd.DataFrame:
    """Fit, forecast, value and cluster every well; write the results unless
    write=False.

    econ_limit_bpm fixes one economic limit for every well instead of deriving
    each well's from `economics`. ensembles ({api8: (m, 3) params}, e.g. from
    load_ensemble()) skips the bootstrap for the wells it covers; it depends
    only on the data and b_cap. ranges=False skips P90/P10 altogether.
    """
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    meta = load_well_metadata(
        os.path.join(DATA, "martin_selected_30_wells.csv"))
    county = meta.set_index("api8")["county"] if "county" in meta else pd.Series(dtype=str)
    dmin_m = dmin_annual / 12.0
    ensembles = dict(ensembles or {})
    computed = {}
    if ranges:  # the bootstrap refits are most of the run time: one process per core
        missing = [int(a) for a in sorted(prod["api8"].unique()) if int(a) not in ensembles]
        fitted = Parallel(n_jobs=-1)(
            delayed(_ensemble)(fitting_series(prod, a)["oil_bbl"].to_numpy(), b_cap, a)
            for a in missing)
        computed = {a: p for a, p in zip(missing, fitted) if p is not None}
        ensembles.update(computed)

    rows = []
    for api8 in sorted(prod["api8"].unique()):
        fit_df = fitting_series(prod, api8)
        full = prod[prod["api8"] == api8]
        cum = float(full["oil_bbl"].sum())
        first = full.iloc[0]
        first_mop = int(full["month_on_production"].min())
        last_mop = int(full["month_on_production"].max())
        producing = full[full["oil_bbl"] > 0]
        gor = trailing_gor(full["oil_bbl"], full["casinghead_gas_mcf"])
        limit = (float(econ_limit_bpm) if econ_limit_bpm is not None
                 else economic_limit(economics, gor))

        q = fit_df["oil_bbl"].to_numpy()
        qi_max = float(q.max()) if len(q) else None
        fit, clock, weight = fit_decline(q, qi_max=qi_max, b_cap=b_cap)
        row = {
            "api8": int(api8),
            "lease_no": int(first["lease_no"]),
            "well_no": str(first["well_no"]),
            "lease_name": str(first["lease_name"]),
            "operator": str(first["operator_name"]),
            "field": str(first["field_name"]),
            "county": str(county.get(api8, "")),
            "district": RRC_DISTRICT.get(int(first["district"]), str(first["district"])),
            "first_prod": (str(int(producing["cycle_year_month"].iloc[0]))
                           if len(producing) else ""),
            # The lease's first RRC-reported month; for some wells this is
            # months of zero-volume reports before first production.
            "lease_first_reported": str(int(first["first_prod_month"])),
            "zero_months_before_record": int(first["reported_month_on_production"]) - 1,
            "last_reported": str(int(full["cycle_year_month"].max())),
            "months_history": last_mop - first_mop + 1,
            # Forecast months left before the well reaches the total-life cap.
            "forecast_month_cap": max(max_life_months - (last_mop - first_mop + 1), 0),
            "cum_oil_bbl": cum,
            "b_cap": float(b_cap),
            "gor_mcf_per_bbl": gor,
            "econ_limit_bpm": limit,
            **FIT_COLS,
            **{k: v for k, v in fit.items() if k != "status"},
            "status": fit["status"],
            "cum_produced_bbl": cum,
            **NO_FORECAST,
            **dict.fromkeys(RANGE_COLS, np.nan),
        }
        if fit["status"] == "ok":
            # The forecast starts the month after the last reported month, at the
            # producing time elapsed since the peak by then.
            start_t = fit["producing_months"]
            cap = row["forecast_month_cap"]
            row.update(eur_bbl(fit["qi"], fit["Di_monthly"], fit["b"], cum, Dmin=dmin_m,
                               econ_limit_bpm=limit, max_months=cap, start_t=start_t))
            fc = forecast_to_limit(fit["qi"], fit["Di_monthly"], fit["b"], dmin_m, limit,
                                   cap, start_t)
            ncf = cash_flow(fc, economics, gor)
            row["pv10_usd"] = float(present_value(ncf, economics.discount))
            row["ncf_usd"] = float(ncf.sum())
            q_next = float(terminal_rate(start_t, fit["qi"], fit["Di_monthly"], fit["b"],
                                         dmin_m))
            row["shut_in_oil_price"] = shut_in_price(economics, gor, q_next)
            recent = producing["oil_bbl"].tail(RECENT_MONTHS).median()
            row["forecast_vs_recent"] = float(q_next / recent)
            for m in RETENTION_MONTHS:
                row[f"retention_{m}mo"] = float(
                    retention(fit["qi"], fit["Di_monthly"], fit["b"], m))
            if ranges:
                r = uncertainty.ranges(ensembles[int(api8)], start_t, cap, dmin_m, limit,
                                       economics, gor)
                rem, pv = row["remaining_bbl"], row["pv10_usd"]
                row["remaining_p90_bbl"] = min(r["remaining"][0], rem)
                row["remaining_p10_bbl"] = max(r["remaining"][1], rem)
                row["eur_p90_bbl"] = cum + row["remaining_p90_bbl"]
                row["eur_p10_bbl"] = cum + row["remaining_p10_bbl"]
                row["pv10_p90_usd"] = min(r["pv"][0], pv)
                row["pv10_p10_usd"] = max(r["pv"][1], pv)
        row["qc_flags"] = _qc_flags(row, max_life_months)
        row["low_confidence"] = _low_confidence(row)
        rows.append(row)

    results = pd.DataFrame(rows)
    results = cluster_wells(results, n_clusters=N_CLUSTERS)

    # Attach lateral length for EUR/ft normalization.
    lat = meta.set_index("api8")["interval_length_proxy_ft"]
    results["lateral_ft"] = results["api8"].map(lat)
    results["eur_per_1000ft"] = results["eur_bbl"] / (results["lateral_ft"] / 1000.0)

    # Assumptions behind every EUR, so charts redraw forecasts consistently.
    results["econ_limit_derived"] = econ_limit_bpm is None
    results["dmin_annual"] = float(dmin_annual)
    results["max_life_months"] = int(max_life_months)
    for k, v in economics.as_columns().items():
        results[k] = float(v)

    if write:
        os.makedirs(RESULTS, exist_ok=True)
        results.to_csv(RESULTS_CSV, index=False)
        observed_type_curve(prod).to_csv(TYPE_CURVE_CSV, index=False)
        if computed:
            uncertainty.to_frame(ensembles).to_csv(ENSEMBLE_CSV, index=False)
    return results


def load_ensemble(path: str = ENSEMBLE_CSV) -> dict:
    """Read the forecast ensembles run() wrote: {api8: (m, 3) params}."""
    return uncertainty.from_frame(pd.read_csv(path))


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    """Read a results CSV back with the dtypes run() wrote it with."""
    results = pd.read_csv(path, dtype={
        "well_no": str, "district": str, "first_prod": str,
        "lease_first_reported": str, "last_reported": str,
        "bounds_hit": str, "qc_flags": str})
    for col in ("bounds_hit", "qc_flags"):
        results[col] = results[col].fillna("")
    return results


def main() -> None:
    results = run()
    ok = results[results["status"] == "ok"]
    print(f"wells fitted: {len(ok)}/{len(results)}")
    print(f"history: {results['first_prod'].min()} - {results['last_reported'].max()}, "
          f"{results['months_history'].min()}-{results['months_history'].max()} months/well")
    print(f"mean b-factor: {ok['b'].mean():.2f} (median {ok['b'].median():.2f}; "
          f"{int(ok['b_at_cap'].sum())} at the {B_CAP:g} cap)")
    print(f"mean Di (annual): {ok['Di_annual'].mean():.1%} (median {ok['Di_annual'].median():.1%})")
    print(f"mean R^2: {ok['r_squared'].mean():.3f} (median {ok['r_squared'].median():.3f}); "
          f"{(ok['r_squared'] > R2_LOW_CONFIDENCE).sum()}/{len(ok)} wells R^2 > {R2_LOW_CONFIDENCE}; "
          f"log-space: {(ok['r_squared_log'] > R2_LOW_CONFIDENCE).sum()}/{len(ok)}")
    print(f"low-confidence fits: {int(results['low_confidence'].sum())}; "
          f"forecasts capped at {MAX_LIFE_MONTHS // 12}-yr life: {int(ok['forecast_capped'].sum())}")
    print(f"economic limit: {ok['econ_limit_bpm'].min():,.0f}-{ok['econ_limit_bpm'].max():,.0f} "
          f"bbl/mo (median {ok['econ_limit_bpm'].median():,.0f}); GOR "
          f"{ok['gor_mcf_per_bbl'].min():.1f}-{ok['gor_mcf_per_bbl'].max():.1f} mcf/bbl")
    print(f"EUR range: {ok['eur_bbl'].min():,.0f} - {ok['eur_bbl'].max():,.0f} bbl")
    print(f"mean EUR: {ok['eur_bbl'].mean():,.0f} bbl (median {ok['eur_bbl'].median():,.0f}); "
          f"mean P90 / P10: {ok['eur_p90_bbl'].mean():,.0f} / {ok['eur_p10_bbl'].mean():,.0f}")
    print(f"EUR P10/P90 ratio: median {(ok['eur_p10_bbl'] / ok['eur_p90_bbl']).median():.2f} "
          f"(range {(ok['eur_p10_bbl'] / ok['eur_p90_bbl']).min():.2f}-"
          f"{(ok['eur_p10_bbl'] / ok['eur_p90_bbl']).max():.2f})")
    print(f"total remaining: {ok['remaining_bbl'].sum():,.0f} bbl; total PV10: "
          f"${ok['pv10_usd'].sum() / 1e6:,.1f} MM (P50); shut-in oil price median "
          f"${ok['shut_in_oil_price'].median():,.1f}/bbl")
    _, scores = choose_k(results)
    print(f"clusters: k={results['cluster_id'].nunique()} (silhouette by k: "
          + ", ".join(f"{kk}: {s:.3f}" for kk, s in scores.items())
          + f"; bootstrap stability ARI {cluster_stability(results):.2f})")
    print(cluster_summary(results).round(3).to_string(index=False))
    print("results -> results/midland_basin_dca_results.csv, results/observed_type_curve.csv, "
          "results/forecast_ensemble.csv")


if __name__ == "__main__":
    main()
