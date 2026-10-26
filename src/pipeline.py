"""End-to-end pipeline: load -> clean -> fit Arps -> forecast/EUR -> cluster.

Run:  python -m src.pipeline   (from the repo root)
Writes: results/midland_basin_dca_results.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arps import MAX_FORECAST_MONTHS, eur_bbl, fit_hyperbolic, terminal_rate
from src.clustering import cluster_summary, cluster_wells
from src.data_loader import fitting_series, load_production, load_well_metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS, "midland_basin_dca_results.csv")

ECON_LIMIT_BPM = 150.0   # ~5 BOPD economic limit
DMIN_ANNUAL = 0.05       # 5%/yr terminal decline
N_CLUSTERS = 3
R2_LOW_CONFIDENCE = 0.6  # fits below this R² (linear or log) are low-confidence
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
            "bounds_hit": ""}


def _qc_flags(row: dict, max_forecast_months: int) -> str:
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
    if row["n_partial_months"]:
        flags.append(f"{row['n_partial_months']} partial month(s) excluded from log R²")
    if row["shut_in_months_at_end"]:
        flags.append(f"shut in for last {row['shut_in_months_at_end']} reported mo")
    if row["forecast_capped"]:
        flags.append(f"forecast capped at {max_forecast_months // 12} yr")
    return "; ".join(flags)


def _low_confidence(row: dict) -> bool:
    if row["status"] != "ok":
        return True
    ratio = row["forecast_vs_recent"]
    return bool(min(row["r_squared"], row["r_squared_log"]) < R2_LOW_CONFIDENCE
                or row["bounds_hit"]
                or ratio > FORECAST_JUMP_FACTOR or ratio < 1 / FORECAST_JUMP_FACTOR)


def run(econ_limit_bpm: float = ECON_LIMIT_BPM,
        dmin_annual: float = DMIN_ANNUAL,
        max_forecast_months: int = MAX_FORECAST_MONTHS) -> pd.DataFrame:
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    meta = load_well_metadata(
        os.path.join(DATA, "martin_selected_30_wells.csv"))
    county = meta.set_index("api8")["county"] if "county" in meta else pd.Series(dtype=str)
    dmin_m = dmin_annual / 12.0

    rows = []
    for api8 in sorted(prod["api8"].unique()):
        fit_df = fitting_series(prod, api8)
        full = prod[prod["api8"] == api8]
        cum = float(full["oil_bbl"].sum())
        first = full.iloc[0]
        first_mop = int(full["month_on_production"].min())
        last_mop = int(full["month_on_production"].max())
        producing = full[full["oil_bbl"] > 0]

        fit = fit_hyperbolic(
            fit_df["t"].to_numpy(),
            fit_df["oil_bbl"].to_numpy(),
            qi_max=float(fit_df["oil_bbl"].max()) if len(fit_df) else None,
        )
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
            "shut_in_months_at_end": int(last_mop - producing["month_on_production"].max())
                                     if len(producing) else int(last_mop - first_mop + 1),
            "cum_oil_bbl": cum,
            **FIT_COLS,
            **{k: v for k, v in fit.items() if k != "status"},
            "status": fit["status"],
        }
        if fit["status"] == "ok":
            # Months since peak at the last reported month; forecast starts after.
            peak_mop = int(fit_df["month_on_production"].iloc[0])
            t_now = last_mop - peak_mop
            row.update(eur_bbl(fit["qi"], fit["Di_monthly"], fit["b"], cum,
                               Dmin=dmin_m, econ_limit_bpm=econ_limit_bpm,
                               max_months=max_forecast_months, start_t=t_now + 1))
            q_next = terminal_rate(np.array([t_now + 1]), fit["qi"], fit["Di_monthly"],
                                   fit["b"], dmin_m)[0]
            recent = producing["oil_bbl"].tail(RECENT_MONTHS).median()
            row["forecast_vs_recent"] = float(q_next / recent)
        else:
            row.update({"eur_bbl": np.nan, "remaining_bbl": np.nan,
                        "cum_produced_bbl": cum, "forecast_months": 0,
                        "forecast_capped": False, "forecast_vs_recent": np.nan})
        row["qc_flags"] = _qc_flags(row, max_forecast_months)
        row["low_confidence"] = _low_confidence(row)
        rows.append(row)

    results = pd.DataFrame(rows)
    results = cluster_wells(results, n_clusters=N_CLUSTERS)

    # Attach lateral length for EUR/ft normalization.
    lat = meta.set_index("api8")["interval_length_proxy_ft"]
    results["lateral_ft"] = results["api8"].map(lat)
    results["eur_per_1000ft"] = results["eur_bbl"] / (results["lateral_ft"] / 1000.0)

    # Assumptions behind every EUR, so charts redraw forecasts consistently.
    results["econ_limit_bpm"] = float(econ_limit_bpm)
    results["dmin_annual"] = float(dmin_annual)
    results["max_forecast_months"] = int(max_forecast_months)

    os.makedirs(RESULTS, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    return results


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
    print(f"mean b-factor: {ok['b'].mean():.2f}")
    print(f"mean Di (annual): {ok['Di_annual'].mean():.1%}")
    print(f"mean R^2: {ok['r_squared'].mean():.3f} (median {ok['r_squared'].median():.3f}); "
          f"{(ok['r_squared'] > R2_LOW_CONFIDENCE).sum()}/{len(ok)} wells R^2 > {R2_LOW_CONFIDENCE}; "
          f"log-space: {(ok['r_squared_log'] > R2_LOW_CONFIDENCE).sum()}/{len(ok)}")
    print(f"low-confidence fits: {int(results['low_confidence'].sum())}; "
          f"forecasts capped at {MAX_FORECAST_MONTHS // 12} yr: {int(ok['forecast_capped'].sum())}")
    print(f"EUR range: {ok['eur_bbl'].min():,.0f} - {ok['eur_bbl'].max():,.0f} bbl")
    print(f"mean EUR: {ok['eur_bbl'].mean():,.0f} bbl")
    print(cluster_summary(results).round(3).to_string(index=False))
    print("results -> results/midland_basin_dca_results.csv")


if __name__ == "__main__":
    main()
