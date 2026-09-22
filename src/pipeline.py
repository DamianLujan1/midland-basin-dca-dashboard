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

from src.arps import eur_bbl, fit_hyperbolic, forecast_to_limit, terminal_rate
from src.clustering import cluster_wells
from src.data_loader import fitting_series, load_production, load_well_metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

ECON_LIMIT_BPM = 150.0   # ~5 BOPD economic limit
DMIN_ANNUAL = 0.05       # 5%/yr terminal decline


def run(econ_limit_bpm: float = ECON_LIMIT_BPM,
        dmin_annual: float = DMIN_ANNUAL) -> pd.DataFrame:
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    meta = load_well_metadata(
        os.path.join(DATA, "martin_selected_30_wells.csv"))
    dmin_m = dmin_annual / 12.0

    rows = []
    for api8 in sorted(prod["api8"].unique()):
        fit_df = fitting_series(prod, api8)
        full = prod[prod["api8"] == api8]
        cum = float(full["oil_bbl"].sum())
        first = full.iloc[0]

        fit = fit_hyperbolic(
            fit_df["t"].to_numpy(),
            fit_df["oil_bbl"].to_numpy(),
            Dmin=dmin_m,
            qi_max=float(fit_df["oil_bbl"].max()),
        )
        row = {
            "api8": int(api8),
            "lease_no": int(first["lease_no"]),
            "well_no": str(first["well_no"]),
            "lease_name": str(first["lease_name"]),
            "operator": str(first["operator_name"]),
            "field": str(first["field_name"]),
            "county": "MARTIN",
            "district": str(first["district"]),
            "first_prod": str(int(first["first_prod_month"])),
            "months_history": int(len(full)),
            "cum_oil_bbl": cum,
            **{k: v for k, v in fit.items() if k != "status"},
            "status": fit["status"],
        }
        if fit["status"] == "ok":
            # Months since peak at end of observed history; forecast starts after.
            peak_mop = int(fit_df["month_on_production"].iloc[0])
            last_mop = int(full["month_on_production"].max())
            t_now = last_mop - peak_mop + 1
            e = eur_bbl(fit["qi"], fit["Di_monthly"], fit["b"], cum,
                        Dmin=dmin_m, econ_limit_bpm=econ_limit_bpm,
                        start_t=t_now + 1)
            row.update(e)
            fc = forecast_to_limit(fit["qi"], fit["Di_monthly"], fit["b"],
                                   Dmin=dmin_m, econ_limit_bpm=econ_limit_bpm,
                                   start_t=t_now + 1)
            row["forecast_months"] = int(len(fc))
        else:
            row.update({"eur_bbl": np.nan, "remaining_bbl": np.nan,
                        "forecast_months": 0})
        rows.append(row)

    results = pd.DataFrame(rows)
    results = cluster_wells(results, n_clusters=3)

    # Attach lateral length for EUR/ft normalization.
    lat = meta.set_index("api8")["interval_length_proxy_ft"]
    results["lateral_ft"] = results["api8"].map(lat)
    results["eur_per_1000ft"] = results["eur_bbl"] / (results["lateral_ft"] / 1000.0)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "midland_basin_dca_results.csv")
    results.to_csv(out, index=False)
    return results


def main() -> None:
    results = run()
    ok = results[results["status"] == "ok"]
    print(f"wells fitted: {len(ok)}/{len(results)}")
    print(f"mean b-factor: {ok['b'].mean():.2f}")
    print(f"mean Di (annual): {ok['Di_annual'].mean():.1%}")
    print(f"mean R^2: {ok['r_squared'].mean():.3f}")
    print(f"EUR range: {ok['eur_bbl'].min():,.0f} - {ok['eur_bbl'].max():,.0f} bbl")
    print(f"mean EUR: {ok['eur_bbl'].mean():,.0f} bbl")
    print("results -> results/midland_basin_dca_results.csv")


if __name__ == "__main__":
    main()
