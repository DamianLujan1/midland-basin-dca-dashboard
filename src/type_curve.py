"""Observed type curve: production percentiles by month on production.

Same construction as the source dataset's martin_observed_type_curve.csv
(per-month P25/P50/P75 oil, P50 gas and BOE, cumulative P50), but built from
the cleaned production table, so the unreported trailing months are not
counted as zero production, and months with too few wells are dropped.
"""
from __future__ import annotations

import pandas as pd

MIN_WELLS = 10  # percentiles from fewer wells are noise


def observed_type_curve(prod: pd.DataFrame, min_wells: int = MIN_WELLS) -> pd.DataFrame:
    g = prod.assign(boe=prod["oil_bbl"] + prod["casinghead_gas_mcf"] / 6.0).groupby(
        "month_on_production")
    tc = pd.DataFrame({
        "well_count": g["api8"].nunique(),
        "p25_oil_bbl": g["oil_bbl"].quantile(0.25),
        "p50_oil_bbl": g["oil_bbl"].median(),
        "p75_oil_bbl": g["oil_bbl"].quantile(0.75),
        "p50_gas_mcf": g["casinghead_gas_mcf"].median(),
        "p50_boe": g["boe"].median(),
    }).reset_index()
    tc = tc[tc["well_count"] >= min_wells].reset_index(drop=True)
    tc["cumulative_p50_oil_bbl"] = tc["p50_oil_bbl"].cumsum()
    tc["cumulative_p50_gas_mcf"] = tc["p50_gas_mcf"].cumsum()
    return tc
