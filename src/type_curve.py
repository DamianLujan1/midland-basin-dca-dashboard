"""Observed type curve: production percentiles by month on production.

Same construction as the source dataset's martin_observed_type_curve.csv
(per-month P25/P50/P75 oil, P50 gas and BOE, cumulative P50), but built from
the cleaned production table, so the unreported trailing months are not
counted as zero production. It stops where fewer than MIN_SHARE of the wells
are still reporting: beyond that the percentiles describe a shrinking, older
subset of the cohort (here mostly one pad) rather than the cohort.
"""
from __future__ import annotations

import pandas as pd

MIN_SHARE = 0.8


def observed_type_curve(prod: pd.DataFrame, min_share: float = MIN_SHARE) -> pd.DataFrame:
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
    enough = tc["well_count"] >= min_share * prod["api8"].nunique()
    # Keep months up to the first one that falls short, not later ones that recover.
    tc = tc[enough.cumprod().astype(bool)].reset_index(drop=True)
    tc["cumulative_p50_oil_bbl"] = tc["p50_oil_bbl"].cumsum()
    tc["cumulative_p50_gas_mcf"] = tc["p50_gas_mcf"].cumsum()
    return tc
