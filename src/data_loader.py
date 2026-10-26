"""Load and clean Midland Basin monthly production data.

The underlying CSVs carry RRC-sourced monthly production for 30 single-well
oil leases in Martin County, TX (every lease has exactly one well, so
lease-level production == well-level production -- the cleanest possible
case for Texas data, which RRC reports by lease).
"""
from __future__ import annotations

import pandas as pd

# Columns we actually need from the production file.
PROD_COLS = [
    "api8", "district", "lease_no", "well_no", "lease_name", "operator_name",
    "field_name", "first_prod_month", "cycle_year_month", "month_on_production",
    "oil_bbl", "casinghead_gas_mcf", "reported_month_on_production",
]


def load_production(path: str) -> pd.DataFrame:
    """Load monthly production, parse dates, sort by well and time.

    Trailing calendar months in which no well reports any oil or gas are
    dropped: a whole cohort does not shut in at once with zero casinghead gas,
    so those rows are unreported months (RRC reporting lag / calendar padding
    in the source extract), not production. Zero months for a single well
    inside the reported period are kept -- those are that well's shut-ins.
    """
    df = pd.read_csv(path, usecols=PROD_COLS).copy()
    reported = (df.groupby("cycle_year_month")[["oil_bbl", "casinghead_gas_mcf"]]
                  .sum().sum(axis=1))
    reported = reported[reported > 0]
    if len(reported):
        df = df[df["cycle_year_month"] <= reported.index.max()]
    df["date"] = pd.to_datetime(df["cycle_year_month"].astype(str), format="%Y%m")
    df = df.sort_values(["api8", "month_on_production"]).reset_index(drop=True)
    # Decline time is taken from month_on_production; it must be a calendar
    # offset (one per month), not a row counter, or forecasts land on the
    # wrong months.
    month_index = df["date"].dt.year * 12 + df["date"].dt.month
    offset = (month_index - df["month_on_production"]).groupby(df["api8"]).nunique()
    if (offset > 1).any():
        bad = list(offset[offset > 1].index[:5])
        raise ValueError(f"month_on_production is not a calendar offset for wells {bad}")
    return df


def load_well_metadata(path: str) -> pd.DataFrame:
    """Load per-well metadata (completions, lateral length, lease info)."""
    meta = pd.read_csv(path, low_memory=False)
    keep = [
        "api8", "lease_no", "well_no", "lease_name", "operator_name",
        "field_name", "county", "single_well_lease", "history_months",
        "interval_length_proxy_ft", "first_prod_month", "last_prod_month",
    ]
    keep = [c for c in keep if c in meta.columns]
    return meta[keep].copy()


def well_series(prod: pd.DataFrame, api8: int) -> pd.DataFrame:
    """Return the cleaned monthly series for one well (zeros kept, flagged).

    t is calendar months since the first month of the record (first month = 0).
    """
    w = prod[prod["api8"] == api8].copy()
    w["t"] = w["month_on_production"] - int(w["month_on_production"].min())
    return w.reset_index(drop=True)


def fitting_series(prod: pd.DataFrame, api8: int) -> pd.DataFrame:
    """Monthly series for decline fitting: one row per calendar month from the
    peak-rate month to the last reported month.

    Starting at the peak is standard practice for shale wells, where early
    flush production / cleanup distorts the hyperbolic fit. Zero-rate months
    (shut-ins) are kept -- and months without a row are added with zero oil --
    because the producing-time clock (arps.fit_decline) must see them; the fit
    itself uses positive months only. t is calendar months since the peak.
    """
    w = well_series(prod, api8)
    producing = w[w["oil_bbl"] > 0]
    if producing.empty:
        return w.iloc[0:0].assign(t=0)
    peak_mop = int(producing.loc[producing["oil_bbl"].idxmax(), "month_on_production"])
    peak_date = producing.loc[producing["oil_bbl"].idxmax(), "date"]
    w = (w[w["month_on_production"] >= peak_mop]
         .set_index("month_on_production")
         .reindex(range(peak_mop, int(w["month_on_production"].max()) + 1)))
    w["oil_bbl"] = w["oil_bbl"].fillna(0.0)
    w["casinghead_gas_mcf"] = w["casinghead_gas_mcf"].fillna(0.0)
    w["api8"] = api8
    w = w.rename_axis("month_on_production").reset_index()
    w["t"] = w["month_on_production"] - peak_mop
    w["date"] = [peak_date + pd.DateOffset(months=int(t)) for t in w["t"]]
    return w
