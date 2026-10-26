"""Display text shared by the Streamlit app and the static Vercel build.

Area, well count, dates and forecast assumptions are derived from the results
table, so they cannot drift from what the pipeline actually used. The
provenance text describes the shipped dataset and must be updated if the data
source changes.
"""
from __future__ import annotations

from html import escape

import pandas as pd

CLUSTER_NOTE = (
    "Wells grouped by decline shape: how much of its peak rate each well's fitted decline "
    "keeps after one and three producing years — the basis for type curves and "
    "benchmarking. The number of groups is chosen by silhouette score. Hollow markers are "
    "low-confidence fits; they are assigned to the nearest group but do not shape it.")

SOURCE_REPO = "Dansaad2020/ai-ml-dca-permian"
SOURCE_URL = f"https://github.com/{SOURCE_REPO}"


def fmt(v, spec: str, suffix: str = "") -> str:
    """Format a number, or '—' if it is missing."""
    return "—" if pd.isna(v) else format(v, spec) + suffix


def _month(yyyymm) -> str:
    if pd.isna(yyyymm) or str(yyyymm) == "":
        return "—"
    return pd.to_datetime(str(yyyymm), format="%Y%m").strftime("%b %Y")


def _assumptions(results: pd.DataFrame) -> tuple[float, float, int]:
    r = results.iloc[0]
    return float(r["dmin_annual"]), float(r["econ_limit_bpm"]), int(r["max_life_months"])


def _area(results: pd.DataFrame) -> str:
    counties = sorted(c.title() for c in results["county"].dropna().astype(str).unique() if c)
    return (" / ".join(counties) + (" County" if len(counties) == 1 else " Counties")
            + ", TX") if counties else "Texas"


def caption(results: pd.DataFrame) -> str:
    dmin, econ, cap = _assumptions(results)
    return (f"{_area(results)} · {len(results)} horizontal oil wells · "
            f"Hyperbolic Arps with {dmin:.0%}/yr terminal decline · "
            f"Economic limit {econ:,.0f} bbl/mo · {cap // 12}-yr max well life")


def data_note(results: pd.DataFrame, html: bool = False) -> str:
    """Provenance note; with html=True it is escaped and links the source."""
    note = (
        "Production: public Texas Railroad Commission (RRC) monthly production records, "
        f"{_area(results)} — {len(results)} horizontal oil wells, single-well leases, "
        f"{_month(results['first_prod'].min())}–{_month(results['last_reported'].max())} "
        "(last reported month). Obtained via the open dataset {source}, which "
        "republishes RRC production, permit, wellbore and completion records; RRC's "
        "interactive query endpoints were unreachable at build time, so the data was not "
        "pulled directly from rrc.texas.gov. Trailing months with no reported volume for "
        "any well are treated as not yet reported. See README for full provenance."
    )
    if html:
        return escape(note).replace(
            "{source}", f'<a href="{SOURCE_URL}">{SOURCE_REPO}</a>')
    return note.replace("{source}", f"{SOURCE_REPO} (GitHub)")


def methodology(results: pd.DataFrame) -> str:
    dmin, econ, cap = _assumptions(results)
    n_clusters = results["cluster_id"].nunique()
    b_cap = float(results.iloc[0].get("b_cap", 1.5))
    return ("Methodology: Arps hyperbolic decline fit from peak rate on producing time "
            "(shut-in months and months under 50% of trend count only their producing "
            f"fraction and are left out of the fit), b ≤ {b_cap:g}, robust soft-L1 loss; "
            f"{dmin:.0%}/yr terminal decline, EUR to "
            f"{econ:,.0f} bbl/mo economic limit within a {cap // 12}-yr well life · "
            f"k-means (k={n_clusters}) on decline shape (rate kept after 1 and 3 producing "
            "years). Technical estimates, not SPE PRMS reserves.")


def well_label(row) -> str:
    return f"{row['lease_name']} ({row['well_no']}) — API {int(row['api8'])}"


def well_metrics(row) -> dict:
    return {
        "EUR": fmt(row["eur_bbl"], ",.0f", " bbl"),
        "Remaining": fmt(row["remaining_bbl"], ",.0f", " bbl"),
        "b-factor": fmt(row["b"], ".2f"),
        "Di (nom. annual)": fmt(row["Di_annual"], ".0%"),
        "Fit R²": fmt(row["r_squared"], ".3f"),
    }


def well_detail(row) -> dict:
    return {
        "Lease": row["lease_name"], "Well": row["well_no"], "API": str(int(row["api8"])),
        "Operator": row["operator"], "Field": row["field"],
        "First production": _month(row["first_prod"]),
        "Lease first reported": _month(row["lease_first_reported"]),
        "Last reported month": _month(row["last_reported"]),
        "History (mo)": str(int(row["months_history"])),
        "Downtime since peak (mo)": fmt(row["downtime_months"], ".1f"),
        "Curtailed months left out of fit": fmt(row["curtailed_months"], ".0f"),
        "Cum oil (bbl)": fmt(row["cum_oil_bbl"], ",.0f"),
        "qi (bbl/mo)": fmt(row["qi"], ",.0f"),
        "Di annual (nom)": fmt(row["Di_annual"], ".0%"),
        "b-factor": fmt(row["b"], ".3f"),
        "EUR (bbl)": fmt(row["eur_bbl"], ",.0f"),
        "Remaining (bbl)": fmt(row["remaining_bbl"], ",.0f"),
        "Forecast life (mo)": str(int(row["forecast_months"])),
        "Lateral (ft)": fmt(row["lateral_ft"], ",.0f"),
        "EUR / 1,000 ft": fmt(row["eur_per_1000ft"], ",.0f", " bbl"),
        "Decline cluster": row["cluster"],
        "R²": fmt(row["r_squared"], ".3f"),
        "R² (log space)": fmt(row["r_squared_log"], ".3f"),
        "QC flags": row["qc_flags"] or "none",
    }
