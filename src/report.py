"""Display text shared by the Streamlit app and the static Vercel build.

Area, well count, dates, forecast and economic assumptions are derived from
the results table, so they cannot drift from what the pipeline actually used. The
provenance text describes the shipped dataset and must be updated if the data
source changes.
"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from src.arps import B_CAP
from src.economics import Economics
from src.uncertainty import BOOT_SAMPLES

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


def bbl(v) -> str:
    """Forecast volume rounded to the nearest thousand barrels."""
    return "—" if pd.isna(v) else f"{round(float(v), -3):,.0f} bbl"


def bbl_range(lo, hi) -> str:
    return "—" if pd.isna(lo) or pd.isna(hi) else (
        f"{round(float(lo), -3):,.0f}–{round(float(hi), -3):,.0f} bbl")


def usd_mm(v) -> str:
    """Dollars in millions, one decimal."""
    if pd.isna(v):
        return "—"
    return ("−" if v < 0 else "") + f"${abs(float(v)) / 1e6:,.1f} MM"


def usd_mm_range(lo, hi) -> str:
    """'$3.4–8.2 MM'."""
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return usd_mm(lo).removesuffix(" MM") + "–" + usd_mm(hi).replace("$", "", 1)


def price(v) -> str:
    """$/bbl with cents; '−$1.20/bbl' when negative."""
    return "—" if pd.isna(v) else ("−" if v < 0 else "") + f"${abs(float(v)):,.2f}/bbl"


def pv_label(rate) -> str:
    """'PV10' for a 10%/yr discount rate, 'PV20' for 20%, 'PV7.5' for 7.5%. Labels
    take the rate the results were discounted at (column econ_discount); rounding
    keeps 0.1 * 100 = 10.000000000000002 from showing."""
    return f"PV{round(float(rate) * 100, 4):g}"


def _month(yyyymm) -> str:
    if pd.isna(yyyymm) or str(yyyymm) == "":
        return "—"
    return pd.to_datetime(str(yyyymm), format="%Y%m").strftime("%b %Y")


def _assumptions(results: pd.DataFrame) -> tuple[float, int]:
    r = results.iloc[0]
    return float(r["dmin_annual"]), int(r["max_life_months"])


def economics_of(results: pd.DataFrame) -> Economics:
    return Economics.from_row(results.iloc[0])


def econ_limit_text(results: pd.DataFrame) -> str:
    """'economic limit 108–137 bbl/mo where cash flow turns negative' or a fixed limit."""
    lim = results["econ_limit_bpm"]
    lim = lim[np.isfinite(lim)]
    if not bool(results["econ_limit_derived"].iloc[0]):
        return f"economic limit {lim.iloc[0]:,.0f} bbl/mo (fixed)" if len(lim) else "economic limit —"
    if lim.empty:
        return "no rate covers operating cost"
    span = (f"{lim.min():,.0f} bbl/mo" if round(lim.min()) == round(lim.max())
            else f"{lim.min():,.0f}–{lim.max():,.0f} bbl/mo")
    return f"economic limit {span} where cash flow turns negative"


def _area(results: pd.DataFrame) -> str:
    counties = sorted(c.title() for c in results["county"].dropna().astype(str).unique() if c)
    return (" / ".join(counties) + (" County" if len(counties) == 1 else " Counties")
            + ", TX") if counties else "Texas"


def caption(results: pd.DataFrame) -> str:
    dmin, cap = _assumptions(results)
    e = economics_of(results)
    return (f"{_area(results)} · {len(results)} horizontal oil wells · "
            f"Hyperbolic Arps with {dmin:.0%}/yr terminal decline · "
            f"${e.oil_price:,.0f}/bbl oil, ${e.gas_price:,.2f}/mcf gas · "
            f"{econ_limit_text(results).capitalize()} · {cap // 12}-yr max well life")


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


def economics_note(results: pd.DataFrame) -> str:
    """One line on the price deck and costs behind the PV and the economic limit."""
    e = economics_of(results)
    return (f"PDP economics, flat prices: ${e.oil_price:,.0f}/bbl oil, ${e.gas_price:,.2f}/mcf "
            f"casinghead gas at each well's last-12-month GOR; {e.nri:.1%} NRI, 100% WI; "
            f"Texas severance {e.oil_sev:.1%} oil / {e.gas_sev:.1%} gas, ad valorem "
            f"{e.ad_valorem:.1%}; LOE ${e.loe_fixed:,.0f}/well-month + ${e.loe_var:,.2f}/bbl; "
            f"PV at {e.discount:.0%}/yr, mid-month, from the month after the last report. "
            "No capex, workovers or abandonment cost.")


RANGE_NOTE = (
    "P50 is the fitted forecast. P90 and P10 are the 10th and 90th percentiles of a "
    f"{BOOT_SAMPLES:,}-member ensemble (block bootstrap of the fit plus a level factor), "
    "calibrated so 80% of held-out 6–18 month volumes fell inside the range in a "
    "leave-one-well-out hindcast. Beyond 18 months the range is the model's extrapolation. "
    f"For wells with b held at the {B_CAP:g} cap the range is skewed low: members can "
    "only have lower b, so b adds no upside.")


def methodology(results: pd.DataFrame) -> str:
    dmin, cap = _assumptions(results)
    n_clusters = results["cluster_id"].nunique()
    b_cap = float(results.iloc[0].get("b_cap", 1.2))
    return ("Methodology: Arps hyperbolic decline fit from peak rate on producing time "
            "(shut-in months and months well under trend count only part of a month and "
            f"are down-weighted in the fit), b ≤ {b_cap:g}, robust soft-L1 loss; "
            f"{dmin:.0%}/yr terminal decline, forecast to the "
            f"{econ_limit_text(results)} within a {cap // 12}-yr well life; P90/P10 from a "
            "hindcast-calibrated ensemble · "
            f"k-means (k={n_clusters}) on decline shape (rate kept after 1 and 3 producing "
            "years). Technical estimates, not SPE PRMS reserves.")


def well_label(row) -> str:
    return f"{row['lease_name']} ({row['well_no']}) — API {int(row['api8'])}"


def well_metrics(row) -> dict:
    return {
        "EUR (P50)": bbl(row["eur_bbl"]),
        "Remaining (P50)": bbl(row["remaining_bbl"]),
        f"{pv_label(row['econ_discount'])} (P50)": usd_mm(row["pv10_usd"]),
        "b-factor": fmt(row["b"], ".2f"),
        "Fit R²": fmt(row["r_squared"], ".3f"),
    }


def well_ranges(row) -> dict:
    """P90–P10 ranges shown under the well metrics, keyed like well_metrics."""
    return {
        "EUR (P50)": bbl_range(row["eur_p90_bbl"], row["eur_p10_bbl"]),
        "Remaining (P50)": bbl_range(row["remaining_p90_bbl"], row["remaining_p10_bbl"]),
        f"{pv_label(row['econ_discount'])} (P50)": usd_mm_range(row["pv10_p90_usd"],
                                                                row["pv10_p10_usd"]),
    }


def _p3(lo, mid, hi, f) -> str:
    return " / ".join(f(v) for v in (lo, mid, hi))


def well_detail(row) -> dict:
    return {
        "Lease": row["lease_name"], "Well": row["well_no"], "API": str(int(row["api8"])),
        "Operator": row["operator"], "Field": row["field"],
        "First production": _month(row["first_prod"]),
        "Lease first reported": _month(row["lease_first_reported"]),
        "Last reported month": _month(row["last_reported"]),
        "History (mo)": str(int(row["months_history"])),
        "Downtime since peak (mo)": fmt(row["downtime_months"], ".1f"),
        "Months down-weighted in the fit (under 70% of trend)": (
            f"{row['downweighted_months']:.0f} ({row['curtailed_months']:.0f} under 50%)"
            if pd.notna(row.get("downweighted_months")) else "—"),
        "Cum oil (bbl)": fmt(row["cum_oil_bbl"], ",.0f"),
        "qi (bbl/mo)": fmt(row["qi"], ",.0f"),
        "Di annual (nom)": fmt(row["Di_annual"], ".0%"),
        "b-factor": fmt(row["b"], ".3f"),
        "EUR P90 / P50 / P10": _p3(row["eur_p90_bbl"], row["eur_bbl"], row["eur_p10_bbl"], bbl),
        "Remaining P90 / P50 / P10": _p3(row["remaining_p90_bbl"], row["remaining_bbl"],
                                         row["remaining_p10_bbl"], bbl),
        "Forecast life (mo)": str(int(row["forecast_months"])),
        "GOR, last 12 producing mo (mcf/bbl)": fmt(row["gor_mcf_per_bbl"], ".2f"),
        "Economic limit (bbl/mo)": fmt(row["econ_limit_bpm"], ",.0f"),
        f"{pv_label(row['econ_discount'])} P90 / P50 / P10": _p3(
            row["pv10_p90_usd"], row["pv10_usd"], row["pv10_p10_usd"], usd_mm),
        "Undiscounted net cash flow": usd_mm(row["ncf_usd"]),
        "Shut-in oil price (covers LOE now)": price(row["shut_in_oil_price"]),
        "Lateral (ft)": fmt(row["lateral_ft"], ",.0f"),
        "EUR / 1,000 ft": bbl(row["eur_per_1000ft"]),
        "Decline cluster": row["cluster"],
        "R²": fmt(row["r_squared"], ".3f"),
        "R² (log space)": fmt(row["r_squared_log"], ".3f"),
        "QC flags": row["qc_flags"] or "none",
    }


def cohort_metrics(ok: pd.DataFrame) -> dict:
    return {
        "Mean EUR (P50)": bbl(ok["eur_bbl"].mean()),
        "Mean EUR P90–P10": bbl_range(ok["eur_p90_bbl"].mean(), ok["eur_p10_bbl"].mean()),
        "Total remaining (P50)": bbl(ok["remaining_bbl"].sum()),
        f"Total {pv_label(ok['econ_discount'].iloc[0])} (P50)": usd_mm(ok["pv10_usd"].sum()),
        "Mean b-factor": f"{ok['b'].mean():.2f}",
        "Mean Di (nom. annual)": f"{ok['Di_annual'].mean():.0%}",
    }


# Ranking table: (column, source, kind). kind sets the scaling and display:
# "mbbl" thousand barrels, "usdmm" $ millions, others as named. "{pv}" in a column
# name becomes the results' pv_label, so take column names from ranking_table().
RANKING = [
    ("Lease", "lease_name", "text"), ("Well", "well_no", "text"), ("API", "api8", "int"),
    ("Cluster", "cluster", "text"), ("b", "b", "b"), ("Di ann.", "Di_annual", "pct"),
    ("Cum (Mbbl)", "cum_oil_bbl", "mbbl"),
    ("EUR P90 (Mbbl)", "eur_p90_bbl", "mbbl"), ("EUR P50 (Mbbl)", "eur_bbl", "mbbl"),
    ("EUR P10 (Mbbl)", "eur_p10_bbl", "mbbl"), ("Remaining P50 (Mbbl)", "remaining_bbl", "mbbl"),
    ("{pv} P50 ($MM)", "pv10_usd", "usdmm"), ("{pv} P90–P10 ($MM)", None, "pvrange"),
    ("Shut-in oil price ($/bbl)", "shut_in_oil_price", "price"),
    ("EUR/1k ft (Mbbl)", "eur_per_1000ft", "mbbl1"), ("R²", "r_squared", "r2"),
    ("QC flags", "qc_flags", "text"),
]


def ranking_table(ok: pd.DataFrame) -> pd.DataFrame:
    """Numeric ranking table (Mbbl, $MM), sorted by P50 EUR, with the PV columns
    named for ok's discount rate."""
    ok = ok.sort_values("eur_bbl", ascending=False)
    pv = pv_label(ok["econ_discount"].iloc[0])
    scale = {"mbbl": 1e-3, "mbbl1": 1e-3, "usdmm": 1e-6, "pct": 100.0}
    out = {}
    for name, col, kind in RANKING:
        name = name.replace("{pv}", pv)
        if kind == "pvrange":
            out[name] = [usd_mm_range(lo, hi).removesuffix(" MM").replace("$", "")
                         for lo, hi in zip(ok["pv10_p90_usd"], ok["pv10_p10_usd"])]
        elif kind == "text":
            out[name] = ok[col].fillna("").astype(str).to_numpy()
        else:
            out[name] = ok[col].to_numpy(float) * scale.get(kind, 1.0)
    return pd.DataFrame(out)


RANKING_FORMATS = {"int": "{:.0f}", "b": "{:.2f}", "pct": "{:.0f}%", "mbbl": "{:,.0f}",
                   "mbbl1": "{:,.1f}", "usdmm": "{:,.1f}", "price": "{:,.2f}", "r2": "{:.3f}"}


def ranking_kinds(table: pd.DataFrame) -> dict[str, str]:
    """Column name -> kind (see RANKING) for a table from ranking_table()."""
    if len(table.columns) != len(RANKING):
        raise ValueError("expected a table from ranking_table()")
    return dict(zip(table.columns, (kind for _, _, kind in RANKING)))


def ranking_cells(table: pd.DataFrame) -> list[list[str]]:
    """The ranking table as display strings (static page)."""
    kinds = ranking_kinds(table)
    rows = []
    for _, r in table.iterrows():
        rows.append([(str(v) if kinds[c] in ("text", "pvrange") else
                      "—" if pd.isna(v) else RANKING_FORMATS[kinds[c]].format(v)) or "—"
                     for c, v in r.items()])
    return rows
