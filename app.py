"""Midland Basin — Automated Decline Curve Analysis dashboard.

Run:  streamlit run app.py   (from the repo root)
"""
from __future__ import annotations

import os
from dataclasses import asdict, astuple, replace

import pandas as pd
import streamlit as st

from src.clustering import cluster_summary
from src.data_loader import load_production
from src.economics import Economics
from src.figures import cluster_figure, type_curve_figure, well_figure
from src.pipeline import (DMIN_ANNUAL, ECONOMICS, ENSEMBLE_CSV, MAX_LIFE_MONTHS,
                          RESULTS_CSV, load_ensemble, load_results)
from src.pipeline import run as run_pipeline
from src.report import (CLUSTER_NOTE, RANGE_NOTE, caption, cohort_metrics, data_note,
                        download_frame, econ_limit_text, economics_note, methodology,
                        ranking_kinds, ranking_table, well_detail, well_label, well_metrics,
                        well_ranges)
from src.type_curve import observed_type_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

st.set_page_config(
    page_title="Midland Basin DCA Dashboard",
    page_icon="🛢️",
    layout="wide",
)

NUM = st.column_config.NumberColumn


@st.cache_data
def load_all(results_mtime: float):
    """Load production, results and forecast ensembles; keyed on the results
    file's mtime so a re-run pipeline is picked up without restarting."""
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    return prod, load_results(RESULTS_CSV), load_ensemble(ENSEMBLE_CSV)


@st.cache_data(show_spinner="Re-forecasting all wells with these assumptions…")
def scenario(dmin_annual: float, max_life_months: int, economics: tuple,
             results_mtime: float):
    """Results under non-base-case assumptions (not written to disk). The fits
    and ensembles do not depend on these, so the saved ensembles are reused.
    Keyed on the results file's mtime too, so a re-run pipeline invalidates
    cached cases along with the base case."""
    return run_pipeline(dmin_annual=dmin_annual, max_life_months=max_life_months,
                        economics=Economics(*economics),
                        ensembles=load_all(results_mtime)[2], write=False)


# Sidebar sliders: key -> (label, min, max, step, model value / slider value,
# display format). Percent sliders divide by 100, so a slider at the base value
# gives back exactly the base-case number.
SLIDERS = {
    "oil_price": ("Oil price (\\$/bbl, flat)", 30.0, 120.0, 1.0, 1, "%.0f"),
    "gas_price": ("Gas price (\\$/mcf, flat)", 0.0, 6.0, 0.25, 1, "%.2f"),
    "nri": ("Net revenue interest (%)", 62.5, 87.5, 0.5, 100, "%.1f"),
    "loe_fixed": ("Fixed LOE (\\$/well-month)", 1000.0, 15000.0, 500.0, 1, "%.0f"),
    "loe_var": ("Variable LOE (\\$/bbl)", 0.0, 12.0, 0.5, 1, "%.2f"),
    "discount": ("Discount rate (%/yr)", 5.0, 20.0, 1.0, 100, "%.0f"),
    "dmin": ("Terminal decline Dmin (%/yr, nominal)", 4.0, 12.0, 1.0, 100, "%.0f"),
    "life_months": ("Maximum well life (years)", 20.0, 60.0, 5.0, 1 / 12, "%.0f"),
}
BASE_MODEL = {**asdict(ECONOMICS), "dmin": DMIN_ANNUAL, "life_months": MAX_LIFE_MONTHS}
BASE = {k: round(BASE_MODEL[k] * v[4], 6) for k, v in SLIDERS.items()}


def to_model(values: dict) -> tuple[float, int, Economics]:
    """Slider values -> (Dmin, life in months, Economics)."""
    m = {k: v / SLIDERS[k][4] for k, v in values.items()}
    econ = replace(ECONOMICS, **{k: m[k] for k in asdict(ECONOMICS) if k in m})
    return m["dmin"], int(round(m["life_months"])), econ


def md(text: str) -> str:
    """Escape dollar signs, which Streamlit markdown would read as LaTeX math."""
    return text.replace("$", "\\$")


def _reset_assumptions():
    st.session_state.update(BASE)


# ----------------------------------------------------------------------------
if not (os.path.exists(RESULTS_CSV) and os.path.exists(ENSEMBLE_CSV)):
    # First run (e.g. fresh Streamlit Cloud deploy): build results.
    with st.spinner("Fitting decline curves and forecast ranges (one-time setup)…"):
        run_pipeline()
prod, results, ensembles = load_all(os.path.getmtime(RESULTS_CSV))

with st.sidebar:
    st.header("Assumptions")
    for key, value in BASE.items():
        st.session_state.setdefault(key, value)
    values = {}
    for i, (key, (label, lo, hi, step, _, fmt)) in enumerate(SLIDERS.items()):
        if i == 0:
            st.subheader("Prices & costs")
        elif key == "dmin":
            st.subheader("Decline")
        values[key] = st.slider(label, lo, hi, step=step, format=fmt, key=key)
    st.button("Reset to base case", on_click=_reset_assumptions)
    st.caption(f"Texas severance {ECONOMICS.oil_sev:.1%} oil / {ECONOMICS.gas_sev:.1%} gas "
               f"and ad valorem {ECONOMICS.ad_valorem:.1%} are fixed. Each well's economic "
               "limit is where its cash flow turns negative. Fits do not change; the "
               "forecast tail, EUR and value do. The static Vercel page shows the base case.")
if values != BASE:
    dmin, life_months, econ = to_model(values)
    results = scenario(dmin, life_months, astuple(econ), os.path.getmtime(RESULTS_CSV))
    st.sidebar.info("Showing a sensitivity case, not the base case.")
st.sidebar.caption(econ_limit_text(results).capitalize() + ".")
ok = results[results["status"] == "ok"].copy()

st.title("🛢️ Midland Basin — Automated Decline Curve Analysis")
st.caption(md(caption(results)))
st.info(data_note(results), icon="📊")
if ok.empty:
    st.error("No wells were fitted successfully, so there is nothing to show.")
    st.stop()

tab_well, tab_cohort, tab_clusters = st.tabs(
    ["Well detail", "Cohort overview", "Decline clusters"])

# ----------------------------------------------------------------------------
with tab_well:
    labels = {well_label(r): int(r["api8"]) for _, r in ok.iterrows()}
    choice = st.selectbox("Select well / lease", sorted(labels.keys()))
    row = ok[ok["api8"] == labels[choice]].iloc[0]
    if row["qc_flags"]:
        st.warning(md(f"QC: {row['qc_flags']}"), icon="⚠️")

    ranges = well_ranges(row)
    for col, (k, v) in zip(st.columns(5), well_metrics(row).items()):
        col.metric(k, v)
        if k in ranges:
            col.caption(md(f"P90–P10: {ranges[k]}"))
    st.caption(RANGE_NOTE)

    st.plotly_chart(well_figure(prod, row, ensembles.get(int(row["api8"]))), width="stretch")

    st.subheader("Well economics & fit")
    detail = well_detail(row)
    st.table(pd.DataFrame({"Value": [str(v) for v in detail.values()]},
                          index=list(detail.keys())))
    st.caption(md(economics_note(results)))

# ----------------------------------------------------------------------------
with tab_cohort:
    fitted = f"{len(ok)} wells" + (f" fitted of {len(results)}" if len(ok) < len(results) else "")
    st.subheader(f"Cohort summary — {fitted}")
    items = list(cohort_metrics(ok).items())
    for row_items in (items[:3], items[3:]):
        for col, (k, v) in zip(st.columns(3), row_items):
            col.metric(k, v)
    st.caption("Totals add up each well's P50; a sum of well P90s would understate the "
               "cohort's own low case, so only P50 totals are shown.")

    st.subheader("EUR ranking")
    formats = {"int": "%d", "b": "%.2f", "pct": "%.0f%%", "mbbl": "%,.0f", "mbbl1": "%,.1f",
               "usdmm": "%,.1f", "price": "%,.2f", "r2": "%.3f"}
    ranking = ranking_table(ok)  # PV column names follow the discount rate
    st.dataframe(
        ranking, width="stretch", hide_index=True,
        column_config={name: NUM(format=formats[kind])
                       for name, kind in ranking_kinds(ranking).items() if kind in formats})
    st.caption(md("Mbbl = thousand barrels, $MM = million dollars; forecast volumes are "
                  "rounded to the nearest thousand barrels. " + economics_note(results)))
    st.download_button(
        "⬇ Download full results CSV (current assumptions)",
        download_frame(results).to_csv(index=False).encode(),
        "midland_basin_dca_results.csv", "text/csv")

    st.subheader("Observed type curve")
    st.caption("Monthly oil by month on production across the cohort (P25–P75 band, "
               "P50 line). It stops where fewer than 80% of the wells are still "
               "reporting; later months would describe a small, older subset.")
    st.plotly_chart(type_curve_figure(observed_type_curve(prod)), width="stretch")

# ----------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Decline-shape clusters")
    st.caption(CLUSTER_NOTE)
    st.plotly_chart(cluster_figure(ok), width="stretch")

    summ = cluster_summary(results)
    for col in ("mean_Di_annual", "mean_retention_12mo", "mean_retention_36mo"):
        summ[col] *= 100
    for col in ("mean_EUR_bbl", "total_EUR_bbl", "mean_EUR_per_1000ft"):
        summ[col] /= 1000
    st.dataframe(
        summ, width="stretch", hide_index=True,
        column_config={
            "cluster": "Cluster", "wells": "Wells",
            "mean_retention_12mo": NUM("Rate kept, 1 yr", format="%.0f%%"),
            "mean_retention_36mo": NUM("Rate kept, 3 yr", format="%.0f%%"),
            "mean_b": NUM("Mean b", format="%.2f"),
            "mean_Di_annual": NUM("Mean Di ann.", format="%.0f%%"),
            "mean_EUR_bbl": NUM("Mean EUR (Mbbl)", format="%,.0f"),
            "total_EUR_bbl": NUM("Total EUR (Mbbl)", format="%,.0f"),
            "mean_EUR_per_1000ft": NUM("Mean EUR/1k ft (Mbbl)", format="%,.1f"),
            "mean_R2": NUM("Mean R²", format="%.3f"),
        })

st.caption("Built as a freelance portfolio project · " + methodology(results))
