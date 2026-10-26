"""Midland Basin — Automated Decline Curve Analysis dashboard.

Run:  streamlit run app.py   (from the repo root)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.clustering import cluster_summary
from src.data_loader import load_production
from src.figures import cluster_figure, type_curve_figure, well_figure
from src.pipeline import (DMIN_ANNUAL, ECON_LIMIT_BPM, MAX_LIFE_MONTHS, RESULTS_CSV,
                          load_results)
from src.pipeline import run as run_pipeline
from src.report import (caption, data_note, methodology, well_detail, well_label,
                        well_metrics)
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
    """Load production + results; keyed on the results file's mtime so a
    re-run pipeline is picked up without restarting the server."""
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    return prod, load_results(RESULTS_CSV)


@st.cache_data(show_spinner="Re-forecasting all wells with these assumptions…")
def scenario(dmin_annual: float, econ_limit_bpm: float, max_life_months: int):
    """Results under non-base-case forecast assumptions (not written to disk)."""
    return run_pipeline(econ_limit_bpm=econ_limit_bpm, dmin_annual=dmin_annual,
                        max_life_months=max_life_months, write=False)


BASE = {"dmin_pct": round(DMIN_ANNUAL * 100), "econ": int(ECON_LIMIT_BPM),
        "life_yr": MAX_LIFE_MONTHS // 12}


def _reset_assumptions():
    st.session_state.update(BASE)


# ----------------------------------------------------------------------------
if not os.path.exists(RESULTS_CSV):
    # First run (e.g. fresh Streamlit Cloud deploy): build results.
    with st.spinner("Fitting decline curves (one-time setup)…"):
        run_pipeline()
prod, results = load_all(os.path.getmtime(RESULTS_CSV))

with st.sidebar:
    st.header("Forecast assumptions")
    for key, value in BASE.items():
        st.session_state.setdefault(key, value)
    dmin_pct = st.slider("Terminal decline Dmin (%/yr, nominal)", 4, 12, key="dmin_pct")
    econ = st.slider("Economic limit (bbl/month)", 50, 400, step=25, key="econ")
    life_yr = st.slider("Maximum well life (years)", 20, 60, step=5, key="life_yr")
    st.button("Reset to base case", on_click=_reset_assumptions)
    st.caption("Fits do not change; only the forecast tail and EUR do. The static "
               "Vercel page shows the base case.")
if (dmin_pct, econ, life_yr) != (BASE["dmin_pct"], BASE["econ"], BASE["life_yr"]):
    results = scenario(dmin_pct / 100, float(econ), life_yr * 12)
    st.sidebar.info("Showing a sensitivity case, not the base case.")
ok = results[results["status"] == "ok"].copy()

st.title("🛢️ Midland Basin — Automated Decline Curve Analysis")
st.caption(caption(results))
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
        st.warning(f"QC: {row['qc_flags']}", icon="⚠️")

    for col, (k, v) in zip(st.columns(5), well_metrics(row).items()):
        col.metric(k, v)

    st.plotly_chart(well_figure(prod, row), width="stretch")

    st.subheader("Well economics & fit")
    detail = well_detail(row)
    st.table(pd.DataFrame({"Value": [str(v) for v in detail.values()]},
                          index=list(detail.keys())))

# ----------------------------------------------------------------------------
with tab_cohort:
    fitted = f"{len(ok)} wells" + (f" fitted of {len(results)}" if len(ok) < len(results) else "")
    st.subheader(f"Cohort summary — {fitted}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean b-factor", f"{ok['b'].mean():.2f}")
    m2.metric("Mean Di (nom. annual)", f"{ok['Di_annual'].mean():.0%}")
    m3.metric("Mean EUR", f"{ok['eur_bbl'].mean():,.0f} bbl")
    m4.metric("Total remaining", f"{ok['remaining_bbl'].sum():,.0f} bbl")

    st.subheader("EUR ranking")
    table = pd.DataFrame({
        "Lease": ok["lease_name"], "Well": ok["well_no"], "API": ok["api8"],
        "qi (bbl/mo)": ok["qi"], "Di ann.": ok["Di_annual"] * 100, "b": ok["b"],
        "Cum (bbl)": ok["cum_oil_bbl"], "EUR (bbl)": ok["eur_bbl"],
        "Remaining (bbl)": ok["remaining_bbl"], "EUR/1k ft": ok["eur_per_1000ft"],
        "Cluster": ok["cluster"], "R²": ok["r_squared"], "QC flags": ok["qc_flags"],
    })
    st.dataframe(
        table.sort_values("EUR (bbl)", ascending=False), width="stretch", hide_index=True,
        column_config={
            "API": NUM(format="%d"), "qi (bbl/mo)": NUM(format="%,.0f"),
            "Di ann.": NUM(format="%.0f%%"), "b": NUM(format="%.2f"),
            "Cum (bbl)": NUM(format="%,.0f"), "EUR (bbl)": NUM(format="%,.0f"),
            "Remaining (bbl)": NUM(format="%,.0f"), "EUR/1k ft": NUM(format="%,.0f"),
            "R²": NUM(format="%.3f"),
        })
    st.download_button(
        "⬇ Download full results CSV (current assumptions)",
        results.to_csv(index=False).encode(),
        "midland_basin_dca_results.csv", "text/csv")

    st.subheader("Observed type curve")
    st.caption("Monthly oil by month on production across the cohort (P25–P75 band, "
               "P50 line); months with fewer than 10 reporting wells are omitted.")
    st.plotly_chart(type_curve_figure(observed_type_curve(prod)), width="stretch")

# ----------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Decline-behavior clusters (k-means on log Di, b)")
    st.caption("Wells with similar decline shapes group together — the basis for "
               "type curves and performance benchmarking. Clusters are named by their "
               "centroid's initial decline (Di) and b-factor; a high b flattens the "
               "decline, so high-b wells can have the largest EURs. "
               "Hollow markers are low-confidence fits.")
    st.plotly_chart(cluster_figure(ok), width="stretch")

    summ = cluster_summary(results)
    summ["mean_Di_annual"] *= 100
    st.dataframe(
        summ, width="stretch", hide_index=True,
        column_config={
            "cluster": "Cluster", "wells": "Wells",
            "mean_b": NUM("Mean b", format="%.2f"),
            "mean_Di_annual": NUM("Mean Di ann.", format="%.0f%%"),
            "mean_EUR_bbl": NUM("Mean EUR (bbl)", format="%,.0f"),
            "total_EUR_bbl": NUM("Total EUR (bbl)", format="%,.0f"),
            "mean_EUR_per_1000ft": NUM("Mean EUR/1k ft (bbl)", format="%,.0f"),
            "mean_R2": NUM("Mean R²", format="%.3f"),
        })

st.caption("Built as a freelance portfolio project · " + methodology(results))
