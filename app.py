"""Midland Basin — Automated Decline Curve Analysis dashboard.

Run:  streamlit run app.py   (from the repo root)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.arps import forecast_to_limit, terminal_rate
from src.data_loader import fitting_series, load_production

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(
    page_title="Midland Basin DCA Dashboard",
    page_icon="🛢️",
    layout="wide",
)

DATA_NOTE = (
    "Production: public Texas Railroad Commission (RRC) monthly production data, "
    "Martin County, TX — 30 horizontal oil wells, single-well leases, "
    "Jan 2022–Jun 2026. Obtained via the open dataset "
    "Dansaad2020/ai-ml-dca-permian (GitHub), which republishes RRC PDQ/wellbore/completion "
    "records; RRC's interactive query endpoints were unreachable at build time, so the "
    "data was not pulled directly from rrc.texas.gov. See README for full provenance."
)


@st.cache_data
def load_all():
    results_path = os.path.join(RESULTS, "midland_basin_dca_results.csv")
    if not os.path.exists(results_path):
        # First run (e.g. fresh Streamlit Cloud deploy): build results.
        from src.pipeline import run as run_pipeline
        with st.spinner("Fitting decline curves for 30 wells (one-time setup)…"):
            run_pipeline()
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    results = pd.read_csv(results_path)
    return prod, results


def well_figure(prod, row, dmin_m: float, econ_bpm: float):
    """History + fitted decline + forecast chart for one well."""
    api8 = int(row["api8"])
    full = prod[prod["api8"] == api8].copy()
    fit_df = fitting_series(prod, api8)
    peak_mop = int(fit_df["month_on_production"].iloc[0])
    last_mop = int(full["month_on_production"].max())
    t_now = last_mop - peak_mop + 1

    hist = full[full["month_on_production"] >= peak_mop].copy()
    t_hist = (hist["month_on_production"] - peak_mop + 1).to_numpy()
    q_fit_hist = terminal_rate(t_hist, row["qi"], row["Di_monthly"], row["b"], dmin_m)

    fc = forecast_to_limit(row["qi"], row["Di_monthly"], row["b"],
                           Dmin=dmin_m, econ_limit_bpm=econ_bpm,
                           start_t=t_now + 1)
    last_date = full["date"].max()
    fc_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                             periods=len(fc), freq="MS")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=full["date"], y=full["oil_bbl"], name="Actual oil (bbl/mo)",
                         marker_color="#9ecae1"))
    fig.add_trace(go.Scatter(x=hist["date"], y=q_fit_hist, name="Fitted decline",
                             line=dict(color="#e6550d", width=3)))
    if len(fc):
        fig.add_trace(go.Scatter(x=fc_dates, y=fc, name="Forecast to econ. limit",
                                 line=dict(color="#e6550d", width=3, dash="dash")))
        fig.add_vline(x=last_date, line_dash="dot", line_color="gray",
                      annotation_text="end of history")
    fig.update_layout(
        title=f"{row['lease_name']} ({row['well_no']}) — API {api8}",
        xaxis_title="Date", yaxis_title="Oil (bbl/month)",
        hovermode="x unified", height=460, legend=dict(orientation="h", y=1.08),
    )
    return fig


# ----------------------------------------------------------------------------
prod, results = load_all()
ok = results[results["status"] == "ok"].copy()

st.title("🛢️ Midland Basin — Automated Decline Curve Analysis")
st.caption("Martin County, TX · 30 horizontal oil wells · "
           "Hyperbolic Arps with 5%/yr terminal decline · Economic limit 150 bbl/mo")
st.info(DATA_NOTE, icon="📊")

tab_well, tab_cohort, tab_clusters = st.tabs(
    ["Well detail", "Cohort overview", "Decline clusters"])

# ----------------------------------------------------------------------------
with tab_well:
    labels = {f"{r.lease_name} ({r.well_no}) — API {int(r.api8)}": int(r.api8)
              for r in ok.itertuples()}
    choice = st.selectbox("Select well / lease", sorted(labels.keys()))
    row = ok[ok["api8"] == labels[choice]].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EUR", f"{row['eur_bbl']:,.0f} bbl")
    c2.metric("Remaining", f"{row['remaining_bbl']:,.0f} bbl")
    c3.metric("b-factor", f"{row['b']:.2f}")
    c4.metric("Di (nom. annual)", f"{row['Di_annual']:.0%}")
    c5.metric("Fit R²", f"{row['r_squared']:.3f}")

    st.plotly_chart(well_figure(prod, row, 0.05 / 12, 150.0), use_container_width=True)

    st.subheader("Well economics & fit")
    detail = pd.DataFrame([{
        "Lease": row["lease_name"], "Well": row["well_no"], "API": int(row["api8"]),
        "Operator": row["operator"], "Field": row["field"],
        "First prod": row["first_prod"], "History (mo)": int(row["months_history"]),
        "Cum oil (bbl)": f"{row['cum_oil_bbl']:,.0f}",
        "qi (bbl/mo)": f"{row['qi']:,.0f}",
        "Di annual (nom)": f"{row['Di_annual']:.0%}",
        "b-factor": round(row["b"], 3),
        "EUR (bbl)": f"{row['eur_bbl']:,.0f}",
        "Remaining (bbl)": f"{row['remaining_bbl']:,.0f}",
        "Forecast life (mo)": int(row["forecast_months"]),
        "Lateral (ft)": f"{row['lateral_ft']:,.0f}",
        "EUR / 1,000 ft": f"{row['eur_per_1000ft']:,.0f} bbl",
        "Decline cluster": row["cluster"],
        "R²": round(row["r_squared"], 3),
    }]).T
    detail.columns = ["Value"]
    st.table(detail)

# ----------------------------------------------------------------------------
with tab_cohort:
    st.subheader("Cohort summary — 30 wells")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean b-factor", f"{ok['b'].mean():.2f}")
    m2.metric("Mean Di (nom. annual)", f"{ok['Di_annual'].mean():.0%}")
    m3.metric("Mean EUR", f"{ok['eur_bbl'].mean():,.0f} bbl")
    m4.metric("Total remaining", f"{ok['remaining_bbl'].sum():,.0f} bbl")

    st.subheader("EUR ranking")
    table = ok[["lease_name", "well_no", "api8", "qi", "Di_annual", "b",
                "cum_oil_bbl", "eur_bbl", "remaining_bbl", "eur_per_1000ft",
                "cluster", "r_squared"]].copy()
    table.columns = ["Lease", "Well", "API", "qi (bbl/mo)", "Di ann.",
                     "b", "Cum (bbl)", "EUR (bbl)", "Remaining (bbl)",
                     "EUR/1k ft", "Cluster", "R²"]
    st.dataframe(table.sort_values("EUR (bbl)", ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download full results CSV",
        results.to_csv(index=False).encode(),
        "midland_basin_dca_results.csv", "text/csv")

# ----------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Decline-behavior clusters (k-means on Di, b)")
    st.caption("Wells with similar decline shapes group together — the basis for "
               "type curves and performance benchmarking.")
    fig = px.scatter(
        ok, x="b", y="Di_annual", color="cluster", hover_name="lease_name",
        hover_data=["api8", "eur_bbl", "r_squared"],
        labels={"b": "b-factor", "Di_annual": "Di (nominal annual)",
                "cluster": "Cluster"},
        height=480,
    )
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    summ = (ok.groupby("cluster")
              .agg(wells=("api8", "count"), mean_b=("b", "mean"),
                   mean_Di_ann=("Di_annual", "mean"),
                   mean_EUR=("eur_bbl", "mean"),
                   total_EUR=("eur_bbl", "sum"),
                   mean_R2=("r_squared", "mean"))
              .round({"mean_b": 2, "mean_Di_ann": 2, "mean_EUR": 0,
                      "total_EUR": 0, "mean_R2": 3})
              .reset_index())
    st.table(summ)

st.caption("Built as a freelance portfolio project · methodology: Arps hyperbolic "
           "decline fit from peak rate (zeros excluded), 5%/yr terminal decline, "
           "EUR to 150 bbl/mo economic limit · k-means (k=3) on (Di, b).")
