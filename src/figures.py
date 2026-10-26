"""Plotly figures shared by the Streamlit app and the static Vercel build."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.arps import forecast_to_limit, terminal_rate
from src.data_loader import fitting_series


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


def cluster_figure(ok: pd.DataFrame):
    """Scatter of fitted (b, Di) colored by decline cluster."""
    fig = px.scatter(
        ok, x="b", y="Di_annual", color="cluster", hover_name="lease_name",
        hover_data=["api8", "eur_bbl", "r_squared"],
        labels={"b": "b-factor", "Di_annual": "Di (nominal annual)",
                "cluster": "Cluster"},
        height=480,
    )
    fig.update_traces(marker=dict(size=11))
    return fig
