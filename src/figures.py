"""Plotly figures shared by the Streamlit app and the static Vercel build."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.arps import B_CAP, fit_decline, forecast_to_limit, terminal_rate
from src.clustering import cluster_order
from src.data_loader import fitting_series

CLUSTER_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]


def well_figure(prod, row):
    """History + fitted decline + forecast chart for one well.

    The forecast is redrawn with the assumptions stored on the results row, so
    it is the same forecast that produced the row's EUR and remaining volume.
    """
    api8 = int(row["api8"])
    dmin_m = float(row["dmin_annual"]) / 12.0
    econ_bpm = float(row["econ_limit_bpm"])
    max_months = int(row["forecast_month_cap"])
    full = prod[prod["api8"] == api8].copy()
    # History from the peak, on the producing-time clock the fit used (the fit
    # is deterministic, so refitting reproduces it): the fitted line pauses
    # through shut-ins and curtailments instead of declining across them.
    hist = fitting_series(prod, api8)
    _, clock = fit_decline(hist["oil_bbl"].to_numpy(), qi_max=float(hist["oil_bbl"].max()),
                           b_cap=float(row.get("b_cap", B_CAP)))
    q_fit_hist = terminal_rate(clock, row["qi"], row["Di_monthly"], row["b"], dmin_m)

    fc = forecast_to_limit(row["qi"], row["Di_monthly"], row["b"],
                           Dmin=dmin_m, econ_limit_bpm=econ_bpm,
                           max_months=max_months, start_t=float(row["producing_months"]))
    last_date = full["date"].max()
    fc_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                             periods=len(fc), freq="MS")
    fc_name = (f"Forecast (capped at {int(row['max_life_months']) // 12}-yr life)"
               if len(fc) == max_months else f"Forecast to {econ_bpm:,.0f} bbl/mo limit")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=full["date"], y=full["oil_bbl"], name="Actual oil (bbl/mo)",
                         marker_color="#9ecae1"))
    fig.add_trace(go.Scatter(x=hist["date"], y=q_fit_hist, name="Fitted decline",
                             line=dict(color="#e6550d", width=3)))
    if len(fc):
        fig.add_trace(go.Scatter(x=fc_dates, y=fc, name=fc_name,
                                 line=dict(color="#e6550d", width=3, dash="dash")))
    fig.add_vline(x=last_date, line_dash="dot", line_color="gray",
                  annotation_text="last reported month")
    fig.update_layout(
        title=f"{row['lease_name']} ({row['well_no']}) — API {api8}",
        xaxis_title="Date", yaxis_title="Oil (bbl/month)",
        hovermode="x unified", height=460, legend=dict(orientation="h", y=1.08),
        updatemenus=[dict(
            type="buttons", direction="left", x=1.0, xanchor="right", y=1.18,
            buttons=[dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                     dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}])],
        )],
    )
    return fig


def type_curve_figure(tc: pd.DataFrame):
    """Observed P25-P75 band and P50 oil rate by month on production."""
    x = tc["month_on_production"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=tc["p75_oil_bbl"], line=dict(width=0),
                             name="P75", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=tc["p25_oil_bbl"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(230,85,13,0.18)", name="P25–P75",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=tc["p50_oil_bbl"], name="P50 oil (bbl/mo)",
                             line=dict(color="#e6550d", width=3),
                             customdata=tc[["p25_oil_bbl", "p75_oil_bbl", "well_count"]],
                             hovertemplate="Month %{x}: P50 %{y:,.0f} bbl<br>P25–P75 "
                                           "%{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
                                           "<br>%{customdata[2]} wells<extra></extra>"))
    fig.update_layout(xaxis_title="Month on production", yaxis_title="Oil (bbl/month)",
                      height=380, legend=dict(orientation="h", y=1.1), hovermode="x")
    return fig


def cluster_figure(ok: pd.DataFrame):
    """Rate kept after 1 vs 3 producing years (the clustering features),
    colored by decline cluster."""
    order = cluster_order(ok)
    plot = ok.assign(QC=ok["low_confidence"].map({True: "Low confidence", False: "OK"}))
    fig = px.scatter(
        plot, x="retention_12mo", y="retention_36mo", color="cluster", symbol="QC",
        hover_name="lease_name",
        hover_data={"api8": True, "eur_bbl": ":,.0f", "b": ":.2f", "Di_annual": ":.0%",
                    "r_squared": ":.3f", "retention_12mo": ":.0%",
                    "retention_36mo": ":.0%", "qc_flags": True},
        category_orders={"cluster": order, "QC": ["OK", "Low confidence"]},
        color_discrete_map=dict(zip(order, CLUSTER_COLORS)),
        symbol_map={"OK": "circle", "Low confidence": "circle-open"},
        labels={"retention_12mo": "Rate kept after 1 producing year (% of peak)",
                "retention_36mo": "Rate kept after 3 producing years (% of peak)",
                "b": "b-factor", "Di_annual": "Di (nominal annual)",
                "cluster": "Cluster", "eur_bbl": "EUR (bbl)", "r_squared": "R²",
                "qc_flags": "QC flags"},
        height=480,
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=2)))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig
