"""Build a static, Vercel-ready version of the dashboard into public/.

Uses the same analysis code (src/) and the same RRC-sourced data (data/) as the
Streamlit app; results are precomputed at build time, charts render client-side
with Plotly.js.

Run:  python scripts/build_static.py   (from the repo root)
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from src.data_loader import load_production  # noqa: E402
from src.figures import cluster_figure, well_figure  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402

DATA = os.path.join(ROOT, "data")
PUBLIC = os.path.join(ROOT, "public")

DATA_NOTE = (
    "Production: real Texas Railroad Commission (RRC) monthly production records, "
    "Martin County, TX — 30 horizontal oil wells (Pioneer Natural Resources, Spraberry "
    "Trend Area), single-well leases, Jan 2022–Jun 2026. Obtained via the open dataset "
    '<a href="https://github.com/Dansaad2020/ai-ml-dca-permian">Dansaad2020/ai-ml-dca-permian</a>, '
    "which republishes public RRC production, permit, wellbore and completion records; "
    "RRC's interactive query endpoints were unreachable at build time, so the data was "
    "not pulled directly from rrc.texas.gov. No production numbers were fabricated."
)


def fmt(v, spec):
    return "" if pd.isna(v) else format(v, spec)


def main() -> None:
    results = run_pipeline()
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    ok = results[results["status"] == "ok"].copy()

    wells = {}
    for r in ok.itertuples():
        row = ok[ok["api8"] == r.api8].iloc[0]
        fig = well_figure(prod, row, 0.05 / 12, 150.0)
        wells[str(int(r.api8))] = {
            "label": f"{r.lease_name} ({r.well_no}) — API {int(r.api8)}",
            "fig": json.loads(fig.to_json()),
            "metrics": {
                "EUR": f"{r.eur_bbl:,.0f} bbl",
                "Remaining": f"{r.remaining_bbl:,.0f} bbl",
                "b-factor": f"{r.b:.2f}",
                "Di (nom. annual)": f"{r.Di_annual:.0%}",
                "Fit R²": f"{r.r_squared:.3f}",
            },
            "detail": {
                "Lease": r.lease_name, "Well": r.well_no, "API": str(int(r.api8)),
                "Operator": r.operator, "Field": r.field,
                "First prod": str(r.first_prod), "History (mo)": str(int(r.months_history)),
                "Cum oil (bbl)": f"{r.cum_oil_bbl:,.0f}",
                "qi (bbl/mo)": f"{r.qi:,.0f}",
                "Di annual (nom)": f"{r.Di_annual:.0%}",
                "b-factor": f"{r.b:.3f}",
                "EUR (bbl)": f"{r.eur_bbl:,.0f}",
                "Remaining (bbl)": f"{r.remaining_bbl:,.0f}",
                "Forecast life (mo)": str(int(r.forecast_months)),
                "Lateral (ft)": fmt(r.lateral_ft, ",.0f"),
                "EUR / 1,000 ft": fmt(r.eur_per_1000ft, ",.0f") + " bbl",
                "Decline cluster": r.cluster,
                "R²": f"{r.r_squared:.3f}",
            },
        }

    cohort = {
        "Mean b-factor": f"{ok['b'].mean():.2f}",
        "Mean Di (nom. annual)": f"{ok['Di_annual'].mean():.0%}",
        "Mean EUR": f"{ok['eur_bbl'].mean():,.0f} bbl",
        "Total remaining": f"{ok['remaining_bbl'].sum():,.0f} bbl",
    }
    rank = ok.sort_values("eur_bbl", ascending=False)
    rank_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in [
            r.lease_name, r.well_no, int(r.api8), f"{r.qi:,.0f}", f"{r.Di_annual:.0%}",
            f"{r.b:.2f}", f"{r.cum_oil_bbl:,.0f}", f"{r.eur_bbl:,.0f}",
            f"{r.remaining_bbl:,.0f}", fmt(r.eur_per_1000ft, ",.0f"), r.cluster,
            f"{r.r_squared:.3f}"]) + "</tr>"
        for r in rank.itertuples())

    summ = (ok.groupby("cluster")
              .agg(wells=("api8", "count"), mean_b=("b", "mean"),
                   mean_Di_ann=("Di_annual", "mean"), mean_EUR=("eur_bbl", "mean"),
                   total_EUR=("eur_bbl", "sum"), mean_R2=("r_squared", "mean"))
              .reset_index())
    summ_rows = "".join(
        f"<tr><td>{html.escape(s.cluster)}</td><td>{s.wells}</td><td>{s.mean_b:.2f}</td>"
        f"<td>{s.mean_Di_ann:.0%}</td><td>{s.mean_EUR:,.0f}</td><td>{s.total_EUR:,.0f}</td>"
        f"<td>{s.mean_R2:.3f}</td></tr>" for s in summ.itertuples())

    payload = {
        "wells": dict(sorted(wells.items(), key=lambda kv: kv[1]["label"])),
        "clusterFig": json.loads(cluster_figure(ok).to_json()),
    }

    tpl = open(os.path.join(ROOT, "scripts", "template.html"), encoding="utf-8").read()
    page = (tpl.replace("{{DATA_NOTE}}", DATA_NOTE)
               .replace("{{COHORT}}", "".join(
                   f'<div class="metric"><span>{k}</span><b>{v}</b></div>'
                   for k, v in cohort.items()))
               .replace("{{RANK_ROWS}}", rank_rows)
               .replace("{{SUMM_ROWS}}", summ_rows)
               .replace("{{PAYLOAD}}", json.dumps(payload, separators=(",", ":"))
                        .replace("</", "<\\/")))

    os.makedirs(PUBLIC, exist_ok=True)
    with open(os.path.join(PUBLIC, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    shutil.copy(os.path.join(ROOT, "results", "midland_basin_dca_results.csv"),
                os.path.join(PUBLIC, "midland_basin_dca_results.csv"))
    print(f"wrote public/index.html ({len(page)/1e6:.2f} MB), {len(wells)} wells")


if __name__ == "__main__":
    main()
