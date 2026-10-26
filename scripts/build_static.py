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
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plotly.offline import get_plotlyjs_version  # noqa: E402

from src.clustering import cluster_summary  # noqa: E402
from src.data_loader import load_production  # noqa: E402
from src.figures import cluster_figure, type_curve_figure, well_figure  # noqa: E402
from src.pipeline import (RESULTS_CSV, TYPE_CURVE_CSV, load_ensemble,  # noqa: E402
                          run as run_pipeline)
from src.report import (CLUSTER_NOTE, RANGE_NOTE, caption, cohort_metrics,  # noqa: E402
                        data_note, economics_note, fmt, methodology, ranking_cells,
                        ranking_table, well_detail, well_label, well_metrics, well_ranges)
from src.type_curve import observed_type_curve  # noqa: E402

DATA = os.path.join(ROOT, "data")
PUBLIC = os.path.join(ROOT, "public")
# Same plotly.js version the installed plotly.py generates figure JSON for.
PLOTLY_SRC = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def cells(values) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"


def script_json(obj) -> str:
    """JSON that is safe inside a <script> element: no '<', '>' or '&' can end
    the element or open a comment, whatever the data contains."""
    return (json.dumps(obj, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def main() -> None:
    results = run_pipeline()
    ensembles = load_ensemble()
    prod = load_production(
        os.path.join(DATA, "martin_selected_30_monthly_production_normalized.csv"))
    ok = results[results["status"] == "ok"].copy()

    wells = {}
    for _, row in ok.iterrows():
        api8 = int(row["api8"])
        wells[str(api8)] = {
            "label": well_label(row),
            "qc": row["qc_flags"],
            "fig": json.loads(well_figure(prod, row, ensembles.get(api8)).to_json()),
            "metrics": well_metrics(row),
            "ranges": well_ranges(row),
            "detail": {k: str(v) for k, v in well_detail(row).items()},
        }

    ranking = ranking_table(ok)  # PV column names follow the discount rate
    rank_head = "<tr>" + "".join(f"<th>{html.escape(name)}</th>" for name in ranking.columns) + "</tr>"
    rank_rows = "".join(cells(r) for r in ranking_cells(ranking))
    summ_rows = "".join(cells([
        s.cluster, s.wells, fmt(s.mean_retention_12mo, ".0%"),
        fmt(s.mean_retention_36mo, ".0%"), fmt(s.mean_b, ".2f"), fmt(s.mean_Di_annual, ".0%"),
        fmt(s.mean_EUR_bbl / 1e3, ",.0f"), fmt(s.total_EUR_bbl / 1e3, ",.0f"),
        fmt(s.mean_EUR_per_1000ft / 1e3, ",.1f"), fmt(s.mean_R2, ".3f")])
        for s in cluster_summary(results).itertuples())

    payload = {
        "wells": dict(sorted(wells.items(), key=lambda kv: kv[1]["label"])),
        "clusterFig": json.loads(cluster_figure(ok).to_json()),
        "typeCurveFig": json.loads(type_curve_figure(observed_type_curve(prod)).to_json()),
    }
    fitted = f"{len(ok)} wells" + (f" fitted of {len(results)}" if len(ok) < len(results) else "")
    parts = {
        "PLOTLY_SRC": PLOTLY_SRC,
        "CAPTION": html.escape(caption(results)),
        "DATA_NOTE": data_note(results, html=True),
        "COHORT_TITLE": html.escape(fitted),
        "COHORT": "".join(
            f'<div class="metric"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
            for k, v in cohort_metrics(ok).items()),
        "RANK_HEAD": rank_head,
        "RANK_ROWS": rank_rows,
        "RANGE_NOTE": html.escape(RANGE_NOTE),
        "ECON_NOTE": html.escape(economics_note(results)),
        "SUMM_ROWS": summ_rows,
        "FOOTER": html.escape(methodology(results)),
        "CLUSTER_NOTE": html.escape(CLUSTER_NOTE),
        "PAYLOAD": script_json(payload),
    }

    with open(os.path.join(ROOT, "scripts", "template.html"), encoding="utf-8") as f:
        tpl = f.read()
    # One pass, so text inserted for one placeholder is never re-scanned for another.
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: parts[m.group(1)], tpl)

    os.makedirs(PUBLIC, exist_ok=True)
    with open(os.path.join(PUBLIC, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    shutil.copy(RESULTS_CSV, os.path.join(PUBLIC, "midland_basin_dca_results.csv"))
    shutil.copy(TYPE_CURVE_CSV, os.path.join(PUBLIC, "observed_type_curve.csv"))
    print(f"wrote public/index.html ({len(page)/1e6:.2f} MB), {len(wells)} wells")


if __name__ == "__main__":
    main()
