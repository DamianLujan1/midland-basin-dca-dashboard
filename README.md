# Midland Basin — Automated Decline Curve Analysis

A production forecasting dashboard for horizontal oil wells in the **Midland Basin**
(Permian), built as a freelance petroleum data-science portfolio project. Given a set
of wells, it automatically fits Arps hyperbolic decline curves, forecasts production
to an economic limit, estimates EUR, and groups wells by decline behavior with
k-means clustering.

![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## What it does

- **Automated Arps fitting** — hyperbolic decline with 5%/yr terminal exponential
  decline, fit per well with `scipy` (log-space least squares, qi capped at the
  observed peak rate)
- **EUR & remaining reserves** — forecast to a 150 bbl/month (~5 BOPD) economic limit
- **Decline-behavior clustering** — k-means on (Di, b) to group wells that decline
  alike (the basis for type curves and benchmarking)
- **Interactive dashboard** — per-well history vs. fit vs. forecast chart, cohort EUR
  ranking, cluster explorer, one-click CSV download

## Data — read this first

> **Provenance flag:** the production numbers in `data/` are **real Texas Railroad
> Commission (RRC) monthly production records**, but they were **not downloaded
> directly from rrc.texas.gov** for this build. RRC's interactive query endpoints
> (Production Data Query web tool, EWA wellbore/production services, MFT bulk
> downloads) were unreachable or non-functional from the build environment, and a
> third-party RRC scraper could not be used. Instead, the data comes from the open
> dataset **[Dansaad2020/ai-ml-dca-permian](https://github.com/Dansaad2020/ai-ml-dca-permian)**
> (`data/processed/martin_selected_30_monthly_production_normalized.csv` and
> `martin_selected_30_wells.csv`), whose author sourced it from public RRC
> production, permit, wellbore, and completion records. **No production numbers were
> fabricated** — every rate in the dashboard traces to an RRC filing.

- **Coverage:** 30 horizontal oil wells, **Martin County, TX** —
  Midland Basin. Operator: Pioneer Natural Resources USA, Inc. Field: Spraberry
  (Trend Area).
- **Period:** January 2022 – June 2026 (33–54 months of history per well, 1,220
  monthly records total).
- **Granularity:** every well sits on a **single-well lease**, so RRC lease-level
  reporting equals well-level production — the cleanest possible case for Texas
  data, which the RRC reports by lease rather than by well.
- **Retrieved:** September 22, 2026.

## Methodology

1. **Cleaning** — zero-rate months are treated as shut-in and excluded from fitting;
   the fit starts at each well's peak-rate month (standard shale practice — flush
   production distorts early decline).
2. **Arps fit** — `q(t) = qi / (1 + b·Di·t)^(1/b)` fit in log space via
   `scipy.optimize.curve_fit`; bounds `Di ∈ [0.005, 0.80]/mo`, `b ∈ [0.05, 2.0]`,
   `qi ≤ observed peak`. Once instantaneous decline falls to 5%/yr nominal, the
   forecast switches to terminal exponential decline.
3. **EUR** — cumulative produced + forecast to 150 bbl/mo economic limit.
4. **Clustering** — k-means (k=3) on standardized (Di, b); clusters named by
   decline steepness.
5. **QC** — every fit carries an R²; wells with R² < 0.6 are low-confidence
   (typically operational noise: restarts, choke changes).

### Findings from this cohort

- Mean b-factor **0.84**, mean nominal initial decline **~340%/yr** — classic
  steep shale decline from flush production.
- Mean EUR **~506,000 bbl** per well (range ~165k–960k bbl).
- Mean fit R² **0.83** (median 0.87); 28 of 30 wells fit with R² > 0.6.
- Three decline families emerge: steep-decline wells (high Di, fast early drop),
  moderate, and mild — the mild cluster holds the highest EUR per 1,000 ft of
  lateral on average.

### Limitations

- 30-well convenience cohort (one operator, one county) — not a statistical sample
  of the Midland Basin.
- Economic limit and terminal decline are assumptions; EURs are technical
  estimates, not SPE PRMS reserves.
- Gas is reported (casinghead) but the forecast/EUR is oil-only in this version.

## Run it locally

```bash
git clone <this-repo>
cd permian-dca-dashboard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fit all wells (writes results/midland_basin_dca_results.csv)
python -m src.pipeline

# Launch the dashboard
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy free on Streamlit Cloud

1. Push this repo to GitHub (data files included — they're small).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo, branch, and `app.py`.
3. Deploy. On first launch the app auto-runs the pipeline if
   `results/midland_basin_dca_results.csv` is missing, so no extra steps are needed.

## Project structure

```
permian-dca-dashboard/
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── martin_selected_30_monthly_production_normalized.csv  # RRC monthly production (via GitHub dataset, see above)
│   ├── martin_selected_30_wells.csv                          # well metadata / completions
│   ├── martin_observed_type_curve.csv
│   └── martin_real_cohort_analysis_qc.json
├── results/
│   └── midland_basin_dca_results.csv   # pipeline output: fits, EUR, clusters
└── src/
    ├── data_loader.py     # load + clean production series
    ├── arps.py            # Arps models, curve fitting, EUR
    ├── clustering.py     # k-means on (Di, b)
    └── pipeline.py        # end-to-end: python -m src.pipeline
```

## Extending it

The pipeline is county-agnostic: drop any RRC-sourced monthly production CSV with
the columns in `data_loader.PROD_COLS` into `data/` and point `pipeline.py` at it
to run the same workflow on Midland, Howard, Glasscock, Reagan, or Upton counties.
