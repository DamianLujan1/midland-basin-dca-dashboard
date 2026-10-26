# Midland Basin — Automated Decline Curve Analysis

A production forecasting dashboard for horizontal oil wells in the **Midland Basin**
(Permian), built as a freelance petroleum data-science portfolio project. Given a set
of wells, it automatically fits Arps hyperbolic decline curves, forecasts production
to an economic limit, estimates EUR, flags low-confidence fits, and groups wells by
decline behavior with k-means clustering.

![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Python](https://img.shields.io/badge/Python-3.12+-blue)

## What it does

- **Automated Arps fitting** — hyperbolic decline with terminal exponential decline,
  fit per well with `scipy` (robust log-space least squares from the peak month, qi
  capped at the observed peak rate)
- **EUR & remaining reserves** — forecast from the month after the last reported month
  to a 150 bbl/month (~5 BOPD) economic limit within a 50-year well life; terminal
  decline, economic limit and well life are adjustable in the app
- **QC flags** — low R² (linear or log space), parameters stuck on a fit bound,
  forecasts that start far from the current producing rate, wells shut in at the last
  report, and time-capped forecasts
- **Decline-behavior clustering** — k-means on (log Di, b), k chosen by silhouette, to
  group wells that decline alike (the basis for type curves and benchmarking)
- **Observed type curve** — P25/P50/P75 monthly oil by month on production
- **Interactive dashboard** — per-well history vs. fit vs. forecast chart (linear/log),
  cohort EUR ranking, type curve, cluster explorer, one-click CSV download

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

- **Coverage:** 30 horizontal oil wells, **Martin County, TX** (RRC District 08) —
  Midland Basin. Operator: Pioneer Natural Resources USA, Inc. Field: Spraberry
  (Trend Area).
- **Period:** January 2022 – March 2026, the last reported month (30–51 months of
  history per well; 1,130 well-months analyzed).
- **Granularity:** every well sits on a **single-well lease**, so RRC lease-level
  reporting equals well-level production — the cleanest possible case for Texas
  data, which the RRC reports by lease rather than by well.
- **Retrieved:** September 22, 2026.

### Known data issues

- **Unreported trailing months.** The source file runs to June 2026, but every one of
  the 30 wells reports exactly zero oil *and* zero gas for April–June 2026 (90 of its
  1,220 rows), while the cohort made ~85,500 bbl in March 2026. That is RRC reporting
  lag or calendar padding in the extract, not 30 simultaneous shut-ins, so the loader
  drops trailing months in which no well reports any volume. Zero months for a single
  well inside the reported period are kept as that well's shut-ins.
- **Lease reported before first production — no production missing.** For 18 of the 30
  wells the lease's first RRC-reported month (`lease_first_reported`) is 4–9 months
  before its first month with positive production. Those months were reported with
  zero oil *and* zero casinghead gas (the source dataset dropped them as
  "pre-positive" rows), and the timing shows they are leases set up before pad-level
  turn-in-line, not lost production: every pad's wells start producing in the same
  month regardless of lease date (all 12 DAVIS-LACY wells in July 2023 against five
  different lease dates; all 8 EPLEY-MARTIN wells in September 2022, five of which
  have no gap at all), and the "gap" wells ramp up from month 1 exactly like the
  others (EPLEY-MARTIN 31E: 1,375 → 11,617 → 18,712 bbl), which a well already months
  into production would not do. DAVIS-LACY 31A, whose first month is its peak, is the
  one ambiguous case. The dashboard's "First production" is the first positive month.
- **Upstream type curve and QC files.** `data/martin_observed_type_curve.csv` and
  `data/martin_real_cohort_analysis_qc.json` are kept exactly as the source dataset
  published them, but they were built with the padded zero months included, so their
  P50 collapses at months-on-production 34–36 (P50 684 bbl at month 34 vs 3,607 in the
  reported data) and their `all_wells_present_through_mop` should be 30, not 33. The
  pipeline rebuilds the observed type curve with the same method from the cleaned data
  (`results/observed_type_curve.csv`, months with at least 10 reporting wells; shown on
  the Cohort tab); use that one.

## Methodology

1. **Cleaning** — trailing unreported months are dropped (see above); zero-rate months
   are treated as shut-in and excluded from fitting; the fit starts at each well's
   peak-rate month (standard shale practice — flush production distorts early decline).
2. **Arps fit** — `q(t) = qi / (1 + b·Di·t)^(1/b)` with `t` = calendar months since the
   peak month (peak = 0, shut-in gaps kept), fit in log space via
   `scipy.optimize.curve_fit` with a robust soft-L1 loss (`f_scale` 0.3) so a single
   partial or restart month cannot dominate the fit; bounds `Di ∈ [0.005, 0.80]/mo`,
   `b ∈ [0.05, 2.0]`, `qi ≤ observed peak`. Once instantaneous decline falls to the
   terminal rate Dmin, the forecast switches to exponential decline.
3. **EUR** — cumulative produced + forecast from the month after the last reported
   month to the economic limit, within a maximum total well life counted from first
   production. Base case: Dmin 8%/yr nominal, 150 bbl/mo (~5 BOPD), 50-year life — see
   *Forecast assumptions* below.
4. **Clustering** — k-means on standardized (log Di, b) — Di spans more than an order of
   magnitude, so its log is the natural scale. The model is fitted on wells that are not
   low-confidence and the rest are assigned to the nearest cluster, so a few poor fits
   cannot move the boundaries; k is chosen from 2–4 by silhouette score, and the
   pipeline reports a bootstrap stability score (adjusted Rand index) for the result.
   Clusters are named by where they sit on each axis (e.g. "High Di · high b"); clusters
   within 20% of each other on an axis share that axis's level, and wells with a
   parameter stuck on a fit bound are left out of the naming centroids. Di alone is
   misleading, because a high-Di well with a high b flattens quickly and can
   out-produce a low-Di, near-exponential well.
5. **QC** — every fit carries R² in linear space and in log space (partial months
   producing under 10% of the fitted rate are left out of the log R² and counted
   separately). A well is **low-confidence** when either R² < 0.6, Di or b sits on a
   fit bound, or the first forecast month is more than 1.35× above or below the median
   of its last 6 producing months; low-confidence wells get hollow markers on the
   cluster chart. Partial months, a shut-in at the last report and forecasts that hit
   the 50-year cap are listed in the `qc_flags` column and the per-well QC warning, but
   do not make a well low-confidence on their own.

### Findings from this cohort

Computed from `results/midland_basin_dca_results.csv` (`python -m src.pipeline` prints
the headline numbers and the cluster table):

- Mean b-factor **0.97** (median 0.94; 12 of 30 wells above 1), mean nominal initial
  decline **~385%/yr** (median 387%/yr) — classic steep shale decline from flush
  production.
- Mean EUR **~517,000 bbl** per well (median ~522,000; range ~169k–891k bbl).
- Mean fit R² **0.88** (median 0.93); all 30 wells have R² > 0.6, but only 26 of 30 in
  log space, which also scores the late-life tail.
- **8 of 30 wells are low-confidence, including the #2 and #5 EUR wells**
  (EPLEY-MARTIN 31E, 864k bbl; DAVIS-LACY 31K, 776k bbl). Both rebound after a
  mid-life curtailment (31E fell to 271 bbl in one month; 31K to 3,208 and 1,583 bbl),
  and a single hyperbolic through the rebound gives a very flat tail (b 1.84 and 2.0).
  Treat the top of the EUR ranking with caution.
- No forecast reaches the 50-year life cap under the base case.
- Two clusters along the Di–b trend. k=2 is chosen by silhouette (0.52 vs 0.44 for
  k=3) and is stable under bootstrap resampling (adjusted Rand index 0.84; the earlier
  k=3 split scored ~0.6, i.e. it changed with the sample):

  | Cluster | Wells | Mean b | Mean Di (nom.) | Mean EUR (bbl) | Mean EUR / 1,000 ft |
  |---|---|---|---|---|---|
  | High Di · high b | 20 | 1.22 | 498%/yr | 571,000 | 57,000 |
  | Low Di · low b | 10 | 0.47 | 158%/yr | 408,000 | 41,000 |

  The high-Di/high-b cluster has the higher EUR per 1,000 ft of lateral; 5 of its 20
  wells are low-confidence.

### Forecast assumptions

The base case uses an 8%/yr nominal terminal decline (the middle of the 6–10% range in
common use for Permian horizontals), a 150 bbl/mo (~5 BOPD) economic limit and a
50-year total well life counted from first production. The fits do not depend on
these; only the forecast tail does. In the Streamlit app all three are sliders in the
sidebar (the static Vercel page shows the base case). `python scripts/sensitivity.py`
reproduces this table:

| Case | Dmin (nom./yr) | Econ. limit (bbl/mo) | Max life (yr) | Mean EUR (bbl) | Median EUR (bbl) | vs base |
|---|---|---|---|---|---|---|
| Base case | 8.0% | 150 | 50 | 516,633 | 522,138 | +0.0% |
| Optimistic tail (Dmin 5%) | 5.0% | 150 | 50 | 562,213 | 525,907 | +8.8% |
| Conservative tail (Dmin 10%) | 10.0% | 150 | 50 | 493,481 | 503,112 | -4.5% |
| Higher economic limit (250 bbl/mo) | 8.0% | 250 | 50 | 503,913 | 507,174 | -2.5% |
| 30-year well life | 8.0% | 150 | 30 | 507,428 | 517,380 | -1.8% |
| Source dataset's QC assumptions | 9.6% | 250 | 30 | 486,453 | 492,227 | -5.8% |

### Limitations

- 30-well convenience cohort (one operator, one county) — not a statistical sample
  of the Midland Basin.
- Economic limit, terminal decline and well life are assumptions; EURs are technical
  estimates, not SPE PRMS reserves. See *Forecast assumptions* for how much they move
  EUR.
- One hyperbolic segment per well: wells whose rate dips and then rebounds are not
  re-fitted from the restart, which inflates b. That explains the high-b
  low-confidence wells (e.g. EPLEY-MARTIN 31E, DAVIS-LACY 31K); the other flags are b
  pinned at its lower bound, a weak log-space fit, or a forecast that starts well away
  from the current producing rate.
- Gas is reported (casinghead) but the forecast/EUR is oil-only in this version.

## Run it locally

```bash
git clone <this-repo>
cd midland-basin-dca-dashboard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fit all wells (writes results/midland_basin_dca_results.csv)
python -m src.pipeline

# Launch the dashboard
streamlit run app.py
```

Then open http://localhost:8501.

Tests: `pip install -r requirements-dev.txt && pytest`.

## Deploy free on Streamlit Cloud

1. Push this repo to GitHub (data files included — they're small).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo, branch, and `app.py`.
3. Deploy. On first launch the app auto-runs the pipeline if
   `results/midland_basin_dca_results.csv` is missing, so no extra steps are needed.

## Deploy on Vercel (static build)

Streamlit needs a long-running server, so the Vercel deployment is a static build of
the same dashboard: `scripts/build_static.py` runs the pipeline (`src/`) on the data in
`data/` and writes `public/index.html` (Plotly charts, well selector, cohort table,
clusters, CSV download). `vercel.json` serves `public/` as-is. The page loads Plotly.js
from cdn.plot.ly at the version matching the installed `plotly` package; if the CDN is
blocked the tables and CSV download still work.

```bash
python scripts/build_static.py   # regenerate public/ after changing data or code
vercel deploy --prod
```

## Project structure

```
midland-basin-dca-dashboard/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # pinned runtime dependencies
├── requirements-dev.txt   # + pytest
├── vercel.json            # static deploy of public/
├── data/
│   ├── martin_selected_30_monthly_production_normalized.csv  # RRC monthly production (via GitHub dataset, see above)
│   ├── martin_selected_30_wells.csv                          # well metadata / completions
│   ├── martin_observed_type_curve.csv                        # upstream artifact, unused (see Known data issues)
│   └── martin_real_cohort_analysis_qc.json                   # upstream artifact, unused (see Known data issues)
├── results/
│   └── midland_basin_dca_results.csv   # pipeline output: fits, EUR, QC flags, clusters
├── public/                # static Vercel build (generated)
├── scripts/
│   ├── build_static.py    # builds public/ from src/ + data/
│   └── template.html
├── src/
│   ├── data_loader.py     # load + clean production series
│   ├── arps.py            # Arps models, curve fitting, EUR
│   ├── clustering.py      # k-means on (log Di, b), k by silhouette
│   ├── figures.py         # Plotly charts shared by both front ends
│   ├── report.py          # captions / labels shared by both front ends
│   └── pipeline.py        # end-to-end: python -m src.pipeline
└── tests/                 # pytest suite
```

## Extending it

The analysis code is county-agnostic: drop any RRC-sourced monthly production CSV with
the columns in `data_loader.PROD_COLS` (and a matching wells file) into `data/`, point
the file names in `pipeline.py` (and `app.py` / `scripts/build_static.py`) at it, and
the same workflow runs on Midland, Howard, Glasscock, Reagan, or Upton counties;
county names and well counts in the UI come from the data.
