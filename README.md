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
  fit per well with `scipy` on producing time (shut-in and curtailed months do not
  count as decline time), robust log-space least squares from the peak month, qi capped
  at the observed peak rate, b capped at 1.5
- **EUR & remaining reserves** — forecast from the month after the last reported month
  to a 150 bbl/month (~5 BOPD) economic limit within a 50-year well life; terminal
  decline, economic limit and well life are adjustable in the app
- **Backtested** — the fitting method was chosen by hindcasting held-out months
  (`scripts/backtest.py`)
- **QC flags** — low R² (linear or log space), parameters stuck on a fit bound, b held
  at its cap, forecasts that start far from the current producing rate, and wells shut
  in or curtailed at the last report
- **Decline-shape clustering** — k-means on the share of peak rate each well keeps after
  one and three producing years, k chosen by silhouette (the basis for type curves and
  benchmarking)
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

1. **Cleaning** — trailing unreported months are dropped (see above). The fit starts
   at each well's peak-rate month (standard shale practice — flush production distorts
   early decline) and uses positive months only; zero months are kept as shut-ins for
   the decline clock.
2. **Decline clock** — a well does not deplete while it is shut in, so decline time is
   *producing* time, not calendar time. Monthly volumes carry no days-on, so a month
   producing under 50% of the fitted trend is counted as producing only `q / q_trend`
   of the month (a zero month for none of it), and the fit and the clock are updated
   alternately (up to 3 passes). Without this, a well that is shut in and then
   rebounds is fitted as a much flatter decline.
3. **Arps fit** — `q(t) = qi / (1 + b·Di·t)^(1/b)` with `t` = producing months since
   the peak month, fit in log space via `scipy.optimize.curve_fit` with a robust
   soft-L1 loss (`f_scale` 0.3) so a single partial or restart month cannot dominate;
   bounds `Di ∈ [0.005, 0.80]/mo`, `b ∈ [0.05, 1.5]` (1.5 is the plausible upper end for
   Midland Basin horizontals with a terminal decline), `qi ≤ observed peak`. Once the
   instantaneous decline falls to the terminal rate Dmin, the forecast switches to
   exponential decline.
4. **EUR** — cumulative produced + forecast from the month after the last reported
   month (each forecast month a full producing month) to the economic limit, within a
   maximum total well life counted from first production. Base case: Dmin 8%/yr
   nominal, 150 bbl/mo (~5 BOPD), 50-year life — see *Forecast assumptions* below.
5. **Clustering** — k-means on the log of the share of peak rate the fitted decline
   keeps after 12 and 36 producing months. That is the decline shape type curves are
   about, and unlike Di or b on their own it is well determined: a fit can trade Di
   against b without changing the curve much, which made clusters on (Di, b) unstable.
   The model is fitted on wells that are not low-confidence (the rest are assigned to
   the nearest cluster, so a few poor fits cannot move the boundaries), k is chosen
   from 2–4 by silhouette score, and the pipeline reports a bootstrap stability score
   (adjusted Rand index). Clusters are named from their shape ("Steep decline", or
   "Steep year 1, moderate after" when the two horizons disagree).
6. **QC** — every fit carries R² in linear space and in log space on its own decline
   clock (partial months producing under 10% of the fitted rate are left out of the
   log R² and counted separately). A well is **low-confidence** when either R² < 0.6,
   Di or b sits on a fit bound (b at its 1.5 cap does not count — that is the cap
   working), or the first forecast month is more than 1.35× above or below the median
   of its last 6 producing months; low-confidence wells get hollow markers on the
   cluster chart. Partial months, b held at the cap, wells shut in or curtailed at the
   last report ("confirm status": the forecast assumes a return to trend) and
   forecasts that hit the life cap are listed in the `qc_flags` column and the
   per-well QC warning.

### How the fitting method was chosen

Five candidate fixes for wells that dip and rebound mid-life (restart segmentation, a
b cap, recency weighting, rate-time matching, and the producing-time clock with a b
cap) were each scored by hindcasting: refit every well with its last 6, 12 or 18
reported months hidden and compare the forecast with what it actually produced. The
producing-time clock with b ≤ 1.5 was the only candidate that was significantly more
accurate than the earlier method (calendar time, b up to 2.0) at 12–18 months while
staying even at shorter horizons (checked from 3 to 24 months).
`python scripts/backtest.py` reproduces the comparison:

| Holdout | Wells | Mean abs. volume error, earlier → current | Paired change (90% CI) | Median bias, earlier → current |
|---|---|---|---|---|
| 6 months | 30 | 17.4% → 17.6% | +0.2% (-0.5% to +1.0%) | -4.2% → -4.7% |
| 12 months | 30 | 26.6% → 23.5% | -3.1% (-5.6% to -0.9%) | +10.6% → +7.7% |
| 18 months | 25 | 23.4% → 21.4% | -2.0% (-3.6% to -0.7%) | +5.3% → +3.6% |

The gain is real but modest: about 10% less held-out error at 12–18 months, less
over-forecasting, and no change at 6 months. With 30 wells, these are the honest
error bars on a one-year forecast.

### Findings from this cohort

Computed from `results/midland_basin_dca_results.csv` (`python -m src.pipeline` prints
the headline numbers and the cluster table):

- Mean b-factor **0.92** (median 0.93; 10 of 30 wells above 1, 5 held at the 1.5 cap),
  mean nominal initial decline **~364%/yr** (median 375%/yr) — classic steep shale
  decline from flush production.
- Mean EUR **~504,000 bbl** per well (median ~507,000; range ~169k–877k bbl).
- 27 of 30 wells show more than half a month of estimated downtime since peak (median
  0.8 months; the most is WELCH-COX E39V, 6.2 months of shut-ins and curtailments).
- Mean fit R² **0.88** (median 0.935); 29 of 30 wells have R² > 0.6, 26 of 30 in log
  space, which also scores the late-life tail.
- **7 of 30 wells are low-confidence, including the #2 EUR well** (EPLEY-MARTIN 31E,
  784k bbl). It fell to 271 bbl in one month and then rebounded far above its earlier
  trend; even on producing time one curve fits it poorly (log R² 0.44) and its b sits
  on the cap. DAVIS-LACY 31K (#7, 681k) is similar. Treat both with caution.
- DAVIS-LACY 31D (~80 bbl/mo in its last three months) and WELCH-COX E39V (shut in for
  its last two reported months) are forecast as returning to trend; confirm their
  status before relying on their remaining volumes.
- No forecast reaches the 50-year life cap under the base case.
- Two decline-shape clusters. k=2 is chosen by silhouette (0.57 vs 0.49 for k=3) and
  is stable under bootstrap resampling (adjusted Rand index 0.91):

  | Cluster | Wells | Peak rate kept after 1 yr / 3 yr | Mean b | Mean Di (nom.) | Mean EUR (bbl) | Mean EUR / 1,000 ft |
  |---|---|---|---|---|---|---|
  | Steep decline | 15 | 17% / 6% | 0.92 | 487%/yr | 467,000 | 47,000 |
  | Shallow decline | 15 | 32% / 13% | 0.93 | 242%/yr | 541,000 | 54,000 |

  The two clusters have the same mean b; what separates them is how fast the first
  years decline. 2 of the steep wells and 5 of the shallow ones are low-confidence.

### Forecast assumptions

The base case uses an 8%/yr nominal terminal decline (the middle of the 6–10% range in
common use for Permian horizontals), a 150 bbl/mo (~5 BOPD) economic limit and a
50-year total well life counted from first production. The fits do not depend on
these; only the forecast tail does. In the Streamlit app all three are sliders in the
sidebar (the static Vercel page shows the base case). `python scripts/sensitivity.py`
reproduces this table:

| Case | Dmin (nom./yr) | Econ. limit (bbl/mo) | Max life (yr) | Mean EUR (bbl) | Median EUR (bbl) | vs base |
|---|---|---|---|---|---|---|
| Base case | 8.0% | 150 | 50 | 504,301 | 507,387 | +0.0% |
| Optimistic tail (Dmin 5%) | 5.0% | 150 | 50 | 542,881 | 530,236 | +7.7% |
| Conservative tail (Dmin 10%) | 10.0% | 150 | 50 | 484,414 | 503,338 | -3.9% |
| Higher economic limit (250 bbl/mo) | 8.0% | 250 | 50 | 491,606 | 495,402 | -2.5% |
| 30-year well life | 8.0% | 150 | 30 | 496,780 | 506,539 | -1.5% |
| Source dataset's QC assumptions | 9.6% | 250 | 30 | 477,204 | 493,201 | -5.4% |

### Limitations

- 30-well convenience cohort (one operator, one county) — not a statistical sample
  of the Midland Basin.
- Economic limit, terminal decline and well life are assumptions; EURs are technical
  estimates, not SPE PRMS reserves. See *Forecast assumptions* for how much they move
  EUR.
- For the 5 wells held at b = 1.5, b is set by the cap, not the data, so their EURs
  carry that assumption (a lower cap lowers them).
- Downtime is inferred from the fit because the data has no days-on, so a permanent
  loss of productivity below 50% of trend would be read as downtime; the "confirm
  status" flag marks wells where this matters at the end of history.
- One curve per well: a well whose rebound is a genuine re-completion (EPLEY-MARTIN 31E
  may be one) would be better forecast from the restart alone, which needs the
  operator's confirmation.
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

Tests: `pip install -r requirements-dev.txt && pytest`. Backtest: `python scripts/backtest.py`; assumption sensitivity: `python scripts/sensitivity.py`.

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
│   ├── midland_basin_dca_results.csv   # pipeline output: fits, EUR, QC flags, clusters
│   └── observed_type_curve.csv         # P25/P50/P75 by month on production
├── public/                # static Vercel build (generated)
├── scripts/
│   ├── build_static.py    # builds public/ from src/ + data/
│   ├── backtest.py        # hindcast of the fitting method
│   ├── sensitivity.py     # EUR vs forecast assumptions
│   └── template.html
├── src/
│   ├── data_loader.py     # load + clean production series
│   ├── arps.py            # Arps models, curve fitting, EUR
│   ├── clustering.py      # k-means on decline shape, k by silhouette
│   ├── figures.py         # Plotly charts shared by both front ends
│   ├── report.py          # captions / labels shared by both front ends
│   ├── type_curve.py      # observed P25/P50/P75 type curve
│   └── pipeline.py        # end-to-end: python -m src.pipeline
└── tests/                 # pytest suite
```

## Extending it

The analysis code is county-agnostic: drop any RRC-sourced monthly production CSV with
the columns in `data_loader.PROD_COLS` (and a matching wells file) into `data/`, point
the file names in `pipeline.py` (and `app.py` / `scripts/build_static.py`) at it, and
the same workflow runs on Midland, Howard, Glasscock, Reagan, or Upton counties;
county names and well counts in the UI come from the data.
