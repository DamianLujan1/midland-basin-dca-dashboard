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
  count as decline time and are left out of the fit), robust log-space least squares
  from the peak month, qi capped at the observed peak rate, b capped at 1.5
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
- **Lease reported before first production — no evidence of missing production.** For
  18 of the 30 wells the lease's first RRC-reported month (`lease_first_reported`) is
  4–9 months before its first month with positive production. The source dataset
  rebases month-on-production to the first month with positive oil *or* casinghead gas
  and drops the earlier "pre-positive" rows (its `reports/source_reproducibility_notes.md`),
  so those leases reported no oil or gas in those months. The timing points to leases
  set up before pad-level turn-in-line rather than production reported elsewhere: every
  pad's wells start producing in the same month regardless of lease date (all 12
  DAVIS-LACY wells in July 2023 against five different lease dates; all 8 EPLEY-MARTIN
  wells in September 2022, five of which have no gap at all), and most "gap" wells ramp
  up from month 1 like the others (EPLEY-MARTIN 31E: 1,375 → 11,617 → 18,712 bbl), which
  a well already months into production would not do. Two gap wells peak in their
  first month and are ambiguous: DAVIS-LACY 31A, and FORTUNE-ASHLEY 31M, which has no
  pad-mates to compare with (two wells without a gap also peak in month 1, so this is
  weak evidence either way). The dashboard's "First production" is the first positive
  month.
- **Upstream type curve and QC files.** `data/martin_observed_type_curve.csv` and
  `data/martin_real_cohort_analysis_qc.json` are kept exactly as the source dataset
  published them, but they were built with the padded zero months included, so their
  P50 collapses at months-on-production 34–36 (684 bbl at month 34 vs 3,607 in the
  reported data) and drops to zero at 44–47, and their `all_wells_present_through_mop`
  should be 30, not 33. The pipeline rebuilds the observed type curve with the same
  method from the cleaned data (`results/observed_type_curve.csv`, shown on the Cohort
  tab) and stops it at month 33, the last month in which at least 80% of the wells
  report: later months come from a 14-well subset, 8 of them one pad (EPLEY-MARTIN),
  four of which were curtailed in the same month.

## Methodology

1. **Cleaning** — trailing unreported months are dropped (see above). The fit starts
   at each well's peak-rate month (standard shale practice — flush production distorts
   early decline); zero months are kept as shut-ins for the decline clock.
2. **Decline clock** — a well does not deplete while it is shut in, so decline time is
   *producing* time, not calendar time. Monthly volumes carry no days-on, so a month
   producing under 50% of the fitted trend is counted as producing only `q / q_trend`
   of the month (a zero month for none of it). Such a month's volume says nothing
   about the trend beyond that, so like a zero month it is left out of the fit. The
   fit and the clock are updated alternately until the clock moves by less than
   0.001 month (1–5 fits per well here); if a month sitting at the 50% line keeps
   flipping, it is counted as producing. On calendar time, a well that is shut in and
   then rebounds reads as a flatter decline.
3. **Arps fit** — `q(t) = qi / (1 + b·Di·t)^(1/b)` with `t` = producing months since
   the peak month, fit in log space via `scipy.optimize.curve_fit` with a robust
   soft-L1 loss (`f_scale` 0.3); bounds `Di ∈ [0.005, 0.80]/mo`, `b ∈ [0.05, 1.5]`,
   `qi ≤ observed peak`. The 1.5 cap is an analog-based upper end for Midland Basin
   horizontals with a terminal decline, not a value the backtest selected (it favours
   lower caps slightly at 12–18 months); see the b-cap rows under *Forecast
   assumptions*. Once the instantaneous decline falls to the terminal rate Dmin, the
   forecast switches to exponential decline.
4. **EUR** — cumulative produced + forecast from the month after the last reported
   month (each forecast month a full producing month) to the economic limit, within a
   maximum total well life counted from first production. Base case: Dmin 8%/yr
   nominal, 150 bbl/mo (~5 BOPD), 50-year life — see *Forecast assumptions* below.
5. **Clustering** — k-means on the log of the share of peak rate the fitted decline
   keeps after 12 and 36 producing months (for wells with less than 36 producing
   months the second value is the fit's extrapolation). That is the decline shape type
   curves are about, and unlike Di or b on their own it is well determined: a fit can
   trade Di against b without changing the curve much, which made clusters on (Di, b)
   unstable. The model is fitted on wells that are not low-confidence (the rest are
   assigned to the nearest cluster, so a few poor fits cannot move the boundaries), k
   is chosen from 2–4 by silhouette score, and the pipeline reports a bootstrap
   stability score (adjusted Rand index). Clusters are named from their shape ("Steep
   decline", or "Steep year 1, moderate after" when the two horizons disagree).
6. **QC** — every fit carries R² in linear and log space over the months it was
   fitted on. A well is **low-confidence** when either R² < 0.6, Di or b sits on a fit
   bound (b at its 1.5 cap does not count — that is the cap working), the first
   forecast month is more than 1.35× above or below the median of its last 6
   producing months, or its last 2+ reported months are shut in or curtailed (the
   forecast assumes a return to trend: "confirm status"). Low-confidence wells get
   hollow markers on the cluster chart. b held at the cap, 3+ months read as curtailed
   and left out of the fit, and forecasts that hit the life cap are also listed in the
   `qc_flags` column and the per-well QC warning; the well detail shows each well's
   downtime and curtailed-month count.

### How the fitting method was chosen

Five candidate fixes for wells that dip and rebound mid-life (restart segmentation, a
b cap, recency weighting, rate-time matching, and the producing-time clock with a b
cap) were scored by hindcasting: refit every well with its last 6, 12 or 18 reported
months hidden and compare the forecast with what it actually produced. The clock with
a b cap was kept as the most accurate candidate at 12–18 months. An independent review
then found the first implementation counted curtailed months twice (as downtime and
as on-trend points); fixing that improved the 12-month result further.
`python scripts/backtest.py` reproduces this and splits the gain between the two
parts:

| Holdout | Wells | earlier (calendar, b ≤ 2) | b cap only | clock only | current |
|---|---|---|---|---|---|
| 6 months | 30 | 17.4% (-4.2%) | 17.7% (-5.1%) | 17.3% (-1.1%) | 17.9% (-3.1%) |
| 12 months | 30 | 26.6% (+10.6%) | 25.2% (+9.9%) | 24.0% (+10.6%) | 22.4% (+9.3%) |
| 18 months | 25 | 23.4% (+5.3%) | 22.5% (+3.6%) | 22.7% (+5.3%) | 21.4% (+3.6%) |

*Mean absolute held-out volume error (median bias).* Paired against the earlier
method, the current one is 4.2 points better at 12 months (90% CI 1.3–7.7) and 2.0
better at 18 (0.4–3.7), and not significantly different at 6 (+0.5, CI −2.0 to +3.3).
Each part alone gives some of the gain; together they beat the clock alone at 12 and
18 months and the cap alone at 12. Caveats: the candidates and their settings were
chosen on these same 30 wells, so the intervals are optimistic, and the 12- and
18-month windows overlap. The gain is real but modest — about a sixth less error on a
one-year forecast — and these are the honest error bars on such a forecast.

### Findings from this cohort

Computed from `results/midland_basin_dca_results.csv` (`python -m src.pipeline` prints
the headline numbers and the cluster table):

- Mean b-factor **0.92** (median 0.94; 11 of 30 wells above 1, 3 held at the 1.5 cap),
  mean nominal initial decline **~357%/yr** (median 375%/yr) — classic steep shale
  decline from flush production.
- Mean EUR **~511,000 bbl** per well (median ~500,000; range ~169k–895k bbl).
- 27 of 30 wells have at least one month after peak read as curtailed (median downtime
  0.9 months; the most is WELCH-COX E39V, 6.3 months). Nine wells have 3+ curtailed
  months left out of their fit, among them EPLEY-MARTIN 31E: its 2023 dip
  (down to 271 bbl) is read as downtime and the rebound to ~12,000 bbl/mo as the well
  resuming its trend, which gives b 0.78 and 596k bbl instead of the flat b ≈ 1.8 tail
  and 864k bbl a single calendar-time curve gives under the same assumptions. If that
  rebound was a re-completion, a restart-segment forecast would be better.
- Mean fit R² **0.93** (median 0.95) over the months each well was fitted on; all 30
  wells have R² > 0.6 in linear and log space.
- **4 of 30 wells are low-confidence:** WELCH-COX E39V (#7 by EUR) and DAVIS-LACY 31D
  (#28) end the record shut in or curtailed (E39V for its last two months, 31D at
  ~80 bbl/mo for three), and both are forecast as returning to trend — confirm their
  status before relying on their remaining volumes; BOHANAN-BLOCKER 3B's forecast
  starts at 0.70× its recent rate (its last six reported months ran above its fitted
  trend);
  DAVIS-LACY 31H has b on the 0.05 floor.
- 3 wells are held at the 1.5 b cap (SCHARBAUER-DOVE 7F #4, DAVIS-LACY 31K #6,
  GLYNN-ATCHISON 44M), holding 12% of cohort EUR; their EURs carry that cap.
- No forecast reaches the 50-year life cap under the base case.
- Two decline-shape clusters. k=2 is chosen by silhouette (0.57 vs 0.48 for k=3) and
  is stable under bootstrap resampling (adjusted Rand index 0.91):

  | Cluster | Wells | Peak rate kept after 1 yr / 3 yr | Mean b | Mean Di (nom.) | Mean EUR (bbl) | Mean EUR / 1,000 ft |
  |---|---|---|---|---|---|---|
  | Steep decline | 16 | 18% / 6% | 0.94 | 473%/yr | 481,000 | 48,000 |
  | Shallow decline | 14 | 33% / 13% | 0.90 | 224%/yr | 544,000 | 54,000 |

  The two clusters have about the same mean b; what separates them is how fast the
  first years decline. Each holds 2 low-confidence wells. The shallow cluster's 13%
  higher mean EUR is not statistically significant with 14 and 16 wells (Mann-Whitney
  p ≈ 0.3).

### Forecast assumptions

The base case uses an 8%/yr nominal terminal decline (between the 5% of the earlier
version and the source dataset's own 9.6%), a 150 bbl/mo (~5 BOPD) economic limit and
a 50-year total well life counted from first production. The fits do not depend on
these; only the forecast tail does. In the Streamlit app all three are sliders in the
sidebar (the static Vercel page shows the base case). The b cap does change the fits
of the wells it binds, so it is shown here but not offered as a slider.
`python scripts/sensitivity.py` reproduces this table:

| Case | Dmin (nom./yr) | Econ. limit (bbl/mo) | Max life (yr) | b cap | Mean EUR (bbl) | Median EUR (bbl) | vs base |
|---|---|---|---|---|---|---|---|
| Base case | 8.0% | 150 | 50 | 1.5 | 510,803 | 499,563 | +0.0% |
| Flatter tail (Dmin 5%) | 5.0% | 150 | 50 | 1.5 | 548,634 | 507,404 | +7.4% |
| Steeper tail (Dmin 10%) | 10.0% | 150 | 50 | 1.5 | 491,171 | 492,035 | -3.8% |
| Higher economic limit (250 bbl/mo) | 8.0% | 250 | 50 | 1.5 | 497,628 | 484,607 | -2.6% |
| 30-year well life | 8.0% | 150 | 30 | 1.5 | 503,088 | 498,678 | -1.5% |
| Source dataset's QC assumptions | 9.6% | 250 | 30 | 1.5 | 483,505 | 480,956 | -5.3% |
| Lower b cap (1.2) | 8.0% | 150 | 50 | 1.2 | 498,464 | 498,678 | -2.4% |
| b cap 2.0 (effectively none) | 8.0% | 150 | 50 | 2 | 516,382 | 499,563 | +1.1% |

### Limitations

- 30-well convenience cohort (one operator, one county) — not a statistical sample
  of the Midland Basin.
- Economic limit, terminal decline, well life and the b cap are assumptions; EURs are
  technical estimates, not SPE PRMS reserves. See *Forecast assumptions* for how much
  they move EUR.
- Downtime is inferred from the fit because the data has no days-on: a month under
  50% of trend is read as part-month downtime. A genuine, lasting loss of
  productivity would be read the same way, and because those months leave the fit, the
  fitted trend can sit above them; the "confirm status" flag and the curtailed-month
  count mark the wells where this matters.
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
