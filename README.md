# Midland Basin — Automated Decline Curve Analysis

A production forecasting and PDP valuation dashboard for horizontal oil wells in the
**Midland Basin** (Permian), built as a freelance petroleum data-science portfolio
project. Given a set of wells, it fits Arps hyperbolic decline curves, forecasts each
well to the rate where its cash flow turns negative, estimates EUR with calibrated
P90/P50/P10 ranges, values the remaining production (PV10), flags low-confidence fits,
and groups wells by decline shape with k-means clustering.

![CI](https://github.com/DamianLujan1/midland-basin-dca-dashboard/actions/workflows/ci.yml/badge.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue)

## What it does

- **Automated Arps fitting** — hyperbolic decline with terminal exponential decline,
  fit per well with `scipy` on producing time (shut-in months and months far below
  trend count only part of a month and are down-weighted in the fit), robust log-space
  least squares from the peak month, qi capped at the observed peak rate, b capped at 1.2
- **EUR & remaining reserves** — forecast from the month after the last reported month
  to each well's economic limit within a 50-year well life
- **P90 / P50 / P10 ranges** — a 200-member ensemble per well (block bootstrap of the
  fit plus a level factor), calibrated on a leave-one-well-out hindcast so the P90–P10
  range holds 80% of held-out volumes; shown as a band on every forecast chart
- **PDP economics** — monthly cash flow after royalty, Texas severance and ad valorem
  taxes and lease operating expense (gas valued at each well's recent GOR); PV10, the
  economic limit (where cash flow turns negative) and the shut-in oil price per well
- **Backtested** — the fitting method and the b cap were chosen by hindcasting held-out
  months (`scripts/backtest.py`), the ranges calibrated the same way
  (`scripts/calibrate_ranges.py`)
- **QC flags** — low R² (linear or log space), parameters stuck on a fit bound, b held
  at its cap, forecasts that start far from the current producing rate, and wells shut
  in or curtailed at the last report
- **Decline-shape clustering** — k-means on the share of peak rate each well keeps after
  one and three producing years, k chosen by silhouette (the basis for type curves and
  benchmarking)
- **Observed type curve** — P25/P50/P75 monthly oil by month on production
- **Interactive dashboard** — prices, costs, terminal decline and well life as sliders;
  per-well history vs. fit vs. forecast chart with the P90–P10 band (linear/log), cohort
  ranking, type curve, cluster explorer, one-click CSV download
- **CI** — tests on Python 3.12 and 3.13, and a check that the committed results and
  static page match a fresh build

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
  DAVIS-LACY wells and FORTUNE-ASHLEY 31M in July 2023 against six different lease
  dates; all 8 EPLEY-MARTIN wells in September 2022, five of which have no gap at all),
  and most "gap" wells ramp up from month 1 like the others (EPLEY-MARTIN 31E: 1,375 →
  11,617 → 18,712 bbl), which a well already months into production would not do. Two
  gap wells peak in their first month and are ambiguous: DAVIS-LACY 31A, and
  FORTUNE-ASHLEY 31M — well 213H, which continues the DAVIS-LACY pad's 201H–212H
  numbering and starts with it, so it is very likely that pad's 13th well (two wells
  without a gap also peak in month 1, so first-month peaks are weak evidence either
  way). The dashboard's "First production" is the first positive month.
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
   *producing* time, not calendar time. Monthly volumes carry no days-on, so the fit
   runs in two passes. The first fits every producing month, with zero months counted
   as downtime. Each month's ratio `r = q / q_trend` to that first trend then sets how
   much of the month it produced and how much it counts in the second fit: under 30% of
   trend, it produced at trend for `r` of the month and is left out of the fit; over
   70%, it is a full month at full weight (the shortfall is noise); in between, its
   share of a month rises linearly from 0.3 to 1 and its weight from 0 to 1. Because
   nothing switches at a threshold, a small change in one month makes a small change in
   the fit (no single ±10% month change moves any well's EUR by more than 5.3%). On
   calendar time, a well that is shut in and then rebounds reads as a flatter decline.
3. **Arps fit** — `q(t) = qi / (1 + b·Di·t)^(1/b)` with `t` = producing months since
   the peak month, fit in log space via `scipy.optimize.curve_fit` with a robust
   soft-L1 loss (`f_scale` 0.3) and the weights above; bounds `Di ∈ [0.005, 0.80]/mo`,
   `b ∈ [0.05, 1.2]`, `qi ≤ observed peak`. The b cap is a judgement call informed by
   the hindcast, not an analog value from the literature (see *How the fitting method
   was chosen*). Once the instantaneous decline falls to the terminal rate Dmin, the
   forecast switches to exponential decline.
4. **Economic limit and EUR** — each well's forecast runs from the month after the last
   reported month (each forecast month a full producing month) until its monthly cash
   flow would turn negative (step 5), within a maximum total well life counted from
   first production. EUR = cumulative produced + that forecast. Base case: Dmin 8%/yr
   nominal, 50-year life, $65/bbl oil.
5. **PDP economics** (`src/economics.py`) — per forecast month, with oil rate `q` and gas
   `GOR·q` (GOR held at the well's last 12 producing months):
   `net revenue = NRI · (1 − ad valorem) · [q·P_oil·(1 − 4.6%) + GOR·q·P_gas·(1 − 7.5%)]`
   (Texas oil and gas production tax rates; royalty owners bear their share of taxes),
   `cash flow = net revenue − fixed LOE − variable LOE · q` at 100% working interest. The
   economic limit is the rate where cash flow is zero, `fixed / (net revenue per bbl −
   variable)`. PV10 discounts the monthly cash flow at 10%/yr, mid-month, from April
   2026. The **shut-in oil price** is the oil price at which this month's rate just
   covers its operating cost — the price below which the well should be shut in now.
   No capital, workovers, abandonment cost or hedges.
6. **P90 / P50 / P10** (`src/uncertainty.py`) — P50 is the fitted forecast. Each well
   also gets a 200-member ensemble of Arps curves: the fit's log residuals are
   resampled in blocks of three consecutive months (keeping their month-to-month
   correlation), added back to the fitted curve and refitted with the same clock,
   weights, bounds and loss; each refit's rate is then scaled by a level factor
   `exp(0.21·z)`, `z` standard normal. P90 and P10 are the ensemble's 10th and 90th
   percentiles of remaining volume, EUR and PV10 (each separately, never crossing P50),
   with each member stopping at the economic limit; the chart band is the pointwise
   10th–90th percentile of the monthly rate. The level factor's 0.21 is calibrated (see
   *How the ranges were calibrated*); the bootstrap alone covers fewer than half of the
   held-out volumes, because a well's own residuals miss what moves real production
   off trend (curtailments, offset frac hits, operating changes).
7. **Clustering** — k-means on the log of the share of peak rate the fitted decline
   keeps after 12 and 36 producing months (for wells with less than 36 producing
   months the second value is the fit's extrapolation). That is the decline shape type
   curves are about, and unlike Di or b on their own it is well determined: a fit can
   trade Di against b without changing the curve much, which made clusters on (Di, b)
   unstable. The model is fitted on wells that are not low-confidence (the rest are
   assigned to the nearest cluster, so a few poor fits cannot move the boundaries), k
   is chosen from 2–4 by silhouette score, and the pipeline reports a bootstrap
   stability score (adjusted Rand index). Clusters are named from their shape ("Steep
   decline", or "Steep year 1, moderate after" when the two horizons disagree).
8. **QC** — every fit carries weighted R² in linear and log space over the months it
   was fitted on. A well is **low-confidence** when either R² < 0.6, Di or b sits on a
   fit bound (b at its 1.2 cap does not count — that is the cap working), the first
   forecast month is more than 1.35× above or below the median of its last 6 producing
   months, or its last 2+ reported months are shut in or under 50% of the first-pass
   trend (the forecast assumes a return to trend: "confirm status"). Low-confidence
   wells get hollow markers on the cluster chart. b held at the cap, 3+ months under 50%
   of trend, forecasts that hit the life cap and wells already below their economic
   limit are also listed in the `qc_flags` column and the per-well QC warning.

### How the fitting method was chosen

Five candidate fixes for wells that dip and rebound mid-life (restart segmentation, a
b cap, recency weighting, rate-time matching, and the producing-time clock with a b
cap) were scored by hindcasting: refit every well with its last 6, 12, 18 or 24
reported months hidden and compare the forecast with what it actually produced. The
clock with a b cap was kept. Independent reviews then changed two details. Leaving
months read as curtailed out of the fit (instead of also fitting them as on-trend
points) made no measurable difference (paired −1.1 points at 12 months, 90% CI −3.0 to
+0.5; an earlier version of this README called it an improvement). Doing so at a hard
50% threshold, iterated to a self-consistent clock, made EUR jump — DAVIS-LACY 31B moved
16% when one month moved by 23 bbl — and gave three wells two self-consistent answers;
the two-pass fit with a continuous ramp (step 2) replaced it, at a cost of about 0.6
points of 12-month accuracy (measured at the old 1.5 cap: 23.0% vs 22.4%, CI 0.0 to
+1.2). `python scripts/backtest.py` reproduces the current comparison:

| Holdout | Wells | earlier (calendar, b ≤ 2) | calendar, b ≤ 1.2 | clock, b ≤ 2 | clock, b ≤ 1.5 | clock, b ≤ 1.0 | **current (clock, b ≤ 1.2)** |
|---|---|---|---|---|---|---|---|
| 6 months | 30 | 17.4% (-4.2%) | 18.5% (-6.0%) | 16.8% (-0.4%) | 17.3% (-1.9%) | 18.7% (-5.7%) | 17.9% (-3.8%) |
| 12 months | 30 | 26.6% (+10.6%) | 23.9% (+3.9%) | 24.5% (+12.6%) | 23.0% (+12.6%) | 20.3% (+0.9%) | 21.5% (+5.0%) |
| 18 months | 25 | 23.4% (+5.3%) | 20.6% (+2.8%) | 23.5% (+5.4%) | 22.3% (+4.7%) | 19.5% (-1.2%) | 20.6% (+3.8%) |
| 24 months | 14 | 27.5% (-6.5%) | 23.8% (-13.4%) | 28.2% (+2.7%) | 25.5% (-4.8%) | 22.0% (-14.2%) | 22.5% (-11.7%) |

*Mean absolute held-out volume error (median bias).* The b cap is the larger lever.
Against a 1.5 cap, 1.2 cuts the error by 1.5 points at 12 months (90% CI 0.5–2.7) and
1.7 at 18 (0.1–3.4), and is not significantly different at 6 months (+0.7, CI −0.2 to
+1.6) or 24 (−3.0, CI −6.4 to +0.1). A 1.0 cap is better still at 12 months (by 1.2
points) but under-forecasts the 24-month window by 14% (median), so 1.2 was kept. Given
the cap, the clock helps at 12 months (2.4 points better than calendar time at the same
cap, CI 0.0–6.1) and is level at 18–24. Against the first version (calendar time,
b ≤ 2) the current method is 5.1 points better at 12 months (1.9–8.8), 2.8 at 18
(1.2–4.6) and 5.0 at 24 (1.1–9.2), and level at 6 (+0.6, CI −1.6 to +2.9). Caveats: the
candidates, the ramp and the cap were all chosen on these same 30 wells, so the
intervals are optimistic, and the windows overlap. The gain is real but modest — about a
fifth less error on a one-year forecast — and these are the honest error bars on such a
forecast.

### How the ranges were calibrated

`python scripts/calibrate_ranges.py` hides each well's last 6, 12 or 18 reported months,
builds the ensemble from the rest exactly as the pipeline does, and asks whether the
held-out volume falls inside P90–P10. For each window it finds the smallest level
factor that would cover it; the factor used (0.21) is the split-conformal 80% quantile
of those (0.203, rounded up). Coverage is then scored leave-one-well-out — each well's
windows with the factor calibrated on the other 29 wells:

| Holdout | Windows | Coverage of P90–P10, leave-one-well-out | Mean P10/P90 volume ratio | Coverage, bootstrap alone | Coverage, constant width | Interval score: ensemble / constant width |
|---|---|---|---|---|---|---|
| 6 months | 30 | 77% | 1.86 | 40% | 80% | 0.951 / 0.884 |
| 12 months | 30 | 80% | 1.90 | 43% | 80% | 0.981 / 1.004 |
| 18 months | 25 | 84% | 2.07 | 64% | 80% | 1.034 / 1.003 |
| All | 85 | 80% | 1.94 | 48% | 80% | 0.986 / 0.961 |

What this does and does not show:

- Over 6–18 months the ranges are calibrated (80% held out, 77–84% by horizon) — but no
  sharper than a band of fixed relative width around the fit (fit × e^±0.31), which
  scores slightly better overall on the interval score (Gneiting & Raftery, 2007; lower
  is better: width plus a penalty for misses). The per-well bootstrap widths do not
  predict which wells will miss over a year or so.
- What the ensemble adds is a spread that grows with the forecast horizon, as the
  uncertainty in b and Di compounds, while every member stays a bounded Arps curve. A
  fixed-width band would claim a 30-year remaining volume is as certain as next year's.
  That growth is the model's extrapolation: with 30–51 months of history, nothing here
  can test a range for remaining life measured in decades. The level factor is applied
  to the whole remaining life, which is conservative if part of a short-term miss is
  transient.
- Remaining-volume P10/P90 ratios run 1.8–6.5 (median 2.6); EUR ratios 1.07–2.16
  (median 1.36), narrower because cumulative production is known.

### Findings from this cohort

Computed from `results/midland_basin_dca_results.csv` (`python -m src.pipeline` prints
the headline numbers and the cluster table):

- Mean b-factor **0.86** (median 0.96; 11 of 30 wells above 1), mean nominal initial
  decline **~324%/yr** (median 312%/yr) — classic steep shale decline from flush
  production. **8 wells are held at the 1.2 b cap** (DAVIS-LACY 31I #1, SCHARBAUER-DOVE
  7D #4 and 7F #8, DAVIS-LACY 31K #9, EPLEY-MARTIN 31D #11, FORTUNE-ASHLEY 31M #12,
  EPLEY-MARTIN 31H #18, GLYNN-ATCHISON 44M #25); they hold 33% of cohort EUR, and their
  EURs carry the cap (a 1.5 cap adds 2.2% to the cohort mean, see below).
- Mean EUR **~501,000 bbl** per well (median ~502,000; range ~169,000–865,000 bbl); mean
  P90–P10 **~402,000–577,000 bbl**. The widest ranges are DAVIS-LACY 31K (332,000–717,000)
  — at the b cap after a curtailment — EPLEY-MARTIN 31E and DAVIS-LACY 31F.
- **Economic limits run 108–137 bbl/mo** (median 130, ~4 BOPD): wells with more gas per
  barrel stay economic to lower oil rates. Forecast lives run 2–38 years (median 28).
- **PDP PV10 $139 MM** for the 30 wells at the base case (median well $4.7 MM, range
  $0.6–9.4 MM; undiscounted net cash flow $230 MM). PV10 moves about 26% for a $15/bbl
  oil price change while EUR barely moves (the limit shifts, the decline does not).
- **Shut-in oil prices are low: median $2.28/bbl**, and negative for 6 wells, whose gas
  revenue at $2/mcf and a GOR of up to 10 mcf/bbl already covers their operating cost
  at today's rate. At $0 gas the economic limit rises to a median 145 bbl/mo and PV10
  falls 11%.
- 27 of 30 wells have at least one month after peak under 50% of their first-pass
  trend (median downtime 1.5 months; the most is WELCH-COX E39V, 6.4 months). Eight have
  3+ such months, among them EPLEY-MARTIN 31E: its 2023 dip (down to 271 bbl) is read
  as downtime and the rebound to ~12,000 bbl/mo as the well resuming its trend, which
  gives b 0.98 and 673,000 bbl instead of the capped b 1.2 and 748,000 bbl a single
  calendar-time curve gives under the same assumptions. If that rebound was a
  re-completion, a restart-segment forecast would be better.
- Mean fit R² **0.92** (median 0.95) over the months each well was fitted on; all 30
  wells have R² > 0.6 in linear space and 29 in log space.
- **5 of 30 wells are low-confidence:** WELCH-COX E39V (#5 by EUR) and DAVIS-LACY 31D
  (#28) end the record shut in or curtailed (E39V for its last two months, 31D at
  ~80 bbl/mo for three) and are forecast as returning to trend — confirm their status
  before relying on their remaining volumes; 31D also has b on the 0.05 floor and a
  forecast starting at 2.03× its recent rate. BOHANAN-BLOCKER 3B's (#7) forecast starts
  at 0.70× its recent rate (its last six reported months ran above its fitted trend).
  GLYNN-ATCHISON 44M (#25) has a log-space R² of 0.60 at the b cap, and DAVIS-LACY 31H
  (#30) has b on the 0.05 floor.
- No forecast reaches the 50-year life cap under the base case.
- Two decline-shape clusters. k=2 is chosen by silhouette (0.58 vs 0.49 for k=3) and
  is fairly stable under bootstrap resampling (adjusted Rand index 0.84):

  | Cluster | Wells | Peak rate kept after 1 yr / 3 yr | Mean b | Mean Di (nom.) | Mean EUR (bbl) | Mean EUR / 1,000 ft |
  |---|---|---|---|---|---|---|
  | Steep decline | 14 | 18% / 6% | 0.89 | 453%/yr | 488,000 | 49,000 |
  | Shallow decline | 16 | 33% / 12% | 0.84 | 211%/yr | 512,000 | 51,000 |

  The two clusters have about the same mean b; what separates them is how fast the
  first years decline. The steep cluster holds 2 low-confidence wells, the shallow 3.
  The shallow cluster's 5% higher mean EUR is not statistically significant (Mann-Whitney
  p ≈ 0.8).

### Forecast and economic assumptions

| Assumption | Base case | Basis |
|---|---|---|
| Oil price | $65/bbl, flat, realized at the lease | Assumption, not a forecast or strip (slider $30–120) |
| Gas price | $2.00/mcf, flat, wellhead casinghead gas incl. NGL value net of processing | Assumption, the least certain input: Permian (Waha) residue gas has traded near or below zero at times (slider $0–6) |
| Gas volume | each well's last-12-producing-month GOR (1.9–10.4 mcf/bbl, median 3.5), held flat | Data; GORs usually rise with time, so gas is if anything understated |
| Royalty / NRI | 75% NRI (25% royalty), 100% working interest | Assumption (slider 62.5–87.5%) |
| Severance tax | 4.6% of oil value, 7.5% of gas value | Texas production tax rates |
| Ad valorem tax | 2.5% of revenue | Assumption |
| Operating cost | $6,000 per well-month + $4.00/bbl oil | Round-number assumption, not a quote: $3.1–6.9/boe at today's rates (median $3.9), ~$32/boe at the economic limit (sliders) |
| Discounting | 10%/yr, mid-month, from April 2026 | PV10 convention (slider 5–20%) |
| Terminal decline Dmin | 8%/yr nominal | Between the earlier 5% and the source dataset's own 9.6% (slider 4–12%) |
| Maximum well life | 50 years from first production | Assumption (slider 20–60) |
| b cap | 1.2 | Hindcast (above); not a slider because it changes the fits |

The fits do not depend on the price, cost, Dmin or life assumptions; only the forecast
tail, EUR and value do. In the Streamlit app all of them except the tax rates are
sliders (the static Vercel page shows the base case).
`python scripts/sensitivity.py` reproduces this table (P50 values):

| Case | Median econ. limit (bbl/mo) | Mean EUR (bbl) | Median EUR (bbl) | EUR vs base | Total PV10 ($MM) | PV10 vs base |
|---|---|---|---|---|---|---|
| Base case | 130 | 501,000 | 502,000 | +0.0% | 138.8 | +0.0% |
| Oil $50/bbl | 169 | 496,000 | 496,000 | -1.0% | 102.9 | -25.8% |
| Oil $80/bbl | 106 | 504,000 | 505,000 | +0.6% | 174.8 | +25.9% |
| Gas $0/mcf | 145 | 499,000 | 499,000 | -0.4% | 122.9 | -11.4% |
| Fixed LOE +50% ($9,000/mo) | 196 | 492,000 | 492,000 | -1.7% | 129.5 | -6.7% |
| Flatter tail (Dmin 5%) | 130 | 534,000 | 518,000 | +6.6% | 141.8 | +2.2% |
| Steeper tail (Dmin 10%) | 130 | 483,000 | 494,000 | -3.5% | 135.8 | -2.2% |
| 30-year well life | 130 | 493,000 | 502,000 | -1.6% | 138.6 | -0.1% |
| Source dataset's QC assumptions (fixed 250 bbl/mo) | 250 | 473,000 | 480,000 | -5.5% | 135.4 | -2.4% |
| b cap 1.0 | 130 | 484,000 | 502,000 | -3.4% | 129.8 | -6.5% |
| b cap 1.5 (previous) | 130 | 512,000 | 502,000 | +2.2% | 144.8 | +4.4% |
| b cap 2.0 (first version's bound) | 130 | 518,000 | 502,000 | +3.4% | 148.1 | +6.7% |
| No b cap (b ≤ 5) | 130 | 521,000 | 502,000 | +4.1% | 150.2 | +8.2% |

The cohort is insensitive to the economic assumptions in volume (the economic limit sits
far below today's rates) and sensitive in value; Dmin and the b cap move the tail.

### Limitations

- 30-well convenience cohort (one operator, one county) — not a statistical sample
  of the Midland Basin.
- Prices, costs, terminal decline, well life and the b cap are assumptions; EURs and
  PV10 are technical estimates, not SPE PRMS reserves. See *Forecast and economic
  assumptions* for how much they move the results.
- The P90–P10 ranges are calibrated only for 6–18 months ahead; for remaining life they
  are the model's extrapolation. Well ranges cannot be added up: unless every well
  misses the same way, the sum of the wells' P90s is below the cohort's own P90, so only
  P50 totals are shown. Pad-mates share operations (curtailments, frac hits), so the wells
  are not independent either.
- Downtime is inferred from the fit because the data has no days-on: a month well
  under trend is read as part-month downtime. A genuine, lasting loss of productivity
  would be read the same way, and because those months are down-weighted, the fitted
  trend can sit above them; the "confirm status" flag and the month counts mark the
  wells where this matters.
- One curve per well: a well whose rebound is a genuine re-completion (EPLEY-MARTIN 31E
  may be one) would be better forecast from the restart alone, which needs the
  operator's confirmation.
- The economics are PDP only, flat-price, 100% WI, with no capital, workovers,
  abandonment or hedges; the gas forecast is the oil forecast times a flat GOR.

## Run it locally

Python 3.12 or 3.13 (`.python-version` pins 3.13, which CI's output check uses).

```bash
git clone <this-repo>
cd midland-basin-dca-dashboard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fit, forecast, value and cluster all wells (~20 s). Writes
# results/midland_basin_dca_results.csv, observed_type_curve.csv, forecast_ensemble.csv
python -m src.pipeline

# Launch the dashboard
streamlit run app.py
```

Then open http://localhost:8501.

- Tests: `pip install -r requirements-dev.txt && pytest` (~30 s).
- Hindcast of the fitting method and b cap: `python scripts/backtest.py`.
- Calibration check of the P90–P10 ranges: `python scripts/calibrate_ranges.py` (~1 min).
- Sensitivity to prices, costs and decline assumptions: `python scripts/sensitivity.py`.

`pyproject.toml` holds the project metadata, the supported Python versions and the pytest
settings; its pins must match `requirements.txt` (a test checks), which is what
Streamlit Community Cloud installs from.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- **Tests** on Python 3.12 and 3.13.
- **Committed outputs are current** — rebuilds `results/` and `public/` with
  `scripts/build_static.py` and compares them with the committed copies
  (`scripts/check_outputs.py`): same files, columns and rows, identical text, numbers
  equal to a relative 1e-6 (a rebuild on another machine can differ in the last digits
  of fitted values; that is not staleness). If it fails, run
  `python scripts/build_static.py` and commit `results/` and `public/`.

## Deploy free on Streamlit Cloud

1. Push this repo to GitHub (data files included — they're small).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo, branch, and `app.py`; choose Python 3.12 or 3.13 under advanced
   settings.
3. Deploy. On first launch the app runs the pipeline if `results/` is missing, so no
   extra steps are needed.

## Deploy on Vercel (static build)

Streamlit needs a long-running server, so the Vercel deployment is a static build of
the same dashboard: `scripts/build_static.py` runs the pipeline (`src/`) on the data in
`data/` and writes `public/index.html` (Plotly charts with the P90–P10 bands, well
selector, cohort table, clusters, CSV download). It shows the base case; the sliders
need the Streamlit app. `vercel.json` serves the committed `public/` as-is (no build
step on Vercel), and CI checks that it is current. The page loads Plotly.js from
cdn.plot.ly at the version matching the installed `plotly` package; if the CDN is
blocked the tables and CSV download still work.

```bash
python scripts/build_static.py   # regenerate public/ after changing data or code
vercel deploy --prod
```

## Project structure

```
midland-basin-dca-dashboard/
├── app.py                 # Streamlit dashboard
├── pyproject.toml         # metadata, Python versions, pytest settings
├── .python-version        # 3.13
├── requirements.txt       # pinned runtime dependencies
├── requirements-dev.txt   # + pytest
├── vercel.json            # static deploy of public/
├── .github/workflows/ci.yml
├── data/
│   ├── martin_selected_30_monthly_production_normalized.csv  # RRC monthly production (via GitHub dataset, see above)
│   ├── martin_selected_30_wells.csv                          # well metadata / completions
│   ├── martin_observed_type_curve.csv                        # upstream artifact, unused (see Known data issues)
│   └── martin_real_cohort_analysis_qc.json                   # upstream artifact, unused (see Known data issues)
├── results/
│   ├── midland_basin_dca_results.csv   # pipeline output: fits, EUR P90/P50/P10, PV10, QC flags, clusters
│   ├── forecast_ensemble.csv           # 200 Arps curves per well behind P90/P10
│   └── observed_type_curve.csv         # P25/P50/P75 by month on production
├── public/                # static Vercel build (generated)
├── scripts/
│   ├── build_static.py    # builds public/ from src/ + data/
│   ├── backtest.py        # hindcast of the fitting method and b cap
│   ├── calibrate_ranges.py # calibration and coverage check of P90/P10
│   ├── sensitivity.py     # EUR and PV10 vs assumptions
│   ├── check_outputs.py   # CI: committed outputs vs a fresh build
│   └── template.html
├── src/
│   ├── data_loader.py     # load + clean production series
│   ├── arps.py            # Arps models, curve fitting, forecasts, EUR
│   ├── economics.py       # PDP cash flow, PV10, economic limit, shut-in price
│   ├── uncertainty.py     # forecast ensemble and P90/P10 ranges
│   ├── clustering.py      # k-means on decline shape, k by silhouette
│   ├── figures.py         # Plotly charts shared by both front ends
│   ├── report.py          # captions / labels / tables shared by both front ends
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
