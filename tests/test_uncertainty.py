"""Unit tests for the forecast ensemble behind P90 / P10."""
import io

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import uncertainty
from src.arps import fit_decline, hyperbolic_rate
from src.economics import Economics

DMIN = 0.08 / 12


def _well(noise=0.12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 40, dtype=float)
    return hyperbolic_rate(t, 25000.0, 0.3, 0.9) * np.exp(rng.normal(0, noise, len(t)))


def test_block_indices_are_runs_of_consecutive_months():
    idx = uncertainty.block_indices(10, 50, np.random.default_rng(0), block=3)
    assert idx.shape == (50, 10) and idx.min() >= 0 and idx.max() <= 9
    for row in idx:
        for start in range(0, 9, 3):
            run = row[start:start + 3]
            assert (np.diff(run) == 1).all()


def test_level_scores_are_evenly_spread_normal():
    z = uncertainty.stratified_normal(200, np.random.default_rng(1))
    np.testing.assert_allclose(np.sort(z), norm.ppf((np.arange(200) + 0.5) / 200))


def test_resampled_months_come_from_the_pool_only():
    pool = np.ones(30, dtype=bool)
    pool[[4, 5, 17, 29]] = False
    src = uncertainty.resample_months(pool, 300, np.random.default_rng(2))
    assert src.shape == (300, 30)
    assert pool[src].all()
    half = pool.astype(float)
    half[12] = 0.5  # a month half in the pool is drawn about half as often
    counts = np.bincount(uncertainty.resample_months(half, 2000, np.random.default_rng(5))
                         .ravel(), minlength=30)
    assert 0.35 < counts[12] / np.median(counts[2:-2][pool[2:-2]]) < 0.65
    # Every month in the pool is drawn; away from the ends of the history (which
    # fewer blocks cover) at about the same rate.
    counts = np.bincount(src.ravel(), minlength=30)
    assert (counts[pool] > 0).all()
    inner = counts[2:-2][pool[2:-2]]
    assert inner.min() > 0.7 * inner.mean()


def test_one_month_joining_the_pool_changes_few_draws():
    # The draws used to depend on the number of usable months, so one month's
    # weight crossing zero redrew every bootstrap sample.
    pool = np.ones(36, dtype=bool)
    pool[[10, 22]] = False
    a = uncertainty.resample_months(pool, 500, np.random.default_rng(4))
    pool[22] = True
    b = uncertainty.resample_months(pool, 500, np.random.default_rng(4))
    assert (a != b).mean() < 0.06
    # A month edging into the pool changes almost nothing.
    soft = pool.astype(float)
    soft[10] = 0.98
    c = uncertainty.resample_months(soft, 500, np.random.default_rng(4))
    soft[10] = 1.0
    d = uncertainty.resample_months(soft, 500, np.random.default_rng(4))
    assert (c != d).mean() < 0.005


def test_noise_free_well_bootstraps_to_its_own_fit():
    q = _well(noise=0.0)
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    p, z = uncertainty.members(q, clock, weight, fit, q.max(), 1.2, seed=0, samples=20)
    np.testing.assert_allclose(p, [[fit["qi"], fit["Di_monthly"], fit["b"]]] * 20, rtol=1e-5)
    assert len(z) == 20


def test_down_weighted_months_do_not_pull_the_ensemble_low():
    # Curtailed months sit below trend because of downtime; resampling their
    # residuals into full months used to read as extra dips and bias every member
    # low. (A well with no curtailment has ~55% of members below its fit: the
    # forecast volume is not linear in b and Di.)
    q = _well(noise=0.1, seed=11)
    t = np.arange(len(q), dtype=float)
    trend = hyperbolic_rate(t, 25000.0, 0.3, 0.9)
    for k, share in ((8, 0.45), (15, 0.5), (22, 0.4), (29, 0.55), (33, 0.5)):
        q[k] = share * trend[k]
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    assert ((weight > 0) & (weight < 1)).sum() >= 4
    p, _ = uncertainty.members(q, clock, weight, fit, q.max(), 1.2, seed=3, samples=300)
    start = fit["producing_months"]
    vol = uncertainty.ensemble_rates(p, start, 120, DMIN).sum(axis=1)
    det = uncertainty.ensemble_rates(np.array([[fit["qi"], fit["Di_monthly"], fit["b"]]]),
                                     start, 120, DMIN).sum()
    assert 0.3 < (vol < det).mean() < 0.6


def test_ensemble_is_reproducible_by_seed():
    q = _well()
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    a = uncertainty.ensemble(q, clock, weight, fit, q.max(), 1.2, seed=5, samples=30)
    b = uncertainty.ensemble(q, clock, weight, fit, q.max(), 1.2, seed=5, samples=30)
    c = uncertainty.ensemble(q, clock, weight, fit, q.max(), 1.2, seed=6, samples=30)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert (a[:, 2] <= 1.2 + 1e-9).all()


def test_ranges_widen_with_noise_and_respect_the_limit():
    e = Economics()
    widths = []
    for noise in (0.03, 0.15):
        q = _well(noise=noise, seed=3)
        fit, clock, weight = fit_decline(q, qi_max=q.max())
        assert not fit["b_at_cap"]  # at the cap b can only fall, which narrows the range
        p = uncertainty.ensemble(q, clock, weight, fit, q.max(), 1.2, seed=1, samples=100)
        r = uncertainty.ranges(p, fit["producing_months"], 400, DMIN, 130.0, e, 3.0)
        lo, hi = r["remaining"]
        widths.append(hi / lo)
        assert r["pv"][0] < r["pv"][1]
    assert widths[1] > widths[0] > 1
    assert uncertainty.ranges(p, 40.0, 0, DMIN, 130.0, e, 3.0)["remaining"] == (0.0, 0.0)


def test_frame_round_trip():
    ens = {101: np.array([[1.0, 0.1, 0.5], [2.0, 0.2, 0.6]]), 7: np.array([[3.0, 0.3, 0.7]])}
    back = uncertainty.from_frame(uncertainty.to_frame(ens))
    assert set(back) == {101, 7}
    for k in ens:
        np.testing.assert_array_equal(back[k], ens[k])


def test_saved_precision_round_trips_exactly():
    q = _well()
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    p = uncertainty.ensemble(q, clock, weight, fit, q.max(), 1.2, seed=2, samples=20)
    back = uncertainty.from_frame(pd.read_csv(io.StringIO(
        uncertainty.to_frame({1: p}).to_csv(index=False))))[1]
    np.testing.assert_array_equal(back, p)
