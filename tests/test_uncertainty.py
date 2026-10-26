"""Unit tests for the forecast ensemble behind P90 / P10."""
import numpy as np
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


def test_level_factors_are_evenly_spread_lognormal():
    f = uncertainty.level_factors(200, np.random.default_rng(1), sigma=0.2)
    np.testing.assert_allclose(np.sort(np.log(f)) / 0.2, norm.ppf((np.arange(200) + 0.5) / 200))


def test_noise_free_well_bootstraps_to_its_own_fit():
    q = _well(noise=0.0)
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    p = uncertainty.bootstrap_fits(q, clock, weight, fit, q.max(), 1.2,
                                   np.random.default_rng(0), samples=20)
    np.testing.assert_allclose(p, [[fit["qi"], fit["Di_monthly"], fit["b"]]] * 20, rtol=1e-5)


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
    for noise in (0.05, 0.25):
        q = _well(noise=noise, seed=3)
        fit, clock, weight = fit_decline(q, qi_max=q.max())
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
