"""Unit tests for the Arps model, fitting, and forecast/EUR bookkeeping."""
import numpy as np
import pytest

from src.arps import (
    B_CAP,
    CURTAIL_RAMP,
    _switch_time,
    eur_bbl,
    fit_decline,
    fit_hyperbolic,
    forecast_to_limit,
    hyperbolic_rate,
    month_weights,
    months_above,
    terminal_rate,
)

DMIN = 0.05 / 12


def test_switch_time_is_where_decline_hits_dmin():
    qi, Di, b = 30000.0, 0.3, 0.9
    t_star = _switch_time(qi, Di, b, DMIN)
    assert Di / (1 + b * Di * t_star) == pytest.approx(DMIN)


def test_terminal_rate_continuous_at_switch():
    qi, Di, b = 30000.0, 0.3, 0.9
    t_star = _switch_time(qi, Di, b, DMIN)
    q = terminal_rate(np.array([t_star - 1e-6, t_star + 1e-6]), qi, Di, b, DMIN)
    assert q[0] == pytest.approx(q[1], rel=1e-6)


def test_terminal_rate_declines_at_dmin_after_switch():
    qi, Di, b = 30000.0, 0.3, 1.5
    t_star = _switch_time(qi, Di, b, DMIN)
    q = terminal_rate(np.array([t_star + 10, t_star + 11]), qi, Di, b, DMIN)
    assert np.log(q[0] / q[1]) == pytest.approx(DMIN)


def test_qi_is_rate_at_t0():
    assert hyperbolic_rate(np.array([0.0]), 25000.0, 0.2, 1.0)[0] == pytest.approx(25000.0)


def test_fit_recovers_synthetic_parameters():
    t = np.arange(0, 40, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.25, 0.9)
    fit = fit_hyperbolic(t, q, qi_max=q.max())
    assert fit["status"] == "ok"
    assert fit["qi"] == pytest.approx(30000.0, rel=1e-3)
    assert fit["Di_monthly"] == pytest.approx(0.25, rel=1e-3)
    assert fit["b"] == pytest.approx(0.9, rel=1e-3)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_single_partial_month_does_not_wreck_fit():
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 40000.0, 0.3, 1.0)
    q[24] = 4.0  # restart month with a few barrels
    fit = fit_hyperbolic(t, q, qi_max=q.max())
    assert fit["b"] == pytest.approx(1.0, abs=0.15)
    assert fit["Di_monthly"] == pytest.approx(0.3, rel=0.15)


def test_partial_month_left_out_of_log_r2_only():
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 40000.0, 0.3, 1.0)
    q[24] = 4.0
    fit = fit_hyperbolic(t, q, qi_max=q.max())
    assert fit["n_partial_months"] == 1
    assert fit["r_squared_log"] > 0.99
    clean = fit_hyperbolic(t, hyperbolic_rate(t, 40000.0, 0.3, 1.0), qi_max=40000.0)
    assert clean["n_partial_months"] == 0


def test_bounds_hit_reported():
    t = np.arange(0, 30, dtype=float)
    q = 30000.0 * np.exp(-0.2 * t)  # exponential: b wants 0, hits the 0.05 floor
    assert fit_hyperbolic(t, q, qi_max=q.max())["bounds_hit"] == "b"


def test_insufficient_data():
    fit = fit_hyperbolic(np.arange(5.0), np.full(5, 1000.0))
    assert fit["status"] == "insufficient_data"


def test_forecast_starts_at_start_t_and_stops_at_limit():
    qi, Di, b = 30000.0, 0.3, 0.9
    fc = forecast_to_limit(qi, Di, b, DMIN, econ_limit_bpm=150.0, start_t=10)
    assert fc[0] == pytest.approx(terminal_rate(np.array([10.0]), qi, Di, b, DMIN)[0])
    assert fc[-1] >= 150.0
    nxt = terminal_rate(np.array([10.0 + len(fc)]), qi, Di, b, DMIN)[0]
    assert nxt < 150.0


def test_eur_is_cum_plus_forecast():
    qi, Di, b = 30000.0, 0.3, 0.9
    fc = forecast_to_limit(qi, Di, b, DMIN, 150.0, start_t=12)
    e = eur_bbl(qi, Di, b, 123456.0, DMIN, 150.0, start_t=12)
    assert e["remaining_bbl"] == pytest.approx(fc.sum())
    assert e["eur_bbl"] == pytest.approx(123456.0 + fc.sum())


def test_shut_in_months_do_not_count_as_decline_time():
    # The well follows its decline in producing time and is shut in for 4
    # months; on restart it picks up where it left off (no depletion while shut in).
    prod_t = np.arange(0, 36, dtype=float)
    q_prod = hyperbolic_rate(prod_t, 30000.0, 0.25, 0.9)
    q = np.r_[q_prod[:15], np.zeros(4), q_prod[15:]]
    fit, clock, _ = fit_decline(q, qi_max=q.max())
    assert fit["b"] == pytest.approx(0.9, abs=0.02)
    assert fit["Di_monthly"] == pytest.approx(0.25, rel=0.02)
    assert fit["downtime_months"] == pytest.approx(4.0)
    assert fit["producing_months"] == pytest.approx(36.0)
    assert clock[15] == clock[18] == pytest.approx(15.0)  # clock paused through the shut-in


def test_calendar_time_fit_of_the_same_well_would_inflate_b():
    prod_t = np.arange(0, 36, dtype=float)
    q_prod = hyperbolic_rate(prod_t, 30000.0, 0.25, 0.9)
    q = np.r_[q_prod[:15], np.zeros(4), q_prod[15:]]
    cal = fit_hyperbolic(np.arange(len(q), dtype=float), q, qi_max=q.max())
    assert cal["b"] > 1.0  # the rebound reads as a flatter tail


def test_curtailed_month_counts_as_partly_producing():
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.25, 0.9)
    q[20] *= 0.25  # month produced at 25% of trend
    fit, _, weight = fit_decline(q, qi_max=q.max())
    assert fit["downtime_months"] == pytest.approx(0.75, abs=0.02)
    assert weight[20] == 0.0  # and is left out of the fit
    assert fit["curtailed_months"] == 1 and fit["curtailed_months_at_end"] == 0


def test_curtailed_months_at_end_are_counted():
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.25, 0.9)
    q[-3:] = [100.0, 0.0, 80.0]
    fit, _, _ = fit_decline(q, qi_max=q.max())
    assert fit["curtailed_months_at_end"] == 3


def test_b_is_capped_and_reported_separately_from_bound_hits():
    t = np.arange(0, 40, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.4, 2.0)
    fit = fit_hyperbolic(t, q, qi_max=q.max())
    assert fit["b"] == pytest.approx(B_CAP)
    assert fit["b_at_cap"] and "b" not in fit["bounds_hit"]


def test_forecast_accepts_fractional_start():
    qi, Di, b = 30000.0, 0.3, 0.9
    fc = forecast_to_limit(qi, Di, b, DMIN, 150.0, start_t=10.4)
    assert fc[0] == pytest.approx(terminal_rate(np.array([10.4]), qi, Di, b, DMIN)[0])
    assert fc[1] == pytest.approx(terminal_rate(np.array([11.4]), qi, Di, b, DMIN)[0])


def test_returned_clock_and_weights_are_the_ones_the_fit_used():
    prod_t = np.arange(0, 36, dtype=float)
    q_prod = hyperbolic_rate(prod_t, 30000.0, 0.25, 0.9)
    q = np.r_[q_prod[:15], np.zeros(4), q_prod[15:]]
    q[25] *= 0.4  # one curtailed month as well, inside the ramp
    fit, clock, weight = fit_decline(q, qi_max=q.max())
    assert 0 < weight[15 + 4 + 6] < 1
    assert clock[-1] + 1 == pytest.approx(fit["producing_months"])
    p = [np.log(fit["qi"]), fit["Di_monthly"], fit["b"]]
    refit = fit_hyperbolic(clock, q, qi_max=q.max(), weights=weight, p0=p)
    for key in ("qi", "Di_monthly", "b"):
        assert refit[key] == pytest.approx(fit[key], rel=1e-6)


def test_curtailed_months_do_not_drag_the_trend_down():
    # On-trend well whose last three months produced at 25% of trend: they are
    # downtime on the clock and must not also be fitted as on-trend points.
    prod_t = np.arange(0, 40, dtype=float)
    q = hyperbolic_rate(prod_t, 20000.0, 0.25, 1.0)
    q[-3:] *= 0.25
    fit, _, weight = fit_decline(q, qi_max=q.max())
    assert fit["b"] == pytest.approx(1.0, abs=0.01)
    assert (weight[-3:] == 0).all()
    assert fit["curtailed_months"] == 3 and fit["curtailed_months_at_end"] == 3
    # Each month is compared with the first pass's trend, which the robust
    # fit lets those months pull down only slightly.
    assert 2.0 <= fit["downtime_months"] <= 2.3


def test_month_weights_ramp():
    lo, hi = CURTAIL_RAMP
    trend = np.full(7, 1000.0)
    q = trend * np.array([0.0, 0.2, lo, (lo + hi) / 2, hi, 1.0, 1.4])
    frac, weight = month_weights(q, trend)
    np.testing.assert_allclose(frac, [0.0, 0.2, lo, lo + (1 - lo) / 2, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(weight, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])


def test_weight_zero_is_the_same_as_leaving_the_month_out():
    rng = np.random.default_rng(1)
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.3, 0.9) * np.exp(rng.normal(0, 0.1, len(t)))
    w = np.ones_like(q)
    w[[5, 17]] = 0.0
    a = fit_hyperbolic(t, q, qi_max=q.max(), weights=w)
    b = fit_hyperbolic(np.delete(t, [5, 17]), np.delete(q, [5, 17]), qi_max=q.max())
    for key in ("qi", "Di_monthly", "b"):
        assert a[key] == pytest.approx(b[key], rel=1e-6)


def test_eur_moves_smoothly_as_one_month_crosses_the_curtailment_ramp():
    # The earlier hard 50% cut-off made EUR jump when one month crossed it.
    rng = np.random.default_rng(7)
    t = np.arange(0, 40, dtype=float)
    base = hyperbolic_rate(t, 25000.0, 0.3, 0.9) * np.exp(rng.normal(0, 0.12, len(t)))
    eurs = []
    for share in np.arange(0.15, 0.95, 0.01):
        q = base.copy()
        q[34] = share * hyperbolic_rate(np.array([34.0]), 25000.0, 0.3, 0.9)[0]
        fit, _, _ = fit_decline(q, qi_max=q.max())
        eurs.append(eur_bbl(fit["qi"], fit["Di_monthly"], fit["b"], q.sum(), DMIN, 150.0,
                            start_t=fit["producing_months"])["eur_bbl"])
    steps = np.abs(np.diff(eurs)) / np.array(eurs[:-1])
    assert steps.max() < 0.005


def test_terminal_rate_broadcasts_over_an_ensemble():
    t = np.arange(0, 300, dtype=float)
    params = np.array([[30000.0, 0.3, 0.9], [20000.0, 0.004, 0.5], [5000.0, 0.8, 0.05]])
    together = terminal_rate(t[None, :], params[:, :1], params[:, 1:2], params[:, 2:3], DMIN)
    for i, (qi, Di, b) in enumerate(params):
        np.testing.assert_allclose(together[i], terminal_rate(t, qi, Di, b, DMIN), rtol=1e-12)


def test_forecast_matches_a_month_by_month_loop():
    qi, Di, b = 30000.0, 0.3, 0.9
    fc = forecast_to_limit(qi, Di, b, DMIN, 150.0, max_months=600, start_t=7.3)
    loop = []
    for i in range(600):
        qm = float(terminal_rate(7.3 + i, qi, Di, b, DMIN))
        if qm < 150.0:
            break
        loop.append(qm)
    np.testing.assert_allclose(fc, loop, rtol=1e-12)
    assert len(forecast_to_limit(qi, Di, b, DMIN, float("inf"), start_t=7.3)) == 0
    assert len(forecast_to_limit(qi, Di, b, DMIN, 150.0, max_months=0)) == 0


def test_months_above_stops_at_the_first_month_below():
    assert months_above(np.array([5.0, 4.0, 2.0, 6.0]), 3.0) == 2
    np.testing.assert_array_equal(months_above(np.array([[5.0, 1.0], [5.0, 5.0]]), 3.0), [1, 2])
