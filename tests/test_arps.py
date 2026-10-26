"""Unit tests for the Arps model, fitting, and forecast/EUR bookkeeping."""
import numpy as np
import pytest

from src.arps import (
    B_CAP,
    _switch_time,
    eur_bbl,
    fit_decline,
    fit_hyperbolic,
    forecast_to_limit,
    hyperbolic_rate,
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
    fit, clock = fit_decline(q, qi_max=q.max())
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
    q[20] *= 0.3  # month produced at 30% of trend
    fit, _ = fit_decline(q, qi_max=q.max())
    assert fit["downtime_months"] == pytest.approx(0.7, abs=0.05)
    assert fit["curtailed_months_at_end"] == 0


def test_curtailed_months_at_end_are_counted():
    t = np.arange(0, 30, dtype=float)
    q = hyperbolic_rate(t, 30000.0, 0.25, 0.9)
    q[-3:] = [100.0, 0.0, 80.0]
    fit, _ = fit_decline(q, qi_max=q.max())
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


def test_returned_clock_is_the_one_the_fit_used():
    prod_t = np.arange(0, 36, dtype=float)
    q_prod = hyperbolic_rate(prod_t, 30000.0, 0.25, 0.9)
    q = np.r_[q_prod[:15], np.zeros(4), q_prod[15:]]
    q[25] *= 0.3  # one curtailed month as well
    fit, clock = fit_decline(q, qi_max=q.max())
    frac = np.diff(np.r_[clock, fit["producing_months"]])
    full = (frac >= 1.0) & (q > 0)
    refit = fit_hyperbolic(clock, np.where(full, q, 0.0), qi_max=q.max())
    for key in ("qi", "Di_monthly", "b"):
        assert refit[key] == pytest.approx(fit[key], rel=1e-9)


def test_curtailed_months_do_not_drag_the_trend_down():
    # On-trend well whose last three months produced at 30% of trend: they are
    # downtime on the clock and must not also be fitted as on-trend points.
    prod_t = np.arange(0, 40, dtype=float)
    q = hyperbolic_rate(prod_t, 20000.0, 0.25, 1.0)
    q[-3:] *= 0.3
    fit, _ = fit_decline(q, qi_max=q.max())
    assert fit["b"] == pytest.approx(1.0, abs=0.01)
    assert fit["curtailed_months"] == 3 and fit["curtailed_months_at_end"] == 3
    # Each curtailed month is compared with the trend at the (paused) clock, so
    # slightly more than 0.7 of each reads as downtime.
    assert 2.1 <= fit["downtime_months"] <= 2.2


def test_result_does_not_depend_on_the_pass_limit(monkeypatch):
    import src.arps as arps
    rng = np.random.default_rng(3)
    t = np.arange(0, 40, dtype=float)
    q = hyperbolic_rate(t, 25000.0, 0.3, 1.1) * np.exp(rng.normal(0, 0.15, len(t)))
    q[[8, 9, 22, 30]] *= [0.0, 0.45, 0.52, 0.2]
    results = []
    for passes in (10, 11, 30):
        monkeypatch.setattr(arps, "MAX_PASSES", passes)
        fit, _ = fit_decline(q, qi_max=q.max())
        results.append((fit["b"], fit["producing_months"]))
    assert results[0] == results[1] == results[2]
