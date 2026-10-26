"""Unit tests for the Arps model, fitting, and forecast/EUR bookkeeping."""
import numpy as np
import pytest

from src.arps import (
    _switch_time,
    eur_bbl,
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
