"""Unit tests for the PDP cash-flow model."""
import numpy as np
import pandas as pd
import pytest

from src.economics import (Economics, cash_flow, discount_factors, economic_limit,
                           net_revenue_per_bbl, present_value, shut_in_price, trailing_gor)

E = Economics()


def test_net_revenue_per_bbl_by_hand():
    # 75% NRI x (1 - 2.5% ad valorem) x ($65 x (1 - 4.6%) + 3 mcf x $2 x (1 - 7.5%))
    assert net_revenue_per_bbl(E, 3.0) == pytest.approx(0.75 * 0.975 * (65 * 0.954 + 3 * 2 * 0.925))


def test_cash_flow_is_zero_at_the_economic_limit():
    gor = 3.5
    lim = economic_limit(E, gor)
    assert cash_flow(np.array([lim]), E, gor)[0] == pytest.approx(0.0, abs=1e-9)
    assert cash_flow(np.array([lim * 1.01]), E, gor)[0] > 0
    assert cash_flow(np.array([lim * 0.99]), E, gor)[0] < 0
    assert 100 < lim < 150  # ~4 BOPD at the base case


def test_no_rate_is_economic_when_margin_does_not_cover_variable_cost():
    poor = Economics(oil_price=4.0, gas_price=0.0)
    assert economic_limit(poor, 3.0) == float("inf")


def test_months_after_shut_in_carry_no_fixed_cost():
    assert cash_flow(np.array([0.0]), E, 3.0)[0] == 0.0


def test_present_value_mid_month():
    ncf = np.full(24, 1000.0)
    expected = sum(1000.0 / 1.1 ** ((i + 0.5) / 12) for i in range(24))
    assert present_value(ncf, 0.10) == pytest.approx(expected)
    assert discount_factors(12, 0.10)[0] == pytest.approx(1.1 ** (-0.5 / 12))
    np.testing.assert_allclose(present_value(np.vstack([ncf, 2 * ncf]), 0.10),
                               [expected, 2 * expected])


def test_shut_in_price_is_where_this_months_cash_flow_is_zero():
    gor, q = 2.5, 1800.0
    p = shut_in_price(E, gor, q)
    at_p = Economics(oil_price=p)
    assert cash_flow(np.array([q]), at_p, gor)[0] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(shut_in_price(E, gor, 0.0))


def test_trailing_gor_uses_last_producing_months():
    oil = pd.Series([100.0] * 20 + [0.0, 0.0])
    gas = pd.Series([100.0] * 8 + [300.0] * 12 + [50.0, 50.0])
    assert trailing_gor(oil, gas) == pytest.approx(3.0)
    assert trailing_gor(pd.Series([0.0]), pd.Series([10.0])) == 0.0


def test_assumptions_round_trip_through_results_columns():
    e = Economics(oil_price=71.0, loe_fixed=4500.0)
    assert Economics.from_row(pd.Series(e.as_columns())) == e
