"""Tests for decline-behavior clustering."""
import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster_order, cluster_summary, cluster_wells


def _results(groups=((6.0, 1.7), (4.0, 1.0), (1.5, 0.4))):
    rng = np.random.default_rng(0)
    rows = []
    for g, (di, b) in enumerate(groups):
        for i in range(5):
            rows.append({"api8": 100 * g + i, "status": "ok", "bounds_hit": "",
                         "Di_annual": di + rng.normal(0, 0.1), "b": b + rng.normal(0, 0.03),
                         "eur_bbl": 1e5 * (g + 1), "eur_per_1000ft": 1e4, "r_squared": 0.9})
    rows.append({"api8": 999, "status": "failed", "Di_annual": np.nan, "b": np.nan,
                 "eur_bbl": np.nan, "eur_per_1000ft": np.nan, "r_squared": np.nan})
    return pd.DataFrame(rows)


def test_names_describe_both_axes_and_ids_order_by_di():
    out = cluster_wells(_results(), n_clusters=3)
    assert cluster_order(out) == ["High Di · high b", "Mid Di · mid b", "Low Di · low b"]
    hi = out[out["cluster"] == "High Di · high b"]
    assert (hi["cluster_id"] == 0).all() and hi["Di_annual"].min() > 5
    assert out.loc[out["api8"] == 999, "cluster"].item() == "Not fitted"


def test_axis_that_does_not_separate_clusters_shares_a_level():
    # High and mid groups differ on b but not on Di: both are 'High Di'.
    out = cluster_wells(_results(((5.0, 1.8), (4.8, 1.0), (1.5, 0.4))))
    assert cluster_order(out) == ["High Di · high b", "High Di · mid b", "Low Di · low b"]


def test_bound_stuck_well_does_not_set_the_name():
    res = _results(((5.0, 1.8), (4.8, 1.0), (1.5, 0.4)))
    res.loc[0, ["Di_annual", "bounds_hit"]] = [9.6, "Di"]  # one Di stuck on the bound
    names = set(cluster_wells(res)["cluster"])
    assert {"High Di · high b", "High Di · mid b"} <= names


def test_does_not_mutate_input_and_reclustering_is_safe():
    res = _results()
    cols = list(res.columns)
    once = cluster_wells(res)
    assert list(res.columns) == cols
    twice = cluster_wells(once)
    pd.testing.assert_series_equal(once["cluster"], twice["cluster"])


@pytest.mark.parametrize("k", [2, 4, 6])
def test_other_k_values_get_unique_names(k):
    out = cluster_wells(_results(), n_clusters=k)
    names = out.loc[out["status"] == "ok", "cluster"]
    assert names.nunique() == k


def test_insufficient_data_returns_copy():
    res = _results().head(2)
    out = cluster_wells(res, n_clusters=3)
    assert "cluster" not in res
    assert (out["cluster"] == "Insufficient data").all()


def test_summary_counts_only_fitted_wells():
    summ = cluster_summary(cluster_wells(_results()))
    assert summ["wells"].sum() == 15
    assert list(summ["cluster"]) == ["High Di · high b", "Mid Di · mid b", "Low Di · low b"]
