"""Tests for decline-shape clustering."""
import numpy as np
import pandas as pd
import pytest

from src.clustering import (choose_k, cluster_order, cluster_stability, cluster_summary,
                            cluster_wells, retention)


def _results(groups=((0.6, 0.4), (0.4, 0.2), (0.2, 0.1))):
    """Five wells per (rate kept after 1 yr, after 3 yr) group, plus a failed fit."""
    rng = np.random.default_rng(0)
    rows = []
    for g, (r12, r36) in enumerate(groups):
        for i in range(5):
            rows.append({"api8": 100 * g + i, "status": "ok", "bounds_hit": "",
                         "retention_12mo": r12 * np.exp(rng.normal(0, 0.03)),
                         "retention_36mo": r36 * np.exp(rng.normal(0, 0.03)),
                         "Di_annual": 3.0, "b": 1.0, "eur_bbl": 1e5 * (g + 1),
                         "eur_per_1000ft": 1e4, "r_squared": 0.9})
    rows.append({"api8": 999, "status": "failed", "bounds_hit": "",
                 "retention_12mo": np.nan, "retention_36mo": np.nan,
                 "Di_annual": np.nan, "b": np.nan, "eur_bbl": np.nan,
                 "eur_per_1000ft": np.nan, "r_squared": np.nan})
    return pd.DataFrame(rows)


def test_retention_is_fraction_of_peak_rate_kept():
    # Harmonic decline (b=1): q(t)/qi = 1 / (1 + Di t).
    assert retention(1000.0, 0.25, 1.0, 12) == pytest.approx(1 / (1 + 0.25 * 12))
    assert retention(1000.0, 0.25, 1.0, 0) == pytest.approx(1.0)


def test_names_describe_decline_shape_and_ids_run_steep_to_shallow():
    out = cluster_wells(_results(), n_clusters=3)
    assert cluster_order(out) == ["Steep decline", "Moderate decline", "Shallow decline"]
    steep = out[out["cluster"] == "Steep decline"]
    assert (steep["cluster_id"] == 0).all() and steep["retention_12mo"].max() < 0.3
    assert out.loc[out["api8"] == 999, "cluster"].item() == "Not fitted"


def test_horizons_that_disagree_are_both_named():
    # A and B keep the same rate after 1 yr but differ after 3 yr.
    out = cluster_wells(_results(((0.3, 0.2), (0.3, 0.1), (0.6, 0.4))), n_clusters=3)
    assert set(cluster_order(out)) == {"Steep year 1, moderate after", "Steep decline",
                                      "Shallow decline"}


def test_bound_stuck_well_does_not_set_the_name():
    res = _results()
    res.loc[0, ["retention_12mo", "bounds_hit"]] = [0.95, "Di"]  # one odd, bound-stuck fit
    assert set(cluster_wells(res, n_clusters=3)["cluster"]) >= {
        "Steep decline", "Moderate decline", "Shallow decline"}


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
    summ = cluster_summary(cluster_wells(_results(), n_clusters=3))
    assert summ["wells"].sum() == 15
    assert list(summ["cluster"]) == ["Steep decline", "Moderate decline", "Shallow decline"]


def test_k_chosen_by_silhouette():
    two = _results(((0.6, 0.4), (0.2, 0.1)))
    out = cluster_wells(two)
    assert out.loc[out["status"] == "ok", "cluster"].nunique() == 2
    k, scores = choose_k(two)
    assert k == 2 and scores[2] == max(scores.values())


def test_low_confidence_wells_do_not_move_boundaries():
    res = _results()
    res["low_confidence"] = False
    extreme = res.index[:2]
    clean = cluster_wells(res.drop(extreme), n_clusters=3)
    res.loc[extreme, ["retention_12mo", "retention_36mo", "low_confidence"]] = [
        [0.95, 0.9, True], [0.02, 0.005, True]]
    out = cluster_wells(res, n_clusters=3)
    kept = out[~out["api8"].isin(res.loc[extreme, "api8"])].set_index("api8")["cluster"]
    pd.testing.assert_series_equal(kept, clean.set_index("api8")["cluster"])
    assert (out["cluster"] != "Not fitted").sum() == len(res) - 1


def test_stability_is_high_for_well_separated_groups():
    out = cluster_wells(_results(((0.6, 0.4), (0.2, 0.1))))
    assert cluster_stability(out, n_boot=20) > 0.95
