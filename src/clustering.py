"""K-means clustering of wells by decline behavior.

Each well is represented by its fitted Arps parameters (annualized initial
decline Di and b-factor). Wells with similar decline shapes land in the same
cluster -- the "which wells behave alike" view that operators use to build
type curves and benchmarking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_LEVELS = {
    2: ["High", "Low"],
    3: ["High", "Mid", "Low"],
    4: ["High", "Mid-high", "Mid-low", "Low"],
    5: ["High", "Mid-high", "Mid", "Mid-low", "Low"],
}


# Clusters whose naming centroids are within this fraction of each other on an
# axis share that axis's level word.
SAME_LEVEL = 0.2


def _levels(k: int) -> list[str]:
    return _LEVELS.get(k, [f"#{i + 1}" for i in range(k)])


def _axis_words(values: np.ndarray) -> list[str] | None:
    """Level word per cluster on one axis (positive values), or None when all
    clusters sit at the same level and the axis does not separate them."""
    order = np.argsort(-values)
    tiers = np.zeros(len(values), dtype=int)
    for prev, cur in zip(order, order[1:]):
        step = (values[prev] - values[cur]) / values[prev] >= SAME_LEVEL
        tiers[cur] = tiers[prev] + int(step)
    n_tiers = tiers.max() + 1
    if n_tiers == 1:
        return None
    words = _levels(n_tiers)
    return [words[t] for t in tiers]


def _cluster_names(ok: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> dict:
    """Name clusters by the axes on which they actually differ.

    Naming centroids leave out wells with a parameter on a fit bound, so a
    single bound-stuck Di cannot make a cluster look 'High Di'.
    """
    clean = (ok["bounds_hit"].fillna("") == "").to_numpy() if "bounds_hit" in ok else \
        np.ones(len(ok), dtype=bool)
    cent = np.empty((n_clusters, 2))
    for k in range(n_clusters):
        members = labels == k
        use = members & clean if (members & clean).any() else members
        cent[k] = ok.loc[use, ["Di_annual", "b"]].mean().to_numpy()
    di, b = _axis_words(cent[:, 0]), _axis_words(cent[:, 1])
    names = {}
    for k in range(n_clusters):
        parts = [f"{words[k]} {axis}" for words, axis in ((di, "Di"), (b, "b")) if words]
        if not parts:
            names[k] = f"Cluster {k + 1}"
            continue
        names[k] = " · ".join([parts[0]] + [p[0].lower() + p[1:] for p in parts[1:]])
    if len(set(names.values())) < n_clusters:  # same level on both axes
        names = {k: f"{n} ({k + 1})" for k, n in names.items()}
    return names


def cluster_wells(results: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Return a copy of the per-well results table with cluster labels added.

    cluster_id 0 is the cluster with the highest centroid Di. Clusters are
    named by where their centroid sits on both axes, e.g. 'High Di · high b'
    (an axis is left out of the name when it does not separate the clusters):
    Di alone is misleading, because a high-Di well with a high b-factor
    flattens quickly and can out-produce a low-Di, near-exponential well.
    """
    results = results.drop(columns=["cluster_id", "cluster"], errors="ignore")
    ok = results[results["status"] == "ok"].copy()
    if len(ok) < n_clusters:
        return results.assign(cluster_id=np.nan, cluster="Insufficient data")

    X = ok[["Di_annual", "b"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(scaler.transform(X))
    centroids = scaler.inverse_transform(km.cluster_centers_)

    di_rank = np.argsort(np.argsort(-centroids[:, 0]))  # 0 = highest Di
    names = _cluster_names(ok, labels, n_clusters)

    ok["cluster_id"] = di_rank[labels]
    ok["cluster"] = [names[k] for k in labels]
    results = results.merge(
        ok[["api8", "cluster_id", "cluster"]], on="api8", how="left"
    )
    results["cluster"] = results["cluster"].fillna("Not fitted")
    return results


def cluster_order(results: pd.DataFrame) -> list[str]:
    """Cluster names ordered by cluster_id, for stable legends and colors."""
    ok = results.dropna(subset=["cluster_id"])
    return list(ok.sort_values("cluster_id")["cluster"].drop_duplicates())


def cluster_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster aggregates: well count, mean b, mean Di, mean/total EUR."""
    ok = results[results["status"] == "ok"]
    return (
        ok.groupby(["cluster_id", "cluster"])
        .agg(
            wells=("api8", "count"),
            mean_b=("b", "mean"),
            mean_Di_annual=("Di_annual", "mean"),
            mean_EUR_bbl=("eur_bbl", "mean"),
            total_EUR_bbl=("eur_bbl", "sum"),
            mean_EUR_per_1000ft=("eur_per_1000ft", "mean"),
            mean_R2=("r_squared", "mean"),
        )
        .reset_index()
        .drop(columns="cluster_id")
    )
