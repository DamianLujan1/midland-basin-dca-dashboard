"""K-means clustering of wells by decline behavior.

Each well is represented by its fitted Arps parameters (annualized initial
decline Di and b-factor). Wells with similar decline shapes land in the same
cluster -- the "which wells behave alike" view that operators use to build
type curves and benchmark performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_wells(results: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Add cluster labels to the per-well results table.

    Clusters are named by decline steepness from their centroids:
    'Steep decline', 'Moderate decline', 'Mild decline'.
    """
    ok = results[results["status"] == "ok"].copy()
    if len(ok) < n_clusters:
        results["cluster"] = "Insufficient data"
        return results

    X = ok[["Di_annual", "b"]].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    centroids = StandardScaler().fit(X).inverse_transform(km.cluster_centers_)

    # Order clusters by centroid Di (steepest first) for stable naming.
    order = np.argsort(-centroids[:, 0])
    names = ["Steep decline", "Moderate decline", "Mild decline",
             "Cluster 4", "Cluster 5"][:n_clusters]
    label_to_name = {int(order[i]): names[i] for i in range(n_clusters)}

    ok["cluster_id"] = labels
    ok["cluster"] = ok["cluster_id"].map(label_to_name)
    results = results.merge(
        ok[["api8", "cluster_id", "cluster"]], on="api8", how="left"
    )
    results["cluster"] = results["cluster"].fillna("Not fitted")
    return results


def cluster_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster aggregates: well count, mean b, mean Di, mean/total EUR."""
    ok = results[results["status"] == "ok"]
    return (
        ok.groupby("cluster")
        .agg(
            wells=("api8", "count"),
            mean_b=("b", "mean"),
            mean_Di_annual=("Di_annual", "mean"),
            mean_EUR_bbl=("eur_bbl", "mean"),
            total_EUR_bbl=("eur_bbl", "sum"),
            mean_R2=("r_squared", "mean"),
        )
        .round(3)
        .reset_index()
    )
