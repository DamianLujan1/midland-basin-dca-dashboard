"""K-means clustering of wells by decline shape.

Each well is represented by how much of its peak rate the fitted decline keeps
after 12 and after 36 producing months (log of q(t)/qi). That is the decline
shape that matters for type curves and benchmarking, and unlike Di or b on
their own it is well determined by the data: a fit can trade Di against b
without changing the curve much, which makes clusters on (Di, b) unstable.

The model is fitted on wells whose fits are not low-confidence; low-confidence
wells are then assigned to the nearest cluster, so a few poor fits cannot move
the cluster boundaries. Unless a number of clusters is given, k is chosen from
K_RANGE by silhouette score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.arps import hyperbolic_rate

K_RANGE = (2, 3, 4)
RETENTION_MONTHS = (12, 36)
FEATURES = [f"retention_{m}mo" for m in RETENTION_MONTHS]

# Level words for retained rate, highest retention first.
_LEVELS = {
    2: ["Shallow", "Steep"],
    3: ["Shallow", "Moderate", "Steep"],
    4: ["Shallow", "Moderately shallow", "Moderately steep", "Steep"],
    5: ["Shallow", "Moderately shallow", "Moderate", "Moderately steep", "Steep"],
}

# Clusters whose naming centroids are within this fraction of each other on an
# axis share that axis's level word.
SAME_LEVEL = 0.2


def retention(qi, Di, b, months: int):
    """Fraction of the peak-month rate the fitted hyperbolic keeps after `months`
    producing months."""
    return hyperbolic_rate(np.full(np.shape(qi), float(months)), qi, Di, b) / qi


def _levels(k: int) -> list[str]:
    return _LEVELS.get(k, [f"#{i + 1}" for i in range(k)])


def _features(df: pd.DataFrame) -> np.ndarray:
    return np.log(df[FEATURES].to_numpy(float))


def _fit_mask(ok: pd.DataFrame) -> np.ndarray:
    """Wells the model is fitted on: not low-confidence (all, if too few)."""
    if "low_confidence" not in ok:
        return np.ones(len(ok), dtype=bool)
    mask = ~ok["low_confidence"].astype(bool).to_numpy()
    return mask if mask.sum() > max(K_RANGE) else np.ones(len(ok), dtype=bool)


def _kmeans(X_fit: np.ndarray, k: int, seed: int = 42):
    scaler = StandardScaler().fit(X_fit)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(X_fit))
    return scaler, km


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


def _naming_centroids(ok: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Mean retention per cluster, leaving out wells with a parameter on a fit
    bound so a single bound-stuck fit cannot set a cluster's name."""
    clean = (ok["bounds_hit"].fillna("") == "").to_numpy() if "bounds_hit" in ok else \
        np.ones(len(ok), dtype=bool)
    cent = np.empty((n_clusters, len(FEATURES)))
    for k in range(n_clusters):
        members = labels == k
        use = members & clean if (members & clean).any() else members
        cent[k] = ok.loc[use, FEATURES].mean().to_numpy()
    return cent


def _cluster_names(cent: np.ndarray) -> dict:
    """'Steep decline' when both horizons agree, 'Steep year 1, moderate after'
    when they differ; a horizon that does not separate the clusters is left out."""
    n_clusters = len(cent)
    early, late = _axis_words(cent[:, 0]), _axis_words(cent[:, 1])
    names = {}
    for k in range(n_clusters):
        if early and late and early[k] != late[k]:
            names[k] = f"{early[k]} year 1, {late[k].lower()} after"
        elif early or late:
            names[k] = f"{(early or late)[k]} decline"
        else:
            names[k] = f"Cluster {k + 1}"
    if len(set(names.values())) < n_clusters:  # same level on both horizons
        names = {k: f"{n} ({k + 1})" for k, n in names.items()}
    return names


def choose_k(results: pd.DataFrame) -> tuple[int, dict]:
    """k in K_RANGE with the best silhouette on the fit set, plus all scores."""
    ok = results[results["status"] == "ok"]
    X_fit = _features(ok)[_fit_mask(ok)]
    scores = {}
    for k in K_RANGE:
        if k < len(X_fit):
            scaler, km = _kmeans(X_fit, k)
            scores[k] = float(silhouette_score(scaler.transform(X_fit), km.labels_))
    return max(scores, key=scores.get), scores


def cluster_wells(results: pd.DataFrame, n_clusters: int | None = None) -> pd.DataFrame:
    """Return a copy of the per-well results table with cluster labels added.

    Needs the FEATURES columns (see retention). n_clusters=None picks k by
    silhouette (see choose_k). cluster_id 0 is the steepest cluster (least
    rate kept after one year).
    """
    results = results.drop(columns=["cluster_id", "cluster"], errors="ignore")
    ok = results[results["status"] == "ok"].copy()
    min_k = n_clusters or min(K_RANGE)
    if len(ok) <= min_k:
        return results.assign(cluster_id=np.nan, cluster="Insufficient data")
    if n_clusters is None:
        n_clusters, _ = choose_k(results)

    X = _features(ok)
    scaler, km = _kmeans(X[_fit_mask(ok)], n_clusters)
    labels = km.predict(scaler.transform(X))
    cent = _naming_centroids(ok, labels, n_clusters)
    names = _cluster_names(cent)
    rank = np.argsort(np.argsort(cent[:, 0]))  # 0 = least rate kept after a year

    ok["cluster_id"] = rank[labels]
    ok["cluster"] = [names[k] for k in labels]
    results = results.merge(
        ok[["api8", "cluster_id", "cluster"]], on="api8", how="left"
    )
    results["cluster"] = results["cluster"].fillna("Not fitted")
    return results


def cluster_stability(results: pd.DataFrame, n_boot: int = 100) -> float:
    """Mean adjusted Rand index between the clustering in `results` and the
    clusterings bootstrap-resampled fit sets give for all fitted wells
    (1 = identical every time; below ~0.8 the partition is not robust)."""
    ok = results[results["status"] == "ok"]
    labels = ok["cluster_id"].to_numpy()
    k = int(np.nanmax(labels)) + 1
    X = _features(ok)
    fit_idx = np.flatnonzero(_fit_mask(ok))
    rng = np.random.default_rng(0)
    aris = []
    for i in range(n_boot):
        boot = rng.choice(fit_idx, len(fit_idx), replace=True)
        scaler, km = _kmeans(X[boot], k, seed=i)
        aris.append(adjusted_rand_score(labels, km.predict(scaler.transform(X))))
    return float(np.mean(aris))


def cluster_order(results: pd.DataFrame) -> list[str]:
    """Cluster names ordered by cluster_id, for stable legends and colors."""
    ok = results.dropna(subset=["cluster_id"])
    return list(ok.sort_values("cluster_id")["cluster"].drop_duplicates())


def cluster_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster aggregates: well count, retention, mean b and Di, EUR."""
    ok = results[results["status"] == "ok"]
    return (
        ok.groupby(["cluster_id", "cluster"])
        .agg(
            wells=("api8", "count"),
            mean_retention_12mo=("retention_12mo", "mean"),
            mean_retention_36mo=("retention_36mo", "mean"),
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
