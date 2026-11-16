"""
Cluster formation for splitting a CVRP into subproblems.

    make_clusters(coordinates, demands, target_customers, method, seed=None)

    1. The number of clusters is
           k = ceil(n / target_customers)
       limited to between 1 and the number of customers n.
    2. The chosen method clusters all customers into k clusters.

target_customers is the number of customers one cluster should hold, m.

Methods:
    sweep            demand-aware: contiguous angular sectors of equal demand
    kmeans           distance-based, ignores demand
    kmeans_weighted  distance-based, each customer weighted by its demand
    agglomerative    distance-based, ignores demand

Empty clusters are dropped, so the result can have fewer than k clusters.

clustering_quality measures the clusters before any routes exist.

Indices are 0-based and exclude the depot. coordinates[0] and demands[0]
belong to the depot.
"""

import math

import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CLUSTERING METHODS
# ============================================================
#
# A method takes the full coordinate array (depot in row 0), the full
# demand array (depot in position 0), a number of clusters and a seed.
# It returns one cluster label per customer. Register new methods in
# METHODS.
# ============================================================

def sweep_clustering(coordinates, demands, n_clusters, seed=None):
    """
    Demand-aware angular sweep around the depot.

    Customers are sorted by polar angle around the depot, measured from a
    start angle. The start angle is drawn from seed, so different seeds
    cut the circle at different places. seed=None uses the unshifted
    order (angles starting just after the negative x-axis).

    The sorted sequence is cut into n_clusters contiguous sectors of about
    equal demand. Each customer is placed by the midpoint of its demand
    interval along the sweep, so a customer falls in the sector that
    contains the middle of its demand.

    If total demand is zero, the sweep falls back to equal customer counts.
    """

    depot = coordinates[0]
    customers = coordinates[1:]
    customer_demands = np.asarray(demands, dtype=float)[1:]

    if seed is None:
        shift = 0.0
    else:
        shift = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi)

    offsets = customers - depot
    angles = np.mod(
        np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi - shift,
        2.0 * np.pi,
    )
    order = np.argsort(angles)

    ordered_demands = customer_demands[order]
    total = ordered_demands.sum()

    if total > 0:

        cumulative = np.cumsum(ordered_demands)

        midpoints = cumulative - ordered_demands / 2

        thresholds = total * np.arange(1, n_clusters) / n_clusters

        sector_of_position = np.searchsorted(thresholds, midpoints, side="right")

    else:

        sector_of_position = np.empty(len(order), dtype=int)

        positions = np.array_split(np.arange(len(order)), n_clusters)

        for sector, chunk in enumerate(positions):
            sector_of_position[chunk] = sector

    labels = np.empty(len(customers), dtype=int)
    labels[order] = sector_of_position

    return labels


def kmeans_clustering(coordinates, demands, n_clusters, seed=42):
    """Unweighted k-means. Demand is not used."""

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=seed,
    )

    return model.fit_predict(coordinates[1:])


def kmeans_weighted_clustering(coordinates, demands, n_clusters, seed=42):
    """
    Demand-weighted k-means. Each customer pulls its centroid in proportion
    to its demand, so centroids sit where the load is rather than where the
    headcount is. Customers with zero demand get weight 1 so they still count.
    """

    weights = np.maximum(np.asarray(demands, dtype=float)[1:], 1.0)

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=seed,
    )

    return model.fit_predict(coordinates[1:], sample_weight=weights)


def agglomerative_clustering(coordinates, demands, n_clusters, seed=None):
    """Demand is not used. Deterministic, so seed is unused."""

    model = AgglomerativeClustering(n_clusters=n_clusters)

    return model.fit_predict(coordinates[1:])


METHODS = {
    "sweep": sweep_clustering,
    "kmeans": kmeans_clustering,
    "kmeans_weighted": kmeans_weighted_clustering,
    "agglomerative": agglomerative_clustering,
}


# ============================================================
# PUBLIC API
# ============================================================

def cluster_count(num_customers, target_customers):
    """ceil(num_customers / target_customers), kept between 1 and num_customers."""

    if target_customers < 1:
        raise ValueError(f"target_customers must be at least 1, got {target_customers}.")

    return max(1, min(num_customers, math.ceil(num_customers / target_customers)))


def make_clusters(coordinates, demands, target_customers, method, seed=None):
    """
    Cluster the customers into cluster_count(...) clusters.

    Returns a list of clusters, each a list of 0-based customer indices
    (depot excluded). Empty clusters are dropped.
    """

    if method not in METHODS:
        raise ValueError(
            f"Unknown clustering method '{method}'. "
            f"Available: {list(METHODS)}"
        )

    num_customers = len(coordinates) - 1

    n_clusters = cluster_count(num_customers, target_customers)

    labels = METHODS[method](coordinates, demands, n_clusters, seed)

    clusters = [
        np.flatnonzero(labels == label).tolist()
        for label in range(n_clusters)
    ]

    return [cluster for cluster in clusters if cluster]


# ============================================================
# CLUSTER QUALITY (before solving)
# ============================================================

SILHOUETTE_SAMPLE_SIZE = 3000


def _coefficient_of_variation(values):

    values = np.asarray(values, dtype=float)

    mean = values.mean()

    return float(values.std() / mean) if mean > 0 else 0.0


def _silhouette(points, labels, seed):
    """Sampled silhouette score, or None if it cannot be computed."""

    if len(np.unique(labels)) < 2:
        return None

    try:
        score = silhouette_score(
            points,
            labels,
            sample_size=min(len(points), SILHOUETTE_SAMPLE_SIZE),
            random_state=seed,
        )
    except ValueError:
        return None

    return float(score)


def _round(value, digits):

    return None if value is None else round(value, digits)


def clustering_quality(coordinates, demands, clusters, target_customers, seed=0):
    """
    Quality of the clustering, before any routes exist.

    Returns (summary, per_cluster): summary is a dict of overall values,
    per_cluster has one dict per cluster.
    """

    points = np.asarray(coordinates, dtype=float)[1:]
    customer_demands = np.asarray(demands)[1:]

    labels = np.empty(len(points), dtype=int)

    for label, cluster in enumerate(clusters):
        labels[np.asarray(cluster)] = label

    centroids = np.array([
        points[np.asarray(cluster)].mean(axis=0)
        for cluster in clusters
    ])

    per_cluster = []
    all_radii = []
    cluster_demands = []

    for label, cluster in enumerate(clusters):

        idx = np.asarray(cluster)

        demand = int(customer_demands[idx].sum())
        radii = np.linalg.norm(points[idx] - centroids[label], axis=1)

        cluster_demands.append(demand)
        all_radii.append(radii)

        per_cluster.append({
            "cluster": label + 1,
            "customers": len(idx),
            "customer_ratio": round(len(idx) / target_customers, 3),
            "demand": demand,
            "mean_radius": round(float(radii.mean()), 1),
            "max_radius": round(float(radii.max()), 1),
        })

    distances = np.linalg.norm(
        points[:, None, :] - centroids[None, :, :],
        axis=2,
    )

    nearest_share = float(np.mean(np.argmin(distances, axis=1) == labels))

    sizes = [row["customers"] for row in per_cluster]

    summary = {
        "n_clusters": len(clusters),
        "customers_min": min(sizes),
        "customers_max": max(sizes),
        "demand_min": min(cluster_demands),
        "demand_max": max(cluster_demands),
        "demand_cv": round(_coefficient_of_variation(cluster_demands), 3),
        "mean_customer_radius": round(float(np.concatenate(all_radii).mean()), 1),
        "nearest_centroid_share": round(nearest_share, 4),
        "silhouette": _round(_silhouette(points, labels, seed), 4),
    }

    return summary, per_cluster