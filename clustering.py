"""
Clustering methods and cluster formation for splitting a CVRP into subproblems.

    cluster_count(num_customers, min_vehicles, sizing, value)
    make_clusters(coordinates, demands, n_clusters, method, seed=None)
    clustering_quality(coordinates, demands, clusters, seed=None)

Cluster count
    sizing="routes"     k = ceil(min_vehicles / value)     value = routes per cluster
    sizing="customers"  k = ceil(num_customers / value)    value = customers per cluster
    k is kept between 1 and the number of customers.

Methods:
    sweep            demand-aware angular sectors of equal demand (deterministic)
    kmeans           distance-based, ignores demand (random unless seeded)
    agglomerative    distance-based, ignores demand (deterministic)

Empty clusters are dropped, so the result can have fewer than k clusters.

clustering_quality measures the clusters before any routes exist.

Indices are 0-based and exclude the depot. coordinates[0] and demands[0]
belong to the depot.
"""

import math

import numpy as np
from scipy.spatial.distance import cdist
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

    Customers are sorted by polar angle around the depot, measured from the
    negative x-axis. The sorted sequence is cut into n_clusters contiguous
    sectors of about equal demand. Each customer is placed by the midpoint
    of its demand interval along the sweep, so a customer falls in the
    sector that contains the middle of its demand.

    If total demand is zero, the sweep falls back to equal customer counts.
    The method is deterministic; seed is accepted for interface compatibility.
    """

    depot = coordinates[0]
    customers = coordinates[1:]
    customer_demands = np.asarray(demands, dtype=float)[1:]

    offsets = customers - depot
    angles = np.mod(
        np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi,
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

        for sector, chunk in enumerate(np.array_split(np.arange(len(order)), n_clusters)):
            sector_of_position[chunk] = sector

    labels = np.empty(len(customers), dtype=int)
    labels[order] = sector_of_position

    return labels


def kmeans_clustering(coordinates, demands, n_clusters, seed=None):
    """Unweighted k-means on customer coordinates. Demand is not used."""

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=seed,
    )

    return model.fit_predict(coordinates[1:])


def agglomerative_clustering(coordinates, demands, n_clusters, seed=None):
    """
    Agglomerative clustering on customer coordinates. Demand is not used.
    Deterministic; seed is accepted for interface compatibility.
    """

    model = AgglomerativeClustering(n_clusters=n_clusters)

    return model.fit_predict(coordinates[1:])


METHODS = {
    "sweep": sweep_clustering,
    "kmeans": kmeans_clustering,
    "agglomerative": agglomerative_clustering,
}


# ============================================================
# PUBLIC API
# ============================================================

def cluster_count(num_customers, min_vehicles, sizing, value):
    """
    Number of clusters k.

        sizing="routes"     k = ceil(min_vehicles / value)
        sizing="customers"  k = ceil(num_customers / value)

    min_vehicles is the lower bound ceil(total demand / capacity), so the
    real number of routes in a cluster can be somewhat above value.
    k is kept between 1 and num_customers.
    """

    if value < 1:
        raise ValueError(f"{sizing} size must be at least 1, got {value}.")

    if sizing == "routes":
        needed = math.ceil(min_vehicles / value)

    elif sizing == "customers":
        needed = math.ceil(num_customers / value)

    else:
        raise ValueError(f"sizing must be 'routes' or 'customers', got '{sizing}'.")

    return max(1, min(num_customers, needed))


def make_clusters(coordinates, demands, n_clusters, method, seed=None):
    """
    Cluster the customers into n_clusters clusters with the given method.

    Returns a list of clusters, each a list of 0-based customer indices
    (depot excluded). Empty clusters are dropped.
    """

    if method not in METHODS:
        raise ValueError(
            f"Unknown clustering method '{method}'. "
            f"Available: {list(METHODS)}"
        )

    num_customers = len(coordinates) - 1

    if not 1 <= n_clusters <= num_customers:
        raise ValueError(
            f"n_clusters must be between 1 and {num_customers}, got {n_clusters}."
        )

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

    return round(float(score), 4)


def clustering_quality(coordinates, demands, clusters, seed=None):
    """
    Summary of the clustering before any routes exist. Returns a dict with:

        demand_cv                spread of cluster demands (std / mean)
        mean_customer_radius     mean distance from a customer to its cluster centroid
        nearest_centroid_share   share of customers whose own centroid is the nearest one
        silhouette               sampled silhouette score, or None

    seed only fixes the silhouette sample; None is treated as 0.
    """

    seed = 0 if seed is None else seed

    points = np.asarray(coordinates, dtype=float)[1:]
    customer_demands = np.asarray(demands)[1:]

    labels = np.empty(len(points), dtype=int)

    for label, cluster in enumerate(clusters):
        labels[np.asarray(cluster)] = label

    centroids = np.array([
        points[np.asarray(cluster)].mean(axis=0)
        for cluster in clusters
    ])

    cluster_demands = [
        int(customer_demands[np.asarray(cluster)].sum())
        for cluster in clusters
    ]

    radii = np.linalg.norm(points - centroids[labels], axis=1)

    nearest_share = float(np.mean(np.argmin(cdist(points, centroids), axis=1) == labels))

    return {
        "demand_cv": round(_coefficient_of_variation(cluster_demands), 3),
        "mean_customer_radius": round(float(radii.mean()), 1),
        "nearest_centroid_share": round(nearest_share, 4),
        "silhouette": _silhouette(points, labels, seed),
    }