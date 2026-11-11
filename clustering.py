"""
Clustering algorithms for splitting a CVRP into subproblems.

Every algorithm has the same signature:

    algorithm(coordinates, demands, capacity, n_clusters, seed)

and returns a list of clusters, each a list of customer indices
(0-indexed, depot excluded). Empty clusters are dropped, so the
number of returned clusters can be lower than n_clusters.

DBSCAN has been removed for now.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


# ============================================================
# HELPER
# ============================================================

def _labels_to_clusters(labels):
    """
    Turn a label array (one label per customer) into a list of
    clusters, each a list of customer indices. Non-empty labels only.
    """

    clusters = []

    for label in sorted(set(labels)):

        cluster = np.where(
            labels == label
        )[0].tolist()

        clusters.append(
            cluster
        )

    return clusters


# ============================================================
# SWEEP
# ============================================================

def sweep_clustering(
    coordinates,
    demands,
    capacity,
    n_clusters,
    seed=None
):
    """
    Angular sweep around the depot.

    Customers are sorted by polar angle around the depot, then the
    sorted sequence is cut into n_clusters contiguous sectors of
    (approximately) equal total demand. Each sector is one cluster.

    The sweep is deterministic, so seed is unused.
    """

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    customers = coordinates[1:]
    customer_demands = demands[1:]

    depot = coordinates[0]

    # --------------------------------------------------------
    # Polar angle of every customer relative to the depot
    # --------------------------------------------------------

    dx = customers[:, 0] - depot[0]
    dy = customers[:, 1] - depot[1]

    angles = np.arctan2(dy, dx)

    order = np.argsort(angles)

    # --------------------------------------------------------
    # Cut the angular sequence into n_clusters demand-balanced
    # sectors. Each customer is placed by the midpoint of its
    # demand slice along the sweep, so a large customer falls
    # into the sector where most of its demand lies.
    # --------------------------------------------------------

    sorted_demands = customer_demands[order].astype(float)

    total_demand = sorted_demands.sum()

    cumulative_mid = np.cumsum(sorted_demands) - sorted_demands / 2

    sorted_labels = np.floor(
        cumulative_mid / total_demand * n_clusters
    ).astype(int)

    # Guard against the last customer landing exactly on n_clusters
    sorted_labels = np.minimum(sorted_labels, n_clusters - 1)

    labels = np.empty(len(customers), dtype=int)
    labels[order] = sorted_labels

    return _labels_to_clusters(labels)


# ============================================================
# K-MEANS
# ============================================================

def kmeans_clustering(
    coordinates,
    demands,
    capacity,
    n_clusters,
    seed=42
):

    customers = coordinates[1:]

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=seed
    )

    labels = model.fit_predict(
        customers
    )

    return _labels_to_clusters(labels)


# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================

def agglomerative_clustering(
    coordinates,
    demands,
    capacity,
    n_clusters,
    seed=None
):

    customers = coordinates[1:]

    # Agglomerative clustering is deterministic, so seed is unused
    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(
        customers
    )

    return _labels_to_clusters(labels)


# ============================================================
# REGISTRY
# ============================================================

ALGORITHM_FUNCTIONS = {
    "sweep": sweep_clustering,
    "kmeans": kmeans_clustering,
    "agglomerative": agglomerative_clustering,
}
