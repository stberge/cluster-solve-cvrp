"""
Route solving with PyVRP.

    solve_cluster            one cluster as its own CVRP
    solve_clusters           every cluster, each with its own time limit
    solve_whole_instance     the full instance in one run

Times are runtime limits in seconds. Distances are rounded to whole
numbers, so solver totals match the route statistics in instance_io.
"""

import numpy as np

from pyvrp import Client, Depot, Location, ProblemData, VehicleType, read, solve
from pyvrp.stop import MaxRuntime


# ============================================================
# HELPERS
# ============================================================

def _client_indices(route):
    """Client indices (0-based) of a route in visit order. Depot stops are skipped."""

    return [
        stop.idx
        for stop in route
        if not stop.is_depot()
    ]


def _rounded_distance_matrix(points):

    diff = points[:, None, :] - points[None, :, :]

    return np.round(
        np.sqrt((diff ** 2).sum(axis=2))
    ).astype(np.int64)


def _vehicle_count(total_demand, capacity):
    """Vehicles needed for the demand, with slack for packing losses."""

    needed = int(np.ceil(total_demand / capacity))

    return int(np.ceil(needed * 1.2)) + 2


# ============================================================
# SOLVE ONE CLUSTER
# ============================================================

def solve_cluster(
    depot_coord,
    customer_coords,
    customer_demands,
    original_indices,
    capacity,
    time_limit,
    seed,
):
    """
    Solve one cluster as a multi-vehicle CVRP (depot -> customers -> depot).

    Returns (routes, distance). Routes are lists of ORIGINAL 0-based
    customer indices in visit order. Distance is the total over this
    cluster's routes.
    """

    if len(customer_coords) == 0:
        return [], 0

    # Location 0 is the depot. Client i (0-based) sits at location i + 1.
    points = np.vstack([depot_coord, customer_coords]).astype(float)

    dist = _rounded_distance_matrix(points)

    vehicles = _vehicle_count(sum(customer_demands), capacity)

    data = ProblemData(
        locations=[Location(float(x), float(y)) for x, y in points],
        clients=[
            Client(location=i + 1, delivery=[int(demand)])
            for i, demand in enumerate(customer_demands)
        ],
        depots=[Depot(location=0)],
        vehicle_types=[
            VehicleType(num_available=vehicles, capacity=[int(capacity)])
        ],
        distance_matrices=[dist],
        duration_matrices=[np.zeros_like(dist)],
    )

    result = solve(
        data,
        stop=MaxRuntime(time_limit),
        seed=seed,
        display=False,
    )

    if not result.is_feasible():
        raise RuntimeError(
            f"No feasible solution for a cluster with {len(customer_coords)} "
            f"customers (demand {sum(customer_demands)}, capacity {capacity}, "
            f"vehicles available {vehicles})."
        )

    routes = []
    distance = 0

    for route in result.best.routes():

        indices = [
            original_indices[client]
            for client in _client_indices(route)
        ]

        if indices:
            routes.append(indices)

        distance += int(route.distance())

    return routes, distance


# ============================================================
# SOLVE ALL CLUSTERS
# ============================================================

def solve_clusters(coordinates, demands, clusters, capacity, time_limits, seed):
    """
    Solve every cluster as its own subproblem.

    time_limits holds one runtime (seconds) per cluster, in the same
    order as clusters.

    Returns (routes, route_clusters, total_distance).
    """

    if len(time_limits) != len(clusters):
        raise ValueError("Need exactly one time limit per cluster.")

    depot = coordinates[0]

    routes = []
    route_clusters = []
    total_distance = 0

    for cluster_id, (cluster, time_limit) in enumerate(zip(clusters, time_limits)):

        idx = np.asarray(cluster) + 1

        cluster_routes, distance = solve_cluster(
            depot,
            coordinates[idx],
            demands[idx],
            cluster,
            capacity,
            time_limit,
            seed,
        )

        routes.extend(cluster_routes)
        route_clusters.extend([cluster_id] * len(cluster_routes))
        total_distance += distance

    return routes, route_clusters, total_distance


# ============================================================
# WHOLE INSTANCE
# ============================================================

def solve_whole_instance(instance_path, time_limit, seed):
    """
    Solve the full CVRP in one run. time_limit is the total runtime (seconds).

    Returns a dict with routes (0-based customer indices), distance,
    runtime (seconds) and feasible.
    """

    # round_func="round" rounds distances to the nearest integer, matching
    # the clustered pipeline. The default leaves floats, which the solver
    # truncates, so the two methods would optimise different distances.
    instance = read(instance_path, round_func="round")

    result = solve(
        instance,
        stop=MaxRuntime(time_limit),
        seed=seed,
        display=False,
    )

    routes = [
        indices
        for indices in (_client_indices(route) for route in result.best.routes())
        if indices
    ]

    return {
        "routes": routes,
        "distance": result.cost(),
        "runtime": result.runtime,
        "feasible": result.is_feasible(),
    }