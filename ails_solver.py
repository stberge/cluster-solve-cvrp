"""
Route solving with AILS-II, through the ailsii Python package.

Same interface as the PyVRP version (ILS.py):

    solve_cluster            every cluster, each with its own time limit
    solve_whole_instance     the full instance in one run

Distances and loads are recomputed in Python with route_stats, so the
totals match the route files whatever the solver reports.
"""

import time

import numpy as np

import ailsii

from instance_io import read_vrp, route_stats


# ============================================================
# SETTINGS
# ============================================================

# AILS-II parameters (defaults from the README)
AILS_PARAMS = dict(varphi=40, gamma=30, dMax=30, dMin=15)


# ============================================================
# HELPERS
# ============================================================

def _solve(points, demands, capacity, time_limit):
    """
    Run AILS-II on one CVRP. points and demands include the depot at index 0.
    Returns (routes as 0-based customer indices, solver cost).
    """

    if demands[0] != 0:
        raise ValueError("The depot demand must be 0.")

    data = {
        "x_coordinates": points[:, 0].tolist(),
        "y_coordinates": points[:, 1].tolist(),
        "demands": [int(d) for d in demands],
        "vehicle_capacity": int(capacity),
    }

    params = ailsii.AlgorithmParameters(
        timeLimit=float(time_limit),
        stoppingCriterion="Time",
        rounded=True,
        **AILS_PARAMS,
    )

    result = ailsii.Solver(parameters=params, verbose=False).solve_cvrp(data)

    # The solver numbers the depot 0 and customers 1..n,
    # so customer k (0-based) is index k + 1.
    routes = [[node - 1 for node in route] for route in result.routes]

    routes = [route for route in routes if route]

    return routes, result.cost


def _check_solution(routes, demands, capacity):
    """Every customer visited exactly once, and no route over capacity."""

    n_customers = len(demands) - 1

    visited = sorted(i for route in routes for i in route)

    if visited != list(range(n_customers)):
        raise RuntimeError(
            f"AILS-II visited {len(visited)} customer slots for "
            f"{n_customers} customers."
        )

    for route in routes:

        load = int(sum(demands[i + 1] for i in route))

        if load > capacity:
            raise RuntimeError(f"Route load {load} exceeds capacity {capacity}.")


def _warn_if_different(solver_cost, distance):
    """The solver's cost and the recomputed distance should match."""

    if abs(solver_cost - distance) > 0.5:
        print(
            f"WARNING: solver cost {solver_cost} differs from "
            f"recomputed distance {distance}"
        )


# ============================================================
# SOLVE ONE CLUSTER (private)
# ============================================================

def _solve_one_cluster(
    depot_coord,
    customer_coords,
    customer_demands,
    original_indices,
    capacity,
    time_limit,
):
    """
    Solve one cluster as a multi-vehicle CVRP.

    Returns (routes, distance). Routes are lists of ORIGINAL 0-based
    customer indices in visit order.
    """

    if len(customer_coords) == 0:
        return [], 0

    points = np.vstack([depot_coord, customer_coords]).astype(float)

    demands = np.concatenate([[0], np.asarray(customer_demands, dtype=int)])

    local_routes, solver_cost = _solve(points, demands, capacity, time_limit)

    _check_solution(local_routes, demands, capacity)

    distance = sum(s["distance"] for s in route_stats(points, demands, local_routes))

    _warn_if_different(solver_cost, distance)

    routes = [[original_indices[i] for i in route] for route in local_routes]

    return routes, distance


# ============================================================
# SOLVE ALL CLUSTERS
# ============================================================

def solve_cluster(coordinates, demands, clusters, capacity, time_limits, seed=None):
    """
    Solve every cluster as its own subproblem.

    time_limits holds one runtime (seconds) per cluster, in the same
    order as clusters. seed is accepted for interface compatibility;
    AILS-II has no seed option.

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

        cluster_routes, distance = _solve_one_cluster(
            depot,
            coordinates[idx],
            demands[idx],
            cluster,
            capacity,
            time_limit,
        )

        routes.extend(cluster_routes)
        route_clusters.extend([cluster_id] * len(cluster_routes))
        total_distance += distance

    return routes, route_clusters, total_distance


# ============================================================
# WHOLE INSTANCE
# ============================================================

def solve_whole_instance(instance_path, time_limit, seed=None):
    """
    Solve the full CVRP in one run. time_limit is the total runtime (seconds).

    Returns a dict with routes (0-based customer indices), distance,
    runtime (seconds) and feasible.
    """

    coordinates, demands, capacity = read_vrp(instance_path)

    started = time.perf_counter()

    routes, solver_cost = _solve(coordinates, demands, capacity, time_limit)

    runtime = time.perf_counter() - started

    _check_solution(routes, demands, capacity)

    distance = sum(s["distance"] for s in route_stats(coordinates, demands, routes))

    _warn_if_different(solver_cost, distance)

    return {
        "routes": routes,
        "distance": distance,
        "runtime": runtime,
        "feasible": True,
    }