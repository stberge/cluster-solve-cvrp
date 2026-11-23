"""
AILS-II routing through the ailsii package.

    solve_cluster(...)        every cluster solved on its own
    solve_whole_instance(...) the full instance solved in one call

Each AILS-II call gets a whole-second timeLimit taken from the run's time
share. The budget is never changed by the measurements. Each call's
elapsed time is measured, and the overhead (elapsed minus timeLimit) is
recorded so it can be corrected for later.

Distances are recomputed with route_stats, so reported totals match the
route files. The solver's own cost is kept and compared as a check.
"""

import time

import numpy as np
import ailsii

from instance_io import route_stats


# ============================================================
# SETTINGS
# ============================================================

# AILS-II parameters (README defaults). Stored with every run.
AILS_PARAMS = dict(varphi=40, gamma=30, dMax=30, dMin=15)

# Smallest timeLimit (s) allowed for one cluster.
MIN_LIMIT = 5

# Guard: clusters above this size are rejected. Placeholder value, set it to
# the size you have validated.
MAX_CLUSTER_CUSTOMERS = 3000

PRINT_SOLVER_CALLS = True


# ============================================================
# TIME LIMITS
# ============================================================

def integer_limits(shares):
    """
    Whole-second timeLimit per share. The limits sum to the floor of the
    total share, so the requested search time never exceeds the budget.
    """

    shares = np.asarray(shares, dtype=float)

    if np.any(shares < MIN_LIMIT):
        worst = int(np.argmin(shares))
        raise ValueError(
            f"Share {worst} is {shares[worst]:.2f} s, below the minimum of "
            f"{MIN_LIMIT} s. Use fewer clusters or a larger TOTAL_TIME."
        )

    limits = np.floor(shares).astype(int)

    spare = int(np.floor(shares.sum() + 1e-9)) - int(limits.sum())

    order = np.argsort(-(shares - limits), kind="stable")
    limits[order[:spare]] += 1

    return limits.tolist()


# ============================================================
# HELPERS
# ============================================================

def _solve(points, demands, capacity, time_limit):
    """
    One AILS-II call. points and demands include the depot at index 0.

    Returns routes (0-based customer indices), the solver's cost, the
    timeLimit that was set, the measured elapsed time of the call, and
    the overhead (elapsed minus timeLimit).
    """

    if demands[0] != 0:
        raise ValueError("The depot demand must be 0.")

    data = {
        "x_coordinates": points[:, 0].tolist(),
        "y_coordinates": points[:, 1].tolist(),
        "demands": [int(d) for d in demands],
        "vehicle_capacity": int(capacity),
    }

    started = time.perf_counter()

    params = ailsii.AlgorithmParameters(
        timeLimit=float(time_limit),
        stoppingCriterion="Time",
        rounded=True,
        **AILS_PARAMS,
    )

    result = ailsii.Solver(parameters=params, verbose=False).solve_cvrp(data)

    elapsed = time.perf_counter() - started

    # The solver numbers the depot 0 and customers 1..n,
    # so customer k (0-based) is index k + 1.
    routes = [[node - 1 for node in route] for route in result.routes]
    routes = [route for route in routes if route]

    return {
        "routes": routes,
        "solver_cost": float(result.cost),
        "limit": int(time_limit),
        "elapsed": round(elapsed, 3),
        "overhead": round(elapsed - time_limit, 3),
    }


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


def _cost_ok(solver_cost, distance):
    """True if the solver's cost matches the recomputed distance."""

    ok = abs(solver_cost - distance) <= 0.5

    if not ok:
        print(
            f"WARNING: solver cost {solver_cost} differs from "
            f"recomputed distance {distance}"
        )

    return ok


def _print_call(call):

    if PRINT_SOLVER_CALLS:
        print(
            f"          solver call: timeLimit {call['limit']} s, "
            f"elapsed {call['elapsed']:.2f} s, overhead {call['overhead']:.2f} s"
        )


def _summary(routes, route_clusters, distance, limits, calls):
    """Common result format for both solve functions."""

    overheads = [call["overhead"] for call in calls]

    return {
        "routes": routes,
        "route_clusters": route_clusters,
        "distance": int(distance),
        "limits": limits,
        "n_calls": len(calls),
        "search_s": int(sum(limits)),
        "solver_elapsed_s": round(sum(call["elapsed"] for call in calls), 3),
        "setup_s": round(sum(overheads), 3),
        "setup_max_s": round(max(overheads), 3),
        "cost_ok": all(call["cost_ok"] for call in calls),
    }


# ============================================================
# SOLVE ONE CLUSTER (private)
# ============================================================

def _solve_one_cluster(
    depot_coord,
    customer_coords,
    customer_demands,
    original_indices,
    capacity,
    limit,
):
    """
    Solve one cluster as a multi-vehicle CVRP.

    Returns (routes, distance, call). Routes use ORIGINAL 0-based customer
    indices.
    """

    points = np.vstack([depot_coord, customer_coords]).astype(float)

    demands = np.concatenate([[0], np.asarray(customer_demands, dtype=int)])

    call = _solve(points, demands, capacity, limit)

    _print_call(call)

    _check_solution(call["routes"], demands, capacity)

    distance = sum(s["distance"] for s in route_stats(points, demands, call["routes"]))

    call["cost_ok"] = _cost_ok(call["solver_cost"], distance)

    routes = [[original_indices[i] for i in route] for route in call["routes"]]

    return routes, distance, call


# ============================================================
# SOLVE ALL CLUSTERS
# ============================================================

def solve_cluster(coordinates, demands, clusters, capacity, time_limits):
    """
    Solve every cluster as its own subproblem.

    time_limits holds one search share (seconds) per cluster, in the same
    order as clusters. integer_limits converts them to whole seconds.
    """

    if len(time_limits) != len(clusters):
        raise ValueError("Need exactly one time share per cluster.")

    largest = max(len(cluster) for cluster in clusters)

    if largest > MAX_CLUSTER_CUSTOMERS:
        raise ValueError(
            f"A cluster has {largest} customers, above MAX_CLUSTER_CUSTOMERS "
            f"({MAX_CLUSTER_CUSTOMERS})."
        )

    limits = integer_limits(time_limits)

    depot = coordinates[0]

    routes = []
    route_clusters = []
    calls = []
    distance = 0

    for cluster_id, (cluster, limit) in enumerate(zip(clusters, limits)):

        idx = np.asarray(cluster) + 1

        cluster_routes, cluster_distance, call = _solve_one_cluster(
            depot,
            coordinates[idx],
            demands[idx],
            cluster,
            capacity,
            limit,
        )

        routes.extend(cluster_routes)
        route_clusters.extend([cluster_id] * len(cluster_routes))
        distance += cluster_distance
        calls.append(call)

    return _summary(routes, route_clusters, distance, limits, calls)


# ============================================================
# WHOLE INSTANCE
# ============================================================

def solve_whole_instance(coordinates, demands, capacity, time_limit):
    """Solve the full CVRP in one call with the whole budget as its timeLimit."""

    limit = integer_limits([time_limit])[0]

    call = _solve(coordinates, demands, capacity, limit)

    _print_call(call)

    _check_solution(call["routes"], demands, capacity)

    distance = sum(s["distance"] for s in route_stats(coordinates, demands, call["routes"]))

    call["cost_ok"] = _cost_ok(call["solver_cost"], distance)

    return _summary(call["routes"], None, distance, [limit], [call])