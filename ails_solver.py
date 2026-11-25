"""
AILS-II routing through the ailsii package.

    solve_cluster(...)        every cluster solved on its own, one after another
    solve_whole_instance(...) the full instance solved in one call

Each AILS-II call gets a whole-second timeLimit taken from the run's time
share. The elapsed time of each call is measured, and its overhead (elapsed
minus timeLimit) is recorded. Nothing is subtracted from the budget.

Distances are recomputed with route_stats, so reported totals match the
route files. The solver's own cost is kept and compared as a check.

Trace: best cost over time as (time_s, cluster, distance) rows on the run's
clock. Clusters run one after another, so a cluster's time starts when the
previous cluster has finished. The rows come from the solver's history if
ailsii reports one (see _history); otherwise each call contributes only its
final cost.
"""

import time

import numpy as np
import ailsii

from instance_io import route_stats


# AILS-II parameters (README defaults). Stored with every run.
AILS_PARAMS = dict(varphi=40, gamma=30, dMax=30, dMin=15)

PRINT_SOLVER_CALLS = True


# ============================================================
# TIME LIMITS
# ============================================================

def integer_limits(shares):
    """
    Whole-second timeLimit per share. The limits sum to the floor of the
    total share, so the search time never exceeds the budget. The solver
    takes whole seconds, so every limit must be at least 1.
    """

    shares = np.asarray(shares, dtype=float)

    limits = np.floor(shares).astype(int)

    if np.any(limits < 1):
        worst = int(np.argmin(shares))
        raise ValueError(
            f"Share {worst} is {shares[worst]:.2f} s, under 1 whole second. "
            f"Use fewer clusters or a larger TOTAL_TIME."
        )

    spare = int(np.floor(shares.sum() + 1e-9)) - int(limits.sum())

    order = np.argsort(-(shares - limits), kind="stable")

    limits[order[:spare]] += 1

    return limits.tolist()


# ============================================================
# ONE AILS-II CALL
# ============================================================

def _history(result):
    """
    Improvements found during the search as (seconds, cost) pairs, or None
    if the solver reports no history. Change the attribute name here if
    ailsii uses another one.
    """

    history = getattr(result, "history", None)

    if history is None:
        return None

    return [(float(t), float(c)) for t, c in history]


def _solve(points, demands, capacity, time_limit):
    """
    One AILS-II call. points and demands include the depot at index 0.

    Returns routes (0-based customer indices), the solver's cost, its
    history (or None), the timeLimit that was set, and the measured elapsed
    time and overhead of the call.
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
        "history": _history(result),
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
            f"AILS-II visited {len(visited)} customer slots for {n_customers} customers."
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


def _solve_checked(points, demands, capacity, time_limit):
    """
    One AILS-II call, printed, checked and measured with route_stats.
    Returns (call, distance).
    """

    call = _solve(points, demands, capacity, time_limit)

    if PRINT_SOLVER_CALLS:
        print(
            f"          solver call: timeLimit {call['limit']} s, "
            f"elapsed {call['elapsed']:.2f} s, overhead {call['overhead']:.2f} s"
        )

    _check_solution(call["routes"], demands, capacity)

    distance = sum(s["distance"] for s in route_stats(points, demands, call["routes"]))

    call["cost_ok"] = _cost_ok(call["solver_cost"], distance)

    return call, distance


def _call_trace(call, offset, group):
    """Rows (time_s, group, distance) for one call, on the run's clock."""

    points = list(call["history"] or [])

    points.append((call["elapsed"], call["solver_cost"]))

    return [(round(offset + t, 3), group, cost) for t, cost in points]


# ============================================================
# SOLVE ONE CLUSTER (private)
# ============================================================

def _solve_one_cluster(
    depot,
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

    points = np.vstack([depot, customer_coords]).astype(float)

    demands = np.concatenate([[0], np.asarray(customer_demands, dtype=int)])

    call, distance = _solve_checked(points, demands, capacity, limit)

    routes = [[original_indices[i] for i in route] for route in call["routes"]]

    return routes, distance, call


# ============================================================
# RESULTS
# ============================================================

def _summary(routes, route_clusters, distance, limits, calls, trace):
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
        "overhead_s": round(sum(overheads), 3),
        "overhead_max_s": round(max(overheads), 3),
        "cost_ok": all(call["cost_ok"] for call in calls),
        "trace_complete": all(call["history"] is not None for call in calls),
        "trace": trace,
    }


# ============================================================
# SOLVE ALL CLUSTERS
# ============================================================

def solve_cluster(coordinates, demands, clusters, capacity, time_limits):
    """
    Solve every cluster as its own subproblem, one after another.

    time_limits holds one search share (seconds) per cluster, in the same
    order as clusters. integer_limits converts them to whole seconds.
    """

    if len(time_limits) != len(clusters):
        raise ValueError("Need exactly one time share per cluster.")

    limits = integer_limits(time_limits)

    depot = coordinates[0]

    routes = []
    route_clusters = []
    calls = []
    trace = []
    distance = 0
    clock = 0.0

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
        trace.extend(_call_trace(call, clock, cluster_id))
        clock += call["elapsed"]

    return _summary(routes, route_clusters, distance, limits, calls, trace)


# ============================================================
# WHOLE INSTANCE
# ============================================================

def solve_whole_instance(coordinates, demands, capacity, time_limit):
    """Solve the full CVRP in one call with the whole budget as its timeLimit."""

    limit = integer_limits([time_limit])[0]

    call, distance = _solve_checked(coordinates, demands, capacity, limit)

    trace = _call_trace(call, 0.0, None)

    return _summary(call["routes"], None, distance, [limit], [call], trace)