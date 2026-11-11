"""
PyVRP-based HGS solving (PyVRP 0.12.2 API).

Two modes:
    solve_all_clusters    -> one HGS run per cluster (clustered approach)
    solve_whole_instance  -> one HGS run on the full CVRP (regular HGS)
"""

import numpy as np

from pyvrp import Client, Depot, ProblemData, VehicleType, read, solve
from pyvrp.stop import MaxRuntime


# ============================================================
# READ CUSTOMER INDICES FROM A PYVRP ROUTE
# ============================================================

def get_customer_ids(route, num_depots=1):
    """
    Client indices visited by a PyVRP route, in visit order.

    Route.visits() returns location indices. Depots occupy the
    lowest location indices, so client i (0-based) is returned
    as location i + num_depots. Subtracting num_depots gives the
    0-based client index.
    """
    return [loc - num_depots for loc in route.visits()]


# ============================================================
# SOLVE A SINGLE CLUSTER WITH PYVRP (HGS)
# ============================================================
#
# Each cluster is solved as a multi-vehicle CVRP:
# depot -> customers -> depot, with as many vehicles as the
# cluster's demand needs (plus slack). No capacity repair is
# applied, so HGS decides how to split the cluster into routes.
# ============================================================

def solve_cluster_route(
    depot_coord,
    customer_coords,
    customer_demands,
    original_indices,
    capacity,
    time_limit,
    seed
):
    """
    Solve one cluster with PyVRP (HGS).

    Returns:
        routes:   list of routes, each a list of ORIGINAL instance
                  customer indices (0-indexed) in visit order
        distance: total distance of all routes in this cluster
    """

    num_customers = len(customer_coords)

    if num_customers == 0:
        return [], 0

    # ----------------------------------------------------
    # Location layout:
    #   index 0       = depot
    #   index 1..n    = customers (in cluster order)
    # Client i (0-based) therefore sits at location i + 1.
    # ----------------------------------------------------

    points = np.vstack([
        [depot_coord],
        customer_coords
    ]).astype(float)

    # Integer Euclidean distance matrix (rounded)
    diff = points[:, None, :] - points[None, :, :]

    dist = np.round(
        np.sqrt((diff ** 2).sum(axis=2))
    ).astype(np.int64)

    # ----------------------------------------------------
    # Build the problem data
    # ----------------------------------------------------

    depot = Depot(
        x=float(points[0, 0]),
        y=float(points[0, 1])
    )

    clients = [
        Client(
            x=float(x),
            y=float(y),
            delivery=[int(demand)]
        )
        for (x, y), demand in zip(customer_coords, customer_demands)
    ]

    # Enough vehicles for this cluster's demand, with slack
    # for bin-packing inefficiency
    vehicles_needed = int(
        np.ceil(sum(customer_demands) / capacity)
    )

    vehicle_type = VehicleType(
        num_available=int(np.ceil(vehicles_needed * 1.2)) + 2,
        capacity=[int(capacity)]
    )

    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[dist],
        duration_matrices=[np.zeros_like(dist)]
    )

    # ----------------------------------------------------
    # Solve (PyVRP's solve() runs HGS)
    # ----------------------------------------------------

    result = solve(
        data,
        stop=MaxRuntime(time_limit),
        seed=seed,
        display=False
    )

    if not result.is_feasible():
        raise RuntimeError(
            f"PyVRP found no feasible solution for a cluster "
            f"with {num_customers} customers "
            f"(total demand {sum(customer_demands)}, "
            f"capacity {capacity}, "
            f"vehicles available {vehicle_type.num_available})."
        )

    best = result.best

    # ----------------------------------------------------
    # Map client indices back to ORIGINAL instance indices
    # ----------------------------------------------------

    routes = []
    distance = 0
    visited = []

    for route in best.routes():

        local_ids = get_customer_ids(route, num_depots=1)
        visited.extend(local_ids)

        route_original_indices = [
            original_indices[client_idx]
            for client_idx in local_ids
        ]

        if route_original_indices:
            routes.append(route_original_indices)

        distance += int(route.distance())

    # Sanity check: every customer in the cluster must be visited once
    if sorted(visited) != list(range(num_customers)):
        raise RuntimeError(
            f"Cluster solution does not visit each customer exactly once "
            f"(expected {num_customers} customers, got {len(visited)} visits)."
        )

    return routes, distance


# ============================================================
# SOLVE ALL CLUSTERS FOR ONE CLUSTERING
# ============================================================

def solve_all_clusters(
    coordinates,
    demands,
    clusters,
    capacity,
    time_limit,
    seed
):
    """
    Solve every cluster as its own PyVRP subproblem.

    time_limit is the HGS runtime PER CLUSTER (seconds).

    Returns:
        routes:          flat list of routes; each route is a list
                         of original customer indices
        route_clusters:  cluster id (0-based), one per route
        total_distance:  sum of all route distances
    """

    depot_coord = coordinates[0]

    routes = []
    route_clusters = []
    total_distance = 0

    for cluster_id, cluster in enumerate(clusters):

        customer_coords = [
            coordinates[idx + 1]
            for idx in cluster
        ]

        customer_demands = [
            demands[idx + 1]
            for idx in cluster
        ]

        cluster_routes, distance = solve_cluster_route(
            depot_coord,
            customer_coords,
            customer_demands,
            cluster,
            capacity,
            time_limit,
            seed
        )

        for route in cluster_routes:
            routes.append(route)
            route_clusters.append(cluster_id)

        total_distance += distance

    return routes, route_clusters, total_distance


# ============================================================
# REGULAR HGS (WHOLE INSTANCE)
# ============================================================

def solve_whole_instance(
    instance_path,
    time_limit,
    seed
):
    """
    Solve the full CVRP with PyVRP (HGS).

    time_limit is the TOTAL HGS runtime for the instance (seconds).

    Returns a dict with:
        routes:    list of routes (original customer indices, 0-indexed)
        distance:  result.cost()
        runtime:   runtime reported by PyVRP (seconds)
        feasible:  bool
    """

    # round_func="round" rounds distances to the nearest integer,
    # matching the clustered pipeline.
    instance = read(instance_path, round_func="round")

    result = solve(
        instance,
        stop=MaxRuntime(time_limit),
        seed=seed,
        display=False
    )

    routes = []
    visited = []

    for route in result.best.routes():

        customer_ids = get_customer_ids(route, instance.num_depots)
        visited.extend(customer_ids)

        if customer_ids:
            routes.append(customer_ids)

    if result.is_feasible() and sorted(visited) != list(range(instance.num_clients)):
        raise RuntimeError(
            f"Whole-instance solution does not visit each customer exactly once "
            f"(expected {instance.num_clients} customers, got {len(visited)} visits)."
        )

    return {
        "routes": routes,
        "distance": result.cost(),
        "runtime": result.runtime,
        "feasible": result.is_feasible(),
    }