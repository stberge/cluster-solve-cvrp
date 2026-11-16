"""
Run everything from here.

Change the SETTINGS block, then run this file.

Cluster sizes:
    k = ceil(n / TARGET_CUSTOMERS_PER_CLUSTER), where n is the number of
    customers, limited to between 1 and n. The customers are then clustered
    with the chosen method. Empty clusters are dropped, so the number of
    clusters formed can be lower than k.

Time budget:
    TOTAL_TIME is the total solver runtime of one run. Clustered runs
    split it across their clusters by TIME_SHARE_BASIS ("demand" or
    "customers"). The whole-instance run gets all of TOTAL_TIME.

Output per run (a _2, _3, ... suffix is added if a name already exists):
    results/{instance}/{TOTAL_TIME}s/
        {base}_routes.png      plot
        {base}_routes.txt      notes: settings, cluster quality, routes
        {base}.sol             routes in the reference .sol format
        {base}_clusters.csv    one row per cluster
    results/routing_overview.xlsx    one row per run: settings and distance

Console:
    A settings summary at the start, then for each instance a header with
    its size, and for each run a START block (setup and clustering quality)
    and a DONE block (routes, routes per customer, distance, time).
    routes per customer = number of routes / number of customers.
"""

import os

# Must be set before scikit-learn is imported. Without it, joblib tries to
# count physical cores by running wmic, which newer Windows versions lack.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from clustering import cluster_count, clustering_quality, make_clusters
from ILS import solve_clusters, solve_whole_instance
from instance_io import (
    calculate_number_of_vehicles,
    plot_routes,
    read_vrp,
    route_stats,
    write_routes_txt,
    write_sol,
)
from overview import update_overview


# ============================================================
# SETTINGS
# ============================================================

INSTANCE_DIR = Path(
    r"C:\Users\20235296\Documents\4de Jaar IE\BEP\data\instances\XL\A - test"
)

RESULT_ROOT = Path(
    r"C:\Users\20235296\Documents\4de Jaar IE\BEP\results"
)

OVERVIEW_FILE = RESULT_ROOT / "routing_overview.xlsx"

# Total solver runtime per run (s), shared between clusters.
TOTAL_TIME = 20

# Every configuration runs once per seed
SEEDS = [101]

RUN_CLUSTERED = True
RUN_WHOLE_INSTANCE = False

# Target number of customers per cluster (m). Clusters = ceil(n / m).
TARGET_CUSTOMERS_PER_CLUSTER = [1000]

# Clustering methods: "sweep", "kmeans", "kmeans_weighted", "agglomerative"
CLUSTER_METHODS = [
    "sweep",
    "kmeans",
    "kmeans_weighted",
    "agglomerative",
]

# Share of TOTAL_TIME per cluster: "demand" or "customers"
TIME_SHARE_BASIS = "demand"

SHOW_PLOTS = False


# ============================================================
# CONSOLE
# ============================================================

INDENT = " " * 10
LABEL_WIDTH = 16
RULE = "=" * 72


def stamp(message):
    """Print a message prefixed with the current clock time."""

    print(f"{datetime.now():%H:%M:%S}  {message}")


def kv(label, value):
    """One aligned 'label  value' line for a console block."""

    return f"{label:<{LABEL_WIDTH}}{value}"


def show(lines):
    """Print lines indented under the last stamped message."""

    for line in lines:
        print(f"{INDENT}{line}")


def fmt(value, digits=3):
    """Format a number for printing, or 'n/a' when it is None."""

    return "n/a" if value is None else f"{value:.{digits}f}"


def print_settings():

    targets = ", ".join(str(t) for t in TARGET_CUSTOMERS_PER_CLUSTER)

    print(RULE)
    stamp("SETTINGS")
    show([
        kv("seeds", ", ".join(str(s) for s in SEEDS)),
        kv("total time", f"{TOTAL_TIME} s per run"),
        kv("time share", TIME_SHARE_BASIS),
        kv("methods", ", ".join(CLUSTER_METHODS) if RUN_CLUSTERED else "none"),
        kv("targets", f"{targets} customers/cluster" if RUN_CLUSTERED else "none"),
        kv("whole instance", "yes" if RUN_WHOLE_INSTANCE else "no"),
    ])
    print(RULE)


# ============================================================
# HELPERS
# ============================================================

def new_run_id():
    """Timestamp identifying one run (to the second)."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_folder(filepath):

    return RESULT_ROOT / filepath.stem / f"{TOTAL_TIME}s"


def unique_base_name(folder, base_name):
    """
    Return base_name, or base_name_2, base_name_3, ... if output files
    with that name already exist in folder.
    """

    candidate = base_name
    counter = 2

    while any(
        (folder / f"{candidate}{suffix}").exists()
        for suffix in ("_routes.png", "_routes.txt", ".sol", "_clusters.csv")
    ):
        candidate = f"{base_name}_{counter}"
        counter += 1

    return candidate


def cluster_time_limits(clusters, demands):
    """
    Split TOTAL_TIME across clusters in proportion to TIME_SHARE_BASIS.
    The limits add up to TOTAL_TIME.
    """

    if TIME_SHARE_BASIS == "demand":
        weights = [
            int(demands[np.asarray(cluster) + 1].sum())
            for cluster in clusters
        ]

    elif TIME_SHARE_BASIS == "customers":
        weights = [len(cluster) for cluster in clusters]

    else:
        raise ValueError(
            f"TIME_SHARE_BASIS must be 'demand' or 'customers', "
            f"got '{TIME_SHARE_BASIS}'."
        )

    weights = np.asarray(weights, dtype=float)

    if weights.sum() == 0:
        weights = np.ones(len(clusters))

    return (TOTAL_TIME * weights / weights.sum()).tolist()


def routes_quality(coordinates, demands, capacity, routes, route_clusters, per_cluster):
    """
    Quality of the routes. Adds routes, min_routes, route_overhead and
    distance to each row of per_cluster (in place). Returns a summary dict.
    """

    stats = route_stats(coordinates, demands, routes)

    routes_per_cluster = [0] * len(per_cluster)
    distance_per_cluster = [0] * len(per_cluster)

    for cluster_id, stat in zip(route_clusters, stats):
        routes_per_cluster[cluster_id] += 1
        distance_per_cluster[cluster_id] += stat["distance"]

    for label, row in enumerate(per_cluster):

        min_routes = int(np.ceil(row["demand"] / capacity))

        row["routes"] = routes_per_cluster[label]
        row["min_routes"] = min_routes
        row["route_overhead"] = routes_per_cluster[label] - min_routes
        row["distance"] = distance_per_cluster[label]

    return {
        "route_overhead": sum(row["route_overhead"] for row in per_cluster),
        "distance": sum(distance_per_cluster),
    }


def write_cluster_csv(path, per_cluster):
    """One row per cluster, with before-solve and after-solve measures."""

    if not per_cluster:
        return

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(per_cluster[0]))
        writer.writeheader()
        writer.writerows(per_cluster)


def check_distances(stats, total_distance):
    """Warn if the per-route distances do not add up to the solver total."""

    check = sum(s["distance"] for s in stats)

    if check != total_distance:
        print(
            f"WARNING: route distances sum to {check}, "
            f"solver reports {total_distance}"
        )


def save_outputs(
    folder,
    base_name,
    title,
    coordinates,
    demands,
    capacity,
    routes,
    total_distance,
    header_lines,
    route_groups=None,
    n_groups=1,
):
    """Write the plot, the routes text file and the .sol file for one run."""

    clustered = route_groups is not None

    plot_routes(
        coordinates,
        routes,
        route_groups if clustered else [0] * len(routes),
        n_groups,
        title,
        folder / f"{base_name}_routes.png",
        show=SHOW_PLOTS,
        clustered=clustered,
    )

    stats = write_routes_txt(
        folder / f"{base_name}_routes.txt",
        coordinates,
        demands,
        capacity,
        routes,
        total_distance,
        route_clusters=route_groups,
        header_lines=header_lines,
    )

    write_sol(folder / f"{base_name}.sol", routes, total_distance)

    check_distances(stats, total_distance)


# ============================================================
# CLUSTERED RUN
# ============================================================

def run_clustering(
    filepath, coordinates, demands, capacity, method, target_customers, seed, progress
):

    started = time.perf_counter()
    customers = len(coordinates) - 1
    requested = cluster_count(customers, target_customers)

    print()
    stamp(
        f"{progress} START  clustered | {method} | seed {seed} | "
        f"target {target_customers} customers/cluster"
    )

    clusters = make_clusters(coordinates, demands, target_customers, method, seed)

    n_clusters = len(clusters)

    sizes = [len(cluster) for cluster in clusters]

    cluster_demands = [
        int(demands[np.asarray(cluster) + 1].sum())
        for cluster in clusters
    ]

    time_limits = cluster_time_limits(clusters, demands)

    before, per_cluster = clustering_quality(
        coordinates, demands, clusters, target_customers, seed
    )

    if n_clusters == requested:
        clusters_text = f"{n_clusters}"
    else:
        clusters_text = f"{n_clusters} (requested {requested}, empty dropped)"

    show([
        kv("clusters", clusters_text),
        kv("customers", f"{min(sizes)} to {max(sizes)} per cluster"),
        kv("demand", f"{min(cluster_demands)} to {max(cluster_demands)} "
                     f"per cluster (capacity {capacity})"),
        kv("solver time", f"{min(time_limits):.1f} to {max(time_limits):.1f} s "
                          f"per cluster ({TIME_SHARE_BASIS} share, "
                          f"{TOTAL_TIME} s total)"),
        kv("nearest share", fmt(before["nearest_centroid_share"])),
        kv("silhouette", fmt(before["silhouette"])),
        kv("demand CV", fmt(before["demand_cv"])),
        kv("solving", "..."),
    ])

    routes, route_clusters, total_distance = solve_clusters(
        coordinates,
        demands,
        clusters,
        capacity,
        time_limits=time_limits,
        seed=seed,
    )

    after = routes_quality(
        coordinates, demands, capacity, routes, route_clusters, per_cluster
    )

    for row, time_limit in zip(per_cluster, time_limits):
        row["time_limit_s"] = round(time_limit, 2)

    run_id = new_run_id()

    folder = run_folder(filepath)
    folder.mkdir(parents=True, exist_ok=True)

    base_name = unique_base_name(
        folder,
        f"{filepath.stem}_{method}_cust{target_customers}_k{n_clusters}_{run_id}",
    )

    title = (
        f"{filepath.stem} - {method} "
        f"(target {target_customers} customers/cluster, k = {n_clusters}, "
        f"{TOTAL_TIME}s total)\n"
        f"{len(routes)} routes | distance = {total_distance} | seed {seed}"
    )

    header_lines = [
        f"Run ID: {run_id}",
        f"Instance: {filepath.name}",
        f"Method: clustered ({method})",
        f"Capacity: {capacity}",
        f"Target customers per cluster: {target_customers}",
        f"Clusters: requested {requested}, formed {n_clusters}",
        f"Customers per cluster (min/max): {min(sizes)} / {max(sizes)}",
        f"Demand per cluster (min/max): {min(cluster_demands)} / {max(cluster_demands)}",
        f"Time share basis: {TIME_SHARE_BASIS}",
        f"Solver time per cluster (s, min/max): "
        f"{min(time_limits):.2f} / {max(time_limits):.2f}",
        f"Total solver time (s): {TOTAL_TIME}",
        f"Nearest-centroid share: {before['nearest_centroid_share']}",
        f"Silhouette: {before['silhouette']}",
        f"Demand CV: {before['demand_cv']}",
        f"Route overhead vs cluster lower bounds: {after['route_overhead']}",
        f"Seed: {seed}",
    ]

    save_outputs(
        folder,
        base_name,
        title,
        coordinates,
        demands,
        capacity,
        routes,
        total_distance,
        header_lines,
        route_groups=route_clusters,
        n_groups=n_clusters,
    )

    write_cluster_csv(folder / f"{base_name}_clusters.csv", per_cluster)

    row = {
        "run_id": run_id,
        "instance": filepath.stem,
        "method": "clustered",
        "algorithm": method,
        "target_customers_per_cluster": target_customers,
        "time_share_basis": TIME_SHARE_BASIS,
        "n_clusters_requested": requested,
        "n_clusters_formed": n_clusters,
        "min_cluster_customers": min(sizes),
        "max_cluster_customers": max(sizes),
        "min_cluster_demand": min(cluster_demands),
        "max_cluster_demand": max(cluster_demands),
        "max_cluster_load_pct": round(100 * max(cluster_demands) / capacity, 1),
        "demand_cv": before["demand_cv"],
        "nearest_centroid_share": before["nearest_centroid_share"],
        "silhouette": before["silhouette"],
        "mean_customer_radius": before["mean_customer_radius"],
        "total_distance": total_distance,
        "avg_time_per_cluster_s": round(TOTAL_TIME / n_clusters, 2),
        "total_time_s": TOTAL_TIME,
        "seed": seed,
        "output_folder": str(folder),
        "file_name": base_name,
    }

    update_overview(OVERVIEW_FILE, [row])

    elapsed = time.perf_counter() - started

    stamp(f"{progress} DONE   clustered | {method} | seed {seed} | {elapsed:.0f} s")
    show([
        kv("routes", len(routes)),
        kv("routes/customer", f"{len(routes) / customers:.4f}"),
        kv("distance", total_distance),
        kv("saved as", base_name),
    ])

    return row


# ============================================================
# WHOLE-INSTANCE RUN
# ============================================================

def run_whole_instance(filepath, coordinates, demands, capacity, seed, progress):

    started = time.perf_counter()
    customers = len(coordinates) - 1

    print()
    stamp(f"{progress} START  whole instance | seed {seed} | {TOTAL_TIME} s")

    result = solve_whole_instance(filepath, time_limit=TOTAL_TIME, seed=seed)

    routes = result["routes"]
    total_distance = result["distance"]

    run_id = new_run_id()

    folder = run_folder(filepath)
    folder.mkdir(parents=True, exist_ok=True)

    base_name = unique_base_name(
        folder,
        f"{filepath.stem}_whole_{TOTAL_TIME}s_{run_id}",
    )

    title = (
        f"{filepath.stem} - whole instance ({TOTAL_TIME}s total)\n"
        f"{len(routes)} routes | distance = {total_distance} | seed {seed}"
    )

    header_lines = [
        f"Run ID: {run_id}",
        f"Instance: {filepath.name}",
        "Method: whole instance",
        f"Capacity: {capacity}",
        f"Total solver time (s): {TOTAL_TIME}",
        f"Runtime reported (s): {result['runtime']:.2f}",
        f"Feasible: {result['feasible']}",
        f"Seed: {seed}",
    ]

    save_outputs(
        folder,
        base_name,
        title,
        coordinates,
        demands,
        capacity,
        routes,
        total_distance,
        header_lines,
    )

    row = {
        "run_id": run_id,
        "instance": filepath.stem,
        "method": "whole_instance",
        "algorithm": "-",
        "total_distance": total_distance,
        "total_time_s": TOTAL_TIME,
        "seed": seed,
        "output_folder": str(folder),
        "file_name": base_name,
    }

    update_overview(OVERVIEW_FILE, [row])

    elapsed = time.perf_counter() - started

    stamp(f"{progress} DONE   whole instance | seed {seed} | {elapsed:.0f} s")
    show([
        kv("routes", len(routes)),
        kv("routes/customer", f"{len(routes) / customers:.4f}"),
        kv("distance", total_distance),
        kv("feasible", result["feasible"]),
        kv("saved as", base_name),
    ])

    return row


# ============================================================
# ONE INSTANCE
# ============================================================

def run_instance(filepath, index, count):

    coordinates, demands, capacity = read_vrp(filepath)

    customers = len(coordinates) - 1
    total_demand = int(demands[1:].sum())
    min_vehicles = calculate_number_of_vehicles(demands[1:], capacity)

    # One entry per run for this instance, in the order they will run
    runs = []

    if RUN_CLUSTERED:
        for seed in SEEDS:
            for method in CLUSTER_METHODS:
                for target_customers in TARGET_CUSTOMERS_PER_CLUSTER:
                    runs.append(("clustered", seed, method, target_customers))

    if RUN_WHOLE_INSTANCE:
        for seed in SEEDS:
            runs.append(("whole", seed, None, None))

    print()
    print(RULE)
    stamp(f"INSTANCE {index}/{count}  {filepath.name}")
    show([
        kv("customers", customers),
        kv("total demand", total_demand),
        kv("capacity", capacity),
        kv("min vehicles", min_vehicles),
        kv("runs", len(runs)),
    ])
    print(RULE)

    for number, (kind, seed, method, target_customers) in enumerate(runs, start=1):

        progress = f"[{number}/{len(runs)}]"

        if kind == "clustered":
            run_clustering(
                filepath, coordinates, demands, capacity,
                method, target_customers, seed, progress,
            )

        else:
            run_whole_instance(
                filepath, coordinates, demands, capacity, seed, progress,
            )

    print()
    stamp(f"INSTANCE {index}/{count} finished")


# ============================================================
# MAIN
# ============================================================

def main():

    if not (RUN_CLUSTERED or RUN_WHOLE_INSTANCE):
        print("Nothing to run: RUN_CLUSTERED and RUN_WHOLE_INSTANCE are both False.")
        return

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(INSTANCE_DIR.glob("*.vrp"))

    if not files:
        print("No .vrp files found.")
        return

    print_settings()
    stamp(f"Found {len(files)} instances in {INSTANCE_DIR.name}")

    for index, filepath in enumerate(files, start=1):
        run_instance(filepath, index, len(files))

    print()
    print(RULE)
    stamp("ALL INSTANCES FINISHED")
    show([kv("overview", str(OVERVIEW_FILE))])
    print(RULE)


if __name__ == "__main__":

    main()