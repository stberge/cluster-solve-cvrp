"""
Run everything from here.

Change the SETTINGS block below, then run this file.

Every run is saved separately:
    - its own row in results\\routing_overview.xlsx (identified by run_id)
    - its own .png and .txt files, with the run_id in the file name

Output layout (per run):
    results\\{instance}\\{total_hgs_time}s\\
        {instance}_{algorithm}_k{n_clusters}_{total_hgs_time}s_{run_id}_routes.png
        {instance}_{algorithm}_k{n_clusters}_{total_hgs_time}s_{run_id}_routes.txt
    results\\{instance}\\{hgs_time}s\\          (regular HGS, no cluster level)
        {instance}_HGS_{hgs_time}s_{run_id}_routes.png / .txt

If two runs with identical settings finish within the same second, the
second one gets a _2 (then _3, ...) suffix, so nothing is overwritten.

Modules:
    instance_io.py  - reading, route statistics, text output, plotting
    clustering.py   - sweep, kmeans, agglomerative
    hgs12.py        - PyVRP (HGS) solving, per cluster and whole instance
    overview.py     - Excel overview
"""

from datetime import datetime
from pathlib import Path

from instance_io import (
    read_vrp,
    calculate_number_of_vehicles,
    write_routes_txt,
    plot_routes,
)
from clustering import ALGORITHM_FUNCTIONS
from hgs import solve_all_clusters, solve_whole_instance
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

# Random seed (clustering and PyVRP)
SEED = None  # None = random seed, int = fixed seed for reproducibility

# ----- Regular HGS (whole instance) -----
RUN_REGULAR_HGS = True
REGULAR_HGS_TIME = 1200          # total HGS runtime per instance (s)

# ----- Clustered HGS -----
RUN_CLUSTERED = False

# Number of clusters. One clustered run per value in the list,
# for every algorithm in CLUSTER_ALGORITHMS. E.g. [4, 6, 50].
N_CLUSTERS = [20]

# Clustering algorithms to run: "sweep", "kmeans", "agglomerative"
CLUSTER_ALGORITHMS = [
    "sweep",
    "kmeans",
    "agglomerative",
]

CLUSTER_HGS_TIME = 60          # HGS runtime PER CLUSTER (s)

# ----- Display -----
SHOW_PLOTS = False            # True = open each plot window (blocks)


# ============================================================
# HELPERS
# ============================================================

def new_run_id():
    """Timestamp identifying one run (to the second)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_base_name(folder, base_name):
    """
    Return base_name, or base_name_2, base_name_3, ... if files with
    that name already exist in folder. Guarantees no run is overwritten.
    """

    candidate = base_name
    counter = 2

    while (
        (folder / f"{candidate}_routes.png").exists()
        or (folder / f"{candidate}_routes.txt").exists()
    ):

        candidate = f"{base_name}_{counter}"
        counter += 1

    return candidate


def _check_distances(stats, total_distance):
    """Warn if per-route distances do not add up to the solver total."""

    check = sum(s["distance"] for s in stats)

    if check != total_distance:
        print(
            f"WARNING: route distances sum to {check}, "
            f"solver reports {total_distance}"
        )


# ============================================================
# RUN CLUSTERED HGS FOR ONE INSTANCE
# ============================================================

def run_clustered(filepath, coordinates, demands, capacity):

    rows = []

    for algorithm_name in CLUSTER_ALGORITHMS:

        if algorithm_name not in ALGORITHM_FUNCTIONS:
            raise ValueError(
                f"Unknown clustering algorithm: {algorithm_name}. "
                f"Available: {list(ALGORITHM_FUNCTIONS)}"
            )

        algorithm_func = ALGORITHM_FUNCTIONS[algorithm_name]

        for n_clusters in N_CLUSTERS:

            print()
            print(f"--- {algorithm_name} | n_clusters = {n_clusters} ---")

            # 1. CLUSTER
            clusters = algorithm_func(
                coordinates,
                demands,
                capacity,
                n_clusters,
                SEED
            )

            num_formed = len(clusters)

            cluster_demands = [
                int(demands[[idx + 1 for idx in cluster]].sum())
                for cluster in clusters
            ]

            total_hgs_time = num_formed * CLUSTER_HGS_TIME

            print(f"Clusters formed: {num_formed}")
            print(
                f"Cluster demand min/max: "
                f"{min(cluster_demands)} / {max(cluster_demands)} "
                f"(capacity {capacity})"
            )

            # 2. HGS PER CLUSTER
            print(
                f"Solving {num_formed} cluster(s) with HGS "
                f"({CLUSTER_HGS_TIME}s each, {total_hgs_time}s total)..."
            )

            routes, route_clusters, total_distance = solve_all_clusters(
                coordinates,
                demands,
                clusters,
                capacity,
                time_limit=CLUSTER_HGS_TIME,
                seed=SEED
            )

            print(f"Routes: {len(routes)}")
            print(f"Total distance: {total_distance}")

            # 3. RUN ID, FOLDER AND UNIQUE FILE NAME
            run_id = new_run_id()

            folder = (
                RESULT_ROOT
                / filepath.stem
                / f"{total_hgs_time}s"
            )

            folder.mkdir(parents=True, exist_ok=True)

            base_name = unique_base_name(
                folder,
                f"{filepath.stem}_{algorithm_name}"
                f"_k{n_clusters}_{total_hgs_time}s_{run_id}"
            )

            title = (
                f"{filepath.stem} - {algorithm_name} "
                f"(k = {n_clusters}, {total_hgs_time}s total HGS)\n"
                f"{num_formed} clusters | {len(routes)} routes | "
                f"distance = {total_distance}"
            )

            # 4. OUTPUT
            plot_routes(
                coordinates,
                routes,
                route_clusters,
                num_formed,
                title,
                folder / f"{base_name}_routes.png",
                show=SHOW_PLOTS,
                clustered=True,
            )

            stats = write_routes_txt(
                folder / f"{base_name}_routes.txt",
                coordinates,
                demands,
                capacity,
                routes,
                total_distance,
                route_clusters=route_clusters,
                header_lines=[
                    f"Run ID: {run_id}",
                    f"Instance: {filepath.name}",
                    f"Method: clustered HGS ({algorithm_name})",
                    f"Capacity: {capacity}",
                    f"Clusters requested: {n_clusters}",
                    f"Clusters formed: {num_formed}",
                    f"HGS time per cluster (s): {CLUSTER_HGS_TIME}",
                    f"Total HGS time (s): {total_hgs_time}",
                    f"Seed: {SEED}",
                ]
            )

            _check_distances(stats, total_distance)

            row = {
                "run_id": run_id,
                "instance": filepath.stem,
                "method": "clustered_hgs",
                "algorithm": algorithm_name,
                "n_clusters_requested": n_clusters,
                "n_clusters_formed": num_formed,
                "num_routes": len(routes),
                "total_distance": total_distance,
                "min_cluster_demand": min(cluster_demands),
                "max_cluster_demand": max(cluster_demands),
                "max_cluster_load_pct": round(
                    100 * max(cluster_demands) / capacity, 1
                ),
                "hgs_time_per_cluster_s": CLUSTER_HGS_TIME,
                "total_hgs_time_s": total_hgs_time,
                "seed": SEED,
                "output_folder": str(folder),
                "file_name": base_name,
            }

            rows.append(row)

            update_overview(OVERVIEW_FILE, [row])
            print(f"Run ID: {run_id}")
            print(f"Saved to: {folder}")

    return rows


# ============================================================
# RUN REGULAR HGS FOR ONE INSTANCE
# ============================================================

def run_regular(filepath, coordinates, demands, capacity):

    print()
    print("--- regular HGS (whole instance) ---")
    print(f"Running HGS for {REGULAR_HGS_TIME} seconds...")

    result = solve_whole_instance(
        filepath,
        time_limit=REGULAR_HGS_TIME,
        seed=SEED
    )

    routes = result["routes"]
    total_distance = result["distance"]

    print(f"Routes: {len(routes)}")
    print(f"Distance: {total_distance}")
    print(f"Feasible: {result['feasible']}")

    # 1. RUN ID, FOLDER AND UNIQUE FILE NAME
    run_id = new_run_id()

    folder = (
        RESULT_ROOT
        / filepath.stem
        / f"{REGULAR_HGS_TIME}s"
    )

    folder.mkdir(parents=True, exist_ok=True)

    base_name = unique_base_name(
        folder,
        f"{filepath.stem}_HGS_{REGULAR_HGS_TIME}s_{run_id}"
    )

    title = (
        f"{filepath.stem} - regular HGS "
        f"({REGULAR_HGS_TIME}s total)\n"
        f"{len(routes)} routes | distance = {total_distance}"
    )

    # 2. OUTPUT
    plot_routes(
        coordinates,
        routes,
        [0] * len(routes),
        1,
        title,
        folder / f"{base_name}_routes.png",
        show=SHOW_PLOTS,
        clustered=False,
    )

    stats = write_routes_txt(
        folder / f"{base_name}_routes.txt",
        coordinates,
        demands,
        capacity,
        routes,
        total_distance,
        header_lines=[
            f"Run ID: {run_id}",
            f"Instance: {filepath.name}",
            "Method: regular HGS (whole instance)",
            f"Capacity: {capacity}",
            f"Total HGS time (s): {REGULAR_HGS_TIME}",
            f"Runtime reported (s): {result['runtime']:.2f}",
            f"Feasible: {result['feasible']}",
            f"Seed: {SEED}",
        ]
    )

    _check_distances(stats, total_distance)

    row = {
        "run_id": run_id,
        "instance": filepath.stem,
        "method": "regular_hgs",
        "algorithm": "-",
        "n_clusters_requested": None,
        "n_clusters_formed": None,
        "num_routes": len(routes),
        "total_distance": total_distance,
        "min_cluster_demand": None,
        "max_cluster_demand": None,
        "max_cluster_load_pct": None,
        "hgs_time_per_cluster_s": None,
        "total_hgs_time_s": REGULAR_HGS_TIME,
        "seed": SEED,
        "output_folder": str(folder),
        "file_name": base_name,
    }

    update_overview(OVERVIEW_FILE, [row])
    print(f"Run ID: {run_id}")
    print(f"Saved to: {folder}")

    return [row]


# ============================================================
# RUN ONE INSTANCE
# ============================================================

def run_instance(filepath):

    print()
    print("=" * 60)
    print(f"INSTANCE: {filepath.name}")
    print("=" * 60)

    coordinates, demands, capacity = read_vrp(filepath)

    print(f"Customers: {len(coordinates) - 1}")
    print(f"Total demand: {demands[1:].sum()}")
    print(f"Vehicle capacity (from file): {capacity}")
    print(
        f"Minimum vehicles: "
        f"{calculate_number_of_vehicles(demands[1:], capacity)}"
    )

    if RUN_CLUSTERED:

        run_clustered(
            filepath,
            coordinates,
            demands,
            capacity
        )

    if RUN_REGULAR_HGS:

        run_regular(
            filepath,
            coordinates,
            demands,
            capacity
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not (RUN_CLUSTERED or RUN_REGULAR_HGS):
        print("Nothing to run: RUN_CLUSTERED and RUN_REGULAR_HGS are both False.")
        return

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(
        INSTANCE_DIR.glob("*.vrp")
    )

    if not files:

        print("No .vrp files found.")

        return

    print(f"Found {len(files)} instances.")

    for filepath in files:

        run_instance(filepath)

    print()
    print("=" * 60)
    print(f"ALL INSTANCES FINISHED. Overview: {OVERVIEW_FILE}")
    print("=" * 60)


if __name__ == "__main__":

    main()