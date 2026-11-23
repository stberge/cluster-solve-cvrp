"""
Benchmark runner: clustered runs and the whole-instance baseline on the XL set.

Change the SETTINGS block, then run:
    python run_benchmark.py              run the batch (resumes automatically)
    python run_benchmark.py --overview   rebuild only the Excel overview

Clusters
--------
Clustered runs are sized by routes per cluster R: k = ceil(min_vehicles / R),
where min_vehicles = ceil(total demand / capacity). Real routes per cluster
can be somewhat above R, because min_vehicles is a lower bound.

Seeds
-----
k-means runs draw a fresh random seed every run. The seed is stored in the
run's JSON. Other methods have no seed.

Time
----
Every run gets TOTAL_TIME seconds of AILS-II search, split across its clusters
by TIME_SHARE_BASIS. Each AILS-II call's timeLimit is its share. No setup or
overhead time is subtracted or reported.

Output
------
    RESULT_ROOT/budget_{TOTAL_TIME}s/{instance}/{run_key}.json   source of truth
    RESULT_ROOT/budget_{TOTAL_TIME}s/{instance}/{run_key}_routes.txt
    RESULT_ROOT/budget_{TOTAL_TIME}s/{instance}/{run_key}.sol
    RESULT_ROOT/overview_{TOTAL_TIME}s.xlsx                      built from the JSON files

Run keys look like:
    <instance>__whole__r1
    <instance>__kmeans__R30__r2

A failed run is recorded with status "failed" and the batch continues.
Failed runs are retried when the batch is started again. Finished runs
are skipped.
"""

import os

# Must be set before scikit-learn is imported (see clustering.py).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import json
import platform
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import psutil
import sklearn

from ails_solver import AILS_PARAMS, MAX_CLUSTER_CUSTOMERS, solve_cluster, solve_whole_instance
from clustering import METHODS, cluster_count, clustering_quality, make_clusters
from instance_io import (
    calculate_number_of_vehicles,
    plot_routes,
    read_vrp,
    write_routes_txt,
    write_sol,
)
from overview import build_overview


# ============================================================
# SETTINGS
# ============================================================

INSTANCE_DIR = Path(
    r"C:\Users\20235296\Documents\4de Jaar IE\BEP\data\instances\XL"
)

# Local disk recommended: OneDrive-synced folders can lock files mid-batch.
RESULT_ROOT = Path(r"C:\Users\20235296\Documents\4de Jaar IE\BEP\results")

# Search budget per run (s), identical for every method.
TOTAL_TIME = 600

# Runs per instance for each method. "whole" is the baseline (no clustering).
# 0 skips a method. sweep and agglomerative always give the same clusters,
# but AILS-II is a stochastic search, so repeated runs still differ.
RUNS = {
    "sweep": 1,
    "agglomerative": 1,
    "kmeans": 1,
    "whole": 1,
}

# Cluster sizes: k = ceil(min_vehicles / R) for each R.
ROUTES_PER_CLUSTER = [30, 50, 80, 130, 200, 300]

# Optional: k = ceil(num_customers / m) for each m. Leave empty to skip.
TARGET_CUSTOMERS_PER_CLUSTER = []

# Share of TOTAL_TIME per cluster: "customers" or "demand"
TIME_SHARE_BASIS = "customers"

SAVE_PLOTS = False
SHOW_PLOTS = False

# Derived from the settings above
CLUSTER_METHODS = [m for m, n in RUNS.items() if m != "whole" and n > 0]

SIZINGS = (
    [("routes", r) for r in ROUTES_PER_CLUSTER]
    + [("customers", m) for m in TARGET_CUSTOMERS_PER_CLUSTER]
)

SIZE_LETTER = {"routes": "R", "customers": "m"}


# ============================================================
# CONSOLE
# ============================================================

INDENT = " " * 10
LABEL_WIDTH = 16
RULE = "=" * 72


def stamp(message):
    print(f"{datetime.now():%H:%M:%S}  {message}")


def show(lines):
    for line in lines:
        print(f"{INDENT}{line}")


def kv(label, value):
    return f"{label:<{LABEL_WIDTH}}{value}"


# ============================================================
# LIBRARY VERSIONS AND MEMORY
# ============================================================

def library_versions():

    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }

    try:
        info["ailsii"] = version("ailsii")
    except PackageNotFoundError:
        info["ailsii"] = "unknown"

    return info


VERSIONS = library_versions()


def peak_mb():
    """Peak memory of this process so far (cumulative over the batch)."""

    info = psutil.Process(os.getpid()).memory_info()

    return round(getattr(info, "peak_wset", info.rss) / 2**20, 1)


# ============================================================
# SETTINGS CHECK
# ============================================================

def check_settings():

    unknown = set(RUNS) - {"whole", *METHODS}

    if unknown:
        raise ValueError(
            f"Unknown method(s) in RUNS: {sorted(unknown)}. "
            f"Available: whole, {', '.join(METHODS)}."
        )

    if any(n < 0 for n in RUNS.values()):
        raise ValueError("Run counts in RUNS must be 0 or more.")

    if CLUSTER_METHODS and not SIZINGS:
        raise ValueError(
            "Clustered methods are set in RUNS, but ROUTES_PER_CLUSTER and "
            "TARGET_CUSTOMERS_PER_CLUSTER are both empty."
        )

    if TIME_SHARE_BASIS not in ("customers", "demand"):
        raise ValueError(
            f"TIME_SHARE_BASIS must be 'customers' or 'demand', "
            f"got '{TIME_SHARE_BASIS}'."
        )


# ============================================================
# PLANNING
# ============================================================

def budget_folder():
    return RESULT_ROOT / f"budget_{TOTAL_TIME}s"


def overview_path():
    return RESULT_ROOT / f"overview_{TOTAL_TIME}s.xlsx"


def size_labels():
    return (
        [f"R={r}" for r in ROUTES_PER_CLUSTER]
        + [f"m={m}" for m in TARGET_CUSTOMERS_PER_CLUSTER]
    )


def planned_runs():
    """Every run for one instance, baseline first. Each run has a run number."""

    runs = []

    for run in range(1, RUNS.get("whole", 0) + 1):
        runs.append({
            "kind": "whole",
            "clustering": "-",
            "sizing": None,
            "value": None,
            "run": run,
            "seed": None,
        })

    for method in CLUSTER_METHODS:
        for sizing, value in SIZINGS:
            for run in range(1, RUNS[method] + 1):
                runs.append({
                    "kind": "clustered",
                    "clustering": method,
                    "sizing": sizing,
                    "value": value,
                    "run": run,
                    "seed": None,
                })

    return runs


def run_key(instance, spec):
    """Deterministic name for one run. Used for file names and resuming."""

    if spec["kind"] == "whole":
        return f"{instance}__whole__r{spec['run']}"

    letter = SIZE_LETTER[spec["sizing"]]

    return f"{instance}__{spec['clustering']}__{letter}{spec['value']}__r{spec['run']}"


def describe(spec):

    if spec["kind"] == "whole":
        return f"whole instance | {TOTAL_TIME} s | run {spec['run']}"

    size = f"{SIZE_LETTER[spec['sizing']]}={spec['value']}"
    seed = "" if spec["seed"] is None else f" | seed {spec['seed']}"

    return f"clustered | {spec['clustering']} | {size} | run {spec['run']}{seed}"


def random_seed():
    """Fresh random seed for one run. Unseeded RNG, so every call differs."""

    return int(np.random.default_rng().integers(1, 2**31 - 1))


# ============================================================
# TIME SHARES
# ============================================================

def cluster_time_limits(clusters, demands):
    """Split TOTAL_TIME across clusters in proportion to TIME_SHARE_BASIS."""

    if TIME_SHARE_BASIS == "customers":
        weights = [len(cluster) for cluster in clusters]

    else:
        weights = [int(demands[np.asarray(c) + 1].sum()) for c in clusters]

    weights = np.asarray(weights, dtype=float)

    if weights.sum() == 0:
        weights = np.ones(len(clusters))

    return (TOTAL_TIME * weights / weights.sum()).tolist()


def count_route_overhead(route_clusters, n_clusters, cluster_demands, capacity):
    """Routes beyond the lower bound ceil(demand / capacity), summed over clusters."""

    used = np.bincount(np.asarray(route_clusters, dtype=int), minlength=n_clusters)
    minimum = np.ceil(np.asarray(cluster_demands) / capacity).astype(int)

    return int(np.sum(used - minimum))


# ============================================================
# ONE RUN
# ============================================================

def execute(spec, coordinates, demands, capacity, min_vehicles):
    """Run one configuration and return its result dict."""

    if spec["kind"] == "whole":
        return solve_whole_instance(coordinates, demands, capacity, TOTAL_TIME)

    method = spec["clustering"]
    seed = spec["seed"]
    n_customers = len(coordinates) - 1

    n_clusters = cluster_count(n_customers, min_vehicles, spec["sizing"], spec["value"])

    started = time.perf_counter()
    clusters = make_clusters(coordinates, demands, n_clusters, method, seed)
    clustering_s = time.perf_counter() - started

    sizes = [len(cluster) for cluster in clusters]

    cluster_demands = [
        int(demands[np.asarray(cluster) + 1].sum())
        for cluster in clusters
    ]

    time_limits = cluster_time_limits(clusters, demands)

    quality = clustering_quality(coordinates, demands, clusters, 0 if seed is None else seed)

    result = solve_cluster(coordinates, demands, clusters, capacity, time_limits)

    result.update({
        "clusters_requested": n_clusters,
        "clusters_formed": len(clusters),
        "clustering_s": round(clustering_s, 3),
        "route_overhead": count_route_overhead(
            result["route_clusters"], len(clusters), cluster_demands, capacity
        ),
        "customers_min": min(sizes),
        "customers_max": max(sizes),
        "cluster_demand_min": min(cluster_demands),
        "cluster_demand_max": max(cluster_demands),
        "max_cluster_load_pct": round(100 * max(cluster_demands) / capacity, 1),
        **quality,
    })

    return result


def save_outputs(folder, key, instance, spec, coordinates, demands, capacity, result):
    """Routes text, .sol file and optional plot for one successful run."""

    routes = result["routes"]
    distance = result["distance"]
    route_clusters = result["route_clusters"]

    header = [
        f"Run key: {key}",
        f"Instance: {instance}",
        f"Kind: {describe(spec)}",
        f"Budget (s): {TOTAL_TIME}",
        f"Search time (s): {result['search_s']}",
        f"Solver time measured (s): {result['solver_elapsed_s']}",
        f"Overhead measured (s): {result['setup_s']}",
    ]

    if SAVE_PLOTS:
        plot_routes(
            coordinates,
            routes,
            route_clusters if route_clusters is not None else [0] * len(routes),
            result.get("clusters_formed") or 1,
            f"{instance} - {key}\n{len(routes)} routes | distance {distance}",
            folder / f"{key}_routes.png",
            show=SHOW_PLOTS,
            clustered=route_clusters is not None,
        )

    write_routes_txt(
        folder / f"{key}_routes.txt",
        coordinates,
        demands,
        capacity,
        routes,
        distance,
        route_clusters=route_clusters,
        header_lines=header,
    )

    write_sol(folder / f"{key}.sol", routes, distance)


def write_json_atomic(path, record):
    """Write to a temporary file, then rename, so a crash cannot leave a half file."""

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    os.replace(tmp, path)


def already_done(json_path):

    if not json_path.exists():
        return False

    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    return record.get("status") == "ok"


def base_fields(instance, spec, key, started_at):

    return {
        "run_key": key,
        "instance": instance,
        "kind": spec["kind"],
        "clustering": spec["clustering"],
        "sizing": spec["sizing"],
        "sizing_value": spec["value"],
        "run": spec["run"],
        "seed": spec["seed"],
        "total_budget_s": TOTAL_TIME,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.node(),
    }


def run_one(instance, data, spec, key, folder, progress):

    coordinates, demands, capacity, min_vehicles = data
    n_customers = len(coordinates) - 1
    json_path = folder / f"{key}.json"

    if spec["clustering"] == "kmeans":
        spec = {**spec, "seed": random_seed()}

    started_at = datetime.now().isoformat(timespec="seconds")

    stamp(f"{progress} START  {describe(spec)}")

    t0 = time.perf_counter()

    try:
        result = execute(spec, coordinates, demands, capacity, min_vehicles)

        save_outputs(folder, key, instance, spec, coordinates, demands, capacity, result)

        wall = time.perf_counter() - t0

        routes = result["routes"]

        record = {
            "status": "ok",
            **base_fields(instance, spec, key, started_at),
            "time_share_basis": TIME_SHARE_BASIS if spec["kind"] == "clustered" else None,
            "n_customers": n_customers,
            "capacity": capacity,
            "total_demand": int(demands[1:].sum()),
            "min_vehicles": min_vehicles,
            "n_routes": len(routes),
            "routes_per_customer": round(len(routes) / n_customers, 5),
            "wall_s": round(wall, 3),
            "peak_mb_process_so_far": peak_mb(),
            "ailsii_params": AILS_PARAMS,
            "versions": VERSIONS,
            **{k: v for k, v in result.items() if k != "routes"},
            "routes": routes,
        }

        write_json_atomic(json_path, record)

    except Exception as error:

        record = {
            "status": "failed",
            **base_fields(instance, spec, key, started_at),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-2000:],
        }

        write_json_atomic(json_path, record)

        stamp(f"{progress} FAILED {describe(spec)}: {type(error).__name__}: {error}")
        return

    stamp(f"{progress} DONE   {describe(spec)} | {record['wall_s']:.1f} s wall")

    lines = [
        kv("routes", record["n_routes"]),
        kv("distance", record["distance"]),
        kv("search", f"{record['search_s']} s over {record['n_calls']} call(s)"),
        kv("solver time", f"{record['solver_elapsed_s']:.2f} s measured"),
        kv("overhead", f"{record['setup_s']:.2f} s measured"),
    ]

    if spec["kind"] == "clustered":
        lines.append(
            kv("clusters", f"{record['clusters_formed']} formed / "
                           f"{record['clusters_requested']} requested")
        )

    lines += [
        kv("cost check", "ok" if record["cost_ok"] else "MISMATCH"),
        kv("saved as", key),
    ]

    show(lines)


# ============================================================
# ONE INSTANCE
# ============================================================

def run_instance(path, index, count):

    instance = path.stem

    coordinates, demands, capacity = read_vrp(path)
    min_vehicles = calculate_number_of_vehicles(demands[1:], capacity)
    data = (coordinates, demands, capacity, min_vehicles)

    folder = budget_folder() / instance
    folder.mkdir(parents=True, exist_ok=True)

    runs = planned_runs()

    print()
    print(RULE)
    stamp(f"INSTANCE {index}/{count}  {instance}")
    show([
        kv("customers", len(coordinates) - 1),
        kv("total demand", int(demands[1:].sum())),
        kv("capacity", capacity),
        kv("min vehicles", min_vehicles),
        kv("runs", len(runs)),
    ])
    print(RULE)

    for number, spec in enumerate(runs, start=1):

        progress = f"[{number}/{len(runs)}]"
        key = run_key(instance, spec)

        if already_done(folder / f"{key}.json"):
            stamp(f"{progress} SKIP   {describe(spec)} (already done)")
            continue

        run_one(instance, data, spec, key, folder, progress)

    print()
    stamp(f"INSTANCE {index}/{count} finished")


# ============================================================
# MAIN
# ============================================================

def print_settings(n_instances):

    print(RULE)
    stamp("SETTINGS")
    show([
        kv("instances", f"{n_instances} in {INSTANCE_DIR.name}"),
        kv("search budget", f"{TOTAL_TIME} s per run"),
        kv("time share", TIME_SHARE_BASIS),
        kv("runs", ", ".join(f"{m} x{n}" for m, n in RUNS.items() if n > 0)),
        kv("cluster sizes", ", ".join(size_labels()) or "none"),
        kv("runs/instance", len(planned_runs())),
        kv("max cluster", f"{MAX_CLUSTER_CUSTOMERS} customers"),
        kv("ailsii params", str(AILS_PARAMS)),
        kv("output", str(budget_folder())),
    ])
    print(RULE)


RULE_LINE = RULE


def main():

    if "--overview" in sys.argv:
        build_overview(budget_folder(), overview_path())
        return

    check_settings()

    if not planned_runs():
        print("Nothing to run: every entry in RUNS is 0.")
        return

    files = sorted(INSTANCE_DIR.glob("*.vrp"))

    if not files:
        print("No .vrp files found.")
        return

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print_settings(len(files))

    for index, path in enumerate(files, start=1):
        run_instance(path, index, len(files))

    print()
    print(RULE)
    stamp("ALL INSTANCES FINISHED")

    try:
        build_overview(budget_folder(), overview_path())
    except PermissionError:
        print(
            f"Could not write {overview_path()} (is it open in Excel?). "
            f"Close it, then run: python run_benchmark.py --overview"
        )

    print(RULE)


if __name__ == "__main__":
    main()