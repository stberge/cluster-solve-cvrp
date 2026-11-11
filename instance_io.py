"""
Reading VRPLIB-style instances, route statistics, text output and plotting.

Plot style (clustered):
    - each cluster's area is shaded with its convex hull (clear colour)
    - customers are dots coloured by their cluster
    - routes are black with a white halo
    - the legend shows Route / Customer / Depot and a shading key only,
      never one entry per cluster (there can be 50+ clusters)

Plot style (regular HGS):
    - routes black, customers black, no shading
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from scipy.spatial import ConvexHull


# ============================================================
# READ VRP FILE (coordinates, demands, capacity)
# ============================================================

def read_vrp(filepath):
    """
    Read a VRPLIB-style .vrp file.

    Returns:
        coordinates: np.array of shape (n+1, 2), node 0 = depot
        demands:     np.array of shape (n+1,), node 0 = depot (0)
        capacity:    int, vehicle capacity read from the header
    """

    coordinates = []
    demands = []
    capacity = None

    reading_coords = False
    reading_demands = False

    with open(filepath, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("CAPACITY"):

                # Format: "CAPACITY : 30"
                capacity = int(
                    line.split(":")[1].strip()
                )

                continue

            if line == "NODE_COORD_SECTION":
                reading_coords = True
                reading_demands = False
                continue

            if line == "DEMAND_SECTION":
                reading_coords = False
                reading_demands = True
                continue

            if line == "DEPOT_SECTION":
                reading_coords = False
                reading_demands = False
                continue

            if line == "EOF":
                break

            if reading_coords:

                parts = line.split()

                if len(parts) >= 3:

                    coordinates.append([
                        float(parts[1]),
                        float(parts[2])
                    ])

            elif reading_demands:

                parts = line.split()

                if len(parts) >= 2:

                    demands.append(
                        int(parts[1])
                    )

    if capacity is None:
        raise ValueError(
            f"No CAPACITY field found in {filepath}"
        )

    return (
        np.array(coordinates),
        np.array(demands),
        capacity
    )


# ============================================================
# CALCULATE NUMBER OF VEHICLES
# ============================================================

def calculate_number_of_vehicles(demands, capacity):

    total_demand = demands.sum()

    return int(
        np.ceil(total_demand / capacity)
    )


# ============================================================
# ROUTE STATISTICS
# ============================================================

def route_stats(coordinates, demands, routes):
    """
    Load and distance of every route.

    Distance uses the same rounding as the PyVRP distance matrices
    (each leg rounded to the nearest integer), so the sum over all
    routes matches the solver's total distance.

    Returns: list of dicts {"distance": int, "load": int}, one per route
    """

    depot = coordinates[0]

    stats = []

    for route in routes:

        points = (
            [depot]
            + [coordinates[idx + 1] for idx in route]
            + [depot]
        )

        distance = 0

        for a, b in zip(points[:-1], points[1:]):

            distance += int(
                np.round(np.hypot(a[0] - b[0], a[1] - b[1]))
            )

        load = int(
            sum(demands[idx + 1] for idx in route)
        )

        stats.append({
            "distance": distance,
            "load": load
        })

    return stats


# ============================================================
# WRITE ROUTES TO TEXT FILE
# ============================================================

def write_routes_txt(
    output_path,
    coordinates,
    demands,
    capacity,
    routes,
    total_distance,
    route_clusters=None,
    header_lines=()
):
    """
    Write routes as text. Each route line shows its cluster (if any),
    its load against capacity, and its distance.

    Returns the per-route statistics (see route_stats), so the caller
    can check that they add up to total_distance.
    """

    stats = route_stats(coordinates, demands, routes)

    with open(output_path, "w") as f:

        for line in header_lines:
            f.write(f"{line}\n")

        if header_lines:
            f.write("\n")

        f.write(f"Number of routes: {len(routes)}\n")
        f.write(f"Total distance: {total_distance}\n\n")

        f.write("Routes:\n")

        for route_number, (route, route_stat) in enumerate(
            zip(routes, stats),
            start=1
        ):

            customers = " ".join(
                f"C{idx}"
                for idx in route
            )

            cluster_part = ""

            if route_clusters is not None:

                cluster_id = route_clusters[route_number - 1]

                cluster_part = f" (cluster {cluster_id + 1})"

            f.write(
                f"Route {route_number}{cluster_part}: "
                f"load {route_stat['load']}/{capacity}, "
                f"distance {route_stat['distance']}: "
                f"CD0 {customers} CD0\n"
            )

    return stats


# ============================================================
# PLOT HELPERS
# ============================================================

def _group_colors(n):
    """
    n distinct, saturated colours.

    Up to 10 groups: tab10 (very distinct).
    More groups: evenly spaced hues, shuffled so that neighbouring
    cluster ids do not get similar colours.
    """

    if n <= 10:

        return [plt.cm.tab10(i) for i in range(n)]

    hues = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))

    rng = np.random.default_rng(0)

    return hues[rng.permutation(n)]


def _cluster_hull(points):
    """
    Convex hull vertices of a point set, or None when the hull is
    degenerate (fewer than 3 points, or all points collinear).
    """

    if len(points) < 3:
        return None

    try:
        hull = ConvexHull(points)
    except Exception:
        return None

    return points[hull.vertices]


# ============================================================
# PLOT ROUTES
# ============================================================

def plot_routes(
    coordinates,
    routes,
    route_groups,
    n_groups,
    title,
    output_file,
    show=False,
    clustered=True,
):
    """
    Plot solved routes.

    routes:        list of routes (original customer indices)
    route_groups:  one group id per route (cluster id when clustered)
    n_groups:      number of distinct group ids
    clustered:     True  -> shaded cluster areas, coloured customers
                   False -> plain black routes and customers
    show:          open the figure window (blocks until closed)
    """

    depot = coordinates[0]

    colors = _group_colors(n_groups)

    # Customers belonging to each group
    group_customers = {g: [] for g in range(n_groups)}

    for route, group in zip(routes, route_groups):
        group_customers[group].extend(route)

    fig, ax = plt.subplots(figsize=(10, 10))

    # --------------------------------------------------------
    # 1. CLUSTER AREAS (background)
    # --------------------------------------------------------

    if clustered:

        for group in range(n_groups):

            if not group_customers[group]:
                continue

            points = np.array([
                coordinates[idx + 1]
                for idx in group_customers[group]
            ])

            hull_points = _cluster_hull(points)

            if hull_points is None:
                continue

            ax.add_patch(
                Polygon(
                    hull_points,
                    closed=True,
                    facecolor=to_rgba(colors[group], 0.35),
                    edgecolor=to_rgba(colors[group], 1.0),
                    linewidth=2.0,
                    zorder=0
                )
            )

    # --------------------------------------------------------
    # 2. ROUTES (black, with a white halo for contrast)
    # --------------------------------------------------------

    halo = [pe.withStroke(linewidth=2.6, foreground="white")]

    for route in routes:

        x_values = [depot[0]]
        y_values = [depot[1]]

        for idx in route:

            x, y = coordinates[idx + 1]

            x_values.append(x)
            y_values.append(y)

        x_values.append(depot[0])
        y_values.append(depot[1])

        ax.plot(
            x_values,
            y_values,
            color="black",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
            path_effects=halo
        )

    # --------------------------------------------------------
    # 3. CUSTOMERS (coloured by cluster, black outline)
    # --------------------------------------------------------

    if clustered:

        for group in range(n_groups):

            if not group_customers[group]:
                continue

            points = np.array([
                coordinates[idx + 1]
                for idx in group_customers[group]
            ])

            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=50,
                color=colors[group],
                edgecolors="black",
                linewidths=0.8,
                zorder=5
            )

    else:

        all_customers = [
            idx
            for route in routes
            for idx in route
        ]

        if all_customers:

            points = np.array([
                coordinates[idx + 1]
                for idx in all_customers
            ])

            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=30,
                color="black",
                edgecolors="white",
                linewidths=0.5,
                zorder=5
            )

    # --------------------------------------------------------
    # 4. DEPOT
    # --------------------------------------------------------

    ax.scatter(
        depot[0],
        depot[1],
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        color="white",
        zorder=10
    )

    # --------------------------------------------------------
    # 5. LEGEND (fixed entries, never one per cluster)
    # --------------------------------------------------------

    handles = [
        Line2D([0], [0], color="black", linewidth=1.2, label="Route"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markeredgecolor="black",
               markersize=8, label="Customer (colour = cluster)"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="white", markeredgecolor="black",
               markersize=14, label="Depot"),
    ]

    if clustered:

        handles.append(
            Patch(
                facecolor=to_rgba("grey", 0.35),
                edgecolor="grey",
                label="Cluster area (shaded)"
            )
        )

    ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)
