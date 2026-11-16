"""
Reading VRPLIB-style instances, route statistics, text and .sol output, and plotting.

Clustered plots shade each cluster's area and colour its customers.
Routes are black in both clustered and whole-instance plots.
"""

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull


# ============================================================
# READ VRP FILE
# ============================================================

def read_vrp(filepath):
    """
    Read a VRPLIB-style .vrp file.

    Returns coordinates (n+1, 2), demands (n+1,) and capacity.
    Node 0 is the depot.
    """

    coordinates = []
    demands = []
    capacity = None
    section = None

    with open(filepath) as f:

        for raw_line in f:

            line = raw_line.strip()

            if line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1])

            elif line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
                section = line

            elif line == "EOF":
                break

            elif section == "NODE_COORD_SECTION":

                parts = line.split()

                if len(parts) >= 3:
                    coordinates.append([float(parts[1]), float(parts[2])])

            elif section == "DEMAND_SECTION":

                parts = line.split()

                if len(parts) >= 2:
                    demands.append(int(parts[1]))

    if capacity is None:
        raise ValueError(f"No CAPACITY field found in {filepath}")

    return np.array(coordinates), np.array(demands), capacity


def calculate_number_of_vehicles(demands, capacity):

    return int(np.ceil(demands.sum() / capacity))


# ============================================================
# ROUTE STATISTICS
# ============================================================

def _leg_distance(a, b):
    """Euclidean distance rounded to the nearest integer, as in the solver."""

    return int(np.round(np.hypot(a[0] - b[0], a[1] - b[1])))


def route_stats(coordinates, demands, routes):
    """
    Load and distance of every route. Distances use the same rounding as
    the solver, so the sum over routes matches the solver total.

    Returns a list of {"distance": int, "load": int}, one per route.
    """

    depot = coordinates[0]

    stats = []

    for route in routes:

        stops = [depot, *(coordinates[i + 1] for i in route), depot]

        distance = sum(
            _leg_distance(a, b)
            for a, b in zip(stops[:-1], stops[1:])
        )

        load = int(sum(demands[i + 1] for i in route))

        stats.append({"distance": distance, "load": load})

    return stats


# ============================================================
# TEXT OUTPUT
# ============================================================

def write_routes_txt(
    output_path,
    coordinates,
    demands,
    capacity,
    routes,
    total_distance,
    route_clusters=None,
    header_lines=(),
):
    """
    Write routes as text: one line per route with load, distance and,
    if given, its cluster. Returns the per-route statistics.
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

        for number, (route, stat) in enumerate(zip(routes, stats), start=1):

            customers = " ".join(f"C{i}" for i in route)

            cluster_part = ""

            if route_clusters is not None:
                cluster_part = f" (cluster {route_clusters[number - 1] + 1})"

            f.write(
                f"Route {number}{cluster_part}: "
                f"load {stat['load']}/{capacity}, "
                f"distance {stat['distance']}: "
                f"CD0 {customers} CD0\n"
            )

    return stats


# ============================================================
# .SOL OUTPUT (reference format)
# ============================================================

def write_sol(output_path, routes, total_distance):
    """
    Write routes in the reference .sol format:

        Route #1: 9280 6557 1033
        ...
        Cost 4076853

    Customers are numbered 1..n, so 0-based index idx is written as idx + 1.
    """

    with open(output_path, "w") as f:

        for number, route in enumerate(routes, start=1):

            customers = " ".join(str(i + 1) for i in route)

            f.write(f"Route #{number}: {customers}\n")

        f.write(f"Cost {int(round(total_distance))}\n")


# ============================================================
# PLOTTING
# ============================================================

def _group_colors(n):
    """
    n distinct colours. Up to 10 groups use tab10. More groups use evenly
    spaced hues, shuffled so neighbouring cluster ids look different.
    """

    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]

    hues = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))

    return hues[np.random.default_rng(0).permutation(n)]


def _cluster_hull(points):
    """Convex hull vertices, or None if the hull is degenerate."""

    if len(points) < 3:
        return None

    try:
        hull = ConvexHull(points)
    except Exception:
        return None

    return points[hull.vertices]


def _points(coordinates, indices):

    return np.array([coordinates[i + 1] for i in indices])


def _members_by_group(routes, route_groups, n_groups):

    members = {group: [] for group in range(n_groups)}

    for route, group in zip(routes, route_groups):
        members[group].extend(route)

    return members


def _draw_cluster_areas(ax, coordinates, members, colors):

    for group, indices in members.items():

        if not indices:
            continue

        hull = _cluster_hull(_points(coordinates, indices))

        if hull is None:
            continue

        ax.add_patch(
            Polygon(
                hull,
                closed=True,
                facecolor=to_rgba(colors[group], 0.35),
                edgecolor=to_rgba(colors[group], 1.0),
                linewidth=2.0,
                zorder=0,
            )
        )


def _draw_routes(ax, coordinates, routes):

    depot = coordinates[0]

    halo = [pe.withStroke(linewidth=2.6, foreground="white")]

    for route in routes:

        stops = [depot, *(coordinates[i + 1] for i in route), depot]

        xs, ys = zip(*stops)

        ax.plot(
            xs,
            ys,
            color="black",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
            path_effects=halo,
        )


def _draw_customers_by_cluster(ax, coordinates, members, colors):

    for group, indices in members.items():

        if not indices:
            continue

        pts = _points(coordinates, indices)

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=50,
            color=colors[group],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )


def _draw_customers_plain(ax, coordinates, routes):

    indices = [i for route in routes for i in route]

    if not indices:
        return

    pts = _points(coordinates, indices)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=30,
        color="black",
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )


def _draw_depot(ax, depot):

    ax.scatter(
        depot[0],
        depot[1],
        s=260,
        marker="*",
        color="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )


def _add_legend(ax, clustered):

    handles = [
        Line2D([0], [0], color="black", linewidth=1.2, label="Route"),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="grey", markeredgecolor="black",
            markersize=8, label="Customer (colour = cluster)" if clustered else "Customer",
        ),
        Line2D(
            [0], [0], marker="*", color="w",
            markerfacecolor="white", markeredgecolor="black",
            markersize=14, label="Depot",
        ),
    ]

    if clustered:
        handles.append(
            Patch(
                facecolor=to_rgba("grey", 0.35),
                edgecolor="grey",
                label="Cluster area (shaded)",
            )
        )

    ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.9)


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
    Plot solved routes and save the figure.

    routes:        list of routes (0-based customer indices)
    route_groups:  one group id per route (cluster id when clustered)
    n_groups:      number of distinct group ids
    clustered:     True shades cluster areas and colours customers by cluster;
                   False draws all customers in black
    show:          open the figure window (blocks until closed)
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    if clustered:

        colors = _group_colors(n_groups)

        members = _members_by_group(routes, route_groups, n_groups)

        _draw_cluster_areas(ax, coordinates, members, colors)

    _draw_routes(ax, coordinates, routes)

    if clustered:
        _draw_customers_by_cluster(ax, coordinates, members, colors)
    else:
        _draw_customers_plain(ax, coordinates, routes)

    _draw_depot(ax, coordinates[0])

    _add_legend(ax, clustered)

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)