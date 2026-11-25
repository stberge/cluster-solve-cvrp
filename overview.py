"""
Excel overview rebuilt from the per-run JSON files.

The JSON files are the source of truth. run_benchmark rebuilds the overview
after every run, so the spreadsheet always matches the files on disk. If
Excel has the file open, the rebuild is skipped and the next run tries again.

Sheets:
    runs      status "ok": every finished run
    failed    status "failed": the error and traceback of each failed run
    skipped   status "skipped": clustered runs with k = 1 (the whole-instance run covers them)
    plots     route plot of every ok run that has one
"""

import io
import json
import math
from pathlib import Path

import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# Kept in the JSON files, left out of the spreadsheet
DROP_FROM_OVERVIEW = {"routes", "route_clusters", "limits", "trace"}

# Shown first, in this order. Other fields follow alphabetically.
PRIORITY_COLUMNS = [
    "instance", "kind", "clustering", "sizing", "sizing_value", "run", "seed",
    "status", "reason", "distance", "n_routes", "routes_per_customer", "route_overhead",
    "n_customers", "capacity", "min_vehicles", "total_demand",
    "clusters_requested", "clusters_formed", "n_calls",
    "search_s", "clustering_s", "wall_s", "overhead_s", "overhead_max_s",
    "cost_ok", "trace_complete",
    "silhouette", "nearest_centroid_share", "demand_cv", "mean_customer_radius",
    "max_cluster_load_pct", "customers_min", "customers_max",
    "cluster_demand_min", "cluster_demand_max",
    "time_share_basis", "total_budget_s", "peak_mb_process_so_far",
    "error", "run_key", "started_at", "finished_at", "machine",
]

SORT_KEYS = ["instance", "kind", "clustering", "sizing", "sizing_value", "run"]

# (sheet name, status it holds)
STATUS_SHEETS = [("runs", "ok"), ("failed", "failed"), ("skipped", "skipped")]

PLOT_WIDTH_PX = 560
PLOT_COLUMN_SPAN = 10   # columns from the left plot to the right plot
PX_PER_ROW = 20         # default row height in Excel


def _cell(value):
    """Lists and dicts become JSON text so Excel can hold them."""

    if isinstance(value, (dict, list)):
        return json.dumps(value)

    return value


def _read_records(result_root):
    """(record, plot path or None) for every run JSON under result_root."""

    entries = []

    for path in sorted(Path(result_root).rglob("*.json")):

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"WARNING: skipping unreadable file {path.name}: {error}")
            continue

        plot = path.with_name(f"{path.stem}_routes.png")

        entries.append((record, plot if plot.exists() else None))

    return entries


def _table(records):
    """One sheet's worth of rows as a DataFrame, columns and rows in order."""

    rows = [
        {k: _cell(v) for k, v in record.items() if k not in DROP_FROM_OVERVIEW}
        for record in records
    ]

    df = pd.DataFrame(rows)

    first = [c for c in PRIORITY_COLUMNS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in first)
    keys = [k for k in SORT_KEYS if k in df.columns]

    return (
        df[first + rest]
        .sort_values(keys, na_position="first")
        .reset_index(drop=True)
        .round(4)
    )


def _style(sheet):
    """Header style, column widths, frozen header row and filter."""

    header_fill = PatternFill("solid", start_color="1F4E78")

    for index, cell in enumerate(sheet[1], start=1):

        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        width = min(max(len(str(cell.value or "")) + 2, 12), 40)
        sheet.column_dimensions[get_column_letter(index)].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def _add_plots(sheet, plots):
    """Plots in two columns, each labelled with its run key."""

    cursor = [1, 1]

    for index, (key, path) in enumerate(plots):

        side = index % 2
        column = 1 + side * PLOT_COLUMN_SPAN
        row = cursor[side]

        sheet.cell(row=row, column=column, value=key).font = Font(bold=True)

        # BytesIO so the PNG is not kept open on disk
        image = XLImage(io.BytesIO(path.read_bytes()))

        scale = PLOT_WIDTH_PX / image.width
        image.width = PLOT_WIDTH_PX
        image.height = round(image.height * scale)

        sheet.add_image(image, sheet.cell(row=row + 1, column=column).coordinate)

        cursor[side] = row + 1 + math.ceil(image.height / PX_PER_ROW) + 1


def build_overview(result_root, xlsx_path):
    """Rebuild the whole Excel overview from the run JSON files."""

    entries = _read_records(result_root)

    if not entries:
        print("No run files found.")
        return

    by_status = {
        status: [record for record, _ in entries if record.get("status") == status]
        for _, status in STATUS_SHEETS
    }

    plots = [
        (record["run_key"], plot)
        for record, plot in entries
        if record.get("status") == "ok" and plot is not None
    ]

    counts = ", ".join(f"{len(by_status[status])} {status}" for _, status in STATUS_SHEETS)

    print(f"Overview: {counts} -> {Path(xlsx_path).name}")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        for sheet_name, status in STATUS_SHEETS:

            if by_status[status]:
                _table(by_status[status]).to_excel(writer, sheet_name=sheet_name, index=False)
                _style(writer.book[sheet_name])

        if plots:
            _add_plots(writer.book.create_sheet("plots"), plots)