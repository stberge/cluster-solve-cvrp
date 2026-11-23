"""
Excel overview built from the per-run JSON files.

The JSON files are the source of truth. The overview is rebuilt from them,
so a locked Excel file cannot lose results.

Sheets:
    runs     every run with status "ok"
    failed   every run that raised an error, with the message and traceback
"""

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# Kept in the JSON files, left out of the spreadsheet
DROP_FROM_OVERVIEW = {"routes", "route_clusters", "limits"}

# Shown first, in this order. Other fields follow alphabetically.
# Kept in the JSON files, left out of the spreadsheet
DROP_FROM_OVERVIEW = {"routes", "route_clusters", "limits"}

# Shown first, in this order. Other fields follow alphabetically.
PRIORITY_COLUMNS = [
    "instance", "kind", "clustering", "sizing", "sizing_value", "run", "seed",
    "status", "distance", "n_routes", "routes_per_customer", "route_overhead",
    "n_customers", "capacity", "min_vehicles", "total_demand",
    "clusters_requested", "clusters_formed", "n_calls",
    "search_s", "clustering_s", "wall_s",
    "cost_ok", "silhouette", "nearest_centroid_share", "demand_cv",
    "mean_customer_radius", "max_cluster_load_pct",
    "customers_min", "customers_max", "cluster_demand_min", "cluster_demand_max",
    "time_share_basis", "total_budget_s", "peak_mb_process_so_far",
    "error", "run_key", "started_at", "finished_at", "machine",
]

SORT_KEYS = ["instance", "kind", "clustering", "sizing", "sizing_value", "run"]


def _cell(value):
    """Lists and dicts become JSON text so Excel can hold them."""

    if isinstance(value, (dict, list)):
        return json.dumps(value)

    return value


def _read_rows(result_root):

    ok_rows = []
    failed_rows = []

    for path in sorted(Path(result_root).rglob("*.json")):

        record = json.loads(path.read_text(encoding="utf-8"))

        row = {k: _cell(v) for k, v in record.items() if k not in DROP_FROM_OVERVIEW}

        if record.get("status") == "ok":
            ok_rows.append(row)
        else:
            failed_rows.append(row)

    return ok_rows, failed_rows


def _ordered(df):

    first = [c for c in PRIORITY_COLUMNS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in first)

    return df[first + rest]


def _sorted(df):

    keys = [k for k in SORT_KEYS if k in df.columns]

    return df.sort_values(keys, na_position="first").reset_index(drop=True) if keys else df


def build_overview(result_root, xlsx_path):
    """Rebuild the Excel overview from every run JSON under result_root."""

    xlsx_path = Path(xlsx_path)

    ok_rows, failed_rows = _read_rows(result_root)

    print(f"Overview: {len(ok_rows)} ok, {len(failed_rows)} failed -> {xlsx_path}")

    if not ok_rows and not failed_rows:
        print("No run files found.")
        return

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        if ok_rows:
            runs = _sorted(_ordered(pd.DataFrame(ok_rows))).round(4)
            runs.to_excel(writer, sheet_name="runs", index=False)

        if failed_rows:
            failed = _ordered(pd.DataFrame(failed_rows))
            failed.to_excel(writer, sheet_name="failed", index=False)

    _format(xlsx_path)


def _format(path):
    """Header style, column widths, frozen header row and filter on every sheet."""

    workbook = load_workbook(path)

    header_fill = PatternFill("solid", start_color="1F4E78")

    for sheet in workbook.worksheets:

        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for index, cell in enumerate(sheet[1], start=1):
            name = str(cell.value or "")
            width = min(max(len(name) + 2, 12), 40)
            sheet.column_dimensions[get_column_letter(index)].width = width

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    workbook.save(path)