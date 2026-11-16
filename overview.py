"""
Excel overview of all runs.

One sheet ("runs"), one row per run, identified by run_id. Rows are
appended and never merged, so repeating a configuration adds another row.

The overview holds settings, clustering measures and the distance result.
Route details (route count, route overhead) are only in each run's
_routes.txt and _clusters.csv files.

Overviews written by earlier versions are converted when read. Their
num_routes and route_overhead columns are dropped when the file is rewritten.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


SHEET_NAME = "runs"

INTEGER_COLUMNS = [
    "target_routes_per_cluster",
    "target_customers_per_cluster",
    "n_clusters_requested",
    "n_clusters_formed",
    "min_cluster_customers",
    "max_cluster_customers",
    "min_cluster_demand",
    "max_cluster_demand",
    "total_distance",
    "total_time_s",
    "seed",
]

COLUMNS = [
    "run_id",
    "last_run",
    "instance",
    "method",
    "algorithm",
    "target_routes_per_cluster",     # older runs only
    "target_customers_per_cluster",  # target m
    "time_share_basis",
    "n_clusters_requested",
    "n_clusters_formed",
    "min_cluster_customers",
    "max_cluster_customers",
    "min_cluster_demand",
    "max_cluster_demand",
    "max_cluster_load_pct",
    "demand_cv",
    "nearest_centroid_share",
    "silhouette",
    "mean_customer_radius",
    "total_distance",
    "avg_time_per_cluster_s",
    "total_time_s",
    "seed",
    "output_folder",
    "file_name",
]

# Column names used by earlier versions. Applied in two steps, so the
# old time column ends up under its current name.
LEGACY_COLUMNS_STEP_1 = {
    "hgs_time_per_cluster_s": "time_per_cluster_s",
    "total_hgs_time_s": "total_time_s",
}

LEGACY_COLUMNS_STEP_2 = {
    "time_per_cluster_s": "avg_time_per_cluster_s",
}

LEGACY_METHODS = {
    "clustered_hgs": "clustered",
    "regular_hgs": "whole_instance",
}

COLUMN_WIDTHS = {
    "output_folder": 70,
    "file_name": 55,
    "last_run": 20,
    "run_id": 20,
}


def update_overview(path, rows):
    """
    Append rows (dicts keyed by COLUMNS; missing keys are left blank)
    to the Excel overview at path. Every row is kept.
    """

    path = Path(path)

    new_rows = pd.DataFrame(rows)

    new_rows["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_rows = new_rows.reindex(columns=COLUMNS)

    if path.exists():
        combined = pd.concat([_load_existing(path), new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = _tidy(combined).reindex(columns=COLUMNS)

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name=SHEET_NAME, index=False)

    except PermissionError as error:
        raise PermissionError(
            f"Could not write {path}. Close it in Excel and run again."
        ) from error

    _format_sheet(path)

    return path


def _load_existing(path):
    """Read the current overview, converting legacy column and method names."""

    old_rows = pd.read_excel(path, sheet_name=SHEET_NAME)

    old_rows = old_rows.rename(columns=LEGACY_COLUMNS_STEP_1)
    old_rows = old_rows.rename(columns=LEGACY_COLUMNS_STEP_2)

    if "method" in old_rows.columns:
        old_rows["method"] = old_rows["method"].replace(LEGACY_METHODS)

    return old_rows


def _tidy(df):
    """Integer columns become integers (no '.0'). Missing values stay blank."""

    for column in INTEGER_COLUMNS:

        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    return df


def _format_sheet(path):
    """Header style, column widths, frozen header row and filter."""

    workbook = load_workbook(path)
    sheet = workbook[SHEET_NAME]

    header_fill = PatternFill("solid", start_color="1F4E78")

    for cell in sheet[1]:

        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for index, column in enumerate(COLUMNS, start=1):

        width = COLUMN_WIDTHS.get(column, max(len(column) + 2, 12))

        sheet.column_dimensions[get_column_letter(index)].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    workbook.save(path)