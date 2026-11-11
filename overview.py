"""
Excel overview of all runs.

One sheet ("runs"), one row per run. Running the same configuration
again replaces its row instead of adding a duplicate. A configuration
is identified by KEY_COLUMNS (instance, method, algorithm, clusters
requested, total HGS time and seed).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


SHEET_NAME = "runs"

KEY_COLUMNS = [
    "instance",
    "method",
    "algorithm",
    "n_clusters_requested",
    "total_hgs_time_s",
    "seed",
]

INTEGER_COLUMNS = [
    "n_clusters_requested",
    "n_clusters_formed",
    "num_routes",
    "total_distance",
    "min_cluster_demand",
    "max_cluster_demand",
    "hgs_time_per_cluster_s",
    "total_hgs_time_s",
    "seed",
]

COLUMNS = [
    "last_run",
    "instance",
    "method",
    "algorithm",
    "n_clusters_requested",
    "n_clusters_formed",
    "num_routes",
    "total_distance",
    "min_cluster_demand",
    "max_cluster_demand",
    "max_cluster_load_pct",
    "hgs_time_per_cluster_s",
    "total_hgs_time_s",
    "seed",
    "output_folder",
    "file_name",
]


def _tidy(df):
    """Make integer columns integers (no '.0') and fill missing text."""

    for column in INTEGER_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).astype("Int64")

    return df


def update_overview(path, rows):
    """
    Add the given run rows to the Excel overview at `path`.

    rows: list of dicts, keys matching COLUMNS (missing keys are blank).
    Existing rows with the same KEY_COLUMNS values are replaced.
    """

    path = Path(path)

    new_rows = pd.DataFrame(rows)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    new_rows["last_run"] = stamp

    new_rows = new_rows.reindex(columns=COLUMNS)

    if path.exists():

        old_rows = pd.read_excel(path, sheet_name=SHEET_NAME)

        combined = pd.concat(
            [old_rows, new_rows],
            ignore_index=True
        )

    else:

        combined = new_rows

    combined = _tidy(combined)

    # Keep the last occurrence (the run that just finished)
    combined = combined.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="last"
    )

    combined = combined.sort_values(
        by=["instance", "method", "n_clusters_requested", "algorithm", "total_hgs_time_s"],
        na_position="last"
    )

    combined = combined.reindex(columns=COLUMNS)

    try:

        with pd.ExcelWriter(path, engine="openpyxl") as writer:

            combined.to_excel(
                writer,
                sheet_name=SHEET_NAME,
                index=False
            )

    except PermissionError:

        raise PermissionError(
            f"Could not write {path}. Close it in Excel and run again."
        )

    _format_sheet(path)

    return path


def _format_sheet(path):
    """Header style, column widths, freeze header, filter."""

    workbook = load_workbook(path)
    sheet = workbook[SHEET_NAME]

    header_fill = PatternFill("solid", start_color="1F4E78")

    for cell in sheet[1]:

        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for index, column in enumerate(COLUMNS, start=1):

        width = max(len(column) + 2, 12)

        if column in ("output_folder",):
            width = 70
        elif column in ("file_name",):
            width = 45
        elif column == "last_run":
            width = 18

        sheet.column_dimensions[get_column_letter(index)].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    workbook.save(path)
