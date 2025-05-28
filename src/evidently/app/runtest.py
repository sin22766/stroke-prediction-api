from pathlib import Path
from evidently.ui.workspace import Workspace
from evidently.tests import gt
from evidently import Report
from evidently.metrics import MinValue, RowCount, ColumnCount, DriftedColumnsCount
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently import Dataset
from evidently import DataDefinition, BinaryClassification
from evidently import Report
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot, line_plot_panel


import pandas as pd
from stroke_prediction.config import PROCESSED_DATA_DIR, PROJ_ROOT

ws = Workspace.create(
    str(Path(f"{PROJ_ROOT}\\src\\evidently\\workspace").resolve()))
project = ws.get_project(project_id="0197165d-4206-7a64-8e45-62e1896a1af7")


data_report = Report([
    DriftedColumnsCount(),
    RowCount(),
    ColumnCount(),
], include_tests="True")


data_definition = DataDefinition(
    classification=[BinaryClassification(
        target="stroke", prediction_labels="prediction")]
)

# get data
reff_data = pd.read_parquet(PROCESSED_DATA_DIR / "val-stroke-data.parquet")
curr_data = pd.read_parquet(PROCESSED_DATA_DIR / "test-stroke-data.parquet")

# Run comparison
run = data_report.run(curr_data, reff_data)
ws.add_run(project.id, run)
