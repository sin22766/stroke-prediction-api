from pathlib import Path
from evidently.ui.workspace import Workspace
from evidently.tests import gt
from evidently import Report
from evidently.metrics import MinValue
from evidently.presets import ClassificationPreset
from evidently import Dataset
from evidently import DataDefinition, BinaryClassification
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot, line_plot_panel

import pandas as pd
from stroke_prediction.config import PROCESSED_DATA_DIR, PROJ_ROOT

ws = Workspace.create(str(Path(f"{PROJ_ROOT}\\src\\evidently\\workspace").resolve()))

project = ws.create_project("Stroke Prediction")

project.description = "Stroke Prediction Monitoring"


# Create dashboard panel
project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Dashboard",
        size="full",
        values=[],  # leave empty
        plot_params={"plot_type": "text"},
    ),
    tab="My new tab",  # will create a Tab if there is no Tab with this name
)
# Sum
project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Row count",
        subtitle="Total number of evaluations over time.",
        size="half",
        values=[PanelMetric(legend="Row count", metric="RowCount")],
        plot_params={"plot_type": "counter", "aggregation": "sum"},
    ),
    tab="My tab",
)
# Sum
project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Column count",
        subtitle="Total column number of evaluations over time.",
        size="half",
        values=[PanelMetric(legend="Column count", metric="ColumnCount")],
        plot_params={"plot_type": "counter", "aggregation": "sum"},
    ),
    tab="My tab",
)

ws.update_project(project)  # or project.save()

