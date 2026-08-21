#!/usr/bin/env python3
"""Reproduce the statistical results and reporting artifacts in the manuscript."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analysis.config.study import OUTPUT_ROOT
from analysis.correlations.analyze import correlations
from analysis.data import load_analysis_data
from analysis.design import FACTOR_COLUMNS, generate_center_design, generate_factorial_design
from analysis.diagnostics.design import alias_structure, missingness_summary
from analysis.meta_models.optimizer_selection import optimizer_meta_models
from analysis.models.regression import fit_primary_models
from analysis.reporting.outputs import write_figures, write_primary_tables


def main() -> None:
    for directory in ("data", "tables", "figures", "diagnostics", "latex_tables"):
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)

    frame = load_analysis_data()
    frame.to_csv(OUTPUT_ROOT / "data" / "nine_factor_analysis_dataset.csv", index=False)
    factorial = pd.DataFrame(generate_factorial_design())
    factorial = factorial[["setting_id", *FACTOR_COLUMNS]]
    factorial.to_csv(
        OUTPUT_ROOT / "data" / "nine_factor_factorial_design.csv", index=False
    )
    centers = pd.DataFrame(generate_center_design())
    centers = centers[["setting_id", *FACTOR_COLUMNS]]
    centers.to_csv(
        OUTPUT_ROOT / "data" / "nine_factor_center_design.csv", index=False
    )
    alias_structure().to_csv(
        OUTPUT_ROOT / "data" / "nine_factor_alias_structure.csv", index=False
    )

    models = fit_primary_models(frame)
    correlation_table = correlations(frame)
    meta = optimizer_meta_models(frame)
    missingness = missingness_summary(frame)
    write_primary_tables(models, correlation_table, meta, missingness)
    write_figures(frame, models, correlation_table, meta)

    if len(models.summaries) != 20 or len(correlation_table) != 10:
        raise RuntimeError("Unexpected manuscript analysis result count")
    accuracy = meta.summaries.set_index("dataset")["selection_accuracy"]
    if abs(float(accuracy["CIFAR10"]) - 67.5) > 1e-12:
        raise RuntimeError("Unexpected CIFAR10 optimizer-selection accuracy")
    if abs(float(accuracy["CIFAR100"]) - 66.25) > 1e-12:
        raise RuntimeError("Unexpected CIFAR100 optimizer-selection accuracy")

    print("Nine-factor manuscript analysis reproduced successfully.")
    print(f"Observations: {len(frame)}")
    print(f"Primary models: {len(models.summaries)}")
    print(f"Correlations: {len(correlation_table)}")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
