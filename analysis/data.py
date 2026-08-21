"""Load and validate the two canonical manuscript input files."""

from __future__ import annotations

import pandas as pd

from analysis.config.study import DATA_FILES
from analysis.design import FACTOR_COLUMNS, OPTIMIZERS, generate_design


INPUT_COLUMNS = [
    "observation_id",
    "setting_id",
    "dataset",
    "optimizer",
    "seed",
    "design_slice",
    *FACTOR_COLUMNS,
    "gaussian_lipschitz",
    "test_accuracy_10_epoch",
]


def load_analysis_data() -> pd.DataFrame:
    frames = []
    for dataset, path in DATA_FILES.items():
        frame = pd.read_json(path, lines=True, precise_float=True)
        if list(frame.columns) != INPUT_COLUMNS:
            raise RuntimeError(f"Unexpected columns in {path.name}")
        if len(frame) != 4000:
            raise RuntimeError(f"Expected 4,000 rows in {path.name}; observed {len(frame)}")
        if not frame["dataset"].eq(dataset).all():
            raise RuntimeError(f"Dataset labels do not match {path.name}")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 8000 or not result["observation_id"].is_unique:
        raise RuntimeError("The combined analysis input must contain 8,000 unique observations")
    if set(result["optimizer"]) != set(OPTIMIZERS):
        raise RuntimeError("Optimizer coverage is incomplete")
    if result.groupby(["dataset", "optimizer"]).size().ne(800).any():
        raise RuntimeError("Each dataset and optimizer must contain 800 observations")
    if result.groupby(["dataset", "optimizer", "setting_id"]).size().ne(10).any():
        raise RuntimeError("Each setting must have ten seed observations")

    expected = {
        tuple(int(row[factor]) for factor in FACTOR_COLUMNS)
        for row in generate_design()
    }
    observed = {
        tuple(int(row[factor]) for factor in FACTOR_COLUMNS)
        for _, row in result.drop_duplicates(["setting_id"])[FACTOR_COLUMNS].iterrows()
    }
    if observed != expected:
        raise RuntimeError("Input settings do not match the nine-factor design")
    return result
