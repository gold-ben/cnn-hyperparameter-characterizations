"""Ten predeclared paired Box-Cox Pearson correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.config.study import ALPHA
from analysis.models.regression import boxcox_transform


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    number_of_tests = int(frame.groupby(["dataset", "optimizer"]).ngroups)
    if number_of_tests != 10:
        raise RuntimeError(f"Expected 10 dataset-by-optimizer correlations; observed {number_of_tests}")
    corrected_alpha = ALPHA / number_of_tests
    for (dataset, optimizer), group in frame.groupby(["dataset", "optimizer"]):
        paired = group[["gaussian_lipschitz", "test_accuracy_10_epoch"]].dropna()
        gaussian, gaussian_lambda = boxcox_transform(paired["gaussian_lipschitz"])
        accuracy, accuracy_lambda = boxcox_transform(paired["test_accuracy_10_epoch"])
        valid = gaussian.notna() & accuracy.notna()
        coefficient, p_value = stats.pearsonr(gaussian.loc[valid], accuracy.loc[valid])
        rows.append({
            "dataset": dataset,
            "optimizer": optimizer,
            "paired_n": int(valid.sum()),
            "gaussian_boxcox_lambda": gaussian_lambda,
            "accuracy_boxcox_lambda": accuracy_lambda,
            "pearson_coefficient": float(coefficient),
            "p_value": float(p_value),
            "correction": "Bonferroni across 10 predeclared correlations",
            "corrected_alpha": corrected_alpha,
            "corrected_significant": bool(p_value < corrected_alpha),
        })
    return pd.DataFrame(rows)

