"""Optimizer selection over 80 unique settings per dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
import statsmodels.api as sm

from analysis.models.regression import boxcox_transform, treatment_matrix
from analysis.design import FACTOR_COLUMNS, OPTIMIZERS


@dataclass
class MetaResults:
    summaries: pd.DataFrame
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    comparisons: pd.DataFrame
    confusion: pd.DataFrame


def _inverse_boxcox(values: np.ndarray, fitted_lambda: float) -> np.ndarray:
    modeled = values.copy()
    if fitted_lambda > 0:
        modeled = np.maximum(modeled, np.nextafter(-1.0 / fitted_lambda, np.inf))
    elif fitted_lambda < 0:
        modeled = np.minimum(modeled, np.nextafter(-1.0 / fitted_lambda, -np.inf))
    return inv_boxcox(modeled, fitted_lambda)


def optimizer_meta_models(
    frame: pd.DataFrame,
    alias_classes: list[dict[str, object]] | None = None,
) -> MetaResults:
    setting_columns = ["setting_id", "design_slice", *FACTOR_COLUMNS]
    means = frame.groupby(["dataset", "optimizer", *setting_columns], as_index=False)["test_accuracy_10_epoch"].mean()
    summary_rows = []
    coefficient_frames = []
    prediction_frames = []

    for (dataset, optimizer), group in means.groupby(["dataset", "optimizer"]):
        if len(group) != 80:
            raise RuntimeError(f"Expected 80 meta-model settings for {dataset}/{optimizer}; observed {len(group)}")
        x = treatment_matrix(group, alias_classes=alias_classes)
        if np.linalg.matrix_rank(x.to_numpy(dtype=float)) != x.shape[1]:
            raise RuntimeError(f"Meta-model matrix is rank deficient for {dataset}/{optimizer}")
        transformed, fitted_lambda = boxcox_transform(group["test_accuracy_10_epoch"])
        fit = sm.OLS(transformed.to_numpy(dtype=float), x.to_numpy(dtype=float)).fit()
        model_id = f"{dataset.lower()}__{optimizer}__optimizer_selection"
        summary_rows.append({
            "model_id": model_id, "dataset": dataset, "optimizer": optimizer,
            "settings": len(group), "columns": x.shape[1], "rank": int(np.linalg.matrix_rank(x)),
            "residual_degrees_of_freedom": int(fit.df_resid), "r_squared": float(fit.rsquared),
            "adjusted_r_squared": float(fit.rsquared_adj), "boxcox_lambda": fitted_lambda,
            "aggregation": "mean accuracy across seeds for each setting",
        })
        coefficient_frames.append(pd.DataFrame({
            "model_id": model_id, "dataset": dataset, "optimizer": optimizer,
            "term": x.columns, "coefficient": fit.params, "standard_error": fit.bse,
            "test_statistic": fit.tvalues, "p_value": fit.pvalues,
        }))
        predicted = group[setting_columns].copy()
        predicted["dataset"] = dataset
        predicted["optimizer"] = optimizer
        predicted["predicted_accuracy"] = _inverse_boxcox(np.asarray(fit.predict(x)), fitted_lambda)
        predicted["observed_mean_accuracy"] = group["test_accuracy_10_epoch"].to_numpy()
        prediction_frames.append(predicted)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparisons = []
    confusion_rows = []
    for (dataset, setting_id), group in predictions.groupby(["dataset", "setting_id"]):
        predicted_max = group["predicted_accuracy"].max()
        actual_max = group["observed_mean_accuracy"].max()
        predicted_ties = sorted(group.loc[np.isclose(group["predicted_accuracy"], predicted_max, rtol=0, atol=1e-12), "optimizer"])
        actual_ties = sorted(group.loc[np.isclose(group["observed_mean_accuracy"], actual_max, rtol=0, atol=1e-12), "optimizer"])
        selected = predicted_ties[0]
        actual = actual_ties[0]
        selected_actual = float(group.loc[group["optimizer"] == selected, "observed_mean_accuracy"].iloc[0])
        comparisons.append({
            "dataset": dataset, "setting_id": setting_id,
            "selected_optimizer": selected, "actual_best_optimizer": actual,
            "selected_is_actual_best": selected in actual_ties,
            "predicted_tie_count": len(predicted_ties), "actual_tie_count": len(actual_ties),
            "regret": float(actual_max - selected_actual),
        })
        confusion_rows.append((dataset, actual, selected))
    comparison_frame = pd.DataFrame(comparisons)
    summary = comparison_frame.groupby("dataset").agg(
        settings=("setting_id", "count"),
        correct_predictions=("selected_is_actual_best", "sum"),
        selection_accuracy=("selected_is_actual_best", "mean"),
        predicted_ties=("predicted_tie_count", lambda values: int((values > 1).sum())),
        actual_ties=("actual_tie_count", lambda values: int((values > 1).sum())),
        mean_regret=("regret", "mean"), median_regret=("regret", "median"), maximum_regret=("regret", "max"),
    ).reset_index()
    summary["selection_accuracy"] *= 100.0
    confusion_source = pd.DataFrame(confusion_rows, columns=["dataset", "actual", "selected"])
    confusion = (
        confusion_source.groupby(["dataset", "actual", "selected"]).size().rename("count").reset_index()
    )
    expected = set(OPTIMIZERS)
    if set(predictions["optimizer"].unique()) != expected:
        raise RuntimeError("Meta-model optimizer coverage is incomplete")
    return MetaResults(
        summaries=summary,
        coefficients=pd.concat(coefficient_frames, ignore_index=True),
        predictions=predictions,
        comparisons=comparison_frame,
        confusion=confusion,
    )
