"""Retrospective model-level power calculations for the primary regressions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from analysis.models.regression import ModelBundle


TARGET_POWER = 0.80
POWER_ALPHA = 0.05
TREATMENT_TESTS = 41
DESIGN_SETTINGS = 80


def _power(sample_size: int, effect_size_f_squared: float, model_columns: int) -> float:
    denominator_df = sample_size - model_columns
    if denominator_df <= 0:
        return 0.0
    critical_value = stats.f.ppf(1.0 - POWER_ALPHA, TREATMENT_TESTS, denominator_df)
    noncentrality = effect_size_f_squared * sample_size
    return float(stats.ncf.sf(
        critical_value,
        TREATMENT_TESTS,
        denominator_df,
        noncentrality,
    ))


def model_power_sample_sizes(models: ModelBundle) -> pd.DataFrame:
    """Estimate complete and attempted runs needed for each omnibus treatment test."""
    rows: list[dict[str, object]] = []
    summaries = models.summaries.set_index("model_id")
    for model_id, diagnostics in models.diagnostics.groupby("model_id", sort=False):
        summary = summaries.loc[model_id]
        y = diagnostics["response_transformed"].to_numpy(dtype=float)
        seed = pd.get_dummies(
            diagnostics["seed"].astype(str), drop_first=True, dtype=float
        )
        reduced = np.column_stack((np.ones(len(diagnostics)), seed.to_numpy(dtype=float)))
        reduced_residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        reduced_sse = float(np.dot(reduced_residual, reduced_residual))
        full_residual = diagnostics["residual_transformed"].to_numpy(dtype=float)
        full_sse = float(np.dot(full_residual, full_residual))
        effect_size = max(0.0, (reduced_sse - full_sse) / full_sse)
        partial_r_squared = effect_size / (1.0 + effect_size)
        model_columns = int(summary["design_columns"])
        theoretical = next(
            sample_size
            for sample_size in range(model_columns + 1, 1_000_001)
            if _power(sample_size, effect_size, model_columns) >= TARGET_POWER
        )
        complete_runs = max(
            DESIGN_SETTINGS,
            DESIGN_SETTINGS * math.ceil(theoretical / DESIGN_SETTINGS),
        )
        completion_rate = float(summary["n_complete"] / summary["n_candidate"])
        attempted_runs = math.ceil(complete_runs / completion_rate)
        rows.append({
            "model_id": model_id,
            "dataset": summary["dataset"],
            "optimizer": summary["optimizer"],
            "response": summary["response"],
            "observed_complete_runs": int(summary["n_complete"]),
            "partial_r_squared": partial_r_squared,
            "cohen_f_squared": effect_size,
            "theoretical_minimum_complete_runs": theoretical,
            "design_compatible_complete_runs": complete_runs,
            "runs_per_design_setting": complete_runs // DESIGN_SETTINGS,
            "observed_completion_rate": completion_rate,
            "estimated_attempted_runs": attempted_runs,
            "power_at_design_compatible_n": _power(
                complete_runs, effect_size, model_columns
            ),
            "target_power": TARGET_POWER,
            "alpha": POWER_ALPHA,
        })
    result = pd.DataFrame(rows)
    if len(result) != 20:
        raise RuntimeError(f"Expected 20 model power rows; observed {len(result)}")
    return result
