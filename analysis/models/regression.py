"""Box-Cox OLS models with estimable Resolution-IV interactions and joint curvature."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from analysis.config.study import ALPHA, RESPONSES
from analysis.design import FACTOR_COLUMNS
from analysis.design import generate_design, generate_factorial_design


@dataclass
class ModelBundle:
    coefficients: pd.DataFrame
    summaries: pd.DataFrame
    diagnostics: pd.DataFrame
    center_tests: pd.DataFrame
    boxcox_parameters: pd.DataFrame
    vifs: pd.DataFrame
    model_matrices: pd.DataFrame


def _effect_signature(
    first: str,
    second: str,
    rows: list[dict[str, int | str]],
) -> tuple[int, ...]:
    values = tuple(int(row[first]) * int(row[second]) for row in rows)
    negated = tuple(-value for value in values)
    return min(values, negated)


def _interaction_groups(
    rows: list[dict[str, int | str]],
) -> list[list[tuple[str, str]]]:
    groups: dict[tuple[int, ...], list[tuple[str, str]]] = {}
    for first, second in itertools.combinations(FACTOR_COLUMNS, 2):
        groups.setdefault(_effect_signature(first, second, rows), []).append((first, second))
    return sorted(
        groups.values(),
        key=lambda pairs: tuple(FACTOR_COLUMNS.index(name) for name in pairs[0]),
    )


def factorial_interaction_alias_classes() -> list[dict[str, object]]:
    """Return the 30 two-factor classes defined by the 64-run fraction."""
    classes = []
    for number, pairs in enumerate(_interaction_groups(generate_factorial_design()), start=1):
        labels = [f"{first}:{second}" for first, second in pairs]
        classes.append({
            "alias_class_id": f"2FI-{number:02d}",
            "representative": labels[0],
            "members": labels,
            "aliased": len(labels) > 1,
        })
    if len(classes) != 30:
        raise RuntimeError(f"Expected 30 factorial two-factor classes; observed {len(classes)}")
    return classes


def interaction_alias_classes() -> list[dict[str, object]]:
    """Return the 31 two-factor classes estimable from all 80 design settings."""
    augmented_groups = {
        frozenset(f"{first}:{second}" for first, second in pairs): pairs
        for pairs in _interaction_groups(generate_design())
    }
    classes = []
    for factorial_class in factorial_interaction_alias_classes():
        factorial_members = list(factorial_class["members"])
        subdivisions = []
        for members, pairs in augmented_groups.items():
            overlap = members.intersection(factorial_members)
            if overlap:
                subdivisions.append((min(factorial_members.index(item) for item in overlap), pairs))
        subdivisions.sort(key=lambda item: item[0])
        split = len(subdivisions) > 1
        for subdivision_number, (_, pairs) in enumerate(subdivisions):
            labels = [f"{first}:{second}" for first, second in pairs]
            suffix = chr(ord("a") + subdivision_number) if split else ""
            classes.append({
                "alias_class_id": f"{factorial_class['alias_class_id']}{suffix}",
                "representative": labels[0],
                "members": labels,
                "aliased": len(labels) > 1,
                "alias_scope": "augmented_80_setting_design",
                "source_factorial_alias_class": factorial_class["alias_class_id"],
                "factorial_members": factorial_members,
                "split_by_center_slice": split,
            })
    if len(classes) != 31:
        raise RuntimeError(f"Expected 31 augmented-design two-factor classes; observed {len(classes)}")
    if sum(bool(item["aliased"]) for item in classes) != 5:
        raise RuntimeError("Expected five paired aliases after center-slice augmentation")
    return classes


def treatment_matrix(
    frame: pd.DataFrame,
    alias_classes: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    if len(FACTOR_COLUMNS) != 9:
        raise RuntimeError("Exactly nine experimental factors are required")
    selected_classes = interaction_alias_classes() if alias_classes is None else alias_classes
    matrix = pd.DataFrame(index=frame.index)
    matrix["Intercept"] = 1.0
    for factor in FACTOR_COLUMNS:
        matrix[factor] = pd.to_numeric(frame[factor], errors="raise").astype(float)
    for alias_class in selected_classes:
        first, second = str(alias_class["representative"]).split(":")
        matrix[str(alias_class["representative"])] = matrix[first] * matrix[second]
    matrix["joint_center_departure"] = frame["design_slice"].eq("center").astype(float)
    expected_columns = 11 + len(selected_classes)
    if matrix.shape[1] != expected_columns:
        raise RuntimeError(
            f"Expected {expected_columns} treatment columns; observed {matrix.shape[1]}"
        )
    return matrix


def model_matrix(
    frame: pd.DataFrame,
    alias_classes: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    treatment = treatment_matrix(frame, alias_classes=alias_classes)
    seed_dummies = pd.get_dummies(frame["seed"].astype(str), prefix="seed_block", drop_first=True, dtype=float)
    matrix = pd.concat([treatment, seed_dummies.set_axis(frame.index)], axis=1)
    rank = int(np.linalg.matrix_rank(matrix.to_numpy(dtype=float)))
    if rank != matrix.shape[1]:
        raise RuntimeError(f"Non-estimable model matrix: rank={rank}, columns={matrix.shape[1]}")
    return matrix, list(treatment.columns)


def boxcox_transform(values: pd.Series) -> tuple[pd.Series, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.gt(0)
    transformed = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.sum() < 2:
        return transformed, np.nan
    if numeric.loc[valid].nunique() == 1:
        transformed.loc[valid] = numeric.loc[valid] - 1.0
        return transformed, 1.0
    result, fitted_lambda = stats.boxcox(numeric.loc[valid].to_numpy(dtype=float))
    transformed.loc[valid] = result
    return transformed, float(fitted_lambda)


def _vif(matrix: pd.DataFrame, treatment_columns: list[str]) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    rows = []
    for index, term in enumerate(matrix.columns):
        if term == "Intercept" or term not in treatment_columns:
            continue
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        residual = target - others @ np.linalg.lstsq(others, target, rcond=None)[0]
        sse = float(np.dot(residual, residual))
        sst = float(np.dot(target - target.mean(), target - target.mean()))
        value = np.inf if sse <= np.finfo(float).eps * max(1.0, sst) else max(1.0, sst / sse)
        rows.append({"term": term, "variance_inflation_factor": value})
    return pd.DataFrame(rows)


def fit_primary_models(frame: pd.DataFrame) -> ModelBundle:
    coefficient_frames = []
    summary_rows = []
    diagnostic_frames = []
    center_rows = []
    lambda_rows = []
    vif_frames = []
    matrix_rows = []

    for dataset in sorted(frame["dataset"].unique()):
        for optimizer in sorted(frame["optimizer"].unique()):
            group = frame.loc[(frame["dataset"] == dataset) & (frame["optimizer"] == optimizer)]
            for response in RESPONSES:
                transformed, fitted_lambda = boxcox_transform(group[response])
                valid = transformed.notna()
                model_frame = group.loc[valid].copy()
                y = transformed.loc[valid].to_numpy(dtype=float)
                x, treatment_columns = model_matrix(model_frame)
                model_id = f"{dataset.lower()}__{optimizer}__{response}"
                fit = sm.OLS(y, x.to_numpy(dtype=float)).fit()
                if int(fit.df_resid) <= 0:
                    raise RuntimeError(f"No residual degrees of freedom for {model_id}")
                treatment_tests = len(treatment_columns) - 1
                bonferroni_alpha = ALPHA / treatment_tests
                coefficients = pd.DataFrame({
                    "model_id": model_id,
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "response": response,
                    "term": x.columns,
                    "coefficient": fit.params,
                    "standard_error": fit.bse,
                    "test_statistic": fit.tvalues,
                    "p_value": fit.pvalues,
                })
                coefficients["term_role"] = np.select(
                    [
                        coefficients["term"].eq("Intercept"),
                        coefficients["term"].str.startswith("seed_block_"),
                        coefficients["term"].eq("joint_center_departure"),
                        coefficients["term"].isin([
                            item["representative"]
                            for item in interaction_alias_classes()
                            if item["aliased"]
                        ]),
                        coefficients["term"].str.contains(":", regex=False),
                    ],
                    [
                        "intercept", "seed_block", "joint_curvature",
                        "two_factor_alias_class", "two_factor_interaction",
                    ],
                    default="first_order",
                )
                alias_by_representative = {
                    item["representative"]: item for item in interaction_alias_classes()
                }
                coefficients["augmented_alias_class"] = coefficients["term"].map(
                    lambda term: alias_by_representative.get(term, {}).get("alias_class_id", "")
                )
                coefficients["augmented_alias_members"] = coefficients["term"].map(
                    lambda term: ";".join(alias_by_representative.get(term, {}).get("members", []))
                )
                coefficients["paired_two_factor_alias"] = coefficients["term"].map(
                    lambda term: bool(alias_by_representative.get(term, {}).get("aliased", False))
                )
                coefficients["factorial_alias_split_by_center_slice"] = coefficients["term"].map(
                    lambda term: bool(alias_by_representative.get(term, {}).get("split_by_center_slice", False))
                )
                coefficients["bonferroni_alpha"] = bonferroni_alpha
                coefficients["bonferroni_significant"] = (
                    coefficients["term_role"].isin([
                        "first_order", "two_factor_alias_class",
                        "two_factor_interaction", "joint_curvature",
                    ])
                    & coefficients["p_value"].lt(bonferroni_alpha)
                )
                coefficient_frames.append(coefficients)

                summary_rows.append({
                    "model_id": model_id,
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "response": response,
                    "n_candidate": len(group),
                    "n_complete": int(valid.sum()),
                    "n_missing": int((~valid).sum()),
                    "boxcox_lambda": fitted_lambda,
                    "design_columns": x.shape[1],
                    "design_rank": int(np.linalg.matrix_rank(x.to_numpy(dtype=float))),
                    "residual_degrees_of_freedom": int(fit.df_resid),
                    "r_squared": float(fit.rsquared),
                    "adjusted_r_squared": float(fit.rsquared_adj),
                    "omnibus_f_statistic": float(fit.fvalue),
                    "omnibus_p_value": float(fit.f_pvalue),
                    "significant_treatment_terms": int(coefficients["bonferroni_significant"].sum()),
                    "bonferroni_alpha": bonferroni_alpha,
                })
                lambda_rows.append({
                    "model_id": model_id, "dataset": dataset, "optimizer": optimizer,
                    "response": response, "boxcox_lambda": fitted_lambda, "n": int(valid.sum()),
                })
                matrix_rows.append({
                    "model_id": model_id, "rows": len(x), "columns": x.shape[1],
                    "rank": int(np.linalg.matrix_rank(x.to_numpy(dtype=float))),
                    "factor_count": len(FACTOR_COLUMNS),
                    "interaction_alias_classes": len(interaction_alias_classes()),
                    "paired_interaction_alias_classes": sum(
                        bool(item["aliased"]) for item in interaction_alias_classes()
                    ),
                    "factorial_alias_classes_split_by_center_slice": sum(
                        bool(item["split_by_center_slice"]) for item in interaction_alias_classes()
                    ),
                    "joint_curvature_columns": 1, "seed_block_columns": x.shape[1] - len(treatment_columns),
                })

                influence = fit.get_influence()
                diagnostics = model_frame[["observation_id", "design_slice", "seed"]].copy()
                diagnostics.insert(0, "model_id", model_id)
                diagnostics["response_transformed"] = y
                diagnostics["fitted_transformed"] = fit.fittedvalues
                diagnostics["residual_transformed"] = fit.resid
                diagnostics["standardized_residual"] = influence.resid_studentized_internal
                diagnostics["cooks_distance"] = influence.cooks_distance[0]
                diagnostic_frames.append(diagnostics)

                vifs = _vif(x, treatment_columns)
                vifs.insert(0, "model_id", model_id)
                vif_frames.append(vifs)

                center_coefficient = coefficients.loc[coefficients["term"] == "joint_center_departure"].iloc[0]
                factorial_values = pd.to_numeric(group.loc[group["design_slice"] == "factorial", response], errors="coerce")
                center_values = pd.to_numeric(group.loc[group["design_slice"] == "center", response], errors="coerce")
                center_rows.append({
                    "model_id": model_id,
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "response": response,
                    "factorial_mean_raw": factorial_values.mean(),
                    "center_mean_raw": center_values.mean(),
                    "center_minus_factorial_raw": center_values.mean() - factorial_values.mean(),
                    "direction_raw": "higher" if center_values.mean() > factorial_values.mean() else "lower",
                    "adjusted_center_departure_boxcox": center_coefficient["coefficient"],
                    "standard_error": center_coefficient["standard_error"],
                    "test_statistic": center_coefficient["test_statistic"],
                    "degrees_of_freedom": int(fit.df_resid),
                    "p_value": center_coefficient["p_value"],
                    "bonferroni_alpha": bonferroni_alpha,
                    "bonferroni_significant": bool(center_coefficient["bonferroni_significant"]),
                })

    summaries = pd.DataFrame(summary_rows)
    if len(summaries) != 20:
        raise RuntimeError(f"Expected 20 primary regression models; observed {len(summaries)}")
    return ModelBundle(
        coefficients=pd.concat(coefficient_frames, ignore_index=True),
        summaries=summaries,
        diagnostics=pd.concat(diagnostic_frames, ignore_index=True),
        center_tests=pd.DataFrame(center_rows),
        boxcox_parameters=pd.DataFrame(lambda_rows),
        vifs=pd.concat(vif_frames, ignore_index=True),
        model_matrices=pd.DataFrame(matrix_rows),
    )
