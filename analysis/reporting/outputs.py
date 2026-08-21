"""Write the concise future-facing result and validation artifact set."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config.study import OUTPUT_ROOT, RESPONSES
from analysis.models.regression import ModelBundle, interaction_alias_classes
from analysis.models.power import model_power_sample_sizes
from analysis.meta_models.optimizer_selection import MetaResults
from analysis.reporting.latex_tables import write_latex_tables
from analysis.design import FACTOR_COLUMNS, OPTIMIZERS


FIRST_ORDER_FACTOR_LABELS = {
    "batch_size_coded": "Batch Size",
    "dropout_flag_coded": "Dropout",
    "bn_flag_coded": "Batch Normalization",
    "max_pool_flag_coded": "Max Pooling",
    "initialization_coded": "Weight Initialization",
    "cnn_width_coded": "Convolutional Width",
    "conv_dim_list_coded": "Convolutional Depth",
    "fc_width_coded": "Fully Connected Width",
    "fc_dim_list_coded": "Fully Connected Depth",
}

OPTIMIZER_LABELS = {
    "sgd": "SGD",
    "sgd_nesterov": "SGD + Nesterov",
    "adam": "Adam",
    "adamw": "AdamW",
    "rmsprop": "RMSprop",
}

RESPONSE_LABELS = {
    "gaussian_lipschitz": "Gaussian-Path Maximum Lipschitz",
    "test_accuracy_10_epoch": "10-Epoch Test Accuracy",
}

SIGNIFICANCE_COLORS = {
    "not_significant": "#FFFFFF",
    "significant_positive": "#1B9E77",
    "significant_negative": "#D95F02",
}


def first_order_significance_matrix(
    coefficients: pd.DataFrame,
    dataset: str,
    response: str,
) -> np.ndarray:
    """Return factor-by-optimizer categorical significance states.

    States are 0 for not significant, 1 for a significant positive
    coefficient, and -1 for a significant negative coefficient.
    """
    selected = coefficients.loc[
        coefficients["dataset"].eq(dataset)
        & coefficients["response"].eq(response)
        & coefficients["term_role"].eq("first_order")
    ]
    indexed = selected.set_index(["term", "optimizer"])
    expected = pd.MultiIndex.from_product(
        [FACTOR_COLUMNS, OPTIMIZERS], names=["term", "optimizer"]
    )
    missing = expected.difference(indexed.index)
    if len(missing):
        raise ValueError(
            f"Missing first-order coefficient rows for {dataset}/{response}: "
            f"{list(missing)}"
        )
    ordered = indexed.reindex(expected)
    significant = ordered["bonferroni_significant"].fillna(False).astype(bool)
    coefficients_numeric = pd.to_numeric(ordered["coefficient"], errors="coerce")
    states = np.where(
        significant,
        np.where(coefficients_numeric.gt(0), 1, -1),
        0,
    )
    return states.reshape(len(FACTOR_COLUMNS), len(OPTIMIZERS))


def write_first_order_significance_heatmaps(coefficients: pd.DataFrame) -> None:
    """Write one journal-ready categorical heatmap for each studied response."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # This categorical response-specific output supersedes the earlier
    # continuous, transposed overview figure.
    (figures / "first_order_effect_heatmap.png").unlink(missing_ok=True)
    datasets = ("CIFAR10", "CIFAR100")
    dataset_labels = {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100"}
    state_colors = [
        SIGNIFICANCE_COLORS["significant_negative"],
        SIGNIFICANCE_COLORS["not_significant"],
        SIGNIFICANCE_COLORS["significant_positive"],
    ]
    color_map = ListedColormap(state_colors)
    normalization = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), color_map.N)

    for response in RESPONSES:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12.2, 7.6),
            sharey=True,
            gridspec_kw={"wspace": 0.08},
        )
        for panel_index, (axis, dataset) in enumerate(zip(axes, datasets, strict=True)):
            matrix = first_order_significance_matrix(
                coefficients,
                dataset,
                response,
            )
            axis.imshow(
                matrix,
                aspect="auto",
                cmap=color_map,
                norm=normalization,
                interpolation="nearest",
            )
            axis.set_xticks(range(len(OPTIMIZERS)))
            axis.set_xticklabels(
                [OPTIMIZER_LABELS[optimizer] for optimizer in OPTIMIZERS],
                rotation=32,
                ha="right",
                rotation_mode="anchor",
                fontsize=11,
                fontweight="bold",
            )
            axis.set_yticks(range(len(FACTOR_COLUMNS)))
            if panel_index == 0:
                axis.set_yticklabels(
                    [FIRST_ORDER_FACTOR_LABELS[factor] for factor in FACTOR_COLUMNS],
                    fontsize=11,
                    fontweight="bold",
                )
                axis.set_ylabel(
                    "First-Order Factor",
                    fontsize=13,
                    fontweight="bold",
                    labelpad=12,
                )
            axis.set_xlabel(
                "Optimization Routine",
                fontsize=13,
                fontweight="bold",
                labelpad=10,
            )
            axis.set_title(
                dataset_labels[dataset],
                fontsize=15,
                fontweight="bold",
                pad=10,
            )
            axis.set_xticks(np.arange(-0.5, len(OPTIMIZERS), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(FACTOR_COLUMNS), 1), minor=True)
            axis.grid(which="minor", color="#D0D0D0", linewidth=0.8)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(axis="y", length=0)

        figure.suptitle(
            f"{RESPONSE_LABELS[response]}: Significant First-Order Effects",
            fontsize=17,
            fontweight="bold",
            y=0.975,
        )
        figure.legend(
            handles=[
                Patch(
                    facecolor=SIGNIFICANCE_COLORS["significant_positive"],
                    edgecolor="#707070",
                    label="Significant Positive",
                ),
                Patch(
                    facecolor=SIGNIFICANCE_COLORS["significant_negative"],
                    edgecolor="#707070",
                    label="Significant Negative",
                ),
                Patch(
                    facecolor=SIGNIFICANCE_COLORS["not_significant"],
                    edgecolor="#707070",
                    label="Not Significant",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            prop={"size": 11, "weight": "bold"},
        )
        figure.subplots_adjust(left=0.25, right=0.985, top=0.87, bottom=0.25)
        output_stem = figures / f"{response}_first_order_significance_heatmap"
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=300,
            facecolor="white",
        )
        figure.savefig(
            output_stem.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def variance_inflation_summary(models: ModelBundle) -> pd.DataFrame:
    """Summarize treatment-term VIFs for each full-rank primary model."""
    term_roles = models.coefficients[
        ["model_id", "dataset", "optimizer", "response", "term", "term_role"]
    ].drop_duplicates()
    vifs = models.vifs.merge(
        term_roles,
        on=["model_id", "term"],
        how="left",
        validate="one_to_one",
    )
    if vifs["term_role"].isna().any():
        missing = sorted(vifs.loc[vifs["term_role"].isna(), "term"].unique())
        raise RuntimeError(f"VIF terms lack coefficient roles: {missing}")

    rows = []
    keys = ["model_id", "dataset", "optimizer", "response"]
    for key, group in vifs.groupby(keys, sort=True):
        by_role = group.groupby("term_role")["variance_inflation_factor"]
        two_factor = group.loc[
            group["term_role"].isin(["two_factor_alias_class", "two_factor_interaction"]),
            "variance_inflation_factor",
        ]
        rows.append({
            **dict(zip(keys, key)),
            "treatment_term_count": len(group),
            "maximum_vif": group["variance_inflation_factor"].max(),
            "maximum_first_order_vif": by_role.get_group("first_order").max(),
            "maximum_two_factor_term_vif": two_factor.max(),
            "joint_center_departure_vif": by_role.get_group("joint_curvature").iloc[0],
            "all_vifs_finite": bool(np.isfinite(group["variance_inflation_factor"]).all()),
            "all_vifs_at_most_5": bool(
                group["variance_inflation_factor"].le(5.0 + 1e-10).all()
            ),
            "all_vifs_below_10": bool(group["variance_inflation_factor"].lt(10.0).all()),
        })
    return pd.DataFrame(rows)


def write_primary_tables(
    models: ModelBundle,
    correlation_table: pd.DataFrame,
    meta: MetaResults,
    missingness: pd.DataFrame,
) -> None:
    tables = OUTPUT_ROOT / "tables"
    diagnostics = OUTPUT_ROOT / "diagnostics"
    tables.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    models.coefficients.to_csv(tables / "primary_model_coefficients.csv", index=False)
    models.summaries.to_csv(tables / "primary_model_summaries.csv", index=False)
    model_power_sample_sizes(models).to_csv(
        tables / "model_power_sample_sizes.csv", index=False
    )
    models.center_tests.to_csv(tables / "center_joint_curvature_tests.csv", index=False)
    models.boxcox_parameters.to_csv(tables / "boxcox_parameters.csv", index=False)
    models.vifs.to_csv(diagnostics / "variance_inflation_factors.csv", index=False)
    variance_inflation_summary(models).to_csv(
        tables / "variance_inflation_summary.csv", index=False
    )
    models.diagnostics.to_csv(diagnostics / "residual_diagnostics.csv", index=False)
    correlation_table.to_csv(tables / "gaussian_accuracy_correlations.csv", index=False)
    missingness.to_csv(tables / "gaussian_missingness.csv", index=False)
    meta.summaries.to_csv(tables / "optimizer_selection_summary.csv", index=False)
    meta.coefficients.to_csv(tables / "optimizer_meta_model_coefficients.csv", index=False)
    meta.predictions.to_csv(tables / "optimizer_accuracy_predictions.csv", index=False)
    meta.comparisons.to_csv(tables / "optimizer_selection_comparison.csv", index=False)
    meta.confusion.to_csv(tables / "optimizer_selection_confusion_matrix.csv", index=False)
    write_latex_tables(OUTPUT_ROOT)


def write_figures(
    frame: pd.DataFrame,
    models: ModelBundle,
    correlations: pd.DataFrame,
    meta: MetaResults,
) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    write_first_order_significance_heatmaps(models.coefficients)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=False, sharey=False)
    for axis, ((dataset, optimizer), group) in zip(
        axes.flat, frame.groupby(["dataset", "optimizer"])
    ):
        axis.scatter(
            group["gaussian_lipschitz"],
            group["test_accuracy_10_epoch"],
            s=4,
            alpha=0.35,
        )
        result = correlations.loc[
            (correlations["dataset"] == dataset)
            & (correlations["optimizer"] == optimizer)
        ].iloc[0]
        axis.set_title(
            f"{dataset} / {optimizer}\nr={result['pearson_coefficient']:.3f}, "
            f"n={int(result['paired_n'])}",
            fontsize=8,
        )
        axis.set_xlabel("Gaussian Lipschitz", fontsize=7)
        axis.set_ylabel("10-epoch accuracy", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "gaussian_accuracy_scatter.png", dpi=180)
    plt.close(fig)

    center = models.center_tests.copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = np.where(center["bonferroni_significant"], "#b2182b", "#4d4d4d")
    ax.bar(range(len(center)), center["center_minus_factorial_raw"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(
        range(len(center)), center["model_id"], rotation=75, ha="right", fontsize=6
    )
    ax.set_ylabel("raw center mean minus factorial mean")
    ax.set_title(
        "Joint-center departures; red denotes model-scale Bonferroni significance"
    )
    fig.tight_layout()
    fig.savefig(figures / "center_joint_departures.png", dpi=180)
    plt.close(fig)

    summary = meta.summaries
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["dataset"], summary["selection_accuracy"], color="#2166ac")
    ax.set_ylim(0, 100)
    ax.set_ylabel("selection accuracy (%)")
    ax.set_title("Optimizer-selection meta-model")
    fig.tight_layout()
    fig.savefig(figures / "optimizer_selection_accuracy.png", dpi=180)
    plt.close(fig)
