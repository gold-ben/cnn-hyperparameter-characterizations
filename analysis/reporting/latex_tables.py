"""Render publication CSV outputs as copy-ready portrait LaTeX longtables."""

from __future__ import annotations

from pathlib import Path
import math
import re
import shutil

import pandas as pd


LATEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _escape(value: object) -> str:
    text = str(value)
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in text)


def _scientific(value: float, digits: int = 2) -> str:
    if value == 0:
        return r"$<10^{-300}$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.{digits}f}\times 10^{{{exponent}}}$"


def _two_decimal_places(value: float) -> str:
    """Format a continuous value to exactly two decimal places."""
    return f"{value:.2f}"


def _number(value: object, column: str) -> str:
    if pd.isna(value):
        return "--"
    numeric = float(value)
    if column in {
        "n", "paired_n", "settings", "correct_predictions", "predicted_ties",
        "actual_ties", "total_candidate_observations", "valid_gaussian_observations",
        "missing_gaussian_observations", "affected_nine_factor_settings",
        "affected_seeds", "predicted_tie_count", "actual_tie_count", "count",
        "n_candidate", "n_complete", "n_missing", "design_columns", "design_rank",
        "residual_degrees_of_freedom", "significant_treatment_terms",
        "treatment_term_count", "seed", "degrees_of_freedom",
        "observed_complete_runs", "theoretical_minimum_complete_runs",
        "design_compatible_complete_runs", "runs_per_design_setting",
        "estimated_attempted_runs",
    }:
        return f"{int(round(numeric)):,}"
    if column in {
        "p_value", "omnibus_p_value", "bonferroni_alpha", "corrected_alpha",
    }:
        if numeric == 0 or abs(numeric) < 0.01:
            return _scientific(numeric)
        return _two_decimal_places(numeric)
    if column.endswith("_coded"):
        return f"{numeric:g}"
    return _two_decimal_places(numeric)


def _term(value: object) -> str:
    text = str(value)
    special = {
        "Intercept": "Intercept",
        "joint_center_departure": "Joint center departure",
    }
    if text in special:
        return special[text]
    if text.startswith("seed_block_"):
        return f"Seed block {_escape(text.removeprefix('seed_block_'))}"
    factor_labels = {
        "batch_size_coded": "Batch size",
        "dropout_flag_coded": "Dropout",
        "bn_flag_coded": "Batch normalization",
        "max_pool_flag_coded": "Max pooling",
        "initialization_coded": "Initialization",
        "cnn_width_coded": "CNN width",
        "conv_dim_list_coded": "Convolutional depth",
        "fc_width_coded": "FC width",
        "fc_dim_list_coded": "FC depth",
    }
    pieces = text.split(":")
    labels = [factor_labels.get(piece, piece.removesuffix("_coded").replace("_", " ")) for piece in pieces]
    return r" $\times$ ".join(_escape(label) for label in labels)


def _text(value: object, column: str) -> str:
    if pd.isna(value):
        return "--"
    if column == "term":
        return _term(value)
    text = str(value)
    if column == "dataset":
        return {"cifar10": "CIFAR--10", "cifar100": "CIFAR--100"}.get(text.lower(), _escape(text))
    if column in {"optimizer", "actual", "selected", "selected_optimizer", "actual_best_optimizer"}:
        return {
            "adam": "Adam", "adamw": "AdamW", "rmsprop": "RMSprop",
            "sgd": "SGD", "sgd_nesterov": "SGD--Nesterov",
        }.get(text, _escape(text))
    if column == "response":
        return {
            "gaussian_lipschitz": "Gaussian Lipschitz",
            "test_accuracy_10_epoch": "10-epoch accuracy",
        }.get(text, _escape(text))
    if column == "model_id":
        parts = text.split("__")
        if len(parts) == 3:
            return " / ".join((
                _text(parts[0], "dataset"),
                _text(parts[1], "optimizer"),
                _text(parts[2], "response"),
            ))
        return _escape(text)
    if column in {
        "bonferroni_significant", "corrected_significant", "selected_is_actual_best",
        "all_vifs_finite", "all_vifs_at_most_5", "all_vifs_below_10",
    }:
        yes = text.lower() == "true"
        return r"\textbf{Yes}" if yes else "No"
    if column == "term_role":
        return {
            "intercept": "Intercept", "seed_block": "Seed block",
            "first_order": "Main effect", "two_factor_alias_class": "2FI alias",
            "two_factor_interaction": "2FI",
            "joint_curvature": "Joint center",
        }.get(text, _escape(text))
    return _escape(text.replace("_", " "))


def _cell(value: object, column: str) -> str:
    if isinstance(value, bool) or column in {
        "dataset", "optimizer", "response", "term", "term_role", "model_id",
        "augmented_alias_class",
        "setting_id", "design_slice", "direction_raw", "correction", "actual",
        "selected", "selected_optimizer", "actual_best_optimizer",
        "bonferroni_significant", "corrected_significant", "selected_is_actual_best",
        "all_vifs_finite", "all_vifs_at_most_5", "all_vifs_below_10",
    }:
        return _text(value, column)
    if isinstance(value, (int, float)):
        return _number(value, column)
    try:
        float(value)
    except (TypeError, ValueError):
        return _text(value, column)
    return _number(value, column)


def _render_longtable(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    note: str | None,
    font: str = r"\scriptsize",
    stretch_column: str | None = None,
) -> str:
    alignments = ["l" if column in {
        "dataset", "optimizer", "response", "term", "term_role", "model_id",
        "augmented_alias_class",
        "setting_id", "design_slice", "direction_raw", "correction", "actual",
        "selected", "selected_optimizer", "actual_best_optimizer",
    } else "r" for column in columns]
    environment = "xltabular" if stretch_column else "longtable"
    if stretch_column:
        if stretch_column not in columns:
            raise RuntimeError(f"Stretch column {stretch_column!r} is absent from {columns}")
        alignments[columns.index(stretch_column)] = "X"
    column_specification = "".join(alignments)
    if len(columns) >= 13:
        tabcolsep = "1.5pt"
    elif len(columns) >= 10:
        tabcolsep = "2pt"
    elif len(columns) >= 8:
        tabcolsep = "2.5pt"
    else:
        tabcolsep = "3pt"
    lines = [
        "% Auto-generated from the corresponding CSV; do not edit by hand.",
        r"\begingroup",
        font,
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        (
            rf"\begin{{xltabular}}{{\linewidth}}{{@{{}}{column_specification}@{{}}}}"
            if stretch_column
            else rf"\begin{{longtable}}{{@{{}}{column_specification}@{{}}}}"
        ),
        rf"\caption{{{caption}}}\label{{{label}}} \\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{len(columns)}}}{{l}}{{\tablename\ \thetable\ continued}} \\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{len(columns)}}}{{r}}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(" & ".join(_cell(value, column) for value, column in zip(row, columns)) + r" \\")
    lines.append(rf"\end{{{environment}}}")
    if note:
        lines.extend([
            r"\vspace{-0.5em}",
            r"\begin{minipage}{\linewidth}",
            r"\footnotesize\textit{Note.} " + note,
            r"\end{minipage}",
        ])
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


TABLE_SPECS = {
    "model_power_sample_sizes": {
        "columns": ["dataset", "optimizer", "response", "observed_complete_runs", "partial_r_squared", "cohen_f_squared", "theoretical_minimum_complete_runs", "design_compatible_complete_runs", "estimated_attempted_runs"],
        "headers": ["Dataset", "Optimizer", "Response", r"\shortstack{Observed\\complete}", r"Partial $R^2$", r"Cohen $f^2$", r"\shortstack{Theoretical\\minimum}", r"\shortstack{Design-compatible\\complete}", r"\shortstack{Estimated\\attempted}"],
        "caption": "Retrospective sample-size estimates for the primary nine-factor models.",
        "note": r"Power is 0.80 at $\alpha=0.05$ for the upper-tail 41-df omnibus partial $F$ test of all treatment terms, conditional on seed blocks. The theoretical minimum is unconstrained. Design-compatible counts preserve all 80 settings; attempted counts additionally adjust for the model-specific observed completion rate. These estimates target the omnibus model test, not individual coefficients.",
        "font": r"\tiny",
    },
    "boxcox_parameters": {
        "columns": ["dataset", "optimizer", "response", "boxcox_lambda", "n"],
        "headers": ["Dataset", "Optimizer", "Response", r"Box--Cox $\lambda$", "$n$"],
        "caption": "Model-specific Box--Cox transformation parameters.",
        "note": "Transformations were estimated separately for each dataset, optimizer, and response.",
    },
    "center_joint_curvature_tests": {
        "columns": ["dataset", "optimizer", "response", "factorial_mean_raw", "center_mean_raw", "center_minus_factorial_raw", "direction_raw", "adjusted_center_departure_boxcox", "standard_error", "test_statistic", "degrees_of_freedom", "p_value", "bonferroni_alpha", "bonferroni_significant"],
        "headers": ["Dataset", "Optimizer", "Response", r"\shortstack{Factorial\\mean}", r"\shortstack{Center\\mean}", r"\shortstack{$\Delta$\\raw}", "Direction", r"\shortstack{$\Delta$\\Box--Cox}", "SE", "$t$", "df", "$p$", r"$\alpha_B$", "Sig."],
        "caption": "Joint-center curvature tests for the primary nine-factor models.",
        "note": r"The center term represents one overall joint departure; it does not identify separate pure-quadratic effects. Significance uses the model-specific Bonferroni threshold $\alpha_B$.",
        "font": r"\tiny",
    },
    "gaussian_accuracy_correlations": {
        "columns": ["dataset", "optimizer", "paired_n", "gaussian_boxcox_lambda", "accuracy_boxcox_lambda", "pearson_coefficient", "p_value", "corrected_alpha", "corrected_significant"],
        "headers": ["Dataset", "Optimizer", "$n$", r"$\lambda_G$", r"$\lambda_A$", "$r$", "$p$", r"$\alpha_c$", "Sig."],
        "caption": "Box--Cox-scale correlations between Gaussian Lipschitz estimates and 10-epoch accuracy.",
        "note": "Pearson correlations use complete response pairs. Bonferroni correction was applied across the ten dataset--optimizer tests.",
        "font": r"\footnotesize",
    },
    "gaussian_missingness": {
        "columns": ["dataset", "optimizer", "total_candidate_observations", "valid_gaussian_observations", "missing_gaussian_observations", "missing_percentage", "affected_nine_factor_settings", "affected_seeds"],
        "headers": ["Dataset", "Optimizer", "Candidate", "Valid", "Missing", r"\shortstack{Missing\\(\%)}", r"\shortstack{Settings\\affected}", r"\shortstack{Seeds\\affected}"],
        "caption": "Gaussian-response missingness by dataset and optimizer.",
        "note": "Missingness is response-specific; accuracy was not used to determine inclusion in Gaussian-response models.",
    },
    "optimizer_selection_summary": {
        "columns": ["dataset", "settings", "correct_predictions", "selection_accuracy", "predicted_ties", "actual_ties"],
        "headers": ["Dataset", "Settings", "Correct", r"\shortstack{Accuracy\\(\%)}", r"\shortstack{Pred.\\ties}", r"\shortstack{Actual\\ties}"],
        "caption": "Optimizer-selection meta-model performance.",
        "note": "Accuracy is the percentage of settings for which the selected optimizer is among the observed accuracy-maximizing optimizers.",
    },
    "optimizer_selection_confusion_matrix": {
        "columns": ["dataset", "actual", "selected", "count"],
        "headers": ["Dataset", "Actual optimizer", "Selected optimizer", "Count"],
        "caption": "Optimizer-selection confusion counts.",
        "note": "Rows with zero observations are omitted from the source table.",
    },
    "optimizer_selection_comparison": {
        "columns": ["dataset", "setting_id", "selected_optimizer", "actual_best_optimizer", "selected_is_actual_best", "predicted_tie_count", "actual_tie_count"],
        "headers": ["Dataset", "Setting", "Selected", r"\shortstack{Observed\\best}", "Correct", r"\shortstack{Pred.\\ties}", r"\shortstack{Actual\\ties}"],
        "caption": "Setting-level optimizer-selection comparisons.",
        "note": "Correct indicates that the selected optimizer is among the observed accuracy-maximizing optimizers for that setting.",
    },
    "optimizer_accuracy_predictions": {
        "columns": ["setting_id", "design_slice", "batch_size_coded", "dropout_flag_coded", "bn_flag_coded", "max_pool_flag_coded", "initialization_coded", "cnn_width_coded", "conv_dim_list_coded", "fc_width_coded", "fc_dim_list_coded", "dataset", "optimizer", "predicted_accuracy", "observed_mean_accuracy"],
        "headers": ["Setting", "Slice", "Batch", "Dropout", "BN", "Pool", "Init.", r"\shortstack{CNN\\width}", r"\shortstack{Conv.\\depth}", r"\shortstack{FC\\width}", r"\shortstack{FC\\depth}", "Dataset", "Optimizer", r"\shortstack{Pred.\\acc.}", r"\shortstack{Observed\\acc.}"],
        "caption": "Setting- and optimizer-specific accuracy predictions from the optimizer-selection meta-models.",
        "note": "Factor columns use the coded design scale. Accuracy values are percentages.",
        "font": r"\tiny",
    },
    "optimizer_meta_model_coefficients": {
        "columns": ["dataset", "optimizer", "term", "coefficient", "standard_error", "test_statistic", "p_value"],
        "headers": ["Dataset", "Optimizer", "Term", "Estimate", "SE", "$t$", "$p$"],
        "caption": "Optimizer-specific accuracy meta-model coefficients.",
        "note": r"Estimates are on the 10-epoch accuracy scale. Interaction labels use $\times$.",
    },
    "primary_model_summaries": {
        "columns": ["dataset", "optimizer", "response", "n_candidate", "n_complete", "n_missing", "boxcox_lambda", "design_columns", "design_rank", "residual_degrees_of_freedom", "r_squared", "adjusted_r_squared", "omnibus_f_statistic", "omnibus_p_value", "significant_treatment_terms", "bonferroni_alpha"],
        "headers": ["Dataset", "Optimizer", "Response", "Candidate", "Complete", "Missing", r"$\lambda$", "Columns", "Rank", r"\shortstack{Residual\\df}", "$R^2$", r"\shortstack{Adj.\\$R^2$}", r"$F$", r"$p_F$", r"\shortstack{Sig.\\terms}", r"$\alpha_B$"],
        "caption": "Primary nine-factor Box--Cox OLS model summaries.",
        "note": "Each full-rank matrix contains 41 tested treatment contrasts, nine seed-block columns, and an intercept (51 columns total). The omnibus F test covers all 50 non-intercept coefficients; its denominator degrees of freedom are reported in the residual-df column. Significant treatment terms use the model-specific Bonferroni threshold.",
        "font": r"\tiny",
    },
    "primary_model_coefficients": {
        "columns": ["dataset", "optimizer", "response", "term", "coefficient", "standard_error", "test_statistic", "p_value", "bonferroni_alpha", "bonferroni_significant"],
        "headers": ["Dataset", "Optimizer", "Response", "Term", "Estimate", "SE", "$t$", "$p$", r"$\alpha_B$", "Sig."],
        "caption": "Complete coefficient results for the primary nine-factor Box--Cox OLS models.",
        "note": "One representative is fitted per estimable two-factor interaction (2FI) alias class. Seed blocks are fixed nuisance effects. Significance is evaluated only for treatment terms using the model-specific Bonferroni threshold.",
        "font": r"\tiny",
    },
    "variance_inflation_summary": {
        "columns": ["dataset", "optimizer", "response", "treatment_term_count", "maximum_vif", "maximum_first_order_vif", "maximum_two_factor_term_vif", "joint_center_departure_vif"],
        "headers": ["Dataset", "Optimizer", "Response", "Terms", r"\shortstack{Max.\\VIF}", r"\shortstack{Max.\\main}", r"\shortstack{Max.\\2FI}", r"\shortstack{Center\\VIF}"],
        "caption": "Variance inflation audit for the primary nine-factor models.",
        "note": r"VIFs use the 31 estimable two-factor-interaction groups in the augmented design. On the 64 factorial settings, Batch normalization $\times$ Max pooling and FC width $\times$ FC depth are exactly proportional ($|r|=1$). The 16 center settings distinguish them because the former interaction continues to vary while the latter is zero, reducing their absolute correlation over all 80 settings to $\sqrt{64/80}=0.894$ (pairwise VIF $=5$). Thus the center slice makes these interactions separately estimable, but the information distinguishing them comes entirely from the center settings and substantial collinearity remains. The other five paired two-factor aliases remain exact and are represented once per alias group in the fitted model.",
    },
    "variance_inflation_factors": {
        "columns": ["model_id", "term", "variance_inflation_factor"],
        "headers": ["Model", "Term", "VIF"],
        "caption": "Term-level variance inflation factors for the primary nine-factor models.",
        "note": "Only treatment terms are reported. Model identifiers encode dataset, optimizer, and response. All fitted-term VIFs are finite.",
    },
}


def _write_primary_coefficient_tables(
    frame: pd.DataFrame,
    destination: Path,
    seed_frame: pd.DataFrame,
) -> None:
    """Render the 20 primary fits as 20 scientifically distinct tables."""
    required = {
        "model_id", "dataset", "optimizer", "response", "term", "coefficient",
        "standard_error", "test_statistic", "p_value", "term_role",
        "bonferroni_alpha", "bonferroni_significant",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Primary coefficient table is missing columns: {sorted(missing)}")

    groups = list(frame.groupby("model_id", sort=False))
    if len(groups) != 20 or any(len(group) != 51 for _, group in groups):
        sizes = {model_id: len(group) for model_id, group in groups}
        raise RuntimeError(f"Expected 20 primary models with 51 rows each; observed {sizes}")

    seed_maps = {}
    for dataset, seeds in seed_frame.groupby("dataset")["seed"]:
        ordered = sorted(int(seed) for seed in seeds.unique())
        if len(ordered) != 10:
            raise RuntimeError(f"Expected ten seed levels for {dataset}; observed {ordered}")
        seed_maps[dataset] = {seed: position for position, seed in enumerate(ordered, start=1)}

    model_directory = destination / "primary_model_coefficients"
    model_directory.mkdir(parents=True, exist_ok=True)
    wrapper = [
        "% Twenty separate primary-model coefficient tables.",
        "% Include this file to typeset the complete coefficient appendix.",
    ]
    columns = [
        "term", "coefficient", "standard_error", "test_statistic", "p_value",
        "bonferroni_significant",
    ]
    headers = ["Term", "Estimate", "SE", "$t$", "$p$", "Sig."]
    for model_id, group in groups:
        first = group.iloc[0]
        dataset = _text(first["dataset"], "dataset")
        optimizer = _text(first["optimizer"], "optimizer")
        response = _text(first["response"], "response")
        display_group = group.copy()
        seed_rows = display_group["term"].str.startswith("seed_block_")
        generic_numbers = display_group.loc[seed_rows, "term"].map(
            lambda term: seed_maps[first["dataset"]][int(term.removeprefix("seed_block_"))]
        )
        if set(generic_numbers) != set(range(2, 11)):
            raise RuntimeError(
                f"Expected generic nonreference seed blocks 2--10 for {model_id}; "
                f"observed {sorted(generic_numbers)}"
            )
        display_group.loc[seed_rows, "term"] = generic_numbers.map(
            lambda number: f"Seed block {number}"
        )
        display_group.loc[seed_rows, "_seed_sort"] = generic_numbers
        display_group.loc[~seed_rows, "_seed_sort"] = 0
        display_group = pd.concat([
            display_group.loc[~seed_rows],
            display_group.loc[seed_rows].sort_values("_seed_sort"),
        ])
        filename = re.sub(r"[^a-z0-9_]+", "_", str(model_id).lower()).strip("_") + ".tex"
        label_slug = re.sub(r"[^a-z0-9]+", "-", str(model_id).lower()).strip("-")
        rendered = _render_longtable(
            display_group,
            columns=columns,
            headers=headers,
            caption=f"Primary-model coefficients for {dataset}, {optimizer}, {response}.",
            label=f"tab:nine-factor-coefficients-{label_slug}",
            note=None,
            font=r"\small",
            stretch_column="term",
        )
        (model_directory / filename).write_text(rendered, encoding="utf-8")
        wrapper.extend([
            r"\clearpage",
            rf"\input{{primary_model_coefficients/{filename}}}",
        ])
    (destination / "primary_model_coefficients.tex").write_text(
        "\n".join(wrapper) + "\n", encoding="utf-8"
    )


def write_latex_tables(output_root: Path) -> None:
    """Write one LaTeX fragment per publication CSV and a standalone master."""
    destination = output_root / "latex_tables"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {path.stem: path for path in sorted((output_root / "tables").glob("*.csv"))}
    sources["variance_inflation_factors"] = output_root / "diagnostics" / "variance_inflation_factors.csv"

    generated = []
    for stem, source in sources.items():
        if stem == "primary_model_coefficients":
            seed_frame = pd.read_csv(
                output_root / "data" / "nine_factor_analysis_dataset.csv",
                usecols=["dataset", "seed"],
            )
            _write_primary_coefficient_tables(pd.read_csv(source), destination, seed_frame)
            generated.append("primary_model_coefficients.tex")
            continue
        spec = TABLE_SPECS.get(stem)
        if spec is None:
            raise RuntimeError(f"No LaTeX table specification for {source}")
        frame = pd.read_csv(source)
        missing = set(spec["columns"]) - set(frame.columns)
        if missing:
            raise RuntimeError(f"Missing columns in {source}: {sorted(missing)}")
        label = "tab:nine-factor-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
        stretch_column = spec.get("stretch_column")
        if stretch_column is None:
            stretch_column = next(
                column for column in (
                    "term", "response", "setting_id", "optimizer", "model_id",
                    "dataset", "actual",
                ) if column in spec["columns"]
            )
        rendered = _render_longtable(
            frame,
            columns=spec["columns"],
            headers=spec["headers"],
            caption=spec["caption"],
            label=label,
            note=spec["note"],
            font=spec.get("font", r"\scriptsize"),
            stretch_column=stretch_column,
        )
        target = destination / f"{stem}.tex"
        target.write_text(rendered, encoding="utf-8")
        generated.append(target.name)

    include_lines = ["% Requires booktabs, longtable, and xltabular."]
    for filename in generated:
        include_lines.extend([r"\clearpage", rf"\input{{{filename}}}"])
    (destination / "all_tables.tex").write_text("\n".join(include_lines) + "\n", encoding="utf-8")

    master = r"""\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xltabular}
\usepackage[T1]{fontenc}
\begin{document}
\input{all_tables.tex}
\end{document}
"""
    (destination / "overleaf_master.tex").write_text(master, encoding="utf-8")

    readme = """# Journal-ready LaTeX tables

Upload this directory's contents to one Overleaf directory and set
`overleaf_master.tex` as the main document to compile every table. To use the
fragments in an existing manuscript, load `booktabs`, `longtable`, `xltabular`,
    then `\\input{filename.tex}`. Each fragment has its own caption,
label, repeating header, and continuation footer. Every table uses `xltabular`
    to occupy exactly the active `\\linewidth`. All fragments remain in portrait
    orientation so they can be included safely in journal templates. Long headers
    are split across lines, and inter-column padding decreases for high column counts. Text-heavy
fields use the flexible `X` column so their contents wrap within `\\linewidth`
rather than expanding the table beyond the page.
Primary coefficient tables intentionally omit notes.

The tables are generated from `outputs/tables/*.csv`, plus the
term-level VIF audit. Numerical values are rounded for presentation; the CSVs
remain the machine-readable source of full-precision values. Continuous values
are reported to two decimal places; exact counts, sample sizes, ranks,
degrees of freedom, seeds, and coded design levels are not rounded. The observation-
level residual diagnostic CSV is intentionally not typeset because its 15,733
rows are a data artifact rather than a journal table.

`primary_model_coefficients.tex` is an appendix wrapper, not a combined results
table. It inputs 20 separate files from `primary_model_coefficients/`, one for
each dataset--optimizer--response model. For presentation, recorded seed IDs are
mapped in numeric order to generic Seed blocks 1--10. Seed block 1 is the
reference level, so each coefficient table displays Seed blocks 2--10.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    power = pd.read_csv(output_root / "tables" / "model_power_sample_sizes.csv")
    methodology = f"""# Sample-size methodology

The target was 80% power at an unadjusted alpha of 0.05 for each primary
model's upper-tail 41-degree-of-freedom omnibus partial F test of all treatment
terms (nine main effects, 31 estimable two-factor groups, and joint-center
departure), conditional on the nine fitted seed-block indicators.

For each fitted Box--Cox response model, a reduced seed-only model was compared
with the full model. The retrospective effect size was Cohen's
`f^2 = (SSE_reduced - SSE_full) / SSE_full`. For candidate total sample size
`N`, power was evaluated from a noncentral F distribution with numerator df 41,
denominator df `N - 51`, and noncentrality `N f^2`. The smallest integer `N`
with power at least 0.80 is reported as the unconstrained theoretical minimum.

Those minima ({int(power['theoretical_minimum_complete_runs'].min())}--{int(power['theoretical_minimum_complete_runs'].max())} complete observations) are smaller than the 80 distinct design settings. Therefore the practical minimum is 80 complete runs per model, one per setting. `estimated_attempted_runs` divides 80 by each model's observed completion rate and rounds up; it is {int(power['estimated_attempted_runs'].min())}--{int(power['estimated_attempted_runs'].max())} runs across models.

This is a retrospective planning estimate based on observed effects, not an
independent prospective guarantee. It supports detection of the overall
treatment signal only; planning for a specific coefficient, multiplicity-
adjusted term test, stable effect estimates, interactions, or missingness by
setting can require substantially more replication. The full-precision results
are in `../tables/model_power_sample_sizes.csv` and the LaTeX table is
`model_power_sample_sizes.tex`.
"""
    (destination / "sample_size_methodology.md").write_text(
        methodology, encoding="utf-8"
    )

    archive_base = output_root / "nine_factor_latex_tables_overleaf"
    shutil.make_archive(str(archive_base), "zip", root_dir=destination)
