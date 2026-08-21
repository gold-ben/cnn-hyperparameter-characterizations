from __future__ import annotations

import itertools
import unittest

import numpy as np
import pandas as pd

from analysis.models.regression import (
    factorial_interaction_alias_classes,
    interaction_alias_classes,
    model_matrix,
)
from analysis.data import load_analysis_data
from analysis.reporting.latex_tables import TABLE_SPECS, _number, _render_longtable
from analysis.reporting.outputs import first_order_significance_matrix
from analysis.design import FACTOR_COLUMNS, OPTIMIZERS
from analysis.design import generate_design


class AnalysisTests(unittest.TestCase):
    def test_latex_continuous_values_use_two_decimal_places(self) -> None:
        self.assertEqual(_number(2.0065615, "maximum_vif"), "2.01")
        self.assertEqual(_number(1.9321163, "maximum_vif"), "1.93")
        self.assertEqual(_number(0.00125, "bonferroni_alpha"), r"$1.25\times 10^{-3}$")
        self.assertEqual(_number(4.816e-122, "p_value"), r"$4.82\times 10^{-122}$")
        self.assertEqual(_number(123.456, "omnibus_f_statistic"), "123.46")
        self.assertEqual(
            _number(4.816e-122, "omnibus_p_value"),
            r"$4.82\times 10^{-122}$",
        )
        self.assertEqual(_number(-158.4554, "center_minus_factorial_raw"), "-158.46")
        self.assertEqual(_number(503.399, "test_statistic"), "503.40")
        self.assertEqual(_number(1870.246, "factorial_mean_raw"), "1870.25")

    def test_latex_structural_values_remain_exact(self) -> None:
        self.assertEqual(_number(773, "n_complete"), "773")
        self.assertEqual(_number(-1, "batch_size_coded"), "-1")

    def test_publication_tables_omit_requested_audit_columns(self) -> None:
        for name, spec in TABLE_SPECS.items():
            with self.subTest(table=name):
                self.assertEqual(len(spec["columns"]), len(spec["headers"]))
                self.assertNotIn("landscape", spec)
        self.assertNotIn("term_role", TABLE_SPECS["primary_model_coefficients"]["columns"])
        self.assertNotIn("augmented_alias_class", TABLE_SPECS["primary_model_coefficients"]["columns"])
        self.assertIn("omnibus_f_statistic", TABLE_SPECS["primary_model_summaries"]["columns"])
        self.assertIn("omnibus_p_value", TABLE_SPECS["primary_model_summaries"]["columns"])
        self.assertNotIn("correction", TABLE_SPECS["gaussian_accuracy_correlations"]["columns"])
        for table in ("optimizer_selection_summary", "optimizer_selection_comparison"):
            self.assertFalse(any("regret" in column for column in TABLE_SPECS[table]["columns"]))
        vif_columns = TABLE_SPECS["variance_inflation_summary"]["columns"]
        self.assertFalse(any(column.startswith("all_vifs_") for column in vif_columns))

    def test_latex_fragments_are_portrait_safe(self) -> None:
        rendered = _render_longtable(
            pd.DataFrame({"dataset": ["CIFAR10"], "n": [1]}),
            columns=["dataset", "n"],
            headers=["Dataset", "$n$"],
            caption="Test table.",
            label="tab:test",
            note=None,
            stretch_column="dataset",
        )
        self.assertNotIn(r"\begin{landscape}", rendered)
        self.assertNotIn(r"\end{landscape}", rendered)
        self.assertIn(r"\begin{xltabular}{\linewidth}", rendered)

    def test_alias_class_count(self) -> None:
        factorial_classes = factorial_interaction_alias_classes()
        self.assertEqual(len(factorial_classes), 30)
        self.assertEqual(sum(bool(item["aliased"]) for item in factorial_classes), 6)

        augmented_classes = interaction_alias_classes()
        self.assertEqual(len(augmented_classes), 31)
        self.assertEqual(sum(bool(item["aliased"]) for item in augmented_classes), 5)
        split = [item for item in augmented_classes if item["split_by_center_slice"]]
        self.assertEqual([item["alias_class_id"] for item in split], ["2FI-14a", "2FI-14b"])
        self.assertEqual(
            [item["members"] for item in split],
            [["bn_flag_coded:max_pool_flag_coded"], ["fc_width_coded:fc_dim_list_coded"]],
        )

    def test_center_slice_breaks_one_exact_factorial_alias_but_remains_collinear(self) -> None:
        design = pd.DataFrame(generate_design())
        first = design["bn_flag_coded"] * design["max_pool_flag_coded"]
        second = design["fc_width_coded"] * design["fc_dim_list_coded"]
        factorial = design["design_slice"].eq("factorial")
        center = ~factorial
        self.assertAlmostEqual(abs(first[factorial].corr(second[factorial])), 1.0)
        self.assertTrue(second[center].eq(0).all())
        self.assertEqual(set(first[center]), {-1, 1})
        self.assertAlmostEqual(abs(first.corr(second)), np.sqrt(64.0 / 80.0))

    def test_full_rank_estimable_matrix(self) -> None:
        rows = []
        for design_row in generate_design():
            for seed in range(42, 52):
                row = dict(design_row)
                row["seed"] = seed
                rows.append(row)
        frame = pd.DataFrame(rows)
        matrix, treatment = model_matrix(frame)
        self.assertEqual(matrix.shape[1], 51)
        self.assertEqual(np.linalg.matrix_rank(matrix.to_numpy()), 51)
        self.assertEqual(len(treatment), 42)
        self.assertEqual(sum(column.startswith("seed_block_") for column in matrix), 9)
        self.assertTrue(frame.groupby("seed")["setting_id"].nunique().eq(80).all())
        self.assertTrue(
            frame.groupby(["seed", "design_slice"]).size().unstack().eq(
                pd.Series({"center": 16, "factorial": 64})
            ).all().all()
        )
        np.testing.assert_array_equal(
            matrix["joint_center_departure"].to_numpy(),
            frame["design_slice"].eq("center").astype(float).to_numpy(),
        )
        self.assertEqual(len(FACTOR_COLUMNS), 9)

    def test_canonical_input_shape_and_coverage(self) -> None:
        frame = load_analysis_data()
        self.assertEqual(len(frame), 8000)
        self.assertTrue(frame["observation_id"].is_unique)
        self.assertTrue(frame.groupby("dataset").size().eq(4000).all())
        self.assertTrue(
            frame.groupby(["dataset", "optimizer", "setting_id"]).size().eq(10).all()
        )

    def test_aliasing_is_removed_from_the_fitted_parameterization(self) -> None:
        factorial = pd.DataFrame(
            row for row in generate_design() if row["design_slice"] == "factorial"
        )
        all_interactions = pd.DataFrame({"Intercept": 1.0}, index=factorial.index)
        for factor in FACTOR_COLUMNS:
            all_interactions[factor] = factorial[factor].astype(float)
        for first, second in itertools.combinations(FACTOR_COLUMNS, 2):
            all_interactions[f"{first}:{second}"] = (
                factorial[first].astype(float) * factorial[second].astype(float)
            )
        self.assertEqual(all_interactions.shape[1], 46)
        self.assertEqual(np.linalg.matrix_rank(all_interactions.to_numpy()), 40)

    def test_pure_quadratics_collapse_to_joint_center_departure(self) -> None:
        design = pd.DataFrame(generate_design())
        center = design["design_slice"].eq("center").astype(float)
        squares = design[FACTOR_COLUMNS].astype(float).pow(2)
        square_matrix = pd.concat([
            pd.Series(1.0, index=design.index, name="Intercept"),
            center.rename("joint_center_departure"),
            squares,
        ], axis=1)
        self.assertEqual(np.linalg.matrix_rank(square_matrix.to_numpy()), 2)

    def test_first_order_significance_matrix_is_categorical_and_directional(self) -> None:
        rows = []
        for factor_index, factor in enumerate(FACTOR_COLUMNS):
            for optimizer_index, optimizer in enumerate(OPTIMIZERS):
                rows.append(
                    {
                        "dataset": "CIFAR10",
                        "response": "test_accuracy_10_epoch",
                        "term_role": "first_order",
                        "term": factor,
                        "optimizer": optimizer,
                        "coefficient": -1.0 if optimizer_index % 2 else 1.0,
                        "bonferroni_significant": factor_index == 0,
                    }
                )
        matrix = first_order_significance_matrix(
            pd.DataFrame(rows), "CIFAR10", "test_accuracy_10_epoch"
        )
        self.assertEqual(matrix.shape, (len(FACTOR_COLUMNS), len(OPTIMIZERS)))
        np.testing.assert_array_equal(matrix[0], [1, -1, 1, -1, 1])
        self.assertTrue(np.all(matrix[1:] == 0))


if __name__ == "__main__":
    unittest.main()
