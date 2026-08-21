from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import network_builder
import settings


FACTOR_COLUMNS = [
    "batch_size_coded",
    "dropout_flag_coded",
    "bn_flag_coded",
    "max_pool_flag_coded",
    "initialization_coded",
    "cnn_width_coded",
    "conv_dim_list_coded",
    "fc_width_coded",
    "fc_dim_list_coded",
]


def design_rows() -> list[dict[str, int]]:
    with (EXPERIMENT_ROOT / "experiment_list.csv").open(newline="") as handle:
        return [
            {key: int(float(value)) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


class ExperimentTests(unittest.TestCase):
    def test_projected_design(self) -> None:
        rows = design_rows()
        self.assertEqual(len(rows), 80)
        self.assertEqual(set(rows[0]), set(FACTOR_COLUMNS))
        self.assertEqual(
            len({tuple(row[column] for column in FACTOR_COLUMNS) for row in rows}),
            80,
        )
        for row in rows[:64]:
            self.assertEqual(
                row["cnn_width_coded"],
                row["dropout_flag_coded"]
                * row["bn_flag_coded"]
                * row["max_pool_flag_coded"]
                * row["initialization_coded"],
            )
            self.assertEqual(
                row["conv_dim_list_coded"],
                row["batch_size_coded"]
                * row["bn_flag_coded"]
                * row["max_pool_flag_coded"]
                * row["initialization_coded"],
            )
            self.assertEqual(
                row["fc_dim_list_coded"],
                row["bn_flag_coded"]
                * row["max_pool_flag_coded"]
                * row["fc_width_coded"],
            )

    def test_every_network_uses_leaky_relu(self) -> None:
        for row in design_rows():
            decoded = settings.decode({"dataset_coded": "CIFAR10", **row})
            network = network_builder.Net(
                img_size=decoded["img_size_decoded"],
                input_dim=decoded["input_dim_decoded"],
                output_dim=decoded["output_dim_decoded"],
                dropout_flag=decoded["dropout_flag_decoded"],
                bn_flag=decoded["bn_flag_decoded"],
                max_pool_flag=decoded["max_pool_flag_decoded"],
                conv_dim_list=decoded["conv_dim_list_decoded"],
                fc_dim_list=decoded["fc_dim_list_decoded"],
            )
            nonlinear = [
                module
                for module in network.modules()
                if isinstance(module, (nn.ReLU, nn.LeakyReLU))
            ]
            self.assertTrue(nonlinear)
            self.assertTrue(all(isinstance(module, nn.LeakyReLU) for module in nonlinear))


if __name__ == "__main__":
    unittest.main()
