"""Nine-factor Resolution-IV design used by the manuscript."""

from __future__ import annotations

import itertools


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

OPTIMIZERS = ("sgd", "sgd_nesterov", "adam", "adamw", "rmsprop")

DEFINING_GENERATORS = (
    frozenset(("B", "C", "D", "F", "G")),
    frozenset(("A", "B", "G", "H")),
    frozenset(("C", "D", "I", "J")),
)

LETTER_TO_FACTOR = dict(zip("ABCDFGHIJ", FACTOR_COLUMNS))
FACTOR_TO_LETTER = {value: key for key, value in LETTER_TO_FACTOR.items()}


def generate_factorial_design() -> list[dict[str, int | str]]:
    rows = []
    for run_number, (a, b, c, d, f, i) in enumerate(
        itertools.product((-1, 1), repeat=6), start=1
    ):
        g = b * c * d * f
        h = a * c * d * f
        j = c * d * i
        values = (a, b, c, d, f, g, h, i, j)
        row = {name: int(value) for name, value in zip(FACTOR_COLUMNS, values)}
        row["setting_id"] = f"NF-F{run_number:03d}"
        row["design_slice"] = "factorial"
        rows.append(row)
    return rows


def generate_center_design() -> list[dict[str, int | str]]:
    rows = []
    for run_number, (b, c, d, f) in enumerate(
        itertools.product((-1, 1), repeat=4), start=1
    ):
        values = (0, b, c, d, f, 0, 0, 0, 0)
        row = {name: int(value) for name, value in zip(FACTOR_COLUMNS, values)}
        row["setting_id"] = f"NF-C{run_number:03d}"
        row["design_slice"] = "center"
        rows.append(row)
    return rows


def generate_design() -> list[dict[str, int | str]]:
    return generate_factorial_design() + generate_center_design()


def defining_relation() -> list[frozenset[str]]:
    words = [frozenset()]
    for generator in DEFINING_GENERATORS:
        words += [word.symmetric_difference(generator) for word in list(words)]
    return sorted(set(words), key=lambda word: (len(word), tuple(sorted(word))))


def resolution() -> int:
    return min(len(word) for word in defining_relation() if word)
