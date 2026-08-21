"""Paths and constants for the manuscript analysis."""

from pathlib import Path

from analysis.design import FACTOR_COLUMNS


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
DATA_FILES = {
    "CIFAR10": PACKAGE_ROOT / "data" / "cifar10_nine_factor.jsonl",
    "CIFAR100": PACKAGE_ROOT / "data" / "cifar100_nine_factor.jsonl",
}
RESPONSES = ("gaussian_lipschitz", "test_accuracy_10_epoch")
ALPHA = 0.05

if len(FACTOR_COLUMNS) != 9:
    raise RuntimeError("The analysis configuration must contain exactly nine factors")
