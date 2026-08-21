"""Defining-relation, alias, and missingness tables."""

from __future__ import annotations

import itertools

import pandas as pd

from analysis.models.regression import (
    factorial_interaction_alias_classes,
    interaction_alias_classes,
)
from analysis.design import FACTOR_COLUMNS
from analysis.design import (
    FACTOR_TO_LETTER,
    LETTER_TO_FACTOR,
    defining_relation,
)


def _effect_label(effect: frozenset[str]) -> str:
    if not effect:
        return "I"
    return ":".join(LETTER_TO_FACTOR[letter] for letter in sorted(effect))


def alias_structure() -> pd.DataFrame:
    words = defining_relation()
    effects = [frozenset((FACTOR_TO_LETTER[factor],)) for factor in FACTOR_COLUMNS]
    effects += [
        frozenset((FACTOR_TO_LETTER[first], FACTOR_TO_LETTER[second]))
        for first, second in itertools.combinations(FACTOR_COLUMNS, 2)
    ]
    rows = []
    for effect in effects:
        aliases = sorted(
            (effect.symmetric_difference(word) for word in words),
            key=lambda item: (len(item), tuple(sorted(item))),
        )
        rows.append({
            "effect": _effect_label(effect),
            "effect_order": len(effect),
            "alias_set": " = ".join(_effect_label(alias) for alias in aliases),
            "lowest_alias_order": min(len(alias) for alias in aliases if alias != effect),
            "contains_two_factor_alias": sum(len(alias) == 2 for alias in aliases) > 1,
        })
    result = pd.DataFrame(rows)
    factorial_map = {}
    for item in factorial_interaction_alias_classes():
        for member in item["members"]:
            factorial_map[member] = item
    augmented_map = {}
    for item in interaction_alias_classes():
        for member in item["members"]:
            augmented_map[member] = item

    result["alias_set_scope"] = "factorial_64_run_defining_relation"
    result["factorial_two_factor_alias_class"] = result["effect"].map(
        lambda value: factorial_map.get(value, {}).get("alias_class_id", "")
    )
    result["factorial_two_factor_members"] = result["effect"].map(
        lambda value: ";".join(factorial_map.get(value, {}).get("members", []))
    )
    result["factorial_contains_paired_2fi_alias"] = result["effect"].map(
        lambda value: bool(factorial_map.get(value, {}).get("aliased", False))
    )
    result["augmented_estimable_class"] = result["effect"].map(
        lambda value: augmented_map.get(value, {}).get("alias_class_id", "")
    )
    result["augmented_class_members"] = result["effect"].map(
        lambda value: ";".join(augmented_map.get(value, {}).get("members", []))
    )
    result["augmented_contains_paired_2fi_alias"] = result["effect"].map(
        lambda value: bool(augmented_map.get(value, {}).get("aliased", False))
    )
    result["factorial_alias_split_by_center_slice"] = result["effect"].map(
        lambda value: bool(augmented_map.get(value, {}).get("split_by_center_slice", False))
    )
    result["center_slice_alias_effect"] = result.apply(
        lambda row: (
            "not_applicable_main_effect"
            if row["effect_order"] == 1
            else "factorial_alias_split_by_center_slice"
            if row["factorial_alias_split_by_center_slice"]
            else "exact_paired_alias_retained"
            if row["augmented_contains_paired_2fi_alias"]
            else "singleton_in_augmented_model"
        ),
        axis=1,
    )
    # Backward-compatible columns now explicitly refer to the fitted augmented design.
    result["estimable_alias_class"] = result["augmented_estimable_class"]
    result["modeled_representative"] = result["effect"].map(
        lambda value: augmented_map.get(value, {}).get("representative", "")
    )
    return result


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, optimizer), group in frame.groupby(["dataset", "optimizer"]):
        missing = group["gaussian_lipschitz"].isna()
        rows.append({
            "dataset": dataset,
            "optimizer": optimizer,
            "total_candidate_observations": len(group),
            "valid_gaussian_observations": int((~missing).sum()),
            "missing_gaussian_observations": int(missing.sum()),
            "missing_percentage": 100.0 * float(missing.mean()),
            "affected_nine_factor_settings": ";".join(sorted(group.loc[missing, "setting_id"].unique())),
            "affected_seeds": ";".join(map(str, sorted(group.loc[missing, "seed"].unique()))),
        })
    return pd.DataFrame(rows)
