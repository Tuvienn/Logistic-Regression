"""Phase 01: Data loading and schema preparation.

This phase only loads the Iris dataset, prepares a clean schema, validates the
expected structure, and optionally exports the prepared dataset for review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "target"
SPECIES_COLUMN = "species"
EXPECTED_TARGET_VALUES = {0, 1, 2}
OUTPUT_PATH = Path("outputs/iris_dataset.csv")


@dataclass(frozen=True)
class IrisDataset:
    """Container for the prepared Iris dataset."""

    dataframe: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    target_names: list[str]


def load_iris_dataset(export_csv: bool = True) -> IrisDataset:
    """Load Iris from scikit-learn and prepare a clean DataFrame schema."""

    raw_dataset = load_iris()
    target_names = [str(name) for name in raw_dataset.target_names]
    dataframe = pd.DataFrame(raw_dataset.data, columns=FEATURE_COLUMNS)
    dataframe[TARGET_COLUMN] = raw_dataset.target
    dataframe[SPECIES_COLUMN] = dataframe[TARGET_COLUMN].map(
        dict(enumerate(target_names))
    )

    dataset = IrisDataset(
        dataframe=dataframe,
        features=dataframe[FEATURE_COLUMNS].copy(),
        target=dataframe[TARGET_COLUMN].copy(),
        feature_names=FEATURE_COLUMNS.copy(),
        target_names=target_names,
    )
    validate_iris_dataset(dataset)

    if export_csv:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.dataframe.to_csv(OUTPUT_PATH, index=False)

    return dataset


def validate_iris_dataset(dataset: IrisDataset) -> None:
    """Validate the schema guarantees required by later phases."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, SPECIES_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataset.dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if dataset.dataframe.shape[0] != 150:
        raise ValueError(f"Expected 150 samples, got {dataset.dataframe.shape[0]}")

    if dataset.features.shape[1] != 4:
        raise ValueError(f"Expected 4 feature columns, got {dataset.features.shape[1]}")

    if len(dataset.target_names) != 3:
        raise ValueError(f"Expected 3 target classes, got {len(dataset.target_names)}")

    actual_targets = set(dataset.target.unique())
    if actual_targets != EXPECTED_TARGET_VALUES:
        raise ValueError(
            f"Expected target values {EXPECTED_TARGET_VALUES}, got {actual_targets}"
        )

    expected_species = {"setosa", "versicolor", "virginica"}
    actual_species = set(dataset.dataframe[SPECIES_COLUMN].unique())
    if actual_species != expected_species:
        raise ValueError(
            f"Expected species values {expected_species}, got {actual_species}"
        )


def main() -> None:
    """Run Phase 1 as a standalone validation step."""

    dataset = load_iris_dataset(export_csv=True)
    print("Phase 1 completed: Iris dataset loaded and validated.")
    print(f"Dataset shape: {dataset.dataframe.shape}")
    print(f"Feature columns: {dataset.feature_names}")
    print(f"Target classes: {dataset.target_names}")
    print(f"Exported dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
