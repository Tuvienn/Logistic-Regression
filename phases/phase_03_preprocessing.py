"""Phase 03: Train/test split and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from phases.phase_01_data_loading import IrisDataset, load_iris_dataset


OUTPUT_DIR = Path("outputs/preprocessing")


@dataclass(frozen=True)
class PreprocessedData:
    """Container for Phase 3 preprocessed outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    scaler: StandardScaler


def run_preprocessing(
    dataset: IrisDataset,
    test_size: float = 0.2,
    random_state: int = 42,
    export_artifacts: bool = True,
) -> PreprocessedData:
    """Split data into train/test sets and apply scaling."""

    X = dataset.features
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array, columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array, columns=X.columns, index=X_test.index
    )

    preprocessed = PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
    )

    if export_artifacts:
        export_preprocessing_artifacts(preprocessed)

    return preprocessed


def export_preprocessing_artifacts(data: PreprocessedData) -> None:
    """Export the preprocessed datasets to disk."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data.X_train_scaled.to_csv(OUTPUT_DIR / "X_train_scaled.csv", index=False)
    data.X_test_scaled.to_csv(OUTPUT_DIR / "X_test_scaled.csv", index=False)
    data.y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    data.y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


def main() -> None:
    """Run Phase 3 as a standalone step."""

    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=True)

    print("Phase 3 completed: Train/Test split and Scaling applied.")
    print(f"X_train shape: {preprocessed.X_train_scaled.shape}")
    print(f"X_test shape: {preprocessed.X_test_scaled.shape}")
    print(f"Exported artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
