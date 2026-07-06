"""Phase 04: Baseline models."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from phases.phase_01_data_loading import load_iris_dataset
from phases.phase_03_preprocessing import PreprocessedData, run_preprocessing


@dataclass(frozen=True)
class BaselineReport:
    """Container for Phase 4 baseline results."""

    strategy: str
    accuracy: float


def run_baseline(data: PreprocessedData) -> BaselineReport:
    """Train and evaluate a simple baseline model."""

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(data.X_train_scaled, data.y_train)
    y_pred = baseline.predict(data.X_test_scaled)

    accuracy = float(accuracy_score(data.y_test, y_pred))

    return BaselineReport(strategy="most_frequent", accuracy=accuracy)


def main() -> None:
    """Run Phase 4 as a standalone step."""

    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=False)
    report = run_baseline(preprocessed)

    print("Phase 4 completed: Baseline established.")
    print(f"Baseline strategy: {report.strategy}")
    print(f"Baseline accuracy: {report.accuracy:.4f}")


if __name__ == "__main__":
    main()
