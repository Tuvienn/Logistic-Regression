"""Phase 05: Logistic Regression tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from phases.phase_01_data_loading import load_iris_dataset
from phases.phase_03_preprocessing import PreprocessedData, run_preprocessing


OUTPUT_DIR = Path("outputs/tuning")


@dataclass(frozen=True)
class TunedModelReport:
    """Container for Phase 5 model tuning results."""

    best_estimator: LogisticRegression
    best_params: dict[str, Any]
    best_cv_score: float


def run_tuning(
    data: PreprocessedData, export_artifacts: bool = True
) -> TunedModelReport:
    """Perform hyperparameter tuning for Logistic Regression."""

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "newton-cg", "saga"],
        "max_iter": [1000],
    }

    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        estimator=lr, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )

    grid_search.fit(data.X_train_scaled, data.y_train)

    report = TunedModelReport(
        best_estimator=grid_search.best_estimator_,
        best_params=grid_search.best_params_,
        best_cv_score=float(grid_search.best_score_),
    )

    if export_artifacts:
        export_tuning_artifacts(report)

    return report


def export_tuning_artifacts(report: TunedModelReport) -> None:
    """Export tuning results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "best_params": report.best_params,
        "best_cv_score": report.best_cv_score,
    }

    with open(OUTPUT_DIR / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    """Run Phase 5 as a standalone step."""
    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=False)
    report = run_tuning(preprocessed, export_artifacts=True)

    print("Phase 5 completed: Hyperparameter tuning finished.")
    print(f"Best parameters: {report.best_params}")
    print(f"Best CV accuracy: {report.best_cv_score:.4f}")


if __name__ == "__main__":
    main()
