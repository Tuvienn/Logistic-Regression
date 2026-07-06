"""Phase 07: Prediction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phases.phase_01_data_loading import load_iris_dataset
from phases.phase_03_preprocessing import PreprocessedData, run_preprocessing
from phases.phase_05_model_tuning import TunedModelReport, run_tuning


OUTPUT_DIR = Path("outputs/predictions")


@dataclass(frozen=True)
class PredictionAnalysisReport:
    """Container for Phase 7 prediction results."""

    predictions_df: pd.DataFrame
    misclassified_df: pd.DataFrame


def run_prediction_analysis(
    data: PreprocessedData, model_report: TunedModelReport, export_artifacts: bool = True
) -> PredictionAnalysisReport:
    """Analyze predictions and extract misclassified samples."""

    model = model_report.best_estimator
    y_pred = model.predict(data.X_test_scaled)
    y_proba = model.predict_proba(data.X_test_scaled)

    # Original unscaled features for context
    X_test_original = data.X_test.copy()

    predictions_df = X_test_original.copy()
    predictions_df["true_label"] = data.y_test
    predictions_df["predicted_label"] = y_pred

    classes = model.classes_
    for i, class_name in enumerate(classes):
        predictions_df[f"prob_{class_name}"] = y_proba[:, i]

    predictions_df["is_correct"] = (
        predictions_df["true_label"] == predictions_df["predicted_label"]
    )

    misclassified_df = predictions_df[~predictions_df["is_correct"]].copy()

    report = PredictionAnalysisReport(
        predictions_df=predictions_df, misclassified_df=misclassified_df
    )

    if export_artifacts:
        export_prediction_artifacts(report)

    return report


def export_prediction_artifacts(report: PredictionAnalysisReport) -> None:
    """Export prediction dataframes to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.predictions_df.to_csv(OUTPUT_DIR / "all_test_predictions.csv", index=False)
    if not report.misclassified_df.empty:
        report.misclassified_df.to_csv(
            OUTPUT_DIR / "misclassified_samples.csv", index=False
        )


def main() -> None:
    """Run Phase 7 as a standalone step."""
    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=False)
    tuned = run_tuning(preprocessed, export_artifacts=False)
    report = run_prediction_analysis(preprocessed, tuned, export_artifacts=True)

    print("Phase 7 completed: Predictions analyzed.")
    print(f"Total samples: {len(report.predictions_df)}")
    print(f"Misclassified samples: {len(report.misclassified_df)}")


if __name__ == "__main__":
    main()
