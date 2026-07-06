"""Phase 06: Final model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from phases.phase_01_data_loading import load_iris_dataset
from phases.phase_03_preprocessing import PreprocessedData, run_preprocessing
from phases.phase_05_model_tuning import TunedModelReport, run_tuning


OUTPUT_DIR = Path("outputs/evaluation")


@dataclass(frozen=True)
class EvaluationReport:
    """Container for Phase 6 evaluation results."""

    accuracy: float
    confusion_matrix: pd.DataFrame
    classification_report: str
    classification_report_dict: dict


def run_evaluation(
    data: PreprocessedData, model_report: TunedModelReport, export_artifacts: bool = True
) -> EvaluationReport:
    """Evaluate the best tuned model on the test set."""

    model = model_report.best_estimator
    y_pred = model.predict(data.X_test_scaled)

    accuracy = float(accuracy_score(data.y_test, y_pred))
    cm_array = confusion_matrix(data.y_test, y_pred)

    # Extract unique labels to use as row/col names
    labels = sorted(data.y_test.unique())
    cm_df = pd.DataFrame(cm_array, index=labels, columns=labels)

    report_str = classification_report(data.y_test, y_pred)
    report_dict = classification_report(data.y_test, y_pred, output_dict=True)

    report = EvaluationReport(
        accuracy=accuracy,
        confusion_matrix=cm_df,
        classification_report=report_str,
        classification_report_dict=report_dict,
    )

    if export_artifacts:
        export_evaluation_artifacts(report)

    return report


def export_evaluation_artifacts(report: EvaluationReport) -> None:
    """Export evaluation metrics and plots to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.confusion_matrix.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    report_df = pd.DataFrame(report.classification_report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    _plot_confusion_matrix(report.confusion_matrix)


def _plot_confusion_matrix(cm_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run Phase 6 as a standalone step."""
    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=False)
    tuned = run_tuning(preprocessed, export_artifacts=False)
    report = run_evaluation(preprocessed, tuned, export_artifacts=True)

    print("Phase 6 completed: Final model evaluated.")
    print(f"Test Accuracy: {report.accuracy:.4f}")
    print("Classification Report:")
    print(report.classification_report)


if __name__ == "__main__":
    main()
