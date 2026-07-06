"""Phase 08: Logistic Regression interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phases.phase_01_data_loading import FEATURE_COLUMNS, load_iris_dataset
from phases.phase_03_preprocessing import PreprocessedData, run_preprocessing
from phases.phase_05_model_tuning import TunedModelReport, run_tuning


OUTPUT_DIR = Path("outputs/interpretation")


@dataclass(frozen=True)
class InterpretationReport:
    """Container for Phase 8 model interpretation."""

    coefficients_df: pd.DataFrame
    intercepts: pd.Series


def run_interpretation(
    data: PreprocessedData, model_report: TunedModelReport, export_artifacts: bool = True
) -> InterpretationReport:
    """Extract and analyze model coefficients for interpretation."""

    model = model_report.best_estimator
    classes = model.classes_

    coefficients_df = pd.DataFrame(
        model.coef_, columns=FEATURE_COLUMNS, index=[f"class_{c}" for c in classes]
    )

    intercepts = pd.Series(model.intercept_, index=[f"class_{c}" for c in classes])

    report = InterpretationReport(
        coefficients_df=coefficients_df, intercepts=intercepts
    )

    if export_artifacts:
        export_interpretation_artifacts(report)

    return report


def export_interpretation_artifacts(report: InterpretationReport) -> None:
    """Export model interpretation results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.coefficients_df.to_csv(OUTPUT_DIR / "coefficients.csv")
    report.intercepts.to_csv(OUTPUT_DIR / "intercepts.csv", header=["intercept"])

    _plot_coefficients(report.coefficients_df)


def _plot_coefficients(coefficients_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    coefficients_df.T.plot(kind="bar", ax=ax)

    ax.set_title("Logistic Regression Coefficients by Class")
    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient Value")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coefficients_plot.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run Phase 8 as a standalone step."""
    dataset = load_iris_dataset(export_csv=False)
    preprocessed = run_preprocessing(dataset, export_artifacts=False)
    tuned = run_tuning(preprocessed, export_artifacts=False)
    report = run_interpretation(preprocessed, tuned, export_artifacts=True)

    print("Phase 8 completed: Model interpretation ready.")
    print("Intercepts:")
    print(report.intercepts)
    print("\nCoefficients:")
    print(report.coefficients_df)


if __name__ == "__main__":
    main()
