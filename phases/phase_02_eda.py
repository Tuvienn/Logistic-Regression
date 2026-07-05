"""Phase 02: Exploratory data analysis.

This phase analyzes dataset quality and feature behavior. It does not split,
scale, train, tune, or evaluate any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phases.phase_01_data_loading import (
    FEATURE_COLUMNS,
    SPECIES_COLUMN,
    IrisDataset,
    load_iris_dataset,
)


EDA_OUTPUT_DIR = Path("outputs/eda")


@dataclass(frozen=True)
class EDAReport:
    """Container for Phase 2 exploratory analysis outputs."""

    dataset_shape: tuple[int, int]
    missing_values: pd.Series
    duplicate_count: int
    class_distribution: pd.DataFrame
    feature_summary: pd.DataFrame
    class_feature_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame


def run_eda(dataset: IrisDataset, export_artifacts: bool = True) -> EDAReport:
    """Run EDA checks and summaries for the prepared Iris dataset."""

    dataframe = dataset.dataframe
    report = EDAReport(
        dataset_shape=dataframe.shape,
        missing_values=dataframe.isna().sum(),
        duplicate_count=int(dataframe.duplicated().sum()),
        class_distribution=_build_class_distribution(dataframe),
        feature_summary=dataframe[FEATURE_COLUMNS].describe().T,
        class_feature_summary=_build_class_feature_summary(dataframe),
        correlation_matrix=dataframe[FEATURE_COLUMNS].corr(),
    )

    if export_artifacts:
        export_eda_artifacts(report)

    return report


def export_eda_artifacts(report: EDAReport) -> None:
    """Export EDA tables and plots for notebook/report review."""

    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.missing_values.to_csv(
        EDA_OUTPUT_DIR / "missing_values.csv", header=["missing_count"]
    )
    report.class_distribution.to_csv(
        EDA_OUTPUT_DIR / "class_distribution.csv", index=False
    )
    report.feature_summary.to_csv(EDA_OUTPUT_DIR / "feature_summary.csv")
    report.class_feature_summary.to_csv(
        EDA_OUTPUT_DIR / "class_feature_summary.csv"
    )
    report.correlation_matrix.to_csv(EDA_OUTPUT_DIR / "correlation_matrix.csv")

    _plot_class_distribution(report.class_distribution)
    _plot_correlation_matrix(report.correlation_matrix)


def _build_class_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    class_counts = (
        dataframe[SPECIES_COLUMN]
        .value_counts()
        .rename_axis(SPECIES_COLUMN)
        .reset_index(name="count")
    )
    class_counts["percentage"] = class_counts["count"] / class_counts["count"].sum()
    return class_counts.sort_values(SPECIES_COLUMN).reset_index(drop=True)


def _build_class_feature_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.groupby(SPECIES_COLUMN)[FEATURE_COLUMNS].agg(["mean", "std"])


def _plot_class_distribution(class_distribution: pd.DataFrame) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        class_distribution[SPECIES_COLUMN],
        class_distribution["count"],
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    ax.set_title("Iris Class Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Sample Count")
    ax.bar_label(ax.containers[0])
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_correlation_matrix(correlation_matrix: pd.DataFrame) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)

    for row_index, row_name in enumerate(correlation_matrix.index):
        for column_index, column_name in enumerate(correlation_matrix.columns):
            value = correlation_matrix.loc[row_name, column_name]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def main() -> None:
    """Run Phase 2 as a standalone EDA step."""

    dataset = load_iris_dataset(export_csv=True)
    report = run_eda(dataset, export_artifacts=True)

    print("Phase 2 completed: EDA report generated.")
    print(f"Dataset shape: {report.dataset_shape}")
    print(f"Duplicate rows: {report.duplicate_count}")
    print("Missing values:")
    print(report.missing_values.to_string())
    print("Class distribution:")
    print(report.class_distribution.to_string(index=False))
    print(f"Exported EDA artifacts: {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
