"""Example exploratory comparison using the real dataset.

This script is not part of the command-line interface. It is retained
to demonstrate the exploratory development and analysis process.
"""

from pathlib import Path

from machining_file import MachiningFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"


def print_result(
    comparison_name: str,
    result: dict,
) -> None:
    """Print one comparison result clearly."""
    print(f"\n{comparison_name}")
    print(f"Feature: {result['feature']}")

    print(
        "Mean baseline value: "
        f"{result['mean_baseline']:.6g}"
    )

    print(
        "Mean fault value: "
        f"{result['mean_other']:.6g}"
    )

    print(
        "Segments where fault value is higher: "
        f"{result['percent_segments_higher']:.1f}%"
    )


def main() -> None:
    """Compare baseline data with two fault conditions."""
    baseline = MachiningFile(
        DATA_DIRECTORY
        / "Segmented_Machining_Baseline.mat",
        "BaselineCrop",
    )

    tool_wear = MachiningFile(
        DATA_DIRECTORY
        / "Segmented_Machining_ToolWear.mat",
        "ToolWearCrop",
    )

    misalignment = MachiningFile(
        DATA_DIRECTORY
        / "Segmented_Machining_Misalignment.mat",
        "MisalignmentCrop",
    )

    tool_wear_result = baseline.compare_to(
        tool_wear,
        "SpindleX",
        num_segments=10,
        feature_name="std",
    )

    misalignment_result = baseline.compare_to(
        misalignment,
        "SpindleX",
        num_segments=10,
        feature_name="std",
    )

    print_result(
        "Tool-wear comparison",
        tool_wear_result,
    )

    print_result(
        "Misalignment comparison",
        misalignment_result,
    )


if __name__ == "__main__":
    main()