"""Command-line interface for machining sensor comparison."""

import argparse
import sys
from collections.abc import Sequence

from machining_file import MachiningFile
from machining_file import SUPPORTED_FEATURES


def build_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a statistical feature between two segmented "
            "machining sensor-data files."
        )
    )

    parser.add_argument(
        "--file1",
        required=True,
        help=(
            "Path to the baseline or first MATLAB v7.3 "
            ".mat file."
        ),
    )

    parser.add_argument(
        "--group1",
        required=True,
        help=(
            "Top-level HDF5 group inside file1, "
            "for example BaselineCrop."
        ),
    )

    parser.add_argument(
        "--file2",
        required=True,
        help=(
            "Path to the fault-condition or second "
            ".mat file."
        ),
    )

    parser.add_argument(
        "--group2",
        required=True,
        help=(
            "Top-level HDF5 group inside file2, "
            "for example ToolWearCrop."
        ),
    )

    parser.add_argument(
        "--field",
        default="SpindleX",
        help=(
            "Sensor field to analyse "
            "(default: SpindleX)."
        ),
    )

    parser.add_argument(
        "--segments",
        type=int,
        default=10,
        help=(
            "Number of matching segments to compare "
            "(default: 10)."
        ),
    )

    parser.add_argument(
        "--feature",
        choices=SUPPORTED_FEATURES,
        default="std",
        help=(
            "Time-domain feature to compare "
            "(default: std)."
        ),
    )

    parser.add_argument(
        "--missing-data",
        choices=("drop", "raise"),
        default="drop",
        help=(
            "How to handle NaN/Inf samples: drop them "
            "or raise an error (default: drop)."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the command-line program and return an exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        file1 = MachiningFile(
            args.file1,
            args.group1,
            args.missing_data,
        )

        file2 = MachiningFile(
            args.file2,
            args.group2,
            args.missing_data,
        )

        result = file1.compare_to(
            file2,
            args.field,
            num_segments=args.segments,
            feature_name=args.feature,
        )

    except (
        FileNotFoundError,
        KeyError,
        IndexError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Comparing '{args.file1}' with "
        f"'{args.file2}' for field "
        f"'{args.field}'"
    )

    print(
        f"Feature: {result['feature']}"
    )

    print(
        f"Segments compared: "
        f"{result['segments_compared']}"
    )

    print(
        "Mean feature value (file1): "
        f"{result['mean_baseline']:.6g}"
    )

    print(
        "Mean feature value (file2): "
        f"{result['mean_other']:.6g}"
    )

    print(
        "Segments where file2 value is higher "
        "than file1: "
        f"{result['percent_segments_higher']:.1f}%"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())