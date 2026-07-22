"""Integration tests for the command-line interface."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "src" / "main.py"


def create_cli_file(
    file_path: Path,
    group_name: str,
    values: np.ndarray,
) -> Path:
    """Create a one-segment HDF5 file for CLI testing."""
    with h5py.File(file_path, "w") as h5_file:
        group = h5_file.create_group(group_name)

        signal = h5_file.create_dataset(
            "segment_data",
            data=np.asarray(
                values,
                dtype=float,
            ).reshape(-1, 1),
        )

        field = group.create_dataset(
            "SpindleX",
            shape=(1, 1),
            dtype=h5py.ref_dtype,
        )

        field[0, 0] = signal.ref

    return file_path


def test_cli_runs_successfully(
    tmp_path: Path,
) -> None:
    """Check that the CLI can complete a comparison."""
    baseline_path = create_cli_file(
        tmp_path / "baseline.mat",
        "BaselineCrop",
        np.array([0.0, 1.0, 0.0, 1.0]),
    )

    fault_path = create_cli_file(
        tmp_path / "fault.mat",
        "FaultCrop",
        np.array([-5.0, 5.0, -5.0, 5.0]),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--file1",
            str(baseline_path),
            "--group1",
            "BaselineCrop",
            "--file2",
            str(fault_path),
            "--group2",
            "FaultCrop",
            "--field",
            "SpindleX",
            "--segments",
            "1",
            "--feature",
            "std",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Feature: std" in completed.stdout
    assert "100.0%" in completed.stdout


def test_cli_returns_error_for_missing_file(
    tmp_path: Path,
) -> None:
    """Check that the CLI gives a controlled file error."""
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--file1",
            str(tmp_path / "does_not_exist.mat"),
            "--group1",
            "BaselineCrop",
            "--file2",
            str(tmp_path / "also_missing.mat"),
            "--group2",
            "FaultCrop",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Data file not found" in completed.stderr