"""Unit tests for the MachiningFile class."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from machining_file import MachiningFile  # noqa: E402


def create_fake_mat_file(
    file_path: Path,
    group_name: str,
    segments: list[np.ndarray],
) -> Path:
    """Create a small MATLAB-style HDF5 test file."""
    with h5py.File(file_path, "w") as h5_file:
        group = h5_file.create_group(group_name)

        field_dataset = group.create_dataset(
            "SpindleX",
            shape=(len(segments), 1),
            dtype=h5py.ref_dtype,
        )

        for index, values in enumerate(segments):
            dataset = h5_file.create_dataset(
                f"segment_{index}_data",
                data=np.asarray(
                    values,
                    dtype=float,
                ).reshape(-1, 1),
            )

            field_dataset[index, 0] = dataset.ref

    return file_path


@pytest.fixture
def fake_mat_file(
    tmp_path: Path,
) -> tuple[str, str]:
    """Return a synthetic file with known signal values."""
    file_path = create_fake_mat_file(
        tmp_path / "fake_data.mat",
        "TestCrop",
        [
            np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0]
            ),
            np.array(
                [10.0, 20.0, 30.0]
            ),
        ],
    )

    return str(file_path), "TestCrop"


def test_load_segment_returns_correct_values(
    fake_mat_file: tuple[str, str],
) -> None:
    """Check that a referenced signal is loaded correctly."""
    file_path, group_name = fake_mat_file

    machining_file = MachiningFile(
        file_path,
        group_name,
    )

    result = machining_file.load_segment(
        "SpindleX",
        0,
    )

    expected = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    assert np.array_equal(result, expected)


def test_compute_stats_returns_expected_features(
    fake_mat_file: tuple[str, str],
) -> None:
    """Check calculated features against known values."""
    file_path, group_name = fake_mat_file

    machining_file = MachiningFile(
        file_path,
        group_name,
    )

    stats = machining_file.compute_stats(
        "SpindleX",
        0,
    )

    assert stats["mean"] == pytest.approx(3.0)

    assert stats["std"] == pytest.approx(
        np.std([1, 2, 3, 4, 5])
    )

    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(5.0)

    assert stats["rms"] == pytest.approx(
        np.sqrt(11.0)
    )

    assert stats["peak_to_peak"] == pytest.approx(
        4.0
    )

    assert stats["crest_factor"] == pytest.approx(
        5.0 / np.sqrt(11.0)
    )


def test_num_segments_counts_correctly(
    fake_mat_file: tuple[str, str],
) -> None:
    """Check that the available segments are counted."""
    file_path, group_name = fake_mat_file

    machining_file = MachiningFile(
        file_path,
        group_name,
    )

    assert (
        machining_file.num_segments("SpindleX")
        == 2
    )


def test_compare_to_detects_higher_fault_variability(
    tmp_path: Path,
) -> None:
    """Check the main baseline-versus-fault comparison."""
    baseline_path = create_fake_mat_file(
        tmp_path / "baseline.mat",
        "BaselineCrop",
        [
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 1.0, 2.0]),
        ],
    )

    fault_path = create_fake_mat_file(
        tmp_path / "fault.mat",
        "FaultCrop",
        [
            np.array([-4.0, 4.0, -4.0, 4.0]),
            np.array([-8.0, 8.0, -8.0, 8.0]),
        ],
    )

    baseline = MachiningFile(
        baseline_path,
        "BaselineCrop",
    )

    fault = MachiningFile(
        fault_path,
        "FaultCrop",
    )

    result = baseline.compare_to(
        fault,
        "SpindleX",
        num_segments=2,
        feature_name="std",
    )

    assert (
        result["mean_other"]
        > result["mean_baseline"]
    )

    assert result[
        "percent_segments_higher"
    ] == pytest.approx(100.0)

    assert result["segments_compared"] == 2


def test_compare_to_rejects_too_many_segments(
    fake_mat_file: tuple[str, str],
) -> None:
    """Check that unavailable segment requests are rejected."""
    file_path, group_name = fake_mat_file

    first = MachiningFile(
        file_path,
        group_name,
    )

    second = MachiningFile(
        file_path,
        group_name,
    )

    with pytest.raises(
        ValueError,
        match="only 2 are available",
    ):
        first.compare_to(
            second,
            "SpindleX",
            num_segments=3,
        )


def test_missing_values_can_be_dropped(
    tmp_path: Path,
) -> None:
    """Check the default missing-data handling policy."""
    file_path = create_fake_mat_file(
        tmp_path / "missing.mat",
        "TestCrop",
        [
            np.array(
                [1.0, np.nan, 3.0, np.inf]
            )
        ],
    )

    machining_file = MachiningFile(
        file_path,
        "TestCrop",
        missing_data="drop",
    )

    stats = machining_file.compute_stats(
        "SpindleX",
        0,
    )

    assert stats["mean"] == pytest.approx(2.0)


def test_missing_values_can_raise_an_error(
    tmp_path: Path,
) -> None:
    """Check the strict missing-data handling policy."""
    file_path = create_fake_mat_file(
        tmp_path / "missing.mat",
        "TestCrop",
        [
            np.array(
                [1.0, np.nan, 3.0]
            )
        ],
    )

    machining_file = MachiningFile(
        file_path,
        "TestCrop",
        missing_data="raise",
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        machining_file.compute_stats(
            "SpindleX",
            0,
        )