import sys
import os
import h5py
import numpy as np
import pytest

# This lets the test find machining_file.py, which lives in ../src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from machining_file import MachiningFile


@pytest.fixture
def fake_mat_file(tmp_path):
    """
    Creates a small, fake .mat-style HDF5 file for testing,
    with known values, so we can check our code against a correct answer.
    """
    file_path = tmp_path / "fake_data.mat"

    with h5py.File(file_path, 'w') as f:
        group = f.create_group('TestCrop')

        # Create two tiny "segments" with known values
        segment_0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        segment_1 = np.array([10.0, 20.0, 30.0]).reshape(-1, 1)

        ref0 = f.create_dataset('segment_0_data', data=segment_0)
        ref1 = f.create_dataset('segment_1_data', data=segment_1)

        # Build the "cell array" structure: a column of references
        field_dataset = group.create_dataset('SpindleX', shape=(2, 1), dtype=h5py.ref_dtype)
        field_dataset[0, 0] = ref0.ref
        field_dataset[1, 0] = ref1.ref

    return str(file_path), 'TestCrop'


def test_load_segment_returns_correct_values(fake_mat_file):
    file_path, group_name = fake_mat_file
    mf = MachiningFile(file_path, group_name)

    result = mf.load_segment('SpindleX', 0)

    expected = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert np.array_equal(result, expected)


def test_compute_stats_returns_correct_mean_and_std(fake_mat_file):
    file_path, group_name = fake_mat_file
    mf = MachiningFile(file_path, group_name)

    stats = mf.compute_stats('SpindleX', 0)

    # segment_0 = [1, 2, 3, 4, 5] -> mean = 3.0
    assert stats['mean'] == pytest.approx(3.0)
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0


def test_num_segments_counts_correctly(fake_mat_file):
    file_path, group_name = fake_mat_file
    mf = MachiningFile(file_path, group_name)

    assert mf.num_segments('SpindleX') == 2