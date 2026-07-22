import h5py
import numpy as np


class MachiningFile:
    """
    Represents one .mat sensor data file from the machining dataset.
    Knows how to load segments and compute basic statistics from itself.
    """

    def __init__(self, file_path, group_name):
        self.file_path = file_path
        self.group_name = group_name

    def load_segment(self, field_name, segment_index):
        """Loads one segment of sensor data from this file."""
        with h5py.File(self.file_path, 'r') as f:
            group = f[self.group_name]
            reference = group[field_name][segment_index, 0]
            actual_array = f[reference][:]
        return actual_array

    def compute_stats(self, field_name, segment_index):
        """Computes basic statistics for one segment."""
        signal = self.load_segment(field_name, segment_index)
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
        }

    def num_segments(self, field_name):
        """Returns how many segments exist for a given field (e.g. 420)."""
        with h5py.File(self.file_path, 'r') as f:
            group = f[self.group_name]
            return group[field_name].shape[0]

    def compare_to(self, other_file, field_name, num_segments=10):
        """
        Compares this file to another MachiningFile across several segments.

        Parameters:
            other_file (MachiningFile): the file to compare against (e.g. a fault file)
            field_name (str): sensor field to compare, e.g. 'SpindleX'
            num_segments (int): how many segments to check

        Returns:
            A dictionary with lists of std devs for both files, and a summary.
        """
        self_stds = []
        other_stds = []

        for i in range(num_segments):
            self_stds.append(self.compute_stats(field_name, i)['std'])
            other_stds.append(other_file.compute_stats(field_name, i)['std'])

        self_stds = np.array(self_stds)
        other_stds = np.array(other_stds)

        percent_higher = np.mean(other_stds > self_stds) * 100

        return {
            'baseline_stds': self_stds,
            'other_stds': other_stds,
            'mean_baseline_std': np.mean(self_stds),
            'mean_other_std': np.mean(other_stds),
            'percent_segments_higher': percent_higher,
        }