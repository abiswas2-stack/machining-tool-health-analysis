"""Utilities for loading and comparing segmented machining sensor data."""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np


SUPPORTED_FEATURES = (
    "mean",
    "std",
    "min",
    "max",
    "rms",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
)


class MachiningFile:
    """Represent one MATLAB v7.3/HDF5 machining sensor-data file.

    Parameters
    ----------
    file_path:
        Path to the .mat file.
    group_name:
        Name of the top-level HDF5 group containing segmented sensor fields.
    missing_data:
        Strategy for non-finite values. "drop" removes NaN/Inf values;
        "raise" stops the analysis with a clear error.
    """

    def __init__(
        self,
        file_path: str | Path,
        group_name: str,
        missing_data: str = "drop",
    ) -> None:
        """Initialise a machining sensor-data file."""
        self.file_path = Path(file_path)
        self.group_name = group_name

        if missing_data not in {"drop", "raise"}:
            raise ValueError(
                "missing_data must be either 'drop' or 'raise'."
            )

        self.missing_data = missing_data

    def _validate_file(self) -> None:
        """Raise a clear error when the data file cannot be opened."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"Expected a file but received: {self.file_path}"
            )

    def _get_field_dataset(
        self,
        h5_file: h5py.File,
        field_name: str,
    ) -> h5py.Dataset:
        """Return a sensor field after validating group and field names."""
        if self.group_name not in h5_file:
            available_groups = list(h5_file.keys())

            raise KeyError(
                f"Group '{self.group_name}' was not found in "
                f"{self.file_path}. Available top-level groups: "
                f"{available_groups}"
            )

        group = h5_file[self.group_name]

        if field_name not in group:
            available_fields = list(group.keys())

            raise KeyError(
                f"Field '{field_name}' was not found in group "
                f"'{self.group_name}'. Available fields: "
                f"{available_fields}"
            )

        return group[field_name]

    def load_segment(
        self,
        field_name: str,
        segment_index: int,
    ) -> np.ndarray:
        """Load one sensor segment from the selected field.

        Parameters
        ----------
        field_name:
            Name of the sensor field, such as SpindleX.
        segment_index:
            Zero-based index of the segment to load.

        Returns
        -------
        numpy.ndarray
            Numeric samples from the requested segment.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        KeyError
            If the requested group or field is absent.
        IndexError
            If the requested segment index is unavailable.
        """
        self._validate_file()

        with h5py.File(self.file_path, "r") as h5_file:
            field_dataset = self._get_field_dataset(
                h5_file,
                field_name,
            )

            number_of_segments = field_dataset.shape[0]

            if (
                segment_index < 0
                or segment_index >= number_of_segments
            ):
                raise IndexError(
                    f"segment_index must be between 0 and "
                    f"{number_of_segments - 1}; received "
                    f"{segment_index}."
                )

            reference = field_dataset[segment_index, 0]

            signal = np.asarray(
                h5_file[reference][:],
                dtype=float,
            )

        return signal

    def _clean_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Flatten a signal and apply the missing-data policy."""
        flattened = np.asarray(
            signal,
            dtype=float,
        ).ravel()

        finite_mask = np.isfinite(flattened)
        missing_count = int((~finite_mask).sum())

        if missing_count and self.missing_data == "raise":
            raise ValueError(
                f"Signal contains {missing_count} "
                "NaN or infinite value(s)."
            )

        cleaned = flattened[finite_mask]

        if cleaned.size == 0:
            raise ValueError(
                "Signal contains no finite numeric values."
            )

        return cleaned

    def compute_stats(
        self,
        field_name: str,
        segment_index: int,
    ) -> Dict[str, float]:
        """Compute time-domain features for one sensor segment.

        The returned features are:

        - mean
        - standard deviation
        - minimum
        - maximum
        - root mean square
        - peak-to-peak amplitude
        - Pearson kurtosis
        - crest factor
        """
        raw_signal = self.load_segment(
            field_name,
            segment_index,
        )

        signal = self._clean_signal(raw_signal)

        mean_value = float(np.mean(signal))
        std_value = float(np.std(signal))
        rms_value = float(
            np.sqrt(np.mean(np.square(signal)))
        )

        if std_value == 0.0:
            kurtosis_value = float("nan")
        else:
            fourth_moment = float(
                np.mean(
                    np.power(
                        signal - mean_value,
                        4,
                    )
                )
            )

            kurtosis_value = (
                fourth_moment / (std_value**4)
            )

        if rms_value == 0.0:
            crest_factor_value = float("nan")
        else:
            crest_factor_value = float(
                np.max(np.abs(signal)) / rms_value
            )

        return {
            "mean": mean_value,
            "std": std_value,
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "rms": rms_value,
            "peak_to_peak": float(np.ptp(signal)),
            "kurtosis": kurtosis_value,
            "crest_factor": crest_factor_value,
        }

    def num_segments(
        self,
        field_name: str,
    ) -> int:
        """Return the number of available segments for a field."""
        self._validate_file()

        with h5py.File(self.file_path, "r") as h5_file:
            field_dataset = self._get_field_dataset(
                h5_file,
                field_name,
            )

            return int(field_dataset.shape[0])

    def compare_to(
        self,
        other_file: "MachiningFile",
        field_name: str,
        num_segments: int = 10,
        feature_name: str = "std",
    ) -> Dict[str, object]:
        """Compare one feature across matching segments in two files.

        This method provides an exploratory comparison between two
        conditions. It does not by itself constitute a trained or
        validated fault classifier.

        Parameters
        ----------
        other_file:
            Second MachiningFile, normally representing a fault
            condition.
        field_name:
            Sensor field to compare.
        num_segments:
            Number of matching segments to analyse.
        feature_name:
            Statistical feature used for the comparison.

        Returns
        -------
        dict
            Segment-level values and summary comparison results.
        """
        if num_segments <= 0:
            raise ValueError(
                "num_segments must be greater than zero."
            )

        if feature_name not in SUPPORTED_FEATURES:
            raise ValueError(
                f"Unsupported feature '{feature_name}'. "
                f"Choose from: "
                f"{', '.join(SUPPORTED_FEATURES)}"
            )

        available_segments = min(
            self.num_segments(field_name),
            other_file.num_segments(field_name),
        )

        if num_segments > available_segments:
            raise ValueError(
                f"Requested {num_segments} segments, but only "
                f"{available_segments} are available in both files."
            )

        baseline_values = np.array(
            [
                self.compute_stats(
                    field_name,
                    index,
                )[feature_name]
                for index in range(num_segments)
            ],
            dtype=float,
        )

        other_values = np.array(
            [
                other_file.compute_stats(
                    field_name,
                    index,
                )[feature_name]
                for index in range(num_segments)
            ],
            dtype=float,
        )

        valid_pairs = (
            np.isfinite(baseline_values)
            & np.isfinite(other_values)
        )

        if not np.any(valid_pairs):
            raise ValueError(
                f"Feature '{feature_name}' produced "
                "no finite comparison values."
            )

        valid_baseline = baseline_values[valid_pairs]
        valid_other = other_values[valid_pairs]

        percent_higher = float(
            np.mean(
                valid_other > valid_baseline
            )
            * 100.0
        )

        return {
            "feature": feature_name,
            "baseline_values": baseline_values,
            "other_values": other_values,
            "mean_baseline": float(
                np.mean(valid_baseline)
            ),
            "mean_other": float(
                np.mean(valid_other)
            ),
            "percent_segments_higher": percent_higher,
            "segments_compared": int(valid_pairs.sum()),
        }