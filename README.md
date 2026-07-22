# machining-project
# Machining Tool Health Analysis

A Python tool for analyzing sensor signals from machine tool health monitoring data, with a focus on detecting simulated faults (e.g. tool wear, misalignment) in machining processes.

This project was developed as part of a Data Science and Research Software module assessment, using the dataset:

> Dominguez Caballero, J.A., Moore, J., Stammers, J. (2023). *Sensor signals for machine tool and process health assessment*. The University of Sheffield. Dataset. https://doi.org/10.15131/shef.data.24125715.v1

## What this does

The software loads sensor data (vibration, power, position) recorded during milling operations, computes summary statistics (mean, standard deviation, min, max) for individual signal segments, and compares a baseline (healthy) file against a fault-condition file to assess whether a given sensor field can distinguish between them.

## Where to get the data

The dataset is **not included in this repository** (files are several GB in size). Download the `.mat` files directly from the official source:

https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715

Place downloaded `.mat` files inside a local `data/` folder at the project root (this folder is git-ignored and won't be tracked).

## Installation

1. Clone this repository:
2. Install the required Python libraries:
## Usage

Run the command-line tool from inside the `src` folder, pointing it at two `.mat` files to compare:
**Arguments:**
| Argument | Required | Description |
|---|---|---|
| `--file1` | Yes | Path to the first `.mat` file (typically the baseline/healthy condition) |
| `--group1` | Yes | Name of the top-level group inside file1 (e.g. `BaselineCrop`) |
| `--file2` | Yes | Path to the second `.mat` file (typically a fault condition) |
| `--group2` | Yes | Name of the top-level group inside file2 (e.g. `ToolWearCrop`) |
| `--field` | No (default: `SpindleX`) | Which sensor field to analyze |
| `--segments` | No (default: `10`) | How many recorded segments to compare |

View all options at any time with:
## Running the tests

From the project root:
Tests use a small, synthetically generated `.mat`-style file (created automatically during testing) so they can run without needing the full dataset downloaded.

## Project structure
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
