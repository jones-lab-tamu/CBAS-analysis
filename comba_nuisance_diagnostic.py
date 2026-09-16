"""Quantify descriptive nuisance relationships in existing raw COMBA outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ANIMALS = (
    ("LacZ", "675G"),
    ("LacZ", "675H"),
    ("LacZ", "675I"),
    ("LacZ", "675J"),
    ("Bmal1KO", "714D"),
    ("Bmal1KO", "714E"),
    ("Bmal1KO", "714G"),
    ("Bmal1KO", "714H"),
)
ANIMAL_SUMMARY_COLUMNS = (
    "Animal",
    "Reported_FRP_h",
    "Computational_FRP_h",
    "Start_CT",
    "Complete_cycle_indices",
    "Full_cycles",
    "Adjacent_pairs",
    "Mean_raw_COMBA",
    "Median_nearest_same_label_distance_min",
    "Mean_nonrest_bouts_per_cycle",
)
PAIR_COLUMNS = (
    "Animal",
    "Cycle_pair",
    "S_0h",
    "S_8h",
    "S_16h",
    "Pair_COMBA",
    "Bouts_cycle_A",
    "Bouts_cycle_B",
    "Median_nearest_same_label_distance_min",
)

DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "CBAS_Analysis_Data"
    / "Cohort_Data"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT.parent / "COMBA_Nuisance_Diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify descriptive nuisance relationships from COMBA outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Cohort_Data directory containing the eight animal folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the three tables and four scatter plots.",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _validate_numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    path: Path,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.all(np.isfinite(values.to_numpy(dtype=float))):
            raise ValueError(f"{path} contains non-finite values in {column}")
        frame[column] = values


def read_comba_outputs(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the existing per-animal COMBA summary and pair files."""

    animal_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    for genotype, animal in ANIMALS:
        output_dir = data_root / genotype / animal / "COMBA_Output"
        summary_path = output_dir / "animal_comba_summary.csv"
        pair_path = output_dir / "comba_pair_scores.csv"
        if not summary_path.is_file() or not pair_path.is_file():
            raise FileNotFoundError(
                "Required existing COMBA output is missing: "
                f"{summary_path} or {pair_path}"
            )

        summary = pd.read_csv(summary_path)
        _require_columns(summary, ANIMAL_SUMMARY_COLUMNS, summary_path)
        if len(summary) != 1:
            raise ValueError(f"{summary_path} must contain exactly one row")
        if str(summary.loc[0, "Animal"]) != animal:
            raise ValueError(f"{summary_path} has an unexpected animal ID")
        _validate_numeric(
            summary,
            (
                "Mean_raw_COMBA",
                "Mean_nonrest_bouts_per_cycle",
                "Median_nearest_same_label_distance_min",
            ),
            summary_path,
        )
        animal_rows.append(
            {
                "Genotype": genotype,
                "Animal": animal,
                "Mean_raw_COMBA": summary.loc[0, "Mean_raw_COMBA"],
                "Mean_nonrest_bouts_per_cycle": summary.loc[
                    0, "Mean_nonrest_bouts_per_cycle"
                ],
                "Median_nearest_same_label_distance_min": summary.loc[
                    0, "Median_nearest_same_label_distance_min"
                ],
            }
        )

        pairs = pd.read_csv(pair_path)
        _require_columns(pairs, PAIR_COLUMNS, pair_path)
        if pairs.empty:
            raise ValueError(f"{pair_path} contains no adjacent-cycle pairs")
        if not np.all(pairs["Animal"].astype(str).eq(animal)):
            raise ValueError(f"{pair_path} has an unexpected animal ID")
        _validate_numeric(
            pairs,
            (
                "Pair_COMBA",
                "Bouts_cycle_A",
                "Bouts_cycle_B",
                "Median_nearest_same_label_distance_min",
            ),
            pair_path,
        )
        pair_subset = pairs.loc[:, list(PAIR_COLUMNS)].copy()
        pair_subset.insert(0, "Genotype", genotype)
        pair_frames.append(pair_subset)

    animal_table = pd.DataFrame(animal_rows)
    pair_table = pd.concat(pair_frames, ignore_index=True)
    if len(animal_table) != 8:
        raise AssertionError("Expected eight animal-level observations")
    if len(pair_table) != 21:
        raise AssertionError(
            f"Expected 21 pair-level observations, found {len(pair_table)}"
        )
    pair_table["Mean_pair_bout_count"] = (
        pair_table["Bouts_cycle_A"] + pair_table["Bouts_cycle_B"]
    ) / 2.0
    return animal_table, pair_table


def _correlation_rows(
    frame: pd.DataFrame,
    *,
    level: str,
    x_variable: str,
    y_variable: str,
    independence_note: str,
) -> list[dict[str, object]]:
    x = frame[x_variable].to_numpy(dtype=float)
    y = frame[y_variable].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite values found for {x_variable} or {y_variable}")
    if len(x) < 3:
        raise ValueError("Correlation requires at least three observations")

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return [
        {
            "Level": level,
            "X_variable": x_variable,
            "Y_variable": y_variable,
            "Method": "Pearson",
            "Correlation": float(pearson.statistic),
            "P_value": float(pearson.pvalue),
            "N": len(x),
            "Independence_note": independence_note,
        },
        {
            "Level": level,
            "X_variable": x_variable,
            "Y_variable": y_variable,
            "Method": "Spearman",
            "Correlation": float(spearman.statistic),
            "P_value": float(spearman.pvalue),
            "N": len(x),
            "Independence_note": independence_note,
        },
    ]


def calculate_correlations(
    animal_table: pd.DataFrame,
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _correlation_rows(
            animal_table,
            level="animal-level",
            x_variable="Mean_raw_COMBA",
            y_variable="Mean_nonrest_bouts_per_cycle",
            independence_note="Descriptive only; n=8, not inferential evidence.",
        )
    )
    rows.extend(
        _correlation_rows(
            animal_table,
            level="animal-level",
            x_variable="Mean_raw_COMBA",
            y_variable="Median_nearest_same_label_distance_min",
            independence_note="Descriptive only; n=8, not inferential evidence.",
        )
    )
    rows.extend(
        _correlation_rows(
            pair_table,
            level="pair-level",
            x_variable="Pair_COMBA",
            y_variable="Mean_pair_bout_count",
            independence_note=(
                "Descriptive only; 21 pairs are non-independent within animals."
            ),
        )
    )
    rows.extend(
        _correlation_rows(
            pair_table,
            level="pair-level",
            x_variable="Pair_COMBA",
            y_variable="Median_nearest_same_label_distance_min",
            independence_note=(
                "Descriptive only; 21 pairs are non-independent within animals."
            ),
        )
    )
    return pd.DataFrame(rows)


def _scatter_plot(
    frame: pd.DataFrame,
    *,
    x_variable: str,
    y_variable: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    x = frame[x_variable].to_numpy(dtype=float)
    y = frame[y_variable].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(x, y)
    axis.plot(x_line, slope * x_line + intercept)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def create_plots(
    animal_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    _scatter_plot(
        animal_table,
        x_variable="Mean_raw_COMBA",
        y_variable="Mean_nonrest_bouts_per_cycle",
        title="Animal-level: COMBA vs mean non-rest bout count",
        x_label="Mean raw COMBA",
        y_label="Mean non-rest bouts per cycle",
        output_path=output_dir / "animal_comba_vs_mean_bout_count.png",
    )
    _scatter_plot(
        animal_table,
        x_variable="Mean_raw_COMBA",
        y_variable="Median_nearest_same_label_distance_min",
        title="Animal-level: COMBA vs same-label candidate distance",
        x_label="Mean raw COMBA",
        y_label="Median nearest same-label distance (min)",
        output_path=output_dir / "animal_comba_vs_same_label_distance.png",
    )
    _scatter_plot(
        pair_table,
        x_variable="Pair_COMBA",
        y_variable="Mean_pair_bout_count",
        title="Pair-level: COMBA vs mean pair bout count",
        x_label="Pair COMBA",
        y_label="Mean pair bout count",
        output_path=output_dir / "pair_comba_vs_mean_bout_count.png",
    )
    _scatter_plot(
        pair_table,
        x_variable="Pair_COMBA",
        y_variable="Median_nearest_same_label_distance_min",
        title="Pair-level: COMBA vs same-label candidate distance",
        x_label="Pair COMBA",
        y_label="Median nearest same-label distance (min)",
        output_path=output_dir / "pair_comba_vs_same_label_distance.png",
    )


def run(data_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    animal_table, pair_table = read_comba_outputs(data_root.resolve())
    correlations = calculate_correlations(animal_table, pair_table)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    animal_table.to_csv(output_dir / "comba_animal_nuisance_table.csv", index=False)
    pair_table.to_csv(output_dir / "comba_pair_nuisance_table.csv", index=False)
    correlations.to_csv(output_dir / "comba_nuisance_correlations.csv", index=False)
    create_plots(animal_table, pair_table, output_dir)
    return animal_table, pair_table, correlations


def main() -> None:
    args = parse_args()
    animal_table, pair_table, correlations = run(args.data_root, args.output_dir)
    print(animal_table.to_string(index=False))
    print(pair_table.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
