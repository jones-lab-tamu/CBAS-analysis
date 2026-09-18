"""Compute the equal-weight circular-W1 non-rest behavioral repertoire metric.

This standalone analysis consumes the existing WTA classifier output and the
frozen FRP/CT solutions.  It does not estimate FRP, infer phase, apply an
eligibility threshold, or perform genotype statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase


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
ANIMAL_ORDER = tuple(animal for _, animal in ANIMALS)
GENOTYPE_BY_ANIMAL = {animal: genotype for genotype, animal in ANIMALS}
NONREST_BEHAVIORS = tuple(
    frozen_phase.BEHAVIORS[index]
    for index in frozen_phase.NONRESTING_BEHAVIOR_INDICES
)
BEHAVIOR_INDEX = {
    behavior: frozen_phase.BEHAVIORS.index(behavior)
    for behavior in NONREST_BEHAVIORS
}

N_CYCLES = 4
N_CT_BINS = 288
CT_HOURS = frozen_phase.CT_HOURS
CT_BIN_WIDTH_HOURS = CT_HOURS / N_CT_BINS
RECORDING_START_CT = 18.0
OUTPUT_DIRNAME = "Circular_W1_Repertoire"
W1_TOLERANCE_HOURS = 1e-12
PROBABILITY_TOLERANCE = 1e-12

DISTRIBUTION_COLUMNS = [
    "animal",
    "genotype",
    "behavior",
    "ct_bin_start_hours",
    "ct_bin_center_hours",
    "occupancy",
    "probability",
]
PAIRWISE_COLUMNS = [
    "animal_i",
    "genotype_i",
    "animal_j",
    "genotype_j",
    "behavior",
    "w1_hours",
]
COMPOSITE_COLUMNS = [
    "animal_i",
    "genotype_i",
    "animal_j",
    "genotype_j",
    *[f"w1_{behavior}" for behavior in NONREST_BEHAVIORS],
    "repertoire_w1_mean_hours",
]
MATRIX_LONG_COLUMNS = [
    "animal_i",
    "genotype_i",
    "animal_j",
    "genotype_j",
    "behavior",
    "w1_hours",
]
DECOMPOSITION_COLUMNS = [
    "behavior",
    "n_pairs",
    "mean_w1_hours",
    "median_w1_hours",
    "sd_w1_hours",
    "min_w1_hours",
    "max_w1_hours",
    "iqr_w1_hours",
    "fraction_of_composite_summed_distance",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Cohort_Data directory containing the eight frozen pilot animals.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def circular_w1(
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    *,
    circumference_hours: float = CT_HOURS,
) -> float:
    """Calculate exact circular 1-Wasserstein distance on an equal grid.

    For equally spaced circular grid points, the cumulative probability
    difference across each edge is an edge flow.  The optimal flow offset is
    any median of those cumulative differences.  The sum of absolute residual
    flows times the grid spacing is the exact minimum transport cost on the
    circle, including the wrap-around edge.
    """

    first = np.asarray(probability_a, dtype=float)
    second = np.asarray(probability_b, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError(
            "Circular W1 requires two one-dimensional probability vectors of "
            f"the same shape, got {first.shape} and {second.shape}"
        )
    if first.size == 0:
        raise ValueError("Circular W1 requires at least one grid point")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Circular W1 received a non-finite probability")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise ValueError("Circular W1 received a negative probability")

    first_total = float(first.sum())
    second_total = float(second.sum())
    if first_total <= 0.0 or second_total <= 0.0:
        raise ValueError("Circular W1 requires positive probability mass")
    first = first / first_total
    second = second / second_total

    cumulative_difference = np.cumsum(first - second)
    optimal_offset = float(np.median(cumulative_difference))
    spacing_hours = circumference_hours / first.size
    distance = spacing_hours * float(
        np.abs(cumulative_difference - optimal_offset).sum()
    )
    if not np.isfinite(distance) or distance < -W1_TOLERANCE_HOURS:
        raise ValueError(f"Circular W1 was invalid: {distance}")
    return max(0.0, distance)


def _point_mass(ct_hours: float, n_bins: int = N_CT_BINS) -> np.ndarray:
    probability = np.zeros(n_bins, dtype=float)
    bin_width = CT_HOURS / n_bins
    index = int(round(ct_hours / bin_width)) % n_bins
    probability[index] = 1.0
    return probability


def run_synthetic_validation() -> dict[str, float]:
    """Run the required circular-W1 tests before reading real data."""

    identical = circular_w1(_point_mass(5.0), _point_mass(5.0))
    ct23_vs_ct1 = circular_w1(_point_mass(23.0), _point_mass(1.0))
    one_bin_shift = circular_w1(
        _point_mass(4.0),
        _point_mass(4.0 + CT_BIN_WIDTH_HOURS),
    )
    known_3h_shift = circular_w1(_point_mass(7.0), _point_mass(10.0))

    narrow = np.zeros(N_CT_BINS, dtype=float)
    broad = np.zeros(N_CT_BINS, dtype=float)
    center = int(round(12.0 / CT_BIN_WIDTH_HOURS))
    narrow[[center - 1, center, center + 1]] = [0.25, 0.50, 0.25]
    broad[[center - 4, center - 2, center, center + 2, center + 4]] = [
        0.10,
        0.20,
        0.40,
        0.20,
        0.10,
    ]
    fixed_center_broadening = circular_w1(narrow, broad)

    arbitrary_a = np.zeros(N_CT_BINS, dtype=float)
    arbitrary_b = np.zeros(N_CT_BINS, dtype=float)
    arbitrary_a[[3, 60, 144, 280]] = [0.10, 0.20, 0.50, 0.20]
    arbitrary_b[[8, 55, 150, 275]] = [0.30, 0.10, 0.40, 0.20]
    symmetry_ab = circular_w1(arbitrary_a, arbitrary_b)
    symmetry_ba = circular_w1(arbitrary_b, arbitrary_a)

    composite_components = [
        circular_w1(_point_mass(index * 2.0), _point_mass(index * 2.0 + 0.5))
        for index in range(len(NONREST_BEHAVIORS))
    ]
    composite_mean = float(np.mean(composite_components))
    composite_arithmetic_error = composite_mean - float(
        sum(composite_components) / len(composite_components)
    )

    matrix_distributions = [
        _point_mass(0.0),
        _point_mass(1.0),
        _point_mass(23.0),
    ]
    synthetic_matrix = np.asarray(
        [
            [
                circular_w1(first, second)
                for second in matrix_distributions
            ]
            for first in matrix_distributions
        ],
        dtype=float,
    )
    matrix_diagonal_max = float(np.max(np.abs(np.diag(synthetic_matrix))))
    matrix_symmetry_max = float(
        np.max(np.abs(synthetic_matrix - synthetic_matrix.T))
    )

    checks = {
        "identity_h": identical,
        "ct23_vs_ct1_h": ct23_vs_ct1,
        "one_bin_shift_h": one_bin_shift,
        "known_3h_shift_h": known_3h_shift,
        "fixed_center_broadening_h": fixed_center_broadening,
        "symmetry_difference_h": symmetry_ab - symmetry_ba,
        "composite_arithmetic_error_h": composite_arithmetic_error,
        "synthetic_matrix_diagonal_max_h": matrix_diagonal_max,
        "synthetic_matrix_symmetry_max_h": matrix_symmetry_max,
    }
    expected = {
        "identity_h": 0.0,
        "ct23_vs_ct1_h": 2.0,
        "one_bin_shift_h": 5.0 / 60.0,
        "known_3h_shift_h": 3.0,
        "symmetry_difference_h": 0.0,
        "composite_arithmetic_error_h": 0.0,
        "synthetic_matrix_diagonal_max_h": 0.0,
        "synthetic_matrix_symmetry_max_h": 0.0,
    }
    for name, expected_value in expected.items():
        if not np.isclose(
            checks[name],
            expected_value,
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(
                f"Synthetic circular-W1 check failed for {name}: "
                f"observed={checks[name]:.17g}, expected={expected_value:.17g}"
            )
    if not checks["fixed_center_broadening_h"] > W1_TOLERANCE_HOURS:
        raise RuntimeError("Fixed-center broadening did not produce nonzero W1")

    print(
        "Synthetic circular-W1 checks passed: "
        f"identity={identical:.12g} h; "
        f"CT23-vs-CT1={ct23_vs_ct1:.12g} h; "
        f"one-bin-shift={one_bin_shift:.12g} h; "
        f"3-hour-shift={known_3h_shift:.12g} h; "
        f"fixed-center-broadening={fixed_center_broadening:.12g} h"
    )
    return checks


def _load_classified_samples(
    input_files: list[tuple[int, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load WTA labels, elapsed source time, and existing frame weights."""

    if not input_files:
        raise ValueError("No source files were supplied")
    if any(
        current_index != previous_index + 1
        for (previous_index, _), (current_index, _) in zip(
            input_files, input_files[1:]
        )
    ):
        raise ValueError("Source-file indices are discontinuous")

    first_file_index = input_files[0][0]
    time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    invalid_rows = 0
    for file_index, path in input_files:
        n_rows, row_positions, labels = frozen_phase.read_and_classify(path)
        invalid_rows += n_rows - len(labels)
        if n_rows == 0:
            raise ValueError(f"Empty source file: {path}")
        if len(labels) == 0:
            continue
        file_start_hours = (
            file_index - first_file_index
        ) * frozen_phase.FILE_DURATION_MINUTES / 60.0
        elapsed_hours = file_start_hours + (
            row_positions.astype(float)
            / n_rows
            * frozen_phase.FILE_DURATION_MINUTES
            / 60.0
        )
        row_duration_seconds = (
            frozen_phase.FILE_DURATION_MINUTES * 60.0 / n_rows
        )
        time_parts.append(elapsed_hours)
        label_parts.append(labels.astype(np.int8, copy=False))
        weight_parts.append(
            np.full(len(labels), row_duration_seconds, dtype=float)
        )

    if not label_parts:
        raise ValueError("No classifiable source rows were found")
    return (
        np.concatenate(time_parts),
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
        invalid_rows,
    )


def assign_ct_bins(
    cycle_times: np.ndarray,
    *,
    frp_hours: float,
    start_ct: float = RECORDING_START_CT,
) -> np.ndarray:
    """Assign CT bins within a frozen CT0-to-CT24 cycle.

    The frozen cycle boundaries are defined by the established mapping
    ``start_CT / 24 + t / T``.  Once a frame is selected inside one of those
    CT0-to-CT24 intervals, subtracting that frozen CT0 boundary and mapping
    ``24 * cycle_time / T`` is mathematically identical and preserves the
    exact floating-point behavior used by the prior phase audit.
    """

    if not np.isclose(
        start_ct,
        RECORDING_START_CT,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Unexpected CT anchor: {start_ct}")
    phase_ct_hours = np.mod(CT_HOURS * cycle_times / frp_hours, CT_HOURS)
    safe_phase = np.minimum(
        phase_ct_hours,
        np.nextafter(CT_HOURS, 0.0),
    )
    ct_bins = np.floor(safe_phase / CT_BIN_WIDTH_HOURS).astype(np.int64)
    return ct_bins


def _validate_probability_vector(
    probability: np.ndarray,
    description: str,
) -> None:
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite probability in {description}")
    if np.any(probability < -PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Negative probability in {description}")
    if not np.isclose(
        float(probability.sum()),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError(
            f"Probability did not sum to one in {description}: "
            f"{probability.sum():.17g}"
        )


def _build_animal_phase_data(
    genotype: str,
    animal: str,
    cohort_root: Path,
) -> dict[str, object]:
    """Build pooled four-cycle phase distributions for one animal."""

    animal_dir = cohort_root / genotype / animal
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Missing animal directory: {animal_dir}")

    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(
            f"{animal} has missing source-file indices: {missing_indices}"
        )
    (
        reported_frp_hours,
        computational_frp_hours,
        start_ct,
        complete_cycles,
        frp_phase_output_dir,
    ) = frozen_phase.load_frp_phase_solution(animal_dir)
    if len(complete_cycles) != N_CYCLES:
        raise ValueError(
            f"Expected {N_CYCLES} frozen complete cycles for {animal}, "
            f"found {len(complete_cycles)}"
        )
    if not np.isclose(
        start_ct,
        RECORDING_START_CT,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Frozen CT anchor for {animal} is {start_ct}, "
            f"expected {RECORDING_START_CT}"
        )

    elapsed_hours, labels, weights_seconds, invalid_rows = (
        _load_classified_samples(input_files)
    )
    pooled_counts = np.zeros(
        (N_CT_BINS, len(frozen_phase.BEHAVIORS)),
        dtype=float,
    )
    cycle_frame_counts: list[int] = []
    cycle_indices: list[int] = []
    cycle_tolerance = 8.0 * np.finfo(float).eps * max(
        1.0,
        computational_frp_hours,
    )

    for cycle_index, start_boundary, end_boundary in complete_cycles:
        selected = (elapsed_hours >= start_boundary) & (
            elapsed_hours < end_boundary
        )
        if not selected.any():
            raise ValueError(
                f"Frozen full cycle {cycle_index} contains no classifiable rows "
                f"for {animal}"
            )
        cycle_times = elapsed_hours[selected] - start_boundary
        cycle_position = start_ct / CT_HOURS + elapsed_hours[selected] / computational_frp_hours
        ct_bins = assign_ct_bins(
            cycle_times,
            frp_hours=computational_frp_hours,
            start_ct=start_ct,
        )
        if np.any(cycle_position < cycle_index - cycle_tolerance) or np.any(
            cycle_position > cycle_index + 1.0 + cycle_tolerance
        ):
            raise RuntimeError(
                f"Phase mapping assigned rows outside frozen cycle {cycle_index} "
                f"for {animal}"
            )
        np.add.at(
            pooled_counts,
            (ct_bins, labels[selected]),
            weights_seconds[selected],
        )
        cycle_indices.append(cycle_index)
        cycle_frame_counts.append(int(selected.sum()))

    nonrest_counts = np.take(
        pooled_counts,
        frozen_phase.NONRESTING_BEHAVIOR_INDICES,
        axis=1,
    )
    behavior_occupancy_seconds = nonrest_counts.sum(axis=0)
    if np.any(behavior_occupancy_seconds <= 0.0):
        zero_behaviors = [
            behavior
            for behavior, occupancy in zip(
                NONREST_BEHAVIORS,
                behavior_occupancy_seconds,
            )
            if occupancy <= 0.0
        ]
        raise ValueError(
            f"Zero four-cycle occupancy for {animal}: {zero_behaviors}"
        )
    probabilities = nonrest_counts / behavior_occupancy_seconds[None, :]
    for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
        _validate_probability_vector(
            probabilities[:, behavior_position],
            f"{animal} {behavior}",
        )

    return {
        "genotype": genotype,
        "animal": animal,
        "reported_frp_hours": float(reported_frp_hours),
        "computational_frp_hours": float(computational_frp_hours),
        "start_ct": float(start_ct),
        "complete_cycles": complete_cycles,
        "cycle_indices": cycle_indices,
        "cycle_frame_counts": cycle_frame_counts,
        "invalid_rows": int(invalid_rows),
        "frp_phase_output_dir": frp_phase_output_dir,
        "occupancy_minutes": behavior_occupancy_seconds / 60.0,
        "probabilities": probabilities,
    }


def _distribution_rows(animal_data: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    starts = np.arange(N_CT_BINS, dtype=float) * CT_BIN_WIDTH_HOURS
    centers = starts + CT_BIN_WIDTH_HOURS / 2.0
    for record in animal_data:
        probabilities = np.asarray(record["probabilities"], dtype=float)
        occupancy_minutes = np.asarray(record["occupancy_minutes"], dtype=float)
        for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
            for bin_index in range(N_CT_BINS):
                rows.append(
                    {
                        "animal": record["animal"],
                        "genotype": record["genotype"],
                        "behavior": behavior,
                        "ct_bin_start_hours": float(starts[bin_index]),
                        "ct_bin_center_hours": float(centers[bin_index]),
                        "occupancy": float(
                            probabilities[bin_index, behavior_position]
                            * occupancy_minutes[behavior_position]
                        ),
                        "probability": float(
                            probabilities[bin_index, behavior_position]
                        ),
                    }
                )
    return rows


def _pairwise_outputs(
    animal_data: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    pairwise_rows: list[dict[str, object]] = []
    composite_rows: list[dict[str, object]] = []
    behavior_matrices = {
        behavior: np.zeros((len(ANIMALS), len(ANIMALS)), dtype=float)
        for behavior in NONREST_BEHAVIORS
    }
    composite_matrix = np.zeros((len(ANIMALS), len(ANIMALS)), dtype=float)

    for i in range(len(animal_data)):
        first = animal_data[i]
        first_probabilities = np.asarray(first["probabilities"], dtype=float)
        for j in range(i + 1, len(animal_data)):
            second = animal_data[j]
            second_probabilities = np.asarray(
                second["probabilities"],
                dtype=float,
            )
            components: dict[str, float] = {}
            for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
                distance = circular_w1(
                    first_probabilities[:, behavior_position],
                    second_probabilities[:, behavior_position],
                )
                components[behavior] = distance
                pairwise_rows.append(
                    {
                        "animal_i": first["animal"],
                        "genotype_i": first["genotype"],
                        "animal_j": second["animal"],
                        "genotype_j": second["genotype"],
                        "behavior": behavior,
                        "w1_hours": distance,
                    }
                )
                behavior_matrices[behavior][i, j] = distance
                behavior_matrices[behavior][j, i] = distance

            composite = float(np.mean(list(components.values())))
            composite_matrix[i, j] = composite
            composite_matrix[j, i] = composite
            composite_row = {
                "animal_i": first["animal"],
                "genotype_i": first["genotype"],
                "animal_j": second["animal"],
                "genotype_j": second["genotype"],
            }
            composite_row.update(
                {f"w1_{behavior}": value for behavior, value in components.items()}
            )
            composite_row["repertoire_w1_mean_hours"] = composite
            composite_rows.append(composite_row)

    pairwise_frame = pd.DataFrame(pairwise_rows, columns=PAIRWISE_COLUMNS)
    composite_frame = pd.DataFrame(composite_rows, columns=COMPOSITE_COLUMNS)
    return pairwise_frame, composite_frame, behavior_matrices, composite_matrix


def _validate_real_outputs(
    animal_data: list[dict[str, object]],
    distribution_frame: pd.DataFrame,
    pairwise_frame: pd.DataFrame,
    composite_frame: pd.DataFrame,
    behavior_matrices: dict[str, np.ndarray],
    composite_matrix: np.ndarray,
) -> None:
    if len(animal_data) != 8 or tuple(record["animal"] for record in animal_data) != ANIMAL_ORDER:
        raise RuntimeError("Real-data output does not contain the fixed eight animals")
    if len(NONREST_BEHAVIORS) != 8:
        raise RuntimeError("The non-rest behavior set is not exactly eight states")
    if len(distribution_frame) != 8 * 8 * N_CT_BINS:
        raise RuntimeError(
            f"Unexpected distribution row count: {len(distribution_frame)}"
        )
    grouped = distribution_frame.groupby(
        ["animal", "behavior"],
        sort=False,
    )
    if len(grouped) != 64 or any(len(group) != N_CT_BINS for _, group in grouped):
        raise RuntimeError("Distribution output does not have 288 bins per cell")
    for (animal, behavior), group in grouped:
        probability = group["probability"].to_numpy(dtype=float)
        _validate_probability_vector(probability, f"CSV {animal} {behavior}")
        occupancy = group["occupancy"].to_numpy(dtype=float)
        if not np.isfinite(occupancy).all() or np.any(occupancy < 0.0):
            raise RuntimeError(f"Invalid occupancy in CSV {animal} {behavior}")

    if len(pairwise_frame) != 28 * 8:
        raise RuntimeError(
            f"Unexpected behavior-specific pair row count: {len(pairwise_frame)}"
        )
    expected_pairs = {
        (ANIMAL_ORDER[i], ANIMAL_ORDER[j], behavior)
        for i in range(len(ANIMAL_ORDER))
        for j in range(i + 1, len(ANIMAL_ORDER))
        for behavior in NONREST_BEHAVIORS
    }
    actual_pairs = set(
        map(
            tuple,
            pairwise_frame[["animal_i", "animal_j", "behavior"]]
            .itertuples(index=False, name=None),
        )
    )
    if actual_pairs != expected_pairs:
        raise RuntimeError("Behavior-specific pair output has missing or extra keys")
    if not np.isfinite(pairwise_frame["w1_hours"].to_numpy(dtype=float)).all():
        raise RuntimeError("Behavior-specific W1 output contains non-finite values")

    if len(composite_frame) != 28:
        raise RuntimeError(
            f"Unexpected composite pair row count: {len(composite_frame)}"
        )
    composite_pair_keys = {
        (row.animal_i, row.animal_j)
        for row in composite_frame.itertuples(index=False)
    }
    expected_composite_keys = {
        (ANIMAL_ORDER[i], ANIMAL_ORDER[j])
        for i in range(len(ANIMAL_ORDER))
        for j in range(i + 1, len(ANIMAL_ORDER))
    }
    if composite_pair_keys != expected_composite_keys:
        raise RuntimeError("Composite output has missing or extra animal pairs")
    component_columns = [f"w1_{behavior}" for behavior in NONREST_BEHAVIORS]
    for row in composite_frame.itertuples(index=False):
        components = np.asarray(
            [getattr(row, column) for column in component_columns],
            dtype=float,
        )
        composite = float(row.repertoire_w1_mean_hours)
        if not np.isfinite(components).all() or not np.isfinite(composite):
            raise RuntimeError("Composite output contains non-finite W1 values")
        if not np.isclose(
            composite,
            float(np.mean(components)),
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(
                f"Composite arithmetic check failed for "
                f"{row.animal_i}-{row.animal_j}"
            )

    for behavior, matrix in behavior_matrices.items():
        if not np.allclose(
            matrix,
            matrix.T,
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(f"Behavior-specific matrix is not symmetric: {behavior}")
        if not np.allclose(
            np.diag(matrix),
            0.0,
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(f"Behavior-specific matrix diagonal is nonzero: {behavior}")
    if not np.allclose(
        composite_matrix,
        composite_matrix.T,
        rtol=0.0,
        atol=W1_TOLERANCE_HOURS,
    ):
        raise RuntimeError("Composite distance matrix is not symmetric")
    if not np.allclose(
        np.diag(composite_matrix),
        0.0,
        rtol=0.0,
        atol=W1_TOLERANCE_HOURS,
    ):
        raise RuntimeError("Composite distance matrix diagonal is nonzero")


def _decomposition_frame(
    pairwise_frame: pd.DataFrame,
) -> pd.DataFrame:
    total_pairwise_sum = float(pairwise_frame["w1_hours"].sum())
    rows: list[dict[str, object]] = []
    for behavior in NONREST_BEHAVIORS:
        values = pairwise_frame.loc[
            pairwise_frame["behavior"] == behavior,
            "w1_hours",
        ].to_numpy(dtype=float)
        first_quartile, third_quartile = np.percentile(values, [25.0, 75.0])
        rows.append(
            {
                "behavior": behavior,
                "n_pairs": int(len(values)),
                "mean_w1_hours": float(np.mean(values)),
                "median_w1_hours": float(np.median(values)),
                "sd_w1_hours": float(np.std(values, ddof=1)),
                "min_w1_hours": float(np.min(values)),
                "max_w1_hours": float(np.max(values)),
                "iqr_w1_hours": float(third_quartile - first_quartile),
                "fraction_of_composite_summed_distance": float(
                    values.sum() / total_pairwise_sum
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECOMPOSITION_COLUMNS)


def _save_figures(
    output_dir: Path,
    animal_data: list[dict[str, object]],
    pairwise_frame: pd.DataFrame,
    composite_matrix: np.ndarray,
) -> list[Path]:
    figure_paths: list[Path] = []
    pair_labels = [
        f"{row.animal_i}-{row.animal_j}"
        for row in pairwise_frame.drop_duplicates(
            ["animal_i", "animal_j"]
        ).itertuples(index=False)
    ]
    pair_matrix = pairwise_frame.pivot(
        index=["animal_i", "animal_j"],
        columns="behavior",
        values="w1_hours",
    )
    pair_matrix = pair_matrix.loc[
        [
            (ANIMAL_ORDER[i], ANIMAL_ORDER[j])
            for i in range(len(ANIMAL_ORDER))
            for j in range(i + 1, len(ANIMAL_ORDER))
        ],
        list(NONREST_BEHAVIORS),
    ]
    pair_values = pair_matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(12.5, 9.0), constrained_layout=True)
    image = axis.imshow(pair_values, aspect="auto", cmap="viridis")
    axis.set_title("Behavior-specific circular W1 by animal pair")
    axis.set_xlabel("Behavior")
    axis.set_ylabel("Unordered animal pair")
    axis.set_xticks(np.arange(len(NONREST_BEHAVIORS)))
    axis.set_xticklabels(NONREST_BEHAVIORS, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(pair_labels)))
    axis.set_yticklabels(pair_labels)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Circular W1 (hours)")
    pair_path = output_dir / "behavior_w1_pairwise_heatmap.png"
    figure.savefig(pair_path, dpi=180)
    plt.close(figure)
    figure_paths.append(pair_path)

    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    image = axis.imshow(composite_matrix, cmap="viridis", vmin=0.0)
    axis.set_title("Composite repertoire circular W1")
    axis.set_xlabel("Animal")
    axis.set_ylabel("Animal")
    axis.set_xticks(np.arange(len(ANIMAL_ORDER)))
    axis.set_xticklabels(ANIMAL_ORDER, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(ANIMAL_ORDER)))
    axis.set_yticklabels(ANIMAL_ORDER)
    for row in range(len(ANIMAL_ORDER)):
        for column in range(len(ANIMAL_ORDER)):
            axis.text(
                column,
                row,
                f"{composite_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if composite_matrix[row, column] > composite_matrix.max() * 0.55 else "black",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean circular W1 (hours)")
    matrix_path = output_dir / "repertoire_distance_matrix_heatmap.png"
    figure.savefig(matrix_path, dpi=180)
    plt.close(figure)
    figure_paths.append(matrix_path)

    starts = np.arange(N_CT_BINS, dtype=float) * CT_BIN_WIDTH_HOURS
    centers = starts + CT_BIN_WIDTH_HOURS / 2.0
    for record in animal_data:
        probabilities = np.asarray(record["probabilities"], dtype=float)
        figure, axes = plt.subplots(
            2,
            4,
            figsize=(16.0, 7.0),
            sharex=True,
            constrained_layout=True,
        )
        for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
            axis = axes.flat[behavior_position]
            values = probabilities[:, behavior_position]
            axis.plot(
                np.r_[centers, CT_HOURS],
                np.r_[values, values[0]],
                color="#1f77b4",
                linewidth=0.9,
            )
            axis.set_title(behavior)
            axis.set_xlim(0.0, CT_HOURS)
            axis.set_ylim(bottom=0.0)
            axis.set_xticks([0.0, 6.0, 12.0, 18.0, 24.0])
            axis.grid(axis="y", alpha=0.25)
        axes[0, 0].set_ylabel("Probability")
        axes[1, 0].set_ylabel("Probability")
        for axis in axes[1, :]:
            axis.set_xlabel("CT hours")
        figure.suptitle(
            f"{record['animal']}: normalized 5-minute WTA phase distributions",
            y=1.02,
        )
        path = output_dir / f"phase_distributions_5min_{record['animal']}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(path)

    return figure_paths


def _write_matrix_long(
    output_path: Path,
    behavior_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for behavior in NONREST_BEHAVIORS:
        matrix = behavior_matrices[behavior]
        for i, animal_i in enumerate(ANIMAL_ORDER):
            for j, animal_j in enumerate(ANIMAL_ORDER):
                rows.append(
                    {
                        "animal_i": animal_i,
                        "genotype_i": GENOTYPE_BY_ANIMAL[animal_i],
                        "animal_j": animal_j,
                        "genotype_j": GENOTYPE_BY_ANIMAL[animal_j],
                        "behavior": behavior,
                        "w1_hours": float(matrix[i, j]),
                    }
                )
    frame = pd.DataFrame(rows, columns=MATRIX_LONG_COLUMNS)
    frame.to_csv(output_path, index=False, float_format="%.17g")
    return frame


def _write_validation_summary(
    output_path: Path,
    synthetic_checks: dict[str, float],
    animal_data: list[dict[str, object]],
    distribution_frame: pd.DataFrame,
    pairwise_frame: pd.DataFrame,
    composite_frame: pd.DataFrame,
    decomposition_frame: pd.DataFrame,
    composite_matrix: np.ndarray,
) -> None:
    component_columns = [f"w1_{behavior}" for behavior in NONREST_BEHAVIORS]
    composite_values = composite_frame["repertoire_w1_mean_hours"].to_numpy(dtype=float)
    pairwise_values = pairwise_frame["w1_hours"].to_numpy(dtype=float)
    pairwise_sums = composite_frame[component_columns].sum(axis=1).to_numpy(dtype=float)
    dominance_rows: list[tuple[float, str, str]] = []
    for row, pair_sum in zip(composite_frame.itertuples(index=False), pairwise_sums):
        values = np.asarray([getattr(row, column) for column in component_columns])
        top_index = int(np.argmax(values))
        dominance_rows.append(
            (
                float(values[top_index] / pair_sum),
                f"{row.animal_i}-{row.animal_j}",
                NONREST_BEHAVIORS[top_index],
            )
        )
    top_fraction, top_pair, top_behavior = max(dominance_rows)

    lines = [
        "Equal-weight circular-W1 non-rest behavioral repertoire validation",
        "",
        "Frozen analysis inputs:",
        "  representation: WTA",
        "  behaviors: " + ", ".join(NONREST_BEHAVIORS),
        f"  CT bins: {N_CT_BINS} bins at {CT_BIN_WIDTH_HOURS:.17g} hours",
        f"  external recording anchor: CT{RECORDING_START_CT:g}",
        "  cycles: four frozen complete CT0-to-CT24 cycles per animal",
        "  genotype statistics: not performed",
        "",
        "Synthetic circular-W1 checks:",
        f"  identity: {synthetic_checks['identity_h']:.17g} h",
        f"  CT23 versus CT1: {synthetic_checks['ct23_vs_ct1_h']:.17g} h",
        f"  one 5-minute bin shift: {synthetic_checks['one_bin_shift_h']:.17g} h",
        f"  known 3-hour shift: {synthetic_checks['known_3h_shift_h']:.17g} h",
        f"  fixed-center broadening: {synthetic_checks['fixed_center_broadening_h']:.17g} h",
        f"  symmetry difference: {synthetic_checks['symmetry_difference_h']:.3e} h",
        f"  composite arithmetic error: {synthetic_checks['composite_arithmetic_error_h']:.3e} h",
        f"  synthetic matrix diagonal maximum: {synthetic_checks['synthetic_matrix_diagonal_max_h']:.3e} h",
        f"  synthetic matrix symmetry maximum: {synthetic_checks['synthetic_matrix_symmetry_max_h']:.3e} h",
        "  result: all required synthetic checks passed",
        "",
        "Real-data coverage checks:",
        f"  animals: {len(animal_data)}",
        f"  animal-by-behavior distributions: {len(animal_data) * len(NONREST_BEHAVIORS)}",
        f"  distribution rows: {len(distribution_frame)}",
        f"  behavior-specific pairwise rows: {len(pairwise_frame)}",
        f"  composite animal pairs: {len(composite_frame)}",
        f"  all behavior-specific values finite: {np.isfinite(pairwise_values).all()}",
        f"  all composite values finite: {np.isfinite(composite_values).all()}",
        "  probability, coverage, symmetry, diagonal, and composite-mean checks: passed",
        "",
        "Frozen per-animal inputs used:",
    ]
    for record in animal_data:
        cycle_text = ",".join(str(index) for index in record["cycle_indices"])
        lines.append(
            f"  {record['animal']}: reported_selected_FRP={record['reported_frp_hours']:.17g} h; "
            f"computational_FRP_from_boundaries={record['computational_frp_hours']:.17g} h; "
            f"start_CT={record['start_ct']:.17g}; cycles=[{cycle_text}]; "
            f"invalid_rows={record['invalid_rows']}"
        )

    lines.extend(
        [
            "",
            "Descriptive W1 results across 28 unordered animal pairs:",
            f"  behavior-specific range: {pairwise_values.min():.6g} to {pairwise_values.max():.6g} h",
            f"  composite repertoire range: {composite_values.min():.6g} to {composite_values.max():.6g} h",
            f"  behavior with largest mean W1: {decomposition_frame.loc[decomposition_frame['mean_w1_hours'].idxmax(), 'behavior']}",
            f"  behavior with smallest mean W1: {decomposition_frame.loc[decomposition_frame['mean_w1_hours'].idxmin(), 'behavior']}",
            f"  most concentrated single-behavior pair contribution: {top_fraction:.6g} "
            f"for {top_pair} ({top_behavior})",
            "  grooming was retained on the same 1/8 scale; no abundance weighting was applied",
            "  sparse-state distributions (including rearing and locomotion) were finite, positive, and passed all W1 checks",
            "  optional same-distribution null sampling: skipped",
            "",
            "No behavior was excluded or reweighted after inspecting these outputs.",
            "Outputs are ready for the next conceptual step: deciding how to summarize or statistically evaluate genotype structure.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(cohort_root: Path) -> dict[str, object]:
    synthetic_checks = run_synthetic_validation()
    animal_data = [
        _build_animal_phase_data(genotype, animal, cohort_root)
        for genotype, animal in ANIMALS
    ]
    distribution_frame = pd.DataFrame(
        _distribution_rows(animal_data),
        columns=DISTRIBUTION_COLUMNS,
    )
    (
        pairwise_frame,
        composite_frame,
        behavior_matrices,
        composite_matrix,
    ) = _pairwise_outputs(animal_data)
    _validate_real_outputs(
        animal_data,
        distribution_frame,
        pairwise_frame,
        composite_frame,
        behavior_matrices,
        composite_matrix,
    )
    decomposition_frame = _decomposition_frame(pairwise_frame)

    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = output_dir / "behavior_phase_distributions_5min.csv"
    pairwise_path = output_dir / "pairwise_behavior_circular_w1.csv"
    composite_path = output_dir / "pairwise_repertoire_distance.csv"
    matrix_path = output_dir / "repertoire_distance_matrix.csv"
    matrix_long_path = output_dir / "behavior_w1_matrices_long.csv"
    decomposition_path = output_dir / "behavior_w1_contribution_summary.csv"
    validation_path = output_dir / "validation_summary.txt"

    distribution_frame.to_csv(
        distribution_path,
        index=False,
        float_format="%.17g",
    )
    pairwise_frame.to_csv(
        pairwise_path,
        index=False,
        float_format="%.17g",
    )
    composite_frame.to_csv(
        composite_path,
        index=False,
        float_format="%.17g",
    )
    matrix_frame = pd.DataFrame(
        composite_matrix,
        index=ANIMAL_ORDER,
        columns=ANIMAL_ORDER,
    )
    matrix_frame.index.name = "animal"
    matrix_frame.to_csv(matrix_path, float_format="%.17g")
    matrix_long_frame = _write_matrix_long(matrix_long_path, behavior_matrices)
    decomposition_frame.to_csv(
        decomposition_path,
        index=False,
        float_format="%.17g",
    )
    figure_paths = _save_figures(
        output_dir,
        animal_data,
        pairwise_frame,
        composite_matrix,
    )
    _write_validation_summary(
        validation_path,
        synthetic_checks,
        animal_data,
        distribution_frame,
        pairwise_frame,
        composite_frame,
        decomposition_frame,
        composite_matrix,
    )

    # Re-read the primary CSV to confirm the serialized probability vectors
    # retain the same coverage and normalization contract.
    serialized_distribution = pd.read_csv(distribution_path)
    if len(serialized_distribution) != len(distribution_frame):
        raise RuntimeError("Serialized phase-distribution row count changed")
    for key, group in serialized_distribution.groupby(
        ["animal", "behavior"],
        sort=False,
    ):
        _validate_probability_vector(
            group["probability"].to_numpy(dtype=float),
            f"serialized {key[0]} {key[1]}",
        )

    print(f"Output directory: {output_dir}")
    print(f"Phase-distribution rows: {len(distribution_frame)}")
    print(f"Behavior-specific pairwise rows: {len(pairwise_frame)}")
    print(f"Composite pairwise rows: {len(composite_frame)}")
    print(
        "Behavior-specific W1 range: "
        f"{pairwise_frame['w1_hours'].min():.6g} to "
        f"{pairwise_frame['w1_hours'].max():.6g} h"
    )
    print(
        "Composite repertoire range: "
        f"{composite_frame['repertoire_w1_mean_hours'].min():.6g} to "
        f"{composite_frame['repertoire_w1_mean_hours'].max():.6g} h"
    )
    print("Real-data coverage and symmetry checks passed.")
    for path in [
        distribution_path,
        pairwise_path,
        composite_path,
        matrix_path,
        matrix_long_path,
        decomposition_path,
        validation_path,
        *figure_paths,
    ]:
        print(path)

    return {
        "output_dir": output_dir,
        "synthetic_checks": synthetic_checks,
        "animal_data": animal_data,
        "distribution_frame": distribution_frame,
        "pairwise_frame": pairwise_frame,
        "composite_frame": composite_frame,
        "matrix_frame": matrix_frame,
        "matrix_long_frame": matrix_long_frame,
        "decomposition_frame": decomposition_frame,
        "composite_matrix": composite_matrix,
        "figure_paths": figure_paths,
    }


def main() -> None:
    cohort_root = parse_args()
    run_analysis(cohort_root)


if __name__ == "__main__":
    main()
