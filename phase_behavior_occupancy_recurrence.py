"""Single-animal integration of the validated occupancy recurrence scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase
import phase_behavior_recurrence as recurrence
from phase_behavior_occupancy_recurrence_validation import score_pair


RESOLUTIONS = ((5, 288), (10, 144))
OUTPUT_DIRNAME = "Occupancy_Recurrence_Output"
NUMERICAL_TOLERANCE = max(recurrence.NORMALIZATION_TOLERANCE, 1e-12)

PAIRWISE_COLUMNS = [
    "Animal",
    "resolution_minutes",
    "cycle_A",
    "cycle_B",
    "A_obs",
    "mu_shift",
    "R_pair",
    "n_common_bins",
    "fraction_common_bins",
    "total_nonrest_A",
    "total_nonrest_B",
    "denominator_1_minus_mu_shift",
    "status",
    "diagnostic",
]

SUMMARY_COLUMNS = [
    "Animal",
    "resolution_minutes",
    "n_complete_cycles",
    "n_adjacent_pairs",
    "n_valid_pairs",
    "mean_A_obs",
    "mean_mu_shift",
    "mean_R_pair",
    "min_fraction_common_bins",
    "status",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Run single-animal phase-resolved occupancy recurrence."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing one animal's sequential CBAS output CSV files.",
    )
    return parser.parse_args().input_dir.expanduser().resolve()


def _load_classified_samples(
    input_files: list[tuple[int, Path]],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read valid WTA samples and preserve elapsed source-file time."""

    first_file_index = input_files[0][0]
    relative_time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    row_counts: list[int] = []
    invalid_rows = 0

    for file_index, path in input_files:
        n_rows, row_indices, labels = recurrence.read_and_classify(path)
        row_counts.append(n_rows)
        invalid_rows += n_rows - len(labels)
        if n_rows == 0 or len(labels) == 0:
            continue
        file_start_hours = (
            file_index - first_file_index
        ) * recurrence.FILE_DURATION_MINUTES / 60.0
        elapsed_hours = file_start_hours + (
            row_indices.astype(float)
            / n_rows
            * recurrence.FILE_DURATION_MINUTES
            / 60.0
        )
        relative_time_parts.append(elapsed_hours)
        label_parts.append(labels)

    if not label_parts:
        raise ValueError("No valid behavioral samples were found in the input CSV files")

    nonzero_row_counts = np.asarray(
        [count for count in row_counts if count > 0],
        dtype=float,
    )
    if nonzero_row_counts.size == 0:
        raise ValueError("Could not determine the source sampling rate")
    expected_samples_per_hour = float(
        np.median(nonzero_row_counts)
        / (recurrence.FILE_DURATION_MINUTES / 60.0)
    )
    return (
        np.concatenate(relative_time_parts),
        np.concatenate(label_parts).astype(np.int8, copy=False),
        expected_samples_per_hour,
        invalid_rows,
    )


def _build_complete_cycles(
    relative_times: np.ndarray,
    labels: np.ndarray,
    complete_cycle_specs: list[tuple[int, float, float]],
    frp_hours: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Select samples inside the frozen FRP/CT full-cycle boundaries."""

    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]] = []
    cycle_time_tolerance = 8.0 * np.finfo(float).eps * max(1.0, frp_hours)
    for cycle_index, start_boundary, end_boundary in complete_cycle_specs:
        selected = (relative_times >= start_boundary) & (
            relative_times < end_boundary
        )
        if not selected.any():
            raise ValueError(
                f"Full FRP cycle {cycle_index} contains no classifiable samples"
            )
        cycle_times = relative_times[selected] - start_boundary
        cycle_labels = labels[selected]
        if np.any(cycle_times < -cycle_time_tolerance) or np.any(
            cycle_times > frp_hours + cycle_time_tolerance
        ):
            raise ValueError(
                f"Samples assigned to full cycle {cycle_index} fall outside its FRP interval"
            )
        cycle_times = np.clip(cycle_times, 0.0, np.nextafter(frp_hours, 0.0))
        complete_cycles.append((cycle_index, cycle_times, cycle_labels))
    return complete_cycles


def _occupancy_for_resolution(
    relative_times: np.ndarray,
    labels: np.ndarray,
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    first_cycle_start: float,
    expected_samples_per_hour: float,
    frp_hours: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build occupancy tensors and masks using the existing bin/coverage rules."""

    shifted_times = relative_times - first_cycle_start
    representations, _, valid_counts = recurrence.build_binned_cycles(
        shifted_times,
        labels,
        len(complete_cycles),
        n_bins,
        expected_samples_per_hour,
        frp_hours,
    )
    expected_per_bin = expected_samples_per_hour * frp_hours / n_bins
    if expected_per_bin <= 0.0:
        raise ValueError("Expected samples per bin must be positive")

    # Use the same normalized phase-bin expression as the existing recurrence
    # builder.  Recomputing counts is necessary because that builder exposes
    # valid counts but not the full behavior-by-bin count tensor.
    cycle_indices = np.floor(shifted_times / frp_hours).astype(np.int64)
    phase_fraction = np.mod(shifted_times / frp_hours, 1.0)
    normalized_phase_hours = 24.0 * phase_fraction
    phase_bins = np.floor(
        normalized_phase_hours / 24.0 * n_bins
    ).astype(np.int64)
    in_complete_cycles = (
        (cycle_indices >= 0)
        & (cycle_indices < len(complete_cycles))
        & (phase_bins >= 0)
        & (phase_bins < n_bins)
    )
    counts = np.zeros(
        (len(complete_cycles), n_bins, len(recurrence.BEHAVIORS)),
        dtype=np.int64,
    )
    np.add.at(
        counts,
        (
            cycle_indices[in_complete_cycles],
            phase_bins[in_complete_cycles],
            labels[in_complete_cycles],
        ),
        1,
    )
    if not np.array_equal(counts.sum(axis=2), valid_counts):
        raise ValueError(
            "Existing cycle indexing and CT binning disagreed while building occupancy"
        )
    if representations["full_9state"].shape != (len(complete_cycles), n_bins):
        raise ValueError("Existing cycle representation has an unexpected shape")

    coverage = valid_counts.astype(float) / expected_per_bin
    valid_masks = coverage >= recurrence.MIN_VALID_COVERAGE
    nonrest_counts = np.take(
        counts,
        recurrence.NONRESTING_BEHAVIOR_INDICES,
        axis=2,
    )
    occupancy = np.zeros_like(nonrest_counts, dtype=float)
    np.divide(
        nonrest_counts,
        valid_counts[:, :, None],
        out=occupancy,
        where=valid_counts[:, :, None] > 0,
    )

    if np.any(occupancy < -NUMERICAL_TOLERANCE) or np.any(
        occupancy > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError("Non-rest occupancy exceeded [0, 1]")
    occupancy_sums = occupancy.sum(axis=2)
    if np.any(occupancy_sums > 1.0 + NUMERICAL_TOLERANCE):
        raise ValueError("A bin's non-rest occupancy exceeded one")
    if np.any(valid_masks & (valid_counts == 0)):
        raise ValueError("A usable bin has no valid classified samples")
    # Unusable bins contain only finite placeholders; their masks exclude them
    # from both observed and shifted scoring. Missing/invalid samples never enter
    # counts and are not converted to resting behavior.
    return occupancy, valid_masks, valid_counts


def _score_resolution(
    animal_id: str,
    resolution_minutes: int,
    occupancy: np.ndarray,
    valid_masks: np.ndarray,
    cycle_indices: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    pair_rows: list[dict[str, object]] = []
    pair_results: list[dict[str, object]] = []
    for pair_index in range(len(cycle_indices) - 1):
        result = score_pair(
            occupancy[pair_index],
            occupancy[pair_index + 1],
            valid_masks[pair_index],
            valid_masks[pair_index + 1],
        )
        if result["status"] != "PASS":
            raise RuntimeError(
                f"Undefined adjacent pair at {resolution_minutes}-minute resolution "
                f"({cycle_indices[pair_index]} vs {cycle_indices[pair_index + 1]}): "
                f"{result['status']}: {result['diagnostic']}"
            )
        for field in (
            "A_obs",
            "mu_shift",
            "R_pair",
            "denominator_1_minus_mu_shift",
        ):
            if not np.isfinite(float(result[field])):
                raise ValueError(
                    f"{field} was non-finite for a PASS pair at "
                    f"{resolution_minutes}-minute resolution"
                )
        if abs(float(result["denominator_1_minus_mu_shift"])) <= recurrence.NORMALIZATION_TOLERANCE:
            raise ValueError(
                f"Unstable shift denominator at {resolution_minutes}-minute resolution"
            )
        pair_results.append(result)
        pair_rows.append(
            {
                "Animal": animal_id,
                "resolution_minutes": resolution_minutes,
                "cycle_A": cycle_indices[pair_index],
                "cycle_B": cycle_indices[pair_index + 1],
                "A_obs": result["A_obs"],
                "mu_shift": result["mu_shift"],
                "R_pair": result["R_pair"],
                "n_common_bins": result["n_common_bins"],
                "fraction_common_bins": result["fraction_common_bins"],
                "total_nonrest_A": result["total_nonrest_A"],
                "total_nonrest_B": result["total_nonrest_B"],
                "denominator_1_minus_mu_shift": result[
                    "denominator_1_minus_mu_shift"
                ],
                "status": result["status"],
                "diagnostic": result["diagnostic"],
            }
        )

    summary = {
        "Animal": animal_id,
        "resolution_minutes": resolution_minutes,
        "n_complete_cycles": len(cycle_indices),
        "n_adjacent_pairs": len(cycle_indices) - 1,
        "n_valid_pairs": len(pair_results),
        "mean_A_obs": float(np.mean([result["A_obs"] for result in pair_results])),
        "mean_mu_shift": float(
            np.mean([result["mu_shift"] for result in pair_results])
        ),
        "mean_R_pair": float(np.mean([result["R_pair"] for result in pair_results])),
        "min_fraction_common_bins": float(
            min(result["fraction_common_bins"] for result in pair_results)
        ),
        "status": "PASS",
    }
    return pair_rows, summary


def run_analysis(input_dir: Path) -> dict[str, object]:
    if tuple(frozen_phase.BEHAVIORS) != tuple(recurrence.BEHAVIORS):
        raise RuntimeError("Existing WTA behavior definitions disagree")

    input_files, missing_indices = frozen_phase.discover_input_files(input_dir)
    (
        reported_frp_hours,
        computational_frp_hours,
        start_ct,
        complete_cycle_specs,
        frp_phase_output_dir,
    ) = frozen_phase.load_frp_phase_solution(input_dir)
    if len(complete_cycle_specs) < 2:
        raise ValueError("At least two complete biological cycles are required")

    relative_times, labels, expected_samples_per_hour, invalid_rows = (
        _load_classified_samples(input_files)
    )
    complete_cycles = _build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp_hours,
    )
    cycle_indices = [cycle_index for cycle_index, _, _ in complete_cycles]
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    resolution_details: list[dict[str, object]] = []
    for resolution_minutes, n_bins in RESOLUTIONS:
        occupancy, valid_masks, valid_counts = _occupancy_for_resolution(
            relative_times,
            labels,
            complete_cycles,
            complete_cycle_specs[0][1],
            expected_samples_per_hour,
            computational_frp_hours,
            n_bins,
        )
        resolution_pair_rows, resolution_summary = _score_resolution(
            input_dir.name,
            resolution_minutes,
            occupancy,
            valid_masks,
            cycle_indices,
        )
        pair_rows.extend(resolution_pair_rows)
        summary_rows.append(resolution_summary)
        resolution_details.append(
            {
                "resolution_minutes": resolution_minutes,
                "tensor_shape": occupancy.shape,
                "usable_bins_per_cycle": valid_masks.sum(axis=1).astype(int).tolist(),
                "rest_containing_bins": int(np.sum(occupancy.sum(axis=2) < 1.0 - NUMERICAL_TOLERANCE)),
                "valid_counts": valid_counts,
            }
        )

    output_dir = input_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows, columns=PAIRWISE_COLUMNS).to_csv(
        output_dir / "occupancy_recurrence_pairwise.csv",
        index=False,
    )
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        output_dir / "occupancy_recurrence_summary.csv",
        index=False,
    )

    return {
        "input_files": input_files,
        "missing_indices": missing_indices,
        "reported_frp_hours": reported_frp_hours,
        "computational_frp_hours": computational_frp_hours,
        "start_ct": start_ct,
        "frp_phase_output_dir": frp_phase_output_dir,
        "complete_cycle_specs": complete_cycle_specs,
        "cycle_indices": cycle_indices,
        "expected_samples_per_hour": expected_samples_per_hour,
        "invalid_rows": invalid_rows,
        "resolution_details": resolution_details,
        "pair_rows": pair_rows,
        "summary_rows": summary_rows,
        "output_dir": output_dir,
    }


def main() -> None:
    input_dir = parse_args()
    result = run_analysis(input_dir)
    print(f"Animal: {input_dir.name}")
    print(
        f"Source files: {len(result['input_files'])}; "
        f"missing indices: {result['missing_indices'] or 'none'}"
    )
    print(
        f"FRP: {result['reported_frp_hours']} h reported; "
        f"{result['computational_frp_hours']} h computational"
    )
    print(f"Start CT: {result['start_ct']}")
    print(f"Complete cycle indices: {result['cycle_indices']}")
    print(f"Invalid source rows excluded: {result['invalid_rows']}")
    print(
        "FRP/phase output used: "
        f"{result['frp_phase_output_dir']}"
    )
    for detail in result["resolution_details"]:
        print(
            f"{detail['resolution_minutes']}-minute tensor shape: "
            f"{detail['tensor_shape']}; usable bins/cycle: "
            f"{detail['usable_bins_per_cycle']}; "
            f"bins with rest: {detail['rest_containing_bins']}"
        )
    pair_frame = pd.DataFrame(result["pair_rows"])
    print(pair_frame[PAIRWISE_COLUMNS].to_string(index=False))
    print(f"Outputs written to: {result['output_dir']}")


if __name__ == "__main__":
    main()
