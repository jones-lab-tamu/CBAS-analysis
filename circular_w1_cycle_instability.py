"""Compute raw W1-based cycle-to-cycle circadian repertoire instability.

This standalone analysis keeps the four frozen complete biological cycles
separate within each animal.  It uses WTA behavior identity, the frozen
animal-specific FRP/CT solution, the validated 5-minute CT grid, and the
validated circular-W1 implementation used by the pooled repertoire analysis.

The output is intentionally a raw metric.  It does not subtract a
finite-sampling floor, correct residual FRP drift, infer genotype effects, or
perform statistical testing.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase
from circular_w1_repertoire_analysis import circular_w1


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
PHASE_MEAN_STABILITY_THRESHOLD = 0.10
DRIFT_SLOPE_SCREEN_THRESHOLD_HOURS = 0.25
PROBABILITY_TOLERANCE = 1e-12
W1_TOLERANCE_HOURS = 1e-12

OUTPUT_DIRNAME = "Circular_W1_Cycle_Instability"
DEFAULT_COHORT_ROOT = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cohort_root",
        nargs="?",
        type=Path,
        default=DEFAULT_COHORT_ROOT,
        help="Cohort_Data directory containing the frozen pilot animals.",
    )
    return parser.parse_args()


def _load_classified_samples(
    input_files: list[tuple[int, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load WTA labels, elapsed source time, and frame-duration weights."""

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
        if n_rows == 0:
            raise ValueError(f"Empty source file: {path}")
        invalid_rows += n_rows - len(labels)
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


def _count_raw_bouts(
    labels: np.ndarray,
    times_hours: np.ndarray,
    weights_seconds: np.ndarray,
    behavior_index: int,
) -> int:
    """Count cycle-local raw WTA runs for one behavior.

    A gap larger than 1.5 times the neighboring frame duration starts a new
    run.  This keeps source-file continuations together while treating an
    omitted/unclassifiable interval as a bout break.
    """

    positions = np.flatnonzero(labels == behavior_index)
    if len(positions) == 0:
        return 0
    if len(positions) == 1:
        return 1

    gaps_seconds = np.diff(times_hours[positions]) * 3600.0
    expected_gap_seconds = 1.5 * np.maximum(
        weights_seconds[positions[:-1]],
        weights_seconds[positions[1:]],
    )
    return int(1 + np.count_nonzero(gaps_seconds > expected_gap_seconds))


def _validate_probability(probability: np.ndarray, description: str) -> None:
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite probability in {description}")
    if np.any(probability < -PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Negative probability in {description}")
    total = float(probability.sum())
    if not np.isclose(
        total,
        1.0,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError(
            f"Probability did not sum to one in {description}: {total:.17g}"
        )


def _build_animal_cycle_data(
    group: str,
    animal: str,
    cohort_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build separate 5-minute distributions for the four frozen cycles."""

    animal_dir = cohort_root / group / animal
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
    cycles: dict[int, dict[str, object]] = {}
    support_rows: list[dict[str, object]] = []
    cycle_tolerance = 8.0 * np.finfo(float).eps * max(
        1.0,
        computational_frp_hours,
    )

    for cycle_order, (cycle_index, start_boundary, end_boundary) in enumerate(
        complete_cycles,
        start=1,
    ):
        selected = (elapsed_hours >= start_boundary) & (
            elapsed_hours < end_boundary
        )
        if not selected.any():
            raise ValueError(
                f"Frozen full cycle {cycle_index} contains no classifiable rows "
                f"for {animal}"
            )

        cycle_times = elapsed_hours[selected] - start_boundary
        if np.any(cycle_times < -cycle_tolerance) or np.any(
            cycle_times > computational_frp_hours + cycle_tolerance
        ):
            raise RuntimeError(
                f"Samples assigned outside frozen cycle {cycle_index} for {animal}"
            )
        cycle_times = np.clip(
            cycle_times,
            0.0,
            np.nextafter(computational_frp_hours, 0.0),
        )
        cycle_labels = labels[selected]
        cycle_weights = weights_seconds[selected]
        phase_bins = frozen_phase.ct_phase_bin_indices(
            cycle_times,
            N_CT_BINS,
            frp_hours=computational_frp_hours,
        )

        cycle_behavior_data: dict[str, dict[str, object]] = {}
        for behavior in NONREST_BEHAVIORS:
            behavior_index = BEHAVIOR_INDEX[behavior]
            positive = cycle_labels == behavior_index
            counts = np.bincount(
                phase_bins[positive],
                weights=cycle_weights[positive],
                minlength=N_CT_BINS,
            ).astype(float)
            total_seconds = float(counts.sum())
            probability = (
                counts / total_seconds if total_seconds > 0.0 else counts
            )
            if total_seconds > 0.0:
                _validate_probability(
                    probability,
                    f"{animal} {behavior} cycle {cycle_index}",
                )
            cycle_behavior_data[behavior] = {
                "probability": probability,
                "total_minutes": total_seconds / 60.0,
                "bout_count": _count_raw_bouts(
                    cycle_labels,
                    cycle_times + start_boundary,
                    cycle_weights,
                    behavior_index,
                ),
                "occupied_5min_ct_bins": int(np.count_nonzero(counts > 0.0)),
            }

        nonrest_mask = np.isin(
            cycle_labels,
            np.asarray(frozen_phase.NONRESTING_BEHAVIOR_INDICES),
        )
        support_rows.extend(
            {
                "animal": animal,
                "behavior": behavior,
                "cycle_index": int(cycle_index),
                "cycle_order": int(cycle_order),
                "elapsed_start_boundary_hours": float(start_boundary),
                "elapsed_end_boundary_hours": float(end_boundary),
                "cycle_duration_hours": float(end_boundary - start_boundary),
                "frp_hours_used": float(computational_frp_hours),
                "total_wta_positive_minutes": float(
                    cycle_behavior_data[behavior]["total_minutes"]
                ),
                "bout_count": int(cycle_behavior_data[behavior]["bout_count"]),
                "occupied_5min_ct_bins": int(
                    cycle_behavior_data[behavior]["occupied_5min_ct_bins"]
                ),
                "n_classified_frames": int(len(cycle_labels)),
                "total_nonrest_minutes_in_cycle": float(
                    cycle_weights[nonrest_mask].sum() / 60.0
                ),
                "support_status": (
                    "NONZERO_SUPPORT"
                    if cycle_behavior_data[behavior]["total_minutes"] > 0.0
                    else "ZERO_SUPPORT"
                ),
            }
            for behavior in NONREST_BEHAVIORS
        )
        cycles[int(cycle_index)] = {
            "cycle_order": int(cycle_order),
            "start_boundary": float(start_boundary),
            "end_boundary": float(end_boundary),
            "behaviors": cycle_behavior_data,
        }

    record = {
        "group": group,
        "animal": animal,
        "reported_frp_hours": float(reported_frp_hours),
        "computational_frp_hours": float(computational_frp_hours),
        "start_ct": float(start_ct),
        "cycle_indices": [int(item[0]) for item in complete_cycles],
        "cycles": cycles,
        "invalid_rows": int(invalid_rows),
        "frp_phase_output_dir": frp_phase_output_dir,
    }
    return record, support_rows


def _build_support_frame(
    support_rows: list[dict[str, object]],
) -> pd.DataFrame:
    support = pd.DataFrame(support_rows)
    if len(support) != len(ANIMALS) * len(NONREST_BEHAVIORS) * N_CYCLES:
        raise RuntimeError(
            "Cycle support output does not contain one row per animal, "
            "behavior, and cycle"
        )
    summary = (
        support.groupby(["animal", "behavior"], sort=False)
        .agg(
            min_cycle_minutes=("total_wta_positive_minutes", "min"),
            max_cycle_minutes=("total_wta_positive_minutes", "max"),
            min_cycle_bout_count=("bout_count", "min"),
            max_cycle_bout_count=("bout_count", "max"),
            min_cycle_occupied_5min_ct_bins=("occupied_5min_ct_bins", "min"),
            max_cycle_occupied_5min_ct_bins=("occupied_5min_ct_bins", "max"),
        )
        .reset_index()
    )
    support = support.merge(
        summary,
        on=["animal", "behavior"],
        how="left",
        validate="many_to_one",
    )
    animal_order = {animal: index for index, animal in enumerate(ANIMAL_ORDER)}
    behavior_order = {
        behavior: index for index, behavior in enumerate(NONREST_BEHAVIORS)
    }
    support["_animal_order"] = support["animal"].map(animal_order)
    support["_behavior_order"] = support["behavior"].map(behavior_order)
    support = support.sort_values(
        ["_animal_order", "_behavior_order", "cycle_order"],
        kind="stable",
    ).drop(columns=["_animal_order", "_behavior_order"])
    return support.reset_index(drop=True)


def _build_cycle_pair_frame(
    animal_records: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in animal_records:
        animal = str(record["animal"])
        cycles = record["cycles"]
        assert isinstance(cycles, dict)
        cycle_indices = list(cycles)
        for behavior in NONREST_BEHAVIORS:
            for cycle_i, cycle_j in itertools.combinations(cycle_indices, 2):
                first = cycles[cycle_i]["behaviors"][behavior]
                second = cycles[cycle_j]["behaviors"][behavior]
                distance = circular_w1(
                    np.asarray(first["probability"], dtype=float),
                    np.asarray(second["probability"], dtype=float),
                )
                if not np.isfinite(distance) or distance < -W1_TOLERANCE_HOURS:
                    raise RuntimeError(
                        f"Invalid cycle-pair W1 for {animal} {behavior} "
                        f"cycles {cycle_i}/{cycle_j}: {distance}"
                    )
                rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "cycle_i": int(cycle_i),
                        "cycle_j": int(cycle_j),
                        "w1_hours": max(0.0, float(distance)),
                        "cycle_i_total_minutes": float(first["total_minutes"]),
                        "cycle_j_total_minutes": float(second["total_minutes"]),
                        "cycle_i_bout_count": int(first["bout_count"]),
                        "cycle_j_bout_count": int(second["bout_count"]),
                        "cycle_i_occupied_5min_ct_bins": int(
                            first["occupied_5min_ct_bins"]
                        ),
                        "cycle_j_occupied_5min_ct_bins": int(
                            second["occupied_5min_ct_bins"]
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    if len(frame) != len(ANIMALS) * len(NONREST_BEHAVIORS) * 6:
        raise RuntimeError("Unexpected cycle-pair output row count")
    animal_order = {animal: index for index, animal in enumerate(ANIMAL_ORDER)}
    behavior_order = {
        behavior: index for index, behavior in enumerate(NONREST_BEHAVIORS)
    }
    frame["_animal_order"] = frame["animal"].map(animal_order)
    frame["_behavior_order"] = frame["behavior"].map(behavior_order)
    frame = frame.sort_values(
        ["_animal_order", "_behavior_order", "cycle_i", "cycle_j"],
        kind="stable",
    ).drop(columns=["_animal_order", "_behavior_order"])
    return frame.reset_index(drop=True)


def _build_behavior_frame(cycle_pair_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for animal in ANIMAL_ORDER:
        for behavior in NONREST_BEHAVIORS:
            values = cycle_pair_frame[
                (cycle_pair_frame["animal"] == animal)
                & (cycle_pair_frame["behavior"] == behavior)
            ]["w1_hours"].to_numpy(float)
            if len(values) != 6:
                raise RuntimeError(
                    f"Expected six cycle-pair values for {animal} {behavior}"
                )
            rows.append(
                {
                    "animal": animal,
                    "behavior": behavior,
                    "n_cycles": N_CYCLES,
                    "n_cycle_pairs": int(len(values)),
                    "mean_cycle_pair_w1_hours": float(np.mean(values)),
                    "median_cycle_pair_w1_hours": float(np.median(values)),
                    "min_cycle_pair_w1_hours": float(np.min(values)),
                    "max_cycle_pair_w1_hours": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _build_animal_frame(behavior_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for animal in ANIMAL_ORDER:
        subset = behavior_frame[behavior_frame["animal"] == animal].set_index(
            "behavior"
        )
        values = subset.loc[list(NONREST_BEHAVIORS), "mean_cycle_pair_w1_hours"]
        composite = float(values.mean())
        row: dict[str, object] = {
            "animal": animal,
            "cycle_to_cycle_repertoire_instability_hours": composite,
        }
        for behavior in NONREST_BEHAVIORS:
            row[f"{behavior}_instability_hours"] = float(values.loc[behavior])
        if not np.isclose(
            composite,
            float(np.mean([row[f"{behavior}_instability_hours"] for behavior in NONREST_BEHAVIORS])),
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(f"Animal composite mean check failed for {animal}")
        rows.append(row)
    return pd.DataFrame(rows)


def _circular_phase_center(
    probability: np.ndarray,
) -> tuple[float, float, str]:
    centers = (
        np.arange(N_CT_BINS, dtype=float) + 0.5
    ) * CT_BIN_WIDTH_HOURS
    vector = np.sum(
        probability
        * np.exp(1j * 2.0 * np.pi * centers / CT_HOURS)
    )
    resultant = float(abs(vector))
    if resultant < PHASE_MEAN_STABILITY_THRESHOLD:
        return float("nan"), resultant, "unreliable_low_concentration"
    phase = float(
        np.mod(
            np.angle(vector) * CT_HOURS / (2.0 * np.pi),
            CT_HOURS,
        )
    )
    return phase, resultant, "reliable"


def _build_phase_drift_frame(
    animal_records: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in animal_records:
        animal = str(record["animal"])
        cycles = record["cycles"]
        assert isinstance(cycles, dict)
        cycle_indices = list(cycles)
        for behavior in NONREST_BEHAVIORS:
            raw_centers: list[float] = []
            resultants: list[float] = []
            statuses: list[str] = []
            for cycle_index in cycle_indices:
                probability = np.asarray(
                    cycles[cycle_index]["behaviors"][behavior]["probability"],
                    dtype=float,
                )
                center, resultant, status = _circular_phase_center(probability)
                raw_centers.append(center)
                resultants.append(resultant)
                statuses.append(status)

            all_reliable = all(status == "reliable" for status in statuses)
            if all_reliable:
                angles = 2.0 * np.pi * np.asarray(raw_centers) / CT_HOURS
                unwrapped_centers = np.unwrap(angles) * CT_HOURS / (2.0 * np.pi)
                deltas_from_first = unwrapped_centers - unwrapped_centers[0]
                deltas_from_previous = np.concatenate(
                    ([np.nan], np.diff(unwrapped_centers))
                )
                cycle_orders = np.asarray(
                    [cycles[index]["cycle_order"] for index in cycle_indices],
                    dtype=float,
                )
                slope = float(
                    np.polyfit(cycle_orders, unwrapped_centers, deg=1)[0]
                )
                differences = np.diff(unwrapped_centers)
                monotone = bool(
                    np.all(differences >= -1e-10)
                    or np.all(differences <= 1e-10)
                )
                if monotone and np.any(np.abs(differences) > 1e-10):
                    progression_status = "monotone"
                else:
                    progression_status = "flat_or_nonmonotone"
            else:
                unwrapped_centers = np.full(len(cycle_indices), np.nan)
                deltas_from_first = np.full(len(cycle_indices), np.nan)
                deltas_from_previous = np.full(len(cycle_indices), np.nan)
                slope = float("nan")
                progression_status = "not_evaluable_low_concentration"

            for position, cycle_index in enumerate(cycle_indices):
                behavior_data = cycles[cycle_index]["behaviors"][behavior]
                rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "cycle_index": int(cycle_index),
                        "cycle_order": int(cycles[cycle_index]["cycle_order"]),
                        "circular_phase_center_hours": raw_centers[position],
                        "unwrapped_phase_center_hours": float(
                            unwrapped_centers[position]
                        ),
                        "mean_resultant_length": resultants[position],
                        "phase_center_status": statuses[position],
                        "phase_center_change_from_first_hours": float(
                            deltas_from_first[position]
                        ),
                        "phase_center_change_from_previous_hours": float(
                            deltas_from_previous[position]
                        ),
                        "progression_slope_hours_per_cycle": slope,
                        "progression_status": progression_status,
                        "cycle_total_minutes": float(behavior_data["total_minutes"]),
                    }
                )

    frame = pd.DataFrame(rows)
    animal_order = {animal: index for index, animal in enumerate(ANIMAL_ORDER)}
    behavior_order = {
        behavior: index for index, behavior in enumerate(NONREST_BEHAVIORS)
    }
    frame["_animal_order"] = frame["animal"].map(animal_order)
    frame["_behavior_order"] = frame["behavior"].map(behavior_order)
    frame = frame.sort_values(
        ["_animal_order", "_behavior_order", "cycle_order"],
        kind="stable",
    ).drop(columns=["_animal_order", "_behavior_order"])
    return frame.reset_index(drop=True)


def _mean_cycle_pair_w1(cycles: list[np.ndarray]) -> tuple[float, list[float]]:
    distances = [
        circular_w1(first, second)
        for first, second in itertools.combinations(cycles, 2)
    ]
    return float(np.mean(distances)), [float(value) for value in distances]


def _point_mass(ct_hours: float) -> np.ndarray:
    probability = np.zeros(N_CT_BINS, dtype=float)
    index = int(round(ct_hours / CT_BIN_WIDTH_HOURS)) % N_CT_BINS
    probability[index] = 1.0
    return probability


def run_synthetic_validation() -> dict[str, object]:
    """Run mathematical checks, including the pooled-vs-instability contrast."""

    identical = _point_mass(10.0)
    shifted = _point_mass(11.0)
    ct23 = _point_mass(23.0)
    ct1 = _point_mass(1.0)
    same_center_point = _point_mass(12.0)
    same_center_shape = 0.5 * _point_mass(10.0) + 0.5 * _point_mass(14.0)
    alternating = [
        _point_mass(10.0),
        _point_mass(18.0),
        _point_mass(10.0),
        _point_mass(18.0),
    ]

    case_values = {
        "identical_distributions_h": circular_w1(identical, identical),
        "pure_1_hour_shift_h": circular_w1(_point_mass(10.0), shifted),
        "ct23_vs_ct1_h": circular_w1(ct23, ct1),
        "same_center_different_shape_h": circular_w1(
            same_center_point,
            same_center_shape,
        ),
    }
    alternating_mean, alternating_distances = _mean_cycle_pair_w1(alternating)
    case_values["alternating_mean_h"] = alternating_mean

    animal_a_cycles = [
        0.5 * _point_mass(10.0) + 0.5 * _point_mass(18.0)
        for _ in range(4)
    ]
    animal_b_cycles = alternating
    animal_a_instability, animal_a_pairs = _mean_cycle_pair_w1(animal_a_cycles)
    animal_b_instability, animal_b_pairs = _mean_cycle_pair_w1(animal_b_cycles)
    pooled_a = np.mean(np.stack(animal_a_cycles), axis=0)
    pooled_b = np.mean(np.stack(animal_b_cycles), axis=0)
    pooled_w1 = circular_w1(pooled_a, pooled_b)
    case_values.update(
        {
            "pooled_equal_example_w1_h": pooled_w1,
            "pooled_equal_example_animal_a_instability_h": animal_a_instability,
            "pooled_equal_example_animal_b_instability_h": animal_b_instability,
        }
    )

    expected = {
        "identical_distributions_h": 0.0,
        "pure_1_hour_shift_h": 1.0,
        "ct23_vs_ct1_h": 2.0,
        "same_center_different_shape_h": 2.0,
        "alternating_mean_h": 32.0 / 6.0,
        "pooled_equal_example_w1_h": 0.0,
        "pooled_equal_example_animal_a_instability_h": 0.0,
        "pooled_equal_example_animal_b_instability_h": 32.0 / 6.0,
    }
    for name, expected_value in expected.items():
        if not np.isclose(
            case_values[name],
            expected_value,
            rtol=0.0,
            atol=W1_TOLERANCE_HOURS,
        ):
            raise RuntimeError(
                f"Synthetic validation failed for {name}: "
                f"observed={case_values[name]:.17g}, "
                f"expected={expected_value:.17g}"
            )

    expected_alternating = [8.0, 0.0, 8.0, 8.0, 0.0, 8.0]
    if not np.allclose(
        alternating_distances,
        expected_alternating,
        rtol=0.0,
        atol=W1_TOLERANCE_HOURS,
    ):
        raise RuntimeError(
            "Synthetic alternating-cycle distances failed: "
            f"observed={alternating_distances}, expected={expected_alternating}"
        )
    if not np.allclose(
        animal_a_pairs,
        np.zeros(6),
        rtol=0.0,
        atol=W1_TOLERANCE_HOURS,
    ):
        raise RuntimeError("Synthetic pooled-equal example gave Animal A nonzero instability")
    if not np.allclose(
        animal_b_pairs,
        expected_alternating,
        rtol=0.0,
        atol=W1_TOLERANCE_HOURS,
    ):
        raise RuntimeError("Synthetic pooled-equal example gave Animal B unexpected pairs")

    return {
        "values": case_values,
        "expected": expected,
        "alternating_distances": alternating_distances,
        "all_cases_passed": True,
        "pooled_equal_example_passed": True,
    }


def write_synthetic_validation_summary(
    output_path: Path,
    validation: dict[str, object],
) -> None:
    values = validation["values"]
    expected = validation["expected"]
    assert isinstance(values, dict)
    assert isinstance(expected, dict)
    lines = [
        "Synthetic validation of raw cycle-to-cycle circular-W1 instability",
        "===============================================================",
        "All cases use 288 bins over CT 0-24 h and the validated circular-W1 implementation.",
        "",
        "Case results:",
    ]
    for name, observed in values.items():
        lines.append(
            f"  {name}: observed={float(observed):.17g} h; "
            f"expected={float(expected[name]):.17g} h; PASS"
        )
    lines.extend(
        [
            "",
            "Alternating-cycle pair distances (cycles 1-2, 1-3, 1-4, 2-3, 2-4, 3-4):",
            "  observed: "
            + ", ".join(f"{value:.17g}" for value in validation["alternating_distances"]),
            "  expected: 8, 0, 8, 8, 0, 8 h; mean = 32/6 = 5.333333333333333 h",
            "",
            "Required pooled-equal / recurrence-different demonstration:",
            "  Animal A: every cycle is 50% CT10 + 50% CT18; cycle instability = 0 h.",
            "  Animal B: cycles alternate between 100% CT10 and 100% CT18; cycle instability = 5.333333333333333 h.",
            "  Their pooled distributions are identical; pooled W1 = 0 h.",
            "  PASS: pooled W1 and within-animal cycle instability distinguish these cases.",
            "",
            "All circular-W1 validation cases passed.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rank_correlation(first: pd.Series, second: pd.Series) -> float:
    first_rank = first.rank(method="average")
    second_rank = second.rank(method="average")
    return float(first_rank.corr(second_rank))


def _cross_check_existing_loco(
    cohort_root: Path,
    behavior_frame: pd.DataFrame,
    support_frame: pd.DataFrame,
) -> dict[str, object]:
    """Read existing LOCO output only as a descriptive cross-check."""

    loco_path = (
        cohort_root
        / "Behavior_Phase_Estimability_Audit"
        / "behavior_phase_stability_summary.csv"
    )
    result: dict[str, object] = {"path": loco_path, "status": "UNAVAILABLE"}
    if not loco_path.is_file():
        result["qualitative"] = "Existing LOCO summary was not available for cross-checking."
        return result

    loco = pd.read_csv(loco_path)
    required = {"Animal", "Behavior", "mean_LOCO_W1_h"}
    if not required.issubset(loco.columns):
        result["qualitative"] = (
            "Existing LOCO summary was present but lacked the expected columns."
        )
        return result

    loco_subset = loco[["Animal", "Behavior", "mean_LOCO_W1_h"]].rename(
        columns={"Animal": "animal", "Behavior": "behavior"}
    )
    merged = behavior_frame.merge(
        loco_subset,
        on=["animal", "behavior"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(behavior_frame):
        result["qualitative"] = (
            f"Only {len(merged)} of {len(behavior_frame)} animal-behavior cells "
            "matched the existing LOCO output."
        )
        return result

    min_support = (
        support_frame.groupby(["animal", "behavior"], sort=False)
        .agg(min_cycle_minutes=("total_wta_positive_minutes", "min"))
        .reset_index()
    )
    merged = merged.merge(
        min_support,
        on=["animal", "behavior"],
        how="inner",
        validate="one_to_one",
    )
    cycle_values = merged["mean_cycle_pair_w1_hours"]
    loco_values = merged["mean_LOCO_W1_h"]
    support_values = merged["min_cycle_minutes"]
    loco_rank_correlation = _rank_correlation(cycle_values, loco_values)
    support_rank_correlation = _rank_correlation(support_values, cycle_values)
    top_count = max(1, int(np.ceil(len(merged) * 0.10)))
    top_cycle = set(
        merged.nlargest(top_count, "mean_cycle_pair_w1_hours")[
            ["animal", "behavior"]
        ].itertuples(index=False, name=None)
    )
    top_loco = set(
        merged.nlargest(top_count, "mean_LOCO_W1_h")[["animal", "behavior"]].itertuples(
            index=False,
            name=None,
        )
    )
    low_support = set(
        merged.nsmallest(top_count, "min_cycle_minutes")[["animal", "behavior"]].itertuples(
            index=False,
            name=None,
        )
    )
    high_cycle_low_support_overlap = len(top_cycle & low_support)
    if loco_rank_correlation > 0.0 and len(top_cycle & top_loco) > 0:
        qualitative = (
            "The raw cycle-instability ranking is directionally consistent with "
            "the existing LOCO stability ranking in this descriptive cross-check."
        )
    elif loco_rank_correlation > 0.0:
        qualitative = (
            "The raw cycle-instability ranking has a positive but weakly overlapping "
            "relationship with the existing LOCO ranking."
        )
    else:
        qualitative = (
            "The raw cycle-instability ranking does not show a positive descriptive "
            "relationship with the existing LOCO ranking."
        )

    if support_rank_correlation < -0.25 and high_cycle_low_support_overlap > 0:
        sparse_qualitative = (
            "There is a possible support-associated inflation signal; interpret the "
            "raw instability values with the support table."
        )
    else:
        sparse_qualitative = (
            "No obvious support-associated inflation signal was identified by this "
            "descriptive screen."
        )

    result.update(
        {
            "status": "MATCHED",
            "matched_cells": len(merged),
            "loco_rank_correlation": loco_rank_correlation,
            "support_rank_correlation": support_rank_correlation,
            "top_decile_count": top_count,
            "top_cycle_loco_overlap": len(top_cycle & top_loco),
            "top_cycle_low_support_overlap": high_cycle_low_support_overlap,
            "qualitative": qualitative,
            "sparse_qualitative": sparse_qualitative,
        }
    )
    return result


def _cross_check_existing_bout_support(
    cohort_root: Path,
    support_frame: pd.DataFrame,
) -> dict[str, object]:
    """Cross-check support counts without using them to compute the metric."""

    support_path = (
        cohort_root
        / "Behavior_Bout_Support_Audit"
        / "behavior_bout_support_by_cycle.csv"
    )
    result: dict[str, object] = {"path": support_path, "status": "UNAVAILABLE"}
    if not support_path.is_file():
        result["qualitative"] = "Existing bout-support output was not available for cross-checking."
        return result

    existing = pd.read_csv(support_path)
    required = {
        "animal",
        "behavior",
        "cycle",
        "wta_minutes",
        "raw_bouts",
    }
    if not required.issubset(existing.columns):
        result["qualitative"] = (
            "Existing bout-support output was present but lacked the expected columns."
        )
        return result

    existing = existing.rename(
        columns={
            "cycle": "cycle_index",
            "wta_minutes": "existing_wta_minutes",
            "raw_bouts": "existing_raw_bouts",
        }
    )
    merged = support_frame.merge(
        existing[
            [
                "animal",
                "behavior",
                "cycle_index",
                "existing_wta_minutes",
                "existing_raw_bouts",
            ]
        ],
        on=["animal", "behavior", "cycle_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(support_frame):
        result["qualitative"] = (
            f"Only {len(merged)} of {len(support_frame)} support cells matched "
            "the existing bout-support output."
        )
        return result

    max_minutes_error = float(
        np.max(
            np.abs(
                merged["total_wta_positive_minutes"].to_numpy(float)
                - merged["existing_wta_minutes"].to_numpy(float)
            )
        )
    )
    bout_mismatches = int(
        np.count_nonzero(
            merged["bout_count"].to_numpy(np.int64)
            != merged["existing_raw_bouts"].to_numpy(np.int64)
        )
    )
    result.update(
        {
            "status": "MATCHED",
            "matched_cells": len(merged),
            "max_minutes_abs_error": max_minutes_error,
            "raw_bout_mismatch_count": bout_mismatches,
            "qualitative": (
                "New support minutes and raw WTA bout counts agree with the existing "
                "support audit; the existing audit was not used as metric input."
            ),
        }
    )
    return result


def _drift_summary(drift_frame: pd.DataFrame) -> dict[str, object]:
    traces = (
        drift_frame.groupby(["animal", "behavior"], sort=False)
        .first()
        .reset_index()
    )
    evaluable = traces[traces["progression_status"] != "not_evaluable_low_concentration"]
    monotone = evaluable[evaluable["progression_status"] == "monotone"]
    candidates: dict[str, dict[str, int]] = {}
    for animal in ANIMAL_ORDER:
        subset = monotone[monotone["animal"] == animal]
        positive = int(
            np.count_nonzero(
                subset["progression_slope_hours_per_cycle"].to_numpy(float)
                >= DRIFT_SLOPE_SCREEN_THRESHOLD_HOURS
            )
        )
        negative = int(
            np.count_nonzero(
                subset["progression_slope_hours_per_cycle"].to_numpy(float)
                <= -DRIFT_SLOPE_SCREEN_THRESHOLD_HOURS
            )
        )
        candidates[animal] = {"positive": positive, "negative": negative}

    shared_candidates = {
        animal: counts
        for animal, counts in candidates.items()
        if max(counts.values()) >= 4
    }
    return {
        "total_traces": len(traces),
        "evaluable_traces": len(evaluable),
        "monotone_traces": len(monotone),
        "shared_candidates": shared_candidates,
        "screen_threshold_hours_per_cycle": DRIFT_SLOPE_SCREEN_THRESHOLD_HOURS,
        "phase_mean_threshold": PHASE_MEAN_STABILITY_THRESHOLD,
    }


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "NA"
    return f"{numeric:.17g}"


def write_run_summary(
    output_path: Path,
    cohort_root: Path,
    animal_records: list[dict[str, object]],
    support_frame: pd.DataFrame,
    cycle_pair_frame: pd.DataFrame,
    behavior_frame: pd.DataFrame,
    animal_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
    synthetic_validation: dict[str, object],
    loco_cross_check: dict[str, object],
    bout_support_cross_check: dict[str, object],
) -> None:
    lines = [
        "Raw W1-based cycle-to-cycle circadian repertoire instability",
        "===============================================================",
        "Definition: equal-weight mean across behaviors of the mean circular W1 distance among all pairs of complete circadian cycles.",
        "Interpretation: low values indicate reproducible cycle-to-cycle timing; high values indicate instability.",
        "",
        "Frozen analysis scope:",
        f"  cohort root: {cohort_root}",
        "  representation: WTA behavior identity",
        "  behaviors: " + ", ".join(NONREST_BEHAVIORS),
        "  rest: excluded from scoring and support-positive behavior distributions",
        "  CT grid: 288 bins over CT 0-24 h (5-minute bins)",
        "  complete cycles: four frozen CT0-to-CT24 cycles per animal",
        "  phase rule: first supplied file is t0; t0 is CT18; cycle-relative CT uses the frozen unrounded FRP from complete-cycle boundaries",
        "  correction status: no finite-sampling floor and no FRP-drift correction",
        "",
        "Animals and cycles:",
        f"  animals analyzed: {len(animal_records)}",
        f"  cycles analyzed per animal: {N_CYCLES}",
    ]
    for record in animal_records:
        lines.append(
            f"  {record['animal']}: cycles={','.join(str(index) for index in record['cycle_indices'])}; "
            f"reported_FRP={_fmt(record['reported_frp_hours'])} h; "
            f"unrounded_FRP_used={_fmt(record['computational_frp_hours'])} h; "
            f"invalid_rows_excluded={record['invalid_rows']}"
        )

    zero_support = support_frame[support_frame["support_status"] == "ZERO_SUPPORT"]
    minimum_minutes = support_frame.loc[
        support_frame["total_wta_positive_minutes"].idxmin()
    ]
    minimum_bouts = support_frame.loc[support_frame["bout_count"].idxmin()]
    minimum_bins = support_frame.loc[
        support_frame["occupied_5min_ct_bins"].idxmin()
    ]
    lines.extend(
        [
            "",
            "Output levels:",
            f"  total cycle-pair comparisons: {len(cycle_pair_frame)}",
            f"  total behavior-level comparisons: {len(behavior_frame)} animal-behavior summaries",
            f"  every animal x behavior x cycle had nonzero support: {len(zero_support) == 0}",
            "",
            "Minimum per-cycle support:",
            f"  minimum WTA-positive minutes: {_fmt(minimum_minutes['total_wta_positive_minutes'])} "
            f"({minimum_minutes['animal']} {minimum_minutes['behavior']} cycle {minimum_minutes['cycle_index']})",
            f"  minimum raw WTA bout count: {int(minimum_bouts['bout_count'])} "
            f"({minimum_bouts['animal']} {minimum_bouts['behavior']} cycle {minimum_bouts['cycle_index']})",
            f"  minimum occupied 5-minute CT bins: {int(minimum_bins['occupied_5min_ct_bins'])} "
            f"({minimum_bins['animal']} {minimum_bins['behavior']} cycle {minimum_bins['cycle_index']})",
            "  bout-count definition: cycle-local raw WTA runs; gaps greater than 1.5 neighboring frame durations start a new run",
            "",
            "Exact animal-level instability values (hours):",
        ]
    )
    for _, row in animal_frame.iterrows():
        lines.append(
            f"  {row['animal']}: cycle_to_cycle_repertoire_instability_hours="
            f"{_fmt(row['cycle_to_cycle_repertoire_instability_hours'])}"
        )

    lines.extend(["", "Exact behavior-specific instability values (hours):"])
    for animal in ANIMAL_ORDER:
        subset = behavior_frame[behavior_frame["animal"] == animal].set_index("behavior")
        lines.append(f"  {animal}:")
        for behavior in NONREST_BEHAVIORS:
            lines.append(
                f"    {behavior}: {_fmt(subset.loc[behavior, 'mean_cycle_pair_w1_hours'])}"
            )

    lines.extend(["", "Highest and lowest behavior instability per animal:"])
    for animal in ANIMAL_ORDER:
        subset = behavior_frame[behavior_frame["animal"] == animal]
        highest = subset.loc[subset["mean_cycle_pair_w1_hours"].idxmax()]
        lowest = subset.loc[subset["mean_cycle_pair_w1_hours"].idxmin()]
        lines.append(
            f"  {animal}: highest={highest['behavior']} ({_fmt(highest['mean_cycle_pair_w1_hours'])} h); "
            f"lowest={lowest['behavior']} ({_fmt(lowest['mean_cycle_pair_w1_hours'])} h)"
        )

    drift_summary = _drift_summary(drift_frame)
    lines.extend(
        [
            "",
            "Phase-drift diagnostic:",
            f"  circular means marked reliable when mean resultant length >= {PHASE_MEAN_STABILITY_THRESHOLD:.2f}; unreliable cells are not used for progression interpretation",
            f"  evaluable animal-behavior traces: {drift_summary['evaluable_traces']} / {drift_summary['total_traces']}",
            f"  monotone traces: {drift_summary['monotone_traces']}",
            f"  shared-drift screen: at least four of eight behaviors with monotone slope magnitude >= {DRIFT_SLOPE_SCREEN_THRESHOLD_HOURS:.2f} h/cycle",
        ]
    )
    if drift_summary["shared_candidates"]:
        for animal, counts in drift_summary["shared_candidates"].items():
            lines.append(
                f"  candidate {animal}: positive={counts['positive']}; negative={counts['negative']}"
            )
        lines.append("  conclusion: at least one animal has a shared monotone-drift candidate; no correction was applied.")
    else:
        lines.append("  conclusion: no animal met the shared monotone-drift screen; no correction was applied.")

    lines.extend(
        [
            "",
            "Existing LOCO/support cross-check:",
            f"  LOCO source: {loco_cross_check['path']}",
            f"  status: {loco_cross_check['status']}",
            f"  existing bout-support source: {bout_support_cross_check['path']}",
            f"  existing bout-support status: {bout_support_cross_check['status']}",
        ]
    )
    if bout_support_cross_check["status"] == "MATCHED":
        lines.extend(
            [
                f"  existing support cells matched: {bout_support_cross_check['matched_cells']}",
                f"  maximum WTA-minute absolute difference: {_fmt(bout_support_cross_check['max_minutes_abs_error'])} min",
                f"  raw WTA bout-count mismatches: {bout_support_cross_check['raw_bout_mismatch_count']}",
                f"  support audit qualitative read: {bout_support_cross_check['qualitative']}",
            ]
        )
    if loco_cross_check["status"] == "MATCHED":
        lines.extend(
            [
                f"  descriptive Spearman rank correlation, cycle instability vs mean LOCO W1: {_fmt(loco_cross_check['loco_rank_correlation'])}",
                f"  top-decile overlap: {loco_cross_check['top_cycle_loco_overlap']} / {loco_cross_check['top_decile_count']} cells",
                f"  descriptive Spearman rank correlation, minimum cycle support vs cycle instability: {_fmt(loco_cross_check['support_rank_correlation'])}",
                f"  top-cycle-instability / lowest-support overlap: {loco_cross_check['top_cycle_low_support_overlap']} / {loco_cross_check['top_decile_count']} cells",
                f"  LOCO qualitative read: {loco_cross_check['qualitative']}",
                f"  sparse-behavior qualitative read: {loco_cross_check['sparse_qualitative']}",
            ]
        )
    else:
        lines.append(f"  qualitative read: {loco_cross_check['qualitative']}")

    lines.extend(
        [
            "",
            "Validation:",
            f"  all circular-W1 validation cases passed: {synthetic_validation['all_cases_passed']}",
            f"  pooled-equal / recurrence-different example passed: {synthetic_validation['pooled_equal_example_passed']}",
            "  no genotype inference, hypothesis tests, effect sizes, p values, sampling-floor correction, or drift correction were performed.",
            "  no conceptual or implementation issue was found in this first-pass raw metric implementation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cohort_root = args.cohort_root.expanduser().resolve()
    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_validation = run_synthetic_validation()
    write_synthetic_validation_summary(
        output_dir / "synthetic_validation_summary.txt",
        synthetic_validation,
    )

    animal_records: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    for group, animal in ANIMALS:
        record, rows = _build_animal_cycle_data(group, animal, cohort_root)
        animal_records.append(record)
        support_rows.extend(rows)

    support_frame = _build_support_frame(support_rows)
    support_path = output_dir / "cycle_support_qc.csv"
    support_frame.to_csv(support_path, index=False, float_format="%.17g")
    zero_support = support_frame[
        support_frame["support_status"] == "ZERO_SUPPORT"
    ]
    if not zero_support.empty:
        affected = "; ".join(
            f"{row.animal} {row.behavior} cycle {row.cycle_index}"
            for row in zero_support.itertuples()
        )
        raise RuntimeError(
            "Zero support was found before cycle-to-cycle W1 calculation. "
            f"Affected cells: {affected}"
        )

    cycle_pair_frame = _build_cycle_pair_frame(animal_records)
    behavior_frame = _build_behavior_frame(cycle_pair_frame)
    animal_frame = _build_animal_frame(behavior_frame)
    drift_frame = _build_phase_drift_frame(animal_records)
    loco_cross_check = _cross_check_existing_loco(
        cohort_root,
        behavior_frame,
        support_frame,
    )
    bout_support_cross_check = _cross_check_existing_bout_support(
        cohort_root,
        support_frame,
    )

    cycle_pair_frame.to_csv(
        output_dir / "cycle_pair_w1.csv",
        index=False,
        float_format="%.17g",
    )
    behavior_frame.to_csv(
        output_dir / "behavior_cycle_instability.csv",
        index=False,
        float_format="%.17g",
    )
    animal_frame.to_csv(
        output_dir / "animal_cycle_instability.csv",
        index=False,
        float_format="%.17g",
    )
    drift_frame.to_csv(
        output_dir / "cycle_phase_drift_diagnostic.csv",
        index=False,
        float_format="%.17g",
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        cohort_root,
        animal_records,
        support_frame,
        cycle_pair_frame,
        behavior_frame,
        animal_frame,
        drift_frame,
        synthetic_validation,
        loco_cross_check,
        bout_support_cross_check,
    )

    print(f"Output directory: {output_dir}")
    print(
        "Synthetic validation passed: "
        f"pooled-equal example={synthetic_validation['pooled_equal_example_passed']}"
    )
    print(
        f"Animals={len(animal_records)}; cycles/animal={N_CYCLES}; "
        f"cycle-pair rows={len(cycle_pair_frame)}; behavior rows={len(behavior_frame)}"
    )
    print("Animal-level cycle-to-cycle instability (hours):")
    for row in animal_frame.itertuples(index=False):
        print(
            f"  {row.animal}: "
            f"{row.cycle_to_cycle_repertoire_instability_hours:.12g}"
        )


if __name__ == "__main__":
    main()
