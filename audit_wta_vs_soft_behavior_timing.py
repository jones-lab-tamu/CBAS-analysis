"""Compare WTA and soft-probability behavior phase representations.

This is a standalone representation audit.  It reads the CBAS probability
columns directly, reuses the frozen FRP/CT cycle definitions and the validated
circular W1 implementation, and does not modify production analyses or raw
data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import frp_phase_analysis as frozen_frp
import phase_behavior_mutual_information as frozen_phase
from audit_behavior_phase_estimability import (
    circular_wasserstein_1,
    run_circular_wasserstein_checks,
)


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
NONREST_INDICES = tuple(frozen_phase.NONRESTING_BEHAVIOR_INDICES)
NONREST_BEHAVIORS = tuple(
    frozen_phase.BEHAVIORS[index] for index in NONREST_INDICES
)
N_CT_BINS = 288
CT_BIN_WIDTH_HOURS = frozen_phase.CT_HOURS / N_CT_BINS
CT_BIN_CENTERS_HOURS = (
    np.arange(N_CT_BINS, dtype=float) + 0.5
) * CT_BIN_WIDTH_HOURS
OUTPUT_DIRNAME = "WTA_vs_Soft_Behavior_Audit"
SOFT_SUM_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-12

SUMMARY_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "WTA_duration_min",
    "SOFT_equivalent_min",
    "soft_to_wta_support_ratio",
    "fraction_soft_mass_nonWTA",
    "mean_soft_prob_on_WTA_frames",
    "mean_soft_prob_on_nonWTA_frames",
    "median_soft_prob_on_nonWTA_frames",
    "p95_soft_prob_on_nonWTA_frames",
    "WTA_phase_CT",
    "SOFT_phase_CT",
    "phase_difference_signed_h",
    "abs_phase_difference_h",
    "WTA_R",
    "SOFT_R",
    "delta_R",
    "W1_WTA_vs_SOFT_h",
    "WTA_LS_period_h",
    "SOFT_LS_period_h",
    "period_difference_h",
    "WTA_mean_LOCO_W1_h",
    "SOFT_mean_LOCO_W1_h",
    "delta_mean_LOCO_W1_h",
    "WTA_max_LOCO_W1_h",
    "SOFT_max_LOCO_W1_h",
]

PHASE_DISTRIBUTION_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "Representation",
    "CT_bin",
    "CT_start_h",
    "CT_end_h",
    "Probability",
]

LOCO_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "Representation",
    "omitted_cycle",
    "circular_W1_full_vs_LOCO_h",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Audit WTA versus soft behavior timing representations."
    )
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the eight animal folders.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def _circular_mean_and_resultant(
    probability: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(probability, dtype=float)
    total = float(values.sum())
    if values.ndim != 1 or values.size != N_CT_BINS or total <= 0.0:
        raise ValueError("Circular phase summary requires a positive 288-bin distribution")
    values = values / total
    vector = np.sum(
        values
        * np.exp(1j * 2.0 * np.pi * CT_BIN_CENTERS_HOURS / frozen_phase.CT_HOURS)
    )
    resultant = float(abs(vector))
    if resultant <= PROBABILITY_TOLERANCE:
        return float("nan"), resultant
    phase = float(
        np.mod(
            np.angle(vector)
            * frozen_phase.CT_HOURS
            / (2.0 * np.pi),
            frozen_phase.CT_HOURS,
        )
    )
    if np.isclose(
        phase,
        frozen_phase.CT_HOURS,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        phase = 0.0
    return phase, resultant


def _signed_circular_difference(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return float("nan")
    return float(
        np.mod(
            second - first + frozen_phase.CT_HOURS / 2.0,
            frozen_phase.CT_HOURS,
        )
        - frozen_phase.CT_HOURS / 2.0
    )


def _run_circular_mean_check() -> float:
    """Check a distribution straddling CT0/24 before real-data processing."""

    probability = np.zeros(N_CT_BINS, dtype=float)
    probability[0] = 0.5
    probability[-1] = 0.5
    phase, _resultant = _circular_mean_and_resultant(probability)
    wrapped_error = abs(_signed_circular_difference(0.0, phase))
    if not np.isclose(wrapped_error, 0.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError(
            f"Circular mean CT0/24 wrap check failed: phase={phase}, "
            f"wrapped error={wrapped_error}"
        )
    print(f"Circular mean CT0/24 wrap check: phase={phase:.12g} CT")
    return phase


def _read_probability_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read direct CBAS probabilities and verify the existing WTA labels."""

    frame = pd.read_csv(path, usecols=frozen_phase.BEHAVIORS)
    probabilities = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(frozen_phase.BEHAVIORS):
        raise ValueError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite CBAS probability found in {path}")

    row_sums = probabilities.sum(axis=1)
    deviations = np.abs(row_sums - 1.0)
    if deviations.size and float(deviations.max()) > SOFT_SUM_TOLERANCE:
        raise RuntimeError(
            f"Soft probability row-sum check failed in {path}: "
            f"maximum deviation={deviations.max():.17g}"
        )

    wta_labels = np.argmax(probabilities, axis=1).astype(np.int8)
    n_rows, row_indices, existing_labels = frozen_phase.read_and_classify(path)
    if n_rows != len(probabilities) or not np.array_equal(
        row_indices,
        np.arange(n_rows, dtype=np.int64),
    ):
        raise RuntimeError(
            f"The existing WTA classifier found unclassifiable rows in {path}"
        )
    if not np.array_equal(wta_labels, existing_labels):
        mismatch = int(np.count_nonzero(wta_labels != existing_labels))
        raise RuntimeError(
            f"WTA/argmax mismatch in {path}: {mismatch} rows"
        )
    return probabilities, wta_labels


def _new_row_sum_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "max_abs_deviation": 0.0,
    }


def _update_row_sum_stats(stats: dict[str, float], probabilities: np.ndarray) -> None:
    row_sums = probabilities.sum(axis=1)
    if row_sums.size == 0:
        return
    stats["count"] += float(row_sums.size)
    stats["sum"] += float(row_sums.sum())
    stats["min"] = min(stats["min"], float(row_sums.min()))
    stats["max"] = max(stats["max"], float(row_sums.max()))
    stats["max_abs_deviation"] = max(
        stats["max_abs_deviation"],
        float(np.abs(row_sums - 1.0).max()),
    )


def _distribution_rows(
    group: str,
    animal: str,
    behavior: str,
    representation: str,
    counts: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError(
            f"Zero {representation} support for {animal} {behavior}; "
            "no distribution will be fabricated"
        )
    probability = counts / total
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise RuntimeError(
            f"Invalid {representation} phase distribution for {animal} {behavior}"
        )
    if not np.isclose(
        probability.sum(),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError(
            f"{representation} phase distribution did not sum to one for "
            f"{animal} {behavior}"
        )
    rows = [
        {
            "Group": group,
            "Animal": animal,
            "Behavior": behavior,
            "Representation": representation,
            "CT_bin": bin_index,
            "CT_start_h": bin_index * CT_BIN_WIDTH_HOURS,
            "CT_end_h": (bin_index + 1) * CT_BIN_WIDTH_HOURS,
            "Probability": float(probability[bin_index]),
        }
        for bin_index in range(N_CT_BINS)
    ]
    return rows, probability


def _loco_distances(
    full_probability: np.ndarray,
    cycle_counts: np.ndarray,
    cycle_indices: list[int],
) -> tuple[list[float], list[dict[str, object]]]:
    distances: list[float] = []
    rows: list[dict[str, object]] = []
    for omitted_position, omitted_cycle in enumerate(cycle_indices):
        remaining_counts = np.delete(
            cycle_counts,
            omitted_position,
            axis=0,
        ).sum(axis=0)
        remaining_total = float(remaining_counts.sum())
        if remaining_total <= 0.0:
            rows.append(
                {
                    "omitted_cycle": omitted_cycle,
                    "circular_W1_full_vs_LOCO_h": np.nan,
                }
            )
            continue
        remaining_probability = remaining_counts / remaining_total
        distance = circular_wasserstein_1(
            full_probability,
            remaining_probability,
        )
        if not np.isfinite(distance) or distance < 0.0:
            raise RuntimeError(
                f"Invalid LOCO circular W1: omitted cycle {omitted_cycle}, "
                f"distance={distance}"
            )
        distances.append(distance)
        rows.append(
            {
                "omitted_cycle": omitted_cycle,
                "circular_W1_full_vs_LOCO_h": distance,
            }
        )
    return distances, rows


def _period_estimates(
    file_times: np.ndarray,
    file_means: np.ndarray,
) -> np.ndarray:
    """Use the existing FRP period grid and Lomb--Scargle implementation."""

    periods = frozen_frp.period_grid()
    estimates = np.full(file_means.shape[1], np.nan, dtype=float)
    for behavior_index in range(file_means.shape[1]):
        values = file_means[:, behavior_index]
        try:
            estimates[behavior_index], _power = frozen_frp.estimate_lomb_scargle_period(
                file_times,
                values,
                periods,
            )
        except ValueError:
            estimates[behavior_index] = np.nan
    return estimates


def _audit_animal(
    group: str,
    animal: str,
    cohort_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    animal_dir = cohort_root / group / animal
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Missing animal directory: {animal_dir}")

    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(f"Source indices are missing for {animal}: {missing_indices}")
    (
        _reported_frp_hours,
        computational_frp_hours,
        _start_ct,
        complete_cycle_specs,
        _frp_phase_output_dir,
    ) = frozen_phase.load_frp_phase_solution(animal_dir)
    if len(complete_cycle_specs) != 4:
        raise ValueError(
            f"Expected four complete cycles for {animal}, "
            f"found {len(complete_cycle_specs)}"
        )
    cycle_indices = [cycle_index for cycle_index, _, _ in complete_cycle_specs]
    n_behaviors = len(frozen_phase.BEHAVIORS)
    n_nonrest = len(NONREST_BEHAVIORS)
    wta_cycle_counts = np.zeros((4, N_CT_BINS, n_nonrest), dtype=float)
    soft_cycle_counts = np.zeros((4, N_CT_BINS, n_nonrest), dtype=float)
    wta_support_seconds = np.zeros(n_nonrest, dtype=float)
    soft_support_seconds = np.zeros(n_nonrest, dtype=float)
    soft_nonwta_seconds = np.zeros(n_nonrest, dtype=float)
    soft_on_wta_sum = np.zeros(n_nonrest, dtype=float)
    soft_on_nonwta_sum = np.zeros(n_nonrest, dtype=float)
    wta_frame_counts = np.zeros(n_nonrest, dtype=float)
    nonwta_frame_counts = np.zeros(n_nonrest, dtype=float)
    nonwta_probability_values: list[list[np.ndarray]] = [
        [] for _ in range(n_nonrest)
    ]
    file_times: list[float] = []
    file_wta_means: list[np.ndarray] = []
    file_soft_means: list[np.ndarray] = []
    row_sum_stats = _new_row_sum_stats()
    wta_mismatch_count = 0
    complete_cycle_selected_counts = np.zeros(4, dtype=np.int64)
    first_file_index = input_files[0][0]

    for file_index, path in input_files:
        probabilities, wta_labels = _read_probability_file(path)
        n_rows = len(probabilities)
        if n_rows == 0:
            continue
        _update_row_sum_stats(row_sum_stats, probabilities)
        file_times.append(
            (file_index - first_file_index + 0.5)
            * frozen_phase.FILE_DURATION_MINUTES
            / 60.0
        )
        file_wta_means.append(
            np.asarray(
                [np.mean(wta_labels == index) for index in range(n_behaviors)],
                dtype=float,
            )
        )
        file_soft_means.append(probabilities.mean(axis=0))
        row_duration_seconds = frozen_phase.FILE_DURATION_MINUTES * 60.0 / n_rows
        row_indices = np.arange(n_rows, dtype=np.int64)
        elapsed_hours = (
            (file_index - first_file_index)
            * frozen_phase.FILE_DURATION_MINUTES
            / 60.0
            + row_indices.astype(float)
            / n_rows
            * frozen_phase.FILE_DURATION_MINUTES
            / 60.0
        )

        for cycle_position, (_cycle_index, start_boundary, end_boundary) in enumerate(
            complete_cycle_specs
        ):
            selected = (elapsed_hours >= start_boundary) & (
                elapsed_hours < end_boundary
            )
            selected_count = int(np.count_nonzero(selected))
            if selected_count == 0:
                continue
            complete_cycle_selected_counts[cycle_position] += selected_count
            cycle_times = elapsed_hours[selected] - start_boundary
            cycle_times = np.clip(
                cycle_times,
                0.0,
                np.nextafter(computational_frp_hours, 0.0),
            )
            phase_bins = frozen_phase.ct_phase_bin_indices(
                cycle_times,
                N_CT_BINS,
                frp_hours=computational_frp_hours,
            )
            selected_probabilities = probabilities[selected][:, NONREST_INDICES]
            selected_labels = wta_labels[selected]
            selected_weight = row_duration_seconds
            for behavior_position, behavior_index in enumerate(NONREST_INDICES):
                wta_mask = selected_labels == behavior_index
                wta_weights = wta_mask.astype(float) * selected_weight
                soft_weights = selected_probabilities[:, behavior_position] * selected_weight
                wta_cycle_counts[cycle_position, :, behavior_position] += np.bincount(
                    phase_bins,
                    weights=wta_weights,
                    minlength=N_CT_BINS,
                )
                soft_cycle_counts[cycle_position, :, behavior_position] += np.bincount(
                    phase_bins,
                    weights=soft_weights,
                    minlength=N_CT_BINS,
                )
                wta_support_seconds[behavior_position] += float(wta_weights.sum())
                soft_support_seconds[behavior_position] += float(soft_weights.sum())
                nonwta_mask = ~wta_mask
                soft_nonwta_seconds[behavior_position] += float(
                    (selected_probabilities[nonwta_mask, behavior_position] * selected_weight).sum()
                )
                wta_count = int(np.count_nonzero(wta_mask))
                nonwta_count = int(np.count_nonzero(nonwta_mask))
                wta_frame_counts[behavior_position] += wta_count
                nonwta_frame_counts[behavior_position] += nonwta_count
                soft_on_wta_sum[behavior_position] += float(
                    selected_probabilities[wta_mask, behavior_position].sum()
                )
                soft_on_nonwta_sum[behavior_position] += float(
                    selected_probabilities[nonwta_mask, behavior_position].sum()
                )
                if nonwta_count:
                    nonwta_probability_values[behavior_position].append(
                        selected_probabilities[nonwta_mask, behavior_position]
                    )

    if not file_times:
        raise ValueError(f"No valid source frames were found for {animal}")
    if not np.array_equal(complete_cycle_selected_counts > 0, np.ones(4, dtype=bool)):
        raise ValueError(
            f"One or more complete cycles contain no classified frames for {animal}: "
            f"{complete_cycle_selected_counts.tolist()}"
        )
    file_times_array = np.asarray(file_times, dtype=float)
    file_wta_means_array = np.stack(file_wta_means, axis=0)
    file_soft_means_array = np.stack(file_soft_means, axis=0)
    wta_periods = _period_estimates(file_times_array, file_wta_means_array)
    soft_periods = _period_estimates(file_times_array, file_soft_means_array)

    phase_rows: list[dict[str, object]] = []
    loco_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
        wta_full_rows, wta_full_probability = _distribution_rows(
            group,
            animal,
            behavior,
            "WTA",
            wta_cycle_counts[:, :, behavior_position].sum(axis=0),
        )
        soft_full_rows, soft_full_probability = _distribution_rows(
            group,
            animal,
            behavior,
            "SOFT",
            soft_cycle_counts[:, :, behavior_position].sum(axis=0),
        )
        phase_rows.extend(wta_full_rows)
        phase_rows.extend(soft_full_rows)
        wta_phase, wta_resultant = _circular_mean_and_resultant(wta_full_probability)
        soft_phase, soft_resultant = _circular_mean_and_resultant(soft_full_probability)
        signed_phase_difference = _signed_circular_difference(wta_phase, soft_phase)
        w1_wta_soft = circular_wasserstein_1(
            wta_full_probability,
            soft_full_probability,
        )

        representation_loco: dict[str, tuple[list[float], float, float]] = {}
        for representation, full_probability, cycle_counts in (
            ("WTA", wta_full_probability, wta_cycle_counts[:, :, behavior_position]),
            ("SOFT", soft_full_probability, soft_cycle_counts[:, :, behavior_position]),
        ):
            distances, distance_rows = _loco_distances(
                full_probability,
                cycle_counts,
                cycle_indices,
            )
            for distance_row in distance_rows:
                loco_rows.append(
                    {
                        "Group": group,
                        "Animal": animal,
                        "Behavior": behavior,
                        "Representation": representation,
                        **distance_row,
                    }
                )
            representation_loco[representation] = (
                distances,
                float(np.mean(distances)) if distances else float("nan"),
                float(np.max(distances)) if distances else float("nan"),
            )

        nonwta_values = (
            np.concatenate(nonwta_probability_values[behavior_position])
            if nonwta_probability_values[behavior_position]
            else np.empty(0, dtype=float)
        )
        wta_duration_minutes = wta_support_seconds[behavior_position] / 60.0
        soft_equivalent_minutes = soft_support_seconds[behavior_position] / 60.0
        soft_ratio = (
            soft_equivalent_minutes / wta_duration_minutes
            if wta_duration_minutes > 0.0
            else float("nan")
        )
        summary_rows.append(
            {
                "Group": group,
                "Animal": animal,
                "Behavior": behavior,
                "WTA_duration_min": wta_duration_minutes,
                "SOFT_equivalent_min": soft_equivalent_minutes,
                "soft_to_wta_support_ratio": soft_ratio,
                "fraction_soft_mass_nonWTA": (
                    soft_nonwta_seconds[behavior_position]
                    / soft_support_seconds[behavior_position]
                    if soft_support_seconds[behavior_position] > 0.0
                    else float("nan")
                ),
                "mean_soft_prob_on_WTA_frames": (
                    soft_on_wta_sum[behavior_position]
                    / wta_frame_counts[behavior_position]
                    if wta_frame_counts[behavior_position] > 0.0
                    else float("nan")
                ),
                "mean_soft_prob_on_nonWTA_frames": (
                    soft_on_nonwta_sum[behavior_position]
                    / nonwta_frame_counts[behavior_position]
                    if nonwta_frame_counts[behavior_position] > 0.0
                    else float("nan")
                ),
                "median_soft_prob_on_nonWTA_frames": (
                    float(np.median(nonwta_values)) if nonwta_values.size else float("nan")
                ),
                "p95_soft_prob_on_nonWTA_frames": (
                    float(np.quantile(nonwta_values, 0.95))
                    if nonwta_values.size
                    else float("nan")
                ),
                "WTA_phase_CT": wta_phase,
                "SOFT_phase_CT": soft_phase,
                "phase_difference_signed_h": signed_phase_difference,
                "abs_phase_difference_h": abs(signed_phase_difference),
                "WTA_R": wta_resultant,
                "SOFT_R": soft_resultant,
                "delta_R": soft_resultant - wta_resultant,
                "W1_WTA_vs_SOFT_h": w1_wta_soft,
                "WTA_LS_period_h": wta_periods[NONREST_INDICES[behavior_position]],
                "SOFT_LS_period_h": soft_periods[NONREST_INDICES[behavior_position]],
                "period_difference_h": (
                    soft_periods[NONREST_INDICES[behavior_position]]
                    - wta_periods[NONREST_INDICES[behavior_position]]
                    if np.isfinite(wta_periods[NONREST_INDICES[behavior_position]])
                    and np.isfinite(soft_periods[NONREST_INDICES[behavior_position]])
                    else float("nan")
                ),
                "WTA_mean_LOCO_W1_h": representation_loco["WTA"][1],
                "SOFT_mean_LOCO_W1_h": representation_loco["SOFT"][1],
                "delta_mean_LOCO_W1_h": (
                    representation_loco["SOFT"][1]
                    - representation_loco["WTA"][1]
                    if np.isfinite(representation_loco["WTA"][1])
                    and np.isfinite(representation_loco["SOFT"][1])
                    else float("nan")
                ),
                "WTA_max_LOCO_W1_h": representation_loco["WTA"][2],
                "SOFT_max_LOCO_W1_h": representation_loco["SOFT"][2],
            }
        )
    result = {
        "summary_rows": summary_rows,
        "phase_rows": phase_rows,
        "loco_rows": loco_rows,
        "row_sum_stats": row_sum_stats,
        "wta_mismatch_count": wta_mismatch_count,
        "wta_periods": wta_periods,
        "soft_periods": soft_periods,
        "cycle_indices": cycle_indices,
        "file_times": file_times_array,
        "invalid_rows": 0,
    }
    return phase_rows, loco_rows, result


def _select_polar_examples(summary_frame: pd.DataFrame) -> list[dict[str, str]]:
    """Select abundant/intermediate/sparse examples without group information."""

    median_support = summary_frame.groupby("Behavior")["WTA_duration_min"].median()
    abundant = "grooming"
    sparse_candidates = ["rearing", "locomotion"]
    sparse = min(sparse_candidates, key=lambda behavior: median_support[behavior])
    intermediate_candidates = [
        behavior
        for behavior in NONREST_BEHAVIORS
        if behavior not in {abundant, sparse}
    ]
    target = float(np.median(median_support.loc[intermediate_candidates]))
    intermediate = min(
        intermediate_candidates,
        key=lambda behavior: abs(float(median_support[behavior]) - target),
    )
    selected_behaviors = [abundant, intermediate, sparse]
    examples: list[dict[str, str]] = []
    for behavior in selected_behaviors:
        rows = summary_frame.loc[summary_frame["Behavior"] == behavior].copy()
        median_w1 = float(rows["W1_WTA_vs_SOFT_h"].median())
        rows["distance_from_median"] = (
            rows["W1_WTA_vs_SOFT_h"] - median_w1
        ).abs()
        selected = rows.sort_values(
            ["distance_from_median", "Animal"],
            kind="stable",
        ).iloc[0]
        examples.append(
            {
                "Behavior": behavior,
                "Animal": str(selected["Animal"]),
                "Group": str(selected["Group"]),
            }
        )
    return examples


def _distribution_lookup(
    distribution_frame: pd.DataFrame,
    animal: str,
    behavior: str,
    representation: str,
) -> np.ndarray:
    rows = distribution_frame.loc[
        (distribution_frame["Animal"] == animal)
        & (distribution_frame["Behavior"] == behavior)
        & (distribution_frame["Representation"] == representation)
    ].sort_values("CT_bin")
    if len(rows) != N_CT_BINS:
        raise ValueError(
            f"Expected {N_CT_BINS} distribution rows for {animal} {behavior} "
            f"{representation}, found {len(rows)}"
        )
    return rows["Probability"].to_numpy(dtype=float)


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    *,
    diverging: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
    if diverging:
        limit = float(np.max(np.abs(matrix)))
        limit = max(limit, 1e-12)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
    else:
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            interpolation="nearest",
        )
    axis.set_title(title)
    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("Animal")
    axis.set_ylabel("Behavior")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_selected_polar_plots(
    output_dir: Path,
    distribution_frame: pd.DataFrame,
    examples: list[dict[str, str]],
    summary_frame: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(
        1,
        len(examples),
        figsize=(16, 5.5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    theta = 2.0 * np.pi * CT_BIN_CENTERS_HOURS / frozen_phase.CT_HOURS
    theta_closed = np.concatenate((theta, theta[:1]))
    for axis, example in zip(axes, examples):
        animal = example["Animal"]
        behavior = example["Behavior"]
        wta = _distribution_lookup(distribution_frame, animal, behavior, "WTA")
        soft = _distribution_lookup(distribution_frame, animal, behavior, "SOFT")
        wta_phase, wta_r = _circular_mean_and_resultant(wta)
        soft_phase, soft_r = _circular_mean_and_resultant(soft)
        axis.plot(theta_closed, np.r_[wta, wta[:1]], linewidth=1.5, label="WTA")
        axis.plot(theta_closed, np.r_[soft, soft[:1]], linewidth=1.5, label="Soft")
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_xticks(np.arange(4) * np.pi / 2.0)
        axis.set_xticklabels(["CT0", "CT6", "CT12", "CT18"])
        radial_limit = max(float(wta.max()), float(soft.max()), 1e-6)
        axis.set_ylim(0.0, radial_limit * 1.35)
        vector_radius = radial_limit * 1.15
        axis.plot(
            [2.0 * np.pi * wta_phase / 24.0] * 2,
            [0.0, vector_radius],
            linestyle="--",
            linewidth=1.0,
            label=f"WTA mean (R={wta_r:.2f})",
        )
        axis.plot(
            [2.0 * np.pi * soft_phase / 24.0] * 2,
            [0.0, vector_radius],
            linestyle=":",
            linewidth=1.2,
            label=f"Soft mean (R={soft_r:.2f})",
        )
        axis.set_title(f"{animal}: {behavior}")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.15, -0.08), fontsize=8)
    figure.suptitle(
        "Selected WTA versus soft phase distributions\n"
        "Animals selected by median WTA-soft W1 within each behavior"
    )
    path = output_dir / "wta_vs_soft_selected_polar_plots.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_selected_overlays(
    output_dir: Path,
    distribution_frame: pd.DataFrame,
    examples: list[dict[str, str]],
) -> Path:
    figure, axes = plt.subplots(
        len(examples),
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, example in zip(axes, examples):
        animal = example["Animal"]
        behavior = example["Behavior"]
        wta = _distribution_lookup(distribution_frame, animal, behavior, "WTA")
        soft = _distribution_lookup(distribution_frame, animal, behavior, "SOFT")
        axis.plot(CT_BIN_CENTERS_HOURS, wta, label="WTA", linewidth=1.2)
        axis.plot(CT_BIN_CENTERS_HOURS, soft, label="Soft", linewidth=1.2)
        axis.set_ylabel("P(CT | behavior)")
        axis.set_title(f"{animal}: {behavior}", loc="left")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Circadian time (CT hours)")
    axes[-1].set_xlim(0.0, frozen_phase.CT_HOURS)
    axes[-1].set_xticks([0.0, 6.0, 12.0, 18.0, 24.0])
    axes[0].legend(loc="upper right")
    figure.suptitle("Selected WTA versus soft phase-distribution overlays")
    path = output_dir / "wta_vs_soft_selected_distribution_overlays.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_period_comparison(
    output_dir: Path,
    summary_frame: pd.DataFrame,
) -> Path | None:
    if not np.isfinite(summary_frame[["WTA_LS_period_h", "SOFT_LS_period_h"]].to_numpy()).all():
        return None
    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    x = summary_frame["WTA_LS_period_h"].to_numpy(dtype=float)
    y = summary_frame["SOFT_LS_period_h"].to_numpy(dtype=float)
    axis.scatter(x, y, color="tab:blue", alpha=0.75, s=28)
    axis.plot([20.0, 28.0], [20.0, 28.0], color="black", linestyle="--", linewidth=1.0)
    axis.set_xlim(20.0, 28.0)
    axis.set_ylim(20.0, 28.0)
    axis.set_xlabel("WTA Lomb--Scargle period (h)")
    axis.set_ylabel("Soft Lomb--Scargle period (h)")
    axis.set_title("WTA versus soft behavior-period estimates")
    axis.grid(alpha=0.25)
    path = output_dir / "wta_vs_soft_period_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_audit(cohort_root: Path) -> dict[str, object]:
    w1_checks = run_circular_wasserstein_checks()
    mean_wrap_check = _run_circular_mean_check()
    all_summary_rows: list[dict[str, object]] = []
    all_phase_rows: list[dict[str, object]] = []
    all_loco_rows: list[dict[str, object]] = []
    row_sum_stats_by_animal: dict[str, dict[str, float]] = {}
    mismatch_by_animal: dict[str, int] = {}
    for group, animal in ANIMALS:
        phase_rows, loco_rows, result = _audit_animal(
            group,
            animal,
            cohort_root,
        )
        all_summary_rows.extend(result["summary_rows"])
        all_phase_rows.extend(phase_rows)
        all_loco_rows.extend(loco_rows)
        row_sum_stats_by_animal[animal] = result["row_sum_stats"]
        mismatch_by_animal[animal] = result["wta_mismatch_count"]

    summary_frame = pd.DataFrame(all_summary_rows, columns=SUMMARY_COLUMNS)
    phase_frame = pd.DataFrame(
        all_phase_rows,
        columns=PHASE_DISTRIBUTION_COLUMNS,
    )
    loco_frame = pd.DataFrame(all_loco_rows, columns=LOCO_COLUMNS)
    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "wta_vs_soft_behavior_summary.csv"
    phase_csv = output_dir / "wta_vs_soft_phase_distributions.csv"
    loco_csv = output_dir / "wta_vs_soft_loco_stability.csv"
    summary_frame.to_csv(summary_csv, index=False)
    phase_frame.to_csv(phase_csv, index=False)
    loco_frame.to_csv(loco_csv, index=False)

    animal_order = [animal for _, animal in ANIMALS]
    behavior_order = list(NONREST_BEHAVIORS)
    phase_matrix = summary_frame.pivot(
        index="Behavior",
        columns="Animal",
        values="abs_phase_difference_h",
    ).loc[behavior_order, animal_order].to_numpy(dtype=float)
    w1_matrix = summary_frame.pivot(
        index="Behavior",
        columns="Animal",
        values="W1_WTA_vs_SOFT_h",
    ).loc[behavior_order, animal_order].to_numpy(dtype=float)
    delta_r_matrix = summary_frame.pivot(
        index="Behavior",
        columns="Animal",
        values="delta_R",
    ).loc[behavior_order, animal_order].to_numpy(dtype=float)
    phase_difference_figure = output_dir / "wta_vs_soft_phase_difference_heatmap.png"
    w1_figure = output_dir / "wta_vs_soft_distribution_distance_heatmap.png"
    delta_r_figure = output_dir / "wta_vs_soft_R_difference_heatmap.png"
    _save_heatmap(
        phase_matrix,
        behavior_order,
        animal_order,
        "WTA versus soft absolute circular phase difference",
        "Absolute phase difference (h)",
        phase_difference_figure,
    )
    _save_heatmap(
        w1_matrix,
        behavior_order,
        animal_order,
        "WTA versus soft phase-distribution distance",
        "Circular W1 (h)",
        w1_figure,
    )
    _save_heatmap(
        delta_r_matrix,
        behavior_order,
        animal_order,
        "Soft minus WTA resultant vector strength",
        "Delta R",
        delta_r_figure,
        diverging=True,
    )
    examples = _select_polar_examples(summary_frame)
    polar_figure = _save_selected_polar_plots(
        output_dir,
        phase_frame,
        examples,
        summary_frame,
    )
    overlay_figure = _save_selected_overlays(
        output_dir,
        phase_frame,
        examples,
    )
    period_figure = _save_period_comparison(output_dir, summary_frame)
    return {
        "output_dir": output_dir,
        "summary_csv": summary_csv,
        "phase_csv": phase_csv,
        "loco_csv": loco_csv,
        "summary_frame": summary_frame,
        "phase_frame": phase_frame,
        "loco_frame": loco_frame,
        "row_sum_stats_by_animal": row_sum_stats_by_animal,
        "mismatch_by_animal": mismatch_by_animal,
        "w1_checks": w1_checks,
        "mean_wrap_check": mean_wrap_check,
        "figures": [
            phase_difference_figure,
            w1_figure,
            delta_r_figure,
            polar_figure,
            overlay_figure,
            period_figure,
        ],
        "polar_examples": examples,
    }


def main() -> None:
    cohort_root = parse_args()
    result = run_audit(cohort_root)
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Phase CSV: {result['phase_csv']}")
    print(f"LOCO CSV: {result['loco_csv']}")
    print("Soft probability row-sum validation:")
    for animal, stats in result["row_sum_stats_by_animal"].items():
        mean = stats["sum"] / stats["count"] if stats["count"] else float("nan")
        print(
            f"  {animal}: mean={mean:.17g}, min={stats['min']:.17g}, "
            f"max={stats['max']:.17g}, max_abs_deviation="
            f"{stats['max_abs_deviation']:.3e}"
        )
    print(f"WTA/argmax mismatches: {result['mismatch_by_animal']}")
    print("Figures:")
    for figure in result["figures"]:
        if figure is not None:
            print(f"  {figure}")
    print("Group-blind polar examples:")
    for example in result["polar_examples"]:
        print(f"  {example['Animal']} / {example['Behavior']}")


if __name__ == "__main__":
    main()
