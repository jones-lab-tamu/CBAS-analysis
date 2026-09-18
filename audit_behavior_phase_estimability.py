"""Audit support and leave-one-cycle-out stability of behavior phase distributions.

This is a standalone estimability diagnostic.  It reuses the existing input
discovery, behavior classification, and frozen FRP/CT cycle boundaries without
changing the production FRP, MI, or recurrence analyses.
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
NONREST_BEHAVIORS = tuple(
    frozen_phase.BEHAVIORS[index]
    for index in frozen_phase.NONRESTING_BEHAVIOR_INDICES
)
N_CT_BINS = 288
CT_BIN_WIDTH_HOURS = frozen_phase.CT_HOURS / N_CT_BINS
OUTPUT_DIRNAME = "Behavior_Phase_Estimability_Audit"
WASSERSTEIN_TOLERANCE_HOURS = 1e-12
PROBABILITY_TOLERANCE = 1e-12

SUPPORT_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "status",
    "total_occupancy_seconds",
    "total_occupancy_minutes",
    "fraction_nonrest",
    "cycles_with_nonzero_occupancy",
    "cycle1_occupancy_minutes",
    "cycle2_occupancy_minutes",
    "cycle3_occupancy_minutes",
    "cycle4_occupancy_minutes",
    "pooled_occupied_CT_bins",
    "cycle1_occupied_CT_bins",
    "cycle2_occupied_CT_bins",
    "cycle3_occupied_CT_bins",
    "cycle4_occupied_CT_bins",
]

LOCO_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "omitted_cycle",
    "remaining_total_occupancy_minutes",
    "status",
    "circular_W1_full_vs_LOCO_h",
]

SUMMARY_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "status",
    "total_occupancy_minutes",
    "cycles_with_nonzero_occupancy",
    "mean_LOCO_W1_h",
    "median_LOCO_W1_h",
    "max_LOCO_W1_h",
    "min_LOCO_W1_h",
    "number_defined_LOCO",
    "number_undefined_LOCO",
]

DISTRIBUTION_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "distribution_type",
    "omitted_cycle",
    "CT_bin",
    "CT_start_h",
    "CT_end_h",
    "probability",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Audit behavior phase-distribution support and stability."
    )
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the eight animal folders.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def circular_wasserstein_1(
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    *,
    circumference_hours: float = frozen_phase.CT_HOURS,
) -> float:
    """Calculate exact discrete circular 1-Wasserstein distance in hours.

    The probability vectors are masses on equally spaced CT grid points.  For
    a cut around the circle, the edge flow is the cumulative mass difference.
    Minimizing the sum of absolute edge flows over the cut offset gives the
    median of those cumulative differences.  Multiplying by the grid spacing
    yields the minimum-cost transport on the circular grid, including transport
    across the CT0/24 boundary.
    """

    first = np.asarray(probability_a, dtype=float)
    second = np.asarray(probability_b, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError(
            "Circular W1 requires two one-dimensional probability vectors of "
            f"the same shape, got {first.shape} and {second.shape}"
        )
    if first.size == 0:
        raise ValueError("Circular W1 requires at least one CT bin")
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
    if not np.isfinite(distance) or distance < -WASSERSTEIN_TOLERANCE_HOURS:
        raise ValueError(f"Circular W1 was invalid: {distance}")
    return max(0.0, distance)


def _point_mass(ct_hours: float) -> np.ndarray:
    probability = np.zeros(N_CT_BINS, dtype=float)
    index = int(round(ct_hours / CT_BIN_WIDTH_HOURS)) % N_CT_BINS
    probability[index] = 1.0
    return probability


def run_circular_wasserstein_checks() -> dict[str, float]:
    """Run required synthetic checks before real-data processing."""

    identical = circular_wasserstein_1(_point_mass(5.0), _point_mass(5.0))
    ct23_vs_ct1 = circular_wasserstein_1(_point_mass(23.0), _point_mass(1.0))
    adjacent = circular_wasserstein_1(_point_mass(4.0), _point_mass(4.0 + CT_BIN_WIDTH_HOURS))
    symmetry_ab = circular_wasserstein_1(_point_mass(7.0), _point_mass(18.0))
    symmetry_ba = circular_wasserstein_1(_point_mass(18.0), _point_mass(7.0))
    ct0_vs_ct12 = circular_wasserstein_1(_point_mass(0.0), _point_mass(12.0))
    expected_adjacent = CT_BIN_WIDTH_HOURS

    checks = {
        "identical_h": identical,
        "ct23_vs_ct1_h": ct23_vs_ct1,
        "adjacent_5min_h": adjacent,
        "symmetry_difference_h": symmetry_ab - symmetry_ba,
        "ct0_vs_ct12_h": ct0_vs_ct12,
    }
    expected = {
        "identical_h": 0.0,
        "ct23_vs_ct1_h": 2.0,
        "adjacent_5min_h": expected_adjacent,
        "symmetry_difference_h": 0.0,
        "ct0_vs_ct12_h": 12.0,
    }
    for name, value in checks.items():
        if not np.isclose(
            value,
            expected[name],
            rtol=0.0,
            atol=WASSERSTEIN_TOLERANCE_HOURS,
        ):
            raise RuntimeError(
                f"Circular W1 synthetic check failed for {name}: "
                f"observed={value:.17g}, expected={expected[name]:.17g}"
            )
    print(
        "Circular W1 checks: "
        f"identical={identical:.12g} h; "
        f"CT23-vs-CT1={ct23_vs_ct1:.12g} h; "
        f"adjacent-5min={adjacent:.12g} h; "
        f"symmetry difference={checks['symmetry_difference_h']:.3e} h; "
        f"CT0-vs-CT12={ct0_vs_ct12:.12g} h"
    )
    return checks


def _load_classified_samples_with_weights(
    input_files: list[tuple[int, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read WTA labels, elapsed times, and classified source-time weights."""

    first_file_index = input_files[0][0]
    relative_time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    invalid_rows = 0
    for file_index, path in input_files:
        n_rows, row_indices, labels = frozen_phase.read_and_classify(path)
        invalid_rows += n_rows - len(labels)
        if n_rows == 0 or len(labels) == 0:
            continue
        file_start_hours = (
            file_index - first_file_index
        ) * frozen_phase.FILE_DURATION_MINUTES / 60.0
        relative_times = file_start_hours + (
            row_indices.astype(float)
            / n_rows
            * frozen_phase.FILE_DURATION_MINUTES
            / 60.0
        )
        classified_seconds = np.full(
            len(labels),
            frozen_phase.FILE_DURATION_MINUTES * 60.0 / n_rows,
            dtype=float,
        )
        relative_time_parts.append(relative_times)
        label_parts.append(labels.astype(np.int8, copy=False))
        weight_parts.append(classified_seconds)

    if not label_parts:
        raise ValueError("No valid behavioral samples were found")
    return (
        np.concatenate(relative_time_parts),
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
        invalid_rows,
    )


def _build_cycle_counts(
    relative_times: np.ndarray,
    labels: np.ndarray,
    classified_seconds: np.ndarray,
    complete_cycle_specs: list[tuple[int, float, float]],
    frp_hours: float,
) -> tuple[list[int], np.ndarray]:
    """Build per-cycle CT-bin behavior occupancy using frozen boundaries."""

    cycle_indices: list[int] = []
    cycle_counts: list[np.ndarray] = []
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, frp_hours)
    for cycle_index, start_boundary, end_boundary in complete_cycle_specs:
        selected = (relative_times >= start_boundary) & (
            relative_times < end_boundary
        )
        if not selected.any():
            raise ValueError(
                f"Full FRP cycle {cycle_index} contains no classifiable samples"
            )
        cycle_times = relative_times[selected] - start_boundary
        if np.any(cycle_times < -tolerance) or np.any(cycle_times > frp_hours + tolerance):
            raise ValueError(
                f"Samples assigned to full cycle {cycle_index} fall outside its FRP interval"
            )
        cycle_times = np.clip(cycle_times, 0.0, np.nextafter(frp_hours, 0.0))
        phase_bins = frozen_phase.ct_phase_bin_indices(
            cycle_times,
            N_CT_BINS,
            frp_hours=frp_hours,
        )
        counts = np.zeros(
            (N_CT_BINS, len(frozen_phase.BEHAVIORS)),
            dtype=float,
        )
        np.add.at(
            counts,
            (phase_bins, labels[selected]),
            classified_seconds[selected],
        )
        cycle_indices.append(cycle_index)
        cycle_counts.append(counts)

    return cycle_indices, np.stack(cycle_counts, axis=0)


def _validate_probability(probability: np.ndarray, description: str) -> None:
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite probability in {description}")
    if np.any(probability < -PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Negative probability in {description}")
    total = float(probability.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError(
            f"Probability did not sum to one in {description}: {total:.17g}"
        )


def _distribution_rows(
    group: str,
    animal: str,
    behavior: str,
    distribution_type: str,
    omitted_cycle: int | str,
    counts: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError(
            f"Cannot build {distribution_type} distribution with zero occupancy "
            f"for {animal} {behavior}"
        )
    probability = counts / total
    _validate_probability(
        probability,
        f"{animal} {behavior} {distribution_type} {omitted_cycle}",
    )
    rows = [
        {
            "Group": group,
            "Animal": animal,
            "Behavior": behavior,
            "distribution_type": distribution_type,
            "omitted_cycle": omitted_cycle,
            "CT_bin": bin_index,
            "CT_start_h": bin_index * CT_BIN_WIDTH_HOURS,
            "CT_end_h": (bin_index + 1) * CT_BIN_WIDTH_HOURS,
            "probability": float(probability[bin_index]),
        }
        for bin_index in range(N_CT_BINS)
    ]
    return rows, probability


def _audit_animal(
    group: str,
    animal: str,
    cohort_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    animal_dir = cohort_root / group / animal
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Missing animal directory: {animal_dir}")

    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(
            f"Source indices are missing for {animal}: {missing_indices}"
        )
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

    relative_times, labels, classified_seconds, invalid_rows = (
        _load_classified_samples_with_weights(input_files)
    )
    cycle_indices, all_cycle_counts = _build_cycle_counts(
        relative_times,
        labels,
        classified_seconds,
        complete_cycle_specs,
        computational_frp_hours,
    )
    if len(cycle_indices) != 4:
        raise RuntimeError(f"Unexpected cycle count for {animal}: {cycle_indices}")

    nonrest_counts = np.take(
        all_cycle_counts,
        frozen_phase.NONRESTING_BEHAVIOR_INDICES,
        axis=2,
    )
    pooled_counts = nonrest_counts.sum(axis=0)
    total_nonrest_seconds = float(pooled_counts.sum())
    if total_nonrest_seconds <= 0.0:
        raise ValueError(f"No non-rest occupancy was found for {animal}")

    support_rows: list[dict[str, object]] = []
    loco_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for behavior_index, behavior in enumerate(NONREST_BEHAVIORS):
        behavior_by_cycle = nonrest_counts[:, :, behavior_index]
        pooled_behavior = behavior_by_cycle.sum(axis=0)
        total_seconds = float(pooled_behavior.sum())
        cycle_seconds = behavior_by_cycle.sum(axis=1)
        cycle_minutes = cycle_seconds / 60.0
        cycle_nonzero_bins = np.count_nonzero(behavior_by_cycle > 0.0, axis=1)
        pooled_nonzero_bins = int(np.count_nonzero(pooled_behavior > 0.0))
        cycles_with_nonzero = int(np.count_nonzero(cycle_seconds > 0.0))
        support_status = "ZERO_OCCUPANCY" if total_seconds <= 0.0 else "NONZERO_OCCUPANCY"
        support_rows.append(
            {
                "Group": group,
                "Animal": animal,
                "Behavior": behavior,
                "status": support_status,
                "total_occupancy_seconds": total_seconds,
                "total_occupancy_minutes": total_seconds / 60.0,
                "fraction_nonrest": total_seconds / total_nonrest_seconds,
                "cycles_with_nonzero_occupancy": cycles_with_nonzero,
                "cycle1_occupancy_minutes": float(cycle_minutes[0]),
                "cycle2_occupancy_minutes": float(cycle_minutes[1]),
                "cycle3_occupancy_minutes": float(cycle_minutes[2]),
                "cycle4_occupancy_minutes": float(cycle_minutes[3]),
                "pooled_occupied_CT_bins": pooled_nonzero_bins,
                "cycle1_occupied_CT_bins": int(cycle_nonzero_bins[0]),
                "cycle2_occupied_CT_bins": int(cycle_nonzero_bins[1]),
                "cycle3_occupied_CT_bins": int(cycle_nonzero_bins[2]),
                "cycle4_occupied_CT_bins": int(cycle_nonzero_bins[3]),
            }
        )

        if total_seconds <= 0.0:
            for omitted_cycle in cycle_indices:
                loco_rows.append(
                    {
                        "Group": group,
                        "Animal": animal,
                        "Behavior": behavior,
                        "omitted_cycle": omitted_cycle,
                        "remaining_total_occupancy_minutes": 0.0,
                        "status": "UNDEFINED_ZERO_REMAINING_OCCUPANCY",
                        "circular_W1_full_vs_LOCO_h": np.nan,
                    }
                )
            summary_rows.append(
                {
                    "Group": group,
                    "Animal": animal,
                    "Behavior": behavior,
                    "status": "ZERO_OCCUPANCY",
                    "total_occupancy_minutes": 0.0,
                    "cycles_with_nonzero_occupancy": 0,
                    "mean_LOCO_W1_h": np.nan,
                    "median_LOCO_W1_h": np.nan,
                    "max_LOCO_W1_h": np.nan,
                    "min_LOCO_W1_h": np.nan,
                    "number_defined_LOCO": 0,
                    "number_undefined_LOCO": len(cycle_indices),
                }
            )
            continue

        full_rows, full_probability = _distribution_rows(
            group,
            animal,
            behavior,
            "FULL",
            "",
            pooled_behavior,
        )
        distribution_rows.extend(full_rows)

        loco_distances: list[float] = []
        undefined_loco = 0
        for omitted_position, omitted_cycle in enumerate(cycle_indices):
            remaining_counts = np.delete(
                behavior_by_cycle,
                omitted_position,
                axis=0,
            ).sum(axis=0)
            remaining_seconds = float(remaining_counts.sum())
            if remaining_seconds <= 0.0:
                undefined_loco += 1
                loco_rows.append(
                    {
                        "Group": group,
                        "Animal": animal,
                        "Behavior": behavior,
                        "omitted_cycle": omitted_cycle,
                        "remaining_total_occupancy_minutes": 0.0,
                        "status": "UNDEFINED_ZERO_REMAINING_OCCUPANCY",
                        "circular_W1_full_vs_LOCO_h": np.nan,
                    }
                )
                continue

            loco_rows_for_distribution, loco_probability = _distribution_rows(
                group,
                animal,
                behavior,
                "LOCO",
                omitted_cycle,
                remaining_counts,
            )
            distribution_rows.extend(loco_rows_for_distribution)
            distance = circular_wasserstein_1(full_probability, loco_probability)
            if not np.isfinite(distance) or distance < 0.0:
                raise RuntimeError(
                    f"Invalid circular W1 for {animal} {behavior}, "
                    f"omitted cycle {omitted_cycle}: {distance}"
                )
            loco_distances.append(distance)
            loco_rows.append(
                {
                    "Group": group,
                    "Animal": animal,
                    "Behavior": behavior,
                    "omitted_cycle": omitted_cycle,
                    "remaining_total_occupancy_minutes": remaining_seconds / 60.0,
                    "status": "DEFINED",
                    "circular_W1_full_vs_LOCO_h": distance,
                }
            )

        defined_count = len(loco_distances)
        if defined_count == len(cycle_indices):
            stability_status = "DEFINED_ALL_LOCO"
        else:
            stability_status = "DEFINED_PARTIAL_LOCO"
        summary_rows.append(
            {
                "Group": group,
                "Animal": animal,
                "Behavior": behavior,
                "status": stability_status,
                "total_occupancy_minutes": total_seconds / 60.0,
                "cycles_with_nonzero_occupancy": cycles_with_nonzero,
                "mean_LOCO_W1_h": float(np.mean(loco_distances))
                if loco_distances
                else np.nan,
                "median_LOCO_W1_h": float(np.median(loco_distances))
                if loco_distances
                else np.nan,
                "max_LOCO_W1_h": float(np.max(loco_distances))
                if loco_distances
                else np.nan,
                "min_LOCO_W1_h": float(np.min(loco_distances))
                if loco_distances
                else np.nan,
                "number_defined_LOCO": defined_count,
                "number_undefined_LOCO": undefined_loco,
            }
        )

    if not np.isclose(
        sum(row["fraction_nonrest"] for row in support_rows),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError(f"Non-rest support fractions did not sum to one for {animal}")
    print(
        f"{animal}: cycles={cycle_indices}, invalid rows excluded={invalid_rows}, "
        f"non-rest occupancy={total_nonrest_seconds / 60.0:.6f} minutes"
    )
    return support_rows, loco_rows, distribution_rows, {
        "summary_rows": summary_rows,
        "invalid_rows": invalid_rows,
    }


def _save_figure(
    output_dir: Path,
    support_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> Path:
    animal_order = [animal for _, animal in ANIMALS]
    behavior_order = list(NONREST_BEHAVIORS)
    support_pivot = support_frame.pivot(
        index="Animal",
        columns="Behavior",
        values="total_occupancy_minutes",
    ).loc[animal_order, behavior_order]
    cycles_pivot = support_frame.pivot(
        index="Animal",
        columns="Behavior",
        values="cycles_with_nonzero_occupancy",
    ).loc[animal_order, behavior_order]
    stability_pivot = summary_frame.pivot(
        index="Animal",
        columns="Behavior",
        values="max_LOCO_W1_h",
    ).loc[animal_order, behavior_order]

    occupancy_minutes = support_pivot.to_numpy(dtype=float)
    cycles_with_nonzero = cycles_pivot.to_numpy(dtype=float)
    max_loco_w1 = stability_pivot.to_numpy(dtype=float)
    display_epsilon_minutes = 1e-6
    log_occupancy = np.log10(occupancy_minutes + display_epsilon_minutes)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7), constrained_layout=True)
    image_a = axes[0].imshow(log_occupancy, aspect="auto", cmap="viridis")
    axes[0].set_title("A. log10(total occupancy minutes)")
    colorbar_a = fig.colorbar(image_a, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar_a.set_label("log10(minutes)")

    masked_w1 = np.ma.masked_invalid(max_loco_w1)
    finite_w1 = max_loco_w1[np.isfinite(max_loco_w1)]
    w1_max = float(finite_w1.max()) if finite_w1.size else 1.0
    if w1_max <= 0.0:
        w1_max = 1.0
    image_b = axes[1].imshow(
        masked_w1,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=w1_max,
    )
    axes[1].set_title("B. max LOCO circular W1")
    colorbar_b = fig.colorbar(image_b, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar_b.set_label("hours")

    image_c = axes[2].imshow(
        cycles_with_nonzero,
        aspect="auto",
        cmap="YlGn",
        vmin=0.0,
        vmax=4.0,
    )
    axes[2].set_title("C. cycles with nonzero occupancy")
    colorbar_c = fig.colorbar(image_c, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar_c.set_label("number of cycles")
    for row_index in range(cycles_with_nonzero.shape[0]):
        for column_index in range(cycles_with_nonzero.shape[1]):
            axes[2].text(
                column_index,
                row_index,
                str(int(cycles_with_nonzero[row_index, column_index])),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    for axis in axes:
        axis.set_xticks(np.arange(len(behavior_order)))
        axis.set_xticklabels(behavior_order, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(animal_order)))
        axis.set_yticklabels(animal_order)
        axis.set_xlabel("Behavior")
        axis.set_ylabel("Animal")
    fig.suptitle(
        "Behavior phase estimability audit: support and leave-one-cycle-out stability"
    )
    path = output_dir / "behavior_phase_estimability_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_audit(cohort_root: Path) -> dict[str, object]:
    checks = run_circular_wasserstein_checks()
    support_rows: list[dict[str, object]] = []
    loco_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for group, animal in ANIMALS:
        animal_support, animal_loco, animal_distributions, extra = _audit_animal(
            group,
            animal,
            cohort_root,
        )
        support_rows.extend(animal_support)
        loco_rows.extend(animal_loco)
        distribution_rows.extend(animal_distributions)
        summary_rows.extend(extra["summary_rows"])

    support_frame = pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS)
    loco_frame = pd.DataFrame(loco_rows, columns=LOCO_COLUMNS)
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    distribution_frame = pd.DataFrame(
        distribution_rows,
        columns=DISTRIBUTION_COLUMNS,
    )
    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    support_csv = output_dir / "behavior_phase_support_audit.csv"
    loco_csv = output_dir / "behavior_phase_loco_stability.csv"
    summary_csv = output_dir / "behavior_phase_stability_summary.csv"
    distribution_csv = output_dir / "behavior_phase_distributions.csv"
    support_frame.to_csv(support_csv, index=False)
    loco_frame.to_csv(loco_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    distribution_frame.to_csv(distribution_csv, index=False)
    figure_path = _save_figure(output_dir, support_frame, summary_frame)
    return {
        "checks": checks,
        "output_dir": output_dir,
        "support_csv": support_csv,
        "loco_csv": loco_csv,
        "summary_csv": summary_csv,
        "distribution_csv": distribution_csv,
        "figure": figure_path,
        "support_frame": support_frame,
        "loco_frame": loco_frame,
        "summary_frame": summary_frame,
        "distribution_frame": distribution_frame,
    }


def main() -> None:
    cohort_root = parse_args()
    result = run_audit(cohort_root)
    print(f"Support CSV: {result['support_csv']}")
    print(f"LOCO CSV: {result['loco_csv']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Distribution CSV: {result['distribution_csv']}")
    print(f"Figure: {result['figure']}")
    print(result["summary_frame"].to_string(index=False))


if __name__ == "__main__":
    main()
