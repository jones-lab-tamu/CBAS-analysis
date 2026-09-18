"""Audit existing conditional MI and occupancy-recurrence metrics on real data.

This is a diagnostic-only layer.  It reads the existing FRP/MI and occupancy
recurrence outputs, reuses their calculation helpers, and writes audit tables
under ``<cohort_root>/Metric_Audit_Output``.

Percentile conventions:

* MI observed percentile is the inclusive empirical CDF fraction
  ``mean(null_value <= observed_value)``.  Ties are counted in full.
* Recurrence zero-shift percentile is the inclusive empirical CDF fraction
  ``mean(all_alignment_affinity <= affinity_at_shift_0)``.  Ties are counted
  in full.
* Recurrence zero-shift rank is one-based minimum rank: ``1 +`` the number of
  all-alignment affinities strictly greater than the zero-shift affinity.
  Circular shifts use the existing convention ``np.roll(cycle_B, shift)``;
  best-shift minutes are the nonnegative shift-bin index multiplied by five.

The Critical C manipulation uses seed 20260916.  At each 5-minute CT bin it
permutes the complete nine-state behavior-count vector across the four cycles.
This preserves the pooled CT-by-behavior counts exactly while changing their
cycle-specific assignment.  No new null distribution is generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase
import phase_behavior_occupancy_recurrence as occupancy_integration
import phase_behavior_recurrence as recurrence
from phase_behavior_occupancy_recurrence_validation import (
    _affinity_on_support,
    score_pair,
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

MI_AUDIT_COLUMNS = [
    "Group",
    "Animal",
    "MI8_observed_bits",
    "MI8_null_mean_bits",
    "MI8_excess_bits",
    "MI8_null_sd",
    "MI8_z",
    "MI8_observed_percentile",
]

RECURRENCE_PAIR_COLUMNS = [
    "Group",
    "Animal",
    "cycle_A",
    "cycle_B",
    "A_obs",
    "mu_shift",
    "R_pair",
    "zero_shift_rank",
    "zero_shift_percentile",
    "best_shift_bins",
    "best_shift_minutes",
    "best_shift_affinity",
]

RECURRENCE_SUMMARY_COLUMNS = [
    "Group",
    "Animal",
    "mean_R_pair",
    "median_zero_shift_percentile",
    "zero_shift_ranks",
    "best_shift_minutes",
]

CRITICAL_C_COLUMNS = [
    "Animal",
    "condition",
    "MI8_observed_bits",
    "MI8_excess_bits",
    "Recurrence5_mean_R_pair",
    "pooled_distribution_max_abs_error",
]

PRIMARY_MI_PHASE_BINS = frozen_phase.PRIMARY_PHASE_BINS
RECURRENCE_BIN_MINUTES = 5
RECURRENCE_N_BINS = 288
CRITICAL_C_SEED = 20260916
NUMERICAL_TOLERANCE = 1e-12


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Audit existing conditional MI and occupancy recurrence outputs."
    )
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the animal folders.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def _require_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite: {result}")
    return result


def _read_mi_audit(animal_dir: Path, group: str, animal: str) -> dict[str, object]:
    mi_dir = animal_dir / "MI_Output"
    results = pd.read_csv(mi_dir / "mi_results.csv")
    conditional = results.loc[
        results["component"].eq("conditional_8state_nonrest")
    ]
    if len(conditional) != 1:
        raise ValueError(
            f"Expected one conditional MI row for {animal}, found {len(conditional)}"
        )
    result = conditional.iloc[0]

    null_frame = pd.read_csv(mi_dir / "mi_null_distribution.csv")
    null_values = null_frame["MI_conditional_8state_nonrest"].to_numpy(dtype=float)
    if null_values.size == 0 or not np.all(np.isfinite(null_values)):
        raise ValueError(f"Conditional MI null values are empty or non-finite for {animal}")

    observed = _require_finite(result["MI_raw_bits"], f"{animal} observed MI")
    null_mean = _require_finite(result["MI_null_mean_bits"], f"{animal} MI null mean")
    null_sd = _require_finite(result["MI_null_SD_bits"], f"{animal} MI null SD")
    excess = _require_finite(result["MI_excess_bits"], f"{animal} MI excess")
    z_score = _require_finite(result["MI_z"], f"{animal} MI z-score")
    if not np.isclose(excess, observed - null_mean, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"Conditional MI excess does not match observed minus null mean for {animal}")

    summary = pd.read_csv(mi_dir / "animal_mi_summary.csv")
    if len(summary) != 1:
        raise ValueError(f"Expected one MI summary row for {animal}")
    summary_excess = _require_finite(
        summary.loc[0, "MI_conditional8_excess_bits"],
        f"{animal} conditional MI summary excess",
    )
    if not np.isclose(excess, summary_excess, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"MI audit excess disagrees with existing animal summary for {animal}")

    observed_percentile = float(np.mean(null_values <= observed))
    if not np.isfinite(observed_percentile):
        raise ValueError(f"MI observed percentile is not finite for {animal}")

    return {
        "Group": group,
        "Animal": animal,
        "MI8_observed_bits": observed,
        "MI8_null_mean_bits": null_mean,
        "MI8_excess_bits": excess,
        "MI8_null_sd": null_sd,
        "MI8_z": z_score,
        "MI8_observed_percentile": observed_percentile,
        "null_values": null_values,
    }


def _load_recurrence_inputs(
    animal_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    float,
    list[tuple[int, float, float]],
]:
    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(f"Unexpected missing source indices for {animal_dir.name}: {missing_indices}")
    (
        _reported_frp,
        computational_frp,
        _start_ct,
        complete_cycle_specs,
        _frp_output_dir,
    ) = frozen_phase.load_frp_phase_solution(animal_dir)
    relative_times, labels, expected_samples_per_hour, invalid_rows = (
        occupancy_integration._load_classified_samples(input_files)
    )
    complete_cycles = occupancy_integration._build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp,
    )
    cycle_indices = [cycle_index for cycle_index, _, _ in complete_cycles]
    if len(cycle_indices) < 2:
        raise ValueError(f"At least two complete cycles are required for {animal_dir.name}")
    if invalid_rows:
        raise ValueError(f"Unexpected invalid source rows for {animal_dir.name}: {invalid_rows}")
    occupancy, valid_masks, _valid_counts = occupancy_integration._occupancy_for_resolution(
        relative_times,
        labels,
        complete_cycles,
        complete_cycle_specs[0][1],
        expected_samples_per_hour,
        computational_frp,
        RECURRENCE_N_BINS,
    )
    return (
        relative_times,
        labels,
        occupancy,
        valid_masks,
        cycle_indices,
        computational_frp,
        complete_cycle_specs,
    )


def _cycle_behavior_counts_from_integration(
    relative_times: np.ndarray,
    labels: np.ndarray,
    first_cycle_start: float,
    n_cycles: int,
    frp_hours: float,
    n_bins: int,
) -> np.ndarray:
    """Reproduce the integration module's exact binning operation order."""

    shifted_times = relative_times - first_cycle_start
    cycle_indices = np.floor(shifted_times / frp_hours).astype(np.int64)
    phase_fraction = np.mod(shifted_times / frp_hours, 1.0)
    normalized_phase_hours = 24.0 * phase_fraction
    phase_bins = np.floor(
        normalized_phase_hours / 24.0 * n_bins
    ).astype(np.int64)
    in_complete_cycles = (
        (cycle_indices >= 0)
        & (cycle_indices < n_cycles)
        & (phase_bins >= 0)
        & (phase_bins < n_bins)
    )
    counts = np.zeros(
        (n_cycles, n_bins, len(frozen_phase.BEHAVIORS)),
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
    return counts


def _cycle_behavior_counts_for_mi(
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    frp_hours: float,
    n_bins: int,
) -> np.ndarray:
    """Build cycle-specific counts with the existing MI binning operation."""

    counts = []
    for _cycle_index, cycle_times, labels in complete_cycles:
        phase_bins = frozen_phase.ct_phase_bin_indices(
            cycle_times,
            n_bins,
            frp_hours=frp_hours,
        )
        counts.append(frozen_phase.contingency_table(phase_bins, labels, n_bins))
    return np.stack(counts, axis=0)


def _occupancy_from_full_counts(
    full_counts: np.ndarray,
    expected_samples_per_bin: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_counts = full_counts.sum(axis=2)
    nonrest_counts = np.take(
        full_counts,
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
    valid_masks = (
        valid_counts.astype(float) / expected_samples_per_bin
        >= recurrence.MIN_VALID_COVERAGE
    )
    return occupancy, valid_masks


def _alignment_audit(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    if tensor_a.shape != tensor_b.shape or tensor_a.shape[0] != RECURRENCE_N_BINS:
        raise ValueError("Expected two 288-bin recurrence tensors with matching shapes")

    affinities = np.empty(RECURRENCE_N_BINS, dtype=float)
    for shift in range(RECURRENCE_N_BINS):
        shifted_b = np.roll(tensor_b, shift, axis=0)
        shifted_mask_b = np.roll(valid_b, shift)
        support = valid_a & shifted_mask_b
        if not np.any(support):
            raise ValueError(f"Circular shift {shift} has no common usable bins")
        affinities[shift], _, _ = _affinity_on_support(
            tensor_a,
            shifted_b,
            support,
        )

    existing = score_pair(tensor_a, tensor_b, valid_a, valid_b)
    if existing["status"] != "PASS":
        raise ValueError(
            f"Existing recurrence scorer returned {existing['status']}: "
            f"{existing['diagnostic']}"
        )
    if not np.isclose(
        affinities[0],
        float(existing["A_obs"]),
        atol=NUMERICAL_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("Audit shift 0 affinity disagrees with existing A_obs")
    if not np.isclose(
        affinities[1:].mean(),
        float(existing["mu_shift"]),
        atol=NUMERICAL_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("Audit nonzero-shift mean disagrees with existing mu_shift")

    zero_affinity = affinities[0]
    zero_shift_rank = int(1 + np.count_nonzero(affinities > zero_affinity))
    zero_shift_percentile = float(np.mean(affinities <= zero_affinity))
    best_shift_bins = int(np.argmax(affinities))
    result = {
        "A_obs": float(existing["A_obs"]),
        "mu_shift": float(existing["mu_shift"]),
        "R_pair": float(existing["R_pair"]),
        "zero_shift_rank": zero_shift_rank,
        "zero_shift_percentile": zero_shift_percentile,
        "best_shift_bins": best_shift_bins,
        "best_shift_minutes": best_shift_bins * RECURRENCE_BIN_MINUTES,
        "best_shift_affinity": float(affinities[best_shift_bins]),
        "denominator": float(existing["denominator_1_minus_mu_shift"]),
    }
    for name in (
        "A_obs",
        "mu_shift",
        "R_pair",
        "zero_shift_percentile",
        "best_shift_affinity",
        "denominator",
    ):
        if not np.isfinite(float(result[name])):
            raise ValueError(f"Recurrence audit value is non-finite: {name}")
    if abs(result["denominator"]) <= recurrence.NORMALIZATION_TOLERANCE:
        raise ValueError("Existing recurrence denominator is unstable")
    return result, affinities


def _read_existing_pairwise(
    animal_dir: Path,
    cycle_a: int,
    cycle_b: int,
) -> pd.Series:
    path = animal_dir / "Occupancy_Recurrence_Output" / "occupancy_recurrence_pairwise.csv"
    frame = pd.read_csv(path)
    rows = frame.loc[
        frame["resolution_minutes"].eq(5)
        & frame["cycle_A"].eq(cycle_a)
        & frame["cycle_B"].eq(cycle_b)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one existing 5-minute pairwise row for {animal_dir.name} {cycle_a}-{cycle_b}")
    return rows.iloc[0]


def _audit_recurrence_animal(
    group: str,
    animal: str,
    animal_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    (
        relative_times,
        labels,
        occupancy,
        valid_masks,
        cycle_indices,
        computational_frp,
        complete_cycle_specs,
    ) = _load_recurrence_inputs(animal_dir)
    if cycle_indices != [spec[0] for spec in complete_cycle_specs]:
        raise ValueError(f"Cycle indexing mismatch for {animal}")

    pair_rows: list[dict[str, object]] = []
    pair_audits: list[dict[str, object]] = []
    for pair_index in range(len(cycle_indices) - 1):
        cycle_a = cycle_indices[pair_index]
        cycle_b = cycle_indices[pair_index + 1]
        audit, _affinities = _alignment_audit(
            occupancy[pair_index],
            occupancy[pair_index + 1],
            valid_masks[pair_index],
            valid_masks[pair_index + 1],
        )
        existing = _read_existing_pairwise(animal_dir, cycle_a, cycle_b)
        for field in ("A_obs", "mu_shift", "R_pair"):
            if not np.isclose(
                float(existing[field]),
                float(audit[field]),
                atol=NUMERICAL_TOLERANCE,
                rtol=0.0,
            ):
                raise ValueError(f"Existing pairwise {field} disagrees for {animal} {cycle_a}-{cycle_b}")
        pair_audits.append(audit)
        pair_rows.append(
            {
                "Group": group,
                "Animal": animal,
                "cycle_A": cycle_a,
                "cycle_B": cycle_b,
                "A_obs": audit["A_obs"],
                "mu_shift": audit["mu_shift"],
                "R_pair": audit["R_pair"],
                "zero_shift_rank": audit["zero_shift_rank"],
                "zero_shift_percentile": audit["zero_shift_percentile"],
                "best_shift_bins": audit["best_shift_bins"],
                "best_shift_minutes": audit["best_shift_minutes"],
                "best_shift_affinity": audit["best_shift_affinity"],
            }
        )

    summary_path = animal_dir / "Occupancy_Recurrence_Output" / "occupancy_recurrence_summary.csv"
    summary = pd.read_csv(summary_path)
    summary5 = summary.loc[summary["resolution_minutes"].eq(5)]
    if len(summary5) != 1:
        raise ValueError(f"Expected one existing 5-minute recurrence summary row for {animal}")
    summary5 = summary5.iloc[0]
    mean_r_pair = float(np.mean([audit["R_pair"] for audit in pair_audits]))
    if not np.isclose(mean_r_pair, float(summary5["mean_R_pair"]), atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"Existing recurrence summary disagrees for {animal}")

    summary_row = {
        "Group": group,
        "Animal": animal,
        "mean_R_pair": mean_r_pair,
        "median_zero_shift_percentile": float(
            np.median([audit["zero_shift_percentile"] for audit in pair_audits])
        ),
        "zero_shift_ranks": ";".join(
            f"{row['cycle_A']}-{row['cycle_B']}={row['zero_shift_rank']}"
            for row in pair_rows
        ),
        "best_shift_minutes": ";".join(
            f"{row['cycle_A']}-{row['cycle_B']}={row['best_shift_minutes']}"
            for row in pair_rows
        ),
    }

    complete_cycles = []
    input_files, _missing = frozen_phase.discover_input_files(animal_dir)
    relative_times, labels, _expected_samples_per_hour, _invalid_rows = (
        occupancy_integration._load_classified_samples(input_files)
    )
    complete_cycles = occupancy_integration._build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp,
    )
    cycle_counts = _cycle_behavior_counts_from_integration(
        relative_times,
        labels,
        complete_cycle_specs[0][1],
        len(cycle_indices),
        computational_frp,
        RECURRENCE_N_BINS,
    )
    expected_samples_per_bin = (
        _expected_samples_per_hour * computational_frp / RECURRENCE_N_BINS
    )
    reconstructed_occupancy, reconstructed_masks = _occupancy_from_full_counts(
        cycle_counts,
        expected_samples_per_bin,
    )
    if not np.allclose(
        reconstructed_occupancy,
        occupancy,
        atol=NUMERICAL_TOLERANCE,
        rtol=0.0,
    ) or not np.array_equal(reconstructed_masks, valid_masks):
        raise ValueError(f"Critical C occupancy reconstruction disagreed with existing integration for {animal}")

    details = {
        "occupancy": occupancy,
        "valid_masks": valid_masks,
        "cycle_counts": cycle_counts,
        "expected_samples_per_bin": expected_samples_per_bin,
        "complete_cycles": complete_cycles,
        "computational_frp": computational_frp,
        "mi_audit": None,
        "pair_audits": pair_audits,
    }
    return pair_rows, summary_row, details


def _aggregate_fine_counts(fine_counts: np.ndarray, n_phase_bins: int) -> np.ndarray:
    n_cycles, n_bins, n_states = fine_counts.shape
    if n_bins % n_phase_bins != 0:
        raise ValueError("Fine CT bins are not divisible by the MI phase-bin count")
    bins_per_phase_bin = n_bins // n_phase_bins
    return fine_counts.reshape(
        n_cycles,
        n_phase_bins,
        bins_per_phase_bin,
        n_states,
    ).sum(axis=2).sum(axis=0)


def _critical_c(
    animal: str,
    mi_audit: dict[str, object],
    recurrence_details: dict[str, object],
) -> list[dict[str, object]]:
    if animal != "675G":
        raise ValueError("Critical C is specified for 675G")

    mi_cycle_counts = _cycle_behavior_counts_for_mi(
        recurrence_details["complete_cycles"],
        float(recurrence_details["computational_frp"]),
        RECURRENCE_N_BINS,
    )
    rng = np.random.default_rng(CRITICAL_C_SEED)
    reassigned_mi_counts = np.empty_like(mi_cycle_counts)
    original_occupancy = np.asarray(recurrence_details["occupancy"], dtype=float)
    original_masks = np.asarray(recurrence_details["valid_masks"], dtype=bool)
    reassigned_occupancy = np.empty_like(original_occupancy)
    reassigned_masks = np.empty_like(original_masks)
    for bin_index in range(mi_cycle_counts.shape[1]):
        source_cycles = rng.permutation(mi_cycle_counts.shape[0])
        reassigned_mi_counts[:, bin_index, :] = mi_cycle_counts[
            source_cycles,
            bin_index,
            :,
        ]
        reassigned_occupancy[:, bin_index, :] = original_occupancy[
            source_cycles,
            bin_index,
            :,
        ]
        reassigned_masks[:, bin_index] = original_masks[source_cycles, bin_index]

    pooled_before = mi_cycle_counts.sum(axis=0)
    pooled_after = reassigned_mi_counts.sum(axis=0)
    pooled_error = float(np.max(np.abs(pooled_before - pooled_after)))
    if pooled_error > NUMERICAL_TOLERANCE:
        raise ValueError(f"Critical C pooled distribution was not preserved: {pooled_error}")

    original_primary_counts = _aggregate_fine_counts(
        mi_cycle_counts,
        PRIMARY_MI_PHASE_BINS,
    )
    reassigned_primary_counts = _aggregate_fine_counts(
        reassigned_mi_counts,
        PRIMARY_MI_PHASE_BINS,
    )
    original_conditional_counts = frozen_phase.conditional_nonrest_counts(
        original_primary_counts
    )
    reassigned_conditional_counts = frozen_phase.conditional_nonrest_counts(
        reassigned_primary_counts
    )
    original_mi = frozen_phase.mutual_information_bits(original_conditional_counts)
    reassigned_mi = frozen_phase.mutual_information_bits(reassigned_conditional_counts)
    expected_original_mi = float(mi_audit["MI8_observed_bits"])
    if not np.isclose(original_mi, expected_original_mi, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError("Critical C original MI disagreed with the existing MI output")
    if not np.isclose(original_mi, reassigned_mi, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(
            "Critical C changed pooled conditional MI beyond tolerance: "
            f"{original_mi} vs {reassigned_mi}"
        )

    original_pair_results = [
        score_pair(
            original_occupancy[index],
            original_occupancy[index + 1],
            original_masks[index],
            original_masks[index + 1],
        )
        for index in range(original_occupancy.shape[0] - 1)
    ]
    reassigned_pair_results = [
        score_pair(
            reassigned_occupancy[index],
            reassigned_occupancy[index + 1],
            reassigned_masks[index],
            reassigned_masks[index + 1],
        )
        for index in range(reassigned_occupancy.shape[0] - 1)
    ]
    for condition, pair_results in (
        ("original", original_pair_results),
        ("cycle_reassigned", reassigned_pair_results),
    ):
        for pair in pair_results:
            if pair["status"] != "PASS":
                raise ValueError(
                    f"Critical C {condition} recurrence pair undefined: "
                    f"{pair['status']}: {pair['diagnostic']}"
                )
    original_r = float(np.mean([float(pair["R_pair"]) for pair in original_pair_results]))
    reassigned_r = float(np.mean([float(pair["R_pair"]) for pair in reassigned_pair_results]))
    null_mean = float(mi_audit["MI8_null_mean_bits"])
    return [
        {
            "Animal": animal,
            "condition": "original",
            "MI8_observed_bits": original_mi,
            "MI8_excess_bits": original_mi - null_mean,
            "Recurrence5_mean_R_pair": original_r,
            "pooled_distribution_max_abs_error": 0.0,
        },
        {
            "Animal": animal,
            "condition": "cycle_reassigned",
            "MI8_observed_bits": reassigned_mi,
            "MI8_excess_bits": reassigned_mi - null_mean,
            "Recurrence5_mean_R_pair": reassigned_r,
            "pooled_distribution_max_abs_error": pooled_error,
        },
    ]


def main() -> None:
    cohort_root = parse_args()
    output_dir = cohort_root / "Metric_Audit_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mi_rows: list[dict[str, object]] = []
    recurrence_pair_rows: list[dict[str, object]] = []
    recurrence_summary_rows: list[dict[str, object]] = []
    critical_c_rows: list[dict[str, object]] = []
    recurrence_details_by_animal: dict[str, dict[str, object]] = {}

    for group, animal in ANIMALS:
        animal_dir = cohort_root / group / animal
        mi_audit = _read_mi_audit(animal_dir, group, animal)
        mi_rows.append({column: mi_audit[column] for column in MI_AUDIT_COLUMNS})
        pair_rows, summary_row, recurrence_details = _audit_recurrence_animal(
            group,
            animal,
            animal_dir,
        )
        recurrence_pair_rows.extend(pair_rows)
        recurrence_summary_rows.append(summary_row)
        recurrence_details_by_animal[animal] = recurrence_details
        if animal == "675G":
            critical_c_rows.extend(_critical_c(animal, mi_audit, recurrence_details))

    pd.DataFrame(mi_rows, columns=MI_AUDIT_COLUMNS).to_csv(
        output_dir / "mi_metric_audit.csv",
        index=False,
    )
    pd.DataFrame(recurrence_pair_rows, columns=RECURRENCE_PAIR_COLUMNS).to_csv(
        output_dir / "recurrence_metric_audit_pairwise.csv",
        index=False,
    )
    pd.DataFrame(recurrence_summary_rows, columns=RECURRENCE_SUMMARY_COLUMNS).to_csv(
        output_dir / "recurrence_metric_audit_summary.csv",
        index=False,
    )
    pd.DataFrame(critical_c_rows, columns=CRITICAL_C_COLUMNS).to_csv(
        output_dir / "real_data_critical_c_audit.csv",
        index=False,
    )

    print(f"Wrote metric audit outputs to: {output_dir}")
    print("MI observed percentile convention: inclusive fraction of null values <= observed")
    print("Recurrence zero-shift rank convention: one-based minimum rank among all 288 alignments")
    print(f"Critical C animal: 675G; fixed seed: {CRITICAL_C_SEED}")
    print("Critical C pooled distribution max absolute error: 0")


if __name__ == "__main__":
    main()
