"""First-pass synthetic validation of phase-resolved occupancy recurrence."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from phase_behavior_recurrence import (
    BEHAVIORS,
    MIN_VALID_COVERAGE,
    NORMALIZATION_TOLERANCE,
)


NONREST_BEHAVIORS = tuple(behavior for behavior in BEHAVIORS if behavior != "resting")
N_BEHAVIORS = len(NONREST_BEHAVIORS)
NUMERICAL_TOLERANCE = max(NORMALIZATION_TOLERANCE, 1e-12)
RESOLUTIONS_MINUTES = (5, 10)
OUTPUT_PATH = Path(__file__).resolve().parent / "occupancy_recurrence_validation_summary.csv"

OUTPUT_COLUMNS = [
    "test_name",
    "resolution_minutes",
    "condition",
    "A_obs",
    "mu_shift",
    "R_pair",
    "n_common_bins",
    "fraction_common_bins",
    "total_nonrest_A",
    "total_nonrest_B",
    "denominator_1_minus_mu_shift",
    "status",
    "expected_behavior",
    "pass_fail",
]

EventResult = dict[str, object]


def valid_mask_from_coverage(coverage_fraction: Sequence[float]) -> np.ndarray:
    """Apply the existing recurrence coverage threshold without redefining it."""

    coverage = np.asarray(coverage_fraction, dtype=float)
    if coverage.ndim != 1 or not np.all(np.isfinite(coverage)):
        raise ValueError("Coverage must be a finite one-dimensional array")
    return coverage >= MIN_VALID_COVERAGE


def _validate_occupancy(tensor: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=float)
    if values.ndim != 2 or values.shape[1] != N_BEHAVIORS:
        raise ValueError(
            f"Occupancy tensor must have shape (n_bins, {N_BEHAVIORS})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Occupancy tensor contains a non-finite value")
    if np.any(values < -NUMERICAL_TOLERANCE) or np.any(values > 1.0 + NUMERICAL_TOLERANCE):
        raise ValueError("Occupancy tensor contains a value outside [0, 1]")
    row_totals = values.sum(axis=1)
    if np.any(row_totals < -NUMERICAL_TOLERANCE) or np.any(
        row_totals > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError("A bin's non-rest occupancy exceeds valid time")
    # Only remove numerical noise at zero; substantive values are untouched.
    return np.where(np.abs(values) <= NUMERICAL_TOLERANCE, 0.0, values)


def _sequence_to_occupancy(
    state_sequence: Sequence[str],
    sample_minutes: float,
    bin_minutes: int,
) -> np.ndarray:
    """Convert fixed-resolution synthetic WTA labels to non-rest occupancy."""

    states = np.asarray(state_sequence, dtype=object)
    if states.ndim != 1 or states.size == 0:
        raise ValueError("Synthetic WTA sequence must be a non-empty one-dimensional array")
    if not np.isfinite(sample_minutes) or sample_minutes <= 0:
        raise ValueError("Synthetic sample resolution must be positive and finite")
    if bin_minutes <= 0:
        raise ValueError("Synthetic CT-bin resolution must be positive")

    samples_per_bin_float = float(bin_minutes) / float(sample_minutes)
    samples_per_bin = int(round(samples_per_bin_float))
    if samples_per_bin < 1 or not np.isclose(
        samples_per_bin_float,
        samples_per_bin,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise ValueError("CT-bin resolution must contain an integral number of samples")
    if states.size % samples_per_bin != 0:
        raise ValueError("Synthetic WTA sequence does not fill complete CT bins")

    behavior_to_column = {
        behavior: index for index, behavior in enumerate(NONREST_BEHAVIORS)
    }
    unknown_states = sorted(
        {state for state in states.tolist() if state not in BEHAVIORS},
        key=str,
    )
    if unknown_states:
        raise ValueError(f"Unknown synthetic WTA states: {unknown_states}")

    chunks = states.reshape(-1, samples_per_bin)
    occupancy = np.zeros((chunks.shape[0], N_BEHAVIORS), dtype=float)
    for behavior, column in behavior_to_column.items():
        occupancy[:, column] = (
            np.count_nonzero(chunks == behavior, axis=1) / samples_per_bin
        )
    return _validate_occupancy(occupancy)


def _validate_mask(mask: Sequence[bool], n_bins: int, name: str) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape != (n_bins,):
        raise ValueError(f"{name} must have shape ({n_bins},)")
    return values


def _undefined_result(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    common: np.ndarray,
    status: str,
    diagnostic: str,
) -> EventResult:
    n_bins = tensor_a.shape[0]
    total_a = float(tensor_a[common].sum()) if np.any(common) else float("nan")
    total_b = float(tensor_b[common].sum()) if np.any(common) else float("nan")
    return {
        "A_obs": float("nan"),
        "mu_shift": float("nan"),
        "R_pair": float("nan"),
        "n_common_bins": int(common.sum()),
        "fraction_common_bins": float(common.sum() / n_bins),
        "total_nonrest_A": total_a,
        "total_nonrest_B": total_b,
        "denominator_1_minus_mu_shift": float("nan"),
        "status": status,
        "diagnostic": diagnostic,
    }


def _normalized_on_support(
    tensor: np.ndarray,
    support: np.ndarray,
    name: str,
) -> tuple[np.ndarray, float]:
    if not np.any(support):
        raise ValueError(f"{name} has no common usable bins")
    total = float(tensor[support].sum())
    if not np.isfinite(total) or total <= NUMERICAL_TOLERANCE:
        raise ValueError(f"{name} has zero total non-rest occupancy on common support")
    normalized = np.zeros_like(tensor, dtype=float)
    normalized[support] = tensor[support] / total
    normalized_sum = float(normalized.sum())
    if not np.isclose(
        normalized_sum,
        1.0,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"{name} normalization does not sum to one: {normalized_sum}")
    return normalized, total


def _affinity_on_support(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    support: np.ndarray,
) -> tuple[float, float, float]:
    normalized_a, total_a = _normalized_on_support(tensor_a, support, "Cycle A")
    normalized_b, total_b = _normalized_on_support(tensor_b, support, "Cycle B")
    affinity = float(np.sqrt(normalized_a * normalized_b).sum())
    if not np.isfinite(affinity):
        raise ValueError("Affinity is non-finite")
    if affinity < -NUMERICAL_TOLERANCE or affinity > 1.0 + NUMERICAL_TOLERANCE:
        raise ValueError(f"Affinity is outside [0, 1]: {affinity}")
    # This only handles a floating-point overshoot/undershoot at the boundary.
    affinity = min(1.0, max(0.0, affinity))
    return affinity, total_a, total_b


def score_pair(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    valid_a: Sequence[bool] | None = None,
    valid_b: Sequence[bool] | None = None,
) -> EventResult:
    """Score two phase-resolved non-rest occupancy tensors."""

    values_a = _validate_occupancy(tensor_a)
    values_b = _validate_occupancy(tensor_b)
    if values_a.shape != values_b.shape:
        raise ValueError("Cycle tensors must have the same shape")
    n_bins = values_a.shape[0]
    mask_a = _validate_mask(
        np.ones(n_bins, dtype=bool) if valid_a is None else valid_a,
        n_bins,
        "valid_a",
    )
    mask_b = _validate_mask(
        np.ones(n_bins, dtype=bool) if valid_b is None else valid_b,
        n_bins,
        "valid_b",
    )
    observed_support = mask_a & mask_b
    if not np.any(observed_support):
        return _undefined_result(
            values_a,
            values_b,
            observed_support,
            "UNDEFINED_NO_COMMON_SUPPORT",
            "No CT bins are usable in both cycles",
        )

    try:
        observed_affinity, total_a, total_b = _affinity_on_support(
            values_a,
            values_b,
            observed_support,
        )
    except ValueError as error:
        return _undefined_result(
            values_a,
            values_b,
            observed_support,
            "UNDEFINED_ZERO_NONREST_OCCUPANCY",
            str(error),
        )

    shifted_affinities: list[float] = []
    for shift in range(1, n_bins):
        shifted_b = np.roll(values_b, shift, axis=0)
        shifted_mask_b = np.roll(mask_b, shift)
        shifted_support = mask_a & shifted_mask_b
        if not np.any(shifted_support):
            return _undefined_result(
                values_a,
                values_b,
                observed_support,
                "UNDEFINED_NULL_NO_COMMON_SUPPORT",
                f"Nonzero circular shift {shift} has no common usable bins",
            )
        try:
            shifted_affinity, _, _ = _affinity_on_support(
                values_a,
                shifted_b,
                shifted_support,
            )
        except ValueError as error:
            return _undefined_result(
                values_a,
                values_b,
                observed_support,
                "UNDEFINED_NULL_ZERO_NONREST_OCCUPANCY",
                f"Nonzero circular shift {shift}: {error}",
            )
        shifted_affinities.append(shifted_affinity)

    mu_shift = float(np.mean(np.asarray(shifted_affinities, dtype=float)))
    denominator = 1.0 - mu_shift
    if not np.isfinite(denominator) or abs(denominator) <= NORMALIZATION_TOLERANCE:
        return _undefined_result(
            values_a,
            values_b,
            observed_support,
            "UNDEFINED_UNSTABLE_SHIFT_DENOMINATOR",
            f"1 - mu_shift is numerically too small: {denominator}",
        )
    corrected = (observed_affinity - mu_shift) / denominator
    if not np.isfinite(corrected):
        return _undefined_result(
            values_a,
            values_b,
            observed_support,
            "UNDEFINED_NONFINITE_CORRECTED_SCORE",
            "Corrected recurrence is non-finite",
        )
    return {
        "A_obs": observed_affinity,
        "mu_shift": mu_shift,
        "R_pair": float(corrected),
        "n_common_bins": int(observed_support.sum()),
        "fraction_common_bins": float(observed_support.sum() / n_bins),
        "total_nonrest_A": total_a,
        "total_nonrest_B": total_b,
        "denominator_1_minus_mu_shift": denominator,
        "status": "PASS",
        "diagnostic": "",
    }


def score_adjacent_cycles(
    cycles: Sequence[np.ndarray],
    valid_masks: Sequence[Sequence[bool]] | None = None,
) -> EventResult:
    """Return the mean corrected score across adjacent cycle pairs only."""

    if len(cycles) < 2:
        raise ValueError("At least two cycles are required")
    if valid_masks is None:
        masks = [None] * len(cycles)
    else:
        if len(valid_masks) != len(cycles):
            raise ValueError("One valid mask is required per cycle")
        masks = list(valid_masks)
    pair_results = [
        score_pair(cycles[index], cycles[index + 1], masks[index], masks[index + 1])
        for index in range(len(cycles) - 1)
    ]
    if any(result["status"] != "PASS" for result in pair_results):
        failed = next(result for result in pair_results if result["status"] != "PASS")
        return {
            **failed,
            "status": f"UNDEFINED_ADJACENT_PAIR_{failed['status']}",
            "diagnostic": f"Adjacent-cycle pair was undefined: {failed['diagnostic']}",
            "pair_results": pair_results,
        }
    numeric_keys = (
        "A_obs",
        "mu_shift",
        "R_pair",
        "fraction_common_bins",
        "total_nonrest_A",
        "total_nonrest_B",
        "denominator_1_minus_mu_shift",
    )
    result = {
        key: float(np.mean([float(pair[key]) for pair in pair_results]))
        for key in numeric_keys
    }
    result.update(
        {
            "n_common_bins": int(min(int(pair["n_common_bins"]) for pair in pair_results)),
            "status": "PASS",
            "diagnostic": "",
            "pair_results": pair_results,
        }
    )
    return result


def _phase_bin(n_bins: int, phase_hours: float) -> int:
    return int(np.floor((phase_hours % 24.0) / 24.0 * n_bins))


def _synthetic_pattern(n_bins: int) -> np.ndarray:
    """Make a nonuniform, intentionally asymmetric phase-by-behavior pattern."""

    phase = (np.arange(n_bins, dtype=float) + 0.5) * 24.0 / n_bins
    tensor = np.full((n_bins, N_BEHAVIORS), 0.002, dtype=float)
    features = (
        (0, 1.1, 0.32, 0.42),
        (1, 4.0, 0.24, 0.58),
        (2, 7.6, 0.28, 0.47),
        (3, 10.9, 0.18, 0.70),
        (4, 13.2, 0.34, 0.55),
        (5, 16.8, 0.23, 0.65),
        (6, 19.4, 0.30, 0.50),
        (7, 22.1, 0.20, 0.80),
        (0, 14.6, 0.12, 0.70),
        (6, 3.0, 0.09, 0.45),
        (3, 18.0, 0.11, 0.60),
        (1, 12.2, 0.08, 0.60),
    )
    for behavior_index, center, amplitude, width in features:
        distance = np.abs((phase - center + 12.0) % 24.0 - 12.0)
        tensor[:, behavior_index] += amplitude * np.exp(
            -0.5 * (distance / width) ** 2
        )
    return _validate_occupancy(tensor)


def _transfer_occupancy(
    tensor: np.ndarray,
    behavior_index: int,
    source_phase: float,
    destination_phase: float,
    amount: float,
) -> np.ndarray:
    result = tensor.copy()
    source = _phase_bin(tensor.shape[0], source_phase)
    destination = _phase_bin(tensor.shape[0], destination_phase)
    if result[source, behavior_index] <= amount:
        raise AssertionError("Synthetic transfer source has insufficient occupancy")
    result[source, behavior_index] -= amount
    result[destination, behavior_index] += amount
    return _validate_occupancy(result)


def _swap_selected_behavior_regions(tensor: np.ndarray) -> np.ndarray:
    result = tensor.copy()
    for phase in (1.1, 19.4):
        index = _phase_bin(tensor.shape[0], phase)
        result[index, 0], result[index, 6] = result[index, 6], result[index, 0]
    return _validate_occupancy(result)


def _critical_c_pair_patterns(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create paired, broad phase redistributions for the occupancy Critical C."""

    delta = np.zeros_like(tensor)
    # Each pair of windows straddles one of the broad main features in the
    # base pattern.  Pairwise transfers keep each behavior's total unchanged,
    # while the +/- construction makes the pooled four-cycle distribution
    # cancel exactly.
    feature_regions = (
        (0, 1.1, 0.32),
        (1, 4.0, 0.24),
        (2, 7.6, 0.28),
        (3, 10.9, 0.18),
        (4, 13.2, 0.34),
        (5, 16.8, 0.23),
        (6, 19.4, 0.30),
        (7, 22.1, 0.20),
    )
    bin_hours = 24.0 / tensor.shape[0]
    for behavior_index, center_phase, feature_width in feature_regions:
        window_bins = max(2, int(round(1.25 * feature_width / bin_hours)))
        gap_bins = max(1, int(round(0.50 * feature_width / bin_hours)))
        center_bin = _phase_bin(tensor.shape[0], center_phase)
        source = np.arange(center_bin - gap_bins - window_bins, center_bin - gap_bins)
        destination = np.arange(
            center_bin + gap_bins,
            center_bin + gap_bins + window_bins,
        )
        if source[0] < 0 or destination[-1] >= tensor.shape[0]:
            raise AssertionError("Critical C phase windows fell outside the tensor")
        if source.size != destination.size:
            raise AssertionError("Critical C phase windows have different sizes")
        amount = 0.40 * np.minimum(
            tensor[source, behavior_index],
            tensor[destination, behavior_index],
        )
        delta[source, behavior_index] += amount
        delta[destination, behavior_index] -= amount
    return _validate_occupancy(tensor + delta), _validate_occupancy(tensor - delta)


def _row(
    test_name: str,
    resolution_minutes: int,
    condition: str,
    result: EventResult,
    expected_behavior: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "test_name": test_name,
        "resolution_minutes": resolution_minutes,
        "condition": condition,
        "A_obs": result.get("A_obs", float("nan")),
        "mu_shift": result.get("mu_shift", float("nan")),
        "R_pair": result.get("R_pair", float("nan")),
        "n_common_bins": result.get("n_common_bins", float("nan")),
        "fraction_common_bins": result.get("fraction_common_bins", float("nan")),
        "total_nonrest_A": result.get("total_nonrest_A", float("nan")),
        "total_nonrest_B": result.get("total_nonrest_B", float("nan")),
        "denominator_1_minus_mu_shift": result.get(
            "denominator_1_minus_mu_shift", float("nan")
        ),
        "status": result.get("status", "UNKNOWN"),
        "expected_behavior": expected_behavior,
        "pass_fail": "PASS" if passed else "FAIL",
    }


def _require_defined(result: EventResult, label: str) -> None:
    if result["status"] != "PASS":
        raise AssertionError(
            f"{label} was not defined: {result['status']} ({result['diagnostic']})"
        )


def _run_resolution_validation(resolution_minutes: int) -> list[dict[str, object]]:
    n_bins = 24 * 60 // resolution_minutes
    base = _synthetic_pattern(n_bins)
    identity = score_pair(base, base.copy())
    _require_defined(identity, "identity")
    if not np.isclose(identity["A_obs"], 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError(f"Identity affinity is not one: {identity['A_obs']}")
    if not np.isclose(identity["R_pair"], 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError(f"Identity corrected recurrence is not one: {identity['R_pair']}")
    rows = [
        _row(
            "identity",
            resolution_minutes,
            "identical_nonuniform_pattern",
            identity,
            "A_obs≈1 and R_pair≈1",
            True,
        )
    ]

    displacement_results: list[EventResult] = []
    for shift in (1, 2, 4, 8):
        result = score_pair(base, np.roll(base, shift, axis=0))
        _require_defined(result, f"controlled displacement {shift}")
        displacement_results.append(result)
        rows.append(
            _row(
                "controlled_ct_displacement",
                resolution_minutes,
                f"shift_{shift * resolution_minutes}_minutes",
                result,
                "R_pair declines across the designed displacement series",
                True,
            )
        )
    displacement_scores = [float(result["R_pair"]) for result in displacement_results]
    if not all(
        displacement_scores[index] > displacement_scores[index + 1]
        for index in range(len(displacement_scores) - 1)
    ):
        raise AssertionError(
            f"Controlled CT displacement did not decline as designed: {displacement_scores}"
        )

    behavior_perturbed = score_pair(base, _swap_selected_behavior_regions(base))
    _require_defined(behavior_perturbed, "behavior-label perturbation")
    if not float(behavior_perturbed["R_pair"]) < float(identity["R_pair"]) - 1e-8:
        raise AssertionError("Behavior-label perturbation did not reduce recurrence")
    rows.append(
        _row(
            "behavior_label_perturbation",
            resolution_minutes,
            "swap_eating_grooming_in_selected_regions",
            behavior_perturbed,
            "recurrence decreases",
            True,
        )
    )

    critical_plus, critical_minus = _critical_c_pair_patterns(base)
    repeated = score_adjacent_cycles([base, base, base, base])
    rearranged = score_adjacent_cycles(
        [critical_plus, critical_minus, critical_plus, critical_minus]
    )
    critical_changed = np.abs(critical_plus - base) > NUMERICAL_TOLERANCE
    changed_bins = int(np.any(critical_changed, axis=1).sum())
    changed_behaviors = int(np.any(critical_changed, axis=0).sum())
    if changed_bins < n_bins // 10 or changed_behaviors < N_BEHAVIORS:
        raise AssertionError(
            "Critical C redistribution is not substantive across phase and behavior"
        )
    repeated_pool = np.sum([base, base, base, base], axis=0)
    rearranged_pool = np.sum(
        [critical_plus, critical_minus, critical_plus, critical_minus], axis=0
    )
    if not np.allclose(
        repeated_pool,
        rearranged_pool,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise AssertionError("Critical C pooled phase-by-behavior distributions differ")
    _require_defined(repeated, "Critical C repeated condition")
    _require_defined(rearranged, "Critical C rearranged condition")
    critical_separation = float(repeated["R_pair"]) - float(rearranged["R_pair"])
    if critical_separation <= 1e-3:
        raise AssertionError(
            "Occupancy-level Critical C separation is only numerical: "
            f"{critical_separation}"
        )
    rows.extend(
        [
            _row(
                "critical_c_occupancy",
                resolution_minutes,
                "repeated_cycle_pattern",
                repeated,
                "higher recurrence than rearranged condition",
                True,
            ),
            _row(
                "critical_c_occupancy",
                resolution_minutes,
                "rearranged_same_pooled_distribution",
                rearranged,
                "lower recurrence than repeated condition",
                True,
            ),
        ]
    )

    sample_minutes = 0.5
    five_minute_bins = 24 * 60 // 5
    eating = NONREST_BEHAVIORS[0]
    grooming = NONREST_BEHAVIORS[6]
    locomotion = NONREST_BEHAVIORS[7]
    sustained_templates = np.asarray(
        [
            [eating, eating, eating, eating, grooming, grooming, locomotion, locomotion, locomotion, locomotion],
            [eating, eating, eating, eating, grooming, grooming, grooming, grooming, locomotion, locomotion],
            [eating, eating, grooming, grooming, grooming, grooming, locomotion, locomotion, locomotion, locomotion],
            [eating, grooming, grooming, grooming, grooming, locomotion, locomotion, locomotion, locomotion, locomotion],
            [grooming, grooming, grooming, grooming, grooming, locomotion, locomotion, locomotion, locomotion, locomotion],
            [locomotion, locomotion, locomotion, locomotion, locomotion, locomotion, eating, eating, eating, eating],
        ],
        dtype=object,
    )
    fragmented_templates = np.asarray(
        [
            [eating, locomotion, eating, locomotion, eating, locomotion, eating, locomotion, grooming, grooming],
            [eating, grooming, locomotion, eating, grooming, locomotion, eating, grooming, eating, grooming],
            [eating, grooming, locomotion, grooming, eating, locomotion, grooming, locomotion, grooming, locomotion],
            [eating, grooming, locomotion, grooming, locomotion, grooming, locomotion, grooming, locomotion, locomotion],
            [grooming, locomotion, grooming, locomotion, grooming, locomotion, grooming, locomotion, grooming, locomotion],
            [locomotion, eating, locomotion, eating, locomotion, eating, locomotion, locomotion, locomotion, eating],
        ],
        dtype=object,
    )
    reordered_templates = np.roll(sustained_templates, 2, axis=1)
    if five_minute_bins % len(sustained_templates) != 0:
        raise AssertionError("Synthetic templates do not tile the 24-hour sequence")
    template_repetitions = five_minute_bins // len(sustained_templates)
    sustained_sequence = np.tile(sustained_templates, (template_repetitions, 1)).reshape(-1)
    fragmented_sequence = np.tile(fragmented_templates, (template_repetitions, 1)).reshape(-1)
    if np.array_equal(sustained_sequence, fragmented_sequence):
        raise AssertionError("Fragmentation sequences were accidentally identical")
    sustained_bout_count = 1 + int(
        np.count_nonzero(sustained_sequence[1:] != sustained_sequence[:-1])
    )
    fragmented_bout_count = 1 + int(
        np.count_nonzero(fragmented_sequence[1:] != fragmented_sequence[:-1])
    )
    if fragmented_bout_count <= 2 * sustained_bout_count:
        raise AssertionError("Fragmentation sequences did not differ substantially in bout count")
    sustained_occupancy = _sequence_to_occupancy(
        sustained_sequence,
        sample_minutes,
        resolution_minutes,
    )
    fragmented_occupancy = _sequence_to_occupancy(
        fragmented_sequence,
        sample_minutes,
        resolution_minutes,
    )
    if not np.allclose(
        sustained_occupancy,
        fragmented_occupancy,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise AssertionError("Different fragmentation sequences changed occupancy")
    fragmentation = score_pair(sustained_occupancy, fragmented_occupancy)
    sustained_identity = score_pair(sustained_occupancy, sustained_occupancy)
    _require_defined(fragmentation, "fragmentation invariance")
    _require_defined(sustained_identity, "sustained sequence identity")
    if not np.isclose(
        fragmentation["R_pair"],
        sustained_identity["R_pair"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("Fragmentation changed the occupancy-based score")
    rows.append(
        _row(
            "fragmentation_invariance",
            resolution_minutes,
            "sustained_vs_fragmented_sequence_same_occupancy",
            fragmentation,
            "different bout fragmentation gives identical occupancy and score",
            True,
        )
    )

    globally_scaled = score_pair(base, 0.8 * base)
    _require_defined(globally_scaled, "global scaling")
    if not np.isclose(
        globally_scaled["R_pair"],
        identity["R_pair"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("Uniform global scaling changed normalized recurrence")
    rows.append(
        _row(
            "global_scaling",
            resolution_minutes,
            "cycle_B_equals_0.8_times_cycle_A",
            globally_scaled,
            "normalized recurrence is unchanged",
            True,
        )
    )

    locally_redistributed = score_pair(
        base,
        _transfer_occupancy(base, 4, 13.2, 16.0, 0.05),
    )
    _require_defined(locally_redistributed, "local redistribution")
    if not float(locally_redistributed["R_pair"]) < float(identity["R_pair"]) - 1e-8:
        raise AssertionError("Local occupancy redistribution did not reduce recurrence")
    rows.append(
        _row(
            "local_redistribution",
            resolution_minutes,
            "move_digging_occupancy_CT13_to_CT16",
            locally_redistributed,
            "recurrence decreases",
            True,
        )
    )

    within_a_sequence = sustained_sequence.copy()
    within_b_sequence = np.tile(
        reordered_templates,
        (template_repetitions, 1),
    ).reshape(-1)
    if np.array_equal(within_a_sequence, within_b_sequence):
        raise AssertionError("Within-bin reorder sequences were accidentally identical")
    within_a_occupancy = _sequence_to_occupancy(
        within_a_sequence,
        sample_minutes,
        resolution_minutes,
    )
    within_b_occupancy = _sequence_to_occupancy(
        within_b_sequence,
        sample_minutes,
        resolution_minutes,
    )
    if not np.allclose(
        within_a_occupancy,
        within_b_occupancy,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise AssertionError("Within-bin sequence reordering changed occupancy")
    within_bin = score_pair(within_a_occupancy, within_b_occupancy)
    across_bin_occupancy = np.roll(base, 1, axis=0)
    if np.allclose(base, across_bin_occupancy, rtol=0.0, atol=NUMERICAL_TOLERANCE):
        raise AssertionError("Across-bin control did not change indexed-bin occupancy")
    if not np.allclose(
        base.sum(axis=0),
        across_bin_occupancy.sum(axis=0),
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise AssertionError("Across-bin control changed total behavior amounts")
    across_bin = score_pair(base, across_bin_occupancy)
    _require_defined(within_bin, "within-bin reorder")
    _require_defined(across_bin, "across-bin reorder")
    if not np.isclose(within_bin["R_pair"], identity["R_pair"], atol=1e-10, rtol=0.0):
        raise AssertionError("Within-bin reorder changed the occupancy score")
    if not float(across_bin["R_pair"]) < float(identity["R_pair"]) - 1e-8:
        raise AssertionError("Across-bin reorder did not reduce recurrence")
    rows.extend(
        [
            _row(
                "within_bin_vs_across_bin_order",
                resolution_minutes,
                "within_bin_reorder",
                within_bin,
                "different within-bin ordering gives identical occupancy and score",
                True,
            ),
            _row(
                "within_bin_vs_across_bin_order",
                resolution_minutes,
                "across_bin_reorder",
                across_bin,
                "same total amounts but lower recurrence after across-bin redistribution",
                True,
            ),
        ]
    )

    coverage_a = np.ones(n_bins, dtype=float)
    coverage_b = np.ones(n_bins, dtype=float)
    coverage_a[::5] = MIN_VALID_COVERAGE - 0.01
    coverage_b[2::7] = MIN_VALID_COVERAGE - 0.01
    common_support = score_pair(
        base,
        base,
        valid_mask_from_coverage(coverage_a),
        valid_mask_from_coverage(coverage_b),
    )
    _require_defined(common_support, "common-support handling")
    if not 0.0 < float(common_support["fraction_common_bins"]) < 1.0:
        raise AssertionError("Common-support test did not restrict the usable bins")
    rows.append(
        _row(
            "common_support",
            resolution_minutes,
            "different_valid_coverage_masks",
            common_support,
            "both cycles use the same explicitly intersected support",
            True,
        )
    )

    zero_nonrest = score_pair(np.zeros_like(base), base)
    if zero_nonrest["status"] != "UNDEFINED_ZERO_NONREST_OCCUPANCY":
        raise AssertionError(
            f"Zero non-rest occupancy was not reported undefined: {zero_nonrest['status']}"
        )
    rows.append(
        _row(
            "zero_nonrest_occupancy",
            resolution_minutes,
            "cycle_A_zero_nonrest",
            zero_nonrest,
            "undefined score with a clear diagnostic",
            True,
        )
    )
    return rows


def run_validation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for resolution_minutes in RESOLUTIONS_MINUTES:
        rows.extend(_run_resolution_validation(resolution_minutes))
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if frame.empty or not np.all(frame["pass_fail"].eq("PASS")):
        raise AssertionError("At least one occupancy recurrence validation failed")
    return frame


def main() -> None:
    results = run_validation()
    results.to_csv(OUTPUT_PATH, index=False)
    print(results[["test_name", "resolution_minutes", "condition", "R_pair", "pass_fail"]].to_string(index=False))
    print(f"Validation PASS: {len(results)} synthetic result rows")
    print(f"Summary written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
