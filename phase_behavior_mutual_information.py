"""Single-animal Phase x Behavior mutual-information analysis.

This script reads one animal's CBAS model-output CSV files from a command-line
input directory, assigns a winner-take-all behavior to each classifiable row,
reconstructs elapsed time, and writes the requested MI tables, null
distribution, plot, and text summary to ``<input_dir>\\MI_Output``.

The analysis is intentionally single-animal and focused. It uses a runtime
free-running period and a relative phase anchor; it does not assign biological
CT or perform any genotype or recurrence analysis.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEHAVIORS = [
    "eating",
    "drinking",
    "rearing",
    "climbing",
    "digging",
    "nesting",
    "resting",
    "grooming",
    "locomotion",
]
RESTING_BEHAVIOR_INDEX = BEHAVIORS.index("resting")
NONRESTING_BEHAVIOR_INDICES = tuple(
    index for index, behavior in enumerate(BEHAVIORS) if behavior != "resting"
)

DEFAULT_FRP_HOURS = 24.0
FILE_DURATION_MINUTES = 10.0
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260911
PHASE_BIN_COUNTS = (8, 12, 24)
PRIMARY_PHASE_BINS = 12
PHASE_ORIGIN_OFFSETS_MINUTES = tuple(range(0, 120, 10))
DECOMPOSITION_TOLERANCE_BITS = 1e-12

INPUT_FILENAME_PATTERN = re.compile(
    r"^(?P<animal>.+)_(?P<index>\d{5})_curated_aug_model_outputs\.csv$"
)
LEGACY_OUTPUT_FILENAMES = (
    "behavior_sequence.csv",
    "mi_summary.csv",
    "mi_decomposition_12bin.csv",
    "phase_behavior_counts_12bin.csv",
    "phase_behavior_probabilities_12bin.csv",
    "mi_null_distribution_12bin.csv",
    "mi_phase_origin_sensitivity.csv",
    "behavioral_time_budget.png",
    "phase_behavior_heatmap_12bin.png",
    "mi_null_distribution_12bin.png",
    "mi_sensitivity.png",
    "mi_phase_origin_sensitivity.png",
    "mi_decomposition_12bin.png",
    "mi_summary.txt",
    "MANUAL_DIFF.txt",
)


def discover_input_files(input_dir: Path) -> tuple[list[tuple[int, Path]], list[int]]:
    """Find and numerically sort the expected source CSV files."""

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )

    csv_files = sorted(input_dir.glob("*.csv"))

    parsed: list[tuple[int, Path]] = []
    animals: set[str] = set()
    for path in csv_files:
        match = INPUT_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        animals.add(match.group("animal"))
        parsed.append((int(match.group("index")), path))

    if not parsed:
        raise FileNotFoundError(
            f"No matching CBAS source CSV files found in {input_dir}"
        )

    if len(animals) != 1:
        raise ValueError(
            "Expected one single-animal filename prefix, found: "
            + ", ".join(sorted(animals))
        )

    parsed.sort(key=lambda item: item[0])
    indices = [index for index, _ in parsed]
    duplicate_indices = sorted(
        index for index, count in pd.Series(indices).value_counts().items() if count > 1
    )
    if duplicate_indices:
        raise ValueError(f"Duplicate numeric sequence indices found: {duplicate_indices}")

    if indices[0] != 0:
        raise ValueError(
            "File sequence must include 00000 because it defines the phase anchor."
        )

    expected = set(range(indices[0], indices[-1] + 1))
    missing_indices = sorted(expected.difference(indices))
    return parsed, missing_indices


def remove_known_legacy_outputs(output_dir: Path) -> None:
    """Remove only named generated artifacts superseded by the six outputs."""

    for filename in LEGACY_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Expected legacy output to be a file: {path}")
            path.unlink()


def read_and_classify(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Read required columns and return valid row positions and winner labels.

    A row is classifiable when at least one required probability is a finite
    numeric value. Non-finite values are ignored for that row's argmax. Ties
    are resolved by the order in ``BEHAVIORS``.
    """

    try:
        frame = pd.read_csv(path, usecols=BEHAVIORS)
    except ValueError as error:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as header_error:  # pragma: no cover - defensive error path
            raise ValueError(
                f"Could not read the header of {path.name}: {header_error}"
            ) from error
        missing = [behavior for behavior in BEHAVIORS if behavior not in header]
        if missing:
            raise ValueError(
                f"Required behavior columns missing from {path.name}: {missing}"
            ) from error
        raise ValueError(f"Could not read required behavior columns from {path.name}: {error}") from error

    n_rows = len(frame)
    if n_rows == 0:
        return 0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    valid = finite.any(axis=1)

    labels = np.full(n_rows, -1, dtype=np.int8)
    if valid.any():
        valid_values = np.where(finite[valid], numeric[valid], -np.inf)
        labels[valid] = np.argmax(valid_values, axis=1).astype(np.int8)

    row_positions = np.arange(n_rows, dtype=np.int64)
    return n_rows, row_positions[valid], labels[valid]


def phase_bin_indices(
    phase_hours: np.ndarray,
    n_bins: int,
    *,
    frp_hours: float,
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Assign phase values to equal-width half-open bins at a chosen origin."""

    hours_per_bin = frp_hours / n_bins
    rotated_phase = np.mod(phase_hours - origin_hours, frp_hours)
    safe_phase = np.minimum(rotated_phase, np.nextafter(frp_hours, 0.0))
    return np.floor(safe_phase / hours_per_bin).astype(np.int64)


def contingency_table(bin_indices: np.ndarray, labels: np.ndarray, n_bins: int) -> np.ndarray:
    """Return a phase-bin by behavior count table."""

    flat = np.bincount(
        bin_indices * len(BEHAVIORS) + labels,
        minlength=n_bins * len(BEHAVIORS),
    )
    return flat.reshape(n_bins, len(BEHAVIORS)).astype(np.int64, copy=False)


def rest_nonrest_counts(counts: np.ndarray) -> np.ndarray:
    """Collapse the 9-state behavior axis to resting versus non-resting."""

    resting = np.take(counts, RESTING_BEHAVIOR_INDEX, axis=-1)[..., None]
    nonresting = np.take(counts, NONRESTING_BEHAVIOR_INDICES, axis=-1).sum(
        axis=-1,
        keepdims=True,
    )
    return np.concatenate((resting, nonresting), axis=-1)


def conditional_nonrest_counts(counts: np.ndarray) -> np.ndarray:
    """Remove resting without changing phase bins or concatenating samples."""

    return np.take(counts, NONRESTING_BEHAVIOR_INDICES, axis=-1)


def mutual_information_bits(counts: np.ndarray) -> float:
    """Calculate MI from a contingency table using the requested definition."""

    total = counts.sum()
    if total <= 0:
        return float("nan")

    joint = counts.astype(float) / total
    phase_probability = joint.sum(axis=1)
    behavior_probability = joint.sum(axis=0)
    expected = phase_probability[:, None] * behavior_probability[None, :]
    positive = joint > 0

    ratio = np.divide(joint, expected, out=np.ones_like(joint), where=positive)
    return float(np.sum(np.where(positive, joint * np.log2(ratio), 0.0)))


def vectorized_mutual_information_bits(counts: np.ndarray) -> np.ndarray:
    """Calculate MI for many contingency tables at once."""

    totals = counts.sum(axis=(1, 2)).astype(float)
    joint = counts.astype(float) / totals[:, None, None]
    phase_probability = joint.sum(axis=2)
    behavior_probability = joint.sum(axis=1)
    expected = phase_probability[:, :, None] * behavior_probability[:, None, :]
    positive = joint > 0
    ratio = np.divide(joint, expected, out=np.ones_like(joint), where=positive)
    return np.sum(np.where(positive, joint * np.log2(ratio), 0.0), axis=(1, 2))


def behavioral_entropy_bits(counts: np.ndarray) -> float:
    """Calculate entropy of the behavior marginal in bits."""

    total = counts.sum()
    if total <= 0:
        return float("nan")

    probability = counts.sum(axis=0).astype(float) / total
    positive = probability > 0
    return float(-np.sum(probability[positive] * np.log2(probability[positive])))


def vectorized_behavioral_entropy_bits(counts: np.ndarray) -> np.ndarray:
    """Calculate behavior entropy for many contingency tables."""

    totals = counts.sum(axis=(1, 2)).astype(float)
    probability = counts.sum(axis=1).astype(float) / totals[:, None]
    positive = probability > 0
    safe_probability = np.where(positive, probability, 1.0)
    return -np.sum(
        np.where(positive, probability * np.log2(safe_probability), 0.0), axis=1
    )


def build_prefix_counts(labels: np.ndarray) -> np.ndarray:
    """Build per-behavior cumulative counts for fast interval queries."""

    prefix = np.zeros((len(labels) + 1, len(BEHAVIORS)), dtype=np.int32)
    for behavior_index in range(len(BEHAVIORS)):
        prefix[1:, behavior_index] = np.cumsum(
            labels == behavior_index, dtype=np.int32
        )
    return prefix


def shifted_cycle_counts(
    event_times: np.ndarray,
    labels: np.ndarray,
    offsets_hours: np.ndarray,
    n_bins: int,
    *,
    frp_hours: float,
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Count shifted cycle behaviors for each random circular time offset.

    ``event_times`` are within-cycle times in [0, frp_hours). For an offset d, a
    source event at time t is assigned to phase (t + d) modulo frp_hours. Prefix
    counts and binary searches make this exact for irregular valid-row times
    without shuffling the behavioral labels.
    """

    prefix = build_prefix_counts(labels)
    n_events = len(labels)
    bin_width = frp_hours / n_bins
    bin_starts = np.mod(
        origin_hours + np.arange(n_bins, dtype=float) * bin_width,
        frp_hours,
    )

    starts = np.mod(bin_starts[None, :] - offsets_hours[:, None], frp_hours)
    ends = starts + bin_width
    wraps = ends > frp_hours

    left = np.searchsorted(event_times, starts, side="left")
    right = np.searchsorted(event_times, np.minimum(ends, frp_hours), side="left")
    counts = np.empty(
        (len(offsets_hours), n_bins, len(BEHAVIORS)), dtype=np.int64
    )

    non_wrapped = ~wraps
    if non_wrapped.any():
        counts[non_wrapped] = prefix[right[non_wrapped]] - prefix[left[non_wrapped]]

    if wraps.any():
        wrapped_right = np.searchsorted(
            event_times, ends[wraps] - frp_hours, side="left"
        )
        counts[wraps] = (
            prefix[n_events]
            - prefix[left[wraps]]
            + prefix[wrapped_right]
        )

    return counts


def calculate_null_tables(
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    offsets_hours: np.ndarray,
    n_bins: int,
    *,
    frp_hours: float,
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Aggregate independently shifted complete cycles for all permutations."""

    null_counts = np.zeros(
        (len(offsets_hours), n_bins, len(BEHAVIORS)), dtype=np.int64
    )
    for cycle_number, (_, event_times, labels) in enumerate(complete_cycles):
        null_counts += shifted_cycle_counts(
            event_times,
            labels,
            offsets_hours[:, cycle_number],
            n_bins,
            frp_hours=frp_hours,
            origin_hours=origin_hours,
        )
    return null_counts


def observed_counts_from_complete_cycles(
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    n_bins: int,
    *,
    frp_hours: float,
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Build observed phase-by-behavior counts from exactly the null cycles."""

    counts = np.zeros((n_bins, len(BEHAVIORS)), dtype=np.int64)
    for _, event_times, labels in complete_cycles:
        phase_hours = np.mod(event_times, frp_hours)
        counts += contingency_table(
            phase_bin_indices(
                phase_hours,
                n_bins,
                frp_hours=frp_hours,
                origin_hours=origin_hours,
            ),
            labels,
            n_bins,
        )
    return counts


def calculate_phase_origin_sensitivity(
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    offsets_hours: np.ndarray,
    primary_observed_counts: np.ndarray,
    primary_null_counts: np.ndarray,
    primary_metrics: dict[str, float],
    complete_sample_count: int,
    *,
    frp_hours: float,
) -> pd.DataFrame:
    """Calculate corrected MI while rotating the 12-bin phase origin."""

    rows: list[dict[str, float | int]] = []
    for offset_minutes in PHASE_ORIGIN_OFFSETS_MINUTES:
        origin_hours = offset_minutes / 60.0
        if offset_minutes == 0:
            observed_counts = primary_observed_counts
            null_counts = primary_null_counts
        else:
            observed_counts = observed_counts_from_complete_cycles(
                complete_cycles,
                PRIMARY_PHASE_BINS,
                frp_hours=frp_hours,
                origin_hours=origin_hours,
            )
            null_counts = calculate_null_tables(
                complete_cycles,
                offsets_hours,
                PRIMARY_PHASE_BINS,
                frp_hours=frp_hours,
                origin_hours=origin_hours,
            )

        metrics, _, _ = analyze_observed_and_null(
            observed_counts,
            null_counts,
            expected_sample_count=complete_sample_count,
        )
        rows.append(
            {
                "phase_bin_offset_minutes": offset_minutes,
                "MI_raw_bits": metrics["MI_raw_bits"],
                "MI_null_mean": metrics["MI_null_mean"],
                "MI_null_SD": metrics["MI_null_SD"],
                "MI_excess_bits": metrics["MI_excess_bits"],
                "MI_z": metrics["MI_z"],
                "NMI_raw": metrics["NMI_raw"],
                "NMI_null_mean": metrics["NMI_null_mean"],
                "NMI_excess": metrics["NMI_excess"],
            }
        )

    sensitivity = pd.DataFrame(rows)
    zero_offset = sensitivity.iloc[0]
    for metric_name in (
        "MI_raw_bits",
        "MI_null_mean",
        "MI_null_SD",
        "MI_excess_bits",
        "MI_z",
        "NMI_raw",
        "NMI_null_mean",
        "NMI_excess",
    ):
        if not np.isclose(zero_offset[metric_name], primary_metrics[metric_name]):
            raise ValueError(
                f"The 0-minute phase-origin result does not match the primary "
                f"12-bin result for {metric_name}."
            )
    return sensitivity


def analyze_observed_and_null(
    observed_counts: np.ndarray,
    null_counts: np.ndarray,
    expected_sample_count: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate requested observed and permutation-corrected MI metrics."""

    if expected_sample_count is not None:
        if int(observed_counts.sum()) != expected_sample_count:
            raise ValueError(
                "Observed contingency table does not contain the expected "
                "complete-cycle sample count."
            )
        null_sample_counts = null_counts.sum(axis=(1, 2))
        if not np.all(null_sample_counts == expected_sample_count):
            raise ValueError(
                "At least one null contingency table does not contain the "
                "expected complete-cycle sample count."
            )

    observed_mi = mutual_information_bits(observed_counts)
    observed_entropy = behavioral_entropy_bits(observed_counts)
    observed_nmi = (
        observed_mi / observed_entropy if observed_entropy > 0 else float("nan")
    )

    null_mi = vectorized_mutual_information_bits(null_counts)
    null_entropy = vectorized_behavioral_entropy_bits(null_counts)
    null_nmi = np.divide(
        null_mi,
        null_entropy,
        out=np.full_like(null_mi, np.nan, dtype=float),
        where=null_entropy > 0,
    )

    null_mean = float(np.mean(null_mi))
    null_sd = float(np.std(null_mi, ddof=1))
    metrics = {
        "MI_raw_bits": observed_mi,
        "MI_null_mean": null_mean,
        "MI_null_SD": null_sd,
        "MI_excess_bits": observed_mi - null_mean,
        "MI_z": (
            (observed_mi - null_mean) / null_sd if null_sd > 0 else float("nan")
        ),
        "H_behavior_bits": observed_entropy,
        "NMI_raw": observed_nmi,
        "NMI_null_mean": float(np.mean(null_nmi)),
        "NMI_excess": observed_nmi - float(np.mean(null_nmi)),
    }
    return metrics, null_mi, null_nmi


def calculate_primary_mi_decomposition(
    full_observed_counts: np.ndarray,
    full_null_counts: np.ndarray,
    full_metrics: dict[str, float],
    full_null_mi: np.ndarray,
    full_null_nmi: np.ndarray,
    complete_sample_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Calculate and validate the 9-state MI decomposition at 12 bins."""

    # The null tables already contain intact 9-state sequences after the
    # existing circular shifts. Conditioning below therefore happens second.
    rest_observed_counts = rest_nonrest_counts(full_observed_counts)
    rest_null_counts = rest_nonrest_counts(full_null_counts)
    conditional_observed_counts = conditional_nonrest_counts(full_observed_counts)
    conditional_null_counts = conditional_nonrest_counts(full_null_counts)

    nonrest_sample_count = int(conditional_observed_counts.sum())
    rest_metrics, rest_null_mi, rest_null_nmi = analyze_observed_and_null(
        rest_observed_counts,
        rest_null_counts,
        expected_sample_count=complete_sample_count,
    )
    conditional_metrics, conditional_null_mi, conditional_null_nmi = analyze_observed_and_null(
        conditional_observed_counts,
        conditional_null_counts,
        expected_sample_count=nonrest_sample_count,
    )

    resting_sample_count = int(rest_observed_counts[:, 0].sum())
    p_rest = resting_sample_count / complete_sample_count
    p_nonrest = nonrest_sample_count / complete_sample_count
    if not np.isclose(p_rest + p_nonrest, 1.0, atol=DECOMPOSITION_TOLERANCE_BITS, rtol=0.0):
        raise ValueError("P_rest + P_nonrest does not equal 1 within tolerance.")

    if not np.all(rest_null_counts.sum(axis=(1, 2)) == complete_sample_count):
        raise ValueError("Rest/non-rest null tables changed the complete-cycle sample count.")
    if not np.all(conditional_null_counts.sum(axis=(1, 2)) == nonrest_sample_count):
        raise ValueError("Conditional null tables do not preserve the non-rest sample count.")

    observed_reconstructed_mi = (
        rest_metrics["MI_raw_bits"]
        + p_nonrest * conditional_metrics["MI_raw_bits"]
    )
    observed_decomposition_error = float(
        full_metrics["MI_raw_bits"] - observed_reconstructed_mi
    )
    null_reconstructed_mi = rest_null_mi + p_nonrest * conditional_null_mi
    null_decomposition_errors = full_null_mi - null_reconstructed_mi
    maximum_null_decomposition_error = float(
        np.max(np.abs(null_decomposition_errors))
    )

    weighted_conditional_excess = (
        p_nonrest * conditional_metrics["MI_excess_bits"]
    )
    reconstructed_excess = (
        rest_metrics["MI_excess_bits"] + weighted_conditional_excess
    )
    excess_decomposition_error = float(
        full_metrics["MI_excess_bits"] - reconstructed_excess
    )

    if abs(observed_decomposition_error) > DECOMPOSITION_TOLERANCE_BITS:
        raise ValueError(
            "Observed MI decomposition identity exceeded the tolerance: "
            f"{observed_decomposition_error:.17g} bits."
        )
    if maximum_null_decomposition_error > DECOMPOSITION_TOLERANCE_BITS:
        raise ValueError(
            "A null MI decomposition identity exceeded the tolerance: "
            f"{maximum_null_decomposition_error:.17g} bits."
        )
    if abs(excess_decomposition_error) > DECOMPOSITION_TOLERANCE_BITS:
        raise ValueError(
            "Excess-MI decomposition identity exceeded the tolerance: "
            f"{excess_decomposition_error:.17g} bits."
        )

    def component_row(
        component: str,
        n_samples: int,
        n_behavior_states: int,
        metrics: dict[str, float],
        weighted_raw: float,
        weighted_excess: float,
    ) -> dict[str, float | int | str]:
        return {
            "component": component,
            "n_samples": n_samples,
            "n_behavior_states": n_behavior_states,
            "H_behavior_bits": metrics["H_behavior_bits"],
            "MI_raw_bits": metrics["MI_raw_bits"],
            "MI_null_mean_bits": metrics["MI_null_mean"],
            "MI_null_SD_bits": metrics["MI_null_SD"],
            "MI_excess_bits": metrics["MI_excess_bits"],
            "MI_z": metrics["MI_z"],
            "NMI_raw": metrics["NMI_raw"],
            "NMI_null_mean": metrics["NMI_null_mean"],
            "NMI_excess": metrics["NMI_excess"],
            "P_rest": p_rest,
            "P_nonrest": p_nonrest,
            "weighted_MI_raw_bits": weighted_raw,
            "weighted_MI_excess_bits": weighted_excess,
        }

    decomposition_frame = pd.DataFrame(
        [
            component_row(
                "full_9state",
                complete_sample_count,
                len(BEHAVIORS),
                full_metrics,
                full_metrics["MI_raw_bits"],
                full_metrics["MI_excess_bits"],
            ),
            component_row(
                "rest_vs_nonrest",
                complete_sample_count,
                2,
                rest_metrics,
                rest_metrics["MI_raw_bits"],
                rest_metrics["MI_excess_bits"],
            ),
            component_row(
                "conditional_8state_nonrest",
                nonrest_sample_count,
                len(NONRESTING_BEHAVIOR_INDICES),
                conditional_metrics,
                p_nonrest * conditional_metrics["MI_raw_bits"],
                p_nonrest * conditional_metrics["MI_excess_bits"],
            ),
        ]
    )
    details = {
        "p_rest": float(p_rest),
        "p_nonrest": float(p_nonrest),
        "n_resting_samples": float(resting_sample_count),
        "n_nonresting_samples": float(nonrest_sample_count),
        "observed_9state_mi": float(full_metrics["MI_raw_bits"]),
        "observed_rest_nonrest_mi": float(rest_metrics["MI_raw_bits"]),
        "observed_conditional_8state_mi": float(conditional_metrics["MI_raw_bits"]),
        "weighted_conditional_mi": float(
            p_nonrest * conditional_metrics["MI_raw_bits"]
        ),
        "reconstructed_total_mi": float(observed_reconstructed_mi),
        "observed_decomposition_error_bits": observed_decomposition_error,
        "maximum_null_decomposition_error_bits": maximum_null_decomposition_error,
        "mi9_excess": float(full_metrics["MI_excess_bits"]),
        "rest_nonrest_excess": float(rest_metrics["MI_excess_bits"]),
        "conditional_8state_excess": float(conditional_metrics["MI_excess_bits"]),
        "weighted_conditional_excess": float(weighted_conditional_excess),
        "reconstructed_excess": float(reconstructed_excess),
        "excess_decomposition_error_bits": excess_decomposition_error,
    }
    null_frame = pd.DataFrame(
        {
            "permutation": np.arange(1, len(full_null_mi) + 1),
            "MI_9state": full_null_mi,
            "MI_rest_nonrest": rest_null_mi,
            "MI_conditional_8state_nonrest": conditional_null_mi,
            "weighted_MI_conditional_8state_nonrest": p_nonrest * conditional_null_mi,
            "NMI_9state": full_null_nmi,
            "NMI_rest_nonrest": rest_null_nmi,
            "NMI_conditional_8state_nonrest": conditional_null_nmi,
        }
    )
    return decomposition_frame, null_frame, details


def make_phase_labels(n_bins: int, *, frp_hours: float) -> list[str]:
    hours_per_bin = frp_hours / n_bins
    return [
        f"{int(start):02d}-{int(start + hours_per_bin):02d}"
        for start in np.arange(n_bins, dtype=float) * hours_per_bin
    ]


def save_phase_behavior_profile(
    counts: np.ndarray,
    path: Path,
    *,
    frp_hours: float,
) -> None:
    """Save the primary phase-by-behavior composition as proportions."""

    phase_totals = counts.sum(axis=1)
    if np.any(phase_totals <= 0):
        raise ValueError("Every primary phase bin must contain at least one sample.")
    proportions = counts.astype(float) / phase_totals[:, None]
    n_bins = counts.shape[0]
    hours_per_bin = frp_hours / n_bins
    profile = pd.DataFrame(proportions, columns=BEHAVIORS)
    profile.insert(0, "phase_bin", np.arange(n_bins, dtype=int))
    profile.insert(
        1,
        "phase_start_hours",
        np.arange(n_bins, dtype=float) * hours_per_bin,
    )
    profile.insert(
        2,
        "phase_end_hours",
        (np.arange(n_bins, dtype=float) + 1) * hours_per_bin,
    )
    profile.to_csv(path, index=False)


def build_sensitivity_frame(
    summary_frame: pd.DataFrame,
    phase_origin_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Combine existing width and phase-origin sensitivity metrics."""

    columns = [
        "sensitivity_type",
        "setting",
        "MI_raw_bits",
        "MI_null_mean_bits",
        "MI_null_SD_bits",
        "MI_excess_bits",
        "MI_z",
        "NMI_raw",
        "NMI_null_mean",
        "NMI_excess",
    ]
    rows: list[dict[str, float | int | str]] = []
    for _, row in summary_frame.sort_values("phase_bins").iterrows():
        rows.append(
            {
                "sensitivity_type": "phase_bin_count",
                "setting": int(row["phase_bins"]),
                "MI_raw_bits": row["MI_raw_bits"],
                "MI_null_mean_bits": row["MI_null_mean"],
                "MI_null_SD_bits": row["MI_null_SD"],
                "MI_excess_bits": row["MI_excess_bits"],
                "MI_z": row["MI_z"],
                "NMI_raw": row["NMI_raw"],
                "NMI_null_mean": row["NMI_null_mean"],
                "NMI_excess": row["NMI_excess"],
            }
        )
    for _, row in phase_origin_sensitivity.iterrows():
        rows.append(
            {
                "sensitivity_type": "phase_origin_minutes",
                "setting": int(row["phase_bin_offset_minutes"]),
                "MI_raw_bits": row["MI_raw_bits"],
                "MI_null_mean_bits": row["MI_null_mean"],
                "MI_null_SD_bits": row["MI_null_SD"],
                "MI_excess_bits": row["MI_excess_bits"],
                "MI_z": row["MI_z"],
                "NMI_raw": row["NMI_raw"],
                "NMI_null_mean": row["NMI_null_mean"],
                "NMI_excess": row["NMI_excess"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_overview_plot(
    output_dir: Path,
    primary_counts: np.ndarray,
    primary_metrics: dict[str, float],
    primary_null_mi: np.ndarray,
    decomposition_details: dict[str, float],
    *,
    frp_hours: float,
) -> None:
    """Save the three-panel overview figure for the primary analysis."""

    phase_totals = primary_counts.sum(axis=1)
    profile = primary_counts.astype(float) / phase_totals[:, None]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    image = axes[0].imshow(
        profile.T,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(profile),
        origin="lower",
    )
    axes[0].set_title("A. Phase x behavior profile")
    axes[0].set_xlabel("Relative circadian phase (hours)")
    axes[0].set_ylabel("Behavior")
    axes[0].set_xticks(np.arange(PRIMARY_PHASE_BINS))
    axes[0].set_xticklabels(
        make_phase_labels(PRIMARY_PHASE_BINS, frp_hours=frp_hours),
        rotation=45,
        ha="right",
    )
    axes[0].set_yticks(np.arange(len(BEHAVIORS)))
    axes[0].set_yticklabels(BEHAVIORS)
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within phase bin")

    axes[1].hist(
        primary_null_mi,
        bins=40,
        color="#4c78a8",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].axvline(
        primary_metrics["MI_raw_bits"],
        color="#d62728",
        linewidth=2,
        label=f"Observed = {primary_metrics['MI_raw_bits']:.5g}",
    )
    axes[1].axvline(
        primary_metrics["MI_null_mean"],
        color="#2ca02c",
        linewidth=2,
        label=f"Null mean = {primary_metrics['MI_null_mean']:.5g}",
    )
    axes[1].set_title("B. Full 9-state null distribution")
    axes[1].set_xlabel("MI (bits)")
    axes[1].set_ylabel("Permutation count")
    axes[1].legend(fontsize=9)

    full_excess = decomposition_details["mi9_excess"]
    rest_excess = decomposition_details["rest_nonrest_excess"]
    weighted_conditional_excess = decomposition_details["weighted_conditional_excess"]
    axes[2].bar(
        [0],
        [rest_excess],
        color="#4c78a8",
        label="Rest/non-rest",
    )
    axes[2].bar(
        [0],
        [weighted_conditional_excess],
        bottom=[rest_excess],
        color="#f58518",
        label="P_nonrest * conditional 8-state",
    )
    for bottom, value in ((0.0, rest_excess), (rest_excess, weighted_conditional_excess)):
        percentage = 100.0 * value / full_excess if full_excess else float("nan")
        axes[2].text(
            0,
            bottom + value / 2,
            f"{value:.5g} bits\n({percentage:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    axes[2].set_title("C. Full 9-state MI excess decomposition")
    axes[2].set_ylabel("MI excess (bits)")
    axes[2].set_xticks([0])
    axes[2].set_xticklabels([f"Full 9-state\n= {full_excess:.5g} bits"])
    axes[2].set_ylim(0, full_excess * 1.28)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(axis="y", alpha=0.3)

    figure.savefig(output_dir / "mi_overview.png", dpi=200)
    plt.close(figure)


def save_animal_mi_summary(
    path: Path,
    animal_id: str,
    frp_hours: float,
    frp_source: str,
    decomposition_details: dict[str, float],
) -> None:
    """Save the compact one-row downstream analysis summary."""

    columns = [
        "animal_id",
        "FRP_hours",
        "FRP_source",
        "P_rest",
        "P_nonrest",
        "MI9_excess_bits",
        "MI_rest_nonrest_excess_bits",
        "MI_conditional8_excess_bits",
        "MI_conditional8_weighted_excess_bits",
    ]
    row = {
        "animal_id": animal_id,
        "FRP_hours": frp_hours,
        "FRP_source": frp_source,
        "P_rest": decomposition_details["p_rest"],
        "P_nonrest": decomposition_details["p_nonrest"],
        "MI9_excess_bits": decomposition_details["mi9_excess"],
        "MI_rest_nonrest_excess_bits": decomposition_details["rest_nonrest_excess"],
        "MI_conditional8_excess_bits": decomposition_details["conditional_8state_excess"],
        "MI_conditional8_weighted_excess_bits": decomposition_details[
            "weighted_conditional_excess"
        ],
    }
    pd.DataFrame([row], columns=columns).to_csv(path, index=False)


def write_run_summary(
    path: Path,
    input_dir: Path,
    animal_prefix: str,
    frp_hours: float,
    frp_source: str,
    input_count: int,
    first_file_index: int,
    last_file_index: int,
    missing_indices: list[int],
    total_duration_hours: float,
    invalid_rows: int,
    anchor_row_index: int,
    complete_cycle_indices: list[int],
    complete_sample_count: int,
    all_available_sample_count: int,
    decomposition_details: dict[str, float],
) -> None:
    """Write a compact run and validation summary without duplicating results."""

    missing_text = "none" if not missing_indices else ", ".join(map(str, missing_indices))
    cycle_text = "none" if not complete_cycle_indices else ", ".join(map(str, complete_cycle_indices))
    p_sum = decomposition_details["p_rest"] + decomposition_details["p_nonrest"]
    p_sum_pass = np.isclose(
        p_sum,
        1.0,
        atol=DECOMPOSITION_TOLERANCE_BITS,
        rtol=0.0,
    )
    lines = [
        "Single-animal Phase x Behavior MI run summary",
        "",
        "Input / run information",
        f"input_directory: {input_dir.resolve()}",
        f"animal_prefix: {animal_prefix}",
        f"source_file_count: {input_count}",
        f"first_source_file_index: {first_file_index}",
        f"last_source_file_index: {last_file_index}",
        f"missing_source_indices: {missing_text}",
        f"total_valid_samples: {all_available_sample_count}",
        f"total_duration_hours: {total_duration_hours:.6f}",
        f"complete_cycles_used: [{cycle_text}]",
        f"complete_cycle_sample_count: {complete_sample_count}",
        f"partial_cycle_samples_excluded_from_corrected_MI: {all_available_sample_count - complete_sample_count}",
        f"rows_excluded_as_unclassifiable: {invalid_rows}",
        f"FRP_hours: {frp_hours}",
        f"FRP_source: {frp_source}",
        f"phase_anchor: first valid row of source file 00000, row index {anchor_row_index}, defines relative phase 0",
        f"primary_phase_bin_count: {PRIMARY_PHASE_BINS}",
        f"permutation_count: {N_PERMUTATIONS}",
        f"random_seed: {RANDOM_SEED}",
        "",
        "Behavioral composition",
        f"P_rest: {decomposition_details['p_rest']:.12g}",
        f"P_nonrest: {decomposition_details['p_nonrest']:.12g}",
        "",
        "Validation",
        f"PASS: observed and null complete-cycle sample counts match at {complete_sample_count} samples.",
        f"PASS: observed decomposition error = {decomposition_details['observed_decomposition_error_bits']:.12g} bits (tolerance {DECOMPOSITION_TOLERANCE_BITS:.0e}).",
        f"PASS: maximum null decomposition error = {decomposition_details['maximum_null_decomposition_error_bits']:.12g} bits (tolerance {DECOMPOSITION_TOLERANCE_BITS:.0e}).",
        f"PASS: excess decomposition error = {decomposition_details['excess_decomposition_error_bits']:.12g} bits (tolerance {DECOMPOSITION_TOLERANCE_BITS:.0e}).",
        f"{'PASS' if p_sum_pass else 'FAIL'}: P_rest + P_nonrest = {p_sum:.12g}.",
        f"PASS: width sensitivity completed for phase-bin counts {list(PHASE_BIN_COUNTS)}.",
        f"PASS: phase-origin sensitivity completed for offsets {list(PHASE_ORIGIN_OFFSETS_MINUTES)} minutes.",
        "PASS: raw source CSV files were read only and remain unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> tuple[Path, float, str]:
    parser = argparse.ArgumentParser(
        description="Run the single-animal Phase x Behavior mutual-information analysis."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing one animal's sequential CBAS output CSV files.",
    )
    parser.add_argument(
        "--frp-hours",
        dest="frp_hours",
        type=float,
        default=None,
        help="Free-running period in hours; defaults to 24.0.",
    )
    args = parser.parse_args()
    if args.frp_hours is None:
        frp_hours = DEFAULT_FRP_HOURS
        frp_source = "default"
    else:
        frp_hours = args.frp_hours
        frp_source = "user_supplied"
        if not np.isfinite(frp_hours) or frp_hours <= 0:
            parser.error("--frp-hours must be finite and greater than 0.")
    return args.input_dir.expanduser().resolve(), frp_hours, frp_source


def main() -> None:
    input_dir, frp_hours, frp_source = parse_args()
    input_files, missing_indices = discover_input_files(input_dir)
    output_dir = input_dir / "MI_Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_known_legacy_outputs(output_dir)

    animal_match = INPUT_FILENAME_PATTERN.fullmatch(input_files[0][1].name)
    assert animal_match is not None
    animal_prefix = animal_match.group("animal")
    files_per_cycle = int(round(frp_hours * 60.0 / FILE_DURATION_MINUTES))

    all_available_counts_by_bins = {
        n_bins: np.zeros((n_bins, len(BEHAVIORS)), dtype=np.int64)
        for n_bins in PHASE_BIN_COUNTS
    }
    behavior_counts = np.zeros(len(BEHAVIORS), dtype=np.int64)
    file_records: dict[int, dict[str, object]] = {}
    anchor_elapsed_hours: float | None = None
    anchor_row_index: int | None = None
    total_valid_samples = 0
    invalid_rows = 0

    for file_index, path in input_files:
        n_rows, row_indices, labels = read_and_classify(path)
        invalid_rows += n_rows - len(labels)
        if file_index == 0 and len(labels) == 0:
            raise ValueError(
                "File 00000 contains no classifiable rows, so the relative phase anchor cannot be defined."
            )

        file_start_hours = file_index * FILE_DURATION_MINUTES / 60.0
        if n_rows:
            raw_elapsed_hours = file_start_hours + (
                row_indices.astype(float) / n_rows * FILE_DURATION_MINUTES / 60.0
            )
        else:
            raw_elapsed_hours = np.empty(0, dtype=float)

        if file_index == 0:
            anchor_elapsed_hours = float(raw_elapsed_hours[0])
            anchor_row_index = int(row_indices[0])
        assert anchor_elapsed_hours is not None
        relative_elapsed_hours = raw_elapsed_hours - anchor_elapsed_hours
        relative_phase_hours = np.mod(relative_elapsed_hours, frp_hours)
        cycle_index = np.floor(relative_elapsed_hours / frp_hours).astype(np.int64)

        for n_bins in PHASE_BIN_COUNTS:
            bins = phase_bin_indices(
                relative_phase_hours,
                n_bins,
                frp_hours=frp_hours,
            )
            all_available_counts_by_bins[n_bins] += contingency_table(bins, labels, n_bins)

        behavior_counts += np.bincount(labels, minlength=len(BEHAVIORS))
        total_valid_samples += len(labels)
        file_records[file_index] = {
            "path": path.name,
            "n_rows": n_rows,
            "row_indices": row_indices,
            "labels": labels,
            "relative_elapsed_hours": relative_elapsed_hours,
        }

    if total_valid_samples == 0:
        raise ValueError("No valid behavioral samples were found in the input CSV files.")

    max_file_index = input_files[-1][0]
    total_duration_hours = (max_file_index + 1) * FILE_DURATION_MINUTES / 60.0

    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]] = []
    max_cycle_index = max_file_index // files_per_cycle
    for cycle_index in range(max_cycle_index + 1):
        cycle_file_indices = range(
            cycle_index * files_per_cycle,
            (cycle_index + 1) * files_per_cycle,
        )
        if any(index not in file_records for index in cycle_file_indices):
            continue
        records = [file_records[index] for index in cycle_file_indices]
        if any(int(record["labels"].size) == 0 for record in records):
            continue
        cycle_times = np.concatenate(
            [
                np.asarray(record["relative_elapsed_hours"], dtype=float)
                - cycle_index * frp_hours
                for record in records
            ]
        )
        cycle_labels = np.concatenate(
            [np.asarray(record["labels"], dtype=np.int8) for record in records]
        )
        complete_cycles.append((cycle_index, cycle_times, cycle_labels))

    if not complete_cycles:
        raise ValueError(
            f"No complete {frp_hours:g}-hour cycles are available for the permutation null."
        )

    complete_sample_count = int(
        sum(len(cycle_labels) for _, _, cycle_labels in complete_cycles)
    )
    complete_counts_by_bins = {
        n_bins: observed_counts_from_complete_cycles(
            complete_cycles,
            n_bins,
            frp_hours=frp_hours,
        )
        for n_bins in PHASE_BIN_COUNTS
    }
    all_available_mi_by_bins = {
        n_bins: mutual_information_bits(all_available_counts_by_bins[n_bins])
        for n_bins in PHASE_BIN_COUNTS
    }

    rng = np.random.default_rng(RANDOM_SEED)
    offsets_hours = rng.uniform(
        0.0,
        frp_hours,
        size=(N_PERMUTATIONS, len(complete_cycles)),
    )

    summary_rows: list[dict[str, float | int]] = []
    null_results_by_bins: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    null_tables_by_bins: dict[int, np.ndarray] = {}
    for n_bins in PHASE_BIN_COUNTS:
        null_counts = calculate_null_tables(
            complete_cycles,
            offsets_hours,
            n_bins,
            frp_hours=frp_hours,
        )
        null_tables_by_bins[n_bins] = null_counts
        metrics, null_mi, null_nmi = analyze_observed_and_null(
            complete_counts_by_bins[n_bins],
            null_counts,
            expected_sample_count=complete_sample_count,
        )
        null_results_by_bins[n_bins] = (null_mi, null_nmi)
        summary_rows.append(
            {
                "phase_bins": n_bins,
                "hours_per_bin": frp_hours / n_bins,
                **metrics,
                "MI_raw_all_available_bits": all_available_mi_by_bins[n_bins],
                "n_valid_samples": complete_sample_count,
                "n_valid_samples_all_available": total_valid_samples,
                "n_complete_cycles_used_for_null": len(complete_cycles),
                "n_permutations": N_PERMUTATIONS,
                "random_seed": RANDOM_SEED,
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("phase_bins")
    primary_metrics = summary_frame.loc[
        summary_frame["phase_bins"] == PRIMARY_PHASE_BINS
    ].iloc[0].to_dict()
    primary_null_mi, primary_null_nmi = null_results_by_bins[PRIMARY_PHASE_BINS]
    primary_null_counts = null_tables_by_bins[PRIMARY_PHASE_BINS]
    primary_counts = complete_counts_by_bins[PRIMARY_PHASE_BINS]
    decomposition_frame, null_frame, decomposition_details = calculate_primary_mi_decomposition(
        primary_counts,
        primary_null_counts,
        primary_metrics,
        primary_null_mi,
        primary_null_nmi,
        complete_sample_count,
    )

    phase_origin_sensitivity = calculate_phase_origin_sensitivity(
        complete_cycles,
        offsets_hours,
        primary_counts,
        primary_null_counts,
        primary_metrics,
        complete_sample_count,
        frp_hours=frp_hours,
    )
    origin_excess = phase_origin_sensitivity["MI_excess_bits"]
    origin_summary = {
        "phase_origin_MI_excess_min": float(origin_excess.min()),
        "phase_origin_MI_excess_max": float(origin_excess.max()),
        "phase_origin_MI_excess_mean": float(origin_excess.mean()),
        "phase_origin_MI_excess_SD": float(origin_excess.std(ddof=1)),
    }
    for column in origin_summary:
        summary_frame[column] = np.nan
        summary_frame.loc[
            summary_frame["phase_bins"] == PRIMARY_PHASE_BINS,
            column,
        ] = origin_summary[column]

    decomposition_frame.to_csv(
        output_dir / "mi_results.csv",
        index=False,
    )
    save_animal_mi_summary(
        output_dir / "animal_mi_summary.csv",
        animal_id=animal_prefix,
        frp_hours=frp_hours,
        frp_source=frp_source,
        decomposition_details=decomposition_details,
    )
    null_frame.to_csv(
        output_dir / "mi_null_distribution.csv",
        index=False,
    )
    save_phase_behavior_profile(
        primary_counts,
        output_dir / "phase_behavior_profile.csv",
        frp_hours=frp_hours,
    )
    build_sensitivity_frame(
        summary_frame,
        phase_origin_sensitivity,
    ).to_csv(
        output_dir / "mi_sensitivity.csv",
        index=False,
    )
    save_overview_plot(
        output_dir,
        primary_counts,
        primary_metrics,
        primary_null_mi,
        decomposition_details,
        frp_hours=frp_hours,
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        input_dir=input_dir,
        animal_prefix=animal_prefix,
        frp_hours=frp_hours,
        frp_source=frp_source,
        input_count=len(input_files),
        first_file_index=input_files[0][0],
        last_file_index=input_files[-1][0],
        missing_indices=missing_indices,
        total_duration_hours=total_duration_hours,
        invalid_rows=invalid_rows,
        anchor_row_index=anchor_row_index if anchor_row_index is not None else -1,
        complete_cycle_indices=[cycle_index for cycle_index, _, _ in complete_cycles],
        complete_sample_count=complete_sample_count,
        all_available_sample_count=total_valid_samples,
        decomposition_details=decomposition_details,
    )

    print("Phase x Behavior mutual-information demonstration complete.")
    print(f"FRP: {frp_hours} hours ({frp_source})")
    print(f"Input CSV files: {len(input_files)} ({input_files[0][0]} through {input_files[-1][0]})")
    print(f"Valid samples (all available): {total_valid_samples}")
    print(f"Complete-cycle samples used for MI: {complete_sample_count}")
    print(f"Complete null cycles: {[cycle_index for cycle_index, _, _ in complete_cycles]}")
    print(f"Primary 12-bin MI excess: {primary_metrics['MI_excess_bits']:.8g} bits")
    print(
        "Phase-origin MI excess range: "
        f"{origin_summary['phase_origin_MI_excess_min']:.8g} to "
        f"{origin_summary['phase_origin_MI_excess_max']:.8g} bits"
    )
    print(
        "MI decomposition errors: "
        f"observed={decomposition_details['observed_decomposition_error_bits']:.3g}, "
        f"max_null={decomposition_details['maximum_null_decomposition_error_bits']:.3g}, "
        f"excess={decomposition_details['excess_decomposition_error_bits']:.3g} bits"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
