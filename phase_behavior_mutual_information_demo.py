"""Single-animal Phase x Behavior mutual-information demonstration.

This script reads the CBAS model-output CSV files in ``CBAS_Data``, assigns a
winner-take-all behavior to each classifiable row, reconstructs elapsed time,
and writes the requested MI tables, null distribution, plots, and text
summary to ``CBAS_Data\\MI_Demo_Output``.

The analysis is intentionally a focused demonstration. It uses a fixed
24-hour period and a relative phase anchor; it does not assign biological CT
or perform any genotype or recurrence analysis.
"""

from __future__ import annotations

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

FRP_HOURS = 24.0
FILE_DURATION_MINUTES = 10.0
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260911
PHASE_BIN_COUNTS = (8, 12, 24)
PRIMARY_PHASE_BINS = 12
PHASE_ORIGIN_OFFSETS_MINUTES = tuple(range(0, 120, 10))
DECOMPOSITION_TOLERANCE_BITS = 1e-12

INPUT_DIR = Path(r"C:\Users\Jeff\Documents\CBAS_Analysis_Data")
OUTPUT_DIR = INPUT_DIR / "MI_Demo_Output"
INPUT_FILENAME_PATTERN = re.compile(
    r"^(?P<animal>.+)_(?P<index>\d{5})_curated_aug_model_outputs\.csv$"
)


def discover_input_files(input_dir: Path) -> tuple[list[tuple[int, Path]], list[int]]:
    """Find and numerically sort the expected source CSV files."""

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    parsed: list[tuple[int, Path]] = []
    animals: set[str] = set()
    for path in csv_files:
        match = INPUT_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(
                "Could not determine a numeric sequence index from expected "
                f"filename pattern: {path.name}"
            )
        animals.add(match.group("animal"))
        parsed.append((int(match.group("index")), path))

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
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Assign phase values to equal-width half-open bins at a chosen origin."""

    hours_per_bin = FRP_HOURS / n_bins
    rotated_phase = np.mod(phase_hours - origin_hours, FRP_HOURS)
    safe_phase = np.minimum(rotated_phase, np.nextafter(FRP_HOURS, 0.0))
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
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Count shifted cycle behaviors for each random circular time offset.

    ``event_times`` are within-cycle times in [0, 24). For an offset d, a
    source event at time t is assigned to phase (t + d) modulo 24. Prefix
    counts and binary searches make this exact for irregular valid-row times
    without shuffling the behavioral labels.
    """

    prefix = build_prefix_counts(labels)
    n_events = len(labels)
    bin_width = FRP_HOURS / n_bins
    bin_starts = np.mod(
        origin_hours + np.arange(n_bins, dtype=float) * bin_width,
        FRP_HOURS,
    )

    starts = np.mod(bin_starts[None, :] - offsets_hours[:, None], FRP_HOURS)
    ends = starts + bin_width
    wraps = ends > FRP_HOURS

    left = np.searchsorted(event_times, starts, side="left")
    right = np.searchsorted(event_times, np.minimum(ends, FRP_HOURS), side="left")
    counts = np.empty(
        (len(offsets_hours), n_bins, len(BEHAVIORS)), dtype=np.int64
    )

    non_wrapped = ~wraps
    if non_wrapped.any():
        counts[non_wrapped] = prefix[right[non_wrapped]] - prefix[left[non_wrapped]]

    if wraps.any():
        wrapped_right = np.searchsorted(
            event_times, ends[wraps] - FRP_HOURS, side="left"
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
            origin_hours=origin_hours,
        )
    return null_counts


def observed_counts_from_complete_cycles(
    complete_cycles: list[tuple[int, np.ndarray, np.ndarray]],
    n_bins: int,
    origin_hours: float = 0.0,
) -> np.ndarray:
    """Build observed phase-by-behavior counts from exactly the null cycles."""

    counts = np.zeros((n_bins, len(BEHAVIORS)), dtype=np.int64)
    for _, event_times, labels in complete_cycles:
        phase_hours = np.mod(event_times, FRP_HOURS)
        counts += contingency_table(
            phase_bin_indices(phase_hours, n_bins, origin_hours=origin_hours),
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
                origin_hours=origin_hours,
            )
            null_counts = calculate_null_tables(
                complete_cycles,
                offsets_hours,
                PRIMARY_PHASE_BINS,
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
    complete_sample_count: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calculate and validate the 9-state MI decomposition at 12 bins."""

    # The null tables already contain intact 9-state sequences after the
    # existing circular shifts. Conditioning below therefore happens second.
    rest_observed_counts = rest_nonrest_counts(full_observed_counts)
    rest_null_counts = rest_nonrest_counts(full_null_counts)
    conditional_observed_counts = conditional_nonrest_counts(full_observed_counts)
    conditional_null_counts = conditional_nonrest_counts(full_null_counts)

    rest_metrics, rest_null_mi, _ = analyze_observed_and_null(
        rest_observed_counts,
        rest_null_counts,
        expected_sample_count=complete_sample_count,
    )
    nonrest_sample_count = int(conditional_observed_counts.sum())
    conditional_metrics, conditional_null_mi, _ = analyze_observed_and_null(
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
        }

    decomposition_frame = pd.DataFrame(
        [
            component_row("full_9state", complete_sample_count, len(BEHAVIORS), full_metrics),
            component_row("rest_vs_nonrest", complete_sample_count, 2, rest_metrics),
            component_row(
                "conditional_8state_nonrest",
                nonrest_sample_count,
                len(NONRESTING_BEHAVIOR_INDICES),
                conditional_metrics,
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
    return decomposition_frame, details


def make_phase_labels(n_bins: int) -> list[str]:
    hours_per_bin = FRP_HOURS / n_bins
    return [
        f"{int(start):02d}-{int(start + hours_per_bin):02d}"
        for start in np.arange(n_bins, dtype=float) * hours_per_bin
    ]


def save_phase_table(
    counts: np.ndarray,
    path: Path,
    probabilities: bool = False,
) -> None:
    """Save a phase table with labels and phase boundaries."""

    n_bins = counts.shape[0]
    hours_per_bin = FRP_HOURS / n_bins
    data = counts.astype(float) / counts.sum() if probabilities else counts
    table = pd.DataFrame(data, columns=BEHAVIORS)
    table.insert(0, "phase_bin", np.arange(n_bins, dtype=int))
    table.insert(1, "phase_bin_label", make_phase_labels(n_bins))
    table.insert(
        2,
        "phase_start_hours",
        np.arange(n_bins, dtype=float) * hours_per_bin,
    )
    table.insert(
        3,
        "phase_end_hours",
        (np.arange(n_bins, dtype=float) + 1) * hours_per_bin,
    )
    table.to_csv(path, index=False)


def save_plots(
    output_dir: Path,
    primary_counts: np.ndarray,
    primary_metrics: dict[str, float],
    primary_null_mi: np.ndarray,
    summary_frame: pd.DataFrame,
    behavior_counts: np.ndarray,
) -> None:
    """Create the four requested diagnostic figures."""

    primary_conditional = primary_counts / primary_counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(
        primary_conditional,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(primary_conditional),
    )
    ax.set_title("Behavior composition by relative circadian phase")
    ax.set_xlabel("Winner-take-all behavior")
    ax.set_ylabel("Relative phase bin (hours)")
    ax.set_xticks(np.arange(len(BEHAVIORS)))
    ax.set_xticklabels(BEHAVIORS, rotation=45, ha="right")
    ax.set_yticks(np.arange(PRIMARY_PHASE_BINS))
    ax.set_yticklabels(make_phase_labels(PRIMARY_PHASE_BINS))
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("P(behavior | phase)")
    fig.tight_layout()
    fig.savefig(output_dir / "phase_behavior_heatmap_12bin.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(primary_null_mi, bins=40, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(
        primary_metrics["MI_raw_bits"],
        color="#d62728",
        linewidth=2,
        label=f"Observed MI = {primary_metrics['MI_raw_bits']:.5g}",
    )
    ax.axvline(
        primary_metrics["MI_null_mean"],
        color="#2ca02c",
        linewidth=2,
        label=f"Null mean = {primary_metrics['MI_null_mean']:.5g}",
    )
    ax.set_title("12-bin circular-shift null distribution")
    ax.set_xlabel("Mutual information (bits)")
    ax.set_ylabel("Permutation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mi_null_distribution_12bin.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = summary_frame.sort_values("phase_bins")
    ax.plot(
        ordered["phase_bins"],
        ordered["MI_excess_bits"],
        marker="o",
        linewidth=2,
        color="#4c78a8",
    )
    ax.set_xticks(ordered["phase_bins"])
    ax.set_xticklabels(
        [f"{int(bins)} ({hours:g} h)" for bins, hours in zip(ordered["phase_bins"], ordered["hours_per_bin"])]
    )
    ax.set_title("MI excess sensitivity to phase-bin resolution")
    ax.set_xlabel("Phase bins (hours per bin)")
    ax.set_ylabel("MI excess (bits)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mi_sensitivity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    proportions = behavior_counts / behavior_counts.sum()
    bars = ax.bar(BEHAVIORS, proportions, color="#72b7b2")
    ax.set_title("Overall behavioral time budget (all valid samples)")
    ax.set_xlabel("Winner-take-all behavior")
    ax.set_ylabel("Proportion of valid samples")
    ax.set_ylim(0, max(0.1, float(proportions.max()) * 1.18))
    ax.tick_params(axis="x", rotation=45)
    for bar, proportion in zip(bars, proportions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{proportion:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "behavioral_time_budget.png", dpi=200)
    plt.close(fig)


def save_phase_origin_outputs(
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the phase-origin sensitivity table and figure."""

    sensitivity.to_csv(output_dir / "mi_phase_origin_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        sensitivity["phase_bin_offset_minutes"],
        sensitivity["MI_excess_bits"],
        marker="o",
        linewidth=2,
        color="#4c78a8",
    )
    ax.set_title("MI excess sensitivity to 12-bin phase origin")
    ax.set_xlabel("Phase-bin origin offset (minutes)")
    ax.set_ylabel("MI excess (bits)")
    ax.set_xticks(sensitivity["phase_bin_offset_minutes"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mi_phase_origin_sensitivity.png", dpi=200)
    plt.close(fig)


def save_decomposition_plot(
    decomposition_details: dict[str, float],
    output_dir: Path,
) -> None:
    """Plot the two additive MI contributions and the full 9-state total."""

    contribution_labels = [
        "I(P; R)",
        "P(non-rest) * I(P; B8 | non-rest)",
    ]
    contributions = [
        decomposition_details["observed_rest_nonrest_mi"],
        decomposition_details["weighted_conditional_mi"],
    ]
    full_mi = decomposition_details["observed_9state_mi"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        contribution_labels,
        contributions,
        color=["#4c78a8", "#f58518"],
        width=0.65,
    )
    ax.axhline(
        full_mi,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Full I(P; B9) = {full_mi:.5g} bits",
    )
    for bar, value in zip(bars, contributions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.5g}",
            ha="center",
            va="bottom",
        )
    ax.set_title(
        "12-bin Phase x Behavior MI decomposition\n"
        "The second bar is the weighted conditional contribution"
    )
    ax.set_ylabel("Mutual information (bits)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mi_decomposition_12bin.png", dpi=200)
    plt.close(fig)


def write_summary_text(
    path: Path,
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
    all_available_mi_by_bins: dict[int, float],
    summary_frame: pd.DataFrame,
    phase_origin_sensitivity: pd.DataFrame,
    decomposition_details: dict[str, float],
) -> None:
    """Write the requested human-readable summary."""

    primary = summary_frame.loc[summary_frame["phase_bins"] == PRIMARY_PHASE_BINS].iloc[0]
    missing_text = "none" if not missing_indices else ", ".join(map(str, missing_indices))
    cycle_text = "none" if not complete_cycle_indices else ", ".join(map(str, complete_cycle_indices))

    lines = [
        "Single-animal Phase x Behavior Mutual Information demonstration",
        "",
        f"Number of input CSV files found: {input_count}",
        f"First file index: {first_file_index}",
        f"Last file index: {last_file_index}",
        f"Missing file indices: {missing_text}",
        f"Total elapsed recording duration: {total_duration_hours:.6f} h",
        f"Total valid behavioral samples (all available): {all_available_sample_count}",
        f"Complete-cycle valid samples used for observed and null MI: {complete_sample_count}",
        f"Partial-cycle samples excluded from MI_excess: {all_available_sample_count - complete_sample_count}",
        f"Rows excluded because they could not be classified: {invalid_rows}",
        "FRP used: 24.0 h",
        f"Phase anchor: first valid sample of file 00000, row index {anchor_row_index}, defines relative phase 0",
        f"Number of complete cycles used in the null: {len(complete_cycle_indices)}",
        f"Complete cycle indices used in the null: {cycle_text}",
        "",
        "Primary 12-bin results (complete-cycle observed and null data)",
        "Observed MI and permutation-null MI used for MI_excess were calculated from the same complete circadian cycles.",
        f"MI_raw_bits: {primary['MI_raw_bits']:.12g}",
        f"MI_null_mean: {primary['MI_null_mean']:.12g}",
        f"MI_null_SD: {primary['MI_null_SD']:.12g}",
        f"MI_excess_bits: {primary['MI_excess_bits']:.12g}",
        f"MI_z: {primary['MI_z']:.12g}",
        f"H_behavior_bits: {primary['H_behavior_bits']:.12g}",
        f"NMI_raw: {primary['NMI_raw']:.12g}",
        f"NMI_excess: {primary['NMI_excess']:.12g}",
        "",
        "Primary 12-bin MI decomposition (bits; NMI values are separate and not additive)",
        f"P_rest: {decomposition_details['p_rest']:.12g}",
        f"P_nonrest: {decomposition_details['p_nonrest']:.12g}",
        f"Resting complete-cycle samples: {decomposition_details['n_resting_samples']:.0f}",
        f"Non-resting complete-cycle samples: {decomposition_details['n_nonresting_samples']:.0f}",
        f"Observed 9-state MI: {decomposition_details['observed_9state_mi']:.12g} bits",
        f"Observed rest/non-rest MI: {decomposition_details['observed_rest_nonrest_mi']:.12g} bits",
        f"Observed conditional 8-state MI: {decomposition_details['observed_conditional_8state_mi']:.12g} bits",
        f"Weighted conditional contribution (P_nonrest * conditional 8-state MI): {decomposition_details['weighted_conditional_mi']:.12g} bits",
        f"Reconstructed total (rest/non-rest MI + weighted conditional MI): {decomposition_details['reconstructed_total_mi']:.12g} bits",
        f"Observed decomposition error: {decomposition_details['observed_decomposition_error_bits']:.12g} bits",
        f"Maximum null decomposition error: {decomposition_details['maximum_null_decomposition_error_bits']:.12g} bits",
        f"9-state MI_excess: {decomposition_details['mi9_excess']:.12g} bits",
        f"Rest/non-rest MI_excess: {decomposition_details['rest_nonrest_excess']:.12g} bits",
        f"Conditional 8-state MI_excess: {decomposition_details['conditional_8state_excess']:.12g} bits",
        f"Weighted conditional excess contribution: {decomposition_details['weighted_conditional_excess']:.12g} bits",
        f"Reconstructed 9-state MI_excess: {decomposition_details['reconstructed_excess']:.12g} bits",
        f"Excess decomposition error: {decomposition_details['excess_decomposition_error_bits']:.12g} bits",
        "NMI for each component is reported separately in mi_decomposition_12bin.csv; no additive NMI decomposition is used.",
        "",
        "Optional descriptive all-available-data raw MI (not corrected by the complete-cycle null)",
    ]

    for n_bins in sorted(all_available_mi_by_bins):
        lines.append(
            f"{n_bins} bins ({FRP_HOURS / n_bins:g} h/bin): "
            f"MI_raw_all_available_bits={all_available_mi_by_bins[n_bins]:.12g}"
        )

    lines.extend(
        [
            "",
            "Phase-bin-width sensitivity results (complete-cycle observed and null data)",
        ]
    )

    for _, row in summary_frame.sort_values("phase_bins").iterrows():
        lines.append(
            f"{int(row['phase_bins'])} bins ({row['hours_per_bin']:g} h/bin): "
            f"MI_raw_bits={row['MI_raw_bits']:.12g}, "
            f"MI_null_mean={row['MI_null_mean']:.12g}, "
            f"MI_null_SD={row['MI_null_SD']:.12g}, "
            f"MI_excess_bits={row['MI_excess_bits']:.12g}, "
            f"MI_z={row['MI_z']:.12g}, "
            f"NMI_raw={row['NMI_raw']:.12g}, "
            f"NMI_null_mean={row['NMI_null_mean']:.12g}, "
            f"NMI_excess={row['NMI_excess']:.12g}"
        )

    origin_excess = phase_origin_sensitivity["MI_excess_bits"]
    lines.extend(
        [
            "",
            "Phase-bin-origin sensitivity results (12 bins, 2 h/bin, complete-cycle observed and null data)",
            f"MI_excess minimum across offsets: {origin_excess.min():.12g} bits",
            f"MI_excess maximum across offsets: {origin_excess.max():.12g} bits",
            f"MI_excess mean across offsets: {origin_excess.mean():.12g} bits",
            f"MI_excess SD across offsets: {origin_excess.std(ddof=1):.12g} bits",
        ]
    )

    for _, row in phase_origin_sensitivity.iterrows():
        lines.append(
            f"{int(row['phase_bin_offset_minutes'])} min offset: "
            f"MI_raw_bits={row['MI_raw_bits']:.12g}, "
            f"MI_null_mean={row['MI_null_mean']:.12g}, "
            f"MI_null_SD={row['MI_null_SD']:.12g}, "
            f"MI_excess_bits={row['MI_excess_bits']:.12g}, "
            f"MI_z={row['MI_z']:.12g}, "
            f"NMI_raw={row['NMI_raw']:.12g}, "
            f"NMI_null_mean={row['NMI_null_mean']:.12g}, "
            f"NMI_excess={row['NMI_excess']:.12g}"
        )

    lines.extend(
        [
            "",
            "The 0-minute phase-origin result is the standard primary 12-bin result. The origin rotation changes only the phase-bin boundaries, not the behavioral samples or circular-shift offsets.",
            "",
            "This is a single-animal demonstration analysis using a fixed 24.0-hour period and a relative phase anchor. The resulting values demonstrate the Phase × Behavior Mutual Information pipeline and should not be interpreted as a genotype effect or as an absolute biological circadian-phase measurement.",
            "",
            "Implementation notes: random seed = " + str(RANDOM_SEED),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manual_diff(
    path: Path,
    new_output_files: list[Path],
    updated_output_files: list[Path],
    missing_indices: list[int],
    invalid_rows: int,
    complete_cycle_indices: list[int],
    complete_sample_count: int,
    all_available_sample_count: int,
    phase_origin_sensitivity: pd.DataFrame,
    summary_frame: pd.DataFrame,
    validation_checks: list[str],
) -> None:
    """Write a manual before/after diff for this non-Git folder."""

    primary = summary_frame.loc[summary_frame["phase_bins"] == PRIMARY_PHASE_BINS].iloc[0]
    origin_excess = phase_origin_sensitivity["MI_excess_bits"]
    lines = [
        "MANUAL_DIFF.txt",
        "",
        "This is a manually prepared before/after diff for the focused updates.",
        "The folder is not a Git repository, so this records the semantic and output changes instead of claiming to be a Git diff.",
        "",
        f"--- {Path(__file__).resolve()} (analysis behavior)",
        "- Before: observed MI and corrected MI metrics were based on all available valid samples, while permutation-null tables were based on complete cycles.",
        f"+ After: observed MI, MI_excess_bits, MI_z, NMI_raw, and NMI_excess use the complete-cycle contingency tables ({complete_sample_count:,} samples) used by the null.",
        f"+ After: all available valid samples ({all_available_sample_count:,}) are retained separately for descriptive MI_raw_all_available_bits values only.",
        "- Before: phase-bin boundaries were fixed at the standard 0-minute origin and no origin sensitivity analysis was emitted.",
        "+ After: 12-bin phase origins are evaluated at 0, 10, 20, ..., 110 minutes; only bin boundaries rotate, while behavioral samples and circular-shift offsets remain unchanged.",
        "- Before: the summary did not distinguish the corrected complete-cycle sample basis from the optional all-available descriptive basis.",
        "+ After: the summary and CSV expose both sample counts and state that observed and null MI use identical complete circadian cycles.",
        "",
        "--- primary 12-bin result (numeric before/after)",
        "- Before: observed basis = all available valid samples (3,702,000); MI_raw_bits = 0.0917526506595; MI_excess_bits = 0.0585033610481; MI_z = 4.49866993312.",
        f"+ After: observed basis = complete cycles {complete_cycle_indices} ({complete_sample_count:,} samples); MI_raw_bits = {primary['MI_raw_bits']:.12g}; MI_excess_bits = {primary['MI_excess_bits']:.12g}; MI_z = {primary['MI_z']:.12g}.",
        f"+ After: descriptive all-available MI_raw_all_available_bits = {primary['MI_raw_all_available_bits']:.12g}; it is not used for corrected MI.",
        "",
        "--- phase-origin sensitivity",
        "+ Added a 12-row CSV covering phase-bin origins from 0 through 110 minutes in 10-minute increments.",
        "+ Added a PNG showing MI excess across those origins.",
        f"+ After: MI_excess_bits range = {origin_excess.min():.12g} to {origin_excess.max():.12g}; mean = {origin_excess.mean():.12g}; sample SD = {origin_excess.std(ddof=1):.12g}.",
        "",
        "--- output artifacts",
        "+++ Added files",
    ]
    lines.extend(f"+++ {file_path.resolve()}" for file_path in new_output_files)
    lines.extend(["", "~~~ Updated files"])
    lines.extend(f"~~~ {file_path.resolve()}" for file_path in updated_output_files)
    lines.extend(
        [
            "",
            "--- execution evidence",
            f"+ Complete cycles used: {complete_cycle_indices}.",
            f"+ Complete-cycle sample count used for observed and null MI: {complete_sample_count}.",
            f"+ All-available valid sample count: {all_available_sample_count}.",
            "+ Source CSVs were read only and were not modified.",
            "",
            "Assumptions retained:",
            "- Input files must match the expected *_NNNNN_curated_aug_model_outputs.csv pattern and share one animal prefix.",
            "- Source sequence indices are sorted numerically. Missing indices remain gaps in elapsed time and prevent affected cycles from entering the null.",
            "- Each source row is assigned time using row_index / number_of_rows_in_file across the fixed 10-minute file interval.",
            "- A row is classified when at least one required probability is finite and numeric. Non-finite cells are ignored for that row's argmax; ties use the listed behavior order.",
            "- The first valid row of file 00000 is the relative phase anchor. FRP is fixed at 24.0 hours.",
            "- Complete null cycles require every expected 10-minute source file and at least one classifiable row in each file. Partial-cycle samples are excluded from corrected MI but remain available for descriptive outputs.",
            "- Null offsets are independently sampled uniformly over continuous 0-24 hour cycle time for each complete cycle and permutation. The same fixed offsets are reused across phase-bin widths and phase origins.",
            "- Phase-origin offsets are interpreted in minutes and converted to hours modulo the fixed 24-hour period.",
            "- MI_null_SD uses the sample standard deviation (ddof=1) of the 10,000 empirical null values.",
            "",
            "Validation checks:",
        ]
    )
    lines.extend(f"- {check}" for check in validation_checks)
    lines.extend(
        [
            "",
            "Warnings and limitations encountered during implementation or execution:",
        ]
    )
    if missing_indices:
        lines.append(
            f"- Missing source sequence indices were detected: {missing_indices}. Elapsed-time gaps were preserved; affected cycles were excluded from the null."
        )
    else:
        lines.append("- No missing source sequence indices were encountered in this execution.")
    if invalid_rows:
        lines.append(
            f"- {invalid_rows} source rows were excluded because they could not be assigned a winner-take-all behavior."
        )
    else:
        lines.append("- No unclassifiable source rows were encountered in this execution.")
    lines.extend(
        [
            f"- Phase-origin MI_excess range: {phase_origin_sensitivity['MI_excess_bits'].min():.12g} to {phase_origin_sensitivity['MI_excess_bits'].max():.12g} bits; mean={phase_origin_sensitivity['MI_excess_bits'].mean():.12g}; SD={phase_origin_sensitivity['MI_excess_bits'].std(ddof=1):.12g}.",
            "- This is a single-animal demonstration with a fixed period and relative phase; it is not a biological CT estimate, genotype comparison, FRP estimate, or recurrence analysis.",
            "- No commit was created.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files, missing_indices = discover_input_files(INPUT_DIR)
    files_per_cycle = int(round(FRP_HOURS * 60.0 / FILE_DURATION_MINUTES))

    all_available_counts_by_bins = {
        n_bins: np.zeros((n_bins, len(BEHAVIORS)), dtype=np.int64)
        for n_bins in PHASE_BIN_COUNTS
    }
    behavior_counts = np.zeros(len(BEHAVIORS), dtype=np.int64)
    file_records: dict[int, dict[str, object]] = {}
    sequence_path = output_dir / "behavior_sequence.csv"
    anchor_elapsed_hours: float | None = None
    anchor_row_index: int | None = None
    total_valid_samples = 0
    invalid_rows = 0

    first_sequence_write = True
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
        relative_phase_hours = np.mod(relative_elapsed_hours, FRP_HOURS)
        cycle_index = np.floor(relative_elapsed_hours / FRP_HOURS).astype(np.int64)

        for n_bins in PHASE_BIN_COUNTS:
            bins = phase_bin_indices(relative_phase_hours, n_bins)
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

        sequence_frame = pd.DataFrame(
            {
                "file_name": path.name,
                "file_index": file_index,
                "row_index": row_indices,
                "elapsed_hours": raw_elapsed_hours,
                "relative_phase_hours": relative_phase_hours,
                "cycle_index": cycle_index,
                "behavior": [BEHAVIORS[int(label)] for label in labels],
            }
        )
        sequence_frame.to_csv(
            sequence_path,
            mode="w" if first_sequence_write else "a",
            header=first_sequence_write,
            index=False,
        )
        first_sequence_write = False

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
                - cycle_index * FRP_HOURS
                for record in records
            ]
        )
        cycle_labels = np.concatenate(
            [np.asarray(record["labels"], dtype=np.int8) for record in records]
        )
        complete_cycles.append((cycle_index, cycle_times, cycle_labels))

    if not complete_cycles:
        raise ValueError(
            "No complete 24-hour cycles are available for the permutation null."
        )

    complete_sample_count = int(
        sum(len(cycle_labels) for _, _, cycle_labels in complete_cycles)
    )
    complete_counts_by_bins = {
        n_bins: observed_counts_from_complete_cycles(complete_cycles, n_bins)
        for n_bins in PHASE_BIN_COUNTS
    }
    all_available_mi_by_bins = {
        n_bins: mutual_information_bits(all_available_counts_by_bins[n_bins])
        for n_bins in PHASE_BIN_COUNTS
    }

    rng = np.random.default_rng(RANDOM_SEED)
    offsets_hours = rng.uniform(
        0.0,
        FRP_HOURS,
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
                "hours_per_bin": FRP_HOURS / n_bins,
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
    decomposition_frame, decomposition_details = calculate_primary_mi_decomposition(
        primary_counts,
        primary_null_counts,
        primary_metrics,
        primary_null_mi,
        complete_sample_count,
    )

    phase_origin_sensitivity = calculate_phase_origin_sensitivity(
        complete_cycles,
        offsets_hours,
        primary_counts,
        primary_null_counts,
        primary_metrics,
        complete_sample_count,
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

    summary_frame.to_csv(output_dir / "mi_summary.csv", index=False)
    pd.DataFrame(
        {
            "permutation": np.arange(1, N_PERMUTATIONS + 1),
            "null_MI_bits": primary_null_mi,
            "null_NMI": primary_null_nmi,
        }
    ).to_csv(output_dir / "mi_null_distribution_12bin.csv", index=False)
    decomposition_frame.to_csv(
        output_dir / "mi_decomposition_12bin.csv",
        index=False,
    )
    save_phase_table(
        primary_counts,
        output_dir / "phase_behavior_counts_12bin.csv",
        probabilities=False,
    )
    save_phase_table(
        primary_counts,
        output_dir / "phase_behavior_probabilities_12bin.csv",
        probabilities=True,
    )
    save_plots(
        output_dir,
        primary_counts,
        primary_metrics,
        primary_null_mi,
        summary_frame,
        behavior_counts,
    )
    save_phase_origin_outputs(phase_origin_sensitivity, output_dir)
    save_decomposition_plot(decomposition_details, output_dir)
    write_summary_text(
        output_dir / "mi_summary.txt",
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
        all_available_mi_by_bins=all_available_mi_by_bins,
        summary_frame=summary_frame,
        phase_origin_sensitivity=phase_origin_sensitivity,
        decomposition_details=decomposition_details,
    )

    print("Phase x Behavior mutual-information demonstration complete.")
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
