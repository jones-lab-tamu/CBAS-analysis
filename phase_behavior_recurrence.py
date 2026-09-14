"""Single-animal circadian behavioral recurrence analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


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
RESTING_BINARY_STATE = 0
NONRESTING_BINARY_STATE = 1

DEFAULT_FRP_HOURS = 24.0
FILE_DURATION_MINUTES = 10.0
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260911
PRIMARY_PHASE_BINS = 48
SENSITIVITY_RESOLUTIONS = ((20, 72), (30, 48), (60, 24))
MIN_VALID_COVERAGE = 0.50
NORMALIZATION_TOLERANCE = 1e-12
INPUT_FILENAME_PATTERN = re.compile(
    r"^(?P<animal>.+)_(?P<index>\d{5})_curated_aug_model_outputs\.csv$"
)


def discover_input_files(input_dir: Path) -> tuple[list[tuple[int, Path]], list[int]]:
    """Find matching CBAS files, enforce one animal, and preserve sequence gaps."""

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )

    parsed: list[tuple[int, Path]] = []
    animals: set[str] = set()
    for path in sorted(input_dir.glob("*.csv")):
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
    return parsed, sorted(expected.difference(indices))


def read_and_classify(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Read behavior probabilities and apply the existing winner-take-all rule."""

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
        raise ValueError(
            f"Could not read required behavior columns from {path.name}: {error}"
        ) from error

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


def parse_args() -> tuple[Path, float, str]:
    parser = argparse.ArgumentParser(
        description="Run single-animal circadian behavioral recurrence analysis."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing one animal's sequential CBAS output CSV files.",
    )
    parser.add_argument(
        "--frp-hours",
        type=float,
        default=None,
        dest="frp_hours",
        help="Free-running period in hours; defaults to 24.0.",
    )
    args = parser.parse_args()
    if args.frp_hours is None:
        return args.input_dir.expanduser().resolve(), DEFAULT_FRP_HOURS, "default"
    if not np.isfinite(args.frp_hours) or args.frp_hours <= 0:
        parser.error("--frp-hours must be finite and greater than 0.")
    return args.input_dir.expanduser().resolve(), args.frp_hours, "user_supplied"


def build_binned_cycles(
    relative_times: np.ndarray,
    labels: np.ndarray,
    n_complete_cycles: int,
    n_bins: int,
    expected_samples_per_hour: float,
    frp_hours: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Build independent recurrence representations from per-bin counts."""

    if n_complete_cycles < 1:
        raise ValueError("At least one complete cycle is required for recurrence analysis.")

    cycle_indices = np.floor(relative_times / frp_hours).astype(np.int64)
    phase_fraction = np.mod(relative_times / frp_hours, 1.0)
    normalized_phase_hours = 24.0 * phase_fraction
    phase_bins = np.floor(normalized_phase_hours / 24.0 * n_bins).astype(np.int64)
    in_complete_cycles = (
        (cycle_indices >= 0)
        & (cycle_indices < n_complete_cycles)
        & (phase_bins >= 0)
        & (phase_bins < n_bins)
    )

    counts = np.zeros(
        (n_complete_cycles, n_bins, len(BEHAVIORS)),
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

    valid_counts = counts.sum(axis=2)
    expected_per_bin = expected_samples_per_hour * frp_hours / n_bins
    if expected_per_bin <= 0:
        raise ValueError("Could not determine a positive expected sample count per bin.")
    coverage = valid_counts.astype(float) / expected_per_bin
    low_coverage = coverage < MIN_VALID_COVERAGE

    # Full 9-state representation: unique maximum across all behaviors.
    full_max_counts = counts.max(axis=2)
    full_ties = (counts == full_max_counts[:, :, None]).sum(axis=2) > 1
    full_states = np.full((n_complete_cycles, n_bins), -1, dtype=np.int8)
    full_reasons = np.zeros((n_complete_cycles, n_bins), dtype=np.int8)
    full_reasons[low_coverage] = 1
    full_reasons[full_ties & ~low_coverage] = 2
    full_usable = ~low_coverage & ~full_ties
    full_states[full_usable] = np.argmax(counts, axis=2)[full_usable].astype(np.int8)

    # Rest/non-rest representation: compare resting against the summed
    # occupancy of the eight non-resting behaviors, independently of full_states.
    resting_counts = counts[:, :, RESTING_BEHAVIOR_INDEX]
    nonresting_counts = counts[:, :, list(NONRESTING_BEHAVIOR_INDICES)].sum(axis=2)
    binary_ties = resting_counts == nonresting_counts
    binary_states = np.full((n_complete_cycles, n_bins), -1, dtype=np.int8)
    binary_reasons = np.zeros((n_complete_cycles, n_bins), dtype=np.int8)
    binary_reasons[low_coverage] = 1
    binary_reasons[binary_ties & ~low_coverage] = 2
    binary_usable = ~low_coverage & ~binary_ties
    binary_states[binary_usable] = np.where(
        resting_counts[binary_usable] > nonresting_counts[binary_usable],
        RESTING_BINARY_STATE,
        NONRESTING_BINARY_STATE,
    ).astype(np.int8)

    # Conditional 8-state representation: only non-resting bins are eligible;
    # identity and ties are determined from the eight non-resting counts alone.
    nonresting_count_matrix = counts[:, :, list(NONRESTING_BEHAVIOR_INDICES)]
    conditional_max_counts = nonresting_count_matrix.max(axis=2)
    conditional_ties = (
        nonresting_count_matrix == conditional_max_counts[:, :, None]
    ).sum(axis=2) > 1
    conditional_states = np.full((n_complete_cycles, n_bins), -1, dtype=np.int8)
    conditional_reasons = np.zeros((n_complete_cycles, n_bins), dtype=np.int8)
    conditional_reasons[low_coverage] = 1
    conditional_reasons[binary_ties & ~low_coverage] = 2
    conditional_reasons[
        (binary_states == NONRESTING_BINARY_STATE)
        & conditional_ties
        & ~low_coverage
        & ~binary_ties
    ] = 3
    conditional_reasons[
        (binary_states == RESTING_BINARY_STATE) & ~low_coverage & ~binary_ties
    ] = 4
    conditional_usable = (
        ~low_coverage
        & ~binary_ties
        & (binary_states == NONRESTING_BINARY_STATE)
        & ~conditional_ties
    )
    conditional_states[conditional_usable] = np.asarray(
        NONRESTING_BEHAVIOR_INDICES,
        dtype=np.int8,
    )[np.argmax(nonresting_count_matrix, axis=2)[conditional_usable]]

    representations = {
        "full_9state": full_states,
        "rest_vs_nonrest": binary_states,
        "conditional_8state_nonrest": conditional_states,
    }
    reasons = {
        "full_9state": full_reasons,
        "rest_vs_nonrest": binary_reasons,
        "conditional_8state_nonrest": conditional_reasons,
    }
    return representations, reasons, valid_counts


def recurrence_components(
    representations: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Calculate recurrence from the independently derived representations."""

    full_first = representations["full_9state"][:-1]
    full_second = representations["full_9state"][1:]
    binary_first = representations["rest_vs_nonrest"][:-1]
    binary_second = representations["rest_vs_nonrest"][1:]
    conditional_first = representations["conditional_8state_nonrest"][:-1]
    conditional_second = representations["conditional_8state_nonrest"][1:]
    possible = int(full_first.size)
    full_valid = (full_first >= 0) & (full_second >= 0)
    binary_valid = (binary_first >= 0) & (binary_second >= 0)

    def component(
        name: str,
        eligible: np.ndarray,
        matches: np.ndarray,
    ) -> dict[str, float | int]:
        n_usable = int(eligible.sum())
        observed = (
            float(matches[eligible].mean()) if n_usable else float("nan")
        )
        usable_fraction = n_usable / possible if possible else float("nan")
        return {
            "component": name,
            "observed_recurrence": observed,
            "n_possible_comparisons": possible,
            "n_usable_comparisons": n_usable,
            "usable_fraction": usable_fraction,
        }

    conditional_valid = (conditional_first >= 0) & (conditional_second >= 0)
    conditional_eligible = (
        binary_valid
        & (binary_first == NONRESTING_BINARY_STATE)
        & (binary_second == NONRESTING_BINARY_STATE)
        & conditional_valid
    )
    return {
        "full_9state": component(
            "full_9state",
            full_valid,
            full_first == full_second,
        ),
        "rest_vs_nonrest": component(
            "rest_vs_nonrest",
            binary_valid,
            binary_first == binary_second,
        ),
        "conditional_8state_nonrest": component(
            "conditional_8state_nonrest",
            conditional_eligible,
            conditional_first == conditional_second,
        ),
    }


def calculate_null_recurrence(
    representations: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate every representation together and calculate all recurrence values."""

    n_cycles, n_bins = representations["full_9state"].shape
    null_values = np.empty((N_PERMUTATIONS, 3), dtype=float)
    conditional_eligible_counts = np.empty(N_PERMUTATIONS, dtype=np.int64)
    shifts = rng.integers(0, n_bins, size=(N_PERMUTATIONS, n_cycles))

    component_order = (
        "full_9state",
        "rest_vs_nonrest",
        "conditional_8state_nonrest",
    )
    for permutation in range(N_PERMUTATIONS):
        shifted_representations = {
            name: np.empty_like(values) for name, values in representations.items()
        }
        for cycle_index in range(n_cycles):
            shift = int(shifts[permutation, cycle_index])
            for name, values in representations.items():
                shifted_representations[name][cycle_index] = np.roll(
                    values[cycle_index],
                    shift,
                )
        components = recurrence_components(shifted_representations)
        null_values[permutation] = [
            components[name]["observed_recurrence"] for name in component_order
        ]
        conditional_eligible_counts[permutation] = int(
            components["conditional_8state_nonrest"]["n_usable_comparisons"]
        )
    return null_values, conditional_eligible_counts


def summarize_components(
    observed_components: dict[str, dict[str, float | int]],
    null_values: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Add null, excess, and normalized recurrence metrics."""

    component_order = (
        "full_9state",
        "rest_vs_nonrest",
        "conditional_8state_nonrest",
    )
    summaries: dict[str, dict[str, float | int]] = {}
    for column, name in enumerate(component_order):
        observed = float(observed_components[name]["observed_recurrence"])
        null_column = null_values[:, column]
        defined_null = np.isfinite(null_column)
        n_defined_null = int(defined_null.sum())
        null_mean = (
            float(np.mean(null_column[defined_null]))
            if n_defined_null
            else float("nan")
        )
        null_sd = (
            float(np.std(null_column[defined_null], ddof=1))
            if n_defined_null >= 2
            else float("nan")
        )
        excess = observed - null_mean
        denominator = 1.0 - null_mean
        normalized = (
            excess
            / denominator
            if np.isfinite(denominator) and abs(denominator) > NORMALIZATION_TOLERANCE
            else float("nan")
        )
        summaries[name] = {
            **observed_components[name],
            "null_mean": null_mean,
            "null_SD": null_sd,
            "recurrence_excess": excess,
            "recurrence_normalized": normalized,
            "n_defined_null_permutations": n_defined_null,
        }
    return summaries


def run_resolution(
    relative_times: np.ndarray,
    labels: np.ndarray,
    n_complete_cycles: int,
    n_bins: int,
    expected_samples_per_hour: float,
    frp_hours: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    representations, reasons, valid_counts = build_binned_cycles(
        relative_times,
        labels,
        n_complete_cycles,
        n_bins,
        expected_samples_per_hour,
        frp_hours,
    )
    observed_components = recurrence_components(representations)
    null_values, conditional_eligible_counts = calculate_null_recurrence(
        representations,
        rng,
    )
    expected_possible = (n_complete_cycles - 1) * n_bins
    for component in observed_components.values():
        if component["n_possible_comparisons"] != expected_possible:
            raise RuntimeError("Adjacent-cycle comparison count is inconsistent.")
        if not 0 <= component["n_usable_comparisons"] <= expected_possible:
            raise RuntimeError("Usable comparison count is out of bounds.")
    if conditional_eligible_counts.shape != (N_PERMUTATIONS,):
        raise RuntimeError("Conditional null denominator count is inconsistent.")
    summaries = summarize_components(observed_components, null_values)
    return {
        "representations": representations,
        "reasons": reasons,
        "valid_counts": valid_counts,
        "observed_components": observed_components,
        "summaries": summaries,
        "null_values": null_values,
        "conditional_null_eligible_counts": conditional_eligible_counts,
    }


def results_frame(summaries: dict[str, dict[str, float | int]]) -> pd.DataFrame:
    columns = [
        "component",
        "observed_recurrence",
        "null_mean",
        "null_SD",
        "recurrence_excess",
        "recurrence_normalized",
        "n_possible_comparisons",
        "n_usable_comparisons",
        "usable_fraction",
    ]
    return pd.DataFrame(
        [summaries[name] for name in (
            "full_9state",
            "rest_vs_nonrest",
            "conditional_8state_nonrest",
        )],
        columns=columns,
    )


def sensitivity_frame(
    resolution_results: dict[int, dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "bin_minutes",
        "phase_bin_count",
        "component",
        "observed_recurrence",
        "null_mean",
        "null_SD",
        "recurrence_excess",
        "recurrence_normalized",
        "n_usable_comparisons",
        "usable_fraction",
    ]
    rows: list[dict[str, float | int | str]] = []
    for bin_minutes, n_bins in SENSITIVITY_RESOLUTIONS:
        summaries = resolution_results[n_bins]["summaries"]
        for name in (
            "full_9state",
            "rest_vs_nonrest",
            "conditional_8state_nonrest",
        ):
            summary = summaries[name]
            rows.append(
                {
                    "bin_minutes": bin_minutes,
                    "phase_bin_count": n_bins,
                    "component": name,
                    "observed_recurrence": summary["observed_recurrence"],
                    "null_mean": summary["null_mean"],
                    "null_SD": summary["null_SD"],
                    "recurrence_excess": summary["recurrence_excess"],
                    "recurrence_normalized": summary["recurrence_normalized"],
                    "n_usable_comparisons": summary["n_usable_comparisons"],
                    "usable_fraction": summary["usable_fraction"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def save_overview_plot(
    output_dir: Path,
    primary_result: dict[str, object],
    primary_summaries: dict[str, dict[str, float | int]],
    n_complete_cycles: int,
) -> None:
    """Save the cycle-state map, primary null, and excess comparison."""

    states = primary_result["representations"]["full_9state"]
    null_values = primary_result["null_values"]
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    state_colors = [
        "#4c78a8",
        "#f58518",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ab",
    ]
    cmap = ListedColormap(state_colors)
    cmap.set_bad("#d9d9d9")
    masked_states = np.ma.masked_where(states < 0, states)
    image = axes[0].imshow(
        masked_states,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0, 24, 0, n_complete_cycles],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(BEHAVIORS) - 0.5,
    )
    axes[0].set_title("A. Dominant behavioral state by cycle and phase")
    axes[0].set_xlabel("Relative circadian phase (normalized hours)")
    axes[0].set_ylabel("Circadian cycle")
    axes[0].set_xticks([0, 6, 12, 18, 24])
    axes[0].set_yticks(np.arange(n_complete_cycles) + 0.5)
    axes[0].set_yticklabels(np.arange(n_complete_cycles))
    colorbar = figure.colorbar(image, ax=axes[0], ticks=np.arange(len(BEHAVIORS)))
    colorbar.ax.set_yticklabels(BEHAVIORS)
    colorbar.set_label("Usable dominant state; gray = unusable")

    full_summary = primary_summaries["full_9state"]
    axes[1].hist(
        null_values[:, 0],
        bins=40,
        color="#4c78a8",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].axvline(
        full_summary["observed_recurrence"],
        color="#d62728",
        linewidth=2,
        label=f"Observed = {full_summary['observed_recurrence']:.5g}",
    )
    axes[1].axvline(
        full_summary["null_mean"],
        color="#2ca02c",
        linewidth=2,
        label=f"Null mean = {full_summary['null_mean']:.5g}",
    )
    axes[1].set_title("B. Full 9-state recurrence null")
    axes[1].set_xlabel("Recurrence")
    axes[1].set_ylabel("Permutation count")
    axes[1].legend(fontsize=9)

    names = [
        "full 9-state",
        "rest/non-rest",
        "conditional 8-state",
    ]
    summary_order = (
        "full_9state",
        "rest_vs_nonrest",
        "conditional_8state_nonrest",
    )
    excess_values = [primary_summaries[name]["recurrence_excess"] for name in summary_order]
    bars = axes[2].bar(names, excess_values, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("C. Primary recurrence excess")
    axes[2].set_ylabel("Recurrence_Excess")
    axes[2].tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, excess_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.5g}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    axes[2].grid(axis="y", alpha=0.3)

    figure.savefig(output_dir / "recurrence_overview.png", dpi=200)
    plt.close(figure)


def write_run_summary(
    path: Path,
    input_dir: Path,
    animal_id: str,
    input_count: int,
    first_file_index: int,
    last_file_index: int,
    missing_indices: list[int],
    total_valid_samples: int,
    total_duration_hours: float,
    complete_cycle_indices: list[int],
    frp_hours: float,
    frp_source: str,
    anchor_row_index: int,
    primary_result: dict[str, object],
    primary_summaries: dict[str, dict[str, float | int]],
) -> None:
    representations = primary_result["representations"]
    reasons = primary_result["reasons"]
    conditional_null_counts = primary_result["conditional_null_eligible_counts"]
    full_summary = primary_summaries["full_9state"]
    conditional_summary = primary_summaries["conditional_8state_nonrest"]
    missing_text = "none" if not missing_indices else ", ".join(map(str, missing_indices))
    cycle_text = "none" if not complete_cycle_indices else ", ".join(map(str, complete_cycle_indices))
    observed_values = [
        float(primary_summaries[name]["observed_recurrence"])
        for name in primary_summaries
    ]
    null_values = np.asarray(primary_result["null_values"], dtype=float)
    observed_in_range = all(
        (
            np.isfinite(value)
            and 0.0 <= value <= 1.0
        )
        or (
            np.isnan(value)
            and int(primary_summaries[name]["n_usable_comparisons"]) == 0
        )
        for name, value in zip(primary_summaries, observed_values)
    )
    finite_null_values = null_values[np.isfinite(null_values)]
    null_in_range = bool(
        np.all(np.isfinite(null_values) | np.isnan(null_values))
        and np.all((finite_null_values >= 0.0) & (finite_null_values <= 1.0))
    )
    valid_reason_check = all(
        np.all(
            (representations[name] >= 0)
            | (reasons[name] != 0)
        )
        for name in representations
    )
    conditional_min = int(np.min(conditional_null_counts))
    conditional_max = int(np.max(conditional_null_counts))
    normalized_nan_components = [
        name for name, summary in primary_summaries.items()
        if not np.isfinite(float(summary["recurrence_normalized"]))
    ]
    normalized_nan_text = (
        "none" if not normalized_nan_components else ", ".join(normalized_nan_components)
    )
    undefined_null_text = "; ".join(
        f"{name}={N_PERMUTATIONS - int(summary['n_defined_null_permutations'])}"
        for name, summary in primary_summaries.items()
        if int(summary["n_defined_null_permutations"]) < N_PERMUTATIONS
    ) or "none"
    lines = [
        "Single-animal circadian behavioral recurrence run summary",
        "",
        "Input / run information",
        f"input_directory: {input_dir.resolve()}",
        f"animal_id: {animal_id}",
        f"source_file_count: {input_count}",
        f"first_source_file_index: {first_file_index}",
        f"last_source_file_index: {last_file_index}",
        f"missing_source_indices: {missing_text}",
        f"total_valid_samples: {total_valid_samples}",
        f"total_duration_hours: {total_duration_hours:.6f}",
        f"complete_cycles_used: [{cycle_text}]",
        f"FRP_hours: {frp_hours}",
        f"FRP_source: {frp_source}",
        f"phase_anchor: first valid sample of source file 00000, row index {anchor_row_index}, defines relative phase 0",
        f"primary_phase_bin_count: {PRIMARY_PHASE_BINS}",
        f"primary_bin_minutes: {24.0 * 60.0 / PRIMARY_PHASE_BINS:g}",
        f"minimum_valid_coverage: {MIN_VALID_COVERAGE:g}",
        f"permutation_count: {N_PERMUTATIONS}",
        f"random_seed: {RANDOM_SEED}",
        "",
        "Coverage / recurrence QC",
        f"total_possible_adjacent_cycle_comparisons: {full_summary['n_possible_comparisons']}",
        f"primary_9state_usable_comparisons: {full_summary['n_usable_comparisons']}",
        f"primary_9state_usable_fraction: {full_summary['usable_fraction']:.12g}",
        f"primary_conditional8_usable_comparisons: {conditional_summary['n_usable_comparisons']}",
        f"primary_conditional8_usable_fraction: {conditional_summary['usable_fraction']:.12g}",
        f"tie_bin_count: {int(np.sum(reasons['full_9state'] == 2))}",
        f"low_coverage_bin_count: {int(np.sum(reasons['full_9state'] == 1))}",
        f"rest_nonrest_binary_tie_bin_count: {int(np.sum(reasons['rest_vs_nonrest'] == 2))}",
        f"conditional8_binary_tie_bin_count: {int(np.sum(reasons['conditional_8state_nonrest'] == 2))}",
        f"conditional8_nonrest_tie_bin_count: {int(np.sum(reasons['conditional_8state_nonrest'] == 3))}",
        f"conditional8_resting_ineligible_bin_count: {int(np.sum(reasons['conditional_8state_nonrest'] == 4))}",
        f"conditional_null_eligible_comparisons_min: {conditional_min}",
        f"conditional_null_eligible_comparisons_max: {conditional_max}",
        f"normalized_recurrence_nan_components: {normalized_nan_text}",
        f"primary_undefined_null_recurrence_permutations: {undefined_null_text}",
        "",
        "Validation",
        "PASS: source CSV files were read only and remain unchanged.",
        f"PASS: all null permutations completed ({N_PERMUTATIONS}).",
        "PASS: sensitivity analyses completed at 20, 30, and 60 circadian minutes per bin.",
        f"{'PASS' if observed_in_range else 'FAIL'}: defined observed recurrence values are within [0, 1].",
        f"{'PASS' if null_in_range else 'FAIL'}: defined null recurrence values are within [0, 1].",
        f"{'PASS' if valid_reason_check else 'FAIL'}: component-specific validity and tie masks are internally consistent.",
        "PASS: recurrence comparison count consistency checks completed.",
        "PASS: conditional null denominators were recalculated after each permutation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_dir, frp_hours, frp_source = parse_args()
    input_files, missing_indices = discover_input_files(input_dir)
    animal_match = INPUT_FILENAME_PATTERN.fullmatch(input_files[0][1].name)
    assert animal_match is not None
    animal_id = animal_match.group("animal")

    file_row_counts: list[int] = []
    relative_time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    anchor_elapsed_hours: float | None = None
    anchor_row_index: int | None = None
    total_valid_samples = 0
    for file_index, path in input_files:
        n_rows, row_indices, labels = read_and_classify(path)
        file_row_counts.append(n_rows)
        if file_index == 0:
            if len(labels) == 0:
                raise ValueError(
                    "File 00000 contains no classifiable rows, so the relative phase anchor cannot be defined."
                )
            anchor_row_index = int(row_indices[0])
            anchor_elapsed_hours = float(
                file_index * FILE_DURATION_MINUTES / 60.0
                + row_indices[0] / n_rows * FILE_DURATION_MINUTES / 60.0
            )

        if n_rows == 0 or len(labels) == 0:
            continue
        assert anchor_elapsed_hours is not None
        raw_elapsed_hours = file_index * FILE_DURATION_MINUTES / 60.0 + (
            row_indices.astype(float) / n_rows * FILE_DURATION_MINUTES / 60.0
        )
        relative_time_parts.append(raw_elapsed_hours - anchor_elapsed_hours)
        label_parts.append(labels)
        total_valid_samples += len(labels)

    if total_valid_samples == 0:
        raise ValueError("No valid behavioral samples were found in the input CSV files.")
    assert anchor_row_index is not None
    relative_times = np.concatenate(relative_time_parts)
    labels = np.concatenate(label_parts).astype(np.int8, copy=False)

    max_file_index = input_files[-1][0]
    total_duration_hours = (max_file_index + 1) * FILE_DURATION_MINUTES / 60.0
    assert anchor_elapsed_hours is not None
    available_duration_hours = total_duration_hours - anchor_elapsed_hours
    n_complete_cycles = int(np.floor(available_duration_hours / frp_hours + 1e-12))
    if n_complete_cycles < 2:
        raise ValueError(
            "At least two complete circadian cycles are required for adjacent-cycle recurrence."
        )
    complete_cycle_indices = list(range(n_complete_cycles))

    nonzero_row_counts = np.asarray([count for count in file_row_counts if count > 0])
    if nonzero_row_counts.size == 0:
        raise ValueError("Could not determine the source sampling rate.")
    expected_samples_per_hour = float(
        np.median(nonzero_row_counts) / (FILE_DURATION_MINUTES / 60.0)
    )

    rng = np.random.default_rng(RANDOM_SEED)
    resolution_results: dict[int, dict[str, object]] = {}
    for _, n_bins in ((30, PRIMARY_PHASE_BINS), (20, 72), (60, 24)):
        resolution_results[n_bins] = run_resolution(
            relative_times,
            labels,
            n_complete_cycles,
            n_bins,
            expected_samples_per_hour,
            frp_hours,
            rng,
        )

    primary_result = resolution_results[PRIMARY_PHASE_BINS]
    primary_summaries = primary_result["summaries"]
    output_dir = input_dir / "Recurrence_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_frame(primary_summaries).to_csv(
        output_dir / "recurrence_results.csv",
        index=False,
    )
    primary_null_values = np.asarray(primary_result["null_values"], dtype=float)
    pd.DataFrame(
        {
            "permutation": np.arange(1, N_PERMUTATIONS + 1),
            "recurrence_9state": primary_null_values[:, 0],
            "recurrence_rest_nonrest": primary_null_values[:, 1],
            "recurrence_conditional8_nonrest": primary_null_values[:, 2],
        }
    ).to_csv(output_dir / "recurrence_null_distribution.csv", index=False)
    sensitivity_frame(resolution_results).to_csv(
        output_dir / "recurrence_sensitivity.csv",
        index=False,
    )

    primary_full = primary_summaries["full_9state"]
    primary_rest = primary_summaries["rest_vs_nonrest"]
    primary_conditional = primary_summaries["conditional_8state_nonrest"]
    pd.DataFrame(
        [
            {
                "animal_id": animal_id,
                "FRP_hours": frp_hours,
                "FRP_source": frp_source,
                "Recurrence9_Excess": primary_full["recurrence_excess"],
                "Recurrence_RestNonrest_Excess": primary_rest["recurrence_excess"],
                "Recurrence8_Conditional_Excess": primary_conditional["recurrence_excess"],
                "Recurrence9_UsableFraction": primary_full["usable_fraction"],
                "Recurrence8_Conditional_EligibleFraction": primary_conditional[
                    "usable_fraction"
                ],
            }
        ]
    ).to_csv(output_dir / "animal_recurrence_summary.csv", index=False)
    save_overview_plot(
        output_dir,
        primary_result,
        primary_summaries,
        n_complete_cycles,
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        input_dir=input_dir,
        animal_id=animal_id,
        input_count=len(input_files),
        first_file_index=input_files[0][0],
        last_file_index=input_files[-1][0],
        missing_indices=missing_indices,
        total_valid_samples=total_valid_samples,
        total_duration_hours=total_duration_hours,
        complete_cycle_indices=complete_cycle_indices,
        frp_hours=frp_hours,
        frp_source=frp_source,
        anchor_row_index=anchor_row_index,
        primary_result=primary_result,
        primary_summaries=primary_summaries,
    )

    print("Single-animal behavioral recurrence analysis complete.")
    print(f"Animal: {animal_id}")
    print(f"FRP: {frp_hours} hours ({frp_source})")
    print(f"Complete cycles: {complete_cycle_indices}")
    print(
        "Primary recurrence excess: "
        f"9-state={primary_full['recurrence_excess']:.8g}, "
        f"rest/non-rest={primary_rest['recurrence_excess']:.8g}, "
        f"conditional8={primary_conditional['recurrence_excess']:.8g}"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
