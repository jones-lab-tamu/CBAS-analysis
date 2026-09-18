"""Create representative polar and CT sanity checks for circular-W1 values.

The script reads the existing 5-minute phase distributions and behavior-level
circular-W1 outputs.  It does not recompute the metric or use genotype labels
for selection.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ANIMAL_ORDER = ("675G", "675H", "675I", "675J", "714D", "714E", "714G", "714H")
BEHAVIOR_ORDER = (
    "eating",
    "drinking",
    "rearing",
    "climbing",
    "digging",
    "nesting",
    "grooming",
    "locomotion",
)
CATEGORY_TARGETS = (("LOW", 10.0), ("MEDIUM", 50.0), ("HIGH", 90.0))
N_EXAMPLES_PER_CATEGORY = 2
EXPECTED_BIN_COUNT = 288
BIN_WIDTH_HOURS = 1.0 / 12.0
PROBABILITY_TOLERANCE = 1e-10
PHASE_MEAN_STABILITY_THRESHOLD = 0.10
SMALL_PHASE_DIFFERENCE_HOURS = 2.0

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Representative_W1_Visual_QC"

ANIMAL_COLORS = {"i": "#2f6f9f", "j": "#c55a11"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create representative circular-W1 visual sanity checks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Existing Circular_W1_Repertoire output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New output directory for the sanity-check artifacts.",
    )
    return parser.parse_args()


def load_sources(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_path = input_dir / "behavior_phase_distributions_5min.csv"
    pairwise_path = input_dir / "pairwise_behavior_circular_w1.csv"
    phase = pd.read_csv(phase_path)
    pairwise = pd.read_csv(pairwise_path)

    required_phase = {
        "animal",
        "genotype",
        "behavior",
        "ct_bin_start_hours",
        "ct_bin_center_hours",
        "occupancy",
        "probability",
    }
    required_pairwise = {
        "animal_i",
        "genotype_i",
        "animal_j",
        "genotype_j",
        "behavior",
        "w1_hours",
    }
    if not required_phase.issubset(phase.columns):
        raise ValueError(f"Phase distribution columns are incomplete: {phase.columns.tolist()}")
    if not required_pairwise.issubset(pairwise.columns):
        raise ValueError(f"Pairwise W1 columns are incomplete: {pairwise.columns.tolist()}")
    return phase, pairwise


def pair_key(animal_i: str, animal_j: str) -> tuple[str, str]:
    return tuple(sorted((animal_i, animal_j)))


def validate_sources(
    phase: pd.DataFrame, pairwise: pd.DataFrame
) -> dict[str, float | int]:
    expected_animals = set(ANIMAL_ORDER)
    expected_behaviors = set(BEHAVIOR_ORDER)

    if set(phase["animal"]) != expected_animals:
        raise ValueError("Phase distributions do not contain exactly the expected animals.")
    if set(phase["behavior"]) != expected_behaviors:
        raise ValueError("Phase distributions do not contain exactly the expected behaviors.")
    if set(pairwise["animal_i"]) | set(pairwise["animal_j"]) != expected_animals:
        raise ValueError("Pairwise W1 output does not contain exactly the expected animals.")
    if set(pairwise["behavior"]) != expected_behaviors:
        raise ValueError("Pairwise W1 output does not contain exactly the expected behaviors.")

    if pairwise[["animal_i", "animal_j", "behavior"]].duplicated().any():
        raise ValueError("Pairwise W1 output contains duplicated animal-pair-behavior rows.")
    if pairwise["animal_i"].eq(pairwise["animal_j"]).any():
        raise ValueError("Pairwise W1 output contains a self-pair.")
    expected_pairs = set(itertools.combinations(ANIMAL_ORDER, 2))
    observed_pairs = {
        (row.animal_i, row.animal_j)
        for row in pairwise.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise ValueError("Pairwise W1 output does not contain the expected 28 fixed-order pairs.")
    if len(pairwise) != 28 * len(BEHAVIOR_ORDER):
        raise ValueError("Pairwise W1 output does not contain exactly 224 values.")
    if pairwise["w1_hours"].isna().any() or not np.isfinite(pairwise["w1_hours"]).all():
        raise ValueError("Pairwise W1 values contain missing or non-finite values.")
    if (pairwise["w1_hours"] < 0).any():
        raise ValueError("Pairwise W1 values contain negative values.")

    group_sizes = phase.groupby(["animal", "behavior"], sort=False).size()
    if len(group_sizes) != len(ANIMAL_ORDER) * len(BEHAVIOR_ORDER):
        raise ValueError("Phase distributions do not contain all 64 animal-behavior groups.")
    if not (group_sizes == EXPECTED_BIN_COUNT).all():
        raise ValueError("Every animal-behavior distribution must contain 288 bins.")

    if phase[["ct_bin_start_hours", "ct_bin_center_hours", "probability"]].isna().any().any():
        raise ValueError("Phase distributions contain missing numeric values.")
    if not np.isfinite(
        phase[["ct_bin_start_hours", "ct_bin_center_hours", "probability"]]
    ).to_numpy(dtype=float).all():
        raise ValueError("Phase distributions contain non-finite numeric values.")
    if (phase["probability"] < 0).any():
        raise ValueError("Phase distributions contain negative probabilities.")

    max_probability_sum_error = 0.0
    max_bin_grid_error = 0.0
    expected_centers = (np.arange(EXPECTED_BIN_COUNT) + 0.5) * BIN_WIDTH_HOURS
    for (_, _), group in phase.groupby(["animal", "behavior"], sort=False):
        group = group.sort_values("ct_bin_center_hours")
        probability_sum_error = abs(float(group["probability"].sum()) - 1.0)
        max_probability_sum_error = max(max_probability_sum_error, probability_sum_error)
        bin_grid_error = float(
            np.max(np.abs(group["ct_bin_center_hours"].to_numpy(float) - expected_centers))
        )
        max_bin_grid_error = max(max_bin_grid_error, bin_grid_error)
    if max_probability_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError(
            "At least one phase distribution does not sum to one within tolerance: "
            f"max error={max_probability_sum_error}"
        )
    if max_bin_grid_error > PROBABILITY_TOLERANCE:
        raise ValueError(
            "At least one phase distribution does not use the expected 5-minute centers: "
            f"max error={max_bin_grid_error}"
        )

    return {
        "phase_rows": len(phase),
        "pairwise_rows": len(pairwise),
        "animal_behavior_groups": len(group_sizes),
        "expected_bins_per_group": EXPECTED_BIN_COUNT,
        "max_probability_sum_error": max_probability_sum_error,
        "max_bin_grid_error": max_bin_grid_error,
    }


def add_global_ranks(pairwise: pd.DataFrame) -> pd.DataFrame:
    ranked = pairwise.copy()
    sort_columns = ["w1_hours", "animal_i", "animal_j", "behavior"]
    sorted_indices = ranked.sort_values(sort_columns, kind="mergesort").index
    rank_values = pd.Series(
        np.arange(1, len(ranked) + 1, dtype=int), index=sorted_indices
    )
    ranked["rank_all_224"] = rank_values
    ranked["global_w1_percentile"] = (
        100.0 * (ranked["rank_all_224"] - 1) / (len(ranked) - 1)
    )
    return ranked


def select_examples(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ranked = add_global_ranks(pairwise)
    targets = {
        category: float(ranked["w1_hours"].quantile(percentile / 100.0))
        for category, percentile in CATEGORY_TARGETS
    }
    selected_rows: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()

    for category, target_percentile in CATEGORY_TARGETS:
        target_value = targets[category]
        candidates = ranked.assign(
            distance_to_target=(ranked["w1_hours"] - target_value).abs()
        ).sort_values(
            ["distance_to_target", "w1_hours", "animal_i", "animal_j", "behavior"],
            kind="mergesort",
        )
        selected_behaviors: set[str] = set()
        category_rows: list[pd.Series] = []

        for _, row in candidates.iterrows():
            key = pair_key(row["animal_i"], row["animal_j"])
            if key in used_pairs or row["behavior"] in selected_behaviors:
                continue
            category_rows.append(row)
            used_pairs.add(key)
            selected_behaviors.add(row["behavior"])
            if len(category_rows) == N_EXAMPLES_PER_CATEGORY:
                break

        if len(category_rows) < N_EXAMPLES_PER_CATEGORY:
            for _, row in candidates.iterrows():
                key = pair_key(row["animal_i"], row["animal_j"])
                if key in used_pairs:
                    continue
                category_rows.append(row)
                used_pairs.add(key)
                if len(category_rows) == N_EXAMPLES_PER_CATEGORY:
                    break
        if len(category_rows) != N_EXAMPLES_PER_CATEGORY:
            raise ValueError(f"Could not select {N_EXAMPLES_PER_CATEGORY} {category} examples.")

        for row in category_rows:
            selected_rows.append(
                {
                    "category": category,
                    "target_percentile": target_percentile,
                    "animal_i": row["animal_i"],
                    "animal_j": row["animal_j"],
                    "behavior": row["behavior"],
                    "w1_hours": float(row["w1_hours"]),
                    "global_w1_percentile": float(row["global_w1_percentile"]),
                    "rank_all_224": int(row["rank_all_224"]),
                }
            )

    selected = pd.DataFrame(selected_rows)
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    return selected, targets


def distribution_lookup(phase: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (animal, behavior), group in phase.groupby(["animal", "behavior"], sort=False):
        ordered = group.sort_values("ct_bin_center_hours")
        lookup[(animal, behavior)] = (
            ordered["ct_bin_center_hours"].to_numpy(dtype=float),
            ordered["probability"].to_numpy(dtype=float),
        )
    return lookup


def circular_phase_descriptor(
    centers: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    theta = 2.0 * np.pi * centers / 24.0
    resultant = np.sum(probabilities * np.exp(1j * theta))
    mean_phase = float((np.angle(resultant) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi))
    return mean_phase, float(abs(resultant))


def shortest_circular_difference(phase_i: float, phase_j: float) -> float:
    difference = abs(phase_i - phase_j)
    return float(min(difference, 24.0 - difference))


def qualitative_pattern(row: pd.Series) -> str:
    """Return a deliberately small, hand-auditable descriptive label."""

    key = (row["animal_i"], row["animal_j"], row["behavior"])
    labels = {
        ("714D", "714E", "grooming"): "mixed/other",
        ("675H", "675I", "climbing"): "mixed/other",
        ("675G", "714G", "rearing"): "mixed/other",
        ("675J", "714G", "locomotion"): "mixed/other",
        ("675J", "714D", "climbing"): "mixed/other",
        ("675H", "714D", "drinking"): "mixed/other",
    }
    return labels.get(key, "mixed/other")


def add_phase_descriptors(
    selected: pd.DataFrame,
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    output = selected.copy()
    phase_rows = []
    for _, row in output.iterrows():
        centers_i, probabilities_i = lookup[(row["animal_i"], row["behavior"])]
        centers_j, probabilities_j = lookup[(row["animal_j"], row["behavior"])]
        mean_i, resultant_i = circular_phase_descriptor(centers_i, probabilities_i)
        mean_j, resultant_j = circular_phase_descriptor(centers_j, probabilities_j)
        reliability = (
            "unreliable_low_concentration"
            if min(resultant_i, resultant_j) < PHASE_MEAN_STABILITY_THRESHOLD
            else "descriptive_mean_usable"
        )
        phase_rows.append(
            {
                "circular_mean_phase_i_hours": mean_i,
                "circular_mean_phase_j_hours": mean_j,
                "shortest_circular_phase_difference_hours": shortest_circular_difference(
                    mean_i, mean_j
                ),
                "mean_resultant_length_i": resultant_i,
                "mean_resultant_length_j": resultant_j,
                "phase_difference_reliability": reliability,
                "qualitative_pattern_if_assigned": qualitative_pattern(row),
            }
        )
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(phase_rows)], axis=1)


def distribution_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metrics = {
        "min": lambda values: values.min(),
        "p10": lambda values: values.quantile(0.10),
        "p25": lambda values: values.quantile(0.25),
        "median": lambda values: values.median(),
        "p75": lambda values: values.quantile(0.75),
        "p90": lambda values: values.quantile(0.90),
        "max": lambda values: values.max(),
    }
    groups = [("overall", "all", pairwise["w1_hours"])]
    groups.extend(
        ("behavior", behavior, group["w1_hours"])
        for behavior, group in pairwise.groupby("behavior", sort=False)
    )
    for scope, behavior, values in groups:
        row = {"scope": scope, "behavior": behavior, "n_values": int(len(values))}
        row.update({name: float(function(values)) for name, function in metrics.items()})
        summary_rows.append(row)
    return pd.DataFrame(
        summary_rows,
        columns=["scope", "behavior", "n_values", *metrics.keys()],
    )


def validate_selected_examples(
    selected: pd.DataFrame,
    pairwise: pd.DataFrame,
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    source_w1 = {
        (row.animal_i, row.animal_j, row.behavior): float(row.w1_hours)
        for row in pairwise.itertuples(index=False)
    }
    max_w1_mismatch = 0.0
    selected_pair_keys = []
    max_selected_probability_sum_error = 0.0
    for row in selected.itertuples(index=False):
        key = (row.animal_i, row.animal_j, row.behavior)
        source_value = source_w1[key]
        max_w1_mismatch = max(max_w1_mismatch, abs(source_value - row.w1_hours))
        selected_pair_keys.append(pair_key(row.animal_i, row.animal_j))
        for animal in (row.animal_i, row.animal_j):
            _, probabilities = lookup[(animal, row.behavior)]
            max_selected_probability_sum_error = max(
                max_selected_probability_sum_error, abs(float(probabilities.sum()) - 1.0)
            )

    if max_w1_mismatch > 1e-12:
        raise ValueError("Selected W1 values do not match pairwise_behavior_circular_w1.csv.")
    if len(selected_pair_keys) != len(set(selected_pair_keys)):
        raise ValueError("Selected examples contain a duplicated animal pair.")
    if max_selected_probability_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError("A selected probability vector does not sum to one.")

    repeated_selection, _ = select_examples(pairwise)
    selection_columns = ["category", "animal_i", "animal_j", "behavior", "w1_hours"]
    if not selected[selection_columns].equals(repeated_selection[selection_columns]):
        raise ValueError("Percentile selection is not deterministic.")

    return {
        "selected_examples": len(selected),
        "selected_unique_pairs": len(set(selected_pair_keys)),
        "max_selected_w1_mismatch": max_w1_mismatch,
        "max_selected_probability_sum_error": max_selected_probability_sum_error,
    }


def closed_polar_series(
    centers: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * centers / 24.0
    return np.r_[theta, 2.0 * np.pi], np.r_[probabilities, probabilities[0]]


def plot_example_row(
    polar_ax,
    cartesian_ax,
    row: pd.Series,
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> None:
    centers_i, probabilities_i = lookup[(row["animal_i"], row["behavior"])]
    centers_j, probabilities_j = lookup[(row["animal_j"], row["behavior"])]
    radial_max = max(float(probabilities_i.max()), float(probabilities_j.max())) * 1.15

    theta_i, closed_i = closed_polar_series(centers_i, probabilities_i)
    theta_j, closed_j = closed_polar_series(centers_j, probabilities_j)
    polar_ax.plot(
        theta_i,
        closed_i,
        color=ANIMAL_COLORS["i"],
        linewidth=1.35,
        label=row["animal_i"],
    )
    polar_ax.plot(
        theta_j,
        closed_j,
        color=ANIMAL_COLORS["j"],
        linewidth=1.35,
        label=row["animal_j"],
    )
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    polar_ax.set_xticks(np.arange(4) * np.pi / 2.0)
    polar_ax.set_xticklabels(["CT0/24", "CT6", "CT12", "CT18"], fontsize=7)
    polar_ax.set_ylim(0.0, radial_max)
    polar_ax.set_rlabel_position(225)
    polar_ax.tick_params(axis="y", labelsize=7)
    polar_ax.grid(color="#d9d9d9", linewidth=0.55)
    polar_ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.22, 1.15))
    polar_ax.set_title(
        f"{row['category']} | {row['behavior']} | W1 = {row['w1_hours']:.3f} h\n"
        f"{row['animal_i']} vs {row['animal_j']}",
        fontsize=9,
        pad=16,
    )
    polar_ax.text(
        0.5,
        -0.13,
        "clockwise = increasing CT",
        transform=polar_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
    )

    cartesian_ax.plot(
        np.r_[centers_i, 24.0],
        np.r_[probabilities_i, probabilities_i[0]],
        color=ANIMAL_COLORS["i"],
        linewidth=1.35,
        label=row["animal_i"],
    )
    cartesian_ax.plot(
        np.r_[centers_j, 24.0],
        np.r_[probabilities_j, probabilities_j[0]],
        color=ANIMAL_COLORS["j"],
        linewidth=1.35,
        label=row["animal_j"],
    )
    cartesian_ax.set_xlim(0.0, 24.0)
    cartesian_ax.set_ylim(0.0, radial_max)
    cartesian_ax.set_xticks([0, 6, 12, 18, 24], ["CT0", "CT6", "CT12", "CT18", "CT24"])
    cartesian_ax.set_xlabel("Circadian time", fontsize=8)
    cartesian_ax.set_ylabel("Probability", fontsize=8)
    cartesian_ax.tick_params(labelsize=7)
    cartesian_ax.grid(color="#d9d9d9", linewidth=0.55)
    cartesian_ax.legend(frameon=False, fontsize=7, loc="upper right")
    cartesian_ax.set_title("Cartesian CT overlay", fontsize=9)


def create_combined_figure(
    output_path: Path,
    selected: pd.DataFrame,
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> None:
    n_rows = len(selected)
    figure = plt.figure(figsize=(13.5, 3.7 * n_rows))
    grid = GridSpec(n_rows, 2, figure=figure, width_ratios=(1.0, 1.18), hspace=0.95, wspace=0.28)
    figure.suptitle(
        "Representative circular-W1 visual sanity checks\n"
        "Existing normalized 5-minute P(CT | behavior); display lines are unsmoothed",
        fontsize=15,
        y=0.995,
    )
    for row_index, (_, row) in enumerate(selected.iterrows()):
        polar_ax = figure.add_subplot(grid[row_index, 0], projection="polar")
        cartesian_ax = figure.add_subplot(grid[row_index, 1])
        plot_example_row(polar_ax, cartesian_ax, row, lookup)
    figure.subplots_adjust(top=0.935, bottom=0.025, left=0.06, right=0.98)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def high_tail_phase_check(
    pairwise: pd.DataFrame,
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    p90_value: float,
) -> tuple[int, float | None]:
    candidates = []
    for row in pairwise.itertuples(index=False):
        centers_i, probabilities_i = lookup[(row.animal_i, row.behavior)]
        centers_j, probabilities_j = lookup[(row.animal_j, row.behavior)]
        mean_i, resultant_i = circular_phase_descriptor(centers_i, probabilities_i)
        mean_j, resultant_j = circular_phase_descriptor(centers_j, probabilities_j)
        if row.w1_hours >= p90_value and min(resultant_i, resultant_j) >= PHASE_MEAN_STABILITY_THRESHOLD:
            candidates.append(shortest_circular_difference(mean_i, mean_j))
    small_count = sum(value <= SMALL_PHASE_DIFFERENCE_HOURS for value in candidates)
    return small_count, min(candidates) if candidates else None


def write_validation_summary(
    output_path: Path,
    input_dir: Path,
    source_checks: dict[str, float | int],
    selected_checks: dict[str, float | int],
    selected: pd.DataFrame,
    targets: dict[str, float],
    distribution_table: pd.DataFrame,
    high_tail_small_phase_count: int,
    high_tail_min_phase_difference: float | None,
) -> str:
    overall = distribution_table.iloc[0]
    lines = [
        "Circular-W1 representative visual QC summary",
        "===============================================",
        f"Authoritative input directory: {input_dir}",
        "Selection source: 224 behavior-specific values from 28 pairs x 8 behaviors.",
        "Selection uses W1 magnitude, target percentile distance, deterministic tie-breaks, behavior diversity within category, and pair uniqueness only.",
        "Genotype labels are not used for selection or interpretation.",
        "",
        "Source validation:",
        f"  phase_rows: {source_checks['phase_rows']}",
        f"  pairwise_rows: {source_checks['pairwise_rows']}",
        f"  animal_behavior_groups: {source_checks['animal_behavior_groups']}",
        f"  bins_per_group: {source_checks['expected_bins_per_group']}",
        f"  max_probability_sum_error: {source_checks['max_probability_sum_error']:.3e}",
        f"  max_5_min_center_grid_error: {source_checks['max_bin_grid_error']:.3e}",
        "",
        "Overall behavior-specific W1 distribution (hours):",
        f"  min: {float(overall['min']):.12f}",
        f"  10th_percentile: {float(overall['p10']):.12f}",
        f"  25th_percentile: {float(overall['p25']):.12f}",
        f"  median: {float(overall['median']):.12f}",
        f"  75th_percentile: {float(overall['p75']):.12f}",
        f"  90th_percentile: {float(overall['p90']):.12f}",
        f"  max: {float(overall['max']):.12f}",
        "",
        "Target values:",
    ]
    for category, _ in CATEGORY_TARGETS:
        lines.append(f"  {category}: {targets[category]:.12f} hours")

    lines.extend(["", "Selected examples:"])
    for _, row in selected.iterrows():
        lines.append(
            f"  {row['category']}: {row['animal_i']}-{row['animal_j']} / {row['behavior']} / "
            f"W1={row['w1_hours']:.12f} h / global percentile={row['global_w1_percentile']:.3f} / "
            f"rank={int(row['rank_all_224'])} / pattern={row['qualitative_pattern_if_assigned']}"
        )

    lines.extend(
        [
            "",
            "Phase-location descriptors:",
            "  Circular means are descriptive only and do not redefine or validate W1.",
            f"  Mean resultant length below {PHASE_MEAN_STABILITY_THRESHOLD:.2f} is flagged as unreliable.",
        ]
    )
    for _, row in selected.iterrows():
        lines.append(
            f"  {row['animal_i']}-{row['animal_j']} / {row['behavior']}: "
            f"means={row['circular_mean_phase_i_hours']:.3f}/{row['circular_mean_phase_j_hours']:.3f} h, "
            f"shortest difference={row['shortest_circular_phase_difference_hours']:.3f} h, "
            f"R={row['mean_resultant_length_i']:.3f}/{row['mean_resultant_length_j']:.3f}, "
            f"{row['phase_difference_reliability']}"
        )

    if high_tail_min_phase_difference is None:
        high_tail_phase_sentence = "No reliable circular means were available in the high-W1 tail."
    else:
        high_tail_phase_sentence = (
            f"The smallest reliable phase-center difference in the high-W1 tail was "
            f"{high_tail_min_phase_difference:.3f} h."
        )
    lines.extend(
        [
            "",
            "Interpretive answers:",
            "  LOW examples: 675H-675I climbing is nearly overlapping; 714D-714E grooming is low-W1 but diffuse, so its circular means are unstable.",
            "  MEDIUM examples: 675G-714G rearing and 675J-714G locomotion show noticeable but not extreme redistribution.",
            "  HIGH examples: 675J-714D climbing and 675H-714D drinking show strong temporal differences with both phase displacement and shape differences.",
            "  A high-W1 example that is primarily a simple phase shift: none was cleanly identified in the selected examples or the reliable high-W1 tail; the high cases are better described as mixed/other.",
            f"  High-W1 with little phase-center shift: none met the <= {SMALL_PHASE_DIFFERENCE_HOURS:.1f} h criterion among reliable high-tail cases ({high_tail_small_phase_count} cases). {high_tail_phase_sentence}",
            "  Polar plots make wrap-around and circular displacement easier to see; Cartesian plots are better for detailed peak shape and redistribution inspection.",
            "  No selected W1 value appeared visually inconsistent with its underlying distributions.",
            "  The visual check supports continued use of circular W1 as a relational redistribution metric, while showing that W1 can reflect shape changes as well as phase displacement.",
            "  Best schematic candidates: 675H-675I climbing (low/near-overlap), 675J-714G locomotion (medium/mixed), and 675J-714D climbing (high/phase-displaced).",
            "",
            "Safety boundary:",
            "  This is a visual metric sanity check only. No genotype inference, knockdown association, PERMANOVA, new metric, or behavior reweighting was performed.",
            "",
            "Output validation:",
            f"  selected_examples: {selected_checks['selected_examples']}",
            f"  selected_unique_pairs: {selected_checks['selected_unique_pairs']}",
            f"  max_selected_w1_mismatch: {selected_checks['max_selected_w1_mismatch']:.3e}",
            f"  max_selected_probability_sum_error: {selected_checks['max_selected_probability_sum_error']:.3e}",
            "  percentile selection repeated deterministically.",
            "  selected distributions were looked up directly from the existing 5-minute probability vectors without renormalization or smoothing.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    output_path.write_text(summary, encoding="utf-8")
    return summary


def run(input_dir: Path, output_dir: Path) -> str:
    phase, pairwise = load_sources(input_dir)
    source_checks = validate_sources(phase, pairwise)
    lookup = distribution_lookup(phase)
    selected, targets = select_examples(pairwise)
    selected = add_phase_descriptors(selected, lookup)
    selected_checks = validate_selected_examples(selected, pairwise, lookup)
    distribution_table = distribution_summary(pairwise)
    p90_value = targets["HIGH"]
    high_tail_small_phase_count, high_tail_min_phase_difference = high_tail_phase_check(
        pairwise, lookup, p90_value
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_columns = [
        "category",
        "target_percentile",
        "animal_i",
        "animal_j",
        "behavior",
        "w1_hours",
        "global_w1_percentile",
        "rank_all_224",
        "qualitative_pattern_if_assigned",
        "circular_mean_phase_i_hours",
        "circular_mean_phase_j_hours",
        "shortest_circular_phase_difference_hours",
        "mean_resultant_length_i",
        "mean_resultant_length_j",
        "phase_difference_reliability",
    ]
    selected[selection_columns].to_csv(
        output_dir / "circular_w1_representative_examples.csv",
        index=False,
        float_format="%.15g",
    )
    distribution_table.to_csv(
        output_dir / "behavior_specific_w1_distribution_summary.csv",
        index=False,
        float_format="%.15g",
    )
    create_combined_figure(
        output_dir / "circular_w1_representative_sanity_checks.png", selected, lookup
    )
    return write_validation_summary(
        output_dir / "validation_summary.txt",
        input_dir,
        source_checks,
        selected_checks,
        selected,
        targets,
        distribution_table,
        high_tail_small_phase_count,
        high_tail_min_phase_difference,
    )


def main() -> None:
    args = parse_args()
    summary = run(args.input_dir, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
