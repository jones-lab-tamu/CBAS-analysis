"""Create representative multibehavior circular-W1 visualizations.

This standalone presentation/QC script reads the frozen phase distributions
and pairwise circular-W1 outputs.  It does not rerun classification, FRP/CT
assignment, or the upstream circular-W1 repertoire analysis.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ANIMAL_ORDER = (
    "675G",
    "675H",
    "675I",
    "675J",
    "714D",
    "714E",
    "714G",
    "714H",
)
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

PAIR_SPECS = (
    {
        "pair_id": "675H__675I",
        "animal_i": "675H",
        "animal_j": "675I",
        "class_label": "low composite distance",
    },
    {
        "pair_id": "714D__714H",
        "animal_i": "714D",
        "animal_j": "714H",
        "class_label": "high composite distance",
    },
)

BEHAVIOR_COLORS = {
    "eating": "#0072B2",
    "drinking": "#D55E00",
    "rearing": "#009E73",
    "climbing": "#CC79A7",
    "digging": "#E69F00",
    "nesting": "#56B4E9",
    "grooming": "#6A3D9A",
    "locomotion": "#4D4D4D",
}

N_PHASE_BINS = 288
N_DISPLAY_BINS = 48
BIN_WIDTH_HOURS = 1.0 / 12.0
DISPLAY_BIN_WIDTH_HOURS = 0.5
HOURS_PER_CYCLE = 24.0
PROBABILITY_TOLERANCE = 1e-12
CDF_TOLERANCE = 1e-10

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Pair_Heatmap_Visualization"

SOURCE_FILENAMES = (
    "behavior_phase_distributions_5min.csv",
    "pairwise_behavior_circular_w1.csv",
    "pairwise_repertoire_distance.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create representative pair heatmaps and centered circular-W1 CDF examples."
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
        help="New directory for pair heatmap and CDF outputs.",
    )
    return parser.parse_args()


def load_sources(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [input_dir / filename for filename in SOURCE_FILENAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required frozen circular-W1 output(s) are missing: "
            + ", ".join(str(path) for path in missing)
        )
    phase, pairwise, composite = (pd.read_csv(path) for path in paths)
    return phase, pairwise, composite


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_fixed_pair_rows(frame: pd.DataFrame, name: str) -> None:
    expected_pairs = set(itertools.combinations(ANIMAL_ORDER, 2))
    observed_pairs = {
        (str(row.animal_i), str(row.animal_j))
        for row in frame.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise ValueError(f"{name} does not contain the expected 28 fixed-order pairs.")
    if frame[["animal_i", "animal_j"]].duplicated().any():
        raise ValueError(f"{name} contains duplicate animal-pair rows.")
    if (frame["animal_i"] == frame["animal_j"]).any():
        raise ValueError(f"{name} contains a self-pair.")


def _phase_vector(
    phase: pd.DataFrame, animal: str, behavior: str
) -> np.ndarray:
    group = phase[
        (phase["animal"] == animal) & (phase["behavior"] == behavior)
    ].sort_values("ct_bin_start_hours")
    if len(group) != N_PHASE_BINS:
        raise ValueError(f"Expected 288 phase bins for {animal}, {behavior}.")
    return group["probability"].to_numpy(dtype=float)


def validate_sources(
    phase: pd.DataFrame,
    pairwise: pd.DataFrame,
    composite: pd.DataFrame,
) -> dict[str, float | int | dict[tuple[str, str, str], float]]:
    """Validate source files without changing or recalculating the metric."""

    _require_columns(
        phase,
        {
            "animal",
            "behavior",
            "ct_bin_start_hours",
            "ct_bin_center_hours",
            "probability",
        },
        SOURCE_FILENAMES[0],
    )
    _require_columns(
        pairwise,
        {"animal_i", "animal_j", "behavior", "w1_hours"},
        SOURCE_FILENAMES[1],
    )
    composite_w1_columns = {f"w1_{behavior}" for behavior in BEHAVIOR_ORDER}
    _require_columns(
        composite,
        {
            "animal_i",
            "animal_j",
            "repertoire_w1_mean_hours",
            *composite_w1_columns,
        },
        SOURCE_FILENAMES[2],
    )

    if set(phase["animal"].astype(str)) != set(ANIMAL_ORDER):
        raise ValueError("Phase distributions do not contain exactly the expected animals.")
    if set(phase["behavior"].astype(str)) != set(BEHAVIOR_ORDER):
        raise ValueError("Phase distributions do not contain exactly the expected behaviors.")
    phase_groups = phase.groupby(["animal", "behavior"], sort=False).size()
    if len(phase_groups) != len(ANIMAL_ORDER) * len(BEHAVIOR_ORDER):
        raise ValueError("Phase distributions do not contain all 64 animal-behavior groups.")
    if not (phase_groups == N_PHASE_BINS).all():
        raise ValueError("Every animal-behavior phase distribution must contain 288 bins.")

    numeric_phase = phase[["ct_bin_start_hours", "ct_bin_center_hours", "probability"]]
    if numeric_phase.isna().any().any():
        raise ValueError("Phase distributions contain missing numeric values.")
    if not np.isfinite(numeric_phase.to_numpy(dtype=float)).all():
        raise ValueError("Phase distributions contain non-finite numeric values.")
    if (phase["probability"] < 0).any():
        raise ValueError("Phase distributions contain negative probabilities.")

    expected_starts = np.arange(N_PHASE_BINS, dtype=float) * BIN_WIDTH_HOURS
    expected_centers = expected_starts + BIN_WIDTH_HOURS / 2.0
    max_probability_sum_error = 0.0
    max_phase_grid_error = 0.0
    for key, group in phase.groupby(["animal", "behavior"], sort=False):
        ordered = group.sort_values("ct_bin_start_hours")
        probability_sum_error = abs(float(ordered["probability"].sum()) - 1.0)
        max_probability_sum_error = max(max_probability_sum_error, probability_sum_error)
        start_error = float(
            np.max(
                np.abs(
                    ordered["ct_bin_start_hours"].to_numpy(dtype=float)
                    - expected_starts
                )
            )
        )
        center_error = float(
            np.max(
                np.abs(
                    ordered["ct_bin_center_hours"].to_numpy(dtype=float)
                    - expected_centers
                )
            )
        )
        grid_error = max(start_error, center_error)
        max_phase_grid_error = max(max_phase_grid_error, grid_error)
        if probability_sum_error > PROBABILITY_TOLERANCE:
            raise ValueError(f"Phase probabilities for {key} do not sum to one.")
    if max_phase_grid_error > PROBABILITY_TOLERANCE:
        raise ValueError(f"Phase-bin grid error exceeds tolerance: {max_phase_grid_error}")

    expected_pair_count = len(list(itertools.combinations(ANIMAL_ORDER, 2)))
    if len(pairwise) != expected_pair_count * len(BEHAVIOR_ORDER):
        raise ValueError("Pairwise W1 output does not contain exactly 224 rows.")
    if pairwise[["animal_i", "animal_j", "behavior"]].duplicated().any():
        raise ValueError("Pairwise W1 output contains duplicate pair-behavior rows.")
    if set(pairwise["behavior"].astype(str)) != set(BEHAVIOR_ORDER):
        raise ValueError("Pairwise W1 output does not contain the expected behaviors.")
    _validate_fixed_pair_rows(
        pairwise.drop_duplicates(subset=["animal_i", "animal_j"]),
        SOURCE_FILENAMES[1],
    )
    if pairwise["w1_hours"].isna().any() or not np.isfinite(
        pairwise["w1_hours"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Pairwise W1 output contains missing or non-finite values.")
    if (pairwise["w1_hours"] < 0).any():
        raise ValueError("Pairwise W1 output contains negative values.")

    if len(composite) != expected_pair_count:
        raise ValueError("Composite repertoire output does not contain exactly 28 rows.")
    _validate_fixed_pair_rows(composite, SOURCE_FILENAMES[2])
    composite_numeric = composite[
        list(composite_w1_columns) + ["repertoire_w1_mean_hours"]
    ]
    if composite_numeric.isna().any().any() or not np.isfinite(
        composite_numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError("Composite repertoire output contains missing or non-finite values.")
    if (composite_numeric < 0).any().any():
        raise ValueError("Composite repertoire output contains negative values.")

    pairwise_lookup = {
        (str(row.animal_i), str(row.animal_j), str(row.behavior)): float(row.w1_hours)
        for row in pairwise.itertuples(index=False)
    }
    composite_lookup = {
        (str(row.animal_i), str(row.animal_j)): row
        for row in composite.itertuples(index=False)
    }
    max_composite_mean_error = 0.0
    max_cross_file_w1_error = 0.0
    for pair_key, row in composite_lookup.items():
        components = []
        for behavior in BEHAVIOR_ORDER:
            wide_value = float(getattr(row, f"w1_{behavior}"))
            long_value = pairwise_lookup[(*pair_key, behavior)]
            max_cross_file_w1_error = max(
                max_cross_file_w1_error, abs(wide_value - long_value)
            )
            components.append(long_value)
        mean_error = abs(float(row.repertoire_w1_mean_hours) - float(np.mean(components)))
        max_composite_mean_error = max(max_composite_mean_error, mean_error)

    if max_cross_file_w1_error > PROBABILITY_TOLERANCE:
        raise ValueError("Wide and long behavior-level W1 outputs disagree.")
    if max_composite_mean_error > PROBABILITY_TOLERANCE:
        raise ValueError("Composite W1 is not the arithmetic mean of its eight components.")

    return {
        "phase_rows": int(len(phase)),
        "phase_groups": int(len(phase_groups)),
        "pairwise_rows": int(len(pairwise)),
        "composite_rows": int(len(composite)),
        "max_probability_sum_error": max_probability_sum_error,
        "max_phase_grid_error": max_phase_grid_error,
        "max_cross_file_w1_error": max_cross_file_w1_error,
        "max_composite_mean_error": max_composite_mean_error,
        "pairwise_lookup": pairwise_lookup,
        "composite_lookup": composite_lookup,
    }


def aggregate_to_30_minutes(phase: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Aggregate six existing 5-minute probability bins without smoothing."""

    rows: list[dict[str, object]] = []
    max_aggregation_error = 0.0
    max_display_sum_error = 0.0
    for animal in ANIMAL_ORDER:
        for behavior in BEHAVIOR_ORDER:
            probability = _phase_vector(phase, animal, behavior)
            for display_bin in range(N_DISPLAY_BINS):
                start = display_bin * 6
                end = start + 6
                mass = float(probability[start:end].sum())
                start_hours = display_bin * DISPLAY_BIN_WIDTH_HOURS
                rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "behavior_order_index": BEHAVIOR_ORDER.index(behavior),
                        "display_bin_index": display_bin,
                        "ct_bin_start": start_hours,
                        "ct_bin_end": start_hours + DISPLAY_BIN_WIDTH_HOURS,
                        "probability_mass_30min": mass,
                        "relative_density_vs_uniform": mass * N_DISPLAY_BINS,
                    }
                )
            display_group = rows[-N_DISPLAY_BINS:]
            display_sum = sum(float(row["probability_mass_30min"]) for row in display_group)
            max_aggregation_error = max(
                max_aggregation_error, abs(display_sum - float(probability.sum()))
            )
            max_display_sum_error = max(max_display_sum_error, abs(display_sum - 1.0))

    display = pd.DataFrame(rows)
    if not np.isfinite(
        display[["probability_mass_30min", "relative_density_vs_uniform"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("30-minute display data contain non-finite values.")
    if (display["probability_mass_30min"] < 0).any():
        raise ValueError("30-minute display data contain negative probability mass.")
    if not (display.groupby(["animal", "behavior"], sort=False).size() == N_DISPLAY_BINS).all():
        raise ValueError("Each animal-behavior display distribution must contain 48 bins.")
    if max_display_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError("30-minute aggregation does not preserve probability mass.")

    relative_density = display["relative_density_vs_uniform"].to_numpy(dtype=float)
    return display, {
        "display_rows": int(len(display)),
        "display_groups": int(
            display.groupby(["animal", "behavior"], sort=False).ngroups
        ),
        "max_aggregation_error": max_aggregation_error,
        "max_display_sum_error": max_display_sum_error,
        "relative_density_min": float(relative_density.min()),
        "relative_density_max": float(relative_density.max()),
        "relative_density_cells_above_4": int((relative_density > 4.0).sum()),
        "relative_density_cells_above_5": int((relative_density > 5.0).sum()),
    }


def selected_pair_display_data(display: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lookup = {
        (row.animal, row.behavior, int(row.display_bin_index)): row
        for row in display.itertuples(index=False)
    }
    for pair in PAIR_SPECS:
        for animal in (pair["animal_i"], pair["animal_j"]):
            for behavior in BEHAVIOR_ORDER:
                for display_bin in range(N_DISPLAY_BINS):
                    row = lookup[(animal, behavior, display_bin)]
                    rows.append(
                        {
                            "pair_id": pair["pair_id"],
                            "animal": animal,
                            "behavior": behavior,
                            "ct_bin_start": float(row.ct_bin_start),
                            "ct_bin_end": float(row.ct_bin_end),
                            "probability_mass_30min": float(row.probability_mass_30min),
                            "relative_density_vs_uniform": float(
                                row.relative_density_vs_uniform
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def selected_pair_w1_table(
    validation: dict[str, float | int | dict[tuple[str, str, str], float]]
) -> pd.DataFrame:
    pairwise_lookup = validation["pairwise_lookup"]
    composite_lookup = validation["composite_lookup"]
    rows: list[dict[str, object]] = []
    for pair in PAIR_SPECS:
        composite_value = float(
            getattr(composite_lookup[(pair["animal_i"], pair["animal_j"])], "repertoire_w1_mean_hours")
        )
        for behavior in BEHAVIOR_ORDER:
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "animal_i": pair["animal_i"],
                    "animal_j": pair["animal_j"],
                    "behavior": behavior,
                    "behavior_w1_hours": pairwise_lookup[
                        (pair["animal_i"], pair["animal_j"], behavior)
                    ],
                    "composite_repertoire_w1_hours": composite_value,
                }
            )
    return pd.DataFrame(rows)


def _pair_phase_arrays(
    phase: pd.DataFrame, animal_i: str, animal_j: str, behavior: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _phase_vector(phase, animal_i, behavior),
        _phase_vector(phase, animal_j, behavior),
    )


def centered_cdf_difference(
    probability_i: np.ndarray,
    probability_j: np.ndarray,
    cut_index: int,
) -> dict[str, np.ndarray | float | int]:
    """Return the centered circular CDF difference for one fixed CT cut."""

    first = np.asarray(probability_i, dtype=float)
    second = np.asarray(probability_j, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("CDF comparison requires equal-length one-dimensional vectors.")
    first_rotated = np.roll(first, -cut_index)
    second_rotated = np.roll(second, -cut_index)
    cumulative_difference = np.cumsum(first_rotated - second_rotated)
    optimal_constant = float(np.median(cumulative_difference))
    centered = cumulative_difference - optimal_constant
    area_hours = BIN_WIDTH_HOURS * float(np.abs(centered).sum())
    return {
        "centered": centered,
        "area_hours": area_hours,
        "optimal_constant": optimal_constant,
        "cut_index": cut_index,
        "first_rotated": first_rotated,
        "second_rotated": second_rotated,
    }


def choose_cdf_examples(
    pair_behavior: pd.DataFrame,
) -> list[dict[str, str]]:
    """Choose examples by transparent exact-W1 rules."""

    low = pair_behavior[pair_behavior["pair_id"] == "675H__675I"].copy()
    high = pair_behavior[pair_behavior["pair_id"] == "714D__714H"].copy()
    low_min = low.sort_values(["behavior_w1_hours", "behavior"], kind="mergesort").iloc[0]
    high_max = high.sort_values(
        ["behavior_w1_hours", "behavior"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    examples = [
        {
            "pair_id": str(low_min.pair_id),
            "animal_i": str(low_min.animal_i),
            "animal_j": str(low_min.animal_j),
            "behavior": str(low_min.behavior),
            "selection_rule": "minimum behavior-specific W1 within the low pair",
        },
        {
            "pair_id": str(high_max.pair_id),
            "animal_i": str(high_max.animal_i),
            "animal_j": str(high_max.animal_j),
            "behavior": str(high_max.behavior),
            "selection_rule": "maximum behavior-specific W1 within the high pair",
        },
        {
            "pair_id": "714D__714H",
            "animal_i": "714D",
            "animal_j": "714H",
            "behavior": "grooming",
            "selection_rule": "diagnostic grooming example retained because its W1 is small despite the high composite pair",
        },
    ]
    if len({(example["pair_id"], example["behavior"]) for example in examples}) != len(
        examples
    ):
        raise ValueError("CDF example selection produced duplicate examples.")
    return examples


def validate_cdf_examples(
    phase: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    examples: list[dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    source_lookup = {
        (row.pair_id, row.behavior): float(row.behavior_w1_hours)
        for row in pair_behavior.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    max_area_error = 0.0
    max_cut_rotation_error = 0.0
    max_reversal_area_error = 0.0
    max_reversal_sign_error = 0.0
    max_wrap_boundary_error = 0.0
    cut_hours_values = (0.0, 6.0, 12.0)

    for example in examples:
        probability_i, probability_j = _pair_phase_arrays(
            phase,
            example["animal_i"],
            example["animal_j"],
            example["behavior"],
        )
        source_w1 = source_lookup[(example["pair_id"], example["behavior"])]
        base = centered_cdf_difference(probability_i, probability_j, 0)
        base_centered = np.asarray(base["centered"], dtype=float)

        for cut_hours in cut_hours_values:
            cut_index = int(round(cut_hours / BIN_WIDTH_HOURS)) % N_PHASE_BINS
            result = centered_cdf_difference(probability_i, probability_j, cut_index)
            swapped = centered_cdf_difference(probability_j, probability_i, cut_index)
            centered = np.asarray(result["centered"], dtype=float)
            swapped_centered = np.asarray(swapped["centered"], dtype=float)
            rotated_base = np.roll(base_centered, -cut_index)
            rotated_first = np.asarray(result["first_rotated"], dtype=float)
            rotated_second = np.asarray(result["second_rotated"], dtype=float)
            delta = rotated_first - rotated_second

            area_error = abs(float(result["area_hours"]) - source_w1)
            cut_rotation_error = float(np.max(np.abs(centered - rotated_base)))
            reversal_area_error = abs(
                float(swapped["area_hours"]) - float(result["area_hours"])
            )
            reversal_sign_error = float(np.max(np.abs(swapped_centered + centered)))
            wrap_boundary_error = abs(
                float(centered[0] - centered[-1]) - float(delta[0])
            )
            max_area_error = max(max_area_error, area_error)
            max_cut_rotation_error = max(max_cut_rotation_error, cut_rotation_error)
            max_reversal_area_error = max(max_reversal_area_error, reversal_area_error)
            max_reversal_sign_error = max(max_reversal_sign_error, reversal_sign_error)
            max_wrap_boundary_error = max(max_wrap_boundary_error, wrap_boundary_error)

            rows.append(
                {
                    "pair": example["pair_id"],
                    "behavior": example["behavior"],
                    "cut_origin": f"CT{int(cut_hours)}",
                    "source_w1_hours": source_w1,
                    "integrated_centered_cdf_area_hours": float(result["area_hours"]),
                    "absolute_error": area_error,
                    "animal_order": "animal_i_minus_animal_j",
                    "cut_rotation_max_abs_error": cut_rotation_error,
                    "reversal_area_error_hours": reversal_area_error,
                    "reversal_curve_sign_max_abs_error": reversal_sign_error,
                    "wrap_boundary_max_abs_error": wrap_boundary_error,
                }
            )

    maximums = (
        max_area_error,
        max_cut_rotation_error,
        max_reversal_area_error,
        max_reversal_sign_error,
        max_wrap_boundary_error,
    )
    if any(value > CDF_TOLERANCE for value in maximums):
        raise RuntimeError(
            "Centered circular-CDF validation failed: "
            f"area={max_area_error}, rotation={max_cut_rotation_error}, "
            f"reversal_area={max_reversal_area_error}, "
            f"reversal_sign={max_reversal_sign_error}, "
            f"wrap_boundary={max_wrap_boundary_error}"
        )

    return pd.DataFrame(rows), {
        "cdf_examples": int(len(examples)),
        "cdf_cut_checks": int(len(rows)),
        "max_cdf_area_error": max_area_error,
        "max_cut_rotation_error": max_cut_rotation_error,
        "max_reversal_area_error": max_reversal_area_error,
        "max_reversal_sign_error": max_reversal_sign_error,
        "max_wrap_boundary_error": max_wrap_boundary_error,
    }


def _display_matrix(display: pd.DataFrame, animal: str) -> np.ndarray:
    matrix = np.zeros((len(BEHAVIOR_ORDER), N_DISPLAY_BINS), dtype=float)
    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        group = display[
            (display["animal"] == animal) & (display["behavior"] == behavior)
        ].sort_values("display_bin_index")
        if len(group) != N_DISPLAY_BINS:
            raise ValueError(f"Missing 30-minute display data for {animal}, {behavior}.")
        matrix[behavior_index, :] = group[
            "relative_density_vs_uniform"
        ].to_numpy(dtype=float)
    return matrix


def plot_heatmap(
    ax: plt.Axes,
    display: pd.DataFrame,
    animal: str,
    norm: Normalize,
    *,
    wrap_margin: bool,
) -> None:
    matrix = _display_matrix(display, animal)
    if wrap_margin:
        matrix = np.concatenate([matrix[:, -6:], matrix, matrix[:, :6]], axis=1)
        left_edge = -3.0
        tick_positions = (-3.0, 0.0, 6.0, 12.0, 18.0, 24.0, 27.0)
        tick_labels = ("CT21", "CT0", "CT6", "CT12", "CT18", "CT24", "CT3")
    else:
        left_edge = 0.0
        tick_positions = (0.0, 6.0, 12.0, 18.0, 24.0)
        tick_labels = ("CT0", "CT6", "CT12", "CT18", "CT24")

    x_edges = left_edge + np.arange(matrix.shape[1] + 1) * DISPLAY_BIN_WIDTH_HOURS
    y_edges = np.arange(len(BEHAVIOR_ORDER) + 1, dtype=float)
    ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap="viridis",
        norm=norm,
        shading="flat",
    )
    if wrap_margin:
        ax.axvspan(-3.0, 0.0, color="#FFFFFF", alpha=0.18, zorder=3)
        ax.axvspan(24.0, 27.0, color="#FFFFFF", alpha=0.18, zorder=3)
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8, zorder=4)
        ax.axvline(24.0, color="#777777", linestyle="--", linewidth=0.8, zorder=4)
        ax.text(
            -1.5,
            1.01,
            "display repeat",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
        ax.text(
            25.5,
            1.01,
            "display repeat",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(len(BEHAVIOR_ORDER), 0.0)
    ax.set_xticks(tick_positions, tick_labels, fontsize=8)
    ax.set_yticks(
        np.arange(len(BEHAVIOR_ORDER), dtype=float) + 0.5,
        BEHAVIOR_ORDER,
        fontsize=8,
    )
    ax.set_xlabel("circadian time", fontsize=9)
    ax.set_ylabel("behavior", fontsize=9)
    ax.set_title(animal, fontsize=10)
    ax.grid(axis="x", color="#FFFFFF", linewidth=0.45, alpha=0.7)
    ax.tick_params(length=0)


def plot_w1_bars(
    ax: plt.Axes,
    pair: dict[str, str],
    pair_behavior: pd.DataFrame,
    bar_limit: float,
) -> None:
    values = (
        pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]]
        .set_index("behavior")
        .loc[list(BEHAVIOR_ORDER), "behavior_w1_hours"]
        .to_numpy(dtype=float)
    )
    y_positions = np.arange(len(BEHAVIOR_ORDER), dtype=float) + 0.5
    ax.barh(
        y_positions,
        values,
        height=0.72,
        color="#4D4D4D",
        alpha=0.85,
    )
    ax.set_ylim(len(BEHAVIOR_ORDER), 0.0)
    ax.set_xlim(0.0, bar_limit)
    ax.set_yticks(y_positions, BEHAVIOR_ORDER, fontsize=8)
    ax.set_xlabel("circular W1 (hours)", fontsize=9)
    composite_value = float(
        pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]][
            "composite_repertoire_w1_hours"
        ].iloc[0]
    )
    ax.set_title(
        f"{pair['pair_id']}\ncomposite W1 = {composite_value:.6f} h",
        fontsize=9,
    )
    ax.grid(axis="x", color="#BBBBBB", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    for y_position, value in zip(y_positions, values):
        ax.text(
            value + bar_limit * 0.012,
            y_position,
            f"{value:.6f}",
            va="center",
            fontsize=7,
        )


def save_heatmap_figure(
    display: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    relative_density_max: float,
    output_path: Path,
    *,
    wrap_margin: bool,
) -> None:
    figure = plt.figure(figsize=(16, 9.3))
    grid = GridSpec(
        len(PAIR_SPECS),
        3,
        figure=figure,
        width_ratios=(1.35, 1.35, 1.0),
        hspace=0.62,
        wspace=0.35,
    )
    norm = Normalize(vmin=0.0, vmax=relative_density_max)
    heatmap_axes: list[plt.Axes] = []
    bar_limit = float(pair_behavior["behavior_w1_hours"].max()) * 1.22

    for row_index, pair in enumerate(PAIR_SPECS):
        axis_i = figure.add_subplot(grid[row_index, 0])
        axis_j = figure.add_subplot(grid[row_index, 1])
        axis_bar = figure.add_subplot(grid[row_index, 2])
        plot_heatmap(axis_i, display, pair["animal_i"], norm, wrap_margin=wrap_margin)
        plot_heatmap(axis_j, display, pair["animal_j"], norm, wrap_margin=wrap_margin)
        plot_w1_bars(axis_bar, pair, pair_behavior, bar_limit)
        heatmap_axes.extend([axis_i, axis_j])

    title = (
        "Representative pair behavior × CT probability heatmaps"
        if not wrap_margin
        else "Representative pair behavior × CT heatmaps with repeated wrap margins"
    )
    subtitle = (
        "30-minute display bins; relative density is P(CT bin | behavior) divided by uniform 1/48"
        if not wrap_margin
        else "Shaded CT21–24 and CT0–3 margins repeat display data only"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=13, y=0.98)
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=heatmap_axes,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("relative density vs uniform", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.text(
        0.5,
        0.018,
        "Circular W1 was calculated from the original 5-minute distributions; 30-minute aggregation is for visualization only.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(top=0.88, bottom=0.10, left=0.055, right=0.91)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_cdf_figure(
    phase: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    examples: list[dict[str, str]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(examples),
        1,
        figsize=(13, 9.5),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    source_lookup = {
        (row.pair_id, row.behavior): float(row.behavior_w1_hours)
        for row in pair_behavior.itertuples(index=False)
    }
    max_abs_centered = 0.0
    results: list[tuple[plt.Axes, dict[str, str], dict[str, np.ndarray | float | int]]] = []
    for ax, example in zip(axes, examples):
        probability_i, probability_j = _pair_phase_arrays(
            phase,
            example["animal_i"],
            example["animal_j"],
            example["behavior"],
        )
        result = centered_cdf_difference(probability_i, probability_j, 0)
        centered = np.asarray(result["centered"], dtype=float)
        max_abs_centered = max(max_abs_centered, float(np.max(np.abs(centered))))
        results.append((ax, example, result))

    y_limit = max_abs_centered * 1.18 if max_abs_centered > 0 else 1.0
    x_edges = np.arange(N_PHASE_BINS + 1, dtype=float) * BIN_WIDTH_HOURS
    for ax, example, result in results:
        centered = np.asarray(result["centered"], dtype=float)
        y_step = np.r_[centered, centered[-1]]
        color = BEHAVIOR_COLORS[example["behavior"]]
        ax.fill_between(
            x_edges,
            np.zeros_like(y_step),
            y_step,
            step="post",
            color=color,
            alpha=0.25,
        )
        ax.step(
            x_edges,
            y_step,
            where="post",
            color=color,
            linewidth=1.3,
        )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlim(0.0, HOURS_PER_CYCLE)
        ax.set_ylabel("centered\nCDF difference", fontsize=8)
        source_w1 = source_lookup[(example["pair_id"], example["behavior"])]
        ax.set_title(
            f"{example['pair_id']} | {example['behavior']} | "
            f"source W1 = {source_w1:.6f} h",
            fontsize=10,
        )
        ax.text(
            0.99,
            0.88,
            f"shaded area = {float(result['area_hours']):.6f} h\n"
            f"selection: {example['selection_rule']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#444444",
        )
        ax.grid(axis="x", color="#BBBBBB", linewidth=0.5, alpha=0.6)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xticks(
        [0.0, 6.0, 12.0, 18.0, 24.0],
        ["CT0", "CT6", "CT12", "CT18", "CT24"],
    )
    axes[-1].set_xlabel("circadian time", fontsize=9)
    figure.suptitle(
        "Centered circular CDF-difference examples\n"
        "D_centered(t) = D(t) − median(D); shaded absolute area is circular W1",
        fontsize=13,
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Curves use the original 5-minute distributions and a CT0 cut; each row is centered around zero.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.97, hspace=0.58)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _format_behavior_values(
    pair: dict[str, str], pair_behavior: pd.DataFrame
) -> list[str]:
    subset = pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]].set_index("behavior")
    return [
        f"  - {behavior}: {float(subset.loc[behavior, 'behavior_w1_hours']):.17g} h"
        for behavior in BEHAVIOR_ORDER
    ]


def write_run_summary(
    output_path: Path,
    input_dir: Path,
    output_dir: Path,
    validation: dict[str, float | int | dict[tuple[str, str, str], float]],
    display_stats: dict[str, float | int],
    pair_behavior: pd.DataFrame,
    examples: list[dict[str, str]],
    cdf_validation: dict[str, float | int],
) -> None:
    composite_lookup = validation["composite_lookup"]
    lines = [
        "Representative multibehavior circular-W1 pair visualization",
        "===========================================================",
        "",
        f"Input directory: {input_dir}",
        f"Output directory: {output_dir}",
        "",
        "Files read:",
        *[f"- {input_dir / filename}" for filename in SOURCE_FILENAMES],
        "",
        "Scientific/data handling:",
        "- Circular W1 was calculated from the original 5-minute distributions; 30-minute aggregation is for visualization only.",
        "- Each behavior row is independently normalized as P(CT | behavior); no abundance or support weighting is used.",
        "- One common sequential intensity scale is used for the full cohort; no per-row rescaling or genotype-dependent scaling is used.",
        "",
        "Representative pairs and exact composite W1:",
    ]
    for pair in PAIR_SPECS:
        composite_value = float(
            getattr(composite_lookup[(pair["animal_i"], pair["animal_j"])], "repertoire_w1_mean_hours")
        )
        lines.append(
            f"- {pair['pair_id']} ({pair['class_label']}): {composite_value:.17g} h"
        )
    lines.extend(["", "Exact behavior-specific W1 values:"])
    for pair in PAIR_SPECS:
        lines.append(f"- {pair['pair_id']}:")
        lines.extend(_format_behavior_values(pair, pair_behavior))

    relative_min = float(display_stats["relative_density_min"])
    relative_max = float(display_stats["relative_density_max"])
    total_display_cells = int(display_stats["display_rows"])
    lines.extend(
        [
            "",
            "30-minute display scale:",
            f"- observed full-cohort relative-density range: {relative_min:.17g} to {relative_max:.17g}",
            "- display cap: none; all observed cells use the common absolute scale",
            f"- cells clipped: 0 of {total_display_cells} (0.0%)",
            f"- maximum aggregation-versus-5-minute mass error: {float(display_stats['max_aggregation_error']):.3g}",
            f"- maximum 30-minute row-sum error: {float(display_stats['max_display_sum_error']):.3g}",
            "",
            "Visual interpretation:",
            "- The simple CT0–24 heatmap is interpretable because each behavior row shows the complete within-behavior temporal probability profile on a fixed CT axis.",
            "- The wrap-margin version modestly improves continuity for patterns near CT0/CT24; its shaded margins are duplicated display data, not new observations.",
            "- The low pair visually shows broadly similar multibehavior phase organization, with nesting the largest behavior-specific W1 contribution.",
            "- The high pair visually shows broader redistribution across several behavior rows, led by climbing, rearing, and eating W1 contributions.",
            "- Rectangular heatmaps are more faithful than center-plus-spread glyphs for showing the complete distributions being compared.",
            "",
            "Centered CDF validation:",
            "- CDF examples were selected transparently: minimum-W1 behavior from the low pair, maximum-W1 behavior from the high pair, and high-pair grooming as a diagnostic small-W1 example.",
            f"- examples: {cdf_validation['cdf_examples']}; cut checks: {cdf_validation['cdf_cut_checks']}",
            f"- maximum source-W1 versus shaded-area error: {float(cdf_validation['max_cdf_area_error']):.3g} h",
            f"- maximum CT0/CT6/CT12 rotation error: {float(cdf_validation['max_cut_rotation_error']):.3g}",
            f"- maximum animal-order reversal area error: {float(cdf_validation['max_reversal_area_error']):.3g} h",
            f"- maximum animal-order reversal sign error: {float(cdf_validation['max_reversal_sign_error']):.3g}",
            f"- maximum wrap-boundary closure error: {float(cdf_validation['max_wrap_boundary_error']):.3g}",
            "- centered CDF curves use D(t) minus a valid median constant and the zero line; no raw pointwise probability difference is used as the primary explanation.",
            "- The CDF panel is sufficiently understandable as a metric-explanation prototype, with source W1 and integrated shaded area printed for each example.",
            "",
            "Validation:",
            f"- source phase rows: {validation['phase_rows']}; source animal-behavior groups: {validation['phase_groups']}",
            f"- source behavior-W1 rows: {validation['pairwise_rows']}; source composite rows: {validation['composite_rows']}",
            f"- maximum source probability-sum error: {float(validation['max_probability_sum_error']):.3g}",
            f"- maximum source grid error: {float(validation['max_phase_grid_error']):.3g} h",
            f"- maximum wide/long behavior-W1 difference: {float(validation['max_cross_file_w1_error']):.3g} h",
            f"- maximum composite-versus-eight-value mean error: {float(validation['max_composite_mean_error']):.3g} h",
            "- no raw data or upstream circular-W1 output was modified",
            "",
            "Generated files:",
            "- representative_pair_heatmaps_simple.png",
            "- representative_pair_heatmaps_wrap_margin.png",
            "- representative_pair_heatmap_data.csv",
            "- representative_pair_behavior_w1.csv",
            "- circular_w1_cdf_examples.png",
            "- circular_w1_cdf_validation.csv",
            "- run_summary.txt",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    phase, pairwise, composite = load_sources(input_dir)
    validation = validate_sources(phase, pairwise, composite)
    display, display_stats = aggregate_to_30_minutes(phase)
    pair_heatmap_data = selected_pair_display_data(display)
    pair_behavior = selected_pair_w1_table(validation)
    cdf_examples = choose_cdf_examples(pair_behavior)
    cdf_validation_table, cdf_validation = validate_cdf_examples(
        phase, pair_behavior, cdf_examples
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_heatmap_data.to_csv(
        output_dir / "representative_pair_heatmap_data.csv",
        index=False,
        float_format="%.17g",
    )
    pair_behavior.to_csv(
        output_dir / "representative_pair_behavior_w1.csv",
        index=False,
        float_format="%.17g",
    )
    cdf_validation_table.to_csv(
        output_dir / "circular_w1_cdf_validation.csv",
        index=False,
        float_format="%.17g",
    )

    save_heatmap_figure(
        display,
        pair_behavior,
        float(display_stats["relative_density_max"]),
        output_dir / "representative_pair_heatmaps_simple.png",
        wrap_margin=False,
    )
    save_heatmap_figure(
        display,
        pair_behavior,
        float(display_stats["relative_density_max"]),
        output_dir / "representative_pair_heatmaps_wrap_margin.png",
        wrap_margin=True,
    )
    save_cdf_figure(
        phase,
        pair_behavior,
        cdf_examples,
        output_dir / "circular_w1_cdf_examples.png",
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        input_dir,
        output_dir,
        validation,
        display_stats,
        pair_behavior,
        cdf_examples,
        cdf_validation,
    )

    print(f"Wrote pair heatmap outputs to {output_dir}")
    for pair in PAIR_SPECS:
        composite_value = float(
            pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]][
                "composite_repertoire_w1_hours"
            ].iloc[0]
        )
        print(f"{pair['pair_id']}: composite_w1={composite_value:.6f} h")
    print(
        "relative_density_range="
        f"{float(display_stats['relative_density_min']):.6f}-"
        f"{float(display_stats['relative_density_max']):.6f}"
    )
    print(
        "cdf_validation_max_area_error="
        f"{float(cdf_validation['max_cdf_area_error']):.3g} h"
    )


if __name__ == "__main__":
    main()
