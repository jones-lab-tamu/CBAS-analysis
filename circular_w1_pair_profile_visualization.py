"""Create stacked filled phase-distribution profiles for frozen W1 pairs.

This standalone presentation/QC script reads the existing 5-minute phase
distributions and pairwise circular-W1 outputs.  It does not rerun any
upstream classification, FRP/CT assignment, or circular-W1 analysis.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
MOCKUP_BEHAVIOR_LABELS = (
    "Eating",
    "Drinking",
    "Rearing",
    "Climbing",
    "Digging",
    "Nesting",
    "Grooming",
    "Exploring",
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

STANDALONE_PAIR_SPECS = (
    {
        "pair_id": "675I__675J",
        "animal_i": "675I",
        "animal_j": "675J",
        "class_label": "low composite distance",
    },
    {
        "pair_id": "675H__714G",
        "animal_i": "675H",
        "animal_j": "714G",
        "class_label": "medium composite distance",
    },
    PAIR_SPECS[1],
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
LANE_GAP_FRACTION = 0.18

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Pair_Profile_Visualization"

SOURCE_FILENAMES = (
    "behavior_phase_distributions_5min.csv",
    "pairwise_behavior_circular_w1.csv",
    "pairwise_repertoire_distance.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked filled phase-distribution profiles for representative circular-W1 pairs."
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
        help="New directory for profile outputs.",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_sources(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [input_dir / filename for filename in SOURCE_FILENAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required frozen circular-W1 output(s) are missing: "
            + ", ".join(str(path) for path in missing)
        )
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def _phase_vector(phase: pd.DataFrame, animal: str, behavior: str) -> np.ndarray:
    group = phase[
        (phase["animal"] == animal) & (phase["behavior"] == behavior)
    ].sort_values("ct_bin_start_hours")
    if len(group) != N_PHASE_BINS:
        raise ValueError(f"Expected 288 phase bins for {animal}, {behavior}.")
    return group["probability"].to_numpy(dtype=float)


def _validate_pair_rows(frame: pd.DataFrame, name: str) -> None:
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


def validate_sources(
    phase: pd.DataFrame,
    pairwise: pd.DataFrame,
    composite: pd.DataFrame,
) -> dict[str, object]:
    """Validate frozen inputs and return source lookups without changing them."""

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
    _require_columns(
        composite,
        {
            "animal_i",
            "animal_j",
            "repertoire_w1_mean_hours",
            *(f"w1_{behavior}" for behavior in BEHAVIOR_ORDER),
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

    numeric_phase = phase[
        ["ct_bin_start_hours", "ct_bin_center_hours", "probability"]
    ]
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
        max_phase_grid_error = max(
            max_phase_grid_error, start_error, center_error
        )
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
    _validate_pair_rows(
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
    _validate_pair_rows(composite, SOURCE_FILENAMES[2])
    composite_columns = list(
        [f"w1_{behavior}" for behavior in BEHAVIOR_ORDER]
        + ["repertoire_w1_mean_hours"]
    )
    composite_numeric = composite[composite_columns]
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
    max_cross_file_w1_error = 0.0
    max_composite_mean_error = 0.0
    for pair_key, row in composite_lookup.items():
        components = []
        for behavior in BEHAVIOR_ORDER:
            wide_value = float(getattr(row, f"w1_{behavior}"))
            long_value = pairwise_lookup[(*pair_key, behavior)]
            max_cross_file_w1_error = max(
                max_cross_file_w1_error, abs(wide_value - long_value)
            )
            components.append(long_value)
        mean_error = abs(
            float(row.repertoire_w1_mean_hours) - float(np.mean(components))
        )
        max_composite_mean_error = max(max_composite_mean_error, mean_error)

    if max_cross_file_w1_error > PROBABILITY_TOLERANCE:
        raise ValueError("Wide and long behavior-level W1 outputs disagree.")
    if max_composite_mean_error > PROBABILITY_TOLERANCE:
        raise ValueError("Composite W1 is not the arithmetic mean of its eight components.")

    for pair in STANDALONE_PAIR_SPECS:
        pair_key = (pair["animal_i"], pair["animal_j"])
        if pair_key not in composite_lookup:
            raise ValueError(f"Selected pair is missing from composite output: {pair_key}")
        for behavior in BEHAVIOR_ORDER:
            if (*pair_key, behavior) not in pairwise_lookup:
                raise ValueError(f"Selected pair is missing W1 for {behavior}: {pair_key}")

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


def aggregate_to_30_minutes(
    phase: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
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
                bin_start = display_bin * DISPLAY_BIN_WIDTH_HOURS
                mass = float(probability[start:end].sum())
                rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "behavior_order_index": BEHAVIOR_ORDER.index(behavior),
                        "display_bin_index": display_bin,
                        "ct_bin_start": bin_start,
                        "ct_bin_end": bin_start + DISPLAY_BIN_WIDTH_HOURS,
                        "ct_bin_center": bin_start + DISPLAY_BIN_WIDTH_HOURS / 2.0,
                        "probability_mass_30min": mass,
                        "relative_density_vs_uniform": mass * N_DISPLAY_BINS,
                    }
                )
            display_group = rows[-N_DISPLAY_BINS:]
            display_sum = sum(
                float(row["probability_mass_30min"]) for row in display_group
            )
            max_aggregation_error = max(
                max_aggregation_error, abs(display_sum - float(probability.sum()))
            )
            max_display_sum_error = max(max_display_sum_error, abs(display_sum - 1.0))

    display = pd.DataFrame(rows)
    probability_columns = [
        "probability_mass_30min",
        "relative_density_vs_uniform",
    ]
    if not np.isfinite(display[probability_columns].to_numpy(dtype=float)).all():
        raise ValueError("30-minute display data contain non-finite values.")
    if (display["probability_mass_30min"] < 0).any():
        raise ValueError("30-minute display data contain negative probability mass.")
    if not (
        display.groupby(["animal", "behavior"], sort=False).size() == N_DISPLAY_BINS
    ).all():
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
    }


def selected_profile_data(display: pd.DataFrame) -> pd.DataFrame:
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
                            "ct_bin_center": float(row.ct_bin_center),
                            "probability_mass_30min": float(
                                row.probability_mass_30min
                            ),
                            "relative_density_vs_uniform": float(
                                row.relative_density_vs_uniform
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def selected_w1_table(
    validation: dict[str, object],
    pair_specs: tuple[dict[str, str], ...] = PAIR_SPECS,
) -> pd.DataFrame:
    pairwise_lookup = validation["pairwise_lookup"]
    composite_lookup = validation["composite_lookup"]
    assert isinstance(pairwise_lookup, dict)
    assert isinstance(composite_lookup, dict)

    rows: list[dict[str, object]] = []
    for pair in pair_specs:
        composite_value = float(
            getattr(
                composite_lookup[(pair["animal_i"], pair["animal_j"])],
                "repertoire_w1_mean_hours",
            )
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


def profile_matrix(display: pd.DataFrame, animal: str) -> np.ndarray:
    matrix = np.zeros((len(BEHAVIOR_ORDER), N_DISPLAY_BINS), dtype=float)
    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        group = display[
            (display["animal"] == animal) & (display["behavior"] == behavior)
        ].sort_values("display_bin_index")
        if len(group) != N_DISPLAY_BINS:
            raise ValueError(f"Missing display data for {animal}, {behavior}.")
        matrix[behavior_index, :] = group[
            "relative_density_vs_uniform"
        ].to_numpy(dtype=float)
    return matrix


def lane_layout(relative_density_max: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    lane_height = float(relative_density_max)
    if lane_height <= 0:
        raise ValueError("Relative-density maximum must be positive.")
    pitch = lane_height * (1.0 + LANE_GAP_FRACTION)
    y_max = len(BEHAVIOR_ORDER) * pitch
    baselines = np.array(
        [(len(BEHAVIOR_ORDER) - 1 - index) * pitch for index in range(len(BEHAVIOR_ORDER))],
        dtype=float,
    )
    row_centers = baselines + lane_height / 2.0
    return lane_height, y_max, baselines, row_centers


def display_coordinates(*, wrap_margin: bool) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, ...], tuple[str, ...]]:
    if wrap_margin:
        left_edge = -3.0
        edges = left_edge + np.arange(61, dtype=float) * DISPLAY_BIN_WIDTH_HOURS
        centers = left_edge + DISPLAY_BIN_WIDTH_HOURS / 2.0 + np.arange(
            60, dtype=float
        ) * DISPLAY_BIN_WIDTH_HOURS
        x_limits = (-3.0, 27.0)
        tick_positions = (-3.0, 0.0, 6.0, 12.0, 18.0, 24.0, 27.0)
        tick_labels = ("CT21", "CT0", "CT6", "CT12", "CT18", "CT24", "CT3")
    else:
        edges = np.arange(49, dtype=float) * DISPLAY_BIN_WIDTH_HOURS
        centers = DISPLAY_BIN_WIDTH_HOURS / 2.0 + np.arange(
            N_DISPLAY_BINS, dtype=float
        ) * DISPLAY_BIN_WIDTH_HOURS
        x_limits = (0.0, 24.0)
        tick_positions = (0.0, 6.0, 12.0, 18.0, 24.0)
        tick_labels = ("CT0", "CT6", "CT12", "CT18", "CT24")
    return edges, centers, x_limits, tick_positions, tick_labels


def _wrap_values(values: np.ndarray, *, wrap_margin: bool) -> np.ndarray:
    if not wrap_margin:
        return values
    return np.concatenate([values[-6:], values, values[:6]])


def plot_profile_axis(
    ax: plt.Axes,
    display: pd.DataFrame,
    animal: str,
    relative_density_max: float,
    *,
    style: str,
    wrap_margin: bool,
) -> tuple[float, np.ndarray]:
    if style not in {"step", "connected"}:
        raise ValueError(f"Unknown profile style: {style}")

    matrix = profile_matrix(display, animal)
    edges, centers, x_limits, tick_positions, tick_labels = display_coordinates(
        wrap_margin=wrap_margin
    )
    lane_height, y_max, baselines, row_centers = lane_layout(relative_density_max)

    if wrap_margin:
        ax.axvspan(-3.0, 0.0, color="#D9D9D9", alpha=0.38, zorder=0)
        ax.axvspan(24.0, 27.0, color="#D9D9D9", alpha=0.38, zorder=0)
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

    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        baseline = baselines[behavior_index]
        values = _wrap_values(matrix[behavior_index, :], wrap_margin=wrap_margin)
        color = BEHAVIOR_COLORS[behavior]

        ax.axhline(baseline, color="#777777", linewidth=0.6, zorder=1)
        ax.axhline(
            baseline + 1.0,
            color="#999999",
            linestyle=(0, (2, 2)),
            linewidth=0.55,
            zorder=1,
        )

        if style == "step":
            y_step = baseline + np.r_[values, values[-1]]
            ax.fill_between(
                edges,
                baseline,
                y_step,
                step="post",
                color=color,
                alpha=0.58,
                linewidth=0.0,
                zorder=2,
            )
            ax.step(
                edges,
                y_step,
                where="post",
                color=color,
                linewidth=0.85,
                zorder=3,
            )
        else:
            connected_x = np.r_[edges[0], centers, edges[-1]]
            connected_values = np.r_[values[0], values, values[-1]]
            connected_y = baseline + connected_values
            ax.fill_between(
                connected_x,
                baseline,
                connected_y,
                color=color,
                alpha=0.58,
                linewidth=0.0,
                zorder=2,
            )
            ax.plot(
                connected_x,
                connected_y,
                color=color,
                linewidth=0.9,
                zorder=3,
            )

    ax.set_xlim(*x_limits)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(tick_positions, tick_labels, fontsize=8)
    ax.set_yticks(row_centers, BEHAVIOR_ORDER, fontsize=8)
    ax.set_xlabel("circadian time", fontsize=9)
    ax.set_ylabel("behavior lanes\n(relative density height)", fontsize=8)
    ax.set_title(animal, fontsize=10)
    ax.grid(axis="x", color="#FFFFFF", linewidth=0.45, alpha=0.75, zorder=1)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    return y_max, row_centers


def plot_w1_bars(
    ax: plt.Axes,
    pair: dict[str, str],
    pair_behavior: pd.DataFrame,
    bar_limit: float,
    y_max: float,
    row_centers: np.ndarray,
    lane_height: float,
) -> None:
    values = (
        pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]]
        .set_index("behavior")
        .loc[list(BEHAVIOR_ORDER), "behavior_w1_hours"]
        .to_numpy(dtype=float)
    )
    ax.barh(
        row_centers,
        values,
        height=lane_height * 0.58,
        color="#4D4D4D",
        alpha=0.86,
        zorder=2,
    )
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, bar_limit)
    ax.set_yticks(row_centers, BEHAVIOR_ORDER, fontsize=8)
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


def save_profile_figure(
    display: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    relative_density_max: float,
    output_path: Path,
    *,
    style: str,
    wrap_margin: bool,
    compact_geometry: bool = False,
) -> None:
    if compact_geometry:
        figure = plt.figure(figsize=(12.0, 12.8))
        width_ratios = (1.0, 1.0, 0.78)
        hspace = 0.22
        wspace = 0.38
    else:
        figure = plt.figure(figsize=(16, 9.3))
        width_ratios = (1.35, 1.35, 1.0)
        hspace = 0.62
        wspace = 0.35
    grid = plt.GridSpec(
        len(PAIR_SPECS),
        3,
        figure=figure,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace,
    )
    lane_height, y_max, _, row_centers = lane_layout(relative_density_max)
    bar_limit = float(pair_behavior["behavior_w1_hours"].max()) * 1.22

    for row_index, pair in enumerate(PAIR_SPECS):
        axis_i = figure.add_subplot(grid[row_index, 0])
        axis_j = figure.add_subplot(grid[row_index, 1])
        axis_bar = figure.add_subplot(grid[row_index, 2])
        plot_profile_axis(
            axis_i,
            display,
            pair["animal_i"],
            relative_density_max,
            style=style,
            wrap_margin=wrap_margin,
        )
        plot_profile_axis(
            axis_j,
            display,
            pair["animal_j"],
            relative_density_max,
            style=style,
            wrap_margin=wrap_margin,
        )
        plot_w1_bars(
            axis_bar,
            pair,
            pair_behavior,
            bar_limit,
            y_max,
            row_centers,
            lane_height,
        )
        if compact_geometry:
            axis_i.set_ylabel(
                "behavior lanes\n(relative density height)" if row_index == 0 else "",
                fontsize=8,
            )
            axis_j.set_ylabel("")
            axis_bar.set_ylabel("")

    style_label = "filled step" if style == "step" else "filled connected"
    if compact_geometry:
        title = "Representative pair connected phase profiles"
    else:
        title = f"Representative pair {style_label} phase-distribution profiles"
    if wrap_margin:
        subtitle = (
            "Height = relative density vs uniform (1× = uniform); colors identify behaviors; "
            "shaded CT21–24 and CT0–3 margins repeat display data only"
        )
    elif compact_geometry:
        subtitle = (
            "30-minute connected values; height = relative density vs uniform (1× = uniform); "
            "common amplitude scale across all lanes"
        )
    else:
        subtitle = (
            "Height = relative density vs uniform (1× = uniform); colors identify behaviors; "
            "common lane scale across all animals and behaviors"
        )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=12.5 if compact_geometry else 13, y=0.98)
    footer = (
        "Circular W1 uses the original 5-minute distributions; 30-minute aggregation is display-only. "
        "Dashed lines mark 1× uniform; connected segments are straight, with no smoothing."
        if compact_geometry
        else "Circular W1 was calculated from the original 5-minute distributions; "
        "30-minute aggregation is for visualization only. Dashed lane lines mark 1× uniform; no smoothing is used."
    )
    figure.text(
        0.5,
        0.014 if compact_geometry else 0.018,
        footer,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    if compact_geometry:
        figure.subplots_adjust(top=0.91, bottom=0.065, left=0.075, right=0.97)
    else:
        figure.subplots_adjust(top=0.88, bottom=0.10, left=0.055, right=0.91)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_standalone_pair_figure(
    display: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    pair: dict[str, str],
    relative_density_max: float,
    bar_limit: float,
    output_path: Path,
) -> None:
    """Save one compact connected-profile figure for one fixed pair."""

    figure = plt.figure(figsize=(12.0, 6.8))
    grid = plt.GridSpec(
        1,
        3,
        figure=figure,
        width_ratios=(1.0, 1.0, 0.78),
        wspace=0.38,
    )
    lane_height, y_max, _, row_centers = lane_layout(relative_density_max)
    axis_i = figure.add_subplot(grid[0, 0])
    axis_j = figure.add_subplot(grid[0, 1])
    axis_bar = figure.add_subplot(grid[0, 2])
    plot_profile_axis(
        axis_i,
        display,
        pair["animal_i"],
        relative_density_max,
        style="connected",
        wrap_margin=False,
    )
    plot_profile_axis(
        axis_j,
        display,
        pair["animal_j"],
        relative_density_max,
        style="connected",
        wrap_margin=False,
    )
    plot_w1_bars(
        axis_bar,
        pair,
        pair_behavior,
        bar_limit,
        y_max,
        row_centers,
        lane_height,
    )
    axis_i.set_ylabel("behavior lanes\n(relative density height)", fontsize=8)
    axis_j.set_ylabel("")
    axis_bar.set_ylabel("")

    figure.suptitle(
        f"{pair['animal_i']} vs {pair['animal_j']} connected phase profiles\n"
        "30-minute connected values; common relative-density scale across all three pairs",
        fontsize=12.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.018,
        "Circular W1 uses the original 5-minute distributions; 30-minute aggregation is display-only. "
        "Dashed lines mark 1× uniform; connected segments are straight, with no smoothing.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(top=0.86, bottom=0.10, left=0.11, right=0.97)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_mockup_w1_axis(
    ax: plt.Axes,
    pair: dict[str, str],
    pair_behavior: pd.DataFrame,
    bar_limit: float,
    y_max: float,
    row_centers: np.ndarray,
    lane_height: float,
    *,
    show_xlabel: bool,
) -> None:
    values = (
        pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]]
        .set_index("behavior")
        .loc[list(BEHAVIOR_ORDER), "behavior_w1_hours"]
        .to_numpy(dtype=float)
    )
    composite_value = float(
        pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]][
            "composite_repertoire_w1_hours"
        ].iloc[0]
    )
    ax.barh(
        row_centers,
        values,
        height=lane_height * 0.58,
        color="#4D4D4D",
        alpha=0.86,
        zorder=2,
    )
    ax.axvline(
        composite_value,
        color="#D62728",
        linestyle=(0, (3, 2)),
        linewidth=0.85,
        zorder=3,
    )
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, bar_limit)
    ax.set_yticks(row_centers, [""] * len(BEHAVIOR_ORDER))
    ax.set_xlabel("Per-behavior W1 (h)" if show_xlabel else "", fontsize=9.0)
    ax.set_title("Composite W1", fontsize=9.0, pad=1.5)
    ax.grid(axis="x", color="#BBBBBB", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, length=2)
    ax.tick_params(axis="y", labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def save_mockup_panel_figure(
    display: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    relative_density_max: float,
    bar_limit: float,
    output_path: Path,
) -> None:
    """Save a compact three-row Panel C candidate matching the supplied mockup."""

    figure = plt.figure(figsize=(7.2, 13.8))
    grid = plt.GridSpec(
        len(STANDALONE_PAIR_SPECS),
        3,
        figure=figure,
        width_ratios=(1.0, 1.0, 0.62),
        hspace=0.15,
        wspace=0.08,
    )
    lane_height, y_max, _, row_centers = lane_layout(relative_density_max)
    row_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    row_titles = ("Low distance", "Intermediate distance", "High distance")

    for row_index, pair in enumerate(STANDALONE_PAIR_SPECS):
        axis_i = figure.add_subplot(grid[row_index, 0])
        axis_j = figure.add_subplot(grid[row_index, 1])
        axis_bar = figure.add_subplot(grid[row_index, 2])
        plot_profile_axis(
            axis_i,
            display,
            pair["animal_i"],
            relative_density_max,
            style="connected",
            wrap_margin=False,
        )
        plot_profile_axis(
            axis_j,
            display,
            pair["animal_j"],
            relative_density_max,
            style="connected",
            wrap_margin=False,
        )
        plot_mockup_w1_axis(
            axis_bar,
            pair,
            pair_behavior,
            bar_limit,
            y_max,
            row_centers,
            lane_height,
            show_xlabel=row_index == len(STANDALONE_PAIR_SPECS) - 1,
        )

        axis_i.set_title(pair["animal_i"], fontsize=9.0, pad=1.5)
        axis_j.set_title(pair["animal_j"], fontsize=9.0, pad=1.5)
        axis_i.set_ylabel("")
        axis_j.set_ylabel("")
        axis_i.set_yticks(row_centers, MOCKUP_BEHAVIOR_LABELS, fontsize=7.4)
        for tick, behavior in zip(axis_i.get_yticklabels(), BEHAVIOR_ORDER):
            tick.set_color(BEHAVIOR_COLORS[behavior])
        axis_i.tick_params(axis="y", length=0, pad=1)
        axis_j.tick_params(axis="y", labelleft=False, length=0)
        axis_i.grid(False, axis="x")
        axis_j.grid(False, axis="x")

        if row_index < len(STANDALONE_PAIR_SPECS) - 1:
            axis_i.set_xticks([])
            axis_j.set_xticks([])
            axis_i.set_xlabel("")
            axis_j.set_xlabel("")
        else:
            axis_i.set_xticks((0.0, 24.0), ("0", "24"), fontsize=8)
            axis_j.set_xticks((0.0, 24.0), ("0", "24"), fontsize=8)
            axis_i.set_xlabel("")
            axis_j.set_xlabel("")

        for axis in (axis_i, axis_j):
            axis.tick_params(axis="x", length=2, labelsize=8)
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
        row_axes.append((axis_i, axis_j, axis_bar))

    figure.subplots_adjust(
        top=0.94,
        bottom=0.085,
        left=0.16,
        right=0.98,
    )
    for row_title, (axis_i, axis_j, _) in zip(row_titles, row_axes):
        left = axis_i.get_position().x0
        right = axis_j.get_position().x1
        top = axis_i.get_position().y1
        figure.text(
            (left + right) / 2.0,
            top + 0.012,
            row_title,
            ha="center",
            va="bottom",
            fontsize=10.0,
        )

    first_i, first_j, first_bar = row_axes[-1]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bar_xlabel_bbox = first_bar.xaxis.get_label().get_window_extent(renderer=renderer)
    bar_xlabel_center_y = figure.transFigure.inverted().transform(
        (0.0, bar_xlabel_bbox.y0 + bar_xlabel_bbox.height / 2.0)
    )[1]
    figure.text(
        (first_i.get_position().x0 + first_j.get_position().x1) / 2.0,
        bar_xlabel_center_y,
        "CT (h)",
        ha="center",
        va="center",
        fontsize=9.0,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _format_behavior_values(
    pair: dict[str, str], pair_behavior: pd.DataFrame
) -> list[str]:
    subset = pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]].set_index(
        "behavior"
    )
    return [
        f"  - {behavior}: {float(subset.loc[behavior, 'behavior_w1_hours']):.17g} h"
        for behavior in BEHAVIOR_ORDER
    ]


def write_run_summary(
    output_path: Path,
    input_dir: Path,
    output_dir: Path,
    validation: dict[str, object],
    display_stats: dict[str, float | int],
    pair_behavior: pd.DataFrame,
    standalone_pair_behavior: pd.DataFrame,
    standalone_bar_limit: float,
) -> None:
    composite_lookup = validation["composite_lookup"]
    assert isinstance(composite_lookup, dict)
    relative_min = float(display_stats["relative_density_min"])
    relative_max = float(display_stats["relative_density_max"])
    total_display_cells = int(display_stats["display_rows"])
    lines = [
        "Representative stacked filled circular-W1 pair profiles",
        "=======================================================",
        "",
        f"Input directory: {input_dir}",
        f"Output directory: {output_dir}",
        "",
        "Files read:",
        *[f"- {input_dir / filename}" for filename in SOURCE_FILENAMES],
        "",
        "Scientific/data handling:",
        "- Circular W1 was calculated from the original 5-minute distributions; 30-minute aggregation is for visualization only.",
        "- Each behavior profile is P(CT | behavior) displayed as relative density versus uniform; no abundance or support weighting is used.",
        "- The same quantitative lane scale is used for all behaviors, animals, and representative pairs.",
        "- Filled step profiles preserve the 30-minute bin edges; connected profiles join adjacent 30-minute bin centers with straight lines and do not fit curves.",
        "",
        "Existing combined-prototype pairs and exact composite W1:",
    ]
    for pair in PAIR_SPECS:
        composite_value = float(
            getattr(
                composite_lookup[(pair["animal_i"], pair["animal_j"])],
                "repertoire_w1_mean_hours",
            )
        )
        lines.append(
            f"- {pair['pair_id']} ({pair['class_label']}): {composite_value:.17g} h"
        )
    lines.extend(["", "Existing combined-prototype behavior-specific W1 values:"])
    for pair in PAIR_SPECS:
        lines.append(f"- {pair['pair_id']}:")
        lines.extend(_format_behavior_values(pair, pair_behavior))

    lines.extend(["", "Standalone pair figures:"])
    for pair in STANDALONE_PAIR_SPECS:
        subset = standalone_pair_behavior[
            standalone_pair_behavior["pair_id"] == pair["pair_id"]
        ]
        composite_value = float(subset["composite_repertoire_w1_hours"].iloc[0])
        lines.append(
            f"- {pair['pair_id']} ({pair['class_label']}): {composite_value:.17g} h"
        )
        lines.extend(_format_behavior_values(pair, standalone_pair_behavior))

    lines.extend(
        [
            "",
            "30-minute display scale:",
            f"- observed full-cohort relative-density range: {relative_min:.17g} to {relative_max:.17g}",
            "- display cap: none; no clipping was used",
            f"- cells clipped: 0 of {total_display_cells} (0.0%)",
            f"- maximum aggregation-versus-5-minute mass error: {float(display_stats['max_aggregation_error']):.3g}",
            f"- maximum 30-minute row-sum error: {float(display_stats['max_display_sum_error']):.3g}",
            f"- common W1-bar x-axis limit across all three standalone pairs: {standalone_bar_limit:.6f} h",
            "",
            "Low / medium / high review:",
            "- The 675I__675J low pair is broadly similar across the behavior lanes; its largest contributions are nesting and digging, but neither is an isolated visual outlier.",
            "- 675H__714G shows moderate differences distributed across nesting, locomotion, climbing, drinking, and eating; nesting is the largest contributor, but the pattern is mixed rather than a single-row change.",
            "- The medium pair shows mixed behavior-specific phase and shape redistribution rather than one common phase shift or one isolated outlier, and remains a good intermediate example.",
            "- The 714D__714H high pair shows broad multibehavior redistribution; several behavior rows contribute strongly, so no single behavior explains the high composite.",
            "- The three standalone figures form a visually sensible low-to-medium-to-high progression, and all six animals are unique across the three pairs.",
            "",
            "Profile versus heatmap interpretation:",
            "- The compact connected CT0–24 profile is easier for scanning peak locations, shoulders, broad shape, and multimodality than the prior horizontally stretched figure.",
            "- The connected segments join adjacent 30-minute centers by straight lines; they are a visual guide and do not change the underlying binned values or add smoothing.",
            "- The CT0–24 compact version is clearer for the main comparison; wrap margins help inspect CT0/CT24 continuity but are not part of the locked primary display.",
            "- Profiles make peak location, broad versus narrow structure, and multimodality more intuitive than the heatmap. The heatmap remains cleaner for dense animal-to-animal comparison across all cells.",
            "- Diffuse grooming remains diffuse under the common scale; it is not made artificially prominent by per-row autoscaling.",
            "- The common scale makes low-amplitude rows visually smaller, which is an honest quantitative consequence rather than a support encoding; the fixed lane spacing keeps every behavior equally represented.",
            "- The aligned W1 bars connect naturally to the matching behavior rows and preserve fixed behavior order.",
            "- The low pair visually looks more similar than the high pair. Nesting is the largest low-pair behavior contribution; climbing, rearing, and eating lead the high pair.",
            "- Recommendation: lock the compact connected CT0–24 profile as the primary representative-pair display, while retaining the heatmap as a compact full-grid comparison.",
            "- Specific drawback: stacked lanes require more vertical space, and the common scale can make diffuse or low-density behaviors look visually quiet even when they are scientifically retained.",
            "",
            "Validation:",
            f"- source phase rows: {validation['phase_rows']}; source animal-behavior groups: {validation['phase_groups']}",
            f"- source behavior-W1 rows: {validation['pairwise_rows']}; source composite rows: {validation['composite_rows']}",
            f"- maximum source probability-sum error: {float(validation['max_probability_sum_error']):.3g}",
            f"- maximum source grid error: {float(validation['max_phase_grid_error']):.3g} h",
            f"- maximum wide/long behavior-W1 difference: {float(validation['max_cross_file_w1_error']):.3g} h",
            f"- maximum composite-versus-eight-value mean error: {float(validation['max_composite_mean_error']):.3g} h",
            "- no smoothing, kernel density estimate, spline, per-row autoscaling, genotype-dependent scaling, support weighting, raw-data edit, or upstream circular-W1 edit was used",
            "- the compact connected candidate changes figure geometry only; the source values, 30-minute aggregation, and common amplitude scale are unchanged",
            "",
            "Generated files:",
            "- representative_pair_profiles_step_simple.png",
            "- representative_pair_profiles_connected_simple.png",
            "- representative_pair_profiles_step_wrap.png",
            "- representative_pair_profiles_connected_wrap.png",
            "- representative_pair_profiles_connected_compact.png",
            "- pair_profile_675I_675J.png",
            "- pair_profile_675H_714G.png",
            "- pair_profile_714D_714H.png",
            "- panel_C_representative_pairs_mockup_style.png",
            "- representative_pair_profile_data.csv",
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
    profile_data = selected_profile_data(display)
    pair_behavior = selected_w1_table(validation)
    standalone_pair_behavior = selected_w1_table(validation, STANDALONE_PAIR_SPECS)
    standalone_bar_limit = (
        float(standalone_pair_behavior["behavior_w1_hours"].max()) * 1.22
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_data.to_csv(
        output_dir / "representative_pair_profile_data.csv",
        index=False,
        float_format="%.17g",
    )

    relative_density_max = float(display_stats["relative_density_max"])
    figure_specs = (
        ("step", False, "representative_pair_profiles_step_simple.png"),
        ("connected", False, "representative_pair_profiles_connected_simple.png"),
        ("step", True, "representative_pair_profiles_step_wrap.png"),
        ("connected", True, "representative_pair_profiles_connected_wrap.png"),
    )
    for style, wrap_margin, filename in figure_specs:
        save_profile_figure(
            display,
            pair_behavior,
            relative_density_max,
            output_dir / filename,
            style=style,
            wrap_margin=wrap_margin,
        )

    save_profile_figure(
        display,
        pair_behavior,
        relative_density_max,
        output_dir / "representative_pair_profiles_connected_compact.png",
        style="connected",
        wrap_margin=False,
        compact_geometry=True,
    )

    for pair in STANDALONE_PAIR_SPECS:
        save_standalone_pair_figure(
            display,
            standalone_pair_behavior,
            pair,
            relative_density_max,
            standalone_bar_limit,
            output_dir / f"pair_profile_{pair['pair_id'].replace('__', '_')}.png",
        )
    save_mockup_panel_figure(
        display,
        standalone_pair_behavior,
        relative_density_max,
        standalone_bar_limit,
        output_dir / "panel_C_representative_pairs_mockup_style.png",
    )

    write_run_summary(
        output_dir / "run_summary.txt",
        input_dir,
        output_dir,
        validation,
        display_stats,
        pair_behavior,
        standalone_pair_behavior,
        standalone_bar_limit,
    )

    print(f"Wrote profile outputs to {output_dir}")
    for pair in PAIR_SPECS:
        composite_value = float(
            pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]][
                "composite_repertoire_w1_hours"
            ].iloc[0]
        )
        print(f"{pair['pair_id']}: composite_w1={composite_value:.6f} h")
    for pair in STANDALONE_PAIR_SPECS:
        composite_value = float(
            standalone_pair_behavior[
                standalone_pair_behavior["pair_id"] == pair["pair_id"]
            ]["composite_repertoire_w1_hours"].iloc[0]
        )
        print(f"{pair['pair_id']}: composite_w1={composite_value:.6f} h")
    print(f"standalone_w1_bar_limit={standalone_bar_limit:.6f} h")
    print(
        "relative_density_range="
        f"{float(display_stats['relative_density_min']):.6f}-"
        f"{relative_density_max:.6f}"
    )
    print(f"display_rows={int(display_stats['display_rows'])}")


if __name__ == "__main__":
    main()
