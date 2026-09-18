"""Create a descriptive circular repertoire-glyph prototype.

This script reads the frozen circular-W1 repertoire outputs and creates
presentation-oriented glyphs.  It does not recompute any W1 distance, alter
the metric, or use genotype for pair selection or visual encoding.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
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
BEHAVIOR_SHORT_LABELS = {
    "eating": "eat",
    "drinking": "drink",
    "rearing": "rear",
    "climbing": "climb",
    "digging": "dig",
    "nesting": "nest",
    "grooming": "groom",
    "locomotion": "locom.",
}
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
SELECTION_CLASS_ORDER = ("low", "medium", "high")

N_PHASE_BINS = 288
BIN_WIDTH_HOURS = 1.0 / 12.0
HOURS_PER_CYCLE = 24.0
FLOAT_TOLERANCE = 1e-12

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Glyph_Visualization"

SOURCE_FILENAMES = (
    "behavior_phase_distributions_5min.csv",
    "pairwise_behavior_circular_w1.csv",
    "pairwise_repertoire_distance.csv",
    "behavior_w1_contribution_summary.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptive circular-W1 repertoire glyphs from existing outputs."
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
        help="New directory for glyph tables, figures, and the run summary.",
    )
    return parser.parse_args()


def load_sources(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [input_dir / filename for filename in SOURCE_FILENAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required existing circular-W1 output(s) are missing: "
            + ", ".join(str(path) for path in missing)
        )

    phase, pairwise, composite, contribution = (
        pd.read_csv(path) for path in paths
    )
    return phase, pairwise, composite, contribution


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_pair_rows(frame: pd.DataFrame, name: str) -> None:
    expected_pairs = set(itertools.combinations(ANIMAL_ORDER, 2))
    observed_pairs = {
        (str(row.animal_i), str(row.animal_j))
        for row in frame.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"{name} does not contain the 28 expected fixed-order animal pairs."
        )
    if frame[["animal_i", "animal_j"]].duplicated().any():
        raise ValueError(f"{name} contains duplicate animal-pair rows.")
    if (frame["animal_i"] == frame["animal_j"]).any():
        raise ValueError(f"{name} contains a self-pair.")


def validate_sources(
    phase: pd.DataFrame,
    pairwise: pd.DataFrame,
    composite: pd.DataFrame,
    contribution: pd.DataFrame,
) -> dict[str, float | int]:
    """Validate source structure and cross-file W1 consistency only.

    The checks compare already-produced W1 values across the supplied files;
    they do not calculate a new W1 distance.
    """

    _require_columns(
        phase,
        {
            "animal",
            "behavior",
            "ct_bin_start_hours",
            "ct_bin_center_hours",
            "probability",
        },
        "behavior_phase_distributions_5min.csv",
    )
    _require_columns(
        pairwise,
        {"animal_i", "animal_j", "behavior", "w1_hours"},
        "pairwise_behavior_circular_w1.csv",
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
        "pairwise_repertoire_distance.csv",
    )
    _require_columns(contribution, {"behavior"}, "behavior_w1_contribution_summary.csv")

    expected_animals = set(ANIMAL_ORDER)
    expected_behaviors = set(BEHAVIOR_ORDER)
    if set(phase["animal"].astype(str)) != expected_animals:
        raise ValueError("Phase distributions do not contain the expected animals.")
    if set(phase["behavior"].astype(str)) != expected_behaviors:
        raise ValueError("Phase distributions do not contain the expected behaviors.")
    if set(pairwise["behavior"].astype(str)) != expected_behaviors:
        raise ValueError("Pairwise W1 output does not contain the expected behaviors.")
    if set(contribution["behavior"].astype(str)) != expected_behaviors:
        raise ValueError(
            "Behavior contribution summary does not contain the expected behaviors."
        )
    if tuple(contribution["behavior"].astype(str)) != BEHAVIOR_ORDER:
        raise ValueError(
            "Behavior contribution summary is not in the required fixed behavior order."
        )

    phase_group_sizes = phase.groupby(["animal", "behavior"], sort=False).size()
    expected_group_count = len(ANIMAL_ORDER) * len(BEHAVIOR_ORDER)
    if len(phase_group_sizes) != expected_group_count:
        raise ValueError("Phase distributions do not contain all animal-behavior groups.")
    if not (phase_group_sizes == N_PHASE_BINS).all():
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
    max_grid_error = 0.0
    for key, group in phase.groupby(["animal", "behavior"], sort=False):
        ordered = group.sort_values("ct_bin_center_hours")
        probability_sum_error = abs(float(ordered["probability"].sum()) - 1.0)
        max_probability_sum_error = max(max_probability_sum_error, probability_sum_error)
        grid_error = max(
            float(
                np.max(
                    np.abs(
                        ordered["ct_bin_start_hours"].to_numpy(dtype=float)
                        - expected_starts
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        ordered["ct_bin_center_hours"].to_numpy(dtype=float)
                        - expected_centers
                    )
                )
            ),
        )
        max_grid_error = max(max_grid_error, grid_error)
        if probability_sum_error > FLOAT_TOLERANCE:
            raise ValueError(
                f"Phase probabilities for {key} do not sum to one: "
                f"error={probability_sum_error}"
            )
    if max_grid_error > FLOAT_TOLERANCE:
        raise ValueError(f"Phase-bin grid does not match 5-minute CT bins: error={max_grid_error}")

    if len(pairwise) != len(list(itertools.combinations(ANIMAL_ORDER, 2))) * len(
        BEHAVIOR_ORDER
    ):
        raise ValueError("Pairwise W1 output does not contain exactly 224 rows.")
    if pairwise[["animal_i", "animal_j", "behavior"]].duplicated().any():
        raise ValueError("Pairwise W1 output contains duplicate pair-behavior rows.")
    _validate_pair_rows(pairwise.drop_duplicates(subset=["animal_i", "animal_j"]), "Pairwise W1 output")
    if pairwise["w1_hours"].isna().any() or not np.isfinite(
        pairwise["w1_hours"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Pairwise W1 output contains missing or non-finite values.")
    if (pairwise["w1_hours"] < 0).any():
        raise ValueError("Pairwise W1 output contains negative values.")

    if len(composite) != len(list(itertools.combinations(ANIMAL_ORDER, 2))):
        raise ValueError("Composite repertoire output does not contain exactly 28 rows.")
    _validate_pair_rows(composite, "Composite repertoire output")
    composite_numeric = composite[list(composite_w1_columns) + ["repertoire_w1_mean_hours"]]
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
    max_cross_file_error = 0.0
    for row in composite.itertuples(index=False):
        for behavior in BEHAVIOR_ORDER:
            composite_value = float(getattr(row, f"w1_{behavior}"))
            pairwise_value = pairwise_lookup[(str(row.animal_i), str(row.animal_j), behavior)]
            difference = abs(composite_value - pairwise_value)
            max_cross_file_error = max(max_cross_file_error, difference)
            if difference > FLOAT_TOLERANCE:
                raise ValueError(
                    "Composite and long pairwise W1 outputs disagree for "
                    f"{row.animal_i} vs {row.animal_j}, {behavior}: "
                    f"{composite_value} versus {pairwise_value}"
                )

    return {
        "phase_rows": int(len(phase)),
        "pairwise_rows": int(len(pairwise)),
        "composite_rows": int(len(composite)),
        "phase_groups": int(len(phase_group_sizes)),
        "max_probability_sum_error": max_probability_sum_error,
        "max_phase_grid_error": max_grid_error,
        "max_cross_file_w1_error": max_cross_file_error,
    }


def build_glyph_summary(phase: pd.DataFrame) -> pd.DataFrame:
    """Summarize each existing phase distribution without changing it."""

    rows: list[dict[str, object]] = []
    sector_width_radians = 2.0 * np.pi / len(BEHAVIOR_ORDER)
    for animal in ANIMAL_ORDER:
        for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
            group = phase[
                (phase["animal"] == animal) & (phase["behavior"] == behavior)
            ].sort_values("ct_bin_center_hours")
            phase_hours = group["ct_bin_center_hours"].to_numpy(dtype=float)
            probability = group["probability"].to_numpy(dtype=float)
            phase_radians = 2.0 * np.pi * phase_hours / HOURS_PER_CYCLE
            resultant = np.sum(probability * np.exp(1j * phase_radians))
            resultant_length = float(abs(resultant))
            if resultant_length <= 1e-15:
                raise ValueError(
                    f"Circular phase center is undefined for {animal}, {behavior}."
                )

            central_phase_radians = float(np.mod(np.angle(resultant), 2.0 * np.pi))
            central_phase_hours = (
                central_phase_radians * HOURS_PER_CYCLE / (2.0 * np.pi)
            )
            spread_radians = float(np.sqrt(max(0.0, -2.0 * np.log(resultant_length))))
            sector_start_radians = behavior_index * sector_width_radians
            sector_end_radians = (behavior_index + 1) * sector_width_radians
            rows.append(
                {
                    "animal": animal,
                    "behavior": behavior,
                    "behavior_order_index": behavior_index,
                    "sector_start_hours": behavior_index
                    * HOURS_PER_CYCLE
                    / len(BEHAVIOR_ORDER),
                    "sector_end_hours": (behavior_index + 1)
                    * HOURS_PER_CYCLE
                    / len(BEHAVIOR_ORDER),
                    "sector_start_radians": sector_start_radians,
                    "sector_end_radians": sector_end_radians,
                    "central_phase_hours": central_phase_hours,
                    "central_phase_radians": central_phase_radians,
                    "mean_resultant_length": resultant_length,
                    "spread_summary_radians": spread_radians,
                    "spread_summary_hours": spread_radians
                    * HOURS_PER_CYCLE
                    / (2.0 * np.pi),
                    "spread_summary_method": "circular_standard_deviation_sqrt_minus_2_log_R",
                }
            )

    return pd.DataFrame(rows)


def make_pair_id(animal_a: str, animal_b: str) -> str:
    return f"{animal_a}__{animal_b}"


def select_representative_pairs(composite: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Select two low, two median-near, and two high source-distance pairs."""

    distance_column = "repertoire_w1_mean_hours"
    ranked = composite.copy()
    ranked["animal_i"] = ranked["animal_i"].astype(str)
    ranked["animal_j"] = ranked["animal_j"].astype(str)
    ranked["pair_id"] = [
        make_pair_id(row.animal_i, row.animal_j)
        for row in ranked.itertuples(index=False)
    ]
    ranked = ranked.sort_values(
        [distance_column, "animal_i", "animal_j"], kind="mergesort"
    ).reset_index(drop=True)
    ranked["rank_by_composite_distance"] = np.arange(1, len(ranked) + 1)
    median_distance = float(ranked[distance_column].median())
    ranked["distance_to_median_hours"] = abs(
        ranked[distance_column] - median_distance
    )

    low = ranked.head(2).copy()
    high = ranked.tail(2).copy().sort_values(
        [distance_column, "animal_i", "animal_j"], kind="mergesort"
    )
    medium = ranked.sort_values(
        ["distance_to_median_hours", distance_column, "animal_i", "animal_j"],
        kind="mergesort",
    ).head(2)

    selected_parts = []
    for selection_class, part in (
        ("low", low),
        ("medium", medium),
        ("high", high),
    ):
        labeled = part.copy()
        labeled["selection_class"] = selection_class
        selected_parts.append(labeled)
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected["pair_id"].duplicated().any():
        raise ValueError("Representative-pair selection produced a duplicate pair.")
    selected["selection_order"] = selected.groupby("selection_class", sort=False).cumcount() + 1
    return selected, median_distance


def build_pair_behavior_table(
    selected: pd.DataFrame, pairwise: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in selected.itertuples(index=False):
        subset = pairwise[
            (pairwise["animal_i"] == pair.animal_i)
            & (pairwise["animal_j"] == pair.animal_j)
        ].set_index("behavior")
        subset = subset.loc[list(BEHAVIOR_ORDER)]
        for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "selection_class": pair.selection_class,
                    "selection_order": int(pair.selection_order),
                    "animal_a": pair.animal_i,
                    "animal_b": pair.animal_j,
                    "behavior": behavior,
                    "behavior_order_index": behavior_index,
                    "behavior_w1_hours": float(subset.loc[behavior, "w1_hours"]),
                    "composite_repertoire_w1_hours": float(
                        pair.repertoire_w1_mean_hours
                    ),
                }
            )
    return pd.DataFrame(rows)


def _configure_polar_axis(ax: plt.Axes) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.2)
    ax.set_yticks([])
    ax.set_xticks(np.arange(4, dtype=float) * (2.0 * np.pi / 4.0))
    ax.set_xticklabels(["CT0", "CT6", "CT12", "CT18"], fontsize=8)
    ax.tick_params(axis="x", pad=8)
    ax.grid(color="#BBBBBB", linewidth=0.5, alpha=0.7)
    ax.spines["polar"].set_color("#777777")


def plot_glyph(ax: plt.Axes, animal_summary: pd.DataFrame, title: str) -> None:
    """Draw one glyph using a common CT angle and concentric behavior lanes."""

    _configure_polar_axis(ax)
    lane_centers = np.linspace(0.25, 0.91, len(BEHAVIOR_ORDER))
    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        row = animal_summary[animal_summary["behavior"] == behavior].iloc[0]
        theta = float(row["central_phase_radians"])
        spread = min(float(row["spread_summary_radians"]), np.pi)
        radius = float(lane_centers[behavior_index])
        color = BEHAVIOR_COLORS[behavior]

        ax.plot(
            np.linspace(0.0, 2.0 * np.pi, 200),
            np.full(200, radius),
            color="#D9D9D9",
            linewidth=0.65,
            alpha=0.9,
            zorder=1,
        )
        arc_angles = np.linspace(theta - spread, theta + spread, 80)
        ax.plot(
            arc_angles,
            np.full_like(arc_angles, radius),
            color=color,
            linewidth=4.0,
            alpha=0.62,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [theta, theta],
            [radius - 0.055, radius + 0.055],
            color=color,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            [theta],
            [radius],
            s=25,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    ax.set_title(title, fontsize=10, pad=16)


def behavior_legend() -> list[Patch]:
    return [
        Patch(
            facecolor=BEHAVIOR_COLORS[behavior],
            edgecolor="none",
            alpha=0.8,
            label=behavior,
        )
        for behavior in BEHAVIOR_ORDER
    ]


def save_all_animal_figure(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 10.2),
        subplot_kw={"projection": "polar"},
    )
    for ax, animal in zip(axes.flat, ANIMAL_ORDER):
        plot_glyph(
            ax,
            summary[summary["animal"] == animal],
            title=animal,
        )
    figure.suptitle(
        "Equal-weight circular-W1 repertoire glyphs\n"
        "dot = circular phase center; colored arc = circular spread",
        fontsize=13,
        y=0.98,
    )
    figure.legend(
        handles=behavior_legend(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=8,
    )
    figure.text(
        0.5,
        0.008,
        "Behavior colors are identified in the shared legend. Fixed ring order is retained across animals.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(
        top=0.86,
        bottom=0.18,
        left=0.03,
        right=0.97,
        hspace=0.55,
        wspace=0.18,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_pair_comparison_figure(
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    pair_behavior: pd.DataFrame,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(14, 22))
    grid = GridSpec(
        len(selected),
        3,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.45),
        hspace=0.76,
        wspace=0.32,
    )
    max_w1 = float(pair_behavior["behavior_w1_hours"].max())
    bar_limit = max_w1 * 1.28 if max_w1 > 0 else 1.0

    for row_index, pair in enumerate(selected.itertuples(index=False)):
        animal_a_summary = summary[summary["animal"] == pair.animal_i]
        animal_b_summary = summary[summary["animal"] == pair.animal_j]
        axis_a = figure.add_subplot(grid[row_index, 0], projection="polar")
        axis_b = figure.add_subplot(grid[row_index, 1], projection="polar")
        plot_glyph(
            axis_a,
            animal_a_summary,
            title=f"{pair.animal_i}\n{pair.selection_class.upper()}",
        )
        plot_glyph(axis_b, animal_b_summary, title=pair.animal_j)

        bar_axis = figure.add_subplot(grid[row_index, 2])
        values = pair_behavior[
            (pair_behavior["pair_id"] == pair.pair_id)
        ].set_index("behavior").loc[list(BEHAVIOR_ORDER), "behavior_w1_hours"]
        order = np.argsort(-values.to_numpy(dtype=float))
        ordered_behaviors = [BEHAVIOR_ORDER[index] for index in order]
        ordered_values = values.to_numpy(dtype=float)[order]
        y_positions = np.arange(len(ordered_behaviors))
        bar_axis.barh(
            y_positions,
            ordered_values,
            color=[BEHAVIOR_COLORS[behavior] for behavior in ordered_behaviors],
            alpha=0.85,
        )
        bar_axis.set_yticks(
            y_positions,
            [BEHAVIOR_SHORT_LABELS[behavior] for behavior in ordered_behaviors],
            fontsize=8,
        )
        bar_axis.invert_yaxis()
        bar_axis.set_xlim(0.0, bar_limit)
        bar_axis.set_xlabel("behavior W1 (hours)", fontsize=8)
        bar_axis.set_title(
            f"{pair.animal_i} vs {pair.animal_j}\n"
            f"composite W1 = {float(pair.repertoire_w1_mean_hours):.6f} h",
            fontsize=9,
        )
        bar_axis.tick_params(axis="x", labelsize=8)
        bar_axis.grid(axis="x", color="#BBBBBB", linewidth=0.5, alpha=0.6)
        bar_axis.set_axisbelow(True)
        for y_position, value in zip(y_positions, ordered_values):
            bar_axis.text(
                value + bar_limit * 0.012,
                y_position,
                f"{value:.6f}",
                va="center",
                fontsize=7,
            )

    figure.suptitle(
        "Representative equal-weight circular-W1 repertoire comparisons\n"
        "bars show source behavior-level W1 values sorted within each pair",
        fontsize=13,
        y=0.995,
    )
    figure.legend(
        handles=behavior_legend(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
        fontsize=8,
    )
    figure.text(
        0.5,
        0.035,
        "Behavior colors are identified in the shared legend. Fixed ring order is retained across animals.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(top=0.96, bottom=0.075, left=0.04, right=0.98)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_run_summary(
    output_path: Path,
    input_dir: Path,
    output_dir: Path,
    selected: pd.DataFrame,
    median_distance: float,
    validation: dict[str, float | int],
) -> None:
    lines = [
        "Circular-W1 repertoire glyph prototype",
        "======================================",
        "",
        f"Input directory: {input_dir}",
        f"Output directory: {output_dir}",
        "",
        "Files read:",
        *[f"- {input_dir / filename}" for filename in SOURCE_FILENAMES],
        "",
        "Representative-pair selection:",
        "- deterministic sort by existing repertoire_w1_mean_hours, then animal IDs",
        f"- low: two smallest composite distances; medium: two closest to median {median_distance:.17g} h; high: two largest",
        "- no genotype, group, or inferential information was used for selection or glyph encoding",
        "",
        "Selected pairs:",
    ]
    for pair in selected.itertuples(index=False):
        lines.append(
            f"- {pair.selection_class}: {pair.pair_id}; "
            f"composite W1={float(pair.repertoire_w1_mean_hours):.17g} h; "
            f"distance rank={int(pair.rank_by_composite_distance)}"
        )
    lines.extend(
        [
            "",
            "Glyph summaries:",
            "- central phase: probability-weighted circular mean of the existing 5-minute phase distribution",
            "- spread: circular standard deviation s=sqrt(-2*ln(R)), where R is the mean resultant length",
            "- spread_summary_hours is the angular spread converted using 24 hours per circle; it is not a linear time interval",
            "- spread is displayed by angular extent; physical arc length differs across concentric lane radii",
            "- all eight behaviors have equal-status concentric lanes; no filled angular region or abundance encoding is used",
            "",
            "Validation:",
            f"- phase rows read: {validation['phase_rows']}; animal-behavior groups: {validation['phase_groups']}",
            f"- behavior-level W1 rows read: {validation['pairwise_rows']}; composite rows read: {validation['composite_rows']}",
            f"- maximum phase-probability sum error: {validation['max_probability_sum_error']:.3g}",
            f"- maximum phase-grid error: {validation['max_phase_grid_error']:.3g} h",
            f"- maximum cross-file behavior-W1 difference: {validation['max_cross_file_w1_error']:.3g} h",
            "- plotted pair bars and composite labels are sourced from the existing W1 CSV outputs; no W1 metric was recomputed",
            "- no upstream scientific script or raw data file was modified",
            "",
            "Limitations:",
            "- each glyph is a compact central-phase/spread shorthand, not a replacement for the full phase distribution",
            "- display arcs are clipped at a 12-hour half-width when spread is broader, so broad distributions need the source CSV for full interpretation",
            "- pair behavior bars are sorted by magnitude for readability; glyph lanes retain the fixed behavior order",
            "",
            "Generated files:",
            "- representative_pair_selection.csv",
            "- animal_behavior_glyph_summary.csv",
            "- pair_behavior_w1_for_glyphs.csv",
            "- all_animal_repertoire_glyphs.png",
            "- selected_pair_repertoire_glyphs.png",
            "- run_summary.txt",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    phase, pairwise, composite, contribution = load_sources(input_dir)
    validation = validate_sources(phase, pairwise, composite, contribution)
    glyph_summary = build_glyph_summary(phase)
    selected, median_distance = select_representative_pairs(composite)
    pair_behavior = build_pair_behavior_table(selected, pairwise)

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_output = selected[
        [
            "pair_id",
            "selection_class",
            "selection_order",
            "animal_i",
            "animal_j",
            "repertoire_w1_mean_hours",
            "rank_by_composite_distance",
            "distance_to_median_hours",
        ]
    ]
    selection_output.to_csv(
        output_dir / "representative_pair_selection.csv",
        index=False,
        float_format="%.17g",
    )
    glyph_summary.to_csv(
        output_dir / "animal_behavior_glyph_summary.csv",
        index=False,
        float_format="%.17g",
    )
    pair_behavior.to_csv(
        output_dir / "pair_behavior_w1_for_glyphs.csv",
        index=False,
        float_format="%.17g",
    )

    save_all_animal_figure(
        glyph_summary,
        output_dir / "all_animal_repertoire_glyphs.png",
    )
    save_pair_comparison_figure(
        selected,
        glyph_summary,
        pair_behavior,
        output_dir / "selected_pair_repertoire_glyphs.png",
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        input_dir,
        output_dir,
        selected,
        median_distance,
        validation,
    )

    print(f"Wrote glyph outputs to {output_dir}")
    for pair in selected.itertuples(index=False):
        print(
            f"{pair.selection_class}: {pair.pair_id} "
            f"composite_w1={float(pair.repertoire_w1_mean_hours):.6f} h"
        )


if __name__ == "__main__":
    main()
