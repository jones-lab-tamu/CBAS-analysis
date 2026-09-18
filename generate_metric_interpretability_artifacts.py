"""Generate provisional real-data interpretability artifacts for MI and recurrence.

The script reads the existing FRP/MI and occupancy-recurrence outputs and
reuses the current classification, CT assignment, occupancy construction, and
pair scorer.  It does not change or rerun any scientific method.

The 10-minute recurrence figure uses display-only averaging of adjacent
5-minute occupancy bins.  Its shift curves remain the primary 5-minute curves.
The backing shift-curve CSV also contains the existing 10-minute sensitivity
curves for auditability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import phase_behavior_mutual_information as frozen_phase
import phase_behavior_occupancy_recurrence as occupancy_integration
import phase_behavior_recurrence as recurrence
from phase_behavior_occupancy_recurrence_validation import (
    _affinity_on_support,
    score_pair,
)


ANIMALS = {
    "714D": "Bmal1KO",
    "714G": "Bmal1KO",
    "714H": "Bmal1KO",
}
NONREST_BEHAVIORS = tuple(
    behavior
    for behavior in frozen_phase.BEHAVIORS
    if behavior != "resting"
)
MI_ANIMALS = ("714H", "714D", "714G")
RECURRENCE_ANIMALS = ("714G", "714D")
MI_PHASE_BINS = frozen_phase.PRIMARY_PHASE_BINS
RECURRENCE_RESOLUTIONS = ((5, 288), (10, 144))
NUMERICAL_TOLERANCE = 1e-12

MI_PROFILE_COLUMNS = [
    "Animal",
    "CT_bin_start",
    "CT_bin_end",
    "P_CT",
    *NONREST_BEHAVIORS,
    "raw_phase_specific_MI_contribution_bits",
]

MI_SUMMARY_COLUMNS = [
    "Animal",
    "MI8_observed_raw_bits",
    "MI8_null_mean_bits",
    "MI8_conditional_excess_bits",
    "phase_contribution_sum_bits",
    "reconstruction_error_bits",
]

SHIFT_CURVE_COLUMNS = [
    "Animal",
    "resolution_minutes",
    "cycle_A",
    "cycle_B",
    "shift_bins",
    "shift_minutes",
    "raw_affinity",
    "normalized_shift_relative_affinity",
]

PAIR_SUMMARY_COLUMNS = [
    "Animal",
    "resolution_minutes",
    "cycle_A",
    "cycle_B",
    "A_obs",
    "mu_shift",
    "A_obs_minus_mu_shift",
    "R_pair",
    "zero_shift_rank",
    "zero_shift_percentile",
    "best_shift_bins",
    "best_shift_minutes",
    "best_shift_affinity",
]

ANIMAL_SUMMARY_COLUMNS = [
    "Animal",
    "Recurrence5_mean_R_pair",
    "Recurrence10_mean_R_pair",
    "median_zero_shift_percentile_5min",
    "median_zero_shift_percentile_10min",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Generate provisional MI and recurrence interpretability artifacts."
    )
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the animal folders.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def _mi_cycle_inputs(
    animal_dir: Path,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], float]:
    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(f"Missing source indices for {animal_dir.name}: {missing_indices}")
    (
        _reported_frp,
        computational_frp,
        _start_ct,
        complete_cycle_specs,
        _frp_output_dir,
    ) = frozen_phase.load_frp_phase_solution(animal_dir)
    relative_times, labels, _expected_samples_per_hour, invalid_rows = (
        occupancy_integration._load_classified_samples(input_files)
    )
    if invalid_rows:
        raise ValueError(f"Invalid source rows for {animal_dir.name}: {invalid_rows}")
    complete_cycles = occupancy_integration._build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp,
    )
    return complete_cycles, computational_frp


def _read_mi_values(
    animal_dir: Path,
) -> tuple[float, float, float]:
    results = pd.read_csv(animal_dir / "MI_Output" / "mi_results.csv")
    conditional = results.loc[
        results["component"].eq("conditional_8state_nonrest")
    ]
    if len(conditional) != 1:
        raise ValueError(f"Expected one conditional MI result row for {animal_dir.name}")
    row = conditional.iloc[0]
    return (
        float(row["MI_raw_bits"]),
        float(row["MI_null_mean_bits"]),
        float(row["MI_excess_bits"]),
    )


def _mi_profile(
    animal: str,
    animal_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    complete_cycles, computational_frp = _mi_cycle_inputs(animal_dir)
    counts = frozen_phase.observed_counts_from_complete_cycles(
        complete_cycles,
        MI_PHASE_BINS,
        frp_hours=computational_frp,
    )
    conditional_counts = frozen_phase.conditional_nonrest_counts(counts)
    nonrest_by_ct = conditional_counts.sum(axis=1)
    total_nonrest = int(conditional_counts.sum())
    if total_nonrest <= 0 or np.any(nonrest_by_ct <= 0):
        raise ValueError(f"Conditional MI profile has an empty CT bin for {animal}")

    p_ct = nonrest_by_ct.astype(float) / total_nonrest
    p_behavior = conditional_counts.sum(axis=0).astype(float) / total_nonrest
    conditional_probability = conditional_counts / nonrest_by_ct[:, None]
    positive = conditional_probability > 0
    ratios = np.divide(
        conditional_probability,
        p_behavior[None, :],
        out=np.ones_like(conditional_probability),
        where=positive,
    )
    per_bin = p_ct * np.sum(
        np.where(positive, conditional_probability * np.log2(ratios), 0.0),
        axis=1,
    )
    observed = frozen_phase.mutual_information_bits(conditional_counts)
    observed_output, null_mean, excess = _read_mi_values(animal_dir)
    if not np.isclose(observed, observed_output, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"Computed conditional MI disagrees with existing output for {animal}")
    if not np.isclose(per_bin.sum(), observed, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"MI phase contributions do not reconstruct observed MI for {animal}")
    if not np.isclose(excess, observed - null_mean, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(f"Existing MI excess is inconsistent for {animal}")

    profile_rows = []
    for index in range(MI_PHASE_BINS):
        row = {
            "Animal": animal,
            "CT_bin_start": index * 24.0 / MI_PHASE_BINS,
            "CT_bin_end": (index + 1) * 24.0 / MI_PHASE_BINS,
            "P_CT": p_ct[index],
            "raw_phase_specific_MI_contribution_bits": per_bin[index],
        }
        row.update(
            {
                behavior: conditional_probability[index, column]
                for column, behavior in enumerate(NONREST_BEHAVIORS)
            }
        )
        profile_rows.append(row)

    return profile_rows, {
        "Animal": animal,
        "observed": observed,
        "null_mean": null_mean,
        "excess": excess,
        "contribution_sum": float(per_bin.sum()),
        "reconstruction_error": float(per_bin.sum() - observed),
        "probability": conditional_probability,
        "contribution": per_bin,
    }


def _recurrence_inputs(
    animal_dir: Path,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(f"Missing source indices for {animal_dir.name}: {missing_indices}")
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
    if invalid_rows:
        raise ValueError(f"Invalid source rows for {animal_dir.name}: {invalid_rows}")
    complete_cycles = occupancy_integration._build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp,
    )
    occupancy, valid_masks, _valid_counts = occupancy_integration._occupancy_for_resolution(
        relative_times,
        labels,
        complete_cycles,
        complete_cycle_specs[0][1],
        expected_samples_per_hour,
        computational_frp,
        n_bins,
    )
    cycle_indices = [cycle_index for cycle_index, _, _ in complete_cycles]
    return occupancy, valid_masks, cycle_indices, computational_frp


def _alignment_curve(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    bin_minutes: int,
) -> tuple[np.ndarray, dict[str, object]]:
    n_bins = tensor_a.shape[0]
    if tensor_b.shape != tensor_a.shape or n_bins not in (144, 288):
        raise ValueError("Unexpected recurrence tensor shape")
    raw_affinity = np.empty(n_bins, dtype=float)
    for shift in range(n_bins):
        shifted_b = np.roll(tensor_b, shift, axis=0)
        shifted_mask_b = np.roll(valid_b, shift)
        support = valid_a & shifted_mask_b
        if not np.any(support):
            raise ValueError(f"Shift {shift} has no common support")
        raw_affinity[shift], _, _ = _affinity_on_support(
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
    mu_shift = float(existing["mu_shift"])
    denominator = 1.0 - mu_shift
    if abs(denominator) <= recurrence.NORMALIZATION_TOLERANCE:
        raise ValueError("Recurrence shift denominator is unstable")
    if not np.isclose(raw_affinity[0], float(existing["A_obs"]), atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError("Shift-zero affinity disagrees with existing A_obs")
    if not np.isclose(raw_affinity[1:].mean(), mu_shift, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError("Nonzero-shift affinity mean disagrees with existing mu_shift")

    normalized = (raw_affinity - mu_shift) / denominator
    zero_rank = int(1 + np.count_nonzero(raw_affinity > raw_affinity[0]))
    zero_percentile = float(np.mean(raw_affinity <= raw_affinity[0]))
    best_shift = int(np.argmax(raw_affinity))
    details = {
        "A_obs": float(existing["A_obs"]),
        "mu_shift": mu_shift,
        "R_pair": float(existing["R_pair"]),
        "A_obs_minus_mu_shift": float(existing["A_obs"]) - mu_shift,
        "zero_shift_rank": zero_rank,
        "zero_shift_percentile": zero_percentile,
        "best_shift_bins": best_shift,
        "best_shift_minutes": best_shift * bin_minutes,
        "best_shift_affinity": float(raw_affinity[best_shift]),
    }
    return normalized, {**details, "raw_affinity": raw_affinity}


def _existing_recurrence_summary(
    animal_dir: Path,
    resolution_minutes: int,
) -> pd.Series:
    frame = pd.read_csv(
        animal_dir
        / "Occupancy_Recurrence_Output"
        / "occupancy_recurrence_summary.csv"
    )
    rows = frame.loc[frame["resolution_minutes"].eq(resolution_minutes)]
    if len(rows) != 1:
        raise ValueError(f"Expected one recurrence summary row for {animal_dir.name} at {resolution_minutes} minutes")
    return rows.iloc[0]


def _recurrence_audits(
    animal: str,
    animal_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    curve_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    summary_values: dict[str, object] = {"Animal": animal}
    for resolution_minutes, n_bins in RECURRENCE_RESOLUTIONS:
        occupancy, valid_masks, cycle_indices, _frp = _recurrence_inputs(
            animal_dir,
            n_bins,
        )
        pair_details = []
        for pair_index in range(len(cycle_indices) - 1):
            normalized, details = _alignment_curve(
                occupancy[pair_index],
                occupancy[pair_index + 1],
                valid_masks[pair_index],
                valid_masks[pair_index + 1],
                resolution_minutes,
            )
            pair_details.append(details)
            for shift in range(n_bins):
                curve_rows.append(
                    {
                        "Animal": animal,
                        "resolution_minutes": resolution_minutes,
                        "cycle_A": cycle_indices[pair_index],
                        "cycle_B": cycle_indices[pair_index + 1],
                        "shift_bins": shift,
                        "shift_minutes": shift * resolution_minutes,
                        "raw_affinity": details["raw_affinity"][shift],
                        "normalized_shift_relative_affinity": normalized[shift],
                    }
                )
            pair_rows.append(
                {
                    "Animal": animal,
                    "resolution_minutes": resolution_minutes,
                    "cycle_A": cycle_indices[pair_index],
                    "cycle_B": cycle_indices[pair_index + 1],
                    **{
                        key: details[key]
                        for key in (
                            "A_obs",
                            "mu_shift",
                            "A_obs_minus_mu_shift",
                            "R_pair",
                            "zero_shift_rank",
                            "zero_shift_percentile",
                            "best_shift_bins",
                            "best_shift_minutes",
                            "best_shift_affinity",
                        )
                    },
                }
            )

        existing_summary = _existing_recurrence_summary(
            animal_dir,
            resolution_minutes,
        )
        mean_r = float(np.mean([row["R_pair"] for row in pair_details]))
        if not np.isclose(
            mean_r,
            float(existing_summary["mean_R_pair"]),
            atol=NUMERICAL_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError(f"Mean recurrence disagrees with existing output for {animal} at {resolution_minutes} minutes")
        summary_values[f"mean_{resolution_minutes}"] = mean_r
        summary_values[f"median_pct_{resolution_minutes}"] = float(
            np.median([row["zero_shift_percentile"] for row in pair_details])
        )

    return curve_rows, pair_rows, {
        "Animal": animal,
        "Recurrence5_mean_R_pair": summary_values["mean_5"],
        "Recurrence10_mean_R_pair": summary_values["mean_10"],
        "median_zero_shift_percentile_5min": summary_values["median_pct_5"],
        "median_zero_shift_percentile_10min": summary_values["median_pct_10"],
    }


def _format_metric_text(mi: dict[str, object]) -> str:
    return (
        f"raw MI={float(mi['observed']):.4f} bits | "
        f"null mean={float(mi['null_mean']):.4f} | "
        f"excess={float(mi['excess']):.4f}"
    )


def _plot_mi_figure(
    output_path: Path,
    mi_profiles: dict[str, dict[str, object]],
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 7.2),
        gridspec_kw={"height_ratios": [5.0, 0.9]},
        sharex="col",
    )
    heatmap_axes = axes[0, :]
    contribution_axes = axes[1, :]
    contribution_max = max(
        float(np.max(mi_profiles[animal]["contribution"]))
        for animal in ("714H", "714D")
    )
    contribution_max = max(contribution_max, 1e-6)
    heatmap_images = []
    for column, animal in enumerate(("714H", "714D")):
        profile = mi_profiles[animal]
        heatmap_ax = heatmap_axes[column]
        image = heatmap_ax.imshow(
            profile["probability"].T,
            origin="lower",
            aspect="auto",
            extent=[0, 24, 0, len(NONREST_BEHAVIORS)],
            vmin=0,
            vmax=1.0,
            cmap="viridis",
        )
        heatmap_images.append(image)
        heatmap_ax.set_title(f"{animal}\n{_format_metric_text(profile)}", fontsize=10)
        heatmap_ax.set_yticks(np.arange(len(NONREST_BEHAVIORS)) + 0.5)
        heatmap_ax.set_yticklabels(NONREST_BEHAVIORS, fontsize=8)
        heatmap_ax.set_xticks(np.arange(0, 25, 4))
        heatmap_ax.set_xlim(0, 24)
        heatmap_ax.set_ylabel("Non-rest behavior")
        heatmap_ax.grid(False)

        contribution_ax = contribution_axes[column]
        contribution_ax.bar(
            np.arange(MI_PHASE_BINS) * 24.0 / MI_PHASE_BINS + 1.0,
            profile["contribution"],
            width=1.95,
            color="darkorange",
            edgecolor="none",
        )
        contribution_ax.set_ylim(0, contribution_max * 1.15)
        contribution_ax.set_xlim(0, 24)
        contribution_ax.set_xticks(np.arange(0, 25, 4))
        contribution_ax.set_ylabel("c(CT)", fontsize=8)
        contribution_ax.set_xlabel("CT (hours)")
        contribution_ax.tick_params(axis="both", labelsize=8)
        contribution_ax.text(
            0.99,
            0.85,
            f"sum = {profile['contribution_sum']:.6f} bits",
            transform=contribution_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    for heatmap_ax in heatmap_axes:
        heatmap_ax.set_xlabel("")
        heatmap_ax.set_xticklabels([])
    for contribution_ax, animal in zip(contribution_axes, ("714H", "714D")):
        contribution_ax.set_title(f"{animal}: per-CT contribution", fontsize=9)
    fig.colorbar(
        heatmap_images[0],
        ax=list(heatmap_axes),
        fraction=0.025,
        pad=0.02,
        label="P(behavior | CT, non-rest)",
    )
    fig.suptitle(
        "Conditional 8-state MI interpretability: 714H versus 714D",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Heatmaps use 12 equal CT bins and complete cycles only; strips are raw observed MI contributions.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.91, hspace=0.45, wspace=0.18)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    ax: plt.Axes,
    occupancy: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    display_bin_minutes: int,
    show_y_labels: bool,
) -> None:
    display_mask = np.broadcast_to(~valid_mask[None, :], occupancy.T.shape)
    display = np.ma.masked_where(display_mask, occupancy.T)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")
    n_bins = occupancy.shape[0]
    ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[0, 24, 0, len(NONREST_BEHAVIORS)],
        vmin=0,
        vmax=1,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.tick_params(axis="both", labelsize=7)
    ax.set_yticks(np.arange(len(NONREST_BEHAVIORS)) + 0.5)
    if show_y_labels:
        ax.set_yticklabels(NONREST_BEHAVIORS, fontsize=7)
        ax.set_ylabel("Non-rest behavior", fontsize=8)
    else:
        ax.set_yticklabels([])


def _plot_shift_panel(
    ax: plt.Axes,
    pair_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    animal: str,
    resolution_minutes: int,
    use_resolution_for_curve: int,
) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73")
    plotted = []
    for pair_index, pair in enumerate(pair_rows):
        selected = [
            row
            for row in curve_rows
            if row["Animal"] == animal
            and row["resolution_minutes"] == use_resolution_for_curve
            and row["cycle_A"] == pair["cycle_A"]
            and row["cycle_B"] == pair["cycle_B"]
        ]
        selected.sort(key=lambda row: row["shift_bins"])
        shifts = np.asarray([row["shift_bins"] for row in selected], dtype=int)
        n_bins = len(shifts)
        signed_bins = np.where(shifts <= n_bins // 2, shifts, shifts - n_bins)
        order = np.argsort(signed_bins)
        x_hours = signed_bins[order] * use_resolution_for_curve / 60.0
        y_values = np.asarray(
            [row["normalized_shift_relative_affinity"] for row in selected],
            dtype=float,
        )[order]
        label = (
            f"{pair['cycle_A']}-{pair['cycle_B']}: R={pair['R_pair']:.3f}, "
            f"rank={pair['zero_shift_rank']}/{n_bins}"
        )
        line = ax.plot(x_hours, y_values, color=colors[pair_index], linewidth=0.9, alpha=0.75, label=label)[0]
        plotted.append((x_hours, y_values, line))

    if not plotted:
        raise ValueError(f"No shift curves found for {animal}")
    mean_x = plotted[0][0]
    mean_y = np.mean(np.stack([item[1] for item in plotted], axis=0), axis=0)
    ax.plot(mean_x, mean_y, color="#222222", linewidth=2.0, label=f"mean R={np.mean([p['R_pair'] for p in pair_rows]):.3f}")
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.text(0.5, 0.94, "Actual CT alignment", transform=ax.transAxes, fontsize=7, ha="center")
    ax.set_xlim(-12, 12)
    ax.set_xlabel("Relative CT shift (hours)", fontsize=8)
    ax.set_ylabel("G(s)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(f"{animal}: normalized shift alignment", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)


def _plot_recurrence_figure(
    output_path: Path,
    recurrence_data: dict[str, dict[str, object]],
    curve_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    display_minutes: int,
    title: str,
    subtitle: str,
) -> None:
    fig = plt.figure(figsize=(15.0, 14.0))
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[1, 1, 1, 1, 1.8],
        hspace=0.55,
        wspace=0.18,
    )
    heatmap_image = None
    heatmap_axes_by_column = []
    for column, animal in enumerate(("714G", "714D")):
        animal_data = recurrence_data[animal]
        occupancy = animal_data["occupancy5"]
        masks = animal_data["masks5"]
        if display_minutes == 10:
            occupancy = occupancy.reshape(4, 144, 2, len(NONREST_BEHAVIORS)).mean(axis=2)
            masks = masks.reshape(4, 144, 2).all(axis=2)
        column_axes = []
        for cycle_index in range(4):
            ax = fig.add_subplot(grid[cycle_index, column])
            column_axes.append(ax)
            _plot_heatmap(
                ax,
                occupancy[cycle_index],
                masks[cycle_index],
                f"{animal} cycle {cycle_index + 1}",
                display_minutes,
                show_y_labels=column == 0,
            )
            if cycle_index < 3:
                ax.set_xticklabels([])
            if heatmap_image is None:
                heatmap_image = ax.images[0]
        column_axes[-1].set_xlabel("CT (hours)", fontsize=8)
        heatmap_axes_by_column.append(column_axes)
        curve_ax = fig.add_subplot(grid[4, column])
        selected_pairs = [
            row
            for row in pair_rows
            if row["Animal"] == animal and row["resolution_minutes"] == 5
        ]
        _plot_shift_panel(
            curve_ax,
            selected_pairs,
            curve_rows,
            animal,
            display_minutes,
            use_resolution_for_curve=5,
        )
    fig.colorbar(
        heatmap_image,
        ax=[axis for column_axes in heatmap_axes_by_column for axis in column_axes],
        fraction=0.012,
        pad=0.01,
        label="Original within-bin non-rest occupancy fraction",
    )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=9)
    fig.text(
        0.5,
        0.005,
        "Gray denotes missing support; rest is implicit in the remaining fraction. Shift curves use the primary 5-minute scorer.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_metric(
    output_path: Path,
    mi_profile: dict[str, object],
    recurrence_data: dict[str, object],
    curve_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> None:
    fig = plt.figure(figsize=(16.0, 10.5))
    grid = fig.add_gridspec(
        5,
        2,
        width_ratios=[1.0, 1.25],
        height_ratios=[4.4, 0.75, 1.0, 1.0, 1.6],
        hspace=0.55,
        wspace=0.3,
    )

    mi_ax = fig.add_subplot(grid[0, 0])
    mi_image = mi_ax.imshow(
        mi_profile["probability"].T,
        origin="lower",
        aspect="auto",
        extent=[0, 24, 0, len(NONREST_BEHAVIORS)],
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    mi_ax.set_title(
        "714G phase × behavior profile\n"
        f"raw MI={mi_profile['observed']:.4f}; excess={mi_profile['excess']:.4f} bits",
        fontsize=10,
    )
    mi_ax.set_xticks(np.arange(0, 25, 4))
    mi_ax.set_yticks(np.arange(len(NONREST_BEHAVIORS)) + 0.5)
    mi_ax.set_yticklabels(NONREST_BEHAVIORS, fontsize=8)
    mi_ax.set_ylabel("Non-rest behavior")
    mi_ax.set_xlabel("CT (hours)")
    fig.colorbar(mi_image, ax=mi_ax, fraction=0.045, pad=0.03, label="P(behavior | CT, non-rest)")

    contribution_ax = fig.add_subplot(grid[1, 0])
    contribution_ax.bar(
        np.arange(MI_PHASE_BINS) * 24.0 / MI_PHASE_BINS + 1.0,
        mi_profile["contribution"],
        width=1.95,
        color="darkorange",
        edgecolor="none",
    )
    contribution_ax.set_xlim(0, 24)
    contribution_ax.set_ylabel("c(CT)", fontsize=8)
    contribution_ax.set_xticks(np.arange(0, 25, 4))
    contribution_ax.tick_params(axis="both", labelsize=8)
    contribution_ax.set_xlabel("CT (hours)", fontsize=8)

    occupancy = recurrence_data["occupancy5"]
    masks = recurrence_data["masks5"]
    heatmap_image = None
    recurrence_heatmap_axes = []
    for cycle_index in range(4):
        ax = fig.add_subplot(grid[cycle_index, 1])
        recurrence_heatmap_axes.append(ax)
        _plot_heatmap(
            ax,
            occupancy[cycle_index],
            masks[cycle_index],
            f"714G cycle {cycle_index + 1}",
            5,
            show_y_labels=cycle_index == 0,
        )
        if cycle_index < 3:
            ax.set_xticklabels([])
        if heatmap_image is None:
            heatmap_image = ax.images[0]
    recurrence_heatmap_axes[-1].set_xlabel("CT (hours)", fontsize=8)
    fig.colorbar(
        heatmap_image,
        ax=recurrence_heatmap_axes,
        fraction=0.025,
        pad=0.02,
        label="Original within-bin non-rest occupancy fraction",
    )

    curve_ax = fig.add_subplot(grid[4, 1])
    selected_pairs = [
        row
        for row in pair_rows
        if row["Animal"] == "714G" and row["resolution_minutes"] == 5
    ]
    _plot_shift_panel(
        curve_ax,
        selected_pairs,
        curve_rows,
        "714G",
        5,
        use_resolution_for_curve=5,
    )
    fig.suptitle(
        "714G cross-metric interpretability: modest MI and stronger recurrence",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Left: 12-bin conditional MI representation. Right: native 5-minute occupancy and alignment representation.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cohort_root = parse_args()
    output_dir = cohort_root / "Metric_Interpretability_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mi_profiles: dict[str, dict[str, object]] = {}
    mi_profile_rows: list[dict[str, object]] = []
    mi_summary_rows: list[dict[str, object]] = []
    for animal in MI_ANIMALS:
        group = ANIMALS[animal]
        profile_rows, summary = _mi_profile(animal, cohort_root / group / animal)
        mi_profile_rows.extend(profile_rows)
        mi_profiles[animal] = summary
        mi_summary_rows.append(
            {
                "Animal": animal,
                "MI8_observed_raw_bits": summary["observed"],
                "MI8_null_mean_bits": summary["null_mean"],
                "MI8_conditional_excess_bits": summary["excess"],
                "phase_contribution_sum_bits": summary["contribution_sum"],
                "reconstruction_error_bits": summary["reconstruction_error"],
            }
        )

    recurrence_data: dict[str, dict[str, object]] = {}
    all_curve_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    animal_summary_rows: list[dict[str, object]] = []
    for animal in RECURRENCE_ANIMALS:
        group = ANIMALS[animal]
        animal_dir = cohort_root / group / animal
        occupancy5, masks5, cycle_indices, _frp = _recurrence_inputs(animal_dir, 288)
        curves, pairs, animal_summary = _recurrence_audits(animal, animal_dir)
        recurrence_data[animal] = {
            "occupancy5": occupancy5,
            "masks5": masks5,
            "cycle_indices": cycle_indices,
        }
        all_curve_rows.extend(curves)
        all_pair_rows.extend(pairs)
        animal_summary_rows.append(animal_summary)

    pd.DataFrame(mi_profile_rows, columns=MI_PROFILE_COLUMNS).to_csv(
        output_dir / "mi_interpretability_phase_profiles.csv",
        index=False,
    )
    pd.DataFrame(mi_summary_rows, columns=MI_SUMMARY_COLUMNS).to_csv(
        output_dir / "mi_interpretability_summary.csv",
        index=False,
    )
    pd.DataFrame(all_curve_rows, columns=SHIFT_CURVE_COLUMNS).to_csv(
        output_dir / "recurrence_interpretability_shift_curves.csv",
        index=False,
    )
    pd.DataFrame(all_pair_rows, columns=PAIR_SUMMARY_COLUMNS).to_csv(
        output_dir / "recurrence_interpretability_pair_summary.csv",
        index=False,
    )
    pd.DataFrame(animal_summary_rows, columns=ANIMAL_SUMMARY_COLUMNS).to_csv(
        output_dir / "recurrence_interpretability_animal_summary.csv",
        index=False,
    )

    _plot_mi_figure(
        output_dir / "MI_714H_vs_714D.png",
        mi_profiles,
    )
    _plot_recurrence_figure(
        output_dir / "Recurrence_714G_vs_714D_5min.png",
        recurrence_data,
        all_curve_rows,
        all_pair_rows,
        display_minutes=5,
        title="5-minute occupancy recurrence interpretability: 714G versus 714D",
        subtitle="Native 5-minute occupancy display and primary 5-minute normalized shift alignment",
    )
    _plot_recurrence_figure(
        output_dir / "Recurrence_714G_vs_714D_10min_display.png",
        recurrence_data,
        all_curve_rows,
        all_pair_rows,
        display_minutes=10,
        title="10-minute display version: 714G versus 714D",
        subtitle="10-minute display averaging; primary recurrence scored at 5-minute resolution",
    )
    _plot_cross_metric(
        output_dir / "CrossMetric_714G.png",
        mi_profiles["714G"],
        recurrence_data["714G"],
        all_curve_rows,
        all_pair_rows,
    )
    print(f"Wrote interpretability artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
