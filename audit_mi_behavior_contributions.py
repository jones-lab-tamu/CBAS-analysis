"""Audit the conditional 8-state MI contribution by non-rest behavior.

This is a standalone diagnostic.  It reuses the existing MI phase assignment,
complete-cycle construction, and cycle-shift null without changing the
production MI analysis or its outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase
import phase_behavior_occupancy_recurrence as occupancy_integration


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
NONREST_BEHAVIORS = tuple(
    frozen_phase.BEHAVIORS[index]
    for index in frozen_phase.NONRESTING_BEHAVIOR_INDICES
)
OUTPUT_DIRNAME = "MI_Behavior_Contribution_Audit"
PRIMARY_COMPONENT = "conditional_8state_nonrest"
TOLERANCE_BITS = 1e-12

AUDIT_COLUMNS = [
    "Group",
    "Animal",
    "Behavior",
    "P_behavior_nonrest",
    "raw_MI_contribution_bits",
    "fraction_raw_MI",
    "null_mean_MI_contribution_bits",
    "excess_MI_contribution_bits",
]

SUMMARY_COLUMNS = [
    "Group",
    "Animal",
    "MI8_observed_raw_bits",
    "MI8_null_mean_bits",
    "MI8_conditional_excess_bits",
    "top_raw_behavior",
    "top_raw_behavior_contribution_bits",
    "top_raw_behavior_fraction",
    "top2_raw_fraction",
    "top3_raw_fraction",
    "top_excess_behavior",
    "top_excess_behavior_contribution_bits",
    "largest_positive_excess_bits",
    "top2_positive_excess_bits",
    "top3_positive_excess_bits",
    "total_negative_excess_bits",
    "raw_reconstruction_error",
    "excess_reconstruction_error",
]


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Audit conditional 8-state MI contributions by behavior."
    )
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the eight animal folders.",
    )
    return parser.parse_args().cohort_root.expanduser().resolve()


def _primary_mi_metrics(animal_dir: Path) -> dict[str, float]:
    """Read the existing primary conditional MI values for one animal."""

    path = animal_dir / "MI_Output" / "mi_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing MI result file: {path}")
    frame = pd.read_csv(path)
    rows = frame.loc[frame["component"] == PRIMARY_COMPONENT]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one {PRIMARY_COMPONENT!r} row in {path}, found {len(rows)}"
        )
    row = rows.iloc[0]
    return {
        "raw": float(row["MI_raw_bits"]),
        "null_mean": float(row["MI_null_mean_bits"]),
        "excess": float(row["MI_excess_bits"]),
    }


def _behavior_contributions(counts: np.ndarray) -> np.ndarray:
    """Return exact per-behavior MI contributions for one or many tables."""

    values = np.asarray(counts, dtype=float)
    one_table = values.ndim == 2
    if one_table:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D contingency table, got {values.shape}")

    totals = values.sum(axis=(1, 2))
    if np.any(totals <= 0.0):
        raise ValueError("Cannot decompose a contingency table with no observations")

    joint = values / totals[:, None, None]
    phase_probability = joint.sum(axis=2)
    behavior_probability = joint.sum(axis=1)
    expected = phase_probability[:, :, None] * behavior_probability[:, None, :]
    positive = joint > 0.0
    ratio = np.divide(joint, expected, out=np.ones_like(joint), where=positive)
    cell_contributions = np.where(
        positive,
        joint * np.log2(ratio),
        0.0,
    )
    result = cell_contributions.sum(axis=1)
    return result[0] if one_table else result


def _build_complete_cycles(
    animal_dir: Path,
) -> tuple[
    float,
    list[tuple[int, np.ndarray, np.ndarray]],
    int,
]:
    """Build the same complete-cycle event lists used by the MI analysis."""

    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(
            f"Source indices are missing for {animal_dir.name}: {missing_indices}"
        )
    (
        _reported_frp_hours,
        computational_frp_hours,
        _start_ct,
        complete_cycle_specs,
        _frp_phase_output_dir,
    ) = frozen_phase.load_frp_phase_solution(animal_dir)
    if len(complete_cycle_specs) < 2:
        raise ValueError(
            f"At least two complete cycles are required for {animal_dir.name}"
        )

    relative_times, labels, _expected_samples_per_hour, invalid_rows = (
        occupancy_integration._load_classified_samples(input_files)
    )
    complete_cycles = occupancy_integration._build_complete_cycles(
        relative_times,
        labels,
        complete_cycle_specs,
        computational_frp_hours,
    )
    return computational_frp_hours, complete_cycles, invalid_rows


def _audit_animal(group: str, animal: str, cohort_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    animal_dir = cohort_root / group / animal
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Missing animal directory: {animal_dir}")

    computational_frp_hours, complete_cycles, invalid_rows = _build_complete_cycles(
        animal_dir
    )
    primary = _primary_mi_metrics(animal_dir)

    observed_full_counts = frozen_phase.observed_counts_from_complete_cycles(
        complete_cycles,
        frozen_phase.PRIMARY_PHASE_BINS,
        frp_hours=computational_frp_hours,
    )
    observed_conditional_counts = frozen_phase.conditional_nonrest_counts(
        observed_full_counts
    )
    observed_raw_contributions = _behavior_contributions(observed_conditional_counts)
    observed_raw_from_decomposition = float(observed_raw_contributions.sum())
    raw_reconstruction_error = observed_raw_from_decomposition - primary["raw"]
    if not np.isclose(
        observed_raw_from_decomposition,
        primary["raw"],
        rtol=0.0,
        atol=TOLERANCE_BITS,
    ):
        raise RuntimeError(
            f"Raw MI reconstruction failed for {animal}: "
            f"decomposed={observed_raw_from_decomposition:.17g}, "
            f"existing={primary['raw']:.17g}"
        )

    total_nonrest = float(observed_conditional_counts.sum())
    behavior_occupancy = observed_conditional_counts.sum(axis=0) / total_nonrest
    occupancy_error = float(behavior_occupancy.sum() - 1.0)
    if not np.isclose(behavior_occupancy.sum(), 1.0, rtol=0.0, atol=TOLERANCE_BITS):
        raise RuntimeError(
            f"Non-rest occupancy does not sum to one for {animal}: "
            f"error={occupancy_error:.17g}"
        )

    rng = np.random.default_rng(frozen_phase.RANDOM_SEED)
    offsets_hours = rng.uniform(
        0.0,
        computational_frp_hours,
        size=(frozen_phase.N_PERMUTATIONS, len(complete_cycles)),
    )
    null_full_counts = frozen_phase.calculate_null_tables(
        complete_cycles,
        offsets_hours,
        frozen_phase.PRIMARY_PHASE_BINS,
        frp_hours=computational_frp_hours,
    )
    null_conditional_counts = frozen_phase.conditional_nonrest_counts(null_full_counts)
    null_contributions = _behavior_contributions(null_conditional_counts)
    null_mean_contributions = null_contributions.mean(axis=0)
    null_mean_from_decomposition = float(null_mean_contributions.sum())
    null_mean_reconstruction_error = (
        null_mean_from_decomposition - primary["null_mean"]
    )
    if not np.isclose(
        null_mean_from_decomposition,
        primary["null_mean"],
        rtol=0.0,
        atol=TOLERANCE_BITS,
    ):
        raise RuntimeError(
            f"Null MI reconstruction failed for {animal}: "
            f"decomposed={null_mean_from_decomposition:.17g}, "
            f"existing={primary['null_mean']:.17g}"
        )

    excess_contributions = observed_raw_contributions - null_mean_contributions
    excess_reconstruction_error = float(excess_contributions.sum() - primary["excess"])
    if not np.isclose(
        excess_contributions.sum(),
        primary["excess"],
        rtol=0.0,
        atol=TOLERANCE_BITS,
    ):
        raise RuntimeError(
            f"Excess MI reconstruction failed for {animal}: "
            f"decomposed={excess_contributions.sum():.17g}, "
            f"existing={primary['excess']:.17g}"
        )

    raw_fractions = observed_raw_contributions / primary["raw"]
    raw_order = np.argsort(observed_raw_contributions)[::-1]
    excess_order = np.argsort(excess_contributions)[::-1]
    positive_excess = np.sort(excess_contributions[excess_contributions > 0.0])[::-1]

    def positive_sum(count: int) -> float:
        return float(positive_excess[:count].sum())

    audit_rows = [
        {
            "Group": group,
            "Animal": animal,
            "Behavior": behavior,
            "P_behavior_nonrest": float(behavior_occupancy[index]),
            "raw_MI_contribution_bits": float(observed_raw_contributions[index]),
            "fraction_raw_MI": float(raw_fractions[index]),
            "null_mean_MI_contribution_bits": float(null_mean_contributions[index]),
            "excess_MI_contribution_bits": float(excess_contributions[index]),
        }
        for index, behavior in enumerate(NONREST_BEHAVIORS)
    ]
    summary_row = {
        "Group": group,
        "Animal": animal,
        "MI8_observed_raw_bits": primary["raw"],
        "MI8_null_mean_bits": primary["null_mean"],
        "MI8_conditional_excess_bits": primary["excess"],
        "top_raw_behavior": NONREST_BEHAVIORS[raw_order[0]],
        "top_raw_behavior_contribution_bits": float(
            observed_raw_contributions[raw_order[0]]
        ),
        "top_raw_behavior_fraction": float(raw_fractions[raw_order[0]]),
        "top2_raw_fraction": float(raw_fractions[raw_order[:2]].sum()),
        "top3_raw_fraction": float(raw_fractions[raw_order[:3]].sum()),
        "top_excess_behavior": NONREST_BEHAVIORS[excess_order[0]],
        "top_excess_behavior_contribution_bits": float(
            excess_contributions[excess_order[0]]
        ),
        "largest_positive_excess_bits": float(positive_sum(1)),
        "top2_positive_excess_bits": float(positive_sum(2)),
        "top3_positive_excess_bits": float(positive_sum(3)),
        "total_negative_excess_bits": float(
            excess_contributions[excess_contributions < 0.0].sum()
        ),
        "raw_reconstruction_error": float(raw_reconstruction_error),
        "excess_reconstruction_error": float(excess_reconstruction_error),
    }
    print(
        f"{animal}: {len(complete_cycles)} complete cycles, "
        f"invalid rows excluded={invalid_rows}, "
        f"raw reconstruction error={raw_reconstruction_error:.3e}, "
        f"excess reconstruction error={excess_reconstruction_error:.3e}"
    )
    return audit_rows, summary_row


def _zero_centered_norm(values: np.ndarray) -> Normalize:
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        maximum = 1.0
    return TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)


def _save_heatmaps(
    output_dir: Path,
    audit_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> Path:
    animal_order = [animal for _, animal in ANIMALS]
    fraction_matrix = audit_frame.pivot(
        index="Animal", columns="Behavior", values="fraction_raw_MI"
    ).loc[animal_order, list(NONREST_BEHAVIORS)].to_numpy()
    excess_matrix = audit_frame.pivot(
        index="Animal", columns="Behavior", values="excess_MI_contribution_bits"
    ).loc[animal_order, list(NONREST_BEHAVIORS)].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    heatmap_specs = (
        (
            axes[0],
            fraction_matrix,
            "Fraction of observed raw MI",
            "Raw MI fraction",
            _zero_centered_norm(fraction_matrix),
        ),
        (
            axes[1],
            excess_matrix,
            "Signed excess MI contribution (bits)",
            "Excess MI (bits)",
            _zero_centered_norm(excess_matrix),
        ),
    )
    for axis, values, title, colorbar_label, norm in heatmap_specs:
        image = axis.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xticks(np.arange(len(NONREST_BEHAVIORS)))
        axis.set_xticklabels(NONREST_BEHAVIORS, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(animal_order)))
        axis.set_yticklabels(animal_order)
        axis.set_xlabel("Behavior")
        axis.set_ylabel("Animal")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)

    fig.suptitle(
        "Conditional 8-state MI behavior-contribution audit\n"
        "Fixed behavior and animal order; signed scales centered at zero"
    )
    path = output_dir / "mi_behavior_contribution_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_audit(cohort_root: Path) -> dict[str, object]:
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for group, animal in ANIMALS:
        animal_audit_rows, summary_row = _audit_animal(group, animal, cohort_root)
        audit_rows.extend(animal_audit_rows)
        summary_rows.append(summary_row)

    audit_frame = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "mi_behavior_contribution_audit.csv"
    summary_csv = output_dir / "mi_behavior_contribution_summary.csv"
    audit_frame.to_csv(audit_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    figure_path = _save_heatmaps(output_dir, audit_frame, summary_frame)
    return {
        "output_dir": output_dir,
        "audit_csv": audit_csv,
        "summary_csv": summary_csv,
        "figure": figure_path,
        "audit_frame": audit_frame,
        "summary_frame": summary_frame,
    }


def main() -> None:
    cohort_root = parse_args()
    result = run_audit(cohort_root)
    print(f"Audit CSV: {result['audit_csv']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Figure: {result['figure']}")
    print(result["summary_frame"].to_string(index=False))


if __name__ == "__main__":
    main()
