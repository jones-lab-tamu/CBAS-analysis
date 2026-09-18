"""Compute raw and gap-merged WTA bout support for the eight pilot animals.

This is a descriptive support audit only.  It consumes the existing WTA
classifier labels, frozen FRP cycle boundaries, and prior estimability output.
It does not estimate FRP, change CT assignment, choose an eligibility rule, or
calculate a repertoire metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase_behavior_mutual_information as frozen_phase


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
N_CYCLES = 4
FILE_DURATION_MINUTES = frozen_phase.FILE_DURATION_MINUTES
MERGE_THRESHOLDS_SECONDS = (0.5, 1.0, 2.0)
PRIMARY_MERGE_SECONDS = 1.0
GAP_COMPARISON_TOLERANCE_SECONDS = 1e-9
OUTPUT_DIRNAME = "Behavior_Bout_Support_Audit"


def _parse_args() -> tuple[Path, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cohort_root",
        type=Path,
        help="Updated Cohort_Data directory containing the eight animal folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <cohort_root>/Behavior_Bout_Support_Audit).",
    )
    args = parser.parse_args()
    return args.cohort_root.expanduser().resolve(), (
        args.output_dir.expanduser().resolve() if args.output_dir else None
    )


def _load_authoritative_estimability(cohort_root: Path) -> pd.DataFrame:
    audit_dir = cohort_root / "Behavior_Phase_Estimability_Audit"
    support_path = audit_dir / "behavior_phase_support_audit.csv"
    stability_path = audit_dir / "behavior_phase_stability_summary.csv"
    if not support_path.is_file() or not stability_path.is_file():
        raise FileNotFoundError(
            "Expected existing behavior estimability outputs before bout audit: "
            f"{support_path} and {stability_path}"
        )

    support = pd.read_csv(support_path)
    stability = pd.read_csv(stability_path)
    required_support = {
        "Group",
        "Animal",
        "Behavior",
        "total_occupancy_minutes",
        "cycle1_occupancy_minutes",
        "cycle2_occupancy_minutes",
        "cycle3_occupancy_minutes",
        "cycle4_occupancy_minutes",
        "cycles_with_nonzero_occupancy",
    }
    required_stability = {
        "Group",
        "Animal",
        "Behavior",
        "mean_LOCO_W1_h",
        "max_LOCO_W1_h",
    }
    if not required_support.issubset(support.columns):
        raise ValueError(
            f"Existing support output is missing columns: "
            f"{sorted(required_support.difference(support.columns))}"
        )
    if not required_stability.issubset(stability.columns):
        raise ValueError(
            f"Existing stability output is missing columns: "
            f"{sorted(required_stability.difference(stability.columns))}"
        )

    key_columns = ["Group", "Animal", "Behavior"]
    if support.duplicated(key_columns).any() or stability.duplicated(key_columns).any():
        raise ValueError("Existing estimability outputs contain duplicate animal/behavior rows")
    expected_keys = {
        (group, animal, behavior)
        for group, animal in ANIMALS
        for behavior in NONREST_BEHAVIORS
    }
    support_keys = set(map(tuple, support[key_columns].itertuples(index=False, name=None)))
    stability_keys = set(map(tuple, stability[key_columns].itertuples(index=False, name=None)))
    if support_keys != expected_keys or stability_keys != expected_keys:
        raise ValueError("Existing estimability outputs do not cover exactly the eight pilot animals and eight non-rest behaviors")
    if not np.isfinite(
        support[
            [
                "total_occupancy_minutes",
                "cycle1_occupancy_minutes",
                "cycle2_occupancy_minutes",
                "cycle3_occupancy_minutes",
                "cycle4_occupancy_minutes",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Existing WTA minute support contains non-finite values")
    if not np.isfinite(
        stability[["mean_LOCO_W1_h", "max_LOCO_W1_h"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Existing LOCO W1 support contains non-finite values")
    return support.merge(
        stability[key_columns + ["mean_LOCO_W1_h", "max_LOCO_W1_h"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )


def _load_animal_frames(
    animal_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]], int]:
    input_files, missing_indices = frozen_phase.discover_input_files(animal_dir)
    if missing_indices:
        raise ValueError(f"{animal_dir.name} has missing source-file indices: {missing_indices}")
    if any(
        current_index != previous_index + 1
        for (previous_index, _), (current_index, _) in zip(input_files, input_files[1:])
    ):
        raise ValueError(f"{animal_dir.name} source-file sequence is discontinuous")

    first_file_index = input_files[0][0]
    label_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    file_spans: list[tuple[int, int, int]] = []
    invalid_rows = 0
    global_start = 0
    for file_index, path in input_files:
        n_rows, row_positions, labels = frozen_phase.read_and_classify(path)
        invalid_rows += n_rows - len(labels)
        if n_rows == 0:
            raise ValueError(f"Empty source file would create a recording discontinuity: {path}")
        if len(labels) != n_rows or not np.array_equal(
            row_positions, np.arange(n_rows, dtype=np.int64)
        ):
            raise ValueError(
                f"Unclassifiable source rows would create a recording discontinuity: {path}"
            )
        file_start_hours = (
            file_index - first_file_index
        ) * FILE_DURATION_MINUTES / 60.0
        elapsed_hours = file_start_hours + (
            np.arange(n_rows, dtype=float) / n_rows
            * FILE_DURATION_MINUTES
            / 60.0
        )
        row_duration_seconds = FILE_DURATION_MINUTES * 60.0 / n_rows
        label_parts.append(labels.astype(np.int8, copy=False))
        time_parts.append(elapsed_hours)
        weight_parts.append(
            np.full(n_rows, row_duration_seconds, dtype=float)
        )
        global_end = global_start + n_rows
        file_spans.append((file_index, global_start, global_end))
        global_start = global_end

    if not label_parts:
        raise ValueError(f"No classified source frames found for {animal_dir}")
    return (
        np.concatenate(label_parts),
        np.concatenate(time_parts),
        np.concatenate(weight_parts),
        file_spans,
        invalid_rows,
    )


def _bout_counts_for_behavior(
    labels: np.ndarray,
    frame_weights_seconds: np.ndarray,
    behavior_index: int,
) -> tuple[int, dict[float, int]]:
    """Count raw runs and merge adjacent behavior runs across short gaps."""

    positions = np.flatnonzero(labels == behavior_index)
    if len(positions) == 0:
        return 0, {threshold: 0 for threshold in MERGE_THRESHOLDS_SECONDS}

    run_breaks = np.flatnonzero(np.diff(positions) > 1) + 1
    run_starts = np.concatenate(([0], run_breaks))
    run_ends = np.concatenate((run_breaks - 1, [len(positions) - 1]))
    raw_count = int(len(run_starts))
    gap_durations_seconds = np.asarray(
        [
            frame_weights_seconds[
                positions[run_ends[index]] + 1 : positions[run_starts[index + 1]]
            ].sum()
            for index in range(len(run_starts) - 1)
        ],
        dtype=float,
    )
    merged_counts = {
        threshold: int(
            1
            + np.count_nonzero(
                gap_durations_seconds >= threshold - GAP_COMPARISON_TOLERANCE_SECONDS
            )
        )
        if len(positions)
        else 0
        for threshold in MERGE_THRESHOLDS_SECONDS
    }
    return raw_count, merged_counts


def _cycle_bout_rows(
    labels: np.ndarray,
    times_hours: np.ndarray,
    weights_seconds: np.ndarray,
    complete_cycles: list[tuple[int, float, float]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], int]:
    cycle_rows: list[dict[str, object]] = []
    per_cycle: dict[int, dict[str, object]] = {}
    selected_frame_total = 0
    for cycle_index, start_boundary, end_boundary in complete_cycles:
        selected = (times_hours >= start_boundary) & (times_hours < end_boundary)
        if not selected.any():
            raise ValueError(f"Frozen full cycle {cycle_index} contains no classified frames")
        cycle_labels = labels[selected]
        cycle_weights = weights_seconds[selected]
        selected_frame_total += int(selected.sum())
        raw_counts: dict[str, int] = {}
        merged_counts: dict[float, dict[str, int]] = {
            threshold: {} for threshold in MERGE_THRESHOLDS_SECONDS
        }
        for behavior_position, behavior in enumerate(NONREST_BEHAVIORS):
            behavior_index = frozen_phase.NONRESTING_BEHAVIOR_INDICES[behavior_position]
            raw_count, behavior_merged_counts = _bout_counts_for_behavior(
                cycle_labels,
                cycle_weights,
                behavior_index,
            )
            raw_counts[behavior] = raw_count
            for threshold in MERGE_THRESHOLDS_SECONDS:
                merged_counts[threshold][behavior] = behavior_merged_counts[threshold]
            cycle_rows.append(
                {
                    "animal": "",
                    "genotype": "",
                    "behavior": behavior,
                    "cycle": cycle_index,
                    "wta_minutes": float(
                        cycle_weights[cycle_labels == behavior_index].sum() / 60.0
                    ),
                    "raw_bouts": raw_count,
                    "bouts_0p5s": behavior_merged_counts[0.5],
                    "bouts_1s": behavior_merged_counts[1.0],
                    "bouts_2s": behavior_merged_counts[2.0],
                }
            )
        per_cycle[cycle_index] = {
            "raw": raw_counts,
            "merged": merged_counts,
            "selected_frames": int(selected.sum()),
        }
    return cycle_rows, per_cycle, selected_frame_total


def _build_summary_rows(
    group: str,
    animal: str,
    authoritative: pd.DataFrame,
    per_cycle: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for behavior in NONREST_BEHAVIORS:
        authoritative_row = authoritative[
            (authoritative["Group"] == group)
            & (authoritative["Animal"] == animal)
            & (authoritative["Behavior"] == behavior)
        ].iloc[0]
        raw_cycle = [int(per_cycle[index]["raw"][behavior]) for index in range(1, N_CYCLES + 1)]
        merged_cycle = {
            threshold: [
                int(per_cycle[index]["merged"][threshold][behavior])
                for index in range(1, N_CYCLES + 1)
            ]
            for threshold in MERGE_THRESHOLDS_SECONDS
        }
        counts_by_rule = {
            "raw": raw_cycle,
            "0p5s": merged_cycle[0.5],
            "1s": merged_cycle[1.0],
            "2s": merged_cycle[2.0],
        }
        row: dict[str, object] = {
            "animal": animal,
            "genotype": group,
            "behavior": behavior,
            "total_wta_minutes": float(authoritative_row["total_occupancy_minutes"]),
            "cycles_with_nonzero_support": int(
                authoritative_row["cycles_with_nonzero_occupancy"]
            ),
            "raw_bouts_total": int(sum(raw_cycle)),
            "bouts_0p5s_total": int(sum(merged_cycle[0.5])),
            "bouts_1s_total": int(sum(merged_cycle[1.0])),
            "bouts_2s_total": int(sum(merged_cycle[2.0])),
            "bouts_1s_min_cycle": int(min(merged_cycle[1.0])),
            "bouts_1s_max_cycle": int(max(merged_cycle[1.0])),
            "bouts_1s_mean_cycle": float(np.mean(merged_cycle[1.0])),
            "bouts_1s_median_cycle": float(np.median(merged_cycle[1.0])),
            "mean_loco_w1_hours": float(authoritative_row["mean_LOCO_W1_h"]),
            "max_loco_w1_hours": float(authoritative_row["max_LOCO_W1_h"]),
        }
        for cycle_position in range(N_CYCLES):
            row[f"cycle{cycle_position + 1}_wta_minutes"] = float(
                authoritative_row[f"cycle{cycle_position + 1}_occupancy_minutes"]
            )
            row[f"raw_bouts_cycle{cycle_position + 1}"] = raw_cycle[cycle_position]
            row[f"bouts_0p5s_cycle{cycle_position + 1}"] = merged_cycle[0.5][cycle_position]
            row[f"bouts_1s_cycle{cycle_position + 1}"] = merged_cycle[1.0][cycle_position]
            row[f"bouts_2s_cycle{cycle_position + 1}"] = merged_cycle[2.0][cycle_position]
        rows.append(row)

        sensitivity_row: dict[str, object] = {
            "animal": animal,
            "genotype": group,
            "behavior": behavior,
        }
        for rule, values in counts_by_rule.items():
            total = int(sum(values))
            sensitivity_row[f"{rule}_bouts_total"] = total
            if rule != "raw":
                difference = total - int(sum(raw_cycle))
                sensitivity_row[f"{rule}_absolute_change_from_raw"] = difference
                sensitivity_row[f"{rule}_percent_change_from_raw"] = (
                    100.0 * difference / sum(raw_cycle) if sum(raw_cycle) else float("nan")
                )
        sensitivity_rows.append(sensitivity_row)
    return rows, sensitivity_rows


def _build_cycle_output_rows(
    group: str,
    animal: str,
    cycle_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output_rows = []
    for row in cycle_rows:
        copied = dict(row)
        copied["animal"] = animal
        copied["genotype"] = group
        output_rows.append(copied)
    return output_rows


def _save_scatter(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(
        frame[x_column],
        frame[y_column],
        color="tab:blue",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        s=38,
    )
    # Labels make sparse, high-instability, and other notable cells
    # identifiable without using genotype as a visual encoding.
    for _, row in frame.iterrows():
        axis.annotate(
            f"{row['animal']}\n{row['behavior']}",
            (row[x_column], row[y_column]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            alpha=0.65,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def run_audit(cohort_root: Path, output_dir: Path) -> dict[str, object]:
    authoritative = _load_authoritative_estimability(cohort_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    cycle_output_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    boundary_continuations = 0

    for group, animal in ANIMALS:
        animal_dir = cohort_root / group / animal
        if not animal_dir.is_dir():
            raise FileNotFoundError(f"Missing animal directory: {animal_dir}")
        (
            _reported_frp_hours,
            _computational_frp_hours,
            _start_ct,
            complete_cycles,
            _phase_dir,
        ) = frozen_phase.load_frp_phase_solution(animal_dir)
        if len(complete_cycles) != N_CYCLES:
            raise ValueError(f"Expected four frozen complete cycles for {animal}")
        labels, times_hours, weights_seconds, file_spans, invalid_rows = _load_animal_frames(
            animal_dir
        )
        if invalid_rows:
            raise ValueError(f"Invalid rows encountered for {animal}: {invalid_rows}")

        cycle_rows, per_cycle, selected_frame_total = _cycle_bout_rows(
            labels,
            times_hours,
            weights_seconds,
            complete_cycles,
        )
        summary_for_animal, sensitivity_for_animal = _build_summary_rows(
            group,
            animal,
            authoritative,
            per_cycle,
        )
        summary_rows.extend(summary_for_animal)
        sensitivity_rows.extend(sensitivity_for_animal)
        cycle_output_rows.extend(_build_cycle_output_rows(group, animal, cycle_rows))

        # Count non-rest same-label continuations that cross a consecutive
        # source-file boundary inside a frozen cycle.  These must remain one
        # bout; the counting function receives the concatenated sequence.
        for _file_index, _, file_end in file_spans[:-1]:
            boundary_left = file_end - 1
            boundary_right = file_end
            if labels[boundary_left] != labels[boundary_right]:
                continue
            if int(labels[boundary_left]) not in frozen_phase.NONRESTING_BEHAVIOR_INDICES:
                continue
            for _cycle_index, start_boundary, end_boundary in complete_cycles:
                same_cycle = (
                    start_boundary <= times_hours[boundary_left] < end_boundary
                    and start_boundary <= times_hours[boundary_right] < end_boundary
                )
                if same_cycle:
                    boundary_continuations += 1
                    break

        authoritative_for_animal = authoritative[
            (authoritative["Group"] == group) & (authoritative["Animal"] == animal)
        ]
        for _, row in authoritative_for_animal.iterrows():
            summary_row = next(
                candidate
                for candidate in summary_for_animal
                if candidate["behavior"] == row["Behavior"]
            )
            computed_cycle_minutes = [
                float(
                    weights_seconds[
                        (times_hours >= complete_cycles[cycle - 1][1])
                        & (times_hours < complete_cycles[cycle - 1][2])
                        & (labels == frozen_phase.NONRESTING_BEHAVIOR_INDICES[NONREST_BEHAVIORS.index(row["Behavior"])])
                    ].sum()
                    / 60.0
                )
                for cycle in range(1, N_CYCLES + 1)
            ]
            authoritative_cycle_minutes = [
                float(row[f"cycle{cycle}_occupancy_minutes"])
                for cycle in range(1, N_CYCLES + 1)
            ]
            if not np.allclose(computed_cycle_minutes, authoritative_cycle_minutes, atol=1e-7, rtol=0.0):
                raise AssertionError(
                    f"WTA minute mismatch with authoritative estimability output for {animal} {row['Behavior']}"
                )
            validation.append(
                {
                    "animal": animal,
                    "selected_cycle_frames": selected_frame_total,
                    "raw_bouts": int(summary_row["raw_bouts_total"]),
                    "bouts_1s": int(summary_row["bouts_1s_total"]),
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    cycle_frame = pd.DataFrame(cycle_output_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    if len(summary_frame) != 64 or summary_frame[["animal", "behavior"]].drop_duplicates().shape[0] != 64:
        raise AssertionError("Summary output does not contain exactly 64 animal-behavior rows")
    if len(cycle_frame) != 256 or cycle_frame[["animal", "behavior", "cycle"]].drop_duplicates().shape[0] != 256:
        raise AssertionError("Cycle output does not contain exactly 64 x 4 rows")
    if len(sensitivity_frame) != 64:
        raise AssertionError("Sensitivity output does not contain exactly 64 rows")

    for _, row in summary_frame.iterrows():
        if not (
            row["raw_bouts_total"] >= row["bouts_0p5s_total"]
            >= row["bouts_1s_total"]
            >= row["bouts_2s_total"]
        ):
            raise AssertionError(f"Merged bout counts are not monotone for {row['animal']} {row['behavior']}")
        if row["total_wta_minutes"] > 0.0 and row["raw_bouts_total"] <= 0:
            raise AssertionError(f"Positive WTA support has zero raw bouts for {row['animal']} {row['behavior']}")

    summary_path = output_dir / "behavior_bout_support_summary.csv"
    cycle_path = output_dir / "behavior_bout_support_by_cycle.csv"
    sensitivity_path = output_dir / "bout_merge_sensitivity_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    cycle_frame.to_csv(cycle_path, index=False)
    sensitivity_frame.to_csv(sensitivity_path, index=False)

    _save_scatter(
        summary_frame,
        "total_wta_minutes",
        "bouts_1s_total",
        "total WTA minutes",
        "total 1 s merged WTA bouts",
        "WTA minutes versus 1 s merged bout count",
        output_dir / "wta_minutes_vs_bouts_1s.png",
    )
    _save_scatter(
        summary_frame,
        "bouts_1s_total",
        "max_loco_w1_hours",
        "total 1 s merged WTA bouts",
        "maximum LOCO circular W1 (hours)",
        "1 s merged bout count versus maximum LOCO W1",
        output_dir / "bouts_1s_vs_max_loco_w1.png",
    )
    _save_scatter(
        summary_frame,
        "total_wta_minutes",
        "max_loco_w1_hours",
        "total WTA minutes",
        "maximum LOCO circular W1 (hours)",
        "WTA minutes versus maximum LOCO W1",
        output_dir / "wta_minutes_vs_max_loco_w1.png",
    )

    print(f"Output directory: {output_dir}")
    print(f"Animal-behavior rows: {len(summary_frame)}")
    print(f"Animal-behavior-cycle rows: {len(cycle_frame)}")
    print(f"Total selected complete-cycle frames: {sum(item['selected_cycle_frames'] for item in validation[::len(NONREST_BEHAVIORS)])}")
    print(f"Non-rest file-boundary bout continuations preserved: {boundary_continuations}")
    print(
        "Spearman correlations (descriptive only): "
        f"minutes vs max W1={_spearman(summary_frame['total_wta_minutes'], summary_frame['max_loco_w1_hours']):.6f}; "
        f"1 s bouts vs max W1={_spearman(summary_frame['bouts_1s_total'], summary_frame['max_loco_w1_hours']):.6f}"
    )
    print("All bout-count monotonicity and coverage checks passed.")
    return {
        "summary": summary_frame,
        "cycle": cycle_frame,
        "sensitivity": sensitivity_frame,
        "validation": validation,
        "summary_path": summary_path,
        "cycle_path": cycle_path,
        "sensitivity_path": sensitivity_path,
        "output_dir": output_dir,
    }


def main() -> None:
    cohort_root, requested_output_dir = _parse_args()
    output_dir = requested_output_dir or cohort_root / OUTPUT_DIRNAME
    run_audit(cohort_root, output_dir)


if __name__ == "__main__":
    main()
