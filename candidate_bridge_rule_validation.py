"""Validate a small set of confidence-aware A->B->A bridge rules.

This script is deliberately limited to the synthetic validation requested for
the 783E demo mouse.  It reads the raw probability columns, reconstructs the
raw winner-take-all (WTA) bouts, and evaluates exactly three duration limits
and four B-vs-A margin thresholds.  Candidate corrections are made only on
copies of the raw WTA labels in memory; no source CSV or corrected full-frame
dataset is written.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as mi


DEFAULT_INPUT_DIR = Path(r"C:\Users\Jeff\Documents\CBAS_Analysis_Data")
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Bridge_Rule_Validation"
)

EXPECTED_FILE_COUNT = 617
EXPECTED_ROWS_PER_FILE = 6000
SAMPLING_RATE_HZ = 10.0
FRAMES_PER_CYCLE = int(24 * 60 * 60 * SAMPLING_RATE_HZ)
EXPECTED_COMPLETE_CYCLES = 4
EXPECTED_RAW_NONREST_BOUTS = 47756
EXPECTED_RAW_NONREST_BY_CYCLE = (13668, 11326, 10863, 11899)
LOCAL_WINDOW_FRAMES = 20

RESTING_BEHAVIOR = "resting"
NONREST_BEHAVIORS = tuple(
    behavior for behavior in mi.BEHAVIORS if behavior != RESTING_BEHAVIOR
)
BEHAVIOR_INDEX = {behavior: index for index, behavior in enumerate(mi.BEHAVIORS)}

DURATION_RULES = (
    ("duration == 1", 1),
    ("duration <= 2", 2),
    ("duration <= 4", 4),
)
MARGIN_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)

DURATION_GROUPS = (
    ("1_frame", 1, 1),
    ("2_frames", 2, 2),
    ("3_4_frames", 3, 4),
    ("5_plus_frames", 5, math.inf),
)


@dataclass
class Bout:
    bout_id: int
    behavior: str
    start_frame_global: int
    end_frame_global: int
    start_file_index: int
    end_file_index: int
    start_row_index: int
    end_row_index: int
    duration_frames: int
    mean_p_B: float
    median_p_B: float
    min_p_B: float
    max_p_B: float
    mean_top2_margin: float
    median_top2_margin: float
    min_top2_margin: float
    cycle: int = -1
    previous_behavior: str | None = None
    next_behavior: str | None = None
    starts_at_recording_boundary: bool = False
    ends_at_recording_boundary: bool = False
    same_flank_ABA: bool = False
    flank_behavior: str | None = None
    mean_margin_B_vs_A: float = float("nan")
    median_margin_B_vs_A: float = float("nan")
    min_margin_B_vs_A: float = float("nan")
    max_margin_B_vs_A: float = float("nan")


@dataclass
class CorrectedBout:
    behavior: str
    start_frame_global: int
    end_frame_global: int
    duration_frames: int
    cycle: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the specified raw-WTA A->B->A bridge candidates."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing one animal's sequential CBAS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived validation outputs.",
    )
    return parser.parse_args()


def classify_numeric(numeric: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Apply the same finite-aware argmax rule as the existing raw-WTA audit."""

    finite = np.isfinite(numeric)
    valid = finite.any(axis=1)
    labels = np.full(numeric.shape[0], -1, dtype=np.int8)
    if valid.any():
        safe = np.where(finite[valid], numeric[valid], -np.inf)
        labels[valid] = np.argmax(safe, axis=1).astype(np.int8)
    invalid_rows = int(np.sum(~valid))
    partial_rows = int(np.sum(valid & ~finite.all(axis=1)))
    return labels, valid, invalid_rows, partial_rows


def load_file_numeric(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame = pd.read_csv(path, usecols=mi.BEHAVIORS)
    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels, valid, invalid_rows, partial_rows = classify_numeric(numeric)
    return numeric, labels, invalid_rows, partial_rows


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def finalize_current(current: dict[str, object], bouts: list[Bout]) -> None:
    """Convert one streaming raw run into a compact bout record."""

    winner_values = np.asarray(current["winner_values"], dtype=float)
    top2_values = np.asarray(current["top2_values"], dtype=float)
    start_frame = int(current["start_frame_global"])
    end_frame = int(current["end_frame_global"])
    bouts.append(
        Bout(
            bout_id=len(bouts),
            behavior=str(current["behavior"]),
            start_frame_global=start_frame,
            end_frame_global=end_frame,
            start_file_index=int(current["start_file_index"]),
            end_file_index=int(current["end_file_index"]),
            start_row_index=int(current["start_row_index"]),
            end_row_index=int(current["end_row_index"]),
            duration_frames=end_frame - start_frame + 1,
            mean_p_B=float(np.mean(winner_values)),
            median_p_B=float(np.median(winner_values)),
            min_p_B=float(np.min(winner_values)),
            max_p_B=float(np.max(winner_values)),
            mean_top2_margin=float(np.mean(top2_values)),
            median_top2_margin=float(np.median(top2_values)),
            min_top2_margin=float(np.min(top2_values)),
        )
    )


def build_raw_sequence(input_dir: Path) -> tuple[list[Bout], dict[str, object]]:
    """Read all source files and reconstruct continuous raw-WTA bouts."""

    input_files, missing_indices = mi.discover_input_files(input_dir)
    indices = [index for index, _ in input_files]
    if len(input_files) != EXPECTED_FILE_COUNT or missing_indices:
        raise ValueError(
            "Expected 617 consecutive source files; "
            f"found {len(input_files)} with missing indices {missing_indices}."
        )
    if indices != list(range(EXPECTED_FILE_COUNT)):
        raise ValueError(
            f"Unexpected numeric file order: {indices[:5]} ... {indices[-5:]}"
        )

    first_header: list[str] | None = None
    header_consistent = True
    file_records: list[dict[str, object]] = []
    label_chunks: list[np.ndarray] = []
    row_counts: list[int] = []
    total_rows = 0
    invalid_rows = 0
    partial_probability_rows = 0
    wta_mismatch_rows = 0
    boundary_count = 0
    boundary_same_state_count = 0
    previous_last_label: int | None = None

    bouts: list[Bout] = []
    current: dict[str, object] | None = None

    for file_index, path in input_files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if first_header is None:
            first_header = header
        elif header != first_header:
            header_consistent = False
        if any(behavior not in header for behavior in mi.BEHAVIORS):
            raise ValueError(f"Missing probability column in {path.name}")

        numeric, labels, invalid, partial = load_file_numeric(path)
        n_rows = numeric.shape[0]
        file_start_global = total_rows
        file_end_global = file_start_global + n_rows - 1
        total_rows += n_rows
        row_counts.append(n_rows)
        label_chunks.append(labels)
        invalid_rows += invalid
        partial_probability_rows += partial

        valid_positions = np.flatnonzero(labels >= 0)
        if valid_positions.size:
            safe = np.where(
                np.isfinite(numeric[valid_positions]),
                numeric[valid_positions],
                -np.inf,
            )
            expected_labels = np.argmax(safe, axis=1)
            wta_mismatch_rows += int(
                np.sum(expected_labels != labels[valid_positions])
            )

        if file_index > 0:
            boundary_count += 1
            if previous_last_label is not None and labels.size and labels[0] >= 0:
                if int(labels[0]) == previous_last_label:
                    boundary_same_state_count += 1

        file_records.append(
            {
                "file_index": file_index,
                "path": path,
                "n_rows": n_rows,
                "global_start": file_start_global,
                "global_end": file_end_global,
            }
        )

        if valid_positions.size:
            gap_breaks = np.flatnonzero(np.diff(valid_positions) > 1) + 1
            segment_starts = np.concatenate(([0], gap_breaks))
            segment_ends = np.concatenate((gap_breaks, [len(valid_positions)]))
            for segment_start, segment_end in zip(segment_starts, segment_ends):
                segment_positions = valid_positions[segment_start:segment_end]
                segment_labels = labels[segment_positions]
                segment_numeric = numeric[segment_positions]
                safe_segment = np.where(
                    np.isfinite(segment_numeric), segment_numeric, -np.inf
                )
                top1 = np.max(safe_segment, axis=1)
                second = np.partition(safe_segment, -2, axis=1)[:, -2]
                top2_margin = top1 - second
                winner_values = segment_numeric[
                    np.arange(len(segment_labels)), segment_labels
                ]

                changes = np.flatnonzero(segment_labels[1:] != segment_labels[:-1]) + 1
                run_starts = np.concatenate(([0], changes))
                run_ends = np.concatenate((changes, [len(segment_labels)]))
                for run_start, run_end in zip(run_starts, run_ends):
                    run_label = int(segment_labels[run_start])
                    first_position = int(segment_positions[run_start])
                    last_position = int(segment_positions[run_end - 1])
                    run_global_start = file_start_global + first_position
                    run_global_end = file_start_global + last_position
                    can_extend = (
                        current is not None
                        and int(current["state_index"]) == run_label
                        and int(current["end_frame_global"]) + 1
                        == run_global_start
                    )
                    if not can_extend:
                        if current is not None:
                            finalize_current(current, bouts)
                        current = {
                            "state_index": run_label,
                            "behavior": mi.BEHAVIORS[run_label],
                            "start_frame_global": run_global_start,
                            "end_frame_global": run_global_end,
                            "start_file_index": file_index,
                            "end_file_index": file_index,
                            "start_row_index": first_position,
                            "end_row_index": last_position,
                            "winner_values": winner_values[run_start:run_end].tolist(),
                            "top2_values": top2_margin[run_start:run_end].tolist(),
                        }
                    else:
                        current["end_frame_global"] = run_global_end
                        current["end_file_index"] = file_index
                        current["end_row_index"] = last_position
                        current["winner_values"].extend(
                            winner_values[run_start:run_end].tolist()
                        )
                        current["top2_values"].extend(
                            top2_margin[run_start:run_end].tolist()
                        )

        previous_last_label = (
            int(labels[-1]) if labels.size and labels[-1] >= 0 else None
        )

    if current is not None:
        finalize_current(current, bouts)

    if not row_counts or any(count != EXPECTED_ROWS_PER_FILE for count in row_counts):
        raise ValueError(
            "Expected exactly 6,000 rows per file; "
            f"observed min={min(row_counts)}, max={max(row_counts)}."
        )
    if not header_consistent:
        raise ValueError("Source CSV headers are not consistent across files")

    all_labels = np.concatenate(label_chunks).astype(np.int8, copy=False)
    for index, bout in enumerate(bouts):
        bout.bout_id = index
        bout.previous_behavior = bouts[index - 1].behavior if index > 0 else None
        bout.next_behavior = (
            bouts[index + 1].behavior if index + 1 < len(bouts) else None
        )
        bout.starts_at_recording_boundary = bout.start_frame_global == 0
        bout.ends_at_recording_boundary = bout.end_frame_global == total_rows - 1
        bout.same_flank_ABA = bool(
            bout.previous_behavior is not None
            and bout.previous_behavior == bout.next_behavior
            and bout.previous_behavior != bout.behavior
        )
        if bout.same_flank_ABA:
            bout.flank_behavior = bout.previous_behavior
        bout.cycle = bout.start_frame_global // FRAMES_PER_CYCLE

    duration_sum_ok = (
        sum(bout.duration_frames for bout in bouts) == total_rows - invalid_rows
    )
    metadata: dict[str, object] = {
        "input_dir": input_dir,
        "input_files": input_files,
        "file_records": file_records,
        "missing_indices": missing_indices,
        "file_count": len(input_files),
        "row_counts": row_counts,
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "partial_probability_rows": partial_probability_rows,
        "wta_mismatch_rows": wta_mismatch_rows,
        "header_consistent": header_consistent,
        "boundary_same_state_count": boundary_same_state_count,
        "boundary_count": boundary_count,
        "duration_sum_ok": duration_sum_ok,
        "bouts": bouts,
        "labels": all_labels,
        "frames_per_cycle": FRAMES_PER_CYCLE,
        "complete_cycles": min(
            EXPECTED_COMPLETE_CYCLES, total_rows // FRAMES_PER_CYCLE
        ),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
    }
    return bouts, metadata


def update_aba_margins(bouts: list[Bout], metadata: dict[str, object]) -> None:
    """Calculate raw-frame B-vs-A margins for non-rest same-flank bouts."""

    selected = [
        bout
        for bout in bouts
        if bout.same_flank_ABA and bout.behavior in NONREST_BEHAVIORS
    ]
    if not selected:
        return

    values_by_bout: dict[int, list[float]] = {
        bout.bout_id: [] for bout in selected
    }
    records = metadata["file_records"]
    cursor = 0

    for record in records:
        file_start = int(record["global_start"])
        file_end = int(record["global_end"])
        while cursor < len(selected) and selected[cursor].end_frame_global < file_start:
            cursor += 1
        overlapping: list[Bout] = []
        index = cursor
        while (
            index < len(selected)
            and selected[index].start_frame_global <= file_end
        ):
            overlapping.append(selected[index])
            index += 1
        if not overlapping:
            continue

        numeric, _, _, _ = load_file_numeric(Path(record["path"]))
        for bout in overlapping:
            start = max(bout.start_frame_global, file_start) - file_start
            end = min(bout.end_frame_global, file_end) - file_start
            b_index = BEHAVIOR_INDEX[bout.behavior]
            a_index = BEHAVIOR_INDEX[str(bout.flank_behavior)]
            differences = numeric[start : end + 1, b_index] - numeric[
                start : end + 1, a_index
            ]
            values_by_bout[bout.bout_id].extend(differences.tolist())

    for bout in selected:
        values = np.asarray(values_by_bout[bout.bout_id], dtype=float)
        if len(values) != bout.duration_frames:
            raise AssertionError(
                f"B-vs-A margin frame count mismatch for bout {bout.bout_id}: "
                f"{len(values)} versus {bout.duration_frames}"
            )
        bout.mean_margin_B_vs_A = float(np.mean(values))
        bout.median_margin_B_vs_A = float(np.median(values))
        bout.min_margin_B_vs_A = float(np.min(values))
        bout.max_margin_B_vs_A = float(np.max(values))


def analysis_bouts(bouts: list[Bout], complete_cycles: int) -> list[Bout]:
    """Return the non-rest bouts in the four complete cycles used by the audit."""

    return [
        bout
        for bout in bouts
        if bout.cycle < complete_cycles and bout.behavior in NONREST_BEHAVIORS
    ]


def is_eligible(bout: Bout, duration_limit: int) -> bool:
    return (
        bout.duration_frames <= duration_limit
        and bout.same_flank_ABA
        and bout.behavior in NONREST_BEHAVIORS
        and np.isfinite(bout.mean_margin_B_vs_A)
    )


def select_candidate_bouts(
    raw_analysis_bouts: list[Bout], duration_limit: int, margin_threshold: float
) -> list[Bout]:
    """Select only from original raw boutes; never from a corrected sequence."""

    return [
        bout
        for bout in raw_analysis_bouts
        if is_eligible(bout, duration_limit)
        and bout.mean_margin_B_vs_A < margin_threshold
    ]


def reconstruct_corrected_bouts(labels: np.ndarray) -> list[CorrectedBout]:
    """Run-length encode one corrected in-memory WTA label sequence."""

    if labels.size == 0:
        return []
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [len(labels) - 1]))
    corrected: list[CorrectedBout] = []
    for start, end in zip(starts, ends):
        label = int(labels[start])
        if label < 0:
            continue
        start_int = int(start)
        end_int = int(end)
        corrected.append(
            CorrectedBout(
                behavior=mi.BEHAVIORS[label],
                start_frame_global=start_int,
                end_frame_global=end_int,
                duration_frames=end_int - start_int + 1,
                cycle=start_int // FRAMES_PER_CYCLE,
            )
        )
    return corrected


def apply_candidate(
    raw_labels: np.ndarray,
    selected: list[Bout],
    total_rows: int,
    complete_cycles: int,
) -> dict[str, object]:
    """Apply one candidate once to a copy of raw labels and count corrected runs."""

    corrected_labels = raw_labels.copy()
    frames_relabeled = 0
    for bout in selected:
        start = bout.start_frame_global
        end = bout.end_frame_global
        raw_slice = raw_labels[start : end + 1]
        expected_label = BEHAVIOR_INDEX[bout.behavior]
        flank_label = BEHAVIOR_INDEX[str(bout.flank_behavior)]
        if not np.all(raw_slice == expected_label):
            raise AssertionError(
                f"Raw label mismatch in candidate bout {bout.bout_id}"
            )
        corrected_labels[start : end + 1] = flank_label
        frames_relabeled += bout.duration_frames

    corrected_bouts = reconstruct_corrected_bouts(corrected_labels)
    corrected_complete = [
        bout
        for bout in corrected_bouts
        if bout.cycle < complete_cycles and bout.behavior in NONREST_BEHAVIORS
    ]
    corrected_all = [
        bout for bout in corrected_bouts if bout.behavior in NONREST_BEHAVIORS
    ]
    corrected_by_cycle = [
        sum(
            bout.cycle == cycle and bout.behavior in NONREST_BEHAVIORS
            for bout in corrected_bouts
        )
        for cycle in range(complete_cycles)
    ]
    return {
        "frames_relabeled": frames_relabeled,
        "corrected_bouts": corrected_bouts,
        "corrected_complete_count": len(corrected_complete),
        "corrected_all_count": len(corrected_all),
        "corrected_by_cycle": corrected_by_cycle,
        "total_rows": total_rows,
    }


def make_candidate_results(
    raw_analysis_bouts: list[Bout], metadata: dict[str, object]
) -> tuple[pd.DataFrame, dict[tuple[int, float], dict[str, object]]]:
    raw_labels = metadata["labels"]
    total_rows = int(metadata["total_rows"])
    complete_cycles = int(metadata["complete_cycles"])
    raw_complete_count = len(raw_analysis_bouts)
    raw_all_count = sum(bout.behavior in NONREST_BEHAVIORS for bout in metadata["bouts"])
    raw_by_cycle = [
        sum(bout.cycle == cycle for bout in raw_analysis_bouts)
        for cycle in range(complete_cycles)
    ]

    rows: list[dict[str, object]] = []
    results: dict[tuple[int, float], dict[str, object]] = {}
    for duration_rule, duration_limit in DURATION_RULES:
        for threshold in MARGIN_THRESHOLDS:
            selected = select_candidate_bouts(
                raw_analysis_bouts, duration_limit, threshold
            )
            applied = apply_candidate(
                raw_labels, selected, total_rows, complete_cycles
            )
            corrected_by_cycle = list(applied["corrected_by_cycle"])
            corrected_complete_count = int(applied["corrected_complete_count"])
            corrected_all_count = int(applied["corrected_all_count"])
            frames_relabeled = int(applied["frames_relabeled"])
            ten_to_fifteen = [
                bout
                for bout in raw_analysis_bouts
                if 10 <= bout.duration_frames <= 15
            ]
            ten_to_fifteen_bridged = [
                bout
                for bout in selected
                if 10 <= bout.duration_frames <= 15
            ]
            row: dict[str, object] = {
                "duration_rule": duration_rule,
                "duration_limit_frames": duration_limit,
                "margin_threshold": threshold,
                "raw_nonrest_bout_count_complete_cycles": raw_complete_count,
                "raw_nonrest_bout_count_all_recording": raw_all_count,
                "number_bouts_bridged": len(selected),
                "number_frames_relabeled": frames_relabeled,
                "fraction_all_nonrest_bouts_bridged": (
                    len(selected) / raw_complete_count
                ),
                "fraction_all_recording_frames_relabeled": frames_relabeled / total_rows,
                "corrected_total_nonrest_bout_count": corrected_complete_count,
                "corrected_total_nonrest_bout_count_all_recording": corrected_all_count,
                "corrected_nonrest_bouts_per_complete_cycle": ",".join(
                    str(value) for value in corrected_by_cycle
                ),
                "percent_reduction_nonrest_bout_count": (
                    100.0 * (raw_complete_count - corrected_complete_count)
                    / raw_complete_count
                ),
                "ten_to_fifteen_total_nonrest_bouts": len(ten_to_fifteen),
                "ten_to_fifteen_bridged_bouts": len(ten_to_fifteen_bridged),
            }
            for cycle, count in enumerate(raw_by_cycle):
                row[f"raw_nonrest_bouts_cycle_{cycle}"] = count
            for cycle, count in enumerate(corrected_by_cycle):
                row[f"corrected_nonrest_bouts_cycle_{cycle}"] = count

            key = (duration_limit, threshold)
            results[key] = {
                "duration_rule": duration_rule,
                "duration_limit_frames": duration_limit,
                "margin_threshold": threshold,
                "selected": selected,
                "row": row,
                "applied": applied,
            }
            rows.append(row)

    return pd.DataFrame(rows), results


def make_behavior_output(
    raw_analysis_bouts: list[Bout],
    results: dict[tuple[int, float], dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results.values():
        for behavior in NONREST_BEHAVIORS:
            raw_behavior = [
                bout for bout in raw_analysis_bouts if bout.behavior == behavior
            ]
            bridged = [bout for bout in result["selected"] if bout.behavior == behavior]
            rows.append(
                {
                    "duration_rule": result["duration_rule"],
                    "duration_limit_frames": result["duration_limit_frames"],
                    "margin_threshold": result["margin_threshold"],
                    "behavior_B": behavior,
                    "raw_B_bout_count": len(raw_behavior),
                    "bridged_B_bout_count": len(bridged),
                    "fraction_B_bouts_bridged": len(bridged) / len(raw_behavior)
                    if raw_behavior
                    else float("nan"),
                    "median_duration_bridged_frames": finite_median(
                        bout.duration_frames for bout in bridged
                    ),
                    "median_mean_p_B_bridged": finite_median(
                        bout.mean_p_B for bout in bridged
                    ),
                    "median_mean_margin_B_vs_A_bridged": finite_median(
                        bout.mean_margin_B_vs_A for bout in bridged
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_duration_output(
    raw_analysis_bouts: list[Bout],
    results: dict[tuple[int, float], dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_ids_by_key = {
        key: {bout.bout_id for bout in result["selected"]}
        for key, result in results.items()
    }
    for key, result in results.items():
        selected_ids = selected_ids_by_key[key]
        duration_limit = int(result["duration_limit_frames"])
        for group_name, lower, upper in DURATION_GROUPS:
            group_bouts = [
                bout
                for bout in raw_analysis_bouts
                if lower <= bout.duration_frames <= upper
            ]
            aba_group = [bout for bout in group_bouts if bout.same_flank_ABA]
            eligible = [
                bout
                for bout in aba_group
                if bout.duration_frames <= duration_limit
                and np.isfinite(bout.mean_margin_B_vs_A)
            ]
            bridged = [bout for bout in eligible if bout.bout_id in selected_ids]
            rows.append(
                {
                    "duration_rule": result["duration_rule"],
                    "duration_limit_frames": duration_limit,
                    "margin_threshold": result["margin_threshold"],
                    "duration_group": group_name,
                    "all_original_nonrest_bouts_in_group": len(group_bouts),
                    "same_flank_ABA_bouts_in_group": len(aba_group),
                    "eligible_ABA_bouts_in_group": len(eligible),
                    "bridged_bouts_in_group": len(bridged),
                    "fraction_eligible_group_removed": (
                        len(bridged) / len(eligible) if eligible else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_preserved_output(
    raw_analysis_bouts: list[Bout],
    results: dict[tuple[int, float], dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results.values():
        duration_limit = int(result["duration_limit_frames"])
        threshold = float(result["margin_threshold"])
        eligible = [
            bout
            for bout in raw_analysis_bouts
            if bout.same_flank_ABA
            and bout.behavior in NONREST_BEHAVIORS
            and bout.duration_frames <= duration_limit
            and np.isfinite(bout.mean_margin_B_vs_A)
        ]
        preserved = [
            bout for bout in eligible if bout.mean_margin_B_vs_A >= threshold
        ]
        rows.append(
            {
                "duration_rule": result["duration_rule"],
                "duration_limit_frames": duration_limit,
                "margin_threshold": threshold,
                "eligible_short_ABA_bout_count": len(eligible),
                "preserved_high_confidence_short_bout_count": len(preserved),
                "fraction_eligible_preserved": len(preserved) / len(eligible)
                if eligible
                else float("nan"),
                "median_mean_p_B_preserved": finite_median(
                    bout.mean_p_B for bout in preserved
                ),
                "median_mean_margin_B_vs_A_preserved": finite_median(
                    bout.mean_margin_B_vs_A for bout in preserved
                ),
            }
        )
    return pd.DataFrame(rows)


def select_diverse(records: list[Bout], count: int) -> list[Bout]:
    """Select deterministically in behavior round-robin order."""

    by_behavior: dict[str, list[Bout]] = {behavior: [] for behavior in NONREST_BEHAVIORS}
    for bout in sorted(records, key=lambda item: (item.behavior, item.start_frame_global)):
        by_behavior[bout.behavior].append(bout)

    selected: list[Bout] = []
    while len(selected) < count:
        added = False
        for behavior in NONREST_BEHAVIORS:
            if by_behavior[behavior]:
                selected.append(by_behavior[behavior].pop(0))
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    return selected


def make_manual_sample(
    raw_analysis_bouts: list[Bout],
) -> tuple[pd.DataFrame, dict[str, object]]:
    preferred_limit = 2
    threshold = 0.10

    def groups(limit: int) -> tuple[list[Bout], list[Bout], list[Bout]]:
        candidates = select_candidate_bouts(raw_analysis_bouts, limit, threshold)
        selected_ids = {bout.bout_id for bout in candidates}
        one_frame = [bout for bout in candidates if bout.duration_frames == 1]
        two_to_four = [
            bout for bout in candidates if 2 <= bout.duration_frames <= 4
        ]
        preserved = [
            bout
            for bout in raw_analysis_bouts
            if bout.same_flank_ABA
            and bout.behavior in NONREST_BEHAVIORS
            and bout.duration_frames <= limit
            and np.isfinite(bout.mean_margin_B_vs_A)
            and bout.mean_margin_B_vs_A >= threshold
            and bout.bout_id not in selected_ids
        ]
        return one_frame, two_to_four, preserved

    one_frame, two_to_four, preserved = groups(preferred_limit)
    used_limit = preferred_limit
    if min(len(one_frame), len(two_to_four), len(preserved)) < 10:
        one_frame, two_to_four, preserved = groups(4)
        used_limit = 4

    selections = (
        ("bridged_one_frame", one_frame),
        ("bridged_two_to_four_frames", two_to_four),
        ("preserved_high_confidence_short", preserved),
    )
    rows: list[dict[str, object]] = []
    sample_number = 0
    for selection_group, records in selections:
        for bout in select_diverse(records, 10):
            sample_number += 1
            rows.append(
                {
                    "sample_id": sample_number,
                    "selection_group": selection_group,
                    "candidate_duration_rule": (
                        "duration <= 2" if used_limit == 2 else "duration <= 4"
                    ),
                    "candidate_duration_limit_frames": used_limit,
                    "candidate_margin_threshold": threshold,
                    "bout_id": bout.bout_id,
                    "behavior_B": bout.behavior,
                    "flanking_behavior_A": bout.flank_behavior,
                    "duration_frames": bout.duration_frames,
                    "duration_seconds": bout.duration_frames / SAMPLING_RATE_HZ,
                    "mean_p_B": bout.mean_p_B,
                    "mean_margin_B_vs_A": bout.mean_margin_B_vs_A,
                    "bridged_under_candidate": selection_group != "preserved_high_confidence_short",
                    "previous_behavior": bout.previous_behavior,
                    "next_behavior": bout.next_behavior,
                    "same_flank_ABA": bout.same_flank_ABA,
                    "starts_at_recording_boundary": bout.starts_at_recording_boundary,
                    "ends_at_recording_boundary": bout.ends_at_recording_boundary,
                    "start_file_index": bout.start_file_index,
                    "end_file_index": bout.end_file_index,
                    "start_row_index": bout.start_row_index,
                    "end_row_index": bout.end_row_index,
                    "start_frame_global": bout.start_frame_global,
                    "end_frame_global": bout.end_frame_global,
                }
            )
    return pd.DataFrame(rows), {
        "duration_limit": used_limit,
        "margin_threshold": threshold,
        "available_one_frame": len(one_frame),
        "available_two_to_four": len(two_to_four),
        "available_preserved": len(preserved),
    }


def write_probability_traces(
    sample: pd.DataFrame,
    raw_labels: np.ndarray,
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    records = metadata["file_records"]
    file_cache: dict[int, np.ndarray] = {}
    trace_rows: list[dict[str, object]] = []

    for sample_row in sample.to_dict(orient="records"):
        sample_id = int(sample_row["sample_id"])
        bout_start = int(sample_row["start_frame_global"])
        bout_end = int(sample_row["end_frame_global"])
        window_start = max(0, bout_start - LOCAL_WINDOW_FRAMES)
        window_end = min(int(metadata["total_rows"]) - 1, bout_end + LOCAL_WINDOW_FRAMES)
        for record in records:
            file_start = int(record["global_start"])
            file_end = int(record["global_end"])
            overlap_start = max(window_start, file_start)
            overlap_end = min(window_end, file_end)
            if overlap_start > overlap_end:
                continue
            file_index = int(record["file_index"])
            if file_index not in file_cache:
                file_cache[file_index], _, _, _ = load_file_numeric(Path(record["path"]))
            numeric = file_cache[file_index]
            for global_frame in range(overlap_start, overlap_end + 1):
                local_row = global_frame - file_start
                label = int(raw_labels[global_frame])
                row: dict[str, object] = {
                    "sample_id": sample_id,
                    "selection_group": sample_row["selection_group"],
                    "bout_id": sample_row["bout_id"],
                    "global_frame": global_frame,
                    "source_file_index": file_index,
                    "source_row_index": local_row,
                    "relative_frame": global_frame - bout_start,
                    "relative_time_seconds": (
                        (global_frame - bout_start) / SAMPLING_RATE_HZ
                    ),
                    "within_sampled_bout": bout_start <= global_frame <= bout_end,
                    "WTA_state": mi.BEHAVIORS[label] if label >= 0 else None,
                }
                for behavior_index, behavior in enumerate(mi.BEHAVIORS):
                    row[f"p_{behavior}"] = float(numeric[local_row, behavior_index])
                trace_rows.append(row)

    pd.DataFrame(trace_rows).to_csv(
        output_dir / "candidate_manual_review_probability_traces.csv", index=False
    )


def fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6g}"
    return str(value)


def write_run_summary(
    output_dir: Path,
    metadata: dict[str, object],
    bouts: list[Bout],
    raw_analysis_bouts: list[Bout],
    summary: pd.DataFrame,
    behavior: pd.DataFrame,
    manual_info: dict[str, object],
    elapsed_seconds: float,
) -> None:
    complete_cycles = int(metadata["complete_cycles"])
    raw_by_cycle = [
        sum(bout.cycle == cycle for bout in raw_analysis_bouts)
        for cycle in range(complete_cycles)
    ]
    lines = [
        "Candidate A->B->A bridge-rule validation: 783E demo mouse",
        "===========================================================",
        "",
        f"input_directory: {metadata['input_dir']}",
        f"source_file_count: {metadata['file_count']}",
        f"source_rows: {metadata['total_rows']}",
        f"rows_per_file: {min(metadata['row_counts'])}-{max(metadata['row_counts'])}",
        f"sampling_rate_hz: {metadata['sampling_rate_hz']}",
        f"recording_duration_hours: {metadata['total_rows'] / SAMPLING_RATE_HZ / 3600.0:.9f}",
        f"complete_cycles_used_for_bout_impact: {complete_cycles}",
        f"raw_total_bouts_all_recording: {len(bouts)}",
        f"raw_nonrest_bouts_all_recording: {sum(bout.behavior in NONREST_BEHAVIORS for bout in bouts)}",
        f"raw_nonrest_bouts_complete_cycles: {len(raw_analysis_bouts)}",
        f"raw_nonrest_bouts_per_complete_cycle: {','.join(str(value) for value in raw_by_cycle)}",
        "candidate_scope: non-rest raw boutes whose onset is in the four complete cycles, matching the prior audit",
        "correction_scope: full raw WTA label array retained in memory; selected complete-cycle boutes relabeled once per candidate",
        "",
        "Validation",
        "----------",
        f"file_count_617: {'PASS' if metadata['file_count'] == EXPECTED_FILE_COUNT else 'FAIL'}",
        f"row_count_6000_each: {'PASS' if all(count == EXPECTED_ROWS_PER_FILE for count in metadata['row_counts']) else 'FAIL'}",
        f"WTA_argmax_mismatches: {metadata['wta_mismatch_rows']}",
        f"invalid_probability_rows: {metadata['invalid_rows']}",
        f"partial_probability_rows: {metadata['partial_probability_rows']}",
        f"header_consistent: {'PASS' if metadata['header_consistent'] else 'FAIL'}",
        f"same_WTA_state_at_file_boundaries: {metadata['boundary_same_state_count']}/{metadata['boundary_count']}",
        f"bout_duration_sum_matches_valid_rows: {'PASS' if metadata['duration_sum_ok'] else 'FAIL'}",
        f"raw_nonrest_bout_count_matches_prior_audit: {'PASS' if len(raw_analysis_bouts) == EXPECTED_RAW_NONREST_BOUTS else 'FAIL'}",
        f"raw_nonrest_cycle_counts_match_prior_audit: {'PASS' if tuple(raw_by_cycle) == EXPECTED_RAW_NONREST_BY_CYCLE else 'FAIL'}",
        "candidate_detection_source: original raw boutes only",
        "candidate_application: each threshold/limit uses a fresh raw-label copy; no recursive or cascading pass",
        "source_CSVs: read only; no raw source file was written",
        "",
        "Candidate overview",
        "------------------",
    ]
    overview_columns = (
        "duration_rule",
        "margin_threshold",
        "number_bouts_bridged",
        "fraction_all_nonrest_bouts_bridged",
        "corrected_total_nonrest_bout_count",
        "percent_reduction_nonrest_bout_count",
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            " | ".join(f"{column}={fmt(row[column])}" for column in overview_columns)
        )

    lines.extend(
        [
            "",
            "Manual-review sample",
            "--------------------",
            f"representative_duration_rule: duration <= {manual_info['duration_limit']}",
            f"representative_margin_threshold: {manual_info['margin_threshold']}",
            f"available_bridged_one_frame: {manual_info['available_one_frame']}",
            f"available_bridged_two_to_four_frames: {manual_info['available_two_to_four']}",
            f"available_preserved_high_confidence_short: {manual_info['available_preserved']}",
            "",
            "Descriptive interpretation",
            "--------------------------",
            "No candidate rule was selected using COMBA, recurrence, genotype, or any biological outcome.",
            "The compact outputs should be read as classifier-behavior sensitivity checks: stricter margins and shorter duration limits are more conservative, while duration <=4 with threshold 0.20 is the most aggressive member of the requested set.",
            "The explicit 10-15-frame rows are included in bridge_rule_summary.csv; their bridged count is expected to be zero because every candidate duration limit is at most four frames.",
            "",
            f"runtime_seconds: {elapsed_seconds:.3f}",
            "outputs contain summaries and local raw probability traces only; no corrected full-frame dataset was created",
        ]
    )

    # Add a concise behavior-effect note for the most aggressive requested rule.
    aggressive = behavior[
        (behavior["duration_limit_frames"] == 4)
        & (behavior["margin_threshold"] == 0.20)
    ].sort_values("fraction_B_bouts_bridged", ascending=False)
    if not aggressive.empty:
        top = aggressive.iloc[0]
        lines.insert(
            lines.index("", lines.index("Descriptive interpretation") + 1),
            "Most affected behavior under duration <=4, margin <0.20: "
            f"{top['behavior_B']} ({fmt(top['fraction_B_bouts_bridged'])} of its raw complete-cycle bouts bridged).",
        )

    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bouts, metadata = build_raw_sequence(input_dir)
    complete_cycles = int(metadata["complete_cycles"])
    if complete_cycles != EXPECTED_COMPLETE_CYCLES:
        raise ValueError(
            f"Expected four complete cycles at 10 Hz; found {complete_cycles}."
        )
    update_aba_margins(bouts, metadata)
    raw_analysis = analysis_bouts(bouts, complete_cycles)
    if len(raw_analysis) != EXPECTED_RAW_NONREST_BOUTS:
        raise ValueError(
            "Raw non-rest bout count does not match the prior audit: "
            f"{len(raw_analysis)} versus {EXPECTED_RAW_NONREST_BOUTS}."
        )

    summary, results = make_candidate_results(raw_analysis, metadata)
    behavior = make_behavior_output(raw_analysis, results)
    duration = make_duration_output(raw_analysis, results)
    preserved = make_preserved_output(raw_analysis, results)
    manual_sample, manual_info = make_manual_sample(raw_analysis)

    summary.to_csv(output_dir / "bridge_rule_summary.csv", index=False)
    behavior.to_csv(output_dir / "bridge_rule_by_behavior.csv", index=False)
    duration.to_csv(output_dir / "bridge_rule_by_duration.csv", index=False)
    preserved.to_csv(
        output_dir / "preserved_high_confidence_short_bouts.csv", index=False
    )
    manual_sample.to_csv(output_dir / "candidate_manual_review_sample.csv", index=False)
    write_probability_traces(
        manual_sample, metadata["labels"], metadata, output_dir
    )
    write_run_summary(
        output_dir,
        metadata,
        bouts,
        raw_analysis,
        summary,
        behavior,
        manual_info,
        time.perf_counter() - started,
    )

    print(
        f"Validated {len(summary)} candidate rules; "
        f"raw complete-cycle non-rest bouts={len(raw_analysis)}; "
        f"outputs={output_dir}"
    )


if __name__ == "__main__":
    main()
