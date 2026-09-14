"""Descriptive confidence audit of raw WTA bouts for the 783E demo mouse.

This script does not smooth, bridge, filter, or relabel the source sequence.  It
only describes the raw WTA bouts and the probability support behind them.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as mi


DEFAULT_INPUT_DIR = Path(r"C:\Users\Jeff\Documents\CBAS_Analysis_Data")
OUTPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Short_Bout_Confidence_Audit"
)
EXPECTED_FILE_COUNT = 617
EXPECTED_ROWS_PER_FILE = 6000
SAMPLING_RATE_HZ = 10.0
FILE_DURATION_SECONDS = 600.0
LOCAL_WINDOW_FRAMES = 20

NONREST_BEHAVIORS = tuple(
    behavior for behavior in mi.BEHAVIORS if behavior != "resting"
)

DURATION_CATEGORIES = (
    ("1_frame", 1, 1),
    ("2_4_frames", 2, 4),
    ("5_9_frames", 5, 9),
    ("10_15_frames", 10, 15),
    ("16_plus_frames", 16, math.inf),
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
    mean_p_winner: float
    median_p_winner: float
    min_p_winner: float
    max_p_winner: float
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
    _winner_values: list[float] = field(default_factory=list, repr=False)
    _top2_values: list[float] = field(default_factory=list, repr=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit confidence of raw WTA bouts for one CBAS mouse."
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
        default=OUTPUT_DIR,
        help="Directory for audit outputs.",
    )
    return parser.parse_args()


def duration_category(duration_frames: int) -> str:
    for name, lower, upper in DURATION_CATEGORIES:
        if lower <= duration_frames <= upper:
            return name
    raise ValueError(f"Unexpected duration in frames: {duration_frames}")


def finite_summary(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return (float("nan"),) * 4
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.25)),
        float(np.quantile(finite, 0.75)),
        float(np.min(finite)),
    )


def finalize_current(current: dict[str, object], bouts: list[Bout]) -> None:
    if not current:
        return
    winner_values = np.asarray(current["winner_values"], dtype=float)
    top2_values = np.asarray(current["top2_values"], dtype=float)
    bout_id = len(bouts)
    bouts.append(
        Bout(
            bout_id=bout_id,
            behavior=str(current["behavior"]),
            start_frame_global=int(current["start_frame_global"]),
            end_frame_global=int(current["end_frame_global"]),
            start_file_index=int(current["start_file_index"]),
            end_file_index=int(current["end_file_index"]),
            start_row_index=int(current["start_row_index"]),
            end_row_index=int(current["end_row_index"]),
            duration_frames=int(current["end_frame_global"])
            - int(current["start_frame_global"])
            + 1,
            mean_p_winner=float(np.nanmean(winner_values)),
            median_p_winner=float(np.nanmedian(winner_values)),
            min_p_winner=float(np.nanmin(winner_values)),
            max_p_winner=float(np.nanmax(winner_values)),
            mean_top2_margin=float(np.nanmean(top2_values)),
            median_top2_margin=float(np.nanmedian(top2_values)),
            min_top2_margin=float(np.nanmin(top2_values)),
            _winner_values=[],
            _top2_values=[],
        )
    )


def classify_numeric_frame(
    numeric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    finite = np.isfinite(numeric)
    valid = finite.any(axis=1)
    invalid_rows = int(np.sum(~valid))
    partial_rows = int(np.sum(valid & ~finite.all(axis=1)))
    labels = np.full(numeric.shape[0], -1, dtype=np.int8)
    if valid.any():
        values_for_argmax = np.where(finite[valid], numeric[valid], -np.inf)
        labels[valid] = np.argmax(values_for_argmax, axis=1).astype(np.int8)
    return labels, valid, invalid_rows, partial_rows


def load_file_numeric(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame = pd.read_csv(path, usecols=mi.BEHAVIORS)
    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels, valid, invalid_rows, partial_rows = classify_numeric_frame(numeric)
    return numeric, labels, invalid_rows, partial_rows


def build_bouts(
    input_dir: Path,
) -> tuple[list[Bout], dict[str, object]]:
    input_files, missing_indices = mi.discover_input_files(input_dir)
    indices = [index for index, _ in input_files]
    if len(input_files) != EXPECTED_FILE_COUNT or missing_indices:
        raise ValueError(
            "Expected 617 consecutive source files; "
            f"found {len(input_files)} with missing indices {missing_indices}."
        )
    if indices != list(range(EXPECTED_FILE_COUNT)):
        raise ValueError(f"Unexpected numeric file order: {indices[:5]} ... {indices[-5:]}")

    first_header: list[str] | None = None
    header_consistent = True
    file_records: list[dict[str, object]] = []
    total_rows = 0
    invalid_rows = 0
    partial_probability_rows = 0
    wta_mismatch_rows = 0
    boundary_same_state_count = 0
    boundary_count = 0
    row_counts: list[int] = []

    bouts: list[Bout] = []
    current: dict[str, object] | None = None
    previous_file_index: int | None = None
    previous_file_last_label: int | None = None

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
        row_counts.append(n_rows)
        invalid_rows += invalid
        partial_probability_rows += partial
        total_rows += n_rows
        valid_positions = np.flatnonzero(labels >= 0)
        valid_labels = labels[valid_positions]
        if valid_positions.size:
            values_for_argmax = np.where(
                np.isfinite(numeric[valid_positions]),
                numeric[valid_positions],
                -np.inf,
            )
            expected_labels = np.argmax(values_for_argmax, axis=1)
            wta_mismatch_rows += int(np.sum(expected_labels != valid_labels))

        if previous_file_index is not None:
            boundary_count += 1
            if valid_labels.size and previous_file_last_label is not None:
                if int(valid_labels[0]) == previous_file_last_label:
                    boundary_same_state_count += 1

        file_start_global = total_rows - n_rows
        file_records.append(
            {
                "file_index": file_index,
                "path": path,
                "n_rows": n_rows,
                "global_start": file_start_global,
                "global_end": file_start_global + n_rows - 1,
            }
        )

        if valid_positions.size:
            gap_breaks = np.flatnonzero(np.diff(valid_positions) > 1) + 1
            segment_starts = np.concatenate(([0], gap_breaks))
            segment_ends = np.concatenate((gap_breaks, [len(valid_positions)]))
            for segment_start, segment_end in zip(segment_starts, segment_ends):
                segment_positions = valid_positions[segment_start:segment_end]
                segment_labels = valid_labels[segment_start:segment_end]
                segment_numeric = numeric[segment_positions]
                finite_segment = np.isfinite(segment_numeric)
                safe_segment = np.where(finite_segment, segment_numeric, -np.inf)
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
                        and int(current["end_frame_global"]) + 1 == run_global_start
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

                previous_file_index = file_index
                previous_file_last_label = int(segment_labels[-1])

        if valid_positions.size == 0:
            if current is not None:
                finalize_current(current, bouts)
                current = None
            previous_file_index = file_index
            previous_file_last_label = None

    if current is not None:
        finalize_current(current, bouts)

    if not row_counts or any(count != EXPECTED_ROWS_PER_FILE for count in row_counts):
        raise ValueError(
            "Expected exactly 6,000 rows per file; "
            f"observed min={min(row_counts)}, max={max(row_counts)}."
        )

    for index, bout in enumerate(bouts):
        bout.bout_id = index
        bout.previous_behavior = bouts[index - 1].behavior if index > 0 else None
        bout.next_behavior = bouts[index + 1].behavior if index + 1 < len(bouts) else None
        bout.starts_at_recording_boundary = bout.start_frame_global == 0
        bout.ends_at_recording_boundary = bout.end_frame_global == total_rows - 1
        bout.same_flank_ABA = bool(
            bout.previous_behavior is not None
            and bout.previous_behavior == bout.next_behavior
            and bout.previous_behavior != bout.behavior
        )
        if bout.same_flank_ABA:
            bout.flank_behavior = bout.previous_behavior

    duration_sum_ok = sum(bout.duration_frames for bout in bouts) == total_rows - invalid_rows
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
        "sampling_rate_hz": float(np.median(np.asarray(row_counts) / FILE_DURATION_SECONDS)),
    }
    return bouts, metadata


def update_aba_margins(
    bouts: list[Bout], metadata: dict[str, object]
) -> None:
    selected = {
        bout.bout_id: bout
        for bout in bouts
        if bout.same_flank_ABA and bout.behavior in NONREST_BEHAVIORS
    }
    if not selected:
        return
    file_records = metadata["file_records"]
    values_by_bout: dict[int, list[float]] = defaultdict(list)
    behavior_index = {behavior: index for index, behavior in enumerate(mi.BEHAVIORS)}
    for record in file_records:
        file_start = int(record["global_start"])
        file_end = int(record["global_end"])
        overlapping = [
            bout
            for bout in selected.values()
            if bout.start_frame_global <= file_end and bout.end_frame_global >= file_start
        ]
        if not overlapping:
            continue
        numeric, _, _, _ = load_file_numeric(Path(record["path"]))
        for bout in overlapping:
            start = max(bout.start_frame_global, file_start) - file_start
            end = min(bout.end_frame_global, file_end) - file_start
            b_index = behavior_index[bout.behavior]
            a_index = behavior_index[str(bout.flank_behavior)]
            differences = numeric[start : end + 1, b_index] - numeric[start : end + 1, a_index]
            values_by_bout[bout.bout_id].extend(differences.tolist())

    for bout_id, bout in selected.items():
        values = np.asarray(values_by_bout[bout_id], dtype=float)
        if len(values) != bout.duration_frames:
            raise AssertionError(
                f"A->B->A margin frame count mismatch for bout {bout_id}: "
                f"{len(values)} versus {bout.duration_frames}"
            )
        bout.mean_margin_B_vs_A = float(np.mean(values))
        bout.median_margin_B_vs_A = float(np.median(values))
        bout.min_margin_B_vs_A = float(np.min(values))
        bout.max_margin_B_vs_A = float(np.max(values))


def add_cycle_attribute(bouts: list[Bout], sampling_rate_hz: float) -> None:
    frames_per_cycle = int(round(24.0 * 3600.0 * sampling_rate_hz))
    for bout in bouts:
        # The audit uses the fixed 10 Hz, 24-hour demo timebase.
        bout.cycle = bout.start_frame_global // frames_per_cycle


def bout_frame_row(bout: Bout, sampling_rate_hz: float) -> dict[str, object]:
    return {
        "bout_id": bout.bout_id,
        "behavior": bout.behavior,
        "start_frame_global": bout.start_frame_global,
        "end_frame_global": bout.end_frame_global,
        "duration_frames": bout.duration_frames,
        "duration_seconds": bout.duration_frames / sampling_rate_hz,
        "duration_category": duration_category(bout.duration_frames),
        "previous_behavior": bout.previous_behavior,
        "next_behavior": bout.next_behavior,
        "starts_at_recording_boundary": bout.starts_at_recording_boundary,
        "ends_at_recording_boundary": bout.ends_at_recording_boundary,
        "same_flank_ABA": bout.same_flank_ABA,
        "flank_behavior": bout.flank_behavior,
        "start_file_index": bout.start_file_index,
        "end_file_index": bout.end_file_index,
        "start_row_index": bout.start_row_index,
        "end_row_index": bout.end_row_index,
        "mean_p_B": bout.mean_p_winner,
        "median_p_B": bout.median_p_winner,
        "min_p_B": bout.min_p_winner,
        "max_p_B": bout.max_p_winner,
        "mean_top2_margin": bout.mean_top2_margin,
        "median_top2_margin": bout.median_top2_margin,
        "min_top2_margin": bout.min_top2_margin,
        "mean_margin_B_vs_A": bout.mean_margin_B_vs_A,
        "median_margin_B_vs_A": bout.median_margin_B_vs_A,
        "min_margin_B_vs_A": bout.min_margin_B_vs_A,
        "max_margin_B_vs_A": bout.max_margin_B_vs_A,
    }


def make_primary_summary(bouts: list[Bout], sampling_rate_hz: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_behavior = {
        behavior: [bout for bout in bouts if bout.behavior == behavior]
        for behavior in NONREST_BEHAVIORS
    }
    for behavior in NONREST_BEHAVIORS:
        behavior_bouts = by_behavior[behavior]
        for category, _, _ in DURATION_CATEGORIES:
            selected = [
                bout
                for bout in behavior_bouts
                if duration_category(bout.duration_frames) == category
            ]
            aba = [bout for bout in selected if bout.same_flank_ABA]
            p_median, p_q25, p_q75, _ = finite_summary(
                bout.mean_p_winner for bout in selected
            )
            margin_median, margin_q25, margin_q75, _ = finite_summary(
                bout.mean_margin_B_vs_A for bout in aba
            )
            top_median, top_q25, top_q75, _ = finite_summary(
                bout.mean_top2_margin for bout in selected
            )
            rows.append(
                {
                    "behavior": behavior,
                    "duration_category": category,
                    "total_bouts": len(selected),
                    "fraction_of_behavior_bouts": len(selected) / len(behavior_bouts)
                    if behavior_bouts
                    else float("nan"),
                    "ABA_bouts": len(aba),
                    "fraction_ABA": len(aba) / len(selected) if selected else float("nan"),
                    "median_mean_p_B": p_median,
                    "q25_mean_p_B": p_q25,
                    "q75_mean_p_B": p_q75,
                    "IQR_mean_p_B": p_q75 - p_q25
                    if np.isfinite(p_q25) and np.isfinite(p_q75)
                    else float("nan"),
                    "median_mean_top2_margin": top_median,
                    "q25_mean_top2_margin": top_q25,
                    "q75_mean_top2_margin": top_q75,
                    "IQR_mean_top2_margin": top_q75 - top_q25
                    if np.isfinite(top_q25) and np.isfinite(top_q75)
                    else float("nan"),
                    "median_mean_margin_B_vs_A": margin_median,
                    "q25_mean_margin_B_vs_A": margin_q25,
                    "q75_mean_margin_B_vs_A": margin_q75,
                    "IQR_mean_margin_B_vs_A": margin_q75 - margin_q25
                    if np.isfinite(margin_q25) and np.isfinite(margin_q75)
                    else float("nan"),
                    "sampling_rate_hz": sampling_rate_hz,
                }
            )
    return pd.DataFrame(rows)


def make_overall_summary(bouts: list[Bout]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, _, _ in DURATION_CATEGORIES:
        selected = [
            bout for bout in bouts if duration_category(bout.duration_frames) == category
        ]
        aba = [bout for bout in selected if bout.same_flank_ABA]
        rows.append(
            {
                "duration_category": category,
                "total_bouts": len(selected),
                "fraction_ABA": len(aba) / len(selected) if selected else float("nan"),
                "median_mean_p_B": finite_summary(
                    bout.mean_p_winner for bout in selected
                )[0],
                "median_mean_top2_margin": finite_summary(
                    bout.mean_top2_margin for bout in selected
                )[0],
                "median_mean_margin_B_vs_A_ABA": finite_summary(
                    bout.mean_margin_B_vs_A for bout in aba
                )[0],
            }
        )
    return pd.DataFrame(rows)


def make_triplet_summary(bouts: list[Bout]) -> pd.DataFrame:
    groups: dict[tuple[str, str, str], list[Bout]] = defaultdict(list)
    for bout in bouts:
        if bout.same_flank_ABA and bout.duration_frames <= 4:
            groups[(str(bout.previous_behavior), bout.behavior, str(bout.next_behavior))].append(
                bout
            )
    rows: list[dict[str, object]] = []
    for (behavior_a, behavior_b, behavior_next), group in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        rows.append(
            {
                "previous_behavior_A": behavior_a,
                "bout_behavior_B": behavior_b,
                "next_behavior_A": behavior_next,
                "triplet": f"{behavior_a} -> {behavior_b} -> {behavior_next}",
                "count": len(group),
                "median_duration_frames_B": float(
                    np.median([bout.duration_frames for bout in group])
                ),
                "median_mean_p_B": float(
                    np.median([bout.mean_p_winner for bout in group])
                ),
                "median_mean_margin_B_vs_A": float(
                    np.median([bout.mean_margin_B_vs_A for bout in group])
                ),
            }
        )
    return pd.DataFrame(rows)


def select_diverse(records: list[Bout], count: int, excluded: set[int]) -> list[Bout]:
    available = [bout for bout in records if bout.bout_id not in excluded]
    by_behavior: dict[str, list[Bout]] = defaultdict(list)
    for bout in sorted(available, key=lambda item: (item.behavior, item.start_frame_global)):
        by_behavior[bout.behavior].append(bout)
    pointers = {behavior: 0 for behavior in by_behavior}
    selected: list[Bout] = []
    while len(selected) < count and by_behavior:
        progressed = False
        for behavior in sorted(by_behavior):
            pointer = pointers[behavior]
            if pointer < len(by_behavior[behavior]) and len(selected) < count:
                selected.append(by_behavior[behavior][pointer])
                pointers[behavior] += 1
                progressed = True
        if not progressed:
            break
    return selected


def select_manual_review_sample(bouts: list[Bout]) -> list[tuple[str, Bout]]:
    selected: list[tuple[str, Bout]] = []
    used: set[int] = set()
    one_frame = [bout for bout in bouts if bout.same_flank_ABA and bout.duration_frames == 1]
    two_four = [
        bout for bout in bouts if bout.same_flank_ABA and 2 <= bout.duration_frames <= 4
    ]
    ten_fifteen = [bout for bout in bouts if 10 <= bout.duration_frames <= 15]
    for label, records in (
        ("one_frame_ABA", one_frame),
        ("two_to_four_frame_ABA", two_four),
        ("ten_to_fifteen_frame_nonrest", ten_fifteen),
    ):
        chosen = select_diverse(records, 8, used)
        selected.extend((label, bout) for bout in chosen)
        used.update(bout.bout_id for bout in chosen)
    return selected


def read_trace_file(
    record: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    numeric, labels, _, _ = load_file_numeric(Path(record["path"]))
    return numeric, labels


def write_manual_sample_outputs(
    selected: list[tuple[str, Bout]],
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    file_records = metadata["file_records"]
    file_starts = [int(record["global_start"]) for record in file_records]
    file_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sample_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    def file_for_frame(global_frame: int) -> dict[str, object]:
        index = int(np.searchsorted(file_starts, global_frame, side="right") - 1)
        return file_records[index]

    for sample_number, (selection_group, bout) in enumerate(selected, start=1):
        sample_id = f"sample_{sample_number:03d}"
        sample_rows.append(
            {
                "sample_id": sample_id,
                "selection_group": selection_group,
                "bout_id": bout.bout_id,
                "behavior_B": bout.behavior,
                "duration_frames": bout.duration_frames,
                "duration_seconds": bout.duration_frames / SAMPLING_RATE_HZ,
                "previous_behavior": bout.previous_behavior,
                "next_behavior": bout.next_behavior,
                "same_flank_ABA": bout.same_flank_ABA,
                "flank_behavior_A": bout.flank_behavior,
                "mean_p_B": bout.mean_p_winner,
                "mean_top2_margin": bout.mean_top2_margin,
                "mean_margin_B_vs_A": bout.mean_margin_B_vs_A,
                "start_file_index": bout.start_file_index,
                "end_file_index": bout.end_file_index,
                "start_row_index": bout.start_row_index,
                "end_row_index": bout.end_row_index,
                "start_frame_global": bout.start_frame_global,
                "end_frame_global": bout.end_frame_global,
            }
        )
        window_start = max(0, bout.start_frame_global - LOCAL_WINDOW_FRAMES)
        window_end = min(
            int(metadata["total_rows"]) - 1,
            bout.end_frame_global + LOCAL_WINDOW_FRAMES,
        )
        for global_frame in range(window_start, window_end + 1):
            record = file_for_frame(global_frame)
            file_index = int(record["file_index"])
            if file_index not in file_cache:
                file_cache[file_index] = read_trace_file(record)
            numeric, labels = file_cache[file_index]
            local_row = global_frame - int(record["global_start"])
            label = int(labels[local_row])
            trace_rows.append(
                {
                    "sample_id": sample_id,
                    "relative_frame": global_frame - bout.start_frame_global,
                    "relative_time_seconds": (global_frame - bout.start_frame_global)
                    / SAMPLING_RATE_HZ,
                    "WTA_state": mi.BEHAVIORS[label] if label >= 0 else None,
                    **{
                        behavior: float(numeric[local_row, index])
                        for index, behavior in enumerate(mi.BEHAVIORS)
                    },
                }
            )

    pd.DataFrame(sample_rows).to_csv(output_dir / "manual_review_sample.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(
        output_dir / "sampled_short_bout_probability_traces.csv", index=False
    )


def write_run_summary(
    output_dir: Path,
    metadata: dict[str, object],
    analysis_bouts: list[Bout],
    selected: list[tuple[str, Bout]],
    runtime_seconds: float,
) -> None:
    overall = pd.read_csv(output_dir / "confidence_by_duration_overall.csv")
    triplets = pd.read_csv(output_dir / "short_ABA_triplets.csv")
    lines = [
        "Short-bout confidence audit: 783E demo mouse",
        "==============================================",
        "",
        f"input_directory: {metadata['input_dir']}",
        f"source_file_count: {metadata['file_count']}",
        f"source_rows: {metadata['total_rows']}",
        f"rows_per_file: {min(metadata['row_counts'])}-{max(metadata['row_counts'])}",
        f"sampling_rate_hz: {metadata['sampling_rate_hz']}",
        f"recording_duration_hours: {metadata['total_rows'] / metadata['sampling_rate_hz'] / 3600.0:.9f}",
        f"analysis_scope: {metadata['complete_cycles']} complete 24-hour cycles, matching the prior demo diagnostic",
        f"analysis_nonrest_bouts: {len(analysis_bouts)}",
        "",
        "Validation",
        "----------",
        f"file_count_617: {'PASS' if metadata['file_count'] == 617 else 'FAIL'}",
        f"row_count_6000_each: {'PASS' if min(metadata['row_counts']) == max(metadata['row_counts']) == 6000 else 'FAIL'}",
        f"invalid_probability_rows: {metadata['invalid_rows']}",
        f"partial_probability_rows: {metadata['partial_probability_rows']}",
        f"WTA_argmax_mismatches: {metadata['wta_mismatch_rows']}",
        f"header_consistent: {'PASS' if metadata['header_consistent'] else 'FAIL'}",
        f"same_WTA_state_at_file_boundaries: {metadata['boundary_same_state_count']}/{metadata['boundary_count']}",
        f"bout_duration_sum_matches_valid_rows: {'PASS' if metadata['duration_sum_ok'] else 'FAIL'}",
        "raw probabilities and WTA labels were not transformed, smoothed, filtered, or bridged",
        "",
        "Duration-class overview",
        "-----------------------",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"{row.duration_category}: bouts={row.total_bouts}; "
            f"ABA_fraction={row.fraction_ABA:.6g}; "
            f"median_mean_p_B={row.median_mean_p_B:.6g}; "
            f"median_top2_margin={row.median_mean_top2_margin:.6g}; "
            f"median_B_vs_A_margin_ABA={row.median_mean_margin_B_vs_A_ABA:.6g}"
        )
    lines.extend(
        [
            "",
            "Most common 1-4-frame ABA triplets",
            "-----------------------------------",
        ]
    )
    if triplets.empty:
        lines.append("none")
    else:
        for row in triplets.head(10).itertuples(index=False):
            lines.append(
                f"{row.triplet}: count={row.count}; median_frames={row.median_duration_frames_B:.3g}; "
                f"median_mean_p_B={row.median_mean_p_B:.6g}; "
                f"median_B_vs_A_margin={row.median_mean_margin_B_vs_A:.6g}"
            )
    lines.extend(
        [
            "",
            "Manual review sample",
            "--------------------",
            f"sample_count: {len(selected)}",
            "",
            "Interpretation",
            "--------------",
            "This audit is descriptive only. Short duration is not treated as evidence that a bout is false.",
            "No margin threshold or bridge rule was selected or applied.",
            "",
            f"runtime_seconds: {runtime_seconds:.6f}",
            "raw CSV files were read only and were not modified",
        ]
    )
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    bouts, metadata = build_bouts(input_dir)
    add_cycle_attribute(bouts, float(metadata["sampling_rate_hz"]))
    complete_cycles = int(
        math.floor(
            metadata["total_rows"]
            / (24.0 * 3600.0 * float(metadata["sampling_rate_hz"]))
        )
    )
    metadata["complete_cycles"] = complete_cycles
    update_aba_margins(bouts, metadata)
    analysis_bouts = [
        bout
        for bout in bouts
        if bout.cycle < complete_cycles and bout.behavior in NONREST_BEHAVIORS
    ]
    if len(analysis_bouts) == 0:
        raise ValueError(
            f"No non-rest bouts were found in the {complete_cycles} complete cycles"
        )

    bout_table = pd.DataFrame(
        [bout_frame_row(bout, float(metadata["sampling_rate_hz"])) for bout in analysis_bouts]
    )
    primary = make_primary_summary(analysis_bouts, float(metadata["sampling_rate_hz"]))
    overall = make_overall_summary(analysis_bouts)
    triplets = make_triplet_summary(analysis_bouts)
    selected = select_manual_review_sample(analysis_bouts)

    bout_table.to_csv(output_dir / "bout_confidence_table.csv", index=False)
    primary.to_csv(output_dir / "confidence_by_behavior_duration.csv", index=False)
    overall.to_csv(output_dir / "confidence_by_duration_overall.csv", index=False)
    triplets.to_csv(output_dir / "short_ABA_triplets.csv", index=False)
    write_manual_sample_outputs(selected, metadata, output_dir)
    write_run_summary(
        output_dir,
        metadata,
        analysis_bouts,
        selected,
        time.perf_counter() - start_time,
    )
    print(
        f"Short-bout confidence audit complete: {len(analysis_bouts)} non-rest bouts; "
        f"outputs={output_dir}"
    )


if __name__ == "__main__":
    main()
