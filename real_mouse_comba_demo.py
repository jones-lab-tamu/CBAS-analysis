"""Run the frozen provisional event-level COMBA diagnostic on one CBAS mouse.

This is intentionally a one-animal diagnostic runner.  It reuses the validated
MI reader for raw WTA classification and the validated synthetic COMBA
implementation for scoring and the marked-block null.  It does not smooth,
bridge, filter, or otherwise alter the raw classifier sequence.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as mi
import synthetic_comba_validation as comba


FRP_HOURS = 24.0
FRP_SOURCE = "demo_fixed"
NULL_REPLICATES = 100
SIGMA_MINUTES = 3
WEIGHTING = "sqrt_count"
NULL_FAMILY = "marked_block"
NULL_BLOCK_SIZE = 3
OUTPUT_DIRECTORY_NAME = "COMBA_Demo_Output"

# This is only a memory-safety check before the unmodified naive DP is called.
# It is not a COMBA scoring threshold and does not change any score.
NAIVE_SCORE_CELL_LIMIT = 50_000_000

NONREST_BEHAVIORS = tuple(
    behavior for behavior in mi.BEHAVIORS if behavior != "resting"
)
RESTING_INDEX = mi.BEHAVIORS.index("resting")
FILE_DURATION_SECONDS = mi.FILE_DURATION_MINUTES * 60.0


@dataclass
class Bout:
    behavior: str
    onset_hours: float
    phase_hours: float
    cycle: int
    duration_frames: int
    start_file_index: int
    end_file_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the provisional event-level COMBA diagnostic for one CBAS mouse."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(r"C:\Users\Jeff\Documents\CBAS_Analysis_Data"),
        help="Folder containing one animal's sequential CBAS CSV files.",
    )
    parser.add_argument(
        "--allow-large-naive-scoring",
        action="store_true",
        help=(
            "Bypass the diagnostic memory-safety stop and attempt the unmodified "
            "naive cyclic DP even for very large raw-WTA event counts."
        ),
    )
    return parser.parse_args()


def expected_sampling_rate(file_row_counts: list[int]) -> tuple[float, float, float]:
    rates = np.asarray(file_row_counts, dtype=float) / FILE_DURATION_SECONDS
    return float(np.median(rates)), float(np.min(rates)), float(np.max(rates))


def close_bout(
    bouts: list[Bout],
    state_index: int | None,
    onset_hours: float | None,
    phase_hours: float | None,
    cycle: int | None,
    duration_frames: int,
    start_file_index: int | None,
    end_file_index: int | None,
) -> None:
    if state_index is None:
        return
    assert onset_hours is not None
    assert phase_hours is not None
    assert cycle is not None
    assert start_file_index is not None
    assert end_file_index is not None
    bouts.append(
        Bout(
            behavior=mi.BEHAVIORS[state_index],
            onset_hours=float(onset_hours),
            phase_hours=float(phase_hours),
            cycle=int(cycle),
            duration_frames=int(duration_frames),
            start_file_index=int(start_file_index),
            end_file_index=int(end_file_index),
        )
    )


def load_and_build_bouts(
    input_dir: Path,
) -> tuple[list[Bout], dict[str, object]]:
    """Read the raw files, apply raw WTA, and build maximal contiguous bouts."""

    input_files, missing_indices = mi.discover_input_files(input_dir)
    indices = [index for index, _ in input_files]
    if len(input_files) != 617 or missing_indices or indices != list(range(617)):
        raise ValueError(
            "Expected exactly consecutive files 00000-00616; "
            f"found count={len(input_files)}, missing={missing_indices}, "
            f"first={indices[0] if indices else None}, "
            f"last={indices[-1] if indices else None}."
        )

    first_header: list[str] | None = None
    file_row_counts: list[int] = []
    invalid_rows = 0
    file_first_labels: dict[int, int] = {}
    file_last_labels: dict[int, int] = {}
    file_boundary_gaps_seconds: list[float] = []
    bouts: list[Bout] = []

    anchor_elapsed_hours: float | None = None
    previous_file_index: int | None = None
    previous_row_position: int | None = None
    previous_row_count: int | None = None
    previous_time_hours: float | None = None

    state_index: int | None = None
    onset_hours: float | None = None
    onset_phase_hours: float | None = None
    onset_cycle: int | None = None
    duration_frames = 0
    start_file_index: int | None = None
    end_file_index: int | None = None

    for file_index, path in input_files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if first_header is None:
            first_header = header
        elif header != first_header:
            raise ValueError(f"Inconsistent CSV columns in {path.name}")
        missing_behavior_columns = [
            behavior for behavior in mi.BEHAVIORS if behavior not in header
        ]
        if missing_behavior_columns:
            raise ValueError(
                f"Missing behavior columns in {path.name}: {missing_behavior_columns}"
            )

        n_rows, row_positions, labels = mi.read_and_classify(path)
        file_row_counts.append(n_rows)
        invalid_rows += n_rows - len(labels)
        if labels.size:
            if np.any((labels < 0) | (labels >= len(mi.BEHAVIORS))):
                raise ValueError(f"Invalid WTA label in {path.name}")
            file_first_labels[file_index] = int(labels[0])
            file_last_labels[file_index] = int(labels[-1])

        if n_rows == 0 or labels.size == 0:
            close_bout(
                bouts,
                state_index,
                onset_hours,
                onset_phase_hours,
                onset_cycle,
                duration_frames,
                start_file_index,
                end_file_index,
            )
            state_index = None
            onset_hours = None
            onset_phase_hours = None
            onset_cycle = None
            duration_frames = 0
            start_file_index = None
            end_file_index = None
            previous_file_index = file_index
            previous_row_position = None
            previous_row_count = n_rows
            previous_time_hours = None
            continue

        raw_elapsed_hours = (
            file_index * mi.FILE_DURATION_MINUTES / 60.0
            + row_positions.astype(float) / n_rows * mi.FILE_DURATION_MINUTES / 60.0
        )
        if file_index == 0:
            anchor_elapsed_hours = float(raw_elapsed_hours[0])
        assert anchor_elapsed_hours is not None
        relative_times = raw_elapsed_hours - anchor_elapsed_hours

        if previous_time_hours is not None:
            boundary_gap = (relative_times[0] - previous_time_hours) * 3600.0
            file_boundary_gaps_seconds.append(float(boundary_gap))

        for position, label, elapsed_hours in zip(
            row_positions.tolist(), labels.tolist(), relative_times.tolist()
        ):
            contiguous = False
            if previous_file_index is not None and previous_row_position is not None:
                if file_index == previous_file_index:
                    contiguous = position == previous_row_position + 1
                elif file_index == previous_file_index + 1:
                    contiguous = (
                        previous_row_count is not None
                        and previous_row_position == previous_row_count - 1
                        and position == 0
                    )

            if not contiguous and state_index is not None:
                close_bout(
                    bouts,
                    state_index,
                    onset_hours,
                    onset_phase_hours,
                    onset_cycle,
                    duration_frames,
                    start_file_index,
                    end_file_index,
                )
                state_index = None
                onset_hours = None
                onset_phase_hours = None
                onset_cycle = None
                duration_frames = 0
                start_file_index = None
                end_file_index = None

            if state_index != int(label):
                if state_index is not None:
                    close_bout(
                        bouts,
                        state_index,
                        onset_hours,
                        onset_phase_hours,
                        onset_cycle,
                        duration_frames,
                        start_file_index,
                        end_file_index,
                    )
                state_index = int(label)
                onset_hours = float(elapsed_hours)
                onset_phase_hours = float(np.mod(elapsed_hours, FRP_HOURS))
                onset_cycle = int(math.floor(elapsed_hours / FRP_HOURS))
                duration_frames = 1
                start_file_index = file_index
            else:
                duration_frames += 1
            end_file_index = file_index
            previous_file_index = file_index
            previous_row_position = position
            previous_row_count = n_rows
            previous_time_hours = float(elapsed_hours)

    close_bout(
        bouts,
        state_index,
        onset_hours,
        onset_phase_hours,
        onset_cycle,
        duration_frames,
        start_file_index,
        end_file_index,
    )

    if anchor_elapsed_hours is None:
        raise ValueError("No valid sample was found in file 00000")

    sampling_rate_hz, sampling_rate_min, sampling_rate_max = expected_sampling_rate(
        file_row_counts
    )
    total_duration_hours = (
        len(input_files) * mi.FILE_DURATION_MINUTES / 60.0 - anchor_elapsed_hours
    )
    complete_cycles = int(math.floor(total_duration_hours / FRP_HOURS))
    complete_bouts = [
        bout for bout in bouts if 0 <= bout.cycle < complete_cycles
    ]
    boundary_same_label_count = sum(
        1
        for index in range(1, len(input_files))
        if index in file_first_labels
        and index - 1 in file_last_labels
        and file_first_labels[index] == file_last_labels[index - 1]
    )
    all_duration_frames = sum(bout.duration_frames for bout in bouts)
    valid_rows = sum(file_row_counts) - invalid_rows
    duration_sum_ok = all_duration_frames == valid_rows
    temporal_continuity_ok = bool(
        file_boundary_gaps_seconds
        and min(file_boundary_gaps_seconds) > 0
        and max(file_boundary_gaps_seconds) < 1.0
    )

    metadata: dict[str, object] = {
        "input_files": input_files,
        "missing_indices": missing_indices,
        "animal_id": input_files[0][1].name.split("_")[0],
        "file_count": len(input_files),
        "first_file_index": indices[0],
        "last_file_index": indices[-1],
        "file_row_counts": file_row_counts,
        "rows_per_file_min": min(file_row_counts),
        "rows_per_file_max": max(file_row_counts),
        "rows_per_file_unique": len(set(file_row_counts)),
        "invalid_rows": invalid_rows,
        "valid_rows": valid_rows,
        "anchor_elapsed_hours": anchor_elapsed_hours,
        "total_duration_hours": total_duration_hours,
        "complete_cycles": complete_cycles,
        "excluded_partial_duration_hours": total_duration_hours
        - complete_cycles * FRP_HOURS,
        "sampling_rate_hz": sampling_rate_hz,
        "sampling_rate_min_hz": sampling_rate_min,
        "sampling_rate_max_hz": sampling_rate_max,
        "boundary_same_label_count": boundary_same_label_count,
        "boundary_count": len(input_files) - 1,
        "boundary_gap_min_seconds": min(file_boundary_gaps_seconds),
        "boundary_gap_max_seconds": max(file_boundary_gaps_seconds),
        "duration_sum_ok": duration_sum_ok,
        "temporal_continuity_ok": temporal_continuity_ok,
        "all_bouts": bouts,
        "complete_bouts": complete_bouts,
    }
    return bouts, metadata


def complete_cycle_events(
    complete_bouts: list[Bout], complete_cycles: int
) -> list[list[comba.Event]]:
    cycles: list[list[comba.Event]] = [[] for _ in range(complete_cycles)]
    for bout in complete_bouts:
        if bout.behavior in NONREST_BEHAVIORS:
            cycles[bout.cycle].append((bout.behavior, bout.phase_hours))
    for cycle in cycles:
        cycle.sort(key=lambda event: event[1])
    return cycles


def cycle_mass(events: list[comba.Event]) -> float:
    return float(comba.event_weights(events, WEIGHTING).sum())


def bout_diagnostics(
    complete_bouts: list[Bout], sampling_rate_hz: float, complete_cycles: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for behavior in NONREST_BEHAVIORS:
        selected = [bout for bout in complete_bouts if bout.behavior == behavior]
        durations_frames = np.asarray(
            [bout.duration_frames for bout in selected], dtype=float
        )
        durations_seconds = durations_frames / sampling_rate_hz
        n = len(selected)
        rows.append(
            {
                "behavior": behavior,
                "total_bouts": n,
                "bouts_per_complete_cycle": n / complete_cycles
                if complete_cycles
                else float("nan"),
                "median_duration_seconds": float(np.median(durations_seconds))
                if n
                else float("nan"),
                "duration_q25_seconds": float(np.quantile(durations_seconds, 0.25))
                if n
                else float("nan"),
                "duration_q75_seconds": float(np.quantile(durations_seconds, 0.75))
                if n
                else float("nan"),
                "minimum_duration_seconds": float(np.min(durations_seconds))
                if n
                else float("nan"),
                "fraction_1_frame": float(np.mean(durations_frames == 1))
                if n
                else float("nan"),
                "fraction_2_4_frames": float(
                    np.mean((durations_frames >= 2) & (durations_frames <= 4))
                )
                if n
                else float("nan"),
                "fraction_5_9_frames": float(
                    np.mean((durations_frames >= 5) & (durations_frames <= 9))
                )
                if n
                else float("nan"),
                "fraction_approximately_1_second": float(
                    np.mean(np.abs(durations_seconds - 1.0) <= 0.1)
                )
                if n
                else float("nan"),
                "fraction_gt_1_second": float(np.mean(durations_seconds > 1.0))
                if n
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def event_counts_by_cycle(
    complete_bouts: list[Bout], complete_cycles: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cycle in range(complete_cycles):
        counts = {behavior: 0 for behavior in NONREST_BEHAVIORS}
        for bout in complete_bouts:
            if bout.cycle == cycle and bout.behavior in counts:
                counts[bout.behavior] += 1
        rows.append(
            {
                "cycle": cycle,
                **counts,
                "total_nonrest_bouts": sum(counts.values()),
            }
        )
    return pd.DataFrame(rows)


def null_replicate_scores(
    cycles: list[list[comba.Event]], animal_id: str
) -> np.ndarray:
    """Use the existing marked-block generator and cyclic score unchanged."""

    case_id = f"real_mouse_{animal_id}"
    rng = np.random.default_rng(
        comba.stable_seed(
            case_id,
            float(SIGMA_MINUTES),
            WEIGHTING,
            NULL_FAMILY,
            NULL_BLOCK_SIZE,
        )
    )
    replicate_scores = np.empty(NULL_REPLICATES, dtype=float)
    empty_missing = [[] for _ in cycles]
    for replicate in range(NULL_REPLICATES):
        pair_scores: list[float] = []
        for index in range(len(cycles) - 1):
            events_a, events_b = comba.eligible_pair(
                cycles[index],
                cycles[index + 1],
                empty_missing[index],
                empty_missing[index + 1],
                SIGMA_MINUTES,
            )
            null_a = comba.generate_null_cycle(
                events_a, NULL_FAMILY, rng, NULL_BLOCK_SIZE
            )
            null_b = comba.generate_null_cycle(
                events_b, NULL_FAMILY, rng, NULL_BLOCK_SIZE
            )
            pair_scores.append(
                comba.score_pair(
                    null_a,
                    null_b,
                    SIGMA_MINUTES,
                    WEIGHTING,
                    "monotone",
                )["score"]
            )
        replicate_scores[replicate] = float(np.mean(pair_scores))
    return replicate_scores


def write_run_summary(
    path: Path,
    metadata: dict[str, object],
    cycles: list[list[comba.Event]],
    pair_rows: list[dict[str, object]],
    observed_mean: float,
    null_mean: float,
    null_sd: float,
    null_se: float,
    corrected_r: float,
    runtimes: dict[str, float],
    scoring_status: str,
    warning: str,
) -> None:
    counts_frame = event_counts_by_cycle(
        metadata["complete_bouts"], metadata["complete_cycles"]
    )
    lines = [
        "Single-mouse provisional COMBA demo",
        "===================================",
        "",
        f"input_directory: {metadata['input_directory']}",
        f"animal_id: {metadata['animal_id']}",
        f"source_file_count: {metadata['file_count']}",
        f"source_file_indices: {metadata['first_file_index']}-{metadata['last_file_index']}",
        f"missing_source_indices: {metadata['missing_indices'] or 'none'}",
        f"FRP_hours: {FRP_HOURS}",
        f"FRP_source: {FRP_SOURCE}",
        "phase_anchor: first valid sample in file 00000",
        f"total_recording_hours: {metadata['total_duration_hours']:.9f}",
        f"complete_cycles: {metadata['complete_cycles']}",
        f"excluded_partial_duration_hours: {metadata['excluded_partial_duration_hours']:.9f}",
        f"adjacent_cycle_pairs: {max(0, int(metadata['complete_cycles']) - 1)}",
        f"sampling_rate_hz: {metadata['sampling_rate_hz']:.9f}",
        f"sampling_rate_range_hz: {metadata['sampling_rate_min_hz']:.9f}-{metadata['sampling_rate_max_hz']:.9f}",
        "",
        "Validation checks",
        "-----------------",
        f"expected_617_files: {'PASS' if metadata['file_count'] == 617 else 'FAIL'}",
        f"numeric_consecutive_order: {'PASS' if not metadata['missing_indices'] else 'FAIL'}",
        f"behavior_columns_consistent: {'PASS' if metadata['header_consistent'] else 'FAIL'}",
        f"rows_per_file: min={metadata['rows_per_file_min']}, max={metadata['rows_per_file_max']}, unique={metadata['rows_per_file_unique']}",
        f"invalid_rows: {metadata['invalid_rows']}",
        f"temporal_continuity: {'PASS' if metadata['temporal_continuity_ok'] else 'FAIL'}; boundary_gap_seconds={metadata['boundary_gap_min_seconds']:.6g}-{metadata['boundary_gap_max_seconds']:.6g}",
        f"same_WTA_state_across_file_boundaries: {metadata['boundary_same_label_count']}/{metadata['boundary_count']}",
        f"bout_duration_frame_sum_matches_valid_rows: {'PASS' if metadata['duration_sum_ok'] else 'FAIL'}",
        "raw_WTA: no smoothing, bridging, duration filtering, or confidence filtering",
        "",
        "Complete-cycle non-rest event counts",
        "------------------------------------",
    ]
    lines.extend(",".join(str(value) for value in row) for row in counts_frame.itertuples(index=False, name=None))
    lines.extend(
        [
            "",
            "Adjacent-cycle COMBA",
            "--------------------",
            f"scoring_status: {scoring_status}",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"cycles {row['cycle_a']}-{row['cycle_b']}: raw_COMBA={row['raw_COMBA']}, events={row['events_cycle_a']}/{row['events_cycle_b']}, mass={row['mass_cycle_a']:.6g}/{row['mass_cycle_b']:.6g}, status={row['status']}"
        )
    lines.extend(
        [
            f"observed_mean_COMBA: {observed_mean}",
            "",
            "Preferred marked-block null",
            "---------------------------",
            f"null_family: {NULL_FAMILY}",
            f"null_block_size: {NULL_BLOCK_SIZE}",
            f"null_replicates: {NULL_REPLICATES}",
            f"null_mean: {null_mean}",
            f"null_SD: {null_sd}",
            f"null_SE: {null_se}",
            f"corrected_COMBA_R: {corrected_r}",
            "",
            "Runtime seconds",
            "---------------",
            f"loading_and_WTA_bout_construction: {runtimes['loading_seconds']:.6f}",
            f"observed_COMBA_scoring: {runtimes['observed_seconds']:.6f}",
            f"100_replicate_marked_block_null: {runtimes['null_seconds']:.6f}",
            f"total_runtime: {runtimes['total_seconds']:.6f}",
            "",
            "Warnings / interpretation",
            "-------------------------",
            warning,
            "This is one mouse only; no genotype or significance interpretation is made.",
            "Duration flags: approximately 1 second means 0.9-1.1 seconds; duration fractions are diagnostic flags and may overlap frame bins.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = input_dir / OUTPUT_DIRECTORY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    bouts, metadata = load_and_build_bouts(input_dir)
    metadata["input_directory"] = input_dir
    metadata["header_consistent"] = True
    loading_seconds = time.perf_counter() - run_start
    complete_cycles = int(metadata["complete_cycles"])
    complete_bouts = metadata["complete_bouts"]
    cycles = complete_cycle_events(complete_bouts, complete_cycles)
    event_counts_frame = event_counts_by_cycle(complete_bouts, complete_cycles)
    diagnostics_frame = bout_diagnostics(
        complete_bouts,
        float(metadata["sampling_rate_hz"]),
        complete_cycles,
    )

    pair_rows: list[dict[str, object]] = []
    for index in range(max(0, complete_cycles - 1)):
        pair_rows.append(
            {
                "cycle_a": index,
                "cycle_b": index + 1,
                "raw_COMBA": float("nan"),
                "events_cycle_a": len(cycles[index]),
                "events_cycle_b": len(cycles[index + 1]),
                "mass_cycle_a": cycle_mass(cycles[index]),
                "mass_cycle_b": cycle_mass(cycles[index + 1]),
                "status": "not_run",
            }
        )

    max_pair_cells = max(
        (
            (len(cycles[index]) + 1) * (len(cycles[index + 1]) + 1)
            for index in range(max(0, complete_cycles - 1))
        ),
        default=0,
    )
    cyclic_work_units = sum(
        min(len(cycles[index]), len(cycles[index + 1]))
        * (len(cycles[index]) + 1)
        * (len(cycles[index + 1]) + 1)
        for index in range(max(0, complete_cycles - 1))
    )

    observed_mean = float("nan")
    null_mean = float("nan")
    null_sd = float("nan")
    null_se = float("nan")
    corrected_r = float("nan")
    observed_seconds = 0.0
    null_seconds = 0.0
    scoring_status = "not_run"
    warning = ""
    if max_pair_cells > NAIVE_SCORE_CELL_LIMIT and not args.allow_large_naive_scoring:
        warning = (
            "Raw WTA produced a naive cyclic-DP event-pair matrix larger than the "
            f"diagnostic safety limit: max_pair_dp_cells={max_pair_cells}; "
            f"estimated cyclic DP work units={cyclic_work_units}. Observed COMBA "
            "and the 100-replicate null were not run to avoid an unbounded memory/time "
            "allocation. This is a runtime diagnostic, not a COMBA score threshold."
        )
    else:
        observed_start = time.perf_counter()
        for index in range(max(0, complete_cycles - 1)):
            result = comba.score_pair(
                cycles[index],
                cycles[index + 1],
                SIGMA_MINUTES,
                WEIGHTING,
                "monotone",
            )
            if not 0.0 <= result["score"] <= 1.0 + 1e-12:
                raise AssertionError("Observed pair score fell outside [0, 1]")
            pair_rows[index]["raw_COMBA"] = result["score"]
            pair_rows[index]["mass_cycle_a"] = result["mass_a"]
            pair_rows[index]["mass_cycle_b"] = result["mass_b"]
            pair_rows[index]["status"] = "scored"
        observed_seconds = time.perf_counter() - observed_start
        observed_mean = float(np.mean([row["raw_COMBA"] for row in pair_rows]))

        null_start = time.perf_counter()
        null_scores = null_replicate_scores(cycles, str(metadata["animal_id"]))
        null_seconds = time.perf_counter() - null_start
        if np.any((null_scores < -1e-12) | (null_scores > 1.0 + 1e-12)):
            raise AssertionError("Null score fell outside [0, 1]")
        null_mean = float(np.mean(null_scores))
        null_sd = float(np.std(null_scores, ddof=1))
        null_se = null_sd / math.sqrt(NULL_REPLICATES)
        corrected_r = comba.chance_correct(observed_mean, null_mean)
        scoring_status = "scored"
        warning = (
            "Observed and null pairs used the same validated cyclic matcher; "
            "original onset phases were not shifted."
        )

    pair_frame = pd.DataFrame(pair_rows)
    summary_row = {
        "animal_id": metadata["animal_id"],
        "FRP_hours": FRP_HOURS,
        "FRP_source": FRP_SOURCE,
        "total_recording_hours": metadata["total_duration_hours"],
        "complete_cycles": complete_cycles,
        "adjacent_cycle_pairs": max(0, complete_cycles - 1),
        "total_nonrest_bouts_complete_cycles": len(complete_bouts)
        - sum(1 for bout in complete_bouts if bout.behavior == "resting"),
        "median_nonrest_bouts_per_cycle": float(
            np.median(event_counts_frame["total_nonrest_bouts"])
        ),
        "max_nonrest_bouts_per_cycle": int(
            event_counts_frame["total_nonrest_bouts"].max()
        ),
        "observed_mean_COMBA": observed_mean,
        "null_replicates": NULL_REPLICATES,
        "null_mean": null_mean,
        "null_SD": null_sd,
        "null_SE": null_se,
        "corrected_COMBA_R": corrected_r,
        "runtime_observed_seconds": observed_seconds,
        "runtime_null_seconds": null_seconds,
        "runtime_total_seconds": 0.0,
        "scoring_status": scoring_status,
        "max_pair_dp_cells": max_pair_cells,
        "estimated_cyclic_dp_work_units": cyclic_work_units,
    }

    event_counts_frame.to_csv(output_dir / "comba_event_counts_by_cycle.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "comba_bout_diagnostics.csv", index=False)
    pair_frame.to_csv(output_dir / "comba_pair_scores.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "animal_comba_summary.csv", index=False
    )
    runtimes = {
        "loading_seconds": loading_seconds,
        "observed_seconds": observed_seconds,
        "null_seconds": null_seconds,
        "total_seconds": time.perf_counter() - run_start,
    }
    summary_row["runtime_total_seconds"] = runtimes["total_seconds"]
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "animal_comba_summary.csv", index=False
    )
    write_run_summary(
        output_dir / "run_summary.txt",
        metadata,
        cycles,
        pair_rows,
        observed_mean,
        null_mean,
        null_sd,
        null_se,
        corrected_r,
        runtimes,
        scoring_status,
        warning,
    )
    print(
        f"COMBA demo complete: status={scoring_status}; "
        f"outputs={output_dir}; total_seconds={runtimes['total_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
