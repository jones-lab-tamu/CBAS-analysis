"""Run the frozen raw practical COMBA analysis for one animal.

The FRP/CT solution and winner-take-all reader are reused from the existing
single-animal MI script.  The three-origin practical scorer is reused from
``practical_comba_validation``; this file only adapts raw frame labels to
animal-specific complete-cycle event lists and writes the requested outputs.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import phase_behavior_mutual_information as mi
import practical_comba_validation as scorer


OUTPUT_DIRECTORY_NAME = "COMBA_Output"
NONREST_BEHAVIORS = tuple(
    behavior for behavior in mi.BEHAVIORS if behavior != "resting"
)
FILE_DURATION_MINUTES = mi.FILE_DURATION_MINUTES


@dataclass(frozen=True)
class Bout:
    behavior: str
    onset_elapsed_hours: float
    valid_row_count: int
    start_file_index: int
    end_file_index: int


@dataclass(frozen=True)
class ExtractedData:
    bouts: tuple[Bout, ...]
    total_valid_rows: int
    total_invalid_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen raw practical COMBA for one animal."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="One animal directory containing raw CBAS files and FRP_Phase_Output.",
    )
    return parser.parse_args()


def _close_bout(
    completed: list[Bout],
    current_behavior: str | None,
    current_onset_hours: float | None,
    current_valid_row_count: int,
    current_start_file_index: int | None,
    current_end_file_index: int | None,
) -> None:
    if current_behavior is None:
        return
    if (
        current_onset_hours is None
        or current_start_file_index is None
        or current_end_file_index is None
    ):
        raise AssertionError("Incomplete current bout state")
    completed.append(
        Bout(
            behavior=current_behavior,
            onset_elapsed_hours=current_onset_hours,
            valid_row_count=current_valid_row_count,
            start_file_index=current_start_file_index,
            end_file_index=current_end_file_index,
        )
    )


def extract_wta_bouts(
    input_files: Sequence[tuple[int, Path]],
) -> ExtractedData:
    """Extract raw maximal contiguous WTA bouts, including resting bouts."""

    if not input_files:
        raise ValueError("No input files were supplied")

    first_file_index = input_files[0][0]
    completed: list[Bout] = []
    total_valid_rows = 0
    total_invalid_rows = 0

    current_behavior: str | None = None
    current_onset_hours: float | None = None
    current_valid_row_count = 0
    current_start_file_index: int | None = None
    current_end_file_index: int | None = None

    previous_file_index: int | None = None
    previous_row_position: int | None = None
    previous_file_row_count: int | None = None

    for file_index, path in input_files:
        n_rows, row_positions, labels = mi.read_and_classify(path)
        total_valid_rows += len(labels)
        total_invalid_rows += n_rows - len(labels)

        if n_rows == 0 or len(labels) == 0:
            _close_bout(
                completed,
                current_behavior,
                current_onset_hours,
                current_valid_row_count,
                current_start_file_index,
                current_end_file_index,
            )
            current_behavior = None
            current_onset_hours = None
            current_valid_row_count = 0
            current_start_file_index = None
            current_end_file_index = None
            previous_file_index = file_index
            previous_row_position = None
            previous_file_row_count = n_rows
            continue

        file_start_hours = (
            file_index - first_file_index
        ) * FILE_DURATION_MINUTES / 60.0
        elapsed_hours = file_start_hours + (
            row_positions.astype(float) / n_rows * FILE_DURATION_MINUTES / 60.0
        )

        for row_position, label, elapsed_hours_for_row in zip(
            row_positions, labels, elapsed_hours
        ):
            row_position_int = int(row_position)
            contiguous = (
                previous_file_index is not None
                and previous_row_position is not None
                and previous_file_row_count is not None
                and (
                    (
                        file_index == previous_file_index
                        and row_position_int == previous_row_position + 1
                    )
                    or (
                        file_index == previous_file_index + 1
                        and previous_row_position == previous_file_row_count - 1
                        and row_position_int == 0
                    )
                )
            )
            if not contiguous:
                _close_bout(
                    completed,
                    current_behavior,
                    current_onset_hours,
                    current_valid_row_count,
                    current_start_file_index,
                    current_end_file_index,
                )
                current_behavior = None
                current_onset_hours = None
                current_valid_row_count = 0
                current_start_file_index = None
                current_end_file_index = None

            behavior = mi.BEHAVIORS[int(label)]
            if current_behavior != behavior:
                _close_bout(
                    completed,
                    current_behavior,
                    current_onset_hours,
                    current_valid_row_count,
                    current_start_file_index,
                    current_end_file_index,
                )
                current_behavior = behavior
                current_onset_hours = float(elapsed_hours_for_row)
                current_valid_row_count = 1
                current_start_file_index = file_index
                current_end_file_index = file_index
            else:
                current_valid_row_count += 1
                current_end_file_index = file_index

            previous_file_index = file_index
            previous_row_position = row_position_int
            previous_file_row_count = n_rows

    _close_bout(
        completed,
        current_behavior,
        current_onset_hours,
        current_valid_row_count,
        current_start_file_index,
        current_end_file_index,
    )

    return ExtractedData(
        bouts=tuple(completed),
        total_valid_rows=total_valid_rows,
        total_invalid_rows=total_invalid_rows,
    )


def _normalize_within_cycle_time(
    elapsed_hours: float,
    cycle_start_hours: float,
    computational_frp_hours: float,
) -> float:
    """Return within-cycle time with only floating-point boundary protection."""

    within_cycle_hours = elapsed_hours - cycle_start_hours
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, computational_frp_hours)
    if within_cycle_hours < -tolerance or within_cycle_hours >= computational_frp_hours + tolerance:
        raise ValueError(
            "Event falls outside its assigned complete-cycle duration: "
            f"within={within_cycle_hours}, FRP={computational_frp_hours}"
        )
    return float(
        min(
            max(within_cycle_hours, 0.0),
            np.nextafter(computational_frp_hours, 0.0),
        )
    )


def assign_complete_cycle_events(
    bouts: Sequence[Bout],
    complete_cycle_specs: Sequence[tuple[int, float, float]],
    computational_frp_hours: float,
) -> dict[int, list[scorer.Event]]:
    """Assign raw non-rest bout onsets to half-open complete-cycle intervals."""

    events_by_cycle: dict[int, list[scorer.Event]] = {}
    for cycle_index, cycle_start_hours, cycle_end_hours in complete_cycle_specs:
        events: list[scorer.Event] = []
        for bout in bouts:
            if bout.behavior not in NONREST_BEHAVIORS:
                continue
            if not (cycle_start_hours <= bout.onset_elapsed_hours < cycle_end_hours):
                continue
            within_cycle_hours = _normalize_within_cycle_time(
                bout.onset_elapsed_hours,
                cycle_start_hours,
                computational_frp_hours,
            )
            phase_hours = mi.CT_HOURS * within_cycle_hours / computational_frp_hours
            events.append((bout.behavior, float(phase_hours)))
        events_by_cycle[cycle_index] = events
    return events_by_cycle


def median_nearest_same_label_distance_minutes(
    events_a: Sequence[scorer.Event],
    events_b: Sequence[scorer.Event],
) -> float:
    """Summarize same-label candidate proximity symmetrically across a pair."""

    phases_by_behavior_a: dict[str, list[float]] = defaultdict(list)
    phases_by_behavior_b: dict[str, list[float]] = defaultdict(list)
    for behavior, phase in events_a:
        phases_by_behavior_a[behavior].append(phase)
    for behavior, phase in events_b:
        phases_by_behavior_b[behavior].append(phase)

    distances_hours: list[float] = []
    for behavior, phase in events_a:
        candidates = phases_by_behavior_b.get(behavior, [])
        if candidates:
            distances_hours.append(
                min(scorer.circular_phase_distance(phase, candidate) for candidate in candidates)
            )
    for behavior, phase in events_b:
        candidates = phases_by_behavior_a.get(behavior, [])
        if candidates:
            distances_hours.append(
                min(scorer.circular_phase_distance(phase, candidate) for candidate in candidates)
            )
    if not distances_hours:
        return float("nan")
    return float(np.median(np.asarray(distances_hours, dtype=float)) * 60.0)


def _validate_pair_score(
    result: scorer.PairScore,
    events_a: Sequence[scorer.Event],
    events_b: Sequence[scorer.Event],
) -> None:
    if len(result.origin_scores) != 3:
        raise ValueError("Practical scorer did not return exactly three origin scores")
    if tuple(scorer.ORIGINS_HOURS) != (0.0, 8.0, 16.0):
        raise ValueError("Practical scorer origins differ from the frozen 0/8/16 h set")
    if not np.all(np.isfinite(np.asarray(result.origin_scores, dtype=float))):
        raise ValueError("COMBA origin score is non-finite")
    if not np.isfinite(result.score):
        raise ValueError("COMBA pair score is non-finite")
    if not all(0.0 <= score <= 1.0 + 1e-12 for score in result.origin_scores):
        raise ValueError(f"COMBA origin score is outside [0, 1]: {result.origin_scores}")
    if not 0.0 <= result.score <= 1.0 + 1e-12:
        raise ValueError(f"COMBA pair score is outside [0, 1]: {result.score}")
    expected_median = float(np.median(np.asarray(result.origin_scores, dtype=float)))
    if not math.isclose(result.score, expected_median, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Pair COMBA score is not the median of the three origins")
    expected_denominator = scorer.event_mass(events_a) + scorer.event_mass(events_b)
    if not np.allclose(
        np.asarray(result.origin_denominators, dtype=float),
        expected_denominator,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Origin denominators do not match the two-cycle event mass")


def _cycle_qc_frame(
    animal_id: str,
    complete_cycle_specs: Sequence[tuple[int, float, float]],
    events_by_cycle: dict[int, list[scorer.Event]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cycle_index, _, _ in complete_cycle_specs:
        counts = Counter(behavior for behavior, _ in events_by_cycle[cycle_index])
        row: dict[str, object] = {
            "Animal": animal_id,
            "Cycle": cycle_index,
            "Total_nonrest_bout_count": sum(counts.values()),
        }
        row.update(
            {
                f"Bouts_{behavior}": counts.get(behavior, 0)
                for behavior in NONREST_BEHAVIORS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_animal(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run one animal and write the three requested CSV outputs."""

    started = time.perf_counter()
    input_dir = input_dir.resolve()
    input_files, missing_indices = mi.discover_input_files(input_dir)
    animal_match = mi.INPUT_FILENAME_PATTERN.fullmatch(input_files[0][1].name)
    if animal_match is None:
        raise AssertionError("Input filename did not match the discovered-file pattern")
    animal_id = animal_match.group("animal")

    (
        reported_frp_hours,
        computational_frp_hours,
        start_ct,
        complete_cycle_specs,
        frp_phase_output_dir,
    ) = mi.load_frp_phase_solution(input_dir)
    extracted = extract_wta_bouts(input_files)
    events_by_cycle = assign_complete_cycle_events(
        extracted.bouts,
        complete_cycle_specs,
        computational_frp_hours,
    )

    cycle_qc = _cycle_qc_frame(animal_id, complete_cycle_specs, events_by_cycle)
    pair_rows: list[dict[str, object]] = []
    pair_scores: list[float] = []
    pair_candidate_distances: list[float] = []
    pair_runtime_seconds = 0.0

    for (cycle_a, _, _), (cycle_b, _, _) in zip(
        complete_cycle_specs, complete_cycle_specs[1:]
    ):
        events_a = events_by_cycle[cycle_a]
        events_b = events_by_cycle[cycle_b]
        result = scorer.practical_score_pair(
            events_a,
            events_b,
            measure_runtime=True,
        )
        _validate_pair_score(result, events_a, events_b)
        candidate_distance = median_nearest_same_label_distance_minutes(
            events_a, events_b
        )
        if not np.isfinite(candidate_distance):
            raise ValueError(
                f"No same-label candidate exists for adjacent pair {cycle_a}->{cycle_b}"
            )
        pair_runtime_seconds += sum(result.origin_runtime_seconds)
        pair_scores.append(float(result.score))
        pair_candidate_distances.append(candidate_distance)
        pair_rows.append(
            {
                "Animal": animal_id,
                "Cycle_pair": f"{cycle_a}->{cycle_b}",
                "S_0h": result.origin_scores[0],
                "S_8h": result.origin_scores[1],
                "S_16h": result.origin_scores[2],
                "Pair_COMBA": result.score,
                "Bouts_cycle_A": len(events_a),
                "Bouts_cycle_B": len(events_b),
                "Median_nearest_same_label_distance_min": candidate_distance,
            }
        )

    if len(pair_scores) != len(complete_cycle_specs) - 1:
        raise ValueError("Adjacent complete-cycle pair count is inconsistent")
    if not pair_scores:
        raise ValueError("No adjacent complete-cycle pairs were available")
    if not extracted.bouts:
        raise ValueError("No WTA bouts were extracted")
    total_nonrest_bouts = int(cycle_qc["Total_nonrest_bout_count"].sum())
    if total_nonrest_bouts == 0:
        raise ValueError("No non-rest bout onsets were found in complete cycles")

    mean_raw_comba = float(np.mean(np.asarray(pair_scores, dtype=float)))
    median_candidate_distance = float(
        np.median(np.asarray(pair_candidate_distances, dtype=float))
    )
    mean_nonrest_bouts_per_cycle = total_nonrest_bouts / len(complete_cycle_specs)
    elapsed_seconds = time.perf_counter() - started

    summary = pd.DataFrame(
        [
            {
                "Animal": animal_id,
                "Reported_FRP_h": reported_frp_hours,
                "Computational_FRP_h": computational_frp_hours,
                "Start_CT": start_ct,
                "Complete_cycle_indices": ",".join(
                    str(cycle_index) for cycle_index, _, _ in complete_cycle_specs
                ),
                "Full_cycles": len(complete_cycle_specs),
                "Adjacent_pairs": len(pair_scores),
                "Mean_raw_COMBA": mean_raw_comba,
                "Median_nearest_same_label_distance_min": median_candidate_distance,
                "Mean_nonrest_bouts_per_cycle": mean_nonrest_bouts_per_cycle,
            }
        ]
    )

    output_dir = input_dir / OUTPUT_DIRECTORY_NAME
    output_dir.mkdir(exist_ok=True)
    summary.to_csv(output_dir / "animal_comba_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "comba_pair_scores.csv", index=False)
    cycle_qc.to_csv(output_dir / "comba_cycle_qc.csv", index=False)

    full_cycle_text = ", ".join(
        f"{cycle_index}: {start:.12g}-{end:.12g} h"
        for cycle_index, start, end in complete_cycle_specs
    )
    missing_text = ", ".join(str(index) for index in missing_indices) or "none"
    run_summary = "\n".join(
        [
            "Raw practical COMBA run",
            f"Animal: {animal_id}",
            f"Input directory: {input_dir}",
            f"Source files: {len(input_files)} ({input_files[0][0]}-{input_files[-1][0]})",
            f"Missing source-file indices: {missing_text}",
            f"FRP solution directory: {frp_phase_output_dir}",
            f"Reported FRP (h): {reported_frp_hours:.12g}",
            f"Computational FRP (h): {computational_frp_hours:.12g}",
            f"Start CT: {start_ct:.12g}",
            f"Complete cycles: {full_cycle_text}",
            f"Valid rows: {extracted.total_valid_rows}",
            f"Invalid rows: {extracted.total_invalid_rows}",
            f"Extracted WTA bouts: {len(extracted.bouts)}",
            f"Complete-cycle non-rest bout onsets: {total_nonrest_bouts}",
            f"Adjacent pairs: {len(pair_scores)}",
            f"Mean raw COMBA: {mean_raw_comba:.12g}",
            f"Median nearest same-label distance (min): {median_candidate_distance:.12g}",
            f"Mean non-rest bouts per complete cycle: {mean_nonrest_bouts_per_cycle:.12g}",
            "Origins: 0 h, 8 h, 16 h",
            "Sigma: 3 min; match cutoff: 30 min; square-root count weighting",
            "Scorer: practical_comba_validation.practical_score_pair",
            "Null correction: none",
            "Bridge/flicker correction: none",
            f"Scorer runtime (s): {pair_runtime_seconds:.6f}",
            f"Total runtime (s): {elapsed_seconds:.6f}",
            "QC: PASS",
        ]
    )
    (output_dir / "run_summary.txt").write_text(run_summary + "\n", encoding="utf-8")
    return summary, pd.DataFrame(pair_rows), cycle_qc, run_summary


def main() -> None:
    args = parse_args()
    summary, pair_scores, cycle_qc, _ = run_animal(args.input_dir)
    print(summary.to_string(index=False))
    print(pair_scores.to_string(index=False))
    print(cycle_qc.to_string(index=False))


if __name__ == "__main__":
    main()
