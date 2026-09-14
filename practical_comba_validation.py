"""Validate the frozen three-origin median sparse COMBA candidate.

The existing ``synthetic_comba_validation.py`` contains the synthetic cases
and the earlier provisional scorer.  This companion script reuses only the
case construction and marked-block null generator.  It implements the
requested practical scorer separately so that the existing validation remains
unchanged.

No real mouse data are read by this script.  Candidate matching is performed
with sparse same-label edges and a Fenwick tree; no dense event-by-event matrix
is constructed.
"""

from __future__ import annotations

import argparse
import math
import time
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import synthetic_comba_validation as existing


PERIOD_HOURS = 24.0
ORIGINS_HOURS = (0.0, 8.0, 16.0)
SIGMA_MINUTES = 3.0
MATCH_CUTOFF_MINUTES = 30.0
MATCH_CUTOFF_HOURS = MATCH_CUTOFF_MINUTES / 60.0
NULL_REPLICATES = 100
MARKED_BLOCK_SIZE = 3
DEFAULT_SEED = existing.DEFAULT_SEED
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Practical_COMBA_Validation"
)
BENCHMARK_EVENT_COUNTS = (10_000, 12_000, 14_000)
BENCHMARK_COMPOSITIONS = ("balanced", "locomotion_heavy")
ROTATION_OFFSETS_HOURS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0)

Event = tuple[str, float]


@dataclass(frozen=True)
class OrderedEvent:
    """An event with original phase retained and origin-relative order added."""

    behavior: str
    phase: float
    relative_phase: float
    original_index: int


@dataclass
class PairScore:
    score: float
    origin_scores: tuple[float, float, float]
    origin_matched_credit: tuple[float, float, float]
    origin_denominators: tuple[float, float, float]
    candidate_edges: tuple[int, int, int]
    origin_runtime_seconds: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the three-origin median sparse COMBA candidate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dedicated synthetic output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic synthetic seed.",
    )
    parser.add_argument(
        "--null-reps",
        type=int,
        default=NULL_REPLICATES,
        help="Marked-block null replicates per targeted case (default: 100).",
    )
    return parser.parse_args()


def circular_phase_distance(phase_a: float, phase_b: float) -> float:
    """Return shortest circular phase distance in hours."""

    absolute = abs((float(phase_a) % PERIOD_HOURS) - (float(phase_b) % PERIOD_HOURS))
    return min(absolute, PERIOD_HOURS - absolute)


def gaussian_credit(delta_hours: float) -> float:
    """Return the frozen single-Gaussian timing credit."""

    sigma_hours = SIGMA_MINUTES / 60.0
    return math.exp(-((float(delta_hours) ** 2) / (2.0 * sigma_hours**2)))


def order_events(events: Sequence[Event], origin_hours: float) -> list[OrderedEvent]:
    """Order by phase after the chosen origin without changing original phases."""

    ordered = [
        OrderedEvent(
            behavior=str(behavior),
            phase=float(phase) % PERIOD_HOURS,
            relative_phase=(float(phase) - origin_hours) % PERIOD_HOURS,
            original_index=index,
        )
        for index, (behavior, phase) in enumerate(events)
    ]
    ordered.sort(key=lambda event: (event.relative_phase, event.original_index))
    return ordered


def event_mass(events: Sequence[Event]) -> float:
    """Return total event mass under a cycle's square-root count weights."""

    counts = Counter(str(behavior) for behavior, _ in events)
    return float(
        sum(1.0 / math.sqrt(counts[str(behavior)]) for behavior, _ in events)
    )


def _fenwick_query(tree: list[float], prefix_length: int) -> float:
    """Maximum value over zero-based positions [0, prefix_length)."""

    best = 0.0
    index = prefix_length
    while index > 0:
        value = tree[index]
        if value > best:
            best = value
        index -= index & -index
    return best


def _fenwick_update(tree: list[float], position: int, value: float) -> None:
    """Apply a max update at one zero-based B position."""

    index = position + 1
    while index < len(tree):
        if value > tree[index]:
            tree[index] = value
        index += index & -index


def _indexed_b_events(
    ordered_b: Sequence[OrderedEvent],
) -> dict[str, tuple[list[float], list[int]]]:
    """Index B phases by label, duplicating the phase circle for lookup only."""

    by_behavior: dict[str, list[tuple[float, int]]] = {}
    for index, event in enumerate(ordered_b):
        by_behavior.setdefault(event.behavior, []).append(
            (event.relative_phase, index)
        )

    indexed: dict[str, tuple[list[float], list[int]]] = {}
    for behavior, values in by_behavior.items():
        relative_phases = [value[0] for value in values]
        indices = [value[1] for value in values]
        indexed[behavior] = (
            [phase - PERIOD_HOURS for phase in relative_phases]
            + relative_phases
            + [phase + PERIOD_HOURS for phase in relative_phases],
            indices + indices + indices,
        )
    return indexed


def sparse_increasing_chain_match(
    ordered_a: Sequence[OrderedEvent],
    ordered_b: Sequence[OrderedEvent],
) -> tuple[float, int]:
    """Return maximum sparse increasing-chain credit and candidate-edge count.

    All candidate edges for one A event are queried before any of that A
    event's updates are applied.  Queries through ``j - 1`` enforce strict B
    ordering, and delayed updates enforce one-to-one use of A events.
    """

    if not ordered_a or not ordered_b:
        return 0.0, 0

    counts_a = Counter(event.behavior for event in ordered_a)
    counts_b = Counter(event.behavior for event in ordered_b)
    b_index = _indexed_b_events(ordered_b)
    tree = [0.0] * (len(ordered_b) + 1)
    best_total = 0.0
    candidate_edge_count = 0

    for event_a in ordered_a:
        phase_values, b_indices = b_index.get(event_a.behavior, ([], []))
        if not phase_values:
            continue

        lower = bisect_left(
            phase_values,
            event_a.relative_phase - MATCH_CUTOFF_HOURS,
        )
        upper = bisect_right(
            phase_values,
            event_a.relative_phase + MATCH_CUTOFF_HOURS,
        )
        weight_scale = min(
            1.0 / math.sqrt(counts_a[event_a.behavior]),
            1.0 / math.sqrt(counts_b[event_a.behavior]),
        )
        pending_updates: list[tuple[int, float]] = []

        # Query all edges first.  No update from this A event is visible here.
        for lookup_index in range(lower, upper):
            b_position = b_indices[lookup_index]
            b_event = ordered_b[b_position]
            delta = circular_phase_distance(event_a.phase, b_event.phase)
            if delta > MATCH_CUTOFF_HOURS:
                continue
            candidate_edge_count += 1
            edge_weight = weight_scale * gaussian_credit(delta)
            prior_value = _fenwick_query(tree, b_position)
            candidate_value = prior_value + edge_weight
            pending_updates.append((b_position, candidate_value))
            if candidate_value > best_total:
                best_total = candidate_value

        # Only now can this A event contribute to later A events.
        for b_position, candidate_value in pending_updates:
            _fenwick_update(tree, b_position, candidate_value)

    return float(best_total), candidate_edge_count


def practical_score_pair(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    *,
    measure_runtime: bool = False,
) -> PairScore:
    """Score one pair at exactly the three frozen origins."""

    mass_a = event_mass(events_a)
    mass_b = event_mass(events_b)
    denominator = mass_a + mass_b
    origin_scores: list[float] = []
    origin_credits: list[float] = []
    origin_denominators: list[float] = []
    edge_counts: list[int] = []
    runtimes: list[float] = []

    for origin_hours in ORIGINS_HOURS:
        started = time.perf_counter()
        ordered_a = order_events(events_a, origin_hours)
        ordered_b = order_events(events_b, origin_hours)
        matched_credit, edge_count = sparse_increasing_chain_match(
            ordered_a, ordered_b
        )
        elapsed = time.perf_counter() - started if measure_runtime else 0.0
        score = 0.0 if denominator == 0.0 else 2.0 * matched_credit / denominator
        origin_scores.append(float(score))
        origin_credits.append(float(matched_credit))
        origin_denominators.append(float(denominator))
        edge_counts.append(int(edge_count))
        runtimes.append(float(elapsed))

    if not np.allclose(origin_denominators, denominator, rtol=0.0, atol=1e-12):
        raise AssertionError("Origin-specific denominators differ")
    if len(origin_scores) != 3:
        raise AssertionError("The practical score must retain exactly three origins")

    return PairScore(
        score=float(np.median(np.asarray(origin_scores, dtype=float))),
        origin_scores=tuple(origin_scores),
        origin_matched_credit=tuple(origin_credits),
        origin_denominators=tuple(origin_denominators),
        candidate_edges=tuple(edge_counts),
        origin_runtime_seconds=tuple(runtimes),
    )


def run_implementation_invariants() -> bool:
    """Check the fixed-origin and sparse-matching contracts on tiny examples."""

    assert ORIGINS_HOURS == (0.0, 8.0, 16.0)
    assert math.isclose(circular_phase_distance(23.99, 0.01), 0.02)
    assert gaussian_credit(0.0) == 1.0

    original = [("eating", 23.99), ("drinking", 0.01)]
    ordered = order_events(original, 0.0)
    assert [event.phase for event in ordered] == [0.01, 23.99]
    assert sorted(event.phase for event in ordered) == sorted(phase for _, phase in original)

    # A single A event must not chain to two B events.  Delayed updates make
    # both queries see the prior tree, rather than the current A update.
    one_a = [OrderedEvent("eating", 0.0, 0.0, 0)]
    two_b = [
        OrderedEvent("eating", 0.0, 0.0, 0),
        OrderedEvent("eating", 0.1, 0.1, 1),
    ]
    credit, _ = sparse_increasing_chain_match(one_a, two_b)
    assert credit <= 1.0 / math.sqrt(2.0) + 1e-12

    # Reverse label order can match at most one event under strict j-1
    # queries, even though both labels and phases are individually eligible.
    reverse_a = [
        OrderedEvent("eating", 0.0, 0.0, 0),
        OrderedEvent("drinking", 1.0, 1.0, 1),
    ]
    reverse_b = [
        OrderedEvent("drinking", 1.0, 1.0, 0),
        OrderedEvent("eating", 0.0, 0.0, 1),
    ]
    reverse_credit, _ = sparse_increasing_chain_match(reverse_a, reverse_b)
    assert reverse_credit <= 1.0 + 1e-12

    # Same-label restriction and the strict 30-minute edge cutoff.
    no_label_credit, no_label_edges = sparse_increasing_chain_match(
        [OrderedEvent("eating", 0.0, 0.0, 0)],
        [OrderedEvent("drinking", 0.0, 0.0, 0)],
    )
    assert no_label_credit == 0.0 and no_label_edges == 0
    cutoff_credit, cutoff_edges = sparse_increasing_chain_match(
        [OrderedEvent("eating", 0.0, 0.0, 0)],
        [OrderedEvent("eating", 0.51, 0.51, 0)],
    )
    assert cutoff_credit == 0.0 and cutoff_edges == 0

    pair = practical_score_pair(
        [("eating", 0.0), ("drinking", 12.0)],
        [("eating", 0.0), ("drinking", 12.0)],
    )
    assert np.allclose(pair.origin_denominators, pair.origin_denominators[0])
    assert math.isclose(pair.score, float(np.median(pair.origin_scores)))
    return True


def evaluate_case(case: dict) -> dict:
    """Evaluate adjacent pairs and report mean origin-specific scores."""

    cycles = existing.clone_cycles(case["cycles"])
    missing = case["missing_intervals"]
    pair_results: list[dict] = []
    for index in range(len(cycles) - 1):
        events_a, events_b = existing.eligible_pair(
            cycles[index],
            cycles[index + 1],
            missing[index],
            missing[index + 1],
            SIGMA_MINUTES,
        )
        pair = practical_score_pair(events_a, events_b)
        pair_results.append(
            {
                "cycle_a": index,
                "cycle_b": index + 1,
                "events_a": list(events_a),
                "events_b": list(events_b),
                "score": pair.score,
                "origin_scores": pair.origin_scores,
                "matched_credit": float(np.mean(pair.origin_matched_credit)),
                "denominator": pair.origin_denominators[0],
                "eligible_events": len(events_a) + len(events_b),
                "candidate_edges": pair.candidate_edges,
            }
        )

    if not pair_results:
        raise ValueError(f"Case has no adjacent pair: {case['case_id']}")

    origin_means = tuple(
        float(np.mean([pair["origin_scores"][origin] for pair in pair_results]))
        for origin in range(3)
    )
    lag_scores: dict[int, float] = {}
    for lag in (2, 3):
        lag_pairs = []
        for index in range(len(cycles) - lag):
            events_a, events_b = existing.eligible_pair(
                cycles[index],
                cycles[index + lag],
                missing[index],
                missing[index + lag],
                SIGMA_MINUTES,
            )
            lag_pairs.append(practical_score_pair(events_a, events_b).score)
        if lag_pairs:
            lag_scores[lag] = float(np.mean(lag_pairs))

    return {
        "score": float(np.mean([pair["score"] for pair in pair_results])),
        "origin_scores": origin_means,
        "matched_credit": float(np.mean([pair["matched_credit"] for pair in pair_results])),
        "total_mass": float(np.mean([pair["denominator"] for pair in pair_results])),
        "eligible_events": float(np.mean([pair["eligible_events"] for pair in pair_results])),
        "lag_scores": lag_scores,
        "pair_results": pair_results,
    }


def _score_from_events(events_a: Sequence[Event], events_b: Sequence[Event]) -> dict:
    pair = practical_score_pair(events_a, events_b)
    return {
        "score": pair.score,
        "origin_scores": pair.origin_scores,
        "candidate_edges": pair.candidate_edges,
    }


def _check_truth_cases(
    cases_by_id: dict[str, dict],
    results: dict[str, dict],
    null_results: dict[str, dict],
) -> dict[str, tuple[str, str]]:
    """Apply fixed qualitative checks to the existing synthetic cases."""

    status: dict[str, tuple[str, str]] = {}

    def add(case_id: str, passed: bool, reason: str) -> None:
        status[case_id] = ("pass" if passed else "fail", reason)

    baseline = results["case_01_perfect_repeat"]["score"]
    add(
        "case_01_perfect_repeat",
        baseline >= 0.95,
        f"score={baseline:.3f}; expected very high recurrence",
    )

    for case_id in (
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    ):
        if case_id not in null_results:
            add(case_id, False, "targeted marked-block null result missing")
    if "case_02_randomized_behavior_identity" in null_results:
        corrected = null_results["case_02_randomized_behavior_identity"]["corrected_score"]
        add(
            "case_02_randomized_behavior_identity",
            abs(corrected) <= 0.10,
            f"marked-block corrected score={corrected:.3f}; expected near chance",
        )

    reference_case = cases_by_id["case_03_same_pooled_phase_different_realizations"]
    reference = evaluate_case(
        {
            **reference_case,
            "cycles": reference_case["reference_cycles"],
            "missing_intervals": [[] for _ in reference_case["reference_cycles"]],
        }
    )
    rearranged = results["case_03_same_pooled_phase_different_realizations"]["score"]
    reduction = reference["score"] - rearranged
    pooled_equal = existing.pooled_phase_equality_by_behavior(
        reference_case["reference_cycles"], reference_case["cycles"]
    )
    add(
        "case_03_same_pooled_phase_different_realizations",
        all(pooled_equal.values())
        and reference["score"] >= 0.95
        and rearranged <= 0.70
        and reduction >= 0.30,
        f"pooled_equal={all(pooled_equal.values())}; reference={reference['score']:.3f}; rearranged={rearranged:.3f}; reduction={reduction:.3f}",
    )

    checks = {
        "case_04_30_second_jitter": (0.95, "30-second jitter should remain high"),
        "case_05_1_minute_jitter": (0.85, "1-minute jitter should remain high"),
        "case_08_brief_rearing_repeated": (0.95, "reproducible brief behavior should remain detectable"),
        "case_11_stable_composition_change": (0.95, "stable composition change should recur"),
        "case_12_reproducibly_increased_bout_number": (0.95, "reproducible extra bouts should recur"),
        "case_18_rare_behavior_absent": (0.95, "remaining repertoire should recur"),
        "case_19_frequency_imbalance": (0.95, "frequency imbalance should not erase recurrence"),
        "case_20_one_rare_event_every_cycle": (0.95, "repeated rare behavior should recur"),
        "case_24_rest_changed_only": (0.95, "rest-only changes should not affect non-rest score"),
        "case_26_absolute_phase_animal_A": (0.95, "animal A should recur within animal"),
        "case_26_absolute_phase_animal_B": (0.95, "animal B should recur within animal"),
        "case_29_cycle_boundary": (0.95, "circular boundary timing should remain high"),
        "case_31_boundary_crossing_order": (0.85, "boundary-crossing order should remain high"),
    }
    for case_id, (threshold, note) in checks.items():
        score = results[case_id]["score"]
        add(case_id, score >= threshold, f"score={score:.3f}; {note}")

    jitter_score = results["case_06_2_to_5_minute_jitter"]["score"]
    add(
        "case_06_2_to_5_minute_jitter",
        0.30 <= jitter_score <= 0.90,
        f"score={jitter_score:.3f}; expected intermediate timing loss",
    )
    displacement_score = results["case_07_10_minute_displacement"]["score"]
    add(
        "case_07_10_minute_displacement",
        displacement_score <= 0.60,
        f"score={displacement_score:.3f}; expected substantial timing loss",
    )

    brief_repeated = results["case_08_brief_rearing_repeated"]["score"]
    brief_random = results["case_09_brief_rearing_randomized"]["score"]
    add(
        "case_09_brief_rearing_randomized",
        brief_repeated - brief_random >= 0.01,
        f"repeated={brief_repeated:.3f}; randomized={brief_random:.3f}; difference={brief_repeated - brief_random:.3f}",
    )
    add(
        "case_10_duration_only_change",
        abs(results["case_10_duration_only_change"]["score"] - baseline) <= 1e-12,
        f"score={results['case_10_duration_only_change']['score']:.3f}; duration is not represented",
    )

    reproducible_extra = results["case_12_reproducibly_increased_bout_number"]["score"]
    random_extra = results["case_13_random_extra_bouts"]["score"]
    add(
        "case_13_random_extra_bouts",
        reproducible_extra - random_extra >= 0.01 and random_extra >= 0.50,
        f"reproducible={reproducible_extra:.3f}; random={random_extra:.3f}",
    )

    for case_id in ("case_14_one_inserted_event", "case_15_one_deleted_event"):
        score = results[case_id]["score"]
        add(case_id, score >= 0.80, f"score={score:.3f}; expected localized modest decrease")

    order_score = results["case_16_adjacent_order_swap"]["score"]
    add(
        "case_16_adjacent_order_swap",
        order_score >= 0.75 and order_score < baseline - 1e-6,
        f"score={order_score:.3f}; perfect-repeat baseline={baseline:.3f}",
    )
    scrambled_score = results["case_17_local_order_scrambling"]["score"]
    add(
        "case_17_local_order_scrambling",
        scrambled_score <= baseline - 0.05,
        f"score={scrambled_score:.3f}; perfect-repeat baseline={baseline:.3f}",
    )

    singleton = results["case_21_one_rare_event_once"]["score"]
    repeated_rare = results["case_20_one_rare_event_every_cycle"]["score"]
    add(
        "case_21_one_rare_event_once",
        abs(repeated_rare - singleton) <= 0.10,
        f"repeated={repeated_rare:.3f}; singleton={singleton:.3f}",
    )

    for case_id in ("case_22_generic_local_syntax", "case_23_stable_circadian_local_variability"):
        if case_id not in null_results:
            continue
        corrected = null_results[case_id]["corrected_score"]
        raw_score = results[case_id]["score"]
        if case_id == "case_22_generic_local_syntax":
            passed = abs(corrected) <= 0.10
            reason = f"raw={raw_score:.3f}; marked-block corrected={corrected:.3f}; expected near chance"
        else:
            passed = raw_score >= 0.75 and corrected >= 0.20
            reason = f"raw={raw_score:.3f}; marked-block corrected={corrected:.3f}; expected above null"
        add(case_id, passed, reason)

    missing_gap = results["case_27_missing_data_gap"]
    add(
        "case_27_missing_data_gap",
        missing_gap["score"] >= 0.95
        and missing_gap["eligible_events"] < results["case_01_perfect_repeat"]["eligible_events"],
        f"score={missing_gap['score']:.3f}; eligible_events={missing_gap['eligible_events']:.1f}; expected fewer eligible events",
    )
    missing_rare = results["case_28_gap_covers_rare_event"]
    add(
        "case_28_gap_covers_rare_event",
        missing_rare["score"] >= 0.95
        and missing_rare["eligible_events"] < results["case_20_one_rare_event_every_cycle"]["eligible_events"],
        f"score={missing_rare['score']:.3f}; eligible_events={missing_rare['eligible_events']:.1f}; expected masked rare event",
    )

    global_shift = results["case_25_cycle_specific_global_phase_shift"]["score"]
    add(
        "case_25_cycle_specific_global_phase_shift",
        global_shift < 0.98,
        f"score={global_shift:.3f}; cycle-specific shifts should reduce recurrence",
    )

    drift = results["case_30_frp_phase_drift"]
    lag_2 = drift["lag_scores"].get(2, float("nan"))
    lag_3 = drift["lag_scores"].get(3, float("nan"))
    add(
        "case_30_frp_phase_drift",
        drift["score"] >= 0.90 and lag_2 < drift["score"] and lag_3 < lag_2,
        f"adjacent={drift['score']:.3f}; lag2={lag_2:.3f}; lag3={lag_3:.3f}",
    )

    return status


def make_truth_rows(
    cases: Sequence[dict],
    results: dict[str, dict],
    statuses: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for case in cases:
        result = results[case["case_id"]]
        rows.append(
            {
                "case_id": case["case_id"],
                "case_name": case["name"],
                "category": case["category"],
                "origin_0h_score": result["origin_scores"][0],
                "origin_8h_score": result["origin_scores"][1],
                "origin_16h_score": result["origin_scores"][2],
                "median_score": result["score"],
                "expected_qualitative_outcome": case["expected_direction"],
                "pass_fail": statuses[case["case_id"]][0],
                "assessment": statuses[case["case_id"]][1],
            }
        )
    return pd.DataFrame(rows)


def run_marked_block_null(
    case: dict,
    observed: dict,
    n_replicates: int,
    seed: int,
) -> dict:
    """Run only the existing max-block-size-3 marked-block null."""

    rng = np.random.default_rng(
        existing.stable_seed(
            "practical_marked_block_3", case["case_id"], seed, n_replicates
        )
    )
    cycles = existing.clone_cycles(case["cycles"])
    missing = case["missing_intervals"]
    if any(missing_intervals for missing_intervals in missing):
        raise ValueError(
            "Targeted coherent marked-block null cases must not require "
            "pair-specific masking."
        )
    replicate_scores = np.empty(n_replicates, dtype=float)
    for replicate in range(n_replicates):
        # Generate one coherent null recording, then reuse each exact cycle in
        # both neighboring adjacent-pair comparisons.
        null_cycles = [
            existing.generate_null_cycle(
                cycle, "marked_block", rng, MARKED_BLOCK_SIZE
            )
            for cycle in cycles
        ]
        pair_inputs = [
            (null_cycles[index], null_cycles[index + 1])
            for index in range(len(null_cycles) - 1)
        ]
        if not all(
            pair_inputs[index - 1][1] is pair_inputs[index][0]
            for index in range(1, len(pair_inputs))
        ):
            raise AssertionError("Coherent null cycle reuse check failed")
        pair_scores = [
            practical_score_pair(null_a, null_b).score
            for null_a, null_b in pair_inputs
        ]
        replicate_scores[replicate] = float(np.mean(pair_scores))

    null_mean = float(np.mean(replicate_scores))
    null_sd = float(np.std(replicate_scores, ddof=1))
    corrected = existing.chance_correct(observed["score"], null_mean)
    if case["case_id"] == "case_01_perfect_repeat":
        passed = corrected >= 0.70
    elif case["case_id"] in {
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
    }:
        passed = abs(corrected) <= 0.10
    else:
        passed = corrected >= 0.20
    return {
        "case_id": case["case_id"],
        "case_name": case["name"],
        "observed_score": observed["score"],
        "marked_block_size": MARKED_BLOCK_SIZE,
        "n_replicates": n_replicates,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "corrected_score": corrected,
        "pass_fail": "pass" if passed else "fail",
        "coherent_cycle_reuse_pass": True,
    }


def make_boundary_case(target_origin_hours: float) -> tuple[list[Event], list[Event]]:
    """Make two otherwise identical cycles with one 36-second boundary crossing."""

    common = [
        ("eating", target_origin_hours - 6.0),
        ("rearing", target_origin_hours + 0.03),
        ("climbing", target_origin_hours + 3.0),
        ("digging", target_origin_hours + 6.0),
        ("nesting", target_origin_hours + 9.0),
        ("grooming", target_origin_hours + 12.0),
        ("locomotion", target_origin_hours + 15.0),
    ]
    cycle_a = common + [("drinking", target_origin_hours - 0.005)]
    cycle_b = common + [("drinking", target_origin_hours + 0.005)]
    return cycle_a, cycle_b


def run_boundary_jitter_tests() -> tuple[pd.DataFrame, bool]:
    rows: list[dict] = []
    for target_origin in ORIGINS_HOURS:
        cycle_a, cycle_b = make_boundary_case(target_origin)
        result = _score_from_events(cycle_a, cycle_b)
        scores = result["origin_scores"]
        target_index = ORIGINS_HOURS.index(target_origin)
        other_scores = [score for index, score in enumerate(scores) if index != target_index]
        passed = (
            scores[target_index] < 0.98
            and min(other_scores) >= 0.95
            and result["score"] >= 0.95
        )
        rows.append(
            {
                "target_boundary_origin_hours": target_origin,
                "origin_0h_score": scores[0],
                "origin_8h_score": scores[1],
                "origin_16h_score": scores[2],
                "median_score": result["score"],
                "perturbation_seconds": 36.0,
                "expected_qualitative_outcome": "target origin may drop; three-origin median remains high",
                "pass_fail": "pass" if passed else "fail",
            }
        )
    frame = pd.DataFrame(rows)
    return frame, bool((frame["pass_fail"] == "pass").all())


def make_rotation_pair() -> tuple[list[Event], list[Event]]:
    """Use one fixed boundary-sensitive pair for common phase rotation."""

    # The 36-second crossing is fixed in the pair.  Rotating both cycles
    # moves that same local order conflict through the three fixed cuts, which
    # makes the single-origin and median sensitivities directly inspectable.
    return make_boundary_case(0.0)


def run_rotation_test() -> tuple[pd.DataFrame, dict[str, object]]:
    cycle_a, cycle_b = make_rotation_pair()
    rows: list[dict] = []
    for offset in ROTATION_OFFSETS_HOURS:
        shifted_a = existing.shift_cycle(cycle_a, offset)
        shifted_b = existing.shift_cycle(cycle_b, offset)
        result = _score_from_events(shifted_a, shifted_b)
        rows.append(
            {
                "common_phase_offset_hours": offset,
                "origin_0h_score": result["origin_scores"][0],
                "origin_8h_score": result["origin_scores"][1],
                "origin_16h_score": result["origin_scores"][2],
                "median_score": result["score"],
            }
        )
    frame = pd.DataFrame(rows)
    ranges = {
        "origin_0h_range": float(frame["origin_0h_score"].max() - frame["origin_0h_score"].min()),
        "origin_8h_range": float(frame["origin_8h_score"].max() - frame["origin_8h_score"].min()),
        "origin_16h_range": float(frame["origin_16h_score"].max() - frame["origin_16h_score"].min()),
        "median_range": float(frame["median_score"].max() - frame["median_score"].min()),
    }
    single_origin_max_range = max(
        ranges["origin_0h_range"], ranges["origin_8h_range"], ranges["origin_16h_range"]
    )
    ranges["pass"] = bool(
        single_origin_max_range > 0.0
        and ranges["median_range"] <= 0.75 * single_origin_max_range
    )
    return frame, ranges


def make_benchmark_events(
    event_count: int, composition: str, seed: int
) -> tuple[list[Event], list[Event]]:
    """Generate only the event lists needed for a runtime benchmark."""

    rng = np.random.default_rng(seed)
    if composition == "balanced":
        labels = [existing.BEHAVIORS[index % len(existing.BEHAVIORS)] for index in range(event_count)]
    elif composition == "locomotion_heavy":
        locomotion_count = int(round(0.80 * event_count))
        remainder = event_count - locomotion_count
        labels = ["locomotion"] * locomotion_count
        labels.extend(
            existing.BEHAVIORS[index % (len(existing.BEHAVIORS) - 1)]
            for index in range(remainder)
        )
    else:
        raise ValueError(f"Unknown benchmark composition: {composition}")

    labels_a = list(labels)
    labels_b = list(labels)
    rng.shuffle(labels_a)
    rng.shuffle(labels_b)
    phases_a = rng.uniform(0.0, PERIOD_HOURS, size=event_count)
    phases_b = rng.uniform(0.0, PERIOD_HOURS, size=event_count)
    events_a = sorted(zip(labels_a, phases_a.tolist()), key=lambda event: event[1])
    events_b = sorted(zip(labels_b, phases_b.tolist()), key=lambda event: event[1])
    return events_a, events_b


def run_runtime_benchmarks(seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for composition_index, composition in enumerate(BENCHMARK_COMPOSITIONS):
        for event_count in BENCHMARK_EVENT_COUNTS:
            events_a, events_b = make_benchmark_events(
                event_count,
                composition,
                seed + 100_000 * composition_index + event_count,
            )
            result = practical_score_pair(events_a, events_b, measure_runtime=True)
            rows.append(
                {
                    "event_count_A": event_count,
                    "event_count_B": event_count,
                    "composition_condition": composition,
                    "candidate_edges_origin_0h": result.candidate_edges[0],
                    "candidate_edges_origin_8h": result.candidate_edges[1],
                    "candidate_edges_origin_16h": result.candidate_edges[2],
                    "runtime_origin_0h_seconds": result.origin_runtime_seconds[0],
                    "runtime_origin_8h_seconds": result.origin_runtime_seconds[1],
                    "runtime_origin_16h_seconds": result.origin_runtime_seconds[2],
                    "total_runtime_three_origin_seconds": sum(
                        result.origin_runtime_seconds
                    ),
                    "median_score_diagnostic": result.score,
                    "dense_matrix_constructed": False,
                }
            )
    return pd.DataFrame(rows)


def write_run_summary(
    output_dir: Path,
    cases: Sequence[dict],
    truth: pd.DataFrame,
    nulls: pd.DataFrame,
    boundary: pd.DataFrame,
    rotation_ranges: dict[str, object],
    benchmark: pd.DataFrame | None,
    implementation_pass: bool,
    scientific_pass: bool,
    runtime_pass: bool | None,
    elapsed_seconds: float,
) -> None:
    central = truth[truth["case_id"].str.match(r"case_(0[1-9]|1[0-9]|2[0-3])_")]
    failed_cases = truth.loc[truth["pass_fail"] == "fail", "case_id"].tolist()
    lines = [
        "Three-origin median sparse COMBA candidate validation",
        "====================================================",
        "",
        "scope: existing synthetic COMBA cases only; no real mouse data were read",
        f"origins_hours: {ORIGINS_HOURS}",
        f"sigma_minutes: {SIGMA_MINUTES}",
        f"match_cutoff_minutes: {MATCH_CUTOFF_MINUTES}",
        "event_weight: 1/sqrt(n_behavior_in_cycle)",
        "matching: same-label sparse maximum-weight strictly increasing chain with delayed Fenwick updates",
        "timing: original phases retained; shortest circular distance used",
        f"implementation_invariants_pass: {implementation_pass}",
        "",
        "Truth cases",
        "-----------",
        f"existing_cases_run: {len(cases)}",
        f"central_cases_01_to_23_pass: {bool((central['pass_fail'] == 'pass').all())}",
        f"all_existing_truth_rows_pass: {bool((truth['pass_fail'] == 'pass').all())}",
        f"failed_cases: {failed_cases or 'none'}",
        "",
        "Boundary-jitter tests",
        "---------------------",
    ]
    for row in boundary.to_dict(orient="records"):
        lines.append(
            f"target={row['target_boundary_origin_hours']:.0f}h; "
            f"S0={row['origin_0h_score']:.3f}; S8={row['origin_8h_score']:.3f}; "
            f"S16={row['origin_16h_score']:.3f}; median={row['median_score']:.3f}; "
            f"{row['pass_fail']}"
        )
    lines.extend(
        [
            "",
            "Common phase-rotation sensitivity",
            "----------------------------------",
            f"origin_0h_range: {rotation_ranges['origin_0h_range']:.6f}",
            f"origin_8h_range: {rotation_ranges['origin_8h_range']:.6f}",
            f"origin_16h_range: {rotation_ranges['origin_16h_range']:.6f}",
            f"three_origin_median_range: {rotation_ranges['median_range']:.6f}",
            f"rotation_robustness_pass: {rotation_ranges['pass']}",
            "",
            "Targeted marked-block null",
            "---------------------------",
        ]
    )
    for row in nulls.to_dict(orient="records"):
        lines.append(
            f"{row['case_id']}: observed={row['observed_score']:.3f}; "
            f"null_mean={row['null_mean']:.3f}; null_sd={row['null_sd']:.3f}; "
            f"corrected={row['corrected_score']:.3f}; {row['pass_fail']}"
        )

    lines.extend(["", "Runtime benchmark", "------------------"])
    if benchmark is None:
        lines.append("not run because a scientific acceptance gate failed")
    else:
        for row in benchmark.to_dict(orient="records"):
            lines.append(
                f"{row['composition_condition']} {int(row['event_count_A'])}x{int(row['event_count_B'])}: "
                f"edges=({int(row['candidate_edges_origin_0h'])},"
                f"{int(row['candidate_edges_origin_8h'])},"
                f"{int(row['candidate_edges_origin_16h'])}); "
                f"runtime=({row['runtime_origin_0h_seconds']:.3f},"
                f"{row['runtime_origin_8h_seconds']:.3f},"
                f"{row['runtime_origin_16h_seconds']:.3f}) s; "
                f"total={row['total_runtime_three_origin_seconds']:.3f} s"
            )
        lines.append(f"runtime_gate_pass: {runtime_pass}")

    lines.extend(
        [
            "",
            "Final verdict",
            "-------------",
            f"scientific_gate_pass: {scientific_pass}",
            f"runtime_gate_pass: {runtime_pass}",
            (
                "READY FOR 783E BENCHMARK"
                if scientific_pass and runtime_pass
                else "NOT READY"
            ),
            "",
            "No 783E real-mouse benchmark was run. No production MI/recurrence script was modified.",
            f"runtime_seconds: {elapsed_seconds:.3f}",
        ]
    )
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.null_reps <= 0:
        raise ValueError("--null-reps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    implementation_pass = run_implementation_invariants()
    cases = existing.build_cases(args.seed)
    cases_by_id = {case["case_id"]: case for case in cases}
    results = {case["case_id"]: evaluate_case(case) for case in cases}

    null_rows: list[dict] = []
    for case_id in (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    ):
        null_rows.append(
            run_marked_block_null(
                cases_by_id[case_id],
                results[case_id],
                args.null_reps,
                args.seed,
            )
        )
    nulls = pd.DataFrame(null_rows)
    null_results = {
        row["case_id"]: row for row in null_rows
    }
    statuses = _check_truth_cases(cases_by_id, results, null_results)
    truth = make_truth_rows(cases, results, statuses)

    boundary, boundary_pass = run_boundary_jitter_tests()
    rotation, rotation_ranges = run_rotation_test()
    central_pass = bool(
        (truth[truth["case_id"].str.match(r"case_(0[1-9]|1[0-9]|2[0-3])_")]["pass_fail"] == "pass").all()
    )
    null_pass = bool((nulls["pass_fail"] == "pass").all())
    scientific_pass = central_pass and null_pass and boundary_pass and bool(rotation_ranges["pass"])

    benchmark: pd.DataFrame | None = None
    runtime_pass: bool | None = None
    if scientific_pass:
        benchmark = run_runtime_benchmarks(args.seed)
        # A measured three-origin total below 60 s at 14k events is the
        # practical gate for this validation run, not a production guarantee.
        largest = benchmark[
            benchmark["event_count_A"] == max(BENCHMARK_EVENT_COUNTS)
        ]["total_runtime_three_origin_seconds"]
        runtime_pass = bool(float(largest.max()) < 60.0)

    truth.to_csv(output_dir / "practical_comba_truth_cases.csv", index=False)
    rotation.to_csv(output_dir / "practical_comba_origin_rotation.csv", index=False)
    if benchmark is not None:
        benchmark.to_csv(output_dir / "practical_comba_runtime_benchmark.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "event_count_A",
                "event_count_B",
                "composition_condition",
                "candidate_edges_origin_0h",
                "candidate_edges_origin_8h",
                "candidate_edges_origin_16h",
                "runtime_origin_0h_seconds",
                "runtime_origin_8h_seconds",
                "runtime_origin_16h_seconds",
                "total_runtime_three_origin_seconds",
            ]
        ).to_csv(output_dir / "practical_comba_runtime_benchmark.csv", index=False)
    nulls.to_csv(output_dir / "practical_comba_null_summary.csv", index=False)
    boundary.to_csv(output_dir / "practical_comba_boundary_jitter.csv", index=False)
    write_run_summary(
        output_dir,
        cases,
        truth,
        nulls,
        boundary,
        rotation_ranges,
        benchmark,
        implementation_pass,
        scientific_pass,
        runtime_pass,
        time.perf_counter() - started,
    )

    print(
        f"truth_cases={len(cases)} central_pass={central_pass} "
        f"boundary_pass={boundary_pass} rotation_pass={rotation_ranges['pass']} "
        f"scientific_pass={scientific_pass} runtime_pass={runtime_pass} "
        f"output_dir={output_dir}"
    )
    if not scientific_pass or runtime_pass is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
