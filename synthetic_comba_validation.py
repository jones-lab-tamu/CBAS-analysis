"""Synthetic validation of the provisional event-level COMBA metric.

This script is deliberately isolated from the production recurrence analysis.  It
operates on known non-rest bout events ``(behavior, onset_phase_hours)`` and
writes audit-friendly validation outputs outside the code repository.

The synthetic cases and acceptance language are intentionally explicit.  This
is a validation script, not a production implementation or a general-purpose
behavioral analysis library.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PERIOD_HOURS = 24.0
BEHAVIORS = (
    "eating",
    "drinking",
    "rearing",
    "climbing",
    "digging",
    "nesting",
    "grooming",
    "locomotion",
)
SIGMA_MINUTES = (1, 2, 3, 5, 10)
DELTA_MINUTES = (0.5, 1, 2, 5, 10, 30)
WEIGHTING_RULES = ("raw_event", "equal_total", "sqrt_count")
NULL_REPLICATES = 2_000
N_CYCLES = 6
PRIMARY_SIGMA_MINUTES = 3
PRIMARY_WEIGHTING = "sqrt_count"
DEFAULT_SEED = 20260914
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\COMBA_Synthetic_Validation"
)

Event = tuple[str, float]


def clone_cycles(cycles: Sequence[Sequence[Event]]) -> list[list[Event]]:
    """Copy and normalize a list of synthetic cycles."""

    return [
        sorted(
            [(behavior, float(phase) % PERIOD_HOURS) for behavior, phase in cycle],
            key=lambda event: event[1],
        )
        for cycle in cycles
    ]


def baseline_cycle() -> list[Event]:
    """Return a small transparent cycle containing all eight non-rest behaviors."""

    phase_by_behavior = {
        "eating": (1.25, 8.75, 16.50, 22.25),
        "drinking": (2.10, 11.00, 20.70),
        "rearing": (5.25, 14.75),
        "climbing": (4.10, 9.80, 17.60),
        "digging": (2.75, 12.75),
        "nesting": (6.80, 21.80),
        "grooming": (0.75, 3.55, 7.20, 13.20, 19.10),
        "locomotion": (
            0.20,
            0.55,
            1.00,
            1.80,
            3.20,
            4.50,
            6.00,
            8.20,
            10.50,
            12.10,
            15.00,
            18.40,
            20.00,
            21.40,
        ),
    }
    return sorted(
        [
            (behavior, phase)
            for behavior, phases in phase_by_behavior.items()
            for phase in phases
        ],
        key=lambda event: event[1],
    )


def make_case(
    case_id: str,
    name: str,
    cycles: Sequence[Sequence[Event]],
    expected_direction: str,
    category: str,
    notes: str,
    missing_intervals: Sequence[Sequence[tuple[float, float]]] | None = None,
    **metadata: object,
) -> dict:
    """Create a plain dictionary case record; no state is shared between cases."""

    copied_cycles = clone_cycles(cycles)
    if missing_intervals is None:
        copied_missing = [[] for _ in copied_cycles]
    else:
        copied_missing = [list(intervals) for intervals in missing_intervals]
        if len(copied_missing) != len(copied_cycles):
            raise ValueError(f"Missing-data mask length mismatch for {case_id}")
    return {
        "case_id": case_id,
        "name": name,
        "cycles": copied_cycles,
        "missing_intervals": copied_missing,
        "expected_direction": expected_direction,
        "category": category,
        "notes": notes,
        **metadata,
    }


def shift_cycle(cycle: Sequence[Event], shift_hours: float) -> list[Event]:
    return [
        (behavior, (phase + shift_hours) % PERIOD_HOURS)
        for behavior, phase in cycle
    ]


def remove_behavior(cycle: Sequence[Event], behavior_to_remove: str) -> list[Event]:
    return [
        event for event in cycle if event[0] != behavior_to_remove
    ]


def remove_event_at_phase(
    cycle: Sequence[Event], behavior: str, phase: float
) -> list[Event]:
    """Remove the nearest matching event once."""

    candidates = [
        (index, abs(circular_phase_distance(cycle[index][1], phase)))
        for index, event in enumerate(cycle)
        if event[0] == behavior
    ]
    if not candidates:
        raise ValueError(f"Could not find {behavior} near {phase} h")
    remove_index = min(candidates, key=lambda item: item[1])[0]
    return [event for index, event in enumerate(cycle) if index != remove_index]


def jittered_cycles(
    cycles: Sequence[Sequence[Event]], amplitude_minutes: float, seed: int
) -> list[list[Event]]:
    """Apply independent bounded onset jitter with a fixed seed."""

    rng = np.random.default_rng(seed)
    jittered: list[list[Event]] = []
    amplitude_hours = amplitude_minutes / 60.0
    for cycle in cycles:
        current = [
            (
                behavior,
                (phase + rng.uniform(-amplitude_hours, amplitude_hours))
                % PERIOD_HOURS,
            )
            for behavior, phase in cycle
        ]
        jittered.append(sorted(current, key=lambda event: event[1]))
    return jittered


def randomized_labels_cycles(
    cycles: Sequence[Sequence[Event]], seed: int
) -> list[list[Event]]:
    """Randomize labels independently by cycle while preserving exact counts."""

    rng = np.random.default_rng(seed)
    randomized: list[list[Event]] = []
    for cycle in cycles:
        ordered = sorted(cycle, key=lambda event: event[1])
        labels = np.asarray([event[0] for event in ordered], dtype=object)
        shuffled = labels[rng.permutation(len(labels))]
        randomized.append(
            [(str(label), event[1]) for label, event in zip(shuffled, ordered)]
        )
    return randomized


def add_fixed_events(
    cycles: Sequence[Sequence[Event]], additions: Sequence[Event]
) -> list[list[Event]]:
    return [list(cycle) + list(additions) for cycle in cycles]


def add_random_extra_events(
    cycles: Sequence[Sequence[Event]],
    behavior: str,
    count: int,
    seed: int,
) -> list[list[Event]]:
    rng = np.random.default_rng(seed)
    return [
        list(cycle)
        + [
            (behavior, float(rng.uniform(0.5, 23.5)))
            for _ in range(count)
        ]
        for cycle in cycles
    ]


def local_order_template() -> tuple[list[Event], list[float], list[str]]:
    """Baseline plus a compact eight-event local motif for order tests."""

    # Events are deliberately close enough that timing credit does not mask
    # the order perturbation itself.
    phases = [10.00 + 0.01 * index for index in range(len(BEHAVIORS))]
    labels = list(BEHAVIORS)
    base = [event for event in baseline_cycle() if not 9.90 <= event[1] <= 10.40]
    return base + list(zip(labels, phases)), phases, labels


def replace_local_labels(
    cycle: Sequence[Event], phases: Sequence[float], labels: Sequence[str]
) -> list[Event]:
    phase_set = {round(phase, 6) for phase in phases}
    retained = [event for event in cycle if round(event[1], 6) not in phase_set]
    return retained + list(zip(labels, phases))


def generic_syntax_cycles() -> list[list[Event]]:
    """Keep short three-event motifs while moving their phase locations."""

    motifs = (
        ("eating", "grooming", "locomotion"),
        ("drinking", "locomotion", "grooming"),
        ("rearing", "locomotion", "climbing"),
        ("digging", "nesting", "locomotion"),
        ("climbing", "grooming", "eating"),
        ("nesting", "drinking", "locomotion"),
    )
    centers_by_cycle = (
        (1.0, 4.0, 7.0, 10.0, 14.0, 18.0),
        (2.0, 5.0, 9.0, 12.0, 16.0, 21.0),
        (3.0, 6.0, 11.0, 15.0, 19.0, 22.0),
        (1.5, 4.5, 8.5, 13.5, 17.5, 20.5),
        (2.5, 6.5, 10.5, 14.5, 18.5, 21.5),
        (0.5, 5.5, 9.5, 12.5, 16.5, 20.5),
    )
    cycles: list[list[Event]] = []
    within_motif_offsets = (-0.02, 0.0, 0.02)
    for centers in centers_by_cycle:
        events = []
        for motif, center in zip(motifs, centers):
            events.extend(
                [
                    (label, center + offset)
                    for label, offset in zip(motif, within_motif_offsets)
                ]
            )
        cycles.append(sorted(events, key=lambda event: event[1]))
    return cycles


def pooled_behavior_phases(cycles: Sequence[Sequence[Event]]) -> dict[str, tuple[float, ...]]:
    """Return exact sorted phase multisets pooled separately by behavior."""

    return {
        behavior: tuple(
            sorted(
                float(phase)
                for cycle in cycles
                for event_behavior, phase in cycle
                if event_behavior == behavior
            )
        )
        for behavior in BEHAVIORS
    }


def pooled_phase_equality_by_behavior(
    reference_cycles: Sequence[Sequence[Event]],
    rearranged_cycles: Sequence[Sequence[Event]],
) -> dict[str, bool]:
    """Check exact pooled phase equality for every behavior."""

    reference = pooled_behavior_phases(reference_cycles)
    rearranged = pooled_behavior_phases(rearranged_cycles)
    return {
        behavior: reference[behavior] == rearranged[behavior]
        for behavior in BEHAVIORS
    }


def build_cases(seed: int) -> list[dict]:
    """Build all predefined truth cases with no data-dependent tuning."""

    base = baseline_cycle()
    repeated = [base for _ in range(N_CYCLES)]
    cases: list[dict] = []

    cases.append(
        make_case(
            "case_01_perfect_repeat",
            "Perfect repeat",
            repeated,
            "very high recurrence",
            "core",
            "Identical non-rest event sequence in every cycle.",
        )
    )

    cases.append(
        make_case(
            "case_02_randomized_behavior_identity",
            "Preserved activity, randomized behavior identities",
            randomized_labels_cycles(repeated, seed + 2),
            "corrected score near chance",
            "null-critical",
            "Event slots and exact per-cycle behavior counts are preserved; labels are independently permuted.",
        )
    )

    critical_c_phase_pairs = {
        behavior: (1.00 + 2.50 * index, 1.50 + 2.50 * index)
        for index, behavior in enumerate(BEHAVIORS)
    }
    critical_c_reference_cycle = [
        (behavior, phase)
        for behavior, phases in critical_c_phase_pairs.items()
        for phase in phases
    ]
    critical_c_reference = [
        critical_c_reference_cycle for _ in range(N_CYCLES)
    ]
    critical_c_rearranged = []
    for cycle_index in range(N_CYCLES):
        selected_phase_index = cycle_index % 2
        critical_c_rearranged.append(
            [
                (behavior, phases[selected_phase_index])
                for behavior, phases in critical_c_phase_pairs.items()
                for _ in range(2)
            ]
        )
    critical_c_invariant = pooled_phase_equality_by_behavior(
        critical_c_reference,
        critical_c_rearranged,
    )
    if not all(critical_c_invariant.values()):
        raise AssertionError(
            "Stronger Critical C pooled phase invariant failed during construction."
        )
    cases.append(
        make_case(
            "case_03_same_pooled_phase_different_realizations",
            "Exact pooled phase profile, rearranged cycle realizations",
            critical_c_rearranged,
            "high reference recurrence, substantially lower rearranged recurrence",
            "core",
            "The reference repeats two behavior-specific phase tokens per cycle; the rearranged version alternates both tokens into separate cycles. Pooled per-behavior phase multisets are checked exactly.",
            reference_cycles=critical_c_reference,
        )
    )

    cases.append(
        make_case(
            "case_04_30_second_jitter",
            "30-second jitter",
            jittered_cycles(repeated, 0.5, seed + 4),
            "almost no loss",
            "timing",
            "Independent onset jitter bounded by 30 seconds.",
        )
    )
    cases.append(
        make_case(
            "case_05_1_minute_jitter",
            "1-minute jitter",
            jittered_cycles(repeated, 1.0, seed + 5),
            "high recurrence",
            "timing",
            "Independent onset jitter bounded by 1 minute.",
        )
    )
    cases.append(
        make_case(
            "case_06_2_to_5_minute_jitter",
            "2–5-minute jitter",
            jittered_cycles(repeated, 4.0, seed + 6),
            "graded intermediate decrease",
            "timing",
            "Independent onset jitter bounded by 4 minutes, representing the requested several-minute range.",
        )
    )
    cases.append(
        make_case(
            "case_07_10_minute_displacement",
            "10-minute or larger displacement",
            jittered_cycles(repeated, 12.0, seed + 7),
            "low timing credit for affected events",
            "timing",
            "Independent onset displacement bounded by 12 minutes.",
        )
    )

    rare_rearing = [
        remove_behavior(cycle, "rearing") + [("rearing", 5.25)]
        for cycle in repeated
    ]
    cases.append(
        make_case(
            "case_08_brief_rearing_repeated",
            "Brief rearing repeated",
            rare_rearing,
            "meaningful positive recurrence contribution",
            "rare-behavior",
            "One reproducible rearing event per cycle; duration is intentionally absent.",
        )
    )

    rearing_phase_choices = (2.30, 8.90, 12.80, 18.30, 22.40, 15.60)
    randomized_rearing: list[list[Event]] = []
    for index, cycle in enumerate(repeated):
        randomized_rearing.append(
            remove_behavior(cycle, "rearing")
            + [("rearing", rearing_phase_choices[index])]
        )
    cases.append(
        make_case(
            "case_09_brief_rearing_randomized",
            "Brief rearing randomized",
            randomized_rearing,
            "lower recurrence than repeated rearing",
            "rare-behavior",
            "The same one-event rearing count is retained, but its phase changes by cycle.",
        )
    )

    cases.append(
        make_case(
            "case_10_duration_only_change",
            "Duration-only change",
            repeated,
            "no change in score",
            "representation",
            "Conceptual durations vary between cycles, but the event-level representation contains only identity and onset phase.",
            duration_annotations="not passed to COMBA; conceptual durations are [1, 5] seconds by cycle",
        )
    )

    cases.append(
        make_case(
            "case_11_stable_composition_change",
            "Stable composition change",
            add_fixed_events(
                repeated,
                [("locomotion", phase) for phase in (2.30, 6.60, 11.70, 16.80, 22.00)],
            ),
            "high within-animal recurrence",
            "count",
            "The locomotion count is increased identically in every cycle.",
        )
    )

    cases.append(
        make_case(
            "case_12_reproducibly_increased_bout_number",
            "Reproducibly increased bout number",
            add_fixed_events(
                repeated,
                [("rearing", phase) for phase in (7.50, 18.75)],
            ),
            "high recurrence",
            "count",
            "Extra rearing events occur at reproducible phases on every cycle.",
        )
    )

    cases.append(
        make_case(
            "case_13_random_extra_bouts",
            "Random extra bouts",
            add_random_extra_events(repeated, "locomotion", 5, seed + 13),
            "lower recurrence than reproducible extra bouts",
            "count",
            "The same number of extra locomotion events is added, with unpredictable phases by cycle.",
        )
    )

    inserted = clone_cycles(repeated)
    inserted[2] = inserted[2] + [("grooming", 13.55)]
    cases.append(
        make_case(
            "case_14_one_inserted_event",
            "One inserted event",
            inserted,
            "localized modest decrease",
            "order",
            "One grooming event is inserted between existing events in cycle 3.",
            perturbation_phase=13.55,
        )
    )

    deleted = clone_cycles(repeated)
    deleted[2] = remove_event_at_phase(deleted[2], "grooming", 13.20)
    cases.append(
        make_case(
            "case_15_one_deleted_event",
            "One deleted event",
            deleted,
            "localized modest decrease",
            "order",
            "One grooming event is deleted from cycle 3.",
            perturbation_phase=13.20,
        )
    )

    order_base, motif_phases, motif_labels = local_order_template()
    order_repeated = [order_base for _ in range(N_CYCLES)]
    swapped = clone_cycles(order_repeated)
    swapped_labels = motif_labels.copy()
    swapped_labels[1], swapped_labels[2] = swapped_labels[2], swapped_labels[1]
    swapped[2] = replace_local_labels(swapped[2], motif_phases, swapped_labels)
    cases.append(
        make_case(
            "case_16_adjacent_order_swap",
            "Adjacent order swap",
            swapped,
            "moderate monotone reduction, not collapse",
            "order",
            "Two adjacent labels in an eight-event local motif are exchanged in cycle 3.",
        )
    )

    scrambled = clone_cycles(order_repeated)
    permutations = (
        [motif_labels[index] for index in (0, 2, 1, 3, 5, 4, 7, 6)],
        [motif_labels[index] for index in (1, 3, 0, 5, 2, 7, 4, 6)],
        [motif_labels[index] for index in (2, 0, 4, 1, 6, 3, 7, 5)],
        [motif_labels[index] for index in (7, 5, 3, 1, 0, 2, 4, 6)],
        [motif_labels[index] for index in (1, 6, 2, 7, 4, 0, 5, 3)],
    )
    for index, labels in enumerate(permutations, start=1):
        scrambled[index] = replace_local_labels(scrambled[index], motif_phases, labels)
    cases.append(
        make_case(
            "case_17_local_order_scrambling",
            "Local order scrambling",
            scrambled,
            "monotone lower than order-free",
            "order",
            "Labels and local phase window are preserved while the eight-event motif order changes by cycle.",
        )
    )

    cases.append(
        make_case(
            "case_18_rare_behavior_absent",
            "Rare behavior absent throughout",
            [remove_behavior(cycle, "digging") for cycle in repeated],
            "high recurrence of remaining repertoire",
            "composition",
            "Digging is absent from every cycle; no missing-behavior penalty is added.",
        )
    )

    frequency_cycle = [
        ("locomotion", phase) for phase in np.linspace(0.50, 23.50, 100)
    ] + [("rearing", 6.00), ("rearing", 18.00)]
    cases.append(
        make_case(
            "case_19_frequency_imbalance",
            "100 locomotion plus 2 rearing events",
            [frequency_cycle for _ in range(N_CYCLES)],
            "rare contribution non-negligible without dominance",
            "weighting",
            "Both frequent locomotion and sparse rearing are perfectly reproducible.",
        )
    )
    cases.append(
        make_case(
            "case_20_one_rare_event_every_cycle",
            "One rare event every cycle",
            rare_rearing,
            "meaningful rare-event contribution",
            "weighting",
            "Exactly one reproducible rearing event occurs in every cycle.",
        )
    )
    singleton = [remove_behavior(cycle, "rearing") for cycle in repeated]
    singleton[2] = singleton[2] + [("rearing", 5.25)]
    cases.append(
        make_case(
            "case_21_one_rare_event_once",
            "One rare event once in the recording",
            singleton,
            "limited influence",
            "weighting",
            "A single rearing event occurs in only one cycle.",
        )
    )

    cases.append(
        make_case(
            "case_22_generic_local_syntax",
            "Generic local syntax without circadian recurrence",
            generic_syntax_cycles(),
            "corrected recurrence near chance",
            "null-critical",
            "Repeated short motifs are preserved, but motif phase locations change between cycles.",
        )
    )

    stable_syntax = clone_cycles(order_repeated)
    stable_syntax_labels = motif_labels.copy()
    stable_syntax_labels[1], stable_syntax_labels[2] = stable_syntax_labels[2], stable_syntax_labels[1]
    stable_syntax[3] = replace_local_labels(stable_syntax[3], motif_phases, stable_syntax_labels)
    cases.append(
        make_case(
            "case_23_stable_circadian_local_variability",
            "Stable circadian organization with local-syntax variability",
            stable_syntax,
            "clearly above null",
            "null-critical",
            "Circadian event phases are stable; one local motif has limited order variation.",
        )
    )

    cases.append(
        make_case(
            "case_24_rest_changed_only",
            "Rest changed only",
            repeated,
            "no change in score",
            "representation",
            "Only conceptual rest occupancy changes; rest is absent from the event set.",
        )
    )

    global_shift_minutes = (0, 2, 5, 10, 15, 20)
    cases.append(
        make_case(
            "case_25_cycle_specific_global_phase_shift",
            "Cycle-specific global phase shift",
            [shift_cycle(base, minutes / 60.0) for minutes in global_shift_minutes],
            "lower recurrence with shift magnitude",
            "timing",
            "The whole repertoire is shifted by a different phase offset in each cycle.",
            shift_minutes=global_shift_minutes,
        )
    )

    cases.append(
        make_case(
            "case_26_absolute_phase_animal_A",
            "Different absolute phase between animals: animal A",
            repeated,
            "high within-animal recurrence",
            "cross-animal",
            "Animal A repeats perfectly at the baseline absolute phase.",
        )
    )
    cases.append(
        make_case(
            "case_26_absolute_phase_animal_B",
            "Different absolute phase between animals: animal B",
            [shift_cycle(base, 6.0) for _ in range(N_CYCLES)],
            "high within-animal recurrence",
            "cross-animal",
            "Animal B repeats perfectly at a different absolute phase; only within-animal scores are compared.",
        )
    )

    missing_27 = [[] for _ in range(N_CYCLES)]
    missing_27[2] = [(9.70, 10.00)]
    cases.append(
        make_case(
            "case_27_missing_data_gap",
            "Missing-data gap",
            repeated,
            "little score change with less available evidence",
            "missingness",
            "A phase gap in cycle 3 is made common to both members of each affected pair after dilation.",
            missing_intervals=missing_27,
        )
    )

    missing_28 = [[] for _ in range(N_CYCLES)]
    missing_28[2] = [(5.20, 5.30)]
    cases.append(
        make_case(
            "case_28_gap_covers_rare_event",
            "Gap covers rare repeated event",
            rare_rearing,
            "neither reward nor penalty for the masked event",
            "missingness",
            "The gap covers the repeated rare rearing event in cycle 3; common-coverage logic masks both pair members.",
            missing_intervals=missing_28,
        )
    )

    boundary_cycles = []
    for index, cycle in enumerate(repeated):
        boundary_phase = 23.99 if index % 2 == 0 else 0.01
        boundary_cycles.append(list(cycle) + [("rearing", boundary_phase)])
    cases.append(
        make_case(
            "case_29_cycle_boundary",
            "Cycle boundary",
            boundary_cycles,
            "high timing similarity across 24-hour boundary",
            "circular-phase",
            "A repeated rearing event alternates between 23.99 h and 0.01 h.",
        )
    )

    drift_minutes = (0, 0.5, 1.0, 1.5, 2.0, 2.5)
    cases.append(
        make_case(
            "case_30_frp_phase_drift",
            "FRP / phase drift sensitivity",
            [shift_cycle(base, minutes / 60.0) for minutes in drift_minutes],
            "small adjacent effect, larger longer-lag discrepancy",
            "timing",
            "A small systematic phase drift accumulates across cycles; adjacent and longer-lag scores are both reported.",
            drift_minutes=drift_minutes,
        )
    )

    boundary_order_cycles = [
        [
            ("rearing", 23.97),
            ("grooming", 23.99),
            ("locomotion", 0.02),
        ],
        [
            ("rearing", 23.99),
            ("grooming", 0.01),
            ("locomotion", 0.04),
        ],
    ]
    cases.append(
        make_case(
            "case_31_boundary_crossing_order",
            "Boundary-crossing sequence order",
            boundary_order_cycles,
            "high monotone score and small D_order",
            "circular-phase",
            "The same short ordered sequence crosses the 0/24-hour cut between two cycles; this test checks sequence order separately from circular timing distance.",
        )
    )

    return cases


def circular_phase_distance(phase_a: float, phase_b: float) -> float:
    """Return the shortest distance on the 24-hour phase circle, in hours."""

    absolute = abs((float(phase_a) % PERIOD_HOURS) - (float(phase_b) % PERIOD_HOURS))
    return min(absolute, PERIOD_HOURS - absolute)


def gaussian_credit(delta_hours: float, sigma_minutes: float) -> float:
    sigma_hours = sigma_minutes / 60.0
    return math.exp(-((delta_hours**2) / (2.0 * sigma_hours**2)))


def phase_distance_to_interval(
    phase: float, start: float, end: float
) -> float:
    """Distance from a phase to a possibly wrapping circular interval."""

    phase = float(phase) % PERIOD_HOURS
    start = float(start) % PERIOD_HOURS
    end = float(end) % PERIOD_HOURS
    if start <= end:
        if start <= phase <= end:
            return 0.0
    elif phase >= start or phase <= end:
        return 0.0
    return min(circular_phase_distance(phase, start), circular_phase_distance(phase, end))


def phase_is_excluded(
    phase: float,
    intervals: Iterable[tuple[float, float]],
    dilation_hours: float,
) -> bool:
    """Apply the specified common-coverage dilation to missing intervals."""

    return any(
        phase_distance_to_interval(phase, start, end) <= dilation_hours
        for start, end in intervals
    )


def eligible_pair(
    cycle_a: Sequence[Event],
    cycle_b: Sequence[Event],
    missing_a: Sequence[tuple[float, float]],
    missing_b: Sequence[tuple[float, float]],
    sigma_minutes: float,
) -> tuple[list[Event], list[Event]]:
    """Use the union of both gaps, dilated by h=3*sigma, on both cycles."""

    combined_intervals = list(missing_a) + list(missing_b)
    dilation_hours = 3.0 * sigma_minutes / 60.0
    if not combined_intervals:
        return list(cycle_a), list(cycle_b)
    filtered_a = [
        event
        for event in cycle_a
        if not phase_is_excluded(event[1], combined_intervals, dilation_hours)
    ]
    filtered_b = [
        event
        for event in cycle_b
        if not phase_is_excluded(event[1], combined_intervals, dilation_hours)
    ]
    return filtered_a, filtered_b


def event_weights(events: Sequence[Event], weighting: str) -> np.ndarray:
    counts = Counter(event[0] for event in events)
    if weighting == "raw_event":
        return np.ones(len(events), dtype=float)
    if weighting == "equal_total":
        return np.asarray([1.0 / counts[event[0]] for event in events], dtype=float)
    if weighting == "sqrt_count":
        return np.asarray(
            [1.0 / math.sqrt(counts[event[0]]) for event in events], dtype=float
        )
    raise ValueError(f"Unknown weighting rule: {weighting}")


def credit_matrix(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    sigma_minutes: float,
) -> np.ndarray:
    matrix = np.zeros((len(events_a), len(events_b)), dtype=float)
    for index_a, (behavior_a, phase_a) in enumerate(events_a):
        for index_b, (behavior_b, phase_b) in enumerate(events_b):
            if behavior_a == behavior_b:
                matrix[index_a, index_b] = min(
                    weights_a[index_a], weights_b[index_b]
                ) * gaussian_credit(
                    circular_phase_distance(phase_a, phase_b), sigma_minutes
                )
    return matrix


def monotone_match(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    credits: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Maximum-weight optional noncrossing matching by dynamic programming."""

    n_a = len(events_a)
    n_b = len(events_b)
    dp = np.zeros((n_a + 1, n_b + 1), dtype=float)
    back = np.zeros((n_a + 1, n_b + 1), dtype=np.int8)
    # back: 1 skips A, 2 skips B, 3 matches the two current events.
    for index_a in range(1, n_a + 1):
        for index_b in range(1, n_b + 1):
            skip_a = dp[index_a - 1, index_b]
            skip_b = dp[index_a, index_b - 1]
            if skip_a >= skip_b:
                best = skip_a
                direction = 1
            else:
                best = skip_b
                direction = 2
            if events_a[index_a - 1][0] == events_b[index_b - 1][0]:
                match = dp[index_a - 1, index_b - 1] + credits[index_a - 1, index_b - 1]
                # Prefer a valid match on exact ties; the score remains globally optimal.
                if match >= best - 1e-15:
                    best = match
                    direction = 3
            dp[index_a, index_b] = best
            back[index_a, index_b] = direction

    pairs: list[tuple[int, int]] = []
    index_a, index_b = n_a, n_b
    while index_a > 0 and index_b > 0:
        direction = back[index_a, index_b]
        if direction == 3:
            pairs.append((index_a - 1, index_b - 1))
            index_a -= 1
            index_b -= 1
        elif direction == 1:
            index_a -= 1
        else:
            index_b -= 1
    pairs.reverse()
    return float(dp[n_a, n_b]), pairs


def cyclic_monotone_match(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    credits: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Maximize the existing monotone DP over cyclic cuts of one sequence.

    The shorter sequence is rotated by index only.  Its original event phases
    and the corresponding entries of ``credits`` are preserved.
    """

    if not events_a or not events_b:
        return 0.0, []

    if len(events_a) <= len(events_b):
        fixed_events = list(events_b)
        rotating_events = list(events_a)
        fixed_is_a = False
        base_credits = credits.T
    else:
        fixed_events = list(events_a)
        rotating_events = list(events_b)
        fixed_is_a = True
        base_credits = credits

    best_credit = -math.inf
    best_pairs: list[tuple[int, int]] = []
    n_rotating = len(rotating_events)
    for cut in range(n_rotating):
        rotating_indices = list(range(cut, n_rotating)) + list(range(cut))
        rotated_events = [rotating_events[index] for index in rotating_indices]
        rotated_credits = base_credits[:, rotating_indices]
        candidate_credit, candidate_pairs = monotone_match(
            fixed_events,
            rotated_events,
            rotated_credits,
        )
        if candidate_credit > best_credit + 1e-15:
            best_credit = candidate_credit
            if fixed_is_a:
                best_pairs = [
                    (fixed_index, rotating_indices[rotating_index])
                    for fixed_index, rotating_index in candidate_pairs
                ]
            else:
                best_pairs = [
                    (rotating_indices[rotating_index], fixed_index)
                    for fixed_index, rotating_index in candidate_pairs
                ]

    return float(best_credit), best_pairs


def order_free_match(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    credits: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Maximum-weight same-label one-to-one matching without an order constraint."""

    labels = sorted(set(event[0] for event in events_a) | set(event[0] for event in events_b))
    pairs: list[tuple[int, int]] = []
    total = 0.0
    for label in labels:
        rows = [index for index, event in enumerate(events_a) if event[0] == label]
        columns = [index for index, event in enumerate(events_b) if event[0] == label]
        if not rows or not columns:
            continue
        submatrix = credits[np.ix_(rows, columns)]
        if len(rows) <= len(columns):
            row_indices, column_indices = linear_sum_assignment(-submatrix)
            local_pairs = [(rows[row], columns[column]) for row, column in zip(row_indices, column_indices)]
        else:
            column_indices, row_indices = linear_sum_assignment(-submatrix.T)
            local_pairs = [(rows[row], columns[column]) for column, row in zip(column_indices, row_indices)]
        pairs.extend(local_pairs)
        total += float(sum(credits[row, column] for row, column in local_pairs))
    pairs.sort()
    return total, pairs


def score_pair(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    sigma_minutes: float,
    weighting: str,
    matching: str = "monotone",
) -> dict:
    """Score one eligible cycle pair and expose supporting quantities."""

    events_a = sorted(events_a, key=lambda event: event[1])
    events_b = sorted(events_b, key=lambda event: event[1])
    weights_a = event_weights(events_a, weighting)
    weights_b = event_weights(events_b, weighting)
    credits = credit_matrix(events_a, events_b, weights_a, weights_b, sigma_minutes)
    if matching == "monotone":
        matched_credit, pairs = cyclic_monotone_match(events_a, events_b, credits)
    elif matching == "order_free":
        matched_credit, pairs = order_free_match(events_a, events_b, credits)
    else:
        raise ValueError(f"Unknown matching mode: {matching}")

    mass_a = float(weights_a.sum())
    mass_b = float(weights_b.sum())
    denominator = mass_a + mass_b
    score = 0.0 if denominator == 0 else 2.0 * matched_credit / denominator
    matched_by_behavior: dict[str, float] = defaultdict(float)
    for index_a, index_b in pairs:
        matched_by_behavior[events_a[index_a][0]] += float(credits[index_a, index_b])
    mass_by_behavior = defaultdict(float)
    for event, weight in zip(events_a, weights_a):
        mass_by_behavior[event[0]] += float(weight)
    for event, weight in zip(events_b, weights_b):
        mass_by_behavior[event[0]] += float(weight)

    return {
        "score": float(score),
        "matched_credit": float(matched_credit),
        "matched_pairs": len(pairs),
        "pairs": pairs,
        "events_a": list(events_a),
        "events_b": list(events_b),
        "mass_a": mass_a,
        "mass_b": mass_b,
        "eligible_events": len(events_a) + len(events_b),
        "matched_by_behavior": dict(matched_by_behavior),
        "mass_by_behavior": dict(mass_by_behavior),
    }


def evaluate_animal(
    case: dict,
    sigma_minutes: float,
    weighting: str,
    matching: str,
) -> dict:
    """Compute the mean adjacent-cycle score and optional longer-lag diagnostics."""

    cycles = clone_cycles(case["cycles"])
    missing = case["missing_intervals"]
    pair_results: list[dict] = []
    for index in range(len(cycles) - 1):
        events_a, events_b = eligible_pair(
            cycles[index],
            cycles[index + 1],
            missing[index],
            missing[index + 1],
            sigma_minutes,
        )
        result = score_pair(events_a, events_b, sigma_minutes, weighting, matching)
        result["cycle_a"] = index
        result["cycle_b"] = index + 1
        pair_results.append(result)

    def aggregate_behavior(key: str) -> dict[str, float]:
        aggregate: dict[str, float] = defaultdict(float)
        for pair in pair_results:
            for behavior, value in pair[key].items():
                aggregate[behavior] += value
        if pair_results:
            return {behavior: value / len(pair_results) for behavior, value in aggregate.items()}
        return {}

    matched_by_behavior = aggregate_behavior("matched_by_behavior")
    total_matched = float(sum(matched_by_behavior.values()))
    rearing_credit_fraction = (
        matched_by_behavior.get("rearing", 0.0) / total_matched
        if total_matched > 0
        else 0.0
    )
    mass_by_behavior = aggregate_behavior("mass_by_behavior")
    eligible_by_behavior: dict[str, float] = defaultdict(float)
    for pair in pair_results:
        for event in pair["events_a"] + pair["events_b"]:
            eligible_by_behavior[event[0]] += 1.0
    if pair_results:
        eligible_by_behavior = {
            behavior: value / len(pair_results)
            for behavior, value in eligible_by_behavior.items()
        }

    lag_scores: dict[int, float] = {}
    for lag in (2, 3):
        lag_pairs = []
        for index in range(len(cycles) - lag):
            events_a, events_b = eligible_pair(
                cycles[index],
                cycles[index + lag],
                missing[index],
                missing[index + lag],
                sigma_minutes,
            )
            lag_pairs.append(
                score_pair(events_a, events_b, sigma_minutes, weighting, matching)["score"]
            )
        if lag_pairs:
            lag_scores[lag] = float(np.mean(lag_pairs))

    return {
        "score": float(np.mean([pair["score"] for pair in pair_results])),
        "matched_credit": float(np.mean([pair["matched_credit"] for pair in pair_results])),
        "matched_pairs": float(np.mean([pair["matched_pairs"] for pair in pair_results])),
        "total_mass": float(np.mean([pair["mass_a"] + pair["mass_b"] for pair in pair_results])),
        "eligible_events": float(np.mean([pair["eligible_events"] for pair in pair_results])),
        "matched_by_behavior": matched_by_behavior,
        "mass_by_behavior": mass_by_behavior,
        "eligible_by_behavior": dict(eligible_by_behavior),
        "rearing_credit_fraction": rearing_credit_fraction,
        "lag_scores": lag_scores,
        "pair_results": pair_results,
    }


def run_matching_invariants(cases_by_id: dict[str, dict]) -> dict[str, dict[str, float | bool]]:
    """Run the requested phase-origin and score-symmetry checks."""

    original_a = baseline_cycle()
    original_b = baseline_cycle()
    original_score = score_pair(
        original_a,
        original_b,
        PRIMARY_SIGMA_MINUTES,
        PRIMARY_WEIGHTING,
        "monotone",
    )["score"]
    common_offset_hours = 7.25
    shifted_score = score_pair(
        shift_cycle(original_a, common_offset_hours),
        shift_cycle(original_b, common_offset_hours),
        PRIMARY_SIGMA_MINUTES,
        PRIMARY_WEIGHTING,
        "monotone",
    )["score"]
    common_origin_difference = abs(original_score - shifted_score)

    symmetry_case = cases_by_id["case_14_one_inserted_event"]
    symmetry_a, symmetry_b = symmetry_case["cycles"][1:3]
    score_ab = score_pair(
        symmetry_a,
        symmetry_b,
        PRIMARY_SIGMA_MINUTES,
        PRIMARY_WEIGHTING,
        "monotone",
    )["score"]
    score_ba = score_pair(
        symmetry_b,
        symmetry_a,
        PRIMARY_SIGMA_MINUTES,
        PRIMARY_WEIGHTING,
        "monotone",
    )["score"]
    symmetry_difference = abs(score_ab - score_ba)
    # This is a direct implementation invariant, not a new scientific score.
    assert symmetry_difference <= 1e-12, (
        f"Cyclic monotone score lost symmetry: {score_ab} versus {score_ba}"
    )

    return {
        "common_phase_origin": {
            "offset_hours": common_offset_hours,
            "original_score": original_score,
            "shifted_score": shifted_score,
            "difference": common_origin_difference,
            "pass": common_origin_difference <= 1e-12,
        },
        "symmetry": {
            "score_ab": score_ab,
            "score_ba": score_ba,
            "difference": symmetry_difference,
            "pass": symmetry_difference <= 1e-12,
        },
    }


def partition_marked_blocks(
    events: Sequence[Event], max_block_size: int, max_elapsed_hours: float = 2.0
) -> list[list[str]]:
    """Partition one cycle into short contiguous label blocks."""

    if max_block_size <= 0:
        raise ValueError("max_block_size must be positive")
    blocks: list[list[str]] = []
    current: list[str] = []
    first_phase: float | None = None
    for behavior, phase in sorted(events, key=lambda event: event[1]):
        should_break = bool(
            current
            and (
                len(current) >= max_block_size
                or first_phase is None
                or phase - first_phase > max_elapsed_hours
            )
        )
        if should_break:
            blocks.append(current)
            current = []
            first_phase = None
        if not current:
            first_phase = phase
        current.append(behavior)
    if current:
        blocks.append(current)
    return blocks


def generate_null_cycle(
    events: Sequence[Event],
    null_family: str,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> list[Event]:
    """Generate exactly one of the three requested null cycles."""

    ordered = sorted(events, key=lambda event: event[1])
    if not ordered:
        return []
    labels = [event[0] for event in ordered]
    phases = [event[1] for event in ordered]

    if null_family == "complete_label":
        null_labels = [labels[index] for index in rng.permutation(len(labels))]
    elif null_family == "circular_rotation":
        if len(labels) <= 1:
            null_labels = labels
        else:
            offset = int(rng.integers(1, len(labels)))
            null_labels = labels[offset:] + labels[:offset]
    elif null_family == "marked_block":
        if block_size is None:
            raise ValueError("marked_block null requires block_size")
        blocks = partition_marked_blocks(ordered, block_size)
        if len(blocks) <= 1:
            null_labels = labels
        else:
            permutation = rng.permutation(len(blocks))
            null_labels = [
                label
                for block_index in permutation
                for label in blocks[int(block_index)]
            ]
    else:
        raise ValueError(f"Unknown null family: {null_family}")

    return list(zip(null_labels, phases))


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return (DEFAULT_SEED + zlib.crc32(text)) % (2**32)


def null_mean_for_case(
    case: dict,
    sigma_minutes: float,
    weighting: str,
    null_family: str,
    block_size: int | None,
    n_replicates: int,
) -> tuple[float, float]:
    """Return the mean and sample SD of animal-level null scores."""

    rng = np.random.default_rng(
        stable_seed(case["case_id"], sigma_minutes, weighting, null_family, block_size)
    )
    replicate_scores = np.empty(n_replicates, dtype=float)
    cycles = clone_cycles(case["cycles"])
    missing = case["missing_intervals"]
    for replicate in range(n_replicates):
        pair_scores = []
        for index in range(len(cycles) - 1):
            events_a, events_b = eligible_pair(
                cycles[index],
                cycles[index + 1],
                missing[index],
                missing[index + 1],
                sigma_minutes,
            )
            null_a = generate_null_cycle(events_a, null_family, rng, block_size)
            null_b = generate_null_cycle(events_b, null_family, rng, block_size)
            pair_scores.append(
                score_pair(
                    null_a,
                    null_b,
                    sigma_minutes,
                    weighting,
                    "monotone",
                )["score"]
            )
        replicate_scores[replicate] = float(np.mean(pair_scores))
    return float(np.mean(replicate_scores)), float(np.std(replicate_scores, ddof=1))


def chance_correct(observed: float, null_mean: float) -> float:
    denominator = 1.0 - null_mean
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((observed - null_mean) / denominator)


def downstream_retention(result: dict, perturbation_phase: float) -> float:
    """Measure whether later matches survive an insertion/deletion."""

    retention_values = []
    for pair in result["pair_results"]:
        later_a = [event for event in pair["events_a"] if event[1] > perturbation_phase]
        later_b = [event for event in pair["events_b"] if event[1] > perturbation_phase]
        denominator = min(len(later_a), len(later_b))
        if denominator == 0:
            continue
        matched_later = sum(
            1
            for index_a, index_b in pair["pairs"]
            if pair["events_a"][index_a][1] > perturbation_phase
            and pair["events_b"][index_b][1] > perturbation_phase
        )
        retention_values.append(matched_later / denominator)
    return float(np.mean(retention_values)) if retention_values else float("nan")


def support_json(result: dict, extra: dict | None = None) -> str:
    payload = {
        "matched_pairs_mean": round(result["matched_pairs"], 6),
        "eligible_events_mean": round(result["eligible_events"], 6),
        "total_mass_mean": round(result["total_mass"], 6),
        "rearing_credit_fraction": round(result["rearing_credit_fraction"], 6),
        "lag_2_score": round(result["lag_scores"].get(2, float("nan")), 6),
        "lag_3_score": round(result["lag_scores"].get(3, float("nan")), 6),
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, sort_keys=True, allow_nan=True)


def add_result_row(
    rows: list[dict],
    case: dict,
    result: dict,
    metric_variant: str,
    matching: str,
    sigma_minutes: float,
    weighting: str,
    null_condition: str,
    block_size: int | None,
    null_mean: float | None = None,
    corrected_score: float | None = None,
    status: str = "ambiguous",
    extra_support: dict | None = None,
) -> None:
    rows.append(
        {
            "case_id": case["case_id"],
            "case_name": case["name"],
            "category": case["category"],
            "metric_variant": metric_variant,
            "matching": matching,
            "sigma_minutes": sigma_minutes,
            "weighting": weighting,
            "null_condition": null_condition,
            "marked_block_size": block_size if block_size is not None else "",
            "raw_score": result["score"],
            "null_mean": "" if null_mean is None else null_mean,
            "corrected_score": "" if corrected_score is None else corrected_score,
            "mean_matched_credit": result["matched_credit"],
            "mean_total_mass": result["total_mass"],
            "mean_eligible_events": result["eligible_events"],
            "order_free_score": "",
            "d_order": "",
            "expected_direction": case["expected_direction"],
            "pass_fail_ambiguous": status,
            "supporting_quantities": support_json(result, extra_support),
        }
    )


def status_for_case(
    case_id: str,
    primary: dict,
    primary_free: dict,
    primary_results: dict[str, dict],
    preferred_corrected: dict[str, float],
    reference_results: dict[str, dict],
    pooled_invariants: dict[str, dict[str, bool]],
) -> tuple[str, str]:
    """Apply predefined qualitative acceptance checks to the primary variant."""

    score = primary["score"]
    free_score = primary_free["score"]
    baseline_result = primary_results["case_01_perfect_repeat"]

    def high(threshold: float = 0.95) -> tuple[str, str]:
        return ("pass", f"raw score {score:.3f} >= {threshold:.2f}") if score >= threshold else (
            "fail" if score < threshold - 0.15 else "ambiguous",
            f"raw score {score:.3f} did not clearly reach {threshold:.2f}",
        )

    if case_id in {
        "case_01_perfect_repeat",
        "case_08_brief_rearing_repeated",
        "case_10_duration_only_change",
        "case_11_stable_composition_change",
        "case_12_reproducibly_increased_bout_number",
        "case_18_rare_behavior_absent",
        "case_20_one_rare_event_every_cycle",
        "case_24_rest_changed_only",
        "case_26_absolute_phase_animal_A",
        "case_26_absolute_phase_animal_B",
        "case_29_cycle_boundary",
    }:
        return high()
    if case_id == "case_02_randomized_behavior_identity":
        corrected = preferred_corrected.get(case_id, float("nan"))
        if np.isfinite(corrected) and abs(corrected) <= 0.10:
            return "pass", f"preferred marked-block corrected score {corrected:.3f} is near chance"
        if np.isfinite(corrected) and abs(corrected) > 0.20:
            return "fail", f"preferred corrected score {corrected:.3f} is positively displaced from chance"
        return "ambiguous", "preferred null-corrected score was not decisively near chance"
    if case_id == "case_03_same_pooled_phase_different_realizations":
        reference = reference_results[case_id]["score"]
        invariant = all(pooled_invariants[case_id].values())
        reduction = reference - score
        if not invariant:
            return "fail", "pooled per-behavior phase invariant failed"
        if reference >= 0.95 and score <= 0.70 and reduction >= 0.30:
            return "pass", (
                f"exact pooled-phase invariant; reference {reference:.3f}, "
                f"rearranged {score:.3f}, reduction {reduction:.3f}"
            )
        if reference < 0.85 or score > 0.85:
            return "fail", (
                f"invariant held but recurrence separation was insufficient: "
                f"reference {reference:.3f}, rearranged {score:.3f}"
            )
        return "ambiguous", (
            f"invariant held; reference {reference:.3f}, rearranged {score:.3f}, "
            f"reduction {reduction:.3f}"
        )
    if case_id == "case_04_30_second_jitter":
        return high(0.95)
    if case_id == "case_05_1_minute_jitter":
        return high(0.85)
    if case_id == "case_06_2_to_5_minute_jitter":
        if 0.30 <= score <= 0.90:
            return "pass", f"raw score {score:.3f} is intermediate"
        if score < 0.15 or score > 0.98:
            return "fail", f"raw score {score:.3f} is not intermediate"
        return "ambiguous", f"raw score {score:.3f} is borderline intermediate"
    if case_id == "case_07_10_minute_displacement":
        if score <= 0.60:
            return "pass", f"raw score {score:.3f} is substantially reduced"
        if score > 0.80:
            return "fail", f"raw score {score:.3f} remains high under large displacement"
        return "ambiguous", f"raw score {score:.3f} is intermediate"
    if case_id == "case_09_brief_rearing_randomized":
        repeated_score = primary_results["case_08_brief_rearing_repeated"]["score"]
        difference = repeated_score - score
        if difference >= 0.01:
            return "pass", f"score is {difference:.3f} below repeated rearing"
        if difference < 0.002:
            return "fail", "randomized rare-event phase caused no detectable reduction"
        return "ambiguous", f"reduction from repeated rearing is only {difference:.3f}"
    if case_id == "case_13_random_extra_bouts":
        reference = primary_results["case_12_reproducibly_increased_bout_number"]["score"]
        difference = reference - score
        if difference >= 0.01 and score >= 0.50:
            return "pass", f"random extras reduce score by {difference:.3f}"
        if difference < 0.002:
            return "fail", "random extras were not distinguished from reproducible extras"
        return "ambiguous", f"random-extra reduction is {difference:.3f}"
    if case_id in {"case_14_one_inserted_event", "case_15_one_deleted_event"}:
        retention = downstream_retention(primary, float(case_id.endswith("inserted_event") and 13.55 or 13.20))
        if score >= 0.80 and retention >= 0.90:
            return "pass", f"score {score:.3f}; downstream retention {retention:.3f}"
        if score < 0.60 or retention < 0.70:
            return "fail", f"score {score:.3f}; downstream retention {retention:.3f}"
        return "ambiguous", f"score {score:.3f}; downstream retention {retention:.3f}"
    if case_id == "case_16_adjacent_order_swap":
        difference = free_score - score
        if score >= 0.75 and difference >= 0.01:
            return "pass", f"monotone score {score:.3f}; D_order {difference:.3f}"
        if score < 0.55:
            return "fail", f"monotone score {score:.3f} is too collapsed"
        return "ambiguous", f"monotone score {score:.3f}; D_order {difference:.3f}"
    if case_id == "case_17_local_order_scrambling":
        difference = free_score - score
        if score >= 0.45 and difference >= 0.05:
            return "pass", f"D_order {difference:.3f} detects local scrambling"
        if difference < 0.01:
            return "fail", f"D_order {difference:.3f} does not detect scrambling"
        return "ambiguous", f"D_order {difference:.3f} is borderline"
    if case_id == "case_19_frequency_imbalance":
        raw = primary_results[case_id]
        if score >= 0.98 and raw["rearing_credit_fraction"] >= 0.05:
            return "pass", f"score {score:.3f}; rearing credit fraction {raw['rearing_credit_fraction']:.3f}"
        if raw["rearing_credit_fraction"] < 0.02:
            return "fail", "rearing contribution is effectively invisible"
        return "ambiguous", f"rearing credit fraction {raw['rearing_credit_fraction']:.3f}"
    if case_id == "case_21_one_rare_event_once":
        reference = primary_results["case_20_one_rare_event_every_cycle"]["score"]
        if abs(reference - score) <= 0.10:
            return "pass", f"singleton changes score by only {reference - score:.3f}"
        if abs(reference - score) > 0.20:
            return "fail", f"singleton changes score by {reference - score:.3f}"
        return "ambiguous", f"singleton change is {reference - score:.3f}"
    if case_id == "case_22_generic_local_syntax":
        corrected = preferred_corrected.get(case_id, float("nan"))
        if np.isfinite(corrected) and abs(corrected) <= 0.10:
            return "pass", f"preferred corrected score {corrected:.3f} is near chance"
        if np.isfinite(corrected) and abs(corrected) > 0.20:
            return "fail", f"preferred corrected score {corrected:.3f} is substantial"
        return "ambiguous", "preferred null-corrected score is borderline"
    if case_id == "case_23_stable_circadian_local_variability":
        corrected = preferred_corrected.get(case_id, float("nan"))
        if np.isfinite(corrected) and corrected >= 0.20 and score >= 0.75:
            return "pass", f"raw {score:.3f}; preferred corrected {corrected:.3f}"
        if np.isfinite(corrected) and corrected < 0.05:
            return "fail", f"stable recurrence is largely conditioned away ({corrected:.3f})"
        return "ambiguous", f"raw {score:.3f}; preferred corrected {corrected:.3f}"
    if case_id == "case_25_cycle_specific_global_phase_shift":
        if 0.20 < score < 0.98:
            return "pass", f"global shifts reduce score to {score:.3f}"
        if score >= 0.99:
            return "fail", "global shifts did not reduce recurrence"
        return "ambiguous", f"global-shift score {score:.3f}"
    if case_id == "case_27_missing_data_gap":
        if score >= 0.95 and primary["eligible_events"] < baseline_result["eligible_events"]:
            return "pass", f"score {score:.3f} with fewer eligible events"
        if score < 0.80:
            return "fail", f"missingness caused a large score change ({score:.3f})"
        return "ambiguous", f"score {score:.3f}; eligible events {primary['eligible_events']:.1f}"
    if case_id == "case_28_gap_covers_rare_event":
        if score >= 0.95 and primary["eligible_by_behavior"].get("rearing", 0.0) < 2.0:
            return "pass", f"masked rare event is not scored; raw score {score:.3f}"
        if score < 0.80:
            return "fail", f"gap caused a large score change ({score:.3f})"
        return "ambiguous", f"score {score:.3f}; eligible rearing {primary['eligible_by_behavior'].get('rearing', 0.0):.1f}"
    if case_id == "case_30_frp_phase_drift":
        lag_2 = primary["lag_scores"].get(2, float("nan"))
        lag_3 = primary["lag_scores"].get(3, float("nan"))
        if score >= 0.90 and lag_2 < score and lag_3 < lag_2:
            return "pass", f"adjacent {score:.3f}; lag-2 {lag_2:.3f}; lag-3 {lag_3:.3f}"
        if score < 0.75 or lag_3 >= lag_2:
            return "fail", f"adjacent {score:.3f}; lag-2 {lag_2:.3f}; lag-3 {lag_3:.3f}"
        return "ambiguous", f"adjacent {score:.3f}; lag-2 {lag_2:.3f}; lag-3 {lag_3:.3f}"
    if case_id == "case_31_boundary_crossing_order":
        difference = free_score - score
        if score >= 0.85 and difference <= 0.05:
            return "pass", f"monotone {score:.3f}; order-free {free_score:.3f}; D_order {difference:.3f}"
        if score < 0.70 or difference > 0.15:
            return "fail", f"monotone {score:.3f}; order-free {free_score:.3f}; D_order {difference:.3f}"
        return "ambiguous", f"monotone {score:.3f}; order-free {free_score:.3f}; D_order {difference:.3f}"
    return high()


def make_truth_rows(
    cases: Sequence[dict],
    metric_cache: dict[tuple[str, float, str, str], dict],
    primary_results: dict[str, dict],
    primary_free_results: dict[str, dict],
    preferred_corrected: dict[str, float],
    reference_results: dict[str, dict],
    pooled_invariants: dict[str, dict[str, bool]],
) -> list[dict]:
    rows: list[dict] = []
    case_status: dict[str, str] = {}
    case_reason: dict[str, str] = {}
    for case in cases:
        status, reason = status_for_case(
            case["case_id"],
            primary_results[case["case_id"]],
            primary_free_results[case["case_id"]],
            primary_results,
            preferred_corrected,
            reference_results,
            pooled_invariants,
        )
        case_status[case["case_id"]] = status
        case_reason[case["case_id"]] = reason

    for case in cases:
        case_id = case["case_id"]
        case_support = {"assessment_reason": case_reason[case_id]}
        if case_id in reference_results:
            reference_score = reference_results[case_id]["score"]
            rearranged_score = primary_results[case_id]["score"]
            case_support.update(
                {
                    "pooled_phase_invariant_by_behavior": pooled_invariants[case_id],
                    "pooled_phase_invariant": all(pooled_invariants[case_id].values()),
                    "reference_score": reference_score,
                    "rearranged_score": rearranged_score,
                    "reference_minus_rearranged": reference_score - rearranged_score,
                }
            )
        for sigma in SIGMA_MINUTES:
            result = metric_cache[(case_id, float(sigma), PRIMARY_WEIGHTING, "monotone")]
            add_result_row(
                rows,
                case,
                result,
                "sigma_sweep",
                "monotone",
                sigma,
                PRIMARY_WEIGHTING,
                "none",
                None,
                status=case_status[case_id],
                extra_support=case_support,
            )
        for weighting in WEIGHTING_RULES:
            result = metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), weighting, "monotone")]
            add_result_row(
                rows,
                case,
                result,
                "weighting_sweep",
                "monotone",
                PRIMARY_SIGMA_MINUTES,
                weighting,
                "none",
                None,
                status=case_status[case_id],
                extra_support=case_support,
            )

        monotone = primary_results[case_id]
        order_free = primary_free_results[case_id]
        add_result_row(
            rows,
            case,
            monotone,
            "order_comparison_monotone",
            "monotone",
            PRIMARY_SIGMA_MINUTES,
            PRIMARY_WEIGHTING,
            "none",
            None,
            status=case_status[case_id],
            extra_support={
                **case_support,
                "order_free_score": order_free["score"],
                "d_order": order_free["score"] - monotone["score"],
            },
        )
        rows[-1]["order_free_score"] = order_free["score"]
        rows[-1]["d_order"] = order_free["score"] - monotone["score"]

        add_result_row(
            rows,
            case,
            order_free,
            "order_comparison_free",
            "order_free",
            PRIMARY_SIGMA_MINUTES,
            PRIMARY_WEIGHTING,
            "none",
            None,
            status=case_status[case_id],
            extra_support={
                **case_support,
                "monotone_score": monotone["score"],
                "d_order": order_free["score"] - monotone["score"],
            },
        )
        rows[-1]["order_free_score"] = order_free["score"]
        rows[-1]["d_order"] = order_free["score"] - monotone["score"]

    return rows


def append_null_rows(
    rows: list[dict],
    cases_by_id: dict[str, dict],
    primary_results: dict[str, dict],
    null_results: dict[tuple[str, str, int | None], tuple[float, float]],
    case_status: dict[str, str],
    case_reason: dict[str, str],
) -> None:
    critical_ids = (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    )
    null_specs = (
        ("null_a_complete_label", "complete_label", None),
        ("null_b_circular_mark_rotation", "circular_rotation", None),
        ("null_c_marked_block_3", "marked_block", 3),
    )
    for case_id in critical_ids:
        case = cases_by_id[case_id]
        observed = primary_results[case_id]
        for condition_name, family, block_size in null_specs:
            null_mean, null_sd = null_results[(case_id, family, block_size)]
            corrected = chance_correct(observed["score"], null_mean)
            status = "ambiguous"
            if case_id in {"case_02_randomized_behavior_identity", "case_22_generic_local_syntax"}:
                if abs(corrected) <= 0.10:
                    status = "pass"
                elif abs(corrected) > 0.20:
                    status = "fail"
            elif case_id == "case_01_perfect_repeat":
                if corrected >= 0.70:
                    status = "pass"
                elif corrected < 0.40:
                    status = "fail"
            elif case_id == "case_23_stable_circadian_local_variability":
                if corrected >= 0.20:
                    status = "pass"
                elif corrected < 0.05:
                    status = "fail"
            add_result_row(
                rows,
                case,
                observed,
                "null_corrected",
                "monotone",
                PRIMARY_SIGMA_MINUTES,
                PRIMARY_WEIGHTING,
                condition_name,
                block_size,
                null_mean=null_mean,
                corrected_score=corrected,
                status=status,
                extra_support={
                    "null_sd": null_sd,
                    "assessment_reason": case_reason[case_id],
                },
            )

        for block_size in (2, 5):
            null_mean, null_sd = null_results[(case_id, "marked_block", block_size)]
            corrected = chance_correct(observed["score"], null_mean)
            status = "ambiguous"
            if case_id in {"case_02_randomized_behavior_identity", "case_22_generic_local_syntax"}:
                if abs(corrected) <= 0.10:
                    status = "pass"
                elif abs(corrected) > 0.20:
                    status = "fail"
            elif case_id == "case_01_perfect_repeat":
                if corrected >= 0.70:
                    status = "pass"
                elif corrected < 0.40:
                    status = "fail"
            elif case_id == "case_23_stable_circadian_local_variability":
                if corrected >= 0.20:
                    status = "pass"
                elif corrected < 0.05:
                    status = "fail"
            add_result_row(
                rows,
                case,
                observed,
                "marked_block_sensitivity",
                "monotone",
                PRIMARY_SIGMA_MINUTES,
                PRIMARY_WEIGHTING,
                f"null_c_marked_block_{block_size}",
                block_size,
                null_mean=null_mean,
                corrected_score=corrected,
                status=status,
                extra_support={"null_sd": null_sd},
            )


def make_parameter_summary(
    metric_cache: dict[tuple[str, float, str, str], dict],
    primary_results: dict[str, dict],
    primary_free_results: dict[str, dict],
    reference_results: dict[str, dict],
    pooled_invariants: dict[str, dict[str, bool]],
    matching_invariants: dict[str, dict[str, float | bool]],
    null_results: dict[tuple[str, str, int | None], tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for sigma in SIGMA_MINUTES:
        for delta in DELTA_MINUTES:
            rows.append(
                {
                    "parameter_family": "sigma_kernel",
                    "parameter": f"sigma_{sigma}_minutes",
                    "case_id": "",
                    "delta_minutes": delta,
                    "value": gaussian_credit(delta / 60.0, sigma),
                    "raw_score": "",
                    "null_mean": "",
                    "corrected_score": "",
                    "rearing_credit_fraction": "",
                    "notes": "K(delta) = exp(-delta^2/(2*sigma^2))",
                }
            )
        for case_id in (
            "case_04_30_second_jitter",
            "case_05_1_minute_jitter",
            "case_06_2_to_5_minute_jitter",
            "case_07_10_minute_displacement",
        ):
            result = metric_cache[(case_id, float(sigma), PRIMARY_WEIGHTING, "monotone")]
            rows.append(
                {
                    "parameter_family": "sigma_truth_case",
                    "parameter": f"sigma_{sigma}_minutes",
                    "case_id": case_id,
                    "delta_minutes": "",
                    "value": "",
                    "raw_score": result["score"],
                    "null_mean": "",
                    "corrected_score": "",
                    "rearing_credit_fraction": result["rearing_credit_fraction"],
                    "notes": "Primary sqrt-count monotone score",
                }
            )

    stress_case_ids = (
        "case_19_frequency_imbalance",
        "case_20_one_rare_event_every_cycle",
        "case_21_one_rare_event_once",
        "case_11_stable_composition_change",
        "case_12_reproducibly_increased_bout_number",
        "case_13_random_extra_bouts",
        "case_18_rare_behavior_absent",
    )
    for weighting in WEIGHTING_RULES:
        values = {
            case_id: metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), weighting, "monotone")]
            for case_id in stress_case_ids
        }
        frequency = values["case_19_frequency_imbalance"]
        rows.append(
            {
                "parameter_family": "weighting_rule",
                "parameter": weighting,
                "case_id": "case_19_frequency_imbalance",
                "delta_minutes": "",
                "value": "",
                "raw_score": frequency["score"],
                "null_mean": "",
                "corrected_score": "",
                "rearing_credit_fraction": frequency["rearing_credit_fraction"],
                "case_20_score": values["case_20_one_rare_event_every_cycle"]["score"],
                "case_21_score": values["case_21_one_rare_event_once"]["score"],
                "case_11_score": values["case_11_stable_composition_change"]["score"],
                "case_12_score": values["case_12_reproducibly_increased_bout_number"]["score"],
                "case_13_score": values["case_13_random_extra_bouts"]["score"],
                "case_18_score": values["case_18_rare_behavior_absent"]["score"],
                "notes": "Raw/equal-total/square-root comparison; all scores use sigma=3 min.",
                }
            )
    critical_c_id = "case_03_same_pooled_phase_different_realizations"
    critical_c_reference = reference_results[critical_c_id]["score"]
    critical_c_rearranged = primary_results[critical_c_id]["score"]
    rows.append(
        {
            "parameter_family": "stronger_critical_c",
            "parameter": "exact_pooled_phase_invariant",
            "case_id": critical_c_id,
            "value": all(pooled_invariants[critical_c_id].values()),
            "raw_score": critical_c_rearranged,
            "reference_score": critical_c_reference,
            "reference_minus_rearranged": critical_c_reference - critical_c_rearranged,
            "notes": "Exact sorted pooled phase equality checked separately for every behavior.",
        }
    )
    boundary_order_id = "case_31_boundary_crossing_order"
    boundary_monotone = primary_results[boundary_order_id]["score"]
    boundary_free = primary_free_results[boundary_order_id]["score"]
    rows.append(
        {
            "parameter_family": "boundary_order_test",
            "parameter": "0_24_phase_cut",
            "case_id": boundary_order_id,
            "raw_score": boundary_monotone,
            "order_free_score": boundary_free,
            "d_order": boundary_free - boundary_monotone,
            "notes": "Sequence-order test with events crossing the 24-to-0 phase boundary.",
        }
    )
    common_origin = matching_invariants["common_phase_origin"]
    rows.append(
        {
            "parameter_family": "matching_invariant",
            "parameter": "common_phase_origin",
            "case_id": "baseline_pair",
            "value": common_origin["pass"],
            "raw_score": common_origin["original_score"],
            "common_shifted_score": common_origin["shifted_score"],
            "difference": common_origin["difference"],
            "notes": "Same phase offset added to both cycles; event phases remain circularly comparable.",
        }
    )
    symmetry = matching_invariants["symmetry"]
    rows.append(
        {
            "parameter_family": "matching_invariant",
            "parameter": "score_symmetry",
            "case_id": "case_14_one_inserted_event_pair",
            "value": symmetry["pass"],
            "raw_score": symmetry["score_ab"],
            "reverse_score": symmetry["score_ba"],
            "difference": symmetry["difference"],
            "notes": "S(A,B) versus S(B,A) using the cyclic monotone matcher.",
        }
    )

    critical_ids = (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    )
    null_specs = (
        ("complete_label", None),
        ("circular_rotation", None),
        ("marked_block", 2),
        ("marked_block", 3),
        ("marked_block", 5),
    )
    for family, block_size in null_specs:
        for case_id in critical_ids:
            null_mean, null_sd = null_results[(case_id, family, block_size)]
            observed = primary_results[case_id]["score"]
            rows.append(
                {
                    "parameter_family": "null_family",
                    "parameter": family,
                    "case_id": case_id,
                    "delta_minutes": "",
                    "value": "",
                    "raw_score": observed,
                    "null_mean": null_mean,
                    "corrected_score": chance_correct(observed, null_mean),
                    "rearing_credit_fraction": primary_results[case_id]["rearing_credit_fraction"],
                    "null_sd": null_sd,
                    "marked_block_size": block_size if family == "marked_block" else "",
                    "notes": "Animal-level mean adjacent-cycle null; no negative truncation.",
                }
            )
    return pd.DataFrame(rows)


def write_figure(
    output_dir: Path,
    metric_cache: dict[tuple[str, float, str, str], dict],
    primary_results: dict[str, dict],
    primary_free_results: dict[str, dict],
    null_results: dict[tuple[str, str, int | None], tuple[float, float]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    jitter_ids = (
        "case_04_30_second_jitter",
        "case_05_1_minute_jitter",
        "case_06_2_to_5_minute_jitter",
        "case_07_10_minute_displacement",
    )
    jitter_labels = ("30 s", "1 min", "2–5 min", "10+ min")
    jitter_scores = [primary_results[case_id]["score"] for case_id in jitter_ids]
    axes[0, 0].plot(jitter_labels, jitter_scores, marker="o", color="#315d8c")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Timing displacement")
    axes[0, 0].set_ylabel("Mean adjacent COMBA")
    axes[0, 0].grid(axis="y", alpha=0.25)

    weighting_labels = {"raw_event": "raw", "equal_total": "equal total", "sqrt_count": "sqrt count"}
    x = np.arange(len(WEIGHTING_RULES))
    rearing_fractions = [
        metric_cache[("case_19_frequency_imbalance", 3.0, weighting, "monotone")]["rearing_credit_fraction"]
        for weighting in WEIGHTING_RULES
    ]
    axes[0, 1].bar(x, rearing_fractions, color=("#9c5b3d", "#c18f2f", "#4b8063"))
    axes[0, 1].set_xticks(x, [weighting_labels[weighting] for weighting in WEIGHTING_RULES])
    axes[0, 1].set_ylim(0, max(0.55, max(rearing_fractions) * 1.25))
    axes[0, 1].set_title("Frequency imbalance: rearing credit")
    axes[0, 1].set_ylabel("Fraction of matched credit")
    axes[0, 1].grid(axis="y", alpha=0.25)

    order_ids = ("case_16_adjacent_order_swap", "case_17_local_order_scrambling")
    order_labels = ("adjacent swap", "local scrambling")
    order_x = np.arange(len(order_ids))
    monotone_scores = [primary_results[case_id]["score"] for case_id in order_ids]
    free_scores = [primary_free_results[case_id]["score"] for case_id in order_ids]
    axes[1, 0].bar(order_x - 0.18, monotone_scores, width=0.36, label="monotone", color="#315d8c")
    axes[1, 0].bar(order_x + 0.18, free_scores, width=0.36, label="order-free", color="#c18f2f")
    axes[1, 0].set_xticks(order_x, order_labels)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Order constraint")
    axes[1, 0].set_ylabel("Mean adjacent COMBA")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    null_labels = ("complete", "rotation", "block 3")
    null_specs = (("complete_label", None), ("circular_rotation", None), ("marked_block", 3))
    null_ids = (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    )
    null_x = np.arange(len(null_ids))
    width = 0.25
    colors = ("#315d8c", "#9c5b3d", "#4b8063")
    for offset, ((family, block_size), label, color) in enumerate(zip(null_specs, null_labels, colors)):
        values = [
            chance_correct(
                primary_results[case_id]["score"],
                null_results[(case_id, family, block_size)][0],
            )
            for case_id in null_ids
        ]
        axes[1, 1].bar(null_x + (offset - 1) * width, values, width=width, label=label, color=color)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(null_x, ("perfect", "Critical A", "syntax", "stable circadian"), rotation=15)
    axes[1, 1].set_title("Null-corrected recurrence")
    axes[1, 1].set_ylabel("R = (obs − null)/(1 − null)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Synthetic validation of provisional COMBA", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "comba_synthetic_validation.png", dpi=180)
    plt.close(fig)


def kernel_acceptance() -> tuple[bool, list[int]]:
    accepted = []
    for sigma in SIGMA_MINUTES:
        one_minute = gaussian_credit(1.0 / 60.0, sigma)
        five_minutes = gaussian_credit(5.0 / 60.0, sigma)
        ten_minutes = gaussian_credit(10.0 / 60.0, sigma)
        if one_minute >= 0.80 and 0.05 <= five_minutes <= 0.75 and ten_minutes <= 0.20:
            accepted.append(sigma)
    return bool(accepted), accepted


def write_acceptance_summary(
    output_dir: Path,
    cases: Sequence[dict],
    case_status: dict[str, str],
    case_reason: dict[str, str],
    primary_results: dict[str, dict],
    primary_free_results: dict[str, dict],
    reference_results: dict[str, dict],
    pooled_invariants: dict[str, dict[str, bool]],
    matching_invariants: dict[str, dict[str, float | bool]],
    null_results: dict[tuple[str, str, int | None], tuple[float, float]],
    metric_cache: dict[tuple[str, float, str, str], dict],
    n_replicates: int,
) -> str:
    kernel_ok, accepted_sigmas = kernel_acceptance()
    order_stress = ("case_16_adjacent_order_swap", "case_17_local_order_scrambling")
    order_d = {
        case_id: primary_free_results[case_id]["score"] - primary_results[case_id]["score"]
        for case_id in order_stress
    }
    insertion_retention = downstream_retention(
        primary_results["case_14_one_inserted_event"], 13.55
    )
    deletion_retention = downstream_retention(
        primary_results["case_15_one_deleted_event"], 13.20
    )

    weighting_fractions = {
        weighting: metric_cache[("case_19_frequency_imbalance", 3.0, weighting, "monotone")]["rearing_credit_fraction"]
        for weighting in WEIGHTING_RULES
    }
    weighting_ok = (
        weighting_fractions["raw_event"] < weighting_fractions["sqrt_count"] < weighting_fractions["equal_total"]
        and weighting_fractions["sqrt_count"] >= 0.05
        and primary_results["case_11_stable_composition_change"]["score"] >= 0.95
        and primary_results["case_12_reproducibly_increased_bout_number"]["score"] >= 0.95
    )

    null_checks = {}
    for case_id in (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    ):
        corrected = chance_correct(
            primary_results[case_id]["score"],
            null_results[(case_id, "marked_block", 3)][0],
        )
        null_checks[case_id] = corrected
    null_ok = (
        null_checks["case_01_perfect_repeat"] >= 0.70
        and abs(null_checks["case_02_randomized_behavior_identity"]) <= 0.10
        and abs(null_checks["case_22_generic_local_syntax"]) <= 0.10
        and null_checks["case_23_stable_circadian_local_variability"] >= 0.20
    )

    cascade_ok = insertion_retention >= 0.90 and deletion_retention >= 0.90
    order_ok = order_d["case_16_adjacent_order_swap"] >= 0.01 and order_d["case_17_local_order_scrambling"] >= 0.05 and cascade_ok
    critical_c_id = "case_03_same_pooled_phase_different_realizations"
    critical_c_reference = reference_results[critical_c_id]["score"]
    critical_c_rearranged = primary_results[critical_c_id]["score"]
    critical_c_reduction = critical_c_reference - critical_c_rearranged
    critical_c_invariant = all(pooled_invariants[critical_c_id].values())
    critical_c_ok = (
        critical_c_invariant
        and critical_c_reference >= 0.95
        and critical_c_rearranged <= 0.70
        and critical_c_reduction >= 0.30
    )
    boundary_order_id = "case_31_boundary_crossing_order"
    boundary_monotone = primary_results[boundary_order_id]["score"]
    boundary_free = primary_free_results[boundary_order_id]["score"]
    boundary_d_order = boundary_free - boundary_monotone
    boundary_order_ok = boundary_monotone >= 0.85 and boundary_d_order <= 0.05
    common_origin = matching_invariants["common_phase_origin"]
    common_origin_ok = bool(common_origin["pass"])
    symmetry = matching_invariants["symmetry"]
    symmetry_ok = bool(symmetry["pass"])
    global_shift_score = primary_results["case_25_cycle_specific_global_phase_shift"]["score"]
    global_shift_ok = global_shift_score < 0.98
    failed_cases = [case_id for case_id, status in case_status.items() if status == "fail"]
    ambiguous_cases = [case_id for case_id, status in case_status.items() if status == "ambiguous"]
    all_case_ok = not failed_cases and not ambiguous_cases
    criteria = (
        kernel_ok,
        weighting_ok,
        null_ok,
        order_ok,
        critical_c_ok,
        boundary_order_ok,
        common_origin_ok,
        symmetry_ok,
        global_shift_ok,
    )
    if all_case_ok and all(criteria):
        recommendation = "GO"
        overall = "The predefined synthetic acceptance criteria were met."
    elif any(not condition for condition in criteria) or failed_cases:
        recommendation = "MODIFY"
        overall = "At least one predefined synthetic acceptance criterion failed or was not robust."
    else:
        recommendation = "MODIFY"
        overall = "The central cases were not decisively classified across all predefined checks."

    case_by_id = {case["case_id"]: case for case in cases}
    lines = [
        "Synthetic validation of provisional COMBA",
        "==========================================",
        "",
        "Scope: known non-rest bout events only; no real CBAS data, genotype labels, classifier simulation, or production scripts were used.",
        f"Null replicates per reported null condition: {n_replicates}",
        "",
        f"Overall conclusion: {overall}",
        f"Recommendation: {recommendation}",
        "",
        "Truth-case assessment",
        "---------------------",
        "Clearly passed:",
    ]
    passed_cases = [case_id for case_id, status in case_status.items() if status == "pass"]
    if passed_cases:
        lines.extend(f"- {case_id}: {case_reason[case_id]}" for case_id in passed_cases)
    else:
        lines.append("- none")
    lines.append("Failed:")
    if failed_cases:
        lines.extend(f"- {case_id}: {case_reason[case_id]}" for case_id in failed_cases)
    else:
        lines.append("- none")
    lines.append("Ambiguous:")
    if ambiguous_cases:
        lines.extend(f"- {case_id}: {case_reason[case_id]}" for case_id in ambiguous_cases)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Monotone versus order-free matching",
            "------------------------------------",
            f"Case 16 D_order = {order_d['case_16_adjacent_order_swap']:.4f}; Case 17 D_order = {order_d['case_17_local_order_scrambling']:.4f}.",
            f"Downstream retention after the inserted event: {insertion_retention:.3f}; after the deleted event: {deletion_retention:.3f}.",
            "The order test " + ("passed without evidence of cascading mismatch." if order_ok else "did not meet the predefined order criteria; inspect the reported D_order and retention values."),
            "",
            "Stronger Critical C",
            "--------------------",
            "Exact pooled per-behavior phase equality: " + ("PASS" if critical_c_invariant else "FAIL"),
            f"Reference recurrence = {critical_c_reference:.3f}; rearranged recurrence = {critical_c_rearranged:.3f}; reduction = {critical_c_reduction:.3f}.",
            "The stronger Critical C test " + ("passed." if critical_c_ok else "failed or was ambiguous."),
            "",
            "Boundary-crossing sequence order",
            "--------------------------------",
            f"Monotone score = {boundary_monotone:.3f}; order-free score = {boundary_free:.3f}; D_order = {boundary_d_order:.3f}.",
            "The boundary-order test " + ("passed." if boundary_order_ok else "failed; the current monotone implementation was not changed."),
            "",
            "Common phase-origin and symmetry invariants",
            "--------------------------------------------",
            f"Common offset = {common_origin['offset_hours']:.2f} h; original score = {common_origin['original_score']:.12f}; shifted score = {common_origin['shifted_score']:.12f}; difference = {common_origin['difference']:.3e}.",
            "Common-phase-origin invariance: " + ("PASS" if common_origin_ok else "FAIL"),
            f"Symmetry check: S(A,B) = {symmetry['score_ab']:.12f}; S(B,A) = {symmetry['score_ba']:.12f}; difference = {symmetry['difference']:.3e}.",
            "Score symmetry: " + ("PASS" if symmetry_ok else "FAIL"),
            f"Global phase-shift sensitivity remains active: Case 25 score = {global_shift_score:.3f}.",
            "",
            "Single-Gaussian sigma",
            "---------------------",
            f"Kernel values are tabulated at 30 seconds, 1, 2, 5, 10, and 30 minutes for sigma values {SIGMA_MINUTES}.",
            f"Sigma values satisfying the predefined qualitative kernel criteria (high at 1 minute, partial at 5 minutes, low at 10 minutes): {accepted_sigmas or 'none'}.",
            "The sigma test " + ("passed with a narrow defensible range." if kernel_ok else "was parameter-sensitive or failed to identify a defensible range."),
            "No parameter was selected by genotype separation; this task contains no genotype labels.",
            "",
            "Weighting",
            "---------",
            "Case 19 rearing matched-credit fractions: "
            + ", ".join(f"{weighting}={value:.3f}" for weighting, value in weighting_fractions.items()),
            "Square-root weighting " + ("met the predefined middle-ground checks." if weighting_ok else "did not meet all predefined middle-ground checks."),
            "Stable increased-count cases and random-extra-bout scores are included in comba_parameter_summary.csv.",
            "",
            "Null model",
            "----------",
            "Marked-block size sensitivity was evaluated at maximum block sizes 2, 3, and 5; block permutations were independent by cycle and used the same eligible event slots after missingness.",
            "Marked-block size 3 corrected scores for the critical cases: "
            + "; ".join(f"{case_id}={value:.3f}" for case_id, value in null_checks.items()),
            "The null test " + ("passed the critical A, generic-syntax, perfect-repeat, and stable-circadian checks." if null_ok else "did not pass all critical null checks."),
            "Complete label permutation, circular mark rotation, and all marked-block sizes are reported in the parameter summary and critical-case rows.",
            "",
            "Unresolved / limitations",
            "------------------------",
            "- These cases validate the event-level provisional metric only; classifier confidence, bout segmentation, bridging, and real-data coverage remain untested.",
            "- Qualitative truth cases use transparent fixed thresholds stated in the script; borderline cases are retained as ambiguous rather than rationalized.",
            "- This script stops at synthetic validation and does not implement production COMBA.",
            "",
            "Case notes",
            "----------",
        ]
    )
    for case_id in ambiguous_cases + failed_cases:
        lines.append(f"- {case_id}: {case_by_id[case_id]['notes']}")

    summary = "\n".join(lines) + "\n"
    (output_dir / "comba_acceptance_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic event-level validation of provisional COMBA."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Synthetic output directory (default: the requested COMBA validation folder).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed seed for synthetic perturbations; output remains deterministic.",
    )
    parser.add_argument(
        "--null-reps",
        type=int,
        default=NULL_REPLICATES,
        help="Null replicates per reported null condition (default: 2000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.null_reps <= 0:
        raise ValueError("--null-reps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.seed)
    cases_by_id = {case["case_id"]: case for case in cases}
    matching_invariants = run_matching_invariants(cases_by_id)
    metric_cache: dict[tuple[str, float, str, str], dict] = {}
    for case in cases:
        case_id = case["case_id"]
        for sigma in SIGMA_MINUTES:
            for weighting in (PRIMARY_WEIGHTING,):
                metric_cache[(case_id, float(sigma), weighting, "monotone")] = evaluate_animal(
                    case, sigma, weighting, "monotone"
                )
        for weighting in WEIGHTING_RULES:
            metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), weighting, "monotone")] = evaluate_animal(
                case, PRIMARY_SIGMA_MINUTES, weighting, "monotone"
            )
        metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), PRIMARY_WEIGHTING, "order_free")] = evaluate_animal(
            case, PRIMARY_SIGMA_MINUTES, PRIMARY_WEIGHTING, "order_free"
        )

    primary_results = {
        case_id: metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), PRIMARY_WEIGHTING, "monotone")]
        for case_id in cases_by_id
    }
    primary_free_results = {
        case_id: metric_cache[(case_id, float(PRIMARY_SIGMA_MINUTES), PRIMARY_WEIGHTING, "order_free")]
        for case_id in cases_by_id
    }
    reference_results: dict[str, dict] = {}
    pooled_invariants: dict[str, dict[str, bool]] = {}
    for case in cases:
        if "reference_cycles" not in case:
            continue
        reference_case = dict(case)
        reference_case["cycles"] = case["reference_cycles"]
        reference_case["missing_intervals"] = [
            [] for _ in case["reference_cycles"]
        ]
        reference_results[case["case_id"]] = evaluate_animal(
            reference_case,
            PRIMARY_SIGMA_MINUTES,
            PRIMARY_WEIGHTING,
            "monotone",
        )
        pooled_invariants[case["case_id"]] = pooled_phase_equality_by_behavior(
            case["reference_cycles"],
            case["cycles"],
        )
        if not all(pooled_invariants[case["case_id"]].values()):
            raise AssertionError(
                f"Pooled phase invariant failed for {case['case_id']}"
            )

    critical_ids = (
        "case_01_perfect_repeat",
        "case_02_randomized_behavior_identity",
        "case_22_generic_local_syntax",
        "case_23_stable_circadian_local_variability",
    )
    null_results: dict[tuple[str, str, int | None], tuple[float, float]] = {}
    for case_id in critical_ids:
        for family, block_size in (
            ("complete_label", None),
            ("circular_rotation", None),
            ("marked_block", 2),
            ("marked_block", 3),
            ("marked_block", 5),
        ):
            null_results[(case_id, family, block_size)] = null_mean_for_case(
                cases_by_id[case_id],
                PRIMARY_SIGMA_MINUTES,
                PRIMARY_WEIGHTING,
                family,
                block_size,
                args.null_reps,
            )

    preferred_corrected = {
        case_id: chance_correct(
            primary_results[case_id]["score"],
            null_results[(case_id, "marked_block", 3)][0],
        )
        for case_id in critical_ids
    }
    case_status = {}
    case_reason = {}
    for case in cases:
        status, reason = status_for_case(
            case["case_id"],
            primary_results[case["case_id"]],
            primary_free_results[case["case_id"]],
            primary_results,
            preferred_corrected,
            reference_results,
            pooled_invariants,
        )
        case_status[case["case_id"]] = status
        case_reason[case["case_id"]] = reason

    truth_rows = make_truth_rows(
        cases,
        metric_cache,
        primary_results,
        primary_free_results,
        preferred_corrected,
        reference_results,
        pooled_invariants,
    )
    append_null_rows(
        truth_rows,
        cases_by_id,
        primary_results,
        null_results,
        case_status,
        case_reason,
    )
    truth_frame = pd.DataFrame(truth_rows)
    truth_frame.to_csv(output_dir / "comba_truth_case_results.csv", index=False)

    parameter_frame = make_parameter_summary(
        metric_cache,
        primary_results,
        primary_free_results,
        reference_results,
        pooled_invariants,
        matching_invariants,
        null_results,
    )
    parameter_frame.to_csv(output_dir / "comba_parameter_summary.csv", index=False)

    write_figure(
        output_dir,
        metric_cache,
        primary_results,
        primary_free_results,
        null_results,
    )
    summary = write_acceptance_summary(
        output_dir,
        cases,
        case_status,
        case_reason,
        primary_results,
        primary_free_results,
        reference_results,
        pooled_invariants,
        matching_invariants,
        null_results,
        metric_cache,
        args.null_reps,
    )

    print(summary)
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
