"""Stress-test the frozen practical COMBA scorer across event densities.

This script generates synthetic non-rest event pairs only.  It reuses
``practical_comba_validation.practical_score_pair`` and does not alter the
production scorer or any real-animal output.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import practical_comba_validation as scorer


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
PERIOD_HOURS = 24.0
DENSITY_LEVELS = (80, 240, 400, 800, 1600)
REPLICATES_PER_CELL = 50
RANDOM_SEED = 20260915

RETAINED_FRACTION = 0.70
REPLACEMENT_FRACTION = 0.30
JITTER_SD_MINUTES = 5.0

# One fixed wrapped-Gaussian phase template per behavior is used for both
# conditions and every density.  The centers cover the full circadian cycle.
PHASE_CENTERS_HOURS = {
    "eating": 1.5,
    "drinking": 4.5,
    "rearing": 7.5,
    "climbing": 10.5,
    "digging": 13.5,
    "nesting": 16.5,
    "grooming": 19.5,
    "locomotion": 22.5,
}
PHASE_TEMPLATE_SD_HOURS = 2.0

CONDITION_FIXED_IMPERFECT = "fixed_imperfect_recurrence"
CONDITION_NO_EVENT_RECURRENCE = "no_event_recurrence"
CONDITIONS = (CONDITION_FIXED_IMPERFECT, CONDITION_NO_EVENT_RECURRENCE)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "CBAS_Analysis_Data"
    / "COMBA_Density_Stress_Test"
)

Event = tuple[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the focused synthetic COMBA event-density stress test."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the requested stress-test outputs.",
    )
    return parser.parse_args()


def validate_frozen_scorer() -> None:
    """Check the fixed scorer contract before generating synthetic data."""

    if not scorer.run_implementation_invariants():
        raise AssertionError("Frozen practical scorer invariants failed")
    if scorer.SIGMA_MINUTES != 3.0:
        raise AssertionError(f"Unexpected scorer sigma: {scorer.SIGMA_MINUTES}")
    if scorer.MATCH_CUTOFF_MINUTES != 30.0:
        raise AssertionError(
            f"Unexpected scorer cutoff: {scorer.MATCH_CUTOFF_MINUTES}"
        )
    if tuple(scorer.ORIGINS_HOURS) != (0.0, 8.0, 16.0):
        raise AssertionError(f"Unexpected scorer origins: {scorer.ORIGINS_HOURS}")


def _retained_and_replacement_counts(density_n: int) -> tuple[int, int, int]:
    if density_n % len(BEHAVIORS) != 0:
        raise ValueError("Density must divide evenly across the eight behaviors")
    per_behavior = density_n // len(BEHAVIORS)
    retained = int(round(RETAINED_FRACTION * per_behavior))
    replacement = int(round(REPLACEMENT_FRACTION * per_behavior))
    if retained + replacement != per_behavior:
        raise AssertionError("Retained and replacement counts do not cover a cycle")
    if not np.isclose(retained / per_behavior, RETAINED_FRACTION):
        raise AssertionError("Density does not realize the fixed retained fraction")
    if not np.isclose(replacement / per_behavior, REPLACEMENT_FRACTION):
        raise AssertionError("Density does not realize the fixed replacement fraction")
    return per_behavior, retained, replacement


def sample_behavior_phases(
    behavior: str,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the fixed broad wrapped-Gaussian phase template."""

    center = PHASE_CENTERS_HOURS[behavior]
    return np.mod(
        rng.normal(center, PHASE_TEMPLATE_SD_HOURS, size=count),
        PERIOD_HOURS,
    )


def sample_cycle(density_n: int, rng: np.random.Generator) -> tuple[Event, ...]:
    """Generate one cycle with exact balanced behavior composition."""

    per_behavior, _, _ = _retained_and_replacement_counts(density_n)
    events: list[Event] = []
    for behavior in BEHAVIORS:
        phases = sample_behavior_phases(behavior, per_behavior, rng)
        events.extend((behavior, float(phase)) for phase in phases)
    events.sort(key=lambda event: event[1])
    return tuple(events)


def make_fixed_imperfect_pair(
    density_n: int,
    rng: np.random.Generator,
) -> tuple[tuple[Event, ...], tuple[Event, ...]]:
    """Make B by retaining/jittering A events and replacing the deletions."""

    cycle_a = sample_cycle(density_n, rng)
    per_behavior, retained, replacement = _retained_and_replacement_counts(density_n)
    phases_by_behavior: dict[str, list[float]] = defaultdict(list)
    for behavior, phase in cycle_a:
        phases_by_behavior[behavior].append(phase)

    jitter_sd_hours = JITTER_SD_MINUTES / 60.0
    cycle_b: list[Event] = []
    for behavior in BEHAVIORS:
        source_phases = np.asarray(phases_by_behavior[behavior], dtype=float)
        if len(source_phases) != per_behavior:
            raise AssertionError("Cycle A behavior composition is not balanced")
        retained_indices = rng.permutation(per_behavior)[:retained]
        retained_phases = np.mod(
            source_phases[retained_indices]
            + rng.normal(0.0, jitter_sd_hours, size=retained),
            PERIOD_HOURS,
        )
        cycle_b.extend((behavior, float(phase)) for phase in retained_phases)

        replacement_phases = sample_behavior_phases(behavior, replacement, rng)
        cycle_b.extend((behavior, float(phase)) for phase in replacement_phases)

    if len(cycle_b) != density_n:
        raise AssertionError("Cycle B has the wrong event density")
    cycle_b.sort(key=lambda event: event[1])
    return cycle_a, tuple(cycle_b)


def make_no_event_recurrence_pair(
    density_n: int,
    rng: np.random.Generator,
) -> tuple[tuple[Event, ...], tuple[Event, ...]]:
    """Generate A and B independently from the same fixed phase templates."""

    cycle_a = sample_cycle(density_n, rng)
    cycle_b = sample_cycle(density_n, rng)
    if cycle_a == cycle_b:
        raise AssertionError("Independent cycles unexpectedly matched exactly")
    return cycle_a, cycle_b


def _behavior_phases(events: Sequence[Event]) -> dict[str, list[float]]:
    phases_by_behavior: dict[str, list[float]] = defaultdict(list)
    for behavior, phase in events:
        phases_by_behavior[behavior].append(phase)
    return phases_by_behavior


def median_nearest_same_label_distance_minutes(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
) -> float:
    """Calculate the symmetric nearest same-label phase distance summary."""

    phases_a = _behavior_phases(events_a)
    phases_b = _behavior_phases(events_b)
    distances_hours: list[float] = []
    for behavior, phase in events_a:
        candidates = phases_b.get(behavior, [])
        if candidates:
            distances_hours.append(
                min(
                    scorer.circular_phase_distance(phase, candidate)
                    for candidate in candidates
                )
            )
    for behavior, phase in events_b:
        candidates = phases_a.get(behavior, [])
        if candidates:
            distances_hours.append(
                min(
                    scorer.circular_phase_distance(phase, candidate)
                    for candidate in candidates
                )
            )
    if not distances_hours:
        raise ValueError("No same-label candidate distances were available")
    return float(np.median(np.asarray(distances_hours, dtype=float)) * 60.0)


def _validate_pair_result(
    result: scorer.PairScore,
    events_a: Sequence[Event],
    events_b: Sequence[Event],
) -> None:
    origin_scores = np.asarray(result.origin_scores, dtype=float)
    if origin_scores.shape != (3,):
        raise AssertionError("The scorer did not return three origin scores")
    if not np.all(np.isfinite(origin_scores)) or not np.isfinite(result.score):
        raise AssertionError("Synthetic COMBA score is non-finite")
    if np.any(origin_scores < 0.0) or np.any(origin_scores > 1.0 + 1e-12):
        raise AssertionError("Synthetic origin score is outside [0, 1]")
    if not 0.0 <= result.score <= 1.0 + 1e-12:
        raise AssertionError("Synthetic pair score is outside [0, 1]")
    if not np.isclose(result.score, np.median(origin_scores), rtol=0.0, atol=1e-12):
        raise AssertionError("Pair COMBA is not the median of its origin scores")
    denominator = scorer.event_mass(events_a) + scorer.event_mass(events_b)
    if not np.allclose(result.origin_denominators, denominator, rtol=0.0, atol=1e-12):
        raise AssertionError("Origin denominators differ from the frozen event mass")


def generate_replicate_table() -> pd.DataFrame:
    """Generate and score all fixed-condition/density/replicate cells."""

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for density_n in DENSITY_LEVELS:
            for replicate in range(1, REPLICATES_PER_CELL + 1):
                if condition == CONDITION_FIXED_IMPERFECT:
                    cycle_a, cycle_b = make_fixed_imperfect_pair(density_n, rng)
                else:
                    cycle_a, cycle_b = make_no_event_recurrence_pair(density_n, rng)

                expected_count = density_n // len(BEHAVIORS)
                for events in (cycle_a, cycle_b):
                    counts = {behavior: 0 for behavior in BEHAVIORS}
                    for behavior, _ in events:
                        counts[behavior] += 1
                    if any(count != expected_count for count in counts.values()):
                        raise AssertionError("Synthetic behavior composition changed")

                result = scorer.practical_score_pair(cycle_a, cycle_b)
                _validate_pair_result(result, cycle_a, cycle_b)
                nearest_distance = median_nearest_same_label_distance_minutes(
                    cycle_a, cycle_b
                )
                rows.append(
                    {
                        "Condition": condition,
                        "Density_N": density_n,
                        "Replicate": replicate,
                        "COMBA": result.score,
                        "S_0h": result.origin_scores[0],
                        "S_8h": result.origin_scores[1],
                        "S_16h": result.origin_scores[2],
                        "Bouts_cycle_A": len(cycle_a),
                        "Bouts_cycle_B": len(cycle_b),
                        "Median_nearest_same_label_distance_min": nearest_distance,
                    }
                )

    table = pd.DataFrame(rows)
    expected_rows = len(CONDITIONS) * len(DENSITY_LEVELS) * REPLICATES_PER_CELL
    if len(table) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} replicates, found {len(table)}")
    cell_counts = table.groupby(["Condition", "Density_N"], sort=False).size()
    if not np.all(cell_counts.to_numpy() == REPLICATES_PER_CELL):
        raise AssertionError("A condition/density cell has the wrong replicate count")
    return table


def summarize_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        replicates.groupby(["Condition", "Density_N"], sort=False)
        .agg(
            Mean_COMBA=("COMBA", "mean"),
            SD_COMBA=("COMBA", "std"),
            Median_COMBA=("COMBA", "median"),
            Min_COMBA=("COMBA", "min"),
            Max_COMBA=("COMBA", "max"),
            Mean_nearest_same_label_distance_min=(
                "Median_nearest_same_label_distance_min",
                "mean",
            ),
        )
        .reset_index()
    )
    return summary


def _correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> tuple[float, float]:
    if method == "Pearson":
        result = stats.pearsonr(x, y)
    elif method == "Spearman":
        result = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def calculate_correlations(
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        replicate_subset = replicates[replicates["Condition"] == condition]
        mean_subset = summary[summary["Condition"] == condition]
        for observation_level, subset, x_column, y_column, note in (
            (
                "replicate-level",
                replicate_subset,
                "Density_N",
                "COMBA",
                "Descriptive only; replicate points share fixed generating parameters.",
            ),
            (
                "density-mean-level",
                mean_subset,
                "Density_N",
                "Mean_COMBA",
                "Descriptive only; five density-level means.",
            ),
        ):
            x = subset[x_column].to_numpy(dtype=float)
            y = subset[y_column].to_numpy(dtype=float)
            for method in ("Pearson", "Spearman"):
                correlation, p_value = _correlation(x, y, method)
                rows.append(
                    {
                        "Condition": condition,
                        "Observation_level": observation_level,
                        "X_variable": x_column,
                        "Y_variable": y_column,
                        "Method": method,
                        "Correlation": correlation,
                        "P_value": p_value,
                        "N": len(subset),
                        "Independence_note": note,
                    }
                )
    return pd.DataFrame(rows)


def create_density_plot(
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    condition: str,
    title: str,
    output_path: Path,
) -> None:
    replicate_subset = replicates[replicates["Condition"] == condition]
    mean_subset = summary[summary["Condition"] == condition]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(
        replicate_subset["Density_N"],
        replicate_subset["COMBA"],
        alpha=0.25,
        s=18,
        label="Replicates",
    )
    axis.errorbar(
        mean_subset["Density_N"],
        mean_subset["Mean_COMBA"],
        yerr=mean_subset["SD_COMBA"],
        fmt="o",
        color="black",
        capsize=3,
        label="Density mean ± SD",
    )
    axis.set_xlabel("Density_N")
    axis.set_ylabel("COMBA")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_run_summary(
    output_dir: Path,
    replicates: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    retained_counts = {
        density_n: _retained_and_replacement_counts(density_n)[1:]
        for density_n in DENSITY_LEVELS
    }
    lines = [
        "Synthetic practical COMBA event-density stress test",
        f"Output directory: {output_dir}",
        "Scorer function: practical_comba_validation.practical_score_pair",
        "Scorer sigma (min): 3.0",
        "Scorer match cutoff (min): 30.0",
        "Scorer origins (h): 0.0, 8.0, 16.0",
        "Scorer matching: same-behavior, one-to-one, strictly order-preserving maximum-weight chain",
        "Scorer weighting: square-root count weighting",
        "Scorer aggregation: median across origins",
        f"Density levels: {', '.join(str(value) for value in DENSITY_LEVELS)}",
        f"Replicates per condition/density cell: {REPLICATES_PER_CELL}",
        f"Random seed: {RANDOM_SEED}",
        "Condition 1: fixed imperfect recurrence",
        f"  retained fraction: {RETAINED_FRACTION:.2f}",
        f"  replacement fraction: {REPLACEMENT_FRACTION:.2f}",
        f"  retained-event jitter SD (min): {JITTER_SD_MINUTES:.1f}",
        "  replacement events: newly sampled from the fixed behavior phase template",
        "Condition 2: no event-level recurrence",
        "  cycle A and cycle B: independent draws from the same fixed behavior phase templates",
        f"  phase-template centers (h): {PHASE_CENTERS_HOURS}",
        f"  phase-template wrapped-Gaussian SD (h): {PHASE_TEMPLATE_SD_HOURS:.1f}",
        "  phase-template parameters are unchanged across density levels",
        "Behavior composition: exactly equal across all eight non-rest labels at every density",
        "No null correction, chance correction, regression, covariate adjustment, or genotype analysis",
        f"Total replicate rows: {len(replicates)}",
        f"Total runtime (s): {elapsed_seconds:.6f}",
        f"Retained/replacement counts by density (per behavior): {retained_counts}",
        "Validation: PASS",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    validate_frozen_scorer()
    replicates = generate_replicate_table()
    summary = summarize_replicates(replicates)
    correlations = calculate_correlations(replicates, summary)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(output_dir / "comba_density_replicates.csv", index=False)
    summary.to_csv(output_dir / "comba_density_summary.csv", index=False)
    correlations.to_csv(output_dir / "comba_density_correlations.csv", index=False)
    create_density_plot(
        replicates,
        summary,
        CONDITION_FIXED_IMPERFECT,
        "Fixed imperfect recurrence across event density",
        output_dir / "fixed_imperfect_recurrence_density.png",
    )
    create_density_plot(
        replicates,
        summary,
        CONDITION_NO_EVENT_RECURRENCE,
        "No event-level recurrence across event density",
        output_dir / "no_event_recurrence_density.png",
    )
    write_run_summary(output_dir, replicates, time.perf_counter() - started)
    return replicates, summary, correlations


def main() -> None:
    args = parse_args()
    replicates, summary, correlations = run(args.output_dir)
    print(summary.to_string(index=False))
    print(correlations.to_string(index=False))
    print(f"replicate_rows={len(replicates)}")


if __name__ == "__main__":
    main()
