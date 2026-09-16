"""Run Gate 1B on fixed imperfect-recurrence synthetic pairs."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import practical_comba_validation as scorer
from comba_density_stress_test import (
    BEHAVIORS,
    PHASE_CENTERS_HOURS,
    PHASE_TEMPLATE_SD_HOURS,
    JITTER_SD_MINUTES,
    RETAINED_FRACTION,
    REPLACEMENT_FRACTION,
    make_fixed_imperfect_pair,
)


PERIOD_HOURS = 24.0
JITTER_BIN_MINUTES = 30.0
JITTER_BIN_HOURS = JITTER_BIN_MINUTES / 60.0
JITTER_BIN_COUNT = 48
SURROGATES_PER_PAIR = 50
OBSERVED_PAIRS_PER_DENSITY = 10
DENSITY_LEVELS = (80, 400, 1600, 8000)
RANDOM_SEED = 20260915

Event = tuple[str, float]

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "CBAS_Analysis_Data"
    / "COMBA_Conditional_Jitter_Fixed_Recurrence_Density_Screen"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal synthetic conditional-jitter density screen."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the density-screen outputs.",
    )
    return parser.parse_args()


def validate_frozen_scorer() -> None:
    if not scorer.run_implementation_invariants():
        raise AssertionError("Frozen practical COMBA invariants failed")
    if scorer.SIGMA_MINUTES != 3.0:
        raise AssertionError(f"Unexpected sigma: {scorer.SIGMA_MINUTES}")
    if scorer.MATCH_CUTOFF_MINUTES != 30.0:
        raise AssertionError(f"Unexpected cutoff: {scorer.MATCH_CUTOFF_MINUTES}")
    if tuple(scorer.ORIGINS_HOURS) != (0.0, 8.0, 16.0):
        raise AssertionError(f"Unexpected origins: {scorer.ORIGINS_HOURS}")


def validate_fixed_recurrence_contract() -> None:
    """Check that the existing synthetic generator has the frozen parameters."""

    if len(BEHAVIORS) != 8:
        raise AssertionError(f"Unexpected behavior count: {len(BEHAVIORS)}")
    if not np.isclose(RETAINED_FRACTION, 0.70, rtol=0.0, atol=0.0):
        raise AssertionError(f"Unexpected retained fraction: {RETAINED_FRACTION}")
    if not np.isclose(REPLACEMENT_FRACTION, 0.30, rtol=0.0, atol=0.0):
        raise AssertionError(
            f"Unexpected replacement fraction: {REPLACEMENT_FRACTION}"
        )
    if not np.isclose(JITTER_SD_MINUTES, 5.0, rtol=0.0, atol=0.0):
        raise AssertionError(f"Unexpected retained-event jitter SD: {JITTER_SD_MINUTES}")
    if not np.isclose(PHASE_TEMPLATE_SD_HOURS, 2.0, rtol=0.0, atol=0.0):
        raise AssertionError(
            f"Unexpected phase-template SD: {PHASE_TEMPLATE_SD_HOURS}"
        )
    expected_centers = (1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5)
    actual_centers = tuple(PHASE_CENTERS_HOURS[behavior] for behavior in BEHAVIORS)
    if actual_centers != expected_centers:
        raise AssertionError(f"Unexpected phase-template centers: {actual_centers}")


def _interval_index(phase_hours: float) -> int:
    if not 0.0 <= phase_hours < PERIOD_HOURS:
        raise ValueError(f"Phase is outside CT0-24: {phase_hours}")
    return min(int(np.floor(phase_hours / JITTER_BIN_HOURS)), JITTER_BIN_COUNT - 1)


def conditional_jitter_cycle(
    events: Sequence[Event],
    rng: np.random.Generator,
) -> tuple[Event, ...]:
    """Jitter each onset uniformly within its original 30-minute CT interval."""

    surrogate: list[Event] = []
    for behavior, phase_hours in events:
        interval = _interval_index(float(phase_hours))
        lower = interval * JITTER_BIN_HOURS
        surrogate_phase = lower + float(rng.uniform(0.0, JITTER_BIN_HOURS))
        surrogate.append((behavior, surrogate_phase))
    return tuple(surrogate)


def verify_conditional_jitter_contract(
    source: Sequence[Event],
    surrogate: Sequence[Event],
) -> None:
    """Fail loudly if one surrogate violates any preserved-count contract."""

    if len(source) != len(surrogate):
        raise AssertionError("Surrogate total event count changed")
    if [behavior for behavior, _ in source] != [behavior for behavior, _ in surrogate]:
        raise AssertionError("Surrogate behavior labels changed")
    if Counter(behavior for behavior, _ in source) != Counter(
        behavior for behavior, _ in surrogate
    ):
        raise AssertionError("Surrogate behavior totals changed")

    source_interval_counts = Counter(
        (behavior, _interval_index(float(phase))) for behavior, phase in source
    )
    surrogate_interval_counts = Counter(
        (behavior, _interval_index(float(phase)))
        for behavior, phase in surrogate
    )
    if source_interval_counts != surrogate_interval_counts:
        raise AssertionError("Surrogate behavior-by-30-minute counts changed")

    for (source_behavior, source_phase), (surrogate_behavior, surrogate_phase) in zip(
        source, surrogate
    ):
        if source_behavior != surrogate_behavior:
            raise AssertionError("Surrogate event label changed")
        interval = _interval_index(float(source_phase))
        lower = interval * JITTER_BIN_HOURS
        upper = lower + JITTER_BIN_HOURS
        if not lower <= surrogate_phase < upper:
            raise AssertionError("Surrogate phase crossed its source interval")
        if not 0.0 <= surrogate_phase < PERIOD_HOURS:
            raise AssertionError("Surrogate phase left CT0-24")


def score_pair_with_conditional_jitter(
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return observed score, null mean, dynamic range, and corrected score."""

    observed = scorer.practical_score_pair(events_a, events_b)
    _validate_observed_score(observed)
    observed_origin_scores = np.asarray(observed.origin_scores, dtype=float)
    if not np.isclose(
        observed.score,
        np.median(observed_origin_scores),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Observed pair score is not the origin median")

    surrogate_scores: list[float] = []
    for _ in range(SURROGATES_PER_PAIR):
        surrogate_a = conditional_jitter_cycle(events_a, rng)
        surrogate_b = conditional_jitter_cycle(events_b, rng)
        verify_conditional_jitter_contract(events_a, surrogate_a)
        verify_conditional_jitter_contract(events_b, surrogate_b)
        surrogate_result = scorer.practical_score_pair(surrogate_a, surrogate_b)
        surrogate_origins = np.asarray(surrogate_result.origin_scores, dtype=float)
        if surrogate_origins.shape != (3,) or not np.all(np.isfinite(surrogate_origins)):
            raise AssertionError("Surrogate origin score is missing or non-finite")
        if not np.isclose(
            surrogate_result.score,
            np.median(surrogate_origins),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("Surrogate score is not the origin median")
        if not 0.0 <= surrogate_result.score <= 1.0 + 1e-12:
            raise AssertionError("Surrogate COMBA is outside [0, 1]")
        surrogate_scores.append(float(surrogate_result.score))

    mu_0 = float(np.mean(np.asarray(surrogate_scores, dtype=float)))
    dynamic_range = 1.0 - mu_0
    if not np.isfinite(dynamic_range) or dynamic_range == 0.0:
        raise AssertionError("Conditional-jitter dynamic range is unusable")
    corrected = (float(observed.score) - mu_0) / dynamic_range
    return {
        "S_obs": float(observed.score),
        "mu_0": mu_0,
        "R_p": float(corrected),
        "one_minus_mu_0": dynamic_range,
    }


def _validate_observed_score(result: scorer.PairScore) -> None:
    origin_scores = np.asarray(result.origin_scores, dtype=float)
    if origin_scores.shape != (3,):
        raise AssertionError("Observed scorer did not return three origin scores")
    if not np.all(np.isfinite(origin_scores)) or not np.isfinite(result.score):
        raise AssertionError("Observed COMBA is non-finite")
    if np.any(origin_scores < 0.0) or np.any(origin_scores > 1.0 + 1e-12):
        raise AssertionError("Observed COMBA origin score is outside [0, 1]")
    if not 0.0 <= result.score <= 1.0 + 1e-12:
        raise AssertionError("Observed COMBA is outside [0, 1]")
    if not np.isclose(result.score, np.median(origin_scores), rtol=0.0, atol=1e-12):
        raise AssertionError("Observed pair score is not the origin median")


def _summarize_density_screen(replicates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        replicates.groupby("Density_N", sort=False)
        .agg(
            Mean_S_obs=("S_obs", "mean"),
            SD_S_obs=("S_obs", "std"),
            Mean_mu_0=("mu_0", "mean"),
            SD_mu_0=("mu_0", "std"),
            Mean_R_p=("R_p", "mean"),
            SD_R_p=("R_p", "std"),
            Median_R_p=("R_p", "median"),
            Mean_one_minus_mu_0=("One_minus_mu_0", "mean"),
            Minimum_one_minus_mu_0=("One_minus_mu_0", "min"),
        )
        .reset_index()
    )
    return summary


def _plot_density_screen(
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(
        replicates["Density_N"],
        replicates["R_p"],
        alpha=0.3,
        s=24,
        label="Observed pairs",
    )
    axis.errorbar(
        summary["Density_N"],
        summary["Mean_R_p"],
        yerr=summary["SD_R_p"],
        fmt="o",
        color="black",
        capsize=3,
        label="Density mean ± SD",
    )
    axis.axhline(0.0, color="gray", linewidth=1.0)
    axis.set_xlabel("Density_N")
    axis.set_ylabel("Corrected COMBA (R_p)")
    axis.set_title("Fixed imperfect recurrence — conditional-jitter corrected COMBA")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_density_screen(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only Gate 1B's fixed imperfect-recurrence density screen."""

    validate_frozen_scorer()
    validate_fixed_recurrence_contract()
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    runtime_by_density: dict[int, float] = {}
    started = time.perf_counter()

    for density_n in DENSITY_LEVELS:
        density_started = time.perf_counter()
        for replicate in range(1, OBSERVED_PAIRS_PER_DENSITY + 1):
            events_a, events_b = make_fixed_imperfect_pair(density_n, rng)
            expected_behavior_count = density_n // len(BEHAVIORS)
            expected_behavior_counts = Counter(
                {behavior: expected_behavior_count for behavior in BEHAVIORS}
            )
            for events in (events_a, events_b):
                if len(events) != density_n:
                    raise AssertionError("Synthetic cycle has the wrong event density")
                if Counter(behavior for behavior, _ in events) != expected_behavior_counts:
                    raise AssertionError("Synthetic behavior composition changed")
                if any(
                    not 0.0 <= float(phase) < PERIOD_HOURS
                    for _, phase in events
                ):
                    raise AssertionError("Synthetic phase left CT0-24")
            result = score_pair_with_conditional_jitter(events_a, events_b, rng)
            rows.append(
                {
                    "Density_N": density_n,
                    "Replicate": replicate,
                    "S_obs": result["S_obs"],
                    "mu_0": result["mu_0"],
                    "R_p": result["R_p"],
                    "One_minus_mu_0": result["one_minus_mu_0"],
                }
            )
            print(
                f"completed density={density_n} replicate={replicate}/{OBSERVED_PAIRS_PER_DENSITY}",
                flush=True,
            )
        runtime_by_density[density_n] = time.perf_counter() - density_started

    replicates = pd.DataFrame(rows)
    if len(replicates) != len(DENSITY_LEVELS) * OBSERVED_PAIRS_PER_DENSITY:
        raise AssertionError("Unexpected number of observed-pair rows")
    if not np.all(np.isfinite(replicates.to_numpy(dtype=float, na_value=np.nan))):
        raise AssertionError("Density screen produced a non-finite value")
    if np.any(replicates[["S_obs"]].to_numpy() < 0.0) or np.any(
        replicates[["S_obs"]].to_numpy() > 1.0 + 1e-12
    ):
        raise AssertionError("Raw observed COMBA is outside [0, 1]")

    summary = _summarize_density_screen(replicates)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(
        output_dir / "fixed_recurrence_density_screen_replicates.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "fixed_recurrence_density_screen_summary.csv",
        index=False,
    )
    _plot_density_screen(
        replicates,
        summary,
        output_dir / "fixed_recurrence_density_screen.png",
    )

    lines = [
        "Gate 1B: fixed imperfect recurrence — conditional-jitter corrected COMBA",
        "Condition: fixed imperfect recurrence",
        "Scorer function: practical_comba_validation.practical_score_pair",
        "Sigma (min): 3.0",
        "Candidate cutoff (min): 30.0",
        "Origins (h): 0.0, 8.0, 16.0",
        "Matching: same-behavior, one-to-one, strictly order-preserving maximum-weight chain",
        "Weighting: square-root count weighting",
        "Aggregation: median across origins",
        "Conditional-jitter: 48 fixed 30-minute intervals; independent uniform redraw within source interval",
        f"Retained fraction: {RETAINED_FRACTION}",
        f"Replacement fraction: {REPLACEMENT_FRACTION}",
        f"Retained-event jitter SD (min): {JITTER_SD_MINUTES}",
        f"Random seed: {RANDOM_SEED}",
        f"Density levels: {', '.join(str(value) for value in DENSITY_LEVELS)}",
        f"Observed pairs per density: {OBSERVED_PAIRS_PER_DENSITY}",
        f"Surrogates per observed pair: {SURROGATES_PER_PAIR}",
        f"Total observed-pair rows: {len(replicates)}",
        f"Phase-template centers (h): {PHASE_CENTERS_HOURS}",
        f"Phase-template wrapped-Gaussian SD (h): {PHASE_TEMPLATE_SD_HOURS}",
        f"Runtime by density (s): {runtime_by_density}",
        f"Total runtime (s): {time.perf_counter() - started:.6f}",
        "Contract checks: PASS",
        "No-event recurrence, alternate nulls, truth cases, correlations, or real-animal analysis run",
        "Gate result: review density-level corrected means; no automatic numeric threshold applied",
    ]
    (output_dir / "fixed_recurrence_density_screen_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return replicates, summary


def main() -> None:
    args = parse_args()
    replicates, summary = run_density_screen(args.output_dir)
    print(summary.to_string(index=False))
    print(f"replicate_rows={len(replicates)}")


if __name__ == "__main__":
    main()
