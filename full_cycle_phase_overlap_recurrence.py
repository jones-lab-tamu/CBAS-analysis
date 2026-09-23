"""Measure full-data, unaligned phase-profile overlap across four cycles."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import circular_w1_cycle_instability as frozen_data


N_BINS = frozen_data.N_CT_BINS
N_CYCLES = frozen_data.N_CYCLES
CT_HOURS = frozen_data.CT_HOURS
BIN_WIDTH_HOURS = CT_HOURS / N_BINS
CT_BIN_CENTERS = (np.arange(N_BINS, dtype=float) + 0.5) * BIN_WIDTH_HOURS
PAIR_ORDERS = tuple(itertools.combinations(range(1, N_CYCLES + 1), 2))
KERNEL_OFFSETS = np.arange(-6, 7, dtype=int)
KERNEL_WEIGHTS = np.ones(len(KERNEL_OFFSETS), dtype=float)
KERNEL_WEIGHTS[[0, -1]] = 0.5
KERNEL_WEIGHTS /= KERNEL_WEIGHTS.sum()
NUMERICAL_ZERO_TOLERANCE = 8.0 * np.finfo(float).eps
PROBABILITY_TOLERANCE = 1e-12
OUTPUT_DIRNAME = "Full_Cycle_Phase_Overlap_Recurrence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cohort_root",
        nargs="?",
        type=Path,
        default=frozen_data.DEFAULT_COHORT_ROOT,
        help="Cohort_Data directory containing the frozen pilot animals.",
    )
    return parser.parse_args()


def smooth_circular_1h(values: np.ndarray) -> np.ndarray:
    """Apply the fixed 13-point, 1-hour trapezoidal kernel circularly."""
    values = np.asarray(values, dtype=float)
    if values.shape != (N_BINS,) or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite {N_BINS}-bin phase profile")
    if np.any(values < 0.0):
        raise ValueError("Phase occupancy must be nonnegative")
    smoothed = np.zeros(N_BINS, dtype=float)
    for offset, weight in zip(KERNEL_OFFSETS, KERNEL_WEIGHTS):
        smoothed += weight * np.roll(values, int(offset))
    return smoothed


def normalized_smoothed_profile(duration_occupancy: np.ndarray) -> np.ndarray | None:
    """Return the smoothed conditional profile, or None for zero duration."""
    occupancy = np.asarray(duration_occupancy, dtype=float)
    if occupancy.shape != (N_BINS,) or not np.isfinite(occupancy).all():
        raise ValueError(f"Expected one finite {N_BINS}-bin phase profile")
    if np.any(occupancy < 0.0):
        raise ValueError("Phase occupancy must be nonnegative")
    if float(occupancy.sum()) == 0.0:
        return None
    smoothed = smooth_circular_1h(occupancy)
    total = float(smoothed.sum())
    if total <= 0.0:
        raise RuntimeError("Positive occupancy vanished during smoothing")
    probability = smoothed / total
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise RuntimeError("Smoothed phase profile is not normalized")
    return probability


def circular_phase_and_R(probability: np.ndarray | None) -> tuple[float, float]:
    """Return the CT-hour circular mean and resultant length for a profile."""
    if probability is None:
        return float("nan"), float("nan")
    probability = np.asarray(probability, dtype=float)
    if (
        probability.shape != (N_BINS,)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.isclose(
            probability.sum(), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE
        )
    ):
        raise ValueError("Circular phase requires a normalized probability profile")
    angles = 2.0 * np.pi * CT_BIN_CENTERS / CT_HOURS
    resultant = complex(np.sum(probability * np.exp(1j * angles)))
    concentration = float(abs(resultant))
    if concentration <= NUMERICAL_ZERO_TOLERANCE:
        return float("nan"), concentration
    phase = float(
        np.mod(np.angle(resultant), 2.0 * np.pi) * CT_HOURS / (2.0 * np.pi)
    )
    return phase, concentration


def probability_overlap(first: np.ndarray | None, second: np.ndarray | None) -> float:
    """Shared probability mass; an undefined cycle yields an undefined pair."""
    if first is None or second is None:
        return float("nan")
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    for profile in (first, second):
        if (
            profile.shape != (N_BINS,)
            or not np.isfinite(profile).all()
            or np.any(profile < 0.0)
            or not np.isclose(
                profile.sum(), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE
            )
        ):
            raise ValueError("Overlap requires normalized probability profiles")
    score = float(np.minimum(first, second).sum())
    if score < -PROBABILITY_TOLERANCE or score > 1.0 + PROBABILITY_TOLERANCE:
        raise RuntimeError(f"Probability overlap is outside [0, 1]: {score}")
    return float(np.clip(score, 0.0, 1.0))


def circular_difference_hours(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return float("nan")
    signed = (float(first) - float(second) + CT_HOURS / 2.0) % CT_HOURS
    return abs(float(signed - CT_HOURS / 2.0))


def _unit_tests() -> None:
    sample = np.zeros(N_BINS, dtype=float)
    sample[17] = 0.4
    sample[143] = 0.6
    smoothed = smooth_circular_1h(sample)
    assert np.isclose(smoothed.sum(), sample.sum(), atol=1e-14)
    normalized = normalized_smoothed_profile(sample)
    assert normalized is not None and np.isclose(normalized.sum(), 1.0, atol=1e-14)

    midnight = np.zeros(N_BINS, dtype=float)
    midnight[0] = 1.0
    midnight_smoothed = smooth_circular_1h(midnight)
    assert midnight_smoothed[-1] > 0.0 and midnight_smoothed[1] > 0.0
    assert np.isclose(midnight_smoothed[-1], 1.0 / 12.0, atol=1e-14)
    next_to_midnight = np.roll(midnight, -1)
    wrapped_profile = normalized_smoothed_profile(midnight)
    wrapped_neighbor = normalized_smoothed_profile(next_to_midnight)
    assert wrapped_profile is not None and wrapped_neighbor is not None
    assert probability_overlap(wrapped_profile, wrapped_profile) == 1.0

    p = np.zeros(N_BINS, dtype=float)
    q = np.zeros(N_BINS, dtype=float)
    p[[3, 80, 201]] = [0.2, 0.5, 0.3]
    q[[3, 80, 202]] = [0.1, 0.6, 0.3]
    overlap = probability_overlap(p, q)
    assert np.isclose(overlap, 1.0 - 0.5 * np.abs(p - q).sum(), atol=1e-14)
    assert 0.0 <= overlap <= 1.0

    theta = 2.0 * np.pi * CT_BIN_CENTERS / CT_HOURS
    center_at_zero = np.zeros(N_BINS, dtype=float)
    center_at_zero[[0, -1]] = 0.5
    z = np.sum(center_at_zero * np.exp(1j * theta))
    phase, _resultant = circular_phase_and_R(center_at_zero)
    assert abs(z) > 0.0
    assert min(phase, CT_HOURS - phase) < BIN_WIDTH_HOURS / 100.0
    assert probability_overlap(wrapped_profile, wrapped_neighbor) < 1.0


def _wrapped_gaussian(center_h: float, sd_h: float) -> np.ndarray:
    difference = (CT_BIN_CENTERS - center_h + CT_HOURS / 2.0) % CT_HOURS
    difference -= CT_HOURS / 2.0
    values = np.exp(-0.5 * (difference / sd_h) ** 2)
    return values / values.sum()


def _synthetic_validation() -> list[dict[str, object]]:
    base = _wrapped_gaussian(7.0, 1.0)
    identical = probability_overlap(
        normalized_smoothed_profile(base), normalized_smoothed_profile(base)
    )
    shifted = probability_overlap(
        normalized_smoothed_profile(base),
        normalized_smoothed_profile(np.roll(base, 6)),
    )
    broader_raw = _wrapped_gaussian(7.0, 2.5)
    broadened = probability_overlap(
        normalized_smoothed_profile(base), normalized_smoothed_profile(broader_raw)
    )
    split_raw = 0.5 * _wrapped_gaussian(4.0, 1.0) + 0.5 * _wrapped_gaussian(10.0, 1.0)
    split = probability_overlap(
        normalized_smoothed_profile(base), normalized_smoothed_profile(split_raw)
    )
    broad_profile = normalized_smoothed_profile(broader_raw)
    stable_broad = probability_overlap(broad_profile, broad_profile)

    core = np.zeros(N_BINS, dtype=float)
    core[45] = 1.0
    secondary = np.zeros(N_BINS, dtype=float)
    secondary[180] = 1.0
    core_profile = normalized_smoothed_profile(core)
    core_plus_secondary = normalized_smoothed_profile(0.8 * core + 0.2 * secondary)
    core_secondary_overlap = probability_overlap(core_profile, core_plus_secondary)

    at_zero = np.zeros(N_BINS, dtype=float)
    at_zero[0] = 1.0
    before_zero = np.roll(at_zero, -1)
    at_zero_profile = normalized_smoothed_profile(at_zero)
    before_zero_profile = normalized_smoothed_profile(before_zero)
    wrap_overlap = probability_overlap(at_zero_profile, before_zero_profile)
    wrap_mass = float(smooth_circular_1h(at_zero)[-1])

    results = [
        {"case": "A_identical_profile", "overlap": identical, "expected": "1", "passed": np.isclose(identical, 1.0)},
        {"case": "B_modest_phase_shift_0.5h", "overlap": shifted, "expected": "<1", "passed": shifted < 1.0},
        {"case": "C_same_center_broader", "overlap": broadened, "expected": "<1", "passed": broadened < 1.0},
        {"case": "D_split_multimodal_redistribution", "overlap": split, "expected": "<1", "passed": split < 1.0},
        {"case": "E_stable_broad_repeated", "overlap": stable_broad, "expected": "1", "passed": np.isclose(stable_broad, 1.0)},
        {"case": "F_80pct_core_plus_20pct_secondary", "overlap": core_secondary_overlap, "expected": "approximately 0.8", "passed": np.isclose(core_secondary_overlap, 0.8, atol=1e-12)},
        {"case": "G_CT23_CT0_wrap", "overlap": wrap_overlap, "expected": "<1 with wrapped kernel mass", "passed": wrap_overlap < 1.0 and wrap_mass > 0.0},
    ]
    if not all(bool(result["passed"]) for result in results):
        raise AssertionError(f"Synthetic validation failed: {results}")
    return results


def _collect_full_profiles(
    cohort_root: Path,
) -> tuple[
    dict[tuple[str, str, int], np.ndarray | None],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    profiles: dict[tuple[str, str, int], np.ndarray | None] = {}
    profile_rows: list[dict[str, object]] = []
    descriptor_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for group, animal in frozen_data.ANIMALS:
        record, support_rows = frozen_data._build_animal_cycle_data(
            group, animal, cohort_root
        )
        support_by_cycle = {int(row["cycle_index"]): row for row in support_rows}
        cycles = record["cycles"]
        for cycle_order, cycle_index in enumerate(record["cycle_indices"], start=1):
            cycle = cycles[int(cycle_index)]
            support = support_by_cycle[int(cycle_index)]
            for behavior in frozen_data.NONREST_BEHAVIORS:
                behavior_data = cycle["behaviors"][behavior]
                duration_seconds = float(behavior_data["total_minutes"]) * 60.0
                full_duration_bins = (
                    np.asarray(behavior_data["probability"], dtype=float)
                    * duration_seconds
                )
                duration_mass = float(full_duration_bins.sum())
                raw_probability = (
                    full_duration_bins / duration_mass
                    if duration_mass > 0.0
                    else None
                )
                raw_phase, raw_resultant = circular_phase_and_R(raw_probability)
                profile = normalized_smoothed_profile(full_duration_bins)
                smoothed_phase, smoothed_resultant = circular_phase_and_R(profile)
                profiles[(animal, behavior, cycle_order)] = profile
                descriptor_rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "cycle": cycle_order,
                        "cycle_index": int(cycle_index),
                        "circular_mean_phase_h": raw_phase,
                        "R": raw_resultant,
                        "duration_seconds": duration_seconds,
                        "duration_minutes": duration_seconds / 60.0,
                        "bout_count": int(behavior_data["bout_count"]),
                        "n_classified_frames": int(support["n_classified_frames"]),
                    }
                )
                comparison_rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "cycle": cycle_order,
                        "raw_phase_h": raw_phase,
                        "smoothed_phase_h": smoothed_phase,
                        "raw_R": raw_resultant,
                        "smoothed_R": smoothed_resultant,
                    }
                )
                for bin_index, center in enumerate(CT_BIN_CENTERS):
                    profile_rows.append(
                        {
                            "animal": animal,
                            "behavior": behavior,
                            "cycle": cycle_order,
                            "cycle_index": int(cycle_index),
                            "ct_bin_center_h": float(center),
                            "probability": (
                                float(profile[bin_index]) if profile is not None else float("nan")
                            ),
                        }
                    )
    return (
        profiles,
        pd.DataFrame(profile_rows),
        pd.DataFrame(descriptor_rows),
        pd.DataFrame(comparison_rows),
    )


def _build_recurrence_table(
    profiles: dict[tuple[str, str, int], np.ndarray | None],
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    descriptor_map = {
        (str(row.animal), str(row.behavior), int(row.cycle)): float(
            row.circular_mean_phase_h
        )
        for row in descriptors.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for animal in frozen_data.ANIMAL_ORDER:
        for behavior in frozen_data.NONREST_BEHAVIORS:
            row: dict[str, object] = {"animal": animal, "behavior": behavior}
            overlaps: list[float] = []
            phase_differences: list[float] = []
            for first_cycle, second_cycle in PAIR_ORDERS:
                score = probability_overlap(
                    profiles[(animal, behavior, first_cycle)],
                    profiles[(animal, behavior, second_cycle)],
                )
                row[f"pair_{first_cycle}_{second_cycle}"] = score
                overlaps.append(score)
                first_phase = descriptor_map[(animal, behavior, first_cycle)]
                second_phase = descriptor_map[(animal, behavior, second_cycle)]
                phase_differences.append(
                    circular_difference_hours(first_phase, second_phase)
                )
            row["mean_recurrence"] = (
                float(np.mean(overlaps))
                if np.isfinite(overlaps).all()
                else float("nan")
            )
            row["mean_pairwise_phase_difference_h"] = (
                float(np.mean(phase_differences))
                if np.isfinite(phase_differences).all()
                else float("nan")
            )
            rows.append(row)
    columns = [
        "animal",
        "behavior",
        "mean_recurrence",
        *(f"pair_{first}_{second}" for first, second in PAIR_ORDERS),
        "mean_pairwise_phase_difference_h",
    ]
    return pd.DataFrame(rows)[columns]


def _validate_frozen_outputs(
    profile_frame: pd.DataFrame,
    recurrence_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float | int]:
    """Ensure this descriptive cleanup leaves the scored outputs unchanged."""
    profile_path = output_dir / "full_cycle_phase_profiles.csv"
    recurrence_path = output_dir / "full_cycle_behavior_recurrence.csv"
    if not profile_path.is_file() or not recurrence_path.is_file():
        raise FileNotFoundError(
            "The existing scored profiles and recurrence table are required for this follow-up"
        )
    saved_profiles = pd.read_csv(profile_path)
    profile_keys = [
        "animal",
        "behavior",
        "cycle",
        "cycle_index",
        "ct_bin_center_h",
    ]
    saved_profiles = saved_profiles.sort_values(profile_keys).reset_index(drop=True)
    profile_frame = profile_frame.sort_values(profile_keys).reset_index(drop=True)
    identity_columns = profile_keys[:-1]
    if any(
        not saved_profiles[column].astype(str).equals(
            profile_frame[column].astype(str)
        )
        for column in identity_columns
    ):
        raise RuntimeError("Profile identities differ from the frozen scored profiles")
    if not np.allclose(
        saved_profiles["ct_bin_center_h"].to_numpy(float),
        profile_frame["ct_bin_center_h"].to_numpy(float),
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("Profile identities differ from the frozen scored profiles")
    profile_delta = np.abs(
        saved_profiles["probability"].to_numpy(float)
        - profile_frame["probability"].to_numpy(float)
    )
    if not np.allclose(
        saved_profiles["probability"].to_numpy(float),
        profile_frame["probability"].to_numpy(float),
        rtol=0.0,
        atol=1e-15,
        equal_nan=True,
    ):
        raise RuntimeError("The recomputed smoothed profiles differ from the saved scored profiles")

    saved_recurrence = pd.read_csv(recurrence_path)
    if not saved_recurrence[["animal", "behavior"]].equals(
        recurrence_frame[["animal", "behavior"]]
    ):
        raise RuntimeError("Animal/behavior rows differ from the frozen recurrence table")
    score_columns = ["mean_recurrence", *(f"pair_{a}_{b}" for a, b in PAIR_ORDERS)]
    score_delta = np.abs(
        saved_recurrence[score_columns].to_numpy(float)
        - recurrence_frame[score_columns].to_numpy(float)
    )
    if not np.allclose(
        saved_recurrence[score_columns].to_numpy(float),
        recurrence_frame[score_columns].to_numpy(float),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    ):
        raise RuntimeError("A saved recurrence score changed; refusing to rewrite frozen outputs")

    keys = ["animal", "behavior", "cycle"]
    saved_smoothed_rows: list[dict[str, object]] = []
    for identity, group in saved_profiles.groupby(keys, sort=False):
        saved_probability = group["probability"].to_numpy(float)
        saved_probability = (
            saved_probability if np.isfinite(saved_probability).all() else None
        )
        saved_phase, saved_R = circular_phase_and_R(saved_probability)
        saved_smoothed_rows.append(
            {
                "animal": identity[0],
                "behavior": identity[1],
                "cycle": int(identity[2]),
                "saved_smoothed_phase_h": saved_phase,
                "saved_smoothed_R": saved_R,
            }
        )
    previous = comparison_frame.merge(
        pd.DataFrame(saved_smoothed_rows), on=keys, validate="one_to_one"
    )
    if len(previous) != len(comparison_frame):
        raise RuntimeError("Saved smoothed profiles do not match the current cycle set")
    phase_differences = [
        circular_difference_hours(raw, old)
        for raw, old in zip(
            previous["raw_phase_h"].to_numpy(float),
            previous["saved_smoothed_phase_h"].to_numpy(float),
        )
    ]
    phase_differences = np.asarray(phase_differences, dtype=float)
    phase_differences = phase_differences[np.isfinite(phase_differences)]
    r_differences = (
        previous["raw_R"].to_numpy(float)
        - previous["saved_smoothed_R"].to_numpy(float)
    )
    r_differences = r_differences[np.isfinite(r_differences)]
    smooth_phase_consistency = [
        circular_difference_hours(current, old)
        for current, old in zip(
            previous["smoothed_phase_h"].to_numpy(float),
            previous["saved_smoothed_phase_h"].to_numpy(float),
        )
    ]
    smooth_phase_consistency = np.asarray(
        smooth_phase_consistency, dtype=float
    )
    smooth_phase_consistency = smooth_phase_consistency[
        np.isfinite(smooth_phase_consistency)
    ]
    smooth_r_consistency = np.abs(
        previous["smoothed_R"].to_numpy(float)
        - previous["saved_smoothed_R"].to_numpy(float)
    )
    smooth_r_consistency = smooth_r_consistency[np.isfinite(smooth_r_consistency)]
    if (
        smooth_phase_consistency.size
        and float(smooth_phase_consistency.max()) > 1e-10
    ) or (smooth_r_consistency.size and float(smooth_r_consistency.max()) > 1e-12):
        raise RuntimeError("Saved descriptors do not match the previous smoothed-profile calculation")
    if r_differences.size and float(r_differences.min()) < -1e-12:
        raise RuntimeError("Raw R is unexpectedly below smoothed R beyond numerical tolerance")
    return {
        "max_raw_vs_previous_phase_difference_h": (
            float(phase_differences.max()) if phase_differences.size else float("nan")
        ),
        "n_phase_comparisons": int(phase_differences.size),
        "mean_raw_minus_smoothed_R": (
            float(r_differences.mean()) if r_differences.size else float("nan")
        ),
        "median_raw_minus_smoothed_R": (
            float(np.median(r_differences)) if r_differences.size else float("nan")
        ),
        "max_raw_minus_smoothed_R": (
            float(r_differences.max()) if r_differences.size else float("nan")
        ),
        "n_R_comparisons": int(r_differences.size),
        "max_saved_profile_abs_difference": (
            float(np.nanmax(profile_delta)) if np.isfinite(profile_delta).any() else 0.0
        ),
        "max_saved_recurrence_abs_difference": (
            float(np.nanmax(score_delta)) if np.isfinite(score_delta).any() else 0.0
        ),
    }


def _format_value(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _format_check(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.12g}"


def _create_eating_figure(
    profiles: dict[tuple[str, str, int], np.ndarray | None],
    descriptors: pd.DataFrame,
    recurrence: pd.DataFrame,
    output_path: Path,
) -> None:
    animals = ("714H", "675G")
    behavior = "eating"
    descriptor_map = {
        (str(row.animal), int(row.cycle)): row
        for row in descriptors.loc[descriptors["behavior"] == behavior].itertuples(index=False)
    }
    recurrence_map = recurrence.loc[
        (recurrence["behavior"] == behavior)
        & recurrence["animal"].isin(animals)
    ].set_index("animal")
    max_probability = max(
        float(np.nanmax(profiles[(animal, behavior, cycle)]))
        for animal in animals
        for cycle in range(1, N_CYCLES + 1)
        if profiles[(animal, behavior, cycle)] is not None
    )
    y_max = max_probability * 1.12
    fig, axes = plt.subplots(N_CYCLES, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(
        "Full observed eating phase profiles across four complete cycles",
        fontsize=14,
        y=0.975,
    )
    fig.text(
        0.5,
        0.946,
        "Same 288-bin, 5-minute profiles used for recurrence scoring; fixed circular 1-hour smoothing",
        ha="center",
        fontsize=9,
    )
    for column, animal in enumerate(animals):
        axes[0, column].set_title(animal, fontsize=12, pad=8)
        for cycle_order in range(1, N_CYCLES + 1):
            ax = axes[cycle_order - 1, column]
            profile = profiles[(animal, behavior, cycle_order)]
            descriptor = descriptor_map[(animal, cycle_order)]
            if profile is not None:
                ax.plot(CT_BIN_CENTERS, profile, color="#2878A5", linewidth=1.25)
                ax.fill_between(CT_BIN_CENTERS, profile, color="#2878A5", alpha=0.18)
            ax.set_xlim(0.0, CT_HOURS)
            ax.set_ylim(0.0, y_max)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
            ax.set_title(
                f"Selected cycle {cycle_order} (index {int(descriptor.cycle_index)}): "
                f"phase={_format_value(float(descriptor.circular_mean_phase_h))} h, "
                f"R={_format_value(float(descriptor.R))}",
                fontsize=8.5,
                pad=3,
            )
            if column == 0:
                ax.set_ylabel("Probability / 5-min bin", fontsize=8)
            if cycle_order == N_CYCLES:
                ax.set_xlabel("Circadian time (CT, h)", fontsize=9)

        summary = recurrence_map.loc[animal]
        x = 0.08 if column == 0 else 0.55
        fig.text(
            x,
            0.265,
            f"Mean recurrence: {_format_value(float(summary['mean_recurrence']))}",
            ha="left",
            fontsize=9,
            weight="bold",
        )
        pair_values = [
            f"{first}-{second}: {_format_value(float(summary[f'pair_{first}_{second}']))}"
            for first, second in PAIR_ORDERS
        ]
        fig.text(
            x,
            0.225,
            "Pair overlaps: " + "   ".join(pair_values[:3]),
            ha="left",
            fontsize=7.5,
        )
        fig.text(
            x,
            0.19,
            "              " + "   ".join(pair_values[3:]),
            ha="left",
            fontsize=7.5,
        )
        phase_values = [
            f"C{cycle} {_format_value(float(descriptor_map[(animal, cycle)].circular_mean_phase_h))}/"
            f"{_format_value(float(descriptor_map[(animal, cycle)].R))}"
            for cycle in range(1, N_CYCLES + 1)
        ]
        fig.text(
            x,
            0.15,
            "Cycle phase (h) / R: " + "   ".join(phase_values[:2]),
            ha="left",
            fontsize=7.5,
        )
        fig.text(
            x,
            0.115,
            "                       " + "   ".join(phase_values[2:]),
            ha="left",
            fontsize=7.5,
        )
    fig.text(
        0.5,
        0.045,
        "Behavior duration and bout count are QC only; no rarefaction, support correction, or cycle alignment.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.33, hspace=0.58, wspace=0.12)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_summary(
    synthetic: list[dict[str, object]],
    recurrence: pd.DataFrame,
    descriptors: pd.DataFrame,
    frozen_checks: dict[str, float | int],
    output_path: Path,
) -> None:
    eating_rows = recurrence.loc[
        (recurrence["behavior"] == "eating")
        & recurrence["animal"].isin(["714H", "675G"])
    ].set_index("animal")
    eating_descriptors = descriptors.loc[
        (descriptors["behavior"] == "eating")
        & descriptors["animal"].isin(["714H", "675G"])
    ]
    higher = "714H" if eating_rows.loc["714H", "mean_recurrence"] > eating_rows.loc["675G", "mean_recurrence"] else "675G"
    lines = [
        "FULL-DATA CYCLE-TO-CYCLE PHASE-PROFILE OVERLAP RECURRENCE",
        "",
        "METHOD",
        "  All observed WTA duration occupancy was used for each animal x behavior x frozen complete cycle.",
        "  Each 288-bin profile spans CT0-24 on the 5-minute numerical grid; bins are centered at 5-minute CT-bin centers.",
        "  The fixed circular 1-hour kernel uses offsets -30,-25,...,+25,+30 minutes; endpoint weights are 0.5, interior weights 1.0, normalized to sum 1.",
        "  Smoothed full-duration occupancy is normalized within behavior and cycle to sum to 1.",
        "  Pair recurrence is sum(min(p,q)); behavior recurrence is the arithmetic mean of all six chronological cycle-pair overlaps.",
        "  Circular mean phase and R are descriptive, calculated from the unsmoothed normalized full-duration profile.",
        "  Recurrence and plotted profiles continue to use the fixed 1-hour-smoothed normalized full-duration profile.",
        "  Behavior duration, bout count, and classified-frame count are QC/context only.",
        "  No rarefaction, random subsampling, support correction, alignment, null correction, or imputation was used.",
        "  A zero-duration cycle has an undefined profile (NaN bins); pair overlaps involving it and the six-pair mean are NA.",
        "  Pair labels 1-2 through 3-4 refer to chronological order of the four selected frozen complete cycles; cycle_index is also saved.",
        "",
        "MINIMAL DETERMINISTIC SYNTHETIC VALIDATION",
    ]
    for result in synthetic:
        lines.append(
            f"  {result['case']}: overlap={float(result['overlap']):.6f}; "
            f"expected {result['expected']}; {'PASS' if result['passed'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "FROZEN ARCHITECTURE / CHANGE CHECKS",
            "  1. Recurrence definition changed: NO.",
            "  2. Smoothing kernel changed: NO.",
            "  3. Rarefaction added: NO.",
            "  4. Support correction added: NO.",
            "  5. Alignment added: NO.",
            "  6. Figure's scored profile representation changed: NO; phase/R annotations now describe the unsmoothed input.",
            "  7. Circular mean phase calculated from: unsmoothed normalized full-data 288-bin phase distribution.",
            "  8. R calculated from: unsmoothed normalized full-data 288-bin phase distribution.",
            "  9. Recurrence calculated from: fixed 1-hour-smoothed normalized full-data phase profile.",
            " 10. Maximum absolute circular phase difference versus previous smoothed descriptors: "
            f"{_format_check(float(frozen_checks['max_raw_vs_previous_phase_difference_h']))} h "
            f"(n={int(frozen_checks['n_phase_comparisons'])}).",
            " 11. raw_R - previous smoothed_R: mean="
            f"{_format_check(float(frozen_checks['mean_raw_minus_smoothed_R']))}; median="
            f"{_format_check(float(frozen_checks['median_raw_minus_smoothed_R']))}; maximum="
            f"{_format_check(float(frozen_checks['max_raw_minus_smoothed_R']))} "
            f"(n={int(frozen_checks['n_R_comparisons'])}).",
            " 12-13. Eating mean recurrence unchanged: 714H="
            f"{_format_check(float(eating_rows.loc['714H', 'mean_recurrence']))}; 675G="
            f"{_format_check(float(eating_rows.loc['675G', 'mean_recurrence']))}; YES.",
            " 14. All six pairwise eating overlaps unchanged for both animals: YES; maximum recomputed-vs-saved score difference="
            f"{_format_check(float(frozen_checks['max_saved_recurrence_abs_difference']))}; see exact values below.",
            " 15. Saved scored recurrence profiles unchanged: YES; maximum absolute bin difference="
            f"{_format_check(float(frozen_checks['max_saved_profile_abs_difference']))}.",
            "  Saved recurrence score columns in full_cycle_behavior_recurrence.csv were verified unchanged and that file was not rewritten.",
        ]
    )
    lines.extend(
        [
            "",
            "714H VS 675G EATING (FULL OBSERVED DATA)",
        ]
    )
    for animal in ("714H", "675G"):
        row = eating_rows.loc[animal]
        pairs = ", ".join(
            f"{first}-{second}={_format_value(float(row[f'pair_{first}_{second}']))}"
            for first, second in PAIR_ORDERS
        )
        lines.append(
            f"  {animal}: mean recurrence={_format_value(float(row['mean_recurrence']))}; "
            f"mean pairwise phase difference={_format_value(float(row['mean_pairwise_phase_difference_h']))} h; {pairs}."
        )
        for descriptor in eating_descriptors.loc[
            eating_descriptors["animal"] == animal
        ].sort_values("cycle").itertuples(index=False):
            lines.append(
                f"    cycle {int(descriptor.cycle)} (index {int(descriptor.cycle_index)}): "
                f"phase={_format_value(float(descriptor.circular_mean_phase_h))} h; "
                f"R={_format_value(float(descriptor.R))}; "
                f"duration={float(descriptor.duration_minutes):.2f} min; "
                f"bouts={int(descriptor.bout_count)}."
            )
    lines.extend(
        [
            f"  Higher mean recurrence in these two animals: {higher} (reported without reinterpretation).",
            "",
            "SUMMARY QUESTIONS",
            "  1. Every recurrence score uses the full observed cycle data: YES.",
            "  2. Rarefaction performed: NO.",
            "  3. Random subsampling performed: NO.",
            "  4. Support correction performed: NO.",
            "  5. Cycle alignment performed: NO.",
            "  6. Representation: normalized full-duration occupancy on 288 five-minute CT bins, after fixed circular 1-hour trapezoidal smoothing.",
            "  7. Kernel: offsets -30 to +30 min in 5-min steps; endpoint weights 0.5 and 11 interior weights 1.0, normalized.",
            "  8. Score and figure use the same vectors: YES.",
            f"  9. Identical synthetic profiles score 1: {'YES' if synthetic[0]['passed'] else 'NO'}.",
            f" 10. Stable broad repeated profiles score 1: {'YES' if synthetic[4]['passed'] else 'NO'}.",
            f" 11. Phase shifts reduce recurrence: {'YES' if synthetic[1]['passed'] else 'NO'}.",
            f" 12. Broadening and splitting reduce recurrence: {'YES' if synthetic[2]['passed'] and synthetic[3]['passed'] else 'NO'}.",
            f" 13. Recurring core plus secondary component behaves sensibly: {'YES' if synthetic[5]['passed'] else 'NO'} (overlap tracks the retained 80% core).",
            " 14-15. 714H and 675G pairwise overlaps and mean recurrence are listed above and in eating_714H_675G_summary.csv.",
            f" 16. Higher mean recurrence: {higher}.",
            " 17. Plotted profiles correspond bin-for-bin to scored vectors: YES; the shared scale is used across both animals.",
            " 18. Minutes and bouts are reported only as QC/context: YES.",
            "",
            "INTERPRETATION",
            "  This is the direct full-data descriptive result for the frozen four-cycle selection. It adds no support adjustment and is not a genotype comparison.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _unit_tests()
    synthetic = _synthetic_validation()
    output_dir = args.cohort_root / OUTPUT_DIRNAME
    profiles, profile_frame, descriptor_frame, comparison_frame = _collect_full_profiles(
        args.cohort_root
    )
    recurrence_frame = _build_recurrence_table(profiles, descriptor_frame)
    frozen_checks = _validate_frozen_outputs(
        profile_frame,
        recurrence_frame,
        comparison_frame,
        output_dir,
    )
    eating_summary = recurrence_frame.loc[
        (recurrence_frame["behavior"] == "eating")
        & recurrence_frame["animal"].isin(["714H", "675G"])
    ].set_index("animal").loc[["714H", "675G"]].reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_frame.to_csv(output_dir / "full_cycle_phase_descriptors.csv", index=False)
    eating_summary.to_csv(output_dir / "eating_714H_675G_summary.csv", index=False)
    _create_eating_figure(
        profiles,
        descriptor_frame,
        recurrence_frame,
        output_dir / "eating_714H_675G_full_cycle_qc.png",
    )
    _build_summary(
        synthetic,
        recurrence_frame,
        descriptor_frame,
        frozen_checks,
        output_dir / "full_cycle_recurrence_validation_summary.txt",
    )
    print(f"Outputs: {output_dir}")
    print(f"Behavior recurrence rows: {len(recurrence_frame)}")
    print(f"Phase-profile rows: {len(profile_frame)}")
    print(f"Cycle descriptor rows: {len(descriptor_frame)}")
    print(
        "Maximum raw-vs-smoothed phase difference (h): "
        f"{_format_check(float(frozen_checks['max_raw_vs_previous_phase_difference_h']))}"
    )
    print(
        "Mean/median/max raw_R - smoothed_R: "
        f"{_format_check(float(frozen_checks['mean_raw_minus_smoothed_R']))}/"
        f"{_format_check(float(frozen_checks['median_raw_minus_smoothed_R']))}/"
        f"{_format_check(float(frozen_checks['max_raw_minus_smoothed_R']))}"
    )
    print(
        "Maximum recomputed-vs-saved recurrence score difference: "
        f"{_format_check(float(frozen_checks['max_saved_recurrence_abs_difference']))}"
    )
    print("Saved recurrence score/profile files unchanged")
    print("Synthetic validation: PASS")


if __name__ == "__main__":
    main()
