"""Descriptive census of ambiguity in CBAS frame-level classifier outputs.

This is a read-only audit of the existing nine-state classifier probabilities.
It does not choose an ambiguity threshold, alter the classifier, or calculate
any biological endpoint.  The input is a cohort root containing the existing
animal folders and frozen FRP/phase outputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
BEHAVIORS = tuple(frozen_phase.BEHAVIORS)
N_BEHAVIORS = len(BEHAVIORS)
N_CT_BINS = 288
FILE_DURATION_MINUTES = 10.0
CT_HOURS = 24.0

RATIO_CUTOFFS = (0.70, 0.80, 0.90, 0.95)
TOP2_SUM_CUTOFFS = (0.60, 0.70, 0.80, 0.90)
P2_CUTOFFS = (0.10, 0.20, 0.30, 0.35, 0.40)
WINDOW_FRAMES = (1, 2, 5, 10)

DEFINITION_LABELS = (
    "A_ratio_ge_0.80",
    "B_ratio_ge_0.90",
    "C_ratio_ge_0.80_top2_ge_0.70",
    "D_p2_ge_0.35_ratio_ge_0.70",
    "diffuse_ratio_ge_0.80_top2_lt_0.60",
)
NEAR_PAIR_DEFINITIONS = (
    "ratio_ge_0.80",
    "ratio_ge_0.90",
    "ratio_ge_0.80_top2_ge_0.70",
    "diffuse_ratio_ge_0.80_top2_lt_0.60",
)
CATEGORY_LABELS = (
    "high_confidence_p1_ge_0.80",
    "clear_winner_p1_ge_0.50_ratio_lt_0.50",
    "pair_ambiguity_ratio_ge_0.80_top2_ge_0.70",
    "diffuse_ratio_ge_0.80_top2_lt_0.60",
    "intermediate_or_other",
)

METRIC_SPECS = {
    "p1": (0.0, 1.05),
    "p2": (0.0, 1.05),
    "p3": (0.0, 1.05),
    "margin_12": (0.0, 1.05),
    "ratio_21": (0.0, 1.05),
    "top2_sum": (0.0, 1.10),
    "top3_sum": (0.0, 1.10),
    "frame_entropy_bits": (0.0, np.log2(N_BEHAVIORS)),
}
METRIC_QUANTILES = (
    ("median", 0.50),
    ("p05", 0.05),
    ("p25", 0.25),
    ("p75", 0.75),
    ("p95", 0.95),
    ("p99", 0.99),
)


def _empty_definition_counts() -> dict[str, int]:
    return {label: 0 for label in DEFINITION_LABELS}


def _format_pair(code: int) -> tuple[str, str]:
    return BEHAVIORS[code // N_BEHAVIORS], BEHAVIORS[code % N_BEHAVIORS]


def _definition_masks(
    p2: np.ndarray,
    ratio: np.ndarray,
    top2_sum: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the fixed descriptive strata requested by the audit contract."""

    return {
        "A_ratio_ge_0.80": ratio >= 0.80,
        "B_ratio_ge_0.90": ratio >= 0.90,
        "C_ratio_ge_0.80_top2_ge_0.70": (ratio >= 0.80) & (top2_sum >= 0.70),
        "D_p2_ge_0.35_ratio_ge_0.70": (p2 >= 0.35) & (ratio >= 0.70),
        "diffuse_ratio_ge_0.80_top2_lt_0.60": (ratio >= 0.80) & (top2_sum < 0.60),
    }


def _pair_definition_masks(
    p2: np.ndarray,
    ratio: np.ndarray,
    top2_sum: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ratio_ge_0.80": ratio >= 0.80,
        "ratio_ge_0.90": ratio >= 0.90,
        "ratio_ge_0.80_top2_ge_0.70": (ratio >= 0.80) & (top2_sum >= 0.70),
        "diffuse_ratio_ge_0.80_top2_lt_0.60": (ratio >= 0.80) & (top2_sum < 0.60),
    }


def _category_masks(
    p1: np.ndarray,
    ratio: np.ndarray,
    top2_sum: np.ndarray,
) -> dict[str, np.ndarray]:
    masks = {
        "high_confidence_p1_ge_0.80": p1 >= 0.80,
        "clear_winner_p1_ge_0.50_ratio_lt_0.50": (p1 >= 0.50) & (ratio < 0.50),
        "pair_ambiguity_ratio_ge_0.80_top2_ge_0.70": (ratio >= 0.80)
        & (top2_sum >= 0.70),
        "diffuse_ratio_ge_0.80_top2_lt_0.60": (ratio >= 0.80) & (top2_sum < 0.60),
    }
    classified = np.zeros(len(p1), dtype=bool)
    for mask in masks.values():
        classified |= mask
    masks["intermediate_or_other"] = ~classified
    return masks


def _hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    target = q * total
    index = int(np.searchsorted(np.cumsum(counts), target, side="left"))
    index = min(max(index, 0), len(counts) - 1)
    return float((edges[index] + edges[index + 1]) / 2.0)


class HistogramCollection:
    """Fixed-bin streaming distributions, avoiding a second copy of all frames."""

    N_BINS = 2000

    def __init__(self) -> None:
        self.edges = {
            metric: np.linspace(low, high, self.N_BINS + 1)
            for metric, (low, high) in METRIC_SPECS.items()
        }
        self.counts = {
            metric: np.zeros(self.N_BINS, dtype=np.int64)
            for metric in METRIC_SPECS
        }

    def update(self, values: dict[str, np.ndarray]) -> None:
        for metric, value in values.items():
            histogram, _ = np.histogram(value, bins=self.edges[metric])
            self.counts[metric] += histogram.astype(np.int64, copy=False)

    def rows(self, scope: str, group: str = "", animal: str = "") -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for metric in METRIC_SPECS:
            row: dict[str, object] = {
                "scope": scope,
                "Group": group,
                "Animal": animal,
                "metric": metric,
                "frames": int(self.counts[metric].sum()),
            }
            for label, quantile in METRIC_QUANTILES:
                row[label] = _hist_quantile(
                    self.counts[metric], self.edges[metric], quantile
                )
            rows.append(row)
        return rows


class PairAccumulator:
    """Counts and streaming approximate medians for ordered winner/runner pairs."""

    N_BINS = 2000

    def __init__(self) -> None:
        self.counts = np.zeros(N_BEHAVIORS * N_BEHAVIORS, dtype=np.int64)
        self.edges = {
            metric: np.linspace(low, high, self.N_BINS + 1)
            for metric, (low, high) in {
                "p1": (0.0, 1.05),
                "p2": (0.0, 1.05),
                "ratio": (0.0, 1.05),
                "top2_sum": (0.0, 1.10),
            }.items()
        }
        self.histograms = {
            metric: np.zeros((N_BEHAVIORS * N_BEHAVIORS, self.N_BINS), dtype=np.int64)
            for metric in self.edges
        }

    def update(
        self,
        pair_codes: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        ratio: np.ndarray,
        top2_sum: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        if mask is None:
            codes = pair_codes
            values = {"p1": p1, "p2": p2, "ratio": ratio, "top2_sum": top2_sum}
        else:
            codes = pair_codes[mask]
            values = {
                "p1": p1[mask],
                "p2": p2[mask],
                "ratio": ratio[mask],
                "top2_sum": top2_sum[mask],
            }
        if len(codes) == 0:
            return
        for code in np.unique(codes):
            selected = codes == code
            code_int = int(code)
            self.counts[code_int] += int(selected.sum())
            for metric, value in values.items():
                histogram, _ = np.histogram(value[selected], bins=self.edges[metric])
                self.histograms[metric][code_int] += histogram.astype(
                    np.int64, copy=False
                )

    def add(self, other: "PairAccumulator") -> None:
        self.counts += other.counts
        for metric in self.histograms:
            self.histograms[metric] += other.histograms[metric]

    def median(self, metric: str, code: int) -> float:
        return _hist_quantile(
            self.histograms[metric][code], self.edges[metric], 0.50
        )


def _read_probabilities(path: Path) -> tuple[np.ndarray, ...]:
    frame = pd.read_csv(path, usecols=list(BEHAVIORS))
    probabilities = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_BEHAVIORS:
        raise ValueError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    if len(probabilities) == 0:
        return (
            probabilities,
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite classifier probability in {path}")
    row_sums = probabilities.sum(axis=1)
    if float(np.max(np.abs(row_sums - 1.0))) > 1e-6:
        raise ValueError(
            f"Classifier probabilities do not sum to one within tolerance in {path}"
        )
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    p1 = sorted_probabilities[:, 0]
    p2 = sorted_probabilities[:, 1]
    p3 = sorted_probabilities[:, 2]
    if np.any(p1 < p2) or np.any(p2 < p3):
        raise AssertionError(f"Top probabilities are not ordered in {path}")
    if np.any(p1 <= 0.0):
        raise ValueError(f"Non-positive top probability prevents ratio calculation in {path}")
    ratio = p2 / p1
    top2_sum = p1 + p2
    top3_sum = top2_sum + p3
    entropy = -np.sum(
        probabilities * np.log2(np.where(probabilities > 0.0, probabilities, 1.0)),
        axis=1,
    )
    return probabilities, order[:, 0].astype(np.int8), order[:, 1].astype(np.int8), p1, p2, p3, ratio, top2_sum, top3_sum, entropy


def _read_file_metrics(path: Path) -> dict[str, np.ndarray]:
    (
        probabilities,
        top1,
        top2,
        p1,
        p2,
        p3,
        ratio,
        top2_sum,
        top3_sum,
        entropy,
    ) = _read_probabilities(path)
    expected_rows, row_positions, existing_labels = frozen_phase.read_and_classify(path)
    if expected_rows != len(probabilities):
        raise AssertionError(f"Row-count mismatch between readers in {path}")
    if not np.array_equal(row_positions, np.arange(len(probabilities), dtype=np.int64)):
        raise AssertionError(f"Existing classifier omitted rows in {path}")
    if not np.array_equal(existing_labels, top1):
        mismatch = int(np.count_nonzero(existing_labels != top1))
        raise AssertionError(f"WTA mismatch with existing classifier in {path}: {mismatch}")
    return {
        "probabilities": probabilities,
        "top1": top1,
        "top2": top2,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "margin_12": p1 - p2,
        "ratio_21": ratio,
        "top2_sum": top2_sum,
        "top3_sum": top3_sum,
        "frame_entropy_bits": entropy,
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _scope_row(
    scope: str,
    group: str,
    animal: str,
    total_frames: int,
    qualifying_frames: int,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "Group": group,
        "Animal": animal,
        "total_frames": int(total_frames),
        "qualifying_frames": int(qualifying_frames),
        "fraction_frames": (
            float(qualifying_frames / total_frames) if total_frames else float("nan")
        ),
    }
    row.update(extra)
    return row


def _pair_rows(
    accumulator: PairAccumulator,
    definition: str,
    scope: str,
    group: str,
    animal: str,
    total_frames: int,
    definition_frames: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for code, count in enumerate(accumulator.counts):
        if count == 0:
            continue
        winner, runner_up = _format_pair(code)
        row = {
            "scope": scope,
            "definition": definition,
            "Animal": animal,
            "Group": group,
            "Winner": winner,
            "RunnerUp": runner_up,
            "count": int(count),
            "fraction": float(count / total_frames) if total_frames else float("nan"),
            "fraction_of_definition_frames": (
                float(count / definition_frames) if definition_frames else float("nan")
            ),
            "median_p1": accumulator.median("p1", code),
            "median_p2": accumulator.median("p2", code),
            "median_ratio": accumulator.median("ratio", code),
            "median_top2_sum": accumulator.median("top2_sum", code),
        }
        rows.append(row)
    return rows


def _top_pair(accumulator: PairAccumulator) -> tuple[str, str, int]:
    if int(accumulator.counts.sum()) == 0:
        return "", "", 0
    code = int(np.argmax(accumulator.counts))
    winner, runner_up = _format_pair(code)
    return winner, runner_up, int(accumulator.counts[code])


def _nearest_transition_distances(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(labels)
    if n_frames == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)
    positions = np.arange(n_frames, dtype=np.int64)
    transition = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        transition[1:] = labels[1:] != labels[:-1]
    transition_positions = positions[transition]
    if len(transition_positions) == 0:
        return np.full(n_frames, np.inf, dtype=float), transition_positions
    left = np.maximum.accumulate(np.where(transition, positions, -1))
    right = np.minimum.accumulate(
        np.where(transition, positions, n_frames)[::-1]
    )[::-1]
    distances = np.minimum(positions - left, right - positions).astype(float)
    return distances, transition_positions


def _nearest_positive_time_distances(
    candidate_times_hours: np.ndarray,
    positive_times_hours: np.ndarray,
) -> np.ndarray:
    if len(candidate_times_hours) == 0 or len(positive_times_hours) == 0:
        return np.full(len(candidate_times_hours), np.nan, dtype=float)
    insertion = np.searchsorted(positive_times_hours, candidate_times_hours)
    right_index = np.minimum(insertion, len(positive_times_hours) - 1)
    left_index = np.maximum(insertion - 1, 0)
    right_distance = np.abs(
        positive_times_hours[right_index] - candidate_times_hours
    )
    left_distance = np.abs(
        positive_times_hours[left_index] - candidate_times_hours
    )
    return np.minimum(left_distance, right_distance) * 3600.0


def _process_animal(
    cohort_root: Path,
    group: str,
    animal: str,
    pooled_histogram: HistogramCollection,
    group_histograms: dict[str, HistogramCollection],
    animal_histograms: dict[str, HistogramCollection],
    density_p1_p2: np.ndarray,
    density_ratio_top2: np.ndarray,
    density_p1_p2_edges: np.ndarray,
    density_ratio_edges: np.ndarray,
    density_top2_edges: np.ndarray,
    pooled_pair_all: PairAccumulator,
    pooled_pair_near: dict[str, PairAccumulator],
    group_pair_all: dict[str, PairAccumulator],
    group_pair_near: dict[str, dict[str, PairAccumulator]],
) -> dict[str, object]:
    input_dir = cohort_root / group / animal
    files, missing_indices = frozen_phase.discover_input_files(input_dir)
    if missing_indices:
        raise ValueError(f"{animal} has missing source-file indices: {missing_indices}")
    (
        reported_frp_hours,
        computational_frp_hours,
        start_ct,
        complete_cycles,
        _phase_dir,
    ) = frozen_phase.load_frp_phase_solution(input_dir)
    if len(complete_cycles) != 4:
        raise ValueError(
            f"Expected the frozen four-cycle pilot solution for {animal}, found "
            f"{len(complete_cycles)} complete cycles"
        )

    animal_histograms[animal] = HistogramCollection()
    group_histograms.setdefault(group, HistogramCollection())
    pair_all = PairAccumulator()
    pair_near = {definition: PairAccumulator() for definition in NEAR_PAIR_DEFINITIONS}
    group_pair_all.setdefault(group, PairAccumulator())
    group_pair_near.setdefault(
        group,
        {definition: PairAccumulator() for definition in NEAR_PAIR_DEFINITIONS},
    )

    animal_definition_counts = _empty_definition_counts()
    threshold_counts = {
        (ratio_cutoff, top2_cutoff): 0
        for ratio_cutoff in RATIO_CUTOFFS
        for top2_cutoff in TOP2_SUM_CUTOFFS
    }
    p2_probe_counts = {f"p2_ge_{cutoff:.2f}": 0 for cutoff in P2_CUTOFFS}
    p2_probe_counts.update(
        {
            "p2_ge_0.30_ratio_ge_0.70": 0,
            "p2_ge_0.35_ratio_ge_0.70": 0,
            "p2_ge_0.35_ratio_ge_0.80": 0,
            "p2_ge_0.40_ratio_ge_0.80": 0,
        }
    )
    category_counts = {label: 0 for label in CATEGORY_LABELS}
    file_rows: list[dict[str, object]] = []
    ct_total = np.zeros(N_CT_BINS, dtype=np.int64)
    ct_ambiguous = {
        "A_ratio_ge_0.80": np.zeros(N_CT_BINS, dtype=np.int64),
        "C_ratio_ge_0.80_top2_ge_0.70": np.zeros(N_CT_BINS, dtype=np.int64),
        "diffuse_ratio_ge_0.80_top2_lt_0.60": np.zeros(N_CT_BINS, dtype=np.int64),
    }
    all_labels: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    positive_times: list[list[np.ndarray]] = [[] for _ in range(N_BEHAVIORS)]
    candidate_times: dict[str, list[list[np.ndarray]]] = {
        definition: [[] for _ in range(N_BEHAVIORS)]
        for definition in NEAR_PAIR_DEFINITIONS
    }
    total_frames = 0
    row_sum_total = 0.0
    row_sum_min = float("inf")
    row_sum_max = float("-inf")
    row_sum_max_abs_deviation = 0.0
    selected_cycle_frame_count = 0
    selected_cycle_bin_count = 0
    cycle_assignment_boundary_disagreements = 0
    median_frame_duration_seconds: list[float] = []

    for file_index, path in files:
        metrics = _read_file_metrics(path)
        probabilities = metrics.pop("probabilities")
        n_rows = len(probabilities)
        if n_rows == 0:
            continue
        p1 = metrics["p1"]
        p2 = metrics["p2"]
        p3 = metrics["p3"]
        ratio = metrics["ratio_21"]
        top2_sum = metrics["top2_sum"]
        top1 = metrics["top1"]
        top2 = metrics["top2"]
        pair_codes = top1.astype(np.int64) * N_BEHAVIORS + top2.astype(np.int64)
        definition_masks = _definition_masks(p2, ratio, top2_sum)
        pair_definition_masks = _pair_definition_masks(p2, ratio, top2_sum)
        category_masks = _category_masks(p1, ratio, top2_sum)
        file_pair_counts = np.bincount(
            pair_codes, minlength=N_BEHAVIORS * N_BEHAVIORS
        )
        file_top_pair_code = int(np.argmax(file_pair_counts))
        file_top_winner, file_top_runner = _format_pair(file_top_pair_code)
        file_c_pair_counts = np.bincount(
            pair_codes[definition_masks["C_ratio_ge_0.80_top2_ge_0.70"]],
            minlength=N_BEHAVIORS * N_BEHAVIORS,
        )
        if int(file_c_pair_counts.sum()):
            file_c_pair_code = int(np.argmax(file_c_pair_counts))
            file_c_winner, file_c_runner = _format_pair(file_c_pair_code)
            file_c_pair_count = int(file_c_pair_counts[file_c_pair_code])
        else:
            file_c_winner, file_c_runner, file_c_pair_count = "", "", 0

        row_sums = probabilities.sum(axis=1)
        total_frames += n_rows
        row_sum_total += float(row_sums.sum())
        row_sum_min = min(row_sum_min, float(row_sums.min()))
        row_sum_max = max(row_sum_max, float(row_sums.max()))
        row_sum_max_abs_deviation = max(
            row_sum_max_abs_deviation,
            float(np.max(np.abs(row_sums - 1.0))),
        )

        confidence_metrics = {
            metric: metrics[metric] for metric in METRIC_SPECS
        }
        pooled_histogram.update(confidence_metrics)
        animal_histograms[animal].update(confidence_metrics)
        group_histograms[group].update(confidence_metrics)
        density_p1_p2 += np.histogram2d(
            p1, p2, bins=(density_p1_p2_edges, density_p1_p2_edges)
        )[0].astype(np.int64, copy=False)
        density_ratio_top2 += np.histogram2d(
            ratio, top2_sum, bins=(density_ratio_edges, density_top2_edges)
        )[0].astype(np.int64, copy=False)

        all_labels.append(top1)
        elapsed_hours = (file_index - files[0][0] + np.arange(n_rows) / n_rows) * (
            FILE_DURATION_MINUTES / 60.0
        )
        all_times.append(elapsed_hours)
        median_frame_duration_seconds.append(FILE_DURATION_MINUTES * 60.0 / n_rows)
        for behavior_index in range(N_BEHAVIORS):
            positive_times[behavior_index].append(elapsed_hours[top1 == behavior_index])

        for definition, mask in definition_masks.items():
            count = int(mask.sum())
            animal_definition_counts[definition] += count
        for category_label, mask in category_masks.items():
            category_counts[category_label] += int(mask.sum())

        for ratio_cutoff in RATIO_CUTOFFS:
            for top2_cutoff in TOP2_SUM_CUTOFFS:
                threshold_counts[(ratio_cutoff, top2_cutoff)] += int(
                    np.count_nonzero((ratio >= ratio_cutoff) & (top2_sum >= top2_cutoff))
                )
        for cutoff in P2_CUTOFFS:
            p2_probe_counts[f"p2_ge_{cutoff:.2f}"] += int(np.count_nonzero(p2 >= cutoff))
        p2_probe_counts["p2_ge_0.30_ratio_ge_0.70"] += int(
            np.count_nonzero((p2 >= 0.30) & (ratio >= 0.70))
        )
        p2_probe_counts["p2_ge_0.35_ratio_ge_0.70"] += int(
            np.count_nonzero((p2 >= 0.35) & (ratio >= 0.70))
        )
        p2_probe_counts["p2_ge_0.35_ratio_ge_0.80"] += int(
            np.count_nonzero((p2 >= 0.35) & (ratio >= 0.80))
        )
        p2_probe_counts["p2_ge_0.40_ratio_ge_0.80"] += int(
            np.count_nonzero((p2 >= 0.40) & (ratio >= 0.80))
        )

        pair_all.update(pair_codes, p1, p2, ratio, top2_sum)
        for definition, mask in pair_definition_masks.items():
            pair_near[definition].update(
                pair_codes, p1, p2, ratio, top2_sum, mask=mask
            )
            for runner_up in range(N_BEHAVIORS):
                candidate_times[definition][runner_up].append(
                    elapsed_hours[mask & (top2 == runner_up)]
                )

        ct_hours = np.mod(
            start_ct + CT_HOURS * elapsed_hours / computational_frp_hours,
            CT_HOURS,
        )
        ct_bins = frozen_phase.phase_bin_indices(
            ct_hours,
            N_CT_BINS,
            frp_hours=CT_HOURS,
        )
        np.add.at(ct_total, ct_bins, 1)
        for definition in ct_ambiguous:
            np.add.at(ct_ambiguous[definition], ct_bins, definition_masks[definition])
        for cycle_index, cycle_start, cycle_end in complete_cycles:
            selected = (elapsed_hours >= cycle_start) & (elapsed_hours < cycle_end)
            if not selected.any():
                continue
            selected_cycle_times = elapsed_hours[selected] - cycle_start
            expected_bins = frozen_phase.ct_phase_bin_indices(
                selected_cycle_times,
                N_CT_BINS,
                frp_hours=computational_frp_hours,
            )
            # The existing MI code uses the absolute-time expression for the
            # all-available table and the cycle-relative expression for the
            # complete-cycle table.  At an exact floating-point bin boundary,
            # those two equivalent expressions can differ by one bin.  Keep
            # the absolute-time result for this all-recording CT profile and
            # record any such boundary-only discrepancy for validation.
            cycle_assignment_boundary_disagreements += int(
                np.count_nonzero(expected_bins != ct_bins[selected])
            )
            selected_cycle_frame_count += int(selected.sum())
            selected_cycle_bin_count += int(len(np.unique(expected_bins)))

        category_file_counts = [int(category_masks[label].sum()) for label in CATEGORY_LABELS]
        file_row: dict[str, object] = {
            "Group": group,
            "Animal": animal,
            "file_index": file_index,
            "filename": path.name,
            "elapsed_midpoint_hours": (file_index - files[0][0] + 0.5)
            * FILE_DURATION_MINUTES
            / 60.0,
            "total_frames": n_rows,
            "mean_row_probability_sum": float(row_sums.mean()),
            "min_row_probability_sum": float(row_sums.min()),
            "max_row_probability_sum": float(row_sums.max()),
            "ratio_ge_0.80_frames": int(definition_masks["A_ratio_ge_0.80"].sum()),
            "ratio_ge_0.80_fraction": float(definition_masks["A_ratio_ge_0.80"].mean()),
            "ratio_ge_0.90_frames": int(definition_masks["B_ratio_ge_0.90"].sum()),
            "ratio_ge_0.90_fraction": float(definition_masks["B_ratio_ge_0.90"].mean()),
            "pair_ambiguity_frames": int(definition_masks["C_ratio_ge_0.80_top2_ge_0.70"].sum()),
            "pair_ambiguity_fraction": float(definition_masks["C_ratio_ge_0.80_top2_ge_0.70"].mean()),
            "p2_ge_0.35_ratio_ge_0.70_frames": int(definition_masks["D_p2_ge_0.35_ratio_ge_0.70"].sum()),
            "p2_ge_0.35_ratio_ge_0.70_fraction": float(definition_masks["D_p2_ge_0.35_ratio_ge_0.70"].mean()),
            "diffuse_frames": int(definition_masks["diffuse_ratio_ge_0.80_top2_lt_0.60"].sum()),
            "diffuse_fraction": float(definition_masks["diffuse_ratio_ge_0.80_top2_lt_0.60"].mean()),
            "top_pair_all": f"{file_top_winner}->{file_top_runner}",
            "top_pair_all_count": int(file_pair_counts[file_top_pair_code]),
            "top_pair_all_fraction": float(
                file_pair_counts[file_top_pair_code] / n_rows
            ),
            "top_pair_c_pair": f"{file_c_winner}->{file_c_runner}" if file_c_winner else "",
            "top_pair_c_count": file_c_pair_count,
            "top_pair_c_fraction": (
                float(file_c_pair_count / definition_masks["C_ratio_ge_0.80_top2_ge_0.70"].sum())
                if file_c_pair_count
                else float("nan")
            ),
        }
        for index, category_label in enumerate(CATEGORY_LABELS):
            file_row[f"{category_label}_frames"] = category_file_counts[index]
            file_row[f"{category_label}_fraction"] = float(category_file_counts[index] / n_rows)
        file_rows.append(file_row)

    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int8)
    times = np.concatenate(all_times) if all_times else np.empty(0, dtype=float)
    if len(labels) != total_frames:
        raise AssertionError(f"Concatenated frame count mismatch for {animal}")
    transition_distances, transition_positions = _nearest_transition_distances(labels)
    frame_duration_seconds = float(np.median(median_frame_duration_seconds))

    transition_rows: list[dict[str, object]] = []
    # Pair candidate times and pair counts retain the exact representative
    # near-tie observations without duplicating all probability arrays.
    c_candidate_times = [
        np.concatenate(candidate_times["ratio_ge_0.80_top2_ge_0.70"][runner])
        if candidate_times["ratio_ge_0.80_top2_ge_0.70"][runner]
        else np.empty(0, dtype=float)
        for runner in range(N_BEHAVIORS)
    ]
    c_candidate_all_times = (
        np.concatenate([values for values in c_candidate_times if len(values)])
        if any(len(values) for values in c_candidate_times)
        else np.empty(0, dtype=float)
    )
    # Convert exact candidate timestamps back to frame positions.  Source
    # files are ordered and contiguous in this audit, so searchsorted is exact
    # up to the floating-point representation of the 10-minute grid.
    candidate_positions = np.searchsorted(times, c_candidate_all_times)
    candidate_positions = np.clip(candidate_positions, 0, max(total_frames - 1, 0))
    if len(candidate_positions):
        close_distances = transition_distances[candidate_positions]
    else:
        close_distances = np.empty(0, dtype=float)
    all_finite_distances = transition_distances[np.isfinite(transition_distances)]
    for scope, distances, scope_frames in (
        ("ALL_FRAMES", all_finite_distances, total_frames),
        ("C_ratio_ge_0.80_top2_ge_0.70", close_distances[np.isfinite(close_distances)], len(c_candidate_all_times)),
    ):
        median_distance_frames = (
            float(np.median(distances)) if len(distances) else float("nan")
        )
        p25_distance_frames = (
            float(np.quantile(distances, 0.25)) if len(distances) else float("nan")
        )
        p75_distance_frames = (
            float(np.quantile(distances, 0.75)) if len(distances) else float("nan")
        )
        for window in WINDOW_FRAMES:
            if scope == "ALL_FRAMES":
                qualifying = int(np.count_nonzero(transition_distances <= window))
                expected_fraction = qualifying / total_frames if total_frames else float("nan")
            else:
                qualifying = int(np.count_nonzero(close_distances <= window))
                expected_fraction = (
                    int(np.count_nonzero(transition_distances <= window)) / total_frames
                    if total_frames
                    else float("nan")
                )
            observed_fraction = qualifying / scope_frames if scope_frames else float("nan")
            transition_rows.append(
                {
                    "scope": scope,
                    "Group": group,
                    "Animal": animal,
                    "window_frames": window,
                    "window_seconds": window * frame_duration_seconds,
                    "frames_in_scope": scope_frames,
                    "frames_within_window": qualifying,
                    "fraction_within_window": observed_fraction,
                    "expected_all_frame_fraction": expected_fraction,
                    "enrichment_ratio": (
                        observed_fraction / expected_fraction
                        if np.isfinite(observed_fraction)
                        and np.isfinite(expected_fraction)
                        and expected_fraction > 0.0
                        else float("nan")
                    ),
                    "median_distance_frames": median_distance_frames,
                    "p25_distance_frames": p25_distance_frames,
                    "p75_distance_frames": p75_distance_frames,
                }
            )

    runner_rows: list[dict[str, object]] = []
    for runner_up in range(N_BEHAVIORS):
        candidate = c_candidate_times[runner_up]
        positive = (
            np.concatenate(positive_times[runner_up])
            if positive_times[runner_up]
            else np.empty(0, dtype=float)
        )
        distances_seconds = _nearest_positive_time_distances(candidate, positive)
        finite = distances_seconds[np.isfinite(distances_seconds)]
        row: dict[str, object] = {
            "definition": "C_ratio_ge_0.80_top2_ge_0.70",
            "Group": group,
            "Animal": animal,
            "RunnerUp": BEHAVIORS[runner_up],
            "ambiguous_frames": len(candidate),
            "median_distance_seconds": float(np.median(finite)) if len(finite) else float("nan"),
            "p25_distance_seconds": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
            "p75_distance_seconds": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
            "p90_distance_seconds": float(np.quantile(finite, 0.90)) if len(finite) else float("nan"),
        }
        for threshold in (1.0, 2.0, 5.0, 10.0):
            row[f"fraction_within_{int(threshold)}s"] = (
                float(np.count_nonzero(finite <= threshold) / len(finite))
                if len(finite)
                else float("nan")
            )
        runner_rows.append(row)

    definition_frames_for_rows = {
        **animal_definition_counts,
        "ratio_ge_0.80": animal_definition_counts["A_ratio_ge_0.80"],
        "ratio_ge_0.90": animal_definition_counts["B_ratio_ge_0.90"],
        "ratio_ge_0.80_top2_ge_0.70": animal_definition_counts["C_ratio_ge_0.80_top2_ge_0.70"],
        "diffuse_ratio_ge_0.80_top2_lt_0.60": animal_definition_counts[
            "diffuse_ratio_ge_0.80_top2_lt_0.60"
        ],
    }
    return {
        "group": group,
        "animal": animal,
        "reported_frp_hours": reported_frp_hours,
        "computational_frp_hours": computational_frp_hours,
        "start_ct": start_ct,
        "complete_cycles": complete_cycles,
        "files": len(files),
        "missing_indices": missing_indices,
        "total_frames": total_frames,
        "definition_counts": animal_definition_counts,
        "threshold_counts": threshold_counts,
        "p2_probe_counts": p2_probe_counts,
        "category_counts": category_counts,
        "file_rows": file_rows,
        "ct_total": ct_total,
        "ct_ambiguous": ct_ambiguous,
        "transition_rows": transition_rows,
        "runner_rows": runner_rows,
        "pair_all": pair_all,
        "pair_near": pair_near,
        "definition_frames_for_rows": definition_frames_for_rows,
        "row_sum_total": row_sum_total,
        "row_sum_min": row_sum_min,
        "row_sum_max": row_sum_max,
        "row_sum_max_abs_deviation": row_sum_max_abs_deviation,
        "selected_cycle_frame_count": selected_cycle_frame_count,
        "selected_cycle_bin_count": selected_cycle_bin_count,
        "cycle_assignment_boundary_disagreements": cycle_assignment_boundary_disagreements,
        "transition_count": len(transition_positions),
        "frame_duration_seconds": frame_duration_seconds,
    }


def _save_density_figure(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    diagonal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.pcolormesh(x_edges, y_edges, density.T, shading="auto")
    fig.colorbar(image, ax=ax, label="frames")
    if diagonal:
        upper = min(x_edges[-1], y_edges[-1])
        ax.plot([0.0, upper], [0.0, upper], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pair_heatmap(accumulator: PairAccumulator, path: Path) -> None:
    matrix = accumulator.counts.reshape(N_BEHAVIORS, N_BEHAVIORS)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="frames")
    ax.set_xticks(range(N_BEHAVIORS), BEHAVIORS, rotation=45, ha="right")
    ax.set_yticks(range(N_BEHAVIORS), BEHAVIORS)
    ax.set_xlabel("runner-up")
    ax.set_ylabel("winner")
    ax.set_title("Ordered winner → runner-up counts: ratio ≥ 0.80 and top2 ≥ 0.70")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_rate_figure(results: list[dict[str, object]], path: Path) -> None:
    labels = [str(result["animal"]) for result in results]
    definitions = (
        ("A_ratio_ge_0.80", "A ratio ≥ 0.80"),
        ("B_ratio_ge_0.90", "B ratio ≥ 0.90"),
        ("C_ratio_ge_0.80_top2_ge_0.70", "C ratio ≥ 0.80, top2 ≥ 0.70"),
        ("D_p2_ge_0.35_ratio_ge_0.70", "D p2 ≥ 0.35, ratio ≥ 0.70"),
    )
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, (definition, label) in enumerate(definitions):
        values = [
            int(result["definition_counts"][definition]) / int(result["total_frames"])
            for result in results
        ]
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("fraction of frames")
    ax.set_title("Descriptive ambiguity rates by animal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_ct_figure(ct_rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(ct_rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, definition in zip(
        axes,
        ("A_ratio_ge_0.80", "C_ratio_ge_0.80_top2_ge_0.70"),
    ):
        subset = frame[frame["definition"] == definition]
        for animal in [animal for _, animal in ANIMALS]:
            animal_subset = subset[subset["Animal"] == animal].sort_values("CT_bin")
            if animal_subset.empty:
                continue
            axis.plot(
                animal_subset["CT_start_h"],
                animal_subset["fraction_ambiguous"],
                linewidth=0.8,
                label=animal,
            )
        axis.set_ylabel("fraction")
        axis.set_title(definition)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("CT bin start (hours)")
    axes[0].legend(ncol=4, fontsize=8)
    fig.suptitle("Ambiguity profiles across assigned circadian phase")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_transition_figure(transition_rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(transition_rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for scope, label in (
        ("ALL_FRAMES", "all frames"),
        ("C_ratio_ge_0.80_top2_ge_0.70", "pair-ambiguous frames"),
    ):
        subset = frame[frame["scope"] == scope]
        if subset.empty:
            continue
        grouped = subset.groupby("window_frames", as_index=False)["fraction_within_window"].mean()
        ax.plot(grouped["window_frames"], grouped["fraction_within_window"], marker="o", label=label)
    ax.set_xlabel("window around WTA transition (frames)")
    ax.set_ylabel("mean fraction within window")
    ax.set_title("Proximity of ambiguity to WTA transitions")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_confidence_rows(
    pooled_histogram: HistogramCollection,
    group_histograms: dict[str, HistogramCollection],
    animal_histograms: dict[str, HistogramCollection],
) -> list[dict[str, object]]:
    rows = pooled_histogram.rows("POOLED")
    for group, histogram in group_histograms.items():
        rows.extend(histogram.rows("GENOTYPE_QC", group=group))
    for animal, histogram in animal_histograms.items():
        group = next(group for group, candidate in ANIMALS if candidate == animal)
        rows.extend(histogram.rows("ANIMAL", group=group, animal=animal))
    return rows


def _build_threshold_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    pooled_counts = defaultdict(int)
    group_counts: dict[str, defaultdict[tuple[float, float], int]] = {}
    total_frames = sum(int(result["total_frames"]) for result in results)
    group_totals = defaultdict(int)
    rows: list[dict[str, object]] = []
    for result in results:
        group = str(result["group"])
        group_counts.setdefault(group, defaultdict(int))
        group_totals[group] += int(result["total_frames"])
        for key, count in result["threshold_counts"].items():
            pooled_counts[key] += int(count)
            group_counts[group][key] += int(count)
    for ratio_cutoff in RATIO_CUTOFFS:
        for top2_cutoff in TOP2_SUM_CUTOFFS:
            key = (ratio_cutoff, top2_cutoff)
            rows.append(
                _scope_row(
                    "POOLED",
                    "",
                    "",
                    total_frames,
                    pooled_counts[key],
                    ratio_cutoff=ratio_cutoff,
                    top2_sum_cutoff=top2_cutoff,
                )
            )
            for group, counts in group_counts.items():
                rows.append(
                    _scope_row(
                        "GENOTYPE_QC",
                        group,
                        "",
                        group_totals[group],
                        counts[key],
                        ratio_cutoff=ratio_cutoff,
                        top2_sum_cutoff=top2_cutoff,
                    )
                )
            for result in results:
                rows.append(
                    _scope_row(
                        "ANIMAL",
                        str(result["group"]),
                        str(result["animal"]),
                        int(result["total_frames"]),
                        int(result["threshold_counts"][key]),
                        ratio_cutoff=ratio_cutoff,
                        top2_sum_cutoff=top2_cutoff,
                    )
                )
    return rows


def _build_runner_threshold_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    probe_labels = [f"p2_ge_{cutoff:.2f}" for cutoff in P2_CUTOFFS] + [
        "p2_ge_0.30_ratio_ge_0.70",
        "p2_ge_0.35_ratio_ge_0.70",
        "p2_ge_0.35_ratio_ge_0.80",
        "p2_ge_0.40_ratio_ge_0.80",
    ]
    pooled = defaultdict(int)
    group_counts: dict[str, defaultdict[str, int]] = {}
    group_totals = defaultdict(int)
    for result in results:
        group = str(result["group"])
        group_counts.setdefault(group, defaultdict(int))
        group_totals[group] += int(result["total_frames"])
        for label in probe_labels:
            pooled[label] += int(result["p2_probe_counts"][label])
            group_counts[group][label] += int(result["p2_probe_counts"][label])
    total_frames = sum(int(result["total_frames"]) for result in results)
    rows: list[dict[str, object]] = []
    for label in probe_labels:
        if label.startswith("p2_ge_") and "ratio" not in label:
            p2_cutoff = float(label.removeprefix("p2_ge_"))
            ratio_cutoff = ""
        else:
            p2_cutoff = {
                "p2_ge_0.30_ratio_ge_0.70": 0.30,
                "p2_ge_0.35_ratio_ge_0.70": 0.35,
                "p2_ge_0.35_ratio_ge_0.80": 0.35,
                "p2_ge_0.40_ratio_ge_0.80": 0.40,
            }[label]
            ratio_cutoff = {
                "p2_ge_0.30_ratio_ge_0.70": 0.70,
                "p2_ge_0.35_ratio_ge_0.70": 0.70,
                "p2_ge_0.35_ratio_ge_0.80": 0.80,
                "p2_ge_0.40_ratio_ge_0.80": 0.80,
            }[label]
        rows.append(
            _scope_row(
                "POOLED",
                "",
                "",
                total_frames,
                pooled[label],
                definition=label,
                p2_cutoff=p2_cutoff,
                ratio_cutoff=ratio_cutoff,
            )
        )
        for group, counts in group_counts.items():
            rows.append(
                _scope_row(
                    "GENOTYPE_QC",
                    group,
                    "",
                    group_totals[group],
                    counts[label],
                    definition=label,
                    p2_cutoff=p2_cutoff,
                    ratio_cutoff=ratio_cutoff,
                )
            )
        for result in results:
            rows.append(
                _scope_row(
                    "ANIMAL",
                    str(result["group"]),
                    str(result["animal"]),
                    int(result["total_frames"]),
                    int(result["p2_probe_counts"][label]),
                    definition=label,
                    p2_cutoff=p2_cutoff,
                    ratio_cutoff=ratio_cutoff,
                )
            )
    return rows


def _build_ct_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bin_width = CT_HOURS / N_CT_BINS
    for result in results:
        total = result["ct_total"]
        for definition, counts in result["ct_ambiguous"].items():
            for ct_bin in range(N_CT_BINS):
                total_frames = int(total[ct_bin])
                ambiguous_frames = int(counts[ct_bin])
                rows.append(
                    {
                        "Animal": result["animal"],
                        "Group": result["group"],
                        "CT_bin": ct_bin,
                        "CT_start_h": ct_bin * bin_width,
                        "definition": definition,
                        "ambiguous_frames": ambiguous_frames,
                        "total_frames": total_frames,
                        "fraction_ambiguous": (
                            ambiguous_frames / total_frames if total_frames else float("nan")
                        ),
                    }
                )
    return rows


def _build_low_confidence_rows(
    results: list[dict[str, object]],
    pooled_pair_near: dict[str, PairAccumulator],
    group_pair_near: dict[str, dict[str, PairAccumulator]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_frames = sum(int(result["total_frames"]) for result in results)
    pooled_count = sum(
        int(result["definition_counts"]["diffuse_ratio_ge_0.80_top2_lt_0.60"])
        for result in results
    )
    scopes = [("POOLED", "", "", total_frames, pooled_count, pooled_pair_near["diffuse_ratio_ge_0.80_top2_lt_0.60"])]
    for group, accumulator in group_pair_near.items():
        group_results = [result for result in results if result["group"] == group]
        group_total = sum(int(result["total_frames"]) for result in group_results)
        group_count = sum(
            int(result["definition_counts"]["diffuse_ratio_ge_0.80_top2_lt_0.60"])
            for result in group_results
        )
        scopes.append(
            (
                "GENOTYPE_QC",
                group,
                "",
                group_total,
                group_count,
                accumulator["diffuse_ratio_ge_0.80_top2_lt_0.60"],
            )
        )
    for result in results:
        scopes.append(
            (
                "ANIMAL",
                str(result["group"]),
                str(result["animal"]),
                int(result["total_frames"]),
                int(result["definition_counts"]["diffuse_ratio_ge_0.80_top2_lt_0.60"]),
                result["pair_near"]["diffuse_ratio_ge_0.80_top2_lt_0.60"],
            )
        )
    for scope, group, animal, total, count, accumulator in scopes:
        winner, runner_up, pair_count = _top_pair(accumulator)
        rows.append(
            {
                "scope": scope,
                "Group": group,
                "Animal": animal,
                "definition": "diffuse_ratio_ge_0.80_top2_lt_0.60",
                "qualifying_frames": count,
                "total_frames": total,
                "fraction_frames": count / total if total else float("nan"),
                "top_pair": f"{winner}->{runner_up}" if winner else "",
                "top_pair_count": pair_count,
                "top_pair_fraction_of_diffuse": pair_count / count if count else float("nan"),
            }
        )
    return rows


def _build_frame_summary_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build one aggregate QC row per animal for the frame summary table."""

    rows: list[dict[str, object]] = []
    for result in results:
        total_frames = int(result["total_frames"])
        row: dict[str, object] = {
            "Group": result["group"],
            "Animal": result["animal"],
            "source_file_count": int(result["files"]),
            "total_frames": total_frames,
            "missing_source_index_count": len(result["missing_indices"]),
            "reported_frp_hours": result["reported_frp_hours"],
            "computational_frp_hours": result["computational_frp_hours"],
            "start_ct": result["start_ct"],
            "complete_cycle_count": len(result["complete_cycles"]),
            "selected_complete_cycle_frames": result["selected_cycle_frame_count"],
            "wta_transition_count": result["transition_count"],
            "median_frame_duration_seconds": result["frame_duration_seconds"],
            "mean_row_probability_sum": result["row_sum_total"] / total_frames,
            "min_row_probability_sum": result["row_sum_min"],
            "max_row_probability_sum": result["row_sum_max"],
            "max_abs_row_sum_deviation": result["row_sum_max_abs_deviation"],
            "cycle_boundary_rounding_disagreements": result[
                "cycle_assignment_boundary_disagreements"
            ],
        }
        for definition, count in result["definition_counts"].items():
            row[f"{definition}_frames"] = int(count)
            row[f"{definition}_fraction"] = count / total_frames if total_frames else float("nan")
        for category, count in result["category_counts"].items():
            row[f"{category}_frames"] = int(count)
            row[f"{category}_fraction"] = count / total_frames if total_frames else float("nan")
        rows.append(row)
    return rows


def _build_pair_outputs(
    results: list[dict[str, object]],
    pooled_pair_all: PairAccumulator,
    pooled_pair_near: dict[str, PairAccumulator],
    group_pair_all: dict[str, PairAccumulator],
    group_pair_near: dict[str, dict[str, PairAccumulator]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    near_rows: list[dict[str, object]] = []
    total_frames = sum(int(result["total_frames"]) for result in results)
    all_rows.extend(_pair_rows(pooled_pair_all, "ALL", "POOLED", "", "", total_frames, total_frames))
    for group, accumulator in group_pair_all.items():
        group_total = sum(int(result["total_frames"]) for result in results if result["group"] == group)
        all_rows.extend(_pair_rows(accumulator, "ALL", "GENOTYPE_QC", group, "", group_total, group_total))
    for result in results:
        all_rows.extend(
            _pair_rows(
                result["pair_all"],
                "ALL",
                "ANIMAL",
                str(result["group"]),
                str(result["animal"]),
                int(result["total_frames"]),
                int(result["total_frames"]),
            )
        )
    for definition in NEAR_PAIR_DEFINITIONS:
        pooled_definition_frames = sum(
            int(result["definition_frames_for_rows"][definition]) for result in results
        )
        near_rows.extend(
            _pair_rows(
                pooled_pair_near[definition],
                definition,
                "POOLED",
                "",
                "",
                total_frames,
                pooled_definition_frames,
            )
        )
        for group, accumulators in group_pair_near.items():
            group_results = [result for result in results if result["group"] == group]
            group_total = sum(int(result["total_frames"]) for result in group_results)
            group_definition_frames = sum(
                int(result["definition_frames_for_rows"][definition])
                for result in group_results
            )
            near_rows.extend(
                _pair_rows(
                    accumulators[definition],
                    definition,
                    "GENOTYPE_QC",
                    group,
                    "",
                    group_total,
                    group_definition_frames,
                )
            )
        for result in results:
            near_rows.extend(
                _pair_rows(
                    result["pair_near"][definition],
                    definition,
                    "ANIMAL",
                    str(result["group"]),
                    str(result["animal"]),
                    int(result["total_frames"]),
                    int(result["definition_frames_for_rows"][definition]),
                )
            )
    return all_rows, near_rows


def run_audit(cohort_root: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_histogram = HistogramCollection()
    group_histograms: dict[str, HistogramCollection] = {}
    animal_histograms: dict[str, HistogramCollection] = {}
    density_p1_p2_edges = np.linspace(0.0, 1.05, 106)
    density_ratio_edges = np.linspace(0.0, 1.05, 106)
    density_top2_edges = np.linspace(0.0, 1.10, 111)
    density_p1_p2 = np.zeros((len(density_p1_p2_edges) - 1,) * 2, dtype=np.int64)
    density_ratio_top2 = np.zeros(
        (len(density_ratio_edges) - 1, len(density_top2_edges) - 1), dtype=np.int64
    )
    pooled_pair_all = PairAccumulator()
    pooled_pair_near = {definition: PairAccumulator() for definition in NEAR_PAIR_DEFINITIONS}
    group_pair_all: dict[str, PairAccumulator] = {}
    group_pair_near: dict[str, dict[str, PairAccumulator]] = {}
    results: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    runner_rows: list[dict[str, object]] = []

    for index, (group, animal) in enumerate(ANIMALS, start=1):
        print(f"[{index}/{len(ANIMALS)}] auditing {group}/{animal}", flush=True)
        result = _process_animal(
            cohort_root,
            group,
            animal,
            pooled_histogram,
            group_histograms,
            animal_histograms,
            density_p1_p2,
            density_ratio_top2,
            density_p1_p2_edges,
            density_ratio_edges,
            density_top2_edges,
            pooled_pair_all,
            pooled_pair_near,
            group_pair_all,
            group_pair_near,
        )
        results.append(result)
        file_rows.extend(result["file_rows"])
        transition_rows.extend(result["transition_rows"])
        runner_rows.extend(result["runner_rows"])
        pooled_pair_all.add(result["pair_all"])
        for definition in NEAR_PAIR_DEFINITIONS:
            pooled_pair_near[definition].add(result["pair_near"][definition])
        group_pair_all[str(result["group"])].add(result["pair_all"])
        for definition in NEAR_PAIR_DEFINITIONS:
            group_pair_near[str(result["group"])][definition].add(
                result["pair_near"][definition]
            )

    confidence_rows = _build_confidence_rows(
        pooled_histogram, group_histograms, animal_histograms
    )
    frame_summary_rows = _build_frame_summary_rows(results)
    threshold_rows = _build_threshold_rows(results)
    runner_threshold_rows = _build_runner_threshold_rows(results)
    ct_rows = _build_ct_rows(results)
    pair_rows, pair_near_rows = _build_pair_outputs(
        results,
        pooled_pair_all,
        pooled_pair_near,
        group_pair_all,
        group_pair_near,
    )
    low_confidence_rows = _build_low_confidence_rows(
        results, pooled_pair_near, group_pair_near
    )

    _write_csv(frame_summary_rows, output_dir / "ambiguity_frame_summary.csv")
    _write_csv(threshold_rows, output_dir / "ambiguity_threshold_surface.csv")
    _write_csv(runner_threshold_rows, output_dir / "ambiguity_runnerup_thresholds.csv")
    _write_csv(pair_rows, output_dir / "ambiguity_pair_counts.csv")
    _write_csv(pair_near_rows, output_dir / "ambiguity_pair_counts_neartie.csv")
    _write_csv(ct_rows, output_dir / "ambiguity_ct_profile.csv")
    _write_csv(file_rows, output_dir / "ambiguity_file_profile.csv")
    _write_csv(confidence_rows, output_dir / "ambiguity_confidence_summary.csv")
    _write_csv(transition_rows, output_dir / "ambiguity_transition_proximity.csv")
    _write_csv(runner_rows, output_dir / "ambiguity_runnerup_bout_proximity.csv")
    _write_csv(low_confidence_rows, output_dir / "ambiguity_low_confidence_summary.csv")

    _save_density_figure(
        density_p1_p2,
        density_p1_p2_edges,
        density_p1_p2_edges,
        "p1",
        "p2",
        "Top-1 versus top-2 probability density",
        output_dir / "ambiguity_p1_p2_density.png",
        diagonal=True,
    )
    _save_density_figure(
        density_ratio_top2,
        density_ratio_edges,
        density_top2_edges,
        "p2 / p1",
        "top2 sum",
        "Ratio versus top-two probability-sum density",
        output_dir / "ambiguity_ratio_top2sum_density.png",
    )
    _save_pair_heatmap(
        pooled_pair_near["ratio_ge_0.80_top2_ge_0.70"],
        output_dir / "ambiguity_pair_heatmap.png",
    )
    _save_rate_figure(results, output_dir / "ambiguity_rate_by_animal.png")
    _save_ct_figure(ct_rows, output_dir / "ambiguity_CT_profiles.png")
    _save_transition_figure(
        transition_rows, output_dir / "ambiguity_transition_proximity.png"
    )

    total_frames = sum(int(result["total_frames"]) for result in results)
    max_row_deviation = max(
        float(result["row_sum_max_abs_deviation"]) for result in results
    )
    selected_cycle_frames = sum(int(result["selected_cycle_frame_count"]) for result in results)
    selected_cycle_bins = sum(int(result["selected_cycle_bin_count"]) for result in results)
    print(f"Wrote ambiguity census to {output_dir}", flush=True)
    print(f"Animals: {len(results)}; frames: {total_frames}", flush=True)
    print(f"Maximum probability row-sum deviation: {max_row_deviation:.3g}", flush=True)
    print(
        "Frozen CT validation: "
        f"{selected_cycle_frames} complete-cycle frames, {selected_cycle_bins} cycle-bin observations",
        flush=True,
    )
    print("WTA validation: existing classifier labels equal top-1 for every frame", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_root", type=Path, help="Pilot Cohort_Data root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <cohort_root>/Ambiguity_Census)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.cohort_root / "Ambiguity_Census"
    run_audit(args.cohort_root, output_dir)


if __name__ == "__main__":
    main()
