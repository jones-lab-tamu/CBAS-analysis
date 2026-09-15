"""Estimate FRP and assign CT/cycle labels for one animal's CBAS files.

This is a deliberately small first-stage analysis.  It reduces each 10-minute
file to a winner-take-all resting fraction, estimates period with only
Lomb--Scargle and cosinor searches, and assigns CT using the schedule-derived
starting phase.  It does not analyze individual behavior phases.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

import phase_behavior_mutual_information as mi


FILE_PATTERN = re.compile(
    r"^(?P<animal>.+)_(?P<index>\d{5})_curated_aug_model_outputs\.csv$"
)

BIN_DURATION_HOURS = mi.FILE_DURATION_MINUTES / 60.0
MIN_FRP_HOURS = 20.0
MAX_FRP_HOURS = 28.0
PERIOD_GRID_STEP_HOURS = 0.001
PERIOD_REPORT_DECIMALS = 2
DEFAULT_START_CT = 18.0
OUTPUT_NAMES = {
    "frp_phase_summary.csv",
    "frp_phase_bins.csv",
    "frp_phase_cycles.csv",
    "frp_phase_diagnostic.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate FRP and assign CT/cycles for one CBAS animal."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="One animal directory containing CBAS model-output CSV files.",
    )
    parser.add_argument(
        "--start-ct",
        type=float,
        default=DEFAULT_START_CT,
        help="CT at elapsed time zero (default: 18.0).",
    )
    return parser.parse_args()


def discover_files(
    input_dir: Path,
) -> tuple[list[tuple[int, Path]], list[int], str]:
    """Find and numerically order source files without filling gaps."""

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )

    parsed: list[tuple[int, Path]] = []
    prefixes: set[str] = set()
    for path in input_dir.glob("*.csv"):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        prefixes.add(match.group("animal"))
        parsed.append((int(match.group("index")), path))

    if not parsed:
        raise FileNotFoundError(
            f"No CBAS source CSV files found in {input_dir}"
        )
    if len(prefixes) != 1:
        raise ValueError(
            "Expected one filename prefix in the animal directory; found "
            + ", ".join(sorted(prefixes))
        )

    parsed.sort(key=lambda item: item[0])
    indices = [index for index, _ in parsed]
    duplicates = sorted(
        index for index, count in pd.Series(indices).value_counts().items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate numeric file indices found: {duplicates}")

    first_index = indices[0]
    last_index = indices[-1]
    expected = set(range(first_index, last_index + 1))
    missing = sorted(expected.difference(indices))
    return parsed, missing, next(iter(prefixes))


def build_rest_bins(
    input_files: list[tuple[int, Path]],
    missing_indices: list[int],
) -> tuple[list[dict[str, object]], float, list[str]]:
    """Build one rest fraction for every expected file slot.

    Missing slots are retained as rows with a missing flag and no signal
    value.  Their elapsed positions are therefore not compressed out of the
    period-estimation time coordinate.
    """

    first_index = input_files[0][0]
    last_index = input_files[-1][0]
    observed_by_index: dict[int, dict[str, object]] = {}
    warnings: list[str] = []

    for file_index, path in input_files:
        n_rows, _, labels = mi.read_and_classify(path)
        valid_count = int(labels.size)
        invalid_count = int(n_rows - valid_count)
        if invalid_count:
            warnings.append(
                f"{path.name} contains {invalid_count} unclassifiable rows"
            )
        rest_fraction = (
            float(np.mean(labels == mi.RESTING_BEHAVIOR_INDEX))
            if valid_count
            else float("nan")
        )
        if not valid_count:
            warnings.append(f"{path.name} contains no valid behavior samples")
        observed_by_index[file_index] = {
            "filename": path.name,
            "rest_fraction": rest_fraction,
        }

    bins: list[dict[str, object]] = []
    for file_index in range(first_index, last_index + 1):
        elapsed_midpoint = (
            file_index - first_index + 0.5
        ) * BIN_DURATION_HOURS
        if file_index in observed_by_index:
            record = observed_by_index[file_index]
            bins.append(
                {
                    "file_index": file_index,
                    "filename": record["filename"],
                    "elapsed_midpoint_hours": elapsed_midpoint,
                    "rest_fraction": record["rest_fraction"],
                    "missing_bin": False,
                }
            )
        else:
            bins.append(
                {
                    "file_index": file_index,
                    "filename": "",
                    "elapsed_midpoint_hours": elapsed_midpoint,
                    "rest_fraction": float("nan"),
                    "missing_bin": True,
                }
            )

    duration_hours = (last_index - first_index + 1) * BIN_DURATION_HOURS
    if missing_indices:
        warnings.append(
            "Missing file indices within the supplied sequence: "
            + ", ".join(str(index) for index in missing_indices)
        )
    if first_index != 0:
        warnings.append(
            f"File sequence starts at index {first_index}; elapsed time is anchored "
            "to the first supplied file because earlier files are not present"
        )
    return bins, duration_hours, warnings


def period_grid() -> np.ndarray:
    count = int(round((MAX_FRP_HOURS - MIN_FRP_HOURS) / PERIOD_GRID_STEP_HOURS))
    return np.linspace(MIN_FRP_HOURS, MAX_FRP_HOURS, count + 1)


def estimate_lomb_scargle_period(
    times_hours: np.ndarray,
    values: np.ndarray,
    periods_hours: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the period at the strongest normalized Lomb--Scargle peak."""

    if times_hours.size < 3:
        raise ValueError("At least three valid 10-minute bins are needed for FRP")
    centered = values - float(np.mean(values))
    if not np.isfinite(centered).all() or float(np.std(centered)) == 0.0:
        raise ValueError("The rest-fraction signal has no usable variation")

    angular_frequencies = 2.0 * np.pi / periods_hours
    power = lombscargle(
        times_hours,
        centered,
        angular_frequencies,
        precenter=False,
        normalize=True,
    )
    if not np.isfinite(power).all():
        raise ValueError("Lomb--Scargle returned non-finite power values")
    peak_index = int(np.argmax(power))
    return float(periods_hours[peak_index]), power


def cosinor_residual_for_period(
    times_hours: np.ndarray,
    values: np.ndarray,
    period_hours: float,
) -> float:
    angular_phase = 2.0 * np.pi * times_hours / period_hours
    design = np.column_stack(
        (np.ones(times_hours.size), np.cos(angular_phase), np.sin(angular_phase))
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    return float(np.sum(residuals**2))


def estimate_cosinor_period(
    times_hours: np.ndarray,
    values: np.ndarray,
    periods_hours: np.ndarray,
) -> float:
    """Return the grid period with minimum simple cosinor residual error."""

    residuals = np.asarray(
        [
            cosinor_residual_for_period(times_hours, values, period)
            for period in periods_hours
        ],
        dtype=float,
    )
    if not np.isfinite(residuals).all():
        raise ValueError("Cosinor period search returned non-finite residuals")
    return float(periods_hours[int(np.argmin(residuals))])


def fit_cosinor(
    times_hours: np.ndarray,
    values: np.ndarray,
    period_hours: float,
) -> np.ndarray:
    """Fit the simple intercept/cosine/sine model for the diagnostic overlay."""

    angular_phase = 2.0 * np.pi * times_hours / period_hours
    design = np.column_stack(
        (np.ones(times_hours.size), np.cos(angular_phase), np.sin(angular_phase))
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(coefficients, dtype=float)


def assign_ct_and_cycle(
    elapsed_hours: float,
    frp_hours: float,
    start_ct: float,
) -> tuple[int, float]:
    """Map elapsed real time to zero-based cycle index and conventional CT."""

    start_phase_fraction = start_ct / 24.0
    unwrapped_cycle_position = start_phase_fraction + elapsed_hours / frp_hours
    cycle_index = int(math.floor(unwrapped_cycle_position))
    phase_fraction = unwrapped_cycle_position - cycle_index
    ct = 24.0 * phase_fraction
    if math.isclose(ct, 24.0, rel_tol=0.0, abs_tol=1e-10):
        ct = 0.0
    return cycle_index, float(ct)


def assign_bins(
    bins: list[dict[str, object]],
    frp_hours: float,
    start_ct: float,
) -> None:
    for record in bins:
        cycle_index, ct = assign_ct_and_cycle(
            float(record["elapsed_midpoint_hours"]), frp_hours, start_ct
        )
        record["cycle_index"] = cycle_index
        record["CT"] = ct


def build_cycle_rows(
    bins: list[dict[str, object]],
    duration_hours: float,
    frp_hours: float,
    start_ct: float,
) -> list[dict[str, object]]:
    """Summarize each represented biological cycle and its missing bins."""

    start_phase_fraction = start_ct / 24.0
    cycle_indices = sorted({int(record["cycle_index"]) for record in bins})
    rows: list[dict[str, object]] = []
    tolerance = 1e-10

    for cycle_index in cycle_indices:
        cycle_bins = [
            record
            for record in bins
            if int(record["cycle_index"]) == cycle_index
        ]
        elapsed_start = (cycle_index - start_phase_fraction) * frp_hours
        elapsed_end = (cycle_index + 1 - start_phase_fraction) * frp_hours
        observed_count = sum(
            not bool(record["missing_bin"]) for record in cycle_bins
        )
        missing_count = sum(bool(record["missing_bin"]) for record in cycle_bins)
        spans_available_recording = (
            elapsed_start >= -tolerance and elapsed_end <= duration_hours + tolerance
        )
        full_cycle = bool(spans_available_recording and missing_count == 0)
        rows.append(
            {
                "cycle_index": cycle_index,
                "elapsed_start_boundary_hours": elapsed_start,
                "elapsed_end_boundary_hours": elapsed_end,
                "observed_bin_count": observed_count,
                "expected_duration_hours": frp_hours,
                "missing_bin_count": missing_count,
                "full_cycle_flag": full_cycle,
            }
        )

    status_by_cycle = {
        int(row["cycle_index"]): ("full" if row["full_cycle_flag"] else "partial")
        for row in rows
    }
    for record in bins:
        record["cycle_status"] = status_by_cycle[int(record["cycle_index"])]
    return rows


def ensure_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        unexpected = {
            path.name for path in output_dir.iterdir() if path.name not in OUTPUT_NAMES
        }
        if unexpected:
            raise ValueError(
                "Output directory contains files outside this task's four outputs: "
                f"{sorted(unexpected)}"
            )
    else:
        output_dir.mkdir(parents=True)


def write_diagnostic_figure(
    output_path: Path,
    bins: list[dict[str, object]],
    cycle_rows: list[dict[str, object]],
    duration_hours: float,
    selected_frp_hours: float,
    start_ct: float,
    cosinor_coefficients: np.ndarray,
) -> None:
    times = np.asarray(
        [float(record["elapsed_midpoint_hours"]) for record in bins], dtype=float
    )
    values = np.asarray(
        [float(record["rest_fraction"]) for record in bins], dtype=float
    )
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(
        times,
        values,
        color="tab:blue",
        linewidth=0.8,
        marker=".",
        markersize=2.5,
        label="10-minute rest fraction",
    )

    model_times = np.linspace(0.0, duration_hours, 2000)
    model_phase = 2.0 * np.pi * model_times / selected_frp_hours
    model_values = (
        cosinor_coefficients[0]
        + cosinor_coefficients[1] * np.cos(model_phase)
        + cosinor_coefficients[2] * np.sin(model_phase)
    )
    axis.plot(
        model_times,
        model_values,
        color="tab:orange",
        linewidth=1.5,
        label=f"Cosinor overlay at selected FRP ({selected_frp_hours:.2f} h)",
    )

    boundary_labeled = False
    for row in cycle_rows:
        boundary = float(row["elapsed_start_boundary_hours"])
        if 0.0 <= boundary <= duration_hours:
            axis.axvline(
                boundary,
                color="tab:red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                label="CT0 / cycle boundary" if not boundary_labeled else None,
            )
            boundary_labeled = True
            axis.text(
                boundary,
                1.02,
                f"CT0\ncycle {int(row['cycle_index'])}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:red",
            )

    axis.set_title(
        f"Rest rhythm and CT assignment (start CT={start_ct:.1f}, "
        f"selected FRP={selected_frp_hours:.2f} h)",
        pad=34,
    )
    axis.set_xlabel("Elapsed time from first supplied file (hours)")
    axis.set_ylabel("Rest fraction")
    axis.set_ylim(-0.02, 1.08)
    axis.set_xlim(0.0, duration_hours)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_analysis(input_dir: Path, start_ct: float) -> dict[str, object]:
    started = time.perf_counter()
    if not np.isfinite(start_ct) or not 0.0 <= start_ct < 24.0:
        raise ValueError("start_ct must be finite and in the half-open range [0, 24)")

    input_files, missing_indices, filename_prefix = discover_files(input_dir)
    bins, duration_hours, warnings = build_rest_bins(input_files, missing_indices)
    if filename_prefix != input_dir.name:
        warnings.append(
            f"Filename prefix is {filename_prefix}, which differs from input folder "
            f"name {input_dir.name}"
        )
    warnings.append(
        f"start_ct={start_ct:.1f} is a schedule-derived input assumption, not inferred "
        "or optimized from the rest rhythm"
    )

    observed_bins = [
        record
        for record in bins
        if not bool(record["missing_bin"])
        and np.isfinite(float(record["rest_fraction"]))
    ]
    times = np.asarray(
        [float(record["elapsed_midpoint_hours"]) for record in observed_bins],
        dtype=float,
    )
    values = np.asarray(
        [float(record["rest_fraction"]) for record in observed_bins],
        dtype=float,
    )
    periods = period_grid()
    lomb_period, _ = estimate_lomb_scargle_period(times, values, periods)
    cosinor_period = estimate_cosinor_period(times, values, periods)
    selected_frp = lomb_period
    warnings.append(
        f"Period grid step was {PERIOD_GRID_STEP_HOURS:g} h; period fields are "
        f"reported to {PERIOD_REPORT_DECIMALS} decimals and CT assignment used the "
        "unrounded Lomb--Scargle peak"
    )
    assign_bins(bins, selected_frp, start_ct)
    cycle_rows = build_cycle_rows(bins, duration_hours, selected_frp, start_ct)
    cosinor_coefficients = fit_cosinor(times, values, selected_frp)

    if math.isclose(lomb_period, MIN_FRP_HOURS, abs_tol=PERIOD_GRID_STEP_HOURS):
        warnings.append("Lomb--Scargle peak is at the lower 20-hour search boundary")
    if math.isclose(lomb_period, MAX_FRP_HOURS, abs_tol=PERIOD_GRID_STEP_HOURS):
        warnings.append("Lomb--Scargle peak is at the upper 28-hour search boundary")
    if math.isclose(cosinor_period, MIN_FRP_HOURS, abs_tol=PERIOD_GRID_STEP_HOURS):
        warnings.append("Cosinor minimum is at the lower 20-hour search boundary")
    if math.isclose(cosinor_period, MAX_FRP_HOURS, abs_tol=PERIOD_GRID_STEP_HOURS):
        warnings.append("Cosinor minimum is at the upper 28-hour search boundary")

    full_cycle_count = sum(bool(row["full_cycle_flag"]) for row in cycle_rows)
    output_dir = input_dir / "FRP_Phase_Output"
    ensure_output_directory(output_dir)

    summary_row = {
        "animal_id": input_dir.name,
        "first_file_index": input_files[0][0],
        "last_file_index": input_files[-1][0],
        "number_of_files": len(input_files),
        "missing_file_count": len(missing_indices),
        "total_elapsed_recording_duration_hours": duration_hours,
        "lomb_scargle_frp_hours": round(lomb_period, PERIOD_REPORT_DECIMALS),
        "cosinor_frp_hours": round(cosinor_period, PERIOD_REPORT_DECIMALS),
        "ls_minus_cosinor_hours": round(
            lomb_period - cosinor_period, PERIOD_REPORT_DECIMALS
        ),
        "selected_frp_hours": round(selected_frp, PERIOD_REPORT_DECIMALS),
        "start_ct_used": start_ct,
        "number_of_full_ct0_ct24_cycles": full_cycle_count,
        "notes_warnings": "; ".join(dict.fromkeys(warnings)) or "None",
    }
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "frp_phase_summary.csv", index=False
    )

    bins_frame = pd.DataFrame(
        bins,
        columns=[
            "file_index",
            "filename",
            "elapsed_midpoint_hours",
            "rest_fraction",
            "cycle_index",
            "CT",
            "cycle_status",
            "missing_bin",
        ],
    )
    bins_frame.to_csv(output_dir / "frp_phase_bins.csv", index=False)
    pd.DataFrame(cycle_rows).to_csv(
        output_dir / "frp_phase_cycles.csv", index=False
    )
    write_diagnostic_figure(
        output_dir / "frp_phase_diagnostic.png",
        bins,
        cycle_rows,
        duration_hours,
        selected_frp,
        start_ct,
        cosinor_coefficients,
    )

    total_runtime = float(time.perf_counter() - started)
    ct0_boundaries = [
        float(row["elapsed_start_boundary_hours"])
        for row in cycle_rows
        if 0.0 <= float(row["elapsed_start_boundary_hours"]) <= duration_hours
    ]
    return {
        "input_dir": input_dir,
        "filename_prefix": filename_prefix,
        "input_files": input_files,
        "missing_indices": missing_indices,
        "duration_hours": duration_hours,
        "lomb_period": lomb_period,
        "cosinor_period": cosinor_period,
        "selected_frp": selected_frp,
        "cycle_rows": cycle_rows,
        "summary_row": summary_row,
        "ct0_boundaries": ct0_boundaries,
        "warnings": list(dict.fromkeys(warnings)),
        "runtime_seconds": total_runtime,
        "output_dir": output_dir,
    }


def main() -> None:
    args = parse_args()
    result = run_analysis(args.input_dir.expanduser().resolve(), args.start_ct)
    summary = result["summary_row"]
    print(f"Input directory: {result['input_dir']}")
    print(
        f"Files: {summary['number_of_files']} "
        f"(indices {summary['first_file_index']} through {summary['last_file_index']}); "
        f"missing within span: {summary['missing_file_count']}"
    )
    print(f"Recording duration: {summary['total_elapsed_recording_duration_hours']:.3f} h")
    print(f"Lomb--Scargle FRP: {summary['lomb_scargle_frp_hours']:.2f} h")
    print(f"Cosinor FRP: {summary['cosinor_frp_hours']:.2f} h")
    print(f"LS minus cosinor: {summary['ls_minus_cosinor_hours']:.2f} h")
    print(f"Selected FRP: {summary['selected_frp_hours']:.2f} h")
    print(f"Start CT: {summary['start_ct_used']:.1f}")
    print(f"Full CT0--CT24 cycles: {summary['number_of_full_ct0_ct24_cycles']}")
    print(
        "CT0 boundaries (elapsed h): "
        + (", ".join(f"{value:.3f}" for value in result["ct0_boundaries"]) or "none")
    )
    print("Warnings: " + (summary["notes_warnings"] or "None"))
    print(f"Output directory: {result['output_dir']}")
    print(f"Runtime: {result['runtime_seconds']:.3f} seconds")


if __name__ == "__main__":
    main()
