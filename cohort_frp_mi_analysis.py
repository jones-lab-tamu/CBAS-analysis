"""Run the validated single-animal FRP and phase-by-behavior MI analyses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import frp_phase_analysis as frp


SCRIPT_DIR = Path(__file__).resolve().parent
MI_SCRIPT = SCRIPT_DIR / "phase_behavior_mutual_information.py"
OUTPUT_DIRECTORY_NAME = "FRP_MI_Cohort_Output"

# These are the known output directories from the existing analyses and this
# wrapper.  They are not candidate animal-input directories.
EXCLUDED_DIRECTORY_NAMES = {
    "FRP_Phase_Output",
    "MI_Output",
    "COMBA_Output",
    "FRP_MI_Cohort_Output",
}

METRICS_COLUMNS = [
    "Group",
    "Animal",
    "FRP_LS_h",
    "FRP_cosinor_h",
    "MI9_excess_bits",
    "NMI9_excess",
    "MI8_conditional_excess_bits",
    "NMI8_conditional_excess",
    "MI_rest_nonrest_excess_bits",
    "MI8_weighted_excess_bits",
    "P_rest",
    "P_nonrest",
]

QC_COLUMNS = [
    "Group",
    "Animal",
    "Input_path",
    "Source_file_count",
    "Missing_source_indices",
    "Recording_duration_h",
    "Reported_FRP_h",
    "Computational_FRP_h",
    "Complete_cycles_used",
    "Number_complete_cycles",
    "Complete_cycle_sample_count",
    "Partial_cycle_samples_excluded",
    "Null_sample_count_check",
    "Decomposition_check",
    "Phase_origin_sensitivity_min",
    "Phase_origin_sensitivity_max",
    "FRP_status",
    "MI_status",
    "Overall_status",
    "Error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fresh FRP and phase-by-behavior MI analyses for a cohort."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="One cohort root or one or more explicit animal folders.",
    )
    return parser.parse_args()


def _matching_source_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and frp.FILE_PATTERN.fullmatch(path.name) is not None
    )


def discover_animal_directories(root: Path) -> list[Path]:
    """Recursively find directories containing the validated CBAS filenames."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {root}")
    if _matching_source_files(root):
        return [root]

    discovered: set[Path] = set()
    for path in root.rglob("*.csv"):
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts):
            continue
        if path.is_file() and frp.FILE_PATTERN.fullmatch(path.name) is not None:
            discovered.add(path.parent.resolve())
    return sorted(discovered, key=lambda path: str(path).lower())


def resolve_animal_directories(input_paths: list[Path]) -> tuple[list[Path], bool]:
    """Return animal folders and whether the input was explicit-folder mode."""

    resolved = [path.expanduser().resolve() for path in input_paths]
    if len(resolved) == 1 and not _matching_source_files(resolved[0]):
        return discover_animal_directories(resolved[0]), False

    animals: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            animals.append(path)
            seen.add(path)
    return animals, True


def output_base_directory(input_paths: list[Path]) -> Path:
    resolved = [path.expanduser().resolve() for path in input_paths]
    if len(resolved) == 1:
        return resolved[0]
    return Path(os.path.commonpath([str(path) for path in resolved]))


def group_for_animal(
    animal_dir: Path,
    supplied_root: Path,
    explicit_folder_mode: bool,
) -> str:
    if explicit_folder_mode:
        return ""
    relative = animal_dir.relative_to(supplied_root)
    if len(relative.parts) < 2:
        return ""
    return relative.parts[-2]


def _read_one_row(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"Expected output file is missing: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one row in {path}; found {len(frame)}")
    return frame.iloc[0]


def _require_columns(row: pd.Series, columns: set[str], path: Path) -> None:
    missing = columns.difference(row.index)
    if missing:
        raise ValueError(f"Output file {path} is missing columns: {sorted(missing)}")


def _finite_float(row: pd.Series, column: str, path: Path) -> float:
    value = float(row[column])
    if not np.isfinite(value):
        raise ValueError(f"Output file {path} has a non-finite {column}")
    return value


def _integer_value(row: pd.Series, column: str, path: Path) -> int:
    value = _finite_float(row, column, path)
    integer = int(value)
    if value != integer:
        raise ValueError(f"Output file {path} has a non-integer {column}: {value}")
    return integer


def _read_mi_run_summary(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected output file is missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return lines, values


def _run_mi_subprocess(animal_dir: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(MI_SCRIPT), str(animal_dir)],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    details = completed.stderr.strip() or completed.stdout.strip()
    if not details:
        details = "no subprocess output"
    raise RuntimeError(
        f"MI subprocess exited with code {completed.returncode}: {details[-3000:]}"
    )


def read_animal_outputs(
    animal_dir: Path,
    group: str,
    frp_result: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Read the fresh FRP and MI files and build metric/QC rows."""

    frp_dir = animal_dir / "FRP_Phase_Output"
    mi_dir = animal_dir / "MI_Output"
    frp_summary_path = frp_dir / "frp_phase_summary.csv"
    frp_summary = _read_one_row(frp_summary_path)
    _require_columns(
        frp_summary,
        {
            "lomb_scargle_frp_hours",
            "cosinor_frp_hours",
            "number_of_files",
            "total_elapsed_recording_duration_hours",
        },
        frp_summary_path,
    )

    mi_summary_path = mi_dir / "animal_mi_summary.csv"
    mi_summary = _read_one_row(mi_summary_path)
    _require_columns(
        mi_summary,
        {
            "FRP_hours",
            "computational_FRP_hours",
            "complete_cycle_indices",
            "n_complete_cycles_used",
            "MI9_excess_bits",
            "NMI9_excess",
            "MI_conditional8_excess_bits",
            "MI_rest_nonrest_excess_bits",
            "MI_conditional8_weighted_excess_bits",
            "P_rest",
            "P_nonrest",
        },
        mi_summary_path,
    )

    mi_results_path = mi_dir / "mi_results.csv"
    if not mi_results_path.is_file():
        raise FileNotFoundError(f"Expected output file is missing: {mi_results_path}")
    mi_results = pd.read_csv(mi_results_path)
    if "component" not in mi_results.columns:
        raise ValueError(f"MI results are missing the component column: {mi_results_path}")
    full_rows = mi_results.loc[mi_results["component"] == "full_9state"]
    conditional_rows = mi_results.loc[
        mi_results["component"] == "conditional_8state_nonrest"
    ]
    if len(full_rows) != 1 or len(conditional_rows) != 1:
        raise ValueError("MI results do not contain exactly one full and conditional row")
    full_row = full_rows.iloc[0]
    conditional_row = conditional_rows.iloc[0]
    _require_columns(
        conditional_row,
        {"NMI_excess"},
        mi_results_path,
    )
    complete_sample_count = _integer_value(full_row, "n_samples", mi_results_path)

    sensitivity_path = mi_dir / "mi_sensitivity.csv"
    if not sensitivity_path.is_file():
        raise FileNotFoundError(f"Expected output file is missing: {sensitivity_path}")
    sensitivity = pd.read_csv(sensitivity_path)
    required_sensitivity_columns = {"sensitivity_type", "MI_excess_bits"}
    if not required_sensitivity_columns.issubset(sensitivity.columns):
        raise ValueError(f"MI sensitivity output is missing required columns: {sensitivity_path}")
    origin_values = sensitivity.loc[
        sensitivity["sensitivity_type"] == "phase_origin_minutes",
        "MI_excess_bits",
    ].astype(float)
    if origin_values.empty or not np.all(np.isfinite(origin_values.to_numpy())):
        raise ValueError("MI phase-origin sensitivity values are missing or non-finite")

    run_summary_path = mi_dir / "run_summary.txt"
    run_lines, run_values = _read_mi_run_summary(run_summary_path)
    null_check = "PASS" if any(
        line.startswith("PASS: observed and null complete-cycle sample counts match")
        for line in run_lines
    ) else "FAIL"
    decomposition_lines = [
        line for line in run_lines if "decomposition error" in line
    ]
    decomposition_check = (
        "PASS"
        if len(decomposition_lines) >= 3
        and all(line.startswith("PASS:") for line in decomposition_lines)
        else "FAIL"
    )
    if null_check != "PASS":
        raise ValueError("MI null complete-cycle sample-count validation did not pass")
    if decomposition_check != "PASS":
        raise ValueError("MI decomposition validation did not pass")
    partial_samples = int(run_values["partial_cycle_samples_excluded_from_corrected_MI"])

    full_cycle_count = sum(
        bool(row["full_cycle_flag"]) for row in frp_result["cycle_rows"]  # type: ignore[index]
    )
    mi_cycle_count = _integer_value(mi_summary, "n_complete_cycles_used", mi_summary_path)
    if full_cycle_count != mi_cycle_count:
        raise ValueError(
            f"FRP and MI complete-cycle counts disagree: {full_cycle_count} vs {mi_cycle_count}"
        )

    metrics = {
        "Group": group,
        "Animal": animal_dir.name,
        "FRP_LS_h": _finite_float(
            frp_summary, "lomb_scargle_frp_hours", frp_summary_path
        ),
        "FRP_cosinor_h": _finite_float(
            frp_summary, "cosinor_frp_hours", frp_summary_path
        ),
        "MI9_excess_bits": _finite_float(
            mi_summary, "MI9_excess_bits", mi_summary_path
        ),
        "NMI9_excess": _finite_float(mi_summary, "NMI9_excess", mi_summary_path),
        "MI8_conditional_excess_bits": _finite_float(
            mi_summary, "MI_conditional8_excess_bits", mi_summary_path
        ),
        "NMI8_conditional_excess": _finite_float(
            conditional_row, "NMI_excess", mi_results_path
        ),
        "MI_rest_nonrest_excess_bits": _finite_float(
            mi_summary, "MI_rest_nonrest_excess_bits", mi_summary_path
        ),
        "MI8_weighted_excess_bits": _finite_float(
            mi_summary, "MI_conditional8_weighted_excess_bits", mi_summary_path
        ),
        "P_rest": _finite_float(mi_summary, "P_rest", mi_summary_path),
        "P_nonrest": _finite_float(mi_summary, "P_nonrest", mi_summary_path),
    }
    qc = {
        "Source_file_count": _integer_value(
            frp_summary, "number_of_files", frp_summary_path
        ),
        "Missing_source_indices": (
            ",".join(str(value) for value in frp_result["missing_indices"])  # type: ignore[index]
            or "none"
        ),
        "Recording_duration_h": _finite_float(
            frp_summary,
            "total_elapsed_recording_duration_hours",
            frp_summary_path,
        ),
        "Reported_FRP_h": _finite_float(mi_summary, "FRP_hours", mi_summary_path),
        "Computational_FRP_h": _finite_float(
            mi_summary, "computational_FRP_hours", mi_summary_path
        ),
        "Complete_cycles_used": str(mi_summary["complete_cycle_indices"]),
        "Number_complete_cycles": mi_cycle_count,
        "Complete_cycle_sample_count": complete_sample_count,
        "Partial_cycle_samples_excluded": partial_samples,
        "Null_sample_count_check": null_check,
        "Decomposition_check": decomposition_check,
        "Phase_origin_sensitivity_min": float(origin_values.min()),
        "Phase_origin_sensitivity_max": float(origin_values.max()),
    }
    return metrics, qc


def _error_message(stage: str, error: Exception) -> str:
    detail = str(error).strip() or repr(error)
    return f"{stage}: {type(error).__name__}: {detail}"


def process_animal(
    animal_dir: Path,
    group: str,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    qc = {column: "" for column in QC_COLUMNS}
    qc.update(
        {
            "Group": group,
            "Animal": animal_dir.name,
            "Input_path": str(animal_dir),
            "FRP_status": "FAIL",
            "MI_status": "FAIL",
            "Overall_status": "FAIL",
        }
    )

    try:
        frp_result = frp.run_analysis(animal_dir, frp.DEFAULT_START_CT)
        qc["FRP_status"] = "PASS"
    except Exception as error:  # keep one-animal failure from stopping the cohort
        qc["Error_message"] = _error_message("FRP", error)
        return None, qc

    try:
        _run_mi_subprocess(animal_dir)
        metrics, output_qc = read_animal_outputs(animal_dir, group, frp_result)
        qc.update(output_qc)
        qc["MI_status"] = "PASS"
        qc["Overall_status"] = "PASS"
        return metrics, qc
    except Exception as error:  # keep one-animal failure from stopping the cohort
        qc["Error_message"] = _error_message("MI/aggregation", error)
        return None, qc


def write_cohort_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, object]],
    qc_rows: list[dict[str, object]],
    supplied_paths: list[Path],
    animals: list[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows, columns=METRICS_COLUMNS).to_csv(
        output_dir / "cohort_frp_mi_metrics.csv",
        index=False,
    )
    pd.DataFrame(qc_rows, columns=QC_COLUMNS).to_csv(
        output_dir / "cohort_frp_mi_qc.csv",
        index=False,
    )

    successful = [
        str(row["Animal"])
        for row in qc_rows
        if row["Overall_status"] == "PASS"
    ]
    failed = [
        f"{row['Animal']}: {row['Error_message']}"
        for row in qc_rows
        if row["Overall_status"] != "PASS"
    ]
    lines = [
        "Cohort FRP + phase-by-behavior MI run summary",
        f"supplied_root_or_folders: {'; '.join(str(path) for path in supplied_paths)}",
        f"number_of_animals_discovered: {len(animals)}",
        f"number_completed_successfully: {len(successful)}",
        f"number_failed: {len(failed)}",
        "successful_animals: " + (", ".join(successful) or "none"),
        "failed_animals: " + ("; ".join(failed) or "none"),
        f"scientific_metrics_csv: {output_dir / 'cohort_frp_mi_metrics.csv'}",
        f"qc_csv: {output_dir / 'cohort_frp_mi_qc.csv'}",
        "FRP_and_MI_rerun_fresh: yes",
        "FRP_source_code_modified: no",
        "MI_source_code_modified: no",
    ]
    (output_dir / "cohort_run_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    animals, explicit_folder_mode = resolve_animal_directories(args.input_paths)
    supplied_paths = [path.expanduser().resolve() for path in args.input_paths]
    if len({path for path in animals}) != len(animals):
        raise ValueError("Duplicate animal folders were discovered")

    supplied_root = supplied_paths[0]
    output_dir = output_base_directory(supplied_paths) / OUTPUT_DIRECTORY_NAME
    metrics_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    print(f"Discovered animals: {len(animals)}", flush=True)
    for index, animal_dir in enumerate(animals, start=1):
        group = group_for_animal(animal_dir, supplied_root, explicit_folder_mode)
        print(
            f"[{index}/{len(animals)}] running {group + '/' if group else ''}{animal_dir.name}",
            flush=True,
        )
        metrics, qc = process_animal(animal_dir, group)
        qc_rows.append(qc)
        if metrics is not None:
            metrics_rows.append(metrics)
            print(f"[{index}/{len(animals)}] PASS {animal_dir.name}", flush=True)
        else:
            print(
                f"[{index}/{len(animals)}] FAIL {animal_dir.name}: {qc['Error_message']}",
                flush=True,
            )

    write_cohort_outputs(
        output_dir,
        metrics_rows,
        qc_rows,
        supplied_paths,
        animals,
    )
    print(f"Completed successfully: {len(metrics_rows)}/{len(animals)}", flush=True)
    print(f"Outputs written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
