"""Generate the generalized composite circular-W1 A-C cohort figure.

The script reads frozen circular-W1 outputs and does not rerun or modify any
upstream analysis.  Cohort metadata may be supplied as a CSV with at least
``animal`` and ``group`` columns.  When no metadata path is supplied, the
group columns already present in ``pairwise_repertoire_distance.csv`` are
used as the current-data fallback.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEHAVIOR_ORDER = (
    "eating",
    "drinking",
    "rearing",
    "climbing",
    "digging",
    "nesting",
    "grooming",
    "locomotion",
)
BEHAVIOR_DISPLAY_NAMES = {
    "eating": "Eating",
    "drinking": "Drinking",
    "rearing": "Rearing",
    "climbing": "Climbing",
    "digging": "Digging",
    "nesting": "Nesting",
    "grooming": "Grooming",
    "locomotion": "Exploring",
}
BEHAVIOR_COLORS = {
    "eating": "#0072B2",
    "drinking": "#D55E00",
    "rearing": "#009E73",
    "climbing": "#CC79A7",
    "digging": "#E69F00",
    "nesting": "#56B4E9",
    "grooming": "#6A3D9A",
    "locomotion": "#4D4D4D",
}
DEFAULT_GROUP_DISPLAY_NAMES = {"LacZ": "LacZ", "Bmal1KO": "Bmal1 KO"}
DEFAULT_GROUP_COLORS = {"LacZ": "#2f6f9f", "Bmal1KO": "#c55a11"}
FALLBACK_GROUP_COLORS = (
    "#2f6f9f",
    "#c55a11",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#6A3D9A",
)

N_PHASE_BINS = 288
N_DISPLAY_BINS = 48
BIN_WIDTH_HOURS = 1.0 / 12.0
DISPLAY_BIN_WIDTH_HOURS = 0.5
HOURS_PER_CYCLE = 24.0
LANE_GAP_FRACTION = 0.18
PROBABILITY_TOLERANCE = 1e-12
W1_TOLERANCE = 1e-10

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Cohort_Figure_Visualization"
SOURCE_FILENAMES = (
    "behavior_phase_distributions_5min.csv",
    "pairwise_behavior_circular_w1.csv",
    "pairwise_repertoire_distance.csv",
)


@dataclass(frozen=True)
class CohortMetadata:
    animals: tuple[str, ...]
    group_by_animal: dict[str, str]
    group_order: tuple[str, ...]
    group_display_names: dict[str, str]
    group_colors: dict[str, str]
    control_group: str
    experimental_group: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the generalized composite circular-W1 A-C figure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the frozen circular-W1 CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the integrated figure and validation tables.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional metadata CSV with animal and group columns. "
            "Without it, groups are inferred from pairwise_repertoire_distance.csv."
        ),
    )
    parser.add_argument(
        "--control-group",
        default=None,
        help="Raw metadata group label used as the control/reference group.",
    )
    parser.add_argument(
        "--experimental-group",
        default=None,
        help="Raw metadata group label used as the experimental group.",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _canonical_pair(animal_i: str, animal_j: str) -> tuple[str, str]:
    if animal_i == animal_j:
        raise ValueError(f"Self-pair is not allowed: {animal_i}")
    return (animal_i, animal_j) if animal_i < animal_j else (animal_j, animal_i)


def load_sources(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [input_dir / filename for filename in SOURCE_FILENAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required frozen circular-W1 output(s) are missing: "
            + ", ".join(str(path) for path in missing)
        )
    phase, pairwise, composite = (pd.read_csv(path) for path in paths)
    return phase, pairwise, composite


def _metadata_from_composite(composite: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        composite,
        {"animal_i", "genotype_i", "animal_j", "genotype_j"},
        SOURCE_FILENAMES[2],
    )
    rows: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for row in composite.itertuples(index=False):
        for animal_column, group_column in (
            ("animal_i", "genotype_i"),
            ("animal_j", "genotype_j"),
        ):
            animal_value = getattr(row, animal_column)
            group_value = getattr(row, group_column)
            if pd.isna(animal_value) or pd.isna(group_value):
                raise ValueError(f"{SOURCE_FILENAMES[2]} contains missing group metadata.")
            animal = str(animal_value)
            group = str(group_value)
            if animal in seen and seen[animal] != group:
                raise ValueError(
                    f"Animal {animal} has conflicting group metadata: "
                    f"{seen[animal]!r} and {group!r}."
                )
            if animal not in seen:
                seen[animal] = group
                rows.append({"animal": animal, "group": group})
    return pd.DataFrame(rows)


def _ordered_unique(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(values.astype(str).tolist()))


def load_metadata(
    metadata_path: Path | None,
    composite: pd.DataFrame,
    *,
    control_group: str | None,
    experimental_group: str | None,
) -> CohortMetadata:
    if metadata_path is None:
        metadata = _metadata_from_composite(composite)
        source = f"inferred from {SOURCE_FILENAMES[2]}"
    else:
        metadata = pd.read_csv(metadata_path)
        _require_columns(metadata, {"animal", "group"}, metadata_path.name)
        source = str(metadata_path)

    metadata = metadata.copy()
    metadata["animal"] = metadata["animal"].astype(str)
    metadata["group"] = metadata["group"].astype(str)
    if metadata["animal"].duplicated().any():
        raise ValueError("Metadata must contain one row per animal.")
    if metadata["animal"].eq("").any() or metadata["group"].eq("").any():
        raise ValueError("Metadata animal and group values must be non-empty.")

    if "display_order" in metadata.columns:
        display_order = pd.to_numeric(metadata["display_order"], errors="coerce")
        if display_order.isna().any():
            raise ValueError("Metadata display_order must be numeric when supplied.")
        metadata["_display_order"] = display_order
    else:
        metadata["_display_order"] = np.arange(len(metadata), dtype=float)

    first_group_order = _ordered_unique(metadata["group"])
    if "group_order" in metadata.columns:
        group_order_values = pd.to_numeric(metadata["group_order"], errors="coerce")
        if group_order_values.isna().any():
            raise ValueError("Metadata group_order must be numeric when supplied.")
        metadata["_group_order"] = group_order_values
        group_order = tuple(
            metadata.groupby("group", sort=False)["_group_order"]
            .min()
            .sort_values(kind="stable")
            .index.astype(str)
        )
    else:
        group_order = tuple(first_group_order)

    if len(group_order) != 2:
        raise ValueError(
            "The descriptive control-versus-experimental figure requires exactly two groups."
        )

    group_by_animal = dict(zip(metadata["animal"], metadata["group"]))
    ordered_animals = tuple(
        animal
        for group in group_order
        for animal in metadata.loc[
            metadata["group"] == group
        ].sort_values("_display_order", kind="stable")["animal"]
    )

    group_display_names: dict[str, str] = {}
    if "group_display_name" in metadata.columns:
        for group, rows in metadata.groupby("group", sort=False):
            values = [str(value) for value in rows["group_display_name"] if not pd.isna(value)]
            if not values or len(set(values)) != 1:
                raise ValueError(
                    "Each group must have one non-empty group_display_name when supplied."
                )
            group_display_names[str(group)] = values[0]
    for group in group_order:
        group_display_names.setdefault(
            group, DEFAULT_GROUP_DISPLAY_NAMES.get(group, group)
        )

    group_colors: dict[str, str] = {}
    if "group_color" in metadata.columns:
        for group, rows in metadata.groupby("group", sort=False):
            values = [str(value) for value in rows["group_color"] if not pd.isna(value)]
            if not values or len(set(values)) != 1:
                raise ValueError(
                    "Each group must have one non-empty group_color when supplied."
                )
            group_colors[str(group)] = values[0]
    fallback_index = 0
    for group in group_order:
        if group not in group_colors:
            group_colors[group] = DEFAULT_GROUP_COLORS.get(
                group,
                FALLBACK_GROUP_COLORS[fallback_index % len(FALLBACK_GROUP_COLORS)],
            )
            fallback_index += 1

    control = control_group or group_order[0]
    experimental = experimental_group or next(
        group for group in group_order if group != control
    )
    if control not in group_order or experimental not in group_order:
        raise ValueError("Control and experimental groups must be present in metadata.")
    if control == experimental:
        raise ValueError("Control and experimental groups must be different.")

    return CohortMetadata(
        animals=ordered_animals,
        group_by_animal=group_by_animal,
        group_order=group_order,
        group_display_names=group_display_names,
        group_colors=group_colors,
        control_group=control,
        experimental_group=experimental,
        source=source,
    )


def validate_and_build_matrix(
    phase: pd.DataFrame,
    pairwise: pd.DataFrame,
    composite: pd.DataFrame,
    metadata: CohortMetadata,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate frozen inputs and build a symmetric matrix from source pairs."""

    _require_columns(
        phase,
        {
            "animal",
            "behavior",
            "ct_bin_start_hours",
            "ct_bin_center_hours",
            "probability",
        },
        SOURCE_FILENAMES[0],
    )
    _require_columns(
        pairwise,
        {"animal_i", "animal_j", "behavior", "w1_hours"},
        SOURCE_FILENAMES[1],
    )
    _require_columns(
        composite,
        {
            "animal_i",
            "animal_j",
            "repertoire_w1_mean_hours",
            *(f"w1_{behavior}" for behavior in BEHAVIOR_ORDER),
        },
        SOURCE_FILENAMES[2],
    )

    phase = phase.copy()
    pairwise = pairwise.copy()
    composite = composite.copy()
    for frame in (phase, pairwise, composite):
        for column in ("animal", "animal_i", "animal_j", "behavior"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    expected_animals = set(metadata.animals)
    phase_animals = set(phase["animal"])
    if phase_animals != expected_animals:
        raise ValueError(
            "Phase distributions do not contain exactly the metadata animals: "
            f"missing={sorted(expected_animals - phase_animals)}, "
            f"extra={sorted(phase_animals - expected_animals)}"
        )
    if set(phase["behavior"]) != set(BEHAVIOR_ORDER):
        raise ValueError("Phase distributions do not contain the fixed eight behaviors.")

    phase_groups = phase.groupby(["animal", "behavior"], sort=False).size()
    if len(phase_groups) != len(metadata.animals) * len(BEHAVIOR_ORDER):
        raise ValueError("Phase distributions do not contain every animal-behavior group.")
    if not (phase_groups == N_PHASE_BINS).all():
        raise ValueError("Every animal-behavior phase distribution must contain 288 bins.")

    numeric_phase = phase[
        ["ct_bin_start_hours", "ct_bin_center_hours", "probability"]
    ]
    if numeric_phase.isna().any().any() or not np.isfinite(numeric_phase.to_numpy(float)).all():
        raise ValueError("Phase distributions contain missing or non-finite numeric values.")
    if (phase["probability"] < 0).any():
        raise ValueError("Phase distributions contain negative probabilities.")

    expected_starts = np.arange(N_PHASE_BINS, dtype=float) * BIN_WIDTH_HOURS
    expected_centers = expected_starts + BIN_WIDTH_HOURS / 2.0
    max_probability_sum_error = 0.0
    max_phase_grid_error = 0.0
    for key, group in phase.groupby(["animal", "behavior"], sort=False):
        ordered = group.sort_values("ct_bin_start_hours")
        sum_error = abs(float(ordered["probability"].sum()) - 1.0)
        start_error = float(
            np.max(
                np.abs(ordered["ct_bin_start_hours"].to_numpy(float) - expected_starts)
            )
        )
        center_error = float(
            np.max(
                np.abs(ordered["ct_bin_center_hours"].to_numpy(float) - expected_centers)
            )
        )
        max_probability_sum_error = max(max_probability_sum_error, sum_error)
        max_phase_grid_error = max(max_phase_grid_error, start_error, center_error)
        if sum_error > PROBABILITY_TOLERANCE:
            raise ValueError(f"Phase probabilities for {key} do not sum to one.")
    if max_phase_grid_error > PROBABILITY_TOLERANCE:
        raise ValueError(f"Phase-bin grid error exceeds tolerance: {max_phase_grid_error}")

    expected_pair_count = len(metadata.animals) * (len(metadata.animals) - 1) // 2
    expected_pairs = {
        _canonical_pair(animal_i, animal_j)
        for animal_i, animal_j in itertools.combinations(metadata.animals, 2)
    }
    pairwise_lookup: dict[tuple[str, str, str], float] = {}
    for row in pairwise.itertuples(index=False):
        animal_i = str(row.animal_i)
        animal_j = str(row.animal_j)
        behavior = str(row.behavior)
        pair_key = _canonical_pair(animal_i, animal_j)
        if pair_key not in expected_pairs:
            raise ValueError(f"Pairwise W1 contains an animal outside metadata: {pair_key}")
        if behavior not in BEHAVIOR_ORDER:
            raise ValueError(f"Pairwise W1 contains an unexpected behavior: {behavior}")
        key = (*pair_key, behavior)
        if key in pairwise_lookup:
            raise ValueError(f"Duplicate pairwise behavior row: {key}")
        value = float(row.w1_hours)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid pairwise W1 value for {key}: {value}")
        pairwise_lookup[key] = value
    if len(pairwise_lookup) != expected_pair_count * len(BEHAVIOR_ORDER):
        raise ValueError("Pairwise W1 output is incomplete or contains duplicate rows.")

    composite_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in composite.itertuples(index=False):
        pair_key = _canonical_pair(str(row.animal_i), str(row.animal_j))
        if pair_key not in expected_pairs:
            raise ValueError(f"Composite W1 contains an animal outside metadata: {pair_key}")
        if pair_key in composite_lookup:
            raise ValueError(f"Duplicate composite pair row: {pair_key}")
        values = {
            behavior: float(getattr(row, f"w1_{behavior}"))
            for behavior in BEHAVIOR_ORDER
        }
        values["repertoire_w1_mean_hours"] = float(row.repertoire_w1_mean_hours)
        if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
            raise ValueError(f"Composite W1 contains non-finite values for {pair_key}")
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Composite W1 contains negative values for {pair_key}")
        composite_lookup[pair_key] = values
    if set(composite_lookup) != expected_pairs:
        raise ValueError("Composite W1 output does not contain exactly the metadata pairs.")

    max_cross_file_w1_error = 0.0
    max_composite_mean_error = 0.0
    for pair_key, values in composite_lookup.items():
        components = []
        for behavior in BEHAVIOR_ORDER:
            long_value = pairwise_lookup[(*pair_key, behavior)]
            max_cross_file_w1_error = max(
                max_cross_file_w1_error, abs(values[behavior] - long_value)
            )
            components.append(long_value)
        mean_error = abs(values["repertoire_w1_mean_hours"] - float(np.mean(components)))
        max_composite_mean_error = max(max_composite_mean_error, mean_error)
    if max_cross_file_w1_error > W1_TOLERANCE:
        raise ValueError("Wide and long behavior-level W1 outputs disagree.")
    if max_composite_mean_error > W1_TOLERANCE:
        raise ValueError("Composite W1 is not the mean of its eight behavior values.")

    matrix_values = np.zeros((len(metadata.animals), len(metadata.animals)), dtype=float)
    animal_indices = {animal: index for index, animal in enumerate(metadata.animals)}
    for (animal_i, animal_j), values in composite_lookup.items():
        i = animal_indices[animal_i]
        j = animal_indices[animal_j]
        value = values["repertoire_w1_mean_hours"]
        matrix_values[i, j] = value
        matrix_values[j, i] = value
    matrix = pd.DataFrame(
        matrix_values,
        index=list(metadata.animals),
        columns=list(metadata.animals),
    )
    if not np.allclose(matrix_values, matrix_values.T, atol=W1_TOLERANCE):
        raise ValueError("Composite distance matrix is not symmetric.")
    if not np.allclose(np.diag(matrix_values), 0.0, atol=W1_TOLERANCE):
        raise ValueError("Composite distance matrix diagonal is not zero.")
    validation = {
        "phase_rows": int(len(phase)),
        "phase_groups": int(len(phase_groups)),
        "pairwise_rows": int(len(pairwise)),
        "composite_rows": int(len(composite)),
        "unique_pair_count": expected_pair_count,
        "max_probability_sum_error": max_probability_sum_error,
        "max_phase_grid_error": max_phase_grid_error,
        "max_cross_file_w1_error": max_cross_file_w1_error,
        "max_composite_mean_error": max_composite_mean_error,
        "phase": phase,
        "pairwise_lookup": pairwise_lookup,
        "composite_lookup": composite_lookup,
    }
    return matrix, validation


def aggregate_to_30_minutes(
    phase: pd.DataFrame, animals: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    max_aggregation_error = 0.0
    max_display_sum_error = 0.0
    for animal in animals:
        for behavior in BEHAVIOR_ORDER:
            group = phase[
                (phase["animal"] == animal) & (phase["behavior"] == behavior)
            ].sort_values("ct_bin_start_hours")
            probability = group["probability"].to_numpy(float)
            for display_bin in range(N_DISPLAY_BINS):
                start = display_bin * 6
                end = start + 6
                start_hours = display_bin * DISPLAY_BIN_WIDTH_HOURS
                mass = float(probability[start:end].sum())
                rows.append(
                    {
                        "animal": animal,
                        "behavior": behavior,
                        "display_bin_index": display_bin,
                        "ct_bin_start": start_hours,
                        "ct_bin_end": start_hours + DISPLAY_BIN_WIDTH_HOURS,
                        "ct_bin_center": start_hours + DISPLAY_BIN_WIDTH_HOURS / 2.0,
                        "probability_mass_30min": mass,
                        "relative_density_vs_uniform": mass * N_DISPLAY_BINS,
                    }
                )
            display_sum = float(sum(row["probability_mass_30min"] for row in rows[-N_DISPLAY_BINS:]))
            max_aggregation_error = max(
                max_aggregation_error, abs(display_sum - float(probability.sum()))
            )
            max_display_sum_error = max(max_display_sum_error, abs(display_sum - 1.0))

    display = pd.DataFrame(rows)
    if not np.isfinite(
        display[["probability_mass_30min", "relative_density_vs_uniform"]].to_numpy(float)
    ).all():
        raise ValueError("30-minute display data contain non-finite values.")
    if max_display_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError("30-minute aggregation does not preserve probability mass.")
    relative_density = display["relative_density_vs_uniform"].to_numpy(float)
    return display, {
        "display_rows": int(len(display)),
        "display_groups": int(display.groupby(["animal", "behavior"], sort=False).ngroups),
        "max_aggregation_error": max_aggregation_error,
        "max_display_sum_error": max_display_sum_error,
        "relative_density_min": float(relative_density.min()),
        "relative_density_max": float(relative_density.max()),
    }


def compute_distance_to_control_reference(
    matrix: pd.DataFrame, metadata: CohortMetadata
) -> pd.DataFrame:
    control_animals = [
        animal
        for animal in metadata.animals
        if metadata.group_by_animal[animal] == metadata.control_group
    ]
    if len(control_animals) < 2:
        raise ValueError("At least two control animals are required for the reference definition.")

    rows: list[dict[str, object]] = []
    for animal in metadata.animals:
        group = metadata.group_by_animal[animal]
        if group == metadata.control_group:
            reference_animals = [other for other in control_animals if other != animal]
        elif group == metadata.experimental_group:
            reference_animals = control_animals
        else:
            raise ValueError(f"Animal {animal} is not in the control or experimental group.")
        values = matrix.loc[animal, reference_animals].to_numpy(dtype=float)
        rows.append(
            {
                "animal": animal,
                "group": group,
                "group_display_name": metadata.group_display_names[group],
                "distance_to_control_reference_hours": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def rank_pairwise_distances(
    matrix: pd.DataFrame, animals: tuple[str, ...]
) -> list[dict[str, object]]:
    records = []
    for animal_i, animal_j in itertools.combinations(sorted(animals), 2):
        records.append(
            {
                "pair_id": f"{animal_i}__{animal_j}",
                "animal_i": animal_i,
                "animal_j": animal_j,
                "composite_w1_hours": float(matrix.loc[animal_i, animal_j]),
            }
        )
    records.sort(key=lambda row: (row["composite_w1_hours"], row["animal_i"], row["animal_j"]))
    denominator = max(1, len(records) - 1)
    for index, record in enumerate(records):
        record["global_pair_rank"] = index + 1
        record["global_pair_percentile"] = index / denominator
    return records


def select_representative_pairs(
    records: list[dict[str, object]], animals: tuple[str, ...]
) -> tuple[list[dict[str, object]], float]:
    if len(animals) < 6:
        raise ValueError(
            "Panel C requires at least six animals to select three non-overlapping representative pairs."
        )
    if not records:
        raise ValueError("No pairwise distances are available for Panel C selection.")

    animal_bits = {animal: 1 << index for index, animal in enumerate(sorted(animals))}
    masks = [animal_bits[row["animal_i"]] | animal_bits[row["animal_j"]] for row in records]
    targets = (0.0, 0.5, 1.0)
    best_key: tuple[object, ...] | None = None
    best_rows: list[dict[str, object]] | None = None
    best_score = float("inf")

    for first in range(len(records) - 2):
        for second in range(first + 1, len(records) - 1):
            if masks[first] & masks[second]:
                continue
            used = masks[first] | masks[second]
            for third in range(second + 1, len(records)):
                if used & masks[third]:
                    continue
                ordered = sorted(
                    (records[first], records[second], records[third]),
                    key=lambda row: (
                        row["global_pair_percentile"],
                        row["animal_i"],
                        row["animal_j"],
                    ),
                )
                score = float(
                    sum(
                        abs(float(row["global_pair_percentile"]) - target)
                        for row, target in zip(ordered, targets)
                    )
                )
                tie_key = (
                    score,
                    int(ordered[0]["global_pair_rank"]),
                    -int(ordered[2]["global_pair_rank"]),
                    tuple(row["pair_id"] for row in ordered),
                )
                if best_key is None or tie_key < best_key:
                    best_key = tie_key
                    best_score = score
                    best_rows = [dict(row) for row in ordered]

    if best_rows is None:
        raise ValueError("No valid three-disjoint-pair set exists for Panel C.")
    distance_classes = ("low", "intermediate", "high")
    for distance_class, row in zip(distance_classes, best_rows):
        row["distance_class"] = distance_class
        row["selection_score"] = best_score
    return best_rows, best_score


def selected_pair_behavior_table(
    selected_pairs: list[dict[str, object]], validation: dict[str, object]
) -> pd.DataFrame:
    pairwise_lookup = validation["pairwise_lookup"]
    composite_lookup = validation["composite_lookup"]
    assert isinstance(pairwise_lookup, dict)
    assert isinstance(composite_lookup, dict)
    rows: list[dict[str, object]] = []
    for pair in selected_pairs:
        pair_key = (str(pair["animal_i"]), str(pair["animal_j"]))
        composite_value = float(composite_lookup[pair_key]["repertoire_w1_mean_hours"])
        for behavior in BEHAVIOR_ORDER:
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "animal_i": pair["animal_i"],
                    "animal_j": pair["animal_j"],
                    "behavior": behavior,
                    "behavior_w1_hours": float(pairwise_lookup[(*pair_key, behavior)]),
                    "composite_repertoire_w1_hours": composite_value,
                }
            )
    return pd.DataFrame(rows)


def _profile_matrix(display: pd.DataFrame, animal: str) -> np.ndarray:
    matrix = np.zeros((len(BEHAVIOR_ORDER), N_DISPLAY_BINS), dtype=float)
    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        group = display[
            (display["animal"] == animal) & (display["behavior"] == behavior)
        ].sort_values("display_bin_index")
        if len(group) != N_DISPLAY_BINS:
            raise ValueError(f"Missing 30-minute profile for {animal}, {behavior}.")
        matrix[behavior_index] = group["relative_density_vs_uniform"].to_numpy(float)
    return matrix


def _lane_geometry(relative_density_max: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    lane_height = float(relative_density_max)
    if lane_height <= 0:
        raise ValueError("Common profile amplitude must be positive.")
    pitch = lane_height * (1.0 + LANE_GAP_FRACTION)
    baselines = np.array(
        [(len(BEHAVIOR_ORDER) - 1 - index) * pitch for index in range(len(BEHAVIOR_ORDER))],
        dtype=float,
    )
    return (
        lane_height,
        len(BEHAVIOR_ORDER) * pitch,
        baselines,
        baselines + lane_height / 2.0,
    )


def draw_profile_axis(
    ax: plt.Axes,
    display: pd.DataFrame,
    animal: str,
    relative_density_max: float,
    *,
    show_behavior_labels: bool,
    show_x_labels: bool,
    show_x_label: bool,
) -> tuple[float, np.ndarray]:
    matrix = _profile_matrix(display, animal)
    lane_height, y_max, baselines, row_centers = _lane_geometry(relative_density_max)
    centers = DISPLAY_BIN_WIDTH_HOURS / 2.0 + np.arange(N_DISPLAY_BINS) * DISPLAY_BIN_WIDTH_HOURS
    connected_x = np.r_[0.0, centers, HOURS_PER_CYCLE]

    for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
        baseline = baselines[behavior_index]
        values = matrix[behavior_index]
        connected_values = np.r_[values[0], values, values[-1]]
        connected_y = baseline + connected_values
        ax.axhline(baseline, color="#777777", linewidth=0.55, zorder=1)
        ax.axhline(
            baseline + 1.0,
            color="#999999",
            linestyle=(0, (2, 2)),
            linewidth=0.5,
            zorder=1,
        )
        ax.fill_between(
            connected_x,
            baseline,
            connected_y,
            color=BEHAVIOR_COLORS[behavior],
            alpha=0.58,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            connected_x,
            connected_y,
            color=BEHAVIOR_COLORS[behavior],
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xlim(0.0, HOURS_PER_CYCLE)
    ax.set_ylim(0.0, y_max)
    if show_behavior_labels:
        ax.set_yticks(
            row_centers,
            [BEHAVIOR_DISPLAY_NAMES[behavior] for behavior in BEHAVIOR_ORDER],
            fontsize=7.0,
        )
        for tick, behavior in zip(ax.get_yticklabels(), BEHAVIOR_ORDER):
            tick.set_color(BEHAVIOR_COLORS[behavior])
        ax.tick_params(axis="y", length=0, pad=1)
    else:
        ax.set_yticks(row_centers, [""] * len(row_centers))
        ax.tick_params(axis="y", length=0)
    if show_x_labels:
        ax.set_xticks((0.0, 24.0), ("0", "24"), fontsize=7.5)
    else:
        ax.set_xticks([])
    ax.set_xlabel("CT (h)" if show_x_label else "", fontsize=8.0)
    ax.set_title(animal, fontsize=8.5, pad=1.0)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    return y_max, row_centers


def draw_w1_axis(
    ax: plt.Axes,
    pair: dict[str, object],
    pair_behavior: pd.DataFrame,
    bar_limit: float,
    row_centers: np.ndarray,
    lane_height: float,
    *,
    show_x_label: bool,
) -> None:
    subset = pair_behavior[pair_behavior["pair_id"] == pair["pair_id"]].set_index("behavior")
    values = subset.loc[list(BEHAVIOR_ORDER), "behavior_w1_hours"].to_numpy(float)
    composite_value = float(subset["composite_repertoire_w1_hours"].iloc[0])
    ax.barh(
        row_centers,
        values,
        height=lane_height * 0.58,
        color="#4D4D4D",
        alpha=0.86,
        zorder=2,
    )
    ax.axvline(
        composite_value,
        color="#D62728",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=3,
    )
    ax.set_ylim(
        0.0,
        float(np.max(row_centers) + lane_height / 2.0 + lane_height * LANE_GAP_FRACTION),
    )
    ax.set_xlim(0.0, bar_limit)
    ax.set_yticks(row_centers, [""] * len(row_centers))
    ax.set_xlabel("Per-behavior W1 (h)" if show_x_label else "", fontsize=8.0)
    ax.set_title("Composite W1", fontsize=8.5, pad=1.0)
    ax.grid(axis="x", color="#BBBBBB", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    if show_x_label:
        ax.tick_params(axis="x", labelsize=7.5, length=2, pad=2)
    else:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def draw_panel_a(
    ax: plt.Axes,
    colorbar_ax: plt.Axes,
    matrix: pd.DataFrame,
    metadata: CohortMetadata,
) -> int:
    animals = metadata.animals
    distances = matrix.loc[list(animals), list(animals)].to_numpy(float)
    mask = np.triu(np.ones_like(distances, dtype=bool), k=0)
    visible_values = distances[~mask]
    if len(visible_values) != len(animals) * (len(animals) - 1) // 2:
        raise ValueError("Panel A does not contain exactly one displayed cell per unique pair.")
    vmax = float(np.max(visible_values))
    if vmax <= 0:
        vmax = 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(
        np.ma.masked_where(mask, distances),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="none",
    )
    n_animals = len(animals)
    tick_fontsize = max(5.5, min(8.5, 78.0 / max(n_animals, 1)))
    positions = np.arange(n_animals)
    ax.set_xticks(
        positions,
        animals,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticks(positions, animals, fontsize=tick_fontsize)
    ax.tick_params(axis="both", length=2, pad=2)
    ax.set_xlim(-0.5, n_animals - 0.5)
    ax.set_ylim(n_animals - 0.5, -0.5)
    cumulative = 0
    for group in metadata.group_order:
        group_animals = [
            animal for animal in animals if metadata.group_by_animal[animal] == group
        ]
        start = cumulative
        end = cumulative + len(group_animals) - 1
        center = (start + end) / 2.0
        ax.text(
            center,
            1.015,
            metadata.group_display_names[group],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=max(7.0, tick_fontsize),
            color=metadata.group_colors[group],
        )
        cumulative += len(group_animals)
        if cumulative < n_animals:
            boundary = cumulative - 0.5
            ax.axhline(boundary, color="#8c8c8c", linewidth=0.6, alpha=0.65, zorder=3)
            ax.axvline(boundary, color="#8c8c8c", linewidth=0.6, alpha=0.65, zorder=3)
    for tick, animal in zip(ax.get_xticklabels(), animals):
        tick.set_color(metadata.group_colors[metadata.group_by_animal[animal]])
    for tick, animal in zip(ax.get_yticklabels(), animals):
        tick.set_color(metadata.group_colors[metadata.group_by_animal[animal]])
    ax.set_title(r"Composite circular $W_1$ distance", fontsize=10.5, pad=20)
    ax.set_aspect("equal", adjustable="box")
    colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"Composite $W_1$ (h)", fontsize=7.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    ax.text(-0.17, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    return len(visible_values)


def draw_panel_b(
    ax: plt.Axes,
    reference_table: pd.DataFrame,
    metadata: CohortMetadata,
) -> None:
    panel_groups = (metadata.control_group, metadata.experimental_group)
    positions = {group: float(index) for index, group in enumerate(panel_groups)}
    for group in panel_groups:
        rows = reference_table[reference_table["group"] == group]
        count = len(rows)
        half_width = min(0.28, max(0.12, 0.12 + 0.16 * min(1.0, (count - 1) / 12.0)))
        offsets = np.zeros(1) if count == 1 else np.linspace(-half_width, half_width, count)
        ax.scatter(
            positions[group] + offsets,
            rows["distance_to_control_reference_hours"].to_numpy(float),
            s=48,
            marker="o",
            color=metadata.group_colors[group],
            edgecolor="#222222",
            linewidth=0.55,
            zorder=3,
        )
    y_values = reference_table["distance_to_control_reference_hours"].to_numpy(float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    padding = max(0.05, 0.08 * (y_max - y_min)) if y_max > y_min else 0.1
    ax.set_xlim(-0.45, len(panel_groups) - 0.55)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xticks(
        list(positions.values()),
        [metadata.group_display_names[group] for group in panel_groups],
        fontsize=8.5,
    )
    control_name = metadata.group_display_names[metadata.control_group]
    ax.set_ylabel(rf"Distance to {control_name} reference, $W_1$ (h)", fontsize=8.5)
    ax.set_title("Distance to control reference", fontsize=10.5, pad=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, length=2)
    ax.tick_params(axis="x", length=2, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.text(-0.17, 1.05, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")


def draw_panel_c(
    figure: plt.Figure,
    subplot_spec,
    display: pd.DataFrame,
    selected_pairs: list[dict[str, object]],
    pair_behavior: pd.DataFrame,
    relative_density_max: float,
    bar_limit: float,
) -> None:
    grid = subplot_spec.subgridspec(
        3,
        3,
        width_ratios=(0.82, 0.82, 0.54),
        hspace=0.12,
        wspace=0.06,
    )
    row_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    row_titles = ("Low distance", "Intermediate distance", "High distance")
    for row_index, pair in enumerate(selected_pairs):
        axis_i = figure.add_subplot(grid[row_index, 0])
        axis_j = figure.add_subplot(grid[row_index, 1])
        axis_bar = figure.add_subplot(grid[row_index, 2])
        draw_profile_axis(
            axis_i,
            display,
            str(pair["animal_i"]),
            relative_density_max,
            show_behavior_labels=True,
            show_x_labels=row_index == len(selected_pairs) - 1,
            show_x_label=row_index == len(selected_pairs) - 1,
        )
        _, row_centers = draw_profile_axis(
            axis_j,
            display,
            str(pair["animal_j"]),
            relative_density_max,
            show_behavior_labels=False,
            show_x_labels=row_index == len(selected_pairs) - 1,
            show_x_label=False,
        )
        lane_height, _, _, _ = _lane_geometry(relative_density_max)
        draw_w1_axis(
            axis_bar,
            pair,
            pair_behavior,
            bar_limit,
            row_centers,
            lane_height,
            show_x_label=row_index == len(selected_pairs) - 1,
        )
        row_axes.append((axis_i, axis_j, axis_bar))

    for row_title, (axis_i, axis_j, _) in zip(row_titles, row_axes):
        left = axis_i.get_position().x0
        right = axis_j.get_position().x1
        top = axis_i.get_position().y1
        figure.text(
            (left + right) / 2.0,
            top + 0.004,
            row_title,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    row_axes[0][0].text(-0.18, 1.13, "C", transform=row_axes[0][0].transAxes, fontsize=12, fontweight="bold")


def build_integrated_figure(
    matrix: pd.DataFrame,
    metadata: CohortMetadata,
    reference_table: pd.DataFrame,
    display: pd.DataFrame,
    selected_pairs: list[dict[str, object]],
    pair_behavior: pd.DataFrame,
    relative_density_max: float,
    bar_limit: float,
    png_path: Path,
    svg_path: Path,
) -> None:
    figure_width = max(15.0, 10.6 + 0.25 * max(0, len(metadata.animals) - 8))
    figure = plt.figure(figsize=(figure_width, 12.6))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 2.20),
        height_ratios=(1.60, 1.0),
        wspace=0.28,
        hspace=0.32,
    )
    panel_a_grid = outer[0, 0].subgridspec(
        3,
        2,
        width_ratios=(0.90, 0.035),
        height_ratios=(0.18, 0.64, 0.18),
        wspace=0.06,
        hspace=0.0,
    )
    panel_a_axis = figure.add_subplot(panel_a_grid[:, 0])
    panel_a_colorbar = figure.add_subplot(panel_a_grid[1, 1])
    draw_panel_a(panel_a_axis, panel_a_colorbar, matrix, metadata)
    panel_b_axis = figure.add_subplot(outer[1, 0])
    draw_panel_b(panel_b_axis, reference_table, metadata)
    draw_panel_c(
        figure,
        outer[:, 1],
        display,
        selected_pairs,
        pair_behavior,
        relative_density_max,
        bar_limit,
    )
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    metadata: CohortMetadata,
    matrix: pd.DataFrame,
    validation: dict[str, object],
    reference_table: pd.DataFrame,
    selected_pairs: list[dict[str, object]],
    selection_score: float,
    display_stats: dict[str, float | int],
    bar_limit: float,
) -> None:
    selection_columns = [
        "distance_class",
        "animal_i",
        "animal_j",
        "composite_w1_hours",
        "global_pair_rank",
        "global_pair_percentile",
        "selection_score",
    ]
    pd.DataFrame(selected_pairs, columns=selection_columns).to_csv(
        output_dir / "representative_pair_selection.csv",
        index=False,
        float_format="%.17g",
    )
    reference_table[
        ["animal", "group", "distance_to_control_reference_hours"]
    ].to_csv(
        output_dir / "distance_to_reference.csv",
        index=False,
        float_format="%.17g",
    )

    lines = [
        "Generalized composite circular-W1 cohort figure",
        "===============================================",
        "",
        f"Metadata source: {metadata.source}",
        f"Number of animals: {len(metadata.animals)}",
        "Animals per group:",
    ]
    for group in metadata.group_order:
        count = sum(metadata.group_by_animal[animal] == group for animal in metadata.animals)
        lines.append(f"  {metadata.group_display_names[group]} ({group}): {count}")
    lines.extend(
        [
            f"Total unique pairwise distances: {validation['unique_pair_count']}",
            f"Panel A matrix dimensions: {matrix.shape[0]} x {matrix.shape[1]}",
            "Panel B definition: control animals are averaged against all other control animals; experimental animals are averaged against all control animals.",
            f"Control group: {metadata.group_display_names[metadata.control_group]}",
            f"Experimental group: {metadata.group_display_names[metadata.experimental_group]}",
            "",
            "Automatic Panel C pair selection:",
            "  objective: minimize abs(low percentile - 0.0) + abs(intermediate percentile - 0.5) + abs(high percentile - 1.0)",
            "  tie-breaking: lower low-pair rank, higher high-pair rank, then lexicographic pair IDs",
        ]
    )
    for pair in selected_pairs:
        lines.append(
            f"  {pair['distance_class']}: {pair['animal_i']} vs {pair['animal_j']}; "
            f"composite W1={float(pair['composite_w1_hours']):.17g} h; "
            f"rank={pair['global_pair_rank']}/{validation['unique_pair_count']}; "
            f"percentile={float(pair['global_pair_percentile']):.17g}"
        )
    selected_animals = [
        animal
        for pair in selected_pairs
        for animal in (pair["animal_i"], pair["animal_j"])
    ]
    lines.extend(
        [
            f"  six unique animals: {len(set(selected_animals)) == 6}",
            f"  objective score: {selection_score:.17g}",
            f"Common Panel C amplitude scale: {float(display_stats['relative_density_max']):.17g} relative density vs uniform",
            f"Common Panel C per-behavior W1 scale: 0 to {bar_limit:.17g} h including padding",
            "No smoothing: true; connected segments are straight and 30-minute aggregation is display-only.",
            "",
            "Validation:",
            f"  maximum phase probability-sum error: {float(validation['max_probability_sum_error']):.3g}",
            f"  maximum phase-grid error: {float(validation['max_phase_grid_error']):.3g} h",
            "  composite distance matrix symmetry: pass",
            "  composite distance matrix diagonal: zero",
            f"  maximum wide/long behavior-W1 error: {float(validation['max_cross_file_w1_error']):.3g} h",
            f"  maximum composite-versus-eight-value mean error: {float(validation['max_composite_mean_error']):.3g} h",
            "  Panel A displays every unique pair once in the lower triangle; diagonal and upper triangle are masked.",
            "  Panel B values were computed from the locked control-reference definition.",
            "  Panel C pairs are genotype-blind, rank-based, deterministic, and animal-disjoint.",
            "  No raw data or upstream circular-W1 analysis files were modified.",
        ]
    )
    (output_dir / "figure_generation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    phase, pairwise, composite = load_sources(input_dir)
    metadata = load_metadata(
        args.metadata.expanduser().resolve() if args.metadata is not None else None,
        composite,
        control_group=args.control_group,
        experimental_group=args.experimental_group,
    )
    matrix, validation = validate_and_build_matrix(phase, pairwise, composite, metadata)
    reference_table = compute_distance_to_control_reference(matrix, metadata)
    pair_records = rank_pairwise_distances(matrix, metadata.animals)
    selected_pairs, selection_score = select_representative_pairs(
        pair_records, metadata.animals
    )
    pair_behavior = selected_pair_behavior_table(selected_pairs, validation)
    display, display_stats = aggregate_to_30_minutes(
        validation["phase"], metadata.animals
    )
    relative_density_max = float(display_stats["relative_density_max"])
    max_behavior_w1 = float(pair_behavior["behavior_w1_hours"].max())
    bar_limit = max_behavior_w1 * 1.18 if max_behavior_w1 > 0 else 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    build_integrated_figure(
        matrix,
        metadata,
        reference_table,
        display,
        selected_pairs,
        pair_behavior,
        relative_density_max,
        bar_limit,
        output_dir / "repertoire_circular_w1_figure.png",
        output_dir / "repertoire_circular_w1_figure.svg",
    )
    write_outputs(
        output_dir,
        metadata,
        matrix,
        validation,
        reference_table,
        selected_pairs,
        selection_score,
        display_stats,
        bar_limit,
    )

    print(f"Wrote generalized A-C figure outputs to {output_dir}")
    for pair in selected_pairs:
        print(
            f"{pair['distance_class']}: {pair['animal_i']}__{pair['animal_j']} "
            f"composite_w1={float(pair['composite_w1_hours']):.6f} h "
            f"percentile={float(pair['global_pair_percentile']):.6f}"
        )
    print(f"selection_score={selection_score:.6f}")


if __name__ == "__main__":
    main()
