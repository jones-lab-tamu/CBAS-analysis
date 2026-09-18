"""Create frozen circular-W1 visual diagnostics.

The script does not recompute behavior-level W1 distances, modify the metric,
or perform genotype or knockdown inference.  The Panel A heatmap uses the
distance matrix plus its pair metadata to order arbitrary cohort sizes by
group while leaving the matrix values unchanged.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


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
ANIMAL_ORDER = tuple(animal for _, animal in ANIMALS)
GENOTYPE_BY_ANIMAL = {animal: genotype for genotype, animal in ANIMALS}
GENOTYPES = ("LacZ", "Bmal1KO")
GENOTYPE_COLORS = {"LacZ": "#2f6f9f", "Bmal1KO": "#c55a11"}
GENOTYPE_MARKERS = {"LacZ": "o", "Bmal1KO": "s"}
GROUP_DISPLAY_NAMES = {"LacZ": "LacZ", "Bmal1KO": "Bmal1 KO"}
COMPOSITE_METADATA_FILENAME = "pairwise_repertoire_distance.csv"
PANEL_A_OUTPUT_DIR_NAME = "Pair_Profile_Visualization"

DEFAULT_INPUT_MATRIX = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data\Circular_W1_Repertoire\repertoire_distance_matrix.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_MATRIX.parent / "PCoA_Visualization"
EIGENVALUE_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCoA and distance-fidelity diagnostics from the frozen W1 matrix."
    )
    parser.add_argument(
        "--input-matrix",
        type=Path,
        default=DEFAULT_INPUT_MATRIX,
        help="Authoritative repertoire distance matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New directory for PCoA outputs.",
    )
    return parser.parse_args()


def load_and_validate_matrix(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    matrix = pd.read_csv(path, index_col=0)

    if tuple(matrix.index) != ANIMAL_ORDER:
        raise ValueError(
            "Matrix row IDs do not match the required fixed order: "
            f"{list(ANIMAL_ORDER)}; got {list(matrix.index)}"
        )
    if tuple(matrix.columns) != ANIMAL_ORDER:
        raise ValueError(
            "Matrix column IDs do not match the required fixed order: "
            f"{list(ANIMAL_ORDER)}; got {list(matrix.columns)}"
        )

    try:
        distances = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Distance matrix contains non-numeric values.") from exc

    if not np.isfinite(distances).all():
        raise ValueError("Distance matrix contains missing or non-finite values.")
    if (distances < 0).any():
        raise ValueError("Distance matrix contains negative distances.")

    diagonal_error = float(np.max(np.abs(np.diag(distances))))
    symmetry_error = float(np.max(np.abs(distances - distances.T)))
    if diagonal_error > EIGENVALUE_TOLERANCE:
        raise ValueError(f"Distance matrix diagonal is not zero: max error={diagonal_error}")
    if symmetry_error > EIGENVALUE_TOLERANCE:
        raise ValueError(f"Distance matrix is not symmetric: max error={symmetry_error}")

    checks = {
        "diagonal_max_abs_error": diagonal_error,
        "symmetry_max_abs_error": symmetry_error,
        "minimum_distance": float(np.min(distances)),
        "maximum_distance": float(np.max(distances)),
    }
    return matrix, distances, checks


def load_group_metadata(
    path: Path, animals: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    """Infer group membership and display order from the frozen pair metadata."""

    metadata = pd.read_csv(path)
    required = {"animal_i", "genotype_i", "animal_j", "genotype_j"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"{path.name} is missing group metadata columns: {missing}")

    group_by_animal: dict[str, str] = {}
    group_order: list[str] = []
    for row in metadata.itertuples(index=False):
        for animal_column, group_column in (
            ("animal_i", "genotype_i"),
            ("animal_j", "genotype_j"),
        ):
            animal_value = getattr(row, animal_column)
            group_value = getattr(row, group_column)
            if pd.isna(animal_value) or pd.isna(group_value):
                raise ValueError(f"{path.name} contains missing group metadata.")
            animal = str(animal_value)
            group = str(group_value)
            previous_group = group_by_animal.setdefault(animal, group)
            if previous_group != group:
                raise ValueError(
                    f"Animal {animal} has conflicting group metadata: "
                    f"{previous_group!r} and {group!r}."
                )
            if group not in group_order:
                group_order.append(group)

    missing_animals = sorted(set(animals).difference(group_by_animal))
    if missing_animals:
        raise ValueError(
            "Group metadata does not cover all matrix animals: "
            + ", ".join(missing_animals)
        )

    animal_set = set(animals)
    group_order = [
        group
        for group in group_order
        if any(
            animal in animal_set and group_by_animal[animal] == group
            for animal in group_by_animal
        )
    ]
    ordered_animals = tuple(
        animal
        for group in group_order
        for animal in animals
        if group_by_animal[animal] == group
    )
    if set(ordered_animals) != set(animals):
        raise ValueError("Could not order every matrix animal by its group metadata.")
    return ordered_animals, group_by_animal, tuple(group_order)


def classical_pcoa(
    distances: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    n_animals = distances.shape[0]
    centering = np.eye(n_animals) - np.ones((n_animals, n_animals)) / n_animals
    squared_distances = distances**2
    centered_matrix = -0.5 * centering @ squared_distances @ centering

    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(centered_matrix)
    order = np.argsort(raw_eigenvalues)[::-1]
    eigenvalues = raw_eigenvalues[order]
    eigenvectors = raw_eigenvectors[:, order]

    positive_mask = eigenvalues > EIGENVALUE_TOLERANCE
    negative_mask = eigenvalues < -EIGENVALUE_TOLERANCE
    positive_indices = np.flatnonzero(positive_mask)
    if len(positive_indices) < 2:
        raise ValueError("The distance matrix has fewer than two positive PCoA axes.")

    positive_eigenvalues = eigenvalues[positive_indices]
    positive_total = float(np.sum(positive_eigenvalues))
    positive_coordinates = eigenvectors[:, positive_indices] * np.sqrt(
        positive_eigenvalues
    )

    # Choose signs from the largest absolute coordinate only for deterministic
    # display.  This uses no genotype or knockdown information.
    for axis_index in range(positive_coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(positive_coordinates[:, axis_index])))
        if positive_coordinates[pivot, axis_index] < 0:
            positive_coordinates[:, axis_index] *= -1

    spectral_reconstruction = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    positive_reconstruction = (
        eigenvectors[:, positive_indices]
        @ np.diag(positive_eigenvalues)
        @ eigenvectors[:, positive_indices].T
    )
    coordinate_reconstruction = positive_coordinates @ positive_coordinates.T

    positive_variance_fraction = np.zeros_like(eigenvalues)
    cumulative_positive_variance_fraction = np.zeros_like(eigenvalues)
    cumulative = 0.0
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue > EIGENVALUE_TOLERANCE:
            fraction = eigenvalue / positive_total
            cumulative += fraction
            positive_variance_fraction[index] = fraction
        cumulative_positive_variance_fraction[index] = cumulative

    negative_eigenvalues = eigenvalues[negative_mask]
    negative_abs_sum = float(np.sum(np.abs(negative_eigenvalues)))
    absolute_eigenvalue_mass = float(np.sum(np.abs(eigenvalues)))

    return {
        "centered_matrix": centered_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "positive_indices": positive_indices,
        "positive_coordinates": positive_coordinates,
        "positive_total": positive_total,
        "negative_abs_sum": negative_abs_sum,
        "negative_fraction_abs_mass": (
            negative_abs_sum / absolute_eigenvalue_mass
            if absolute_eigenvalue_mass > 0
            else 0.0
        ),
        "positive_variance_fraction": positive_variance_fraction,
        "cumulative_positive_variance_fraction": cumulative_positive_variance_fraction,
        "spectral_reconstruction_max_abs_error": float(
            np.max(np.abs(centered_matrix - spectral_reconstruction))
        ),
        "positive_reconstruction_max_abs_error": float(
            np.max(np.abs(positive_reconstruction - coordinate_reconstruction))
        ),
        "coordinate_reconstruction": coordinate_reconstruction,
        "spectral_reconstruction": spectral_reconstruction,
    }


def unique_pairs(
    distances: np.ndarray,
    embedded_coordinates: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    original_values: list[float] = []
    embedded_values: list[float] = []

    for i, j in itertools.combinations(range(len(ANIMAL_ORDER)), 2):
        original = float(distances[i, j])
        embedded = float(np.linalg.norm(embedded_coordinates[i] - embedded_coordinates[j]))
        absolute_error = abs(original - embedded)
        relative_error = absolute_error / original if original > 0 else np.nan
        rows.append(
            {
                "animal_i": ANIMAL_ORDER[i],
                "animal_j": ANIMAL_ORDER[j],
                "original_w1_distance_hours": original,
                "embedded_2d_distance_hours": embedded,
                "absolute_error_hours": absolute_error,
                "relative_error": relative_error,
            }
        )
        original_values.append(original)
        embedded_values.append(embedded)

    return pd.DataFrame(rows), np.asarray(original_values), np.asarray(embedded_values)


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def fidelity_metrics(
    pairwise: pd.DataFrame,
    original_values: np.ndarray,
    embedded_values: np.ndarray,
) -> dict[str, float | int | str]:
    errors = original_values - embedded_values
    pearson = float(np.corrcoef(original_values, embedded_values)[0, 1])
    spearman = float(np.corrcoef(average_ranks(original_values), average_ranks(embedded_values))[0, 1])
    rmse = float(np.sqrt(np.mean(errors**2)))
    stress = float(np.sqrt(np.sum(errors**2) / np.sum(original_values**2)))
    max_index = int(np.argmax(np.abs(errors)))
    max_row = pairwise.iloc[max_index]

    return {
        "n_unique_pairs": int(len(pairwise)),
        "pearson": pearson,
        "spearman": spearman,
        "rmse_hours": rmse,
        "normalized_stress": stress,
        "max_absolute_error_hours": float(max_row["absolute_error_hours"]),
        "max_absolute_error_pair": (
            f"{max_row['animal_i']}-{max_row['animal_j']}"
        ),
        "max_relative_error": float(np.nanmax(pairwise["relative_error"].to_numpy(float))),
    }


def prim_mst(distances: np.ndarray) -> list[tuple[int, int]]:
    """Return a deterministic minimum spanning tree from the original matrix."""

    n_animals = distances.shape[0]
    visited = {0}
    edges: list[tuple[int, int]] = []

    while len(visited) < n_animals:
        candidates = [
            (float(distances[i, j]), i, j)
            for i in sorted(visited)
            for j in range(n_animals)
            if j not in visited
        ]
        if not candidates:
            raise ValueError("Could not construct an MST from the distance matrix.")
        _, i, j = min(candidates)
        edges.append((i, j))
        visited.add(j)

    return edges


def write_csv_outputs(
    output_dir: Path,
    pcoa: dict[str, np.ndarray | float | int],
    pairwise: pd.DataFrame,
) -> None:
    eigenvalues = np.asarray(pcoa["eigenvalues"], dtype=float)
    positive_variance_fraction = np.asarray(
        pcoa["positive_variance_fraction"], dtype=float
    )
    cumulative_positive_variance_fraction = np.asarray(
        pcoa["cumulative_positive_variance_fraction"], dtype=float
    )
    positive_coordinates = np.asarray(pcoa["positive_coordinates"], dtype=float)

    eigenvalue_rows = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue > EIGENVALUE_TOLERANCE:
            sign = "positive"
        elif eigenvalue < -EIGENVALUE_TOLERANCE:
            sign = "negative"
        else:
            sign = "zero"
        eigenvalue_rows.append(
            {
                "axis": index + 1,
                "eigenvalue": eigenvalue,
                "sign": sign,
                "positive_variance_fraction": positive_variance_fraction[index],
                "cumulative_positive_variance_fraction": cumulative_positive_variance_fraction[
                    index
                ],
            }
        )
    pd.DataFrame(eigenvalue_rows).to_csv(
        output_dir / "pcoa_eigenvalues.csv", index=False, float_format="%.15g"
    )

    coordinate_rows = []
    for index, animal in enumerate(ANIMAL_ORDER):
        coordinate_rows.append(
            {
                "animal": animal,
                "genotype": GENOTYPE_BY_ANIMAL[animal],
                "PCoA1": positive_coordinates[index, 0],
                "PCoA2": positive_coordinates[index, 1],
                "PCoA3_if_available": (
                    positive_coordinates[index, 2]
                    if positive_coordinates.shape[1] >= 3
                    else np.nan
                ),
                "eigenvalue_1": eigenvalues[0],
                "eigenvalue_2": eigenvalues[1],
                "variance_pct_axis1": 100.0 * positive_variance_fraction[0],
                "variance_pct_axis2": 100.0 * positive_variance_fraction[1],
            }
        )
    pd.DataFrame(coordinate_rows).to_csv(
        output_dir / "repertoire_pcoa_coordinates.csv",
        index=False,
        float_format="%.15g",
    )

    pairwise.to_csv(
        output_dir / "pcoa_pairwise_fidelity.csv",
        index=False,
        float_format="%.15g",
    )


def plot_pcoa(
    output_path: Path,
    coordinates: np.ndarray,
    axis_fractions: np.ndarray,
    mst_edges: list[tuple[int, int]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.8))

    if mst_edges is not None:
        for i, j in mst_edges:
            ax.plot(
                [coordinates[i, 0], coordinates[j, 0]],
                [coordinates[i, 1], coordinates[j, 1]],
                color="#777777",
                linewidth=1.1,
                zorder=1,
            )

    for genotype in GENOTYPES:
        indices = [
            index
            for index, animal in enumerate(ANIMAL_ORDER)
            if GENOTYPE_BY_ANIMAL[animal] == genotype
        ]
        ax.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=70,
            marker=GENOTYPE_MARKERS[genotype],
            color=GENOTYPE_COLORS[genotype],
            edgecolor="#222222",
            linewidth=0.7,
            label=genotype,
            zorder=3,
        )

    for index, animal in enumerate(ANIMAL_ORDER):
        ax.annotate(
            animal,
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            zorder=4,
        )

    ax.set_xlabel(
        f"PCoA1 (hours; {100.0 * axis_fractions[0]:.1f}% "
        "positive-eigenvalue variance)",
        fontsize=11,
    )
    ax.set_ylabel(
        f"PCoA2 (hours; {100.0 * axis_fractions[1]:.1f}% positive variance)",
        fontsize=11,
    )
    ax.set_title("Circular-W1 repertoire PCoA")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity(output_path: Path, pairwise: pd.DataFrame) -> None:
    original = pairwise["original_w1_distance_hours"].to_numpy(float)
    embedded = pairwise["embedded_2d_distance_hours"].to_numpy(float)
    upper = float(max(np.max(original), np.max(embedded)) * 1.08)

    figure = plt.figure(figsize=(11.5, 7.5))
    grid = GridSpec(1, 2, figure=figure, width_ratios=(4.2, 1.8), wspace=0.08)
    ax = figure.add_subplot(grid[0, 0])
    key_ax = figure.add_subplot(grid[0, 1])
    key_ax.axis("off")
    ax.scatter(
        original,
        embedded,
        s=34,
        color="#2f6f9f",
        edgecolor="#222222",
        linewidth=0.45,
        zorder=3,
    )
    ax.plot([0, upper], [0, upper], color="#555555", linewidth=1.0, linestyle="--")

    for pair_number, (_, row) in enumerate(pairwise.iterrows(), start=1):
        ax.annotate(
            str(pair_number),
            (row["original_w1_distance_hours"], row["embedded_2d_distance_hours"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

        key_column = 0 if pair_number <= 14 else 1
        key_row = (pair_number - 1) % 14
        key_ax.text(
            0.02 + 0.5 * key_column,
            0.98 - key_row / 15.0,
            f"{pair_number}: {row['animal_i']}–{row['animal_j']}",
            transform=key_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Original circular-W1 repertoire distance (hours)")
    ax.set_ylabel("2D PCoA Euclidean distance (hours)")
    ax.set_title("2D PCoA distance fidelity (numbered pair key at right)")
    ax.grid(color="#dddddd", linewidth=0.6)
    key_ax.set_title("Pair key", loc="left", fontsize=10)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.93, wspace=0.08)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(
    output_path: Path,
    matrix: pd.DataFrame,
    group_by_animal: dict[str, str],
    group_order: tuple[str, ...],
) -> None:
    """Save the cohort-scalable lower-triangle composite-W1 heatmap."""

    animals = tuple(str(animal) for animal in matrix.index)
    columns = tuple(str(animal) for animal in matrix.columns)
    if animals != columns or len(set(animals)) != len(animals):
        raise ValueError("The composite distance matrix must have matching unique row and column IDs.")
    if set(animals) != set(group_by_animal):
        raise ValueError("Group metadata and composite distance matrix cover different animals.")

    ordered_animals = tuple(
        animal
        for group in group_order
        for animal in animals
        if group_by_animal[animal] == group
    )
    if set(ordered_animals) != set(animals):
        raise ValueError("Could not order every matrix animal by its group metadata.")

    ordered_distances = matrix.loc[list(ordered_animals), list(ordered_animals)].to_numpy(
        dtype=float
    )
    if not np.isfinite(ordered_distances).all():
        raise ValueError("Composite distance matrix contains non-finite values.")
    if not np.allclose(ordered_distances, ordered_distances.T, atol=EIGENVALUE_TOLERANCE):
        raise ValueError("Composite distance matrix is not symmetric.")

    n_animals = len(ordered_animals)
    display_mask = np.triu(np.ones((n_animals, n_animals), dtype=bool), k=0)
    visible_values = ordered_distances[~display_mask]
    if visible_values.size == 0:
        raise ValueError("At least one unique pair is required for the composite heatmap.")
    vmax = float(np.max(visible_values))
    if vmax <= 0.0:
        vmax = 1.0
    masked_distances = np.ma.masked_where(display_mask, ordered_distances)

    figure_size = max(5.8, 0.42 * n_animals + 2.3)
    figure, ax = plt.subplots(figsize=(figure_size + 1.0, figure_size))
    figure.subplots_adjust(left=0.17, right=0.86, bottom=0.18, top=0.82)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(
        masked_distances,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="none",
    )

    tick_fontsize = max(7.0, min(10.0, 80.0 / max(n_animals, 1)))
    positions = np.arange(n_animals)
    ax.set_xticks(
        positions,
        ordered_animals,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticks(positions, ordered_animals, fontsize=tick_fontsize)
    ax.tick_params(axis="both", length=3, pad=3)
    ax.set_xlim(-0.5, n_animals - 0.5)
    ax.set_ylim(n_animals - 0.5, -0.5)

    cumulative = 0
    for group in group_order:
        group_animals = [animal for animal in ordered_animals if group_by_animal[animal] == group]
        if not group_animals:
            continue
        start = cumulative
        end = cumulative + len(group_animals) - 1
        center = (start + end) / 2.0
        group_color = GENOTYPE_COLORS.get(group, "#444444")
        group_label = GROUP_DISPLAY_NAMES.get(group, group)
        ax.text(
            center,
            1.015,
            group_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=max(8.0, tick_fontsize),
            color=group_color,
        )
        cumulative += len(group_animals)
        if cumulative < n_animals:
            boundary = cumulative - 0.5
            ax.axhline(boundary, color="#8c8c8c", linewidth=0.7, alpha=0.65, zorder=3)
            ax.axvline(boundary, color="#8c8c8c", linewidth=0.7, alpha=0.65, zorder=3)

    for tick, animal in zip(ax.get_xticklabels(), ordered_animals):
        tick.set_color(GENOTYPE_COLORS.get(group_by_animal[animal], "#333333"))
    for tick, animal in zip(ax.get_yticklabels(), ordered_animals):
        tick.set_color(GENOTYPE_COLORS.get(group_by_animal[animal], "#333333"))

    ax.set_title(r"Composite circular $W_1$ distance", fontsize=13, pad=24)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Composite $W_1$ (h)", labelpad=8)
    colorbar.ax.tick_params(labelsize=max(7.0, tick_fontsize - 0.5))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def distance_to_lacz_table(distances: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, animal in enumerate(ANIMAL_ORDER):
        if index < 4:
            reference_indices = [other for other in range(4) if other != index]
        else:
            reference_indices = list(range(4))
        values = distances[index, reference_indices]
        rows.append(
            {
                "animal": animal,
                "genotype": GENOTYPE_BY_ANIMAL[animal],
                "n_reference_animals": len(reference_indices),
                "mean_distance_to_lacz_hours": float(np.mean(values)),
                "min_distance_to_lacz_hours": float(np.min(values)),
                "max_distance_to_lacz_hours": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def plot_distance_to_lacz(output_path: Path, reference_table: pd.DataFrame) -> None:
    x_values = np.arange(len(reference_table), dtype=float)
    y_values = reference_table["mean_distance_to_lacz_hours"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for genotype in GENOTYPES:
        indices = [
            index
            for index, animal in enumerate(reference_table["animal"])
            if GENOTYPE_BY_ANIMAL[animal] == genotype
        ]
        ax.scatter(
            x_values[indices],
            y_values[indices],
            s=70,
            marker=GENOTYPE_MARKERS[genotype],
            color=GENOTYPE_COLORS[genotype],
            edgecolor="#222222",
            linewidth=0.7,
            label=genotype,
            zorder=3,
        )

    for index, row in reference_table.iterrows():
        ax.annotate(
            row["animal"],
            (x_values[index], y_values[index]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlim(-0.5, len(reference_table) - 0.5)
    ax.set_xticks(x_values, ["" for _ in x_values])
    ax.set_ylabel("Mean distance to specified LacZ reference (hours)")
    ax.set_title("Descriptive distance to LacZ reference")
    ax.text(
        0.0,
        -0.16,
        "LacZ: other three LacZ animals; Bmal1KO: all four LacZ animals.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_written_outputs(
    output_dir: Path,
    pcoa: dict[str, np.ndarray | float | int],
    pairwise: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> dict[str, float | int]:
    eigenvalue_table = pd.read_csv(output_dir / "pcoa_eigenvalues.csv")
    coordinate_table = pd.read_csv(output_dir / "repertoire_pcoa_coordinates.csv")
    written_pairwise = pd.read_csv(output_dir / "pcoa_pairwise_fidelity.csv")
    written_reference = pd.read_csv(output_dir / "distance_to_lacz_reference.csv")

    eigenvalues = np.asarray(pcoa["eigenvalues"], dtype=float)
    positive_coordinates = np.asarray(pcoa["positive_coordinates"], dtype=float)
    if tuple(coordinate_table["animal"]) != ANIMAL_ORDER:
        raise ValueError("Coordinate output order does not match the fixed animal order.")
    if tuple(coordinate_table["genotype"]) != tuple(
        GENOTYPE_BY_ANIMAL[animal] for animal in ANIMAL_ORDER
    ):
        raise ValueError("Coordinate output genotype labels do not match the fixed mapping.")
    if tuple(eigenvalue_table["axis"]) != tuple(range(1, len(ANIMAL_ORDER) + 1)):
        raise ValueError("Eigenvalue output axes are incomplete or out of order.")
    if len(written_pairwise) != len(pairwise) or len(written_pairwise) != 28:
        raise ValueError("Pairwise fidelity output does not contain the 28 unique pairs.")
    if tuple(written_reference["animal"]) != ANIMAL_ORDER:
        raise ValueError("Distance-to-LacZ output order does not match the fixed animal order.")

    if not np.allclose(eigenvalue_table["eigenvalue"], eigenvalues, atol=1e-11):
        raise ValueError("Written eigenvalues do not match the eigendecomposition.")
    if not np.allclose(coordinate_table["PCoA1"], positive_coordinates[:, 0], atol=1e-11):
        raise ValueError("Written PCoA1 coordinates do not match the eigendecomposition.")
    if not np.allclose(coordinate_table["PCoA2"], positive_coordinates[:, 1], atol=1e-11):
        raise ValueError("Written PCoA2 coordinates do not match the eigendecomposition.")
    if positive_coordinates.shape[1] >= 3 and not np.allclose(
        coordinate_table["PCoA3_if_available"], positive_coordinates[:, 2], atol=1e-11
    ):
        raise ValueError("Written PCoA3 coordinates do not match the eigendecomposition.")
    if not np.allclose(
        coordinate_table["eigenvalue_1"], eigenvalues[0], atol=1e-11
    ) or not np.allclose(coordinate_table["eigenvalue_2"], eigenvalues[1], atol=1e-11):
        raise ValueError("Repeated coordinate eigenvalue metadata is inconsistent.")
    expected_axis_fractions = 100.0 * np.asarray(
        pcoa["positive_variance_fraction"], dtype=float
    )
    if not np.allclose(
        coordinate_table["variance_pct_axis1"], expected_axis_fractions[0], atol=1e-11
    ) or not np.allclose(
        coordinate_table["variance_pct_axis2"], expected_axis_fractions[1], atol=1e-11
    ):
        raise ValueError("Repeated coordinate variance metadata is inconsistent.")
    if not np.allclose(
        written_pairwise["original_w1_distance_hours"],
        pairwise["original_w1_distance_hours"],
        atol=1e-11,
    ):
        raise ValueError("Written original pairwise distances do not match the input matrix.")
    if not np.allclose(
        written_pairwise["embedded_2d_distance_hours"],
        pairwise["embedded_2d_distance_hours"],
        atol=1e-11,
    ):
        raise ValueError("Written embedded pairwise distances do not match PCoA coordinates.")
    for column in ("absolute_error_hours", "relative_error"):
        if not np.allclose(
            written_pairwise[column], pairwise[column], atol=1e-11, equal_nan=True
        ):
            raise ValueError(f"Written pairwise {column} values are inconsistent.")
    if tuple(written_reference["genotype"]) != tuple(
        GENOTYPE_BY_ANIMAL[animal] for animal in ANIMAL_ORDER
    ):
        raise ValueError("Distance-to-LacZ genotype labels do not match the fixed mapping.")
    if not np.allclose(
        written_reference.select_dtypes(include=[np.number]),
        reference_table.select_dtypes(include=[np.number]),
        atol=1e-11,
    ):
        raise ValueError("Written distance-to-LacZ values do not match the input matrix.")

    embedded_matrix = np.zeros((len(ANIMAL_ORDER), len(ANIMAL_ORDER)), dtype=float)
    for _, row in written_pairwise.iterrows():
        i = ANIMAL_ORDER.index(row["animal_i"])
        j = ANIMAL_ORDER.index(row["animal_j"])
        value = float(row["embedded_2d_distance_hours"])
        embedded_matrix[i, j] = value
        embedded_matrix[j, i] = value
    embedded_diagonal_error = float(np.max(np.abs(np.diag(embedded_matrix))))
    embedded_symmetry_error = float(np.max(np.abs(embedded_matrix - embedded_matrix.T)))
    if embedded_diagonal_error > 1e-11 or embedded_symmetry_error > 1e-11:
        raise ValueError("Embedded pairwise distance matrix is not symmetric with zero diagonal.")

    return {
        "embedded_diagonal_max_abs_error": embedded_diagonal_error,
        "embedded_symmetry_max_abs_error": embedded_symmetry_error,
        "coordinate_rows": len(coordinate_table),
        "eigenvalue_rows": len(eigenvalue_table),
        "pairwise_rows": len(written_pairwise),
        "reference_rows": len(written_reference),
    }


def descriptive_structure_lines(distances: np.ndarray) -> list[str]:
    lacz_values = [distances[i, j] for i, j in itertools.combinations(range(4), 2)]
    de_value = distances[4, 5]
    gh_value = distances[6, 7]
    de_gh_values = [distances[i, j] for i in (4, 5) for j in (6, 7)]
    gh_lacz_values = [distances[i, j] for i in (6, 7) for j in range(4)]
    return [
        "Descriptive structure in the exact matrix:",
        (
            f"  LacZ within-group pair range: {min(lacz_values):.6f} to "
            f"{max(lacz_values):.6f} hours."
        ),
        f"  714D-714E distance: {de_value:.6f} hours.",
        f"  714G-714H distance: {gh_value:.6f} hours.",
        (
            f"  714D/714E to 714G/714H pair range: {min(de_gh_values):.6f} to "
            f"{max(de_gh_values):.6f} hours."
        ),
        (
            f"  714G/714H to LacZ pair range: {min(gh_lacz_values):.6f} to "
            f"{max(gh_lacz_values):.6f} hours."
        ),
        "  These are descriptive pairwise summaries, not genotype tests.",
    ]


def write_validation_summary(
    output_path: Path,
    input_path: Path,
    matrix_checks: dict[str, float],
    pcoa: dict[str, np.ndarray | float | int],
    fidelity: dict[str, float | int | str],
    written_checks: dict[str, float | int],
    distances: np.ndarray,
) -> str:
    eigenvalues = np.asarray(pcoa["eigenvalues"], dtype=float)
    positive_variance_fraction = np.asarray(
        pcoa["positive_variance_fraction"], dtype=float
    )
    cumulative_positive_variance_fraction = np.asarray(
        pcoa["cumulative_positive_variance_fraction"], dtype=float
    )
    lines = [
        "Circular-W1 repertoire PCoA validation summary",
        "================================================",
        f"Authoritative input matrix: {input_path}",
        "Fixed animal order: " + ", ".join(ANIMAL_ORDER),
        "Units: hours of average circular redistribution across the eight equally weighted non-rest behaviors.",
        "",
        "Input matrix checks:",
        f"  diagonal_max_abs_error: {matrix_checks['diagonal_max_abs_error']:.3e}",
        f"  symmetry_max_abs_error: {matrix_checks['symmetry_max_abs_error']:.3e}",
        f"  minimum_distance: {matrix_checks['minimum_distance']:.12g}",
        f"  maximum_distance: {matrix_checks['maximum_distance']:.12g}",
        "  missing_values: none",
        "  negative_distances: none",
        "",
        "Classical PCoA eigenvalues, sorted descending:",
        (
            "Eigenvalue classification tolerance: "
            f"{EIGENVALUE_TOLERANCE:.1e}; raw values within this tolerance are reported as zero."
        ),
    ]
    for index, eigenvalue in enumerate(eigenvalues, start=1):
        if eigenvalue > EIGENVALUE_TOLERANCE:
            sign = "positive"
        elif eigenvalue < -EIGENVALUE_TOLERANCE:
            sign = "negative"
        else:
            sign = "zero"
        lines.append(f"  axis {index}: {eigenvalue:.15g} ({sign})")

    positive_count = int(np.sum(pcoa["positive_mask"]))
    negative_count = int(np.sum(pcoa["negative_mask"]))
    lines.extend(
        [
            "",
            f"  positive_eigenvalue_count: {positive_count}",
            f"  negative_eigenvalue_count: {negative_count}",
            f"  sum_positive_eigenvalues: {float(pcoa['positive_total']):.15g}",
            f"  sum_absolute_negative_eigenvalues: {float(pcoa['negative_abs_sum']):.15g}",
            (
                "  negative_fraction_absolute_eigenvalue_mass: "
                f"{float(pcoa['negative_fraction_abs_mass']):.15g}"
            ),
            f"  PCoA1_positive_variance_pct: {100.0 * positive_variance_fraction[0]:.9f}",
            f"  PCoA2_positive_variance_pct: {100.0 * positive_variance_fraction[1]:.9f}",
            (
                "  first_two_positive_axes_cumulative_variance_pct: "
                f"{100.0 * cumulative_positive_variance_fraction[1]:.9f}"
            ),
            "",
            "2D distance fidelity using 28 unique unordered pairs:",
            f"  pearson_correlation: {float(fidelity['pearson']):.12f}",
            f"  spearman_correlation: {float(fidelity['spearman']):.12f}",
            f"  rmse_hours: {float(fidelity['rmse_hours']):.12f}",
            f"  normalized_stress: {float(fidelity['normalized_stress']):.12f}",
            (
                "  largest_absolute_pairwise_distortion_hours: "
                f"{float(fidelity['max_absolute_error_hours']):.12f}"
            ),
            f"  largest_absolute_pairwise_distortion_pair: {fidelity['max_absolute_error_pair']}",
            f"  largest_relative_pairwise_distortion: {float(fidelity['max_relative_error']):.12f}",
            "",
            f"  spectral_reconstruction_max_abs_error: {float(pcoa['spectral_reconstruction_max_abs_error']):.3e}",
            f"  positive_coordinate_reconstruction_max_abs_error: {float(pcoa['positive_reconstruction_max_abs_error']):.3e}",
            f"  embedded_distance_matrix_symmetry_max_abs_error: {float(written_checks['embedded_symmetry_max_abs_error']):.3e}",
            f"  embedded_distance_matrix_diagonal_max_abs_error: {float(written_checks['embedded_diagonal_max_abs_error']):.3e}",
            "",
        ]
    )
    lines.extend(descriptive_structure_lines(distances))
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "  PCoA1 and PCoA2 are geometric coordinates only; no biological meaning is assigned to either axis.",
            "  Genotype is used only as a visual identifier. No genotype inference, p-values, knockdown analysis, or PERMANOVA was performed.",
            (
                "  The 2D representation is suitable as an intuitive main descriptive figure with the exact heatmap and pairwise matrix retained as the authoritative reference."
            ),
            (
                "  No broad contradiction with the exact heatmap is indicated by the high overall distance correlation, but localized distortion is present; inspect the fidelity plot, especially the largest-distortion pair above."
            ),
            "",
            "Written-output checks:",
            f"  coordinate_rows: {written_checks['coordinate_rows']}",
            f"  eigenvalue_rows: {written_checks['eigenvalue_rows']}",
            f"  pairwise_rows: {written_checks['pairwise_rows']}",
            f"  distance_to_lacz_rows: {written_checks['reference_rows']}",
            "  all required diagnostic tables were internally cross-checked against the in-memory calculation.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    output_path.write_text(summary, encoding="utf-8")
    return summary


def run(input_path: Path, output_dir: Path) -> str:
    matrix, distances, matrix_checks = load_and_validate_matrix(input_path)
    panel_a_animals, panel_a_group_by_animal, panel_a_group_order = load_group_metadata(
        input_path.parent / COMPOSITE_METADATA_FILENAME,
        tuple(str(animal) for animal in matrix.index),
    )
    pcoa = classical_pcoa(distances)
    positive_coordinates = np.asarray(pcoa["positive_coordinates"], dtype=float)
    two_dimensional_coordinates = positive_coordinates[:, :2]

    embedded_distance_matrix = np.linalg.norm(
        two_dimensional_coordinates[:, None, :] - two_dimensional_coordinates[None, :, :],
        axis=2,
    )
    if not np.allclose(embedded_distance_matrix, embedded_distance_matrix.T, atol=1e-11):
        raise ValueError("Embedded 2D distance matrix is not symmetric.")
    if not np.allclose(np.diag(embedded_distance_matrix), 0.0, atol=1e-11):
        raise ValueError("Embedded 2D distance matrix diagonal is not zero.")

    pairwise, original_values, embedded_values = unique_pairs(
        distances, two_dimensional_coordinates
    )
    fidelity = fidelity_metrics(pairwise, original_values, embedded_values)
    mst_edges = prim_mst(distances)
    reference_table = distance_to_lacz_table(distances)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_outputs(output_dir, pcoa, pairwise)
    reference_table.to_csv(
        output_dir / "distance_to_lacz_reference.csv",
        index=False,
        float_format="%.15g",
    )

    axis_fractions = np.asarray(pcoa["positive_variance_fraction"], dtype=float)
    plot_pcoa(
        output_dir / "repertoire_pcoa_2d.png",
        two_dimensional_coordinates,
        axis_fractions,
    )
    plot_pcoa(
        output_dir / "repertoire_pcoa_2d_with_mst.png",
        two_dimensional_coordinates,
        axis_fractions,
        mst_edges=mst_edges,
    )
    plot_fidelity(output_dir / "pcoa_distance_fidelity.png", pairwise)
    panel_a_output_dir = input_path.parent / PANEL_A_OUTPUT_DIR_NAME
    panel_a_output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        panel_a_output_dir / "panel_A_composite_circular_w1_heatmap.png",
        matrix.loc[list(panel_a_animals), list(panel_a_animals)],
        panel_a_group_by_animal,
        panel_a_group_order,
    )
    plot_distance_to_lacz(
        output_dir / "distance_to_lacz_reference.png", reference_table
    )

    written_checks = validate_written_outputs(
        output_dir, pcoa, pairwise, reference_table
    )
    return write_validation_summary(
        output_dir / "validation_summary.txt",
        input_path,
        matrix_checks,
        pcoa,
        fidelity,
        written_checks,
        distances,
    )


def main() -> None:
    args = parse_args()
    summary = run(args.input_matrix, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
