"""Create a blinded visual atlas from the saved scored phase profiles."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


PROFILE_PATH = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data"
    r"\Full_Cycle_Phase_Overlap_Recurrence\full_cycle_phase_profiles.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data"
    r"\Full_Cycle_Phase_Overlap_Recurrence\Visual_Recurrence_Atlas"
)
SEED = 20260924
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
GENOTYPE_BY_ANIMAL = {
    "675G": "LacZ",
    "675H": "LacZ",
    "675I": "LacZ",
    "675J": "LacZ",
    "714D": "Bmal1KO",
    "714E": "Bmal1KO",
    "714G": "Bmal1KO",
    "714H": "Bmal1KO",
}
ANIMALS = tuple(GENOTYPE_BY_ANIMAL)
N_CYCLES = 4
N_BINS = 288
CT_HOURS = 24.0
CT_TICKS = (0, 6, 12, 18, 24)
PROFILE_COLOR = "#3E638F"
FILL_COLOR = "#7797BA"


def _load_profiles() -> dict[
    tuple[str, str], tuple[tuple[np.ndarray, np.ndarray], ...]
]:
    """Load saved profile vectors without recomputing or transforming them."""
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(f"Saved scored profiles not found: {PROFILE_PATH}")
    profiles = pd.read_csv(
        PROFILE_PATH,
        usecols=("animal", "behavior", "cycle", "ct_bin_center_h", "probability"),
    )
    if set(profiles["animal"].unique()) != set(ANIMALS):
        raise ValueError("The saved profile file does not contain the expected cohort")
    if set(profiles["behavior"].unique()) != set(BEHAVIORS):
        raise ValueError("The saved profile file does not contain the expected behaviors")
    if set(profiles["cycle"].astype(int).unique()) != set(range(1, N_CYCLES + 1)):
        raise ValueError("The saved profiles must contain cycles 1 through 4")

    expected_centers = (np.arange(N_BINS, dtype=float) + 0.5) * CT_HOURS / N_BINS
    block_profiles: dict[
        tuple[str, str], tuple[tuple[np.ndarray, np.ndarray], ...]
    ] = {}
    for animal in ANIMALS:
        for behavior in BEHAVIORS:
            cycles: list[tuple[np.ndarray, np.ndarray]] = []
            for cycle in range(1, N_CYCLES + 1):
                selected = profiles.loc[
                    (profiles["animal"] == animal)
                    & (profiles["behavior"] == behavior)
                    & (profiles["cycle"].astype(int) == cycle)
                ].sort_values("ct_bin_center_h")
                if len(selected) != N_BINS:
                    raise ValueError("Every saved cycle profile must contain 288 phase bins")
                centers = selected["ct_bin_center_h"].to_numpy(dtype=float)
                values = selected["probability"].to_numpy(dtype=float)
                if not np.allclose(centers, expected_centers, rtol=0.0, atol=1e-10):
                    raise ValueError("Saved phase-bin centers are incomplete or out of order")
                if not np.isfinite(values).all() or np.any(values < 0.0):
                    raise ValueError("Saved profile values must be finite and nonnegative")
                if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
                    raise ValueError("Each saved scored profile must sum to one")
                cycles.append((centers, values))
            block_profiles[(animal, behavior)] = tuple(cycles)

    if len(block_profiles) != 64:
        raise ValueError("Expected exactly 64 animal-behavior profile blocks")
    return block_profiles


def _make_atlas_records(
    block_profiles: dict[
        tuple[str, str], tuple[tuple[np.ndarray, np.ndarray], ...]
    ],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Shuffle blocks reproducibly, then create IDs and a separately ordered key."""
    canonical = [
        (animal, behavior)
        for animal in ANIMALS
        for behavior in BEHAVIORS
    ]
    if len(canonical) != 64 or len(set(canonical)) != 64:
        raise ValueError("Expected 64 unique animal-behavior combinations")

    order = np.random.default_rng(SEED).permutation(len(canonical))
    repeated_order = np.random.default_rng(SEED).permutation(len(canonical))
    if not np.array_equal(order, repeated_order):
        raise AssertionError("The fixed-seed block order is not reproducible")
    atlas_records: list[dict[str, object]] = []
    id_by_block: dict[tuple[str, str], str] = {}
    for number, canonical_index in enumerate(order, start=1):
        animal, behavior = canonical[int(canonical_index)]
        anonymous_id = f"R{number:03d}"
        id_by_block[(animal, behavior)] = anonymous_id
        atlas_records.append(
            {
                "anonymous_id": anonymous_id,
                "animal": animal,
                "genotype": GENOTYPE_BY_ANIMAL[animal],
                "behavior": behavior,
                "cycles": block_profiles[(animal, behavior)],
            }
        )

    # Preserve canonical source ordering rather than matching atlas page order.
    key_rows = [
        {
            "anonymous_id": id_by_block[(animal, behavior)],
            "animal": animal,
            "genotype": GENOTYPE_BY_ANIMAL[animal],
            "behavior": behavior,
        }
        for animal, behavior in canonical
    ]
    return atlas_records, key_rows


def _render_block(figure: plt.Figure, cell, block: dict[str, object]) -> list[plt.Axes]:
    cycles = block["cycles"]
    block_max = max(float(values.max()) for _centers, values in cycles)
    shared_y_max = block_max * 1.05
    if shared_y_max <= 0.0:
        raise ValueError("A profile block must contain positive probability values")

    block_grid = cell.subgridspec(
        5,
        2,
        height_ratios=(0.18, 1, 1, 1, 1),
        width_ratios=(1.2, 6.0),
        wspace=0.025,
        hspace=0.14,
    )
    header = figure.add_subplot(block_grid[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.32,
        str(block["anonymous_id"]),
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="center",
    )

    axes: list[plt.Axes] = []
    for cycle_index, (centers, values) in enumerate(cycles, start=1):
        row = cycle_index
        cycle_label = figure.add_subplot(block_grid[row, 0])
        cycle_label.axis("off")
        cycle_label.text(
            1.0,
            0.5,
            f"Cycle {cycle_index}",
            fontsize=7.2,
            ha="right",
            va="center",
        )
        if axes:
            ax = figure.add_subplot(block_grid[row, 1], sharex=axes[0], sharey=axes[0])
        else:
            ax = figure.add_subplot(block_grid[row, 1])
        ax.fill_between(centers, 0.0, values, color=FILL_COLOR, alpha=0.24, linewidth=0)
        line, = ax.plot(centers, values, color=PROFILE_COLOR, linewidth=0.9)
        if not np.array_equal(line.get_ydata(), values) or not np.array_equal(
            line.get_xdata(), centers
        ):
            raise AssertionError("A plotted profile differs from its saved probability vector")
        ax.set_xlim(0.0, CT_HOURS)
        ax.set_ylim(0.0, shared_y_max)
        ax.set_xticks(CT_TICKS)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1.5)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(axis="x", color="#e7e7e7", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if cycle_index == N_CYCLES:
            ax.set_xlabel("CT (h)", fontsize=7.2, labelpad=2)
        else:
            ax.set_xlabel("")
        axes.append(ax)

    limits = [ax.get_ylim() for ax in axes]
    if any(limit != limits[0] for limit in limits[1:]):
        raise AssertionError("The four cycles in one block must share a y-axis scale")
    if not np.isclose(limits[0][1], shared_y_max, rtol=0.0, atol=1e-15):
        raise AssertionError("The block y-axis maximum must come from its four saved cycles")
    return axes


def _validate_blinded_text(
    figure: plt.Figure,
    anonymous_ids: set[str],
) -> None:
    visible_text = []
    for ax in figure.axes:
        visible_text.extend(text.get_text() for text in ax.texts)
        visible_text.extend(text.get_text() for text in ax.get_xticklabels())
        visible_text.extend(text.get_text() for text in ax.get_yticklabels())
        visible_text.extend((ax.get_xlabel(), ax.get_ylabel(), ax.get_title()))
    visible_text = [value for value in visible_text if value]
    allowed_ids = anonymous_ids
    for value in visible_text:
        if value.startswith("R") and value not in allowed_ids:
            raise AssertionError("Unexpected visible identifier in the atlas")
        if "cycle " in value.lower() and value.lower() not in {
            f"cycle {number}" for number in range(1, N_CYCLES + 1)
        }:
            raise AssertionError("Unexpected visible cycle label in the atlas")
        private_terms = (*ANIMALS, *GENOTYPE_BY_ANIMAL.values(), *BEHAVIORS)
        if any(term.casefold() in value.casefold() for term in private_terms):
            raise AssertionError("A private animal, genotype, or behavior label is visible")


def main() -> None:
    block_profiles = _load_profiles()
    atlas_records, key_rows = _make_atlas_records(block_profiles)
    atlas_ids = [str(block["anonymous_id"]) for block in atlas_records]
    if atlas_ids != [f"R{number:03d}" for number in range(1, 65)]:
        raise AssertionError("Anonymous identifiers are not unique and sequential")
    expected_blocks = {(animal, behavior) for animal in ANIMALS for behavior in BEHAVIORS}
    if {(row["animal"], row["behavior"]) for row in key_rows} != expected_blocks:
        raise AssertionError("The blinding key does not map 64 unique blocks")
    if len({row["anonymous_id"] for row in key_rows}) != 64:
        raise AssertionError("The blinding key contains duplicate anonymous IDs")
    key_order = [row["anonymous_id"] for row in key_rows]
    if key_order == atlas_ids:
        raise AssertionError("The blinding key row order must not reproduce atlas order")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "visual_recurrence_atlas.pdf"
    key_path = OUTPUT_DIR / "visual_recurrence_blinding_key.csv"
    rating_path = OUTPUT_DIR / "visual_recurrence_rating_sheet.csv"

    blocks_per_page = 4
    page_count = (len(atlas_records) + blocks_per_page - 1) // blocks_per_page
    if page_count != 16:
        raise AssertionError("Expected 16 atlas pages with four blocks per page")
    figure_width, figure_height = 8.5, 11.0
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(atlas_records), blocks_per_page):
            page_blocks = atlas_records[page_start : page_start + blocks_per_page]
            figure = plt.figure(figsize=(figure_width, figure_height))
            page_grid = figure.add_gridspec(
                2,
                2,
                left=0.07,
                right=0.98,
                bottom=0.055,
                top=0.97,
                hspace=0.22,
                wspace=0.18,
            )
            for slot, block in enumerate(page_blocks):
                row, column = divmod(slot, 2)
                _render_block(figure, page_grid[row, column], block)
            _validate_blinded_text(figure, set(atlas_ids))
            pdf.savefig(figure)
            plt.close(figure)

    pd.DataFrame(
        key_rows,
        columns=("anonymous_id", "animal", "genotype", "behavior"),
    ).to_csv(key_path, index=False)
    pd.DataFrame(
        {
            "anonymous_id": atlas_ids,
            "visual_recurrence_rating": [""] * len(atlas_ids),
            "notes": [""] * len(atlas_ids),
        },
        columns=("anonymous_id", "visual_recurrence_rating", "notes"),
    ).to_csv(rating_path, index=False)

    key_frame = pd.read_csv(key_path, keep_default_na=False)
    rating_frame = pd.read_csv(rating_path, keep_default_na=False)
    if len(key_frame) != 64 or key_frame["anonymous_id"].nunique() != 64:
        raise AssertionError("The saved blinding key failed its uniqueness check")
    if len(rating_frame) != 64 or rating_frame["anonymous_id"].tolist() != atlas_ids:
        raise AssertionError("The saved rating sheet does not match atlas order")
    if not (rating_frame["visual_recurrence_rating"].eq("").all() and rating_frame["notes"].eq("").all()):
        raise AssertionError("Rating and notes fields must be blank")

    print("Anonymous blocks generated: 64")
    print(f"Atlas pages: {page_count}")
    print(f"Blocks per page: {blocks_per_page}")
    print("Validation checks passed: 64 unique blocks; 4 cycles x 288 bins; shared within-block y-scales; fixed-seed order; unique key; blank rating sheet.")
    print("Visible atlas labels contain anonymous IDs, cycle labels, and CT-axis markings only.")
    print(f"Output: {pdf_path}")
    print(f"Output: {key_path}")
    print(f"Output: {rating_path}")


if __name__ == "__main__":
    main()
