"""Create a first-pass three-panel demo figure from saved recurrence outputs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


DATA_DIR = (
    r"C:\Users\Jeff\Documents\CBAS_Analysis_Data\Cohort_Data"
    r"\Full_Cycle_Phase_Overlap_Recurrence"
)
OUTPUT_DIR = rf"{DATA_DIR}\Demo_Figure"
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
BEHAVIOR_LABELS = {behavior: behavior.capitalize() for behavior in BEHAVIORS}
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
COHORT_ORDER = tuple(GENOTYPE_BY_ANIMAL)
N_CYCLES = 4
N_BINS = 288
CT_HOURS = 24.0
LOW_COLOR = "#D55E00"
HIGH_COLOR = "#0072B2"
NEUTRAL_COLOR = "#777777"


def _load_saved_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the accepted score table and its exact saved profiles."""
    recurrence_path = rf"{DATA_DIR}\full_cycle_behavior_recurrence.csv"
    profiles_path = rf"{DATA_DIR}\full_cycle_phase_profiles.csv"
    recurrence = pd.read_csv(recurrence_path)
    profiles = pd.read_csv(profiles_path)

    recurrence_columns = {"animal", "behavior", "mean_recurrence"}
    profile_columns = {
        "animal",
        "behavior",
        "cycle",
        "ct_bin_center_h",
        "probability",
    }
    if not recurrence_columns.issubset(recurrence.columns):
        raise ValueError(f"Missing recurrence columns: {recurrence_columns - set(recurrence.columns)}")
    if not profile_columns.issubset(profiles.columns):
        raise ValueError(f"Missing profile columns: {profile_columns - set(profiles.columns)}")
    if set(recurrence["animal"]) != set(COHORT_ORDER):
        raise ValueError("The saved recurrence table does not contain the expected 8 animals")
    if set(profiles["animal"]) != set(COHORT_ORDER):
        raise ValueError("The saved profile table does not contain the expected 8 animals")
    if set(recurrence["behavior"]) != set(BEHAVIORS):
        raise ValueError("The saved recurrence table has an unexpected behavior set")
    if set(profiles["behavior"]) != set(BEHAVIORS):
        raise ValueError("The saved profile table has an unexpected behavior set")
    if recurrence.duplicated(["animal", "behavior"]).any():
        raise ValueError("The saved recurrence table contains duplicate animal/behavior rows")
    if recurrence.groupby("animal").size().ne(len(BEHAVIORS)).any():
        raise ValueError("Each animal must have exactly eight behavior recurrence values")

    scores = recurrence["mean_recurrence"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Behavior recurrence values must be finite and within [0, 1]")

    expected_centers = (np.arange(N_BINS, dtype=float) + 0.5) * CT_HOURS / N_BINS
    if set(profiles["cycle"].astype(int)) != set(range(1, N_CYCLES + 1)):
        raise ValueError("The saved profiles must contain cycles 1 through 4")
    expected_groups = len(COHORT_ORDER) * len(BEHAVIORS) * N_CYCLES
    groups = profiles.groupby(["animal", "behavior", "cycle"], sort=False)
    if len(groups) != expected_groups:
        raise ValueError("The saved profiles do not contain every animal/behavior/cycle")
    for identity, group in groups:
        ordered = group.sort_values("ct_bin_center_h")
        if len(ordered) != N_BINS:
            raise ValueError(f"Expected {N_BINS} saved bins for profile {identity}")
        centers = ordered["ct_bin_center_h"].to_numpy(dtype=float)
        values = ordered["probability"].to_numpy(dtype=float)
        if not np.allclose(centers, expected_centers, rtol=0.0, atol=1e-10):
            raise ValueError(f"Unexpected CT-bin centers for saved profile {identity}")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"Invalid saved probability values for profile {identity}")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Saved scored profile does not sum to one: {identity}")

    return recurrence, profiles


def _animal_order_and_representatives(
    recurrence: pd.DataFrame,
) -> tuple[list[str], dict[str, float], str, str]:
    animal_means = recurrence.groupby("animal", sort=False)["mean_recurrence"].mean().to_dict()
    cohort_rank = {animal: index for index, animal in enumerate(COHORT_ORDER)}
    panel_order: list[str] = []
    for genotype in ("LacZ", "Bmal1KO"):
        members = [
            animal
            for animal in COHORT_ORDER
            if GENOTYPE_BY_ANIMAL[animal] == genotype
        ]
        panel_order.extend(
            sorted(members, key=lambda animal: (animal_means[animal], cohort_rank[animal]))
        )

    lowest = min(panel_order, key=lambda animal: (animal_means[animal], cohort_rank[animal]))
    highest = max(panel_order, key=lambda animal: (animal_means[animal], -cohort_rank[animal]))
    return panel_order, animal_means, lowest, highest


def _profile_matrix(profiles: pd.DataFrame, animal: str) -> np.ndarray:
    rows = []
    for behavior in BEHAVIORS:
        for cycle in range(1, N_CYCLES + 1):
            selected = profiles.loc[
                (profiles["animal"] == animal)
                & (profiles["behavior"] == behavior)
                & (profiles["cycle"].astype(int) == cycle)
            ].sort_values("ct_bin_center_h")
            rows.append(selected["probability"].to_numpy(dtype=float))
    return np.vstack(rows)


def _format_animal_tick(animal: str) -> str:
    return f"{animal}\n{GENOTYPE_BY_ANIMAL[animal]}"


def _create_figure(
    recurrence: pd.DataFrame,
    profiles: pd.DataFrame,
    animal_order: list[str],
    animal_means: dict[str, float],
    lowest: str,
    highest: str,
) -> tuple[plt.Figure, float, float]:
    recurrence_lookup = recurrence.set_index(["animal", "behavior"])["mean_recurrence"]
    matrix = recurrence.pivot(
        index="behavior", columns="animal", values="mean_recurrence"
    ).reindex(index=BEHAVIORS, columns=animal_order)
    if matrix.isna().any().any():
        raise ValueError("Could not assemble the complete behavior-by-animal recurrence matrix")

    low_profile = _profile_matrix(profiles, lowest)
    high_profile = _profile_matrix(profiles, highest)
    displayed_profiles = np.concatenate((low_profile.ravel(), high_profile.ravel()))
    profile_min = float(displayed_profiles.min())
    profile_max = float(displayed_profiles.max())
    if profile_max <= 0.0:
        raise ValueError("Displayed saved profiles have no positive probability values")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(17, 14.5))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.82, 1.72),
        left=0.065,
        right=0.955,
        top=0.94,
        bottom=0.045,
        hspace=0.12,
    )
    top = outer[0].subgridspec(
        1, 3, width_ratios=(0.93, 1.22, 0.035), wspace=0.29
    )
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    cax_b = fig.add_subplot(top[0, 2])

    fig.suptitle(
        "Cycle-to-cycle recurrence of circadian behavior timing",
        fontsize=15,
        fontweight="medium",
        y=0.982,
    )

    # Panel A: one individual dot per animal, grouped and ordered by genotype.
    positions = {animal: index for index, animal in enumerate((animal_order[:4]))}
    positions.update({animal: index + 5 for index, animal in enumerate(animal_order[4:])})
    for animal in animal_order:
        color = LOW_COLOR if animal == lowest else HIGH_COLOR if animal == highest else NEUTRAL_COLOR
        size = 80 if animal in (lowest, highest) else 48
        ax_a.scatter(
            positions[animal],
            animal_means[animal],
            s=size,
            color=color,
            edgecolor="white" if animal not in (lowest, highest) else "black",
            linewidth=0.8 if animal in (lowest, highest) else 0.4,
            zorder=3,
        )
    ax_a.axvline(4.0, color="#c7c7c7", linewidth=0.9, zorder=1)
    ax_a.set_xlim(-0.65, 8.65)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_a.set_ylabel("Mean recurrence across 8 behaviors")
    ax_a.set_xticks([positions[animal] for animal in animal_order])
    ax_a.set_xticklabels([_format_animal_tick(animal) for animal in animal_order])
    ax_a.tick_params(axis="x", length=0, pad=4)
    ax_a.tick_params(axis="y", length=0)
    ax_a.grid(axis="y", color="#e7e7e7", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.text(-0.12, 1.06, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")
    ax_a.text(
        0.0,
        1.06,
        "Mean cycle-to-cycle recurrence",
        transform=ax_a.transAxes,
        fontsize=10,
        va="center",
    )
    ax_a.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="", color=LOW_COLOR, markersize=6,
                   markeredgecolor="black", label="Lowest"),
            Line2D([], [], marker="o", linestyle="", color=HIGH_COLOR, markersize=6,
                   markeredgecolor="black", label="Highest"),
        ),
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=7.5,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.2,
    )

    # Panel B: fixed 0-1 sequential recurrence scale and matching animal order.
    image_b = ax_b.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_b.set_yticks(np.arange(len(BEHAVIORS)))
    ax_b.set_yticklabels([BEHAVIOR_LABELS[behavior] for behavior in BEHAVIORS])
    ax_b.set_xticks(np.arange(len(animal_order)))
    ax_b.set_xticklabels([_format_animal_tick(animal) for animal in animal_order])
    ax_b.tick_params(axis="both", length=0, pad=4)
    ax_b.axvline(3.5, color="white", linewidth=1.4)
    for animal, color in ((lowest, LOW_COLOR), (highest, HIGH_COLOR)):
        column = animal_order.index(animal)
        ax_b.add_patch(
            Rectangle(
                (column - 0.5, -0.5),
                1.0,
                len(BEHAVIORS),
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                clip_on=False,
            )
        )
    ax_b.text(-0.10, 1.06, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    ax_b.text(
        0.0,
        1.06,
        "Behavior × animal recurrence",
        transform=ax_b.transAxes,
        fontsize=10,
        va="center",
    )
    cbar_b = fig.colorbar(image_b, cax=cax_b, ticks=np.linspace(0.0, 1.0, 5))
    cbar_b.set_label("Recurrence", fontsize=8)
    cbar_b.ax.tick_params(labelsize=7, length=2, pad=2)

    # Panel C: exact saved 288-bin scored vectors, stacked by behavior and cycle.
    panel_c = outer[1].subgridspec(
        3,
        5,
        height_ratios=(0.105, 0.09, 1.0),
        width_ratios=(1.55, 5.25, 1.55, 5.25, 0.34),
        wspace=0.035,
        hspace=0.015,
    )
    title_c = fig.add_subplot(panel_c[0, :4])
    title_c.axis("off")
    title_c.text(
        0.0,
        0.55,
        "C   Representative cycle profiles from the saved scored probabilities",
        fontsize=10.5,
        fontweight="medium",
        ha="left",
        va="center",
    )
    title_c.text(
        1.0,
        0.55,
        "Behavior labels are followed by recurrence; rows 1–4 are cycles top to bottom",
        fontsize=7.8,
        color="#555555",
        ha="right",
        va="center",
    )

    header_low = fig.add_subplot(panel_c[1, 0:2])
    header_high = fig.add_subplot(panel_c[1, 2:4])
    for header_ax, animal, label, color in (
        (header_low, lowest, "LOWEST", LOW_COLOR),
        (header_high, highest, "HIGHEST", HIGH_COLOR),
    ):
        header_ax.axis("off")
        header_ax.text(
            0.5,
            0.74,
            f"{label}  ·  {animal} ({GENOTYPE_BY_ANIMAL[animal]})",
            color=color,
            fontsize=10,
            fontweight="semibold",
            ha="center",
            va="center",
        )
        header_ax.text(
            0.5,
            0.18,
            f"Mean recurrence = {animal_means[animal]:.3f}",
            color="#444444",
            fontsize=8.5,
            ha="center",
            va="center",
        )

    label_low = fig.add_subplot(panel_c[2, 0])
    ax_profile_low = fig.add_subplot(panel_c[2, 1])
    label_high = fig.add_subplot(panel_c[2, 2])
    ax_profile_high = fig.add_subplot(panel_c[2, 3], sharex=ax_profile_low, sharey=ax_profile_low)
    cax_c = fig.add_subplot(panel_c[2, 4])

    for label_ax, animal in ((label_low, lowest), (label_high, highest)):
        label_ax.set_xlim(0.0, 1.0)
        label_ax.set_ylim(32.0, 0.0)
        label_ax.axis("off")
        for behavior_index, behavior in enumerate(BEHAVIORS):
            score = float(recurrence_lookup.loc[(animal, behavior)])
            center = behavior_index * N_CYCLES + N_CYCLES / 2.0
            label_ax.text(
                1.0,
                center,
                f"{BEHAVIOR_LABELS[behavior]}\n{score:.2f}",
                ha="right",
                va="center",
                fontsize=8.1,
                linespacing=1.2,
                color="#222222",
            )
            if behavior_index:
                label_ax.axhline(behavior_index * N_CYCLES, color="#b9b9b9", linewidth=0.65)

    image_c = None
    for profile_ax, values in ((ax_profile_low, low_profile), (ax_profile_high, high_profile)):
        image_c = profile_ax.imshow(
            values,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, CT_HOURS, values.shape[0], 0.0),
            cmap="magma",
            vmin=0.0,
            vmax=profile_max,
        )
        profile_ax.set_xlim(0.0, CT_HOURS)
        profile_ax.set_ylim(values.shape[0], 0.0)
        profile_ax.set_xticks((0, 6, 12, 18, 24))
        profile_ax.set_xlabel("Circadian time (CT, h)", labelpad=4)
        profile_ax.set_yticks(np.arange(32, dtype=float) + 0.5)
        profile_ax.set_yticklabels([str(cycle) for _ in BEHAVIORS for cycle in range(1, 5)])
        profile_ax.tick_params(axis="x", length=2, pad=3, labelsize=8)
        profile_ax.tick_params(axis="y", length=0, pad=2, labelsize=6.5, colors="#555555")
        for row_boundary in range(1, values.shape[0]):
            is_behavior_boundary = row_boundary % N_CYCLES == 0
            profile_ax.axhline(
                row_boundary,
                color="white",
                linewidth=0.75 if is_behavior_boundary else 0.35,
                alpha=0.9,
            )

    assert image_c is not None
    cbar_c = fig.colorbar(image_c, cax=cax_c, ticks=np.linspace(0.0, profile_max, 4))
    cbar_c.set_label("Probability per 5-min bin", fontsize=8.5, labelpad=8)
    cbar_c.ax.tick_params(labelsize=7, length=2, pad=2)
    cbar_c.ax.set_yticklabels([f"{tick:.3f}" for tick in np.linspace(0.0, profile_max, 4)])

    return fig, profile_min, profile_max


def main() -> None:
    recurrence, profiles = _load_saved_outputs()
    animal_order, animal_means, lowest, highest = _animal_order_and_representatives(recurrence)
    figure, profile_min, profile_max = _create_figure(
        recurrence, profiles, animal_order, animal_means, lowest, highest
    )

    from pathlib import Path

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "cycle_recurrence_demo.png"
    pdf_path = output_dir / "cycle_recurrence_demo.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("ANIMAL-LEVEL MEAN RECURRENCE (equal-weight mean of 8 behavior scores)")
    for animal in COHORT_ORDER:
        print(f"  {animal} ({GENOTYPE_BY_ANIMAL[animal]}): {animal_means[animal]:.6f}")
    print(f"Panel A/B animal order: {', '.join(animal_order)}")
    print(
        f"Lowest representative: {lowest} ({GENOTYPE_BY_ANIMAL[lowest]}), "
        f"mean recurrence={animal_means[lowest]:.6f}"
    )
    print(
        f"Highest representative: {highest} ({GENOTYPE_BY_ANIMAL[highest]}), "
        f"mean recurrence={animal_means[highest]:.6f}"
    )
    for animal in (lowest, highest):
        print(f"Behavior recurrence values for {animal}:")
        animal_scores = recurrence.loc[recurrence["animal"] == animal].set_index("behavior")
        for behavior in BEHAVIORS:
            print(f"  {behavior}: {animal_scores.loc[behavior, 'mean_recurrence']:.6f}")
    print(
        "Panel C displayed probability range: "
        f"minimum={profile_min:.9f}, maximum={profile_max:.9f}; scale=[0, {profile_max:.9f}]"
    )
    print("Panel C source: exact probability vectors from full_cycle_phase_profiles.csv.")
    print("No row-specific normalization, smoothing, binning, alignment, or rescaling was applied.")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
