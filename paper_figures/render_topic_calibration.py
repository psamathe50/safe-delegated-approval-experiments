"""Render the saved, validation-selected B=17 calibration comparison.

This script only reads the canonical JSON and draws the eight stored test
operating points. It does not fit score rules, select test thresholds, or run
an experiment. Run from any directory with Python, NumPy, and Matplotlib.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "veto-review-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "veto-review-font-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper_style import GRID, INK, METHOD_COLORS, configure


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "experiments"
    / "one_step_panel"
    / "data"
    / "topic_threshold_budget_results.json"
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "generated"
    / "revised_topic_calibration"
)

COMMITTEE = METHOD_COLORS["learned_weight"]
INDIVIDUAL = METHOD_COLORS["best_single"]
FAMILIES = (
    ("global_cardinal_topic_threshold_control", "Global committee", COMMITTEE),
    ("learned_topic_cardinal", "Topic committees", COMMITTEE),
    (
        "train_selected_global_singleton_topic_threshold_control",
        "Global individual",
        INDIVIDUAL,
    ),
    ("train_selected_topic_singleton", "Topic individuals", INDIVIDUAL),
)


def stored_points() -> list[dict]:
    """Read the existing reference target; retain every matched test point."""
    data = json.loads(SOURCE.read_text())
    if data["validation_failure_budgets"]["primary_95_soundness_budget"] != 17:
        raise ValueError("The saved primary validation budget is no longer 17.")
    if data["test_labels_used_for_selection"]:
        raise ValueError("Expected selection without test labels.")
    if data["epsilon_good"] != 0.5:
        raise ValueError("Expected gain margin 0.5.")

    records = []
    for key, label, color in FAMILIES:
        family = data["families"][key]
        row = {"label": label, "color": color}
        for setting in ("shared_threshold", "topic_threshold_vector"):
            point = family[setting]["primary_95_validation_target"]
            if point["validation_failure_budget"] != 17:
                raise ValueError(f"Unexpected validation budget: {key}/{setting}")
            if point["validation"]["prompts"] != 353:
                raise ValueError("Expected 353 validation prompts.")
            if point["test"]["prompts"] != 352:
                raise ValueError("Expected 352 held-out prompts.")
            row[setting] = point["test"]
        records.append(row)
    return records


def main() -> None:
    records = stored_points()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    configure()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.85, bottom=0.18, wspace=0.25)

    panels = (
        ("sound_prompts", "Soundness (90-100%)", (90, 100), [90, 95, 100]),
        (
            "epsilon_good_complete_prompts",
            "Completeness (0-50%)",
            (0, 50),
            [0, 25, 50],
        ),
    )
    for axis, (metric, title, limits, ticks) in zip(axes, panels):
        axis.set_xlabel(title, labelpad=8)
        axis.set_xlim(*limits)
        axis.set_xticks(ticks)
        axis.set_ylim(-0.58, 3.58)
        axis.set_yticks(range(4))
        axis.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        axis.axhline(1.5, color=GRID, linewidth=0.8, zorder=1)
        axis.tick_params(axis="both", length=0, pad=6)
        for side in ("left", "right", "top"):
            axis.spines[side].set_visible(False)

        for index, record in enumerate(records):
            center = 3 - index
            color = record["color"]
            one = record["shared_threshold"][metric]
            five = record["topic_threshold_vector"][metric]
            one_x, five_x = 100 * one / 352, 100 * five / 352
            one_y, five_y = center + 0.11, center - 0.11
            axis.plot(
                [one_x, five_x],
                [one_y, five_y],
                color=color,
                alpha=0.42,
                linewidth=1.0,
                zorder=2,
            )
            axis.plot(
                one_x,
                one_y,
                marker="o",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.25,
                linestyle="none",
                zorder=3,
            )
            axis.plot(
                five_x,
                five_y,
                marker="o",
                markersize=5.2,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=3,
            )
            # Labels always sit outside the connector's vertical interval:
            # shared above its open marker, topic below its filled marker.
            # This also works when topic calibration reduces soundness.
            for (x, y, count), offset, vertical in zip(
                ((one_x, one_y, one), (five_x, five_y, five)),
                (4.5, -4.5),
                ("bottom", "top"),
            ):
                axis.annotate(
                    f"{100 * count / 352:.1f}",
                    (x, y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=vertical,
                    fontsize=9.0,
                    color=color,
                )

    axes[0].set_yticklabels([row["label"] for row in reversed(records)])
    axes[0].tick_params(axis="y", pad=11)
    axes[1].tick_params(axis="y", labelleft=False)
    separator_y = fig.transFigure.inverted().transform(
        axes[0].transData.transform((90, 1.5))
    )[1]
    fig.add_artist(Line2D(
        [0.045, 0.985], [separator_y, separator_y],
        transform=fig.transFigure, color=GRID, linewidth=0.8, zorder=0,
    ))

    legend = (
        Line2D(
            [], [], color=INK, marker="o", linestyle="none", markersize=5.2,
            markerfacecolor="white", markeredgewidth=1.2,
            label="Shared threshold",
        ),
        Line2D(
            [], [], color=INK, marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=INK, label="Topic thresholds",
        ),
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.985),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.4,
        fontsize=9.1,
    )
    fig.savefig(
        OUTPUT.with_suffix(".pdf"),
        metadata={
            "Title": "Validation-selected topic calibration on RewardBench 2",
            "Author": "",
            "Creator": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
