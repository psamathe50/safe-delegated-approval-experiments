"""Compute binary-rule results and make their paper figures.

The k-sweeps compare pure-response performance with soundness and completeness
over the full proposal domain.  All full-domain quantities are computed from
the exact hyperplane arrangement rather than sampled response lotteries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
STRONGREJECT_PDF = PLOTS / "strongreject_binary.pdf"
STRONGREJECT_PNG = PLOTS / "strongreject_binary.png"
REWARDBENCH_PDF = PLOTS / "rewardbench_binary.pdf"
REWARDBENCH_PNG = PLOTS / "rewardbench_binary.png"
REWARDBENCH_CATEGORY_PDF = PLOTS / "rewardbench_categories.pdf"
REWARDBENCH_CATEGORY_PNG = PLOTS / "rewardbench_categories.png"
REWARDBENCH_COMPLETENESS_CATEGORY_PDF = (
    PLOTS / "rewardbench_completeness_categories.pdf"
)
REWARDBENCH_COMPLETENESS_CATEGORY_PNG = (
    PLOTS / "rewardbench_completeness_categories.png"
)
BINARY_OVERALL_PDF = PLOTS / "binary_overall.pdf"
BINARY_OVERALL_PNG = PLOTS / "binary_overall.png"

INK = "#24272B"
BLUE = "#2F66A8"
ORANGE = "#D46A32"
GREEN = "#26876E"
PURPLE = "#7655A5"
RED = "#B44A55"
GRAY = "#747775"
GRID = "#DFE1E2"
TOLERANCE = 1e-8
CATEGORY_ORDER = ("Factuality", "Focus", "Math", "Precise IF", "Safety")
CATEGORY_COLORS = (BLUE, ORANGE, GREEN, PURPLE, RED)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A7AAAC",
            "axes.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", pad=6)


def strongreject_frontier() -> tuple[np.ndarray, np.ndarray]:
    """Authorization and soundness rates on the five-human-label cohort."""
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=True) as data:
        complete = data["label_count"] == 5
        human = data["human"][complete]
        scores = data["evaluator_scores"][complete]

    harmful = human > 0.5 + TOLERANCE
    acceptable = ~harmful
    objections = np.sum(scores > 0.5 + TOLERANCE, axis=1)
    k = np.arange(scores.shape[1])
    authorization = np.asarray(
        [np.mean(objections[acceptable] <= value) for value in k]
    )
    soundness = np.asarray(
        [np.mean(objections[harmful] > value) for value in k]
    )

    assert human.size == 1084
    assert int(harmful.sum()) == 190
    assert int(acceptable.sum()) == 894
    assert int(np.sum(objections[acceptable] <= 0)) == 267
    assert int(np.sum(objections[acceptable] <= 1)) == 419
    assert int(np.sum(objections[harmful] > 0)) == 190
    assert int(np.sum(objections[harmful] > 1)) == 190
    return 100 * authorization, 100 * soundness


def rewardbench_frontiers() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Category frontiers for correct authorization and endpoint safety."""
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"]
        candidate_counts = data["candidate_counts"]
        num_correct = data["num_correct"]
        categories = data["subsets"].astype(str)

    standard = np.flatnonzero(candidate_counts == 4)
    assert len(standard) == 1763
    assert np.all(num_correct[standard] == 1)

    # The unique labeled-correct response is first, followed by three labeled-
    # incorrect responses.  A reviewer objects when it scores a proposed pure
    # response strictly below the uniform four-response fallback.
    correct_objections = []
    incorrect_objections = []
    for prompt in standard:
        prompt_scores = scores[:, offsets[prompt] : offsets[prompt + 1]]
        baseline = prompt_scores.mean(axis=1)
        objections = np.sum(
            prompt_scores < baseline[:, None] - TOLERANCE,
            axis=0,
        )
        correct_objections.append(int(objections[0]))
        incorrect_objections.append(objections[1:])
    correct_objections = np.asarray(correct_objections)
    incorrect_objections = np.asarray(incorrect_objections)
    k = np.arange(scores.shape[0])
    assert int(np.sum(correct_objections <= 0)) == 73
    assert int(np.sum(correct_objections <= 1)) == 208
    assert incorrect_objections.shape == (1763, 3)
    assert int(np.sum(incorrect_objections > 0)) == 5279
    assert int(np.sum(incorrect_objections > 1)) == 5243
    assert int(np.sum(np.all(incorrect_objections > 0, axis=1))) == 1754
    assert int(np.sum(np.all(incorrect_objections > 1, axis=1))) == 1718
    standard_categories = categories[standard]
    frontiers: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "All prompts": (
            100
            * np.asarray(
                [np.mean(correct_objections <= value) for value in k]
            ),
            100
            * np.asarray(
                [
                    np.mean(np.all(incorrect_objections > value, axis=1))
                    for value in k
                ]
            ),
        )
    }
    for category in CATEGORY_ORDER:
        selected = standard_categories == category
        authorization = np.asarray(
            [np.mean(correct_objections[selected] <= value) for value in k]
        )
        rejection = np.asarray(
            [
                np.mean(
                    np.all(incorrect_objections[selected] > value, axis=1)
                )
                for value in k
            ]
        )
        frontiers[category] = (100 * authorization, 100 * rejection)

    expected_sizes = {
        "Factuality": 475,
        "Focus": 495,
        "Math": 183,
        "Precise IF": 160,
        "Safety": 450,
    }
    assert {
        category: int(np.sum(standard_categories == category))
        for category in CATEGORY_ORDER
    } == expected_sizes
    return frontiers


def rewardbench_depth_survival(
    n_thresholds: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Full-domain soundness and completeness for every count threshold."""
    with np.load(
        DATA / "rewardbench2_depth.npz", allow_pickle=True
    ) as data:
        standard = data["candidate_counts"] == 4
        depth = data["depth"][standard].astype(int)
        maximum_beneficial_objections = data[
            "maximum_beneficial_objections"
        ][standard].astype(int)
        exact = data["depth_exact"][standard].astype(bool)
        categories = data["subsets"][standard].astype(str)

    assert depth.size == 1763
    assert np.all(exact)
    assert int(np.sum(depth >= 1)) == 1590
    assert int(np.sum(depth >= 2)) == 1361
    assert int(np.sum(depth >= 3)) == 1139
    assert int(np.sum(depth >= 6)) == 669

    # A nontrivial count rule can tolerate k=0,...,N-1 objections.  Plot the
    # complete range rather than truncating the curves at a visually convenient
    # value of k.
    k = np.arange(n_thresholds)
    survival: dict[str, np.ndarray] = {
        "All prompts": 100 * np.asarray([np.mean(depth >= value + 1) for value in k])
    }
    completeness: dict[str, np.ndarray] = {
        "All prompts": 100
        * np.asarray(
            [
                np.mean(maximum_beneficial_objections <= value)
                for value in k
            ]
        )
    }
    expected_sizes = {
        "Factuality": 475,
        "Focus": 495,
        "Math": 183,
        "Precise IF": 160,
        "Safety": 450,
    }
    for category in CATEGORY_ORDER:
        selected = categories == category
        assert int(np.sum(selected)) == expected_sizes[category]
        survival[category] = 100 * np.asarray(
            [np.mean(depth[selected] >= value + 1) for value in k]
        )
        completeness[category] = 100 * np.asarray(
            [
                np.mean(maximum_beneficial_objections[selected] <= value)
                for value in k
            ]
        )
    return k, survival, completeness


def format_k_axis(
    axis: plt.Axes, *, xmax: int, xticks: list[int], ylabel: str
) -> None:
    axis.set_xlim(0, xmax)
    axis.set_ylim(0, 103)
    axis.set_xticks(xticks)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.65, color="#8F9293")
    axis.set_xlabel(r"objections tolerated $k$")
    axis.set_ylabel(ylabel)


def save_split_figure(
    figure: plt.Figure,
    pdf_path: Path,
    png_path: Path,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    print(pdf_path)
    print(png_path)


def make_split_figures(
    strong_authorization: np.ndarray,
    strong_soundness: np.ndarray,
    rewardbench_curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save the three empirical comparisons as independent paper figures."""
    n_rb_thresholds = rewardbench_curves["All prompts"][0].size
    rb_k, rb_survival, rb_completeness = rewardbench_depth_survival(
        n_rb_thresholds
    )

    figure, axis = plt.subplots(figsize=(4.65, 3.0))
    strong_k = np.arange(strong_soundness.size)
    axis.plot(
        strong_k,
        strong_authorization,
        color=BLUE,
        linewidth=1.55,
        marker="o",
        markersize=3.2,
        label="Pure-response completeness (= full-domain completeness)",
    )
    axis.plot(
        strong_k,
        strong_soundness,
        color=RED,
        linewidth=1.55,
        linestyle="--",
        marker="o",
        markersize=3.2,
        label="Full-domain soundness",
    )
    format_k_axis(
        axis,
        xmax=int(strong_k[-1]),
        xticks=list(strong_k),
        ylabel="fraction of responses",
    )
    axis.legend(frameon=False, fontsize=7.2, loc="center right")
    figure.tight_layout()
    save_split_figure(figure, STRONGREJECT_PDF, STRONGREJECT_PNG)

    figure, axis = plt.subplots(figsize=(5.1, 3.1))
    authorization, pure_soundness = rewardbench_curves["All prompts"]
    axis.plot(
        rb_k,
        authorization,
        color=BLUE,
        linewidth=1.45,
        marker="o",
        markersize=2.2,
        label="Pure-response completeness",
        zorder=3,
    )
    axis.plot(
        rb_k,
        rb_completeness["All prompts"],
        color=GREEN,
        linewidth=1.45,
        linestyle="-.",
        marker="s",
        markersize=2.1,
        label="Full-domain completeness",
        zorder=3,
    )
    axis.plot(
        rb_k,
        rb_survival["All prompts"],
        color=RED,
        linewidth=1.45,
        linestyle="--",
        marker="o",
        markersize=2.2,
        label="Full-domain soundness",
        zorder=3,
    )
    axis.plot(
        rb_k,
        pure_soundness,
        color=GRAY,
        linewidth=1.35,
        linestyle=":",
        marker="o",
        markersize=2.0,
        label="Pure-response soundness",
        zorder=2,
    )
    format_k_axis(
        axis,
        xmax=int(rb_k[-1]),
        xticks=[0, 10, 20, 30, 40, int(rb_k[-1])],
        ylabel="fraction of prompts",
    )
    axis.legend(frameon=False, fontsize=7.1, loc="center right")
    figure.tight_layout()
    save_split_figure(figure, REWARDBENCH_PDF, REWARDBENCH_PNG)

    figure, axis = plt.subplots(figsize=(6.0, 3.45))
    selected_k = np.asarray([0, 5, 10, 20, 30, int(rb_k[-1])])
    axis.plot(
        pure_soundness,
        authorization,
        color=INK,
        linewidth=1.55,
        linestyle="--",
        marker="o",
        markevery=selected_k,
        markersize=3.0,
        label="All prompts",
        zorder=3,
    )
    for category, color in zip(CATEGORY_ORDER, CATEGORY_COLORS):
        category_authorization, category_soundness = rewardbench_curves[category]
        axis.plot(
            category_soundness,
            category_authorization,
            color=color,
            linewidth=1.25,
            marker="o",
            markevery=selected_k,
            markersize=2.6,
            label=category,
            zorder=2,
        )
    label_offsets = {
        0: (-26, 5),
        5: (4, -10),
        10: (4, -10),
        20: (4, -10),
        30: (4, -10),
        int(rb_k[-1]): (4, -11),
    }
    for value in selected_k:
        axis.annotate(
            rf"$k={value}$",
            (pure_soundness[value], authorization[value]),
            xytext=label_offsets[int(value)],
            textcoords="offset points",
            fontsize=6.4,
            color=INK,
        )
    axis.set_xlim(-1, 101)
    axis.set_ylim(0, 103)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.65, color="#8F9293")
    axis.set_xlabel("pure-response soundness")
    axis.set_ylabel("pure-response completeness")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.9,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.81))
    save_split_figure(
        figure,
        REWARDBENCH_CATEGORY_PDF,
        REWARDBENCH_CATEGORY_PNG,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    for axis, values, title in (
        (axes[0], rb_survival, r"$\bf{(a)}$ Full-domain soundness"),
        (axes[1], rb_completeness, r"$\bf{(b)}$ Full-domain completeness"),
    ):
        axis.plot(
            rb_k,
            values["All prompts"],
            color=INK,
            linewidth=1.55,
            linestyle="--",
            marker="o",
            markevery=5,
            markersize=3.0,
            label="All prompts",
            zorder=3,
        )
        for category, color in zip(CATEGORY_ORDER, CATEGORY_COLORS):
            axis.plot(
                rb_k,
                values[category],
                color=color,
                linewidth=1.2,
                marker="o",
                markevery=5,
                markersize=2.6,
                label=category,
                zorder=2,
            )
        axis.set_xlim(0, int(rb_k[-1]))
        axis.set_ylim(0, 103)
        axis.set_xticks([0, 10, 20, 30, 40, int(rb_k[-1])])
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")
        axis.set_xlabel(r"objections tolerated $k$")
        axis.set_title(title, loc="left", pad=6)
    axes[0].set_ylabel("fraction of prompts")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.9,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82), w_pad=2.2)
    save_split_figure(
        figure,
        REWARDBENCH_COMPLETENESS_CATEGORY_PDF,
        REWARDBENCH_COMPLETENESS_CATEGORY_PNG,
    )


def make_overall_binary_figure(
    strong_authorization: np.ndarray,
    strong_soundness: np.ndarray,
    rewardbench_curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Place the two directly comparable overall k-sweeps side by side."""
    n_rb_thresholds = rewardbench_curves["All prompts"][0].size
    rb_k, rb_survival, rb_completeness = rewardbench_depth_survival(
        n_rb_thresholds
    )
    strong_k = np.arange(strong_soundness.size)
    rb_authorization, rb_pure_soundness = rewardbench_curves["All prompts"]

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    axes[0].plot(
        strong_k,
        strong_authorization,
        color=GREEN,
        linewidth=1.45,
        linestyle="-.",
        label="Full-domain completeness",
    )
    axes[0].plot(
        strong_k,
        strong_authorization,
        color=BLUE,
        linewidth=1.55,
        marker="o",
        markersize=3.2,
        label="Pure-response completeness",
    )
    axes[0].plot(
        strong_k,
        strong_soundness,
        color=RED,
        linewidth=1.55,
        linestyle="--",
        marker="o",
        markersize=3.2,
        label="Full-domain soundness",
    )
    format_k_axis(
        axes[0],
        xmax=int(strong_k[-1]),
        xticks=list(strong_k),
        ylabel="fraction of responses",
    )
    panel_title(axes[0], "a", "StrongREJECT")

    axes[1].plot(
        rb_k,
        rb_authorization,
        color=BLUE,
        linewidth=1.45,
        marker="o",
        markersize=2.1,
        label="Pure-response completeness",
        zorder=3,
    )
    axes[1].plot(
        rb_k,
        rb_completeness["All prompts"],
        color=GREEN,
        linewidth=1.45,
        linestyle="-.",
        marker="s",
        markersize=2.1,
        label="Full-domain completeness",
        zorder=3,
    )
    axes[1].plot(
        rb_k,
        rb_survival["All prompts"],
        color=RED,
        linewidth=1.45,
        linestyle="--",
        marker="o",
        markersize=2.1,
        label="Full-domain soundness",
        zorder=3,
    )
    axes[1].plot(
        rb_k,
        rb_pure_soundness,
        color=GRAY,
        linewidth=1.35,
        linestyle=":",
        marker="o",
        markersize=1.9,
        label="Pure-response soundness",
        zorder=2,
    )
    format_k_axis(
        axes[1],
        xmax=int(rb_k[-1]),
        xticks=[0, 10, 20, 30, 40, int(rb_k[-1])],
        ylabel="fraction of prompts",
    )
    panel_title(axes[1], "b", "RewardBench 2")

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=1.9,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2.2)
    save_split_figure(figure, BINARY_OVERALL_PDF, BINARY_OVERALL_PNG)


def main() -> None:
    configure_style()
    strong_authorization, strong_soundness = strongreject_frontier()
    assert np.isclose(strong_authorization[0], 100 * 267 / 894)
    assert np.isclose(strong_soundness[0], 100.0)

    category_frontiers = rewardbench_frontiers()
    make_split_figures(
        strong_authorization,
        strong_soundness,
        category_frontiers,
    )
    make_overall_binary_figure(
        strong_authorization,
        strong_soundness,
        category_frontiers,
    )


if __name__ == "__main__":
    main()
