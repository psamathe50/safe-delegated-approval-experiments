"""Generate the paper figure for topic-level gains and angular alignment.

The figure uses the pinned RewardBench 2 score archive.  It deliberately
separates individual reviewer observations from the equal-weight cardinal
aggregate, and it separates the distribution of every reviewer--prompt angle
from the promptwise closest-reviewer diagnostic.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"

CATEGORIES = ("Factuality", "Focus", "Math", "Precise IF", "Safety")
GOOD_COLOR = "#0072B2"
BAD_COLOR = "#D55E00"
TOPIC_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
TOPIC_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
TOLERANCE = 1e-12


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.weight": "normal",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_topic_distributions() -> dict[str, dict[str, np.ndarray]]:
    path = DATA / "rewardbench2_scores.npz"
    with np.load(path, allow_pickle=True) as archive:
        scores = archive["candidate_scores"].astype(float)
        offsets = archive["offsets"].astype(int)
        candidate_counts = archive["candidate_counts"].astype(int)
        num_correct = archive["num_correct"].astype(int)
        subsets = archive["subsets"].astype(str)

    standard = np.flatnonzero(candidate_counts == 4)
    if len(standard) != 1763 or not np.all(num_correct[standard] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")
    if scores.shape[0] != 48:
        raise RuntimeError("unexpected RewardBench 2 reviewer count")

    principal = np.array([1.0, 0.0, 0.0, 0.0])
    principal -= principal.mean()
    principal /= np.linalg.norm(principal)

    lists: dict[str, dict[str, list[float]]] = {
        category: {
            "reviewer_good": [],
            "reviewer_bad": [],
            "panel_good": [],
            "panel_bad": [],
            "all_angles": [],
            "best_angle": [],
        }
        for category in CATEGORIES
    }

    for prompt in standard:
        category = subsets[prompt]
        if category not in lists:
            raise RuntimeError(f"unexpected RewardBench category: {category}")
        start, stop = offsets[prompt], offsets[prompt + 1]
        prompt_scores = scores[:, start:stop]
        centered = prompt_scores - prompt_scores.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        if np.any(norms <= TOLERANCE):
            raise RuntimeError(f"constant reviewer score vector on prompt {prompt}")
        normalized = centered / norms

        target = lists[category]
        target["reviewer_good"].extend(normalized[:, 0])
        target["reviewer_bad"].extend(normalized[:, 1:].reshape(-1))
        equal_weight = normalized.mean(axis=0)
        target["panel_good"].append(float(equal_weight[0]))
        target["panel_bad"].extend(equal_weight[1:])

        cosine = np.clip(normalized @ principal, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        target["all_angles"].extend(angles)
        target["best_angle"].append(float(np.min(angles)))

    output: dict[str, dict[str, np.ndarray]] = {}
    for category, values in lists.items():
        output[category] = {
            key: np.asarray(series, dtype=float)
            for key, series in values.items()
        }
    return output


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)


def add_violin(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    width: float,
    annotate_median: bool = False,
) -> float:
    arguments = {
        "positions": [position],
        "widths": width,
        "showmeans": False,
        "showmedians": False,
        "showextrema": False,
        "points": 160,
        "bw_method": 0.18,
    }
    if "orientation" in inspect.signature(axis.violinplot).parameters:
        arguments["orientation"] = "horizontal"
    else:
        arguments["vert"] = False
    parts = axis.violinplot([values], **arguments)
    body = parts["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.24)
    body.set_linewidth(0.7)

    q10, q25, median, q75, q90 = np.quantile(
        values, (0.10, 0.25, 0.50, 0.75, 0.90)
    )
    axis.hlines(position, q10, q90, color=color, linewidth=0.65, zorder=3)
    axis.hlines(position, q25, q75, color=color, linewidth=2.2, zorder=4)
    axis.scatter(
        [median],
        [position],
        s=12,
        facecolor="white",
        edgecolor=color,
        linewidth=0.85,
        zorder=5,
    )
    if annotate_median:
        axis.annotate(
            f"{median:.1f}°",
            (median, position),
            xytext=(4, 3),
            textcoords="offset points",
            color=color,
            fontsize=6.3,
            ha="left",
            va="bottom",
        )
    return float(median)


def draw_gain_panel(
    axis: plt.Axes,
    distributions: dict[str, dict[str, np.ndarray]],
    good_key: str,
    bad_key: str,
    title: str,
) -> None:
    base_positions = np.arange(len(CATEGORIES))[::-1].astype(float)
    for category, base in zip(CATEGORIES, base_positions):
        add_violin(
            axis,
            distributions[category][good_key],
            base + 0.17,
            GOOD_COLOR,
            0.28,
        )
        add_violin(
            axis,
            distributions[category][bad_key],
            base - 0.17,
            BAD_COLOR,
            0.28,
        )

    axis.axvline(0.0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.7)
    axis.set_xlim(-np.sqrt(3 / 4), np.sqrt(3 / 4))
    axis.set_xticks((-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75))
    axis.set_ylim(-0.55, len(CATEGORIES) - 0.45)
    axis.set_yticks(base_positions)
    axis.set_yticklabels(CATEGORIES)
    axis.set_xlabel("normalized gain over the uniform fallback")
    axis.set_title(title, loc="left", pad=5)
    style_axis(axis)


def draw_angle_panel(
    axis: plt.Axes,
    distributions: dict[str, dict[str, np.ndarray]],
    key: str,
    title: str,
    x_max: float,
    ticks: tuple[float, ...],
) -> None:
    for index, category in enumerate(CATEGORIES):
        values = np.sort(distributions[category][key])
        cumulative = np.arange(1, len(values) + 1) / len(values)
        color = TOPIC_COLORS[index]
        median = float(np.median(values))
        axis.step(
            values,
            cumulative,
            where="post",
            color=color,
            linestyle=TOPIC_LINESTYLES[index],
            linewidth=1.15,
            label=category,
        )
        axis.scatter(
            [median],
            [0.5],
            s=13,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )
    if x_max > 90:
        axis.axvline(
            90.0,
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.7,
        )
    axis.set_xlim(0.0, x_max)
    axis.set_xticks(ticks)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axis.axhline(
        0.5,
        color="#888888",
        linestyle=(0, (2, 2)),
        linewidth=0.55,
    )
    axis.set_xlabel("angle to the principal (degrees; smaller is closer)")
    axis.set_ylabel("empirical fraction at or below angle")
    axis.set_title(title, loc="left", pad=5)
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=5.9,
        handlelength=2.5,
        borderaxespad=0.4,
    )
    style_axis(axis)


def make_figure(
    distributions: dict[str, dict[str, np.ndarray]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 6.05),
        gridspec_kw={"hspace": 0.43, "wspace": 0.34},
    )
    draw_gain_panel(
        axes[0, 0],
        distributions,
        "reviewer_good",
        "reviewer_bad",
        "(a) Individual reviewer endpoint gains",
    )
    draw_gain_panel(
        axes[0, 1],
        distributions,
        "panel_good",
        "panel_bad",
        "(b) Equal-weight panel-mean endpoint gains",
    )
    draw_angle_panel(
        axes[1, 0],
        distributions,
        "all_angles",
        "(c) Every reviewer-prompt angle",
        180.0,
        (0, 45, 90, 135, 180),
    )
    draw_angle_panel(
        axes[1, 1],
        distributions,
        "best_angle",
        "(d) Closest reviewer per prompt (oracle)",
        90.0,
        (0, 15, 30, 45, 60, 75, 90),
    )

    legend = (
        Patch(facecolor=GOOD_COLOR, edgecolor=GOOD_COLOR, alpha=0.30,
              label="principal-beneficial: labeled-correct endpoint"),
        Patch(facecolor=BAD_COLOR, edgecolor=BAD_COLOR, alpha=0.30,
              label="principal-harmful: labeled-incorrect endpoint"),
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    figure.subplots_adjust(top=0.93, bottom=0.08, left=0.11, right=0.985)
    return figure


def main() -> None:
    configure_matplotlib()
    PLOTS.mkdir(exist_ok=True)
    distributions = load_topic_distributions()
    figure = make_figure(distributions)
    figure.savefig(PLOTS / "rewardbench_topic_distributions.pdf")
    figure.savefig(
        PLOTS / "rewardbench_topic_distributions.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
