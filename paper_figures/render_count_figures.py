"""Render count-rule figures from archived scores and saved exact depths.

No fits, geometric audits, model calls, or canonical-repository writes are made.
The canonical read/aggregation helpers preserve their raw-score tie convention.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "veto-review-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "veto-review-font-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_style import configure, GREEN, OCHRE, GRID, TOPIC_COLORS, CURVE_WIDTH, MAIN_CURVE_WIDTH

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "experiments" / "one_step_panel"
DATA = CANONICAL / "data"
OUT = Path(__file__).resolve().parent / "generated"
TOPICS = tuple(TOPIC_COLORS)
# Distinct cool/warm hues make the two soundness curves easy to distinguish.
# Keep this palette local so recoloring this figure does not alter other plots.
DETERMINISTIC_SOUNDNESS = "#0072B2"
RANDOMIZED_SOUNDNESS = "#B2182B"
# Sparse dash patterns distinguish overlapping topic curves in grayscale.
TOPIC_STYLES = {
    "All prompts": "-", "Factuality": (0, (6, 2.5)),
    "Focus": (0, (3.5, 2)), "Math": (0, (7, 2, 2, 2)),
    "Precise IF": (0, (10, 2.5)), "Safety": (0, (7, 2, 2, 2, 2, 2)),
}
SIZES = {"All prompts": 1763, "Factuality": 475, "Focus": 495, "Math": 183, "Precise IF": 160, "Safety": 450}
PINNED = {
    "rewardbench2_scores.npz": "2b6f6cfa2c6cace3d85e1bc96b5a8c61390b3b91a2bb73f67bd6e89849ed0d7c",
    "strongreject_scores.npz": "e462848048763dc24640535baa2ce2bf9ed9834535faf323c2b73b8ea5396d48",
}


def load_curves():
    for name, expected in PINNED.items():
        observed = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Pinned archive changed: {name}")
    spec = importlib.util.spec_from_file_location("canonical_binary_read_helpers", CANONICAL / "binary_analysis.py")
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    sr_complete, sr_sound = canonical.strongreject_frontier()
    pure = canonical.rewardbench_frontiers()
    k, full_sound, full_complete = canonical.rewardbench_depth_survival(48)

    # Compare every full-domain curve with independently stored per-prompt
    # bounded-domain errors, rather than recomputing any arrangement or LP.
    with np.load(DATA / "approximate_results.npz", allow_pickle=False) as saved:
        categories = saved["categories"]
        error_sound = saved["rewardbench_soundness_error"]
        error_complete = saved["rewardbench_completeness_error"]
        attained = saved["rewardbench_completeness_supremum_attained"]
        assert error_sound.shape == error_complete.shape == attained.shape == (1763, 48)
        sound = error_sound <= 5e-8
        complete = np.isclose(error_complete, 0.0, atol=5e-8, rtol=0.0) & ~attained
        for topic in TOPICS:
            mask = np.ones(1763, dtype=bool) if topic == "All prompts" else categories == topic
            assert int(mask.sum()) == SIZES[topic]
            np.testing.assert_allclose(full_sound[topic], 100 * sound[mask].mean(axis=0), atol=1e-10, rtol=0)
            np.testing.assert_allclose(full_complete[topic], 100 * complete[mask].mean(axis=0), atol=1e-10, rtol=0)
        sr_objections = saved["strongreject_objections"]
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=False) as saved:
        human = saved["human"][saved["label_count"] == 5]
    harmful = human > 0.5 + 1e-8
    assert len(human) == 1084 and int(harmful.sum()) == 190
    for i in range(9):
        np.testing.assert_allclose(sr_sound[i], 100 * np.mean(sr_objections[harmful] > i), atol=1e-10, rtol=0)
        np.testing.assert_allclose(sr_complete[i], 100 * np.mean(sr_objections[~harmful] <= i), atol=1e-10, rtol=0)
    for topic in TOPICS:
        comp, snd = pure[topic]
        assert len(comp) == len(snd) == 48
        assert np.all(np.diff(comp) >= -1e-10) and np.all(np.diff(snd) <= 1e-10)
        assert np.all(np.diff(full_complete[topic]) >= -1e-10)
        assert np.all(np.diff(full_sound[topic]) <= 1e-10)
    print(json.dumps({
        "source": str(DATA.relative_to(ROOT)), "pinned_archives_verified": True,
        "full_curves_checked_against_saved_errors": "all 48 thresholds, all 6 cohorts",
        "strongreject": {"responses": 1084, "harmful": 190, "acceptable": 894,
            "k1_harmful_rejected": round(sr_sound[1] * 190 / 100),
            "k1_acceptable_released": round(sr_complete[1] * 894 / 100),
            "k6_harmful_rejected": round(sr_sound[6] * 190 / 100),
            "k6_acceptable_released": round(sr_complete[6] * 894 / 100)},
        "rewardbench2": {"prompts": 1763, "k0_pure_sound": round(pure["All prompts"][1][0] * 1763 / 100),
            "k0_pure_complete": round(pure["All prompts"][0][0] * 1763 / 100),
            "k0_full_sound": round(full_sound["All prompts"][0] * 1763 / 100),
            "k1_full_sound": round(full_sound["All prompts"][1] * 1763 / 100),
            "k24_full_complete": round(full_complete["All prompts"][24] * 1763 / 100)},
    }, indent=2))
    return sr_complete, sr_sound, pure, k, full_sound, full_complete


def axis_style(ax, xlabel, ylabel=None, xmax=47):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6, pad=5)
    ax.set_ylim(-1.5, 102.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, xmax)
    ax.set_xticks(list(range(9)) if xmax == 8 else [0, 10, 20, 30, 40, 47])
    ax.set_xlabel(xlabel, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=7)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = fig.bbox
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < extent.x0 - 1 or box.y0 < extent.y0 - 1 or box.x1 > extent.x1 + 1 or box.y1 > extent.y1 + 1):
            raise RuntimeError(f"Clipped figure text in {stem}: {text.get_text()}")
    fig.savefig(OUT / f"{stem}.pdf", metadata={"Title": stem.replace("_", " "), "Author": "", "Creator": "Matplotlib"})
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print((OUT / f"{stem}.pdf").relative_to(ROOT))


def plot_count(sr_complete, sr_sound, pure, k, full_sound, full_complete):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.095, right=.985, bottom=.18, top=.735, wspace=.35)
    sr_k = np.arange(9)
    # Every curve retains every evaluated integer threshold. Crosses mark all
    # nine StrongREJECT settings, and every fifth plus the last RewardBench k.
    count_style = dict(lw=MAIN_CURVE_WIDTH, linestyle="-", marker="x", markersize=3.4, markeredgewidth=.85, clip_on=False)
    axes[0].plot(sr_k, sr_sound, color=DETERMINISTIC_SOUNDNESS, **count_style)
    axes[0].plot(sr_k, sr_complete, color=OCHRE, **count_style)
    axis_style(axes[0], r"Objections tolerated $k$", "Conditional response rate (%)", xmax=8)
    axes[0].set_title("(a) StrongREJECT", loc="left", pad=10)
    values = (pure["All prompts"][1], pure["All prompts"][0], full_sound["All prompts"], full_complete["All prompts"])
    colors = (DETERMINISTIC_SOUNDNESS, OCHRE, RANDOMIZED_SOUNDNESS, GREEN)
    labels = ("Deterministic soundness", "Deterministic completeness", "Randomized soundness", "Randomized completeness")
    marker_indices = sorted(set(range(0, len(k), 5)) | {len(k) - 1})
    for y, color in zip(values, colors):
        axes[1].plot(k, y, color=color, markevery=marker_indices, **count_style)
    axis_style(axes[1], r"Objections tolerated $k$", "Fraction of prompts (%)")
    axes[1].set_title("(b) RewardBench 2", loc="left", pad=10)
    # Small horizontal padding keeps the crosses at k=0 and the largest k
    # inside the plotting area without changing any displayed observations.
    axes[0].set_xlim(-.12, 8.12)
    axes[1].set_xlim(0, 50)
    axes[1].set_xticks([0, 10, 20, 30, 40, 50])
    handles = [Line2D([], [], color=c, label=label, **count_style) for c, label in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .99), frameon=False, ncol=2, handlelength=2.1, columnspacing=1.5, labelspacing=.6)
    save(fig, "revised_count_rules")


def topic_legend(fig):
    handles = [Line2D([], [], color=TOPIC_COLORS[topic], lw=CURVE_WIDTH, ls=TOPIC_STYLES[topic], label=topic) for topic in TOPICS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .99), frameon=False, ncol=3, handlelength=2.1, columnspacing=1.8, labelspacing=.6)


def plot_pure_topics(pure):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.76)
    for topic in TOPICS[::-1]:
        complete, sound = pure[topic]
        ax.plot(sound, complete, color=TOPIC_COLORS[topic], lw=CURVE_WIDTH, ls=TOPIC_STYLES[topic])
    axis_style(ax, "Deterministic soundness (%)", "Deterministic completeness (%)", xmax=100)
    ax.set_xlim(-1.5, 101.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    topic_legend(fig)
    save(fig, "revised_rewardbench_topics")


def plot_full_topics(k, sound, complete):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    fig.subplots_adjust(left=.095, right=.985, bottom=.18, top=.735, wspace=.23)
    for ax, values, title in zip(axes, (sound, complete), ("(a) Soundness over lotteries", "(b) Completeness over lotteries")):
        for topic in TOPICS[::-1]:
            ax.plot(k, values[topic], color=TOPIC_COLORS[topic], lw=CURVE_WIDTH, ls=TOPIC_STYLES[topic])
        axis_style(ax, r"Objections tolerated $k$")
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_title(title, loc="left", pad=10)
    axes[0].set_ylabel("Fraction of prompts (%)", labelpad=7)
    topic_legend(fig)
    save(fig, "revised_full_topics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", choices=("count", "topics", "full-topics"),
        default=("count", "topics", "full-topics"),
        help="Only write the named figure outputs (default: all three).",
    )
    args = parser.parse_args()
    configure()
    sr_complete, sr_sound, pure, k, sound, complete = load_curves()
    if "count" in args.figures:
        plot_count(sr_complete, sr_sound, pure, k, sound, complete)
    if "topics" in args.figures:
        plot_pure_topics(pure)
    if "full-topics" in args.figures:
        plot_full_topics(k, sound, complete)


if __name__ == "__main__":
    main()
