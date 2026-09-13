"""Redraw the count-rule approximation heatmaps from saved error arrays.

Only the presentation grid is reaggregated. No model fitting or geometric
optimization is run. The grid, boundary convention, and single-global-k
maximization match approximate_analysis.py:285-379 and 383-395 in the canonical
one_step_panel bundle. Run with Python, NumPy, and Matplotlib.
"""

from __future__ import annotations

import hashlib
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np

from paper_style import COVERAGE_CMAP, INK, configure


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "experiments" / "one_step_panel" / "data"
SOURCE = DATA / "approximate_results.npz"
OUTPUT = HERE / "generated" / "revised_approximate_count"
TOLERANCE = 1e-8
# One binned scale for both panels, with distinct bands near full coverage.
COLOR_BOUNDARIES = np.array([0, .10, .25, .40, .50, .75, .90, .95, .99, 1.0])


def approximate_complete(errors, attained, epsilon):
    """A rejected maximum at the required gain fails; an unattained sup can pass."""
    return (errors < epsilon - 5 * TOLERANCE) | (
        np.isclose(errors, epsilon, atol=5 * TOLERANCE, rtol=0.0) & ~attained
    )


def global_count_grid(soundness, completeness, attained, harm_grid, gain_grid):
    """Aggregate each fixed k first, then maximize the joint prompt fraction."""
    best = np.zeros((len(gain_grid), len(harm_grid)), dtype=float)
    best_k = np.zeros_like(best, dtype=int)
    for k in range(soundness.shape[1]):
        harm_start = np.searchsorted(
            harm_grid, soundness[:, k] - 5 * TOLERANCE, side="left"
        )
        gain_start = np.empty(len(completeness), dtype=int)
        reached = attained[:, k]
        gain_start[~reached] = np.searchsorted(
            gain_grid, completeness[~reached, k] - 5 * TOLERANCE, side="left"
        )
        gain_start[reached] = np.searchsorted(
            gain_grid, completeness[reached, k] + 5 * TOLERANCE, side="right"
        )
        histogram = np.zeros_like(best, dtype=int)
        usable = (harm_start < len(harm_grid)) & (gain_start < len(gain_grid))
        np.add.at(histogram, (gain_start[usable], harm_start[usable]), 1)
        rates = histogram.cumsum(axis=0).cumsum(axis=1) / len(soundness)
        better = rates > best
        best[better], best_k[better] = rates[better], k

    # These three inexpensive checks audit the display aggregation only.
    for iy, ix in (
        (0, 0),
        (len(gain_grid) // 2, len(harm_grid) // 2),
        (len(gain_grid) - 1, len(harm_grid) - 1),
    ):
        joint = (soundness <= harm_grid[ix] + 5 * TOLERANCE) & approximate_complete(
            completeness, attained, gain_grid[iy]
        )
        if not np.isclose(best[iy, ix], joint.mean(axis=0).max(), atol=1e-12, rtol=0):
            raise ValueError("Display aggregation disagrees with the saved-error rule.")
    return best, best_k


def load_saved_arrays():
    cases = []
    with np.load(SOURCE, allow_pickle=False) as source:
        for prefix, label, shape, max_loss, max_gain, gain_count in (
            ("strongreject", "StrongREJECT", (1084, 9), 0.5, 0.5, 101),
            ("rewardbench", "RewardBench 2", (1763, 48), 0.25, 0.75, 151),
        ):
            soundness = source[f"{prefix}_soundness_error"]
            completeness = source[f"{prefix}_completeness_error"]
            attained = source[f"{prefix}_completeness_supremum_attained"]
            if any(a.shape != shape for a in (soundness, completeness, attained)):
                raise ValueError(f"Unexpected saved-array shape for {label}.")
            if attained.dtype != np.dtype(bool):
                raise ValueError("Expected Boolean supremum-attainment flags.")
            for array, bound in ((soundness, max_loss), (completeness, max_gain)):
                if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > bound):
                    raise ValueError(f"Saved errors outside the declared domain: {label}.")
            harm_grid = np.linspace(0.0, max_loss, 101)
            gain_grid = np.linspace(0.0, max_gain, gain_count)
            rates, k = global_count_grid(
                soundness, completeness, attained, harm_grid, gain_grid
            )
            cases.append({
                "label": label, "harm_grid": harm_grid, "gain_grid": gain_grid,
                "rates": rates, "best_k": k, "prompts": shape[0],
                "k_count": shape[1], "soundness": soundness,
                "completeness": completeness, "attained": attained,
            })

    with np.load(DATA / "strongreject_scores.npz", allow_pickle=False) as strong:
        if int(np.sum(strong["label_count"] == 5)) != cases[0]["prompts"]:
            raise ValueError("The saved StrongREJECT cohort no longer matches five labels.")
    if not np.isclose(cases[0]["rates"][0, 0], 1013 / 1084, atol=1e-12, rtol=0):
        raise ValueError("StrongREJECT exact-origin fraction differs from saved report.")
    if cases[0]["best_k"][0, 0] != 6 or cases[1]["rates"][0, 0] != 0:
        raise ValueError("Unexpected exact-origin count-rule result.")
    return cases


def main():
    cases = load_saved_arrays()
    configure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.7))
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1, 1, 0.045),
        left=0.09, right=0.89, bottom=0.19, top=0.87, wspace=0.32,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in (0, 1)]
    color_axis = fig.add_subplot(grid[0, 2])
    colors = ListedColormap(COVERAGE_CMAP(np.linspace(.015, .99, len(COLOR_BOUNDARIES) - 1)))
    norm = BoundaryNorm(COLOR_BOUNDARIES, colors.N, clip=True)

    for index, (axis, case) in enumerate(zip(axes, cases)):
        loss, gain, rates = case["harm_grid"], case["gain_grid"], case["rates"]
        mesh = axis.pcolormesh(
            loss, gain, rates, shading="nearest", cmap=colors,
            norm=norm, rasterized=True,
        )
        if index == 0:
            # Label the four actual positive-gain plateaus, not guessed
            # contour levels. The gain-zero row and unrestricted-loss edge
            # are retained and reported separately below.
            for left, right, bottom, top in (
                (0, .25, 0, .25), (.25, .5, 0, .25),
                (0, .25, .25, .5), (.25, .5, .25, .5),
            ):
                selected = rates[np.ix_((gain > bottom) & (gain <= top),
                                       (loss >= left) & (loss < right))]
                values = np.unique(selected)
                if len(values) != 1:
                    raise ValueError("Expected a constant StrongREJECT interior region.")
                axis.text(
                    (left + right) / 2, (bottom + top) / 2,
                    f"{100 * values[0]:.1f}%", color="white", fontsize=10.5,
                    ha="center", va="center",
                )
            axis.axvline(.25, color="white", linewidth=.7, alpha=.7)
            axis.axhline(.25, color="white", linewidth=.7, alpha=.7)
        else:
            contour = axis.contour(
                loss, gain, rates, levels=(.10, .25, .40),
                colors=INK, linewidths=.8,
            )
            labels = axis.clabel(
                contour, inline=True, inline_spacing=6,
                fmt=lambda value: f"{100 * value:g}%", fontsize=9.5,
                manual=((.13, .10), (.10, .35), (.20, .55)),
                use_clabeltext=False,
            )
            for label in labels:
                label.set_rotation(0)
                label.set_bbox({"facecolor": "#FFF8ED", "edgecolor": "none", "pad": 1.1})
        axis.set_xlim(0, loss[-1])
        axis.set_ylim(0, gain[-1])
        axis.set_xticks(np.linspace(0, loss[-1], 6))
        axis.set_yticks(np.linspace(0, gain[-1], 6))
        axis.tick_params(length=3, width=0.6, pad=4)
        axis.set_xlabel(r"Allowed loss $\epsilon_{\rm harm}$", labelpad=6)
        if index == 0:
            axis.set_ylabel(r"Required gain $\epsilon_{\rm good}$", labelpad=6)
        axis.set_title(
            f"({chr(97 + index)}) {case['label']}", loc="left", weight="normal", pad=9
        )
        # The final column is a real endpoint, not a rendering artifact.
        # Unrestricted soundness does not imply full completeness for k<n.
        edge_color = "white" if index == 0 else INK
        axis.axvline(loss[-1], color=edge_color, linewidth=1.8,
                     linestyle=(0, (2, 2)), zorder=5, clip_on=False)
        axis.annotate(
            "Unrestricted loss", xy=(1, .84), xycoords="axes fraction",
            xytext=(.97, .94), textcoords="axes fraction",
            ha="right", va="top", fontsize=8.3, color=edge_color,
            arrowprops={"arrowstyle": "->", "color": edge_color,
                        "linewidth": .8, "shrinkA": 3, "shrinkB": 2},
        )

    colorbar = fig.colorbar(
        mesh, cax=color_axis, ticks=[0, .25, .5, .75, .9, .95, 1],
        boundaries=COLOR_BOUNDARIES, spacing="uniform", drawedges=True,
    )
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
    colorbar.ax.tick_params(length=2.5, width=0.6, pad=4, labelsize=9.5)
    colorbar.set_label("Both guarantees satisfied (%)", fontsize=10.3, labelpad=9)
    colorbar.outline.set_linewidth(0.5)

    fig.savefig(OUTPUT.with_suffix(".pdf"), metadata={
        "Title": "Approximate count-rule soundness and completeness",
        "Author": "", "Creator": "Matplotlib", "CreationDate": None, "ModDate": None,
    })
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)

    report = {
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "shared_color_bin_boundaries": COLOR_BOUNDARIES.tolist(),
        "boundary_note": "At maximum loss, soundness is unrestricted; completeness still uses the saved k<n family.",
    }
    for case in cases:
        rates = case["rates"]
        report[case["label"]] = {
            "cohort": case["prompts"], "k_values": [0, case["k_count"] - 1],
            "grid_shape_gain_by_loss": list(rates.shape),
            "loss_range": [0.0, float(case["harm_grid"][-1])],
            "gain_range": [0.0, float(case["gain_grid"][-1])],
            "color_range": [0.0, 1.0],
            "observed_joint_fraction_range": [float(rates.min()), float(rates.max())],
            "exact_origin_fraction": float(rates[0, 0]),
            "exact_origin_k": int(case["best_k"][0, 0]),
            "unrestricted_loss_fraction_range": [float(rates[:, -1].min()), float(rates[:, -1].max())],
            "monotone_on_both_grid_axes": bool(
                (np.diff(rates, axis=0) >= -1e-12).all()
                and (np.diff(rates, axis=1) >= -1e-12).all()
            ),
        }
        if case["label"] == "StrongREJECT":
            report[case["label"]]["all_saved_grid_fractions"] = np.unique(rates).tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
