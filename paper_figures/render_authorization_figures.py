"""Render paper figures from saved results; no fitting or experimental runs.

Tradeoff curves connect observed outcomes in threshold order.
The fitted score rules and the saved threshold grid remain fixed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from paper_style import AXIS, CURVE_WIDTH, MAIN_CURVE_WIDTH, GRID, INK, MUTED, METHOD_COLORS, configure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "one_step_panel" / "data" / "cardinal_results.json"
OUT = Path(__file__).resolve().parent / "generated"
STYLES = {
    "count_rule": (r"$k$-tolerant threshold", METHOD_COLORS["count_rule"], "-"),
    "weighted_binary": ("Weighted votes", METHOD_COLORS["weighted_binary"], "-"),
    "learned_weight": ("Cardinal committee", METHOD_COLORS["learned_weight"], "-"),
    "best_single": ("Single reviewer", METHOD_COLORS["best_single"], "-"),
    "equal_weight": ("Equal weights", METHOD_COLORS["equal_weight"], (0, (6, 2.5))),
    "ablated_learned_weight": ("Committee, reviewer removed", METHOD_COLORS["ablated_learned_weight"], (0, (8, 2.5, 3, 2.5))),
}
MAIN = ("learned_weight", "best_single", "weighted_binary", "count_rule")


def style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6, pad=5)
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 25, 50, 75, 100])


def coords(row, domain):
    if domain == "pure":
        return row["test"]["pure_response_soundness"], row["test"]["pure_response_completeness"]
    return row["test_full_domain_soundness"], row["test_approximate_full_domain"]["epsilon_good_completeness"]


def selected_row(data, key):
    family = data[key]
    parameter = "k" if key == "count_rule" else "threshold"
    selected = family["operating_points"]["0.99"]
    matches = [r for r in family["rows"] if r[parameter] == selected[parameter]]
    assert len(matches) == 1
    row = matches[0]
    assert row["validation"]["pure_response_soundness"] >= .99 - 1e-9
    feasible = [r for r in family["rows"] if r["validation"]["pure_response_soundness"] >= .99 - 1e-9]
    best = max(feasible, key=lambda r: (r["validation"]["pure_response_completeness"], r["validation"]["pure_response_soundness"]))
    assert row == best
    return row


def retrospective(data, key, domain, target):
    rows = [r for r in data[key]["rows"] if coords(r, domain)[0] >= target - 1e-9]
    return max(rows, key=lambda r: (coords(r, domain)[1], coords(r, domain)[0]))


def draw_curve(ax, data, key, domain, linewidth=CURVE_WIDTH):
    """Connect observed pairs; preserve threshold order and all distinct outcomes."""
    _, color, dash = STYLES[key]
    xy = np.asarray([coords(r, domain) for r in data[key]["rows"]]) * 100
    # Identical consecutive outcomes add no information. Keep vertical and
    # horizontal segments between distinct measured pairs without smoothing.
    keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]
    xy = xy[keep]
    ax.plot(xy[:, 0], xy[:, 1], color=color, ls=dash, lw=linewidth, zorder=3)


def family_handles(keys, linewidth=CURVE_WIDTH):
    return [Line2D([], [], color=STYLES[k][1], ls=STYLES[k][2], lw=linewidth, label=STYLES[k][0]) for k in keys]


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"Title": stem.replace("_", " "), "Author": "", "Creator": "Matplotlib"})
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def main_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.095, right=.982, bottom=.185, top=.87, wspace=.45)
    panels = (("pure", "(a) Individual answers", "Correct answer authorized (%)"),
              ("full", "(b) Lotteries over answers", "Prompts authorizing every\nqualifying lottery (%)"))
    for ax, (domain, title, ylabel) in zip(axes, panels):
        style_axis(ax, "Test soundness (%)", ylabel)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlim(90, 100.25)
        ax.set_xticks([90, 95, 100])
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=2, width=.5)
        for target in (95, 99):
            ax.axvline(target, color=GRID, lw=.8, zorder=0)
        for key in MAIN:
            draw_curve(ax, data, key, domain, linewidth=MAIN_CURVE_WIDTH)
    axes[1].legend(handles=family_handles(MAIN, linewidth=MAIN_CURVE_WIDTH), loc="upper right", frameon=False,
                   fontsize=9.5, handlelength=2.1, handletextpad=.6, labelspacing=.7)
    axes[1].text(.965, .49, "Both binary rules\nlie near zero", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9, color=MUTED, linespacing=1.35)
    save(fig, "revised_authorization_comparison")


def appendix_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.12, right=.985, bottom=.18, top=.755, wspace=.32)
    fig.legend(handles=family_handles(tuple(STYLES)), loc="upper center",
               bbox_to_anchor=(.53, .998), ncol=3, frameon=False,
               fontsize=9, columnspacing=1.2, handlelength=2.7, labelspacing=.6)
    ax = axes[0]
    style_axis(ax, "Test soundness (%)", "Prompts authorizing every\nqualifying lottery (%)")
    ax.set_title("(a) Soundness versus completeness", loc="left", pad=12)
    ax.set_xlim(0, 100.8)
    ax.set_xticks([0, 25, 50, 75, 100])
    for key in STYLES:
        draw_curve(ax, data, key, "full")
    ax = axes[1]
    style_axis(ax, r"Allowed loss $\varepsilon_{\mathrm{harm}}$", "")
    ax.set_title("(b) Completeness versus allowed loss", loc="left", pad=12)
    ax.set_xlim(0, .245)
    ax.set_xticks([0, .05, .10, .15, .20])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(-1, 45)
    ax.set_yticks([0, 15, 30, 45])
    profile = data["soundness_tolerance_sensitivity"]
    x = np.asarray(profile["epsilon_harm"])
    nonvacuous = x < .25
    assert x[-1] == .25
    for key in STYLES:
        _, color, dash = STYLES[key]
        y = np.asarray(profile["series"][key]["best_completeness"]) * 100
        assert y[-1] == 100
        # Keep all nonvacuous observations. Report the shared exact endpoint
        # explicitly instead of compressing these curves with a 100% jump.
        ax.step(x[nonvacuous], y[nonvacuous], where="post", color=color,
                ls=dash, lw=CURVE_WIDTH, zorder=2)
    ax.text(.97, .97, "At 0.25: 100%\nfor all rules", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK, linespacing=1.35)
    save(fig, "revised_cardinal_appendix")


def margin_comparison(data):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    fig.subplots_adjust(left=.13, right=.978, bottom=.21, top=.95)
    keys = ("learned_weight", "best_single", "ablated_learned_weight")
    style_axis(ax, r"Minimum gain guaranteed approval $\varepsilon_{\mathrm{good}}$", "Completeness (%)")
    ax.set_xlim(0, .755)
    ax.set_xticks([0, .25, .50, .75])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.axvline(.5, color=GRID, lw=.8, zorder=0)
    profile = data["margin_sensitivity"]
    for key in keys:
        _, color, dash = STYLES[key]
        ax.plot(profile["gain_margins"], np.asarray(profile["series"][key]["completeness"]) * 100,
                color=color, ls=dash, lw=CURVE_WIDTH)
    ax.legend(handles=family_handles(keys), loc="upper left", frameon=False,
              fontsize=9.5, handlelength=2.7, labelspacing=.7)
    ax.text(.15, 8, "Zero completeness", color=MUTED, fontsize=9, ha="center")
    save(fig, "revised_margin_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", nargs="+", choices=("main", "appendix", "margin"),
                        default=("main", "appendix", "margin"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    configure()
    # Named checkpoints guard against accidental metric or comparator changes.
    assert round(coords(selected_row(data, "learned_weight"), "pure")[0] * 352) == 347
    assert round(coords(retrospective(data, "best_single", "pure", .99), "pure")[1] * 352) == 164
    assert round(coords(retrospective(data, "learned_weight", "full", .95), "full")[1] * 352) == 101
    renderers = {"main": main_comparison, "appendix": appendix_comparison,
                 "margin": margin_comparison}
    for name in args.figures:
        renderers[name](data)
    provenance = {
        "source_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "test_prompts": 352,
        "curves": "Observed soundness-completeness pairs connected in saved threshold order; no envelope or smoothing.",
        "reported_comparisons": "At each quoted soundness target, maximize completeness over evaluated thresholds meeting that target on test.",
        "main_domains": ["pure responses", "all response lotteries, gain margin 0.5"],
        "main_display_soundness_percent": [90, 100],
        "main_curve_linewidth_points": MAIN_CURVE_WIDTH,
        "error_bars": "None; descriptive outcomes on the fixed test set, not a population confidence statement.",
        "thresholds_refitted": False,
        "text_font": "Computer Modern Roman (bundled cmr10 TrueType); Computer Modern math",
        "percent_format": "Bare tick values with (%) in the axis label",
        "main_guides": {"major_percent": [90, 95, 100], "minor_step_percent": 1, "vertical_guides_percent": [95, 99]},
        "harm_panel_view": {"loss_range": [0, 0.245], "completeness_percent_range": [0, 45], "endpoint_note": "All six saved curves attain 100% at loss 0.25"},
        "margin_view": "Full saved gain domain retained, including zero completeness at small margins",

        "test_checkpoints": {str(target): {key: list(coords(retrospective(data, key, "pure", target), "pure")) for key in MAIN} for target in (.95, .99)},
    }
    (OUT / "revised_authorization_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
