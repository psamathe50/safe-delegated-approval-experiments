"""Shared colors and typography for the figures in main.tex."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CHARCOAL = "#403832"
BLUE = "#3F6E89"
OCHRE = "#BB8135"
ROSE = "#A95F64"
GREEN = "#64846A"
PLUM = "#80688C"
GRAY = "#96918A"
INK = "#302B27"
MUTED = "#7D736A"
GRID = "#E7E2DB"
AXIS = "#B4AAA0"
CURVE_WIDTH = 2.0
MAIN_CURVE_WIDTH = 1.1

METHOD_COLORS = {
    "learned_weight": CHARCOAL,
    "best_single": BLUE,
    "weighted_binary": OCHRE,
    "count_rule": ROSE,
    "equal_weight": GRAY,
    "ablated_learned_weight": PLUM,
}

TOPIC_COLORS = {
    "All prompts": CHARCOAL,
    "Factuality": BLUE,
    "Focus": OCHRE,
    "Math": GREEN,
    "Precise IF": PLUM,
    "Safety": ROSE,
}

# Increasing darkness denotes increasing coverage on a single common scale.
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "paper_coverage", ["#FFF8ED", "#E5C69B", "#C78A59", "#9E5341", "#633832"]
)


def configure():
    # Matplotlib bundles genuine TrueType Computer Modern. Using it avoids
    # embedding an OpenType/CFF Latin Modern file as a PDF TrueType font.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"],
        "mathtext.fontset": "cm", "text.usetex": False,
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 10.8, "axes.titlesize": 11.8,
        "axes.titleweight": "normal", "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": .6,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
