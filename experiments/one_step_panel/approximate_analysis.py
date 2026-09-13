"""Exact approximate soundness and completeness on the bounded domains.

For a rule on a prompt, the soundness error is the largest principal loss of
an authorized proposal.  The completeness error is the supremum principal
gain of a rejected proposal.  RewardBench 2 errors are optimized over the
entire four-response simplex; no response lotteries are sampled.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.linalg import null_space

from geometry import multicandidate_utilities, stratified_prompt_split


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
RESULTS_NPZ = DATA / "approximate_results.npz"
RESULTS_JSON = DATA / "approximate_results.json"
IMPLEMENTATION_PDF = PLOTS / "approximate_implementation.pdf"
IMPLEMENTATION_PNG = PLOTS / "approximate_implementation.png"

TOLERANCE = 1e-8
INK = "#24272B"

# d stores the first three coordinates of q-b.  The fourth is minus their sum.
Q_GRADIENTS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
    ]
)
FACET_NORMALS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
FACET_OFFSETS = np.asarray([-0.25, -0.25, -0.25, 0.25])
PURE_DEVIATIONS = np.eye(4)[:, :3] - 0.25
PAIR_FIRST, PAIR_SECOND = np.triu_indices(48, 1)


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


def q_coordinates(deviations: np.ndarray) -> np.ndarray:
    deviations = np.atleast_2d(deviations)
    return np.column_stack(
        [0.25 + deviations, 0.25 - np.sum(deviations, axis=1)]
    )


def facet_vertices(
    reviewers: np.ndarray,
    facet: int,
) -> np.ndarray:
    """Enumerate every vertex of the reviewer arrangement on one simplex facet."""
    vertices: list[np.ndarray] = [
        PURE_DEVIATIONS[index] for index in range(4) if index != facet
    ]
    facet_actions = [index for index in range(4) if index != facet]

    # Intersections of a reviewer boundary with a simplex edge.
    for first, second in itertools.combinations(facet_actions, 2):
        start = PURE_DEVIATIONS[first]
        direction = PURE_DEVIATIONS[second] - start
        denominator = reviewers @ direction
        numerator = -(reviewers @ start)
        usable = np.abs(denominator) > TOLERANCE
        parameter = np.zeros(len(reviewers))
        parameter[usable] = numerator[usable] / denominator[usable]
        usable &= parameter >= -TOLERANCE
        usable &= parameter <= 1.0 + TOLERANCE
        vertices.extend(start + parameter[usable, None] * direction)

    # Intersections of two reviewer boundaries with the facet.  Their cross
    # product is the common ray through the uniform fallback.
    crosses = np.cross(reviewers[PAIR_FIRST], reviewers[PAIR_SECOND])
    denominator = crosses @ FACET_NORMALS[facet]
    usable = np.abs(denominator) > TOLERANCE
    pair_vertices = (
        crosses[usable]
        * (FACET_OFFSETS[facet] / denominator[usable])[:, None]
    )
    if pair_vertices.size:
        feasible = np.min(q_coordinates(pair_vertices), axis=1) >= -TOLERANCE
        vertices.extend(pair_vertices[feasible])

    matrix = np.asarray(vertices, dtype=float)
    feasible = np.min(q_coordinates(matrix), axis=1) >= -TOLERANCE
    matrix = matrix[feasible]
    _, unique = np.unique(np.round(matrix, 10), axis=0, return_index=True)
    return matrix[np.sort(unique)]


def adjacent_cell_maximum_objections(
    reviewers: np.ndarray,
    deviation: np.ndarray,
    facet: int,
    gains: np.ndarray,
) -> int:
    """Maximum strict objections in a facet cell incident to a vertex."""
    tied = np.flatnonzero(np.abs(gains) <= 5 * TOLERANCE)
    base = int(np.sum(gains < -5 * TOLERANCE))
    q = q_coordinates(deviation)[0]
    active = [
        index
        for index in range(4)
        if index != facet and q[index] <= 5 * TOLERANCE
    ]

    # These cases cover all nondegenerate vertices and avoid an angular sweep
    # for the roughly five thousand ordinary candidates on each prompt.
    if tied.size == 0:
        return base
    if not active and tied.size <= 2:
        return base + int(tied.size)
    if len(active) == 1 and tied.size == 1:
        return base + 1

    basis = null_space(FACET_NORMALS[facet][None, :])
    boundary_angles: list[float] = []
    for row in list(reviewers[tied]) + [Q_GRADIENTS[index] for index in active]:
        coefficients = row @ basis
        if np.linalg.norm(coefficients) <= TOLERANCE:
            continue
        angle = float(
            np.arctan2(coefficients[0], -coefficients[1]) % (2 * np.pi)
        )
        boundary_angles.extend([angle, (angle + np.pi) % (2 * np.pi)])

    if boundary_angles:
        values = np.unique(np.round(boundary_angles, 12))
        candidate_angles = [
            float((left + right) / 2 % (2 * np.pi))
            for left, right in zip(values, np.r_[values[1:], values[0] + 2 * np.pi])
        ]
    else:
        candidate_angles = [0.0]

    maximum = base
    for angle in candidate_angles:
        direction_2d = np.asarray([np.cos(angle), np.sin(angle)])
        direction = basis @ direction_2d
        if any(Q_GRADIENTS[index] @ direction <= TOLERANCE for index in active):
            continue
        maximum = max(
            maximum,
            base
            + int(
                np.sum(reviewers[tied] @ direction < -TOLERANCE)
            ),
        )
    return maximum


def rewardbench_prompt_errors(
    reviewers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return exact error bands for every nontrivial count threshold."""
    n_reviewers = len(reviewers)
    harm_by_objections = np.zeros(n_reviewers + 1)
    rejected_sup_by_objections = np.zeros(n_reviewers + 1)
    rejected_attained_by_objections = np.full(n_reviewers + 1, -np.inf)

    minimum_harmful_objections = n_reviewers + 1
    maximum_beneficial_objections = 0
    for facet in range(4):
        vertices = facet_vertices(reviewers, facet)
        evaluations = reviewers @ vertices.T
        for position, deviation in enumerate(vertices):
            gains = evaluations[:, position]
            objections = int(np.sum(gains < -TOLERANCE))
            principal_gain = float(deviation[0])
            if principal_gain < -TOLERANCE:
                minimum_harmful_objections = min(
                    minimum_harmful_objections, objections
                )
            harm_by_objections[objections] = max(
                harm_by_objections[objections], max(0.0, -principal_gain)
            )
            rejected_attained_by_objections[objections] = max(
                rejected_attained_by_objections[objections],
                principal_gain,
            )

            adjacent = adjacent_cell_maximum_objections(
                reviewers, deviation, facet, gains
            )
            maximum_beneficial_objections = max(
                maximum_beneficial_objections,
                # A beneficial ray reaches the simplex boundary with strictly
                # positive principal gain.  Vertices at gain zero can also be
                # incident to cells lying wholly on the harmful side and must
                # not be used to infer the beneficial-cell maximum.
                adjacent if principal_gain > TOLERANCE else 0,
            )
            rejected_sup_by_objections[adjacent] = max(
                rejected_sup_by_objections[adjacent],
                max(0.0, principal_gain),
            )

    soundness_error = np.maximum.accumulate(harm_by_objections)[:-1]
    completeness_error = np.zeros(n_reviewers)
    completeness_sup_attained = np.zeros(n_reviewers, dtype=bool)
    for k in range(n_reviewers):
        relevant = rejected_sup_by_objections[k + 1 :]
        supremum = float(np.max(relevant)) if relevant.size else 0.0
        completeness_error[k] = supremum
        attained = float(np.max(rejected_attained_by_objections[k + 1 :]))
        completeness_sup_attained[k] = attained >= supremum - 5 * TOLERANCE
    return (
        soundness_error,
        completeness_error,
        completeness_sup_attained,
        minimum_harmful_objections,
        maximum_beneficial_objections,
    )


def strongreject_errors() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=True) as data:
        complete = data["label_count"] == 5
        human = data["human"][complete].astype(float)
        scores = data["evaluator_scores"][complete].astype(float)
    principal_gain = 0.5 - human
    objections = np.sum(scores > 0.5 + TOLERANCE, axis=1)
    k_values = np.arange(scores.shape[1])
    accepted = objections[:, None] <= k_values[None, :]
    soundness_error = np.where(
        accepted, np.maximum(0.0, -principal_gain[:, None]), 0.0
    )
    completeness_error = np.where(
        ~accepted, np.maximum(0.0, principal_gain[:, None]), 0.0
    )
    completeness_supremum_attained = (~accepted) & (
        principal_gain[:, None] >= -TOLERANCE
    )
    return (
        soundness_error,
        completeness_error,
        completeness_supremum_attained,
        objections,
    )


def approximate_complete(
    errors: np.ndarray,
    attained: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Apply the weak-gain boundary convention in epsilon-completeness."""
    below = errors < epsilon - 5 * TOLERANCE
    equal_unattained = np.isclose(
        errors, epsilon, atol=5 * TOLERANCE, rtol=0.0
    ) & ~attained
    return below | equal_unattained


def simultaneous_rates(
    soundness_error: np.ndarray,
    completeness_error: np.ndarray,
    completeness_attained: np.ndarray,
    epsilon_harm: float,
    epsilon_good: float,
) -> np.ndarray:
    """Fraction of cases meeting both tolerances, separately for every k."""
    return np.mean(
        (soundness_error <= epsilon_harm + 5 * TOLERANCE)
        & approximate_complete(
            completeness_error,
            completeness_attained,
            epsilon_good,
        ),
        axis=0,
    )


def achievable_frontier(
    soundness_error: np.ndarray,
    completeness_error: np.ndarray,
    completeness_attained: np.ndarray,
    harm_grid: np.ndarray,
    good_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Best simultaneous rate and its single global k over a tolerance grid."""
    best_rate = np.zeros((len(good_grid), len(harm_grid)), dtype=float)
    best_k = np.zeros_like(best_rate, dtype=int)

    # For a fixed k, each case is certified on an upper-right rectangle of the
    # tolerance grid.  Bin its lower-left corner and use a cumulative sum rather
    # than reevaluating every case at every grid point.
    for k in range(soundness_error.shape[1]):
        harm_start = np.searchsorted(
            harm_grid,
            soundness_error[:, k] - 5 * TOLERANCE,
            side="left",
        )
        good_start = np.empty(len(completeness_error), dtype=int)
        attained = completeness_attained[:, k]
        good_start[~attained] = np.searchsorted(
            good_grid,
            completeness_error[~attained, k] - 5 * TOLERANCE,
            side="left",
        )
        good_start[attained] = np.searchsorted(
            good_grid,
            completeness_error[attained, k] + 5 * TOLERANCE,
            side="right",
        )

        histogram = np.zeros_like(best_rate, dtype=int)
        usable = (harm_start < len(harm_grid)) & (good_start < len(good_grid))
        np.add.at(histogram, (good_start[usable], harm_start[usable]), 1)
        rates = histogram.cumsum(axis=0).cumsum(axis=1) / len(soundness_error)
        improves = rates > best_rate
        best_rate[improves] = rates[improves]
        best_k[improves] = k

    # Check the rectangle calculation at representative boundary and interior
    # points against the definitions used everywhere else in this script.
    checks = {
        (0, 0),
        (len(good_grid) // 2, len(harm_grid) // 2),
        (len(good_grid) - 1, len(harm_grid) - 1),
    }
    for good_index, harm_index in checks:
        direct = simultaneous_rates(
            soundness_error,
            completeness_error,
            completeness_attained,
            float(harm_grid[harm_index]),
            float(good_grid[good_index]),
        )
        if not np.isclose(
            best_rate[good_index, harm_index],
            np.max(direct),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("frontier grid disagrees with direct evaluation")
    return best_rate, best_k


def make_implementation_figure(
    strong_sound: np.ndarray,
    strong_complete: np.ndarray,
    strong_attained: np.ndarray,
    reward_sound: np.ndarray,
    reward_complete: np.ndarray,
    reward_attained: np.ndarray,
) -> None:
    configure_style()
    strong_harm_grid = np.linspace(0.0, 0.5, 101)
    strong_good_grid = np.linspace(0.0, 0.5, 101)
    reward_harm_grid = np.linspace(0.0, 0.25, 101)
    reward_good_grid = np.linspace(0.0, 0.75, 151)
    strong_rates, _ = achievable_frontier(
        strong_sound,
        strong_complete,
        strong_attained,
        strong_harm_grid,
        strong_good_grid,
    )
    reward_rates, _ = achievable_frontier(
        reward_sound,
        reward_complete,
        reward_attained,
        reward_harm_grid,
        reward_good_grid,
    )
    figure = plt.figure(figsize=(9.2, 3.55))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.075,
        right=0.94,
        bottom=0.18,
        top=0.91,
        wspace=0.34,
    )
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    colorbar_axis = figure.add_subplot(grid[0, 2])
    meshes = []
    for axis, rates, harm_grid, good_grid, title, levels, contour_color, labels in (
        (
            axes[0],
            strong_rates,
            strong_harm_grid,
            strong_good_grid,
            r"$\bf{(a)}$ StrongREJECT",
            (95,),
            "white",
            ((0.06, 0.25),),
        ),
        (
            axes[1],
            reward_rates,
            reward_harm_grid,
            reward_good_grid,
            r"$\bf{(b)}$ RewardBench 2",
            (10, 25, 40),
            INK,
            ((0.13, 0.10), (0.10, 0.35), (0.20, 0.55)),
        ),
    ):
        mesh = axis.pcolormesh(
            harm_grid,
            good_grid,
            100 * rates,
            shading="nearest",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            rasterized=True,
        )
        meshes.append(mesh)
        contours = axis.contour(
            harm_grid,
            good_grid,
            100 * rates,
            levels=levels,
            colors=contour_color,
            linewidths=0.75,
        )
        axis.clabel(
            contours,
            inline=True,
            fmt=lambda value: f"{value:g}%",
            fontsize=6.5,
            manual=labels,
        )
        axis.set_xlim(float(harm_grid[0]), float(harm_grid[-1]))
        axis.set_ylim(float(good_grid[0]), float(good_grid[-1]))
        axis.tick_params(length=3, width=0.65, color="#8F9293")
        axis.set_xlabel(r"allowed loss $\epsilon_{\rm harm}$")
        axis.set_ylabel(r"required gain $\epsilon_{\rm good}$")
        axis.set_title(title, loc="left", pad=6)
    axes[0].set_xticks(np.linspace(0.0, 0.5, 6))
    axes[0].set_yticks(np.linspace(0.0, 0.5, 6))
    axes[1].set_xticks(np.linspace(0.0, 0.25, 6))
    axes[1].set_yticks(np.linspace(0.0, 0.75, 6))
    colorbar = figure.colorbar(
        meshes[-1],
        cax=colorbar_axis,
        ticks=[0, 25, 50, 75, 100],
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    colorbar.set_label("best jointly certified fraction", fontsize=7.6)
    figure.savefig(IMPLEMENTATION_PDF, bbox_inches="tight")
    figure.savefig(IMPLEMENTATION_PNG, dpi=320, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        subsets = data["subsets"].astype(str)
        prompt_keys = data["prompt_keys"].astype(str)
        dataset_revision = str(data["dataset_revision"])
        results_revision = str(data["results_revision"])
    standard = np.flatnonzero(candidate_counts == 4)
    if len(standard) != 1763 or not np.all(num_correct[standard] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")

    n_reviewers = scores.shape[0]
    reward_soundness = np.empty((len(standard), n_reviewers))
    reward_completeness = np.empty((len(standard), n_reviewers))
    reward_attained = np.empty((len(standard), n_reviewers), dtype=bool)
    boundary_depth = np.empty(len(standard), dtype=int)
    boundary_beneficial_objections = np.empty(len(standard), dtype=int)
    for position, prompt in enumerate(standard):
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, _ = multicandidate_utilities(scores, start, stop, 1)
        (
            reward_soundness[position],
            reward_completeness[position],
            reward_attained[position],
            boundary_depth[position],
            boundary_beneficial_objections[position],
        ) = rewardbench_prompt_errors(reviewers)
        if (position + 1) % 100 == 0:
            print(f"bounded-domain audit {position + 1}/{len(standard)}")

    with np.load(DATA / "rewardbench2_depth.npz", allow_pickle=True) as data:
        stored_depth = data["depth"][standard].astype(int)
        stored_beneficial = data["maximum_beneficial_objections"][standard].astype(int)
    if not np.array_equal(boundary_depth, stored_depth):
        mismatch = int(np.sum(boundary_depth != stored_depth))
        raise RuntimeError(f"bounded audit disagrees with depth on {mismatch} prompts")
    if not np.array_equal(boundary_beneficial_objections, stored_beneficial):
        mismatch = int(np.sum(boundary_beneficial_objections != stored_beneficial))
        raise RuntimeError(
            f"bounded audit disagrees with completeness on {mismatch} prompts"
        )

    (
        strong_soundness,
        strong_completeness,
        strong_attained,
        strong_objections,
    ) = strongreject_errors()
    np.savez_compressed(
        RESULTS_NPZ,
        standard_prompts=standard,
        prompt_keys=prompt_keys[standard],
        categories=subsets[standard],
        rewardbench_soundness_error=reward_soundness,
        rewardbench_completeness_error=reward_completeness,
        rewardbench_completeness_supremum_attained=reward_attained,
        strongreject_soundness_error=strong_soundness,
        strongreject_completeness_error=strong_completeness,
        strongreject_completeness_supremum_attained=strong_attained,
        strongreject_objections=strong_objections,
        dataset_revision=dataset_revision,
        results_revision=results_revision,
    )

    assignment = stratified_prompt_split(subsets, seed=20260813)
    test_positions = np.flatnonzero(assignment[standard] == "test")

    # A few fixed pairs make the machine-readable summary easy to audit.  The
    # figure itself evaluates the complete two-dimensional tolerance grid.
    reward_pairs = ((0.0, 0.0), (0.05, 0.30), (0.10, 0.30), (0.20, 0.30))
    strong_pairs = ((0.0, 0.0), (0.0, 0.25), (0.25, 0.25))

    def summarize_pairs(
        soundness: np.ndarray,
        completeness: np.ndarray,
        attained: np.ndarray,
        pairs: tuple[tuple[float, float], ...],
    ) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for epsilon_harm, epsilon_good in pairs:
            rates = simultaneous_rates(
                soundness,
                completeness,
                attained,
                epsilon_harm,
                epsilon_good,
            )
            output[f"({epsilon_harm:g},{epsilon_good:g})"] = {
                "k": int(np.argmax(rates)),
                "rate": float(np.max(rates)),
            }
        return output

    summary = {
        "dataset_revision": dataset_revision,
        "results_revision": results_revision,
        "rewardbench2": {
            "prompts": int(len(standard)),
            "soundness_error_range": [
                float(np.min(reward_soundness)),
                float(np.max(reward_soundness)),
            ],
            "completeness_error_range": [
                float(np.min(reward_completeness)),
                float(np.max(reward_completeness)),
            ],
            "achievable_examples": summarize_pairs(
                reward_soundness,
                reward_completeness,
                reward_attained,
                reward_pairs,
            ),
            "test_achievable_examples": summarize_pairs(
                reward_soundness[test_positions],
                reward_completeness[test_positions],
                reward_attained[test_positions],
                reward_pairs,
            ),
        },
        "strongreject": {
            "responses": int(len(strong_soundness)),
            "achievable_examples": summarize_pairs(
                strong_soundness,
                strong_completeness,
                strong_attained,
                strong_pairs,
            ),
        },
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    make_implementation_figure(
        strong_soundness,
        strong_completeness,
        strong_attained,
        reward_soundness,
        reward_completeness,
        reward_attained,
    )
    print(RESULTS_NPZ)
    print(RESULTS_JSON)
    print(IMPLEMENTATION_PDF)


if __name__ == "__main__":
    main()
