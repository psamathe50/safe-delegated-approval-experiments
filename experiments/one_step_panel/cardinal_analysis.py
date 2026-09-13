"""Held-out RewardBench 2 audit of binary and cardinal weighted rules.

For each standard four-response prompt, reviewer scores are centered at the
uniform-response fallback and positively normalized within reviewer and
prompt.  Nonnegative weight vectors are learned on training prompts by
multicandidate softmax loss; regularization is chosen on validation prompts.

We learn matched nonnegative weighted rules from either binary approval bits
or cardinal scores.  Each rule authorizes a pure response when its aggregate
exceeds a common threshold.  Binary and learned-cardinal thresholds are
enumerated from validation scores.  The separate train-selected singleton
audit enumerates thresholds from training scores; all held-out curves are
evaluated on test prompts.  This is a distributional evaluation of learned
rules, not a prompt-specific coverage certificate.

The separate topic-threshold budget audit freezes those score rules, builds
behaviorally exhaustive calibration grids from validation, selects one shared
threshold or five routed thresholds for every validation failure budget, and
only then evaluates each selected policy on test.

Two fixed native-binary diagnostics rank reviewers by validation choice
accuracy.  One follows each next-ranked reviewer alone after dropping its
predecessors; the other adds reviewers in that order and requires unanimous
approval from every ranked prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp

from approximate_analysis import (
    FACET_NORMALS,
    Q_GRADIENTS,
    facet_vertices,
    q_coordinates,
)
from geometry import (
    covers_multicandidate_utility,
    multicandidate_utilities,
    standard_angular_cell_evaluations,
    standard_angular_evaluations,
    stratified_prompt_split,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "rewardbench2_scores.npz"
DEPTH_DATA = HERE / "data" / "rewardbench2_depth.npz"
APPROXIMATE_DATA = HERE / "data" / "approximate_results.npz"
RESULTS = HERE / "data" / "cardinal_results.json"
TOPIC_THRESHOLD_RESULTS = HERE / "data" / "topic_threshold_budget_results.json"
FIGURE = HERE / "plots" / "cardinal_frontier.png"
FIGURE_PDF = HERE / "plots" / "cardinal_frontier.pdf"
PURE_FIGURE = HERE / "plots" / "cardinal_pure_response.png"
PURE_FIGURE_PDF = HERE / "plots" / "cardinal_pure_response.pdf"
PURE_DETAIL_FIGURE = HERE / "plots" / "cardinal_pure_response_full_and_zoom.png"
PURE_DETAIL_FIGURE_PDF = (
    HERE / "plots" / "cardinal_pure_response_full_and_zoom.pdf"
)
MARGIN_FIGURE = HERE / "plots" / "cardinal_margin_sensitivity.png"
MARGIN_FIGURE_PDF = HERE / "plots" / "cardinal_margin_sensitivity.pdf"
RANKED_APPROXIMATE_FIGURE = (
    HERE / "plots" / "ranked_singleton_approximate.png"
)
RANKED_APPROXIMATE_FIGURE_PDF = (
    HERE / "plots" / "ranked_singleton_approximate.pdf"
)
TOPIC_CONDITIONED_FIGURE = (
    HERE / "plots" / "topic_conditioned_comparison.png"
)
TOPIC_CONDITIONED_FIGURE_PDF = (
    HERE / "plots" / "topic_conditioned_comparison.pdf"
)
TOPIC_THRESHOLD_FIGURE = (
    HERE / "plots" / "topic_threshold_budget.png"
)
TOPIC_THRESHOLD_FIGURE_PDF = (
    HERE / "plots" / "topic_threshold_budget.pdf"
)
SINGLETON_GENERALIZATION_FIGURE = (
    HERE / "plots" / "singleton_train_test_generalization.png"
)
SINGLETON_GENERALIZATION_FIGURE_PDF = (
    HERE / "plots" / "singleton_train_test_generalization.pdf"
)
SEED = 20260813
TOPICS = ("Factuality", "Focus", "Math", "Precise IF", "Safety")
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BINARY_LEARNED_SHARE_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
TARGET_SOUNDNESS = (0.99, 0.95, 0.90, 0.80)
APPROXIMATE_GOOD = 0.50
SOUNDNESS_COVERAGE_TARGET = 0.95
TOLERANCE = 1e-9
GEOMETRY_TOLERANCE = 1e-8
CALIBRATION_WEIGHT_DECIMALS = 8

INK = "#24272B"
BLUE = "#2F66A8"
RED = "#B44A55"
GRAY = "#747775"
ORANGE = "#D17A22"
PURPLE = "#7653A6"
PINK = "#B75082"
TEAL = "#168A96"
GRID = "#DFE1E2"


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


def normalized_features(
    scores: np.ndarray,
    offsets: np.ndarray,
    standard_prompts: np.ndarray,
) -> np.ndarray:
    """Return centered, unit-norm score vectors for four-response prompts."""
    features = np.zeros((len(offsets) - 1, 4, scores.shape[0]))
    for prompt in standard_prompts:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        prompt_scores = scores[:, start:stop]
        if prompt_scores.shape[1] != 4:
            raise ValueError("standard prompt does not have four responses")
        centered = prompt_scores - prompt_scores.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        scale[scale < TOLERANCE] = 1.0
        features[prompt] = (centered / scale).T
    return features


def canonical_calibration_weight_units(weights: np.ndarray) -> np.ndarray:
    """Quantize a fitted committee to reproducible integer units.

    Optimizer and BLAS implementations can differ in the last few bits.  A
    threshold exactly on a tolerance-adjusted validation boundary can turn
    those irrelevant differences into a different representative of the same
    validation behavior, which can in turn move a held-out count.  Calibration
    therefore uses fixed eight-decimal integer weight units.  The same
    projection is also used for weighted-binary score grids, where repeated
    approval patterns create exact mathematical ties that ordinary BLAS
    aggregation can spuriously split.  Continuous cardinal frontier fits
    remain unquantized outside the separate calibration audit.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("calibration weights must be one finite vector")
    if np.min(weights) < -TOLERANCE or np.sum(weights) <= TOLERANCE:
        raise ValueError("calibration weights must be nonnegative and nonzero")
    scale = 10**CALIBRATION_WEIGHT_DECIMALS
    units = np.rint(np.maximum(weights, 0.0) * scale).astype(np.int64)
    total_units = int(np.sum(units))
    if total_units <= 0:
        raise RuntimeError("weight quantization removed the whole committee")
    return units


def canonical_calibration_weights(weights: np.ndarray) -> np.ndarray:
    """Return the normalized form of the canonical integer-unit weights."""
    units = canonical_calibration_weight_units(weights)
    return units.astype(float) / int(np.sum(units))


def calibration_weight_canonicalization_metadata(
    fitted_weights: np.ndarray,
    weight_units: np.ndarray,
    models: np.ndarray,
) -> dict[str, object]:
    """Describe and fingerprint one deterministic weight projection."""
    fitted_weights = np.asarray(fitted_weights, dtype=float)
    weight_units = np.asarray(weight_units, dtype=np.int64)
    canonical = weight_units.astype(float) / int(np.sum(weight_units))
    delta = canonical - fitted_weights
    scale = 10**CALIBRATION_WEIGHT_DECIMALS
    scaled = np.maximum(fitted_weights, 0.0) * scale
    half_boundary_distance = np.min(
        np.abs(scaled - (np.floor(scaled) + 0.5))
    )
    unit_bytes = np.asarray(weight_units, dtype="<i8").tobytes()
    reviewer_order = "\0".join(str(model) for model in models).encode("utf-8")
    return {
        "decimal_places": CALIBRATION_WEIGHT_DECIMALS,
        "integer_unit_rounding": "nearest, ties to even",
        "integer_unit_total": int(np.sum(weight_units)),
        "integer_units_sha256": hashlib.sha256(unit_bytes).hexdigest(),
        "reviewer_order_sha256": hashlib.sha256(reviewer_order).hexdigest(),
        "renormalized_to_sum_one": True,
        "aggregation": (
            "math.fsum of feature times integer unit in fixed reviewer "
            "archive order, divided once by the integer-unit total"
        ),
        "maximum_absolute_weight_change": float(np.max(np.abs(delta))),
        "l1_weight_change": float(np.sum(np.abs(delta))),
        "minimum_distance_to_half_unit_boundary": float(
            half_boundary_distance
        ),
    }


def reproducible_weighted_gains(
    features: np.ndarray,
    weight_units: np.ndarray,
) -> np.ndarray:
    """Aggregate integer units in fixed order, dividing only once."""
    features = np.asarray(features, dtype=float)
    weight_units = np.asarray(weight_units, dtype=np.int64)
    if features.ndim < 2 or features.shape[-1] != len(weight_units):
        raise ValueError("feature and calibration-weight shapes disagree")
    total_units = int(np.sum(weight_units))
    if total_units <= 0:
        raise ValueError("calibration weight units must sum to a positive value")
    active = np.flatnonzero(weight_units)
    flattened = features.reshape(-1, features.shape[-1])
    values = np.fromiter(
        (
            math.fsum(
                float(row[index]) * int(weight_units[index])
                for index in active
            )
            / total_units
            for row in flattened
        ),
        dtype=float,
        count=len(flattened),
    )
    return values.reshape(features.shape[:-1])


def fit_nonnegative_weights(
    features: np.ndarray,
    prompts: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fit global nonnegative reviewer weights using the correct response."""
    n_reviewers = features.shape[-1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = features[prompts] @ weights
        normalizer = logsumexp(logits, axis=1)
        loss = np.mean(normalizer - logits[:, 0])
        loss += 0.5 * regularization * float(weights @ weights)
        probabilities = np.exp(logits - normalizer[:, None])
        probabilities[:, 0] -= 1.0
        gradient = np.einsum(
            "ec,ecn->n", probabilities, features[prompts]
        ) / len(prompts)
        gradient += regularization * weights
        return float(loss), gradient

    result = minimize(
        objective,
        np.full(n_reviewers, 0.1),
        jac=True,
        bounds=[(0.0, None)] * n_reviewers,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"nonnegative-weight fit failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if np.min(weights) < -TOLERANCE or np.sum(weights) <= TOLERANCE:
        raise RuntimeError("nonnegative-weight fit returned invalid weights")
    return weights / np.sum(weights)


def endpoint_metrics(
    accepted: np.ndarray,
    prompts: np.ndarray,
) -> dict[str, float]:
    """Correct authorization and rejection of all three incorrect endpoints."""
    selected = accepted[prompts]
    return {
        "pure_response_completeness": float(np.mean(selected[:, 0])),
        "pure_response_soundness": float(
            np.mean(np.all(~selected[:, 1:], axis=1))
        ),
    }


def cardinal_rows(
    gains: np.ndarray,
    validation_prompts: np.ndarray,
    test_prompts: np.ndarray,
) -> list[dict[str, object]]:
    """Evaluate validation-defined thresholds on validation and test prompts."""
    thresholds = np.concatenate(
        [
            np.asarray([np.inf]),
            np.sort(np.unique(gains[validation_prompts].ravel()))[::-1],
            np.asarray([-np.inf]),
        ]
    )
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        accepted = gains >= threshold - TOLERANCE
        rows.append(
            {
                "threshold": (
                    float(threshold)
                    if np.isfinite(threshold)
                    else ("reject_all" if threshold > 0 else "accept_all")
                ),
                "validation": endpoint_metrics(accepted, validation_prompts),
                "test": endpoint_metrics(accepted, test_prompts),
            }
        )
    validation_soundness = np.asarray(
        [row["validation"]["pure_response_soundness"] for row in rows]
    )
    validation_completeness = np.asarray(
        [row["validation"]["pure_response_completeness"] for row in rows]
    )
    if np.any(np.diff(validation_soundness) > TOLERANCE):
        raise RuntimeError("soundness is not monotone as threshold decreases")
    if np.any(np.diff(validation_completeness) < -TOLERANCE):
        raise RuntimeError("completeness is not monotone as threshold decreases")
    return rows


def count_rows(
    features: np.ndarray,
    validation_prompts: np.ndarray,
    test_prompts: np.ndarray,
) -> list[dict[str, object]]:
    """Evaluate anonymous objection thresholds on the same held-out split."""
    objections = np.sum(features < -TOLERANCE, axis=2)
    n_reviewers = features.shape[2]
    rows: list[dict[str, object]] = []
    for k in range(-1, n_reviewers + 1):
        accepted = (
            np.zeros_like(objections, dtype=bool)
            if k < 0
            else objections <= k
        )
        rows.append(
            {
                "k": k,
                "validation": endpoint_metrics(accepted, validation_prompts),
                "test": endpoint_metrics(accepted, test_prompts),
            }
        )
    return rows


def select_operating_points(
    rows: list[dict[str, object]],
    parameter: str,
) -> dict[str, dict[str, object] | None]:
    """Select maximum validation coverage under each validation safety target."""
    selected: dict[str, dict[str, object] | None] = {}
    for target in TARGET_SOUNDNESS:
        feasible = [
            row
            for row in rows
            if row["validation"]["pure_response_soundness"]
            >= target - TOLERANCE
        ]
        if not feasible:
            selected[f"{target:.2f}"] = None
            continue
        best = max(
            feasible,
            key=lambda row: (
                row["validation"]["pure_response_completeness"],
                row["validation"]["pure_response_soundness"],
            ),
        )
        selected[f"{target:.2f}"] = {
            parameter: best[parameter],
            "validation": best["validation"],
            "test": best["test"],
        }
    return selected


def full_domain_metrics(
    gains: np.ndarray,
    threshold: float,
    test_prompts: np.ndarray,
) -> dict[str, float | int]:
    """Audit a cardinal threshold over every response lottery."""
    if threshold == np.inf:
        return {
            "test_prompts": int(len(test_prompts)),
            "full_domain_sound_prompts": int(len(test_prompts)),
            "full_domain_soundness_rate": 1.0,
            "full_domain_complete_prompts": 0,
            "full_domain_completeness_rate": 0.0,
            "median_worst_case_loss": 0.0,
            "maximum_worst_case_loss": 0.0,
        }
    if threshold == -np.inf:
        return {
            "test_prompts": int(len(test_prompts)),
            "full_domain_sound_prompts": 0,
            "full_domain_soundness_rate": 0.0,
            "full_domain_complete_prompts": int(len(test_prompts)),
            "full_domain_completeness_rate": 1.0,
            "median_worst_case_loss": 0.25,
            "maximum_worst_case_loss": 0.25,
        }
    maximum_losses = []
    for prompt in test_prompts:
        aggregate = gains[prompt]
        result = linprog(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            A_ub=-aggregate[None, :],
            b_ub=np.asarray([-threshold]),
            A_eq=np.ones((1, 4)),
            b_eq=np.ones(1),
            bounds=[(0.0, 1.0)] * 4,
            method="highs",
        )
        if result.status == 2:
            maximum_losses.append(0.0)
            continue
        if result.status != 0 or result.fun is None:
            raise RuntimeError(f"full-domain audit failed: {result.message}")
        maximum_losses.append(max(0.0, 0.25 - float(result.fun)))
    losses = np.asarray(maximum_losses)
    completeness_rate = cardinal_full_domain_completeness_rate(
        gains, threshold, test_prompts
    )
    return {
        "test_prompts": int(len(test_prompts)),
        "full_domain_sound_prompts": int(np.sum(losses <= 1e-8)),
        "full_domain_soundness_rate": float(np.mean(losses <= 1e-8)),
        "full_domain_complete_prompts": int(
            round(completeness_rate * len(test_prompts))
        ),
        "full_domain_completeness_rate": completeness_rate,
        "median_worst_case_loss": float(np.median(losses)),
        "maximum_worst_case_loss": float(np.max(losses)),
    }


def cardinal_full_domain_soundness_rate(
    gains: np.ndarray,
    threshold: float,
    prompts: np.ndarray,
) -> float:
    """Closed-form full-lottery soundness rate for one score threshold."""
    if threshold == np.inf:
        return 1.0
    if threshold == -np.inf:
        return 0.0
    selected = gains[prompts]
    correct_gain = selected[:, 0]
    best_incorrect_gain = np.max(selected[:, 1:], axis=1)
    maximum_gain = np.maximum(correct_gain, best_incorrect_gain)
    infeasible = maximum_gain < threshold - TOLERANCE
    min_correct_probability = np.zeros(len(prompts))
    needs_correct_mass = (~infeasible) & (
        best_incorrect_gain < threshold - TOLERANCE
    )
    denominator = correct_gain - best_incorrect_gain
    if np.any(denominator[needs_correct_mass] <= TOLERANCE):
        raise RuntimeError("invalid cardinal full-domain frontier geometry")
    min_correct_probability[needs_correct_mass] = (
        threshold - best_incorrect_gain[needs_correct_mass]
    ) / denominator[needs_correct_mass]
    sound = infeasible | (min_correct_probability >= 0.25 - TOLERANCE)
    return float(np.mean(sound))


def cardinal_full_domain_completeness_rate(
    gains: np.ndarray,
    threshold: float,
    prompts: np.ndarray,
) -> float:
    """Rate of prompts on which every principal-beneficial lottery is accepted."""
    if threshold == np.inf:
        return 0.0
    if threshold == -np.inf:
        return 1.0
    selected = gains[prompts]
    correct_gain = selected[:, 0]
    worst_incorrect_gain = np.min(selected[:, 1:], axis=1)
    # Over q_1 >= 1/4, a linear score is minimized either at the pure correct
    # response or at the boundary lottery placing all remaining mass on the
    # lowest-scored incorrect response.
    minimum_beneficial_gain = np.minimum(
        correct_gain,
        0.25 * correct_gain + 0.75 * worst_incorrect_gain,
    )
    return float(
        np.mean(minimum_beneficial_gain >= threshold - TOLERANCE)
    )


def choice_accuracy(gains: np.ndarray, prompts: np.ndarray) -> float:
    """Choice accuracy with ties split uniformly rather than by array order."""
    selected = gains[prompts]
    maxima = np.max(selected, axis=1, keepdims=True)
    tied = np.isclose(selected, maxima, atol=TOLERANCE, rtol=0.0)
    return float(np.mean(tied[:, 0] / np.sum(tied, axis=1)))


def exact_weighted_binary_audit(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return harmful and beneficial approval extrema and exact depth.

    A harmful lottery has negative principal gain.  Reversing its direction
    and fixing principal gain to one leaves a two-dimensional affine
    hyperplane arrangement.  Reviewer i objects exactly where its affine
    evaluation is strictly positive.  The least total objection weight over
    all arrangement faces therefore gives one minus the greatest approval
    weight attained by any harmful lottery.  Ties count as approvals.
    """
    weights = np.asarray(weights, dtype=float)
    if np.min(weights) < -TOLERANCE or not np.isclose(
        np.sum(weights), 1.0, atol=TOLERANCE
    ):
        raise ValueError("binary weights must be nonnegative and sum to one")
    maximum_harmful_approval = np.empty(len(prompts))
    minimum_beneficial_approval = np.empty(len(prompts))
    exact_depth = np.empty(len(prompts), dtype=int)
    maximum_beneficial_objections = np.empty(len(prompts), dtype=int)
    for position, prompt in enumerate(prompts):
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewer_vectors, principal_vector = multicandidate_utilities(
            scores, start, stop, 1
        )
        evaluations = standard_angular_evaluations(
            reviewer_vectors, principal_vector
        )
        cell_evaluations = standard_angular_cell_evaluations(
            reviewer_vectors, principal_vector
        )
        objections = evaluations > 1e-8
        objection_weight = weights @ objections
        maximum_harmful_approval[position] = 1.0 - float(
            np.min(objection_weight)
        )
        beneficial_objections = cell_evaluations < -1e-8
        beneficial_approval_weight = 1.0 - weights @ beneficial_objections
        minimum_beneficial_approval[position] = float(
            np.min(beneficial_approval_weight)
        )
        exact_depth[position] = int(np.min(np.sum(objections, axis=0)))
        maximum_beneficial_objections[position] = int(
            np.max(np.sum(beneficial_objections, axis=0))
        )
    return (
        maximum_harmful_approval,
        minimum_beneficial_approval,
        exact_depth,
        maximum_beneficial_objections,
    )


def exact_topic_weighted_binary_audit(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
    subsets: np.ndarray,
    weights_by_topic: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Audit a topic-routed weighted-binary rule without sampling lotteries."""
    maximum_harmful_approval = np.empty(len(prompts))
    minimum_beneficial_approval = np.empty(len(prompts))
    exact_depth = np.empty(len(prompts), dtype=int)
    maximum_beneficial_objections = np.empty(len(prompts), dtype=int)
    assigned = np.zeros(len(prompts), dtype=bool)
    prompt_topics = subsets[prompts]
    for topic in TOPICS:
        positions = np.flatnonzero(prompt_topics == topic)
        if len(positions) == 0:
            raise RuntimeError(f"topic {topic} has no prompts to audit")
        topic_prompts = prompts[positions]
        (
            maximum_harmful_approval[positions],
            minimum_beneficial_approval[positions],
            exact_depth[positions],
            maximum_beneficial_objections[positions],
        ) = exact_weighted_binary_audit(
            scores,
            offsets,
            topic_prompts,
            weights_by_topic[topic],
        )
        assigned[positions] = True
    if not np.all(assigned):
        unexpected = np.unique(prompt_topics[~assigned]).tolist()
        raise RuntimeError(f"unrouted weighted-binary topics: {unexpected}")
    return (
        maximum_harmful_approval,
        minimum_beneficial_approval,
        exact_depth,
        maximum_beneficial_objections,
    )


def weighted_binary_full_domain_soundness_rate(
    maximum_harmful_approval: np.ndarray,
    threshold: float,
) -> float:
    """Exact rate of prompts on which no harmful lottery is authorized."""
    if threshold == np.inf:
        return 1.0
    if threshold == -np.inf:
        return 0.0
    return float(
        np.mean(maximum_harmful_approval < threshold - TOLERANCE)
    )


def weighted_binary_full_domain_completeness_rate(
    minimum_beneficial_approval: np.ndarray,
    threshold: float,
) -> float:
    """Exact rate of prompts on which every beneficial lottery is authorized."""
    if threshold == np.inf:
        return 0.0
    if threshold == -np.inf:
        return 1.0
    return float(
        np.mean(minimum_beneficial_approval >= threshold - TOLERANCE)
    )


def adjacent_cell_maximum_objection_weight(
    reviewers: np.ndarray,
    deviation: np.ndarray,
    facet: int,
    gains: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Largest objection weight in a facet cell incident to one vertex."""
    tied = np.flatnonzero(np.abs(gains) <= 5 * GEOMETRY_TOLERANCE)
    base = float(np.sum(weights[gains < -5 * GEOMETRY_TOLERANCE]))
    q = q_coordinates(deviation)[0]
    active = [
        index
        for index in range(4)
        if index != facet and q[index] <= 5 * GEOMETRY_TOLERANCE
    ]

    if tied.size == 0:
        return base
    if not active and tied.size <= 2:
        return base + float(np.sum(weights[tied]))
    if len(active) == 1 and tied.size == 1:
        return base + float(weights[tied[0]])

    basis = null_space(FACET_NORMALS[facet][None, :])
    boundary_angles: list[float] = []
    for row in list(reviewers[tied]) + [Q_GRADIENTS[index] for index in active]:
        coefficients = row @ basis
        if np.linalg.norm(coefficients) <= GEOMETRY_TOLERANCE:
            continue
        angle = float(
            np.arctan2(coefficients[0], -coefficients[1]) % (2 * np.pi)
        )
        boundary_angles.extend([angle, (angle + np.pi) % (2 * np.pi)])

    if boundary_angles:
        values = np.unique(np.round(boundary_angles, 12))
        candidate_angles = [
            float((left + right) / 2 % (2 * np.pi))
            for left, right in zip(
                values,
                np.r_[values[1:], values[0] + 2 * np.pi],
            )
        ]
    else:
        candidate_angles = [0.0]

    maximum = base
    for angle in candidate_angles:
        direction_2d = np.asarray([np.cos(angle), np.sin(angle)])
        direction = basis @ direction_2d
        if any(
            Q_GRADIENTS[index] @ direction <= GEOMETRY_TOLERANCE
            for index in active
        ):
            continue
        maximum = max(
            maximum,
            base
            + float(
                np.sum(
                    weights[tied][
                        reviewers[tied] @ direction < -GEOMETRY_TOLERANCE
                    ]
                )
            ),
        )
    return maximum


def _prefix_maximum(
    keys: np.ndarray,
    values: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    cumulative = np.maximum.accumulate(values[order])
    positions = np.searchsorted(sorted_keys, upper_bounds, side="right")
    output = np.zeros(len(upper_bounds))
    usable = positions > 0
    output[usable] = cumulative[positions[usable] - 1]
    return output


def _suffix_maximum(
    keys: np.ndarray,
    values: np.ndarray,
    lower_bounds: np.ndarray,
    empty_value: float = 0.0,
) -> np.ndarray:
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    suffix = np.maximum.accumulate(values[order][::-1])[::-1]
    positions = np.searchsorted(sorted_keys, lower_bounds, side="right")
    output = np.full(len(lower_bounds), empty_value)
    usable = positions < len(sorted_keys)
    output[usable] = suffix[positions[usable]]
    return output


def weighted_binary_prompt_errors(
    reviewers: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact full-domain errors for every threshold of a weighted binary rule."""
    vertex_objections: list[float] = []
    adjacent_objections: list[float] = []
    principal_gains: list[float] = []
    for facet in range(4):
        vertices = facet_vertices(reviewers, facet)
        evaluations = reviewers @ vertices.T
        for position, deviation in enumerate(vertices):
            gains = evaluations[:, position]
            vertex_objections.append(
                float(np.sum(weights[gains < -GEOMETRY_TOLERANCE]))
            )
            adjacent_objections.append(
                adjacent_cell_maximum_objection_weight(
                    reviewers,
                    deviation,
                    facet,
                    gains,
                    weights,
                )
            )
            principal_gains.append(float(deviation[0]))

    vertex_objections_array = np.asarray(vertex_objections)
    adjacent_objections_array = np.asarray(adjacent_objections)
    principal_gains_array = np.asarray(principal_gains)
    finite = np.isfinite(thresholds)
    objection_bounds = 1.0 - thresholds[finite] + TOLERANCE

    soundness_error = np.zeros(len(thresholds))
    completeness_error = np.zeros(len(thresholds))
    completeness_attained = np.zeros(len(thresholds), dtype=bool)
    soundness_error[finite] = _prefix_maximum(
        vertex_objections_array,
        np.maximum(0.0, -principal_gains_array),
        objection_bounds,
    )
    completeness_error[finite] = _suffix_maximum(
        adjacent_objections_array,
        np.maximum(0.0, principal_gains_array),
        objection_bounds,
    )
    attained_gain = _suffix_maximum(
        vertex_objections_array,
        principal_gains_array,
        objection_bounds,
        empty_value=-np.inf,
    )
    completeness_attained[finite] = (
        attained_gain
        >= completeness_error[finite] - 5 * GEOMETRY_TOLERANCE
    )

    reject_all = np.isposinf(thresholds)
    accept_all = np.isneginf(thresholds)
    completeness_error[reject_all] = 0.75
    completeness_attained[reject_all] = True
    soundness_error[accept_all] = 0.25
    return soundness_error, completeness_error, completeness_attained


def topic_weighted_binary_prompt_errors(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
    subsets: np.ndarray,
    weights_by_topic: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact prompt errors for a shared threshold and topic-routed weights."""
    soundness_error = np.zeros((len(prompts), len(thresholds)))
    completeness_error = np.zeros_like(soundness_error)
    completeness_attained = np.zeros_like(soundness_error, dtype=bool)
    for position, prompt in enumerate(prompts):
        topic = str(subsets[prompt])
        if topic not in weights_by_topic:
            raise RuntimeError(f"no weighted-binary fit for topic {topic}")
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, _ = multicandidate_utilities(scores, start, stop, 1)
        (
            soundness_error[position],
            completeness_error[position],
            completeness_attained[position],
        ) = weighted_binary_prompt_errors(
            reviewers,
            weights_by_topic[topic],
            thresholds,
        )
        if (position + 1) % 50 == 0:
            print(
                "topic-conditioned weighted-binary approximate audit "
                f"{position + 1}/{len(prompts)}"
            )
    return soundness_error, completeness_error, completeness_attained


def cardinal_prompt_errors(
    gains: np.ndarray,
    prompts: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact full-domain errors for every threshold of one cardinal rule."""
    selected = gains[prompts]
    correct = selected[:, 0]
    best_incorrect = np.max(selected[:, 1:], axis=1)
    worst_incorrect = np.min(selected[:, 1:], axis=1)
    soundness_error = np.zeros((len(prompts), len(thresholds)))
    completeness_error = np.zeros_like(soundness_error)
    completeness_attained = np.zeros_like(soundness_error, dtype=bool)

    for column, threshold in enumerate(thresholds):
        if np.isposinf(threshold):
            completeness_error[:, column] = 0.75
            completeness_attained[:, column] = True
            continue
        if np.isneginf(threshold):
            soundness_error[:, column] = 0.25
            continue

        maximum_score = np.maximum(correct, best_incorrect)
        infeasible = maximum_score < threshold - TOLERANCE
        minimum_correct = np.zeros(len(prompts))
        needs_correct = (~infeasible) & (
            best_incorrect < threshold - TOLERANCE
        )
        denominator = correct - best_incorrect
        if np.any(denominator[needs_correct] <= TOLERANCE):
            raise RuntimeError("invalid approximate cardinal soundness geometry")
        minimum_correct[needs_correct] = (
            threshold - best_incorrect[needs_correct]
        ) / denominator[needs_correct]
        minimum_correct = np.clip(minimum_correct, 0.0, 1.0)
        soundness_error[:, column] = np.maximum(
            0.0,
            0.25 - minimum_correct,
        )
        soundness_error[infeasible, column] = 0.0

        pure_rejected = correct < threshold - TOLERANCE
        completeness_error[pure_rejected, column] = 0.75
        completeness_attained[pure_rejected, column] = True
        crossing = (correct >= threshold - TOLERANCE) & (
            worst_incorrect < threshold - TOLERANCE
        )
        crossing_denominator = correct - worst_incorrect
        if np.any(crossing_denominator[crossing] <= TOLERANCE):
            raise RuntimeError("invalid approximate cardinal completeness geometry")
        boundary_correct = np.zeros(len(prompts))
        boundary_correct[crossing] = (
            threshold - worst_incorrect[crossing]
        ) / crossing_denominator[crossing]
        boundary_correct = np.clip(boundary_correct, 0.0, 1.0)
        completeness_error[crossing, column] = np.maximum(
            0.0,
            boundary_correct[crossing] - 0.25,
        )
        # Rejection uses a strict score inequality.  Hence a supremum attained
        # on the score boundary is not itself rejected.
        completeness_attained[crossing, column] = False
    return soundness_error, completeness_error, completeness_attained


def ranked_single_reviewer_errors(
    features: np.ndarray,
    prompts: np.ndarray,
    reviewer_ranking: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Audit native binary approval for each validation-ranked reviewer.

    A single reviewer's native approval set is the halfspace with cardinal
    threshold zero, so the closed-form cardinal audit is exact here.  Columns
    follow validation rank rather than the archive's reviewer order.
    """
    n_reviewers = features.shape[2]
    if not np.array_equal(
        np.sort(reviewer_ranking), np.arange(n_reviewers)
    ):
        raise ValueError("reviewer ranking is not a permutation")
    soundness_error = np.empty((len(prompts), n_reviewers))
    completeness_error = np.empty_like(soundness_error)
    completeness_attained = np.empty_like(soundness_error, dtype=bool)
    for column, reviewer in enumerate(reviewer_ranking):
        soundness, completeness, attained = cardinal_prompt_errors(
            features[:, :, reviewer],
            prompts,
            np.asarray([0.0]),
        )
        soundness_error[:, column] = soundness[:, 0]
        completeness_error[:, column] = completeness[:, 0]
        completeness_attained[:, column] = attained[:, 0]
    return soundness_error, completeness_error, completeness_attained


def ranked_prefix_unanimity_errors(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
    reviewer_ranking: np.ndarray,
    single_completeness_error: np.ndarray,
    single_completeness_attained: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Audit unanimous native approval for every ranked reviewer prefix.

    Prefix ``m`` contains the first ``m`` reviewers in validation rank and
    authorizes a lottery only when all of them approve.  Soundness is evaluated
    on every arrangement vertex on the simplex boundary.  Rejection by a
    unanimous panel is the union of its members' rejection regions, so its
    completeness error is the cumulative maximum of the exact single-reviewer
    errors, with attainment propagated at ties.
    """
    n_reviewers = len(reviewer_ranking)
    if single_completeness_error.shape != (len(prompts), n_reviewers):
        raise ValueError("single-reviewer completeness array has wrong shape")
    if single_completeness_attained.shape != (
        len(prompts),
        n_reviewers,
    ):
        raise ValueError("single-reviewer attainment array has wrong shape")

    soundness_error = np.zeros((len(prompts), n_reviewers))
    rank_positions = np.empty(n_reviewers, dtype=int)
    rank_positions[reviewer_ranking] = np.arange(n_reviewers)
    for position, prompt in enumerate(prompts):
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, _ = multicandidate_utilities(scores, start, stop, 1)
        first_objector_rank: list[np.ndarray] = []
        principal_losses: list[np.ndarray] = []
        for facet in range(4):
            vertices = facet_vertices(reviewers, facet)
            evaluations = reviewers @ vertices.T
            objector_ranks = np.where(
                evaluations < -GEOMETRY_TOLERANCE,
                rank_positions[:, None],
                n_reviewers,
            )
            first_objector_rank.append(np.min(objector_ranks, axis=0))
            principal_losses.append(np.maximum(0.0, -vertices[:, 0]))
        first_objector = np.concatenate(first_objector_rank)
        losses = np.concatenate(principal_losses)
        for column in range(n_reviewers):
            authorized_vertices = first_objector > column
            soundness_error[position, column] = float(
                np.max(losses[authorized_vertices], initial=0.0)
            )
        if (position + 1) % 50 == 0:
            print(
                "ranked-prefix binary audit "
                f"{position + 1}/{len(prompts)}"
            )

    completeness_error = np.zeros_like(single_completeness_error)
    completeness_attained = np.zeros_like(
        single_completeness_attained,
        dtype=bool,
    )
    running_error = np.zeros(len(prompts))
    running_attained = np.zeros(len(prompts), dtype=bool)
    for column in range(n_reviewers):
        candidate_error = single_completeness_error[:, column]
        candidate_attained = single_completeness_attained[:, column]
        greater = candidate_error > (
            running_error + 5 * GEOMETRY_TOLERANCE
        )
        equal = np.isclose(
            candidate_error,
            running_error,
            atol=5 * GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
        running_attained = np.where(
            greater,
            candidate_attained,
            running_attained | (equal & candidate_attained),
        )
        running_error = np.maximum(running_error, candidate_error)
        completeness_error[:, column] = running_error
        completeness_attained[:, column] = running_attained
    return soundness_error, completeness_error, completeness_attained


def epsilon_complete(
    errors: np.ndarray,
    attained: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    below = errors < epsilon - 5 * GEOMETRY_TOLERANCE
    equal_unattained = np.isclose(
        errors,
        epsilon,
        atol=5 * GEOMETRY_TOLERANCE,
        rtol=0.0,
    ) & ~attained
    return below | equal_unattained


def train_test_cardinal_sweep(
    gains: np.ndarray,
    training_prompts: np.ndarray,
    test_prompts: np.ndarray,
) -> tuple[
    list[dict[str, object]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Evaluate one train-defined score-threshold sweep on train and test."""
    thresholds = np.concatenate(
        [
            np.asarray([np.inf]),
            np.sort(np.unique(gains[training_prompts].ravel()))[::-1],
            np.asarray([-np.inf]),
        ]
    )
    split_errors = {}
    test_error_data = None
    for split, prompts in (
        ("train", training_prompts),
        ("test", test_prompts),
    ):
        soundness, completeness, attained = cardinal_prompt_errors(
            gains,
            prompts,
            thresholds,
        )
        split_errors[split] = {
            "soundness": np.mean(
                soundness <= 5 * GEOMETRY_TOLERANCE,
                axis=0,
            ),
            "completeness": np.mean(
                epsilon_complete(completeness, attained, 0.0),
                axis=0,
            ),
            "epsilon_good_completeness": np.mean(
                epsilon_complete(
                    completeness,
                    attained,
                    APPROXIMATE_GOOD,
                ),
                axis=0,
            ),
        }
        if split == "test":
            test_error_data = (soundness, completeness, attained)

    rows: list[dict[str, object]] = []
    for column, threshold in enumerate(thresholds):
        row: dict[str, object] = {
            "threshold": (
                float(threshold)
                if np.isfinite(threshold)
                else ("reject_all" if threshold > 0 else "accept_all")
            )
        }
        for split in ("train", "test"):
            row[split] = {
                "full_domain_soundness": float(
                    split_errors[split]["soundness"][column]
                ),
                "full_domain_completeness": float(
                    split_errors[split]["completeness"][column]
                ),
                "epsilon_good_completeness": float(
                    split_errors[split]["epsilon_good_completeness"][column]
                ),
            }
        rows.append(row)

    for split in ("train", "test"):
        soundness = np.asarray(
            [row[split]["full_domain_soundness"] for row in rows]
        )
        approximate = np.asarray(
            [row[split]["epsilon_good_completeness"] for row in rows]
        )
        if np.any(np.diff(soundness) > TOLERANCE):
            raise RuntimeError(
                f"{split} singleton soundness sweep is nonmonotone"
            )
        if np.any(np.diff(approximate) < -TOLERANCE):
            raise RuntimeError(
                f"{split} singleton completeness sweep is nonmonotone"
            )
    if test_error_data is None:
        raise RuntimeError("train-defined singleton sweep omitted test errors")
    return rows, test_error_data


def train_test_cardinal_fixed_checkpoints(
    gains: np.ndarray,
    training_prompts: np.ndarray,
    test_prompts: np.ndarray,
    thresholds: tuple[float, ...] = (0.45, 0.55),
) -> list[dict[str, object]]:
    """Evaluate predeclared matched thresholds directly on train and test."""
    threshold_array = np.asarray(thresholds, dtype=float)
    checkpoints: list[dict[str, object]] = [
        {"threshold": float(threshold)} for threshold in threshold_array
    ]
    for split, prompts in (
        ("train", training_prompts),
        ("test", test_prompts),
    ):
        soundness, completeness, attained = cardinal_prompt_errors(
            gains,
            prompts,
            threshold_array,
        )
        soundness_rates = np.mean(
            soundness <= 5 * GEOMETRY_TOLERANCE,
            axis=0,
        )
        completeness_rates = np.mean(
            epsilon_complete(
                completeness,
                attained,
                APPROXIMATE_GOOD,
            ),
            axis=0,
        )
        for column, checkpoint in enumerate(checkpoints):
            checkpoint[split] = {
                "full_domain_soundness": float(soundness_rates[column]),
                "epsilon_good_completeness": float(
                    completeness_rates[column]
                ),
            }
    return checkpoints


def validation_behavior_thresholds(
    gains: np.ndarray,
    prompts: np.ndarray,
) -> np.ndarray:
    """Enumerate every validation region of the two Boolean guarantees."""
    selected = gains[prompts]
    if not np.all(np.isfinite(selected)):
        raise RuntimeError("threshold calibration received nonfinite gains")
    correct = selected[:, 0]
    best_incorrect = np.max(selected[:, 1:], axis=1)
    worst_incorrect = np.min(selected[:, 1:], axis=1)
    predicate_tolerance = 5 * GEOMETRY_TOLERANCE

    roots = [
        np.maximum(correct, best_incorrect) + TOLERANCE,
        best_incorrect + TOLERANCE,
        correct + TOLERANCE,
        worst_incorrect + TOLERANCE,
    ]
    sound_crossing = correct > best_incorrect
    roots.append(
        best_incorrect[sound_crossing]
        + (correct[sound_crossing] - best_incorrect[sound_crossing])
        * (0.25 - predicate_tolerance)
    )
    complete_crossing = correct > worst_incorrect
    roots.append(
        worst_incorrect[complete_crossing]
        + (correct[complete_crossing] - worst_incorrect[complete_crossing])
        * (0.25 + APPROXIMATE_GOOD + predicate_tolerance)
    )
    critical = np.sort(np.unique(np.concatenate(roots)))
    if not np.all(np.isfinite(critical)):
        raise RuntimeError("threshold calibration produced nonfinite roots")

    candidates = {float(np.inf), float(-np.inf)}
    for value in critical:
        candidates.add(float(value))
        candidates.add(float(np.nextafter(value, -np.inf)))
        candidates.add(float(np.nextafter(value, np.inf)))
    for lower, upper in zip(critical[:-1], critical[1:]):
        midpoint = lower + (upper - lower) / 2
        if not lower < midpoint < upper:
            midpoint = np.nextafter(lower, upper)
        if lower < midpoint < upper:
            candidates.add(float(midpoint))
    return np.asarray(sorted(candidates, reverse=True), dtype=float)


def serialized_threshold(threshold: float) -> float | str:
    if np.isposinf(threshold):
        return "reject_all"
    if np.isneginf(threshold):
        return "accept_all"
    return float(threshold)


def numeric_threshold(threshold: float | str) -> float:
    if threshold == "reject_all":
        return float(np.inf)
    if threshold == "accept_all":
        return float(-np.inf)
    return float(threshold)


def threshold_count_summary(
    prompts: int,
    sound_prompts: int,
    complete_prompts: int,
) -> dict[str, object]:
    return {
        "prompts": int(prompts),
        "sound_prompts": int(sound_prompts),
        "unsound_prompts": int(prompts - sound_prompts),
        "epsilon_good_complete_prompts": int(complete_prompts),
        "full_domain_soundness": float(sound_prompts / prompts),
        "epsilon_good_completeness": float(complete_prompts / prompts),
    }


def validation_threshold_options(
    gains: np.ndarray,
    validation_prompts: np.ndarray,
    required_thresholds: np.ndarray | None = None,
) -> dict[str, object]:
    """Build validation-only threshold profiles and Pareto-compress them.

    ``required_thresholds`` is used to embed the pooled shared grid literally
    in every topic-specific grid.  When an added threshold has the same
    validation behavior as a canonical topic-local representative, the local
    representative wins the deterministic tie-break.  This preserves the
    original behaviorally exhaustive grid's held-out representative while
    making the shared policy a strict subset of the raw vector search space.
    """
    canonical_thresholds = validation_behavior_thresholds(
        gains,
        validation_prompts,
    )
    canonical_identities = {float(value) for value in canonical_thresholds}
    if required_thresholds is None:
        thresholds = canonical_thresholds
    else:
        required = np.asarray(required_thresholds, dtype=float)
        if required.ndim != 1 or not np.all(
            np.isfinite(required) | np.isinf(required)
        ):
            raise RuntimeError("invalid required threshold grid")
        thresholds = np.sort(
            np.unique(np.concatenate((canonical_thresholds, required)))
        )[::-1]
    soundness, completeness, attained = cardinal_prompt_errors(
        gains,
        validation_prompts,
        thresholds,
    )
    sound = soundness <= 5 * GEOMETRY_TOLERANCE
    complete = epsilon_complete(
        completeness,
        attained,
        APPROXIMATE_GOOD,
    )

    best_by_failure: dict[int, dict[str, object]] = {}
    for column, threshold in enumerate(thresholds):
        validation_sound = int(
            np.sum(sound[:, column])
        )
        validation_complete = int(
            np.sum(complete[:, column])
        )
        failure_count = len(validation_prompts) - validation_sound
        option = {
            "threshold": float(threshold),
            "canonical_local_candidate": (
                float(threshold) in canonical_identities
            ),
            "validation": threshold_count_summary(
                len(validation_prompts),
                validation_sound,
                validation_complete,
            ),
        }
        incumbent = best_by_failure.get(failure_count)
        if incumbent is None or (
            validation_complete,
            option["canonical_local_candidate"],
            float(threshold),
        ) > (
            incumbent["validation"]["epsilon_good_complete_prompts"],
            incumbent["canonical_local_candidate"],
            incumbent["threshold"],
        ):
            best_by_failure[failure_count] = option

    pareto_options = []
    best_complete = -1
    for failure_count in sorted(best_by_failure):
        option = best_by_failure[failure_count]
        complete = option["validation"]["epsilon_good_complete_prompts"]
        if complete > best_complete:
            pareto_options.append(option)
            best_complete = complete
    return {
        "candidate_threshold_count": int(len(thresholds)),
        "canonical_local_candidate_threshold_count": int(
            len(canonical_thresholds)
        ),
        "required_candidate_threshold_count": int(
            0 if required_thresholds is None else len(required_thresholds)
        ),
        "candidate_thresholds": [
            serialized_threshold(threshold) for threshold in thresholds
        ],
        "pareto_option_count": int(len(pareto_options)),
        "pareto_validation_profiles": [
            {
                "threshold": serialized_threshold(option["threshold"]),
                "canonical_local_candidate": bool(
                    option["canonical_local_candidate"]
                ),
                "unsound_prompts": option["validation"]["unsound_prompts"],
                "epsilon_good_complete_prompts": option["validation"][
                    "epsilon_good_complete_prompts"
                ],
            }
            for option in pareto_options
        ],
        "options": pareto_options,
    }


def shared_threshold_budget_rows(
    options: list[dict[str, object]],
    topics: tuple[str, ...],
    maximum_budget: int,
) -> list[dict[str, object]]:
    """Select one shared threshold under every validation failure budget."""
    rows = []
    for budget in range(maximum_budget + 1):
        feasible = [
            option
            for option in options
            if option["validation"]["unsound_prompts"] <= budget
        ]
        if not feasible:
            raise RuntimeError("shared threshold has no feasible reject-all rule")
        selected = max(
            feasible,
            key=lambda option: (
                option["validation"]["epsilon_good_complete_prompts"],
                -option["validation"]["unsound_prompts"],
                option["threshold"],
            ),
        )
        threshold = float(selected["threshold"])
        rows.append(
            {
                "validation_failure_budget": int(budget),
                "used_validation_failures": int(
                    selected["validation"]["unsound_prompts"]
                ),
                "threshold": serialized_threshold(threshold),
                "thresholds_by_topic": {
                    topic: serialized_threshold(threshold) for topic in topics
                },
                "validation": selected["validation"],
            }
        )
    return rows


def topic_threshold_budget_rows(
    options_by_topic: dict[str, list[dict[str, object]]],
    validation_counts: dict[str, int],
    topics: tuple[str, ...],
    maximum_budget: int,
) -> list[dict[str, object]]:
    """Solve the exact multiple-choice threshold allocation for all budgets."""
    states: dict[int, dict[str, object]] = {
        0: {
            "validation_completions": 0,
            "selected_options": [],
        }
    }
    for topic in topics:
        next_states: dict[int, dict[str, object]] = {}
        for used_failures, state in states.items():
            for option in options_by_topic[topic]:
                new_failures = (
                    used_failures
                    + option["validation"]["unsound_prompts"]
                )
                candidate = {
                    "validation_completions": (
                        state["validation_completions"]
                        + option["validation"][
                            "epsilon_good_complete_prompts"
                        ]
                    ),
                    "selected_options": state["selected_options"] + [option],
                }
                candidate_key = (
                    candidate["validation_completions"],
                    tuple(
                        option_row["threshold"]
                        for option_row in candidate["selected_options"]
                    ),
                )
                incumbent = next_states.get(new_failures)
                if incumbent is None:
                    next_states[new_failures] = candidate
                    continue
                incumbent_options = incumbent["selected_options"]
                incumbent_key = (
                    incumbent["validation_completions"],
                    tuple(
                        option_row["threshold"]
                        for option_row in incumbent_options
                    ),
                )
                if candidate_key > incumbent_key:
                    next_states[new_failures] = candidate
        states = next_states

    rows = []
    for budget in range(maximum_budget + 1):
        feasible = [
            (used_failures, state)
            for used_failures, state in states.items()
            if used_failures <= budget
        ]
        if not feasible:
            raise RuntimeError("topic threshold vector omits reject-all")
        used_failures, selected = max(
            feasible,
            key=lambda item: (
                item[1]["validation_completions"],
                -item[0],
                tuple(
                    option["threshold"]
                    for option in item[1]["selected_options"]
                ),
            ),
        )
        selected_options = selected["selected_options"]
        validation_sound = sum(
            option["validation"]["sound_prompts"]
            for option in selected_options
        )
        validation_complete = sum(
            option["validation"]["epsilon_good_complete_prompts"]
            for option in selected_options
        )
        rows.append(
            {
                "validation_failure_budget": int(budget),
                "used_validation_failures": int(used_failures),
                "thresholds_by_topic": {
                    topic: serialized_threshold(option["threshold"])
                    for topic, option in zip(topics, selected_options)
                },
                "validation": threshold_count_summary(
                    sum(validation_counts.values()),
                    validation_sound,
                    validation_complete,
                ),
            }
        )
    return rows


def attach_frozen_split_metrics(
    gains: np.ndarray,
    subsets: np.ndarray,
    prompts: np.ndarray,
    rows: list[dict[str, object]],
    split: str,
    topics: tuple[str, ...],
) -> None:
    """Evaluate already-selected policies and attach pooled/topic counts."""
    cache: dict[str, dict[float, dict[str, object]]] = {}
    for topic in topics:
        topic_prompts = prompts[subsets[prompts] == topic]
        thresholds = np.unique(
            np.asarray(
                [
                    numeric_threshold(row["thresholds_by_topic"][topic])
                    for row in rows
                ],
                dtype=float,
            )
        )
        soundness, completeness, attained = cardinal_prompt_errors(
            gains,
            topic_prompts,
            thresholds,
        )
        sound = soundness <= 5 * GEOMETRY_TOLERANCE
        complete = epsilon_complete(
            completeness,
            attained,
            APPROXIMATE_GOOD,
        )
        cache[topic] = {
            float(threshold): threshold_count_summary(
                len(topic_prompts),
                int(np.sum(sound[:, column])),
                int(np.sum(complete[:, column])),
            )
            for column, threshold in enumerate(thresholds)
        }

    for row in rows:
        by_topic = {
            topic: cache[topic][
                numeric_threshold(row["thresholds_by_topic"][topic])
            ]
            for topic in topics
        }
        pooled = threshold_count_summary(
            sum(metrics["prompts"] for metrics in by_topic.values()),
            sum(metrics["sound_prompts"] for metrics in by_topic.values()),
            sum(
                metrics["epsilon_good_complete_prompts"]
                for metrics in by_topic.values()
            ),
        )
        if split in row and row[split] != pooled:
            raise RuntimeError(
                f"{split} routed counts disagree with selected profiles"
            )
        row[split] = pooled
        row[f"{split}_by_topic"] = by_topic


def validation_selection_payload(
    output: dict[str, object],
    topics: tuple[str, ...],
) -> dict[str, object]:
    """Return only validation-selected state for a leakage-resistant digest."""
    payload = {
        "candidate_source": output["candidate_source"],
        "score_rule": output["score_rule"],
        "validation_topic_counts": output["validation_topic_counts"],
        "policies": {},
    }
    for policy in ("shared_threshold", "topic_threshold_vector"):
        policy_output = output[policy]
        if policy == "shared_threshold":
            grid_and_profiles = {
                "candidate_thresholds": policy_output[
                    "candidate_thresholds"
                ],
                "pareto_validation_profiles": policy_output[
                    "pareto_validation_profiles"
                ],
            }
        else:
            grid_and_profiles = {
                "candidate_thresholds_by_topic": policy_output[
                    "candidate_thresholds_by_topic"
                ],
                "pareto_validation_profiles_by_topic": policy_output[
                    "pareto_validation_profiles_by_topic"
                ],
            }
        payload["policies"][policy] = {
            **grid_and_profiles,
            "rows": [
                {
                    "validation_failure_budget": row[
                        "validation_failure_budget"
                    ],
                    "used_validation_failures": row[
                        "used_validation_failures"
                    ],
                    "thresholds_by_topic": {
                        topic: row["thresholds_by_topic"][topic]
                        for topic in topics
                    },
                    "validation": row["validation"],
                    "validation_by_topic": {
                        topic: row["validation_by_topic"][topic]
                        for topic in topics
                    },
                }
                for row in policy_output["rows"]
            ],
        }
    return payload


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_validation_budget_thresholds(
    validation_gains: np.ndarray,
    validation_subsets: np.ndarray,
    candidate_source: str,
    score_rule: dict[str, object],
    topics: tuple[str, ...] = TOPICS,
) -> dict[str, object]:
    """Select all budget-indexed policies from validation inputs alone."""
    validation_gains = np.asarray(validation_gains, dtype=float)
    validation_subsets = np.asarray(validation_subsets, dtype=str)
    if validation_gains.ndim != 2 or validation_gains.shape[1] != 4:
        raise ValueError("validation gains must have shape (prompts, 4)")
    if validation_subsets.shape != (len(validation_gains),):
        raise ValueError("validation topic labels do not match score rows")
    local_validation_prompts = np.arange(
        len(validation_gains),
        dtype=int,
    )
    validation_counts = {
        topic: int(np.sum(validation_subsets == topic)) for topic in topics
    }
    maximum_budget = len(local_validation_prompts)
    pooled_thresholds = validation_behavior_thresholds(
        validation_gains,
        local_validation_prompts,
    )
    shared_output = validation_threshold_options(
        validation_gains,
        local_validation_prompts,
    )
    if not np.array_equal(
        np.asarray(
            [numeric_threshold(value) for value in shared_output["candidate_thresholds"]]
        ),
        pooled_thresholds,
    ):
        raise RuntimeError("shared threshold grid reconstruction changed")
    shared_rows = shared_threshold_budget_rows(
        shared_output["options"],
        topics,
        maximum_budget,
    )

    options_by_topic = {}
    topic_outputs = {}
    topic_candidate_counts = {}
    topic_local_candidate_counts = {}
    topic_pareto_counts = {}
    pooled_identities = {float(value) for value in pooled_thresholds}
    for topic in topics:
        topic_prompts = local_validation_prompts[
            validation_subsets == topic
        ]
        topic_output = validation_threshold_options(
            validation_gains,
            topic_prompts,
            required_thresholds=pooled_thresholds,
        )
        topic_identities = {
            numeric_threshold(value)
            for value in topic_output["candidate_thresholds"]
        }
        if not pooled_identities.issubset(topic_identities):
            raise RuntimeError(
                f"shared threshold grid is not embedded for {topic}"
            )
        topic_outputs[topic] = topic_output
        options_by_topic[topic] = topic_output["options"]
        topic_candidate_counts[topic] = topic_output[
            "candidate_threshold_count"
        ]
        topic_local_candidate_counts[topic] = topic_output[
            "canonical_local_candidate_threshold_count"
        ]
        topic_pareto_counts[topic] = topic_output["pareto_option_count"]
    vector_rows = topic_threshold_budget_rows(
        options_by_topic,
        validation_counts,
        topics,
        maximum_budget,
    )

    if any("test" in row for row in shared_rows + vector_rows):
        raise RuntimeError("test metrics were attached before policy selection")
    for rows in (shared_rows, vector_rows):
        attach_frozen_split_metrics(
            validation_gains,
            validation_subsets,
            local_validation_prompts,
            rows,
            "validation",
            topics,
        )

    output = {
        "candidate_source": candidate_source,
        "score_rule": score_rule,
        "validation_topic_counts": validation_counts,
        "shared_threshold": {
            "candidate_threshold_count": shared_output[
                "candidate_threshold_count"
            ],
            "candidate_thresholds": shared_output["candidate_thresholds"],
            "pareto_option_count": shared_output["pareto_option_count"],
            "pareto_validation_profiles": shared_output[
                "pareto_validation_profiles"
            ],
            "rows": shared_rows,
        },
        "topic_threshold_vector": {
            "candidate_threshold_counts_by_topic": topic_candidate_counts,
            "canonical_local_candidate_threshold_counts_by_topic": (
                topic_local_candidate_counts
            ),
            "required_pooled_candidate_threshold_count": int(
                len(pooled_thresholds)
            ),
            "pareto_option_counts_by_topic": topic_pareto_counts,
            "candidate_thresholds_by_topic": {
                topic: topic_outputs[topic]["candidate_thresholds"]
                for topic in topics
            },
            "pareto_validation_profiles_by_topic": {
                topic: topic_outputs[topic]["pareto_validation_profiles"]
                for topic in topics
            },
            "rows": vector_rows,
        },
    }

    shared_completeness = np.asarray(
        [
            row["validation"]["epsilon_good_complete_prompts"]
            for row in shared_rows
        ],
        dtype=int,
    )
    vector_completeness = np.asarray(
        [
            row["validation"]["epsilon_good_complete_prompts"]
            for row in vector_rows
        ],
        dtype=int,
    )
    if np.any(vector_completeness < shared_completeness):
        raise RuntimeError(
            "topic-threshold optimum is worse than its nested shared rule"
        )
    for rows in (shared_rows, vector_rows):
        validation_completeness = np.asarray(
            [row["validation"]["epsilon_good_complete_prompts"] for row in rows]
        )
        if np.any(np.diff(validation_completeness) < 0):
            raise RuntimeError("validation-budget optimum is nonmonotone")
        if any(
            row["used_validation_failures"]
            > row["validation_failure_budget"]
            for row in rows
        ):
            raise RuntimeError("calibrated threshold exceeds failure budget")

    output["shared_rule_feasibility_audit"] = {
        "raw_grid_construction": (
            "Each topic grid is the union of its canonical behaviorally "
            "exhaustive validation grid and the pooled shared grid."
        ),
        "shared_vector_embedding": "tau maps to (tau, tau, tau, tau, tau)",
        "all_shared_candidates_in_every_topic_grid": True,
        "budgets_checked": int(maximum_budget + 1),
        "topic_vector_validation_completeness_weakly_dominates_shared": True,
    }
    return output


def validation_budget_threshold_calibration(
    gains: np.ndarray,
    subsets: np.ndarray,
    validation_prompts: np.ndarray,
    test_prompts: np.ndarray,
    candidate_source: str,
    score_rule: dict[str, object],
    topics: tuple[str, ...] = TOPICS,
) -> dict[str, object]:
    """Select with validation-only inputs, then attach calibrated test metrics."""
    validation_gains = np.asarray(
        gains[validation_prompts],
        dtype=float,
    ).copy()
    validation_subsets = np.asarray(
        subsets[validation_prompts],
        dtype=str,
    ).copy()
    output = select_validation_budget_thresholds(
        validation_gains,
        validation_subsets,
        candidate_source,
        score_rule,
        topics,
    )
    shared_rows = output["shared_threshold"]["rows"]
    vector_rows = output["topic_threshold_vector"]["rows"]
    if any("test" in row for row in shared_rows + vector_rows):
        raise RuntimeError("validation-only selection returned test metrics")

    selection_before_test = json_sha256(
        validation_selection_payload(output, topics)
    )
    output["test_topic_counts"] = {
        topic: int(np.sum(subsets[test_prompts] == topic)) for topic in topics
    }
    for rows in (shared_rows, vector_rows):
        attach_frozen_split_metrics(
            gains,
            subsets,
            test_prompts,
            rows,
            "test",
            topics,
        )
    selection_after_test = json_sha256(
        validation_selection_payload(output, topics)
    )
    if selection_after_test != selection_before_test:
        raise RuntimeError("held-out evaluation changed threshold selection")

    output["selection_audit"] = {
        "selection_function_inputs": [
            "frozen validation score array",
            "validation topic labels",
            "candidate-source description",
            "frozen score-rule provenance",
            "fixed topic order",
        ],
        "selection_function_receives_held_out_inputs": False,
        "calibrated_test_metrics_attached_after_selection": True,
        "validation_selection_sha256_before_test_evaluation": (
            selection_before_test
        ),
        "validation_selection_sha256_after_test_evaluation": (
            selection_after_test
        ),
        "selection_unchanged_by_test_evaluation": True,
    }

    primary_budget = len(validation_prompts) - int(
        np.ceil(
            SOUNDNESS_COVERAGE_TARGET * len(validation_prompts) - TOLERANCE
        )
    )
    for policy in ("shared_threshold", "topic_threshold_vector"):
        primary_row = output[policy]["rows"][primary_budget]
        primary = dict(primary_row)
        primary["by_topic"] = {
            topic: {
                "threshold": primary_row["thresholds_by_topic"][topic],
                "validation": primary_row["validation_by_topic"][topic],
                "test": primary_row["test_by_topic"][topic],
            }
            for topic in topics
        }
        output[policy]["primary_95_validation_target"] = primary
    return output


def ranked_single_approximate_tradeoff(
    soundness_error: np.ndarray,
    completeness_error: np.ndarray,
    completeness_attained: np.ndarray,
    reviewer_ranking: np.ndarray,
    models: np.ndarray,
) -> dict[str, object]:
    """Return rank-resolved approximate guarantees for native singletons.

    Reviewer rank is fixed entirely on the validation split before these
    held-out rates are calculated.  A prompt contributes to a heatmap cell
    only when the same native singleton satisfies both the soundness and
    completeness conditions on that prompt.
    """
    if soundness_error.shape != completeness_error.shape:
        raise ValueError("ranked-singleton error arrays have different shapes")
    if completeness_attained.shape != completeness_error.shape:
        raise ValueError("ranked-singleton attainment array has wrong shape")
    n_prompts, n_reviewers = soundness_error.shape
    if len(reviewer_ranking) != n_reviewers:
        raise ValueError("reviewer ranking has wrong length")

    epsilon_harm = np.linspace(0.0, 0.25, 101)
    epsilon_good = np.linspace(0.0, 0.75, 151)
    targets = (0.10, 0.20, 0.30)
    boundaries = {
        f"{target:.2f}": np.full(
            (n_reviewers, len(epsilon_harm)),
            np.nan,
        )
        for target in targets
    }
    rank1_heatmap: np.ndarray | None = None
    checkpoint_harm = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    checkpoint_good = np.asarray([0.0, 0.25, 0.50, 0.75])
    checkpoint_harm_indices = np.searchsorted(epsilon_harm, checkpoint_harm)
    checkpoint_good_indices = np.searchsorted(epsilon_good, checkpoint_good)
    checkpoint_rows: list[dict[str, object]] = []

    for rank_index in range(n_reviewers):
        sound = (
            soundness_error[:, rank_index, None]
            <= epsilon_harm[None, :] + 5 * GEOMETRY_TOLERANCE
        )
        complete = epsilon_complete(
            completeness_error[:, rank_index, None],
            completeness_attained[:, rank_index, None],
            epsilon_good[None, :],
        )
        joint_rates = (
            complete.T.astype(float) @ sound.astype(float)
        ) / n_prompts
        if rank_index == 0:
            rank1_heatmap = joint_rates.copy()

        for target in targets:
            feasible = joint_rates >= target - TOLERANCE
            any_feasible = np.any(feasible, axis=0)
            first_feasible = np.argmax(feasible, axis=0)
            boundaries[f"{target:.2f}"][rank_index, any_feasible] = (
                epsilon_good[first_feasible[any_feasible]]
            )

        reviewer = int(reviewer_ranking[rank_index])
        checkpoint_rows.append(
            {
                "reviewer_rank": int(rank_index + 1),
                "reviewer_index": reviewer,
                "model": str(models[reviewer]),
                "joint_certified_fraction": joint_rates[
                    np.ix_(checkpoint_good_indices, checkpoint_harm_indices)
                ].tolist(),
            }
        )

    if rank1_heatmap is None:
        raise RuntimeError("ranked-singleton tradeoff has no reviewers")

    serialized_boundaries: dict[str, object] = {}
    for target in targets:
        values = boundaries[f"{target:.2f}"]
        serialized_boundaries[f"{target:.2f}"] = {
            "target_joint_fraction": target,
            "by_rank": [
                [
                    None if not np.isfinite(value) else float(value)
                    for value in row
                ]
                for row in values
            ],
        }

    return {
        "definitions": {
            "joint_certified_fraction": (
                "Held-out fraction of prompts on which the same native "
                "single-reviewer rule both authorizes no proposal with "
                "principal loss above epsilon_harm and authorizes every "
                "proposal with principal gain at least epsilon_good."
            ),
            "soundness_condition": (
                "The exact worst authorized principal loss is at most "
                "epsilon_harm; equality satisfies the condition."
            ),
            "completeness_condition": (
                "The exact supremum gain among rejected proposals is below "
                "epsilon_good, or equals epsilon_good without being attained."
            ),
            "boundary": (
                "For each validation-ranked reviewer and epsilon_harm grid "
                "point, the minimum epsilon_good grid point attaining the "
                "target jointly certified held-out fraction; null means the "
                "target is not attained on the displayed grid."
            ),
            "ranking_provenance": (
                "Reviewer order is fixed by validation choice accuracy with "
                "original reviewer index as the tie-break; test prompts do "
                "not select or order reviewers."
            ),
        },
        "test_prompts": int(n_prompts),
        "epsilon_harm_grid": epsilon_harm.tolist(),
        "epsilon_good_grid": epsilon_good.tolist(),
        "rank1_heatmap": {
            "reviewer_rank": 1,
            "values": rank1_heatmap.tolist(),
        },
        "minimum_epsilon_good_by_target": serialized_boundaries,
        "representative_checkpoints": {
            "epsilon_harm": checkpoint_harm.tolist(),
            "epsilon_good": checkpoint_good.tolist(),
            "rates_by_rank": checkpoint_rows,
        },
    }


def prompt_specific_cardinal_reconstruction(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
) -> dict[str, float | int]:
    """Recompute exact prompt-specific nonnegative principal reconstruction."""
    covered = 0
    authorized_correct = 0
    maximum_residual = 0.0
    for prompt in prompts:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewer_vectors, principal_vector = multicandidate_utilities(
            scores, start, stop, 1
        )
        result = linprog(
            np.zeros(len(reviewer_vectors)),
            A_eq=reviewer_vectors.T,
            b_eq=principal_vector,
            bounds=[(0.0, None)] * len(reviewer_vectors),
            method="highs",
        )
        if result.status == 2:
            continue
        if result.status != 0 or result.x is None:
            raise RuntimeError(
                f"prompt-specific reconstruction failed: {result.message}"
            )
        residual = np.linalg.norm(
            reviewer_vectors.T @ result.x - principal_vector,
            ord=np.inf,
        )
        maximum_residual = max(maximum_residual, float(residual))
        if residual > 1e-8:
            raise RuntimeError("prompt-specific reconstruction is inaccurate")
        covered += 1
        # The aggregate is the principal direction, whose gain for the pure
        # correct response relative to the uniform fallback is 3/4.
        authorized_correct += 1
    return {
        "prompts": int(len(prompts)),
        "covered_prompts": int(covered),
        "coverage_rate": float(covered / len(prompts)),
        "authorized_correct_among_covered": int(authorized_correct),
        "pure_response_completeness_among_covered": float(
            authorized_correct / covered if covered else 0.0
        ),
        "maximum_lp_residual": float(maximum_residual),
    }


def single_reviewer_alignment(
    scores: np.ndarray,
    offsets: np.ndarray,
    prompts: np.ndarray,
    reviewer: int,
) -> dict[str, float | int]:
    """Compare one reviewer with the principal's exact utility direction."""
    covered = 0
    angles = []
    for prompt in prompts:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, principal = multicandidate_utilities(scores, start, stop, 1)
        covered += int(
            covers_multicandidate_utility(
                reviewers, principal, panel=np.array([reviewer])
            )
        )
        reviewer_vector = reviewers[reviewer]
        reviewer_norm = np.linalg.norm(reviewer_vector)
        cosine = (
            -1.0
            if reviewer_norm <= TOLERANCE
            else np.clip(
                reviewer_vector @ principal / reviewer_norm,
                -1.0,
                1.0,
            )
        )
        angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "prompts": int(len(prompts)),
        "exactly_covered_prompts": int(covered),
        "median_angle_degrees": float(np.median(angles)),
        "within_10_degrees": int(np.sum(np.asarray(angles) <= 10.0)),
    }


def make_figure(output: dict[str, object]) -> None:
    configure_style()

    def figure_rows(key: str) -> list[dict[str, object]]:
        if key == "train_selected_global_singleton":
            return output["train_selected_singleton_generalization"]["global"][
                "rows"
            ]
        if key == "train_selected_topic_singleton":
            return output["train_selected_singleton_generalization"][
                "topic_conditioned"
            ]["rows"]
        if key == "topic_conditioned_learned_weight":
            return output["topic_conditioned"]["learned_weight"]["rows"]
        return output[key]["rows"]

    def figure_coordinates(
        key: str,
        row: dict[str, object],
    ) -> tuple[float, float]:
        if key in {
            "train_selected_global_singleton",
            "train_selected_topic_singleton",
        }:
            return (
                100 * row["test"]["full_domain_soundness"],
                100 * row["test"]["epsilon_good_completeness"],
            )
        return (
            100 * row["test_full_domain_soundness"],
            100
            * row["test_approximate_full_domain"][
                "epsilon_good_completeness"
            ],
        )

    threshold_specifications = (
        ("count_rule", "Binary count rule", RED, "--", 1.45, None),
        (
            "weighted_binary",
            "Learned weighted binary",
            ORANGE,
            "-",
            1.65,
            None,
        ),
        ("equal_weight", "Equal-weight cardinal", GRAY, "-.", 1.35, None),
        (
            "learned_weight",
            "Learned cardinal (global)",
            INK,
            "-",
            1.85,
            None,
        ),
        (
            "topic_conditioned_learned_weight",
            "Learned cardinal (topic-routed)",
            INK,
            (0, (3, 1.7)),
            1.65,
            None,
        ),
        (
            "ablated_learned_weight",
            "Learned cardinal (no validation-best reviewer)",
            PURPLE,
            (0, (4, 2)),
            1.45,
            None,
        ),
        (
            "train_selected_global_singleton",
            "Train-selected singleton (global)",
            BLUE,
            ":",
            1.65,
            None,
        ),
        (
            "train_selected_topic_singleton",
            "Train-selected singleton (topic-routed)",
            BLUE,
            "-",
            1.55,
            None,
        ),
    )
    ranked_specifications = (
        (
            "ranked_single_reviewer",
            "Ranked next reviewer (binary)",
            PINK,
            (0, (1, 1)),
            1.35,
            "o",
        ),
        (
            "ranked_prefix_unanimity",
            r"Top-$m$ reviewers, unanimity",
            TEAL,
            "-",
            1.65,
            "o",
        ),
    )

    figure = plt.figure(figsize=(9.2, 5.7), facecolor="white")
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.22)
    exact_axis = figure.add_subplot(grid[0, 0])
    ranked_axis = figure.add_subplot(grid[0, 1])
    tolerance_axis = figure.add_subplot(grid[1, :])
    axes = (exact_axis, ranked_axis, tolerance_axis)

    for key, label, color, linestyle, linewidth, _ in threshold_specifications:
        rows = figure_rows(key)
        coordinates = [figure_coordinates(key, row) for row in rows]
        exact_axis.plot(
            [x_value for x_value, _ in coordinates],
            [y_value for _, y_value in coordinates],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=(
                4
                if key
                in {
                    "learned_weight",
                    "topic_conditioned_learned_weight",
                    "train_selected_topic_singleton",
                }
                else 2
            ),
        )
    prefix_rows = output["ranked_prefix_unanimity"]["rows"]
    exact_axis.plot(
        [100 * row["test_full_domain_soundness"] for row in prefix_rows],
        [
            100
            * row["test_approximate_full_domain"][
                "epsilon_good_completeness"
            ]
            for row in prefix_rows
        ],
        color=TEAL,
        linestyle="-",
        linewidth=1.35,
        marker="o",
        markersize=2.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label=r"Top-$m$ reviewers, unanimity",
        zorder=4,
    )
    reconstruction = output["prompt_specific_cardinal_reconstruction"]
    exact_axis.scatter(
        100 * reconstruction["test_full_domain_soundness_rate"],
        100 * reconstruction["test_full_domain_completeness_rate"],
        color="#26876E",
        marker="D",
        s=28,
        label="Prompt-specific reconstruction",
        zorder=5,
    )
    exact_axis.set_xlim(74, 101)
    exact_axis.set_xticks([75, 80, 85, 90, 95, 100])
    exact_axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    exact_axis.set_xlabel("fraction satisfying exact soundness")
    exact_axis.set_title(r"$\bf{(a)}$ Exact soundness", loc="left", pad=6)
    target = 100 * SOUNDNESS_COVERAGE_TARGET
    exact_axis.axvline(
        target,
        color="#8F9293",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    exact_axis.text(
        target - 0.8,
        101.0,
        f"{target:.0f}%",
        color="#6E7173",
        fontsize=6.2,
        ha="right",
        va="top",
    )
    marked: dict[str, tuple[float, float]] = {}
    for key, color, marker in (
        ("learned_weight", INK, "o"),
        ("topic_conditioned_learned_weight", INK, "s"),
        ("train_selected_global_singleton", BLUE, "o"),
        ("train_selected_topic_singleton", BLUE, "s"),
    ):
        candidates = []
        for row in figure_rows(key):
            x_value, y_value = figure_coordinates(key, row)
            if x_value >= target - 1e-10:
                candidates.append((y_value, x_value))
        y_value, x_value = max(candidates)
        marked[key] = (x_value, y_value)
        exact_axis.scatter(
            [x_value],
            [y_value],
            s=22,
            facecolor=color,
            edgecolor="white",
            marker=marker,
            linewidth=0.65,
            zorder=6,
        )
    x_gap = max(
        marked["learned_weight"][0],
        marked["train_selected_global_singleton"][0],
    )
    lower = marked["train_selected_global_singleton"][1]
    upper = marked["learned_weight"][1]
    exact_axis.plot([x_gap, x_gap], [lower, upper], color="#55585A", lw=0.8)
    exact_axis.plot(
        [x_gap - 0.65, x_gap + 0.65],
        [lower, lower],
        color="#55585A",
        lw=0.8,
    )
    exact_axis.plot(
        [x_gap - 0.65, x_gap + 0.65],
        [upper, upper],
        color="#55585A",
        lw=0.8,
    )
    exact_axis.text(
        x_gap - 1.5,
        (lower + upper) / 2,
        f"{upper - lower:.0f} pp",
        color="#55585A",
        fontsize=6.4,
        ha="right",
        va="center",
    )

    visible_prefix = [
        (panel_size, row)
        for panel_size, row in enumerate(prefix_rows, start=1)
        if 100 * row["test_full_domain_soundness"] >= 74.0
    ]
    if visible_prefix:
        prefix_labels = [
            (visible_prefix[0], (3, 5), "left"),
            (visible_prefix[-1], (-3, 7), "right"),
        ]
        for (panel_size, row), offset, alignment in prefix_labels:
            exact_axis.annotate(
                f"$m={panel_size}$",
                xy=(
                    100 * row["test_full_domain_soundness"],
                    100
                    * row["test_approximate_full_domain"][
                        "epsilon_good_completeness"
                    ],
                ),
                xytext=offset,
                textcoords="offset points",
                color=TEAL,
                fontsize=5.8,
                ha=alignment,
                va="bottom",
            )

    for key, label, color, linestyle, linewidth, marker in ranked_specifications:
        rows = figure_rows(key)
        ranked_axis.plot(
            [100 * row["test_full_domain_soundness"] for row in rows],
            [
                100
                * row["test_approximate_full_domain"][
                    "epsilon_good_completeness"
                ]
                for row in rows
            ],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=2.1,
            markeredgewidth=0.0,
            label=label,
            zorder=3,
        )
    for panel_size, offset in ((1, (4, -8)), (24, (4, 5)), (48, (-4, 7))):
        row = prefix_rows[panel_size - 1]
        ranked_axis.annotate(
            f"$m={panel_size}$",
            xy=(
                100 * row["test_full_domain_soundness"],
                100
                * row["test_approximate_full_domain"][
                    "epsilon_good_completeness"
                ],
            ),
            xytext=offset,
            textcoords="offset points",
            color=TEAL,
            fontsize=5.8,
            ha="right" if panel_size == 48 else "left",
            va="bottom" if offset[1] >= 0 else "top",
        )
    ranked_axis.text(
        2.5,
        80.0,
        "every native single-reviewer rule\nhas 0% exact soundness",
        color=PINK,
        fontsize=5.9,
        ha="left",
        va="center",
    )
    ranked_axis.set_xlim(-2.5, 102.5)
    ranked_axis.set_xticks([0, 25, 50, 75, 100])
    ranked_axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ranked_axis.set_xlabel("fraction satisfying exact soundness")
    ranked_axis.set_title(
        r"$\bf{(b)}$ Validation-ranked binary rules",
        loc="left",
        pad=6,
    )

    tolerance = output["soundness_tolerance_sensitivity"]
    epsilon_harm = np.asarray(tolerance["epsilon_harm"], dtype=float)
    for key, label, color, linestyle, linewidth, _ in (
        threshold_specifications + ranked_specifications
    ):
        tolerance_axis.plot(
            epsilon_harm,
            100
            * np.asarray(
                tolerance["series"][key]["best_completeness"],
                dtype=float,
            ),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            drawstyle="steps-post",
            label=label,
            zorder=(
                4
                if key
                in {
                    "learned_weight",
                    "topic_conditioned_learned_weight",
                    "train_selected_topic_singleton",
                }
                else 2
            ),
        )
    tolerance_axis.set_xlim(-0.005, 0.255)
    tolerance_axis.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    tolerance_axis.set_xlabel(r"permitted loss $\epsilon_{\rm harm}$")
    tolerance_axis.set_title(
        r"$\bf{(c)}$ Approximate soundness at 95%",
        loc="left",
        pad=6,
    )

    for axis in axes:
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")
    exact_axis.set_ylabel(
        rf"fraction authorizing every gain $\geq {APPROXIMATE_GOOD:g}$"
    )
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    figure.legend(
        list(handles_by_label.values()),
        list(handles_by_label.keys()),
        frameon=False,
        fontsize=6.7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.15,
        handlelength=2.0,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.75,
        hspace=0.46,
        wspace=0.23,
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=260, bbox_inches="tight")
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(figure)


def make_pure_response_figure(output: dict[str, object]) -> None:
    """Plot held-out pure-endpoint tradeoffs on validation-defined grids."""
    configure_style()
    specifications = (
        ("count_rule", "Binary count rule", RED, "--", 1.45),
        (
            "weighted_binary",
            "Learned weighted binary",
            ORANGE,
            "-",
            1.65,
        ),
        ("equal_weight", "Equal-weight cardinal", GRAY, "-.", 1.35),
        (
            "learned_weight",
            "Learned nonnegative cardinal",
            INK,
            "-",
            1.85,
        ),
        (
            "ablated_learned_weight",
            "Learned cardinal without selected reviewer",
            PURPLE,
            (0, (4, 2)),
            1.45,
        ),
        (
            "best_single",
            "Validation-selected single reviewer",
            BLUE,
            ":",
            1.65,
        ),
    )

    figure, axis = plt.subplots(figsize=(7.35, 4.35), facecolor="white")
    for key, label, color, linestyle, linewidth in specifications:
        rows = output[key]["rows"]
        axis.plot(
            [100 * row["test"]["pure_response_soundness"] for row in rows],
            [
                100 * row["test"]["pure_response_completeness"]
                for row in rows
            ],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=3 if key == "learned_weight" else 2,
        )
        for target, marker, size in ((0.99, "s", 18), (0.95, "o", 20)):
            feasible = [
                row
                for row in rows
                if row["test"]["pure_response_soundness"]
                >= target - TOLERANCE
            ]
            best = max(
                feasible,
                key=lambda row: (
                    row["test"]["pure_response_completeness"],
                    row["test"]["pure_response_soundness"],
                ),
            )
            axis.scatter(
                100 * best["test"]["pure_response_soundness"],
                100 * best["test"]["pure_response_completeness"],
                s=size,
                marker=marker,
                facecolor=color if target == 0.99 else "white",
                edgecolor=color,
                linewidth=0.8,
                zorder=5,
            )

    reconstruction = output["prompt_specific_cardinal_reconstruction"]
    axis.scatter(
        100 * reconstruction["test_full_domain_soundness_rate"],
        100 * reconstruction["test_full_domain_completeness_rate"],
        color="#26876E",
        marker="D",
        s=30,
        label="Prompt-specific reconstruction",
        zorder=6,
    )

    for target, dash in ((95, (0, (2, 2))), (99, (0, (1, 2)))):
        axis.axvline(
            target,
            color="#8F9293",
            linestyle=dash,
            linewidth=0.75,
            zorder=1,
        )
        axis.text(
            target - 0.25,
            101.0,
            f"{target}%",
            color="#6E7173",
            fontsize=6.2,
            ha="right",
            va="top",
        )

    axis.text(
        75.0,
        3.0,
        "threshold candidates fixed on validation\n"
        "filled squares: best at $\\geq 99\\%$ soundness\n"
        "open circles: best at $\\geq 95\\%$ soundness",
        color="#65686A",
        fontsize=6.2,
        ha="left",
        va="bottom",
    )
    axis.set_xlim(74, 101)
    axis.set_xticks([75, 80, 85, 90, 95, 100])
    axis.set_ylim(0, 103)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.set_xlabel("fraction satisfying pure-response soundness")
    axis.set_ylabel("fraction satisfying pure-response completeness")
    axis.set_title(
        "Held-out pure-response frontier",
        loc="left",
        pad=7,
    )
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.65, color="#8F9293")
    figure.legend(
        *axis.get_legend_handles_labels(),
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handlelength=2.1,
    )
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.13, top=0.72)
    PURE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PURE_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(PURE_FIGURE_PDF, bbox_inches="tight")
    plt.close(figure)


def make_pure_response_detail_figure(output: dict[str, object]) -> None:
    """Show the same pure-response frontier at full and detailed scales."""
    configure_style()
    specifications = (
        ("count_rule", "Binary count rule", RED, "--", 1.35),
        (
            "weighted_binary",
            "Learned weighted binary",
            ORANGE,
            "-",
            1.5,
        ),
        ("equal_weight", "Equal-weight cardinal", GRAY, "-.", 1.25),
        (
            "learned_weight",
            "Learned nonnegative cardinal",
            INK,
            "-",
            1.8,
        ),
        (
            "ablated_learned_weight",
            "Learned cardinal without selected reviewer",
            PURPLE,
            (0, (4, 2)),
            1.35,
        ),
        (
            "best_single",
            "Validation-selected single reviewer",
            BLUE,
            ":",
            1.6,
        ),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.75),
        sharey=True,
        facecolor="white",
    )
    for axis in axes:
        for key, label, color, linestyle, linewidth in specifications:
            rows = output[key]["rows"]
            axis.plot(
                [
                    100 * row["test"]["pure_response_soundness"]
                    for row in rows
                ],
                [
                    100 * row["test"]["pure_response_completeness"]
                    for row in rows
                ],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
                zorder=3 if key == "learned_weight" else 2,
            )

    checkpoints: dict[str, tuple[float, float]] = {}
    detail_axis = axes[1]
    for key, _, color, _, _ in specifications:
        if key not in {"learned_weight", "best_single"}:
            continue
        rows = output[key]["rows"]
        feasible = [
            row
            for row in rows
            if row["test"]["pure_response_soundness"]
            >= 0.99 - TOLERANCE
        ]
        best = max(
            feasible,
            key=lambda row: (
                row["test"]["pure_response_completeness"],
                row["test"]["pure_response_soundness"],
            ),
        )
        x_value = 100 * best["test"]["pure_response_soundness"]
        y_value = 100 * best["test"]["pure_response_completeness"]
        checkpoints[key] = (x_value, y_value)
        detail_axis.scatter(
            x_value,
            y_value,
            s=30,
            marker="s",
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )

    reconstruction = output["prompt_specific_cardinal_reconstruction"]
    for axis in axes:
        axis.scatter(
            100 * reconstruction["test_full_domain_soundness_rate"],
            100 * reconstruction["test_full_domain_completeness_rate"],
            color="#26876E",
            marker="D",
            s=27,
            label="Prompt-specific reconstruction",
            zorder=6,
        )

    detail_axis.axvspan(99, 100.2, color="#DDEEE9", alpha=0.55, zorder=0)
    detail_axis.axvline(
        99,
        color="#5D6865",
        linestyle=(0, (2, 2)),
        linewidth=1.05,
        zorder=1,
    )
    detail_axis.text(
        98.94,
        101.0,
        "99%",
        color="#4F5957",
        fontsize=6.5,
        ha="right",
        va="top",
    )

    panel_99 = checkpoints["learned_weight"][1]
    single_99 = checkpoints["best_single"][1]
    detail_axis.text(
        94.55,
        3.0,
        rf"At $\geq99\%$ pure soundness:"
        "\n"
        rf"learned cardinal {panel_99:.1f}%; selected singleton "
        rf"{single_99:.1f}%"
        "\n"
        rf"completeness gap: {panel_99 - single_99:.1f} pp",
        color="#65686A",
        fontsize=6.3,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
    )

    axes[0].set_xlim(-2, 102)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_title(
        r"$\bf{(a)}$ Soundness-completeness pure frontier",
        loc="left",
        pad=7,
    )
    detail_axis.set_xlim(94.4, 100.2)
    detail_axis.set_xticks([95, 97, 99, 100])
    detail_axis.set_title(
        r"$\bf{(b)}$ High-soundness detail",
        loc="left",
        pad=7,
    )

    for axis in axes:
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.set_xlabel("pure-response soundness")
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")

    axes[0].set_ylabel("pure-response completeness")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.15,
        handlelength=2.0,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.16,
        top=0.75,
        wspace=0.15,
    )
    PURE_DETAIL_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PURE_DETAIL_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(PURE_DETAIL_FIGURE_PDF, bbox_inches="tight")
    plt.close(figure)


def make_margin_sensitivity_figure(output: dict[str, object]) -> None:
    """Plot completeness over the full range of required gain margins."""
    configure_style()
    sensitivity = output["margin_sensitivity"]
    margins = np.asarray(sensitivity["gain_margins"], dtype=float)
    specifications = (
        ("learned_weight", "Learned cardinal committee", INK, "-", 1.85),
        (
            "ablated_learned_weight",
            "Committee (no validation-best reviewer)",
            PURPLE,
            (0, (4, 2)),
            1.55,
        ),
        (
            "best_single",
            "Validation-selected single reviewer",
            BLUE,
            ":",
            1.7,
        ),
    )
    figure, axis = plt.subplots(1, 1, figsize=(5.6, 3.15), facecolor="white")
    for key, label, color, linestyle, linewidth in specifications:
        rates = 100 * np.asarray(
            sensitivity["series"][key]["completeness"],
            dtype=float,
        )
        axis.plot(
            margins,
            rates,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        index = int(np.argmin(np.abs(margins - 0.5)))
        axis.scatter(
            [margins[index]],
            [rates[index]],
            s=18,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    axis.axvline(
        0.5,
        color="#8F9293",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    axis.text(
        0.49,
        3.0,
        "gain 0.5",
        color="#6E7173",
        fontsize=6.2,
        rotation=90,
        ha="right",
        va="bottom",
    )
    axis.set_xlim(0, 0.75)
    axis.set_ylim(0, 103)
    axis.set_xticks([0, 0.25, 0.5, 0.75])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.65, color="#8F9293")
    axis.set_xlabel("minimum principal gain required")
    axis.set_ylabel("fraction authorizing every qualifying proposal")
    axis.set_title(
        "At least 95% of prompts satisfy exact soundness",
        loc="left",
        pad=6,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.1,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handlelength=2.1,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    MARGIN_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(MARGIN_FIGURE, dpi=260, bbox_inches="tight")
    figure.savefig(MARGIN_FIGURE_PDF, bbox_inches="tight")
    plt.close(figure)


def make_ranked_single_approximate_figure(
    output: dict[str, object],
) -> None:
    """Plot approximate soundness--completeness tradeoffs by validation rank."""
    configure_style()
    tradeoff = output["ranked_single_reviewer"]["approximate_tradeoff"]
    epsilon_harm = np.asarray(tradeoff["epsilon_harm_grid"], dtype=float)
    epsilon_good = np.asarray(tradeoff["epsilon_good_grid"], dtype=float)
    rank1_heatmap = np.asarray(
        tradeoff["rank1_heatmap"]["values"],
        dtype=float,
    )
    if rank1_heatmap.shape != (len(epsilon_good), len(epsilon_harm)):
        raise ValueError("rank-1 approximate heatmap has wrong shape")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 4.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        facecolor="white",
    )
    heat_axis = axes[0, 0]
    heatmap = heat_axis.pcolormesh(
        epsilon_harm,
        epsilon_good,
        rank1_heatmap,
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    contours = heat_axis.contour(
        epsilon_harm,
        epsilon_good,
        rank1_heatmap,
        levels=[0.10, 0.20, 0.30],
        colors=[INK],
        linewidths=0.75,
    )
    heat_axis.clabel(
        contours,
        fmt=lambda value: f"{value:.0%}",
        inline=True,
        fontsize=6.3,
    )
    heat_colorbar = figure.colorbar(
        heatmap,
        ax=heat_axis,
        pad=0.02,
        fraction=0.047,
    )
    heat_colorbar.set_label("jointly certified held-out fraction", fontsize=7.2)
    heat_colorbar.ax.yaxis.set_major_formatter(
        PercentFormatter(1.0, decimals=0)
    )
    heat_colorbar.ax.tick_params(labelsize=6.6, length=2.5)
    heat_axis.set_title(
        r"$\bf{(a)}$ Rank 1: joint approximate guarantee",
        loc="left",
        pad=6,
    )

    rank_cmap = matplotlib.colormaps["viridis_r"]
    rank_norm = matplotlib.colors.Normalize(vmin=1, vmax=48)
    boundary_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for axis, target, panel in zip(
        boundary_axes,
        (0.10, 0.20, 0.30),
        ("b", "c", "d"),
    ):
        stored = tradeoff["minimum_epsilon_good_by_target"][
            f"{target:.2f}"
        ]
        boundaries = np.asarray(stored["by_rank"], dtype=float)
        if boundaries.shape != (48, len(epsilon_harm)):
            raise ValueError("ranked approximate boundary has wrong shape")
        for rank in range(48, 0, -1):
            color = rank_cmap(rank_norm(rank))
            values = boundaries[rank - 1]
            if rank == 1:
                axis.plot(
                    epsilon_harm,
                    values,
                    color="white",
                    linewidth=3.2,
                    drawstyle="steps-post",
                    zorder=5,
                )
                linewidth = 2.15
                alpha = 1.0
                zorder = 6
            else:
                linewidth = 0.72
                alpha = 0.58
                zorder = 2
            axis.plot(
                epsilon_harm,
                values,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                drawstyle="steps-post",
                zorder=zorder,
            )
            if np.isfinite(values[-1]):
                axis.scatter(
                    [epsilon_harm[-1]],
                    [values[-1]],
                    s=22 if rank == 1 else 7,
                    facecolor=color,
                    edgecolor=INK if rank == 1 else "white",
                    linewidth=0.7 if rank == 1 else 0.25,
                    alpha=1.0 if rank == 1 else 0.78,
                    zorder=10 if rank == 1 else 9,
                )
        axis.set_title(
            rf"$\bf{{({panel})}}$ {target:.0%} joint coverage",
            loc="left",
            pad=6,
        )

    rank_mappable = plt.cm.ScalarMappable(norm=rank_norm, cmap=rank_cmap)
    rank_mappable.set_array([])
    rank_colorbar = figure.colorbar(
        rank_mappable,
        ax=boundary_axes,
        pad=0.02,
        fraction=0.027,
        shrink=0.92,
    )
    rank_colorbar.set_ticks([1, 12, 24, 36, 48])
    rank_colorbar.set_label(
        "validation rank (1 = best)",
        fontsize=7.2,
    )
    rank_colorbar.ax.tick_params(labelsize=6.6, length=2.5)

    for axis in axes.ravel():
        axis.axvline(
            0.25,
            color=RED,
            linestyle=(0, (2, 2)),
            linewidth=0.9,
            zorder=7,
        )
        axis.text(
            0.247,
            0.735,
            "vacuous harm edge",
            color=RED,
            fontsize=5.9,
            rotation=90,
            ha="right",
            va="top",
            zorder=8,
        )
        axis.set_xlim(0.0, 0.255)
        axis.set_ylim(0.0, 0.75)
        axis.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        axis.set_yticks([0.0, 0.25, 0.50, 0.75])
        axis.grid(color=GRID, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")
    axes[1, 0].set_xlabel(r"permitted loss $\epsilon_{\rm harm}$")
    axes[1, 1].set_xlabel(r"permitted loss $\epsilon_{\rm harm}$")
    axes[0, 0].set_ylabel(r"required gain $\epsilon_{\rm good}$")
    axes[1, 0].set_ylabel(r"minimum $\epsilon_{\rm good}$")
    figure.suptitle(
        "Validation-ranked native single reviewers",
        fontsize=10.0,
        fontweight="semibold",
    )

    RANKED_APPROXIMATE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        RANKED_APPROXIMATE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        RANKED_APPROXIMATE_FIGURE_PDF,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_topic_conditioned_figure(output: dict[str, object]) -> None:
    """Compare global rules with shared-threshold topic-conditioned analogs."""
    configure_style()
    specifications = (
        ("weighted_binary", "Weighted binary", ORANGE),
        ("learned_weight", "Nonnegative cardinal", INK),
        ("best_single", "Selected singleton", BLUE),
    )
    topic_output = output["topic_conditioned"]
    sensitivity = output["soundness_tolerance_sensitivity"]
    epsilon_harm = np.asarray(sensitivity["epsilon_harm"], dtype=float)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.55),
        facecolor="white",
    )
    exact_axis, approximate_axis = axes

    for key, label, color in specifications:
        global_rows = output[key]["rows"]
        topic_rows = topic_output[key]["rows"]
        for rows, suffix, linestyle, linewidth, alpha, zorder in (
            (global_rows, "global", "--", 1.25, 0.58, 2),
            (topic_rows, "topic-conditioned", "-", 2.0, 1.0, 4),
        ):
            x_values = np.asarray(
                [100 * row["test_full_domain_soundness"] for row in rows]
            )
            y_values = np.asarray(
                [
                    100
                    * row["test_approximate_full_domain"][
                        "epsilon_good_completeness"
                    ]
                    for row in rows
                ]
            )
            exact_axis.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{label} ({suffix})",
                zorder=zorder,
            )
            feasible = np.flatnonzero(
                x_values >= 100 * SOUNDNESS_COVERAGE_TARGET - TOLERANCE
            )
            if len(feasible):
                best = feasible[int(np.argmax(y_values[feasible]))]
                exact_axis.scatter(
                    [x_values[best]],
                    [y_values[best]],
                    s=20 if suffix == "topic-conditioned" else 13,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=alpha,
                    zorder=zorder + 2,
                )

        for series_key, suffix, linestyle, linewidth, alpha, zorder in (
            (key, "global", "--", 1.25, 0.58, 2),
            (
                f"topic_conditioned_{key}",
                "topic-conditioned",
                "-",
                2.0,
                1.0,
                4,
            ),
        ):
            rates = 100 * np.asarray(
                sensitivity["series"][series_key]["best_completeness"],
                dtype=float,
            )
            approximate_axis.plot(
                epsilon_harm,
                rates,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

    exact_axis.axvline(
        100 * SOUNDNESS_COVERAGE_TARGET,
        color="#8F9293",
        linestyle=(0, (2, 2)),
        linewidth=0.85,
        zorder=1,
    )
    exact_axis.text(
        100 * SOUNDNESS_COVERAGE_TARGET - 0.7,
        101.0,
        "95%",
        color="#6E7173",
        fontsize=6.2,
        ha="right",
        va="top",
    )
    exact_axis.set_xlim(74, 101)
    exact_axis.set_xticks([75, 80, 85, 90, 95, 100])
    exact_axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    exact_axis.set_xlabel("fraction satisfying exact soundness")
    exact_axis.set_ylabel(
        r"fraction authorizing every gain $\geq 0.5$"
    )
    exact_axis.set_title(
        r"$\bf{(a)}$ Pooled held-out frontier",
        loc="left",
        pad=6,
    )

    approximate_axis.axvline(
        0.25,
        color=RED,
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        zorder=5,
    )
    approximate_axis.text(
        0.247,
        97.0,
        "vacuous harm edge",
        color=RED,
        fontsize=6.0,
        rotation=90,
        ha="right",
        va="top",
    )
    approximate_axis.set_xlim(0.0, 0.255)
    approximate_axis.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    approximate_axis.set_xlabel(r"permitted loss $\epsilon_{\rm harm}$")
    approximate_axis.set_title(
        r"$\bf{(b)}$ At least 95% of prompts protected",
        loc="left",
        pad=6,
    )

    for axis in axes:
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")
    approximate_axis.set_ylabel(
        r"fraction authorizing every gain $\geq 0.5$"
    )
    handles, labels = exact_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.0,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.15,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.78,
        wspace=0.24,
    )
    TOPIC_CONDITIONED_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        TOPIC_CONDITIONED_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        TOPIC_CONDITIONED_FIGURE_PDF,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_topic_threshold_budget_figure(output: dict[str, object]) -> None:
    """Plot held-out paths selected by pooled validation failure budgets."""
    configure_style()
    calibration = output["topic_threshold_budget_calibration"]
    families = calibration["families"]
    primary_budget = calibration["validation_failure_budgets"][
        "primary_95_soundness_budget"
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.7),
        facecolor="white",
    )
    cardinal_axis, singleton_axis = axes
    specifications = {
        cardinal_axis: (
            (
                "learned_topic_cardinal",
                "shared_threshold",
                "Topic weights, one shared $\\tau$",
                INK,
                "--",
                1.35,
                0.62,
            ),
            (
                "learned_topic_cardinal",
                "topic_threshold_vector",
                "Topic weights, five $\\tau_t$",
                INK,
                "-",
                2.05,
                1.0,
            ),
            (
                "global_cardinal_topic_threshold_control",
                "shared_threshold",
                "Global weights, one shared $\\tau$",
                PURPLE,
                "--",
                1.25,
                0.56,
            ),
            (
                "global_cardinal_topic_threshold_control",
                "topic_threshold_vector",
                "Global weights, five $\\tau_t$",
                PURPLE,
                "-",
                1.75,
                0.9,
            ),
        ),
        singleton_axis: (
            (
                "train_selected_global_singleton_topic_threshold_control",
                "shared_threshold",
                "Global reviewer; shared $\\tau$",
                ORANGE,
                "--",
                1.25,
                0.56,
            ),
            (
                "train_selected_global_singleton_topic_threshold_control",
                "topic_threshold_vector",
                "Global reviewer; five $\\tau_t$",
                ORANGE,
                "-",
                1.75,
                0.9,
            ),
            (
                "train_selected_topic_singleton",
                "shared_threshold",
                "Topic reviewers; shared $\\tau$",
                BLUE,
                "--",
                1.35,
                0.62,
            ),
            (
                "train_selected_topic_singleton",
                "topic_threshold_vector",
                "Topic reviewers; five $\\tau_t$",
                BLUE,
                "-",
                2.05,
                1.0,
            ),
        ),
    }
    marker_by_policy = {
        "shared_threshold": "o",
        "topic_threshold_vector": "s",
    }
    for axis, axis_specifications in specifications.items():
        for (
            family,
            policy,
            label,
            color,
            linestyle,
            linewidth,
            alpha,
        ) in axis_specifications:
            rows = families[family][policy]["rows"]
            path = rows
            x_values = [
                100 * row["test"]["full_domain_soundness"] for row in path
            ]
            y_values = [
                100 * row["test"]["epsilon_good_completeness"]
                for row in path
            ]
            axis.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=4 if policy == "topic_threshold_vector" else 2,
            )
            primary = rows[primary_budget]
            axis.scatter(
                [100 * primary["test"]["full_domain_soundness"]],
                [100 * primary["test"]["epsilon_good_completeness"]],
                s=32 if policy == "topic_threshold_vector" else 22,
                marker=marker_by_policy[policy],
                facecolor=color,
                edgecolor="white",
                linewidth=0.65,
                alpha=alpha,
                zorder=8,
            )

        main_family = (
            "learned_topic_cardinal"
            if axis is cardinal_axis
            else "train_selected_topic_singleton"
        )
        primary = families[main_family]["topic_threshold_vector"]["rows"][
            primary_budget
        ]
        axis.annotate(
            (
                f"$B={primary_budget}$ validation marker\n"
                f"test: {primary['test']['sound_prompts']}/"
                f"{primary['test']['prompts']} sound, "
                f"{primary['test']['epsilon_good_complete_prompts']}/"
                f"{primary['test']['prompts']} complete"
            ),
            xy=(
                100 * primary["test"]["full_domain_soundness"],
                100 * primary["test"]["epsilon_good_completeness"],
            ),
            xytext=(-90, -28),
            textcoords="offset points",
            fontsize=6.2,
            color="#55585A",
            arrowprops={
                "arrowstyle": "-",
                "color": "#888B8D",
                "linewidth": 0.65,
            },
            ha="left",
            va="top",
            zorder=9,
        )

    for axis in axes:
        axis.axvline(
            100 * SOUNDNESS_COVERAGE_TARGET,
            color="#8F9293",
            linestyle=(0, (2, 2)),
            linewidth=0.85,
            zorder=1,
        )
        axis.set_xlim(-1, 101)
        axis.set_ylim(-1, 101)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color="#8F9293")
        axis.set_xlabel("held-out exact soundness")
    cardinal_axis.set_ylabel(
        r"held-out fraction authorizing every gain $\geq0.5$"
    )
    cardinal_axis.set_title(
        r"$\bf{(a)}$ Learned cardinal scores",
        loc="left",
        pad=6,
    )
    singleton_axis.set_title(
        r"$\bf{(b)}$ Train-selected singleton scores",
        loc="left",
        pad=6,
    )
    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.15,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.72,
        wspace=0.23,
    )
    TOPIC_THRESHOLD_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(TOPIC_THRESHOLD_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(TOPIC_THRESHOLD_FIGURE_PDF, bbox_inches="tight")
    plt.close(figure)


def make_singleton_generalization_figure(
    output: dict[str, object],
) -> None:
    """Plot train and held-out curves for train-selected singleton rules."""
    configure_style()
    comparison = output["train_selected_singleton_generalization"]
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(6.3, 3.55),
        facecolor="white",
    )
    specifications = (
        ("global", "train", "Global singleton (train)", BLUE, "--", 1.35, 0.62),
        ("global", "test", "Global singleton (test)", BLUE, "-", 2.0, 1.0),
        (
            "topic_conditioned",
            "train",
            "Topic singleton (train)",
            PINK,
            "--",
            1.35,
            0.62,
        ),
        (
            "topic_conditioned",
            "test",
            "Topic singleton (test)",
            PINK,
            "-",
            2.0,
            1.0,
        ),
    )
    for policy, split, label, color, linestyle, linewidth, alpha in specifications:
        rows = comparison[policy]["rows"]
        axis.plot(
            [100 * row[split]["full_domain_soundness"] for row in rows],
            [
                100 * row[split]["epsilon_good_completeness"]
                for row in rows
            ],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=3 if split == "test" else 2,
        )
    axis.axvline(
        100 * SOUNDNESS_COVERAGE_TARGET,
        color="#8F9293",
        linestyle=(0, (2, 2)),
        linewidth=0.85,
        zorder=1,
    )
    axis.text(
        100 * SOUNDNESS_COVERAGE_TARGET - 0.7,
        101.0,
        "95%",
        color="#6E7173",
        fontsize=6.2,
        ha="right",
        va="top",
    )
    axis.set_xlim(74, 101)
    axis.set_ylim(0, 103)
    axis.set_xticks([75, 80, 85, 90, 95, 100])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.65, color="#8F9293")
    axis.set_xlabel("fraction satisfying exact soundness")
    axis.set_ylabel(r"fraction authorizing every gain $\geq 0.5$")
    axis.set_title(
        "Train-selected singleton rules: train versus held-out test",
        loc="left",
        pad=6,
    )
    axis.text(
        74.8,
        4.0,
        (
            "choice accuracy  "
            f"global: {comparison['global']['training_choice_accuracy']:.1%} "
            f"train / {comparison['global']['test_choice_accuracy']:.1%} test;  "
            "topic: "
            f"{comparison['topic_conditioned']['training_choice_accuracy']:.1%} "
            "train / "
            f"{comparison['topic_conditioned']['test_choice_accuracy']:.1%} test"
        ),
        color="#66696B",
        fontsize=6.2,
        ha="left",
        va="bottom",
    )
    figure.legend(
        *axis.get_legend_handles_labels(),
        frameon=False,
        fontsize=7.0,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.3,
        handlelength=2.25,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.79))
    SINGLETON_GENERALIZATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        SINGLETON_GENERALIZATION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        SINGLETON_GENERALIZATION_FIGURE_PDF,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    with np.load(DATA, allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        subsets = data["subsets"].astype(str)
        models = data["models"].astype(str)
        dataset_revision = str(data["dataset_revision"])
        results_revision = str(data["results_revision"])
    with np.load(DEPTH_DATA, allow_pickle=True) as data:
        exact_depth = data["depth"].astype(int)
        maximum_beneficial_objections = data[
            "maximum_beneficial_objections"
        ].astype(int)
        depth_is_exact = data["depth_exact"].astype(bool)
        depth_candidate_counts = data["candidate_counts"].astype(int)
        depth_subsets = data["subsets"].astype(str)
        depth_dataset_revision = str(data["dataset_revision"])
        depth_results_revision = str(data["results_revision"])
    if (
        depth_dataset_revision != dataset_revision
        or depth_results_revision != results_revision
        or not np.array_equal(depth_candidate_counts, candidate_counts)
        or not np.array_equal(depth_subsets, subsets)
    ):
        raise RuntimeError("full-domain depths do not match the score archive")

    standard_prompts = np.flatnonzero(candidate_counts == 4)
    if len(standard_prompts) != 1763 or not np.all(num_correct[standard_prompts] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")
    assignment = stratified_prompt_split(subsets, seed=SEED)
    training_prompts = standard_prompts[assignment[standard_prompts] == "train"]
    validation_prompts = standard_prompts[
        assignment[standard_prompts] == "validation"
    ]
    test_prompts = standard_prompts[assignment[standard_prompts] == "test"]
    if (len(training_prompts), len(validation_prompts), len(test_prompts)) != (
        1058,
        353,
        352,
    ):
        raise RuntimeError("unexpected train/validation/test sizes")

    features = normalized_features(scores, offsets, standard_prompts)
    binary_features = (features >= -TOLERANCE).astype(float)
    fitted = []
    for regularization in REGULARIZATION_GRID:
        weights = fit_nonnegative_weights(
            features, training_prompts, regularization
        )
        gains = features @ weights
        fitted.append(
            {
                "regularization": regularization,
                "weights": weights,
                "validation_accuracy": choice_accuracy(
                    gains, validation_prompts
                ),
            }
        )
    selected_fit = max(
        fitted,
        key=lambda row: (row["validation_accuracy"], -row["regularization"]),
    )
    learned_weights = selected_fit["weights"]
    learned_gains = features @ learned_weights
    equal_gains = np.mean(features, axis=2)

    binary_fitted = []
    for regularization in REGULARIZATION_GRID:
        weights = fit_nonnegative_weights(
            binary_features, training_prompts, regularization
        )
        gains = binary_features @ weights
        binary_fitted.append(
            {
                "regularization": regularization,
                "weights": weights,
                "validation_accuracy": choice_accuracy(
                    gains, validation_prompts
                ),
            }
        )
    selected_binary_fit = max(
        binary_fitted,
        key=lambda row: (row["validation_accuracy"], -row["regularization"]),
    )
    raw_weighted_binary_weights = selected_binary_fit["weights"]
    uniform_binary_weights = np.full(
        features.shape[2], 1.0 / features.shape[2]
    )
    binary_shrinkage_fits = []
    for learned_share in BINARY_LEARNED_SHARE_GRID:
        fitted_mixture_weights = (
            learned_share * raw_weighted_binary_weights
            + (1.0 - learned_share) * uniform_binary_weights
        )
        weight_units = canonical_calibration_weight_units(
            fitted_mixture_weights
        )
        weights = weight_units.astype(float) / int(np.sum(weight_units))
        gains = reproducible_weighted_gains(binary_features, weight_units)
        rows = cardinal_rows(gains, validation_prompts, test_prompts)
        operating = select_operating_points(rows, "threshold")["0.99"]
        if operating is None:
            raise RuntimeError("weighted binary rule has no safe operating point")
        binary_shrinkage_fits.append(
            {
                "learned_share": learned_share,
                "fitted_mixture_weights": fitted_mixture_weights,
                "weight_units": weight_units,
                "weights": weights,
                "gains": gains,
                "rows": rows,
                "operating": operating,
                "validation_choice_accuracy": choice_accuracy(
                    gains, validation_prompts
                ),
            }
        )
    selected_binary_shrinkage = max(
        binary_shrinkage_fits,
        key=lambda row: (
            row["operating"]["validation"]["pure_response_completeness"],
            row["operating"]["validation"]["pure_response_soundness"],
            row["validation_choice_accuracy"],
            row["learned_share"],
        ),
    )
    weighted_binary_weights = selected_binary_shrinkage["weights"]
    weighted_binary_weight_units = selected_binary_shrinkage["weight_units"]
    weighted_binary_gains = selected_binary_shrinkage["gains"]

    validation_individual_accuracy = np.asarray(
        [
            choice_accuracy(features[:, :, reviewer], validation_prompts)
            for reviewer in range(features.shape[2])
        ]
    )
    best_reviewer = int(np.argmax(validation_individual_accuracy))
    reviewer_ranking = np.lexsort(
        (
            np.arange(features.shape[2]),
            -validation_individual_accuracy,
        )
    )
    if int(reviewer_ranking[0]) != best_reviewer:
        raise RuntimeError("validation ranking and selected reviewer disagree")
    single_gains = features[:, :, best_reviewer]

    training_individual_accuracy = np.asarray(
        [
            choice_accuracy(features[:, :, reviewer], training_prompts)
            for reviewer in range(features.shape[2])
        ]
    )
    training_reviewer_order = np.lexsort(
        (
            np.arange(features.shape[2]),
            -training_individual_accuracy,
        )
    )
    training_best_reviewer = int(training_reviewer_order[0])
    training_global_single_gains = features[:, :, training_best_reviewer]
    training_topic_single_gains = np.zeros_like(equal_gains)
    training_topic_single_metadata: dict[str, dict[str, object]] = {}

    expected_topic_splits = {
        "Factuality": (285, 95, 95),
        "Focus": (297, 99, 99),
        "Math": (110, 37, 36),
        "Precise IF": (96, 32, 32),
        "Safety": (270, 90, 90),
    }
    topic_cardinal_weights: dict[str, np.ndarray] = {}
    topic_binary_weights: dict[str, np.ndarray] = {}
    topic_single_reviewers: dict[str, int] = {}
    topic_metadata: dict[str, dict[str, object]] = {}
    topic_cardinal_gains = np.zeros_like(equal_gains)
    topic_binary_gains = np.zeros_like(equal_gains)
    topic_single_gains = np.zeros_like(equal_gains)
    routed_standard = np.zeros(len(standard_prompts), dtype=bool)

    for topic in TOPICS:
        topic_all = standard_prompts[subsets[standard_prompts] == topic]
        topic_training = topic_all[assignment[topic_all] == "train"]
        topic_validation = topic_all[assignment[topic_all] == "validation"]
        topic_test = topic_all[assignment[topic_all] == "test"]
        split_sizes = (
            len(topic_training),
            len(topic_validation),
            len(topic_test),
        )
        if split_sizes != expected_topic_splits[topic]:
            raise RuntimeError(f"unexpected split sizes for topic {topic}")

        topic_training_individual_accuracy = np.asarray(
            [
                choice_accuracy(
                    features[:, :, reviewer],
                    topic_training,
                )
                for reviewer in range(features.shape[2])
            ]
        )
        topic_training_reviewer_order = np.lexsort(
            (
                np.arange(features.shape[2]),
                -topic_training_individual_accuracy,
            )
        )
        topic_training_reviewer = int(topic_training_reviewer_order[0])
        training_topic_single_gains[topic_all] = features[
            topic_all,
            :,
            topic_training_reviewer,
        ]
        training_topic_single_metadata[topic] = {
            "reviewer_index": topic_training_reviewer,
            "model": str(models[topic_training_reviewer]),
            "training_choice_accuracy": float(
                topic_training_individual_accuracy[
                    topic_training_reviewer
                ]
            ),
            "validation_choice_accuracy": choice_accuracy(
                training_topic_single_gains,
                topic_validation,
            ),
            "test_choice_accuracy": choice_accuracy(
                training_topic_single_gains,
                topic_test,
            ),
            "prompts": {
                "training": int(len(topic_training)),
                "validation": int(len(topic_validation)),
                "test": int(len(topic_test)),
            },
        }

        cardinal_fits = []
        for regularization in REGULARIZATION_GRID:
            weights = fit_nonnegative_weights(
                features,
                topic_training,
                regularization,
            )
            gains = features @ weights
            cardinal_fits.append(
                {
                    "regularization": regularization,
                    "weights": weights,
                    "validation_accuracy": choice_accuracy(
                        gains,
                        topic_validation,
                    ),
                }
            )
        topic_cardinal_fit = max(
            cardinal_fits,
            key=lambda row: (
                row["validation_accuracy"],
                -row["regularization"],
            ),
        )
        cardinal_weights = topic_cardinal_fit["weights"]
        topic_cardinal_weights[topic] = cardinal_weights
        topic_cardinal_gains[topic_all] = (
            features[topic_all] @ cardinal_weights
        )

        topic_binary_fits = []
        for regularization in REGULARIZATION_GRID:
            weights = fit_nonnegative_weights(
                binary_features,
                topic_training,
                regularization,
            )
            gains = binary_features @ weights
            topic_binary_fits.append(
                {
                    "regularization": regularization,
                    "weights": weights,
                    "validation_accuracy": choice_accuracy(
                        gains,
                        topic_validation,
                    ),
                }
            )
        topic_binary_fit = max(
            topic_binary_fits,
            key=lambda row: (
                row["validation_accuracy"],
                -row["regularization"],
            ),
        )
        topic_binary_shrinkage_fits = []
        for learned_share in BINARY_LEARNED_SHARE_GRID:
            fitted_mixture_weights = (
                learned_share * topic_binary_fit["weights"]
                + (1.0 - learned_share) * uniform_binary_weights
            )
            weight_units = canonical_calibration_weight_units(
                fitted_mixture_weights
            )
            weights = weight_units.astype(float) / int(np.sum(weight_units))
            gains = reproducible_weighted_gains(
                binary_features,
                weight_units,
            )
            # Topic-local thresholds are used only to match the existing
            # validation-only shrinkage selection.  The final routed policy
            # below discards them and sweeps one pooled scalar threshold.
            validation_rows = cardinal_rows(
                gains,
                topic_validation,
                topic_validation,
            )
            operating = select_operating_points(
                validation_rows,
                "threshold",
            )["0.99"]
            if operating is None:
                raise RuntimeError(
                    f"topic {topic} has no binary shrinkage operating point"
                )
            topic_binary_shrinkage_fits.append(
                {
                    "learned_share": learned_share,
                    "fitted_mixture_weights": fitted_mixture_weights,
                    "weight_units": weight_units,
                    "weights": weights,
                    "operating": operating,
                    "validation_choice_accuracy": choice_accuracy(
                        gains,
                        topic_validation,
                    ),
                }
            )
        topic_binary_shrinkage = max(
            topic_binary_shrinkage_fits,
            key=lambda row: (
                row["operating"]["validation"][
                    "pure_response_completeness"
                ],
                row["operating"]["validation"][
                    "pure_response_soundness"
                ],
                row["validation_choice_accuracy"],
                row["learned_share"],
            ),
        )
        binary_weights = topic_binary_shrinkage["weights"]
        binary_weight_units = topic_binary_shrinkage["weight_units"]
        topic_binary_weights[topic] = binary_weights
        topic_binary_gains[topic_all] = reproducible_weighted_gains(
            binary_features[topic_all],
            binary_weight_units,
        )

        topic_individual_accuracy = np.asarray(
            [
                choice_accuracy(
                    features[:, :, reviewer],
                    topic_validation,
                )
                for reviewer in range(features.shape[2])
            ]
        )
        topic_reviewer_order = np.lexsort(
            (
                np.arange(features.shape[2]),
                -topic_individual_accuracy,
            )
        )
        topic_reviewer = int(topic_reviewer_order[0])
        topic_single_reviewers[topic] = topic_reviewer
        topic_single_gains[topic_all] = features[
            topic_all,
            :,
            topic_reviewer,
        ]

        topic_metadata[topic] = {
            "prompts": {
                "training": int(len(topic_training)),
                "validation": int(len(topic_validation)),
                "test": int(len(topic_test)),
            },
            "learned_weight": {
                "selected_regularization": float(
                    topic_cardinal_fit["regularization"]
                ),
                "validation_choice_accuracy": float(
                    topic_cardinal_fit["validation_accuracy"]
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_cardinal_gains,
                    topic_test,
                ),
                "nonzero_weights": int(
                    np.sum(cardinal_weights > 1e-8)
                ),
                "weights": {
                    model: float(weight)
                    for model, weight in zip(models, cardinal_weights)
                    if weight > 1e-8
                },
            },
            "weighted_binary": {
                "selected_regularization": float(
                    topic_binary_fit["regularization"]
                ),
                "raw_validation_choice_accuracy": float(
                    topic_binary_fit["validation_accuracy"]
                ),
                "learned_weight_share": float(
                    topic_binary_shrinkage["learned_share"]
                ),
                "uniform_weight_share": float(
                    1.0 - topic_binary_shrinkage["learned_share"]
                ),
                "validation_choice_accuracy": float(
                    topic_binary_shrinkage[
                        "validation_choice_accuracy"
                    ]
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_binary_gains,
                    topic_test,
                ),
                "shrinkage_selection_99_validation_operating_point": (
                    topic_binary_shrinkage["operating"]["validation"]
                ),
                "weights": {
                    model: float(weight)
                    for model, weight in zip(models, binary_weights)
                    if weight > 1e-8
                },
                "precanonical_weights_by_reviewer": {
                    str(model): float(weight)
                    for model, weight in zip(
                        models,
                        topic_binary_shrinkage[
                            "fitted_mixture_weights"
                        ],
                    )
                },
                "weight_units_by_reviewer": {
                    str(model): int(unit)
                    for model, unit in zip(models, binary_weight_units)
                },
                "weight_canonicalization": (
                    calibration_weight_canonicalization_metadata(
                        topic_binary_shrinkage[
                            "fitted_mixture_weights"
                        ],
                        binary_weight_units,
                        models,
                    )
                ),
            },
            "best_single": {
                "reviewer_index": topic_reviewer,
                "model": str(models[topic_reviewer]),
                "validation_choice_accuracy": float(
                    topic_individual_accuracy[topic_reviewer]
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_single_gains,
                    topic_test,
                ),
                "tie_break": "Original reviewer index, ascending.",
            },
        }
        topic_positions = np.searchsorted(standard_prompts, topic_all)
        routed_standard[topic_positions] = True

    if not np.all(routed_standard):
        raise RuntimeError("not every standard prompt received a topic route")
    for weights_by_topic in (
        topic_cardinal_weights,
        topic_binary_weights,
    ):
        if set(weights_by_topic) != set(TOPICS):
            raise RuntimeError("topic-conditioned weights omit a topic")
        for weights in weights_by_topic.values():
            if np.min(weights) < -TOLERANCE or not np.isclose(
                np.sum(weights),
                1.0,
                atol=TOLERANCE,
            ):
                raise RuntimeError("invalid topic-conditioned weights")

    # Calibration is unusually sensitive to values that lie exactly on one
    # of its tolerance-adjusted roots.  Canonicalize only the two learned
    # committee score rules, then use a fixed-order high-accuracy sum so the
    # validation-selected representative is reconstructible from the weights
    # serialized in cardinal_results.json on every supported platform.
    calibration_global_cardinal_weight_units = (
        canonical_calibration_weight_units(learned_weights)
    )
    calibration_global_cardinal_weights = (
        calibration_global_cardinal_weight_units.astype(float)
        / int(np.sum(calibration_global_cardinal_weight_units))
    )
    calibration_global_cardinal_gains = reproducible_weighted_gains(
        features,
        calibration_global_cardinal_weight_units,
    )
    calibration_topic_cardinal_weight_units = {
        topic: canonical_calibration_weight_units(
            topic_cardinal_weights[topic]
        )
        for topic in TOPICS
    }
    calibration_topic_cardinal_weights = {
        topic: units.astype(float) / int(np.sum(units))
        for topic, units in calibration_topic_cardinal_weight_units.items()
    }
    calibration_topic_cardinal_gains = np.zeros_like(equal_gains)
    for topic in TOPICS:
        topic_all = standard_prompts[subsets[standard_prompts] == topic]
        calibration_topic_cardinal_gains[topic_all] = (
            reproducible_weighted_gains(
                features[topic_all],
                calibration_topic_cardinal_weight_units[topic],
            )
        )

    (
        training_global_single_rows,
        training_global_single_test_errors,
    ) = train_test_cardinal_sweep(
        training_global_single_gains,
        training_prompts,
        test_prompts,
    )
    (
        training_topic_single_rows,
        training_topic_single_test_errors,
    ) = train_test_cardinal_sweep(
        training_topic_single_gains,
        training_prompts,
        test_prompts,
    )
    training_global_single_checkpoints = (
        train_test_cardinal_fixed_checkpoints(
            training_global_single_gains,
            training_prompts,
            test_prompts,
        )
    )
    training_topic_single_checkpoints = (
        train_test_cardinal_fixed_checkpoints(
            training_topic_single_gains,
            training_prompts,
            test_prompts,
        )
    )

    global_cardinal_score_rule = {
        "kind": "learned cardinal weighted score",
        "parameter_scope": "global",
        "weight_fit_split": "training",
        "regularization_selection_split": "validation",
        "selected_regularization": float(selected_fit["regularization"]),
        "weights_by_reviewer": {
            str(model): float(weight)
            for model, weight in zip(
                models,
                calibration_global_cardinal_weights,
            )
        },
        "calibration_weight_units_by_reviewer": {
            str(model): int(unit)
            for model, unit in zip(
                models,
                calibration_global_cardinal_weight_units,
            )
        },
        "calibration_weight_canonicalization": (
            calibration_weight_canonicalization_metadata(
                learned_weights,
                calibration_global_cardinal_weight_units,
                models,
            )
        ),
        "same_frozen_score_rule_in_every_topic": True,
        "frozen_before_threshold_calibration": True,
        "test_labels_used": False,
    }
    topic_cardinal_score_rule = {
        "kind": "learned cardinal weighted score",
        "parameter_scope": "topic-routed",
        "weight_fit_split": "within-topic training",
        "regularization_selection_split": "within-topic validation",
        "topics": {
            topic: {
                "selected_regularization": float(
                    topic_metadata[topic]["learned_weight"][
                        "selected_regularization"
                    ]
                ),
                "weights_by_reviewer": {
                    str(model): float(weight)
                    for model, weight in zip(
                        models,
                        calibration_topic_cardinal_weights[topic],
                    )
                },
                "calibration_weight_units_by_reviewer": {
                    str(model): int(unit)
                    for model, unit in zip(
                        models,
                        calibration_topic_cardinal_weight_units[topic],
                    )
                },
                "calibration_weight_canonicalization": (
                    calibration_weight_canonicalization_metadata(
                        topic_cardinal_weights[topic],
                        calibration_topic_cardinal_weight_units[topic],
                        models,
                    )
                ),
            }
            for topic in TOPICS
        },
        "frozen_before_threshold_calibration": True,
        "test_labels_used": False,
    }
    global_singleton_score_rule = {
        "kind": "single reviewer score",
        "parameter_scope": "global",
        "reviewer_selection_split": "training",
        "reviewer_selection_metric": (
            "correct-response choice accuracy with response ties split "
            "uniformly"
        ),
        "reviewer_index": int(training_best_reviewer),
        "model": str(models[training_best_reviewer]),
        "training_choice_accuracy": float(
            training_individual_accuracy[training_best_reviewer]
        ),
        "same_frozen_score_rule_in_every_topic": True,
        "frozen_before_threshold_calibration": True,
        "test_labels_used": False,
    }
    topic_singleton_score_rule = {
        "kind": "single reviewer score",
        "parameter_scope": "topic-routed",
        "reviewer_selection_split": "within-topic training",
        "reviewer_selection_metric": (
            "correct-response choice accuracy with response ties split "
            "uniformly"
        ),
        "reviewers_by_topic": {
            topic: {
                "reviewer_index": int(
                    training_topic_single_metadata[topic]["reviewer_index"]
                ),
                "model": str(
                    training_topic_single_metadata[topic]["model"]
                ),
                "training_choice_accuracy": float(
                    training_topic_single_metadata[topic][
                        "training_choice_accuracy"
                    ]
                ),
            }
            for topic in TOPICS
        },
        "frozen_before_threshold_calibration": True,
        "test_labels_used": False,
    }

    topic_threshold_calibration_families = {
        "learned_topic_cardinal": validation_budget_threshold_calibration(
            calibration_topic_cardinal_gains,
            subsets,
            validation_prompts,
            test_prompts,
            (
                "After deterministic eight-decimal weight canonicalization, "
                "tolerance-aware robust critical thresholds, adjacent floats, "
                "and open-interval representatives are derived exclusively "
                "from the routed validation scores."
            ),
            topic_cardinal_score_rule,
        ),
        "global_cardinal_topic_threshold_control": (
            validation_budget_threshold_calibration(
                calibration_global_cardinal_gains,
                subsets,
                validation_prompts,
                test_prompts,
                (
                    "After deterministic eight-decimal weight "
                    "canonicalization, tolerance-aware robust critical "
                    "thresholds, adjacent floats, and open-interval "
                    "representatives are derived exclusively from the "
                    "global-cardinal validation scores."
                ),
                global_cardinal_score_rule,
            )
        ),
        "train_selected_global_singleton_topic_threshold_control": (
            validation_budget_threshold_calibration(
                training_global_single_gains,
                subsets,
                validation_prompts,
                test_prompts,
                (
                    "After the global reviewer identity is selected on "
                    "training only, tolerance-aware robust critical "
                    "thresholds, adjacent floats, and open-interval "
                    "representatives are derived exclusively from its "
                    "validation scores."
                ),
                global_singleton_score_rule,
            )
        ),
        "train_selected_topic_singleton": (
            validation_budget_threshold_calibration(
                training_topic_single_gains,
                subsets,
                validation_prompts,
                test_prompts,
                (
                    "After reviewer identities are selected on training only, "
                    "tolerance-aware robust critical thresholds, adjacent "
                    "floats, and open-interval representatives are derived "
                    "exclusively from their routed validation scores."
                ),
                topic_singleton_score_rule,
            )
        ),
    }

    retained_reviewers = np.arange(features.shape[2]) != best_reviewer
    ablated_features = features[:, :, retained_reviewers]
    ablated_fitted = []
    for regularization in REGULARIZATION_GRID:
        weights = fit_nonnegative_weights(
            ablated_features,
            training_prompts,
            regularization,
        )
        gains = ablated_features @ weights
        ablated_fitted.append(
            {
                "regularization": regularization,
                "weights": weights,
                "validation_accuracy": choice_accuracy(
                    gains,
                    validation_prompts,
                ),
            }
        )
    selected_ablated_fit = max(
        ablated_fitted,
        key=lambda row: (row["validation_accuracy"], -row["regularization"]),
    )
    ablated_weights = selected_ablated_fit["weights"]
    ablated_gains = ablated_features @ ablated_weights

    (
        ranked_single_soundness,
        ranked_single_completeness,
        ranked_single_attained,
    ) = ranked_single_reviewer_errors(
        features,
        test_prompts,
        reviewer_ranking,
    )
    (
        ranked_prefix_soundness,
        ranked_prefix_completeness,
        ranked_prefix_attained,
    ) = ranked_prefix_unanimity_errors(
        scores,
        offsets,
        test_prompts,
        reviewer_ranking,
        ranked_single_completeness,
        ranked_single_attained,
    )
    if not (
        np.allclose(
            ranked_prefix_soundness[:, 0],
            ranked_single_soundness[:, 0],
            atol=1e-12,
            rtol=0.0,
        )
        and np.allclose(
            ranked_prefix_completeness[:, 0],
            ranked_single_completeness[:, 0],
            atol=1e-12,
            rtol=0.0,
        )
        and np.array_equal(
            ranked_prefix_attained[:, 0],
            ranked_single_attained[:, 0],
        )
    ):
        raise RuntimeError("top ranked prefix and singleton disagree")

    ranked_single = []
    ranked_prefix = []
    for column, reviewer in enumerate(reviewer_ranking):
        single_exact_soundness = float(
            np.mean(
                ranked_single_soundness[:, column]
                <= 5 * GEOMETRY_TOLERANCE
            )
        )
        single_exact_completeness = float(
            np.mean(
                epsilon_complete(
                    ranked_single_completeness[:, column],
                    ranked_single_attained[:, column],
                    0.0,
                )
            )
        )
        ranked_single.append(
            {
                "removed_top_reviewers": int(column),
                "reviewer_rank": int(column + 1),
                "reviewer_index": int(reviewer),
                "model": str(models[reviewer]),
                "validation_choice_accuracy": float(
                    validation_individual_accuracy[reviewer]
                ),
                "test_choice_accuracy": choice_accuracy(
                    features[:, :, reviewer],
                    test_prompts,
                ),
                "test_full_domain_soundness": single_exact_soundness,
                "test_full_domain_completeness": (
                    single_exact_completeness
                ),
            }
        )
        prefix_exact_soundness = float(
            np.mean(
                ranked_prefix_soundness[:, column]
                <= 5 * GEOMETRY_TOLERANCE
            )
        )
        prefix_exact_completeness = float(
            np.mean(
                epsilon_complete(
                    ranked_prefix_completeness[:, column],
                    ranked_prefix_attained[:, column],
                    0.0,
                )
            )
        )
        ranked_prefix.append(
            {
                "panel_size": int(column + 1),
                "added_reviewer_index": int(reviewer),
                "added_model": str(models[reviewer]),
                "validation_choice_accuracy": float(
                    validation_individual_accuracy[reviewer]
                ),
                "test_full_domain_soundness": prefix_exact_soundness,
                "test_full_domain_completeness": (
                    prefix_exact_completeness
                ),
            }
        )
    prefix_soundness_rates = np.asarray(
        [row["test_full_domain_soundness"] for row in ranked_prefix]
    )
    if np.any(np.diff(prefix_soundness_rates) < -TOLERANCE):
        raise RuntimeError("ranked-prefix soundness is not monotone")

    count = count_rows(features, validation_prompts, test_prompts)
    weighted_binary = selected_binary_shrinkage["rows"]
    single = cardinal_rows(single_gains, validation_prompts, test_prompts)
    equal = cardinal_rows(equal_gains, validation_prompts, test_prompts)
    learned = cardinal_rows(learned_gains, validation_prompts, test_prompts)
    ablated_learned = cardinal_rows(
        ablated_gains,
        validation_prompts,
        test_prompts,
    )
    topic_weighted_binary = cardinal_rows(
        topic_binary_gains,
        validation_prompts,
        test_prompts,
    )
    topic_learned = cardinal_rows(
        topic_cardinal_gains,
        validation_prompts,
        test_prompts,
    )
    topic_single = cardinal_rows(
        topic_single_gains,
        validation_prompts,
        test_prompts,
    )
    for row in count:
        k = int(row["k"])
        row["test_full_domain_soundness"] = float(
            1.0 if k < 0 else np.mean(exact_depth[test_prompts] >= k + 1)
        )
        row["test_full_domain_completeness"] = float(
            0.0
            if k < 0
            else np.mean(maximum_beneficial_objections[test_prompts] <= k)
        )
    (
        weighted_binary_maximum_harmful_approval,
        weighted_binary_minimum_beneficial_approval,
        independently_computed_test_depth,
        independently_computed_test_beneficial_objections,
    ) = exact_weighted_binary_audit(
        scores,
        offsets,
        test_prompts,
        weighted_binary_weights,
    )
    if not np.array_equal(
        independently_computed_test_depth, exact_depth[test_prompts]
    ):
        raise RuntimeError(
            "weighted binary arrangement audit disagrees with exact depth"
        )
    if not np.array_equal(
        independently_computed_test_beneficial_objections,
        maximum_beneficial_objections[test_prompts],
    ):
        raise RuntimeError(
            "weighted binary arrangement audit disagrees with full-domain "
            "completeness"
        )
    for row in weighted_binary:
        threshold = row["threshold"]
        numeric_threshold = (
            np.inf
            if threshold == "reject_all"
            else -np.inf
            if threshold == "accept_all"
            else float(threshold)
        )
        row["test_full_domain_soundness"] = (
            weighted_binary_full_domain_soundness_rate(
                weighted_binary_maximum_harmful_approval,
                numeric_threshold,
            )
        )
        row["test_full_domain_completeness"] = (
            weighted_binary_full_domain_completeness_rate(
                weighted_binary_minimum_beneficial_approval,
                numeric_threshold,
            )
        )
    (
        topic_binary_maximum_harmful_approval,
        topic_binary_minimum_beneficial_approval,
        topic_binary_test_depth,
        topic_binary_test_beneficial_objections,
    ) = exact_topic_weighted_binary_audit(
        scores,
        offsets,
        test_prompts,
        subsets,
        topic_binary_weights,
    )
    if not np.array_equal(topic_binary_test_depth, exact_depth[test_prompts]):
        raise RuntimeError(
            "topic-weighted binary audit disagrees with exact depth"
        )
    if not np.array_equal(
        topic_binary_test_beneficial_objections,
        maximum_beneficial_objections[test_prompts],
    ):
        raise RuntimeError(
            "topic-weighted binary audit disagrees with completeness depth"
        )
    for row in topic_weighted_binary:
        threshold = row["threshold"]
        numeric_threshold = (
            np.inf
            if threshold == "reject_all"
            else -np.inf
            if threshold == "accept_all"
            else float(threshold)
        )
        row["test_full_domain_soundness"] = (
            weighted_binary_full_domain_soundness_rate(
                topic_binary_maximum_harmful_approval,
                numeric_threshold,
            )
        )
        row["test_full_domain_completeness"] = (
            weighted_binary_full_domain_completeness_rate(
                topic_binary_minimum_beneficial_approval,
                numeric_threshold,
            )
        )
    cardinal_error_data: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for key, rows, gains in (
        ("best_single", single, single_gains),
        ("equal_weight", equal, equal_gains),
        ("learned_weight", learned, learned_gains),
        ("ablated_learned_weight", ablated_learned, ablated_gains),
        (
            "topic_conditioned_learned_weight",
            topic_learned,
            topic_cardinal_gains,
        ),
        (
            "topic_conditioned_best_single",
            topic_single,
            topic_single_gains,
        ),
    ):
        for row in rows:
            threshold = row["threshold"]
            numeric_threshold = (
                np.inf
                if threshold == "reject_all"
                else -np.inf
                if threshold == "accept_all"
                else float(threshold)
            )
            row["test_full_domain_soundness"] = (
                cardinal_full_domain_soundness_rate(
                    gains, numeric_threshold, test_prompts
                )
            )
            row["test_full_domain_completeness"] = (
                cardinal_full_domain_completeness_rate(
                    gains, numeric_threshold, test_prompts
                )
            )

    def numeric_thresholds(rows: list[dict[str, object]]) -> np.ndarray:
        return np.asarray(
            [
                np.inf
                if row["threshold"] == "reject_all"
                else -np.inf
                if row["threshold"] == "accept_all"
                else float(row["threshold"])
                for row in rows
            ]
        )

    def attach_approximate_rates(
        rows: list[dict[str, object]],
        soundness_error: np.ndarray,
        completeness_error: np.ndarray,
        completeness_attained: np.ndarray,
    ) -> None:
        approximate_complete = epsilon_complete(
            completeness_error,
            completeness_attained,
            APPROXIMATE_GOOD,
        )
        exact_sound_rate = np.mean(
            soundness_error <= 5 * GEOMETRY_TOLERANCE,
            axis=0,
        )
        exact_complete_rate = np.mean(
            epsilon_complete(
                completeness_error,
                completeness_attained,
                0.0,
            ),
            axis=0,
        )
        stored_sound_rate = np.asarray(
            [row["test_full_domain_soundness"] for row in rows]
        )
        stored_complete_rate = np.asarray(
            [row["test_full_domain_completeness"] for row in rows]
        )
        if not np.allclose(
            exact_sound_rate,
            stored_sound_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("approximate errors disagree with exact soundness")
        if not np.allclose(
            exact_complete_rate,
            stored_complete_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("approximate errors disagree with exact completeness")
        for column, row in enumerate(rows):
            row["test_approximate_full_domain"] = {
                "epsilon_good_completeness": float(
                    np.mean(approximate_complete[:, column])
                ),
            }

    with np.load(APPROXIMATE_DATA, allow_pickle=True) as data:
        approximate_prompts = data["standard_prompts"].astype(int)
        count_soundness = data["rewardbench_soundness_error"].astype(float)
        count_completeness = data[
            "rewardbench_completeness_error"
        ].astype(float)
        count_attained = data[
            "rewardbench_completeness_supremum_attained"
        ].astype(bool)
    if not np.array_equal(approximate_prompts, standard_prompts):
        raise RuntimeError("approximate archive uses a different prompt cohort")
    test_positions = np.searchsorted(standard_prompts, test_prompts)
    count_soundness = np.column_stack(
        [
            np.zeros(len(test_prompts)),
            count_soundness[test_positions],
            np.full(len(test_prompts), 0.25),
        ]
    )
    count_completeness = np.column_stack(
        [
            np.full(len(test_prompts), 0.75),
            count_completeness[test_positions],
            np.zeros(len(test_prompts)),
        ]
    )
    count_attained = np.column_stack(
        [
            np.ones(len(test_prompts), dtype=bool),
            count_attained[test_positions],
            np.zeros(len(test_prompts), dtype=bool),
        ]
    )
    attach_approximate_rates(
        count,
        count_soundness,
        count_completeness,
        count_attained,
    )
    if not np.allclose(
        ranked_prefix_soundness[:, -1],
        count_soundness[:, 1],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "full ranked prefix disagrees with count-rule unanimity soundness"
        )
    if not np.allclose(
        ranked_prefix_completeness[:, -1],
        count_completeness[:, 1],
        atol=1e-12,
        rtol=0.0,
    ) or not np.array_equal(
        ranked_prefix_attained[:, -1],
        count_attained[:, 1],
    ):
        raise RuntimeError(
            "full ranked prefix disagrees with count-rule unanimity completeness"
        )

    weighted_thresholds = numeric_thresholds(weighted_binary)
    weighted_soundness = np.zeros((len(test_prompts), len(weighted_thresholds)))
    weighted_completeness = np.zeros_like(weighted_soundness)
    weighted_attained = np.zeros_like(weighted_soundness, dtype=bool)
    for position, prompt in enumerate(test_prompts):
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, _ = multicandidate_utilities(scores, start, stop, 1)
        (
            weighted_soundness[position],
            weighted_completeness[position],
            weighted_attained[position],
        ) = weighted_binary_prompt_errors(
            reviewers,
            weighted_binary_weights,
            weighted_thresholds,
        )
        if (position + 1) % 50 == 0:
            print(
                "weighted-binary approximate audit "
                f"{position + 1}/{len(test_prompts)}"
            )
    attach_approximate_rates(
        weighted_binary,
        weighted_soundness,
        weighted_completeness,
        weighted_attained,
    )
    topic_weighted_thresholds = numeric_thresholds(topic_weighted_binary)
    (
        topic_weighted_soundness,
        topic_weighted_completeness,
        topic_weighted_attained,
    ) = topic_weighted_binary_prompt_errors(
        scores,
        offsets,
        test_prompts,
        subsets,
        topic_binary_weights,
        topic_weighted_thresholds,
    )
    attach_approximate_rates(
        topic_weighted_binary,
        topic_weighted_soundness,
        topic_weighted_completeness,
        topic_weighted_attained,
    )

    for key, rows, gains in (
        ("best_single", single, single_gains),
        ("equal_weight", equal, equal_gains),
        ("learned_weight", learned, learned_gains),
        ("ablated_learned_weight", ablated_learned, ablated_gains),
        (
            "topic_conditioned_learned_weight",
            topic_learned,
            topic_cardinal_gains,
        ),
        (
            "topic_conditioned_best_single",
            topic_single,
            topic_single_gains,
        ),
    ):
        soundness_error, completeness_error, completeness_attained = (
            cardinal_prompt_errors(
                gains,
                test_prompts,
                numeric_thresholds(rows),
            )
        )
        attach_approximate_rates(
            rows,
            soundness_error,
            completeness_error,
            completeness_attained,
        )
        cardinal_error_data[key] = (
            soundness_error,
            completeness_error,
            completeness_attained,
        )
    for key, test_errors in (
        (
            "train_selected_global_singleton",
            training_global_single_test_errors,
        ),
        (
            "train_selected_topic_singleton",
            training_topic_single_test_errors,
        ),
    ):
        cardinal_error_data[key] = test_errors

    def attach_topic_test_rates(
        rows: list[dict[str, object]],
        soundness_error: np.ndarray,
        completeness_error: np.ndarray,
        completeness_attained: np.ndarray,
    ) -> None:
        exact_complete = epsilon_complete(
            completeness_error,
            completeness_attained,
            0.0,
        )
        approximate_complete = epsilon_complete(
            completeness_error,
            completeness_attained,
            APPROXIMATE_GOOD,
        )
        test_topics = subsets[test_prompts]
        for column, row in enumerate(rows):
            by_topic = {}
            for topic in TOPICS:
                selected = test_topics == topic
                if not np.any(selected):
                    raise RuntimeError(f"test split omits topic {topic}")
                by_topic[topic] = {
                    "prompts": int(np.sum(selected)),
                    "full_domain_soundness": float(
                        np.mean(
                            soundness_error[selected, column]
                            <= 5 * GEOMETRY_TOLERANCE
                        )
                    ),
                    "full_domain_completeness": float(
                        np.mean(exact_complete[selected, column])
                    ),
                    "epsilon_good_completeness": float(
                        np.mean(approximate_complete[selected, column])
                    ),
                }
            row["test_by_topic"] = by_topic
            for topic_key, pooled_value in (
                (
                    "full_domain_soundness",
                    row["test_full_domain_soundness"],
                ),
                (
                    "full_domain_completeness",
                    row["test_full_domain_completeness"],
                ),
                (
                    "epsilon_good_completeness",
                    row["test_approximate_full_domain"][
                        "epsilon_good_completeness"
                    ],
                ),
            ):
                reconstructed = sum(
                    values["prompts"] * values[topic_key]
                    for values in by_topic.values()
                ) / len(test_prompts)
                if not np.isclose(
                    reconstructed,
                    pooled_value,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "topic metrics do not reconstruct pooled rate"
                    )

    attach_topic_test_rates(
        topic_weighted_binary,
        topic_weighted_soundness,
        topic_weighted_completeness,
        topic_weighted_attained,
    )
    attach_topic_test_rates(
        topic_learned,
        *cardinal_error_data["topic_conditioned_learned_weight"],
    )
    attach_topic_test_rates(
        topic_single,
        *cardinal_error_data["topic_conditioned_best_single"],
    )
    attach_approximate_rates(
        ranked_single,
        ranked_single_soundness,
        ranked_single_completeness,
        ranked_single_attained,
    )
    attach_approximate_rates(
        ranked_prefix,
        ranked_prefix_soundness,
        ranked_prefix_completeness,
        ranked_prefix_attained,
    )
    ranked_single_tradeoff = ranked_single_approximate_tradeoff(
        ranked_single_soundness,
        ranked_single_completeness,
        ranked_single_attained,
        reviewer_ranking,
        models,
    )

    count_operating = select_operating_points(count, "k")
    weighted_binary_operating = select_operating_points(
        weighted_binary, "threshold"
    )
    single_operating = select_operating_points(single, "threshold")
    equal_operating = select_operating_points(equal, "threshold")
    learned_operating = select_operating_points(learned, "threshold")
    ablated_learned_operating = select_operating_points(
        ablated_learned,
        "threshold",
    )
    topic_weighted_binary_operating = select_operating_points(
        topic_weighted_binary,
        "threshold",
    )
    topic_learned_operating = select_operating_points(
        topic_learned,
        "threshold",
    )
    topic_single_operating = select_operating_points(
        topic_single,
        "threshold",
    )

    rows_by_key = {
        "count_rule": count,
        "weighted_binary": weighted_binary,
        "equal_weight": equal,
        "learned_weight": learned,
        "ablated_learned_weight": ablated_learned,
        "best_single": single,
        "ranked_single_reviewer": ranked_single,
        "ranked_prefix_unanimity": ranked_prefix,
        "topic_conditioned_weighted_binary": topic_weighted_binary,
        "topic_conditioned_learned_weight": topic_learned,
        "topic_conditioned_best_single": topic_single,
        "train_selected_global_singleton": training_global_single_rows,
        "train_selected_topic_singleton": training_topic_single_rows,
    }
    error_data = {
        "count_rule": (
            count_soundness,
            count_completeness,
            count_attained,
        ),
        "weighted_binary": (
            weighted_soundness,
            weighted_completeness,
            weighted_attained,
        ),
        **cardinal_error_data,
        "ranked_single_reviewer": (
            ranked_single_soundness,
            ranked_single_completeness,
            ranked_single_attained,
        ),
        "ranked_prefix_unanimity": (
            ranked_prefix_soundness,
            ranked_prefix_completeness,
            ranked_prefix_attained,
        ),
        "topic_conditioned_weighted_binary": (
            topic_weighted_soundness,
            topic_weighted_completeness,
            topic_weighted_attained,
        ),
    }

    required_sound_prompts = int(
        np.ceil(SOUNDNESS_COVERAGE_TARGET * len(test_prompts) - TOLERANCE)
    )
    required_tolerance_by_rule = {}
    for key, (soundness_error, _, _) in error_data.items():
        required_tolerance_by_rule[key] = np.sort(
            soundness_error,
            axis=0,
        )[required_sound_prompts - 1]
    soundness_tolerances = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.05, 0.10, 0.15, 0.20, 0.25]),
                *required_tolerance_by_rule.values(),
            ]
        )
    )
    soundness_tolerances = soundness_tolerances[
        (soundness_tolerances >= -GEOMETRY_TOLERANCE)
        & (soundness_tolerances <= 0.25 + GEOMETRY_TOLERANCE)
    ]
    soundness_tolerances = np.unique(
        np.clip(soundness_tolerances, 0.0, 0.25)
    )
    soundness_tolerance_sensitivity: dict[str, object] = {
        "coverage_target": SOUNDNESS_COVERAGE_TARGET,
        "epsilon_good": APPROXIMATE_GOOD,
        "implicit_reject_all_when_no_setting_is_feasible": True,
        "implicit_accept_all_at_vacuous_endpoint": True,
        "epsilon_harm": soundness_tolerances.tolist(),
        "series": {},
    }
    for key, rows in rows_by_key.items():
        soundness_error, completeness_error, completeness_attained = (
            error_data[key]
        )
        completeness_rates = np.mean(
            epsilon_complete(
                completeness_error,
                completeness_attained,
                APPROXIMATE_GOOD,
            ),
            axis=0,
        )
        best_completeness = []
        achieved_soundness = []
        selected_index = []
        for epsilon_harm in soundness_tolerances:
            if epsilon_harm >= 0.25 - 5 * GEOMETRY_TOLERANCE:
                best_completeness.append(1.0)
                achieved_soundness.append(1.0)
                selected_index.append(-2)
                continue
            feasible = np.flatnonzero(
                required_tolerance_by_rule[key]
                <= epsilon_harm + 5 * GEOMETRY_TOLERANCE
            )
            if len(feasible) == 0:
                # The fixed native-binary families need not contain a setting
                # that protects 95% of prompts at a small loss tolerance.  Use
                # the always-safe reject-all rule as their conservative
                # baseline until a ranked setting becomes feasible.
                best_completeness.append(0.0)
                achieved_soundness.append(1.0)
                selected_index.append(-1)
                continue
            best_rate = np.max(completeness_rates[feasible])
            best_indices = feasible[
                np.isclose(
                    completeness_rates[feasible],
                    best_rate,
                    atol=TOLERANCE,
                    rtol=0.0,
                )
            ]
            index = int(best_indices[-1])
            soundness_rate = float(
                np.mean(
                    soundness_error[:, index]
                    <= epsilon_harm + 5 * GEOMETRY_TOLERANCE
                )
            )
            if soundness_rate < SOUNDNESS_COVERAGE_TARGET - TOLERANCE:
                raise RuntimeError("selected threshold misses soundness target")
            best_completeness.append(float(best_rate))
            achieved_soundness.append(soundness_rate)
            selected_index.append(index)
        soundness_tolerance_sensitivity["series"][key] = {
            "best_completeness": best_completeness,
            "achieved_soundness": achieved_soundness,
            "selected_index": selected_index,
        }

    gain_margins = np.linspace(0.0, 0.75, 151)
    margin_sensitivity: dict[str, object] = {
        "soundness_coverage_target": SOUNDNESS_COVERAGE_TARGET,
        "gain_margins": gain_margins.tolist(),
        "series": {},
    }
    for key in (
        "learned_weight",
        "ablated_learned_weight",
        "best_single",
    ):
        rows = rows_by_key[key]
        _, completeness_error, completeness_attained = error_data[key]
        exact_soundness_rates = np.asarray(
            [row["test_full_domain_soundness"] for row in rows]
        )
        feasible = np.flatnonzero(
            exact_soundness_rates
            >= SOUNDNESS_COVERAGE_TARGET - TOLERANCE
        )
        if len(feasible) == 0:
            raise RuntimeError("no threshold meets margin-sensitivity target")
        threshold_index = int(feasible[-1])
        completeness_profile = [
            float(
                np.mean(
                    epsilon_complete(
                        completeness_error[:, threshold_index],
                        completeness_attained[:, threshold_index],
                        margin,
                    )
                )
            )
            for margin in gain_margins
        ]
        margin_sensitivity["series"][key] = {
            "achieved_soundness": float(
                exact_soundness_rates[threshold_index]
            ),
            "threshold": rows[threshold_index]["threshold"],
            "completeness": completeness_profile,
        }

    if not np.all(depth_is_exact[test_prompts]):
        raise RuntimeError("held-out full-domain depths are not exact")
    for target in (f"{value:.2f}" for value in TARGET_SOUNDNESS):
        count_point = count_operating[target]
        if count_point is not None:
            k = int(count_point["k"])
            count_point["test_full_domain_soundness"] = float(
                1.0 if k < 0 else np.mean(exact_depth[test_prompts] >= k + 1)
            )
            count_point["test_full_domain_completeness"] = float(
                0.0
                if k < 0
                else np.mean(
                    maximum_beneficial_objections[test_prompts] <= k
                )
            )
        binary_point = weighted_binary_operating[target]
        if binary_point is not None:
            threshold = binary_point["threshold"]
            numeric_threshold = (
                np.inf
                if threshold == "reject_all"
                else -np.inf
                if threshold == "accept_all"
                else float(threshold)
            )
            binary_point["test_full_domain_soundness"] = (
                weighted_binary_full_domain_soundness_rate(
                    weighted_binary_maximum_harmful_approval,
                    numeric_threshold,
                )
            )
            binary_point["test_full_domain_completeness"] = (
                weighted_binary_full_domain_completeness_rate(
                    weighted_binary_minimum_beneficial_approval,
                    numeric_threshold,
                )
            )
        topic_binary_point = topic_weighted_binary_operating[target]
        if topic_binary_point is not None:
            threshold = topic_binary_point["threshold"]
            numeric_threshold = (
                np.inf
                if threshold == "reject_all"
                else -np.inf
                if threshold == "accept_all"
                else float(threshold)
            )
            topic_binary_point["test_full_domain_soundness"] = (
                weighted_binary_full_domain_soundness_rate(
                    topic_binary_maximum_harmful_approval,
                    numeric_threshold,
                )
            )
            topic_binary_point["test_full_domain_completeness"] = (
                weighted_binary_full_domain_completeness_rate(
                    topic_binary_minimum_beneficial_approval,
                    numeric_threshold,
                )
            )
        for point, gains in (
            (single_operating[target], single_gains),
            (equal_operating[target], equal_gains),
            (learned_operating[target], learned_gains),
            (ablated_learned_operating[target], ablated_gains),
            (topic_single_operating[target], topic_single_gains),
            (topic_learned_operating[target], topic_cardinal_gains),
        ):
            if point is None:
                continue
            threshold = point["threshold"]
            numeric_threshold = (
                np.inf
                if threshold == "reject_all"
                else -np.inf
                if threshold == "accept_all"
                else float(threshold)
            )
            point["test_full_domain"] = full_domain_metrics(
                gains, numeric_threshold, test_prompts
            )
            fast_rate = cardinal_full_domain_soundness_rate(
                gains, numeric_threshold, test_prompts
            )
            if not np.isclose(
                fast_rate,
                point["test_full_domain"]["full_domain_soundness_rate"],
                atol=1e-12,
            ):
                raise RuntimeError("closed-form and LP soundness audits disagree")

    prompt_specific_reconstruction = prompt_specific_cardinal_reconstruction(
        scores, offsets, standard_prompts
    )
    if prompt_specific_reconstruction["covered_prompts"] != int(
        np.sum(exact_depth[standard_prompts] >= 1)
    ):
        raise RuntimeError("reconstruction LP and exact coverage disagree")
    prompt_specific_reconstruction.update(
        {
            "test_prompts": int(len(test_prompts)),
            "test_covered_prompts": int(
                np.sum(exact_depth[test_prompts] >= 1)
            ),
            "test_full_domain_soundness_rate": 1.0,
            "test_full_domain_completeness_rate": float(
                np.mean(exact_depth[test_prompts] >= 1)
            ),
        }
    )
    best_single_alignment = single_reviewer_alignment(
        scores, offsets, standard_prompts, best_reviewer
    )

    output: dict[str, object] = {
        "seed": SEED,
        "dataset_revision": dataset_revision,
        "results_revision": results_revision,
        "cohort": {
            "standard_prompts": int(len(standard_prompts)),
            "training_prompts": int(len(training_prompts)),
            "validation_prompts": int(len(validation_prompts)),
            "test_prompts": int(len(test_prompts)),
        },
        "normalization": (
            "Scores are centered at the prompt-specific uniform fallback and "
            "L2-normalized within reviewer and prompt."
        ),
        "approximate_margins": {
            "epsilon_good": APPROXIMATE_GOOD,
        },
        "soundness_tolerance_sensitivity": (
            soundness_tolerance_sensitivity
        ),
        "margin_sensitivity": margin_sensitivity,
        "topic_threshold_budget_calibration": {
            "seed": SEED,
            "dataset_revision": dataset_revision,
            "results_revision": results_revision,
            "cohort": {
                "training_prompts": int(len(training_prompts)),
                "validation_prompts": int(len(validation_prompts)),
                "test_prompts": int(len(test_prompts)),
            },
            "selection_split": "validation",
            "test_labels_used_for_selection": False,
            "score_functions_frozen_before_threshold_calibration": True,
            "topics_order": list(TOPICS),
            "epsilon_good": APPROXIMATE_GOOD,
            "full_domain_audit": (
                "Exact complete-simplex/full-lottery calculation; response "
                "lotteries are not sampled."
            ),
            "ties_approve": True,
            "validation_failure_budgets": {
                "minimum": 0,
                "maximum": int(len(validation_prompts)),
                "values": list(range(len(validation_prompts) + 1)),
                "primary_95_soundness_budget": int(
                    len(validation_prompts)
                    - np.ceil(
                        SOUNDNESS_COVERAGE_TARGET
                        * len(validation_prompts)
                        - TOLERANCE
                    )
                ),
                "checkpoints": {
                    f"{target:.2f}": int(
                        len(validation_prompts)
                        - np.ceil(
                            target * len(validation_prompts) - TOLERANCE
                        )
                    )
                    for target in (0.99, 0.95, 0.90, 0.80)
                },
            },
            "objective": (
                "For every pooled validation exact-soundness failure budget, "
                "maximize the number of validation prompts on which every "
                "proposal with principal gain at least 0.5 is authorized."
            ),
            "candidate_grid": (
                "For each validation score rule, enumerate every tolerance-"
                "adjusted root at which exact-soundness or 0.5-completeness "
                "can change, both adjacent floating-point values, one "
                "representative of every open interval, reject-all, and "
                "accept-all. The grid is behaviorally exhaustive for the "
                "implemented Boolean validation objective. Every topic grid "
                "also contains the complete pooled shared grid, making each "
                "shared threshold literally feasible as a constant vector."
            ),
            "topic_vector_constraint": (
                "Choose one threshold per observed topic subject to the sum "
                "of its five validation exact-soundness failure counts not "
                "exceeding the pooled budget."
            ),
            "shared_baseline": (
                "Choose one scalar threshold for the same routed scores using "
                "the identical validation objective and failure budget."
            ),
            "tie_break": (
                "Maximize validation completeness, then use fewer validation "
                "failures. When added pooled candidates duplicate a local "
                "validation behavior, retain the canonical topic-local "
                "representative; remaining ties prefer lexicographically "
                "higher thresholds in fixed topic order. Test outcomes never "
                "break a tie."
            ),
            "evaluation": (
                "Every validation-selected shared threshold or five-threshold "
                "vector is frozen before exact held-out test evaluation. The "
                "resulting test path is ordered by validation budget and is "
                "not filtered or optimized using test labels."
            ),
            "families": topic_threshold_calibration_families,
        },
        "train_selected_singleton_generalization": {
            "selection_provenance": (
                "Reviewer identities are selected exclusively by correct-"
                "response choice accuracy on training prompts. Identities "
                "are then frozen before validation or test reporting."
            ),
            "selection_metric": (
                "Correct-response choice accuracy with response ties split "
                "uniformly."
            ),
            "tie_break": "Original reviewer index, ascending.",
            "epsilon_good": APPROXIMATE_GOOD,
            "threshold_policy": {
                "form": (
                    "The topic-conditioned policy uses one scalar threshold "
                    "shared across all five topics."
                ),
                "candidate_source": (
                    "Each policy has its own threshold grid: the distinct "
                    "scores of all four responses on its 1,058 pooled "
                    "training prompts, plus reject-all and accept-all."
                ),
                "evaluation": (
                    "For a given policy, exactly the same train-defined "
                    "threshold values are evaluated on training and held-out "
                    "test prompts. No frontier point selects a reviewer or "
                    "changes a threshold candidate."
                ),
                "fixed_checkpoint_thresholds": [0.45, 0.55],
                "fixed_checkpoint_provenance": (
                    "The two matched thresholds were predeclared and are "
                    "evaluated directly on both splits; they are not "
                    "selected from either frontier."
                ),
                "topic_specific_threshold_vector_used": False,
            },
            "global": {
                "reviewer_index": training_best_reviewer,
                "model": str(models[training_best_reviewer]),
                "training_choice_accuracy": float(
                    training_individual_accuracy[training_best_reviewer]
                ),
                "validation_choice_accuracy": choice_accuracy(
                    training_global_single_gains,
                    validation_prompts,
                ),
                "test_choice_accuracy": choice_accuracy(
                    training_global_single_gains,
                    test_prompts,
                ),
                "threshold_rows": int(len(training_global_single_rows)),
                "fixed_threshold_checkpoints": (
                    training_global_single_checkpoints
                ),
                "rows": training_global_single_rows,
            },
            "topic_conditioned": {
                "reviewers_by_topic": training_topic_single_metadata,
                "training_choice_accuracy": choice_accuracy(
                    training_topic_single_gains,
                    training_prompts,
                ),
                "validation_choice_accuracy": choice_accuracy(
                    training_topic_single_gains,
                    validation_prompts,
                ),
                "test_choice_accuracy": choice_accuracy(
                    training_topic_single_gains,
                    test_prompts,
                ),
                "threshold_rows": int(len(training_topic_single_rows)),
                "fixed_threshold_checkpoints": (
                    training_topic_single_checkpoints
                ),
                "rows": training_topic_single_rows,
            },
        },
        "topic_conditioned": {
            "topics_order": list(TOPICS),
            "routing_label_source": (
                "The prompt-aligned RewardBench 2 `subsets` field in the "
                "pinned score archive; the label is treated as observed "
                "before authorization."
            ),
            "selection_provenance": (
                "Within each topic, weights are fit only on that topic's "
                "training prompts; regularization, binary shrinkage, and "
                "single-reviewer identity use only that topic's validation "
                "prompts. Test prompts are used only for held-out reporting."
            ),
            "threshold_policy": {
                "form": (
                    "One scalar threshold per rule family is shared across "
                    "all five topics."
                ),
                "candidate_source": (
                    "The union of routed aggregate response scores on all "
                    "353 pooled validation prompts, plus reject-all and "
                    "accept-all endpoints."
                ),
                "ties_approve": True,
                "topic_specific_threshold_vector_used": False,
                "comparison_reason": (
                    "A shared threshold isolates topic-conditioned weights "
                    "or reviewer identity without adding topic-specific "
                    "calibration parameters."
                ),
            },
            "binary_shrinkage_selection": (
                "The existing learned-share grid and primary 99% validation "
                "pure-response soundness objective are applied separately "
                "inside each topic before the shared-threshold sweep."
            ),
            "topics": topic_metadata,
            "weighted_binary": {
                "validation_choice_accuracy": choice_accuracy(
                    topic_binary_gains,
                    validation_prompts,
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_binary_gains,
                    test_prompts,
                ),
                "shared_threshold_rows": int(len(topic_weighted_binary)),
                "rows": topic_weighted_binary,
                "operating_points": topic_weighted_binary_operating,
            },
            "learned_weight": {
                "validation_choice_accuracy": choice_accuracy(
                    topic_cardinal_gains,
                    validation_prompts,
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_cardinal_gains,
                    test_prompts,
                ),
                "shared_threshold_rows": int(len(topic_learned)),
                "rows": topic_learned,
                "operating_points": topic_learned_operating,
            },
            "best_single": {
                "validation_choice_accuracy": choice_accuracy(
                    topic_single_gains,
                    validation_prompts,
                ),
                "test_choice_accuracy": choice_accuracy(
                    topic_single_gains,
                    test_prompts,
                ),
                "shared_threshold_rows": int(len(topic_single)),
                "rows": topic_single,
                "operating_points": topic_single_operating,
            },
        },
        "count_rule": {
            "rows": count,
            "operating_points": count_operating,
        },
        "weighted_binary": {
            "selected_regularization": selected_binary_fit["regularization"],
            "learned_weight_share": selected_binary_shrinkage[
                "learned_share"
            ],
            "uniform_weight_share": (
                1.0 - selected_binary_shrinkage["learned_share"]
            ),
            "validation_choice_accuracy": selected_binary_shrinkage[
                "validation_choice_accuracy"
            ],
            "test_choice_accuracy": choice_accuracy(
                weighted_binary_gains, test_prompts
            ),
            "nonzero_weights": int(
                np.sum(weighted_binary_weights > 1e-8)
            ),
            "weights": {
                model: float(weight)
                for model, weight in zip(models, weighted_binary_weights)
                if weight > 1e-8
            },
            "precanonical_weights_by_reviewer": {
                str(model): float(weight)
                for model, weight in zip(
                    models,
                    selected_binary_shrinkage[
                        "fitted_mixture_weights"
                    ],
                )
            },
            "weight_units_by_reviewer": {
                str(model): int(unit)
                for model, unit in zip(
                    models,
                    weighted_binary_weight_units,
                )
            },
            "weight_canonicalization": (
                calibration_weight_canonicalization_metadata(
                    selected_binary_shrinkage[
                        "fitted_mixture_weights"
                    ],
                    weighted_binary_weight_units,
                    models,
                )
            ),
            "rows": weighted_binary,
            "operating_points": weighted_binary_operating,
        },
        "best_single": {
            "model": models[best_reviewer],
            "exact_alignment": best_single_alignment,
            "validation_choice_accuracy": float(
                validation_individual_accuracy[best_reviewer]
            ),
            "test_choice_accuracy": choice_accuracy(single_gains, test_prompts),
            "zero_threshold_endpoint_metrics": endpoint_metrics(
                single_gains >= -TOLERANCE, test_prompts
            ),
            "zero_threshold_full_domain": full_domain_metrics(
                single_gains, 0.0, test_prompts
            ),
            "rows": single,
            "operating_points": single_operating,
        },
        "ranked_single_reviewer": {
            "ranking_split": "validation",
            "ranking_metric": (
                "Correct-response choice accuracy with ties split uniformly."
            ),
            "tie_break": "Original reviewer index, ascending.",
            "rule": (
                "After removing the first k validation-ranked reviewers, use "
                "the native binary approval of reviewer rank k+1."
            ),
            "native_approval_threshold": 0.0,
            "approximate_tradeoff": ranked_single_tradeoff,
            "rows": ranked_single,
        },
        "ranked_prefix_unanimity": {
            "ranking_split": "validation",
            "ranking_metric": (
                "Correct-response choice accuracy with ties split uniformly."
            ),
            "tie_break": "Original reviewer index, ascending.",
            "rule": (
                "Use the first m validation-ranked reviewers and authorize "
                "only when all m give native binary approval."
            ),
            "native_approval_threshold": 0.0,
            "rows": ranked_prefix,
        },
        "equal_weight": {
            "test_choice_accuracy": choice_accuracy(equal_gains, test_prompts),
            "zero_threshold_full_domain": full_domain_metrics(
                equal_gains, 0.0, test_prompts
            ),
            "rows": equal,
            "operating_points": equal_operating,
        },
        "learned_weight": {
            "selected_regularization": selected_fit["regularization"],
            "validation_choice_accuracy": selected_fit["validation_accuracy"],
            "test_choice_accuracy": choice_accuracy(learned_gains, test_prompts),
            "nonzero_weights": int(np.sum(learned_weights > 1e-8)),
            "weights": {
                model: float(weight)
                for model, weight in zip(models, learned_weights)
                if weight > 1e-8
            },
            "zero_threshold_endpoint_metrics": endpoint_metrics(
                learned_gains >= -TOLERANCE, test_prompts
            ),
            "zero_threshold_full_domain": full_domain_metrics(
                learned_gains, 0.0, test_prompts
            ),
            "rows": learned,
            "operating_points": learned_operating,
        },
        "ablated_learned_weight": {
            "removed_model": str(models[best_reviewer]),
            "selected_regularization": selected_ablated_fit[
                "regularization"
            ],
            "validation_choice_accuracy": selected_ablated_fit[
                "validation_accuracy"
            ],
            "test_choice_accuracy": choice_accuracy(
                ablated_gains,
                test_prompts,
            ),
            "nonzero_weights": int(np.sum(ablated_weights > 1e-8)),
            "weights": {
                model: float(weight)
                for model, weight in zip(
                    models[retained_reviewers],
                    ablated_weights,
                )
                if weight > 1e-8
            },
            "zero_threshold_endpoint_metrics": endpoint_metrics(
                ablated_gains >= -TOLERANCE,
                test_prompts,
            ),
            "zero_threshold_full_domain": full_domain_metrics(
                ablated_gains,
                0.0,
                test_prompts,
            ),
            "rows": ablated_learned,
            "operating_points": ablated_learned_operating,
        },
        "prompt_specific_cardinal_reconstruction": (
            prompt_specific_reconstruction
        ),
    }

    topic_threshold_output = output["topic_threshold_budget_calibration"]
    cardinal_output = dict(output)
    del cardinal_output["topic_threshold_budget_calibration"]
    RESULTS.write_text(json.dumps(cardinal_output, indent=2) + "\n")
    TOPIC_THRESHOLD_RESULTS.write_text(
        json.dumps(topic_threshold_output, indent=2) + "\n"
    )
    make_figure(output)
    make_pure_response_figure(output)
    make_pure_response_detail_figure(output)
    make_margin_sensitivity_figure(output)
    make_ranked_single_approximate_figure(output)
    make_topic_conditioned_figure(output)
    make_topic_threshold_budget_figure(output)
    make_singleton_generalization_figure(output)
    print(
        "choice accuracy (test): "
        f"weighted_binary={output['weighted_binary']['test_choice_accuracy']:.3f}, "
        f"single={output['best_single']['test_choice_accuracy']:.3f}, "
        f"equal={output['equal_weight']['test_choice_accuracy']:.3f}, "
        f"learned={output['learned_weight']['test_choice_accuracy']:.3f}"
    )
    print(
        "learned zero-threshold endpoint metrics: "
        f"{output['learned_weight']['zero_threshold_endpoint_metrics']}"
    )
    print(
        "learned zero-threshold full-domain audit: "
        f"{output['learned_weight']['zero_threshold_full_domain']}"
    )
    for target in TARGET_SOUNDNESS:
        key = f"{target:.2f}"
        count_point = output["count_rule"]["operating_points"][key]
        binary_point = output["weighted_binary"]["operating_points"][key]
        learned_point = output["learned_weight"]["operating_points"][key]
        print(
            f"validation target {target:.0%}: "
            f"count={count_point if count_point else None}; "
            f"weighted_binary={binary_point if binary_point else None}; "
            f"learned={learned_point if learned_point else None}"
        )
    print(
        "prompt-specific cardinal reconstruction: "
        f"{output['prompt_specific_cardinal_reconstruction']}"
    )
    print(
        "validation-ranked native binary endpoints: "
        "single_top="
        f"{output['ranked_single_reviewer']['rows'][0]}; "
        "prefix_full="
        f"{output['ranked_prefix_unanimity']['rows'][-1]}"
    )
    print(RESULTS)
    print(TOPIC_THRESHOLD_RESULTS)
    print(FIGURE)
    print(FIGURE_PDF)
    print(PURE_FIGURE)
    print(PURE_FIGURE_PDF)
    print(PURE_DETAIL_FIGURE)
    print(PURE_DETAIL_FIGURE_PDF)
    print(MARGIN_FIGURE)
    print(MARGIN_FIGURE_PDF)
    print(RANKED_APPROXIMATE_FIGURE)
    print(RANKED_APPROXIMATE_FIGURE_PDF)
    print(TOPIC_CONDITIONED_FIGURE)
    print(TOPIC_CONDITIONED_FIGURE_PDF)
    print(TOPIC_THRESHOLD_FIGURE)
    print(TOPIC_THRESHOLD_FIGURE_PDF)
    print(SINGLETON_GENERALIZATION_FIGURE)
    print(SINGLETON_GENERALIZATION_FIGURE_PDF)


if __name__ == "__main__":
    main()
