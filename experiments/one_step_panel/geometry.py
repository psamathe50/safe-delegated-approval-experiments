"""Geometry shared by the RewardBench 2 experiment analyses."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog


def multicandidate_utilities(
    candidate_scores: np.ndarray,
    start: int,
    stop: int,
    n_correct: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent reviewer and principal utilities modulo additive constants."""
    scores = np.asarray(candidate_scores[:, start:stop], dtype=float)
    if not (0 < n_correct < scores.shape[1]):
        raise ValueError("a prompt needs correct and incorrect candidates")

    principal = np.concatenate(
        [np.ones(n_correct), np.zeros(scores.shape[1] - n_correct)]
    )
    reviewer_vectors = scores[:, :-1] - scores[:, -1, None]
    principal_vector = principal[:-1] - principal[-1]

    norms = np.linalg.norm(reviewer_vectors, axis=1)
    nonzero = norms > 1e-12
    reviewer_vectors[nonzero] /= norms[nonzero, None]
    principal_vector /= np.linalg.norm(principal_vector)
    return reviewer_vectors, principal_vector


def covers_multicandidate_utility(
    reviewer_vectors: np.ndarray,
    principal_vector: np.ndarray,
    panel: np.ndarray | None = None,
) -> bool:
    """Test whether a reviewer coalition conically covers the principal."""
    if panel is None:
        panel = np.arange(len(reviewer_vectors))
    panel = np.asarray(panel, dtype=int)
    if panel.size == 0:
        return False
    result = linprog(
        np.zeros(panel.size),
        A_eq=reviewer_vectors[panel].T,
        b_eq=principal_vector,
        bounds=[(0.0, None)] * panel.size,
        method="highs",
    )
    return result.status == 0


def standard_angular_evaluations(
    reviewer_vectors: np.ndarray,
    principal_vector: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Evaluate every face needed for exact strict depth in three dimensions.

    Normalizing a harmful direction by ``principal_vector @ v = 1`` leaves a
    two-dimensional affine arrangement. The returned columns include its
    vertices and unbounded cells, so minimizing the number of positive entries
    is an exact depth calculation rather than a sampled approximation.
    """
    if principal_vector.size != 3:
        raise ValueError("standard angular evaluations require dimension 3")
    if np.linalg.norm(principal_vector) <= tolerance:
        raise ValueError("principal direction must be nonzero")

    origin = principal_vector / (principal_vector @ principal_vector)
    basis = null_space(principal_vector[None, :])
    offsets = reviewer_vectors @ origin
    slopes = reviewer_vectors @ basis
    candidates = [np.zeros(2)]

    for first, second in itertools.combinations(range(len(reviewer_vectors)), 2):
        system = np.stack([slopes[first], slopes[second]])
        if abs(np.linalg.det(system)) > tolerance:
            candidates.append(
                np.linalg.solve(
                    system, -np.array([offsets[first], offsets[second]])
                )
            )

    # Cover degenerate arrangements with parallel lines.
    for index in range(len(reviewer_vectors)):
        denominator = slopes[index] @ slopes[index]
        if denominator > tolerance:
            candidates.append(-offsets[index] * slopes[index] / denominator)

    boundary_angles: list[float] = []
    for slope in slopes:
        if np.linalg.norm(slope) > tolerance:
            angle = (np.arctan2(slope[1], slope[0]) + np.pi / 2) % (2 * np.pi)
            boundary_angles.extend([angle, (angle + np.pi) % (2 * np.pi)])
    boundary_angles = sorted(set(round(angle, 12) for angle in boundary_angles))

    directions = [
        np.array([np.cos(angle), np.sin(angle)]) for angle in boundary_angles
    ]
    if boundary_angles:
        directions.extend(
            np.array(
                [
                    np.cos((first + second) / 2),
                    np.sin((first + second) / 2),
                ]
            )
            for first, second in zip(
                boundary_angles,
                boundary_angles[1:] + [boundary_angles[0] + 2 * np.pi],
            )
        )

    for direction in directions:
        directional_slopes = slopes @ direction
        changing = np.abs(directional_slopes) > tolerance
        crossings = np.abs(offsets[changing] / directional_slopes[changing])
        radius = (crossings.max() if crossings.size else 0.0) + 2.0
        candidates.append(radius * direction)

    candidate_matrix = np.stack(candidates)
    return offsets[:, None] + slopes @ candidate_matrix.T


def standard_angular_cell_evaluations(
    reviewer_vectors: np.ndarray,
    principal_vector: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Evaluate one interior point of every angular arrangement cell.

    ``standard_angular_evaluations`` includes the lower-dimensional faces
    needed to minimize a *strict* objection count: landing on a reviewer
    boundary turns that reviewer into an approver.  Full-domain completeness
    instead maximizes the number of strict objections to a beneficial
    proposal.  That maximum is attained in a full-dimensional cell, not
    necessarily at one of its vertices.  This routine samples both sides of
    every open line segment or ray in the two-dimensional affine arrangement,
    which visits every full-dimensional cell exactly enough for the maximum.
    """
    if principal_vector.size != 3:
        raise ValueError("standard angular cells require dimension 3")
    if np.linalg.norm(principal_vector) <= tolerance:
        raise ValueError("principal direction must be nonzero")

    origin = principal_vector / (principal_vector @ principal_vector)
    basis = null_space(principal_vector[None, :])
    offsets = reviewer_vectors @ origin
    slopes = reviewer_vectors @ basis
    active = np.flatnonzero(np.linalg.norm(slopes, axis=1) > tolerance)
    if active.size == 0:
        return offsets[:, None]

    candidates: list[np.ndarray] = []
    for index in active:
        slope = slopes[index]
        slope_norm = np.linalg.norm(slope)
        normal = slope / slope_norm
        direction = np.array([-normal[1], normal[0]])
        point_on_line = -offsets[index] * slope / (slope @ slope)

        crossings: list[float] = []
        for other in active:
            if other == index:
                continue
            system = np.stack([slope, slopes[other]])
            if abs(np.linalg.det(system)) <= tolerance:
                continue
            intersection = np.linalg.solve(
                system,
                -np.array([offsets[index], offsets[other]]),
            )
            crossings.append(float(direction @ (intersection - point_on_line)))

        if crossings:
            values = np.unique(np.round(crossings, 11))
            span = max(1.0, float(values[-1] - values[0]))
            interval_points = [float(values[0] - span - 1.0)]
            interval_points.extend(
                float((left + right) / 2)
                for left, right in zip(values[:-1], values[1:])
                if right - left > tolerance
            )
            interval_points.append(float(values[-1] + span + 1.0))
        else:
            interval_points = [0.0]

        for coordinate in interval_points:
            point = point_on_line + coordinate * direction
            evaluations = offsets + slopes @ point
            directional_change = slopes @ normal
            bounds = []
            for other in range(len(reviewer_vectors)):
                if abs(evaluations[other]) <= 10 * tolerance:
                    continue
                change = abs(directional_change[other])
                if change > tolerance:
                    bounds.append(abs(evaluations[other]) / change)
            step = 1.0 if not bounds else min(1.0, 0.25 * min(bounds))
            candidates.extend([point - step * normal, point + step * normal])

    candidate_matrix = np.stack(candidates)
    evaluations = offsets[:, None] + slopes @ candidate_matrix.T
    signs = np.where(
        evaluations > tolerance,
        1,
        np.where(evaluations < -tolerance, -1, 0),
    )
    _, unique_columns = np.unique(signs.T, axis=0, return_index=True)
    return evaluations[:, np.sort(unique_columns)]


def stratified_prompt_split(
    subsets: np.ndarray,
    seed: int = 20260813,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> np.ndarray:
    """Assign prompts to train, validation, and test within each category."""
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    rng = np.random.default_rng(seed)
    assignment = np.empty(len(subsets), dtype="U10")
    for subset in np.unique(subsets):
        indices = rng.permutation(np.flatnonzero(subsets == subset))
        n_train = int(round(train_fraction * len(indices)))
        n_validation = int(round(validation_fraction * len(indices)))
        assignment[indices[:n_train]] = "train"
        assignment[indices[n_train : n_train + n_validation]] = "validation"
        assignment[indices[n_train + n_validation :]] = "test"
    return assignment
