"""Compute exact RewardBench 2 rejection depth from reviewer scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry import (
    multicandidate_utilities,
    standard_angular_cell_evaluations,
    standard_angular_evaluations,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TOLERANCE = 1e-8


def main() -> None:
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        prompt_keys = data["prompt_keys"].astype(str)
        subsets = data["subsets"].astype(str)
        dataset_revision = str(data["dataset_revision"])
        results_revision = str(data["results_revision"])

    standard = np.flatnonzero(candidate_counts == 4)
    if len(standard) != 1763 or not np.all(num_correct[standard] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")

    depth = np.full(len(candidate_counts), -1, dtype=np.int16)
    maximum_beneficial_objections = np.full(
        len(candidate_counts), -1, dtype=np.int16
    )
    depth_exact = np.zeros(len(candidate_counts), dtype=bool)
    for prompt in standard:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        reviewers, principal = multicandidate_utilities(
            scores, start, stop, int(num_correct[prompt])
        )
        evaluations = standard_angular_evaluations(
            reviewers, principal, tolerance=TOLERANCE
        )
        cell_evaluations = standard_angular_cell_evaluations(
            reviewers, principal, tolerance=TOLERANCE
        )
        depth[prompt] = int(
            np.min(np.sum(evaluations > TOLERANCE, axis=0))
        )
        maximum_beneficial_objections[prompt] = int(
            np.max(np.sum(cell_evaluations < -TOLERANCE, axis=0))
        )
        depth_exact[prompt] = True

    standard_depth = depth[standard]
    if (
        int(np.sum(standard_depth >= 1)) != 1590
        or int(np.sum(standard_depth >= 2)) != 1361
        or int(np.sum(standard_depth >= 6)) != 669
    ):
        raise RuntimeError("RewardBench 2 depth audit changed unexpectedly")

    np.savez_compressed(
        DATA / "rewardbench2_depth.npz",
        prompt_keys=prompt_keys,
        subsets=subsets,
        candidate_counts=candidate_counts,
        depth=depth,
        maximum_beneficial_objections=maximum_beneficial_objections,
        depth_exact=depth_exact,
        dataset_revision=dataset_revision,
        results_revision=results_revision,
    )
    print(
        "Exact depth: "
        f"{int(np.sum(standard_depth >= 1))}/{len(standard)} prompts covered; "
        f"maximum depth {int(standard_depth.max())}"
    )


if __name__ == "__main__":
    main()
