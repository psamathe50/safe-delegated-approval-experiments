"""Individual alignment and panel depth on RewardBench 2 and StrongREJECT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = DATA / "alignment_results.json"
ANGLE_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
NORM_TOLERANCE = 1e-12


def summarize_angles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_degrees": float(np.mean(values)),
        "median_degrees": float(np.median(values)),
        "p10_degrees": float(np.quantile(values, 0.10)),
        "p90_degrees": float(np.quantile(values, 0.90)),
        "maximum_degrees": float(np.max(values)),
    }


def main() -> None:
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"]
        candidate_counts = data["candidate_counts"]
        num_correct = data["num_correct"]
        subsets = data["subsets"].astype(str)
        prompt_keys = data["prompt_keys"].astype(str)
        dataset_revision = str(data["dataset_revision"])
        results_revision = str(data["results_revision"])

    with np.load(
        DATA / "rewardbench2_depth.npz", allow_pickle=True
    ) as data:
        depth = data["depth"].astype(int)
        depth_candidate_counts = data["candidate_counts"]

    if not np.array_equal(candidate_counts, depth_candidate_counts):
        raise RuntimeError("score and depth archives use different prompt orders")

    standard = np.flatnonzero(candidate_counts == 4)
    if len(standard) != 1763 or not np.all(num_correct[standard] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")

    principal = np.array([1.0, 0.0, 0.0, 0.0])
    centered_principal = principal - np.mean(principal)
    centered_principal /= np.linalg.norm(centered_principal)

    best_angle = []
    best_reviewer = []
    for prompt in standard:
        prompt_scores = scores[:, offsets[prompt] : offsets[prompt + 1]]
        centered_scores = prompt_scores - np.mean(
            prompt_scores, axis=1, keepdims=True
        )
        norms = np.linalg.norm(centered_scores, axis=1)
        if np.any(norms <= NORM_TOLERANCE):
            raise RuntimeError(f"constant reviewer score vector on prompt {prompt}")
        cosine = centered_scores @ centered_principal / norms
        reviewer = int(np.argmax(cosine))
        angle = float(
            np.degrees(np.arccos(np.clip(cosine[reviewer], -1.0, 1.0)))
        )
        best_angle.append(angle)
        best_reviewer.append(reviewer)

    best_angle = np.asarray(best_angle)
    best_reviewer = np.asarray(best_reviewer)
    standard_depth = depth[standard]
    standard_subsets = subsets[standard]
    covered = standard_depth >= 1

    threshold_sensitivity = {}
    for threshold in ANGLE_THRESHOLDS:
        separated = covered & (best_angle > threshold)
        threshold_sensitivity[str(int(threshold))] = {
            "separated_covered_count": int(np.sum(separated)),
            "separated_rate_among_covered": float(
                np.sum(separated) / np.sum(covered)
            ),
            "depth_at_least_2_count": int(
                np.sum(separated & (standard_depth >= 2))
            ),
            "depth_at_least_6_count": int(
                np.sum(separated & (standard_depth >= 6))
            ),
        }

    categories = {}
    for category in sorted(set(standard_subsets)):
        selected = standard_subsets == category
        categories[category] = {
            "n_prompts": int(np.sum(selected)),
            "coverage_rate": float(np.mean(standard_depth[selected] >= 1)),
            "best_angle": summarize_angles(best_angle[selected]),
            "depth_mean": float(np.mean(standard_depth[selected])),
            "depth_median": float(np.median(standard_depth[selected])),
        }

    with np.load(
        DATA / "strongreject_scores.npz", allow_pickle=True
    ) as data:
        complete = data["label_count"] == 5
        human_harm = data["human"][complete].astype(float)
        evaluator_harm = data["evaluator_scores"][complete].astype(float)

    principal_gain = 0.5 - human_harm
    reviewer_gain = 0.5 - evaluator_harm
    strictly_acceptable = principal_gain > NORM_TOLERANCE
    principal_indifferent = np.abs(principal_gain) <= NORM_TOLERANCE
    harmful = principal_gain < -NORM_TOLERANCE
    acceptable_aligned = np.sum(
        reviewer_gain[strictly_acceptable] > NORM_TOLERANCE, axis=1
    )
    harmful_aligned = np.sum(
        reviewer_gain[harmful] < -NORM_TOLERANCE, axis=1
    )
    strongreject = {
        "construction": (
            "The release-versus-abstain feasible span is one-dimensional. "
            "For a nonzero principal gain, a reviewer is individually aligned "
            "exactly when its gain has the same sign."
        ),
        "n_responses": int(len(principal_gain)),
        "strictly_acceptable_count": int(np.sum(strictly_acceptable)),
        "principal_indifferent_count": int(np.sum(principal_indifferent)),
        "harmful_count": int(np.sum(harmful)),
        "strictly_acceptable_aligned_reviewers": {
            "minimum": int(np.min(acceptable_aligned)),
            "median": float(np.median(acceptable_aligned)),
            "maximum": int(np.max(acceptable_aligned)),
        },
        "harmful_aligned_reviewers": {
            "minimum": int(np.min(harmful_aligned)),
            "median": float(np.median(harmful_aligned)),
            "maximum": int(np.max(harmful_aligned)),
        },
    }

    output = {
        "dataset_revision": dataset_revision,
        "results_revision": results_revision,
        "construction": (
            "Euclidean angle between the centered principal vector and the "
            "nearest centered reviewer score vector; additive shifts and "
            "positive rescaling do not change the angle."
        ),
        "n_standard_prompts": int(len(standard)),
        "n_covered_prompts": int(np.sum(covered)),
        "all_standard": summarize_angles(best_angle),
        "covered": summarize_angles(best_angle[covered]),
        "threshold_sensitivity": threshold_sensitivity,
        "categories": categories,
        "strongreject": strongreject,
        "rows": [
            {
                "prompt": int(prompt),
                "prompt_key": str(prompt_keys[prompt]),
                "category": str(subsets[prompt]),
                "depth": int(depth[prompt]),
                "best_reviewer_index": int(best_reviewer[index]),
                "best_angle_degrees": float(best_angle[index]),
            }
            for index, prompt in enumerate(standard)
        ],
    }

    if output["n_covered_prompts"] != 1590:
        raise RuntimeError("covered-prompt count changed")
    if not np.isclose(output["covered"]["median_degrees"], 6.844416, atol=1e-6):
        raise RuntimeError("covered-prompt median angle changed")
    check = threshold_sensitivity["10"]
    if (
        check["separated_covered_count"] != 575
        or check["depth_at_least_2_count"] != 423
        or check["depth_at_least_6_count"] != 125
    ):
        raise RuntimeError("10-degree sensitivity counts changed")
    if (
        strongreject["n_responses"] != 1084
        or strongreject["strictly_acceptable_count"] != 848
        or strongreject["principal_indifferent_count"] != 46
        or strongreject["harmful_count"] != 190
        or strongreject["strictly_acceptable_aligned_reviewers"]["minimum"] != 1
        or strongreject["strictly_acceptable_aligned_reviewers"]["median"] != 7.0
        or strongreject["harmful_aligned_reviewers"]["minimum"] != 2
        or strongreject["harmful_aligned_reviewers"]["median"] != 7.0
    ):
        raise RuntimeError("StrongREJECT directional-alignment counts changed")

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT)
    print(
        "covered median best angle: "
        f"{output['covered']['median_degrees']:.2f} degrees"
    )
    print(
        "best angle > 10 degrees: 575 covered prompts; "
        "423 have depth >= 2 and 125 have depth >= 6"
    )
    print(
        "StrongREJECT aligned reviewers: harmful minimum 2, median 7; "
        "strictly acceptable minimum 1, median 7"
    )


if __name__ == "__main__":
    main()
