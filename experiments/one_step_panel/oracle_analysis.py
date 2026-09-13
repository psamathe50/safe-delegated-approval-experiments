"""Reproduce the prompt-specific exact-sound rule comparison in the paper.

These are oracle audits, not learned deployment rules.  On each benchmark
instance the principal utility is known from the benchmark label.  We compare
unanimity, the most permissive count threshold certified by rejection depth,
and the maximal sound monotone rule from the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from geometry import covers_multicandidate_utility, multicandidate_utilities


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = DATA / "oracle_results.json"
TOLERANCE = 1e-8


def worst_principal_gain(
    scores: np.ndarray,
    approvers: np.ndarray,
) -> float:
    """Minimize RewardBench principal gain over a coalition's approval cone."""
    n_candidates = scores.shape[1]
    baseline = np.full(n_candidates, 1.0 / n_candidates)
    result = linprog(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        A_ub=-scores[approvers] if approvers.size else None,
        b_ub=-(scores[approvers] @ baseline) if approvers.size else None,
        A_eq=np.ones((1, n_candidates)),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * n_candidates,
        method="highs",
    )
    if result.status != 0 or result.fun is None:
        raise RuntimeError(f"worst-case approval LP failed: {result.message}")
    return float(result.fun - 0.25)


def summarize_rewardbench(rows: list[dict[str, object]]) -> dict[str, object]:
    covered = [row for row in rows if row["full_panel_covers"]]
    output: dict[str, object] = {
        "prompts": len(rows),
        "covered_prompts": len(covered),
        "rules": {},
    }
    for key in ("unanimity", "largest_certified_count", "maximal_monotone"):
        authorized = int(sum(bool(row[key]) for row in covered))
        output["rules"][key] = {
            "authorized_covered": authorized,
            "rate_among_covered": authorized / len(covered),
        }
    return output


def rewardbench_results() -> dict[str, object]:
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        prompt_keys = data["prompt_keys"].astype(str)
        subsets = data["subsets"].astype(str)
        dataset_revision = str(data["dataset_revision"])
        results_revision = str(data["results_revision"])
    with np.load(DATA / "rewardbench2_depth.npz", allow_pickle=True) as data:
        depth = data["depth"].astype(int)
        depth_exact = data["depth_exact"].astype(bool)

    standard = np.flatnonzero(candidate_counts == 4)
    if len(standard) != 1763 or not np.all(num_correct[standard] == 1):
        raise RuntimeError("unexpected RewardBench 2 standard-prompt cohort")
    if not np.all(depth_exact[standard]):
        raise RuntimeError("oracle audit requires exact stored depths")

    rows: list[dict[str, object]] = []
    for prompt in standard:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        prompt_scores = scores[:, start:stop]
        baseline_scores = np.mean(prompt_scores, axis=1)
        correct_gains = prompt_scores[:, 0] - baseline_scores
        approvers = np.flatnonzero(correct_gains >= 0.0)
        objectors = int(len(prompt_scores) - len(approvers))
        reviewers, principal = multicandidate_utilities(
            scores, start, stop, 1
        )
        full_panel_covers = bool(depth[prompt] >= 1)
        if covers_multicandidate_utility(reviewers, principal) != full_panel_covers:
            raise RuntimeError(f"coverage/depth mismatch on prompt {prompt}")

        maximal = covers_multicandidate_utility(
            reviewers,
            principal,
            panel=approvers,
        )
        worst_gain = worst_principal_gain(prompt_scores, approvers)
        if maximal != (worst_gain >= -TOLERANCE):
            raise RuntimeError(
                f"cone and worst-case LP disagree on prompt {prompt}"
            )
        largest_count = bool(
            full_panel_covers and objectors <= int(depth[prompt]) - 1
        )
        unanimity = objectors == 0
        if (unanimity or largest_count) and full_panel_covers and not maximal:
            raise RuntimeError("certified count rule exceeds maximal sound rule")

        rows.append(
            {
                "prompt": int(prompt),
                "prompt_key": str(prompt_keys[prompt]),
                "category": str(subsets[prompt]),
                "depth": int(depth[prompt]),
                "correct_response_objectors": objectors,
                "full_panel_covers": full_panel_covers,
                "unanimity": unanimity,
                "largest_certified_count": largest_count,
                "maximal_monotone": bool(maximal),
                "approval_set_worst_principal_gain": worst_gain,
            }
        )

    summary = summarize_rewardbench(rows)
    counts = {
        key: summary["rules"][key]["authorized_covered"]
        for key in summary["rules"]
    }
    if summary["covered_prompts"] != 1590 or counts != {
        "unanimity": 55,
        "largest_certified_count": 581,
        "maximal_monotone": 1409,
    }:
        raise RuntimeError("RewardBench 2 oracle-rule results changed")
    return {
        "dataset_revision": dataset_revision,
        "results_revision": results_revision,
        "construction": (
            "Uniform fallback and labeled-correct pure proposal; rates are "
            "reported among prompts covered by the full panel."
        ),
        "summary": summary,
        "rows": rows,
    }


def strongreject_results() -> dict[str, object]:
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=True) as data:
        complete = data["label_count"] == 5
        human = data["human"][complete].astype(float)
        scores = data["evaluator_scores"][complete].astype(float)

    acceptable = human <= 0.5 + TOLERANCE
    harmful = human > 0.5 + TOLERANCE
    objections = np.sum(scores > 0.5 + TOLERANCE, axis=1)
    n_reviewers = scores.shape[1]
    depth = np.where(acceptable, n_reviewers + 1, objections)
    largest_k = np.minimum(depth - 1, n_reviewers - 1)
    unanimity = objections == 0
    largest_count = objections <= largest_k
    # In a release-versus-abstain decision, every approval set is safe when
    # release is acceptable, and none is safe when release is harmful.
    maximal = acceptable.copy()

    if (
        len(human) != 1084
        or int(np.sum(acceptable)) != 894
        or int(np.sum(harmful)) != 190
        or int(np.min(objections[harmful])) != 2
    ):
        raise RuntimeError("StrongREJECT oracle cohort changed")
    counts = {
        "unanimity": int(np.sum(unanimity & acceptable)),
        "largest_certified_count": int(np.sum(largest_count & acceptable)),
        "maximal_monotone": int(np.sum(maximal & acceptable)),
    }
    if counts != {
        "unanimity": 267,
        "largest_certified_count": 893,
        "maximal_monotone": 894,
    }:
        raise RuntimeError("StrongREJECT oracle-rule results changed")
    return {
        "responses": int(len(human)),
        "acceptable_responses": int(np.sum(acceptable)),
        "harmful_responses": int(np.sum(harmful)),
        "rules": {
            key: {
                "authorized_acceptable": value,
                "rate_among_acceptable": value / int(np.sum(acceptable)),
            }
            for key, value in counts.items()
        },
    }


def main() -> None:
    output = {
        "scope": (
            "Prompt-specific oracle comparison. The benchmark principal is "
            "known separately on every instance; these are not deployable "
            "rules learned without labels."
        ),
        "strongreject": strongreject_results(),
        "rewardbench2": rewardbench_results(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(
        "Oracle rules (StrongREJECT): "
        + ", ".join(
            f"{key}={value['rate_among_acceptable']:.1%}"
            for key, value in output["strongreject"]["rules"].items()
        ),
        flush=True,
    )
    print(
        "Oracle rules (RewardBench 2): "
        + ", ".join(
            f"{key}={value['rate_among_covered']:.1%}"
            for key, value in output["rewardbench2"]["summary"]["rules"].items()
        ),
        flush=True,
    )
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
