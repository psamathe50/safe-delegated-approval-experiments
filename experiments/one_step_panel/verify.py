"""Fast post-run audit for the one-step experiment reproduction bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
TOLERANCE = 1e-8
CALIBRATION_WEIGHT_DECIMALS = 8
CALIBRATION_WEIGHT_UNIT_SHA256 = {
    "global": "4f6871b26dbf6a01a743ffea30a2cca942a47a33676ef2419040e99df2de72eb",
    "Factuality": "42efb235cf33da5f3775b70b0ced72a01d5657b09a435fa501f30ed2b3a1eaed",
    "Focus": "3c4e6844bf3c13f8adde361cbdbb4032e11e86abde0e9cdcca75112bdbfb896a",
    "Math": "9925312546b375c4e6272150975c46274c0512ca0e9532b45412cae5dcbce370",
    "Precise IF": "7a9f9df0aae8a73a4692bc830c5386926ca1cb6839e611e2212461f9f7baaee4",
    "Safety": "d534c3904b5f3b30d852e24cea8ecc848ea3e175c0b4e32ba722fb89f89520f6",
}
BINARY_WEIGHT_UNIT_SHA256 = {
    "global": "dd9d87e2ce6a72bb0a0af0c2d6c8fa79c3efd96919df24a45949ab8772d7580e",
    "Factuality": "a70c8311fc868b3c55986691c3f459fc9b53c597b76bc95b4c35e86f26278399",
    "Focus": "fb8dd249bf4cab8920ffe2adbc4af1935f92ca5ff864b862fc8830d3e489cd62",
    "Math": "4ccf33171d9849a1454a130fd5d90ee3db6d0d948addb5cab4c07a994f2f2bd8",
    "Precise IF": "fb5ce955726a040f0616bad2781b2cc10bf7bd5a562093d3bfbe8c1c978e0e85",
    "Safety": "34f6b3e5c24b7ae5be777991bbd30fe7d81c3c798dcf64ca52b76cb8f4880436",
}
INPUT_SHA256 = {
    "rewardbench2_scores.npz": (
        "2b6f6cfa2c6cace3d85e1bc96b5a8c61390b3b91a2bb73f67bd6e89849ed0d7c"
    ),
    "strongreject_scores.npz": (
        "e462848048763dc24640535baa2ce2bf9ed9834535faf323c2b73b8ea5396d48"
    ),
}
REQUIRED_RESULTS = (
    "rewardbench2_depth.npz",
    "approximate_results.npz",
    "approximate_results.json",
    "alignment_results.json",
    "oracle_results.json",
    "cardinal_results.json",
    "topic_threshold_budget_results.json",
)
REQUIRED_FIGURES = (
    "binary_overall.pdf",
    "approximate_implementation.pdf",
    "rewardbench_categories.pdf",
    "rewardbench_completeness_categories.pdf",
    "cardinal_frontier.pdf",
    "cardinal_pure_response.pdf",
    "cardinal_pure_response_full_and_zoom.pdf",
    "cardinal_margin_sensitivity.pdf",
    "ranked_singleton_approximate.pdf",
    "rewardbench_topic_distributions.pdf",
    "singleton_train_test_generalization.pdf",
    "topic_conditioned_comparison.pdf",
    "topic_threshold_budget.pdf",
)
REQUIRED_PNGS = (
    "cardinal_pure_response.png",
    "cardinal_pure_response_full_and_zoom.png",
    "ranked_singleton_approximate.png",
    "rewardbench_topic_distributions.png",
    "singleton_train_test_generalization.png",
    "topic_conditioned_comparison.png",
    "topic_threshold_budget.png",
)
TOPIC_SPLITS = {
    "Factuality": (285, 95, 95),
    "Focus": (297, 99, 99),
    "Math": (110, 37, 36),
    "Precise IF": (96, 32, 32),
    "Safety": (270, 90, 90),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verified_binary_weight_units(
    record: dict[str, object],
    reviewer_names: list[str],
    expected_hash: str,
    label: str,
) -> np.ndarray:
    """Rebuild and verify one canonical weighted-binary score vector."""
    raw_by_reviewer = record["precanonical_weights_by_reviewer"]
    units_by_reviewer = record["weight_units_by_reviewer"]
    canonical_by_reviewer = record["weights"]
    require(
        list(raw_by_reviewer) == reviewer_names
        and list(units_by_reviewer) == reviewer_names,
        f"{label} weighted-binary reviewer order changed",
    )
    raw = np.asarray(
        [raw_by_reviewer[name] for name in reviewer_names],
        dtype=float,
    )
    scale = 10**CALIBRATION_WEIGHT_DECIMALS
    expected_units = np.rint(np.maximum(raw, 0.0) * scale).astype(
        np.int64
    )
    observed_units = np.asarray(
        [units_by_reviewer[name] for name in reviewer_names],
        dtype=np.int64,
    )
    total_units = int(np.sum(expected_units))
    require(total_units > 0, f"{label} weighted-binary units vanished")
    canonical = expected_units.astype(float) / total_units
    observed = np.asarray(
        [canonical_by_reviewer.get(name, 0.0) for name in reviewer_names],
        dtype=float,
    )
    unit_hash = hashlib.sha256(
        np.asarray(expected_units, dtype="<i8").tobytes()
    ).hexdigest()
    scaled = np.maximum(raw, 0.0) * scale
    delta = canonical - raw
    expected_metadata = {
        "decimal_places": CALIBRATION_WEIGHT_DECIMALS,
        "integer_unit_rounding": "nearest, ties to even",
        "integer_unit_total": total_units,
        "integer_units_sha256": unit_hash,
        "reviewer_order_sha256": hashlib.sha256(
            "\0".join(reviewer_names).encode("utf-8")
        ).hexdigest(),
        "renormalized_to_sum_one": True,
        "aggregation": (
            "math.fsum of feature times integer unit in fixed reviewer "
            "archive order, divided once by the integer-unit total"
        ),
        "maximum_absolute_weight_change": float(
            np.max(np.abs(delta))
        ),
        "l1_weight_change": float(np.sum(np.abs(delta))),
        "minimum_distance_to_half_unit_boundary": float(
            np.min(np.abs(scaled - (np.floor(scaled) + 0.5)))
        ),
    }
    require(
        np.array_equal(observed_units, expected_units)
        and np.array_equal(observed, canonical)
        and unit_hash == expected_hash
        and record["weight_canonicalization"] == expected_metadata,
        f"{label} weighted-binary canonicalization changed",
    )
    return observed_units


def reproducible_weighted_gains(
    features: np.ndarray,
    weight_units: np.ndarray,
) -> np.ndarray:
    """Reconstruct fixed-order integer-unit aggregation used by analysis."""
    weight_units = np.asarray(weight_units, dtype=np.int64)
    require(
        features.ndim >= 2 and features.shape[-1] == len(weight_units),
        "weighted-gain feature and unit shapes disagree",
    )
    total_units = int(np.sum(weight_units))
    require(total_units > 0, "weighted-gain units must be positive")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def epsilon_complete(
    error: np.ndarray,
    attained: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    return (error < epsilon - 5 * TOLERANCE) | (
        np.isclose(error, epsilon, atol=5 * TOLERANCE, rtol=0.0)
        & ~attained
    )


def check_inputs() -> None:
    for name, expected in INPUT_SHA256.items():
        path = DATA / name
        require(path.is_file(), f"missing pinned input: {path}")
        require(sha256(path) == expected, f"input checksum mismatch: {name}")

    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        standard = data["candidate_counts"] == 4
        require(int(np.sum(standard)) == 1763, "RewardBench cohort changed")
        require(
            np.all(data["num_correct"][standard] == 1),
            "RewardBench standard prompt lacks a unique correct response",
        )
        require(data["candidate_scores"].shape[0] == 48, "reviewer count changed")
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=True) as data:
        complete = data["label_count"] == 5
        require(int(np.sum(complete)) == 1084, "StrongREJECT cohort changed")
        require(data["evaluator_scores"].shape[1] == 9, "evaluator count changed")


def check_exact_results() -> None:
    for name in REQUIRED_RESULTS:
        require((DATA / name).is_file(), f"missing derived result: {name}")

    with np.load(DATA / "rewardbench2_depth.npz", allow_pickle=True) as data:
        standard = data["candidate_counts"] == 4
        depth = data["depth"][standard].astype(int)
        beneficial = data["maximum_beneficial_objections"][standard].astype(int)
        require(np.all(data["depth_exact"][standard]), "stored depth is not exact")
    require(len(depth) == 1763, "depth archive has wrong cohort")
    require(int(np.sum(depth >= 1)) == 1590, "coverage count changed")
    require(int(np.sum(depth >= 2)) == 1361, "depth >= 2 count changed")
    require(int(np.sum(depth >= 6)) == 669, "depth >= 6 count changed")
    require(int(np.max(depth)) == 23, "maximum depth changed")
    require(int(np.min(beneficial)) >= 0, "invalid beneficial-objection maximum")

    with np.load(DATA / "approximate_results.npz", allow_pickle=True) as data:
        reward_sound = data["rewardbench_soundness_error"].astype(float)
        reward_complete = data["rewardbench_completeness_error"].astype(float)
        reward_attained = data[
            "rewardbench_completeness_supremum_attained"
        ].astype(bool)
        strong_sound = data["strongreject_soundness_error"].astype(float)
        strong_complete = data["strongreject_completeness_error"].astype(float)
        strong_attained = data[
            "strongreject_completeness_supremum_attained"
        ].astype(bool)
        strong_objections = data["strongreject_objections"].astype(int)
    with np.load(DATA / "strongreject_scores.npz", allow_pickle=True) as data:
        complete = data["label_count"] == 5
        strong_harmful = data["human"][complete].astype(float) > 0.5 + TOLERANCE
    require(reward_sound.shape == (1763, 48), "RewardBench error shape changed")
    require(strong_sound.shape == (1084, 9), "StrongREJECT error shape changed")
    require(
        int(np.min(strong_objections[strong_harmful])) >= 2,
        "a harmful StrongREJECT response has fewer than two objections",
    )
    strong_joint = np.mean(
        (strong_sound <= 5 * TOLERANCE)
        & epsilon_complete(strong_complete, strong_attained, 0.0),
        axis=0,
    )
    reward_joint = np.mean(
        (reward_sound <= 5 * TOLERANCE)
        & epsilon_complete(reward_complete, reward_attained, 0.0),
        axis=0,
    )
    require(int(np.argmax(strong_joint)) == 6, "StrongREJECT best exact k changed")
    require(
        np.isclose(np.max(strong_joint), 0.9345018450184502),
        "StrongREJECT exact joint rate changed",
    )
    require(np.max(reward_joint) == 0.0, "RewardBench exact joint rate changed")


def check_alignment_and_oracles() -> None:
    alignment = json.loads((DATA / "alignment_results.json").read_text())
    require(alignment["n_covered_prompts"] == 1590, "alignment coverage changed")
    require(
        np.isclose(alignment["covered"]["median_degrees"], 6.844416, atol=1e-6),
        "median closest-reviewer angle changed",
    )
    ten = alignment["threshold_sensitivity"]["10"]
    require(
        (
            ten["separated_covered_count"],
            ten["depth_at_least_2_count"],
            ten["depth_at_least_6_count"],
        )
        == (575, 423, 125),
        "10-degree depth counts changed",
    )

    oracle = json.loads((DATA / "oracle_results.json").read_text())
    strong_rules = oracle["strongreject"]["rules"]
    reward_rules = oracle["rewardbench2"]["summary"]["rules"]
    require(
        [strong_rules[key]["authorized_acceptable"] for key in strong_rules]
        == [267, 893, 894],
        "StrongREJECT oracle table changed",
    )
    require(
        [reward_rules[key]["authorized_covered"] for key in reward_rules]
        == [55, 581, 1409],
        "RewardBench oracle table changed",
    )


def check_topic_conditioned_results(
    results: dict[str, object],
    reviewer_names: list[str],
    binary_features: np.ndarray,
    subsets: np.ndarray,
    validation_prompts: np.ndarray,
) -> None:
    topic_output = results["topic_conditioned"]
    topics = list(TOPIC_SPLITS)
    require(
        topic_output["topics_order"] == topics,
        "topic-conditioned routing order changed",
    )
    routing = topic_output["routing_label_source"].lower()
    provenance = topic_output["selection_provenance"].lower()
    require(
        "subsets" in routing and "observed before authorization" in routing,
        "topic-routing observability is not documented",
    )
    require(
        "training prompts" in provenance
        and "validation prompts" in provenance
        and "test prompts are used only for held-out reporting" in provenance,
        "topic-conditioned split provenance is incomplete",
    )
    policy = topic_output["threshold_policy"]
    require(
        policy["ties_approve"]
        and not policy["topic_specific_threshold_vector_used"],
        "topic-conditioned threshold semantics changed",
    )
    require(
        "one scalar threshold" in policy["form"].lower()
        and "all five topics" in policy["form"].lower()
        and "353 pooled validation prompts" in policy["candidate_source"],
        "shared validation-threshold provenance changed",
    )
    require(
        "99% validation" in topic_output["binary_shrinkage_selection"],
        "topic-local binary shrinkage selection is undocumented",
    )

    expected_fits = {
        "Factuality": {
            "cardinal": (0.01, 0.8947368421052632, 0.8631578947368421, 13),
            "binary": (
                0.1,
                0.99,
                0.8368421052631579,
                0.8315789473684211,
                0.7894736842105263,
                0.0,
            ),
            "single": (
                26,
                "allenai/Llama-3.1-70B-Instruct-RM-RB2",
                0.8526315789473684,
                0.8315789473684211,
            ),
        },
        "Focus": {
            "cardinal": (0.0001, 0.98989898989899, 0.98989898989899, 23),
            "binary": (
                0.0001,
                0.99,
                0.9393939393939394,
                0.9393939393939394,
                0.9595959595959596,
                1 / 99,
            ),
            "single": (
                18,
                "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                0.98989898989899,
                0.98989898989899,
            ),
        },
        "Math": {
            "cardinal": (0.01, 0.7837837837837838, 0.7777777777777778, 8),
            "binary": (
                1.0,
                0.99,
                0.6486486486486487,
                0.6486486486486487,
                0.7222222222222222,
                0.0,
            ),
            "single": (
                18,
                "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                0.8378378378378378,
                0.7222222222222222,
            ),
        },
        "Precise IF": {
            "cardinal": (0.0001, 0.71875, 0.5625, 4),
            "binary": (0.1, 0.25, 0.515625, 0.5625, 0.46875, 0.0),
            "single": (
                18,
                "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                0.75,
                0.5625,
            ),
        },
        "Safety": {
            "cardinal": (1.0, 0.9555555555555556, 0.9777777777777777, 38),
            "binary": (
                0.001,
                0.99,
                0.9333333333333333,
                0.9333333333333333,
                0.9444444444444444,
                8 / 90,
            ),
            "single": (
                1,
                "HFXM/RAMO-Llama3.1-8B",
                0.9666666666666667,
                0.9777777777777777,
            ),
        },
    }
    require(
        set(topic_output["topics"]) == set(topics),
        "topic-conditioned fit metadata has the wrong topics",
    )
    topic_binary_units: dict[str, np.ndarray] = {}
    for topic in topics:
        metadata = topic_output["topics"][topic]
        require(
            tuple(metadata["prompts"][split] for split in ("training", "validation", "test"))
            == TOPIC_SPLITS[topic],
            f"{topic} split counts changed",
        )
        expected = expected_fits[topic]
        cardinal = metadata["learned_weight"]
        expected_regularization, expected_validation, expected_test, expected_nz = (
            expected["cardinal"]
        )
        require(
            np.isclose(cardinal["selected_regularization"], expected_regularization)
            and np.isclose(cardinal["validation_choice_accuracy"], expected_validation)
            and np.isclose(cardinal["test_choice_accuracy"], expected_test)
            and cardinal["nonzero_weights"] == expected_nz,
            f"{topic} topic-cardinal fit changed",
        )
        cardinal_weights = np.asarray(list(cardinal["weights"].values()), dtype=float)
        require(
            len(cardinal_weights) == expected_nz
            and np.all(cardinal_weights > 0.0)
            and np.isclose(np.sum(cardinal_weights), 1.0),
            f"{topic} topic-cardinal weights are invalid",
        )

        binary = metadata["weighted_binary"]
        (
            expected_regularization,
            expected_share,
            expected_raw_validation,
            expected_validation,
            expected_test,
            expected_safe_completeness,
        ) = expected["binary"]
        safe_point = binary["shrinkage_selection_99_validation_operating_point"]
        require(
            np.isclose(binary["selected_regularization"], expected_regularization)
            and np.isclose(binary["learned_weight_share"], expected_share)
            and np.isclose(binary["uniform_weight_share"], 1.0 - expected_share)
            and np.isclose(
                binary["raw_validation_choice_accuracy"],
                expected_raw_validation,
            )
            and np.isclose(binary["validation_choice_accuracy"], expected_validation)
            and np.isclose(binary["test_choice_accuracy"], expected_test),
            f"{topic} topic-binary fit changed",
        )
        require(
            safe_point["pure_response_soundness"] == 1.0
            and np.isclose(
                safe_point["pure_response_completeness"],
                expected_safe_completeness,
            ),
            f"{topic} topic-binary 99% shrinkage point changed",
        )
        binary_weights = np.asarray(list(binary["weights"].values()), dtype=float)
        require(
            binary_weights.shape == (48,)
            and np.all(binary_weights > 0.0)
            and np.isclose(np.sum(binary_weights), 1.0),
            f"{topic} topic-binary weights are invalid",
        )
        topic_binary_units[topic] = verified_binary_weight_units(
            binary,
            reviewer_names,
            BINARY_WEIGHT_UNIT_SHA256[topic],
            topic,
        )

        single = metadata["best_single"]
        expected_index, expected_model, expected_validation, expected_test = (
            expected["single"]
        )
        require(
            single["reviewer_index"] == expected_index
            and single["model"] == expected_model
            and np.isclose(single["validation_choice_accuracy"], expected_validation)
            and np.isclose(single["test_choice_accuracy"], expected_test)
            and single["tie_break"] == "Original reviewer index, ascending.",
            f"{topic} validation-best singleton changed",
        )

    expected_families = {
        "weighted_binary": ((1103,), 0.8441926345609065, 0.8409090909090909),
        "learned_weight": ((1414,), 0.9093484419263456, 0.8920454545454546),
        "best_single": ((1414,), 0.9093484419263456, 0.8778409090909091),
    }
    routed_validation_gains = np.empty((len(validation_prompts), 4))
    for topic in topics:
        positions = np.flatnonzero(subsets[validation_prompts] == topic)
        routed_validation_gains[positions] = reproducible_weighted_gains(
            binary_features[validation_prompts[positions]],
            topic_binary_units[topic],
        )
    expected_topic_binary_thresholds = np.concatenate(
        [
            np.asarray([np.inf]),
            np.sort(np.unique(routed_validation_gains.ravel()))[::-1],
            np.asarray([-np.inf]),
        ]
    )
    stored_topic_binary_thresholds = np.asarray(
        [
            np.inf
            if row["threshold"] == "reject_all"
            else -np.inf
            if row["threshold"] == "accept_all"
            else float(row["threshold"])
            for row in topic_output["weighted_binary"]["rows"]
        ]
    )
    require(
        np.array_equal(
            stored_topic_binary_thresholds,
            expected_topic_binary_thresholds,
        ),
        "topic-weighted binary thresholds are not canonical and validation-defined",
    )
    test_counts = {topic: TOPIC_SPLITS[topic][2] for topic in topics}
    for family, (expected_row_counts, expected_validation, expected_test) in (
        expected_families.items()
    ):
        family_output = topic_output[family]
        rows = family_output["rows"]
        require(
            family_output["shared_threshold_rows"] == len(rows)
            and len(rows) in expected_row_counts,
            f"topic-conditioned {family} threshold path size is invalid",
        )
        require(
            np.isclose(family_output["validation_choice_accuracy"], expected_validation)
            and np.isclose(family_output["test_choice_accuracy"], expected_test),
            f"topic-conditioned {family} pooled choice accuracy changed",
        )
        require(
            rows[0]["threshold"] == "reject_all"
            and rows[-1]["threshold"] == "accept_all",
            f"topic-conditioned {family} sentinel thresholds changed",
        )
        finite_thresholds = np.asarray(
            [float(row["threshold"]) for row in rows[1:-1]],
            dtype=float,
        )
        require(
            np.all(np.isfinite(finite_thresholds))
            and np.all(np.diff(finite_thresholds) < 0.0),
            f"topic-conditioned {family} thresholds are not strictly descending",
        )

        validation_soundness = np.asarray(
            [row["validation"]["pure_response_soundness"] for row in rows]
        )
        validation_completeness = np.asarray(
            [row["validation"]["pure_response_completeness"] for row in rows]
        )
        exact_soundness = np.asarray(
            [row["test_full_domain_soundness"] for row in rows]
        )
        exact_completeness = np.asarray(
            [row["test_full_domain_completeness"] for row in rows]
        )
        approximate_completeness = np.asarray(
            [
                row["test_approximate_full_domain"][
                    "epsilon_good_completeness"
                ]
                for row in rows
            ]
        )
        require(
            np.all(np.diff(validation_soundness) <= 1e-12)
            and np.all(np.diff(validation_completeness) >= -1e-12)
            and np.all(np.diff(exact_soundness) <= 1e-12)
            and np.all(np.diff(exact_completeness) >= -1e-12)
            and np.all(np.diff(approximate_completeness) >= -1e-12),
            f"topic-conditioned {family} threshold path is nonmonotone",
        )
        require(
            np.allclose(
                [exact_soundness[0], exact_completeness[0], approximate_completeness[0]],
                [1.0, 0.0, 0.0],
            )
            and np.allclose(
                [exact_soundness[-1], exact_completeness[-1], approximate_completeness[-1]],
                [0.0, 1.0, 1.0],
            ),
            f"topic-conditioned {family} endpoint metrics changed",
        )
        for row in rows:
            by_topic = row["test_by_topic"]
            require(
                set(by_topic) == set(topics),
                f"topic-conditioned {family} row omits a topic",
            )
            for topic in topics:
                require(
                    by_topic[topic]["prompts"] == test_counts[topic],
                    f"topic-conditioned {family} per-topic denominator changed",
                )
                for metric in (
                    "full_domain_soundness",
                    "full_domain_completeness",
                    "epsilon_good_completeness",
                ):
                    count = by_topic[topic][metric] * test_counts[topic]
                    require(
                        np.isclose(count, round(count)),
                        f"topic-conditioned {family} {topic} rate is not empirical",
                    )
            pooled = {
                metric: sum(
                    by_topic[topic][metric] * test_counts[topic]
                    for topic in topics
                )
                / 352
                for metric in (
                    "full_domain_soundness",
                    "full_domain_completeness",
                    "epsilon_good_completeness",
                )
            }
            require(
                np.isclose(pooled["full_domain_soundness"], row["test_full_domain_soundness"])
                and np.isclose(
                    pooled["full_domain_completeness"],
                    row["test_full_domain_completeness"],
                )
                and np.isclose(
                    pooled["epsilon_good_completeness"],
                    row["test_approximate_full_domain"][
                        "epsilon_good_completeness"
                    ],
                ),
                f"topic-conditioned {family} pooled and per-topic rates disagree",
            )

    expected_frontiers = {
        "weighted_binary": (
            "reject_all",
            1.0,
            0,
            {
                "Factuality": (95, 0),
                "Focus": (99, 0),
                "Math": (36, 0),
                "Precise IF": (32, 0),
                "Safety": (90, 0),
            },
        ),
        "learned_weight": (
            0.45957415213565544,
            335 / 352,
            73,
            {
                "Factuality": (89, 12),
                "Focus": (99, 21),
                "Math": (32, 1),
                "Precise IF": (25, 0),
                "Safety": (90, 39),
            },
        ),
        "best_single": (
            0.5532027391050631,
            335 / 352,
            28,
            {
                "Factuality": (89, 0),
                "Focus": (98, 9),
                "Math": (35, 1),
                "Precise IF": (23, 0),
                "Safety": (90, 18),
            },
        ),
    }
    for family, (
        expected_threshold,
        expected_soundness,
        expected_good_count,
        expected_topic_counts,
    ) in expected_frontiers.items():
        rows = topic_output[family]["rows"]
        feasible = [
            row
            for row in rows
            if row["test_full_domain_soundness"] >= 0.95 - 1e-12
        ]
        frontier = max(
            feasible,
            key=lambda row: row["test_approximate_full_domain"][
                "epsilon_good_completeness"
            ],
        )
        if isinstance(expected_threshold, str):
            threshold_matches = frontier["threshold"] == expected_threshold
        else:
            threshold_matches = np.isclose(
                frontier["threshold"],
                expected_threshold,
            )
        require(
            threshold_matches
            and np.isclose(
                frontier["test_full_domain_soundness"],
                expected_soundness,
            )
            and np.isclose(
                frontier["test_approximate_full_domain"][
                    "epsilon_good_completeness"
                ],
                expected_good_count / 352,
            )
            and frontier["test_full_domain_completeness"] == 0.0,
            f"topic-conditioned {family} 95% frontier changed",
        )
        for topic, (expected_sound_count, expected_good_topic_count) in (
            expected_topic_counts.items()
        ):
            stored = frontier["test_by_topic"][topic]
            require(
                np.isclose(
                    stored["full_domain_soundness"] * stored["prompts"],
                    expected_sound_count,
                )
                and np.isclose(
                    stored["epsilon_good_completeness"] * stored["prompts"],
                    expected_good_topic_count,
                ),
                f"topic-conditioned {family} {topic} 95% breakdown changed",
            )

    global_frontier_good = {}
    for family in ("weighted_binary", "learned_weight", "best_single"):
        feasible = [
            row
            for row in results[family]["rows"]
            if row["test_full_domain_soundness"] >= 0.95 - 1e-12
        ]
        global_frontier_good[family] = max(
            row["test_approximate_full_domain"]["epsilon_good_completeness"]
            for row in feasible
        )
    require(
        np.allclose(
            [
                global_frontier_good["weighted_binary"],
                global_frontier_good["learned_weight"],
                global_frontier_good["best_single"],
            ],
            np.asarray([0, 101, 27]) / 352,
        ),
        "global comparison frontiers changed",
    )
    nonreject_binary_soundness = max(
        row["test_full_domain_soundness"]
        for row in topic_output["weighted_binary"]["rows"]
        if row["threshold"] != "reject_all"
    )
    require(
        np.isclose(nonreject_binary_soundness, 315 / 352),
        "topic-conditioned binary nontrivial soundness ceiling changed",
    )

    tolerance = results["soundness_tolerance_sensitivity"]
    epsilon_harm = np.asarray(tolerance["epsilon_harm"], dtype=float)
    expected_sensitivity = {
        "topic_conditioned_weighted_binary": {
            0.0: 0,
            0.05: 0,
            0.10: 2,
            0.15: 2,
            0.20: 11,
            0.25: 352,
        },
        "topic_conditioned_learned_weight": {
            0.0: 73,
            0.05: 73,
            0.10: 73,
            0.15: 73,
            0.20: 73,
            0.25: 352,
        },
        "topic_conditioned_best_single": {
            0.0: 28,
            0.05: 28,
            0.10: 28,
            0.15: 28,
            0.20: 28,
            0.25: 352,
        },
    }
    for key, checkpoints in expected_sensitivity.items():
        series = tolerance["series"][key]
        completeness = np.asarray(series["best_completeness"], dtype=float)
        achieved_soundness = np.asarray(series["achieved_soundness"], dtype=float)
        selected_index = np.asarray(series["selected_index"], dtype=int)
        require(
            completeness.shape == epsilon_harm.shape
            and achieved_soundness.shape == epsilon_harm.shape
            and selected_index.shape == epsilon_harm.shape
            and np.all(np.diff(completeness) >= -1e-12)
            and np.all(achieved_soundness >= 0.95 - 1e-12),
            f"{key} approximate sensitivity is invalid",
        )
        for epsilon, expected_count in checkpoints.items():
            positions = np.flatnonzero(
                np.isclose(epsilon_harm, epsilon, atol=1e-12, rtol=0.0)
            )
            require(len(positions) == 1, f"missing epsilon_harm={epsilon}")
            require(
                np.isclose(completeness[positions[0]], expected_count / 352),
                f"{key} epsilon_harm={epsilon} checkpoint changed",
            )


def check_train_selected_singleton_results(results: dict[str, object]) -> None:
    comparison = results["train_selected_singleton_generalization"]
    provenance = comparison["selection_provenance"].lower()
    policy = comparison["threshold_policy"]
    require(
        "selected exclusively" in provenance
        and "training prompts" in provenance
        and "frozen before validation or test" in provenance,
        "train-selected singleton provenance changed",
    )
    require(
        comparison["tie_break"] == "Original reviewer index, ascending."
        and comparison["selection_metric"]
        == "Correct-response choice accuracy with response ties split uniformly."
        and comparison["epsilon_good"] == 0.5,
        "train-selected singleton rule semantics changed",
    )
    require(
        not policy["topic_specific_threshold_vector_used"]
        and "one scalar threshold" in policy["form"].lower()
        and "1,058 pooled training prompts" in policy["candidate_source"]
        and "exactly the same train-defined threshold values"
        in policy["evaluation"].lower(),
        "train-selected singleton threshold provenance changed",
    )
    require(
        policy["fixed_checkpoint_thresholds"] == [0.45, 0.55]
        and "predeclared" in policy["fixed_checkpoint_provenance"].lower()
        and "not selected from either frontier"
        in policy["fixed_checkpoint_provenance"].lower(),
        "train-selected singleton fixed-threshold provenance changed",
    )

    # Independently reconstruct the split, normalized reviewer scores,
    # training-only reviewer selections, and train-derived threshold grids.
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        subsets = data["subsets"].astype(str)
        models = data["models"].astype(str)
    standard_prompts = np.flatnonzero(
        (candidate_counts == 4) & (num_correct == 1)
    )
    rng = np.random.default_rng(results["seed"])
    assignment = np.empty(len(subsets), dtype="U10")
    for subset in np.unique(subsets):
        indices = rng.permutation(np.flatnonzero(subsets == subset))
        n_training = int(round(0.6 * len(indices)))
        n_validation = int(round(0.2 * len(indices)))
        assignment[indices[:n_training]] = "train"
        assignment[indices[n_training : n_training + n_validation]] = (
            "validation"
        )
        assignment[indices[n_training + n_validation :]] = "test"
    training_prompts = standard_prompts[
        assignment[standard_prompts] == "train"
    ]
    validation_prompts = standard_prompts[
        assignment[standard_prompts] == "validation"
    ]
    test_prompts = standard_prompts[
        assignment[standard_prompts] == "test"
    ]
    require(
        (len(training_prompts), len(validation_prompts), len(test_prompts))
        == (1058, 353, 352),
        "independent singleton split reconstruction changed",
    )

    features = np.zeros((len(subsets), 4, len(models)))
    for prompt in standard_prompts:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        prompt_scores = scores[:, start:stop]
        centered = prompt_scores - prompt_scores.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        scale[scale < 1e-9] = 1.0
        features[prompt] = (centered / scale).T

    def reviewer_accuracies(prompts: np.ndarray) -> np.ndarray:
        selected = features[prompts]
        maxima = np.max(selected, axis=1, keepdims=True)
        tied = np.isclose(selected, maxima, atol=1e-9, rtol=0.0)
        return np.mean(tied[:, 0, :] / np.sum(tied, axis=1), axis=0)

    def choice_accuracy(gains: np.ndarray, prompts: np.ndarray) -> float:
        selected = gains[prompts]
        maxima = np.max(selected, axis=1, keepdims=True)
        tied = np.isclose(selected, maxima, atol=1e-9, rtol=0.0)
        return float(np.mean(tied[:, 0] / np.sum(tied, axis=1)))

    training_accuracy = reviewer_accuracies(training_prompts)
    global_order = np.lexsort((np.arange(len(models)), -training_accuracy))
    global_reviewer = int(global_order[0])
    require(global_reviewer == 18, "training-best global reviewer changed")
    global_gains = features[:, :, global_reviewer]
    global_output = comparison["global"]
    require(
        global_output["reviewer_index"] == global_reviewer
        and global_output["model"] == str(models[global_reviewer])
        and np.isclose(
            global_output["training_choice_accuracy"],
            training_accuracy[global_reviewer],
        )
        and np.isclose(
            global_output["validation_choice_accuracy"],
            choice_accuracy(global_gains, validation_prompts),
        )
        and np.isclose(
            global_output["test_choice_accuracy"],
            choice_accuracy(global_gains, test_prompts),
        ),
        "training-selected global singleton metadata changed",
    )
    require(
        np.allclose(
            [
                global_output["training_choice_accuracy"],
                global_output["validation_choice_accuracy"],
                global_output["test_choice_accuracy"],
            ],
            [942 / 1058, 317 / 353, 313 / 352],
        ),
        "training-selected global singleton choice checkpoints changed",
    )

    expected_topic_reviewers = {
        "Factuality": (18, "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
        "Focus": (18, "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
        "Math": (18, "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
        "Precise IF": (18, "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
        "Safety": (1, "HFXM/RAMO-Llama3.1-8B"),
    }
    topic_output = comparison["topic_conditioned"]
    topic_metadata = topic_output["reviewers_by_topic"]
    require(
        set(topic_metadata) == set(TOPIC_SPLITS),
        "training-selected topic singleton metadata has the wrong topics",
    )
    topic_gains = np.zeros_like(global_gains)
    for topic, (expected_reviewer, expected_model) in (
        expected_topic_reviewers.items()
    ):
        topic_all = standard_prompts[subsets[standard_prompts] == topic]
        topic_training = topic_all[assignment[topic_all] == "train"]
        topic_validation = topic_all[assignment[topic_all] == "validation"]
        topic_test = topic_all[assignment[topic_all] == "test"]
        topic_accuracy = reviewer_accuracies(topic_training)
        topic_order = np.lexsort((np.arange(len(models)), -topic_accuracy))
        reviewer = int(topic_order[0])
        require(
            reviewer == expected_reviewer,
            f"training-best {topic} reviewer changed",
        )
        topic_gains[topic_all] = features[topic_all, :, reviewer]
        stored = topic_metadata[topic]
        require(
            stored["reviewer_index"] == reviewer
            and stored["model"] == expected_model
            and tuple(
                stored["prompts"][split]
                for split in ("training", "validation", "test")
            )
            == TOPIC_SPLITS[topic]
            and np.isclose(
                stored["training_choice_accuracy"],
                topic_accuracy[reviewer],
            )
            and np.isclose(
                stored["validation_choice_accuracy"],
                choice_accuracy(topic_gains, topic_validation),
            )
            and np.isclose(
                stored["test_choice_accuracy"],
                choice_accuracy(topic_gains, topic_test),
            ),
            f"training-selected {topic} singleton metadata changed",
        )

    require(
        np.allclose(
            [
                topic_output["training_choice_accuracy"],
                topic_output["validation_choice_accuracy"],
                topic_output["test_choice_accuracy"],
            ],
            [944 / 1058, 318 / 353, 314 / 352],
        )
        and np.isclose(
            topic_output["training_choice_accuracy"],
            choice_accuracy(topic_gains, training_prompts),
        )
        and np.isclose(
            topic_output["validation_choice_accuracy"],
            choice_accuracy(topic_gains, validation_prompts),
        )
        and np.isclose(
            topic_output["test_choice_accuracy"],
            choice_accuracy(topic_gains, test_prompts),
        ),
        "training-selected topic singleton pooled choice checkpoints changed",
    )
    require(
        topic_metadata["Factuality"]["reviewer_index"]
        != results["topic_conditioned"]["topics"]["Factuality"][
            "best_single"
        ]["reviewer_index"],
        "training topic selection accidentally reused validation selection",
    )

    expected_policies = {
        "global": (global_gains, 4227),
        "topic_conditioned": (topic_gains, 4222),
    }
    for policy_name, (gains, expected_rows) in expected_policies.items():
        stored_policy = comparison[policy_name]
        rows = stored_policy["rows"]
        require(
            stored_policy["threshold_rows"] == expected_rows
            and len(rows) == expected_rows,
            f"{policy_name} train-selected threshold count changed",
        )
        stored_thresholds = np.asarray(
            [
                np.inf
                if row["threshold"] == "reject_all"
                else -np.inf
                if row["threshold"] == "accept_all"
                else float(row["threshold"])
                for row in rows
            ]
        )
        expected_thresholds = np.concatenate(
            [
                np.asarray([np.inf]),
                np.sort(np.unique(gains[training_prompts].ravel()))[::-1],
                np.asarray([-np.inf]),
            ]
        )
        require(
            np.array_equal(stored_thresholds, expected_thresholds),
            f"{policy_name} thresholds are not exactly train-defined",
        )
        require(
            rows[0]["threshold"] == "reject_all"
            and rows[-1]["threshold"] == "accept_all"
            and np.all(np.diff(stored_thresholds[1:-1]) < 0.0),
            f"{policy_name} train-selected threshold ordering changed",
        )
        for split, denominator in (("train", 1058), ("test", 352)):
            soundness = np.asarray(
                [row[split]["full_domain_soundness"] for row in rows]
            )
            exact_completeness = np.asarray(
                [row[split]["full_domain_completeness"] for row in rows]
            )
            approximate_completeness = np.asarray(
                [row[split]["epsilon_good_completeness"] for row in rows]
            )
            require(
                np.all(np.diff(soundness) <= 1e-12)
                and np.all(np.diff(exact_completeness) >= -1e-12)
                and np.all(np.diff(approximate_completeness) >= -1e-12),
                f"{policy_name} {split} singleton path is nonmonotone",
            )
            require(
                np.allclose(
                    [
                        soundness[0],
                        exact_completeness[0],
                        approximate_completeness[0],
                    ],
                    [1.0, 0.0, 0.0],
                )
                and np.allclose(
                    [
                        soundness[-1],
                        exact_completeness[-1],
                        approximate_completeness[-1],
                    ],
                    [0.0, 1.0, 1.0],
                ),
                f"{policy_name} {split} singleton endpoints changed",
            )
            for values in (
                soundness,
                exact_completeness,
                approximate_completeness,
            ):
                counts = values * denominator
                require(
                    np.allclose(counts, np.round(counts)),
                    f"{policy_name} {split} singleton rates are not empirical",
                )

    def frontier_index(
        rows: list[dict[str, object]],
        split: str,
        target: float,
    ) -> int:
        soundness = np.asarray(
            [row[split]["full_domain_soundness"] for row in rows]
        )
        completeness = np.asarray(
            [row[split]["epsilon_good_completeness"] for row in rows]
        )
        feasible = np.flatnonzero(soundness >= target - 1e-12)
        best = np.max(completeness[feasible])
        tied = feasible[
            np.isclose(completeness[feasible], best, atol=1e-12, rtol=0.0)
        ]
        return int(tied[-1])

    expected_frontiers = {
        "global": {
            ("train", 0.95): (841, 0.617746874525141, 1006, 0),
            ("test", 0.95): (941, 0.5524760714558891, 335, 30),
            ("train", 0.90): (1023, 0.47232017762407746, 953, 398),
            ("test", 0.90): (1037, 0.45928956919538383, 317, 150),
        },
        "topic_conditioned": {
            ("train", 0.95): (849, 0.6156050958243589, 1006, 0),
            ("test", 0.95): (947, 0.5524760714558891, 335, 34),
            ("train", 0.90): (1024, 0.4771588256622271, 957, 415),
            ("test", 0.90): (1041, 0.4555393587797906, 317, 156),
        },
    }
    for policy_name, checkpoints in expected_frontiers.items():
        rows = comparison[policy_name]["rows"]
        for (split, target), (
            expected_index,
            expected_threshold,
            expected_sound_count,
            expected_good_count,
        ) in checkpoints.items():
            index = frontier_index(rows, split, target)
            row = rows[index]
            denominator = 1058 if split == "train" else 352
            require(
                index == expected_index
                and np.isclose(row["threshold"], expected_threshold)
                and np.isclose(
                    row[split]["full_domain_soundness"],
                    expected_sound_count / denominator,
                )
                and np.isclose(
                    row[split]["epsilon_good_completeness"],
                    expected_good_count / denominator,
                )
                and row[split]["full_domain_completeness"] == 0.0,
                f"{policy_name} {split} {target:.0%} frontier changed",
            )
        training_index = checkpoints[("train", 0.95)][0]
        require(
            np.isclose(
                rows[training_index]["test"]["full_domain_soundness"],
                338 / 352,
            )
            and rows[training_index]["test"][
                "epsilon_good_completeness"
            ]
            == 0.0,
            f"{policy_name} train-95 threshold test outcome changed",
        )

    tolerance = results["soundness_tolerance_sensitivity"]
    epsilon_harm = np.asarray(tolerance["epsilon_harm"], dtype=float)
    zero = np.flatnonzero(
        np.isclose(epsilon_harm, 0.0, atol=1e-12, rtol=0.0)
    )
    require(len(zero) == 1, "train-selected sensitivity omits exact soundness")
    for key, expected_good_count, expected_index in (
        ("train_selected_global_singleton", 30, 941),
        ("train_selected_topic_singleton", 34, 947),
    ):
        series = tolerance["series"][key]
        completeness = np.asarray(series["best_completeness"], dtype=float)
        achieved_soundness = np.asarray(series["achieved_soundness"], dtype=float)
        selected_index = np.asarray(series["selected_index"], dtype=int)
        require(
            completeness.shape == epsilon_harm.shape
            and achieved_soundness.shape == epsilon_harm.shape
            and selected_index.shape == epsilon_harm.shape
            and np.all(np.diff(completeness) >= -1e-12)
            and np.all(achieved_soundness >= 0.95 - 1e-12)
            and np.isclose(
                completeness[zero[0]],
                expected_good_count / 352,
            )
            and np.isclose(achieved_soundness[zero[0]], 335 / 352)
            and selected_index[zero[0]] == expected_index
            and completeness[-1] == 1.0,
            f"{key} approximate sensitivity changed",
        )

    expected_direct_thresholds = {
        "global": {
            0.45: (
                (0.8790170132325141, 0.45085066162570886),
                (0.8920454545454546, 0.4431818181818182),
            ),
            0.55: (
                (0.9319470699432892, 0.08790170132325142),
                (0.9488636363636364, 0.10227272727272728),
            ),
        },
        "topic_conditioned": {
            0.45: (
                (0.8809073724007561, 0.4725897920604915),
                (0.8948863636363636, 0.45454545454545453),
            ),
            0.55: (
                (0.9319470699432892, 0.1219281663516068),
                (0.9488636363636364, 0.11363636363636363),
            ),
        },
    }
    for policy_name, expected in expected_direct_thresholds.items():
        checkpoints = comparison[policy_name]["fixed_threshold_checkpoints"]
        require(
            len(checkpoints) == 2
            and [checkpoint["threshold"] for checkpoint in checkpoints]
            == [0.45, 0.55],
            f"{policy_name} direct fixed-threshold grid changed",
        )
        for checkpoint in checkpoints:
            threshold = checkpoint["threshold"]
            expected_train, expected_test = expected[threshold]
            require(
                np.allclose(
                    [
                        checkpoint["train"]["full_domain_soundness"],
                        checkpoint["train"]["epsilon_good_completeness"],
                    ],
                    expected_train,
                )
                and np.allclose(
                    [
                        checkpoint["test"]["full_domain_soundness"],
                        checkpoint["test"]["epsilon_good_completeness"],
                    ],
                    expected_test,
                ),
                f"{policy_name} direct fixed-tau {threshold:.2f} checkpoint changed",
            )

    # The direct fixed-threshold audit above is the report checkpoint.  These
    # separate checks ensure that the nearest members of each train-derived
    # plotting grid also remain reproducible; they need not equal tau exactly.
    expected_nearest_grid_thresholds = {
        "global": {
            0.45: (
                0.450230346576342,
                (0.8790170132325141, 0.45085066162570886),
                (0.8920454545454546, 0.4431818181818182),
            ),
            0.55: (
                0.5494357803902453,
                (0.9319470699432892, 0.09546313799621928),
                (0.9488636363636364, 0.10511363636363637),
            ),
        },
        "topic_conditioned": {
            0.45: (
                0.450230346576342,
                (0.8809073724007561, 0.4725897920604915),
                (0.8948863636363636, 0.45454545454545453),
            ),
            0.55: (
                0.5494357803902453,
                (0.9319470699432892, 0.12570888468809074),
                (0.9488636363636364, 0.11931818181818182),
            ),
        },
    }
    for policy_name, fixed in expected_nearest_grid_thresholds.items():
        rows = comparison[policy_name]["rows"]
        finite = np.asarray(
            [
                np.nan if isinstance(row["threshold"], str) else row["threshold"]
                for row in rows
            ],
            dtype=float,
        )
        for target, (expected_threshold, expected_train, expected_test) in (
            fixed.items()
        ):
            index = int(np.nanargmin(np.abs(finite - target)))
            row = rows[index]
            require(
                np.isclose(row["threshold"], expected_threshold)
                and np.allclose(
                    [
                        row["train"]["full_domain_soundness"],
                        row["train"]["epsilon_good_completeness"],
                    ],
                    expected_train,
                )
                and np.allclose(
                    [
                        row["test"]["full_domain_soundness"],
                        row["test"]["epsilon_good_completeness"],
                    ],
                    expected_test,
                ),
                f"{policy_name} nearest-grid tau={target:.2f} checkpoint changed",
            )


def check_topic_threshold_budget_results(
    cardinal_results: dict[str, object],
) -> None:
    calibration = json.loads(
        (DATA / "topic_threshold_budget_results.json").read_text()
    )
    topics = list(TOPIC_SPLITS)
    require(
        calibration["seed"] == cardinal_results["seed"]
        and calibration["dataset_revision"]
        == cardinal_results["dataset_revision"]
        and calibration["results_revision"]
        == cardinal_results["results_revision"]
        and calibration["cohort"]
        == {
            key: cardinal_results["cohort"][key]
            for key in ("training_prompts", "validation_prompts", "test_prompts")
        },
        "topic-threshold result provenance does not match cardinal results",
    )
    require(
        calibration["selection_split"] == "validation"
        and calibration["test_labels_used_for_selection"] is False
        and calibration["score_functions_frozen_before_threshold_calibration"]
        is True
        and calibration["topics_order"] == topics
        and calibration["epsilon_good"] == 0.5
        and calibration["ties_approve"] is True
        and "exact complete-simplex"
        in calibration["full_domain_audit"].lower()
        and "full-lottery" in calibration["full_domain_audit"].lower()
        and "not sampled" in calibration["full_domain_audit"].lower(),
        "topic-threshold calibration provenance changed",
    )
    budgets = calibration["validation_failure_budgets"]
    require(
        budgets["minimum"] == 0
        and budgets["maximum"] == 353
        and budgets["values"] == list(range(354))
        and budgets["primary_95_soundness_budget"] == 17
        and budgets["checkpoints"]
        == {"0.99": 3, "0.95": 17, "0.90": 35, "0.80": 70},
        "topic-threshold failure budgets changed",
    )
    require(
        "behaviorally exhaustive" in calibration["candidate_grid"].lower()
        and "sum" in calibration["topic_vector_constraint"].lower()
        and "test outcomes never break a tie" in calibration["tie_break"].lower()
        and "frozen" in calibration["evaluation"].lower(),
        "topic-threshold selection semantics are incomplete",
    )

    expected_primary = {
        "learned_topic_cardinal": {
            "shared_threshold": (17, 336, 75, 334, 84),
            "topic_threshold_vector": (17, 336, 122, 337, 132),
        },
        "global_cardinal_topic_threshold_control": {
            "shared_threshold": (17, 336, 101, 331, 104),
            "topic_threshold_vector": (17, 336, 137, 334, 144),
        },
        "train_selected_global_singleton_topic_threshold_control": {
            "shared_threshold": (17, 336, 40, 333, 44),
            "topic_threshold_vector": (17, 336, 158, 327, 151),
        },
        "train_selected_topic_singleton": {
            "shared_threshold": (17, 336, 46, 333, 47),
            "topic_threshold_vector": (17, 336, 155, 336, 147),
        },
    }
    families = calibration["families"]
    require(
        set(families) == set(expected_primary),
        "topic-threshold family set changed",
    )
    require(
        all(
            "validation" in family["candidate_source"].lower()
            and "critical" in family["candidate_source"].lower()
            for family in families.values()
        )
        and "training"
        in families[
            "train_selected_global_singleton_topic_threshold_control"
        ]["candidate_source"].lower()
        and "training" in families["train_selected_topic_singleton"][
            "candidate_source"
        ].lower(),
        "topic-threshold candidate provenance changed",
    )

    score_rules = {
        family_name: family["score_rule"]
        for family_name, family in families.items()
    }

    def canonicalized_weights_match(
        rule: dict[str, object],
        fitted_sparse: dict[str, float],
    ) -> bool:
        """Rebuild the calibration committee from stored fitted weights."""
        calibrated = rule["weights_by_reviewer"]
        stored_units = rule["calibration_weight_units_by_reviewer"]
        names = list(calibrated)
        if (
            not set(fitted_sparse).issubset(names)
            or list(stored_units) != names
        ):
            return False
        fitted = np.asarray(
            [fitted_sparse.get(name, 0.0) for name in names],
            dtype=float,
        )
        scale = 10**CALIBRATION_WEIGHT_DECIMALS
        expected_units = np.rint(
            np.maximum(fitted, 0.0) * scale
        ).astype(np.int64)
        total_units = int(np.sum(expected_units))
        if total_units <= 0:
            return False
        expected = expected_units.astype(float) / total_units
        observed = np.asarray([calibrated[name] for name in names])
        observed_units = np.asarray(
            [stored_units[name] for name in names],
            dtype=np.int64,
        )
        delta = expected - fitted
        scaled = np.maximum(fitted, 0.0) * scale
        metadata = rule["calibration_weight_canonicalization"]
        expected_metadata = {
            "decimal_places": CALIBRATION_WEIGHT_DECIMALS,
            "integer_unit_rounding": "nearest, ties to even",
            "integer_unit_total": total_units,
            "integer_units_sha256": hashlib.sha256(
                np.asarray(expected_units, dtype="<i8").tobytes()
            ).hexdigest(),
            "reviewer_order_sha256": hashlib.sha256(
                "\0".join(names).encode("utf-8")
            ).hexdigest(),
            "renormalized_to_sum_one": True,
            "aggregation": (
                "math.fsum of feature times integer unit in fixed reviewer "
                "archive order, divided once by the integer-unit total"
            ),
            "maximum_absolute_weight_change": float(
                np.max(np.abs(delta))
            ),
            "l1_weight_change": float(np.sum(np.abs(delta))),
            "minimum_distance_to_half_unit_boundary": float(
                np.min(np.abs(scaled - (np.floor(scaled) + 0.5)))
            ),
        }
        return (
            np.array_equal(observed_units, expected_units)
            and np.array_equal(observed, expected)
            and metadata == expected_metadata
        )
    require(
        all(
            rule["frozen_before_threshold_calibration"] is True
            and rule["test_labels_used"] is False
            for rule in score_rules.values()
        ),
        "topic-threshold score functions were not frozen before calibration",
    )

    global_cardinal_rule = score_rules[
        "global_cardinal_topic_threshold_control"
    ]
    require(
        global_cardinal_rule["kind"]
        == "learned cardinal weighted score"
        and global_cardinal_rule["parameter_scope"] == "global"
        and global_cardinal_rule["weight_fit_split"] == "training"
        and global_cardinal_rule["regularization_selection_split"]
        == "validation"
        and global_cardinal_rule["same_frozen_score_rule_in_every_topic"],
        "global-cardinal score-rule provenance changed",
    )
    require(
        np.isclose(
            sum(global_cardinal_rule["weights_by_reviewer"].values()),
            1.0,
        )
        and global_cardinal_rule["selected_regularization"]
        == cardinal_results["learned_weight"]["selected_regularization"]
        and global_cardinal_rule["calibration_weight_canonicalization"][
            "integer_units_sha256"
        ]
        == CALIBRATION_WEIGHT_UNIT_SHA256["global"]
        and canonicalized_weights_match(
            global_cardinal_rule,
            cardinal_results["learned_weight"]["weights"],
        ),
        "global-cardinal calibration does not use the frozen learned weights",
    )

    topic_cardinal_rule = score_rules["learned_topic_cardinal"]
    topic_cardinal_results = cardinal_results["topic_conditioned"]["topics"]
    require(
        topic_cardinal_rule["kind"] == "learned cardinal weighted score"
        and topic_cardinal_rule["parameter_scope"] == "topic-routed"
        and topic_cardinal_rule["weight_fit_split"]
        == "within-topic training"
        and topic_cardinal_rule["regularization_selection_split"]
        == "within-topic validation"
        and list(topic_cardinal_rule["topics"]) == topics,
        "topic-cardinal score-rule provenance changed",
    )
    for topic in topics:
        rule = topic_cardinal_rule["topics"][topic]
        stored = topic_cardinal_results[topic]["learned_weight"]
        require(
            np.isclose(sum(rule["weights_by_reviewer"].values()), 1.0)
            and rule["selected_regularization"]
            == stored["selected_regularization"]
            and rule["calibration_weight_canonicalization"][
                "integer_units_sha256"
            ]
            == CALIBRATION_WEIGHT_UNIT_SHA256[topic]
            and canonicalized_weights_match(
                rule,
                stored["weights"],
            ),
            f"{topic} cardinal calibration does not use frozen learned weights",
        )

    singleton_results = cardinal_results[
        "train_selected_singleton_generalization"
    ]
    global_singleton_rule = score_rules[
        "train_selected_global_singleton_topic_threshold_control"
    ]
    stored_global_singleton = singleton_results["global"]
    require(
        global_singleton_rule["kind"] == "single reviewer score"
        and global_singleton_rule["parameter_scope"] == "global"
        and global_singleton_rule["reviewer_selection_split"] == "training"
        and global_singleton_rule["reviewer_selection_metric"]
        .lower()
        .rstrip(".")
        == singleton_results["selection_metric"].lower().rstrip(".")
        and global_singleton_rule["same_frozen_score_rule_in_every_topic"]
        and global_singleton_rule["reviewer_index"]
        == stored_global_singleton["reviewer_index"]
        and global_singleton_rule["model"]
        == stored_global_singleton["model"]
        and np.isclose(
            global_singleton_rule["training_choice_accuracy"],
            stored_global_singleton["training_choice_accuracy"],
        ),
        "global-singleton calibration does not use the training-selected reviewer",
    )
    topic_singleton_rule = score_rules["train_selected_topic_singleton"]
    stored_topic_singletons = singleton_results["topic_conditioned"][
        "reviewers_by_topic"
    ]
    require(
        topic_singleton_rule["kind"] == "single reviewer score"
        and topic_singleton_rule["parameter_scope"] == "topic-routed"
        and topic_singleton_rule["reviewer_selection_split"]
        == "within-topic training"
        and topic_singleton_rule["reviewer_selection_metric"]
        .lower()
        .rstrip(".")
        == singleton_results["selection_metric"].lower().rstrip(".")
        and list(topic_singleton_rule["reviewers_by_topic"]) == topics,
        "topic-singleton score-rule provenance changed",
    )
    for topic in topics:
        rule = topic_singleton_rule["reviewers_by_topic"][topic]
        stored = stored_topic_singletons[topic]
        require(
            rule["reviewer_index"] == stored["reviewer_index"]
            and rule["model"] == stored["model"]
            and np.isclose(
                rule["training_choice_accuracy"],
                stored["training_choice_accuracy"],
            ),
            f"{topic} calibration does not use its training-selected reviewer",
        )

    def threshold_number(value: float | str) -> float:
        if value == "reject_all":
            return float(np.inf)
        if value == "accept_all":
            return float(-np.inf)
        return float(value)

    def threshold_identity(value: float | str) -> tuple[str, float | str]:
        return (
            ("sentinel", value)
            if isinstance(value, str)
            else ("finite", float(value))
        )

    def validation_selection_payload(
        family: dict[str, object],
    ) -> dict[str, object]:
        """Mirror cardinal_analysis.validation_selection_payload exactly."""
        payload = {
            "candidate_source": family["candidate_source"],
            "score_rule": family["score_rule"],
            "validation_topic_counts": family["validation_topic_counts"],
            "policies": {},
        }
        for policy_name in ("shared_threshold", "topic_threshold_vector"):
            policy = family[policy_name]
            if policy_name == "shared_threshold":
                grid_and_profiles = {
                    "candidate_thresholds": policy["candidate_thresholds"],
                    "pareto_validation_profiles": policy[
                        "pareto_validation_profiles"
                    ],
                }
            else:
                grid_and_profiles = {
                    "candidate_thresholds_by_topic": policy[
                        "candidate_thresholds_by_topic"
                    ],
                    "pareto_validation_profiles_by_topic": policy[
                        "pareto_validation_profiles_by_topic"
                    ],
                }
            payload["policies"][policy_name] = {
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
                    for row in policy["rows"]
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

    def check_profile_grid(
        candidate_thresholds: list[float | str],
        profiles: list[dict[str, object]],
        expected_count: int,
        label: str,
        all_canonical: bool,
    ) -> None:
        identities = [
            threshold_identity(threshold)
            for threshold in candidate_thresholds
        ]
        identity_set = set(identities)
        require(
            len(candidate_thresholds) == expected_count
            and len(identity_set) == expected_count
            and candidate_thresholds[0] == "reject_all"
            and candidate_thresholds[-1] == "accept_all"
            and np.all(
                np.diff(
                    [threshold_number(value) for value in candidate_thresholds]
                )
                < 0
            ),
            f"{label} critical candidate grid changed",
        )
        failures = np.asarray(
            [profile["unsound_prompts"] for profile in profiles],
            dtype=int,
        )
        completions = np.asarray(
            [
                profile["epsilon_good_complete_prompts"]
                for profile in profiles
            ],
            dtype=int,
        )
        require(
            len(profiles) > 0
            and np.all(np.diff(failures) > 0)
            and np.all(np.diff(completions) > 0)
            and all(
                threshold_identity(profile["threshold"]) in identity_set
                for profile in profiles
            )
            and all(
                isinstance(profile.get("canonical_local_candidate"), bool)
                for profile in profiles
            )
            and (
                not all_canonical
                or all(
                    profile["canonical_local_candidate"]
                    for profile in profiles
                )
            ),
            f"{label} Pareto validation profiles changed",
        )

    def expected_shared_rows(
        profiles: list[dict[str, object]],
    ) -> list[tuple[int, int, float | str]]:
        selected = []
        for budget in range(354):
            feasible = [
                profile
                for profile in profiles
                if profile["unsound_prompts"] <= budget
            ]
            best = max(
                feasible,
                key=lambda profile: (
                    profile["epsilon_good_complete_prompts"],
                    -profile["unsound_prompts"],
                    threshold_number(profile["threshold"]),
                ),
            )
            selected.append(
                (
                    best["unsound_prompts"],
                    best["epsilon_good_complete_prompts"],
                    best["threshold"],
                )
            )
        return selected

    def expected_vector_rows(
        profiles_by_topic: dict[str, list[dict[str, object]]],
    ) -> list[tuple[int, int, tuple[float | str, ...]]]:
        states: dict[int, tuple[int, tuple[float | str, ...]]] = {
            0: (0, ())
        }
        for topic in topics:
            next_states = {}
            for used_failures, (completions, thresholds) in states.items():
                for profile in profiles_by_topic[topic]:
                    new_failures = used_failures + profile["unsound_prompts"]
                    candidate = (
                        completions
                        + profile["epsilon_good_complete_prompts"],
                        thresholds + (profile["threshold"],),
                    )
                    incumbent = next_states.get(new_failures)
                    if incumbent is None or (
                        candidate[0],
                        tuple(threshold_number(value) for value in candidate[1]),
                    ) > (
                        incumbent[0],
                        tuple(threshold_number(value) for value in incumbent[1]),
                    ):
                        next_states[new_failures] = candidate
            states = next_states

        selected = []
        for budget in range(354):
            used_failures, (completions, thresholds) = max(
                (
                    (used, state)
                    for used, state in states.items()
                    if used <= budget
                ),
                key=lambda item: (
                    item[1][0],
                    -item[0],
                    tuple(
                        threshold_number(value) for value in item[1][1]
                    ),
                ),
            )
            selected.append((used_failures, completions, thresholds))
        return selected

    for family_name, policies in expected_primary.items():
        family = families[family_name]
        require(
            family["validation_topic_counts"]
            == {topic: values[1] for topic, values in TOPIC_SPLITS.items()}
            and family["test_topic_counts"]
            == {topic: values[2] for topic, values in TOPIC_SPLITS.items()},
            f"{family_name} topic counts changed",
        )
        selection_audit = family["selection_audit"]
        recomputed_selection_digest = json_sha256(
            validation_selection_payload(family)
        )
        require(
            selection_audit["selection_function_inputs"]
            == [
                "frozen validation score array",
                "validation topic labels",
                "candidate-source description",
                "frozen score-rule provenance",
                "fixed topic order",
            ]
            and selection_audit[
                "selection_function_receives_held_out_inputs"
            ]
            is False
            and selection_audit[
                "calibrated_test_metrics_attached_after_selection"
            ]
            is True
            and selection_audit["selection_unchanged_by_test_evaluation"]
            is True
            and selection_audit[
                "validation_selection_sha256_before_test_evaluation"
            ]
            == recomputed_selection_digest
            and selection_audit[
                "validation_selection_sha256_after_test_evaluation"
            ]
            == recomputed_selection_digest,
            f"{family_name} validation-only selection digest changed",
        )

        feasibility_audit = family["shared_rule_feasibility_audit"]
        require(
            feasibility_audit["all_shared_candidates_in_every_topic_grid"]
            is True
            and feasibility_audit["budgets_checked"] == 354
            and feasibility_audit[
                "topic_vector_validation_completeness_weakly_dominates_shared"
            ]
            is True
            and "union" in feasibility_audit["raw_grid_construction"].lower()
            and "tau maps to" in feasibility_audit[
                "shared_vector_embedding"
            ].lower(),
            f"{family_name} shared-rule feasibility audit changed",
        )
        for policy_name, expected in policies.items():
            policy = family[policy_name]
            rows = policy["rows"]
            require(len(rows) == 354, f"{family_name} {policy_name} path changed")
            observed_budgets = np.asarray(
                [row["validation_failure_budget"] for row in rows],
                dtype=int,
            )
            used = np.asarray(
                [row["used_validation_failures"] for row in rows],
                dtype=int,
            )
            validation_complete = np.asarray(
                [
                    row["validation"]["epsilon_good_complete_prompts"]
                    for row in rows
                ],
                dtype=int,
            )
            require(
                np.array_equal(observed_budgets, np.arange(354))
                and np.all(used <= observed_budgets)
                and np.all(np.diff(validation_complete) >= 0)
                and validation_complete[-1] == 353,
                f"{family_name} {policy_name} budget optimum is invalid",
            )

            if policy_name == "shared_threshold":
                profiles = policy["pareto_validation_profiles"]
                require(
                    len(profiles) == policy["pareto_option_count"],
                    f"{family_name} shared Pareto count changed",
                )
                check_profile_grid(
                    policy["candidate_thresholds"],
                    profiles,
                    policy["candidate_threshold_count"],
                    f"{family_name} shared",
                    True,
                )
                independent = expected_shared_rows(profiles)
            else:
                profiles_by_topic = policy[
                    "pareto_validation_profiles_by_topic"
                ]
                candidates_by_topic = policy[
                    "candidate_thresholds_by_topic"
                ]
                require(
                    list(profiles_by_topic) == topics
                    and list(candidates_by_topic) == topics
                    and list(
                        policy[
                            "canonical_local_candidate_threshold_counts_by_topic"
                        ]
                    )
                    == topics
                    and list(policy["candidate_threshold_counts_by_topic"])
                    == topics
                    and list(policy["pareto_option_counts_by_topic"])
                    == topics
                    and policy["required_pooled_candidate_threshold_count"]
                    == family["shared_threshold"][
                        "candidate_threshold_count"
                    ],
                    f"{family_name} topic profile order changed",
                )
                shared_identities = {
                    threshold_identity(value)
                    for value in family["shared_threshold"][
                        "candidate_thresholds"
                    ]
                }
                for topic in topics:
                    candidate_count = policy[
                        "candidate_threshold_counts_by_topic"
                    ][topic]
                    canonical_count = policy[
                        "canonical_local_candidate_threshold_counts_by_topic"
                    ][topic]
                    required_count = policy[
                        "required_pooled_candidate_threshold_count"
                    ]
                    topic_identities = {
                        threshold_identity(value)
                        for value in candidates_by_topic[topic]
                    }
                    require(
                        len(profiles_by_topic[topic])
                        == policy["pareto_option_counts_by_topic"][topic]
                        and max(canonical_count, required_count)
                        <= candidate_count
                        <= canonical_count + required_count - 2
                        and shared_identities.issubset(topic_identities),
                        f"{family_name} {topic} Pareto count changed",
                    )
                    check_profile_grid(
                        candidates_by_topic[topic],
                        profiles_by_topic[topic],
                        candidate_count,
                        f"{family_name} {topic}",
                        False,
                    )
                independent = expected_vector_rows(profiles_by_topic)

            for row in rows:
                require(
                    list(row["thresholds_by_topic"]) == topics,
                    f"{family_name} {policy_name} threshold order changed",
                )
                if policy_name == "shared_threshold":
                    require(
                        all(
                            threshold_identity(threshold)
                            == threshold_identity(row["threshold"])
                            for threshold in row["thresholds_by_topic"].values()
                        ),
                        f"{family_name} shared threshold is not shared",
                    )
                for split, denominator, count_position in (
                    ("validation", 353, 1),
                    ("test", 352, 2),
                ):
                    metrics = row[split]
                    require(
                        metrics["prompts"] == denominator
                        and metrics["sound_prompts"]
                        + metrics["unsound_prompts"]
                        == denominator
                        and np.isclose(
                            metrics["full_domain_soundness"],
                            metrics["sound_prompts"] / denominator,
                        )
                        and np.isclose(
                            metrics["epsilon_good_completeness"],
                            metrics["epsilon_good_complete_prompts"]
                            / denominator,
                        ),
                        f"{family_name} {policy_name} {split} rates changed",
                    )
                    by_topic = row[f"{split}_by_topic"]
                    require(
                        list(by_topic) == topics
                        and all(
                            by_topic[topic]["prompts"]
                            == TOPIC_SPLITS[topic][count_position]
                            for topic in topics
                        )
                        and all(
                            by_topic[topic]["sound_prompts"]
                            + by_topic[topic]["unsound_prompts"]
                            == by_topic[topic]["prompts"]
                            and np.isclose(
                                by_topic[topic]["full_domain_soundness"],
                                by_topic[topic]["sound_prompts"]
                                / by_topic[topic]["prompts"],
                            )
                            and np.isclose(
                                by_topic[topic][
                                    "epsilon_good_completeness"
                                ],
                                by_topic[topic][
                                    "epsilon_good_complete_prompts"
                                ]
                                / by_topic[topic]["prompts"],
                            )
                            for topic in topics
                        )
                        and sum(
                            by_topic[topic]["sound_prompts"]
                            for topic in topics
                        )
                        == metrics["sound_prompts"]
                        and sum(
                            by_topic[topic][
                                "epsilon_good_complete_prompts"
                            ]
                            for topic in topics
                        )
                        == metrics["epsilon_good_complete_prompts"],
                        f"{family_name} {policy_name} {split} topic reconstruction changed",
                    )
                require(
                    row["used_validation_failures"]
                    == row["validation"]["unsound_prompts"],
                    f"{family_name} {policy_name} used budget disagrees",
                )

            for row, independently_selected in zip(rows, independent):
                if policy_name == "shared_threshold":
                    expected_used, expected_complete, expected_threshold = (
                        independently_selected
                    )
                    threshold_matches = threshold_identity(row["threshold"]) == (
                        threshold_identity(expected_threshold)
                    )
                else:
                    expected_used, expected_complete, expected_thresholds = (
                        independently_selected
                    )
                    threshold_matches = tuple(
                        threshold_identity(row["thresholds_by_topic"][topic])
                        for topic in topics
                    ) == tuple(
                        threshold_identity(value)
                        for value in expected_thresholds
                    )
                require(
                    row["used_validation_failures"] == expected_used
                    and row["validation"][
                        "epsilon_good_complete_prompts"
                    ]
                    == expected_complete
                    and threshold_matches,
                    f"{family_name} {policy_name} DP optimum changed",
                )

            primary = policy["primary_95_validation_target"]
            selected_checkpoint = (
                primary["used_validation_failures"],
                primary["validation"]["sound_prompts"],
                primary["validation"]["epsilon_good_complete_prompts"],
                primary["test"]["sound_prompts"],
                primary["test"]["epsilon_good_complete_prompts"],
            )
            selected_row = rows[17]
            require(
                primary["validation_failure_budget"] == 17
                and primary["used_validation_failures"]
                == selected_row["used_validation_failures"]
                and primary["thresholds_by_topic"]
                == selected_row["thresholds_by_topic"]
                and primary["validation"] == selected_row["validation"]
                and primary["test"] == selected_row["test"]
                and (expected is None or selected_checkpoint == expected),
                f"{family_name} {policy_name} 95% checkpoint changed",
            )
            for split in ("validation", "test"):
                require(
                    sum(
                        primary["by_topic"][topic][split]["sound_prompts"]
                        for topic in topics
                    )
                    == primary[split]["sound_prompts"]
                    and sum(
                        primary["by_topic"][topic][split][
                            "epsilon_good_complete_prompts"
                        ]
                        for topic in topics
                    )
                    == primary[split]["epsilon_good_complete_prompts"],
                    f"{family_name} {policy_name} topic totals changed",
                )
            require(
                all(
                    threshold_identity(
                        primary["by_topic"][topic]["threshold"]
                    )
                    == threshold_identity(
                        selected_row["thresholds_by_topic"][topic]
                    )
                    and primary["by_topic"][topic]["validation"]
                    == selected_row["validation_by_topic"][topic]
                    and primary["by_topic"][topic]["test"]
                    == selected_row["test_by_topic"][topic]
                    for topic in topics
                ),
                f"{family_name} {policy_name} primary topic detail changed",
            )

        shared_rows = family["shared_threshold"]["rows"]
        vector_rows = family["topic_threshold_vector"]["rows"]
        profiles_by_topic = family["topic_threshold_vector"][
            "pareto_validation_profiles_by_topic"
        ]
        for shared_row, vector_row in zip(shared_rows, vector_rows):
            witness_failures = 0
            witness_completions = 0
            for topic in topics:
                shared_topic = shared_row["validation_by_topic"][topic]
                feasible_witnesses = [
                    profile
                    for profile in profiles_by_topic[topic]
                    if profile["unsound_prompts"]
                    <= shared_topic["unsound_prompts"]
                    and profile["epsilon_good_complete_prompts"]
                    >= shared_topic["epsilon_good_complete_prompts"]
                ]
                require(
                    bool(feasible_witnesses),
                    f"{family_name} {topic} lacks a shared-rule Pareto witness",
                )
                witness = max(
                    feasible_witnesses,
                    key=lambda profile: (
                        profile["epsilon_good_complete_prompts"],
                        -profile["unsound_prompts"],
                        threshold_number(profile["threshold"]),
                    ),
                )
                witness_failures += witness["unsound_prompts"]
                witness_completions += witness[
                    "epsilon_good_complete_prompts"
                ]
            require(
                witness_failures
                <= shared_row["used_validation_failures"]
                and witness_completions
                >= shared_row["validation"][
                    "epsilon_good_complete_prompts"
                ]
                and vector_row["validation"][
                    "epsilon_good_complete_prompts"
                ]
                >= shared_row["validation"][
                    "epsilon_good_complete_prompts"
                ],
                f"{family_name} shared policy is not nested in topic calibration",
            )


def check_cardinal_results() -> None:
    results = json.loads((DATA / "cardinal_results.json").read_text())
    require(
        results["cohort"]
        == {
            "standard_prompts": 1763,
            "training_prompts": 1058,
            "validation_prompts": 353,
            "test_prompts": 352,
        },
        "held-out split changed",
    )
    reconstruction = results["prompt_specific_cardinal_reconstruction"]
    require(
        (reconstruction["test_covered_prompts"], reconstruction["test_prompts"])
        == (315, 352),
        "test reconstruction count changed",
    )
    require(
        reconstruction["maximum_lp_residual"] < 2e-14,
        "prompt-specific reconstruction residual is too large",
    )

    # The pure-response figure reuses the original validation-defined
    # threshold families.  Independently reconstruct those grids and audit
    # the held-out endpoint paths before checking its descriptive markers.
    with np.load(DATA / "rewardbench2_scores.npz", allow_pickle=True) as data:
        scores = data["candidate_scores"].astype(float)
        offsets = data["offsets"].astype(int)
        candidate_counts = data["candidate_counts"].astype(int)
        num_correct = data["num_correct"].astype(int)
        subsets = data["subsets"].astype(str)
        models = data["models"].astype(str)
    standard_prompts = np.flatnonzero(
        (candidate_counts == 4) & (num_correct == 1)
    )
    rng = np.random.default_rng(results["seed"])
    assignment = np.empty(len(subsets), dtype="U10")
    for subset in np.unique(subsets):
        indices = rng.permutation(np.flatnonzero(subsets == subset))
        n_training = int(round(0.6 * len(indices)))
        n_validation = int(round(0.2 * len(indices)))
        assignment[indices[:n_training]] = "train"
        assignment[indices[n_training : n_training + n_validation]] = (
            "validation"
        )
        assignment[indices[n_training + n_validation :]] = "test"
    validation_prompts = standard_prompts[
        assignment[standard_prompts] == "validation"
    ]
    test_prompts = standard_prompts[assignment[standard_prompts] == "test"]
    require(
        (len(validation_prompts), len(test_prompts)) == (353, 352),
        "pure-response split reconstruction changed",
    )

    features = np.zeros((len(subsets), 4, len(models)))
    for prompt in standard_prompts:
        start, stop = int(offsets[prompt]), int(offsets[prompt + 1])
        prompt_scores = scores[:, start:stop]
        centered = prompt_scores - prompt_scores.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        scale[scale < 1e-9] = 1.0
        features[prompt] = (centered / scale).T
    binary_features = (features >= -1e-9).astype(float)

    def stored_weights(key: str) -> np.ndarray:
        weights = np.zeros(len(models))
        by_model = results[key]["weights"]
        for index, model in enumerate(models):
            weights[index] = float(by_model.get(str(model), 0.0))
        return weights

    reviewer_names = [str(model) for model in models]
    weighted_binary_units = verified_binary_weight_units(
        results["weighted_binary"],
        reviewer_names,
        BINARY_WEIGHT_UNIT_SHA256["global"],
        "global",
    )

    selected_model = str(results["best_single"]["model"])
    selected_indices = np.flatnonzero(models == selected_model)
    require(
        len(selected_indices) == 1,
        "validation-selected singleton model is not unique",
    )
    removed_model = str(
        results["ablated_learned_weight"]["removed_model"]
    )
    retained_reviewers = models != removed_model
    ablated_weights = np.asarray(
        [
            results["ablated_learned_weight"]["weights"].get(
                str(model), 0.0
            )
            for model in models[retained_reviewers]
        ],
        dtype=float,
    )
    pure_gains = {
        "weighted_binary": reproducible_weighted_gains(
            binary_features,
            weighted_binary_units,
        ),
        "equal_weight": np.mean(features, axis=2),
        "learned_weight": features @ stored_weights("learned_weight"),
        "ablated_learned_weight": features[:, :, retained_reviewers]
        @ ablated_weights,
        "best_single": features[:, :, int(selected_indices[0])],
    }
    expected_row_counts = {
        "count_rule": 50,
        "weighted_binary": 983,
        "equal_weight": 1414,
        "learned_weight": 1414,
        "ablated_learned_weight": 1414,
        "best_single": 1414,
    }
    for key, expected_rows in expected_row_counts.items():
        rows = results[key]["rows"]
        require(
            len(rows) == expected_rows,
            f"{key} pure-response threshold count changed",
        )
        if key == "count_rule":
            require(
                [row["k"] for row in rows] == list(range(-1, 49)),
                "count-rule pure-response grid changed",
            )
        else:
            stored_thresholds = np.asarray(
                [
                    np.inf
                    if row["threshold"] == "reject_all"
                    else -np.inf
                    if row["threshold"] == "accept_all"
                    else float(row["threshold"])
                    for row in rows
                ]
            )
            expected_thresholds = np.concatenate(
                [
                    np.asarray([np.inf]),
                    np.sort(
                        np.unique(pure_gains[key][validation_prompts].ravel())
                    )[::-1],
                    np.asarray([-np.inf]),
                ]
            )
            require(
                np.allclose(
                    stored_thresholds,
                    expected_thresholds,
                    atol=1e-12,
                    rtol=0.0,
                ),
                f"{key} pure-response thresholds are not validation-defined",
            )

        for split, denominator in (("validation", 353), ("test", 352)):
            soundness = np.asarray(
                [row[split]["pure_response_soundness"] for row in rows]
            )
            completeness = np.asarray(
                [row[split]["pure_response_completeness"] for row in rows]
            )
            require(
                np.all(np.diff(soundness) <= 1e-12)
                and np.all(np.diff(completeness) >= -1e-12),
                f"{key} {split} pure-response path is nonmonotone",
            )
            require(
                np.allclose(
                    [soundness[0], completeness[0]],
                    [1.0, 0.0],
                )
                and np.allclose(
                    [soundness[-1], completeness[-1]],
                    [0.0, 1.0],
                ),
                f"{key} {split} pure-response sentinels changed",
            )
            require(
                np.allclose(
                    soundness * denominator,
                    np.rint(soundness * denominator),
                )
                and np.allclose(
                    completeness * denominator,
                    np.rint(completeness * denominator),
                ),
                f"{key} {split} pure-response rates are not prompt fractions",
            )

        full_soundness = np.asarray(
            [row["test_full_domain_soundness"] for row in rows]
        )
        full_completeness = np.asarray(
            [row["test_full_domain_completeness"] for row in rows]
        )
        test_pure_soundness = np.asarray(
            [row["test"]["pure_response_soundness"] for row in rows]
        )
        test_pure_completeness = np.asarray(
            [row["test"]["pure_response_completeness"] for row in rows]
        )
        require(
            np.all(full_soundness <= test_pure_soundness + 1e-12)
            and np.all(full_completeness <= test_pure_completeness + 1e-12),
            f"{key} full-domain guarantees exceed pure-endpoint guarantees",
        )

    expected_pure_frontiers = {
        0.99: {
            "count_rule": (349, 13),
            "weighted_binary": (349, 31),
            "equal_weight": (349, 124),
            "learned_weight": (349, 213),
            "ablated_learned_weight": (349, 202),
            "best_single": (350, 164),
        },
        0.95: {
            "count_rule": (335, 62),
            "weighted_binary": (335, 171),
            "equal_weight": (336, 203),
            "learned_weight": (336, 282),
            "ablated_learned_weight": (335, 263),
            "best_single": (335, 277),
        },
    }
    for target, expected_by_key in expected_pure_frontiers.items():
        for key, expected_counts in expected_by_key.items():
            feasible = [
                row
                for row in results[key]["rows"]
                if row["test"]["pure_response_soundness"]
                >= target - 1e-8
            ]
            best = max(
                feasible,
                key=lambda row: (
                    row["test"]["pure_response_completeness"],
                    row["test"]["pure_response_soundness"],
                ),
            )
            observed_counts = (
                int(round(352 * best["test"]["pure_response_soundness"])),
                int(round(352 * best["test"]["pure_response_completeness"])),
            )
            require(
                observed_counts == expected_counts,
                f"{key} descriptive pure frontier changed at {target:.0%}",
            )
    require(
        reconstruction["test_full_domain_soundness_rate"] == 1.0
        and np.isclose(
            reconstruction["test_full_domain_completeness_rate"],
            315 / 352,
        ),
        "pure-response reconstruction point changed",
    )

    tolerance = results["soundness_tolerance_sensitivity"]
    epsilon = np.asarray(tolerance["epsilon_harm"], dtype=float)
    require(tolerance["coverage_target"] == 0.95, "soundness target changed")
    require(tolerance["epsilon_good"] == 0.5, "completeness margin changed")
    require(
        len(epsilon) >= 12 and epsilon[0] == 0.0 and epsilon[-1] == 0.25,
        "exact soundness breakpoints changed",
    )
    require(
        tolerance["implicit_reject_all_when_no_setting_is_feasible"],
        "ranked-family reject-all sentinel is missing",
    )
    require(
        tolerance["implicit_accept_all_at_vacuous_endpoint"],
        "vacuous accept-all endpoint is missing",
    )
    require(np.all(np.diff(epsilon) > 0), "soundness breakpoints are not sorted")
    expected_at_zero = {
        "count_rule": 0.0,
        "weighted_binary": 0.0,
        "equal_weight": 0.045454545454545456,
        "learned_weight": 0.2869318181818182,
        "ablated_learned_weight": 0.20738636363636365,
        "best_single": 0.07670454545454546,
        "ranked_single_reviewer": 0.0,
        "ranked_prefix_unanimity": 0.0,
    }
    for key, expected in expected_at_zero.items():
        series = tolerance["series"][key]
        completeness = np.asarray(series["best_completeness"], dtype=float)
        soundness = np.asarray(series["achieved_soundness"], dtype=float)
        require(np.isclose(completeness[0], expected), f"{key} headline changed")
        require(np.all(np.diff(completeness) >= -1e-12), f"{key} is nonmonotone")
        require(np.all(soundness >= 0.95 - 1e-12), f"{key} misses 95% soundness")
        require(completeness[-1] == 1.0, f"{key} vacuous endpoint changed")

    check_topic_conditioned_results(
        results,
        reviewer_names,
        binary_features,
        subsets,
        validation_prompts,
    )
    check_train_selected_singleton_results(results)
    check_topic_threshold_budget_results(results)

    ranked_single = results["ranked_single_reviewer"]["rows"]
    ranked_prefix = results["ranked_prefix_unanimity"]["rows"]
    require(
        len(ranked_single) == 48 and len(ranked_prefix) == 48,
        "ranked binary family size changed",
    )
    require(
        ranked_single[0]["model"] == results["best_single"]["model"],
        "validation ranking no longer starts with the selected reviewer",
    )
    ranked_accuracy = np.asarray(
        [row["validation_choice_accuracy"] for row in ranked_single]
    )
    require(
        np.all(np.diff(ranked_accuracy) <= 1e-12),
        "validation reviewer ranking is not sorted",
    )
    single_soundness = np.asarray(
        [row["test_full_domain_soundness"] for row in ranked_single]
    )
    require(
        np.all(single_soundness == 0.0),
        "native singleton exact-soundness diagnostic changed",
    )
    prefix_soundness = np.asarray(
        [row["test_full_domain_soundness"] for row in ranked_prefix]
    )
    prefix_completeness = np.asarray(
        [
            row["test_approximate_full_domain"][
                "epsilon_good_completeness"
            ]
            for row in ranked_prefix
        ]
    )
    require(
        np.all(np.diff(prefix_soundness) >= -1e-12),
        "ranked-prefix soundness is nonmonotone",
    )
    require(
        np.all(np.diff(prefix_completeness) <= 1e-12),
        "ranked-prefix completeness is nonmonotone",
    )
    require(
        np.isclose(prefix_soundness[-1], 315 / 352)
        and np.isclose(prefix_completeness[-1], 2 / 352),
        "full ranked-prefix endpoint changed",
    )
    require(
        np.isclose(prefix_completeness[0], 332 / 352),
        "top-reviewer native-binary endpoint changed",
    )

    tradeoff = results["ranked_single_reviewer"]["approximate_tradeoff"]
    require(
        {
            "joint_certified_fraction",
            "soundness_condition",
            "completeness_condition",
            "boundary",
        }
        <= set(tradeoff["definitions"]),
        "ranked-singleton approximate definitions are incomplete",
    )
    harm_grid = np.asarray(tradeoff["epsilon_harm_grid"], dtype=float)
    good_grid = np.asarray(tradeoff["epsilon_good_grid"], dtype=float)
    require(
        harm_grid.shape == (101,)
        and np.allclose(harm_grid, np.linspace(0.0, 0.25, 101)),
        "ranked-singleton harm grid changed",
    )
    require(
        good_grid.shape == (151,)
        and np.allclose(good_grid, np.linspace(0.0, 0.75, 151)),
        "ranked-singleton good grid changed",
    )

    rank1_heatmap_record = tradeoff["rank1_heatmap"]
    rank1_heatmap = np.asarray(rank1_heatmap_record["values"], dtype=float)
    require(
        rank1_heatmap_record["reviewer_rank"] == 1
        and rank1_heatmap.shape == (151, 101),
        "rank-one approximate heatmap shape changed",
    )
    require(
        np.all(np.isfinite(rank1_heatmap))
        and np.all((rank1_heatmap >= 0.0) & (rank1_heatmap <= 1.0)),
        "rank-one approximate heatmap contains an invalid rate",
    )
    require(
        np.all(np.diff(rank1_heatmap, axis=0) >= -1e-12)
        and np.all(np.diff(rank1_heatmap, axis=1) >= -1e-12),
        "rank-one approximate heatmap is not coordinatewise monotone",
    )

    target_boundaries = tradeoff["minimum_epsilon_good_by_target"]
    for key, expected_target in (("0.10", 0.10), ("0.20", 0.20), ("0.30", 0.30)):
        boundary_record = target_boundaries[key]
        require(
            np.isclose(
                boundary_record["target_joint_fraction"],
                expected_target,
            ),
            f"ranked-singleton boundary target {key} changed",
        )
        by_rank = boundary_record["by_rank"]
        require(
            len(by_rank) == 48 and all(len(row) == 101 for row in by_rank),
            f"ranked-singleton boundary {key} has the wrong shape",
        )
        for rank, row in enumerate(by_rank, start=1):
            boundary = np.asarray(
                [np.nan if value is None else float(value) for value in row]
            )
            feasible = np.isfinite(boundary)
            require(
                np.all(np.diff(feasible.astype(int)) >= 0),
                f"rank {rank} boundary {key} becomes infeasible as harm relaxes",
            )
            require(
                np.all(
                    (boundary[feasible] >= good_grid[0] - 1e-12)
                    & (boundary[feasible] <= good_grid[-1] + 1e-12)
                ),
                f"rank {rank} boundary {key} leaves the good-margin grid",
            )
            require(
                np.all(np.diff(boundary[feasible]) <= 1e-12),
                f"rank {rank} boundary {key} rises as harm relaxes",
            )

    checkpoints = tradeoff["representative_checkpoints"]
    checkpoint_harm = np.asarray(checkpoints["epsilon_harm"], dtype=float)
    checkpoint_good = np.asarray(checkpoints["epsilon_good"], dtype=float)
    require(
        np.allclose(checkpoint_harm, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        and np.allclose(checkpoint_good, [0.0, 0.25, 0.50, 0.75]),
        "ranked-singleton representative checkpoint grid changed",
    )
    rates_by_rank = checkpoints["rates_by_rank"]
    require(len(rates_by_rank) == 48, "ranked-singleton checkpoints omit ranks")
    for row, ranked_row in zip(rates_by_rank, ranked_single):
        rates = np.asarray(row["joint_certified_fraction"], dtype=float)
        require(
            row["reviewer_rank"] == ranked_row["reviewer_rank"]
            and row["reviewer_index"] == ranked_row["reviewer_index"]
            and row["model"] == ranked_row["model"],
            "ranked-singleton checkpoint metadata disagrees with ranking",
        )
        require(
            rates.shape == (4, 6)
            and np.all(np.isfinite(rates))
            and np.all((rates >= 0.0) & (rates <= 1.0)),
            f"rank {row['reviewer_rank']} checkpoint rates are invalid",
        )
        require(
            np.all(np.diff(rates, axis=0) >= -1e-12)
            and np.all(np.diff(rates, axis=1) >= -1e-12),
            f"rank {row['reviewer_rank']} checkpoints are nonmonotone",
        )

    rank1_checkpoints = np.asarray(
        rates_by_rank[0]["joint_certified_fraction"], dtype=float
    )
    require(
        np.allclose(
            rank1_checkpoints[2, [1, 2, 4, 5]],
            np.asarray([31, 63, 120, 332]) / 352,
        ),
        "rank-one epsilon_good=0.50 checkpoints changed",
    )
    harm_indices = np.searchsorted(harm_grid, checkpoint_harm)
    good_indices = np.searchsorted(good_grid, checkpoint_good)
    require(
        np.allclose(
            rank1_heatmap[np.ix_(good_indices, harm_indices)],
            rank1_checkpoints,
        ),
        "rank-one heatmap and representative checkpoints disagree",
    )
    high_soundness = np.flatnonzero(prefix_soundness >= 0.75)
    require(
        len(high_soundness) > 0
        and ranked_prefix[high_soundness[0]]["panel_size"] == 42
        and np.isclose(prefix_soundness[high_soundness[0]], 271 / 352)
        and np.isclose(prefix_completeness[high_soundness[0]], 33 / 352),
        "ranked-prefix high-soundness overlay entry changed",
    )
    best_model = results["best_single"]["model"]
    selected_weight = results["learned_weight"]["weights"][best_model]
    require(np.isclose(selected_weight, 0.5192016283, atol=1e-9), "weight changed")


def check_figure_outputs() -> None:
    for name in REQUIRED_FIGURES:
        path = PLOTS / name
        require(path.is_file(), f"missing cited figure: {name}")
        require(path.read_bytes()[:5] == b"%PDF-", f"invalid PDF figure: {name}")
    for name in REQUIRED_PNGS:
        path = PLOTS / name
        require(path.is_file(), f"missing PNG figure: {name}")
        require(
            path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
            f"invalid PNG figure: {name}",
        )


def main() -> None:
    checks = (
        ("pinned inputs", check_inputs),
        ("exact depth and error arrays", check_exact_results),
        ("alignment and oracle rules", check_alignment_and_oracles),
        ("held-out weighted rules", check_cardinal_results),
        ("figure outputs", check_figure_outputs),
    )
    for label, function in checks:
        function()
        print(f"[ok] {label}")
    print("All one-step experiment checks passed.")


if __name__ == "__main__":
    main()
