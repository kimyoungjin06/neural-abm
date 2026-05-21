from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neural_abm.evidence_matrix import (
    EvidenceManifest,
    MatrixCase,
    MatrixVariant,
    ToyRunHandler,
    build_effect_summary,
    build_pairwise_effect_summary,
    case_ceiling_run_fields,
    ceiling_gap,
    ceiling_stability_run_fields,
    direction_effect,
    load_manifest,
    precommitment_trajectory_run_fields,
    run_matrix,
    summary_stats,
    time_to_ceiling,
)


@dataclass(frozen=True)
class FakeConfig:
    raw: dict

    def model_dump(self, mode: str = "json") -> dict:
        assert mode == "json"
        return self.raw


def fake_loader(path: Path) -> FakeConfig:
    return FakeConfig(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_toy24_basin_learned_credit_replay_floor_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_learned_credit_replay_floor_quick.yaml")
    )

    assert manifest.label == "toy24_basin_learned_credit_replay_floor_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        floor_variants = [
            variant
            for variant in case.variants
            if "floor50" in variant.name and variant.group == "nabm"
        ]
        assert len(floor_variants) == 1
        updates = floor_variants[0].updates
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_selection"
            ]
            == "confident_agreement"
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_min_selected_rate"
            ]
            == pytest.approx(0.5)
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_floor_source"
            ]
            == "prototype_abs"
        )


def test_toy24_basin_learned_credit_replay_curriculum_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_learned_credit_replay_curriculum_quick.yaml"
        )
    )

    assert manifest.label == "toy24_basin_learned_credit_replay_curriculum_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [curriculum] = [
            variant
            for variant in case.variants
            if "curriculum_floor50_d30" in variant.name and variant.group == "nabm"
        ]
        updates = curriculum.updates
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_selection"
            ]
            == "confident_agreement"
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_min_selected_rate"
            ]
            == pytest.approx(0.5)
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_floor_schedule"
            ]
            == "linear_decay"
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_floor_start_rate"
            ]
            == pytest.approx(1.0)
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_floor_decay_epochs"
            ]
            == 30
        )


def test_toy24_basin_learned_credit_soft_replay_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_learned_credit_soft_replay_quick.yaml"
        )
    )

    assert manifest.label == "toy24_basin_learned_credit_soft_replay_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [soft] = [
            variant
            for variant in case.variants
            if "soft_min50" in variant.name and variant.group == "nabm"
        ]
        updates = soft.updates
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_selection"
            ]
            == "confident_agreement"
        )
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "soft_attention"
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_soft_min_weight"
            ]
            == pytest.approx(0.5)
        )
        assert (
            updates[
                "model.policy.domain.basin_credit."
                "learned_credit_replay_soft_disagreement_weight"
            ]
            == pytest.approx(0.25)
        )


def test_toy24_basin_learned_credit_weight_scorer_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_learned_credit_weight_scorer_quick.yaml"
        )
    )

    assert manifest.label == "toy24_basin_learned_credit_weight_scorer_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [scorer] = [
            variant
            for variant in case.variants
            if variant.name == "learned_candidate_context_weight_scorer_replay"
        ]
        assert scorer.group == "nabm"
        updates = scorer.updates
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "learned_weight"
        )
        assert (
            "learned_credit_replay_weight_model_path"
            in " ".join(updates.keys())
        )


def test_toy24_basin_future_motion_weight_scorer_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_learned_credit_future_motion_weight_scorer_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy24_basin_learned_credit_future_motion_weight_scorer_quick"
    )
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [scorer] = [
            variant
            for variant in case.variants
            if variant.name
            == "learned_candidate_context_future_motion_weight_scorer_replay"
        ]
        assert scorer.group == "nabm"
        updates = scorer.updates
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "learned_weight"
        )
        model_path = updates[
            "model.policy.domain.basin_credit."
            "learned_credit_replay_weight_model_path"
        ]
        assert "future_motion_h5_q90" in str(model_path)


def test_toy24_basin_direction_pressure_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_direction_pressure_quick.yaml")
    )

    assert manifest.label == "toy24_basin_direction_pressure_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [candidate] = [
            variant
            for variant in case.variants
            if variant.name
            == "learned_candidate_context_direction_pressure_scorer_replay"
        ]
        assert candidate.group == "nabm"
        updates = candidate.updates
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "learned_weight"
        )
        model_path = updates[
            "model.policy.domain.basin_credit."
            "learned_credit_replay_weight_model_path"
        ]
        assert "replay_pressure_scorer_h5_q99" in str(model_path)


def test_toy24_basin_pairwise_direction_pressure_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_pairwise_direction_pressure_quick.yaml")
    )

    assert manifest.label == "toy24_basin_pairwise_direction_pressure_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [candidate] = [
            variant
            for variant in case.variants
            if variant.name == "learned_pairwise_direction_pressure_scorer_replay"
        ]
        assert candidate.group == "nabm"
        updates = candidate.updates
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "learned_weight"
        )
        direction_model_path = updates[
            "model.policy.domain.basin_credit.learned_credit_model_path"
        ]
        pressure_model_path = updates[
            "model.policy.domain.basin_credit."
            "learned_credit_replay_weight_model_path"
        ]
        assert "phase_critic_pairwise_direction" in str(direction_model_path)
        assert "replay_pressure_scorer_h5_q99" in str(pressure_model_path)


def test_toy24_basin_future_outcome_direction_pressure_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_future_outcome_direction_pressure_quick.yaml"
        )
    )

    assert manifest.label == "toy24_basin_future_outcome_direction_pressure_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        [candidate] = [
            variant
            for variant in case.variants
            if variant.name
            == "learned_future_outcome_direction_pressure_scorer_replay"
        ]
        assert candidate.group == "nabm"
        updates = candidate.updates
        assert (
            updates[
                "model.policy.domain.basin_credit.learned_credit_replay_mode"
            ]
            == "learned_weight"
        )
        direction_model_path = updates[
            "model.policy.domain.basin_credit.learned_credit_model_path"
        ]
        pressure_model_path = updates[
            "model.policy.domain.basin_credit."
            "learned_credit_replay_weight_model_path"
        ]
        assert "phase_critic_future_outcome_direction" in str(direction_model_path)
        assert "replay_pressure_scorer_h5_q99" in str(pressure_model_path)


def test_toy24_revision_operator_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_revision_operator_quick.yaml")
    )

    assert manifest.label == "toy24_revision_operator_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        assert case.nabm_group == "revision"
        [candidate] = [
            variant
            for variant in case.variants
            if variant.group == "revision"
        ]
        assert candidate.name == "revision_operator_mixed_objective_basin_w0p5_0p5_h1"
        updates = candidate.updates
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert (
            updates["model.coordination.revision_operator_source"]
            == "policy_probability"
        )
        assert (
            updates["model.policy.domain.basin_credit.objective_weight"]
            == pytest.approx(0.5)
        )
        assert (
            updates["model.policy.domain.basin_credit.individual_weight"]
            == pytest.approx(0.0)
        )
        assert (
            updates["model.policy.domain.basin_credit.basin_weight"]
            == pytest.approx(0.5)
        )


def test_toy24_revision_operator_controls_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_revision_operator_controls_quick.yaml")
    )

    assert manifest.label == "toy24_revision_operator_controls_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        assert case.nabm_group == "control"
        variants = {variant.name: variant for variant in case.variants}
        assert variants[
            "revision_operator_mixed_objective_basin_w0p5_0p5_h1"
        ].group == "diagnostic"
        assert variants["revision_operator_commitment_hysteresis"].group == "control"
        assert variants["revision_operator_terminal_argmax_k1"].group == "control"
        assert variants["revision_operator_terminal_argmax_k5"].group == "control"
        commitment_updates = variants["revision_operator_commitment_hysteresis"].updates
        assert commitment_updates["model.coordination.commitment_enabled"] is True
        assert (
            commitment_updates[
                "model.coordination.commitment_min_policy_probability"
            ]
            == pytest.approx(0.9)
        )
        assert (
            variants["revision_operator_terminal_argmax_k1"].updates[
                "model.policy.decision.terminal_argmax_epochs"
            ]
            == 1
        )
        assert (
            variants["revision_operator_terminal_argmax_k5"].updates[
                "model.policy.decision.terminal_argmax_epochs"
            ]
            == 5
        )


def test_toy24_revision_operator_precommitment_controls_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_revision_operator_precommitment_controls_quick.yaml"
        )
    )

    assert manifest.label == "toy24_revision_operator_precommitment_controls_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        assert case.nabm_group == "precommitment_control"
        variants = {variant.name: variant for variant in case.variants}
        assert variants[
            "revision_operator_mixed_objective_basin_w0p5_0p5_h1"
        ].group == "diagnostic"
        assert variants["revision_operator_commitment_hysteresis"].group == "diagnostic"
        assert variants["revision_operator_precommitment_evidence"].group == (
            "precommitment_control"
        )
        assert variants[
            "revision_operator_precommitment_peer_evidence_w0p25"
        ].group == "precommitment_control"
        assert variants[
            "revision_operator_precommitment_peer_evidence_w0p5"
        ].group == "precommitment_control"
        assert variants[
            "revision_operator_precommitment_peer_evidence_w1p0"
        ].group == "precommitment_control"
        assert variants[
            "revision_operator_precommitment_commitment_hysteresis"
        ].group == "precommitment_control"
        precommitment_updates = variants[
            "revision_operator_precommitment_evidence"
        ].updates
        assert precommitment_updates[
            "model.coordination.precommitment_enabled"
        ] is True
        assert (
            precommitment_updates[
                "model.coordination.precommitment_min_policy_probability"
            ]
            == pytest.approx(0.75)
        )
        combined_updates = variants[
            "revision_operator_precommitment_commitment_hysteresis"
        ].updates
        assert combined_updates["model.coordination.precommitment_enabled"] is True
        assert combined_updates["model.coordination.commitment_enabled"] is True
        peer_updates = variants[
            "revision_operator_precommitment_peer_evidence_w0p5"
        ].updates
        assert (
            peer_updates[
                "model.coordination.precommitment_peer_evidence_enabled"
            ]
            is True
        )
        assert (
            peer_updates["model.coordination.precommitment_peer_evidence_weight"]
            == pytest.approx(0.5)
        )


def test_toy24_revision_operator_precommitment_peer_evidence_stability_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy24_revision_operator_precommitment_peer_evidence_stability_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy24_revision_operator_precommitment_peer_evidence_stability_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        assert case.nabm_group == "precommitment_candidate"
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == {
            "reputation_imitation",
            "revision_operator_mixed_objective_basin_w0p5_0p5_h1",
            "revision_operator_precommitment_evidence",
            "revision_operator_precommitment_peer_evidence_w1p0",
        }
        assert variants["reputation_imitation"].group == "baseline"
        assert (
            variants["revision_operator_mixed_objective_basin_w0p5_0p5_h1"].group
            == "diagnostic"
        )
        assert variants["revision_operator_precommitment_evidence"].group == (
            "diagnostic"
        )
        candidate = variants["revision_operator_precommitment_peer_evidence_w1p0"]
        assert candidate.group == "precommitment_candidate"
        assert (
            candidate.updates[
                "model.coordination.precommitment_peer_evidence_enabled"
            ]
            is True
        )
        assert (
            candidate.updates[
                "model.coordination.precommitment_peer_evidence_weight"
            ]
            == pytest.approx(1.0)
        )


def test_toy5_readiness_propagation_holdout_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy5_readiness_propagation_holdout_quick.yaml")
    )

    assert manifest.label == "toy5_readiness_propagation_holdout_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_propagation_holdout"
    assert case.primary_metric == "domain_cascade_size"
    assert case.direction == "maximize"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "readiness_holdout"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.5)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "complex_threshold",
        "neural_output_average",
        "neural_precommitment_evidence",
        "neural_readiness_propagation_w1p0",
    }
    assert variants["complex_threshold"].group == "diagnostic"
    assert variants["neural_output_average"].group == "baseline"
    assert variants["neural_precommitment_evidence"].group == "diagnostic"
    candidate = variants["neural_readiness_propagation_w1p0"]
    assert candidate.group == "readiness_holdout"
    assert candidate.updates["model.policy.rule"] == "neural_policy"
    assert candidate.updates["model.coordination.precommitment_enabled"] is True
    assert (
        candidate.updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ]
        is True
    )
    assert (
        candidate.updates["model.coordination.precommitment_peer_evidence_weight"]
        == pytest.approx(1.0)
    )


def test_toy5_readiness_propagation_hard_argmax_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy5_readiness_propagation_hard_argmax_quick.yaml")
    )

    assert manifest.label == "toy5_readiness_propagation_hard_argmax_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 30
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_propagation_hard_argmax"
    assert case.primary_metric == "domain_cascade_size"
    assert case.direction == "maximize"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "readiness_hard"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_argmax_output_average",
        "neural_argmax_precommitment_evidence",
        "neural_argmax_readiness_propagation_w1p0",
        "neural_argmax_readiness_direction_gated_w1p0",
    }
    baseline = variants["neural_argmax_output_average"]
    assert baseline.group == "baseline"
    assert baseline.updates["model.policy.decision.mode"] == "argmax"
    assert baseline.updates["model.policy.neural_update_backend"] == "tensor_batched"
    candidate = variants["neural_argmax_readiness_propagation_w1p0"]
    assert candidate.group == "readiness_hard"
    assert candidate.updates["model.coordination.precommitment_enabled"] is True
    assert (
        candidate.updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ]
        is True
    )
    assert (
        candidate.updates["model.coordination.precommitment_peer_evidence_weight"]
        == pytest.approx(1.0)
    )
    direction_gated = variants["neural_argmax_readiness_direction_gated_w1p0"]
    assert direction_gated.group == "diagnostic"
    assert (
        direction_gated.updates[
            "model.coordination.precommitment_requires_direction"
        ]
        is True
    )


def test_toy5_readiness_augmented_direction_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_readiness_augmented_direction_hard_argmax_quick.yaml"
        )
    )

    assert manifest.label == "toy5_readiness_augmented_direction_hard_argmax_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 30
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_augmented_direction_hard_argmax"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "readiness_direction"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_argmax_output_average",
        "neural_argmax_precommitment_evidence",
        "neural_argmax_readiness_propagation_w1p0",
        "neural_argmax_local_threshold_direction_w1p0",
        "neural_argmax_readiness_augmented_direction_w1p0",
    }
    local = variants["neural_argmax_local_threshold_direction_w1p0"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    candidate = variants["neural_argmax_readiness_augmented_direction_w1p0"]
    assert candidate.group == "readiness_direction"
    assert candidate.updates["model.coordination.precommitment_enabled"] is True
    assert (
        candidate.updates["model.coordination.precommitment_requires_direction"]
        is True
    )
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ]
        is True
    )


def test_toy5_readiness_augmented_direction_frontier_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_readiness_augmented_direction_frontier_quick.yaml"
        )
    )

    assert manifest.label == "toy5_readiness_augmented_direction_frontier_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 30
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_augmented_direction_frontier"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "readiness_frontier"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_argmax_output_average",
        "neural_argmax_precommitment_evidence",
        "neural_argmax_readiness_propagation_w1p0",
        "neural_argmax_local_threshold_direction_w1p0",
        "neural_argmax_readiness_augmented_direction_w1p0",
    }
    baseline = variants["neural_argmax_output_average"]
    assert baseline.group == "baseline"
    assert baseline.updates["domain.environment.homogeneous_threshold"] == (
        pytest.approx(0.75)
    )
    local = variants["neural_argmax_local_threshold_direction_w1p0"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    assert local.updates["model.coordination.precommitment_evidence_decay"] == (
        pytest.approx(0.5)
    )
    candidate = variants["neural_argmax_readiness_augmented_direction_w1p0"]
    assert candidate.group == "readiness_frontier"
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )
    assert (
        candidate.updates["model.coordination.precommitment_peer_evidence_weight"]
        == pytest.approx(1.0)
    )


def test_toy5_readiness_augmented_direction_frontier_stability_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_readiness_augmented_direction_frontier_stability.yaml"
        )
    )

    assert manifest.label == "toy5_readiness_augmented_direction_frontier_stability"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 30
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_augmented_direction_frontier_stability"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "readiness_frontier"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_argmax_output_average",
        "neural_argmax_precommitment_evidence",
        "neural_argmax_readiness_propagation_w1p0",
        "neural_argmax_local_threshold_direction_w1p0",
        "neural_argmax_readiness_augmented_direction_w1p0",
    }
    baseline = variants["neural_argmax_output_average"]
    assert baseline.group == "baseline"
    assert baseline.updates["domain.environment.homogeneous_threshold"] == (
        pytest.approx(0.75)
    )
    local = variants["neural_argmax_local_threshold_direction_w1p0"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    assert local.updates["model.coordination.precommitment_evidence_decay"] == (
        pytest.approx(0.5)
    )
    candidate = variants["neural_argmax_readiness_augmented_direction_w1p0"]
    assert candidate.group == "readiness_frontier"
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )
    assert (
        candidate.updates["model.coordination.precommitment_peer_evidence_weight"]
        == pytest.approx(1.0)
    )


def test_toy5_readiness_direction_control_stress_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy5_readiness_direction_control_stress.yaml")
    )

    assert manifest.label == "toy5_readiness_direction_control_stress"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 20
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_readiness_direction_control_stress"
    assert case.primary_metric == "domain_non_adoption_rate"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "direction_control"
    assert case.ceiling_metric == "domain_non_adoption_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "complex_threshold_no_seed_control",
        "non_directional_readiness_self_excitation",
        "local_threshold_direction_control",
        "readiness_augmented_direction_control_w1p0",
    }
    baseline = variants["complex_threshold_no_seed_control"]
    assert baseline.group == "baseline"
    assert baseline.updates["model.policy.rule"] == "complex_threshold"
    assert baseline.updates["domain.environment.initial_action_fraction"] == (
        pytest.approx(0.0)
    )
    assert baseline.updates["domain.environment.homogeneous_threshold"] == (
        pytest.approx(0.25)
    )
    non_directional = variants["non_directional_readiness_self_excitation"]
    assert non_directional.group == "diagnostic"
    assert (
        non_directional.updates[
            "model.coordination.precommitment_requires_direction"
        ]
        is False
    )
    assert (
        non_directional.updates[
            "model.coordination.precommitment_min_policy_probability"
        ]
        == pytest.approx(0.0)
    )
    local = variants["local_threshold_direction_control"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    candidate = variants["readiness_augmented_direction_control_w1p0"]
    assert candidate.group == "direction_control"
    assert candidate.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )


def test_toy5_neural_readiness_direction_control_stress_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy5_neural_readiness_direction_control_stress.yaml")
    )

    assert manifest.label == "toy5_neural_readiness_direction_control_stress"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 20
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_neural_readiness_direction_control_stress"
    assert case.primary_metric == "domain_non_adoption_rate"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "direction_control"
    assert case.ceiling_metric == "domain_non_adoption_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_prior_no_seed_control",
        "neural_non_directional_readiness_self_excitation",
        "neural_local_threshold_direction_control",
        "neural_readiness_augmented_direction_control_w1p0",
    }
    baseline = variants["neural_prior_no_seed_control"]
    assert baseline.group == "baseline"
    assert baseline.updates["model.policy.rule"] == "neural_policy"
    assert baseline.updates["model.policy.learning_enabled"] is False
    assert baseline.updates["model.policy.neural_update_backend"] == "tensor_batched"
    assert baseline.updates["model.policy.decision.mode"] == "argmax"
    assert baseline.updates["model.agents.policy_prior_action_probability"] == (
        pytest.approx(0.0)
    )
    assert baseline.updates["domain.environment.initial_action_fraction"] == (
        pytest.approx(0.0)
    )
    non_directional = variants["neural_non_directional_readiness_self_excitation"]
    assert non_directional.group == "diagnostic"
    assert (
        non_directional.updates[
            "model.coordination.precommitment_requires_direction"
        ]
        is False
    )
    local = variants["neural_local_threshold_direction_control"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    candidate = variants["neural_readiness_augmented_direction_control_w1p0"]
    assert candidate.group == "direction_control"
    assert candidate.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )


def test_toy5_neural_threshold_target_direction_control_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_direction_control_stress.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_direction_control_stress"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 20
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_neural_threshold_target_direction_control_stress"
    assert case.primary_metric == "domain_non_adoption_rate"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "direction_control"
    assert case.ceiling_metric == "domain_non_adoption_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_threshold_target_no_seed_control",
        "neural_threshold_target_non_directional_readiness_self_excitation",
        "neural_threshold_target_local_direction_control",
        "neural_threshold_target_augmented_direction_control_w1p0",
    }
    baseline = variants["neural_threshold_target_no_seed_control"]
    assert baseline.group == "baseline"
    assert baseline.updates["model.policy.rule"] == "neural_policy"
    assert baseline.updates["model.policy.learning_enabled"] is True
    assert (
        baseline.updates["model.policy.domain.local_update_rule"]
        == "threshold_target"
    )
    assert baseline.updates["model.agents.policy_prior_action_probability"] == (
        pytest.approx(0.0)
    )
    non_directional = variants[
        "neural_threshold_target_non_directional_readiness_self_excitation"
    ]
    assert non_directional.group == "diagnostic"
    assert (
        non_directional.updates[
            "model.coordination.precommitment_requires_direction"
        ]
        is False
    )
    local = variants["neural_threshold_target_local_direction_control"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )
    candidate = variants["neural_threshold_target_augmented_direction_control_w1p0"]
    assert candidate.group == "direction_control"
    assert candidate.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(1.0)
    )


def test_toy5_neural_threshold_target_frontier_stability_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_frontier_stability.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_frontier_stability"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.toy == "toy5"
    assert case.name == "toy5_neural_threshold_target_frontier_stability"
    assert case.primary_metric == "domain_cascade_size"
    assert case.baseline_group == "baseline"
    assert case.nabm_group == "threshold_frontier"
    assert case.ceiling_metric == "action_rate"
    assert case.ceiling_value == pytest.approx(0.95)
    assert case.ceiling_tolerance == pytest.approx(0.0)
    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "neural_threshold_target_output_average",
        "neural_threshold_target_readiness_propagation_w1p0",
        "neural_threshold_target_local_direction_w1p0",
        "neural_threshold_target_readiness_augmented_w1p0",
        "neural_threshold_target_exposure_anchored_w2p0",
    }
    baseline = variants["neural_threshold_target_output_average"]
    assert baseline.group == "baseline"
    assert baseline.updates["model.policy.rule"] == "neural_policy"
    assert baseline.updates["model.policy.learning_enabled"] is True
    assert (
        baseline.updates["model.policy.domain.local_update_rule"]
        == "threshold_target"
    )
    assert baseline.updates["model.agents.policy_prior_action_probability"] == (
        pytest.approx(0.49)
    )

    non_directional = variants["neural_threshold_target_readiness_propagation_w1p0"]
    assert non_directional.group == "diagnostic"
    assert (
        non_directional.updates[
            "model.coordination.precommitment_requires_direction"
        ]
        is False
    )

    local = variants["neural_threshold_target_local_direction_w1p0"]
    assert local.group == "diagnostic"
    assert local.updates["model.coordination.precommitment_requires_direction"] is True
    assert (
        local.updates["model.coordination.precommitment_direction_source"]
        == "local_threshold"
    )

    augmented = variants["neural_threshold_target_readiness_augmented_w1p0"]
    assert augmented.group == "diagnostic"
    assert (
        augmented.updates["model.coordination.precommitment_direction_source"]
        == "readiness_augmented_threshold"
    )

    candidate = variants["neural_threshold_target_exposure_anchored_w2p0"]
    assert candidate.group == "threshold_frontier"
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )
    assert (
        candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(2.0)
    )
    assert (
        candidate.updates["model.coordination.precommitment_peer_evidence_weight"]
        == pytest.approx(1.0)
    )


def test_toy5_neural_threshold_target_combined_manifest_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_safety_frontier_combined.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_safety_frontier_combined"
    assert manifest.seeds == tuple(range(1, 11))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 2
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_target_no_seed_safety",
        "toy5_threshold_target_seeded_frontier_spread",
    }

    safety = cases["toy5_threshold_target_no_seed_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == "directional_threshold_target"
    assert safety.ceiling_metric == "domain_non_adoption_rate"
    assert safety.ceiling_value == pytest.approx(0.95)
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_target_no_seed_exposure_anchored_prior0p49"
    ]
    assert safety_candidate.group == "directional_threshold_target"
    assert (
        safety_candidate.updates["model.agents.policy_prior_action_probability"]
        == pytest.approx(0.49)
    )
    assert (
        safety_candidate.updates["domain.environment.initial_action_fraction"]
        == pytest.approx(0.0)
    )
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_min_policy_probability"
        ]
        == pytest.approx(0.0)
    )
    assert (
        safety_candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )

    frontier = cases["toy5_threshold_target_seeded_frontier_spread"]
    assert frontier.epochs is None
    assert frontier.primary_metric == "domain_cascade_size"
    assert frontier.nabm_group == "directional_threshold_target"
    assert frontier.ceiling_metric == "action_rate"
    assert frontier.ceiling_value == pytest.approx(0.95)
    frontier_variants = {variant.name: variant for variant in frontier.variants}
    frontier_candidate = frontier_variants[
        "neural_threshold_target_frontier_exposure_anchored_w2p0"
    ]
    assert frontier_candidate.group == "directional_threshold_target"
    assert (
        frontier_candidate.updates["domain.environment.initial_action_fraction"]
        == pytest.approx(0.01)
    )
    assert (
        frontier_candidate.updates[
            "model.coordination.precommitment_min_policy_probability"
        ]
        == pytest.approx(0.5)
    )
    assert (
        frontier_candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(2.0)
    )
    assert (
        frontier_candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )


def test_toy5_neural_threshold_target_structural_robustness_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_structural_robustness_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy5_neural_threshold_target_structural_robustness_quick"
    )
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 3
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_target_no_seed_heterogeneous_safety",
        "toy5_threshold_target_random_seed_frontier_spread",
        "toy5_threshold_target_heterogeneous_frontier_spread",
    }

    safety = cases["toy5_threshold_target_no_seed_heterogeneous_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == "directional_threshold_target_robust"
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_target_no_seed_heterogeneous_exposure_anchor"
    ]
    assert safety_candidate.group == "directional_threshold_target_robust"
    assert (
        safety_candidate.updates["domain.environment.threshold_mode"]
        == "heterogeneous"
    )
    assert (
        safety_candidate.updates["domain.environment.initial_action_fraction"]
        == pytest.approx(0.0)
    )
    assert (
        safety_candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )

    random_frontier = cases["toy5_threshold_target_random_seed_frontier_spread"]
    assert random_frontier.primary_metric == "domain_cascade_size"
    assert random_frontier.nabm_group == "directional_threshold_target_robust"
    random_variants = {variant.name: variant for variant in random_frontier.variants}
    random_candidate = random_variants[
        "neural_threshold_target_random_seed_frontier_exposure_anchor"
    ]
    assert random_candidate.group == "directional_threshold_target_robust"
    assert random_candidate.updates["domain.environment.seed_selection"] == "random"
    assert (
        random_candidate.updates["domain.environment.homogeneous_threshold"]
        == pytest.approx(0.75)
    )
    assert (
        random_candidate.updates[
            "model.coordination.precommitment_readiness_direction_weight"
        ]
        == pytest.approx(2.0)
    )

    heterogeneous = cases["toy5_threshold_target_heterogeneous_frontier_spread"]
    assert heterogeneous.primary_metric == "domain_cascade_size"
    assert heterogeneous.nabm_group == "directional_threshold_target_robust"
    heterogeneous_variants = {
        variant.name: variant for variant in heterogeneous.variants
    }
    heterogeneous_candidate = heterogeneous_variants[
        "neural_threshold_target_heterogeneous_frontier_exposure_anchor"
    ]
    assert heterogeneous_candidate.group == "directional_threshold_target_robust"
    assert (
        heterogeneous_candidate.updates["domain.environment.threshold_mode"]
        == "heterogeneous"
    )
    assert (
        heterogeneous_candidate.updates[
            "domain.environment.heterogeneous_threshold_high"
        ]
        == pytest.approx(0.85)
    )
    assert (
        heterogeneous_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_exposure_with_action_anchor"
    )


def test_toy5_neural_threshold_target_lattice_wavefront_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_lattice_wavefront_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_lattice_wavefront_quick"
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 2
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_target_wavefront_no_seed_safety",
        "toy5_threshold_target_lattice_wavefront_spread",
    }

    safety = cases["toy5_threshold_target_wavefront_no_seed_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == "directional_threshold_target_wavefront"
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_target_wavefront_no_seed_exposure_anchor"
    ]
    assert safety_candidate.group == "directional_threshold_target_wavefront"
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_min_policy_probability"
        ]
        == pytest.approx(0.0)
    )
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )
    assert (
        safety_candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )

    lattice = cases["toy5_threshold_target_lattice_wavefront_spread"]
    assert lattice.primary_metric == "domain_cascade_size"
    assert lattice.nabm_group == "directional_threshold_target_wavefront"
    assert lattice.ceiling_metric == "action_rate"
    assert lattice.ceiling_value == pytest.approx(0.95)
    lattice_variants = {variant.name: variant for variant in lattice.variants}
    mean_diagnostic = lattice_variants[
        "neural_threshold_target_lattice_mean_exposure_anchor"
    ]
    assert mean_diagnostic.group == "diagnostic"
    assert (
        mean_diagnostic.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "mean"
    )
    candidate = lattice_variants[
        "neural_threshold_target_lattice_max_wavefront_anchor"
    ]
    assert candidate.group == "directional_threshold_target_wavefront"
    assert candidate.updates["domain.graph.rewire_probability"] == pytest.approx(0.0)
    assert (
        candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )
    assert (
        candidate.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )


def test_toy5_neural_threshold_target_wavefront_topology_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_wavefront_topology_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_wavefront_topology_quick"
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 4
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_target_wavefront_topology_no_seed_safety",
        "toy5_threshold_target_lattice_k4_wavefront_spread",
        "toy5_threshold_target_lattice_k8_wavefront_spread",
        "toy5_threshold_target_rewired_p0p02_wavefront_spread",
    }

    safety = cases["toy5_threshold_target_wavefront_topology_no_seed_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == "directional_threshold_target_wavefront_topology"
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_target_topology_no_seed_exposure_anchor"
    ]
    assert safety_candidate.group == "directional_threshold_target_wavefront_topology"
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )

    k4 = cases["toy5_threshold_target_lattice_k4_wavefront_spread"]
    assert k4.nabm_group == "directional_threshold_target_wavefront_topology"
    k4_variants = {variant.name: variant for variant in k4.variants}
    k4_mean = k4_variants["neural_threshold_target_lattice_k4_mean_exposure_anchor"]
    assert k4_mean.group == "diagnostic"
    assert k4_mean.updates["domain.graph.k"] == pytest.approx(4)
    assert (
        k4_mean.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "mean"
    )
    k4_candidate = k4_variants[
        "neural_threshold_target_lattice_k4_max_wavefront_anchor"
    ]
    assert k4_candidate.group == "directional_threshold_target_wavefront_topology"
    assert k4_candidate.updates["domain.graph.k"] == pytest.approx(4)
    assert k4_candidate.updates["domain.graph.rewire_probability"] == pytest.approx(
        0.0
    )
    assert (
        k4_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )

    k8 = cases["toy5_threshold_target_lattice_k8_wavefront_spread"]
    k8_variants = {variant.name: variant for variant in k8.variants}
    k8_candidate = k8_variants[
        "neural_threshold_target_lattice_k8_max_wavefront_anchor"
    ]
    assert k8_candidate.group == "directional_threshold_target_wavefront_topology"
    assert k8_candidate.updates["domain.graph.k"] == pytest.approx(8)
    assert k8_candidate.updates["domain.graph.rewire_probability"] == pytest.approx(
        0.0
    )

    rewired = cases["toy5_threshold_target_rewired_p0p02_wavefront_spread"]
    rewired_variants = {variant.name: variant for variant in rewired.variants}
    rewired_candidate = rewired_variants[
        "neural_threshold_target_rewired_p0p02_max_wavefront_anchor"
    ]
    assert rewired_candidate.group == "directional_threshold_target_wavefront_topology"
    assert rewired_candidate.updates["domain.graph.k"] == pytest.approx(6)
    assert rewired_candidate.updates[
        "domain.graph.rewire_probability"
    ] == pytest.approx(0.02)
    assert (
        rewired_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_exposure_with_action_anchor"
    )


def test_toy5_neural_threshold_target_wavefront_stress_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_wavefront_stress_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_wavefront_stress_quick"
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 4
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety",
        "toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread",
        "toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread",
        "toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread",
    }

    safety = cases["toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == "directional_threshold_target_wavefront_stress"
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_target_stress_no_seed_heterogeneous_exposure_anchor"
    ]
    assert safety_candidate.group == "directional_threshold_target_wavefront_stress"
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )
    assert (
        safety_candidate.updates["domain.environment.threshold_mode"]
        == "heterogeneous"
    )

    k4 = cases["toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread"]
    assert k4.nabm_group == "directional_threshold_target_wavefront_stress"
    k4_variants = {variant.name: variant for variant in k4.variants}
    k4_candidate = k4_variants[
        "neural_threshold_target_lattice_k4_heterogeneous_h0p85_max_wavefront_anchor"
    ]
    assert k4_candidate.group == "directional_threshold_target_wavefront_stress"
    assert k4_candidate.updates["domain.graph.k"] == pytest.approx(4)
    assert k4_candidate.updates["domain.graph.rewire_probability"] == pytest.approx(
        0.0
    )
    assert (
        k4_candidate.updates["domain.environment.heterogeneous_threshold_high"]
        == pytest.approx(0.85)
    )
    assert (
        k4_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )

    k6_hard = cases[
        "toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread"
    ]
    k6_variants = {variant.name: variant for variant in k6_hard.variants}
    k6_candidate = k6_variants[
        "neural_threshold_target_lattice_k6_heterogeneous_h0p95_max_wavefront_anchor"
    ]
    assert k6_candidate.group == "directional_threshold_target_wavefront_stress"
    assert k6_candidate.updates["domain.graph.k"] == pytest.approx(6)
    assert (
        k6_candidate.updates["domain.environment.heterogeneous_threshold_high"]
        == pytest.approx(0.95)
    )

    rewired = cases[
        "toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread"
    ]
    rewired_variants = {variant.name: variant for variant in rewired.variants}
    rewired_candidate = rewired_variants[
        "neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_max_wavefront_anchor"
    ]
    assert rewired_candidate.group == "directional_threshold_target_wavefront_stress"
    assert rewired_candidate.updates["domain.graph.rewire_probability"] == pytest.approx(
        0.1
    )
    assert (
        rewired_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_exposure_with_action_anchor"
    )


def test_toy5_neural_threshold_target_threshold_aware_wavefront_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_threshold_aware_wavefront_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy5_neural_threshold_target_threshold_aware_wavefront_quick"
    )
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 4
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety",
        "toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread",
        "toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread",
        "toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread",
    }

    safety = cases["toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    assert safety.nabm_group == (
        "directional_threshold_target_threshold_aware_wavefront"
    )
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_aware_wavefront_no_seed_threshold_anchor"
    ]
    assert safety_candidate.group == (
        "directional_threshold_target_threshold_aware_wavefront"
    )
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_augmented_threshold_with_action_anchor"
    )
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_peer_readiness_aggregation"
        ]
        == "max"
    )

    k4 = cases["toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread"]
    k4_variants = {variant.name: variant for variant in k4.variants}
    k4_exposure = k4_variants[
        "neural_threshold_aware_lattice_k4_h0p85_exposure_anchor"
    ]
    assert k4_exposure.group == "diagnostic"
    assert (
        k4_exposure.updates["model.coordination.precommitment_direction_source"]
        == "readiness_exposure_with_action_anchor"
    )
    k4_candidate = k4_variants[
        "neural_threshold_aware_lattice_k4_h0p85_threshold_anchor"
    ]
    assert k4_candidate.group == (
        "directional_threshold_target_threshold_aware_wavefront"
    )
    assert k4_candidate.updates["domain.graph.k"] == pytest.approx(4)
    assert (
        k4_candidate.updates["domain.environment.heterogeneous_threshold_high"]
        == pytest.approx(0.85)
    )

    k6_hard = cases["toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread"]
    k6_variants = {variant.name: variant for variant in k6_hard.variants}
    k6_candidate = k6_variants[
        "neural_threshold_aware_lattice_k6_h0p95_threshold_anchor"
    ]
    assert k6_candidate.group == (
        "directional_threshold_target_threshold_aware_wavefront"
    )
    assert k6_candidate.updates["domain.graph.rewire_probability"] == pytest.approx(
        0.0
    )
    assert (
        k6_candidate.updates["domain.environment.heterogeneous_threshold_high"]
        == pytest.approx(0.95)
    )

    rewired = cases["toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread"]
    rewired_variants = {variant.name: variant for variant in rewired.variants}
    rewired_candidate = rewired_variants[
        "neural_threshold_aware_rewired_p0p10_h0p95_threshold_anchor"
    ]
    assert rewired_candidate.group == (
        "directional_threshold_target_threshold_aware_wavefront"
    )
    assert rewired_candidate.updates[
        "domain.graph.rewire_probability"
    ] == pytest.approx(0.1)
    assert (
        rewired_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_augmented_threshold_with_action_anchor"
    )


def test_toy5_neural_threshold_target_threshold_aware_grid_contract() -> None:
    manifest = load_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_threshold_aware_grid_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_threshold_aware_grid_quick"
    assert manifest.seeds == tuple(range(1, 6))
    assert manifest.epochs == 50
    assert len(manifest.cases) == 7
    cases = {case.name: case for case in manifest.cases}
    assert set(cases) == {
        "toy5_threshold_aware_grid_no_seed_heterogeneous_safety",
        "toy5_threshold_aware_grid_lattice_k4_h0p85_spread",
        "toy5_threshold_aware_grid_lattice_k4_h0p95_spread",
        "toy5_threshold_aware_grid_lattice_k6_h0p85_spread",
        "toy5_threshold_aware_grid_lattice_k6_h0p95_spread",
        "toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread",
        "toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread",
    }

    for case in cases.values():
        groups = {variant.group for variant in case.variants}
        assert groups == {
            "baseline",
            "negative_control",
            "directional_threshold_target_threshold_aware_grid",
        }
        assert case.nabm_group == "directional_threshold_target_threshold_aware_grid"

    safety = cases["toy5_threshold_aware_grid_no_seed_heterogeneous_safety"]
    assert safety.epochs == 20
    assert safety.primary_metric == "domain_non_adoption_rate"
    safety_variants = {variant.name: variant for variant in safety.variants}
    safety_candidate = safety_variants[
        "neural_threshold_aware_grid_no_seed_threshold_anchor"
    ]
    assert (
        safety_candidate.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_augmented_threshold_with_action_anchor"
    )

    k4_h95 = cases["toy5_threshold_aware_grid_lattice_k4_h0p95_spread"]
    k4_h95_variants = {variant.name: variant for variant in k4_h95.variants}
    k4_h95_candidate = k4_h95_variants[
        "neural_threshold_aware_grid_lattice_k4_h0p95_threshold_anchor"
    ]
    assert k4_h95_candidate.updates["domain.graph.k"] == pytest.approx(4)
    assert k4_h95_candidate.updates[
        "domain.graph.rewire_probability"
    ] == pytest.approx(0.0)
    assert (
        k4_h95_candidate.updates[
            "domain.environment.heterogeneous_threshold_high"
        ]
        == pytest.approx(0.95)
    )

    rewired_h85 = cases[
        "toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread"
    ]
    rewired_h85_variants = {variant.name: variant for variant in rewired_h85.variants}
    rewired_h85_negative = rewired_h85_variants[
        "neural_threshold_aware_grid_rewired_k6_p0p10_h0p85_exposure_anchor"
    ]
    assert rewired_h85_negative.group == "negative_control"
    assert rewired_h85_negative.updates["domain.graph.k"] == pytest.approx(6)
    assert rewired_h85_negative.updates[
        "domain.graph.rewire_probability"
    ] == pytest.approx(0.1)
    assert (
        rewired_h85_negative.updates[
            "model.coordination.precommitment_direction_source"
        ]
        == "readiness_exposure_with_action_anchor"
    )


def fake_runner(config: FakeConfig, config_path: Path) -> SimpleNamespace:
    del config_path
    run_dir = (
        Path(config.raw["run"]["output_dir"])
        / f"{config.raw['run']['name']}_seed{config.raw['run']['seed']:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if "score" in config.raw["domain"]:
        metrics["score"] = config.raw["domain"]["score"]
    with (run_dir / "aggregate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "score"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "score": 0.0})
        writer.writerow({"epoch": 1, "score": metrics.get("score", 0.0)})
    return SimpleNamespace(
        run_dir=run_dir,
        toy=config.raw["domain"]["toy"],
        domain_metrics=metrics,
    )


def write_fake_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "base",
                    "seed": 1,
                    "output_dir": str(tmp_path / "runs"),
                },
                "simulation": {"epochs": 1, "sync_mode": "synchronous"},
                "model": {
                    "coordination": {
                        "mixer": "none",
                        "peer_rule": "none",
                        "alpha": 0.0,
                        "threshold": 0.0,
                    }
                },
                "domain": {"toy": "toy1", "score": 0.0},
                "logging": {
                    "micro_state": False,
                    "interval": 1,
                    "aggregate_metrics": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def fake_manifest(tmp_path: Path, *, metric: str = "score") -> EvidenceManifest:
    base_config = write_fake_base_config(tmp_path)
    return EvidenceManifest(
        label="fake_matrix",
        seeds=(1, 2),
        epochs=3,
        config_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        cases=(
            MatrixCase(
                toy="toy1",
                name="fake_case",
                base_config=base_config,
                primary_metric=metric,
                direction="maximize",
                variants=(
                    MatrixVariant(
                        name="baseline",
                        group="baseline",
                        updates={"domain.score": 1.0},
                    ),
                    MatrixVariant(
                        name="nabm",
                        group="nabm",
                        updates={
                            "domain.score": 2.0,
                            "model.coordination.mixer": "output_average",
                            "model.coordination.peer_rule": "output_similarity",
                            "model.coordination.alpha": 0.25,
                        },
                    ),
                ),
            ),
        ),
    )


def test_summary_stats_mean_std_and_ci95() -> None:
    stats = summary_stats([1.0, 2.0, 3.0])

    assert stats.n == 3
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(1.0)
    assert stats.ci95 == pytest.approx(1.96 / (3**0.5))


def test_direction_effect_supports_maximize_and_minimize() -> None:
    assert direction_effect(baseline=2.0, nabm=5.0, direction="maximize") == 3.0
    assert direction_effect(baseline=5.0, nabm=2.0, direction="minimize") == 3.0


def test_build_effect_summary_matches_groups_by_seed_and_direction() -> None:
    rows = [
        {
            "label": "x",
            "case": "c",
            "toy": "toy3",
            "group": "baseline",
            "seed": 1,
            "primary_metric": "polarization",
            "metric_value": 0.8,
            "direction": "minimize",
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy3",
            "group": "baseline",
            "seed": 2,
            "primary_metric": "polarization",
            "metric_value": 0.6,
            "direction": "minimize",
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy3",
            "group": "nabm",
            "seed": 1,
            "primary_metric": "polarization",
            "metric_value": 0.5,
            "direction": "minimize",
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy3",
            "group": "nabm",
            "seed": 2,
            "primary_metric": "polarization",
            "metric_value": 0.4,
            "direction": "minimize",
        },
    ]

    [summary] = build_effect_summary(rows)

    assert summary["baseline_mean"] == pytest.approx(0.7)
    assert summary["nabm_mean"] == pytest.approx(0.45)
    assert summary["effect_mean"] == pytest.approx(0.25)
    assert summary["effect_n"] == 2


def test_build_pairwise_effect_summary_compares_each_baseline_variant() -> None:
    rows = [
        {
            "label": "x",
            "case": "c",
            "toy": "toy2",
            "variant": "weak_reference",
            "group": "baseline",
            "seed": 1,
            "primary_metric": "payoff",
            "metric_value": 1.0,
            "direction": "maximize",
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy2",
            "variant": "strong_reference",
            "group": "baseline",
            "seed": 1,
            "primary_metric": "payoff",
            "metric_value": 3.0,
            "direction": "maximize",
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy2",
            "variant": "neural",
            "group": "nabm",
            "seed": 1,
            "primary_metric": "payoff",
            "metric_value": 2.0,
            "direction": "maximize",
        },
    ]

    summaries = build_pairwise_effect_summary(rows)

    by_baseline = {row["baseline_variant"]: row for row in summaries}
    assert by_baseline["weak_reference"]["effect_mean"] == pytest.approx(1.0)
    assert by_baseline["strong_reference"]["effect_mean"] == pytest.approx(-1.0)


def test_build_pairwise_effect_summary_reports_ceiling_outcome() -> None:
    rows = [
        {
            "label": "x",
            "case": "c",
            "toy": "toy2",
            "variant": "reputation",
            "group": "baseline",
            "seed": 1,
            "primary_metric": "final_mean_payoff",
            "metric_value": 3.0,
            "direction": "maximize",
            "ceiling_metric": "mean_payoff",
            "ceiling_value": 3.0,
            "ceiling_tolerance": 0.001,
            "final_within_ceiling": True,
            "ever_reached_ceiling": True,
            "time_to_ceiling": 2,
            "ever_ceiling_final_miss": False,
            "late_flip_count_after_first_ceiling": 0.0,
            "late_flip_rate_after_first_ceiling": 0.0,
            "terminal_window_ceiling_rate": 1.0,
            "terminal_window_mean_ceiling_metric": 3.0,
        },
        {
            "label": "x",
            "case": "c",
            "toy": "toy2",
            "variant": "neural",
            "group": "nabm",
            "seed": 1,
            "primary_metric": "final_mean_payoff",
            "metric_value": 3.0,
            "direction": "maximize",
            "ceiling_metric": "mean_payoff",
            "ceiling_value": 3.0,
            "ceiling_tolerance": 0.001,
            "final_within_ceiling": True,
            "ever_reached_ceiling": True,
            "time_to_ceiling": 5,
            "ever_ceiling_final_miss": True,
            "late_flip_count_after_first_ceiling": 2.0,
            "late_flip_rate_after_first_ceiling": 0.02,
            "terminal_window_ceiling_rate": 0.8,
            "terminal_window_mean_ceiling_metric": 2.998,
        },
    ]

    [summary] = build_pairwise_effect_summary(rows)

    assert summary["effect_mean"] == pytest.approx(0.0)
    assert summary["baseline_final_ceiling_rate"] == pytest.approx(1.0)
    assert summary["nabm_final_ceiling_rate"] == pytest.approx(1.0)
    assert summary["baseline_time_to_ceiling_mean"] == pytest.approx(2.0)
    assert summary["nabm_time_to_ceiling_mean"] == pytest.approx(5.0)
    assert summary["time_to_ceiling_effect_mean"] == pytest.approx(-3.0)
    assert summary["nabm_ever_ceiling_final_miss_rate"] == pytest.approx(1.0)
    assert summary["nabm_late_flip_rate_after_first_ceiling_mean"] == pytest.approx(
        0.02
    )
    assert summary["nabm_terminal_window_ceiling_rate_mean"] == pytest.approx(0.8)
    assert summary["ceiling_outcome"] == "baseline_faster_to_ceiling"


def test_run_matrix_writes_run_and_effect_artifacts(tmp_path: Path) -> None:
    result = run_matrix(
        fake_manifest(tmp_path),
        results_dir=tmp_path / "results",
        handlers={"toy1": ToyRunHandler(loader=fake_loader, runner=fake_runner)},
    )

    assert result.runs_path.exists()
    assert result.effects_path.exists()
    assert result.pairwise_effects_path.exists()
    assert result.markdown_path.exists()
    assert len(result.run_rows) == 4
    assert result.effect_rows[0]["effect_mean"] == pytest.approx(1.0)
    assert result.pairwise_effect_rows[0]["effect_mean"] == pytest.approx(1.0)
    generated_config = yaml.safe_load(
        Path(result.run_rows[0]["config_path"]).read_text(encoding="utf-8")
    )
    assert generated_config["simulation"]["epochs"] == 3
    assert generated_config["run"]["output_dir"] == str(tmp_path / "runs")
    assert result.run_rows[0]["ceiling_metric"] == ""


def test_time_to_ceiling_reads_first_reaching_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "aggregate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mean_payoff"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "mean_payoff": 2.0})
        writer.writerow({"epoch": 2, "mean_payoff": 2.998})
        writer.writerow({"epoch": 3, "mean_payoff": 3.0})

    assert (
        time_to_ceiling(
            run_dir=run_dir,
            metric="mean_payoff",
            ceiling_value=3.0,
            tolerance=0.001,
        )
        == 3
    )
    assert ceiling_gap(metric_value=3.1, ceiling_value=3.0) == pytest.approx(0.0)


def test_ceiling_gap_uses_final_ceiling_metric_not_primary_metric(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "aggregate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mean_payoff"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "mean_payoff": 1.0})
        writer.writerow({"epoch": 1, "mean_payoff": 2.5})
    case = MatrixCase(
        toy="toy2",
        name="ceiling_case",
        base_config=tmp_path / "base.yaml",
        primary_metric="final_mean_payoff",
        direction="maximize",
        variants=(),
        ceiling_metric="mean_payoff",
        ceiling_value=3.0,
        ceiling_tolerance=0.001,
    )

    fields = case_ceiling_run_fields(case=case, run_dir=run_dir)

    assert fields["ceiling_metric_value"] == pytest.approx(2.5)
    assert fields["ceiling_gap"] == pytest.approx(0.5)
    assert fields["final_within_ceiling"] is False
    assert fields["ever_reached_ceiling"] is False
    assert fields["ever_ceiling_final_miss"] is False
    assert fields["terminal_window_ceiling_rate"] == pytest.approx(0.0)


def test_ceiling_stability_fields_report_late_flip_after_first_ceiling(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "aggregate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "mean_payoff", "action_flip_count", "action_flip_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "mean_payoff": 1.0,
                "action_flip_count": 0,
                "action_flip_rate": 0.0,
            }
        )
        writer.writerow(
            {
                "epoch": 1,
                "mean_payoff": 2.0,
                "action_flip_count": 0,
                "action_flip_rate": 0.0,
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "mean_payoff": 2.1,
                "action_flip_count": 1,
                "action_flip_rate": 0.25,
            }
        )
        writer.writerow(
            {
                "epoch": 3,
                "mean_payoff": 1.9,
                "action_flip_count": 2,
                "action_flip_rate": 0.5,
            }
        )
        writer.writerow(
            {
                "epoch": 4,
                "mean_payoff": 2.2,
                "action_flip_count": 0,
                "action_flip_rate": 0.0,
            }
        )
        writer.writerow(
            {
                "epoch": 5,
                "mean_payoff": 1.95,
                "action_flip_count": 1,
                "action_flip_rate": 0.25,
            }
        )

    fields = ceiling_stability_run_fields(
        run_dir=run_dir,
        metric="mean_payoff",
        ceiling_value=2.0,
        tolerance=0.0,
        time_to_ceiling_value=1,
        final_within_ceiling=False,
    )

    assert fields["ever_ceiling_final_miss"] is True
    assert fields["late_flip_count_after_first_ceiling"] == pytest.approx(4.0)
    assert fields["late_flip_rate_after_first_ceiling"] == pytest.approx(0.25)
    assert fields["terminal_window_size"] == 5
    assert fields["terminal_window_ceiling_rate"] == pytest.approx(0.6)
    assert fields["terminal_window_mean_ceiling_metric"] == pytest.approx(2.03)


def test_precommitment_trajectory_fields_read_final_aggregate_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fieldnames = [
        "epoch",
        "precommitment_first_ready_epoch",
        "precommitment_all_ready_epoch",
        "precommitment_first_forced_epoch",
        "precommitment_ready_to_forced_delay_mean",
        "precommitment_premature_exit_count",
        "precommitment_high_policy_rate",
        "precommitment_direction_score_mean",
        "precommitment_direction_score_positive_rate",
        "precommitment_direction_ok_rate",
        "precommitment_ready_largest_component_fraction",
        "precommitment_peer_evidence_enabled",
        "precommitment_peer_evidence_weight",
        "precommitment_peer_readiness_aggregation",
        "precommitment_peer_readiness_mean",
        "precommitment_peer_readiness_active_rate",
        "precommitment_peer_evidence_increment_mean",
    ]
    with (run_dir / "aggregate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "precommitment_first_ready_epoch": "",
                "precommitment_all_ready_epoch": "",
                "precommitment_first_forced_epoch": "",
                "precommitment_ready_to_forced_delay_mean": "",
                "precommitment_premature_exit_count": 0,
                "precommitment_high_policy_rate": 0.0,
                "precommitment_direction_score_mean": 0.0,
                "precommitment_direction_score_positive_rate": 0.0,
                "precommitment_direction_ok_rate": 0.0,
                "precommitment_ready_largest_component_fraction": 0.0,
                "precommitment_peer_evidence_enabled": False,
                "precommitment_peer_evidence_weight": 0.0,
                "precommitment_peer_readiness_aggregation": "mean",
                "precommitment_peer_readiness_mean": 0.0,
                "precommitment_peer_readiness_active_rate": 0.0,
                "precommitment_peer_evidence_increment_mean": 0.0,
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "precommitment_first_ready_epoch": 1,
                "precommitment_all_ready_epoch": 2,
                "precommitment_first_forced_epoch": 2,
                "precommitment_ready_to_forced_delay_mean": 1.0,
                "precommitment_premature_exit_count": 3,
                "precommitment_high_policy_rate": 0.7,
                "precommitment_direction_score_mean": 0.1,
                "precommitment_direction_score_positive_rate": 0.6,
                "precommitment_direction_ok_rate": 0.8,
                "precommitment_ready_largest_component_fraction": 0.9,
                "precommitment_peer_evidence_enabled": True,
                "precommitment_peer_evidence_weight": 0.5,
                "precommitment_peer_readiness_aggregation": "max",
                "precommitment_peer_readiness_mean": 0.25,
                "precommitment_peer_readiness_active_rate": 0.4,
                "precommitment_peer_evidence_increment_mean": 0.125,
            }
        )

    fields = precommitment_trajectory_run_fields(run_dir)

    assert fields["precommitment_first_ready_epoch"] == "1"
    assert fields["precommitment_all_ready_epoch"] == "2"
    assert fields["precommitment_first_forced_epoch"] == "2"
    assert fields["precommitment_ready_to_forced_delay_mean"] == "1.0"
    assert fields["precommitment_premature_exit_count"] == "3"
    assert fields["precommitment_high_policy_rate"] == "0.7"
    assert fields["precommitment_direction_score_mean"] == "0.1"
    assert fields["precommitment_direction_score_positive_rate"] == "0.6"
    assert fields["precommitment_direction_ok_rate"] == "0.8"
    assert fields["precommitment_ready_largest_component_fraction"] == "0.9"
    assert fields["precommitment_peer_evidence_enabled"] == "True"
    assert fields["precommitment_peer_evidence_weight"] == "0.5"
    assert fields["precommitment_peer_readiness_aggregation"] == "max"
    assert fields["precommitment_peer_readiness_mean"] == "0.25"
    assert fields["precommitment_peer_readiness_active_rate"] == "0.4"
    assert fields["precommitment_peer_evidence_increment_mean"] == "0.125"


def test_run_matrix_records_ceiling_fields(tmp_path: Path) -> None:
    manifest = fake_manifest(tmp_path)
    case = manifest.cases[0]
    manifest = EvidenceManifest(
        label=manifest.label,
        seeds=manifest.seeds,
        epochs=manifest.epochs,
        config_dir=manifest.config_dir,
        runs_dir=manifest.runs_dir,
        cases=(
            MatrixCase(
                toy=case.toy,
                name=case.name,
                base_config=case.base_config,
                primary_metric=case.primary_metric,
                direction=case.direction,
                variants=case.variants,
                ceiling_metric="score",
                ceiling_value=2.0,
                ceiling_tolerance=0.0,
            ),
        ),
    )

    result = run_matrix(
        manifest,
        results_dir=tmp_path / "results",
        handlers={"toy1": ToyRunHandler(loader=fake_loader, runner=fake_runner)},
    )

    by_variant = {row["variant"]: row for row in result.run_rows if row["seed"] == 1}
    assert by_variant["baseline"]["ceiling_gap"] == pytest.approx(1.0)
    assert by_variant["baseline"]["final_within_ceiling"] is False
    assert by_variant["baseline"]["time_to_ceiling"] == ""
    assert by_variant["nabm"]["ceiling_gap"] == pytest.approx(0.0)
    assert by_variant["nabm"]["final_within_ceiling"] is True
    assert by_variant["nabm"]["time_to_ceiling"] == 1
    assert result.pairwise_effect_rows[0]["ceiling_outcome"] in {
        "nabm_more_final_ceiling_hits",
        "baseline_more_final_ceiling_hits",
    }


def test_run_matrix_rejects_unknown_toy(tmp_path: Path) -> None:
    manifest = fake_manifest(tmp_path)
    bad_case = MatrixCase(
        toy="toy99",
        name="bad_case",
        base_config=manifest.cases[0].base_config,
        primary_metric="score",
        direction="maximize",
        variants=manifest.cases[0].variants,
    )
    bad_manifest = EvidenceManifest(
        label=manifest.label,
        seeds=manifest.seeds,
        epochs=manifest.epochs,
        config_dir=manifest.config_dir,
        runs_dir=manifest.runs_dir,
        cases=(bad_case,),
    )

    with pytest.raises(ValueError, match="Unknown evidence matrix toy"):
        run_matrix(bad_manifest, results_dir=tmp_path / "results", handlers={})


def test_run_matrix_rejects_missing_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing primary metric"):
        run_matrix(
            fake_manifest(tmp_path, metric="missing_score"),
            results_dir=tmp_path / "results",
            handlers={"toy1": ToyRunHandler(loader=fake_loader, runner=fake_runner)},
        )


def test_default_quick_manifest_is_loadable() -> None:
    manifest = load_manifest(Path("experiments/evidence/nabm_effect_matrix_quick.yaml"))

    assert manifest.label == "nabm_effect_matrix_quick"
    assert manifest.seeds == (1, 2, 3)
    assert {case.toy for case in manifest.cases} == {
        "toy1",
        "toy2",
        "toy3",
        "toy4",
        "toy5",
    }
    for case in manifest.cases:
        assert case.primary_metric
        assert case.direction in {"maximize", "minimize"}
        assert case.variants


def test_toy24_state_continuation_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_state_continuation_matrix_quick.yaml")
    )

    assert manifest.label == "toy24_state_continuation_matrix_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        variants = {variant.name: variant for variant in case.variants}
        assert "reputation_imitation" in variants
        assert "neural_material_output_average" in variants
        assert "neural_continuation_balanced" in variants
        assert "neural_continuation_social_heavy" in variants
        assert "neural_continuation_welfare_heavy" in variants
        assert variants["reputation_imitation"].group == "baseline"
        assert variants["neural_continuation_balanced"].group == "nabm"


def test_toy24_objective_family_v2_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_objective_family_v2_quick.yaml")
    )

    assert manifest.label == "toy24_objective_family_v2_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "material",
        "linear_balanced",
        "linear_welfare_heavy",
        "nonlinear_mild",
        "nonlinear_interaction",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"


def test_toy24_teacher_bootstrap_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_teacher_bootstrap_quick.yaml")
    )

    assert manifest.label == "toy24_teacher_bootstrap_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "nonlinear_interaction",
        "teacher_bootstrap_w0p5_e5_linear_welfare",
        "teacher_bootstrap_w1p0_e3_linear_welfare",
        "teacher_bootstrap_w0p5_e5_nonlinear_interaction",
        "teacher_bootstrap_w1p0_e3_nonlinear_interaction",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"
        for name, variant in variants.items():
            if name.startswith("teacher_bootstrap"):
                assert variant.updates["model.policy.domain.bootstrap.enabled"] is True
                assert (
                    variant.updates["model.policy.domain.bootstrap.decay"] == "linear"
                )


def test_toy24_decision_bootstrap_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_decision_bootstrap_quick.yaml")
    )

    assert manifest.label == "toy24_decision_bootstrap_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "target_bootstrap_w1p0_e3_linear_welfare",
        "decision_bootstrap_w1p0_e3_linear_welfare",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "target_decision_bootstrap_w1p0_e3_linear_welfare",
        "target_decision_bootstrap_w1p0_e5_linear_welfare",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"
        assert "nonlinear_interaction" not in variants
        for name, variant in variants.items():
            if name.startswith("decision_bootstrap"):
                assert (
                    variant.updates[
                        "model.policy.domain.bootstrap.decision_enabled"
                    ]
                    is True
                )
                assert (
                    variant.updates[
                        "model.policy.domain.bootstrap.decision_decay"
                    ]
                    == "linear"
                )
                assert "model.policy.domain.bootstrap.enabled" not in variant.updates
            if name.startswith("target_decision_bootstrap"):
                assert variant.updates["model.policy.domain.bootstrap.enabled"] is True
                assert (
                    variant.updates[
                        "model.policy.domain.bootstrap.decision_enabled"
                    ]
                    is True
                )


def test_toy24_policy_distill_bootstrap_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_policy_distill_bootstrap_quick.yaml")
    )

    assert manifest.label == "toy24_policy_distill_bootstrap_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "target_bootstrap_w1p0_e3_linear_welfare",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "distill_bootstrap_w1p0_e3_linear_welfare",
        "distill_bootstrap_w1p0_e5_linear_welfare",
        "target_distill_bootstrap_w1p0_e5_linear_welfare",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"
        assert "nonlinear_interaction" not in variants
        for name, variant in variants.items():
            if name.startswith("distill_bootstrap"):
                assert (
                    variant.updates["model.policy.domain.bootstrap.distill_enabled"]
                    is True
                )
                assert (
                    variant.updates["model.policy.domain.bootstrap.distill_decay"]
                    == "linear"
                )
                assert (
                    variant.updates["model.policy.domain.bootstrap.distill_loss"]
                    == "bce"
                )
                assert (
                    variant.updates["model.policy.domain.bootstrap.distill_scope"]
                    == "all"
                )
            if name.startswith("target_distill_bootstrap"):
                assert variant.updates["model.policy.domain.bootstrap.enabled"] is True
                assert (
                    variant.updates["model.policy.domain.bootstrap.distill_enabled"]
                    is True
                )
            if name.startswith("decision_bootstrap"):
                assert (
                    variant.updates[
                        "model.policy.domain.bootstrap.decision_enabled"
                    ]
                    is True
                )
                assert (
                    variant.updates[
                        "model.policy.domain.bootstrap.decision_decay"
                    ]
                    == "linear"
                )
                assert "model.policy.domain.bootstrap.enabled" not in variant.updates


def test_toy24_teacher_alignment_diagnostics_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_teacher_alignment_diagnostics_quick.yaml")
    )

    assert manifest.label == "toy24_teacher_alignment_diagnostics_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "distill_bootstrap_w1p0_e5_linear_welfare",
        "target_distill_bootstrap_w1p0_e5_linear_welfare",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"
        assert (
            variants["decision_bootstrap_w1p0_e5_linear_welfare"].updates[
                "model.policy.domain.bootstrap.decision_enabled"
            ]
            is True
        )
        assert (
            variants["distill_bootstrap_w1p0_e5_linear_welfare"].updates[
                "model.policy.domain.bootstrap.distill_enabled"
            ]
            is True
        )
        target_distill = variants["target_distill_bootstrap_w1p0_e5_linear_welfare"]
        assert target_distill.updates["model.policy.domain.bootstrap.enabled"] is True
        assert (
            target_distill.updates[
                "model.policy.domain.bootstrap.distill_enabled"
            ]
            is True
        )


def test_toy24_gated_distill_bootstrap_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_gated_distill_bootstrap_quick.yaml")
    )

    assert manifest.label == "toy24_gated_distill_bootstrap_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "distill_bootstrap_w1p0_e5_linear_welfare",
        "stable_distill_bootstrap_w1p0_e5_linear_welfare",
        "gated_distill_bootstrap_w1p0_e5_linear_welfare",
        "stable_gated_distill_bootstrap_w1p0_e5_linear_welfare",
        "target_stable_gated_distill_w1p0_e5_linear_welfare",
    }
    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"
        stable = variants["stable_distill_bootstrap_w1p0_e5_linear_welfare"]
        assert (
            stable.updates[
                "model.policy.domain.bootstrap.distill_stable_teacher_only"
            ]
            is True
        )
        gated = variants["gated_distill_bootstrap_w1p0_e5_linear_welfare"]
        assert (
            gated.updates[
                "model.policy.domain.bootstrap.distill_gradient_gate_enabled"
            ]
            is True
        )
        stable_gated = variants[
            "stable_gated_distill_bootstrap_w1p0_e5_linear_welfare"
        ]
        assert (
            stable_gated.updates[
                "model.policy.domain.bootstrap.distill_stable_teacher_only"
            ]
            is True
        )
        assert (
            stable_gated.updates[
                "model.policy.domain.bootstrap.distill_gradient_gate_enabled"
            ]
            is True
        )
        target_stable_gated = variants[
            "target_stable_gated_distill_w1p0_e5_linear_welfare"
        ]
        assert target_stable_gated.updates[
            "model.policy.domain.bootstrap.enabled"
        ] is True
        assert (
            target_stable_gated.updates[
                "model.policy.domain.bootstrap.distill_gradient_gate_enabled"
            ]
            is True
        )


def test_toy24_scaffolded_decision_replay_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_scaffolded_decision_replay_quick.yaml")
    )

    assert manifest.label == "toy24_scaffolded_decision_replay_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "decision_replay_bc_w1p0_e5_linear_welfare",
        "decision_replay_success_w1p0_e5_linear_welfare",
        "stable_decision_replay_success_w1p0_e5_linear_welfare",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        for name in expected_variants - {"reputation_imitation"}:
            assert variants[name].group == "nabm"

        decision = variants["decision_bootstrap_w1p0_e5_linear_welfare"]
        assert (
            decision.updates["model.policy.domain.bootstrap.decision_enabled"]
            is True
        )
        assert "model.policy.domain.bootstrap.replay_enabled" not in decision.updates

        bc_replay = variants["decision_replay_bc_w1p0_e5_linear_welfare"]
        assert (
            bc_replay.updates["model.policy.domain.bootstrap.replay_enabled"]
            is True
        )
        assert (
            bc_replay.updates[
                "model.policy.domain.bootstrap.replay_require_objective_agreement"
            ]
            is False
        )
        assert (
            bc_replay.updates[
                "model.policy.domain.bootstrap.replay_require_postsocial_alignment_improvement"
            ]
            is False
        )

        success_replay = variants["decision_replay_success_w1p0_e5_linear_welfare"]
        assert (
            success_replay.updates[
                "model.policy.domain.bootstrap.replay_require_objective_agreement"
            ]
            is True
        )
        assert (
            success_replay.updates[
                "model.policy.domain.bootstrap.replay_stable_teacher_only"
            ]
            is False
        )

        stable_success = variants[
            "stable_decision_replay_success_w1p0_e5_linear_welfare"
        ]
        assert (
            stable_success.updates[
                "model.policy.domain.bootstrap.replay_stable_teacher_only"
            ]
            is True
        )


def test_toy24_basin_credit_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_credit_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "basin_credit_w1p0_h1_prototype",
        "mixed_individual_basin_w0p5_0p5_h1",
    }
    ceilings = {
        "toy2": (3.0, 0.001),
        "toy4": (0.6, 0.005),
    }
    for case in manifest.cases:
        assert case.primary_metric == "final_mean_payoff"
        assert case.direction == "maximize"
        ceiling_value, ceiling_tolerance = ceilings[case.toy]
        assert case.ceiling_metric == "mean_payoff"
        assert case.ceiling_value == pytest.approx(ceiling_value)
        assert case.ceiling_tolerance == pytest.approx(ceiling_tolerance)
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        assert (
            variants["decision_bootstrap_w1p0_e5_linear_welfare"].group
            == "diagnostic"
        )
        pure = variants["basin_credit_w1p0_h1_prototype"]
        assert pure.group == "nabm"
        assert pure.updates["model.policy.domain.basin_credit.enabled"] is True
        assert (
            pure.updates["model.policy.domain.basin_credit.critic"]
            == "prototype_phase"
        )
        assert (
            pure.updates["model.policy.domain.basin_credit.credit_method"]
            == "one_step_ablation"
        )
        assert pure.updates["model.policy.domain.basin_credit.basin_weight"] == 1.0
        mixed = variants["mixed_individual_basin_w0p5_0p5_h1"]
        assert mixed.group == "nabm"
        assert (
            mixed.updates["model.policy.domain.basin_credit.individual_weight"]
            == 0.5
        )
        assert mixed.updates["model.policy.domain.basin_credit.basin_weight"] == 0.5


def test_toy24_basin_credit_objective_blend_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_credit_objective_blend_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_objective_blend_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "basin_credit_w1p0_h1_prototype",
        "mixed_individual_basin_w0p5_0p5_h1",
        "mixed_objective_basin_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1",
        "mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1",
        "mixed_objective_basin_directional_social_w0p5_0p5_h1",
    }
    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        assert variants["linear_welfare_heavy"].group == "baseline"
        assert (
            variants["decision_bootstrap_w1p0_e5_linear_welfare"].group
            == "diagnostic"
        )
        assert variants["basin_credit_w1p0_h1_prototype"].group == "nabm"
        material_mixed = variants["mixed_individual_basin_w0p5_0p5_h1"]
        assert material_mixed.group == "diagnostic"
        assert (
            material_mixed.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.0
        )
        assert (
            material_mixed.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.5
        )
        assert (
            material_mixed.updates["model.policy.domain.basin_credit.basin_weight"]
            == 0.5
        )
        objective_mixed = variants["mixed_objective_basin_w0p5_0p5_h1"]
        assert objective_mixed.group == "nabm"
        assert (
            objective_mixed.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            objective_mixed.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            objective_mixed.updates["model.policy.domain.basin_credit.basin_weight"]
            == 0.5
        )
        confidence_mixed = variants[
            "mixed_objective_basin_confidence_social_w0p5_0p5_h1"
        ]
        assert confidence_mixed.group == "nabm"
        assert (
            confidence_mixed.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            confidence_mixed.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            confidence_mixed.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            confidence_mixed.updates["model.coordination.confidence_weighting"]
            == "peer"
        )
        assert (
            confidence_mixed.updates["model.coordination.confidence_weight_floor"]
            == 0.0
        )
        assert (
            confidence_mixed.updates["model.coordination.confidence_weight_power"]
            == 1.0
        )
        confidence_floor = variants[
            "mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1"
        ]
        assert confidence_floor.group == "nabm"
        assert (
            confidence_floor.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            confidence_floor.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            confidence_floor.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            confidence_floor.updates["model.coordination.confidence_weighting"]
            == "peer"
        )
        assert (
            confidence_floor.updates["model.coordination.confidence_weight_floor"]
            == 0.5
        )
        assert (
            confidence_floor.updates["model.coordination.confidence_weight_power"]
            == 1.0
        )
        confidence_tail = variants[
            "mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1"
        ]
        assert confidence_tail.group == "nabm"
        assert (
            confidence_tail.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            confidence_tail.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            confidence_tail.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            confidence_tail.updates["model.coordination.confidence_weighting"]
            == "peer"
        )
        assert (
            confidence_tail.updates["model.coordination.confidence_weight_floor"]
            == 0.0
        )
        assert (
            confidence_tail.updates["model.coordination.confidence_weight_power"]
            == 1.0
        )
        assert (
            confidence_tail.updates["model.coordination.confidence_tail_floor"]
            == 0.5
        )
        assert (
            confidence_tail.updates[
                "model.coordination.confidence_tail_min_policy_rate"
            ]
            == 0.95
        )
        assert (
            confidence_tail.updates[
                "model.coordination.confidence_tail_min_action_rate"
            ]
            == 0.95
        )
        confidence_commitment = variants[
            "mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1"
        ]
        assert confidence_commitment.group == "nabm"
        assert (
            confidence_commitment.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            confidence_commitment.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            confidence_commitment.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            confidence_commitment.updates["model.coordination.confidence_weighting"]
            == "peer"
        )
        assert (
            confidence_commitment.updates["model.coordination.commitment_enabled"]
            is True
        )
        assert (
            confidence_commitment.updates[
                "model.coordination.commitment_min_policy_probability"
            ]
            == 0.9
        )
        assert (
            confidence_commitment.updates[
                "model.coordination.commitment_min_action_streak"
            ]
            == 2
        )
        assert (
            confidence_commitment.updates[
                "model.coordination.commitment_exit_policy_probability"
            ]
            == 0.75
        )
        confidence_precommitment = variants[
            "mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1"
        ]
        assert confidence_precommitment.group == "nabm"
        assert (
            confidence_precommitment.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            confidence_precommitment.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            confidence_precommitment.updates[
                "model.coordination.commitment_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment.updates[
                "model.coordination.precommitment_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment.updates[
                "model.coordination.precommitment_min_policy_probability"
            ]
            == 0.75
        )
        assert (
            confidence_precommitment.updates[
                "model.coordination.precommitment_min_evidence"
            ]
            == 1.5
        )
        assert (
            confidence_precommitment.updates[
                "model.coordination.precommitment_evidence_decay"
            ]
            == 0.8
        )
        confidence_precommitment_feedback = variants[
            "mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1"
        ]
        assert confidence_precommitment_feedback.group == "nabm"
        assert (
            confidence_precommitment_feedback.updates[
                "model.coordination.precommitment_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment_feedback.updates[
                "model.coordination.precommitment_decision_feedback_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment_feedback.updates[
                "model.coordination.precommitment_decision_feedback_weight"
            ]
            == 1.0
        )
        confidence_precommitment_social_feedback = variants[
            "mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1"
        ]
        assert confidence_precommitment_social_feedback.group == "nabm"
        assert (
            confidence_precommitment_social_feedback.updates[
                "model.coordination.precommitment_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment_social_feedback.updates[
                "model.coordination.precommitment_social_feedback_enabled"
            ]
            is True
        )
        assert (
            confidence_precommitment_social_feedback.updates[
                "model.coordination.precommitment_social_feedback_weight"
            ]
            == 1.0
        )
        directional_mixed = variants[
            "mixed_objective_basin_directional_social_w0p5_0p5_h1"
        ]
        assert directional_mixed.group == "nabm"
        assert (
            directional_mixed.updates[
                "model.policy.domain.basin_credit.objective_weight"
            ]
            == 0.5
        )
        assert (
            directional_mixed.updates[
                "model.policy.domain.basin_credit.individual_weight"
            ]
            == 0.0
        )
        assert (
            directional_mixed.updates[
                "model.policy.domain.basin_credit.basin_weight"
            ]
            == 0.5
        )
        assert (
            directional_mixed.updates["model.coordination.confidence_weighting"]
            == "peer_direction"
        )


def test_toy24_basin_credit_replay_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_credit_replay_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_replay_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "basin_credit_w1p0_h1_prototype",
        "mixed_individual_basin_w0p5_0p5_h1",
        "mixed_objective_basin_w0p5_0p5_h1",
        "mixed_objective_basin_replay_all_p2_h1",
        "mixed_objective_basin_replay_all_p3_h1",
    }
    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        assert variants["linear_welfare_heavy"].group == "baseline"
        assert variants["mixed_individual_basin_w0p5_0p5_h1"].group == "diagnostic"
        assert variants["mixed_objective_basin_w0p5_0p5_h1"].group == "nabm"
        replay_p2 = variants["mixed_objective_basin_replay_all_p2_h1"]
        replay_p3 = variants["mixed_objective_basin_replay_all_p3_h1"]
        for replay, passes in [(replay_p2, 2), (replay_p3, 3)]:
            assert replay.group == "nabm"
            assert (
                replay.updates["model.policy.domain.basin_credit.objective_weight"]
                == 0.5
            )
            assert (
                replay.updates["model.policy.domain.basin_credit.individual_weight"]
                == 0.0
            )
            assert replay.updates["model.policy.domain.basin_credit.basin_weight"] == 0.5
            assert (
                replay.updates["model.policy.domain.basin_credit.training_scope"]
                == "all"
            )
            assert (
                replay.updates["model.policy.domain.basin_credit.training_passes"]
                == passes
            )


def test_toy24_basin_credit_adaptive_replay_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_credit_adaptive_replay_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_adaptive_replay_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "basin_credit_w1p0_h1_prototype",
        "mixed_individual_basin_w0p5_0p5_h1",
        "mixed_objective_basin_w0p5_0p5_h1",
        "mixed_objective_basin_replay_revised_p2_h1",
        "mixed_objective_basin_replay_revised_p3_h1",
        "mixed_objective_basin_replay_all_p1_h1",
        "mixed_objective_basin_replay_all_p2_h1",
        "mixed_objective_basin_replay_all_p3_h1",
        "mixed_objective_basin_adaptive_score_p3_min2_h1",
    }
    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        assert variants["reputation_imitation"].group == "baseline"
        assert variants["linear_welfare_heavy"].group == "baseline"
        assert variants["mixed_individual_basin_w0p5_0p5_h1"].group == "diagnostic"
        adaptive = variants["mixed_objective_basin_adaptive_score_p3_min2_h1"]
        assert adaptive.group == "nabm"
        assert (
            adaptive.updates["model.policy.domain.basin_credit.objective_weight"]
            == 0.5
        )
        assert (
            adaptive.updates["model.policy.domain.basin_credit.training_scope"]
            == "all"
        )
        assert (
            adaptive.updates["model.policy.domain.basin_credit.training_passes"]
            == 3
        )
        assert (
            adaptive.updates["model.policy.domain.basin_credit.training_pass_schedule"]
            == "target_score_decay"
        )
        assert (
            adaptive.updates["model.policy.domain.basin_credit.min_training_passes"]
            == 2
        )
        assert (
            adaptive.updates[
                "model.policy.domain.basin_credit.training_pass_score_threshold"
            ]
            == 0.995
        )


def test_toy24_basin_credit_escalation_replay_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_credit_escalation_replay_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_escalation_replay_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    expected_variants = {
        "reputation_imitation",
        "linear_welfare_heavy",
        "decision_bootstrap_w1p0_e5_linear_welfare",
        "basin_credit_w1p0_h1_prototype",
        "mixed_individual_basin_w0p5_0p5_h1",
        "mixed_objective_basin_w0p5_0p5_h1",
        "mixed_objective_basin_replay_all_p2_h1",
        "mixed_objective_basin_replay_all_p3_h1",
        "mixed_objective_basin_adaptive_score_p3_min2_h1",
        "mixed_objective_basin_escalate_credit_p3_min2_h1",
    }
    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert set(variants) == expected_variants
        escalate = variants["mixed_objective_basin_escalate_credit_p3_min2_h1"]
        assert escalate.group == "nabm"
        assert (
            escalate.updates["model.policy.domain.basin_credit.objective_weight"]
            == 0.5
        )
        assert (
            escalate.updates["model.policy.domain.basin_credit.training_scope"]
            == "all"
        )
        assert (
            escalate.updates["model.policy.domain.basin_credit.training_passes"]
            == 3
        )
        assert (
            escalate.updates[
                "model.policy.domain.basin_credit.training_pass_schedule"
            ]
            == "credit_signal_escalation"
        )
        assert (
            escalate.updates["model.policy.domain.basin_credit.min_training_passes"]
            == 2
        )
        assert (
            escalate.updates[
                "model.policy.domain.basin_credit.training_pass_credit_positive_threshold"
            ]
            == 0.6
        )
        assert (
            escalate.updates[
                "model.policy.domain.basin_credit.training_pass_credit_delta_threshold"
            ]
            == 0.0
        )
