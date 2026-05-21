from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from neural_abm.diagnostics.evidence_profile import profile_evidence_artifacts


def write_manifest(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def toy5_manifest(label: str) -> dict:
    return {
        "label": label,
        "seeds": [1, 2],
        "epochs": 50,
        "success_criteria": {
            "main_group": "threshold_aware",
            "require_without_teacher_bootstrap_replay": True,
            "cases": {
                "toy5_threshold_aware_lattice_k4_h0p85_spread": {
                    "final_ceiling_min_hits": 2,
                    "mean_time_to_ceiling_lt": 45,
                }
            },
        },
        "cases": [
            {
                "toy": "toy5",
                "name": "toy5_threshold_aware_lattice_k4_h0p85_spread",
                "base_config": "experiments/configs/toy5_contagion_adoption_baseline.yaml",
                "primary_metric": "domain_cascade_size",
                "direction": "maximize",
                "baseline_group": "baseline",
                "nabm_group": "threshold_aware",
                "ceiling_metric": "action_rate",
                "ceiling_value": 0.95,
                "variants": [
                    {
                        "name": "baseline_output_average",
                        "group": "baseline",
                        "updates": {},
                    },
                    {
                        "name": "mean_readiness_diagnostic",
                        "group": "diagnostic",
                        "updates": {
                            "model.coordination.precommitment_direction_source": (
                                "readiness_augmented_threshold_with_action_anchor"
                            ),
                            "model.coordination.precommitment_peer_readiness_aggregation": "mean",
                            "domain.graph.k": 4,
                            "domain.environment.threshold_mode": "heterogeneous",
                            "domain.environment.heterogeneous_threshold_high": 0.85,
                        },
                    },
                    {
                        "name": "max_threshold_anchor",
                        "group": "threshold_aware",
                        "updates": {
                            "model.coordination.precommitment_direction_source": (
                                "readiness_augmented_threshold_with_action_anchor"
                            ),
                            "model.coordination.precommitment_peer_readiness_aggregation": "max",
                            "domain.graph.k": 4,
                            "domain.environment.threshold_mode": "heterogeneous",
                            "domain.environment.heterogeneous_threshold_high": 0.85,
                        },
                    },
                ],
            }
        ],
    }


def toy5_rows(label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case = "toy5_threshold_aware_lattice_k4_h0p85_spread"
    shared = {
        "label": label,
        "case": case,
        "toy": "toy5",
        "baseline_group": "baseline",
        "nabm_group": "threshold_aware",
        "primary_metric": "domain_cascade_size",
        "direction": "maximize",
        "ceiling_metric": "action_rate",
        "ceiling_value": 0.95,
        "ceiling_tolerance": 0.0,
        "ever_ceiling_final_miss": False,
        "late_flip_rate_after_first_ceiling": 0.0,
        "terminal_window_ceiling_rate": 0.0,
        "precommitment_direction_score_mean": 0.0,
        "precommitment_direction_ok_rate": 0.0,
        "precommitment_peer_readiness_mean": 0.0,
        "precommitment_peer_readiness_aggregation": "mean",
        "config_path": "",
        "run_dir": "",
        "nabm_status": "full",
        "neural_role": "",
        "social_channels": "",
        "reference_policies": "",
    }
    for seed in (1, 2):
        rows.append(
            {
                **shared,
                "variant": "baseline_output_average",
                "group": "baseline",
                "seed": seed,
                "metric_value": 1,
                "ceiling_metric_value": 0.01,
                "ceiling_gap": 0.94,
                "final_within_ceiling": False,
                "ever_reached_ceiling": False,
                "time_to_ceiling": "",
            }
        )
        rows.append(
            {
                **shared,
                "variant": "mean_readiness_diagnostic",
                "group": "diagnostic",
                "seed": seed,
                "metric_value": 49,
                "ceiling_metric_value": 0.49,
                "ceiling_gap": 0.46,
                "final_within_ceiling": False,
                "ever_reached_ceiling": False,
                "time_to_ceiling": "",
            }
        )
        rows.append(
            {
                **shared,
                "variant": "max_threshold_anchor",
                "group": "threshold_aware",
                "seed": seed,
                "metric_value": 100,
                "ceiling_metric_value": 1.0,
                "ceiling_gap": 0.0,
                "final_within_ceiling": True,
                "ever_reached_ceiling": True,
                "time_to_ceiling": 20 + seed,
                "terminal_window_ceiling_rate": 1.0,
                "precommitment_direction_score_mean": 2.4,
                "precommitment_direction_ok_rate": 1.0,
                "precommitment_peer_readiness_mean": 1.0,
                "precommitment_peer_readiness_aggregation": "max",
            }
        )
    return rows


def test_evidence_profile_toy5_adapter_marks_wavefront_contrast(
    tmp_path: Path,
) -> None:
    label = "toy5_profile_test"
    manifest_path = tmp_path / "manifest.yaml"
    runs_path = tmp_path / "runs.csv"
    write_manifest(manifest_path, toy5_manifest(label))
    write_rows(runs_path, toy5_rows(label))

    output = profile_evidence_artifacts(
        manifest_path,
        runs_path=runs_path,
        output_dir=tmp_path,
    )

    assert output.profile.status == "pass"
    assert output.markdown_path is not None
    assert output.json_path is not None
    assert output.cases_path is not None
    assert "toy5_threshold_aware_evidence" in output.profile.notes
    assert "toy5_mean_vs_max_frontier_contrast" in output.profile.notes
    case = output.profile.cases[0]
    assert "toy5_threshold_aware_direction" in case.notes
    assert "toy5_mean_readiness_frontier_stall" in case.notes
    max_variant = next(
        variant for variant in case.variants if variant.variant == "max_threshold_anchor"
    )
    assert max_variant.details["readiness_aggregation"] == "max"
    assert max_variant.details["direction_source"] == (
        "readiness_augmented_threshold_with_action_anchor"
    )

    profile_json = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert profile_json["label"] == label
    assert "max_threshold_anchor" in output.markdown_path.read_text(encoding="utf-8")
    assert "best_main_variant" in output.cases_path.read_text(encoding="utf-8")


def test_evidence_profile_tolerates_missing_optional_diagnostic_fields(
    tmp_path: Path,
) -> None:
    label = "generic_profile_test"
    manifest_path = tmp_path / "manifest.yaml"
    runs_path = tmp_path / "runs.csv"
    write_manifest(
        manifest_path,
        {
            "label": label,
            "seeds": [1],
            "epochs": 2,
            "success_criteria": {
                "main_group": "nabm",
                "cases": {
                    "toy1_case": {
                        "final_ceiling_min_hits": 1,
                        "mean_time_to_ceiling_lt": 2,
                    }
                },
            },
            "cases": [
                {
                    "toy": "toy1",
                    "name": "toy1_case",
                    "base_config": "experiments/configs/toy1_classification_baseline.yaml",
                    "primary_metric": "score",
                    "direction": "maximize",
                    "variants": [
                        {
                            "name": "toy1_main",
                            "group": "nabm",
                            "updates": {},
                        }
                    ],
                }
            ],
        },
    )
    write_rows(
        runs_path,
        [
            {
                "label": label,
                "case": "toy1_case",
                "toy": "toy1",
                "variant": "toy1_main",
                "group": "nabm",
                "baseline_group": "baseline",
                "nabm_group": "nabm",
                "seed": 1,
                "primary_metric": "score",
                "metric_value": 1.0,
                "direction": "maximize",
                "ceiling_metric": "score",
                "ceiling_value": 1.0,
                "ceiling_tolerance": 0.0,
                "ceiling_metric_value": 1.0,
                "ceiling_gap": 0.0,
                "final_within_ceiling": True,
                "ever_reached_ceiling": True,
                "time_to_ceiling": 1,
                "config_path": "",
                "run_dir": "",
                "nabm_status": "full",
                "neural_role": "",
                "social_channels": "",
                "reference_policies": "",
            }
        ],
    )

    output = profile_evidence_artifacts(
        manifest_path,
        runs_path=runs_path,
        write_outputs=False,
    )

    assert output.profile.status == "pass"
    assert output.profile.cases[0].variants[0].metric.mean == 1.0


def toy24_basin_manifest(label: str) -> dict:
    return {
        "label": label,
        "seeds": [1, 2, 3],
        "epochs": 50,
        "success_criteria": {
            "main_group": "nabm",
            "require_without_teacher_bootstrap_replay": True,
            "cases": {
                "toy4_basin_credit": {
                    "final_ceiling_min_hits": 2,
                    "mean_time_to_ceiling_lt": 20,
                }
            },
        },
        "cases": [
            {
                "toy": "toy4",
                "name": "toy4_basin_credit",
                "base_config": "experiments/configs/toy4_public_goods_baseline.yaml",
                "primary_metric": "final_mean_payoff",
                "direction": "maximize",
                "baseline_group": "baseline",
                "nabm_group": "nabm",
                "ceiling_metric": "mean_payoff",
                "ceiling_value": 0.6,
                "ceiling_tolerance": 0.005,
                "variants": [
                    {
                        "name": "reputation_imitation",
                        "group": "baseline",
                        "updates": {"model.policy.rule": "reputation_imitation"},
                    },
                    {
                        "name": "mixed_individual_basin_w0p5_0p5_h1",
                        "group": "diagnostic",
                        "updates": {
                            "model.policy.domain.basin_credit.enabled": True,
                            "model.policy.domain.basin_credit.critic": "prototype_phase",
                            "model.policy.domain.basin_credit.credit_method": "one_step_ablation",
                            "model.policy.domain.basin_credit.objective_weight": 0.0,
                            "model.policy.domain.basin_credit.individual_weight": 0.5,
                            "model.policy.domain.basin_credit.basin_weight": 0.5,
                            "model.policy.domain.basin_credit.horizon": 1,
                            "model.policy.domain.basin_credit.target_basin": "ceiling",
                        },
                    },
                    {
                        "name": "mixed_objective_basin_w0p5_0p5_h1",
                        "group": "nabm",
                        "updates": {
                            "model.policy.domain.basin_credit.enabled": True,
                            "model.policy.domain.basin_credit.critic": "prototype_phase",
                            "model.policy.domain.basin_credit.credit_method": "one_step_ablation",
                            "model.policy.domain.basin_credit.objective_weight": 0.5,
                            "model.policy.domain.basin_credit.individual_weight": 0.0,
                            "model.policy.domain.basin_credit.basin_weight": 0.5,
                            "model.policy.domain.basin_credit.horizon": 1,
                            "model.policy.domain.basin_credit.target_basin": "ceiling",
                            "model.coordination.revision_operator_enabled": True,
                            "model.coordination.revision_operator_source": "policy_probability",
                        },
                    },
                ],
            }
        ],
    }


def toy24_basin_rows(label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shared = {
        "label": label,
        "case": "toy4_basin_credit",
        "toy": "toy4",
        "baseline_group": "baseline",
        "nabm_group": "nabm",
        "primary_metric": "final_mean_payoff",
        "direction": "maximize",
        "ceiling_metric": "mean_payoff",
        "ceiling_value": 0.6,
        "ceiling_tolerance": 0.005,
        "config_path": "",
        "run_dir": "",
        "nabm_status": "full",
        "neural_role": "",
        "social_channels": "",
        "reference_policies": "",
    }
    for seed in (1, 2, 3):
        rows.append(
            {
                **shared,
                "variant": "reputation_imitation",
                "group": "baseline",
                "seed": seed,
                "metric_value": 0.6,
                "ceiling_metric_value": 0.6,
                "ceiling_gap": 0.0,
                "final_within_ceiling": True,
                "ever_reached_ceiling": True,
                "time_to_ceiling": 3,
                "ever_ceiling_final_miss": False,
                "late_flip_count_after_first_ceiling": 0.0,
                "late_flip_rate_after_first_ceiling": 0.0,
                "terminal_window_ceiling_rate": 1.0,
                "terminal_window_mean_ceiling_metric": 0.6,
            }
        )
        rows.append(
            {
                **shared,
                "variant": "mixed_individual_basin_w0p5_0p5_h1",
                "group": "diagnostic",
                "seed": seed,
                "metric_value": 0.01,
                "ceiling_metric_value": 0.01,
                "ceiling_gap": 0.59,
                "final_within_ceiling": False,
                "ever_reached_ceiling": False,
                "time_to_ceiling": "",
                "ever_ceiling_final_miss": False,
                "late_flip_count_after_first_ceiling": 0.0,
                "late_flip_rate_after_first_ceiling": 0.0,
                "terminal_window_ceiling_rate": 0.0,
                "terminal_window_mean_ceiling_metric": 0.01,
            }
        )
    for seed, final_hit, final_metric, time_to_ceiling, late_rate in (
        (1, False, 0.594, 10, 0.006),
        (2, True, 0.6, 12, 0.0),
        (3, True, 0.6, 11, 0.0),
    ):
        rows.append(
            {
                **shared,
                "variant": "mixed_objective_basin_w0p5_0p5_h1",
                "group": "nabm",
                "seed": seed,
                "metric_value": final_metric,
                "ceiling_metric_value": final_metric,
                "ceiling_gap": 0.6 - final_metric,
                "final_within_ceiling": final_hit,
                "ever_reached_ceiling": True,
                "time_to_ceiling": time_to_ceiling,
                "ever_ceiling_final_miss": not final_hit,
                "late_flip_count_after_first_ceiling": 1.0 if late_rate else 0.0,
                "late_flip_rate_after_first_ceiling": late_rate,
                "terminal_window_ceiling_rate": 0.8 if late_rate else 1.0,
                "terminal_window_mean_ceiling_metric": 0.5988 if late_rate else 0.6,
            }
        )
    return rows


def test_evidence_profile_toy24_adapter_marks_basin_revision_hazards(
    tmp_path: Path,
) -> None:
    label = "toy24_profile_test"
    manifest_path = tmp_path / "manifest.yaml"
    runs_path = tmp_path / "runs.csv"
    write_manifest(manifest_path, toy24_basin_manifest(label))
    write_rows(runs_path, toy24_basin_rows(label))

    output = profile_evidence_artifacts(
        manifest_path,
        runs_path=runs_path,
        output_dir=tmp_path,
    )

    assert output.profile.status == "pass"
    assert "toy24_objective_basin_evidence" in output.profile.notes
    assert "toy24_material_basin_collapse_contrast" in output.profile.notes
    assert "toy24_revision_operator_evidence" in output.profile.notes
    assert "toy24_final_epoch_hazard_evidence" in output.profile.notes
    case = output.profile.cases[0]
    assert "toy24_objective_basin_blend" in case.notes
    assert "toy24_material_basin_collapse_diagnostic" in case.notes
    assert "toy24_final_epoch_hazard" in case.issue_codes
    assert "toy24_main_candidate_ceiling_miss" in case.notes
    assert "toy24_best_main_ceiling_miss" in case.issue_codes
    main_variant = next(
        variant
        for variant in case.variants
        if variant.variant == "mixed_objective_basin_w0p5_0p5_h1"
    )
    assert main_variant.details["basin_objective_weight"] == 0.5
    assert main_variant.details["basin_weight"] == 0.5
    assert main_variant.details["revision_operator_enabled"] is True
    assert main_variant.details["ever_ceiling_final_miss_rate"] == 1 / 3
