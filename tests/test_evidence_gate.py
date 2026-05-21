from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.evidence_gate import (
    EvidenceGateInputError,
    EvidenceGateCaseCriterion,
    EvidenceGateCriteria,
    evaluate_evidence_gate,
    load_gate_manifest,
    render_gate_markdown,
    variant_uses_teacher_bootstrap_replay,
)
from neural_abm.evidence_matrix import EvidenceManifest, MatrixCase, MatrixVariant


def make_manifest(
    *,
    variants: tuple[MatrixVariant, ...],
    seeds: tuple[int, ...] = (1, 2, 3),
) -> EvidenceManifest:
    return EvidenceManifest(
        label="fake_gate",
        seeds=seeds,
        epochs=50,
        config_dir=Path("configs"),
        runs_dir=Path("runs"),
        cases=(
            MatrixCase(
                toy="toy2",
                name="toy2_basin_credit",
                base_config=Path("base.yaml"),
                primary_metric="final_mean_payoff",
                direction="maximize",
                variants=variants,
                ceiling_metric="mean_payoff",
                ceiling_value=3.0,
                ceiling_tolerance=0.001,
            ),
        ),
    )


def gate_criteria(
    *,
    final_ceiling_min_hits: int = 3,
    mean_time_to_ceiling_lt: float = 10.0,
) -> EvidenceGateCriteria:
    return EvidenceGateCriteria(
        main_group="nabm",
        require_without_teacher_bootstrap_replay=True,
        cases={
            "toy2_basin_credit": EvidenceGateCaseCriterion(
                final_ceiling_min_hits=final_ceiling_min_hits,
                mean_time_to_ceiling_lt=mean_time_to_ceiling_lt,
            )
        },
    )


def run_row(
    *,
    variant: str,
    group: str,
    seed: int,
    final_hit: bool,
    time_to_ceiling: int | str,
    metric_value: float = 3.0,
    ever_ceiling_final_miss: bool | None = None,
    late_flip_count_after_first_ceiling: float | None = None,
    late_flip_rate_after_first_ceiling: float | None = None,
    terminal_window_ceiling_rate: float | None = None,
    terminal_window_mean_ceiling_metric: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": "fake_gate",
        "case": "toy2_basin_credit",
        "toy": "toy2",
        "variant": variant,
        "group": group,
        "seed": seed,
        "primary_metric": "final_mean_payoff",
        "metric_value": metric_value,
        "direction": "maximize",
        "final_within_ceiling": final_hit,
        "ever_reached_ceiling": time_to_ceiling != "",
        "time_to_ceiling": time_to_ceiling,
    }
    optional_fields = {
        "ever_ceiling_final_miss": ever_ceiling_final_miss,
        "late_flip_count_after_first_ceiling": late_flip_count_after_first_ceiling,
        "late_flip_rate_after_first_ceiling": late_flip_rate_after_first_ceiling,
        "terminal_window_ceiling_rate": terminal_window_ceiling_rate,
        "terminal_window_mean_ceiling_metric": terminal_window_mean_ceiling_metric,
    }
    row.update(
        {key: value for key, value in optional_fields.items() if value is not None}
    )
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_evidence_gate_passes_eligible_nabm_variant() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant("reputation", "baseline", {}),
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    rows = [
        run_row(variant="reputation", group="baseline", seed=1, final_hit=True, time_to_ceiling=8),
        run_row(variant="reputation", group="baseline", seed=2, final_hit=False, time_to_ceiling=""),
        run_row(variant="reputation", group="baseline", seed=3, final_hit=False, time_to_ceiling=""),
        run_row(variant="basin_credit", group="nabm", seed=1, final_hit=True, time_to_ceiling=3),
        run_row(variant="basin_credit", group="nabm", seed=2, final_hit=True, time_to_ceiling=4),
        run_row(variant="basin_credit", group="nabm", seed=3, final_hit=True, time_to_ceiling=5),
    ]

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(),
        run_rows=rows,
    )

    assert summary["status"] == "pass"
    [case] = summary["cases"]
    assert case["status"] == "pass"
    assert case["best_main_variant"]["variant"] == "basin_credit"
    assert case["best_main_variant"]["final_ceiling_hits"] == 3
    assert case["best_main_variant"]["mean_time_to_ceiling"] == pytest.approx(4.0)
    assert case["baseline_improved_by_best_main"] is True


def test_evidence_gate_does_not_count_baseline_or_diagnostic_as_main_success() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant("reputation", "baseline", {}),
            MatrixVariant(
                "decision_bootstrap",
                "diagnostic",
                {"model.policy.domain.bootstrap.decision_enabled": True},
            ),
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    rows = []
    for variant, group in (
        ("reputation", "baseline"),
        ("decision_bootstrap", "diagnostic"),
    ):
        rows.extend(
            run_row(variant=variant, group=group, seed=seed, final_hit=True, time_to_ceiling=2)
            for seed in (1, 2, 3)
        )
    rows.extend(
        run_row(variant="basin_credit", group="nabm", seed=seed, final_hit=False, time_to_ceiling="")
        for seed in (1, 2, 3)
    )

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(),
        run_rows=rows,
    )

    assert summary["status"] == "fail"
    [case] = summary["cases"]
    assert case["status"] == "fail"
    diagnostic = next(
        variant
        for variant in case["variants"]
        if variant["variant"] == "decision_bootstrap"
    )
    assert diagnostic["status"] == "diagnostic_only"
    assert diagnostic["uses_teacher_bootstrap_replay"] is True


def test_evidence_gate_excludes_bootstrap_variant_from_main_claim() -> None:
    bootstrap_variant = MatrixVariant(
        "bootstrap_claim",
        "nabm",
        {"model.policy.domain.bootstrap.decision_enabled": True},
    )
    manifest = make_manifest(variants=(bootstrap_variant,))
    rows = [
        run_row(variant="bootstrap_claim", group="nabm", seed=seed, final_hit=True, time_to_ceiling=2)
        for seed in (1, 2, 3)
    ]

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(),
        run_rows=rows,
    )

    assert summary["status"] == "fail"
    [case] = summary["cases"]
    [variant] = case["variants"]
    assert variant["eligible_main"] is False
    assert variant["status"] == "diagnostic_only"
    assert variant_uses_teacher_bootstrap_replay(bootstrap_variant) is True


def test_evidence_gate_marks_missing_expected_seed_inconclusive() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    rows = [
        run_row(variant="basin_credit", group="nabm", seed=1, final_hit=True, time_to_ceiling=3),
        run_row(variant="basin_credit", group="nabm", seed=2, final_hit=True, time_to_ceiling=4),
    ]

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(final_ceiling_min_hits=2),
        run_rows=rows,
    )

    assert summary["status"] == "inconclusive"
    [case] = summary["cases"]
    [variant] = case["variants"]
    assert variant["status"] == "inconclusive"
    assert variant["missing_seeds"] == [3]


def test_evidence_gate_rejects_unknown_variant_row() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    rows = [
        run_row(
            variant="unexpected_variant",
            group="nabm",
            seed=1,
            final_hit=True,
            time_to_ceiling=3,
        )
    ]

    with pytest.raises(EvidenceGateInputError, match="unknown_rows=1"):
        evaluate_evidence_gate(
            manifest=manifest,
            criteria=gate_criteria(),
            run_rows=rows,
        )


def test_evidence_gate_rejects_duplicate_case_variant_seed_row() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    row = run_row(
        variant="basin_credit",
        group="nabm",
        seed=1,
        final_hit=True,
        time_to_ceiling=3,
    )

    with pytest.raises(EvidenceGateInputError, match="duplicate_rows=1"):
        evaluate_evidence_gate(
            manifest=manifest,
            criteria=gate_criteria(),
            run_rows=[row, dict(row)],
        )


def test_evidence_gate_rejects_missing_required_columns() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    row = run_row(
        variant="basin_credit",
        group="nabm",
        seed=1,
        final_hit=True,
        time_to_ceiling=3,
    )
    del row["metric_value"]

    with pytest.raises(EvidenceGateInputError, match="missing_required_columns=1"):
        evaluate_evidence_gate(
            manifest=manifest,
            criteria=gate_criteria(),
            run_rows=[row],
        )


def test_evidence_gate_marks_malformed_observed_row_inconclusive() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        ),
        seeds=(1,),
    )
    row = run_row(
        variant="basin_credit",
        group="nabm",
        seed=1,
        final_hit=True,
        time_to_ceiling="fast",
    )
    row["final_within_ceiling"] = "maybe"

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(final_ceiling_min_hits=1),
        run_rows=[row],
    )

    assert summary["status"] == "inconclusive"
    assert len(summary["input_validation"]["malformed_rows"]) == 1
    [case] = summary["cases"]
    [variant] = case["variants"]
    assert variant["status"] == "inconclusive"
    assert variant["malformed_rows"][0]["fields"] == [
        "final_within_ceiling",
        "time_to_ceiling",
    ]


def test_evidence_gate_reports_late_instability_metrics() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        )
    )
    rows = [
        run_row(
            variant="basin_credit",
            group="nabm",
            seed=1,
            final_hit=True,
            time_to_ceiling=3,
            ever_ceiling_final_miss=False,
            late_flip_count_after_first_ceiling=0.0,
            late_flip_rate_after_first_ceiling=0.0,
            terminal_window_ceiling_rate=1.0,
            terminal_window_mean_ceiling_metric=3.0,
        ),
        run_row(
            variant="basin_credit",
            group="nabm",
            seed=2,
            final_hit=False,
            time_to_ceiling=4,
            metric_value=2.996,
            ever_ceiling_final_miss=True,
            late_flip_count_after_first_ceiling=1.0,
            late_flip_rate_after_first_ceiling=0.01,
            terminal_window_ceiling_rate=0.8,
            terminal_window_mean_ceiling_metric=2.998,
        ),
        run_row(
            variant="basin_credit",
            group="nabm",
            seed=3,
            final_hit=True,
            time_to_ceiling=5,
            ever_ceiling_final_miss=False,
            late_flip_count_after_first_ceiling=0.0,
            late_flip_rate_after_first_ceiling=0.0,
            terminal_window_ceiling_rate=1.0,
            terminal_window_mean_ceiling_metric=3.0,
        ),
    ]

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(final_ceiling_min_hits=2),
        run_rows=rows,
    )

    [case] = summary["cases"]
    [variant] = case["variants"]
    assert variant["ever_ceiling_final_miss_count"] == 1
    assert variant["late_flip_count_after_first_ceiling_mean"] == pytest.approx(1 / 3)
    assert variant["late_flip_rate_after_first_ceiling_mean"] == pytest.approx(0.01 / 3)
    assert variant["terminal_window_ceiling_rate_mean"] == pytest.approx(0.9333333333)
    assert variant["terminal_window_mean_ceiling_metric_mean"] == pytest.approx(
        (3.0 + 2.998 + 3.0) / 3
    )

    markdown = render_gate_markdown(summary)

    assert "Ever-Final Misses" in markdown
    assert "Late Flip Rate" in markdown


def test_evidence_gate_marks_malformed_optional_instability_field() -> None:
    manifest = make_manifest(
        variants=(
            MatrixVariant(
                "basin_credit",
                "nabm",
                {"model.policy.domain.basin_credit.enabled": True},
            ),
        ),
        seeds=(1,),
    )
    row = run_row(
        variant="basin_credit",
        group="nabm",
        seed=1,
        final_hit=True,
        time_to_ceiling=3,
    )
    row["late_flip_rate_after_first_ceiling"] = "fast"

    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=gate_criteria(final_ceiling_min_hits=1),
        run_rows=[row],
    )

    assert summary["status"] == "inconclusive"
    [malformed] = summary["input_validation"]["malformed_rows"]
    assert malformed["fields"] == ["late_flip_rate_after_first_ceiling"]


def test_basin_credit_workflow_cli_skip_matrix_smoke(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fake_gate_cli.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "label": "fake_gate_cli",
                "seeds": [1],
                "epochs": 1,
                "success_criteria": {
                    "main_group": "nabm",
                    "require_without_teacher_bootstrap_replay": True,
                    "cases": {
                        "toy2_case": {
                            "final_ceiling_min_hits": 1,
                            "mean_time_to_ceiling_lt": 10,
                        }
                    },
                },
                "cases": [
                    {
                        "toy": "toy2",
                        "name": "toy2_case",
                        "base_config": "base.yaml",
                        "primary_metric": "final_mean_payoff",
                        "direction": "maximize",
                        "ceiling_metric": "mean_payoff",
                        "ceiling_value": 3.0,
                        "ceiling_tolerance": 0.001,
                        "variants": [
                            {
                                "name": "basin_credit",
                                "group": "nabm",
                                "updates": {
                                    "model.policy.domain.basin_credit.enabled": True
                                },
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    row = run_row(
        variant="basin_credit",
        group="nabm",
        seed=1,
        final_hit=True,
        time_to_ceiling=3,
    )
    row["label"] = "fake_gate_cli"
    row["case"] = "toy2_case"
    runs_path = tmp_path / "fake_runs.csv"
    write_rows(runs_path, [row])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_basin_credit_evidence_workflow.py",
            "--manifest",
            str(manifest_path),
            "--skip-matrix",
            "--runs-path",
            str(runs_path),
            "--gate-output-dir",
            str(tmp_path / "gate"),
            "--profile-output-dir",
            str(tmp_path / "profiles"),
            "--require-pass",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Evidence gate status: pass" in completed.stdout
    assert "Wrote evidence profile JSON" in completed.stdout
    assert (tmp_path / "gate" / "fake_gate_cli.summary.json").exists()
    assert (tmp_path / "gate" / "fake_gate_cli.summary.md").exists()
    assert (tmp_path / "profiles" / "fake_gate_cli_profile.json").exists()
    assert (tmp_path / "profiles" / "fake_gate_cli_profile.md").exists()
    assert (tmp_path / "profiles" / "fake_gate_cli_profile_cases.csv").exists()


def test_toy24_basin_credit_manifest_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy24_basin_credit_quick.yaml")
    )

    assert manifest.label == "toy24_basin_credit_quick"
    assert criteria.main_group == "nabm"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert criteria.cases["toy2_basin_credit"].final_ceiling_min_hits == 3
    assert criteria.cases["toy2_basin_credit"].mean_time_to_ceiling_lt == pytest.approx(
        10.0
    )
    assert criteria.cases["toy4_basin_credit"].final_ceiling_min_hits == 2
    assert criteria.cases["toy4_basin_credit"].mean_time_to_ceiling_lt == pytest.approx(
        12.0
    )


def test_toy24_precommitment_peer_evidence_closure_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy24_precommitment_peer_evidence_closure_quick.yaml")
    )

    assert manifest.label == "toy24_precommitment_peer_evidence_closure_quick"
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert criteria.main_group == "peer_evidence_closure"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy2_precommitment_peer_evidence_closure"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy2_precommitment_peer_evidence_closure"
    ].mean_time_to_ceiling_lt == pytest.approx(10.0)
    assert (
        criteria.cases[
            "toy4_precommitment_peer_evidence_closure"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy4_precommitment_peer_evidence_closure"
    ].mean_time_to_ceiling_lt == pytest.approx(10.0)

    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert case.nabm_group == "peer_evidence_closure"
        assert set(variants) == {
            "reputation_imitation",
            "objective_basin_w0p5_0p5_h1",
            "revision_objective_basin_w0p5_0p5_h1",
            "revision_precommitment_evidence",
            "revision_precommitment_peer_evidence_w1p0",
        }
        assert variants["revision_precommitment_peer_evidence_w1p0"].group == (
            "peer_evidence_closure"
        )
        assert variants["revision_objective_basin_w0p5_0p5_h1"].updates[
            "model.coordination.revision_operator_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_w1p0"].updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_w1p0"].updates[
            "model.coordination.precommitment_peer_evidence_weight"
        ] == pytest.approx(1.0)


def test_toy24_precommitment_peer_evidence_noisy_reputation_stress_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy24_precommitment_peer_evidence_noisy_reputation_stress_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy24_precommitment_peer_evidence_noisy_reputation_stress_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert criteria.main_group == "peer_evidence_noisy_reputation_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy2_precommitment_peer_evidence_noisy_reputation_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy2_precommitment_peer_evidence_noisy_reputation_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(12.0)
    assert (
        criteria.cases[
            "toy4_precommitment_peer_evidence_noisy_reputation_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy4_precommitment_peer_evidence_noisy_reputation_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(12.0)

    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert case.nabm_group == "peer_evidence_noisy_reputation_stress"
        assert set(variants) == {
            "reputation_imitation_noisy_s1p0",
            "objective_basin_noisy_s1p0",
            "revision_objective_basin_noisy_s1p0",
            "revision_precommitment_evidence_noisy_s1p0",
            "revision_precommitment_peer_evidence_noisy_s1p0",
        }
        for variant in variants.values():
            assert variant.updates["model.state.reputation.noise"] == pytest.approx(
                1.0
            )
        assert variants["revision_precommitment_peer_evidence_noisy_s1p0"].group == (
            "peer_evidence_noisy_reputation_stress"
        )
        assert variants["revision_objective_basin_noisy_s1p0"].updates[
            "model.coordination.revision_operator_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_noisy_s1p0"].updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_noisy_s1p0"].updates[
            "model.coordination.precommitment_peer_evidence_weight"
        ] == pytest.approx(1.0)


def test_toy24_precommitment_peer_evidence_sparse_seed_stress_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy24_precommitment_peer_evidence_sparse_seed_stress_quick.yaml"
        )
    )

    assert manifest.label == "toy24_precommitment_peer_evidence_sparse_seed_stress_quick"
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert criteria.main_group == "peer_evidence_sparse_seed_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy2_precommitment_peer_evidence_sparse_seed_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy2_precommitment_peer_evidence_sparse_seed_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(25.0)
    assert (
        criteria.cases[
            "toy4_precommitment_peer_evidence_sparse_seed_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy4_precommitment_peer_evidence_sparse_seed_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(25.0)

    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert case.nabm_group == "peer_evidence_sparse_seed_stress"
        assert set(variants) == {
            "reputation_imitation_sparse_p0p1",
            "objective_basin_sparse_p0p1",
            "revision_objective_basin_sparse_p0p1",
            "revision_precommitment_evidence_sparse_p0p1",
            "revision_precommitment_peer_evidence_sparse_p0p1",
        }
        for variant in variants.values():
            assert variant.updates[
                "domain.environment.initial_action_probability"
            ] == pytest.approx(0.1)
        assert variants["revision_precommitment_peer_evidence_sparse_p0p1"].group == (
            "peer_evidence_sparse_seed_stress"
        )
        assert variants["revision_objective_basin_sparse_p0p1"].updates[
            "model.coordination.revision_operator_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_sparse_p0p1"].updates[
            "model.coordination.precommitment_peer_evidence_enabled"
        ] is True
        assert variants["revision_precommitment_peer_evidence_sparse_p0p1"].updates[
            "model.coordination.precommitment_peer_evidence_weight"
        ] == pytest.approx(1.0)


def test_toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy24_precommitment_peer_evidence_"
            "open_boundary_sparse_seed_stress_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert criteria.main_group == "peer_evidence_open_boundary_sparse_seed_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(25.0)
    assert (
        criteria.cases[
            "toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(25.0)

    for case in manifest.cases:
        variants = {variant.name: variant for variant in case.variants}
        assert case.nabm_group == "peer_evidence_open_boundary_sparse_seed_stress"
        assert set(variants) == {
            "reputation_imitation_open_sparse_p0p1",
            "objective_basin_open_sparse_p0p1",
            "revision_objective_basin_open_sparse_p0p1",
            "revision_precommitment_evidence_open_sparse_p0p1",
            "revision_precommitment_peer_evidence_open_sparse_p0p1",
        }
        for variant in variants.values():
            assert variant.updates[
                "domain.environment.initial_action_probability"
            ] == pytest.approx(0.1)
            if case.toy == "toy2":
                assert variant.updates["domain.environment.periodic"] is False
            else:
                assert variant.updates["domain.graph.periodic"] is False
        assert (
            variants["revision_precommitment_peer_evidence_open_sparse_p0p1"].group
            == "peer_evidence_open_boundary_sparse_seed_stress"
        )
        assert variants["revision_objective_basin_open_sparse_p0p1"].updates[
            "model.coordination.revision_operator_enabled"
        ] is True
        assert variants[
            "revision_precommitment_peer_evidence_open_sparse_p0p1"
        ].updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        assert variants[
            "revision_precommitment_peer_evidence_open_sparse_p0p1"
        ].updates["model.coordination.precommitment_peer_evidence_weight"] == (
            pytest.approx(1.0)
        )


def test_toy24_precommitment_peer_evidence_domain_uncertainty_calibration_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy24_precommitment_peer_evidence_"
            "domain_uncertainty_calibration_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick"
    )
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 60
    assert criteria.main_group == "peer_evidence_domain_uncertainty_calibration"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy2_precommitment_peer_evidence_stag_hunt_calibration"
        ].final_ceiling_min_hits
        == 1
    )
    assert criteria.cases[
        "toy2_precommitment_peer_evidence_stag_hunt_calibration"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)
    assert (
        criteria.cases[
            "toy4_precommitment_peer_evidence_resource_coupled_calibration"
        ].final_ceiling_min_hits
        == 1
    )
    assert criteria.cases[
        "toy4_precommitment_peer_evidence_resource_coupled_calibration"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    cases = {case.name: case for case in manifest.cases}
    toy2 = cases["toy2_precommitment_peer_evidence_stag_hunt_calibration"]
    toy4 = cases["toy4_precommitment_peer_evidence_resource_coupled_calibration"]
    assert toy2.ceiling_value == pytest.approx(4.0)
    assert toy4.ceiling_value == pytest.approx(0.6)

    toy2_variants = {variant.name: variant for variant in toy2.variants}
    assert set(toy2_variants) == {
        "reputation_imitation_stag_hunt_p0p35",
        "objective_basin_stag_hunt_p0p35",
        "revision_objective_basin_stag_hunt_p0p35",
        "revision_precommitment_evidence_stag_hunt_p0p35",
        "revision_precommitment_peer_evidence_stag_hunt_p0p35",
    }
    for variant in toy2_variants.values():
        assert variant.updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert variant.updates["domain.game.family"] == "stag_hunt"
        assert variant.updates["domain.game.payoff.R"] == pytest.approx(4.0)
        assert variant.updates["domain.game.payoff.T"] == pytest.approx(3.0)
        assert variant.updates["domain.environment.payoff_R"] == pytest.approx(4.0)
        assert variant.updates["domain.environment.payoff_T"] == pytest.approx(3.0)
    assert (
        toy2_variants["revision_precommitment_peer_evidence_stag_hunt_p0p35"].group
        == "peer_evidence_domain_uncertainty_calibration"
    )
    assert toy2_variants["revision_objective_basin_stag_hunt_p0p35"].updates[
        "model.coordination.revision_operator_enabled"
    ] is True
    assert toy2_variants[
        "revision_precommitment_peer_evidence_stag_hunt_p0p35"
    ].updates["model.coordination.precommitment_peer_evidence_enabled"] is True

    toy4_variants = {variant.name: variant for variant in toy4.variants}
    assert set(toy4_variants) == {
        "reputation_imitation_resource_p0p35",
        "objective_basin_resource_p0p35",
        "revision_objective_basin_resource_p0p35",
        "revision_precommitment_evidence_resource_p0p35",
        "revision_precommitment_peer_evidence_resource_p0p35",
    }
    for variant in toy4_variants.values():
        assert variant.updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert variant.updates["domain.environment.resource_enabled"] is True
        assert variant.updates["domain.environment.resource_initial"] == pytest.approx(
            60.0
        )
        assert variant.updates[
            "domain.environment.resource_recovery_rate"
        ] == pytest.approx(0.03)
        assert variant.updates[
            "domain.environment.resource_extraction_per_defector"
        ] == pytest.approx(0.05)
    assert (
        toy4_variants["revision_precommitment_peer_evidence_resource_p0p35"].group
        == "peer_evidence_domain_uncertainty_calibration"
    )
    assert toy4_variants["revision_objective_basin_resource_p0p35"].updates[
        "model.coordination.revision_operator_enabled"
    ] is True
    assert toy4_variants[
        "revision_precommitment_peer_evidence_resource_p0p35"
    ].updates["model.coordination.precommitment_peer_evidence_enabled"] is True


def test_toy4_resource_environment_weight_probe_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy4_resource_environment_weight_probe_quick.yaml")
    )

    assert manifest.label == "toy4_resource_environment_weight_probe_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_environment_weight_probe"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases["toy4_resource_environment_weight_probe"].final_ceiling_min_hits
        == 1
    )
    assert criteria.cases[
        "toy4_resource_environment_weight_probe"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_environment_weight_probe"
    assert case.nabm_group == "resource_environment_weight_probe"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35",
        "revision_precommitment_peer_evidence_resource_envw0p0",
        "revision_precommitment_peer_evidence_resource_envw0p5",
        "revision_precommitment_peer_evidence_resource_envw1p0",
        "revision_precommitment_peer_evidence_resource_envw2p0",
    }
    assert variants["reputation_imitation_resource_p0p35"].group == "baseline"
    assert (
        variants["revision_precommitment_peer_evidence_resource_envw0p0"].group
        == "diagnostic"
    )

    expected_environment_weights = {
        "revision_precommitment_peer_evidence_resource_envw0p5": 0.5,
        "revision_precommitment_peer_evidence_resource_envw1p0": 1.0,
        "revision_precommitment_peer_evidence_resource_envw2p0": 2.0,
    }
    for variant_name, environment_weight in expected_environment_weights.items():
        assert variants[variant_name].group == "resource_environment_weight_probe"
        assert variants[variant_name].updates[
            "model.policy.domain.objective.environment_weight"
        ] == pytest.approx(environment_weight)

    for variant in variants.values():
        assert variant.updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert variant.updates["domain.environment.resource_enabled"] is True
        assert variant.updates["domain.environment.resource_initial"] == pytest.approx(
            60.0
        )
        assert variant.updates[
            "domain.environment.resource_carrying_capacity"
        ] == pytest.approx(100.0)
        assert variant.updates[
            "domain.environment.resource_recovery_rate"
        ] == pytest.approx(0.03)
        assert variant.updates[
            "domain.environment.resource_extraction_per_defector"
        ] == pytest.approx(0.05)
        assert variant.updates[
            "domain.environment.resource_collapse_threshold"
        ] == pytest.approx(0.0)

    for variant_name in {
        "revision_precommitment_peer_evidence_resource_envw0p0",
        *expected_environment_weights,
    }:
        updates = variants[variant_name].updates
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates["model.policy.domain.objective.mode"] == "state_continuation"
        assert updates["model.policy.domain.objective.profile"] == "custom"
        assert updates["model.policy.domain.objective.material_weight"] == pytest.approx(
            1.0
        )
        assert updates["model.policy.domain.objective.social_weight"] == pytest.approx(
            0.5
        )
        assert updates["model.policy.domain.objective.welfare_weight"] == pytest.approx(
            2.0
        )
        assert updates["model.policy.domain.objective.clip_abs"] is None
        assert updates["model.policy.domain.basin_credit.enabled"] is True
        assert (
            updates["model.policy.domain.basin_credit.critic"] == "prototype_phase"
        )
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert (
            updates["model.coordination.revision_operator_source"]
            == "policy_probability"
        )
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_min_policy_probability"]
            == pytest.approx(0.75)
        )
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )
        assert updates["model.coordination.precommitment_peer_evidence_weight"] == (
            pytest.approx(1.0)
        )

    assert variants[
        "revision_precommitment_peer_evidence_resource_envw0p0"
    ].updates["model.policy.domain.objective.environment_weight"] == pytest.approx(0.0)


def test_toy4_resource_lookahead_probe_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy4_resource_lookahead_probe_quick.yaml")
    )

    assert manifest.label == "toy4_resource_lookahead_probe_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_lookahead_probe"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases["toy4_resource_lookahead_probe"].final_ceiling_min_hits == 1
    )
    assert criteria.cases[
        "toy4_resource_lookahead_probe"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_lookahead_probe"
    assert case.nabm_group == "resource_lookahead_probe"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35",
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0",
        "revision_precommitment_peer_evidence_resource_lookahead_envw2p0",
        "revision_precommitment_peer_evidence_resource_lookahead_envw5p0",
        "revision_precommitment_peer_evidence_resource_lookahead_envw10p0",
    }
    assert variants["reputation_imitation_resource_p0p35"].group == "baseline"
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_pressure_envw2p0"
        ].group
        == "diagnostic"
    )

    expected_environment_weights = {
        "revision_precommitment_peer_evidence_resource_lookahead_envw2p0": 2.0,
        "revision_precommitment_peer_evidence_resource_lookahead_envw5p0": 5.0,
        "revision_precommitment_peer_evidence_resource_lookahead_envw10p0": 10.0,
    }
    for variant_name, environment_weight in expected_environment_weights.items():
        updates = variants[variant_name].updates
        assert variants[variant_name].group == "resource_lookahead_probe"
        assert updates["model.policy.domain.objective.profile"] == "custom"
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(environment_weight)
        )
        assert updates[
            "model.policy.domain.resource_environment_pressure_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_lookahead_weight"
        ] == pytest.approx(1.0)

    pressure_updates = variants[
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0"
    ].updates
    assert pressure_updates[
        "model.policy.domain.objective.environment_weight"
    ] == pytest.approx(2.0)
    assert pressure_updates[
        "model.policy.domain.resource_environment_pressure_weight"
    ] == pytest.approx(1.0)
    assert pressure_updates[
        "model.policy.domain.resource_environment_lookahead_weight"
    ] == pytest.approx(0.0)

    for variant in variants.values():
        assert variant.updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert variant.updates["domain.environment.resource_enabled"] is True
        assert variant.updates["domain.environment.resource_initial"] == pytest.approx(
            60.0
        )
        assert variant.updates[
            "domain.environment.resource_carrying_capacity"
        ] == pytest.approx(100.0)
        assert variant.updates[
            "domain.environment.resource_recovery_rate"
        ] == pytest.approx(0.03)
        assert variant.updates[
            "domain.environment.resource_extraction_per_defector"
        ] == pytest.approx(0.05)

    for variant_name in {
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0",
        *expected_environment_weights,
    }:
        updates = variants[variant_name].updates
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates["model.policy.domain.objective.mode"] == "state_continuation"
        assert updates["model.policy.domain.objective.material_weight"] == pytest.approx(
            1.0
        )
        assert updates["model.policy.domain.objective.social_weight"] == pytest.approx(
            0.5
        )
        assert updates["model.policy.domain.objective.welfare_weight"] == pytest.approx(
            2.0
        )
        assert updates["model.policy.domain.objective.clip_abs"] is None
        assert updates["model.policy.domain.basin_credit.enabled"] is True
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )


def test_toy4_resource_threshold_probe_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy4_resource_threshold_probe_quick.yaml")
    )

    assert manifest.label == "toy4_resource_threshold_probe_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_threshold_probe"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases["toy4_resource_threshold_probe"].final_ceiling_min_hits == 1
    )
    assert criteria.cases[
        "toy4_resource_threshold_probe"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_threshold_probe"
    assert case.nabm_group == "resource_threshold_probe"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35",
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0",
        "revision_precommitment_peer_evidence_resource_lookahead_envw10p0",
        "revision_precommitment_peer_evidence_resource_threshold_local_envw0p5",
        "revision_precommitment_peer_evidence_resource_threshold_local_envw1p0",
        "revision_precommitment_peer_evidence_resource_threshold_local_envw2p0",
    }
    assert variants["reputation_imitation_resource_p0p35"].group == "baseline"
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_pressure_envw2p0"
        ].group
        == "diagnostic"
    )
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_lookahead_envw10p0"
        ].group
        == "diagnostic"
    )

    expected_threshold_weights = {
        "revision_precommitment_peer_evidence_resource_threshold_local_envw0p5": 0.5,
        "revision_precommitment_peer_evidence_resource_threshold_local_envw1p0": 1.0,
        "revision_precommitment_peer_evidence_resource_threshold_local_envw2p0": 2.0,
    }
    for variant_name, environment_weight in expected_threshold_weights.items():
        updates = variants[variant_name].updates
        assert variants[variant_name].group == "resource_threshold_probe"
        assert updates["model.policy.domain.objective.profile"] == "custom"
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(environment_weight)
        )
        assert updates[
            "model.policy.domain.resource_environment_pressure_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_lookahead_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert (
            updates["model.policy.domain.resource_environment_threshold_scope"]
            == "local"
        )

    pressure_updates = variants[
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0"
    ].updates
    assert pressure_updates[
        "model.policy.domain.resource_environment_pressure_weight"
    ] == pytest.approx(1.0)
    assert pressure_updates[
        "model.policy.domain.resource_environment_threshold_weight"
    ] == pytest.approx(0.0)

    lookahead_updates = variants[
        "revision_precommitment_peer_evidence_resource_lookahead_envw10p0"
    ].updates
    assert lookahead_updates[
        "model.policy.domain.resource_environment_lookahead_weight"
    ] == pytest.approx(1.0)
    assert lookahead_updates[
        "model.policy.domain.resource_environment_threshold_weight"
    ] == pytest.approx(0.0)

    for variant in variants.values():
        assert variant.updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert variant.updates["domain.environment.resource_enabled"] is True
        assert variant.updates["domain.environment.resource_initial"] == pytest.approx(
            60.0
        )
        assert variant.updates[
            "domain.environment.resource_recovery_rate"
        ] == pytest.approx(0.03)
        assert variant.updates[
            "domain.environment.resource_extraction_per_defector"
        ] == pytest.approx(0.05)

    for variant_name in {
        "revision_precommitment_peer_evidence_resource_pressure_envw2p0",
        "revision_precommitment_peer_evidence_resource_lookahead_envw10p0",
        *expected_threshold_weights,
    }:
        updates = variants[variant_name].updates
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates["model.policy.domain.objective.mode"] == "state_continuation"
        assert updates["model.policy.domain.objective.material_weight"] == pytest.approx(
            1.0
        )
        assert updates["model.policy.domain.objective.social_weight"] == pytest.approx(
            0.5
        )
        assert updates["model.policy.domain.objective.welfare_weight"] == pytest.approx(
            2.0
        )
        assert updates["model.policy.domain.objective.clip_abs"] is None
        assert updates["model.policy.domain.basin_credit.enabled"] is True
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )


def test_toy4_resource_threshold_hardening_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy4_resource_threshold_hardening_quick.yaml")
    )

    assert manifest.label == "toy4_resource_threshold_hardening_quick"
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_threshold_hardening"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases["toy4_resource_threshold_hardening"].final_ceiling_min_hits
        == 4
    )
    assert criteria.cases[
        "toy4_resource_threshold_hardening"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_threshold_hardening"
    assert case.nabm_group == "resource_threshold_hardening"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0"
        ),
        "revision_precommitment_peer_evidence_resource_threshold_local_envw2p0",
    }
    assert variants["reputation_imitation_resource_p0p35"].group == "baseline"
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_threshold_population_envw2p0"
        ].group
        == "diagnostic"
    )
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_threshold_local_envw2p0"
        ].group
        == "resource_threshold_hardening"
    )

    population_updates = variants[
        "revision_precommitment_peer_evidence_resource_threshold_population_envw2p0"
    ].updates
    local_updates = variants[
        "revision_precommitment_peer_evidence_resource_threshold_local_envw2p0"
    ].updates
    assert (
        population_updates["model.policy.domain.resource_environment_threshold_scope"]
        == "population"
    )
    assert (
        local_updates["model.policy.domain.resource_environment_threshold_scope"]
        == "local"
    )

    for updates in (population_updates, local_updates):
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates["model.policy.domain.objective.mode"] == "state_continuation"
        assert updates["model.policy.domain.objective.profile"] == "custom"
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(2.0)
        )
        assert updates[
            "model.policy.domain.resource_environment_pressure_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_lookahead_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert updates["model.policy.domain.basin_credit.enabled"] is True
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )
        assert updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert updates["domain.environment.resource_enabled"] is True
        assert updates["domain.environment.resource_initial"] == pytest.approx(60.0)
        assert updates[
            "domain.environment.resource_recovery_rate"
        ] == pytest.approx(0.03)
        assert updates[
            "domain.environment.resource_extraction_per_defector"
        ] == pytest.approx(0.05)


def test_toy4_resource_threshold_noisy_reputation_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy4_resource_threshold_noisy_reputation_stress_quick.yaml")
    )

    assert manifest.label == "toy4_resource_threshold_noisy_reputation_stress_quick"
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_threshold_noisy_reputation_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy4_resource_threshold_noisy_reputation_stress"
        ].final_ceiling_min_hits
        == 4
    )
    assert criteria.cases[
        "toy4_resource_threshold_noisy_reputation_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_threshold_noisy_reputation_stress"
    assert case.nabm_group == "resource_threshold_noisy_reputation_stress"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35_clean",
        "reputation_imitation_resource_p0p35_noisy_s1p0",
        "reputation_imitation_resource_p0p35_noisy_s2p0",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_reputation_noise_s2p0"
        ),
    }
    assert variants["reputation_imitation_resource_p0p35_clean"].group == "baseline"
    assert variants["reputation_imitation_resource_p0p35_noisy_s1p0"].group == (
        "diagnostic"
    )
    assert variants["reputation_imitation_resource_p0p35_noisy_s2p0"].group == (
        "diagnostic"
    )
    assert (
        variants[
            "revision_precommitment_peer_evidence_resource_threshold_population_envw2p0"
        ].group
        == "diagnostic"
    )
    assert (
        variants[
            (
                "revision_precommitment_peer_evidence_resource_threshold_"
                "local_envw2p0_reputation_noise_s2p0"
            )
        ].group
        == "resource_threshold_noisy_reputation_stress"
    )

    clean_updates = variants["reputation_imitation_resource_p0p35_clean"].updates
    noisy_1_updates = variants[
        "reputation_imitation_resource_p0p35_noisy_s1p0"
    ].updates
    noisy_2_updates = variants[
        "reputation_imitation_resource_p0p35_noisy_s2p0"
    ].updates
    local_updates = variants[
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_reputation_noise_s2p0"
        )
    ].updates
    population_updates = variants[
        "revision_precommitment_peer_evidence_resource_threshold_population_envw2p0"
    ].updates

    for updates, expected_noise in (
        (clean_updates, 0.0),
        (noisy_1_updates, 1.0),
        (noisy_2_updates, 2.0),
    ):
        assert updates["model.policy.rule"] == "reputation_imitation"
        assert updates["model.state.reputation.noise"] == pytest.approx(
            expected_noise
        )
        assert updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert updates["domain.environment.resource_enabled"] is True

    for updates, expected_scope in (
        (population_updates, "population"),
        (local_updates, "local"),
    ):
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates[
            "model.policy.domain.resource_environment_threshold_scope"
        ] == expected_scope
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert updates[
            "model.policy.domain.resource_environment_pressure_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_lookahead_weight"
        ] == pytest.approx(0.0)
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(2.0)
        )
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )
        assert updates["domain.environment.resource_enabled"] is True

    assert local_updates["model.state.reputation.noise"] == pytest.approx(2.0)


def test_toy4_resource_threshold_heterogeneous_extraction_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy4_resource_threshold_heterogeneous_extraction_stress_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy4_resource_threshold_heterogeneous_extraction_stress_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_threshold_heterogeneous_extraction_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy4_resource_threshold_heterogeneous_extraction_stress"
        ].final_ceiling_min_hits
        == 4
    )
    assert criteria.cases[
        "toy4_resource_threshold_heterogeneous_extraction_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_threshold_heterogeneous_extraction_stress"
    assert case.nabm_group == "resource_threshold_heterogeneous_extraction_stress"

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35_heterogeneous_extraction_h1p0",
        (
            "reputation_imitation_resource_p0p35_noisy_s2p0_"
            "heterogeneous_extraction_h1p0"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0_heterogeneous_extraction_h1p0"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_heterogeneous_extraction_h1p0"
        ),
    }
    assert (
        variants[
            "reputation_imitation_resource_p0p35_heterogeneous_extraction_h1p0"
        ].group
        == "baseline"
    )
    assert (
        variants[
            (
                "revision_precommitment_peer_evidence_resource_threshold_"
                "local_envw2p0_heterogeneous_extraction_h1p0"
            )
        ].group
        == "resource_threshold_heterogeneous_extraction_stress"
    )

    for variant in variants.values():
        updates = variant.updates
        assert updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert updates["domain.environment.resource_enabled"] is True
        assert updates["domain.environment.resource_initial"] == pytest.approx(60.0)
        assert updates[
            "domain.environment.resource_extraction_heterogeneity"
        ] == pytest.approx(1.0)
        assert (
            updates["domain.environment.resource_extraction_heterogeneity_mode"]
            == "checkerboard"
        )

    noisy_updates = variants[
        (
            "reputation_imitation_resource_p0p35_noisy_s2p0_"
            "heterogeneous_extraction_h1p0"
        )
    ].updates
    assert noisy_updates["model.state.reputation.noise"] == pytest.approx(2.0)

    population_updates = variants[
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0_heterogeneous_extraction_h1p0"
        )
    ].updates
    local_updates = variants[
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_heterogeneous_extraction_h1p0"
        )
    ].updates
    for updates, expected_scope in (
        (population_updates, "population"),
        (local_updates, "local"),
    ):
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates[
            "model.policy.domain.resource_environment_threshold_scope"
        ] == expected_scope
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(2.0)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )


def test_toy4_resource_threshold_sparse_observation_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy4_resource_threshold_sparse_resource_observation_stress_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy4_resource_threshold_sparse_resource_observation_stress_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert criteria.main_group == "resource_threshold_sparse_resource_observation_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy4_resource_threshold_sparse_resource_observation_stress"
        ].final_ceiling_min_hits
        == 4
    )
    assert criteria.cases[
        "toy4_resource_threshold_sparse_resource_observation_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_resource_threshold_sparse_resource_observation_stress"
    assert case.nabm_group == "resource_threshold_sparse_resource_observation_stress"

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "reputation_imitation_resource_p0p35_clean",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0_global_observation"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_global_observation"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_hidden_observation"
        ),
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_local_sustain_observation"
        ),
    }
    assert variants["reputation_imitation_resource_p0p35_clean"].group == "baseline"
    assert (
        variants[
            (
                "revision_precommitment_peer_evidence_resource_threshold_"
                "local_envw2p0_local_sustain_observation"
            )
        ].group
        == "resource_threshold_sparse_resource_observation_stress"
    )

    expected_observation_modes = {
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0_global_observation"
        ): "global",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_global_observation"
        ): "global",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_hidden_observation"
        ): "hidden",
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_local_sustain_observation"
        ): "local_sustain",
    }
    for variant_name, expected_mode in expected_observation_modes.items():
        updates = variants[variant_name].updates
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates[
            "domain.environment.resource_observation_mode"
        ] == expected_mode
        assert updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert updates["domain.environment.resource_enabled"] is True
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(2.0)
        )
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )
    assert variants[
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "population_envw2p0_global_observation"
        )
    ].updates["model.policy.domain.resource_environment_threshold_scope"] == (
        "population"
    )
    assert variants[
        (
            "revision_precommitment_peer_evidence_resource_threshold_"
            "local_envw2p0_local_sustain_observation"
        )
    ].updates["model.policy.domain.resource_environment_threshold_scope"] == "local"


def test_toy4_resource_threshold_noisy_reputation_local_observation_manifest_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy4_resource_threshold_noisy_reputation_"
            "local_observation_stress_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy4_noisy_rep_local_obs_stress_quick"
    )
    assert manifest.seeds == (1, 2, 3, 4, 5)
    assert manifest.epochs == 60
    assert len(manifest.cases) == 1
    assert (
        criteria.main_group == "toy4_noisy_rep_local_obs_stress"
    )
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy4_noisy_rep_local_obs_stress"
        ].final_ceiling_min_hits
        == 4
    )
    assert criteria.cases[
        "toy4_noisy_rep_local_obs_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(60.0)

    case = manifest.cases[0]
    assert case.toy == "toy4"
    assert case.name == "toy4_noisy_rep_local_obs_stress"
    assert case.nabm_group == "toy4_noisy_rep_local_obs_stress"
    assert case.ceiling_value == pytest.approx(0.6)
    assert case.ceiling_tolerance == pytest.approx(0.005)

    variants = {variant.name: variant for variant in case.variants}
    assert set(variants) == {
        "rep_clean",
        "rep_noisy_s2p0",
        "rev_pop_global_obs_noisy_s2p0",
        "rev_local_global_obs_noisy_s2p0",
        "rev_local_hidden_obs_noisy_s2p0",
        "rev_local_sustain_obs_noisy_s2p0",
    }
    assert variants["rep_clean"].group == "baseline"
    assert variants["rep_noisy_s2p0"].group == "diagnostic"
    assert (
        variants["rev_local_sustain_obs_noisy_s2p0"].group
        == "toy4_noisy_rep_local_obs_stress"
    )

    clean_updates = variants["rep_clean"].updates
    noisy_updates = variants["rep_noisy_s2p0"].updates
    assert clean_updates["model.policy.rule"] == "reputation_imitation"
    assert clean_updates["model.state.reputation.noise"] == pytest.approx(0.0)
    assert noisy_updates["model.policy.rule"] == "reputation_imitation"
    assert noisy_updates["model.state.reputation.noise"] == pytest.approx(2.0)

    expected_observation_modes = {
        "rev_pop_global_obs_noisy_s2p0": ("population", "global"),
        "rev_local_global_obs_noisy_s2p0": ("local", "global"),
        "rev_local_hidden_obs_noisy_s2p0": ("local", "hidden"),
        "rev_local_sustain_obs_noisy_s2p0": ("local", "local_sustain"),
    }
    for variant_name, (expected_scope, expected_mode) in (
        expected_observation_modes.items()
    ):
        updates = variants[variant_name].updates
        assert updates["model.policy.rule"] == "neural_policy"
        assert updates["model.policy.neural_update_backend"] == "loop"
        assert updates["model.state.reputation.noise"] == pytest.approx(2.0)
        assert updates[
            "domain.environment.resource_observation_mode"
        ] == expected_mode
        assert updates[
            "model.policy.domain.resource_environment_threshold_scope"
        ] == expected_scope
        assert updates[
            "model.policy.domain.resource_environment_threshold_weight"
        ] == pytest.approx(1.0)
        assert updates[
            "model.policy.domain.resource_environment_pressure_weight"
        ] == pytest.approx(0.0)
        assert updates[
            "model.policy.domain.resource_environment_lookahead_weight"
        ] == pytest.approx(0.0)
        assert updates["model.policy.domain.objective.environment_weight"] == (
            pytest.approx(2.0)
        )
        assert updates["model.policy.domain.basin_credit.objective_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.policy.domain.basin_credit.basin_weight"] == (
            pytest.approx(0.5)
        )
        assert updates["model.coordination.revision_operator_enabled"] is True
        assert updates["model.coordination.precommitment_enabled"] is True
        assert (
            updates["model.coordination.precommitment_peer_evidence_enabled"] is True
        )
        assert updates[
            "domain.environment.initial_action_probability"
        ] == pytest.approx(0.35)
        assert updates["domain.environment.resource_enabled"] is True


def test_toy24_revision_operator_manifest_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy24_revision_operator_quick.yaml")
    )

    assert manifest.label == "toy24_revision_operator_quick"
    assert criteria.main_group == "revision"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert criteria.cases["toy2_revision_operator"].final_ceiling_min_hits == 3
    assert criteria.cases[
        "toy2_revision_operator"
    ].mean_time_to_ceiling_lt == pytest.approx(10.0)
    assert criteria.cases["toy4_revision_operator"].final_ceiling_min_hits == 2
    assert criteria.cases[
        "toy4_revision_operator"
    ].mean_time_to_ceiling_lt == pytest.approx(12.0)


def test_toy24_revision_operator_controls_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy24_revision_operator_controls_quick.yaml")
    )

    assert manifest.label == "toy24_revision_operator_controls_quick"
    assert criteria.main_group == "control"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert criteria.cases[
        "toy2_revision_operator_controls"
    ].final_ceiling_min_hits == 3
    assert criteria.cases[
        "toy2_revision_operator_controls"
    ].mean_time_to_ceiling_lt == pytest.approx(10.0)
    assert criteria.cases[
        "toy4_revision_operator_controls"
    ].final_ceiling_min_hits == 2
    assert criteria.cases[
        "toy4_revision_operator_controls"
    ].mean_time_to_ceiling_lt == pytest.approx(12.0)


def test_toy24_revision_operator_precommitment_controls_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy24_revision_operator_precommitment_controls_quick.yaml"
        )
    )

    assert manifest.label == "toy24_revision_operator_precommitment_controls_quick"
    assert criteria.main_group == "precommitment_control"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert criteria.cases[
        "toy2_revision_operator_precommitment_controls"
    ].final_ceiling_min_hits == 3
    assert criteria.cases[
        "toy2_revision_operator_precommitment_controls"
    ].mean_time_to_ceiling_lt == pytest.approx(10.0)
    assert criteria.cases[
        "toy4_revision_operator_precommitment_controls"
    ].final_ceiling_min_hits == 2
    assert criteria.cases[
        "toy4_revision_operator_precommitment_controls"
    ].mean_time_to_ceiling_lt == pytest.approx(12.0)


def test_toy5_readiness_direction_control_stress_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy5_readiness_direction_control_stress.yaml")
    )

    assert manifest.label == "toy5_readiness_direction_control_stress"
    assert criteria.main_group == "direction_control"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_readiness_direction_control_stress"
        ].final_ceiling_min_hits
        == 10
    )
    assert criteria.cases[
        "toy5_readiness_direction_control_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)


def test_toy5_neural_readiness_direction_control_stress_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path("experiments/evidence/toy5_neural_readiness_direction_control_stress.yaml")
    )

    assert manifest.label == "toy5_neural_readiness_direction_control_stress"
    assert criteria.main_group == "direction_control"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_neural_readiness_direction_control_stress"
        ].final_ceiling_min_hits
        == 10
    )
    assert criteria.cases[
        "toy5_neural_readiness_direction_control_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)


def test_toy5_neural_threshold_target_direction_control_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_direction_control_stress.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_direction_control_stress"
    assert criteria.main_group == "direction_control"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_neural_threshold_target_direction_control_stress"
        ].final_ceiling_min_hits
        == 10
    )
    assert criteria.cases[
        "toy5_neural_threshold_target_direction_control_stress"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)


def test_toy5_neural_threshold_target_frontier_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_frontier_stability.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_frontier_stability"
    assert criteria.main_group == "threshold_frontier"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_neural_threshold_target_frontier_stability"
        ].final_ceiling_min_hits
        == 8
    )
    assert criteria.cases[
        "toy5_neural_threshold_target_frontier_stability"
    ].mean_time_to_ceiling_lt == pytest.approx(40.0)


def test_toy5_neural_threshold_target_combined_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_safety_frontier_combined.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_safety_frontier_combined"
    assert criteria.main_group == "directional_threshold_target"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_target_no_seed_safety"
        ].final_ceiling_min_hits
        == 10
    )
    assert criteria.cases[
        "toy5_threshold_target_no_seed_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    assert (
        criteria.cases[
            "toy5_threshold_target_seeded_frontier_spread"
        ].final_ceiling_min_hits
        == 8
    )
    assert criteria.cases[
        "toy5_threshold_target_seeded_frontier_spread"
    ].mean_time_to_ceiling_lt == pytest.approx(40.0)


def test_toy5_neural_threshold_target_robustness_success_criteria_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_structural_robustness_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy5_neural_threshold_target_structural_robustness_quick"
    )
    assert criteria.main_group == "directional_threshold_target_robust"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_target_no_seed_heterogeneous_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_no_seed_heterogeneous_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    assert (
        criteria.cases[
            "toy5_threshold_target_random_seed_frontier_spread"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_random_seed_frontier_spread"
    ].mean_time_to_ceiling_lt == pytest.approx(45.0)
    assert (
        criteria.cases[
            "toy5_threshold_target_heterogeneous_frontier_spread"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_heterogeneous_frontier_spread"
    ].mean_time_to_ceiling_lt == pytest.approx(45.0)


def test_toy5_neural_threshold_target_lattice_wavefront_success_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_lattice_wavefront_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_lattice_wavefront_quick"
    assert criteria.main_group == "directional_threshold_target_wavefront"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_target_wavefront_no_seed_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_wavefront_no_seed_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    assert (
        criteria.cases[
            "toy5_threshold_target_lattice_wavefront_spread"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_lattice_wavefront_spread"
    ].mean_time_to_ceiling_lt == pytest.approx(30.0)


def test_toy5_neural_threshold_target_wavefront_topology_success_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_wavefront_topology_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_wavefront_topology_quick"
    assert criteria.main_group == "directional_threshold_target_wavefront_topology"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_target_wavefront_topology_no_seed_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_wavefront_topology_no_seed_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    for case_name in (
        "toy5_threshold_target_lattice_k4_wavefront_spread",
        "toy5_threshold_target_lattice_k8_wavefront_spread",
        "toy5_threshold_target_rewired_p0p02_wavefront_spread",
    ):
        assert criteria.cases[case_name].final_ceiling_min_hits == 5
        assert criteria.cases[
            case_name
        ].mean_time_to_ceiling_lt == pytest.approx(45.0)


def test_toy5_neural_threshold_target_wavefront_stress_success_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_wavefront_stress_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_wavefront_stress_quick"
    assert criteria.main_group == "directional_threshold_target_wavefront_stress"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    for case_name in (
        "toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread",
        "toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread",
        "toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread",
    ):
        assert criteria.cases[case_name].final_ceiling_min_hits == 5
        assert criteria.cases[
            case_name
        ].mean_time_to_ceiling_lt == pytest.approx(45.0)


def test_toy5_neural_threshold_target_threshold_aware_wavefront_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_threshold_aware_wavefront_quick.yaml"
        )
    )

    assert (
        manifest.label
        == "toy5_neural_threshold_target_threshold_aware_wavefront_quick"
    )
    assert (
        criteria.main_group
        == "directional_threshold_target_threshold_aware_wavefront"
    )
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    for case_name in (
        "toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread",
        "toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread",
        "toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread",
    ):
        assert criteria.cases[case_name].final_ceiling_min_hits == 5
        assert criteria.cases[
            case_name
        ].mean_time_to_ceiling_lt == pytest.approx(45.0)


def test_toy5_neural_threshold_target_threshold_aware_grid_contract() -> None:
    manifest, criteria = load_gate_manifest(
        Path(
            "experiments/evidence/"
            "toy5_neural_threshold_target_threshold_aware_grid_quick.yaml"
        )
    )

    assert manifest.label == "toy5_neural_threshold_target_threshold_aware_grid_quick"
    assert criteria.main_group == "directional_threshold_target_threshold_aware_grid"
    assert criteria.require_without_teacher_bootstrap_replay is True
    assert (
        criteria.cases[
            "toy5_threshold_aware_grid_no_seed_heterogeneous_safety"
        ].final_ceiling_min_hits
        == 5
    )
    assert criteria.cases[
        "toy5_threshold_aware_grid_no_seed_heterogeneous_safety"
    ].mean_time_to_ceiling_lt == pytest.approx(1.0)
    for case_name in (
        "toy5_threshold_aware_grid_lattice_k4_h0p85_spread",
        "toy5_threshold_aware_grid_lattice_k4_h0p95_spread",
        "toy5_threshold_aware_grid_lattice_k6_h0p85_spread",
        "toy5_threshold_aware_grid_lattice_k6_h0p95_spread",
        "toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread",
        "toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread",
    ):
        assert criteria.cases[case_name].final_ceiling_min_hits == 5
        assert criteria.cases[
            case_name
        ].mean_time_to_ceiling_lt == pytest.approx(45.0)
