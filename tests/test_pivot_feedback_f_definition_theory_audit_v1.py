from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_definition_theory_audit_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_definition_theory_audit_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_definition_theory_audit_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v5.md"
PRODUCTION = Path("/ssd/pivot-feedback-f-definition-theory-audit-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-definition-theory-audit-v1-20260804.verification.json")
FINAL = Path("/ssd/pivot-feedback-f-final-panel-conformance-v1-20260804")
CUTOFF = Path("/ssd/pivot-feedback-cutoff-primitives-v0-full-20260723")
AUTHORS = Path("/ssd/openalex-g1-author-history-v3-full-20260722")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load(BUILDER, "f_definition_audit_builder_test")
VERIFY = _load(VERIFIER, "f_definition_audit_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_retains_primary_but_repairs_measurement_normalization_alignment() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert contract["schema_version"] == BUILD.CONFIG_SCHEMA
    assert contract["primary_estimand_decision"]["primary_work_inclusion"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT"
    assert contract["primary_estimand_decision"]["mixed_incumbent_new_team_action"] == "EXCLUDE_FROM_PRIMARY"
    assert contract["aligned_primary_replay"]["candidate_universe"].startswith("ALL_ELIGIBLE")
    assert contract["aligned_primary_replay"]["aggregate_median_mad_reuse"] == "FORBIDDEN"
    assert contract["aligned_primary_replay"]["minimum_distinct_current_snapshot_subfield_families"] == 3
    assert contract["aligned_primary_replay"]["threshold_inclusive"] == 3.0
    assert contract["aligned_primary_replay"]["threshold_relaxation"] is False
    assert contract["authorization_after_verified_audit"]["strict_incumbent_aligned_all_topic_rescreen"] == "AUTHORIZED_NEXT"
    assert contract["authorization_after_verified_audit"]["operator_only_private_holdout_materialization"] == "NOT_AUTHORIZED"
    assert contract["completion_accounting"]["pass_f"] is False
    assert contract["completion_accounting"]["pass_g0"] is False


def test_production_composition_and_stage_diagnosis_are_exact() -> None:
    report = json.loads((PRODUCTION / "definition_theory_audit.json").read_text(encoding="utf-8"))

    assert report["status"] == BUILD.STATUS
    assert report["work_composition_totals"]["reference"]["aggregate_author_linked_work_count"] == 494218
    assert report["work_composition_totals"]["reference"]["all_pre_signal_incumbent_work_count"] == 39507
    assert report["work_composition_totals"]["recent"]["aggregate_author_linked_work_count"] == 1141266
    assert report["work_composition_totals"]["recent"]["all_pre_signal_incumbent_work_count"] == 31221
    assert report["net_growth_decomposition"]["no_pre_signal_incumbent_recent_minus_reference"] == 551001
    assert report["stage_decomposition"]["strict_recent_not_greater_episode_count"] == 143
    assert report["stage_decomposition"]["strict_with_aggregate_normalization_surge_episode_count"] == 15
    assert report["stage_decomposition"]["network_leaveout_surge_episode_count"] == 19
    assert report["stage_decomposition"]["peer_leaveout_created_episode_count"] == 4
    assert report["stage_decomposition"]["peer_leaveout_killed_episode_count"] == 0
    assert report["stage_decomposition"]["focal_leaveout_incidence_count"] == 0
    assert report["scope"]["raw_author_or_work_ids_emitted"] is False
    assert report["scope"]["pass_f"] is False
    assert report["scope"]["pass_g0"] is False


def test_repository_summary_binds_production_and_independent_raw_replay() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))

    assert summary["production_artifact"]["output_receipt_sha256"] == _sha(PRODUCTION / "definition_theory_audit.outputs.json")
    assert summary["production_artifact"]["composition_sha256"] == _sha(PRODUCTION / "episode_work_composition.parquet")
    assert summary["repository_bindings"]["config_sha256"] == _sha(CONFIG)
    assert summary["repository_bindings"]["builder_sha256"] == _sha(BUILDER)
    assert summary["repository_bindings"]["verifier_sha256"] == _sha(VERIFIER)
    assert summary["independent_verification"]["verification_sha256"] == _sha(VERIFICATION)
    assert verification["status"] == VERIFY.VERIFY_STATUS
    assert verification["composition_rows_replayed"] == 388
    assert verification["raw_source_composition_replay_match"] is True
    assert verification["pass_f"] is False
    assert verification["pass_g0"] is False


def test_verifier_is_independent_of_builder() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_definition_theory_audit_v1" not in source
    assert "raw composition replay mismatch" in source
    assert "read_parquet" in source


def test_verifier_rejects_improper_f_pass_before_raw_replay(tmp_path: Path) -> None:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    receipt_path = output / "definition_theory_audit.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(VERIFY.DefinitionAuditVerificationError, match="unsafe receipt state: pass_f"):
        VERIFY.verify(
            config_path=CONFIG,
            final_root=FINAL,
            cutoff_root=CUTOFF,
            author_history_root=AUTHORS,
            output_root=output,
            verification_path=tmp_path / "should-not-exist.json",
        )


def test_human_ledger_names_the_single_next_action_and_hard_stops() -> None:
    text = LEDGER.read_text(encoding="utf-8")

    assert "all eligible topics" in text
    assert "aggregate field-year median/MAD" in text
    assert "`pass_f=false` and `pass_g0=false`" in text
    assert "No confirmatory entry or persistence label" in text
    assert "neither a\nnull nor a positive association" in text
