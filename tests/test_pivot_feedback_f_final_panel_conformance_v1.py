from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_final_panel_conformance_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_final_panel_conformance_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_final_panel_conformance_v1.py"
PRODUCTION = Path("/ssd/pivot-feedback-f-final-panel-conformance-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-final-panel-conformance-v1-20260804.verification.json")
ASSIGNMENT = Path("/ssd/pivot-feedback-f-field-episode-assignment-v1-20260803")
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.json"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load(BUILDER, "f_final_panel_builder_test")
VERIFY = _load(VERIFIER, "f_final_panel_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_output(tmp_path: Path) -> Path:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    return output


def _rebind_receipt(output: Path) -> None:
    receipt_path = output / "final_panel_conformance.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        artifact = output / item["relative_path"]
        item["bytes"] = artifact.stat().st_size
        item["sha256"] = _sha(artifact)
        if artifact.suffix == ".parquet":
            item["rows"] = pq.ParquetFile(artifact).metadata.num_rows
    ordered = sorted(receipt["artifacts"], key=lambda row: row["relative_path"])
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        VERIFY._canonical_bytes(
            [(row["relative_path"], row["sha256"]) for row in ordered]
        )
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_contract_removes_circular_outcome_conditioned_f_gate() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert contract["schema_version"] == BUILD.CONFIG_SCHEMA
    assert contract["completion_accounting"]["this_component_is_new_gate"] is False
    assert contract["completion_accounting"]["this_component_alone_may_pass_f"] is False
    assert contract["support_replay"]["label_dependent"]["may_block_f"] is False
    assert contract["support_replay"]["label_dependent"]["status_during_f"].startswith("DEFERRED")
    assert contract["panel_contract"]["team_cap"] == "NONE"
    assert contract["panel_contract"]["attention_work_rule"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT"
    assert contract["authorization_after_block"]["threshold_relaxation"] == "NOT_AUTHORIZED"


def test_builder_replays_only_aggregate_screen_eligible_frozen_field_attempts() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    episodes, slots, binding = BUILD._load_assignment(
        ASSIGNMENT,
        contract["verified_assignment_parent"],
    )

    assert len(episodes) == 194
    assert len(slots) == 4
    disposition = binding["aggregate_screen_field_disposition"]
    assert [row["field_id"] for row in disposition["g1_domain_slot_01"] if row["aggregate_screen_eligible"]] == [17]
    assert [row["field_id"] for row in disposition["g1_domain_slot_02"] if row["aggregate_screen_eligible"]] == [32]
    assert [row["field_id"] for row in disposition["g1_domain_slot_03"] if row["aggregate_screen_eligible"]] == [11]
    assert [row["field_id"] for row in disposition["g4_domain_author_cold_slot"] if row["aggregate_screen_eligible"]] == [36, 27]


def test_verifier_is_independent_of_builder_and_raw_source_rejoin() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_final_panel_conformance_v1" not in source
    assert "read_parquet('/ssd/openalex" not in source
    assert "assignment_and_reserve_order_replayed" in source
    assert "raw ID in episode output" in source


def test_exact_blocked_production_passes_independent_aggregate_verification(tmp_path: Path) -> None:
    result = VERIFY.verify(
        config_path=CONFIG,
        assignment_root=ASSIGNMENT,
        output_root=PRODUCTION,
        verification_path=tmp_path / "verification.json",
    )

    assert result["status"] == VERIFY.BLOCKED_STATUS
    assert result["episode_rows_verified"] == 194
    assert result["slot_rows_verified"] == 4
    assert result["final_panel_conformance_complete"] is False
    assert result["pass_f"] is False
    assert result["pass_g0"] is False
    assert result["g4_open"] is False


def test_repository_summary_binds_blocked_receipt_and_verification() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["status"].startswith("BLOCKED_FINAL_PANEL")
    assert summary["production_artifact"]["output_receipt_sha256"] == _sha(
        PRODUCTION / "final_panel_conformance.outputs.json"
    )
    assert summary["independent_verification"]["verification_sha256"] == _sha(VERIFICATION)
    assert [row["leaveout_conformed_subfield_family_count"] for row in summary["slot_reconciliation"]] == [1, 1, 2, 6]
    assert summary["slot_reconciliation"][-1]["final_selected_field_id"] == 27
    assert summary["fail_closed_state"]["final_panel_conformance_complete"] is False
    assert summary["fail_closed_state"]["holdout_byte_seal_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False


def test_verifier_rejects_rebound_surge_presence_mismatch(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "final_panel_episode_conformance.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    index = next(i for i, row in enumerate(rows) if row["leaveout_surge_present"])
    rows[index]["leaveout_surge_present"] = False
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    _rebind_receipt(output)

    with pytest.raises(VERIFY.FinalPanelVerificationError, match="surge presence flag"):
        VERIFY.verify(
            config_path=CONFIG,
            assignment_root=ASSIGNMENT,
            output_root=output,
            verification_path=tmp_path / "should-not-exist.json",
        )


def test_verifier_rejects_receipt_that_improperly_passes_f(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "final_panel_conformance.outputs.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(VERIFY.FinalPanelVerificationError, match="unsafe receipt state: pass_f"):
        VERIFY.verify(
            config_path=CONFIG,
            assignment_root=ASSIGNMENT,
            output_root=output,
            verification_path=tmp_path / "should-not-exist.json",
        )
