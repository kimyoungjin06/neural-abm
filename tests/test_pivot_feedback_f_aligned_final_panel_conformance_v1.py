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
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_final_panel_conformance_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_aligned_final_panel_conformance_v1.py"
RUNTIME_HELPER = SCRIPTS / "build_pivot_feedback_f_final_panel_conformance_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_aligned_final_panel_conformance_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v7.md"
ALIGNED = Path("/ssd/pivot-feedback-f-aligned-strict-incumbent-rescreen-v1-20260804")
PRODUCTION = Path("/ssd/pivot-feedback-f-aligned-final-panel-conformance-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-aligned-final-panel-conformance-v1-20260804.verification.json")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load(VERIFIER, "f_aligned_final_panel_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_output(tmp_path: Path) -> Path:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    return output


def _rebind_receipt(output: Path) -> None:
    receipt_path = output / "aligned_final_panel_conformance.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        artifact = output / item["relative_path"]
        item["bytes"] = artifact.stat().st_size
        item["sha256"] = _sha(artifact)
        if artifact.suffix == ".parquet":
            item["rows"] = pq.ParquetFile(artifact).metadata.num_rows
    ordered = sorted(receipt["artifacts"], key=lambda row: row["relative_path"])
    tree = [
        (row["relative_path"], row["bytes"], row["sha256"])
        for row in ordered
    ]
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        VERIFY._canonical_bytes(tree)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)


def test_contract_freezes_aligned_panel_without_fallback_or_relaxation() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    panel = contract["panel_contract"]
    assert contract["schema_version"] == VERIFY.CONFIG_SCHEMA
    assert panel["episode_source"] == "RETAINED_ALIGNED_STRICT_INCUMBENT_CANDIDATES_ONLY"
    assert panel["selected_fields_fixed_by_verified_aligned_rescreen"] == [16, 32, 11, 27]
    assert panel["selected_field_replacement_during_leaveout"] == "FORBIDDEN"
    assert panel["later_reserve_fallback_after_aligned_selection"] == "FORBIDDEN"
    assert panel["aligned_screen_normalization"]["aggregate_location_or_scale_reuse"] == "FORBIDDEN"
    assert panel["aligned_screen_normalization"]["threshold_inclusive"] == 3.0
    assert panel["aligned_screen_normalization"]["threshold_relaxation"] is False
    assert panel["final_slot_requires_leaveout_surge_subfield_families"] == 3
    assert panel["team_cap"] == "NONE"
    assert contract["support_replay"]["label_independent"]["minimum_total_opportunity_rows"] == 500
    assert contract["completion_accounting"]["pass_f"] is False
    assert contract["completion_accounting"]["pass_g0"] is False


def test_production_closes_all_aligned_episodes_and_four_slots() -> None:
    report = json.loads(
        (PRODUCTION / "aligned_final_panel_conformance.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == (
        "PASS_OUTCOME_BLIND_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_"
        "CONFORMANCE_ATOMIC_PARTITION_AUTHORIZED_NO_F_PASS"
    )
    accounting = report["replay_accounting"]
    assert accounting["aligned_episode_rows"] == 65
    assert accounting["aligned_count_closure_rows"] == 65
    assert accounting["risk_opportunity_rows"] == 247_970_614
    assert accounting["focal_excluded_attention_work_incidences"] == 0
    assert accounting["network_leaveout_surge_opportunity_rows"] == 247_963_432
    assert accounting["episodes_passing_label_independent_support"] == 65
    assert accounting["episodes_with_leaveout_surge"] == 65
    assert accounting["maximum_declared_team_size"] == 5000
    assert accounting["prior_focal_work_relations_above_team_size_100"] == 50_932_406
    assert accounting["team_cap_applied"] is False
    assert [row["aligned_selected_field_id"] for row in report["slot_support"]] == [16, 32, 11, 27]
    assert [row["leaveout_conformed_subfield_family_count"] for row in report["slot_support"]] == [3, 3, 7, 16]
    assert all(row["final_panel_slot_pass"] for row in report["slot_support"])
    assert report["scope"]["pass_f"] is False
    assert report["scope"]["pass_g0"] is False


def test_exact_production_passes_independent_aligned_verification(tmp_path: Path) -> None:
    result = VERIFY.verify(
        config_path=CONFIG,
        aligned_root=ALIGNED,
        output_root=PRODUCTION,
        verification_output=tmp_path / "verification.json",
    )

    assert result["status"] == VERIFY.PASS_STATUS
    assert result["episode_verification"] == {
        "episode_rows": 65,
        "aligned_count_closure_rows": 65,
        "risk_opportunity_rows": 247_970_614,
        "peer_risk_rows": 40_753_340,
        "network_excluded_attention_work_incidences": 2_229_480,
        "network_leaveout_surge_opportunity_rows": 247_963_432,
        "non_surge_risk_rows": 7_182,
        "episodes_with_non_surge_risk_rows": 19,
        "episodes_with_any_leaveout_count_change": 65,
        "episodes_with_median_leaveout_count_change": 0,
    }
    assert result["selected_fields_verified"] == [16, 32, 11, 27]
    assert result["leaveout_family_counts_verified"] == [3, 3, 7, 16]
    assert result["raw_id_columns_absent"] is True
    assert result["final_panel_conformance_complete"] is True
    assert result["pass_f"] is False
    assert result["pass_g0"] is False
    assert result["g4_open"] is False


def test_verifier_is_independent_of_aligned_builder() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_aligned_final_panel_conformance_v1" not in source
    assert "aligned count closure" in source
    assert "leaveout monotonicity" in source
    assert "raw ID in episode output" in source


def test_verifier_rejects_rebound_surge_presence_mismatch(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "aligned_final_panel_episode_conformance.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    index = next(i for i, row in enumerate(rows) if row["leaveout_surge_present"])
    rows[index]["leaveout_surge_present"] = False
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    path.chmod(0o600)
    _rebind_receipt(output)

    with pytest.raises(
        VERIFY.AlignedFinalPanelVerificationError,
        match="surge presence flag",
    ):
        VERIFY.verify(
            config_path=CONFIG,
            aligned_root=ALIGNED,
            output_root=output,
            verification_output=tmp_path / "should-not-exist.json",
        )


def test_verifier_rejects_receipt_that_improperly_passes_f(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    receipt_path = output / "aligned_final_panel_conformance.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)

    with pytest.raises(
        VERIFY.AlignedFinalPanelVerificationError,
        match="unsafe receipt state: pass_f",
    ):
        VERIFY.verify(
            config_path=CONFIG,
            aligned_root=ALIGNED,
            output_root=output,
            verification_output=tmp_path / "should-not-exist.json",
        )


def test_repository_summary_and_ledger_bind_fail_closed_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    production = summary["production_artifact"]
    bindings = summary["repository_bindings"]
    assert production["output_receipt_sha256"] == _sha(
        PRODUCTION / "aligned_final_panel_conformance.outputs.json"
    )
    assert bindings["config_sha256"] == _sha(CONFIG)
    assert bindings["builder_sha256"] == _sha(BUILDER)
    assert bindings["runtime_helper_sha256"] == _sha(RUNTIME_HELPER)
    assert bindings["verifier_sha256"] == _sha(VERIFIER)
    assert summary["independent_verification"]["verification_sha256"] == _sha(
        VERIFICATION
    )
    assert verification["status"] == VERIFY.PASS_STATUS
    assert summary["fail_closed_state"]["final_panel_conformance_complete"] is True
    assert summary["fail_closed_state"]["atomic_partition_assignment_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "3, 3, 7, and 16" in ledger
    assert "Exactly one next artifact is authorized: atomic assignment" in ledger
    assert "`pass_f=false` and `pass_g0=false`" in ledger
