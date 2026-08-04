#!/usr/bin/env python3
"""Independently verify aligned strict-incumbent final-panel conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_final_panel_conformance_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_aligned_final_panel_conformance.v1"
VERIFY_SCHEMA = "pivot_feedback_f_aligned_final_panel_conformance_verification.v1"
PASS_STATUS = "PASS_INDEPENDENT_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_CONFORMANCE_NO_F_PASS"
BLOCKED_STATUS = "PASS_INDEPENDENT_BLOCKED_ALIGNED_FINAL_PANEL_LEAVEOUT_OR_SUPPORT_CONFORMANCE_NO_F_PASS"

EXPECTED_FILES = {
    "ALIGNED_FINAL_PANEL_CONFORMANCE.md",
    "aligned_final_panel_conformance.json",
    "aligned_final_panel_conformance.outputs.json",
    "aligned_final_panel_episode_conformance.parquet",
    "aligned_final_panel_slot_support.parquet",
}

EPISODE_FIELDS = [
    ("episode_id", "uint16", False),
    ("slot_id", "string", False),
    ("temporal_partition_id", "string", False),
    ("field_attempt_order", "uint8", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("topic_name", "string", False),
    ("subfield_id", "uint16", False),
    ("subfield_name", "string", False),
    ("decision_year", "uint16", False),
    ("aligned_reference_count", "uint64", False),
    ("aligned_recent_count", "uint64", False),
    ("aligned_acceleration", "double", False),
    ("aligned_field_year_median", "double", False),
    ("aligned_field_year_mad", "double", False),
    ("aligned_modified_z", "double", False),
    ("all_incumbent_reference_work_count", "uint64", False),
    ("all_incumbent_recent_work_count", "uint64", False),
    ("aligned_count_closure", "bool", False),
    ("risk_opportunity_count", "uint64", False),
    ("risk_with_prior_target_active_peer_count", "uint64", False),
    ("prior_target_active_peer_relation_count", "uint64", False),
    ("focal_excluded_attention_work_incidence_count", "uint64", False),
    ("network_excluded_attention_work_incidence_count", "uint64", False),
    ("network_leaveout_reference_count_min", "uint64", False),
    ("network_leaveout_reference_count_median", "double", False),
    ("network_leaveout_reference_count_max", "uint64", False),
    ("network_leaveout_recent_count_min", "uint64", False),
    ("network_leaveout_recent_count_median", "double", False),
    ("network_leaveout_recent_count_max", "uint64", False),
    ("network_leaveout_surge_opportunity_count", "uint64", False),
    ("leaveout_surge_present", "bool", False),
    ("minimum_total_support_required", "uint32", False),
    ("label_independent_support_pass", "bool", False),
    ("label_dependent_support_status", "string", False),
    ("sealed_slot", "bool", False),
]

SLOT_FIELDS = [
    ("slot_id", "string", False),
    ("temporal_partition_id", "string", False),
    ("aligned_selected_field_id", "uint8", False),
    ("aligned_candidate_episode_count", "uint16", False),
    ("aligned_subfield_family_count", "uint16", False),
    ("risk_opportunity_count", "uint64", False),
    ("episodes_passing_label_independent_support", "uint16", False),
    ("all_episodes_pass_label_independent_support", "bool", False),
    ("leaveout_surge_episode_count", "uint16", False),
    ("leaveout_conformed_subfield_family_count", "uint16", False),
    ("minimum_leaveout_subfield_families_required", "uint16", False),
    ("final_panel_slot_pass", "bool", False),
    ("status", "string", False),
    ("sealed_slot", "bool", False),
]


class AlignedFinalPanelVerificationError(RuntimeError):
    """Fail-closed aligned final-panel verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignedFinalPanelVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _schema_version(path: Path) -> str:
    return (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"schema_version", b"").decode()


def _assert_schema(path: Path, expected: Sequence[tuple[str, str, bool]], version: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    observed = [(field.name, str(field.type), field.nullable) for field in schema]
    require(observed == list(expected), f"schema fields: {path.name}")
    require((schema.metadata or {}).get(b"schema_version", b"").decode() == version, f"schema version: {path.name}")


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}


def _verify_output_tree(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(root.is_dir() and not root.is_symlink(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"non-file output: {path.name}")
        require((path.stat().st_mode & 0o777) == 0o600, f"output mode: {path.name}")
    receipt_path = root / "aligned_final_panel_conformance.outputs.json"
    receipt = _json(receipt_path)
    contract = _mapping(config["output_contract"], "output contract")
    require(receipt.get("schema_version") == contract["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") in (contract["passing_status"], contract["blocking_status"]), "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt last flag")
    artifacts = _artifact_map(receipt)
    require(set(artifacts) == EXPECTED_FILES - {receipt_path.name}, "receipt artifact set")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count")
    tree = []
    for relative in sorted(artifacts):
        path = root / relative
        row = artifacts[relative]
        require(path.stat().st_size == row["bytes"], f"artifact bytes: {relative}")
        require(_sha256(path) == row["sha256"], f"artifact SHA: {relative}")
        if path.suffix == ".parquet":
            require(pq.ParquetFile(path).metadata.num_rows == row["rows"], f"artifact rows: {relative}")
        tree.append((relative, row["bytes"], row["sha256"]))
    require(hashlib.sha256(_canonical_bytes(tree)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree SHA")
    require(receipt_path.stat().st_mtime_ns > max((root / name).stat().st_mtime_ns for name in artifacts), "receipt not written last")
    require(receipt["confirmatory_entry_or_persistence_labels_read"] == 0, "unsafe receipt state: confirmatory labels")
    for key in ("selected_field_replaced", "raw_author_or_work_ids_emitted", "team_cap_applied", "holdout_byte_seal_complete", "model_fitted", "g4_open", "pass_f", "pass_g0"):
        require(receipt[key] is False, f"unsafe receipt state: {key}")
    return receipt, artifacts


def _load_parent(root: Path, binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"] and root.is_dir() and not root.is_symlink(), "aligned parent root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "aligned parent receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "aligned parent receipt schema")
    require(receipt.get("status") == binding["required_status"], "aligned parent receipt status")
    candidate_path = root / str(binding["candidate_basename"])
    slot_path = root / str(binding["slot_basename"])
    require(_sha256(candidate_path) == binding["candidate_sha256"], "aligned parent candidate SHA")
    require(_sha256(slot_path) == binding["slot_sha256"], "aligned parent slot SHA")
    require(_schema_version(candidate_path) == binding["candidate_schema_version"], "aligned parent candidate schema")
    require(_schema_version(slot_path) == binding["slot_schema_version"], "aligned parent slot schema")
    verification = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == binding["verification_sha256"], "aligned parent verification SHA")
    require(_json(verification).get("status") == binding["verification_status"], "aligned parent verification status")
    candidates = [dict(row) for row in pq.read_table(candidate_path).to_pylist() if row["retained_for_aligned_panel"]]
    slots = pq.read_table(slot_path).to_pylist()
    require(len(candidates) == binding["retained_candidate_rows"] == 65 and len(slots) == 4, "aligned parent row closure")
    return candidates, slots


def _expected_episode_rows(
    candidates: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    slot_order = {str(row["slot_id"]): index for index, row in enumerate(slots)}
    return sorted(
        candidates,
        key=lambda row: (
            slot_order[str(row["slot_id"])],
            int(row["decision_year"]),
            int(row["subfield_id"]),
            int(row["topic_id"]),
        ),
    )


def _verify_episodes(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _assert_schema(path, EPISODE_FIELDS, config["output_contract"]["episode_schema_version"])
    rows = pq.read_table(path).to_pylist()
    require(len(rows) == len(expected) == 65, "episode row closure")
    forbidden = {"author_id", "author_numeric_id", "work_id", "work_numeric_id", "focal_author_id", "peer_author_id"}
    require(not (forbidden & set(pq.ParquetFile(path).schema_arrow.names)), "raw ID in episode output")
    multiplier = float(config["panel_contract"]["aligned_screen_normalization"]["modified_z_multiplier"])
    threshold = float(config["panel_contract"]["aligned_screen_normalization"]["threshold_inclusive"])
    support_minimum = int(config["support_replay"]["label_independent"]["minimum_total_opportunity_rows"])
    changed_episodes = 0
    median_changed_episodes = 0
    non_surge_risk_rows = 0
    episodes_with_non_surge_risk_rows = 0
    peer_risk_rows = 0
    excluded_incidences = 0
    for episode_id, (row, parent) in enumerate(zip(rows, expected, strict=True), 1):
        require(int(row["episode_id"]) == episode_id, "episode ordering")
        for output_name, parent_name in (
            ("slot_id", "slot_id"),
            ("temporal_partition_id", "temporal_partition_id"),
            ("field_attempt_order", "field_attempt_order"),
            ("field_id", "field_id"),
            ("topic_id", "topic_id"),
            ("topic_name", "topic_name"),
            ("subfield_id", "subfield_id"),
            ("subfield_name", "subfield_name"),
            ("decision_year", "decision_year"),
        ):
            require(row[output_name] == parent[parent_name], f"parent episode key: {episode_id} {output_name}")
        for output_name, parent_name in (
            ("aligned_reference_count", "reference_count"),
            ("aligned_recent_count", "recent_count"),
        ):
            require(int(row[output_name]) == int(parent[parent_name]), f"parent count: {episode_id} {output_name}")
        for output_name, parent_name in (
            ("aligned_acceleration", "acceleration"),
            ("aligned_field_year_median", "field_year_median"),
            ("aligned_field_year_mad", "field_year_mad"),
            ("aligned_modified_z", "modified_z"),
        ):
            require(math.isclose(float(row[output_name]), float(parent[parent_name]), rel_tol=1e-12, abs_tol=1e-12), f"parent statistic: {episode_id} {output_name}")
        reference = int(row["aligned_reference_count"])
        recent = int(row["aligned_recent_count"])
        acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
        require(math.isclose(float(row["aligned_acceleration"]), acceleration, rel_tol=1e-12, abs_tol=1e-12), "aligned acceleration")
        mad = float(row["aligned_field_year_mad"])
        require(math.isfinite(mad) and mad > 0, "aligned MAD")
        z_value = multiplier * (acceleration - float(row["aligned_field_year_median"])) / mad
        require(math.isclose(float(row["aligned_modified_z"]), z_value, rel_tol=1e-12, abs_tol=1e-12), "aligned modified z")
        require(recent > reference and acceleration > 0 and z_value >= threshold, "parent candidate rule")
        closure = int(row["all_incumbent_reference_work_count"]) == reference and int(row["all_incumbent_recent_work_count"]) == recent
        require(bool(row["aligned_count_closure"]) == closure and closure, "aligned count closure")
        risk = int(row["risk_opportunity_count"])
        require(risk > 0, "zero risk opportunity")
        require(int(row["minimum_total_support_required"]) == support_minimum, "support minimum")
        require(bool(row["label_independent_support_pass"]) == (risk >= support_minimum), "support flag")
        require(row["label_dependent_support_status"] == "DEFERRED_UNTIL_POST_F_AUTHORIZED_LABEL_MATERIALIZATION", "label-dependent status")
        peer_risk = int(row["risk_with_prior_target_active_peer_count"])
        require(0 <= peer_risk <= risk, "peer risk bound")
        require(int(row["prior_target_active_peer_relation_count"]) >= peer_risk, "peer relation bound")
        require(int(row["focal_excluded_attention_work_incidence_count"]) == 0, "focal anti-target incidence")
        excluded = int(row["network_excluded_attention_work_incidence_count"])
        require(excluded >= 0, "negative excluded incidence")
        for prefix, base_name in (("reference", "all_incumbent_reference_work_count"), ("recent", "all_incumbent_recent_work_count")):
            low = int(row[f"network_leaveout_{prefix}_count_min"])
            median = float(row[f"network_leaveout_{prefix}_count_median"])
            high = int(row[f"network_leaveout_{prefix}_count_max"])
            base = int(row[base_name])
            require(0 <= low <= median <= high <= base, f"leaveout monotonicity: {episode_id} {prefix}")
        surge = int(row["network_leaveout_surge_opportunity_count"])
        require(0 <= surge <= risk, "surge opportunity bound")
        require(bool(row["leaveout_surge_present"]) == (surge > 0), "surge presence flag")
        lost = risk - surge
        non_surge_risk_rows += lost
        episodes_with_non_surge_risk_rows += int(lost > 0)
        peer_risk_rows += peer_risk
        excluded_incidences += excluded
        changed = (
            int(row["network_leaveout_reference_count_min"]) < int(row["all_incumbent_reference_work_count"])
            or int(row["network_leaveout_recent_count_min"]) < int(row["all_incumbent_recent_work_count"])
        )
        median_changed = (
            float(row["network_leaveout_reference_count_median"]) != float(row["all_incumbent_reference_work_count"])
            or float(row["network_leaveout_recent_count_median"]) != float(row["all_incumbent_recent_work_count"])
        )
        changed_episodes += int(changed)
        median_changed_episodes += int(median_changed)
    return rows, {
        "episode_rows": len(rows),
        "aligned_count_closure_rows": sum(bool(row["aligned_count_closure"]) for row in rows),
        "risk_opportunity_rows": sum(int(row["risk_opportunity_count"]) for row in rows),
        "peer_risk_rows": peer_risk_rows,
        "network_excluded_attention_work_incidences": excluded_incidences,
        "network_leaveout_surge_opportunity_rows": sum(int(row["network_leaveout_surge_opportunity_count"]) for row in rows),
        "non_surge_risk_rows": non_surge_risk_rows,
        "episodes_with_non_surge_risk_rows": episodes_with_non_surge_risk_rows,
        "episodes_with_any_leaveout_count_change": changed_episodes,
        "episodes_with_median_leaveout_count_change": median_changed_episodes,
    }


def _verify_slots(
    path: Path,
    parent_slots: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    _assert_schema(path, SLOT_FIELDS, config["output_contract"]["slot_schema_version"])
    observed = pq.read_table(path).to_pylist()
    require(len(observed) == len(parent_slots) == 4, "slot row closure")
    by_id = {str(row["slot_id"]): row for row in observed}
    minimum = int(config["panel_contract"]["final_slot_requires_leaveout_surge_subfield_families"])
    final_pass = True
    for parent in parent_slots:
        slot_id = str(parent["slot_id"])
        row = by_id[slot_id]
        selected = [episode for episode in episodes if episode["slot_id"] == slot_id]
        field_id = int(parent["aligned_selected_field_id"])
        require({int(episode["field_id"]) for episode in selected} == {field_id}, "slot selected-field closure")
        surge = [episode for episode in selected if episode["leaveout_surge_present"]]
        families = {int(episode["subfield_id"]) for episode in surge}
        all_support = all(bool(episode["label_independent_support_pass"]) for episode in selected)
        slot_pass = all_support and len(families) >= minimum
        final_pass &= slot_pass
        require(int(row["aligned_selected_field_id"]) == field_id, "slot selected field")
        require(int(row["aligned_candidate_episode_count"]) == len(selected), "slot episode count")
        require(int(row["aligned_subfield_family_count"]) == len({int(episode["subfield_id"]) for episode in selected}), "slot aligned family count")
        require(int(row["risk_opportunity_count"]) == sum(int(episode["risk_opportunity_count"]) for episode in selected), "slot risk count")
        require(int(row["episodes_passing_label_independent_support"]) == sum(bool(episode["label_independent_support_pass"]) for episode in selected), "slot support count")
        require(bool(row["all_episodes_pass_label_independent_support"]) == all_support, "slot all-support flag")
        require(int(row["leaveout_surge_episode_count"]) == len(surge), "slot surge episode count")
        require(int(row["leaveout_conformed_subfield_family_count"]) == len(families), "slot leaveout family count")
        require(int(row["minimum_leaveout_subfield_families_required"]) == minimum, "slot family minimum")
        require(bool(row["final_panel_slot_pass"]) == slot_pass, "slot pass flag")
        require(bool(row["sealed_slot"]) == bool(parent["sealed_slot"]), "slot seal flag")
    return observed, final_pass


def verify(
    *,
    config_path: Path,
    aligned_root: Path,
    output_root: Path,
    verification_output: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output basename")
    verification_output = verification_output.resolve()
    require(not verification_output.exists() and verification_output.parent.is_dir(), "verification output must be new")
    receipt, artifacts = _verify_output_tree(output_root, config)
    report = _json(output_root / "aligned_final_panel_conformance.json")
    require(report.get("schema_version") == config["output_contract"]["report_schema_version"], "report schema")
    require(report.get("status") == receipt.get("status"), "report receipt status")
    execution = _mapping(report["execution"], "execution")
    require(_sha256(config_path) == execution["config_sha256"], "executed config SHA")
    builder = _safe_repo_path(execution["builder_relative_path"])
    require(_sha256(builder) == execution["builder_sha256"], "executed builder SHA")
    helper = _safe_repo_path(execution["runtime_helper_relative_path"])
    require(_sha256(helper) == execution["runtime_helper_sha256"] == config["runtime_helper"]["sha256"], "runtime helper SHA")
    parent_candidates, parent_slots = _load_parent(aligned_root, _mapping(config["verified_aligned_rescreen_parent"], "aligned parent"))
    expected = _expected_episode_rows(parent_candidates, parent_slots)
    episodes, episode_check = _verify_episodes(output_root / "aligned_final_panel_episode_conformance.parquet", expected, config)
    slots, final_pass = _verify_slots(output_root / "aligned_final_panel_slot_support.parquet", parent_slots, episodes, config)
    expected_status = config["output_contract"]["passing_status"] if final_pass else config["output_contract"]["blocking_status"]
    require(report["status"] == expected_status, "derived final status")
    require(bool(receipt["final_panel_conformance_complete"]) == final_pass, "receipt final-panel state")
    require(bool(receipt["atomic_partition_assignment_authorized_next"]) == final_pass, "receipt atomic authorization")
    require(report["slot_support"] == slots, "report slot table")
    accounting = _mapping(report["replay_accounting"], "replay accounting")
    for report_key, check_key in (
        ("aligned_episode_rows", "episode_rows"),
        ("aligned_count_closure_rows", "aligned_count_closure_rows"),
        ("risk_opportunity_rows", "risk_opportunity_rows"),
        ("network_leaveout_surge_opportunity_rows", "network_leaveout_surge_opportunity_rows"),
    ):
        require(int(accounting[report_key]) == int(episode_check[check_key]), f"report accounting: {report_key}")
    scope = _mapping(report["scope"], "scope")
    require(scope["final_panel_conformance_complete"] is final_pass, "scope final-panel state")
    require(scope["confirmatory_entry_or_persistence_labels_read"] == 0, "confirmatory labels read")
    for key in ("selected_field_replaced", "raw_author_or_work_ids_emitted", "case_control_sampling_applied", "outcome_conditioned_support_evaluated", "realized_citation_effect_read", "holdout_byte_seal_complete", "model_fitted", "g4_open", "pass_f", "pass_g0"):
        require(scope[key] is False, f"unsafe scope: {key}")
    if final_pass:
        require(report["authorization"]["atomic_partition_assignment"] == "AUTHORIZED_NEXT", "atomic partition authorization")
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS if final_pass else BLOCKED_STATUS,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(output_root / "aligned_final_panel_conformance.outputs.json"),
        "artifact_count": len(artifacts),
        "episode_verification": episode_check,
        "slot_rows_verified": len(slots),
        "selected_fields_verified": [int(row["aligned_selected_field_id"]) for row in slots],
        "leaveout_family_counts_verified": [int(row["leaveout_conformed_subfield_family_count"]) for row in slots],
        "row_algebra_and_leaveout_monotonicity_verified": True,
        "raw_id_columns_absent": True,
        "final_panel_conformance_complete": final_pass,
        "next_authorization": report["authorization"],
        "confirmatory_entry_or_persistence_labels_read": 0,
        "holdout_byte_seal_complete": False,
        "pass_f": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
    }
    verification_output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(verification_output, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        aligned_root=args.aligned_root,
        output_root=args.output_root,
        verification_output=args.verification_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
