#!/usr/bin/env python3
"""Independently verify the F final-panel conformance aggregate artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_final_panel_conformance_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_final_panel_conformance.v1"
REPORT_SCHEMA = "pivot_feedback_f_final_panel_conformance_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_f_final_panel_conformance_outputs.v1"
EPISODE_SCHEMA_VERSION = "pivot_feedback_f_final_panel_episode_conformance.v1"
SLOT_SCHEMA_VERSION = "pivot_feedback_f_final_panel_slot_support.v1"
VERIFY_SCHEMA = "pivot_feedback_f_final_panel_conformance_verification.v1"
PASS_STATUS = "PASS_INDEPENDENT_OUTCOME_BLIND_FINAL_PANEL_CONFORMANCE_ARTIFACT_REPLAY"
BLOCKED_STATUS = "PASS_INDEPENDENT_BLOCKED_FINAL_PANEL_LEAVEOUT_REPLAY_NO_F_PASS"


class FinalPanelVerificationError(RuntimeError):
    """Fail-closed final-panel verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalPanelVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    return path


def _validate_binding(binding: Mapping[str, Any]) -> None:
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], f"binding SHA: {path.name}")
    document = _yaml(path) if path.suffix in (".yaml", ".yml") else _json(path)
    require(document.get("schema_version") == binding["required_schema_version"], f"binding schema: {path.name}")
    if "required_status" in binding:
        require(document.get("status") == binding["required_status"], f"binding status: {path.name}")


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt.get("artifacts")
    require(isinstance(rows, list), "receipt artifacts")
    result = {str(row["relative_path"]): dict(row) for row in rows}
    require(len(result) == len(rows), "duplicate receipt artifact")
    return result


def _assignment_rows(config: Mapping[str, Any], assignment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    binding = _mapping(config["verified_assignment_parent"], "assignment binding")
    root = assignment_root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "assignment root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "assignment receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("status") == binding["required_status"], "assignment receipt status")
    candidate_path = root / str(binding["candidate_basename"])
    slot_path = root / str(binding["slot_basename"])
    require(_sha256(candidate_path) == binding["candidate_sha256"], "candidate SHA")
    require(_sha256(slot_path) == binding["slot_sha256"], "slot SHA")
    candidates = pq.read_table(candidate_path).to_pylist()
    slots = pq.read_table(slot_path).to_pylist()
    episodes: list[dict[str, Any]] = []
    dispositions: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        slot_id = str(slot["slot_id"])
        period = str(slot["temporal_partition_id"])
        fields = [int(slot["nominal_primary_field_id"]), *[int(value) for value in slot["reserve_field_ids"]]]
        field_rows = []
        for order, field_id in enumerate(fields, 1):
            rows = [row for row in candidates if int(row["field_id"]) == field_id and str(row["temporal_partition_id"]) == period]
            rows.sort(key=lambda row: (int(row["decision_year"]), int(row["subfield_id"]), int(row["topic_id"])))
            family_count = len({int(row["subfield_id"]) for row in rows})
            eligible = family_count >= 3
            field_rows.append({"field_id": field_id, "eligible": eligible, "candidate_count": len(rows), "family_count": family_count})
            if not eligible:
                continue
            for row in rows:
                episodes.append(
                    {
                        "episode_id": len(episodes) + 1,
                        "slot_id": slot_id,
                        "temporal_partition_id": period,
                        "field_attempt_order": order,
                        "initial_assignment_selected": field_id == int(slot["selected_field_id"]),
                        "field_id": field_id,
                        "topic_id": int(row["topic_id"]),
                        "subfield_id": int(row["subfield_id"]),
                        "decision_year": int(row["decision_year"]),
                        "aggregate_reference_count": int(row["reference_count"]),
                        "aggregate_recent_count": int(row["recent_count"]),
                        "aggregate_modified_z": float(row["modified_z"]),
                        "sealed_slot": bool(slot["sealed"]),
                    }
                )
        dispositions[slot_id] = field_rows
    require(len(episodes) == 194, "expected episode closure")
    return episodes, slots, dispositions


def _schema_version(table: pa.Table) -> str:
    return (table.schema.metadata or {}).get(b"schema_version", b"").decode()


def verify(*, config_path: Path, assignment_root: Path, output_root: Path, verification_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    _validate_binding(_mapping(config["authorization_parent"], "authorization parent"))
    for name, binding in _mapping(config["bound_design_sources"], "bound sources").items():
        _validate_binding(_mapping(binding, name))
    expected, assignment_slots, dispositions = _assignment_rows(config, assignment_root)
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output root")
    receipt_path = output_root / "final_panel_conformance.outputs.json"
    report_path = output_root / "final_panel_conformance.json"
    episode_path = output_root / "final_panel_episode_conformance.parquet"
    slot_path = output_root / "final_panel_slot_support.parquet"
    markdown_path = output_root / "FINAL_PANEL_CONFORMANCE.md"
    receipt = _json(receipt_path)
    report = _json(report_path)
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    status = str(receipt.get("status"))
    require(report.get("status") == status, "report/receipt status")
    allowed = {str(config["output_contract"]["passing_status"]), str(config["output_contract"]["blocking_status"])}
    require(status in allowed, "output status")
    artifacts = _artifact_map(receipt)
    expected_names = {markdown_path.name, report_path.name, episode_path.name, slot_path.name}
    require(set(artifacts) == expected_names, "artifact set")
    rehashed = []
    for name in sorted(expected_names):
        path = output_root / name
        sha = _sha256(path)
        require(artifacts[name]["sha256"] == sha, f"artifact SHA: {name}")
        require(int(artifacts[name]["bytes"]) == path.stat().st_size, f"artifact bytes: {name}")
        rehashed.append((name, sha))
    require(receipt.get("artifact_tree_sha256") == hashlib.sha256(_canonical_bytes(rehashed)).hexdigest(), "artifact tree SHA")
    receipt_mtime = receipt_path.stat().st_mtime_ns
    require(all(receipt_mtime > (output_root / name).stat().st_mtime_ns for name in expected_names), "receipt not written last")
    episode_table = pq.read_table(episode_path)
    slot_table = pq.read_table(slot_path)
    require(_schema_version(episode_table) == EPISODE_SCHEMA_VERSION, "episode schema version")
    require(_schema_version(slot_table) == SLOT_SCHEMA_VERSION, "slot schema version")
    require(len(episode_table) == 194 and len(slot_table) == 4, "artifact row closure")
    forbidden = {"author_id", "author_numeric_id", "work_id", "work_numeric_id", "focal_author_id", "peer_author_id"}
    require(not (forbidden & set(episode_table.column_names)), "raw ID in episode output")
    observed = episode_table.to_pylist()
    key_names = [
        "episode_id", "slot_id", "temporal_partition_id", "field_attempt_order",
        "initial_assignment_selected", "field_id", "topic_id", "subfield_id",
        "decision_year", "aggregate_reference_count", "aggregate_recent_count",
        "aggregate_modified_z", "sealed_slot",
    ]
    for wanted, row in zip(expected, observed, strict=True):
        for name in key_names:
            require(row[name] == wanted[name], f"episode assignment replay: {wanted['episode_id']} {name}")
        risk = int(row["risk_opportunity_count"])
        require(risk >= int(row["minimum_total_support_required"]) == 500, "label-independent support")
        require(bool(row["label_independent_support_pass"]), "episode support flag")
        require(row["label_dependent_support_status"] == "DEFERRED_UNTIL_POST_F_AUTHORIZED_LABEL_MATERIALIZATION", "label-dependent status")
        require(int(row["focal_excluded_attention_work_incidence_count"]) == 0, "focal leaveout incidence")
        require(0 <= int(row["risk_with_prior_target_active_peer_count"]) <= risk, "peer risk bound")
        require(0 <= int(row["network_leaveout_surge_opportunity_count"]) <= risk, "surge opportunity bound")
        require(bool(row["leaveout_surge_present"]) == (int(row["network_leaveout_surge_opportunity_count"]) > 0), "surge presence flag")
        for prefix, base in (("reference", "all_incumbent_reference_work_count"), ("recent", "all_incumbent_recent_work_count")):
            low = int(row[f"network_leaveout_{prefix}_count_min"])
            med = float(row[f"network_leaveout_{prefix}_count_median"])
            high = int(row[f"network_leaveout_{prefix}_count_max"])
            require(0 <= low <= med <= high <= int(row[base]), f"leaveout monotonicity: {row['episode_id']} {prefix}")
    observed_slots = {str(row["slot_id"]): row for row in slot_table.to_pylist()}
    expected_final_pass = True
    for assignment in assignment_slots:
        slot_id = str(assignment["slot_id"])
        slot_row = observed_slots[slot_id]
        rows = [row for row in observed if row["slot_id"] == slot_id]
        attempted = [int(row["field_id"]) for row in dispositions[slot_id] if row["eligible"]]
        require(slot_row["aggregate_screen_attempted_field_ids"] == ",".join(str(value) for value in attempted), "attempted fields")
        require(int(slot_row["initial_selected_field_id"]) == int(assignment["selected_field_id"]), "initial field")
        final_field = None
        best_rows: list[dict[str, Any]] = []
        best_families = -1
        for field_id in attempted:
            candidate = [row for row in rows if int(row["field_id"]) == field_id]
            families = {int(row["subfield_id"]) for row in candidate if row["leaveout_surge_present"]}
            if len(families) > best_families:
                best_rows, best_families = candidate, len(families)
            if len(families) >= 3:
                final_field = field_id
                best_rows = candidate
                break
        final_rows = best_rows
        surge_rows = [row for row in final_rows if row["leaveout_surge_present"]]
        family_count = len({int(row["subfield_id"]) for row in surge_rows})
        slot_pass = final_field is not None
        expected_final_pass &= slot_pass
        require(slot_row["final_selected_field_id"] == final_field, "final field replay")
        require(int(slot_row["leaveout_surge_episode_count"]) == len(surge_rows), "slot surge episodes")
        require(int(slot_row["leaveout_conformed_subfield_family_count"]) == family_count, "slot surge families")
        require(bool(slot_row["final_panel_slot_pass"]) == slot_pass, "slot pass")
    blocking = str(config["output_contract"]["blocking_status"])
    passing = str(config["output_contract"]["passing_status"])
    require(status == (passing if expected_final_pass else blocking), "final status replay")
    require(bool(receipt["final_panel_conformance_complete"]) == expected_final_pass, "receipt final-panel state")
    for key in ("entry_or_persistence_labels_read", "raw_author_or_work_ids_emitted", "team_cap_applied", "holdout_byte_seal_complete", "pass_f", "pass_g0", "model_fitted", "g4_open"):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    scope = _mapping(report["scope"], "report scope")
    require(scope.get("final_panel_conformance_complete") is expected_final_pass, "scope final-panel state")
    require(scope.get("entry_or_persistence_labels_read") is False and scope.get("g4_open") is False, "unsafe report scope")
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS if expected_final_pass else BLOCKED_STATUS,
        "output_root": str(output_root),
        "output_receipt_sha256": _sha256(receipt_path),
        "artifacts_rehashed_including_receipt": len(expected_names) + 1,
        "episode_rows_verified": len(observed),
        "slot_rows_verified": len(observed_slots),
        "assignment_and_reserve_order_replayed": True,
        "row_algebra_and_leaveout_monotonicity_verified": True,
        "raw_id_columns_absent": True,
        "final_panel_conformance_complete": expected_final_pass,
        "holdout_byte_seal_complete": False,
        "pass_f": False,
        "pass_g0": False,
        "g4_open": False,
    }
    verification_path = verification_path.resolve()
    require(not verification_path.exists() and verification_path.parent.is_dir(), "verification output must be new")
    verification_path.write_bytes(json.dumps(verification, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    os.chmod(verification_path, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(config_path=args.config, assignment_root=args.assignment_root, output_root=args.output_root, verification_path=args.verification_output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
