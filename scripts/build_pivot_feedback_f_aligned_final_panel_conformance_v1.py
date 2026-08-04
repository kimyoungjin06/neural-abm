#!/usr/bin/env python3
"""Build aligned strict-incumbent final-panel leaveout/support conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1 as d0
import build_pivot_feedback_f_final_panel_conformance_v1 as prior


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_final_panel_conformance_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_aligned_final_panel_conformance.v1"

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("field_attempt_order", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("topic_name", pa.string(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("subfield_name", pa.string(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("aligned_reference_count", pa.uint64(), nullable=False),
        pa.field("aligned_recent_count", pa.uint64(), nullable=False),
        pa.field("aligned_acceleration", pa.float64(), nullable=False),
        pa.field("aligned_field_year_median", pa.float64(), nullable=False),
        pa.field("aligned_field_year_mad", pa.float64(), nullable=False),
        pa.field("aligned_modified_z", pa.float64(), nullable=False),
        pa.field("all_incumbent_reference_work_count", pa.uint64(), nullable=False),
        pa.field("all_incumbent_recent_work_count", pa.uint64(), nullable=False),
        pa.field("aligned_count_closure", pa.bool_(), nullable=False),
        pa.field("risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("risk_with_prior_target_active_peer_count", pa.uint64(), nullable=False),
        pa.field("prior_target_active_peer_relation_count", pa.uint64(), nullable=False),
        pa.field("focal_excluded_attention_work_incidence_count", pa.uint64(), nullable=False),
        pa.field("network_excluded_attention_work_incidence_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_reference_count_min", pa.uint64(), nullable=False),
        pa.field("network_leaveout_reference_count_median", pa.float64(), nullable=False),
        pa.field("network_leaveout_reference_count_max", pa.uint64(), nullable=False),
        pa.field("network_leaveout_recent_count_min", pa.uint64(), nullable=False),
        pa.field("network_leaveout_recent_count_median", pa.float64(), nullable=False),
        pa.field("network_leaveout_recent_count_max", pa.uint64(), nullable=False),
        pa.field("network_leaveout_surge_opportunity_count", pa.uint64(), nullable=False),
        pa.field("leaveout_surge_present", pa.bool_(), nullable=False),
        pa.field("minimum_total_support_required", pa.uint32(), nullable=False),
        pa.field("label_independent_support_pass", pa.bool_(), nullable=False),
        pa.field("label_dependent_support_status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_final_panel_episode_conformance.v1"},
)

SLOT_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("aligned_selected_field_id", pa.uint8(), nullable=False),
        pa.field("aligned_candidate_episode_count", pa.uint16(), nullable=False),
        pa.field("aligned_subfield_family_count", pa.uint16(), nullable=False),
        pa.field("risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("episodes_passing_label_independent_support", pa.uint16(), nullable=False),
        pa.field("all_episodes_pass_label_independent_support", pa.bool_(), nullable=False),
        pa.field("leaveout_surge_episode_count", pa.uint16(), nullable=False),
        pa.field("leaveout_conformed_subfield_family_count", pa.uint16(), nullable=False),
        pa.field("minimum_leaveout_subfield_families_required", pa.uint16(), nullable=False),
        pa.field("final_panel_slot_pass", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_final_panel_slot_support.v1"},
)


class AlignedFinalPanelBuildError(RuntimeError):
    """Fail-closed aligned final-panel build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignedFinalPanelBuildError(message)


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    return path


def _validate_repo_document(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    document = _json(path) if path.suffix == ".json" else _yaml(path)
    require(document.get("schema_version") == binding["required_schema_version"], f"repository schema: {path.name}")
    if "required_status" in binding:
        require(document.get("status") == binding["required_status"], f"repository status: {path.name}")
    return {"relative_path": binding["relative_path"], "sha256": binding["sha256"]}


def _validate_authorization(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["authorization_parent"], "authorization parent")
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], "authorization parent SHA")
    document = _yaml(path)
    require(document.get("schema_version") == binding["required_schema_version"], "authorization parent schema")
    require(document["authorization_after_pass"].get(binding["required_authorization"]) == binding["required_value"], "aligned final-panel replay not authorized")
    return {"relative_path": binding["relative_path"], "sha256": binding["sha256"]}


def _schema_version(path: Path) -> str:
    return (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"schema_version", b"").decode()


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}


def _load_aligned_parent(
    root: Path, binding: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink() and root.name == binding["expected_root_basename"], "aligned parent root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "aligned receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "aligned receipt schema")
    require(receipt.get("status") == binding["required_status"], "aligned receipt status")
    artifacts = _artifact_map(receipt)
    candidate_path = root / str(binding["candidate_basename"])
    slot_path = root / str(binding["slot_basename"])
    for path, sha, rows, schema in (
        (candidate_path, binding["candidate_sha256"], binding["candidate_rows"], binding["candidate_schema_version"]),
        (slot_path, binding["slot_sha256"], binding["slot_rows"], binding["slot_schema_version"]),
    ):
        require(_sha256(path) == sha, f"aligned artifact SHA: {path.name}")
        require(artifacts[path.name]["sha256"] == sha, f"aligned receipt artifact: {path.name}")
        require(pq.ParquetFile(path).metadata.num_rows == rows, f"aligned artifact rows: {path.name}")
        require(_schema_version(path) == schema, f"aligned artifact schema: {path.name}")
    verification = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == binding["verification_sha256"], "aligned verification SHA")
    require(_json(verification).get("status") == binding["verification_status"], "aligned verification status")
    candidates = pq.read_table(candidate_path).to_pylist()
    retained = [dict(row) for row in candidates if bool(row["retained_for_aligned_panel"])]
    require(len(retained) == binding["retained_candidate_rows"], "aligned retained candidate rows")
    require(all(row["strict_attention_candidate"] and row["aligned_selected_field"] for row in retained), "unaligned retained candidate")
    slots = pq.read_table(slot_path).to_pylist()
    require(all(row["slot_pass"] and row["aligned_selected_field_id"] is not None for row in slots), "aligned slot not passed")
    selected = {str(row["slot_id"]): int(row["aligned_selected_field_id"]) for row in slots}
    require(all(selected[str(row["slot_id"])] == int(row["field_id"]) for row in retained), "retained field differs from aligned slot")
    return retained, slots, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "candidate_sha256": binding["candidate_sha256"],
        "slot_sha256": binding["slot_sha256"],
        "verification_sha256": binding["verification_sha256"],
    }


def _helper_episodes(
    candidates: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    slot_order = {str(row["slot_id"]): index for index, row in enumerate(slots)}
    sealed = {str(row["slot_id"]): bool(row["sealed_slot"]) for row in slots}
    ordered = sorted(
        candidates,
        key=lambda row: (
            slot_order[str(row["slot_id"])],
            int(row["decision_year"]),
            int(row["subfield_id"]),
            int(row["topic_id"]),
        ),
    )
    helper: list[dict[str, Any]] = []
    source: dict[int, dict[str, Any]] = {}
    for episode_id, row in enumerate(ordered, 1):
        require(row["modified_z"] is not None and math.isfinite(float(row["modified_z"])), "aligned candidate z")
        source[episode_id] = dict(row)
        helper.append(
            {
                "episode_id": episode_id,
                "slot_id": str(row["slot_id"]),
                "temporal_partition_id": str(row["temporal_partition_id"]),
                "field_attempt_order": int(row["field_attempt_order"]),
                "initial_assignment_selected": True,
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "subfield_id": int(row["subfield_id"]),
                "decision_year": int(row["decision_year"]),
                "aggregate_reference_count": int(row["reference_count"]),
                "aggregate_recent_count": int(row["recent_count"]),
                "aggregate_modified_z": float(row["modified_z"]),
                "field_year_median": float(row["field_year_median"]),
                "field_year_mad": float(row["field_year_mad"]),
                "sealed_slot": sealed[str(row["slot_id"])],
            }
        )
    return helper, source


def _episode_output(
    replay: Sequence[Mapping[str, Any]],
    source: Mapping[int, Mapping[str, Any]],
    minimum_support: int,
) -> list[dict[str, Any]]:
    rows = []
    for observed in replay:
        episode_id = int(observed["episode_id"])
        aligned = source[episode_id]
        count_closure = (
            int(observed["all_incumbent_reference_work_count"]) == int(aligned["reference_count"])
            and int(observed["all_incumbent_recent_work_count"]) == int(aligned["recent_count"])
        )
        require(count_closure, f"aligned strict-count closure: {episode_id}")
        risk = int(observed["risk_opportunity_count"])
        require(risk > 0, f"zero opportunity episode: {episode_id}")
        require(int(observed["focal_excluded_attention_work_incidence_count"]) == 0, "focal anti-target mismatch")
        rows.append(
            {
                "episode_id": episode_id,
                "slot_id": str(aligned["slot_id"]),
                "temporal_partition_id": str(aligned["temporal_partition_id"]),
                "field_attempt_order": int(aligned["field_attempt_order"]),
                "field_id": int(aligned["field_id"]),
                "topic_id": int(aligned["topic_id"]),
                "topic_name": str(aligned["topic_name"]),
                "subfield_id": int(aligned["subfield_id"]),
                "subfield_name": str(aligned["subfield_name"]),
                "decision_year": int(aligned["decision_year"]),
                "aligned_reference_count": int(aligned["reference_count"]),
                "aligned_recent_count": int(aligned["recent_count"]),
                "aligned_acceleration": float(aligned["acceleration"]),
                "aligned_field_year_median": float(aligned["field_year_median"]),
                "aligned_field_year_mad": float(aligned["field_year_mad"]),
                "aligned_modified_z": float(aligned["modified_z"]),
                "all_incumbent_reference_work_count": int(observed["all_incumbent_reference_work_count"]),
                "all_incumbent_recent_work_count": int(observed["all_incumbent_recent_work_count"]),
                "aligned_count_closure": count_closure,
                "risk_opportunity_count": risk,
                "risk_with_prior_target_active_peer_count": int(observed["risk_with_prior_target_active_peer_count"]),
                "prior_target_active_peer_relation_count": int(observed["prior_target_active_peer_relation_count"]),
                "focal_excluded_attention_work_incidence_count": int(observed["focal_excluded_attention_work_incidence_count"]),
                "network_excluded_attention_work_incidence_count": int(observed["network_excluded_attention_work_incidence_count"]),
                "network_leaveout_reference_count_min": int(observed["network_leaveout_reference_count_min"]),
                "network_leaveout_reference_count_median": float(observed["network_leaveout_reference_count_median"]),
                "network_leaveout_reference_count_max": int(observed["network_leaveout_reference_count_max"]),
                "network_leaveout_recent_count_min": int(observed["network_leaveout_recent_count_min"]),
                "network_leaveout_recent_count_median": float(observed["network_leaveout_recent_count_median"]),
                "network_leaveout_recent_count_max": int(observed["network_leaveout_recent_count_max"]),
                "network_leaveout_surge_opportunity_count": int(observed["network_leaveout_surge_opportunity_count"]),
                "leaveout_surge_present": int(observed["network_leaveout_surge_opportunity_count"]) > 0,
                "minimum_total_support_required": minimum_support,
                "label_independent_support_pass": risk >= minimum_support,
                "label_dependent_support_status": "DEFERRED_UNTIL_POST_F_AUTHORIZED_LABEL_MATERIALIZATION",
                "sealed_slot": bool(observed["sealed_slot"]),
            }
        )
    require(len(rows) == len(source) == 65, "aligned episode replay closure")
    return rows


def _slot_output(
    slots: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    minimum_families: int,
) -> list[dict[str, Any]]:
    rows = []
    for slot in slots:
        slot_id = str(slot["slot_id"])
        selected = [row for row in episodes if row["slot_id"] == slot_id]
        field_id = int(slot["aligned_selected_field_id"])
        require(selected and {int(row["field_id"]) for row in selected} == {field_id}, f"selected field closure: {slot_id}")
        surge = [row for row in selected if row["leaveout_surge_present"]]
        conformed_families = {int(row["subfield_id"]) for row in surge}
        all_support = all(bool(row["label_independent_support_pass"]) for row in selected)
        slot_pass = all_support and len(conformed_families) >= minimum_families
        if not all_support:
            status = "BLOCKED_LABEL_INDEPENDENT_EPISODE_SUPPORT"
        elif len(conformed_families) < minimum_families:
            status = "BLOCKED_FEWER_THAN_THREE_LEAVEOUT_SURGE_SUBFIELD_FAMILIES"
        else:
            status = "PASS_ALIGNED_LEAVEOUT_SHOCK_FAMILY_AND_SUPPORT"
        rows.append(
            {
                "slot_id": slot_id,
                "temporal_partition_id": str(slot["temporal_partition_id"]),
                "aligned_selected_field_id": field_id,
                "aligned_candidate_episode_count": len(selected),
                "aligned_subfield_family_count": len({int(row["subfield_id"]) for row in selected}),
                "risk_opportunity_count": sum(int(row["risk_opportunity_count"]) for row in selected),
                "episodes_passing_label_independent_support": sum(bool(row["label_independent_support_pass"]) for row in selected),
                "all_episodes_pass_label_independent_support": all_support,
                "leaveout_surge_episode_count": len(surge),
                "leaveout_conformed_subfield_family_count": len(conformed_families),
                "minimum_leaveout_subfield_families_required": minimum_families,
                "final_panel_slot_pass": slot_pass,
                "status": status,
                "sealed_slot": bool(slot["sealed_slot"]),
            }
        )
    require(len(rows) == 4 and sum(int(row["aligned_candidate_episode_count"]) for row in rows) == 65, "slot output closure")
    return rows


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")
    os.chmod(path, 0o600)
    return {"relative_path": path.name, "bytes": path.stat().st_size, "rows": len(table), "sha256": _sha256(path)}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Aligned strict-incumbent final-panel conformance",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Slot | Field | Episodes | Leaveout episodes | Leaveout families | Support |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["slot_support"]:
        lines.append(
            f"| {row['slot_id']} | {row['aligned_selected_field_id']} | {row['aligned_candidate_episode_count']} | "
            f"{row['leaveout_surge_episode_count']} | {row['leaveout_conformed_subfield_family_count']} | {row['status']} |"
        )
    lines.extend(["", "No selected field was replaced and no aggregate normalization was reused.", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def _cleanup(temporary: Path | None, output: Path) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute() and temporary.parent == output.parent, "unsafe temporary parent")
    require(temporary.name.startswith(f".{output.name}."), "unsafe temporary prefix")
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    aligned_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= threads <= 32, "threads must be in [1,32]")
    require(16 <= memory_limit_gb <= 384, "memory limit must be in [16,384] GB")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status") == "FROZEN_ALIGNED_FINAL_PANEL_CONFORMANCE_PENDING_PRODUCTION_NO_F_PASS", "config status")
    repository_bindings = {"authorization_parent": _validate_authorization(config)}
    for name, raw in _mapping(config["bound_design_sources"], "bound sources").items():
        repository_bindings[name] = _validate_repo_document(_mapping(raw, name))
    helper_binding = _mapping(config["runtime_helper"], "runtime helper")
    helper_path = _safe_repo_path(helper_binding["relative_path"])
    require(_sha256(helper_path) == helper_binding["sha256"], "runtime helper SHA")
    require(helper_path == Path(prior.__file__).resolve(), "loaded runtime helper path")
    repository_bindings["runtime_helper"] = {"relative_path": helper_binding["relative_path"], "sha256": helper_binding["sha256"]}
    candidates, slots, aligned_binding = _load_aligned_parent(aligned_root, _mapping(config["verified_aligned_rescreen_parent"], "aligned parent"))
    panel = _mapping(config["panel_contract"], "panel contract")
    require(panel["selected_field_replacement_during_leaveout"] == "FORBIDDEN", "selected field replacement allowed")
    require(panel["later_reserve_fallback_after_aligned_selection"] == "FORBIDDEN", "reserve fallback allowed")
    require(panel["aligned_screen_normalization"]["aggregate_location_or_scale_reuse"] == "FORBIDDEN", "aggregate normalization reuse")
    require(panel["team_cap"] == "NONE" and panel["attention_work_rule"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT", "primary leaveout construct")
    selected_fields = [int(row["aligned_selected_field_id"]) for row in slots]
    require(selected_fields == [int(value) for value in panel["selected_fields_fixed_by_verified_aligned_rescreen"]], "fixed selected fields")
    support = _mapping(config["support_replay"], "support replay")
    require(support["label_dependent"]["may_block_f"] is False, "circular label-dependent support")
    completion = _mapping(config["completion_accounting"], "completion")
    require(completion["confirmatory_entry_or_persistence_labels_read"] == 0, "confirmatory labels read")
    for key in ("holdout_byte_seal_complete", "model_fitted", "g4_open", "pass_f", "pass_g0"):
        require(completion[key] is False, f"unsafe completion state: {key}")
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    work_paths, work_binding, work_stats = d0._validate_partitioned_input(
        cutoff_root, _mapping(inputs["cutoff_work_metadata"], "work input"), schema=d0.WORK_INPUT_SCHEMA, workers=threads
    )
    event_paths, event_binding, event_stats = d0._validate_partitioned_input(
        author_history_root, _mapping(inputs["author_history_events"], "event input"), schema=d0.EVENT_INPUT_SCHEMA, workers=threads
    )
    helper_episodes, source = _helper_episodes(candidates, slots)
    output = output.resolve()
    output_contract = _mapping(config["output_contract"], "output contract")
    require(output.name == output_contract["root_basename"], "output basename")
    require(not output.exists() and output.parent.is_dir(), "output must be new")
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        database = temporary / "private-leaveout.duckdb"
        replay, replay_accounting = prior._run_sql_replay(
            database=database,
            episodes=helper_episodes,
            work_glob=cutoff_root.resolve() / "work_metadata" / "*.parquet",
            event_glob=author_history_root.resolve() / "events" / "*.parquet",
            threads=threads,
            memory_limit_gb=memory_limit_gb,
        )
        minimum_support = int(support["label_independent"]["minimum_total_opportunity_rows"])
        episode_rows = _episode_output(replay, source, minimum_support)
        minimum_families = int(panel["final_slot_requires_leaveout_surge_subfield_families"])
        slot_rows = _slot_output(slots, episode_rows, minimum_families)
        final_panel_pass = all(bool(row["final_panel_slot_pass"]) for row in slot_rows)
        status = output_contract["passing_status"] if final_panel_pass else output_contract["blocking_status"]
        d0._validate_unchanged(work_paths, work_stats)
        d0._validate_unchanged(event_paths, event_stats)
        if database.exists():
            database.unlink()
        spill = temporary / "duckdb-spill"
        if spill.exists():
            shutil.rmtree(spill)
        artifacts = [
            _write_parquet(pa.Table.from_pylist(episode_rows, schema=EPISODE_SCHEMA), temporary / "aligned_final_panel_episode_conformance.parquet"),
            _write_parquet(pa.Table.from_pylist(slot_rows, schema=SLOT_SCHEMA), temporary / "aligned_final_panel_slot_support.parquet"),
        ]
        report = {
            "schema_version": output_contract["report_schema_version"],
            "status": status,
            "repository_bindings": repository_bindings,
            "aligned_parent_binding": aligned_binding,
            "input_bindings": {"cutoff_work_metadata": work_binding, "author_history_events": event_binding},
            "scope": {
                "episode_count": len(episode_rows),
                "slot_count": len(slot_rows),
                "final_panel_conformance_complete": final_panel_pass,
                "selected_field_replaced": False,
                "raw_author_or_work_ids_emitted": False,
                "confirmatory_entry_or_persistence_labels_read": 0,
                "case_control_sampling_applied": False,
                "outcome_conditioned_support_evaluated": False,
                "realized_citation_effect_read": False,
                "holdout_byte_seal_complete": False,
                "model_fitted": False,
                "g4_open": False,
                "pass_f": False,
                "pass_g0": False,
            },
            "replay_accounting": {
                **replay_accounting,
                "aligned_episode_rows": len(episode_rows),
                "aligned_count_closure_rows": sum(bool(row["aligned_count_closure"]) for row in episode_rows),
                "episodes_passing_label_independent_support": sum(bool(row["label_independent_support_pass"]) for row in episode_rows),
                "episodes_with_leaveout_surge": sum(bool(row["leaveout_surge_present"]) for row in episode_rows),
                "network_leaveout_surge_opportunity_rows": sum(int(row["network_leaveout_surge_opportunity_count"]) for row in episode_rows),
            },
            "support_boundary": {
                "label_independent_minimum_total": minimum_support,
                "every_episode_must_pass": support["label_independent"]["every_episode_must_pass"],
                "label_dependent_support_status": support["label_dependent"]["status_during_f"],
                "label_dependent_support_may_block_f": support["label_dependent"]["may_block_f"],
            },
            "normalization_boundary": panel["aligned_screen_normalization"],
            "slot_support": slot_rows,
            "authorization": config["authorization_after_pass"] if final_panel_pass else config["authorization_after_block"],
            "execution": {
                "config_relative_path": str(config_path.relative_to(ROOT)),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "runtime_helper_relative_path": helper_binding["relative_path"],
                "runtime_helper_sha256": helper_binding["sha256"],
                "threads": threads,
                "memory_limit_gb": memory_limit_gb,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "aligned_final_panel_conformance.json"
        _write_json(report_path, report)
        artifacts.append({"relative_path": report_path.name, "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)})
        markdown_path = temporary / "ALIGNED_FINAL_PANEL_CONFORMANCE.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append({"relative_path": markdown_path.name, "bytes": markdown_path.stat().st_size, "sha256": _sha256(markdown_path)})
        artifacts.sort(key=lambda row: row["relative_path"])
        receipt = {
            "schema_version": output_contract["receipt_schema_version"],
            "status": status,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes([(row["relative_path"], row["bytes"], row["sha256"]) for row in artifacts])
            ).hexdigest(),
            "receipt_written_last": True,
            "independent_verification_complete": False,
            "final_panel_conformance_complete": final_panel_pass,
            "atomic_partition_assignment_authorized_next": final_panel_pass,
            "selected_field_replaced": False,
            "label_independent_support_complete": True,
            "label_dependent_support_deferred_post_f": True,
            "confirmatory_entry_or_persistence_labels_read": 0,
            "raw_author_or_work_ids_emitted": False,
            "team_cap_applied": False,
            "holdout_byte_seal_complete": False,
            "model_fitted": False,
            "g4_open": False,
            "pass_f": False,
            "pass_g0": False,
        }
        _write_json(temporary / "aligned_final_panel_conformance.outputs.json", receipt)
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup(temporary, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit-gb", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        aligned_root=args.aligned_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    print(json.dumps({"status": report["status"], "replay_accounting": report["replay_accounting"], "slot_support": report["slot_support"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
