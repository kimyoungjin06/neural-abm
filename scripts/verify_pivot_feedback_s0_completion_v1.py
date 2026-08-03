#!/usr/bin/env python3
"""Bind the three independently verified S0 components into one gate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCHEMA = "pivot_feedback_s0_completion_receipt.v1"
PASS_STATUS = (
    "PASS_S0_THRESHOLD_FREE_FIELD_AUTHOR_GRAPH_AND_AFFILIATION_INVENTORY_"
    "D0_DISCOVERY_ONLY_AUTHORIZED"
)
BLOCKED_STATUS = "BLOCKED_S0_COMPLETION_RECEIPT"

CORE = {
    "receipt": "s0_core_inventory.outputs.json",
    "receipt_schema": "pivot_feedback_s0_threshold_free_core_inventory_outputs.v1",
    "receipt_status": (
        "PASS_S0_THRESHOLD_FREE_CORE_FIELD_YEAR_INVENTORY_"
        "WITH_ORCID_AFFILIATION_AND_GRAPH_TOPOLOGY_PENDING"
    ),
    "pass_field": "core_component_pass",
    "report": "s0_core_inventory.json",
    "verification_schema": (
        "pivot_feedback_s0_threshold_free_core_verification.v1"
    ),
    "verification_status": (
        "PASS_INDEPENDENT_S0_CORE_OUTPUT_CONSERVATION_AND_ATTENTION_REPLAY_"
        "DEFERRED_COMPONENTS_STILL_PENDING"
    ),
    "verifier": "scripts/verify_pivot_feedback_s0_threshold_free_inventory.py",
}
ORCID = {
    "receipt": "authorship_orcid_affiliation_report.outputs.json",
    "receipt_schema": (
        "pivot_feedback_s0_authorship_orcid_affiliation_outputs.v1"
    ),
    "receipt_status": (
        "PASS_S0_FULL_AUTHORSHIP_ORCID_AND_AFFILIATION_PROJECTION_"
        "WITH_TOKEN_AND_CURRENT_SNAPSHOT_LIMITATIONS"
    ),
    "pass_field": "component_pass",
    "report": "authorship_orcid_affiliation_report.json",
    "verification_schema": (
        "pivot_feedback_s0_authorship_orcid_affiliation_verification.v1"
    ),
    "verification_status": (
        "PASS_INDEPENDENT_S0_AUTHORSHIP_ORCID_AFFILIATION_CONSERVATION_"
        "COAUTHOR_GRAPH_STILL_NOT_RUN"
    ),
    "verifier": (
        "scripts/verify_pivot_feedback_s0_authorship_orcid_affiliation.py"
    ),
}
GRAPH = {
    "receipt": "coauthor_graph_profile_report.outputs.json",
    "receipt_schema": "pivot_feedback_s0_coauthor_graph_profile_outputs.v1",
    "receipt_status": (
        "PASS_S0_EXACT_NONRANKED_COAUTHOR_GRAPH_PROFILES_"
        "WITH_CURRENT_SNAPSHOT_AUTHOR_TOKEN_LIMITATIONS"
    ),
    "pass_field": "component_pass",
    "report": "coauthor_graph_profile_report.json",
    "verification_schema": (
        "pivot_feedback_s0_coauthor_graph_profile_verification.v1"
    ),
    "verification_status": (
        "PASS_INDEPENDENT_S0_COAUTHOR_GRAPH_PROFILE_TOPOLOGY_CONSERVATION_"
        "ORCID_AFFILIATION_RECEIPT_STILL_REQUIRED"
    ),
    "verifier": "scripts/verify_pivot_feedback_s0_coauthor_graph_profile.py",
}
FORBIDDEN_METRIC_KEYS = {
    "eligible_author_count",
    "matured_entry_opportunity_count",
    "fixed_horizon_outcome_coverage",
    "independent_attention_shock_count",
    "entry_effect_size",
    "persistence_effect_size",
    "overshoot_timing_gap",
    "predictive_score",
    "selected_best_field",
    "selected_best_threshold",
    "g4_holdout_outcome",
}


class S0CompletionError(RuntimeError):
    """Fail-closed S0 completion error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S0CompletionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S0CompletionError(f"invalid {label}: {error}") from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _published_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            str(key)
            for key in value
        } | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def _artifact_path(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "artifact path is invalid")
    pure = PurePosixPath(relative)
    require(
        not pure.is_absolute() and ".." not in pure.parts,
        "artifact path is unsafe",
    )
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "artifact path escapes output root")
    require(path.is_file() and not path.is_symlink(), "artifact is invalid")
    return path


def _validate_component(
    root_value: Path,
    verification_path_value: Path,
    contract: Mapping[str, str],
    *,
    label: str,
) -> dict[str, Any]:
    root = root_value.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), f"{label} root is invalid")
    require(
        stat.S_IMODE(root.stat().st_mode) == 0o700,
        f"{label} root mode mismatch",
    )
    receipt_path = root / contract["receipt"]
    receipt = _read_json(receipt_path, f"{label} receipt")
    require(
        receipt.get("schema_version") == contract["receipt_schema"],
        f"{label} receipt schema mismatch",
    )
    require(
        receipt.get("status") == contract["receipt_status"],
        f"{label} receipt status mismatch",
    )
    require(
        receipt.get(contract["pass_field"]) is True,
        f"{label} component did not pass",
    )
    require(receipt.get("s0_gate_pass") is False, f"{label} self-promoted S0")
    require(receipt.get("d0_authorized") is False, f"{label} self-authorized D0")
    if label != "core":
        require(receipt.get("full_scope") is True, f"{label} is not full scope")

    artifacts_raw = receipt.get("artifacts")
    require(isinstance(artifacts_raw, list), f"{label} artifacts are invalid")
    artifacts = [_mapping(item, f"{label} artifact") for item in artifacts_raw]
    relative_paths = [item.get("relative_path") for item in artifacts]
    require(relative_paths == sorted(relative_paths), f"{label} artifacts are not sorted")
    require(len(relative_paths) == len(set(relative_paths)), f"{label} duplicate artifact")
    require(
        len(artifacts) == receipt.get("artifact_count"),
        f"{label} artifact count mismatch",
    )
    for item in artifacts:
        path = _artifact_path(root, item.get("relative_path"))
        require(
            stat.S_IMODE(path.stat().st_mode) == 0o600,
            f"{label} artifact mode mismatch",
        )
        require(path.stat().st_size == item.get("bytes"), f"{label} artifact size mismatch")
        require(_sha256_file(path) == item.get("sha256"), f"{label} artifact SHA mismatch")
    expected_names = set(relative_paths) | {contract["receipt"]}
    require(
        {path.name for path in root.iterdir()} == expected_names,
        f"{label} output file set mismatch",
    )
    require(
        hashlib.sha256(_canonical_bytes(artifacts)).hexdigest()
        == receipt.get("artifact_tree_sha256"),
        f"{label} artifact tree mismatch",
    )
    receipt_sha = _sha256_file(receipt_path)

    verification_path = verification_path_value.resolve(strict=True)
    require(
        not verification_path.is_relative_to(root),
        f"{label} verification report is inside output root",
    )
    require(
        stat.S_IMODE(verification_path.stat().st_mode) == 0o600,
        f"{label} verification report mode mismatch",
    )
    verification = _read_json(
        verification_path,
        f"{label} independent verification",
    )
    require(
        verification.get("schema_version") == contract["verification_schema"],
        f"{label} verification schema mismatch",
    )
    require(
        verification.get("status") == contract["verification_status"],
        f"{label} verification status mismatch",
    )
    require(
        verification.get("output_root_basename") == root.name,
        f"{label} verification root mismatch",
    )
    require(
        verification.get("output_receipt_sha256") == receipt_sha,
        f"{label} verification receipt SHA mismatch",
    )
    require(verification.get("s0_gate_pass") is False, f"{label} verification self-promoted S0")
    require(verification.get("d0_authorized") is False, f"{label} verification self-authorized D0")
    require(
        verification.get("model_fitting_authorized") is False,
        f"{label} verification authorized model fitting",
    )
    require(
        verification.get("scientific_effect") == "NOT_ESTABLISHED",
        f"{label} verification overstated effect",
    )
    if label != "core":
        require(verification.get("full_scope") is True, f"{label} verification is not full")
        require(verification.get("component_pass") is True, f"{label} verification did not pass")
    verifier_path = (REPO_ROOT / contract["verifier"]).resolve(strict=True)
    verifier_sha = _sha256_file(verifier_path)
    require(
        verification.get("verifier_sha256") == verifier_sha,
        f"{label} verifier SHA mismatch",
    )

    report = _read_json(root / contract["report"], f"{label} producer report")
    require(
        FORBIDDEN_METRIC_KEYS.isdisjoint(_all_keys(report)),
        f"{label} emitted a threshold-dependent metric",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": receipt_sha,
        "artifact_count": len(artifacts) + 1,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "verification_report_basename": verification_path.name,
        "verification_report_sha256": _sha256_file(verification_path),
        "verifier_sha256": verifier_sha,
        "verification": verification,
        "report": report,
    }


def verify(
    *,
    core_root: Path,
    core_verification: Path,
    orcid_root: Path,
    orcid_verification: Path,
    graph_root: Path,
    graph_verification: Path,
) -> dict[str, Any]:
    core = _validate_component(
        core_root,
        core_verification,
        CORE,
        label="core",
    )
    orcid = _validate_component(
        orcid_root,
        orcid_verification,
        ORCID,
        label="orcid_affiliation",
    )
    graph = _validate_component(
        graph_root,
        graph_verification,
        GRAPH,
        label="coauthor_graph",
    )

    core_verification_result = _mapping(
        core["verification"].get("field_year_verification"),
        "core field-year verification",
    )
    orcid_verification_result = _mapping(
        orcid["verification"].get("field_year_verification"),
        "ORCID field-year verification",
    )
    graph_verification_result = _mapping(
        graph["verification"].get("graph_verification"),
        "graph verification",
    )
    require(core_verification_result.get("field_count") == 26, "core field coverage mismatch")
    require(orcid_verification_result.get("field_count") == 26, "ORCID field coverage mismatch")
    require(
        graph_verification_result.get("core_cells")
        == core_verification_result.get("rows"),
        "graph-to-core cell closure mismatch",
    )
    require(
        graph_verification_result.get("rows")
        == 3 * core_verification_result.get("rows"),
        "graph profile row closure mismatch",
    )
    require(
        graph_verification_result.get("all_degree_sums_close") is True,
        "graph degree conservation missing",
    )
    require(
        graph_verification_result.get("all_component_bounds_hold") is True,
        "graph component bounds missing",
    )
    require(
        graph_verification_result.get("all_largest_component_shares_replayed")
        is True,
        "graph component share replay missing",
    )

    core_report = _mapping(core["report"], "core report")
    orcid_report = _mapping(orcid["report"], "ORCID report")
    graph_report = _mapping(graph["report"], "graph report")
    core_provenance = _mapping(core_report.get("provenance"), "core provenance")
    orcid_config = _mapping(orcid_report.get("provenance"), "ORCID provenance")
    graph_binding = _mapping(graph_report.get("core_binding"), "graph core binding")
    require(
        graph_binding.get("receipt_sha256") == core["receipt_sha256"],
        "graph did not bind the verified core receipt",
    )
    require(
        graph_binding.get("field_year_rows") == core_verification_result.get("rows"),
        "graph core row binding mismatch",
    )
    require(
        core_report.get("scope", {}).get("gpu_used") is False
        and orcid_report.get("scope", {}).get("gpu_used") is False
        and graph_report.get("scope", {}).get("gpu_used") is False,
        "an S0 component claims GPU use",
    )
    require(
        isinstance(core_provenance.get("config_sha256"), str)
        and isinstance(orcid_config.get("config_sha256"), str),
        "component provenance is incomplete",
    )

    source = _mapping(orcid_report.get("source_accounting"), "ORCID source accounting")
    tokens = _mapping(orcid_report.get("orcid_token_diagnostics"), "ORCID token diagnostics")
    institutions = _mapping(
        orcid_report.get("institution_diagnostics"),
        "institution diagnostics",
    )
    materialization = _mapping(
        graph_report.get("materialization_accounting"),
        "graph materialization accounting",
    )
    graph_inventory = _mapping(
        graph_report.get("field_year_inventory"),
        "graph field-year inventory",
    )
    graph_sums = _mapping(
        graph_inventory.get("profile_metric_sums"),
        "graph metric sums",
    )

    artifact_count = sum(
        int(component["artifact_count"])
        for component in (core, orcid, graph)
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": PASS_STATUS,
        "scope": {
            "s0_gate_pass": True,
            "threshold_free_only": True,
            "outcomes_read": False,
            "d0_discovery_pilot_authorized": True,
            "d0_entry_end_year": 2012,
            "g0_measurement_freeze_authorized": False,
            "final_panel_materialization_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "component_receipts": {
            label: {
                key: component[key]
                for key in (
                    "root_basename",
                    "receipt_sha256",
                    "artifact_count",
                    "artifact_tree_sha256",
                    "verification_report_basename",
                    "verification_report_sha256",
                    "verifier_sha256",
                )
            }
            for label, component in (
                ("core", core),
                ("orcid_affiliation", orcid),
                ("coauthor_graph", graph),
            )
        },
        "verification_accounting": {
            "artifacts_and_receipts_rehashed": artifact_count,
            "independent_verification_reports_rehashed": 3,
            "total_current_files_rehashed": artifact_count + 3,
            "all_current_artifacts_match_receipts": True,
            "all_verifier_hashes_match_current_repository": True,
            "all_component_outputs_unchanged_since_independent_verification": True,
        },
        "core_inventory": {
            "field_year_rows": core_verification_result["rows"],
            "field_count": core_verification_result["field_count"],
            "null_year_cells": core_verification_result["null_year_cells"],
            "work_rows": core_report["work_accounting"]["rows"],
            "author_event_rows": core_report["author_event_accounting"]["rows"],
            "attention_rows": core["verification"]["attention_replay"]["rows"],
        },
        "orcid_affiliation_inventory": {
            "raw_authorship_rows": source["raw_rows"],
            "retained_authorship_rows": source["retained_rows"],
            "nonnull_orcid_rows": source["nonnull_orcid_rows"],
            "syntax_valid_orcid_rows": source["syntax_valid_orcid_rows"],
            "checksum_valid_orcid_rows": source["checksum_valid_orcid_rows"],
            "stable_single_valid_orcid_author_tokens": (
                tokens["stable_single_valid_orcid_author_tokens"]
            ),
            "multiple_valid_orcid_author_tokens": (
                tokens["multiple_valid_orcid_author_tokens"]
            ),
            "shared_valid_orcids_across_stable_tokens": (
                tokens["shared_valid_orcids_across_stable_tokens"]
            ),
            "author_work_affiliation_rows": source["author_work_affiliation_rows"],
            "author_work_institution_rows": source["author_work_institution_rows"],
            "distinct_institution_tokens": institutions["distinct_institution_tokens"],
            "field_year_rows": orcid_verification_result["rows"],
        },
        "coauthor_graph_inventory": {
            "profile_caps": [10, 50, 100],
            "profiles_ranked_or_selected": False,
            "source_author_event_rows": materialization["source_rows"],
            "field_year_profile_rows": graph_verification_result["rows"],
            "profile_metric_sums": graph_sums,
            "unbounded_all_graph_materialized": False,
        },
        "authorization": {
            "d0_discovery_only_pre2013": "AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "g0_5_final_panel_conformance": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_holdout_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "next_required_implementation": {
            "gate": "D0",
            "role": "DISCOVERY_ONLY_MEASUREMENT_CALIBRATION_AND_FEASIBILITY",
            "entry_year_end": 2012,
            "final_field_choice_by_effect_size_allowed": False,
            "confirmatory_year_changes_allowed": False,
            "workload_class": "CPU_IO",
            "gpu_required": False,
        },
        "completion_verifier_sha256": _sha256_file(Path(__file__).resolve()),
        "claim_boundary": (
            "This receipt combines three independently verified, current, "
            "threshold-free S0 components and authorizes only the pre-2013 D0 "
            "discovery-calibration gate. It establishes current-snapshot "
            "OpenAlex author-token, ORCID-syntax/checksum, affiliation-token, "
            "institution-token, and predeclared non-ranked coauthor-graph "
            "inventories. It does not establish natural-person identity, ORCID "
            "ownership, historical affiliation or collaboration observability, "
            "a preferred graph cap, candidate eligibility, final fields or "
            "thresholds, G0, a final panel, model fitting, empirical or causal "
            "effects, or a scientific effect."
        ),
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    payload = _published_bytes(result)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o644, follow_symlinks=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--core-verification", type=Path, required=True)
    parser.add_argument("--orcid-root", type=Path, required=True)
    parser.add_argument("--orcid-verification", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--graph-verification", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="write a new compact S0 completion receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(
            core_root=args.core_root,
            core_verification=args.core_verification,
            orcid_root=args.orcid_root,
            orcid_verification=args.orcid_verification,
            graph_root=args.graph_root,
            graph_verification=args.graph_verification,
        )
        if args.report is not None:
            _write_report(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        S0CompletionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(json.dumps({"status": BLOCKED_STATUS, "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
