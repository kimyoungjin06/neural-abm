#!/usr/bin/env python3
"""Materialize the rank-5 post-diagnostic pass and global certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730"
)
STATUS = "PASS_RANK5_POST_DIAGNOSTIC_CONTENT_PASS_GLOBAL_CERTIFICATES_FROZEN"


class PostDiagnosticCertificateCloseoutError(RuntimeError):
    """Fail-closed production certificate closeout error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDiagnosticCertificateCloseoutError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input: {path.name}")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _top(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    path = (Path("/ssd") / str(binding[basename_key])).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "top-level input")
    require(path.is_file() and not path.is_symlink(), f"top-level input: {path.name}")
    require(_sha256(path) == binding[sha_key], f"top-level SHA: {path.name}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_POST_DIAGNOSTIC_PASS_CERTIFICATE_CLOSEOUT_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    selection = _mapping(config.get("selection_contract"), "selection")
    selection_doc = _yaml(_repo(selection))
    require(selection_doc.get("schema_version") == selection["required_schema_version"], "selection schema")
    contract = _mapping(selection_doc.get("selection_contract"), "selection contract")
    require(contract.get("candidate_order") == selection["candidate_order"], "candidate order")
    require(contract.get("eager_algorithm", {}).get("first") == selection["first_selected"], "first rule")
    require(contract.get("eager_algorithm", {}).get("second") == selection["second_selected"], "second rule")
    require(contract.get("compatibility", {}).get("distinct_subfield_required") is True, "subfield rule")
    require(contract.get("compatibility", {}).get("minimum_entry_year_separation") == 9, "year rule")
    disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    require(disposition.get("content_state") == "CONTENT_PASS", "content state")
    require(disposition.get("final_accept_rows") == disposition.get("pass_threshold_rows") == 32, "pass lower bound")
    require(disposition.get("retrospective_post_diagnostic") is True, "retrospective")
    require(disposition.get("out_of_sample_validation") is False, "out of sample")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("production_disposition") == "AUTHORIZED_CONTENT_PASS_ONLY", "production")
    require(authorization.get("selection_certificate") == "AUTHORIZED_EXACT_RANKS_ONLY", "certificate")
    require(
        authorization.get("deterministic_reselection")
        == "NOT_AUTHORIZED_UNTIL_CLOSEOUT_INDEPENDENTLY_VERIFIED",
        "reselection",
    )
    for key in (
        "new_api_request",
        "production_pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def load_inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("prior_closeout"), "prior")
    dispositions_path = _bound(prior, "dispositions_basename", "dispositions_sha256")
    frontier_path = _bound(prior, "frontier_basename", "frontier_sha256")
    certificate_path = _bound(prior, "certificate_basename", "certificate_sha256")
    prior_receipt_path = _bound(prior, "receipt_basename", "receipt_sha256")
    prior_verification_path = _top(prior, "verification_basename", "verification_sha256")
    require(_json(prior_receipt_path).get("status") == prior["receipt_status"], "prior receipt")
    require(_json(prior_verification_path).get("status") == prior["verification_status"], "prior verification")
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    require(dispositions.num_rows == prior["disposition_rows"] == 8, "disposition rows")
    require(frontier.num_rows == prior["frontier_rows"] == 51, "frontier rows")
    require(_json(certificate_path).get("global_selection_certificate_issued") is False, "prior certificate")

    result = _mapping(frozen.get("verified_rule_result"), "result")
    replay_path = _bound(result, "replay_basename", "replay_sha256")
    candidate_path = _bound(result, "candidate_basename", "candidate_sha256")
    result_receipt_path = _bound(result, "receipt_basename", "receipt_sha256")
    result_verification_path = _top(result, "verification_basename", "verification_sha256")
    require(_json(result_receipt_path).get("status") == result["receipt_status"], "result receipt")
    require(_json(result_verification_path).get("status") == result["verification_status"], "result verification")
    replay = pq.read_table(replay_path)
    require(replay.num_rows == result["replay_rows"] == 9, "replay rows")
    require(sum(replay.column("deterministic_accept").to_pylist()) == result["accepted_rows"] == 2, "accepted rows")
    candidate = _json(candidate_path)
    require(candidate.get("candidate_content_state") == "CONTENT_PASS", "candidate state")
    require(candidate.get("final_accept_rows") == 32, "candidate accept")
    require(candidate.get("production_disposition_created") is False, "premature disposition")
    hashes = {
        "prior_dispositions_sha256": _sha256(dispositions_path),
        "prior_frontier_sha256": _sha256(frontier_path),
        "prior_certificate_sha256": _sha256(certificate_path),
        "prior_receipt_sha256": _sha256(prior_receipt_path),
        "prior_verification_sha256": _sha256(prior_verification_path),
        "rule_replay_sha256": _sha256(replay_path),
        "rule_candidate_sha256": _sha256(candidate_path),
        "rule_receipt_sha256": _sha256(result_receipt_path),
        "rule_verification_sha256": _sha256(result_verification_path),
    }
    return dispositions, frontier, candidate, hashes


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["subfield_id"] != second["subfield_id"]
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def _select_pair(rows: Sequence[Mapping[str, Any]], field_id: int) -> list[int]:
    field_rows = sorted(
        (row for row in rows if row["field_id"] == field_id),
        key=lambda row: row["field_candidate_rank"],
    )
    first: Mapping[str, Any] | None = None
    for row in field_rows:
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        require(state == "CONTENT_PASS", f"unresolved first candidate: field {field_id}")
        first = row
        break
    require(first is not None, f"missing first candidate: field {field_id}")
    for row in field_rows:
        if row["field_candidate_rank"] <= first["field_candidate_rank"]:
            continue
        if not _compatible(first, row):
            continue
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        require(state == "CONTENT_PASS", f"unresolved compatible candidate: field {field_id}")
        return [int(first["field_candidate_rank"]), int(row["field_candidate_rank"])]
    raise PostDiagnosticCertificateCloseoutError(f"missing second candidate: field {field_id}")


def update_tables(
    dispositions: pa.Table,
    frontier: pa.Table,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[int, list[int]]]:
    frozen = _mapping(config.get("rank5_production_disposition"), "disposition")
    require(candidate.get("final_accept_rows") == frozen["final_accept_rows"], "candidate accept binding")
    require(candidate.get("final_reject_rows") == frozen["final_reject_rows"], "candidate reject binding")
    require(candidate.get("final_unresolved_rows") == frozen["final_unresolved_rows"], "candidate unresolved binding")
    disposition_rows = dispositions.to_pylist()
    rank5 = [
        row
        for row in disposition_rows
        if row["field_id"] == frozen["field_id"]
        and row["field_candidate_rank"] == frozen["field_candidate_rank"]
    ]
    require(len(rank5) == 1, "rank5 disposition")
    rank5[0].update(
        {
            "content_state": frozen["content_state"],
            "disposition_reason": frozen["disposition_reason"],
            "production_rule_version": frozen["production_rule_version"],
            "source_replay_state": frozen["source_replay_state"],
            "post_diagnostic_amendment_used": frozen["post_diagnostic_amendment_used"],
            "human_review_used": frozen["human_review_used"],
            "api_labels_used_as_ground_truth": frozen["api_labels_used_as_ground_truth"],
        }
    )
    output = _mapping(config.get("output_contract"), "output")
    dispositions_out = pa.Table.from_pylist(
        disposition_rows,
        schema=dispositions.schema.with_metadata(
            {b"schema_version": str(output["dispositions_schema_version"]).encode()}
        ),
    )

    frontier_frozen = _mapping(config.get("frontier_v10"), "frontier")
    frontier_rows = frontier.to_pylist()
    rank5_frontier = next(
        row for row in frontier_rows if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    rank6_frontier = next(
        row for row in frontier_rows if row["field_id"] == 27 and row["field_candidate_rank"] == 6
    )
    rank5_frontier.update(
        {
            "frontier_role": frontier_frozen["rank5_frontier_role"],
            "content_state": "CONTENT_PASS",
            "state_reason": frontier_frozen["rank5_state_reason"],
            "targeted_sample_required": False,
        }
    )
    rank6_frontier.update(
        {
            "frontier_role": frontier_frozen["rank6_frontier_role"],
            "targeted_sample_required": False,
        }
    )
    selected = {
        17: _select_pair(frontier_rows, 17),
        27: _select_pair(frontier_rows, 27),
    }
    require(selected == {17: [1, 45], 27: [5, 6]}, "selected pairs")
    counts = {
        "total_frontier_rows": len(frontier_rows),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in frontier_rows),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in frontier_rows),
        "total_content_unresolved_rows": sum(row["content_state"] == "CONTENT_UNRESOLVED" for row in frontier_rows),
        "total_not_evaluated_rows": sum(row["content_state"] == "NOT_EVALUATED" for row in frontier_rows),
        "field_certificates_issued": 2,
        "global_selection_certificate_issued": True,
        "deterministic_reselection_executed": False,
    }
    for key, value in counts.items():
        require(value == frontier_frozen[key], f"frontier count: {key}")
    frontier_out = pa.Table.from_pylist(
        frontier_rows,
        schema=frontier.schema.with_metadata(
            {b"schema_version": str(output["frontier_schema_version"]).encode()}
        ),
    )
    return dispositions_out, frontier_out, counts, selected


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output name")
    require(not output_root.exists(), "output exists")
    dispositions_in, frontier_in, candidate, hashes = load_inputs(config)
    dispositions, frontier, frontier_counts, selected = update_tables(
        dispositions_in, frontier_in, candidate, config
    )
    certificate = _mapping(config.get("terminal_certificate_status"), "certificate")
    require(certificate.get("computer_science_selected_candidate_ranks") == selected[17], "CS certificate")
    require(certificate.get("medicine_selected_candidate_ranks") == selected[27], "Medicine certificate")
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")
    require(certificate.get("deterministic_reselection_executed") is False, "reselection")

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir="/ssd")
    )
    temporary.chmod(0o700)
    try:
        dispositions_path = temporary / str(output["dispositions_basename"])
        frontier_path = temporary / str(output["frontier_basename"])
        pq.write_table(dispositions, dispositions_path, compression="zstd", use_dictionary=True)
        pq.write_table(frontier, frontier_path, compression="zstd", use_dictionary=True)
        dispositions_path.chmod(0o600)
        frontier_path.chmod(0o600)
        certificate_path = temporary / str(output["certificate_basename"])
        _write_json(certificate_path, certificate)
        report_path = temporary / str(output["report_basename"])
        report_path.write_text(
            "\n".join(
                [
                    "# Rank-5 post-diagnostic pass and certificate closeout",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    "- rank-5 counts: 32 accept / 25 reject / 7 unresolved",
                    "- rank-5 production state: `CONTENT_PASS`",
                    "- Computer Science certificate: ranks 1 and 45",
                    "- Medicine certificate: ranks 5 and 6",
                    "- global selection certificate: issued",
                    "- method label: retrospective post-diagnostic amendment",
                    "- deterministic reselection: not executed",
                    "- human review: `NOT_RUN_BY_DESIGN`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report_path.chmod(0o600)
        artifacts = {
            "dispositions": _artifact(dispositions_path, rows=8),
            "frontier": _artifact(frontier_path, rows=51),
            "certificate_status": _artifact(certificate_path),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_hashes": hashes,
            "artifacts": artifacts,
            "cumulative_production_dispositions": 8,
            "rank5_counts": {"accept": 32, "reject": 25, "unresolved": 7},
            "rank5_content_state": "CONTENT_PASS",
            "frontier": frontier_counts,
            "selected_candidate_ranks": {
                "computer_science": selected[17],
                "medicine": selected[27],
            },
            "terminal_certificate_status": certificate,
            "field_certificates_issued": 2,
            "global_selection_certificate_issued": True,
            "deterministic_reselection_executed": False,
            "deterministic_reselection_successor_authorized": False,
            "retrospective_post_diagnostic": True,
            "out_of_sample_validation": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_used_as_ground_truth": False,
            "pubmed_indexing_used_as_ground_truth": False,
            "new_api_requests_created": 0,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        _write_json(temporary / str(output["receipt_basename"]), receipt)
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == Path("/ssd"), "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.config, args.output), indent=2, sort_keys=True))
        return 0
    except (
        PostDiagnosticCertificateCloseoutError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_POST_DIAGNOSTIC_CERTIFICATE_CLOSEOUT", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
