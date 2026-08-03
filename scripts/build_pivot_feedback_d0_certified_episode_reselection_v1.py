#!/usr/bin/env python3
"""Materialize the exact four globally certified D0 episode candidates."""

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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_certified_episode_reselection_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730"
)
STATUS = "PASS_EXACT_FOUR_GLOBAL_CERTIFICATE_BOUND_EPISODE_RESELECTION"


class CertifiedEpisodeReselectionError(RuntimeError):
    """Fail-closed exact episode-reselection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedEpisodeReselectionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _repo(path_text: str, expected_sha256: str) -> Path:
    path = (REPO_ROOT / path_text).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == expected_sha256, f"repository SHA: {path.name}")
    return path


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "bound root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"bound file: {path.name}")
    require(_sha256(path) == binding[sha_key], f"bound SHA: {path.name}")
    return path


def _top(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    path = (Path("/ssd") / str(binding[basename_key])).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "top-level binding")
    require(path.is_file() and not path.is_symlink(), f"top-level file: {path.name}")
    require(_sha256(path) == binding[sha_key], f"top-level SHA: {path.name}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_certified_episode_reselection_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_EXACT_FOUR_CERTIFIED_EPISODE_RESELECTION_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_doc = _yaml(_repo(str(parent["relative_path"]), str(parent["sha256"])))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    parent_auth = _mapping(parent_doc.get("authorization"), "parent authorization")
    for key, value in _mapping(
        parent.get("required_authorization"), "required authorization"
    ).items():
        require(parent_auth.get(key) == value, f"parent authorization: {key}")
    contract = _mapping(config.get("selection_contract"), "selection contract")
    exact_rows = contract.get("exact_rows")
    require(isinstance(exact_rows, list) and len(exact_rows) == 4, "exact rows")
    require(contract.get("candidate_substitution") == "FORBIDDEN", "substitution")
    require(contract.get("outcome_access") == "FORBIDDEN", "outcome access")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("exact_four_episode_reselection") == "AUTHORIZED", "reselection")
    for key in (
        "source_work_rescan",
        "author_opportunity_panel_rebuild",
        "production_pivot_classification",
        "discovery_outcome_opening",
        "new_api_request",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def load_inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    candidates = _mapping(frozen.get("measurement_abi_candidates"), "candidates")
    _repo(str(candidates["config_relative_path"]), str(candidates["config_sha256"]))
    candidate_path = _bound(candidates, "candidate_basename", "candidate_sha256")
    candidate_receipt_path = _bound(candidates, "receipt_basename", "receipt_sha256")
    candidate_receipt = _json(candidate_receipt_path)
    require(candidate_receipt.get("status") == candidates["receipt_status"], "candidate receipt")
    candidate_table = pq.read_table(candidate_path)
    require(candidate_table.num_rows == candidates["candidate_rows"] == 70277, "candidate rows")
    require(
        (candidate_table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == candidates["candidate_schema_version"],
        "candidate schema",
    )

    certificate = _mapping(frozen.get("selection_certificate"), "certificate")
    frontier_path = _bound(certificate, "frontier_basename", "frontier_sha256")
    certificate_path = _bound(certificate, "certificate_basename", "certificate_sha256")
    receipt_path = _bound(certificate, "receipt_basename", "receipt_sha256")
    verification_path = _top(certificate, "verification_basename", "verification_sha256")
    frontier = pq.read_table(frontier_path)
    require(frontier.num_rows == certificate["frontier_rows"] == 51, "frontier rows")
    require(
        (frontier.schema.metadata or {}).get(b"schema_version", b"").decode()
        == certificate["frontier_schema_version"],
        "frontier schema",
    )
    certificate_doc = _json(certificate_path)
    require(certificate_doc.get("schema_version") == certificate["certificate_schema_version"], "certificate schema")
    require(certificate_doc.get("status") == certificate["certificate_status"], "certificate status")
    require(_json(receipt_path).get("status") == certificate["receipt_status"], "closeout receipt")
    require(_json(verification_path).get("status") == certificate["verification_status"], "closeout verification")
    hashes = {
        "measurement_abi_candidates_sha256": _sha256(candidate_path),
        "measurement_abi_receipt_sha256": _sha256(candidate_receipt_path),
        "frontier_v10_sha256": _sha256(frontier_path),
        "terminal_certificate_v5_sha256": _sha256(certificate_path),
        "certificate_closeout_receipt_sha256": _sha256(receipt_path),
        "certificate_closeout_verification_sha256": _sha256(verification_path),
    }
    return candidate_table, frontier, certificate_doc, hashes


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        int(first["subfield_id"]) != int(second["subfield_id"])
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def select_rows(
    candidates: pa.Table,
    frontier: pa.Table,
    certificate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_rows = candidates.to_pylist()
    candidate_index = {
        (int(row["field_id"]), int(row["field_candidate_rank"])): row
        for row in candidate_rows
    }
    require(len(candidate_index) == len(candidate_rows), "candidate key uniqueness")
    frontier_rows = frontier.to_pylist()
    frontier_index = {
        (int(row["field_id"]), int(row["field_candidate_rank"])): row
        for row in frontier_rows
    }
    expected_pairs = {
        17: list(certificate["computer_science_selected_candidate_ranks"]),
        27: list(certificate["medicine_selected_candidate_ranks"]),
    }
    require(expected_pairs == {17: [1, 45], 27: [5, 6]}, "certificate pairs")
    exact_values = config["selection_contract"]["exact_rows"]
    rows: list[dict[str, Any]] = []
    for expected_value in exact_values:
        expected = _mapping(expected_value, "exact row")
        key = (int(expected["field_id"]), int(expected["field_candidate_rank"]))
        require(key in candidate_index and key in frontier_index, f"selection key: {key}")
        candidate = dict(candidate_index[key])
        frontier_row = frontier_index[key]
        require(frontier_row["content_state"] == "CONTENT_PASS", f"content state: {key}")
        require(int(candidate["topic_id"]) == int(expected["topic_id"]), f"topic: {key}")
        require(int(candidate["entry_year"]) == int(expected["entry_year"]), f"entry year: {key}")
        require(int(candidate["subfield_id"]) == int(expected["subfield_id"]), f"subfield: {key}")
        require(
            bool(candidate["episode_selected"])
            is bool(expected["prior_measurement_abi_selected"]),
            f"prior selection: {key}",
        )
        require(
            int(expected["field_candidate_rank"])
            in expected_pairs[int(expected["field_id"])],
            f"certificate membership: {key}",
        )
        candidate.update(
            {
                "certified_episode_index": int(expected["certified_episode_index"]),
                "field_episode_order": int(expected["field_episode_order"]),
                "content_state": "CONTENT_PASS",
                "selection_certificate_version": "v5",
                "prior_measurement_abi_selected": bool(
                    expected["prior_measurement_abi_selected"]
                ),
                "selection_change": str(expected["selection_change"]),
            }
        )
        candidate.pop("episode_selected")
        candidate.pop("episode_order")
        rows.append(candidate)
    rows.sort(key=lambda row: row["certified_episode_index"])
    require(
        [(row["field_id"], row["field_candidate_rank"]) for row in rows]
        == [(17, 1), (17, 45), (27, 5), (27, 6)],
        "selected order",
    )
    for field_id in (17, 27):
        pair = [row for row in rows if row["field_id"] == field_id]
        require(len(pair) == 2 and _compatible(pair[0], pair[1]), f"pair compatibility: {field_id}")
    contract = _mapping(config.get("selection_contract"), "contract")
    require(sum(row["selection_change"] == "RETAINED" for row in rows) == contract["retained_episode_rows"], "retained rows")
    require(
        sum(row["selection_change"] == "REPLACED_PRIOR_FIELD_FIRST" for row in rows)
        == contract["newly_selected_episode_rows"],
        "new selection rows",
    )
    return rows


def _selection_schema(candidate_schema: pa.Schema, schema_version: str) -> pa.Schema:
    fields = [
        field
        for field in candidate_schema
        if field.name not in {"episode_selected", "episode_order"}
    ]
    fields.extend(
        [
            pa.field("certified_episode_index", pa.uint8(), nullable=False),
            pa.field("field_episode_order", pa.uint8(), nullable=False),
            pa.field("content_state", pa.string(), nullable=False),
            pa.field("selection_certificate_version", pa.string(), nullable=False),
            pa.field("prior_measurement_abi_selected", pa.bool_(), nullable=False),
            pa.field("selection_change", pa.string(), nullable=False),
        ]
    )
    return pa.schema(fields, metadata={b"schema_version": schema_version.encode()})


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        artifact["rows"] = rows
    return artifact


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output basename")
    require(not output_root.exists(), "output exists")
    candidates, frontier, certificate, input_hashes = load_inputs(config)
    rows = select_rows(candidates, frontier, certificate, config)
    schema = _selection_schema(candidates.schema, str(output["selection_schema_version"]))
    selection = pa.Table.from_pylist(rows, schema=schema)
    decision = {
        "schema_version": "pivot_feedback_d0_certified_episode_reselection_decision.v1",
        "status": STATUS,
        "selection_source": "TERMINAL_SELECTION_CERTIFICATE_V5_AND_FRONTIER_V10",
        "selected_episode_rows": 4,
        "retained_episode_rows": 3,
        "newly_selected_episode_rows": 1,
        "deselected_prior_episode_rows": 1,
        "selected_field_candidate_ranks": {
            "computer_science": [1, 45],
            "medicine": [5, 6],
        },
        "deselected_prior_episode": config["selection_contract"][
            "deselected_prior_episode"
        ],
        "source_work_rescan_executed": False,
        "author_opportunity_panel_rebuild_executed": False,
        "outcomes_read": False,
        "human_review": "NOT_RUN_BY_DESIGN",
    }
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir="/ssd")
    )
    temporary.chmod(0o700)
    try:
        selection_path = temporary / str(output["selection_basename"])
        pq.write_table(selection, selection_path, compression="zstd", use_dictionary=True)
        selection_path.chmod(0o600)
        decision_path = temporary / str(output["decision_basename"])
        _write_json(decision_path, decision)
        report_path = temporary / str(output["report_basename"])
        report_path.write_text(
            "\n".join(
                [
                    "# Certified episode reselection",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    "- certified episodes: 4",
                    "- Computer Science ranks: 1, 45",
                    "- Medicine ranks: 5, 6",
                    "- retained from measurement-ABI selection: 3",
                    "- newly selected: Medicine rank 5",
                    "- deselected: Medicine rank 1",
                    "- source work rescan: not executed",
                    "- author/opportunity panel rebuild: not executed",
                    "- outcomes read: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report_path.chmod(0o600)
        artifacts = {
            "selection": _artifact(selection_path, rows=4),
            "decision": _artifact(decision_path),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_certified_episode_reselection_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_hashes": input_hashes,
            "artifacts": artifacts,
            "selected_episode_rows": 4,
            "retained_episode_rows": 3,
            "newly_selected_episode_rows": 1,
            "deselected_prior_episode_rows": 1,
            "selected_field_candidate_ranks": {
                "computer_science": [1, 45],
                "medicine": [5, 6],
            },
            "global_selection_certificate_bound": True,
            "deterministic_reselection_executed": True,
            "source_work_rescan_executed": False,
            "author_opportunity_panel_rebuild_executed": False,
            "production_pivot_classification_executed": False,
            "outcomes_read": False,
            "api_requests_created": 0,
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
        CertifiedEpisodeReselectionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_CERTIFIED_EPISODE_RESELECTION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
